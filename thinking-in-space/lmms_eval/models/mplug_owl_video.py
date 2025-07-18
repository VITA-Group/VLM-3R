from datetime import timedelta
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.mplug_owl_video.modeling_mplug_owl import (
    MplugOwlForConditionalGeneration,
)
from lmms_eval.models.mplug_owl_video.processing_mplug_owl import (
    MplugOwlImageProcessor,
    MplugOwlProcessor,
)

eval_logger = logger


@register_model("mplug_owl_video")
class mplug_Owl(lmms):
    def __init__(
        self,
        pretrained: str = "MAGAer13/mplug-owl-llama-7b-video",
        device: Optional[str] = "cuda:0",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        device_map="cuda:0",
        num_frames: Union[str, int] = 4,
        **kwargs,
    ) -> None:
        """
        Install instructions:
        1. Install lmms-eval
        cd lmms-eval
        pip install -e .;
        2. Install other packages with restricted versions
        pip install av sentencepiece protobuf==3.20 transformers==4.28.1 einops;
        """
        super().__init__()

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        # import pdb; pdb.set_trace()
        # This is very slow. Their issue, not mine
        # Also, keep transformers in version 4.28.1
        # They put a Config object inside a config object, this is not acceptable
        # for transformers == 4.39.1, object type not serializable
        # Protobuf needs to be in 3.20.x otherwise error
        # ヽ(｀Д´)ﾉ
        self._model = MplugOwlForConditionalGeneration.from_pretrained(
            pretrained,
            torch_dtype=torch.bfloat16,
        )
        self.image_processor = MplugOwlImageProcessor.from_pretrained(pretrained)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.processor = MplugOwlProcessor(self.image_processor, self.tokenizer)
        self.model.eval()
        self.batch_size_per_gpu = batch_size
        self.num_frames = num_frames

        self.model.to(self.device)

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.process_index
            self._world_size = self.accelerator.num_processes
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def format_prompt(self, question):
        prompts = [f" <|video|> Question : {question} Answer : "]
        return prompts

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            inputs = self.processor(text=self.format_prompt(contexts), videos=visuals, num_frames=self.num_frames, return_tensors="pt")
            pixel_values_videos = inputs["video_pixel_values"]
            if pixel_values_videos.shape[2] != self.num_frames:
                empty_frames = torch.zeros((1, pixel_values_videos.shape[1], self.num_frames - pixel_values_videos.shape[2], *pixel_values_videos.shape[3:]), dtype=pixel_values_videos.dtype)
                pixel_values_videos = torch.cat([pixel_values_videos, empty_frames], dim=2)
                inputs["video_pixel_values"] = pixel_values_videos
            inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            if "max_new_tokens" in gen_kwargs:
                gen_kwargs["max_length"] = gen_kwargs["max_new_tokens"]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_length"] = 128
            if "do_sample" not in gen_kwargs:
                gen_kwargs["do_sample"] = False
            if "top_k" not in gen_kwargs:
                gen_kwargs["top_k"] = 1

            generate_kwargs = {"do_sample": gen_kwargs["do_sample"], "top_k": gen_kwargs["top_k"], "max_length": gen_kwargs["max_length"]}

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generate_kwargs)
            sentence = self.tokenizer.decode(outputs.tolist()[0], skip_special_tokens=True)
            pbar.update(1)
            res.append(sentence)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        return super().loglikelihood(requests)
