import argparse
import torch
import os
import json
import ray
import time
import numpy as np
from tqdm import tqdm
import shortuuid
import fasteners

# Fix bitsandbytes CUDA setup issue in Ray workers
os.environ['BNB_CUDA_VERSION'] = '121'  # CUDA 12.1
os.environ['BITSANDBYTES_NOWELCOME'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3')

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from PIL import Image
import math
from llava.eval.model_vqa import preprocess_qwen

from decord import VideoReader, cpu
import warnings
warnings.filterwarnings("ignore")
import copy


def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    """Load video frames from file."""
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames, frame_time, video_time


def load_precomputed_features(feature_path):
    """Load precomputed features from disk."""
    if os.path.exists(feature_path):
        features = torch.load(feature_path, map_location='cpu')
        return features
    return None


@ray.remote(num_gpus=1)
def eval_model(questions, args):
    # Fix bitsandbytes CUDA setup in worker process
    gpu_ids = ray.get_gpu_ids()
    if gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ids[0])
    os.environ['BNB_CUDA_VERSION'] = '121'  # CUDA 12.1
    os.environ['BITSANDBYTES_NOWELCOME'] = '1'
    
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = "llava_qwen_lora"
    tokenizer, model, processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, torch_dtype="bfloat16"
    )
    model.to(device="cuda")
    
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "a")
    file_lock = fasteners.InterProcessLock(ans_file)
    
    inference_time = []
    feature_load_time = []
    
    for line in tqdm(questions):
        idx = line["question_id"]
        scene_id = line["scene_id"]
        video_path = os.path.join(args.video_folder, f"{scene_id}.mp4")
        qs = line["question"]

        args.conv_mode = "qwen_1_5"

        start_time = time.time()
        
        # Try to load precomputed spatial features first
        spatial_features = None
        if args.feature_dir:
            # Following the pattern from train.py: replace .mp4 with .pt and videos with spatial_features
            feature_path = os.path.join(args.feature_dir, f"{scene_id}.pt")
            if not os.path.exists(feature_path):
                # Try alternative path pattern
                feature_path = os.path.join(args.video_folder, f"{scene_id}.mp4").replace('.mp4', '.pt').replace('videos', 'spatial_features')
            
            if os.path.exists(feature_path):
                # Load precomputed spatial features
                feature_load_time.append(time.time() - start_time)
                spatial_features = torch.load(feature_path, map_location='cuda')
                spatial_features = [spatial_features]
                # Still need to load video for metadata, but can skip processing
                video, frame_time, video_time = load_video(video_path, args.max_frames_num, 1, force_sample=True)
                video = processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
                video = [video]
            else:
                # No precomputed features, process normally
                video, frame_time, video_time = load_video(video_path, args.max_frames_num, 1, force_sample=True)
                video = processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
                video = [video]
        else:
            # No feature directory specified, process normally
            video, frame_time, video_time = load_video(video_path, args.max_frames_num, 1, force_sample=True)
            video = processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
            video = [video]
        
        time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(video[0]) if isinstance(video, list) else video.shape[0]} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
        question = DEFAULT_IMAGE_TOKEN + f"{time_instruction}\n{qs}\nAnswer the question simply."
        
        conv = copy.deepcopy(conv_templates[args.conv_mode])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda")

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=video,
                spatial_features=spatial_features,  # Pass precomputed spatial features
                modalities=["video"],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=512,
                use_cache=True,
            )
        
        inference_time.append(time.time() - start_time)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        
        with file_lock:
            ans_file.write(json.dumps({
                "question_id": idx,
                "prompt": prompt_question,
                "text": outputs,
                "answer_id": ans_id,
                "model_id": model_name,
                "metadata": {}
            }) + "\n")
            ans_file.flush()
    
    ans_file.close()
    
    return {
        'inference_time': inference_time,
        'feature_load_time': feature_load_time
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/llava3d-v1.5-7b-task-v3-tuning")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--video-folder", type=str, default="playground/data/LLaVA-3D-Pretrain")
    parser.add_argument("--feature-dir", type=str, default=None,
                       help="Directory containing precomputed features")
    parser.add_argument("--question-file", type=str, default="playground/data/annotations/llava3d_sqa3d_val_question.json")
    parser.add_argument("--answers-file", type=str, default="./llava3d_sqa3d_val_answer_pred.json")
    parser.add_argument("--max-frames-num", type=int, default=32)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--n_gpu", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with Ray local mode")
    args = parser.parse_args()

    # Load questions
    with open(args.question_file, 'r') as file:
        questions = json.load(file)
    
    if os.path.exists(args.answers_file):
        print(f"The {args.answers_file} already exists! Appending to it.")
    
    # Initialize Ray
    if args.n_gpu == 1 and args.debug:
        # Use Ray local mode for debugging
        ray.init(local_mode=True)
    else:
        ray.init(num_gpus=args.n_gpu)
    
    # Distribute questions across GPUs
    features = []
    for i in range(args.n_gpu):
        features.append(eval_model.remote(questions[i::args.n_gpu], args))
    
    # Get results
    results = ray.get(features)
    
    # Aggregate statistics
    all_inference_times = []
    all_feature_load_times = []
    
    for result in results:
        all_inference_times.extend(result['inference_time'])
        all_feature_load_times.extend(result['feature_load_time'])
    
    print(f"Average inference time: {np.mean(all_inference_times):.2f} seconds")
    if all_feature_load_times:
        print(f"Average feature load time: {np.mean(all_feature_load_times):.4f} seconds")
    print(f"Total questions processed: {len(questions)}")