import argparse
import torch
import os
import json
from tqdm import tqdm
from PIL import Image
import numpy as np

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_anyres_image, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from transformers import AutoConfig

def parse_args():
    """
    Parse command-line arguments for multi-image inference.
    """
    parser = argparse.ArgumentParser(description="Run multi-image inference with a LLaVA model.")

    # Define the command-line arguments
    parser.add_argument("--image_dir", help="Path to the directory containing images.", required=True)
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", required=True)
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=True)
    parser.add_argument("--prompt", type=str, required=True, help="The prompt to use for inference.")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--load_8bit", action='store_true', help="Load the model in 8-bit mode.")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing config for model loading.")

    # Arguments from video_demo that might be relevant for model config
    parser.add_argument("--mm_resampler_type", type=str, default="spatial_pool")
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=4)
    parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--image_aspect_ratio", type=str, default="anyres")
    parser.add_argument("--image_grid_pinpoints", type=str, default="[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]")
    parser.add_argument("--mm_patch_merge_type", type=str, default="spatial_unpad")
    parser.add_argument("--mm_newline_position", type=str, default="no_token")

    return parser.parse_args()

def load_images_from_dir(image_dir):
    """
    Load all images from a directory.
    
    Args:
        image_dir (str): Path to the directory containing images.
        
    Returns:
        list: A list of PIL Image objects.
        list: A list of image file paths, sorted alphabetically.
    """
    image_paths = []
    for filename in sorted(os.listdir(image_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_paths.append(os.path.join(image_dir, filename))
    
    images = [Image.open(p).convert("RGB") for p in image_paths]
    return images, image_paths

def run_inference(args):
    """
    Run multi-image inference.
    """
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    
    # Set model configuration parameters if they exist
    overwrite_config = {}
    if args.overwrite:
        overwrite_config["mm_spatial_pool_mode"] = args.mm_spatial_pool_mode
        overwrite_config["mm_spatial_pool_stride"] = args.mm_spatial_pool_stride
        overwrite_config["mm_newline_position"] = args.mm_newline_position

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, 
        load_8bit=args.load_8bit, 
        overwrite_config=overwrite_config
    )
    
    # Move model to cuda
    model.to("cuda")

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_name = args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.json")
    ans_file = open(answers_file, "w")

    # Load images from the specified directory
    pil_images, image_paths = load_images_from_dir(args.image_dir)
    num_images = len(pil_images)

    if num_images == 0:
        print(f"No images found in directory: {args.image_dir}")
        return

    print(f"Found {num_images} images for inference.")

    # Preprocess images
    image_tensor = image_processor.preprocess(pil_images, return_tensors="pt")["pixel_values"]
    # Split the batched tensor into a list of single image tensors
    images = [img.half().cuda() for img in image_tensor]

    # Prepare the prompt
    question = args.prompt
    # question = 'How many white chairs are there?'
    
    # Create image placeholders
    if model.config.mm_use_im_start_end:
        image_placeholders = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n') * num_images
    else:
        image_placeholders = (DEFAULT_IMAGE_TOKEN + '\n') * num_images

    qs = image_placeholders + question

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    
    if tokenizer.pad_token_id is None and "qwen" in tokenizer.name_or_path.lower():
        print("Setting pad token to bos token for qwen model.")
        tokenizer.pad_token_id = 151643
            
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # Run model inference
    with torch.inference_mode():
        output_ids = model.generate(
            inputs=input_ids,
            images=images,  # Pass the list of image tensors
            attention_mask=attention_masks,
            modalities=["image"] * num_images,  # Specify modality for each image
            do_sample=False,
            temperature=0.0,
            max_new_tokens=1024,
            top_p=0.1,
            num_beams=1,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    
    print(f"Question: {question}\n")
    print(f"Images: {', '.join(image_paths)}\n")
    print(f"Response: {outputs}\n")

    # Save results
    sample_set = {
        "Q": question,
        "image_paths": image_paths,
        "pred": outputs
    }
    ans_file.write(json.dumps(sample_set, ensure_ascii=False) + "\n")
    ans_file.flush()

    ans_file.close()

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)