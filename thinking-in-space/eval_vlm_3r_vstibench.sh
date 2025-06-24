export CUDA_VISIBLE_DEVICES=0 # If you have multiple GPUs, you can set the actual GPU IDs, e.g., "0,1,2"
export LMMS_EVAL_LAUNCHER="accelerate"

# If you have multiple GPUs, you can set --num_processes=the number of GPUs to use
accelerate launch \
    --num_processes=1 \
    -m lmms_eval \
    --model vlm_3r \
    --model_args pretrained=Journey9ni/vlm-3r-llava-qwen2-lora-vstibench,model_base=lmms-lab/LLaVA-NeXT-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=32 \
    --tasks vstibench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix vlm_3r_7b_qwen2_lora \
    --output_path logs/$(TZ="America/New_York" date "+%Y%m%d")/vstibench