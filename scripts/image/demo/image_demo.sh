#!/bin/bash

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

CKPT=$1
CONV_MODE=$2
POOL_STRIDE=$3
POOL_MODE=$4
NEWLINE_POSITION=$5
OVERWRITE_FLAG=$6
IMAGE_DIR=$7
MODEL_BASE=$8

if [ -z "$POOL_STRIDE" ]; then
    SAVE_DIR_SUFFIX=""
else
    SAVE_DIR_SUFFIX="_stride_${POOL_STRIDE}"
fi

OVERWRITE_STR=""
if [ "$OVERWRITE_FLAG" = True ]; then
    OVERWRITE_STR="--overwrite"
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}${SAVE_DIR_SUFFIX}
else
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}${SAVE_DIR_SUFFIX}_overwrite_False
fi

python3 playground/demo/image_demo.py \
    --model-path $CKPT \
    --model-base ${MODEL_BASE} \
    --image_dir ${IMAGE_DIR} \
    --output_dir ./work_dirs/image_demo/$SAVE_DIR \
    --output_name pred \
    $OVERWRITE_STR \
    --mm_spatial_pool_stride ${POOL_STRIDE:-2} \
    --conv-mode $CONV_MODE \
    --mm_spatial_pool_mode ${POOL_MODE:-average} \
    --mm_newline_position ${NEWLINE_POSITION:-grid} \
    --prompt "How many white chairs are there?" 