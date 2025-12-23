#!/bin/bash

# Configuration
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export LORA_PATH_CONTENT="./lora-sdxl-dog/pytorch_lora_weights.safetensors"
export LORA_PATH_STYLE="./lora-sdxl-waterpaintingnew/pytorch_lora_weights.safetensors"
export OUTPUT_FOLDER="./outputs_lazy_ziplora_v20"
export PROMPT="a sbu dog in cba style"

# Algorithm Parameters (V20 The Golden Mean)
# Interpolating between V18 (C 0.71 / S 0.62) and V19 (C 0.55 / S 0.67).
# Target: C 0.65 / S 0.65.
#
# Retention: 0.76 (Midway between V18's 0.77 and V19's 0.74)
#            Should drop Content from 0.71 to ~0.65.
# Fg Alpha:  0.80 (Midway between 0.75 and 0.85)
#            Should boost Style from 0.62 to ~0.65.
# Bg Boost:  2.10 (Midway between 2.0 and 2.2)
#            Supports Style boost.

export RETENTION=0.76
export FG_ALPHA=0.80
export BG_BOOST=2.10

# Create output directory
mkdir -p "$OUTPUT_FOLDER"

echo "========================================"
echo "RUNNING LAZY ZIPLORA V20 (The Golden Mean)"
echo "Content:   $LORA_PATH_CONTENT"
echo "Style:     $LORA_PATH_STYLE"
echo "Retention: $RETENTION"
echo "Fg Alpha:  $FG_ALPHA"
echo "Bg Boost:  $BG_BOOST"
echo "========================================"

# Using V17 Script (Energy Preserving Logic)
python lazy_ziplora_v17.py \
  --content "$LORA_PATH_CONTENT" \
  --style "$LORA_PATH_STYLE" \
  --output "ziplora_v20.safetensors" \
  --retention "$RETENTION" \
  --fg_alpha "$FG_ALPHA" \
  --bg_boost "$BG_BOOST"

if [ $? -ne 0 ]; then
    echo "Merge failed!"
    exit 1
fi

echo "========================================"
echo "RUNNING INFERENCE"
echo "Model: $MODEL_NAME"
echo "Fused LoRA: ziplora_v20.safetensors"
echo "========================================"

# 2. Run Inference
accelerate launch --multi_gpu --num_processes 4 inference_standard.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --fused_lora_path="ziplora_v20.safetensors" \
  --output_folder="$OUTPUT_FOLDER" \
  --prompt="$PROMPT"

echo "Done! Check outputs in $OUTPUT_FOLDER"
