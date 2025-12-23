#!/usr/bin/env bash
set -euo pipefail

MODEL="stabilityai/stable-diffusion-xl-base-1.0"
LORA_CONTENT="./lora-sdxl-dog/pytorch_lora_weights.safetensors"
LORA_STYLE="./lora-sdxl-waterpaintingnew/pytorch_lora_weights.safetensors"
PROMPT="a sbu dog in cba style"

CONTENT_FOLDER="datasets/dog"
STYLE_FOLDER="datasets/waterpainting"
TEXT_CONTENT="sbu dog"
TEXT_STYLE="cba style"

ALPHAS=(2 3 4 4)
BETAS=(1.5 2 2 3)     # 與 ALPHAS 對應或你想要的值

for i in "${!ALPHAS[@]}"; do
  alpha="${ALPHAS[$i]}"
  beta="${BETAS[$i]}"
  out="outputs_a${alpha}_b${beta}"
  echo "==> Running ${out}"
  python inference_sd.py \
    --pretrained_model_name_or_path "$MODEL" \
    --lora_name_or_path_content "$LORA_CONTENT" \
    --lora_name_or_path_style "$LORA_STYLE" \
    --prompt "$PROMPT" \
    --output_folder "$out" \
    --alpha "$alpha" \
    --beta "$beta"

  echo "==> Evaluating ${out}"
  python evaluate_clip_scores.py \
    --generated_folder "$out" \
    --content_folder "$CONTENT_FOLDER" \
    --style_folder "$STYLE_FOLDER" \
    --text_content "$TEXT_CONTENT" \
    --text_style "$TEXT_STYLE" \
    --save_json "$out/clip_scores.json"
done

echo "All runs done."