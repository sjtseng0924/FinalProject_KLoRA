export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export LORA_PATH_CONTENT="./lora-sdxl-bear/pytorch_lora_weights.safetensors"
export LORA_PATH_STYLE_A="./lora-sdxl-waterpaintingnew/pytorch_lora_weights.safetensors"
export LORA_PATH_STYLE_B="./lora-sdxl-anime-new/pytorch_lora_weights.safetensors"
export STYLE_WEIGHT_A=0.497314453125
export STYLE_WEIGHT_B=0.5029296875
export OUTPUT_FOLDER="outputs-sdxl-bear-waterpainting_anime"
export PROMPT="a td bear in xyz and cba style"
python inference_sd_twostyle.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --lora_name_or_path_content="$LORA_PATH_CONTENT" \
  --lora_name_or_path_style_a="$LORA_PATH_STYLE_A" \
  --lora_name_or_path_style_b="$LORA_PATH_STYLE_B" \
  --style_weight_a="$STYLE_WEIGHT_A" \
  --style_weight_b="$STYLE_WEIGHT_B" \
  --output_folder="$OUTPUT_FOLDER" \
  --prompt="$PROMPT" \
  --pattern s*
