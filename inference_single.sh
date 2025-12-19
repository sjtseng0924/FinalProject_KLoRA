python inference_single.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --lora_name_or_path lora-sdxl-oilpainting/pytorch_lora_weights.safetensors \
  --prompt "a cat in jin style image" \
  --output_folder output_oil