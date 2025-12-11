export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export LORA_PATH_CONTENT="./lora-sdxl-cat/pytorch_lora_weights.safetensors"
export LORA_PATH_STYLE="./lora-sdxl-waterpainting/pytorch_lora_weights.safetensors"
export OUTPUT_FOLDER="./outputs_cat"  
export PROMPT="a cat in watercolor style "

python inference_sd.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --lora_name_or_path_content="$LORA_PATH_CONTENT" \
  --lora_name_or_path_style="$LORA_PATH_STYLE" \
  --output_folder="$OUTPUT_FOLDER" \
  --prompt="$PROMPT"

# using gradio 
# python inference_gradio.py \
#   --pretrained_model_name_or_path="$MODEL_NAME" \
#   --lora_name_or_path_content="$LORA_PATH_CONTENT" \
#   --lora_name_or_path_style="$LORA_PATH_STYLE" \
#   --output_folder="$OUTPUT_FOLDER" \
#   --prompt="$PROMPT"
