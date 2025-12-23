export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"

# for subject
# export OUTPUT_DIR="lora-sdxl-catnew"
# export INSTANCE_DIR="datasets/cat"
# export PROMPT="ann's cat"
# export VALID_PROMPT="ann's cat sitting on the sofa"

# for style
export OUTPUT_DIR="lora-sdxl-anime-waterpainting"
export INSTANCE_DIR="datasets/anime_waterpainting"
export PROMPT="efg style"
export VALID_PROMPT="a man in efg style"

# Use only free GPUs (0 and 3) to avoid OOM
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

accelerate launch --num_processes=2 train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="${PROMPT}" \
  --rank=8 \
  --resolution=1024 \
  --train_batch_size=1 \
  --learning_rate=5e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --seed="0" \
  --mixed_precision="fp16" \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --use_8bit_adam 