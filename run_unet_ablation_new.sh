
# ----------- USER SETTINGS -----------
MODEL="stabilityai/stable-diffusion-xl-base-1.0"
LORA_CONTENT="./lora-sdxl-dog/pytorch_lora_weights.safetensors"
LORA_STYLE="./lora-sdxl-waterpaintingnew/pytorch_lora_weights.safetensors"
PROMPT="a sbu dog in cba style"
PATTERN="s*"
NUM_IMAGES=10            # keep tiny for speed/OOM safety; raise after验证
DATA_ROOT="data"        # base output folder

CONTENT_FOLDER="datasets/dog"
STYLE_FOLDER="datasets/waterpainting"
TEXT_CONTENT="sbu dog"
TEXT_STYLE="cba style"

# Baseline alphas (s* uses beta = alpha*0.85). These runs use global alpha/beta, no per-block.
BASELINE_ALPHAS=(2 2.5 3 3.5)

# Grids for per-block alpha/beta (down/mid/up). Adjust as needed (broader sweep).
ALPHA_GRID_DOWN=(3 4)
ALPHA_GRID_MID=(3 4)
ALPHA_GRID_UP=(4 5)
BETA_GRID_DOWN=(3)
BETA_GRID_MID=(2)
BETA_GRID_UP=(1.5)

# -------------------------------------
mkdir -p "$DATA_ROOT"
run_id=$(date +%Y%m%d_%H%M%S)
TAG="ablate_${run_id}"
BASE_OUT="$DATA_ROOT/$TAG"
mkdir -p "$BASE_OUT"
LOG="$BASE_OUT/run.log"
exec > >(tee -a "$LOG") 2>&1

echo "Saving runs to $BASE_OUT"

for a_d in "${ALPHA_GRID_DOWN[@]}"; do
  for a_m in "${ALPHA_GRID_MID[@]}"; do
    for a_u in "${ALPHA_GRID_UP[@]}"; do
      for b_d in "${BETA_GRID_DOWN[@]}"; do
        for b_m in "${BETA_GRID_MID[@]}"; do
          for b_u in "${BETA_GRID_UP[@]}"; do
            out_dir="$BASE_OUT/a${a_d}-${a_m}-${a_u}_b${b_d}-${b_m}-${b_u}"
            echo "==> Running $out_dir"
            CUDA_VISIBLE_DEVICES=1 python inference_sd.py \
              --pretrained_model_name_or_path "$MODEL" \
              --lora_name_or_path_content "$LORA_CONTENT" \
              --lora_name_or_path_style "$LORA_STYLE" \
              --prompt "$PROMPT" \
              --pattern "$PATTERN" \
              --output_folder "$out_dir" \
              --num_images "$NUM_IMAGES" \
              --alpha_blocks "{\"down\":$a_d,\"mid\":$a_m,\"up\":$a_u}" \
              --beta_blocks  "{\"down\":$b_d,\"mid\":$b_m,\"up\":$b_u}" || { echo "[WARN] inference failed at $out_dir"; continue; }

            echo "==> Evaluating $out_dir"
            CUDA_VISIBLE_DEVICES=1 python evaluate_clip_scores.py \
              --generated_folder "$out_dir" \
              --content_folder "$CONTENT_FOLDER" \
              --style_folder "$STYLE_FOLDER" \
              --text_content "$TEXT_CONTENT" \
              --text_style "$TEXT_STYLE" \
              --save_json "$out_dir/score.json" || { echo "[WARN] eval failed at $out_dir"; continue; }
          done
        done
      done
    done
  done
done

echo "Done. All results in $BASE_OUT"