import argparse
from diffusers import DiffusionPipeline
import torch
import os
from utils import insert_sd_klora_to_unet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/home/ubuntu/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b/",
        help="Pretrained model path",
    )
    parser.add_argument(
        "--lora_name_or_path_content",
        type=str,
        help="LoRA path",
        default="loraDataset/content_6/pytorch_lora_weights.safetensors",
    )
    parser.add_argument(
        "--lora_name_or_path_style",
        type=str,
        help="LoRA path",
        default="loraDataset/style_9/pytorch_lora_weights.safetensors",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Output folder path",
        default="output",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt for the image generation",
        default="a sbu cat in szn style",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Pattern for the image generation",
        default="s*",
    )
    return parser.parse_args()


args = parse_args()
pattern = args.pattern
if pattern == "s*":
    alpha = 1.5
    beta = alpha * 0.85
else:
    alpha = 1.5
    beta = 0.5

sum_timesteps = 28000

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) 先用預設精度載入，避免一些權重被鎖成 float 後轉不動
pipe = DiffusionPipeline.from_pretrained(
    args.pretrained_model_name_or_path,
    use_safetensors=True,
)

# 2) 插入 K-LoRA（保持你的原樣）
pipe.unet = insert_sd_klora_to_unet(
    pipe.unet, args.lora_name_or_path_content, args.lora_name_or_path_style, alpha, beta, sum_timesteps, pattern
)

# 3) 明確設定各模組 dtype：UNet/TE -> fp16；VAE -> fp32
pipe.unet.to(dtype=torch.float16)
if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
    pipe.text_encoder.to(dtype=torch.float16)
if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
    pipe.text_encoder_2.to(dtype=torch.float16)

pipe.vae.to(dtype=torch.float32)  # ★ VAE 用 fp32，避免 Half/Float 衝突與黑圖

# 可用就呼叫；沒有也不影響
if hasattr(pipe, "upcast_vae"):
    pipe.upcast_vae()

# 4) 移到裝置；不要再用 dtype=...（我們已經分別設好了）
pipe.to(device)

# 5) 記憶體優化（安全可加可不加）
if hasattr(pipe, "vae"):
    try:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    except Exception:
        pass

# 6) 可靠保險：decode 前把 latents 轉成 fp32（避免某些版本路徑仍丟 half 進來）
_orig_decode = pipe.vae.decode
def _decode_cast_fp32(z, *args, **kwargs):
    if isinstance(z, torch.Tensor) and z.dtype != torch.float32:
        z = z.to(torch.float32)
    return _orig_decode(z, *args, **kwargs)
pipe.vae.decode = _decode_cast_fp32


def run():
    seeds = list(range(40))
    seeds = [see for see in seeds]
    os.makedirs(args.output_folder, exist_ok=True)

    for index, seed in enumerate(seeds):
        generator = torch.Generator(device=device).manual_seed(seed)
        image = pipe(prompt=args.prompt, generator=generator).images[0]
        output_path = os.path.join(args.output_folder, f"output_image_{index}.png")
        image.save(output_path)
        print(output_path)


if __name__ == "__main__":
    run()
