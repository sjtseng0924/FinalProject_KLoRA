import argparse
import torch
import os
from diffusers import DiffusionPipeline
from utils import insert_sd_klora_to_unet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--lora_name_or_path",
        type=str,
        required=True,
        help="Single LoRA path (style OR content)",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="output_single_lora",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a cat, sketch style",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="s*",
    )
    return parser.parse_args()


args = parse_args()

# ===== 重要：只啟用一個 LoRA =====
alpha = 1.0
beta = 0.0              # ❗ 第二個 LoRA 完全關掉
sum_timesteps = 28000

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = DiffusionPipeline.from_pretrained(
    args.pretrained_model_name_or_path,
    use_safetensors=True,
)

# ❗ 關鍵技巧：同一個 LoRA path 傳兩次
pipe.unet = insert_sd_klora_to_unet(
    pipe.unet,
    args.lora_name_or_path,   # content slot
    args.lora_name_or_path,   # style slot（但 beta=0）
    alpha,
    beta,
    sum_timesteps,
    args.pattern,
)

# ===== dtype safety（完全沿用你原本的）=====
pipe.unet.to(dtype=torch.float16)

if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
    pipe.text_encoder.to(dtype=torch.float16)
if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
    pipe.text_encoder_2.to(dtype=torch.float16)

pipe.vae.to(dtype=torch.float32)
if hasattr(pipe, "upcast_vae"):
    pipe.upcast_vae()

pipe.to(device)

try:
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
except Exception:
    pass

# 強制 VAE decode fp32
_orig_decode = pipe.vae.decode
def _decode_cast_fp32(z, *args, **kwargs):
    if isinstance(z, torch.Tensor) and z.dtype != torch.float32:
        z = z.to(torch.float32)
    return _orig_decode(z, *args, **kwargs)
pipe.vae.decode = _decode_cast_fp32


def run():
    os.makedirs(args.output_folder, exist_ok=True)

    for seed in range(40):
        generator = torch.Generator(device=device).manual_seed(seed)
        image = pipe(
            prompt=args.prompt,
            generator=generator,
        ).images[0]

        out = os.path.join(args.output_folder, f"output_{seed}.png")
        image.save(out)
        print(out)


if __name__ == "__main__":
    run()
