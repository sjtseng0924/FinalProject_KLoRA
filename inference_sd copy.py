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
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override alpha; if None, use pattern default",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Override beta; if None, use pattern default",
    )
    parser.add_argument(
        "--alpha_blocks",
        type=str,
        default=None,
        help='JSON for per-block alpha, e.g. {"down":4,"mid":4,"up":4}',
    )
    parser.add_argument(
        "--beta_blocks",
        type=str,
        default=None,
        help='JSON for per-block beta, e.g. {"down":3,"mid":3,"up":3}',
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=10,
        help="Number of images to generate",
    )
    return parser.parse_args()


args = parse_args()
pattern = args.pattern
if pattern == "s*":
    alpha = 6
    beta = alpha * 0.85
else:
    alpha = 1.5
    beta = 0.5

# override alpha/beta if specified
if args.alpha is not None:
    alpha = args.alpha
if args.beta is not None:
    beta = args.beta

import json
alpha_blocks = json.loads(args.alpha_blocks) if args.alpha_blocks else None
beta_blocks = json.loads(args.beta_blocks) if args.beta_blocks else None

sum_timesteps = 28000

# --- BEGIN: Add Dtype Setting ---
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = DiffusionPipeline.from_pretrained(
    args.pretrained_model_name_or_path,
    use_safetensors=True,
)
pipe.unet = insert_sd_klora_to_unet(
    pipe.unet,
    args.lora_name_or_path_content,
    args.lora_name_or_path_style,
    alpha,
    beta,
    sum_timesteps,
    pattern,
    alpha_blocks=alpha_blocks,
    beta_blocks=beta_blocks,
)
# UNet/TE:fp16; VAE:fp32;
pipe.unet.to(dtype=torch.float16)
if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
    pipe.text_encoder.to(dtype=torch.float16)
if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
    pipe.text_encoder_2.to(dtype=torch.float16)
pipe.vae.to(dtype=torch.float32)
if hasattr(pipe, "upcast_vae"):
    pipe.upcast_vae()
pipe.to(device)
if hasattr(pipe, "vae"):
    try:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    except Exception:
        pass
_orig_decode = pipe.vae.decode
def _decode_cast_fp32(z, *args, **kwargs):
    if isinstance(z, torch.Tensor) and z.dtype != torch.float32:
        z = z.to(torch.float32)
    return _orig_decode(z, *args, **kwargs)
pipe.vae.decode = _decode_cast_fp32
# --- END: Add Dtype Setting ---

def run():
    seeds = list(range(args.num_images))
    os.makedirs(args.output_folder, exist_ok=True)

    for index, seed in enumerate(seeds):
        generator = torch.Generator(device=device).manual_seed(seed)
        image = pipe(prompt=args.prompt, generator=generator).images[0]
        output_path = os.path.join(args.output_folder, f"output_image_{index}.png")
        image.save(output_path)
        print(output_path)


if __name__ == "__main__":
    run()
