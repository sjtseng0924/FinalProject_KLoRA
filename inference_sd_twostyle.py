import argparse
import torch
import os
from diffusers import DiffusionPipeline
from safetensors.torch import load_file, save_file
from utils import insert_sd_klora_to_unet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--lora_name_or_path_content",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--lora_name_or_path_style_a",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--lora_name_or_path_style_b",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--style_weight_a",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--style_weight_b",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="output",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a sbu dog in mixed style",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="s*",
    )
    return parser.parse_args()

def merge_style_loras(style_paths, weights, output_path):
    assert len(style_paths) == len(weights)
    merged = {}
    for path, w in zip(style_paths, weights):
        sd = load_file(path)
        for k, v in sd.items():
            if k not in merged:
                merged[k] = w * v
            else:
                merged[k] += w * v
    save_file(merged, output_path)
    return output_path

args = parse_args()

# K-LoRA alpha / beta
if args.pattern == "s*":
    alpha = 9.0
    beta = alpha * 0.85
else:
    alpha = 1.5
    beta = 0.5

sum_timesteps = 28000

device = "cuda" if torch.cuda.is_available() else "cpu"

merged_style_path = "merged_style_lora.safetensors"
merge_style_loras(
    style_paths=[
        args.lora_name_or_path_style_a,
        args.lora_name_or_path_style_b,
    ],
    weights=[
        args.style_weight_a,
        args.style_weight_b,
    ],
    output_path=merged_style_path,
)

pipe = DiffusionPipeline.from_pretrained(
    args.pretrained_model_name_or_path,
    use_safetensors=True,
)

pipe.unet = insert_sd_klora_to_unet(
    pipe.unet,
    args.lora_name_or_path_content,
    merged_style_path,
    alpha,
    beta,
    sum_timesteps,
    args.pattern,
)

pipe.unet.to(dtype=torch.float16)

if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
    pipe.text_encoder.to(dtype=torch.float16)

if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
    pipe.text_encoder_2.to(dtype=torch.float16)

pipe.vae.to(dtype=torch.float32)

if hasattr(pipe, "upcast_vae"):
    pipe.upcast_vae()

pipe.to(device)

# VAE memory safe
if hasattr(pipe, "vae"):
    try:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    except Exception:
        pass

# Force fp32 decode
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

        out_path = os.path.join(
            args.output_folder,
            f"seed_{seed}.png",
        )
        image.save(out_path)
        print(out_path)

if __name__ == "__main__":
    run()
