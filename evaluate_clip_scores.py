import argparse
import os
import math
from typing import List, Tuple, Dict
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def load_images(folder: str) -> List[Image.Image]:
    exts = {'.png', '.jpg', '.jpeg', '.webp'}
    images = []
    for name in sorted(os.listdir(folder)):
        if os.path.splitext(name)[1].lower() in exts:
            try:
                images.append(Image.open(os.path.join(folder, name)).convert('RGB'))
            except Exception:
                pass
    return images


def compute_image_embeddings(model: CLIPModel, processor: CLIPProcessor, images: List[Image.Image], device: str) -> torch.Tensor:
    if not images:
        return torch.empty(0, model.visual_projection.out_features, device=device)
    batch_embeddings = []
    batch_size = 8
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            image_outputs = model.get_image_features(**inputs)
            image_outputs = image_outputs / image_outputs.norm(p=2, dim=-1, keepdim=True)
        batch_embeddings.append(image_outputs)
    return torch.cat(batch_embeddings, dim=0)


def compute_text_embeddings(model: CLIPModel, processor: CLIPProcessor, texts: List[str], device: str) -> torch.Tensor:
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_outputs = model.get_text_features(**inputs)
        text_outputs = text_outputs / text_outputs.norm(p=2, dim=-1, keepdim=True)
    return text_outputs


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a @ b.T)


def aggregate_similarity(generated_emb: torch.Tensor, ref_emb: torch.Tensor) -> Dict[str, float]:
    if ref_emb.numel() == 0 or generated_emb.numel() == 0:
        return {"mean": math.nan, "median": math.nan, "max": math.nan}
    sims = cosine_similarity(generated_emb, ref_emb)  # [G, R]
    # For each generated image, average similarity over references
    per_image = sims.mean(dim=1)
    return {
        "mean": per_image.mean().item(),
        "median": per_image.median().item(),
        "max": per_image.max().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP content/style similarity.")
    parser.add_argument("--generated_folder", type=str, default="outputs")
    parser.add_argument("--content_folder", type=str, default="datasets/dog")
    parser.add_argument("--style_folder", type=str, default="datasets/waterpainting")
    parser.add_argument("--text_content", type=str, default="sbu dog")
    parser.add_argument("--text_style", type=str, default="waterpainting style")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_json", type=str, default="clip_scores.json")
    args = parser.parse_args()

    device = args.device
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)

    gen_images = load_images(args.generated_folder)
    content_images = load_images(args.content_folder)
    style_images = load_images(args.style_folder)

    gen_emb = compute_image_embeddings(model, processor, gen_images, device)
    content_emb = compute_image_embeddings(model, processor, content_images, device)
    style_emb = compute_image_embeddings(model, processor, style_images, device)

    # Image-image similarities
    content_img_metrics = aggregate_similarity(gen_emb, content_emb)
    style_img_metrics = aggregate_similarity(gen_emb, style_emb)

    # Text similarities
    text_emb = compute_text_embeddings(model, processor, [args.text_content, args.text_style], device)
    content_text_emb, style_text_emb = text_emb[0:1], text_emb[1:2]
    # Generated images vs content/style text
    gen_vs_content_text = cosine_similarity(gen_emb, content_text_emb).squeeze(-1)
    gen_vs_style_text = cosine_similarity(gen_emb, style_text_emb).squeeze(-1)

    content_text_metrics = {
        "mean": gen_vs_content_text.mean().item(),
        "median": gen_vs_content_text.median().item(),
        "max": gen_vs_content_text.max().item(),
    }
    style_text_metrics = {
        "mean": gen_vs_style_text.mean().item(),
        "median": gen_vs_style_text.median().item(),
        "max": gen_vs_style_text.max().item(),
    }

    results = {
        "image_content_similarity": content_img_metrics,
        "image_style_similarity": style_img_metrics,
        "text_content_similarity": content_text_metrics,
        "text_style_similarity": style_text_metrics,
        "counts": {
            "generated": len(gen_images),
            "content_refs": len(content_images),
            "style_refs": len(style_images),
        },
    }

    import json
    with open(args.save_json, "w") as f:
        json.dump(results, f, indent=2)

    print("=== CLIP Similarity Results ===")
    for k, v in results.items():
        if isinstance(v, dict):
            print(k, v)


if __name__ == "__main__":
    main()
