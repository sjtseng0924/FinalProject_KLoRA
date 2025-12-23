import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPTokenizer, CLIPTextModel

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    use_safetensors=True,
)

device = "cuda"
pipe.to(device)

pipe.unet.to(dtype=torch.float16)

if pipe.text_encoder is not None:
    pipe.text_encoder.to(dtype=torch.float16)
if pipe.text_encoder_2 is not None:
    pipe.text_encoder_2.to(dtype=torch.float16)
pipe.vae.to(dtype=torch.float32)

if hasattr(pipe, "upcast_vae"):
    pipe.upcast_vae()

_orig_decode = pipe.vae.decode
def _decode_cast_fp32(z, *args, **kwargs):
    if isinstance(z, torch.Tensor) and z.dtype != torch.float32:
        z = z.to(torch.float32)
    return _orig_decode(z, *args, **kwargs)

pipe.vae.decode = _decode_cast_fp32
pipe.load_lora_weights(
    "./lora-sdxl-anime-new",
    weight_name="pytorch_lora_weights.safetensors",
    adapter_name="style_anime",
)

pipe.load_lora_weights(
    "./lora-sdxl-waterpaintingnew",
    weight_name="pytorch_lora_weights.safetensors",
    adapter_name="style_waterpainting",
)

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained(
    "openai/clip-vit-large-patch14",
).to(device).to(dtype=torch.float16)

def embed(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        return text_encoder(**inputs).last_hidden_state.mean(dim=1).squeeze(0)

prompt = "a td bear in xyz and cba style"
style_desc = {
    "style_anime": "anime illustration, clean lineart, vivid color",
    "style_waterpainting": "watercolor painting, soft brush strokes, delicate texture",
}
prompt_emb = embed(prompt)
sims = torch.stack([
    F.cosine_similarity(prompt_emb, embed(d), dim=0)
    for d in style_desc.values()
])
weights = torch.softmax(sims, dim=0).tolist()
print("Stylus-style weights:", weights)

pipe.set_adapters(
    adapter_names=list(style_desc.keys()),
    adapter_weights=weights,
)

image = pipe(
    prompt=prompt,
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]

image.save("stylus_twostyle_safe.png")
print("Saved stylus_twostyle_safe.png")
