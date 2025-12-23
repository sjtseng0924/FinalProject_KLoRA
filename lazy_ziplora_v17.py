import torch
from safetensors.torch import load_file, save_file
import argparse
import os

def lazy_ziplora_v17(content_path, style_path, output_path, retention=0.75, fg_alpha=0.55, bg_boost=2.0):
    """
    Lazy ZipLoRA v17: "Energy-Preserving Blending" (Nlerp)
    
    Diagnosis of "Disappearing Dog":
    - Linear Interpolation ((1-a)C + aS) of orthogonal vectors results in a vector with REDUCED magnitude.
    - If C and S are orthogonal, 0.5C + 0.5S has length 0.707 (30% energy loss).
    - This "Signal Drop" causes the dog to become faint or ghost-like ("disappearing").
    
    Solution: Norm-Aware Linear Interpolation (Nlerp)
    - We blend C and S.
    - Then we RESCALE the result to match the original Content Energy.
    - W_new = Normalize(Blend(C, S)) * Norm(C)
    
    This allows us to use HIGHER Alpha (more style direction) without losing the "Solidity" (Energy) of the dog.
    """
    print(f"Loading Content: {content_path}")
    sd_content = load_file(content_path)
    print(f"Loading Style: {style_path}")
    sd_style = load_file(style_path)
    
    sd_new = {}
    print(f"Starting V17 Merge (Energy-Preserving Nlerp)...")
    print(f"  - Retention: {retention*100}%")
    print(f"  - Fg Alpha:  {fg_alpha} (Renormalized)")
    print(f"  - Bg Boost:  {bg_boost}")
    
    for key in sd_content.keys():
        if key not in sd_style: 
            sd_new[key] = sd_content[key]
            continue
            
        wc = sd_content[key].float()
        ws = sd_style[key].float()
        
        if wc.shape != ws.shape:
            sd_new[key] = wc
            continue

        if len(wc.shape) == 2:
            dim = 0
        elif len(wc.shape) == 4:
            dim = (0, 2, 3)
        else:
            sd_new[key] = wc
            continue
            
        # 1. Protection Mask
        norm_c = torch.linalg.norm(wc, ord=2, dim=dim, keepdim=True)
        norm_s = torch.linalg.norm(ws, ord=2, dim=dim, keepdim=True) + 1e-8
        
        flat_norm_c = norm_c.flatten()
        k = int(len(flat_norm_c) * retention)
        
        if k == 0:
            mask_fg = torch.zeros_like(norm_c, dtype=torch.bool)
        elif k == len(flat_norm_c):
            mask_fg = torch.ones_like(norm_c, dtype=torch.bool)
        else:
            top_k_val = torch.topk(flat_norm_c, k).values[-1]
            mask_fg = norm_c >= top_k_val
            
        mask_bg = ~mask_fg
        
        # 2. Background: Aggressive Replacement (Standard)
        target_norm_bg = torch.sqrt(norm_c * norm_s) * bg_boost
        ws_bg_scaled = ws * (target_norm_bg / norm_s)
        
        # 3. Foreground: Energy-Preserving Blending (Nlerp)
        # Standard Linear Blend first
        # We perform weighted addition: C + alpha * S_scaled
        # Wait, if we use (1-a)C + aS, it's interpolation.
        # Let's use Injection style: C + alpha * S (norm adjusted)
        
        # Option A: Slerp-like approximation
        # W_blend = C + (alpha * S * (Norm_C/Norm_S))
        ws_norm_to_c = ws * (norm_c / norm_s)
        w_blend_raw = wc + (fg_alpha * ws_norm_to_c)
        
        # Energy Restoration Step:
        # Calculate norm of the blended vector
        norm_blend = torch.linalg.norm(w_blend_raw, ord=2, dim=dim, keepdim=True) + 1e-8
        # Rescale to match Original Content Norm
        w_fg_restored = w_blend_raw * (norm_c / norm_blend)
        
        # 4. Merge
        w_new = (mask_fg.float() * w_fg_restored) + (mask_bg.float() * ws_bg_scaled)

        sd_new[key] = w_new.to(dtype=torch.float16)
        
        alpha_key = key.replace(".weight", ".alpha")
        if alpha_key in sd_content:
            sd_new[alpha_key] = sd_content[alpha_key]

    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    print(f"Saving V17 LoRA to: {output_path}")
    save_file(sd_new, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lazy ZipLoRA v17: Energy Preserved")
    parser.add_argument("--content", type=str, required=True)
    parser.add_argument("--style", type=str, required=True)
    parser.add_argument("--output", type=str, default="ziplora_v17.safetensors")
    parser.add_argument("--retention", type=float, default=0.75, help="Retention")
    parser.add_argument("--fg_alpha", type=float, default=0.55, help="Foreground Alpha")
    parser.add_argument("--bg_boost", type=float, default=2.0, help="Background Boost")
    
    args = parser.parse_args()
    
    lazy_ziplora_v17(args.content, args.style, args.output, args.retention, args.fg_alpha, args.bg_boost)
