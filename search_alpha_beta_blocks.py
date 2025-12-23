#!/usr/bin/env python
"""
Optuna search for per-block alpha/beta (down/mid/up) on K-LoRA.
Generates a small batch per trial, evaluates CLIP, maximizes a weighted score.
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile

import optuna


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--lora_name_or_path_content", type=str, required=True)
    p.add_argument("--lora_name_or_path_style", type=str, required=True)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--content_folder", type=str, required=True)
    p.add_argument("--style_folder", type=str, required=True)
    p.add_argument("--text_content", type=str, required=True)
    p.add_argument("--text_style", type=str, required=True)
    p.add_argument("--num_images", type=int, default=6, help="Images per trial (use small for speed)")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--lambda_content", type=float, default=0.5, help="Weight for content score in objective")
    p.add_argument("--lambda_style", type=float, default=0.5, help="Weight for style score in objective")
    p.add_argument("--pattern", type=str, default="s*", help="Pattern passed to inference_sd.py")
    return p.parse_args()


def run_inference(args, alpha_blocks, beta_blocks, out_dir):
    cmd = [
        "python",
        "inference_sd.py",
        "--pretrained_model_name_or_path",
        args.pretrained_model_name_or_path,
        "--lora_name_or_path_content",
        args.lora_name_or_path_content,
        "--lora_name_or_path_style",
        args.lora_name_or_path_style,
        "--prompt",
        args.prompt,
        "--output_folder",
        out_dir,
        "--num_images",
        str(args.num_images),
        "--pattern",
        args.pattern,
        "--alpha_blocks",
        json.dumps(alpha_blocks),
        "--beta_blocks",
        json.dumps(beta_blocks),
    ]
    subprocess.run(cmd, check=True)


def run_eval(args, out_dir, score_path):
    cmd = [
        "python",
        "evaluate_clip_scores.py",
        "--generated_folder",
        out_dir,
        "--content_folder",
        args.content_folder,
        "--style_folder",
        args.style_folder,
        "--text_content",
        args.text_content,
        "--text_style",
        args.text_style,
        "--save_json",
        score_path,
    ]
    subprocess.run(cmd, check=True)


def make_objective(args):
    lam_c = args.lambda_content
    lam_s = args.lambda_style

    def objective(trial: optuna.Trial):
        a_down = trial.suggest_float("a_down", 1.0, 6.0)
        a_mid = trial.suggest_float("a_mid", 1.0, 6.0)
        a_up = trial.suggest_float("a_up", 1.0, 6.0)

        b_down = trial.suggest_float("b_down", 0.0, a_down)
        b_mid = trial.suggest_float("b_mid", 0.0, a_mid)
        b_up = trial.suggest_float("b_up", 0.0, a_up)

        alpha_blocks = {"down": a_down, "mid": a_mid, "up": a_up}
        beta_blocks = {"down": b_down, "mid": b_mid, "up": b_up}

        out_dir = tempfile.mkdtemp(prefix="ab_blocks_")
        score_path = os.path.join(out_dir, "clip_scores.json")
        try:
            run_inference(args, alpha_blocks, beta_blocks, out_dir)
            run_eval(args, out_dir, score_path)
            with open(score_path, "r") as f:
                res = json.load(f)
            img_c = res["image_content_similarity"]["mean"]
            img_s = res["image_style_similarity"]["mean"]
            score = lam_c * img_c + lam_s * img_s
            return score
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    return objective


def main():
    args = parse_args()
    objective = make_objective(args)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)
    print("Best params:", study.best_params)
    print("Best score:", study.best_value)


if __name__ == "__main__":
    main()
