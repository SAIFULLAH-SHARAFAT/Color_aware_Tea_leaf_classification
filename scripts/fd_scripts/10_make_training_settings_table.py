#!/usr/bin/env python3
"""Generate the consolidated training-settings table requested by Reviewer #1."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "results" / "fd_fresh" / "runs"
OUT = REPO / "results" / "fd_fresh" / "tables" / "training_settings.tex"


def load_cfg(run: str) -> dict:
    return json.load(open(RUNS / run / "config" / "config.json"))


def val(c: dict, key: str, default="--"):
    return c.get(key, default)


def main() -> None:
    hsv = load_cfg("R0_s42")
    token = load_cfg("C1_s42")
    rows = [
        ("Backbone", hsv.get("model_name", "Swin-S"), "Swin-S"),
        ("Input size", val(hsv, "input_size", 224), val(token, "img_size", 224)),
        ("Epoch budget", val(hsv, "epochs"), val(token, "epochs")),
        ("Physical batch size", val(hsv, "batch_size"), val(token, "batch_size")),
        ("Gradient accumulation", val(hsv, "gradient_accumulation_steps", 1), val(token, "gradient_accumulation_steps", 1)),
        ("Effective batch size", val(hsv, "batch_size") * val(hsv, "gradient_accumulation_steps", 1), val(token, "batch_size") * val(token, "gradient_accumulation_steps", 1)),
        ("Optimizer", "AdamW", "AdamW"),
        ("Initial learning rate", val(hsv, "lr"), val(token, "lr")),
        ("Weight decay", val(hsv, "weight_decay", 0.05), val(token, "weight_decay", 0.05)),
        ("LR warm-up (epochs)", val(hsv, "warmup_epochs", 5), val(token, "warmup_epochs", 5)),
        ("Minimum LR", val(hsv, "min_lr", 1e-6), val(token, "min_lr", 1e-6)),
        ("DropPath", val(hsv, "drop_path_rate"), val(token, "drop_path_rate")),
        ("Label smoothing", val(hsv, "label_smoothing", 0.1), val(token, "label_smoothing", 0.1)),
        ("EMA decay", val(hsv, "ema_decay", 0.9998), val(token, "ema_decay", 0.9998)),
        ("Gradient clipping", val(hsv, "grad_clip_norm", 1.0), val(token, "grad_clip_norm", 1.0)),
        ("Gate warm-up (epochs)", val(hsv, "gate_warmup_epochs", 5), val(token, "gate_warmup_epochs", 10)),
        ("Hue jitter", val(hsv, "hue_jitter", 0.0), val(token, "hue_jitter", 0.0)),
        ("Checkpoint criterion", "validation Macro-F1", "validation Macro-F1"),
        ("Early-stopping patience", val(hsv, "early_stopping_patience", 25), val(token, "early_stopping_patience", 25)),
        ("Seeds", "42, 1337, 2026", "42, 1337, 2026"),
    ]

    lines = [
        "% generated from completed run configs -- do not edit by hand",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Consolidated training settings for the complementary HSV and primary token-attention experiment families.}",
        r"\label{tab:training_settings}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"\textbf{Setting} & \textbf{HSV family (R0--R3)} & \textbf{Token family (C1/TLA/C2/C4)} " + r"\\",
        r"\midrule",
    ]
    for key, a, b in rows:
        lines.append(f"{key} & {a} & {b} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
