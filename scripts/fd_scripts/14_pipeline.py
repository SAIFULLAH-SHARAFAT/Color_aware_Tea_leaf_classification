#!/usr/bin/env python3
"""Convenience orchestrator for the frozen revision pipeline.

Nothing is hidden: this simply invokes the numbered scripts in documented
order. Training stages require explicit flags so a preflight never launches a
multi-hour GPU campaign by accident.
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]

def run(*parts):
    cmd=[sys.executable,*map(str,parts)]; print("\n$"," ".join(cmd)); subprocess.run(cmd,check=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--prepare",action="store_true",help="verify, preflight, environment, model contracts")
    ap.add_argument("--smoke",action="store_true",help="run all models for seed 42")
    ap.add_argument("--remaining",action="store_true",help="run all models for seeds 1337 and 2026")
    ap.add_argument("--analyze",action="store_true",help="aggregate, bootstrap, robustness, figures, diagnostics")
    ap.add_argument("--export",action="store_true")
    a=ap.parse_args()
    if not any(vars(a).values()): raise SystemExit("Choose one or more of --prepare --smoke --remaining --analyze --export")
    S=REPO/"scripts"
    if a.prepare:
        run(S/"01_verify_frozen_dataset.py","--verify-md5")
        run(S/"02_preflight.py")
        run(S/"03_capture_environment.py")
        run(S/"04_validate_model_contracts.py")
    if a.smoke: run(S/"05_run_experiments.py","--all","--seeds","42")
    if a.remaining: run(S/"05_run_experiments.py","--all","--seeds","1337","2026")
    if a.analyze:
        run(S/"06_aggregate_results.py")
        run(S/"10_make_training_settings_table.py")
        run(S/"13_summarize_init_scale.py")
        run(S/"07_paired_bootstrap.py","--auto","--n-boot","10000")
        run(S/"08_eval_robustness.py","--planned")
        run(S/"09_make_figures.py","--format","pdf")
    if a.export: run(S/"11_export_reproducibility_bundle.py")

if __name__=="__main__": main()
