#!/usr/bin/env python3
"""Cheap preflight before launching any GPU training."""
from __future__ import annotations
import importlib, json, subprocess, sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]

def main():
    needed=["torch","torchvision","timm","numpy","pandas","sklearn","yaml","PIL","tqdm"]
    failed=[]
    for m in needed:
        try:
            mod=importlib.import_module(m); print(f"[OK] {m}: {getattr(mod,'__version__','installed')}")
        except Exception as e:
            failed.append((m,str(e))); print(f"[FAIL] {m}: {e}")
    if failed: raise SystemExit("Missing dependencies: "+str(failed))
    subprocess.run([sys.executable,str(REPO/"scripts/01_verify_frozen_dataset.py")],check=True)
    subprocess.run([sys.executable,str(REPO/"scripts/05_run_experiments.py"),"--list"],check=True)
    subprocess.run([sys.executable,str(REPO/"scripts/05_run_experiments.py"),"--all","--dry-run"],check=True)
    print("PASS: preflight complete. No training was started.")
if __name__=="__main__": main()
