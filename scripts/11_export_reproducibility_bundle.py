#!/usr/bin/env python3
"""Export a compact public reproducibility bundle (no model weights or image data)."""
from __future__ import annotations
import shutil, zipfile
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
OUT=REPO/"results/fresh/reproducibility_bundle"
def main():
    if OUT.exists(): shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for rel in ["configs","src","scripts","docs"]:
        src=REPO/rel
        if src.exists(): shutil.copytree(src,OUT/rel,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    for rel in ["data/manifests","data/hf_dataset_lock.json","results/fresh/tables","results/fresh/environment.json"]:
        src=REPO/rel
        if not src.exists(): continue
        dst=OUT/rel
        dst.parent.mkdir(parents=True,exist_ok=True)
        if src.is_dir(): shutil.copytree(src,dst)
        else: shutil.copy2(src,dst)
    # Per-run configs/metrics/raw IDs+predictions only; omit checkpoints.
    runs=REPO/"results/fresh/runs"
    if runs.exists():
        for run in runs.iterdir():
            if not run.is_dir(): continue
            for sub in ["config","metrics","logs","raw_outputs"]:
                src=run/sub
                if src.exists(): shutil.copytree(src,OUT/"runs"/run.name/sub)
    z=OUT.with_suffix(".zip")
    if z.exists(): z.unlink()
    with zipfile.ZipFile(z,"w",zipfile.ZIP_DEFLATED) as f:
        for p in OUT.rglob("*"):
            if p.is_file(): f.write(p,p.relative_to(OUT.parent))
    print("wrote",z)
if __name__=="__main__": main()
