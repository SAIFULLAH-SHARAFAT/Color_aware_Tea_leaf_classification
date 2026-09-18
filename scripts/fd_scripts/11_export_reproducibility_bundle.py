#!/usr/bin/env python3
"""Export the revision reproducibility bundle (no images/checkpoints).

Future-design scaffolds are deliberately excluded by default so unexecuted ideas
are not confused with methods reported in the revision. Use
--include-future-designs only for a private archival bundle.
"""
from __future__ import annotations
import argparse, shutil, zipfile
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
OUT=REPO/"results/fd_fresh/reproducibility_bundle"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--include-future-designs",action="store_true")
    args=ap.parse_args()
    if OUT.exists(): shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rels=["configs","src","scripts","docs"]
    if args.include_future_designs: rels.append("future_designs")
    for rel in rels:
        src=REPO/rel
        if src.exists(): shutil.copytree(src,OUT/rel,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    for rel in ["README.md","requirements.txt","data/manifests","data/hf_dataset_lock.json",
                "results/fd_fresh/tables","results/fd_fresh/environment.json"]:
        src=REPO/rel
        if not src.exists(): continue
        dst=OUT/rel; dst.parent.mkdir(parents=True,exist_ok=True)
        if src.is_dir(): shutil.copytree(src,dst)
        else: shutil.copy2(src,dst)
    runs=REPO/"results/fd_fresh/runs"
    if runs.exists():
        for run in runs.iterdir():
            if not run.is_dir(): continue
            for sub in ["config","metrics","logs","raw_outputs"]:
                src=run/sub
                if src.exists(): shutil.copytree(src,OUT/"runs"/run.name/sub)
    z=OUT.with_suffix(".zip")
    if z.exists(): z.unlink()
    with zipfile.ZipFile(z,"w",zipfile.ZIP_DEFLATED) as f:
        for q in OUT.rglob("*"):
            if q.is_file(): f.write(q,q.relative_to(OUT.parent))
    print("wrote",z)

if __name__=="__main__": main()
