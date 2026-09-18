#!/usr/bin/env python3
"""Aggregate the non-invasive pre-training HSV/RGB scale diagnostics.

This is a mechanistic diagnostic, not an additional trained architecture.
It reads metrics/init_scale_diagnostics.json written before the first optimizer
update by R1/R2/R3.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
RUNS=REPO/"results/fd_fresh/runs"
OUT=REPO/"results/fd_fresh/tables"

def main():
    rows=[]
    for run in sorted(RUNS.glob("R[123]_s*")):
        p=run/"metrics"/"init_scale_diagnostics.json"
        if not p.exists():
            print("[missing]",p); continue
        d=json.load(open(p)); d["run"]=run.name
        d["experiment"]=run.name.rsplit("_s",1)[0]
        rows.append(d)
    if not rows: raise SystemExit("No init-scale diagnostics found. Run fresh HSV experiments first.")
    df=pd.DataFrame(rows)
    OUT.mkdir(parents=True,exist_ok=True)
    df.to_csv(OUT/"init_scale_diagnostics_per_run.csv",index=False)
    cols=["hsv_projected_to_rgb_rms","gate_mean","full_gate_delta_to_rgb_rms",\
          "first_epoch_delta_to_rgb_rms","feature_cosine_mean","full_gate_logit_delta_rms"]
    agg=df.groupby("experiment")[cols].agg(["mean","std"])
    agg.to_csv(OUT/"init_scale_diagnostics_summary.csv")
    print(df[["run"]+cols].to_string(index=False))
    print("\nSUMMARY\n",agg.to_string())
    print("\nwrote",OUT/"init_scale_diagnostics_per_run.csv")
    print("wrote",OUT/"init_scale_diagnostics_summary.csv")

if __name__=="__main__": main()
