#!/usr/bin/env python3
"""Paired error-overlap analysis for reviewer-facing interpretation."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest
REPO=Path(__file__).resolve().parents[1]
def load(run):
    raw=Path(run)/"raw_outputs"; ids=json.load(open(raw/"test_ids.json")); y=np.load(raw/"test_targets.npy"); p=np.load(raw/"test_predictions.npy")
    return np.asarray(ids,dtype=object),y,p
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--a",required=True); ap.add_argument("--b",required=True); ap.add_argument("--out",required=True); args=ap.parse_args()
    ia,ya,pa=load(args.a); ib,yb,pb=load(args.b); pos={x:i for i,x in enumerate(ib.tolist())}; order=np.array([pos[x] for x in ia.tolist()]); yb,pb=yb[order],pb[order]
    assert np.array_equal(ya,yb)
    ca=pa==ya; cb=pb==ya; a_only=int((ca & ~cb).sum()); b_only=int((~ca & cb).sum()); both=int((ca & cb).sum()); neither=int((~ca & ~cb).sum())
    n=a_only+b_only; pval=float(binomtest(min(a_only,b_only),n,0.5).pvalue) if n else 1.0
    out={"a":Path(args.a).name,"b":Path(args.b).name,"both_correct":both,"a_only_correct":a_only,"b_only_correct":b_only,"both_wrong":neither,"mcnemar_exact_two_sided_p":pval,"n_test":len(ya)}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=="__main__": main()
