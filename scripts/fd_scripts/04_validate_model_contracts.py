#!/usr/bin/env python3
"""Instantiate final architectures before training and verify structural contracts."""
from __future__ import annotations
import sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(REPO/"src"))
import torch
import train_swin_three_models as S
import train_token_models as T

def n(m): return sum(p.numel() for p in m.parameters())
def main():
    # No network access: pretrained=False. This tests architecture, not weights.
    r0=S.SwinRGBHSV(model_name="swin_small_patch4_window7_224.ms_in1k",pretrained=False,use_hsv_branch=False)
    r1=S.SwinRGBHSV(model_name="swin_small_patch4_window7_224.ms_in1k",pretrained=False,use_hsv_branch=True,gate_vector=True)
    c1=T.SwinTCCA(arch="rgb",pretrained=False)
    tla=T.SwinTCCA(arch="tcca",pretrained=False)
    c2=T.SwinTCCA(arch="adapter",pretrained=False,adapter_expansion=3)
    c4=T.SwinTCCA(arch="eca",pretrained=False)
    models={"R0":r0,"R1":r1,"C1":c1,"TLA":tla,"C2":c2,"C4":c4}
    for k,m in models.items(): print(f"{k:4s} total params {n(m):,}")
    # Same RGB backbone and MLP-head capacity across family baselines.
    if n(r0) != n(c1):
        raise SystemExit(f"R0/C1 total parameter mismatch: {n(r0)} vs {n(c1)}")
    D=768; expected=6*D*D+8*D+1
    tla_token=sum(p.numel() for p in tla.tcca.parameters())+sum(p.numel() for p in tla.color_proj4.parameters())
    c2_token=sum(p.numel() for p in c2.adapter.parameters())
    assert tla_token==expected==3545089, (tla_token,expected)
    assert c2_token==expected, (c2_token,expected)
    print("TLA active params:",tla_token)
    print("C2  active params:",c2_token)
    print("PASS: architecture contracts verified.")
if __name__=="__main__": main()
