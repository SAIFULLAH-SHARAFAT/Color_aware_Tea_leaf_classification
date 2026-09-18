#!/usr/bin/env python3
"""Cheap unit tests for future design components (does not require timm)."""
from pathlib import Path
import importlib.util, torch
P=Path(__file__).parent/"01_shared_color_components.py"
spec=importlib.util.spec_from_file_location("components",P); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
# patch stem
stem=m.HSVPatchStem(4,96,4); assert sum(p.numel() for p in stem.parameters())==6240
# couplings
abc=m.LayerCoupling(96,4,"abc"); assert sum(p.numel() for p in abc.parameters())==576
ac=m.LayerCoupling(96,4,"ac"); assert sum(p.numel() for p in ac.parameters())==192
# mixture is normalized
p=torch.randn(5,7); q=torch.randn(5,7)
for mode in ["decode_branch","normalized"]:
    mix=m.ConfidenceMixture(.5,mode); r,a,_,_=mix(p,q)
    assert torch.allclose(r.sum(-1),torch.ones(5),atol=1e-6); assert (a>=.5).all() and (a<=1).all()
# copy control identity at initialization
copy=m.GatedAttentionCopy(32,1.0); x=torch.randn(2,8,32); assert torch.allclose(copy(x),x)
print("PASS: future component contracts")
