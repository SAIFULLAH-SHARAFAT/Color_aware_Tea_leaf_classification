#!/usr/bin/env python3
"""Standalone primitive for the future strictly compute-matched attention-copy control."""
import torch
from torch import nn
class GatedAttentionCopy(nn.Module):
    def __init__(self,dim:int,init:float=1.0):
        super().__init__(); self.g=nn.Parameter(torch.full((dim,),float(init)))
    def forward(self,primary_attention_output):
        return self.g * primary_attention_output

if __name__=="__main__":
    m=GatedAttentionCopy(768); x=torch.randn(2,49,768); y=m(x)
    print("params",sum(p.numel() for p in m.parameters()),"shape",tuple(y.shape),"max init diff",(y-x).abs().max().item())
