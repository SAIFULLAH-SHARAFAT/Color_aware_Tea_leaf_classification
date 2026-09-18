#!/usr/bin/env python3
"""Reusable components for a future shared-weight chromatic Swin study.

These components are runnable and unit-testable independently of timm internals.
The block-level shared-KV Swin adapter is intentionally separate.
"""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F

class HSVPatchStem(nn.Module):
    """Separate color patch embedding; all subsequent large weights can be shared."""
    def __init__(self,in_chans=4,embed_dim=96,patch_size=4,bias=True):
        super().__init__()
        self.proj=nn.Conv2d(in_chans,embed_dim,kernel_size=patch_size,stride=patch_size,bias=bias)
    def forward(self,x): return self.proj(x)

class LayerCoupling(nn.Module):
    """Decode-Branch-inspired elementwise couplings for one Transformer block."""
    def __init__(self,dim:int,mlp_ratio:int=4,mode:str="abc"):
        super().__init__(); assert mode in {"abc","ac"}; self.mode=mode
        self.a=nn.Parameter(torch.zeros(dim))
        self.c=nn.Parameter(torch.zeros(dim))
        self.b=nn.Parameter(torch.zeros(dim*mlp_ratio)) if mode=="abc" else None
    def mix_query(self,q2,q1): return q2 + self.a * q1
    def mix_hidden(self,h2,h1): return h2 if self.b is None else h2 + self.b * h1
    def mix_mlp(self,m2,m1): return m2 + self.c * m1

class ConfidenceMixture(nn.Module):
    """Primary/color probability mixture with collision-probability confidence.

    mode='decode_branch': alpha=clip(sum p^2, min_alpha, 1), matching the
    source idea.

    mode='normalized': maps the K-class uniform collision 1/K to min_alpha and
    a degenerate distribution to 1 before clipping. This is a proposed
    multiclass adaptation, not a claim from Decode-Branch.
    """
    def __init__(self,min_alpha:float=.5,mode:str="decode_branch",detach_alpha:bool=False):
        super().__init__(); assert 0<=min_alpha<=1; assert mode in {"decode_branch","normalized"}
        self.min_alpha=min_alpha; self.mode=mode; self.detach_alpha=detach_alpha
    def alpha(self,p):
        coll=p.square().sum(dim=-1,keepdim=True)
        if self.mode=="normalized":
            k=p.shape[-1]; coll=(coll-1.0/k)/(1.0-1.0/k); coll=coll.clamp(0,1)
            a=self.min_alpha+(1-self.min_alpha)*coll
        else:
            a=coll.clamp(self.min_alpha,1.0)
        return a.detach() if self.detach_alpha else a
    def forward(self,primary_logits,color_logits):
        p=F.softmax(primary_logits,dim=-1); q=F.softmax(color_logits,dim=-1); a=self.alpha(p)
        mix=a*p+(1-a)*q
        return mix,a,p,q
    def nll(self,primary_logits,color_logits,target):
        mix,a,p,q=self(primary_logits,color_logits)
        loss=-torch.log(mix.gather(1,target[:,None]).clamp_min(1e-12)).mean()
        return loss,{"alpha_mean":float(a.detach().mean()),"alpha_min":float(a.detach().min()),"alpha_max":float(a.detach().max())}

class GatedAttentionCopy(nn.Module):
    """Strict-compute-control primitive: branch receives g_A * primary attention."""
    def __init__(self,dim:int,init:float=1.0): super().__init__(); self.g=nn.Parameter(torch.full((dim,),float(init)))
    def forward(self,primary_attention_output): return self.g * primary_attention_output

@torch.no_grad()
def rms_match_linear_(linear:nn.Linear,input_tensor:torch.Tensor,target_rms:float):
    """Data-driven future initializer: rescale a Linear weight to a target output RMS.

    Use only on a declared training calibration batch before optimization. Do not
    tune this using validation/test performance.
    """
    y=linear(input_tensor.float()); cur=y.square().mean().sqrt().item()
    if cur<=0: raise ValueError("current RMS is zero")
    scale=float(target_rms)/cur; linear.weight.mul_(scale)
    if linear.bias is not None: linear.bias.mul_(scale)
    return {"before_rms":cur,"target_rms":float(target_rms),"weight_scale":scale}
