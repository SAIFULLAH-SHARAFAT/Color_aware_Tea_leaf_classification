#!/usr/bin/env python3
"""
Controlled perturbation robustness for the fresh revision.

18 corrupted conditions + one unperturbed condition.
All models are evaluated on identical deterministic corruptions.

The unperturbed preprocessing is EXACTLY aligned to training evaluation:
Resize(int(224*1.14)=255, BICUBIC) -> CenterCrop(224).
"""
from __future__ import annotations
import argparse, hashlib, io, json, sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

REPO = Path(__file__).resolve().parents[1]
IMAGENET_MEAN=(0.485,0.456,0.406)
IMAGENET_STD=(0.229,0.224,0.225)
IMAGE_EXTS=(".jpg",".jpeg",".png",".bmp",".webp")
LABELS=np.arange(7)

def _np(img): return np.asarray(img,dtype=np.float32)/255.0
def _pil(a): return Image.fromarray((np.clip(a,0,1)*255).astype(np.uint8))

def p_identity():
    return lambda im,key: im
def p_brightness(delta):
    return lambda im,key: _pil(_np(im)+delta)
def p_contrast(factor):
    def f(im,key):
        a=_np(im); mu=a.mean()
        return _pil((a-mu)*factor+mu)
    return f
def p_saturation(factor):
    def f(im,key):
        a=_np(im); gray=a@np.array([.299,.587,.114],np.float32)
        return _pil(gray[...,None]+(a-gray[...,None])*factor)
    return f
def p_hue(shift):
    def f(im,key):
        hsv=np.asarray(im.convert("HSV"),dtype=np.float32)
        hsv[...,0]=(hsv[...,0]+shift*256.0)%256.0
        return Image.fromarray(hsv.astype(np.uint8),mode="HSV").convert("RGB")
    return f
def _stable_seed(key):
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],"little")%(2**32)
def p_noise(sigma, tag):
    def f(im,key):
        rng=np.random.default_rng(_stable_seed(f"{tag}|{key}"))
        a=_np(im)
        return _pil(a+rng.normal(0,sigma,a.shape).astype(np.float32))
    return f
def p_blur(radius):
    return lambda im,key: im.filter(ImageFilter.GaussianBlur(radius=radius))
def p_jpeg(q):
    def f(im,key):
        b=io.BytesIO(); im.save(b,format="JPEG",quality=q); b.seek(0)
        return Image.open(b).convert("RGB")
    return f
def p_shadow(side,strength=.35):
    def f(im,key):
        a=_np(im); w=a.shape[1]
        ramp=np.linspace(0,1,w,dtype=np.float32)
        if side=="left": ramp=ramp[::-1]
        mask=(1-strength*ramp)[None,:,None]
        return _pil(a*mask)
    return f
def p_wb(gains):
    return lambda im,key: _pil(_np(im)*np.asarray(gains,np.float32)[None,None,:])

PERTURBATIONS: Dict[str,Tuple[str,Callable]]={
 "unperturbed":("unperturbed",p_identity()),
 "brightness_up":("brightness/contrast",p_brightness(+.20)),
 "brightness_down":("brightness/contrast",p_brightness(-.20)),
 "contrast_low":("brightness/contrast",p_contrast(.75)),
 "contrast_high":("brightness/contrast",p_contrast(1.25)),
 "saturation_low":("saturation",p_saturation(.50)),
 "saturation_high":("saturation",p_saturation(1.50)),
 "hue_pos":("hue shift",p_hue(+.05)),
 "hue_neg":("hue shift",p_hue(-.05)),
 "noise_light":("gaussian noise",p_noise(.03,"noise_light")),
 "noise_heavy":("gaussian noise",p_noise(.08,"noise_heavy")),
 "blur_light":("blur",p_blur(1.0)),
 "blur_heavy":("blur",p_blur(2.5)),
 "jpeg_q60":("jpeg compression",p_jpeg(60)),
 "jpeg_q30":("jpeg compression",p_jpeg(30)),
 "shadow_left":("shadow",p_shadow("left")),
 "shadow_right":("shadow",p_shadow("right")),
 "wb_warm":("white balance",p_wb((1.12,1.03,.90))),
 "wb_cool":("white balance",p_wb((.90,1.02,1.12))),
}
assert len(PERTURBATIONS)-1 == 18

class PerturbedTestSet(Dataset):
    def __init__(self,root:Path,perturb:Callable,img_size=224):
        self.root=root; self.perturb=perturb
        classes=sorted(p.name for p in root.iterdir() if p.is_dir())
        self.samples=[]
        for ci,c in enumerate(classes):
            for fp in sorted((root/c).iterdir()):
                if fp.suffix.lower() in IMAGE_EXTS:
                    self.samples.append((fp,ci))
        self.resize=transforms.Compose([
            transforms.Resize(int(img_size*1.14), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
        ])
        self.tensor=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD),
        ])
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        fp,y=self.samples[i]
        with Image.open(fp) as im:
            im=self.resize(im.convert("RGB"))
            im=self.perturb(im,str(fp.relative_to(self.root)))
            return self.tensor(im),y

def load_model(run_dir:Path,device:str):
    sys.path.insert(0,str(REPO/"src"))
    cfg=json.load(open(run_dir/"config"/"config.json"))
    if "arch" in cfg:
        import train_token_models as T
        model=T.SwinTCCA(
            pretrained=False,
            arch=cfg["arch"],
            drop_path_rate=cfg.get("drop_path_rate",.2),
            num_classes=cfg.get("num_classes",7),
            use_hsv=cfg.get("use_hsv",False),
            hsv_use_sincos=cfg.get("hsv_use_sincos",True),
            use_seg=False,
            tcca_num_heads=cfg.get("tcca_num_heads",8),
            adapter_expansion=cfg.get("adapter_expansion",3),
        )
    else:
        import train_swin_three_models as S
        model=S.SwinRGBHSV(
            model_name=cfg.get("model_name","swin_small_patch4_window7_224"),
            pretrained=False,
            drop_path_rate=cfg.get("drop_path_rate",.2),
            num_classes=cfg.get("num_classes",7),
            use_hsv_branch=cfg.get("use_hsv_branch",False),
            hsv_embed_dim=cfg.get("hsv_embed_dim",128),
            hsv_use_sincos=cfg.get("hsv_use_sincos",True),
            hsv_dropout=cfg.get("hsv_dropout",.1),
            fuse_dropout=cfg.get("fuse_dropout",.2),
            gate_hidden=cfg.get("gate_hidden",256),
            gate_vector=cfg.get("gate_vector",False),
            img_mean=tuple(cfg.get("img_mean",IMAGENET_MEAN)),
            img_std=tuple(cfg.get("img_std",IMAGENET_STD)),
        )
    mdir=run_dir/"model"
    for name in ("ema_model_weights_only.pth","model_weights_only.pth"):
        p=mdir/name
        if p.exists():
            sd=torch.load(p,map_location="cpu")
            sd={k.replace("module.",""):v for k,v in sd.items()}
            model.load_state_dict(sd,strict=True)
            print("   loaded",name,"strict=True")
            break
    else:
        raise SystemExit(f"No weights-only checkpoint found in {mdir}")
    return model.eval().to(device)

@torch.no_grad()
def evaluate(model,loader,device):
    ys=[]; ps=[]
    for x,y in loader:
        x=x.to(device,non_blocking=True)
        with torch.autocast(device_type="cuda",enabled=(device=="cuda")):
            logits=model(x)
        ps.append(logits.float().argmax(1).cpu().numpy()); ys.append(y.numpy())
    y=np.concatenate(ys); p=np.concatenate(ps)
    return float(f1_score(y,p,labels=LABELS,average="macro",zero_division=0))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--runs-dir",default=str(REPO/"results"/"fd_fresh"/"runs"))
    ap.add_argument("--data-root",default=str(REPO/"data"/"Tea_leaf_dataset"))
    ap.add_argument("--runs",nargs="+")
    ap.add_argument("--all",action="store_true")
    ap.add_argument("--planned",action="store_true",help="Use robustness_experiments and robustness_seeds from YAML")
    ap.add_argument("--seeds",type=int,nargs="+",default=None)
    ap.add_argument("--batch-size",type=int,default=64)
    ap.add_argument("--num-workers",type=int,default=2)
    ap.add_argument("--out",default=str(REPO/"results"/"fd_fresh"/"tables"/"robustness.csv"))
    args=ap.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    runs_dir=Path(args.runs_dir); test_root=Path(args.data_root)/"test"
    if args.planned:
        import yaml
        cfg=yaml.safe_load(open(REPO/"configs/experiments.yaml"))
        exps=cfg["defaults"]["robustness_experiments"]
        seeds=args.seeds or cfg["defaults"]["robustness_seeds"]
        targets=[runs_dir/f"{e}_s{s}" for e in exps for s in seeds]
    elif args.runs:
        targets=[runs_dir/r for r in args.runs]
    elif args.all:
        seeds=args.seeds or [42,1337,2026]
        targets=[d for d in sorted(runs_dir.iterdir())
                 if (d/"config"/"config.json").exists()
                 and any(d.name.endswith(f"_s{s}") for s in seeds)]
    else:
        raise SystemExit("choose --runs, --planned, or --all")
    missing=[str(t) for t in targets if not (t/"config"/"config.json").exists()]
    if missing: raise SystemExit("Missing planned run(s): "+", ".join(missing[:8]))

    rows=[]
    for run in targets:
        print("\n"+"="*78+"\n"+run.name+"\n"+"="*78)
        model=load_model(run,device)
        for name,(group,fn) in PERTURBATIONS.items():
            dl=DataLoader(PerturbedTestSet(test_root,fn),
                          batch_size=args.batch_size,shuffle=False,
                          num_workers=args.num_workers,pin_memory=True)
            f=evaluate(model,dl,device)
            rows.append({"run":run.name,"experiment":run.name.rsplit("_s",1)[0],
                         "seed":run.name.rsplit("_s",1)[-1],
                         "perturbation":name,"group":group,"macro_f1":f})
            print(f"   {name:<18} {f:.4f}")
        del model
        if device=="cuda": torch.cuda.empty_cache()
    df=pd.DataFrame(rows); Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(args.out,index=False)
    summaries=[]
    for run,g in df.groupby("run"):
        clean=float(g.loc[g.perturbation=="unperturbed","macro_f1"].iloc[0])
        stress=g[g.perturbation!="unperturbed"]["macro_f1"]
        summaries.append({"run":run,"experiment":run.rsplit("_s",1)[0],
                          "seed":run.rsplit("_s",1)[-1],"unperturbed":clean,
                          "mean_stress":float(stress.mean()),
                          "worst_stress":float(stress.min()),
                          "mean_drop":clean-float(stress.mean()),
                          "mean_retention":float((stress/clean).mean())})
    s=pd.DataFrame(summaries)
    s.to_csv(Path(args.out).with_name("robustness_summary.csv"),index=False)
    grp=df.groupby(["experiment","group"])["macro_f1"].agg(["mean","std"]).reset_index()
    grp.to_csv(Path(args.out).with_name("robustness_grouped.csv"),index=False)
    print(s.round(4).to_string(index=False))

if __name__=="__main__": main()
