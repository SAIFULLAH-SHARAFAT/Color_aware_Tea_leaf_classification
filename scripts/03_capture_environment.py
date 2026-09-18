#!/usr/bin/env python3
from __future__ import annotations
import json, platform, subprocess, sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
def v(name):
    try:
        m=__import__(name); return getattr(m,"__version__","unknown")
    except Exception as e: return f"unavailable: {e}"
def main():
    import torch
    info={"python":sys.version,"platform":platform.platform(),"torch":v("torch"),
          "torchvision":v("torchvision"),"timm":v("timm"),"numpy":v("numpy"),
          "pandas":v("pandas"),"sklearn":v("sklearn"),"pillow":v("PIL"),
          "cuda_available":torch.cuda.is_available(),"cuda_runtime":torch.version.cuda,
          "cudnn":torch.backends.cudnn.version(),
          "gpu":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    try: info["pip_freeze"]=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True).splitlines()
    except Exception: info["pip_freeze"]=[]
    out=REPO/"results/fresh/environment.json"; out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(info,indent=2)); print(json.dumps({k:v for k,v in info.items() if k!='pip_freeze'},indent=2)); print("wrote",out)
if __name__=="__main__": main()
