#!/usr/bin/env python3
"""
05_run_experiments.py
=====================

Run experiments declared in configs/experiments.yaml.

Every run writes its own directory under results/runs/<EXP>_s<SEED>/ containing
the config, logs, metrics, raw predictions, and checkpoints, so downstream
aggregation never depends on anything typed by hand.

USAGE
-----
    # what would run, without running it
    python scripts/05_run_experiments.py --list
    python scripts/05_run_experiments.py --only C1 --dry-run

    # a single run
    python scripts/05_run_experiments.py --only C1 --seeds 42

    # a family
    python scripts/05_run_experiments.py --family token

    # everything not already finished
    python scripts/05_run_experiments.py --all --skip-done

NOTES
-----
* --skip-done treats a run as finished when metrics/test_results.json exists.
* Session-limited environments (Kaggle, Colab) should invoke one or two runs
  per session; --auto_resume in the training scripts recovers an interrupted
  run from its checkpoint directory.
* Parameter-count expectations in the YAML are checked after each run. A
  mismatch means the architecture was not built as declared, which is the
  failure mode a silently ignored flag produces.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

try:
    import yaml
except ImportError:
    raise SystemExit("pyyaml required:  pip install pyyaml")

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "experiments.yaml"


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)



def _sha256_file(path: Path) -> str:
    import hashlib
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def verify_dataset_fingerprint(cfg: dict) -> None:
    """Verify both fingerprint JSON and the canonical manifest contents."""
    import hashlib
    import pandas as pd
    defaults=cfg["defaults"]
    expected=defaults.get("expected_manifest_sha256")
    fp=REPO/defaults["manifest_fingerprint"]
    mp=REPO/defaults["manifest_csv"]
    if not fp.exists() or not mp.exists():
        raise SystemExit("Frozen dataset manifests missing. Run scripts/00 and 01 first.")
    meta=json.load(open(fp))
    df=pd.read_csv(mp)
    cols=["split","class_name","filename","md5","canonical_id","is_derived","op"]
    if "op" not in df.columns: df["op"]=""
    canon=df[cols].sort_values(["split","class_name","filename"]).reset_index(drop=True)
    got=hashlib.sha256(canon.to_csv(index=False).encode()).hexdigest()
    if got != expected or meta.get("manifest_sha256") != expected:
        raise SystemExit(
            "DATASET MANIFEST LOCK MISMATCH. Refusing to launch experiments.\n"
            f" expected: {expected}\n recomputed: {got}\n json: {meta.get('manifest_sha256')}"
        )
    got_counts=(meta.get("n_train"),meta.get("n_val"),meta.get("n_test"),meta.get("n_total"))
    if got_counts != (6090,816,808,7714):
        raise SystemExit(f"Dataset count fingerprint mismatch: {got_counts}")
    print(f"Dataset manifest lock verified: {got}")


def write_run_provenance(cfg: dict, exp_id: str, seed: int, run_name: str, cmd: List[str]) -> None:
    """Attach the exact command, hashes, dataset lock, and HF snapshot ID to each run."""
    import datetime, platform
    defaults=cfg["defaults"]
    run_dir=REPO/defaults["results_dir"]/run_name
    cdir=run_dir/"config"; cdir.mkdir(parents=True,exist_ok=True)
    exp=cfg["experiments"][exp_id]; recipe=cfg["recipes"][exp["recipe"]]
    train_script=REPO/recipe["script"]
    sources={
      "train_script": {"path":str(train_script.relative_to(REPO)),"sha256":_sha256_file(train_script)},
      "experiment_yaml": {"path":"configs/experiments.yaml","sha256":_sha256_file(REPO/"configs/experiments.yaml")},
    }
    shared=REPO/"src/train_swin_three_models.py"
    if shared.exists(): sources["shared_swin_script"]={"path":"src/train_swin_three_models.py","sha256":_sha256_file(shared)}
    hf_lock=REPO/"data/hf_dataset_lock.json"
    provenance={
      "campaign_id":defaults.get("campaign_id"),"experiment":exp_id,"seed":seed,
      "run_name":run_name,"command":cmd,"dataset_manifest_sha256":defaults.get("expected_manifest_sha256"),
      "hf_dataset_lock": json.load(open(hf_lock)) if hf_lock.exists() else None,
      "sources":sources,"python":sys.version,"platform":platform.platform(),
      "recorded_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(cdir/"run_provenance.json","w") as f: json.dump(provenance,f,indent=2)

def flag_args(d: Dict) -> List[str]:
    """Turn a dict into CLI flags. True -> bare flag, False -> omitted."""
    out: List[str] = []
    for k, v in d.items():
        if isinstance(v, bool):
            if v:
                out.append(f"--{k}")
        else:
            out += [f"--{k}", str(v)]
    return out


def build_command(cfg: dict, exp_id: str, seed: int) -> tuple[str, List[str]]:
    defaults = cfg["defaults"]
    exp = cfg["experiments"][exp_id]
    recipe = cfg["recipes"][exp["recipe"]]

    run_name = f"{exp_id}_s{seed}"
    run_dir = REPO / defaults["results_dir"]
    ckpt_dir = REPO / defaults["ckpt_dir"] / run_name

    args: Dict = {}
    args.update(recipe["args"])
    args.update(exp.get("args", {}))

    cmd = [sys.executable, str(REPO / recipe["script"]),
           "--exp_name", run_name,
           "--run_dir", str(run_dir),
           "--data_root", str(REPO / defaults["data_root"]),
           "--ckpt_temp_dir", str(ckpt_dir),
           "--seed", str(seed),
           "--auto_resume"]
    cmd += flag_args(args)
    return run_name, cmd


def check_params(run_dir: Path, exp: dict, exp_id: str) -> None:
    mp = run_dir / "metrics" / "test_results.json"
    if not mp.exists():
        return
    m = json.load(open(mp))

    # Optional total-model guard.
    got_total = m.get("params_m")
    want_total = exp.get("expect_params_m")
    if want_total is not None and got_total is not None:
        if abs(float(got_total) - float(want_total)) > 0.05:
            raise SystemExit(
                f"TOTAL PARAMETER MISMATCH for {exp_id}: built {got_total}M, "
                f"declared {want_total}M. Refusing to continue."
            )
        print(f"   total parameter count verified: {got_total}M")

    # Reviewer controls use the ACTIVE token-module parameter budget.
    want_token = exp.get("expect_token_params")
    if want_token is not None:
        got_token = m.get("token_module_params")
        if got_token is None:
            raise SystemExit(
                f"Missing token_module_params in {mp} for {exp_id}. "
                "Refusing to accept an unverifiable control run."
            )
        if int(got_token) != int(want_token):
            raise SystemExit(
                f"TOKEN PARAMETER MISMATCH for {exp_id}: built {got_token:,}, "
                f"expected {int(want_token):,}. Refusing to continue."
            )
        print(f"   active token-module budget verified: {int(got_token):,}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--only", nargs="+", help="Experiment IDs, e.g. C1 C2")
    ap.add_argument("--family", help="Run a whole family (hsv | token)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", help="Override declared seeds")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    verify_dataset_fingerprint(cfg)
    exps = cfg["experiments"]

    if args.list:
        print(f"{'ID':<6} {'family':<7} {'recipe':<6} {'seeds':<18} label")
        for k, v in exps.items():
            seeds = v.get("seeds", cfg["defaults"]["seeds"])
            print(f"{k:<6} {v.get('family',''):<7} {v['recipe']:<6} "
                  f"{str(seeds):<18} {v['label']}")
        return

    if args.only:
        selected = [e for e in args.only if e in exps]
        missing = [e for e in args.only if e not in exps]
        if missing:
            raise SystemExit(f"unknown experiment(s): {missing}")
    elif args.family:
        selected = [k for k, v in exps.items() if v.get("family") == args.family]
    elif args.all:
        selected = list(exps)
    else:
        raise SystemExit("choose --only, --family, --all or --list")

    jobs = []
    for exp_id in selected:
        seeds = args.seeds or exps[exp_id].get("seeds", cfg["defaults"]["seeds"])
        for s in seeds:
            jobs.append((exp_id, s))

    print(f"{len(jobs)} run(s) selected\n")
    for exp_id, seed in jobs:
        run_name, cmd = build_command(cfg, exp_id, seed)
        run_dir = REPO / cfg["defaults"]["results_dir"] / run_name

        if args.skip_done and (run_dir / "metrics" / "test_results.json").exists():
            print(f"[skip] {run_name} already complete")
            continue

        print("=" * 78)
        print(f"RUN {run_name}  --  {exps[exp_id]['label']}")
        print("=" * 78)
        print(" ".join(cmd) + "\n")
        if args.dry_run:
            continue

        subprocess.run(cmd, check=True)
        check_params(run_dir, exps[exp_id], exp_id)
        write_run_provenance(cfg, exp_id, seed, run_name, cmd)


if __name__ == "__main__":
    main()
