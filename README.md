## Final confirmatory matrix

Complementary HSV family: `R0 R1 R2 R3` (3 seeds each).
Primary token family: `C1 TLA C2 C4` (3 seeds each).

- C1: recipe/head-matched no-token baseline
- TLA: Stage-4 RGB token-level attention
- C2: Stage-4 non-attention MLP with exactly the same 3,545,089 active parameters as TLA
- C4: lightweight Stage-4 ECA-style attention comparator

Historical prompting, Stage-3 post-hoc attention, segmentation, and HSV-TCCA screening are not part of the fresh confirmatory campaign.

## 0. Install
```bash
pip install -r requirements.txt
```

## 1. Download the exact public experiment dataset
```bash
python scripts/00_fetch_frozen_dataset.py --revision main --overwrite
```
This resolves `main` to a concrete Hugging Face commit SHA and stores it in `data/hf_dataset_lock.json`. For maximum archival stability, rerun later with that SHA as `--revision`.

If you already copied the exact Hugging Face release into `data/Tea_leaf_dataset` and `data/manifests`, skip the download and run verification directly.

## 2. Verify the frozen dataset
Fast lock + filesystem + leakage checks:
```bash
python scripts/01_verify_frozen_dataset.py
```
Optional full per-file MD5 verification:
```bash
python scripts/01_verify_frozen_dataset.py --verify-md5
```
Expected: train 6090, validation 816, test 808, total 7714, manifest SHA `d407fdb1...43d3d`, zero cross-split exact hashes, zero source-family overlap, zero derivatives outside train.

## 3. Preflight and record environment
```bash
python scripts/02_preflight.py
python scripts/03_capture_environment.py
python scripts/04_validate_model_contracts.py
```
Do not launch the full campaign if any of these fail.

## 4. Seed-42 smoke campaign
```bash
python scripts/05_run_experiments.py --all --seeds 42
```

## 5. Remaining confirmatory seeds
```bash
python scripts/05_run_experiments.py --all --seeds 1337 2026
```
This yields 24 fresh training runs in total (8 models x 3 seeds).

## 6. Aggregate results
```bash
python scripts/06_aggregate_results.py
python scripts/10_make_training_settings_table.py
```
The main tables are deliberately separated into `token_primary_results.tex` and `hsv_complementary_results.tex`; models trained under different recipe families are not ranked against one another.

## 7. Paired bootstrap
```bash
python scripts/07_paired_bootstrap.py --auto --n-boot 10000
```
Fresh runs save `test_ids.json`; the bootstrap aligns by exact image ID before paired resampling.

## 8. Robustness, all three seeds
```bash
python scripts/08_eval_robustness.py --planned
```
This evaluates the predeclared models under 18 corruptions + unperturbed.

## 9. Figures
```bash
python scripts/09_make_figures.py --format pdf
```

## 10. Optional paired error analysis
```bash
python scripts/12_pairwise_error_analysis.py \
  --a results/fresh/runs/TLA_s42 \
  --b results/fresh/runs/C1_s42 \
  --out results/fresh/tables/error_TLA_vs_C1_s42.json
```

## 11. Export public reproducibility material
```bash
python scripts/11_export_reproducibility_bundle.py
```
The export omits image data and model checkpoints but includes code, configs, manifests, metrics, exact predictions/IDs, tables, logs, and environment metadata.

## Interpretation rules

1. TLA's primary causal comparison is **TLA vs C1**.
2. Mechanism-vs-capacity is **TLA vs C2**.
3. Lightweight-attention comparison is **TLA vs C4**.
4. HSV conclusions are based on **R1/R2/R3 vs R0**, not comparison with TLA.
5. Bootstrap intervals quantify test-sample uncertainty for fixed checkpoints; three-seed variation remains the training-variance evidence.
6. Controlled corruptions are not field validation.
7. Do not add new architectures after inspecting test results; this matrix is the predeclared revision campaign.
