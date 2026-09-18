# Frozen experiment dataset

All fresh experiments use the public frozen release:

- Hugging Face: `saifullah03/tea-leaf-disease-dataset`
- URL: https://huggingface.co/datasets/saifullah03/tea-leaf-disease-dataset
- Dataset tree: `data/Tea_leaf_dataset/`
- Audit artifacts: `data/manifests/`

## Frozen audited build

- train: **6,090** = 3,971 retained originals + 2,119 training-only derivatives
- validation: **816** originals
- test: **808** originals
- total effective instances: **7,714**
- retained originals after the audit: **5,595**
- removed cross-partition perceptual near-duplicate originals: **79** (35 validation, 44 test)
- manifest SHA-256: `d407fdb133ebc95c3b0bc6f815280786bcdef37cd6c3c009192198fd21943d3d`

The audit verifies byte-identical cross-split duplicates, filename-derived source-family confinement, training-only derivative confinement, and the project's two-signal perceptual near-duplicate rule. The complete manifests and removal records are released with the Hugging Face dataset.

## Experimental lock

`scripts/00_fetch_frozen_dataset.py` records the resolved Hugging Face commit SHA, and `scripts/01_verify_frozen_dataset.py` refuses to proceed if the manifest fingerprint, filesystem, counts, exact-hash audit, or source-family audit differs from the frozen build.
