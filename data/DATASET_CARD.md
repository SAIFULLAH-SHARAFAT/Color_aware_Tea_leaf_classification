---
annotations_creators:
- expert-generated
language:
- en
license: mit
task_categories:
- image-classification
task_ids:
- multi-class-image-classification
tags:
- agriculture
- plant-pathology
- computer-vision
- tea-leaf
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/Tea_leaf_dataset/train/*/*
  - split: validation
    path: data/Tea_leaf_dataset/val/*/*
  - split: test
    path: data/Tea_leaf_dataset/test/*/*
---

# Data

Image data is not versioned in this repository. Only manifests, audit outputs,
and fingerprints are tracked, which is enough to verify and rebuild the exact
partition used.

## Layout

```text
data/
  raw/                   TeaLeafBD release as downloaded      (gitignored)
  tea_leaf_clean/        intermediate build                   (gitignored)
  Tea_leaf_dataset/      FINAL audited dataset                (gitignored)
  manifests/             tracked audit artifacts