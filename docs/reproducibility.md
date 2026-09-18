# Reproducibility notes

1. Dataset is locked by manifest SHA-256 and, after download, by the resolved Hugging Face commit SHA.
2. Run commands are generated from `configs/experiments.yaml`.
3. Seeds are 42, 1337, and 2026.
4. Each run stores its own config, exact predictions, test IDs, metrics, and source hashes.
5. Current paper runs are isolated under `results/fresh/`.
6. `future_designs/` is archival design work and is **not** part of the current paper's evidence.
7. If software/library updates alter numerical reproducibility, report the captured `environment.json` rather than silently changing the recipe.
