# Reviewer-to-experiment map

| Reviewer concern | Evidence / artifact |
|---|---|
| Is TLA gain just a different recipe/head? | C1 vs TLA |
| Is TLA gain just extra parameters? | C2 vs TLA; exact active parameter match |
| Is a simple lightweight attention sufficient? | C4 vs TLA |
| HSV gain confounded by head/augmentation? | R0/R1/R2/R3 use matched head and hue policy |
| Bootstrap procedure unclear | `07_paired_bootstrap.py`, 10,000 paired image-level resamples |
| Hyperparameters scattered | `10_make_training_settings_table.py` |
| Robustness vs field generalization | `08_eval_robustness.py`; 18 controlled corruptions, explicitly not field validation |
| HSV branch may start under-scaled | pre-optimization `init_scale_diagnostics.json` + `13_summarize_init_scale.py` |
| Reproducibility/data availability | frozen HF lock, manifests, environment capture, export bundle |

## Interpretation hierarchy

Primary architectural claim: **C1 -> TLA**, interpreted together with C2 and C4.
Complementary chromatic claim: **R0 -> R1/R2/R3**.  Do not infer TLA-vs-HSV
superiority from cross-family values because the training recipes differ.
