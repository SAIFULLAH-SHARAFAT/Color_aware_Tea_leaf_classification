# Future color-branch designs

These files preserve the research designs discussed after reading:

- `2608.12385v2` (Decode-Branch Transformer)
- `2609.13285v2` (Grouped Value Attention)

They are kept separate so `scripts/05_run_experiments.py --all` can never run
them accidentally.  Do not cite results from this folder until a new frozen
experiment matrix is declared and all required multi-seed controls are run.

## Design ladder

1. **F0 shared-weight dual-pass baseline** — same Swin stage weights evaluated
   for RGB and HSV trajectories, but independent self-attention. Establishes
   whether weight sharing itself is viable.
2. **F1 asymmetric shared-KV color branch (a/b/c)** — target design. The RGB
   path is structurally independent for fixed weights; the color path forms its
   own query and reads RGB keys/values. All large Q/K/V/O/MLP matrices are shared.
3. **F2 minimal a/c coupling** — omits the MLP-intermediate b-vector. This is the
   version that can reach roughly 0.025--0.03M extra parameters with a 4-channel
   HSV patch stem. It is a new simplified design, not a faithful copy of
   Decode-Branch.
4. **F3 confidence-mixture readout** — test both the original clipped collision
   gate and a 7-class-normalized collision gate.
5. **F4 compute-matched gated-copy control** — branch receives a learned
   vector-gated copy of primary attention output so attention is evaluated once.

## Parameter accounting for Swin-S

Dims/depths: `[96,192,384,768]`, `[2,2,18,2]`; `sum depth*dim = 9,024`.

- faithful a/b/c vectors with MLP ratio 4: `6 * 9,024 = 54,144`
- minimal a/c vectors: `2 * 9,024 = 18,048`
- 4-channel -> 96 patch stem, 4x4 with bias: `4*96*16 + 96 = 6,240`
- faithful total before optional norms/readout: ~60,384 (~0.060M)
- minimal a/c total before optional norms/readout: ~24,288 (~0.024M)

Run `python future_designs/00_parameter_accounting.py` to regenerate these
figures rather than copying them by hand.

## Important conceptual caveat

The primary RGB **forward graph** does not read the color branch. That does not
mean a jointly trained RGB predictor has identical learned weights to a
separately trained baseline, because branch-loss gradients can update shared
weights. A future paper should distinguish structural forward independence from
training-time co-adaptation.

## Implementation status

The shared-KV Swin design is intentionally
kept as a contract/reference implementation because block-level Swin attention
integration is timm-version-sensitive. Run `05_inspect_timm_swin_api.py` in the
future environment before implementing the adapter, then freeze that timm
version in a separate campaign.
