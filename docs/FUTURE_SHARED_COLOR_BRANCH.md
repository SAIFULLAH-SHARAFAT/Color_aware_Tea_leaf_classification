# Future shared-weight chromatic Swin design

## Hypothesis

A chromatic trajectory can reuse the RGB Swin's large attention and MLP matrices
while adding only a separate HSV patch stem and small per-block coupling vectors.
If class-dependent HSV gains are real rather than variance, a sample-adaptive
readout should exploit color when RGB is uncertain without forcing color into
every prediction.

## Primary/branch asymmetry

For each block, the RGB trajectory computes its ordinary query, key, value,
attention output, and MLP output. The color trajectory computes a separate query
but reads the RGB keys and values. The primary forward graph never reads color
state. Large Q/K/V/O and MLP matrices are shared.

Faithful Decode-Branch-style couplings use vectors `a` in D, `b` in 4D, and `c`
in D. For Swin-S this is 54,144 coupling parameters across all blocks. A
4-channel sin/cos-HSV patch stem adds 6,240, giving about 0.060M before any
optional readout parameters. A deliberately simplified a/c-only design is about
0.024M with the same stem.

## Structural independence vs training independence

For fixed weights the RGB forward path is unchanged by whether the color branch
is evaluated. During joint training, however, branch-loss gradients can update
shared matrices, so the learned RGB predictor need not equal a separately
trained baseline. Future experiments should compare joint training with a
frozen-primary variant if this distinction becomes central.

## Confidence mixture

Two candidate policies are archived:

1. source-style collision gate: `alpha=clip(sum(p^2), 0.5, 1)`;
2. proposed K-class-normalized collision gate mapping uniform `1/K` to 0.5 and
   a point mass to 1.

The second is included because with seven classes the source-style gate can sit
at 0.5 for a wide range of uncertain primary distributions. This is a hypothesis
to test, not a claimed improvement.

## Scale-matched initialization

The current revision logs pre-training RGB RMS, projected-HSV RMS, raw sigmoid
gate statistics, gated contribution RMS, and initial logit change. If future
runs show severe under-scaling before the gate, `rms_match_linear_` in
`future_designs/01_shared_color_components.py` can be used on a declared
training calibration batch before optimization. It must not be tuned against
validation/test accuracy after the fact.

## Required ablations for a future paper

- RGB baseline;
- shared-weight dual self-attention baseline;
- asymmetric shared-KV branch without coupling;
- + coupling;
- + confidence mixture;
- faithful a/b/c vs minimal a/c;
- joint vs frozen-primary training;
- parameter/FLOP/latency accounting;
- gated-copy compute control;
- class-wise error analysis, especially any class for which color degrades;
- three seeds and paired bootstrap;
- controlled robustness, with field generalization kept separate.
