# Dataset provenance 

## Frozen experimental source

The **experimental source of truth is the public Hugging Face dataset**
`saifullah03/tea-leaf-disease-dataset`, together with the tracked final manifest
and fingerprint.  The frozen working set has **7,714 effective instances**:

- train: 6,090
- validation: 816
- test: 808

The manifest lock is:

`d407fdb133ebc95c3b0bc6f815280786bcdef37cd6c3c009192198fd21943d3d`

Every fresh run verifies that lock before training.

## Historical construction record

The authors' project record is that an earlier Mendeley snapshot downloaded at
the start of the project contained **5,674 images**.  That historical snapshot
was split and training-only augmentation was used to construct the processed
`tea-leaf701515` working corpus.  Subsequent leakage/near-duplicate cleaning
produced the frozen 7,714-instance release above.

This statement is a **provenance record of the snapshot actually used**, not a
claim that the current upstream Mendeley version still contains 5,674 images.
For reproducibility, readers should use the frozen Hugging Face release and its
commit SHA rather than reconstructing the experiment from a mutable upstream
version.

## Leakage scope

The released manifest and verification scripts check file-set identity, exact
MD5 overlap, canonical source-family overlap, and confinement of declared
derivatives to training.  The paper should describe those checks exactly and
should not broaden them into an unsupported claim that every possible semantic
near-duplicate in the world has been detected.
