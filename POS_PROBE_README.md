# MIPIC POS Probe

## Experiment

The trained MIPIC encoder was frozen. We extracted final-layer token vectors from the first subword of each word, then compared:

- **Prefix-k:** a linear POS classifier using the first k of 768 dimensions.
- **Learned-k:** a linear POS classifier and learned selector with exactly k active dimensions on every forward pass.

We tested k = 16, 32, 64, 128, and 256. The selector and classifier trained on the train split, validation selected the checkpoint, and the test split was used for final scoring. Both methods used the same train-only coordinate normalization. Values below are mean ± standard deviation across three seeds.

## Dataset

Universal Dependencies **English Web Treebank (EWT)**, with train, dev, and test splits and universal POS tags. The probe is token-level and uses MIPIC's final-layer token vectors, not its sentence-level pooled embedding.

## Test results

| k | Prefix macro-F1 | Learned macro-F1 | Δ macro-F1 | Prefix accuracy | Learned accuracy | Selected in first k | Random-selection expectation | Jaccard across seeds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 24.95 ± 0.11% | 20.84 ± 2.06% | -4.11 pp | 38.97 ± 0.11% | 32.28 ± 2.84% | 8.33 ± 2.95% | 2.08% | 0.022 |
| 32 | 29.40 ± 0.06% | 29.96 ± 1.04% | +0.56 pp | 43.77 ± 0.05% | 44.33 ± 1.53% | 6.25 ± 2.55% | 4.17% | 0.016 |
| 64 | 40.39 ± 0.05% | 42.53 ± 0.72% | +2.14 pp | 52.78 ± 0.04% | 55.05 ± 0.98% | 9.90 ± 1.95% | 8.33% | 0.058 |
| 128 | 50.87 ± 0.22% | 56.10 ± 0.18% | +5.23 pp | 62.54 ± 0.09% | 67.67 ± 0.09% | 17.45 ± 3.85% | 16.67% | 0.125 |
| 256 | 66.34 ± 0.20% | 69.23 ± 0.69% | +2.89 pp | 76.48 ± 0.03% | 79.04 ± 0.27% | 30.47 ± 1.10% | 33.33% | 0.218 |

The random-selection expectation is k/768: the expected fraction of a uniformly selected k-dimension subset that falls in the first k coordinates. `pp` means percentage points. Jaccard measures overlap between selected dimension sets across seeds; 1 means identical sets.

## Inference

The learned selector performs better than the prefix at k=64, 128, and 256, ties closely at k=32, and performs worse at k=16. This indicates that POS-predictive information is available in non-prefix subsets, and that those subsets can outperform prefixes at moderate budgets.

The selected fraction in the first k is close to the uniform-selection expectation for k=32 through 256. The results therefore do **not** show a clear preference for either early or late coordinate locations. Also, low Jaccard scores show that the selector usually chooses different coordinates across seeds, even when its test score is similar. The task-relevant signal may be distributed or redundant across dimensions rather than tied to one stable subset.

These findings apply to POS tagging on the frozen final-layer token vectors. They do not establish how MIPIC's sentence-level pooled embeddings are ordered for other tasks.
