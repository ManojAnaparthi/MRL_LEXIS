# POS Probe: Sparse Learned Gates

## Question

Can a POS tagger use a task-learned sparse subset of MIPIC dimensions when it learns both which dimensions to keep and how strongly to scale each one, under a budget of at most k active dimensions?

For an embedding such as `[2, 3, 1, 5.5, 6]`, learned gates `[0.7, 0, 0, 0.5, 0]` produce `[1.4, 0, 0, 2.75, 0]`. The vector retains its original width, but only the gated coordinates contribute.

## Setup

- **Dataset:** Universal Dependencies English Web Treebank (EWT), with train, validation, and test splits.
- **Representation:** frozen MIPIC final-layer token vectors, using the first subword vector for each word.
- **Task:** token-level universal POS tagging.
- **Representation width:** 768.
- **Prefix baseline:** a linear POS classifier using the first k coordinates.
- **Sparse learned-gate probe:** a linear classifier with a global dimension selector and learned sigmoid coefficients. The forward pass retains the top k selector scores; other coordinates are set to zero. The selector, coefficients, and classifier train jointly using POS cross-entropy. Thus the selected values can be attenuated, and no more than k dimensions are active.
- **Normalization:** coordinate-wise mean and standard deviation computed from train tokens only and shared by both probe conditions.
- **Selection budgets:** k = 16, 32, 64, 128, and 256.
- **Seeds:** 13, 37, and 73.
- **Checkpoint selection:** validation macro-F1. Test metrics are reported for the selected checkpoint.

The table gives means and population standard deviations across three seeds. Macro-F1 and accuracy are shown as percentages. The difference is sparse-gate macro-F1 minus prefix macro-F1, in percentage points. “Selected in first k” is the mean fraction of selected indices below k. Pairwise Jaccard measures overlap among the selected coordinate sets across seeds; 1 means identical sets.

## Test results

| k | Prefix macro-F1 | Sparse-gate macro-F1 | Δ macro-F1 | Prefix accuracy | Sparse-gate accuracy | Mean selected gate value | Selected in first k | Pairwise Jaccard |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 24.95 ± 0.11% | 20.97 ± 2.33% | −3.99 pp | 38.97 ± 0.11% | 33.20 ± 3.26% | 0.9261 ± 0.0120 | 6.25% | 0.011 |
| 32 | 29.40 ± 0.06% | 30.29 ± 1.45% | +0.89 pp | 43.77 ± 0.05% | 44.14 ± 1.74% | 0.9217 ± 0.0169 | 4.17% | 0.005 |
| 64 | 40.39 ± 0.05% | 42.83 ± 1.72% | +2.45 pp | 52.78 ± 0.04% | 54.98 ± 1.99% | 0.9211 ± 0.0153 | 10.94% | 0.073 |
| 128 | 50.87 ± 0.22% | 55.83 ± 1.16% | +4.96 pp | 62.54 ± 0.09% | 67.60 ± 0.62% | 0.9207 ± 0.0160 | 17.19% | 0.115 |
| 256 | 66.34 ± 0.20% | 69.54 ± 1.47% | +3.20 pp | 76.48 ± 0.03% | 78.51 ± 0.26% | 0.9178 ± 0.0175 | 32.42% | 0.214 |

## Findings

The sparse learned-gate probe has higher mean test macro-F1 than the prefix baseline at k = 32, 64, 128, and 256. At k = 16, it is lower by 3.99 percentage points. The largest observed improvement is at k = 128 (+4.96 points). These comparisons are descriptive; three seeds do not establish statistical significance.

The selected coordinates are spread beyond the first k positions. At k = 16, only 6.25% of selected coordinates are in the first 16 on average; at k = 256 the proportion is 32.42%. Pairwise Jaccard is very low at k = 16 and 32, then rises with larger budgets to 0.214 at k = 256. This indicates that selected sets differ substantially across seeds, especially at the smaller budgets; similar performance need not depend on an identical coordinate subset.

Mean selected gate values are around 0.92 across budgets. The learned coefficients therefore scale selected normalized coordinates modestly on average in this run, while the hard top-k selection supplies most of the sparsity. Individual coefficient values vary by coordinate and seed; the mean does not imply all selected coordinates have the same contribution.

## Limitations

- This is token-level POS tagging over frozen final-layer token vectors. It does not establish how sentence-level pooled embeddings behave or whether the result transfers to other tasks.
- The encoder is frozen, so the experiment measures probe-level selection and scaling rather than adaptation of MIPIC representations.
- The hard top-k forward selection uses a straight-through soft top-k surrogate for training gradients.
- Only three seeds and one dataset are included. Results are descriptive and do not prove a universal dimension ordering or statistically reliable performance advantage.
