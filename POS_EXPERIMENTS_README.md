# POS Probes on Frozen MIPIC Representations

## Research question

Can a POS tagger perform better with task-selected MIPIC dimensions than with the first k dimensions of the representation?

## Shared experiment setup

- **Dataset:** Universal Dependencies English Web Treebank (EWT), with train, validation, and test splits.
- **Task:** token-level universal POS tagging. Each word is represented by the final-layer vector at its first subword.
- **Encoder:** frozen MIPIC, producing 768-dimensional token vectors. Only the POS probe is trained.
- **Normalization:** each coordinate is standardized using training-token statistics, shared by both comparison methods.
- **Dimension budgets:** k = 16, 32, 64, 128, and 256.
- **Seeds:** 13, 37, and 73. Tables report mean ± population standard deviation.
- **Model selection:** validation macro-F1; final scores are measured on the test split.

The baseline is a linear POS classifier trained using the first k coordinates. Results apply to these token-level representations and this POS dataset; they do not establish dimension ordering for sentence embeddings or other tasks.

## Experiment 1: Fixed-budget learned selector (`pos_probe.py`)

This method jointly trains a linear classifier and a selector over the full representation. A hard top-k mask keeps exactly k coordinates in each forward pass. A straight-through soft top-k surrogate provides gradients to the selector during training.

| k | Prefix macro-F1 | Learned macro-F1 | Difference |
|---:|---:|---:|---:|
| 16 | 24.95 ± 0.11% | 20.84 ± 2.06% | −4.11 pp |
| 32 | 29.40 ± 0.06% | 29.96 ± 1.04% | +0.56 pp |
| 64 | 40.39 ± 0.05% | 42.53 ± 0.72% | +2.14 pp |
| 128 | 50.87 ± 0.22% | 56.10 ± 0.18% | +5.23 pp |
| 256 | 66.34 ± 0.20% | 69.23 ± 0.69% | +2.89 pp |

The learned selector has higher mean macro-F1 at k = 32, 64, 128, and 256, and lower at k = 16. Its largest mean gain is 5.23 percentage points at k = 128. The selected coordinate sets differ substantially across seeds (mean pairwise Jaccard scores: 0.016–0.218).

## Experiment 2: Sparse learned gates (`pos_probe_spars.py`)

This method adds a learned sigmoid coefficient for every coordinate to the top-k selector. The selected coordinates can be scaled, and no more than k coordinates are active. The classifier, selector, and gate coefficients are trained jointly.

| k | Prefix macro-F1 | Sparse-gate macro-F1 | Difference |
|---:|---:|---:|---:|
| 16 | 24.95 ± 0.11% | 20.97 ± 2.33% | −3.99 pp |
| 32 | 29.40 ± 0.06% | 30.29 ± 1.45% | +0.89 pp |
| 64 | 40.39 ± 0.05% | 42.83 ± 1.72% | +2.45 pp |
| 128 | 50.87 ± 0.22% | 55.83 ± 1.16% | +4.96 pp |
| 256 | 66.34 ± 0.20% | 69.54 ± 1.47% | +3.20 pp |

The sparse-gate method also has higher mean macro-F1 at k = 32–256 and lower at k = 16. Its largest mean gain is 4.96 percentage points at k = 128. Mean selected gate values are about 0.92. Selected coordinate sets remain variable across seeds (mean pairwise Jaccard scores: 0.005–0.214).

## Conclusion

Both current fixed-budget methods show the same overall pattern: learned selection trails the prefix at 16 dimensions but exceeds it on average from 32 through 256 dimensions. The strongest average improvement for both methods is at 128 dimensions. The selected sets vary across seeds, so these results do not identify one stable set of POS dimensions. With three seeds and one dataset, the differences are descriptive and do not establish statistical significance or generalization beyond this setup.

## Experiment 3: Dense L2 gates (`pos_probe_l2.py`)

This probe learns a sigmoid scale for every coordinate and adds an L2 penalty to the gate vector. It keeps all 768 coordinates and has no active-dimension budget. A full-width ungated probe is the control.

| Probe | Test accuracy | Test macro-F1 |
|---|---:|---:|
| Full-width, ungated | 87.63 ± 0.15% | 81.49 ± 0.10% |
| Dense L2 gates | 87.39 ± 0.11% | 81.17 ± 0.14% |
| Gated minus ungated | −0.24 pp | −0.32 pp |

The gated probe was slightly lower on average and had lower macro-F1 on each of the three seeds. Its mean gate was 0.689 ± 0.033, with no gate below 0.5. The mean gate over the first 32, 64, 128, 256, and 512 coordinates was 0.686, 0.686, 0.687, 0.688, and 0.687 respectively, compared with 0.689 over all 768 coordinates. This nearly flat pattern and the low coordinate-index correlation (Spearman rho 0.028 ± 0.012 across seeds) provide no clear evidence that the model weights early coordinates more for POS tagging. Gate values are scales rather than causal feature importance, and can be partly compensated by the linear classifier. See [POS_L2_README.md](POS_L2_README.md) for the full setup, results, and limitations.

## Reproducibility and artifacts

The fixed-budget methods are implemented in `pos_probe.py` and `pos_probe_spars.py`; the dense L2-gated method is in `pos_probe_l2.py`. Details for the first two methods are also recorded in `POS_PROBE_README.md` and `POS_SPARSE_GATES_README.md`. Their per-run POS JSON/CSV files are not included in this workspace, so their per-seed rows and selected-coordinate indices cannot be independently inspected here. The L2 results analyzed above are present in `pos_probe_l2_results.json` and `pos_probe_l2_summary.csv`. The scripts require a trained MIPIC checkpoint by path. The notebook's saved training log records completion of a MIPIC run and an external Kaggle checkpoint, but that checkpoint is not present locally.
