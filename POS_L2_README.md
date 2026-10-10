# Dense L2-Gated POS Probe

## Question

When a POS tagger can scale every MIPIC coordinate independently, does it assign larger weights to earlier coordinates? Unlike the fixed-budget probes, this method keeps all 768 dimensions and does not choose an active-coordinate count.

## Setup

- **Dataset:** Universal Dependencies English Web Treebank (EWT), revision 2.18.
- **Task:** token-level universal POS tagging. Each word uses its first subword's final-layer MIPIC vector.
- **Encoder:** frozen MIPIC; the POS probe alone is trained.
- **Features:** per-coordinate standardized using train-token statistics only.
- **Gate:** one learned sigmoid scale per coordinate, applied to the full-width feature vector before the linear classifier.
- **Objective:** POS cross-entropy plus `0.001 × sum(gate²)`; classifier uses AdamW with weight decay 0.01.
- **Training:** 8 epochs maximum, learning rate 0.002, batch size 2,048. Validation macro-F1 selects the checkpoint; test is used for final scores.
- **Seeds:** 13, 37, and 73. Values below are mean ± population standard deviation.
- **Control:** same full-width linear POS probe without learned gates.

## Test performance

| Probe | Accuracy | Macro-F1 |
|---|---:|---:|
| Full-width, ungated | 87.63 ± 0.15% | 81.49 ± 0.10% |
| Dense L2 gates | 87.39 ± 0.11% | 81.17 ± 0.14% |
| Gated minus ungated | −0.24 pp | −0.32 pp |

The gated probe scored slightly below the ungated control on both reported metrics. Macro-F1 was lower for the gated probe on all three seeds, by approximately 0.19–0.39 percentage points. With three seeds, this is a descriptive comparison, not a significance test.

## Learned coordinate scales

| Gate statistic | Mean ± std across seeds |
|---|---:|
| Mean gate value across dimensions | 0.689 ± 0.033 |
| Within-run standard deviation across dimensions | 0.046 ± 0.006 |
| Minimum gate in a run | 0.638 ± 0.041 |
| Maximum gate in a run | 0.948 ± 0.003 |
| Fraction of gates below 0.5 | 0% |
| Fraction of gates below 0.25 | 0% |
| Mean gate, dimensions 0–127 | 0.687 ± 0.033 |
| Mean gate, dimensions 128–767 | 0.690 ± 0.033 |
| Spearman correlation: dimension index vs. gate | 0.028 ± 0.012 |

The first 128 coordinates received a slightly lower mean gate than the remaining 640, by about 0.0035 on average. This difference is very small. The index-to-gate Spearman correlations were low for every seed (0.020–0.045), with no individual correlation statistically significant at the 0.05 level. These results do not show a meaningful preference for early coordinates.

### Mean gate over initial prefixes

This table averages each seed's gate values over the first N dimensions, then reports mean ± population standard deviation across the three seeds.

| Initial dimensions | Mean gate value |
|---:|---:|
| First 32 | 0.686 ± 0.035 |
| First 64 | 0.686 ± 0.034 |
| First 128 | 0.687 ± 0.033 |
| First 256 | 0.688 ± 0.033 |
| First 512 | 0.687 ± 0.033 |
| All 768 | 0.689 ± 0.033 |

The prefix averages are nearly flat. The first 32 average is slightly lower than the full-vector average, and the first 256 is the highest of these prefixes, but the differences are small relative to seed-to-seed variation. There is no clear pattern of higher weights for the initial coordinates.

The regularizer reduced the average gate below its 0.9 initialization, but no gate fell below 0.5 in any seed. Thus, this run did not produce strongly suppressed or near-zero coordinates. Mean gate values also varied across seeds (0.649–0.730), while their relative ordering by coordinate was only weakly related to index.

## Interpretation and limitations

For this frozen MIPIC representation and POS setup, the L2-gated probe did not improve over the ungated full-width control, and learned gate values did not favor early dimensions. The gates are multiplicative scales, not direct measures of causal feature importance: the linear classifier can compensate for a coordinate's scale with its own weights. In addition, regularization strength affects the gate magnitudes; this is one coefficient setting, one dataset, and three seeds. The result does not establish dimension importance for other tasks or sentence-level embeddings.

## Files

- `pos_probe_l2.py`: dense-gate training and evaluation.
- `mipic-a1-lexis.ipynb`, section 15: Kaggle cell that runs the probe and displays the summary and coordinate gates.
- `pos_probe_l2_results.json`: per-seed metrics and all 768 gate values for each seed.
- `pos_probe_l2_summary.csv`: per-seed performance and gate summary.
- `pos_probe_l2_prefix_gate_means.csv`: mean gate values for the first 32, 64, 128, 256, 512, and 768 dimensions, including per-seed values.

The notebook run also writes `pos_probe_l2_dimension_gates.csv` in its Kaggle output directory.
