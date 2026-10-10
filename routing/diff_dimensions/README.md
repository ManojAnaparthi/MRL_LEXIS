# MRL Dimension Routing: Baseline vs Task Specialization

## Question

Does training a task-specific block of an embedding for each task improve performance on that task compared with standard shared MRL training?

## Setup

Both runs use TinyBERT with 768-dimensional sentence representations and nested dimensions 16, 32, 64, 128, 256, 512, and 768. The seven evaluation tasks are TweetEval sentiment, Banking77, MRPC, WiC, DGEM_FORMAT, SICK, and STS-B. The task-to-block assignments follow increasing block sizes: 0–16 to TweetEval, 16–32 to Banking77, 32–64 to MRPC, 64–128 to WiC, 128–256 to DGEM_FORMAT, 256–512 to SICK, and 512–768 to STS-B.

The baseline trains every dimension block on all training examples. The specialization run masks the contrastive query to the block assigned to each example's source task; the full batch remains the key set. In both runs, frozen embeddings are evaluated on every task with an independent logistic-regression classifier. Scores are F1, except STS-B, which uses Spearman correlation. The JSON files contain one result matrix per run; no multi-seed uncertainty estimates are provided.

## Scores at assigned block and full width

Scores are shown as percentages. Delta is specialization minus baseline, in percentage points.

| Task (assigned width) | Baseline at block | Specialized at block | Δ at block | Baseline 768D | Specialized 768D | Δ at 768D |
|---|---:|---:|---:|---:|---:|---:|
| TweetEval sentiment (16D) | 61.18 | 61.83 | +0.65 | 69.08 | 69.65 | +0.57 |
| Banking77 (32D) | 26.50 | 23.61 | −2.90 | 62.20 | 58.77 | −3.43 |
| MRPC (64D) | 50.10 | 51.62 | +1.52 | 65.73 | 62.84 | −2.88 |
| WiC (128D) | 61.63 | 60.68 | −0.96 | 62.47 | 60.01 | −2.46 |
| DGEM_FORMAT (256D) | 70.66 | 71.61 | +0.95 | 71.58 | 71.91 | +0.32 |
| SICK (512D) | 73.59 | 72.57 | −1.02 | 74.63 | 72.81 | −1.82 |
| STS-B (768D) | 73.30 | 72.87 | −0.43 | 73.30 | 72.87 | −0.43 |

## Result

At its assigned block, specialization is higher on three tasks (TweetEval, MRPC, DGEM_FORMAT) and lower on four (Banking77, WiC, SICK, STS-B). At full width, it is higher only on TweetEval and DGEM_FORMAT; the remaining five scores are lower. Thus, these results do not show a consistent advantage for task-specific routing. The largest assigned-block drop is Banking77 (−2.90 points), while the largest assigned-block gain is MRPC (+1.52 points).

The standard baseline scores also show that larger prefixes generally help several tasks, but gains are task-dependent. The results do not by themselves establish why a task changed: explanations involving task complexity, block capacity, or transfer are hypotheses, not directly tested mechanisms.

## Files and limitations

- `run_baseline_mrl.py` and `run_task_specialization.py`: training and evaluation implementations.
- `results_baseline_mrl.json` and `results_task_specialization.json`: saved score matrices and task/block metadata.
- `mrl_specialization_analysis.md`: prior narrative analysis; this concise report uses the saved JSON values for the comparisons above.

These are single reported runs without standard deviations or significance testing. Results should be treated as descriptive evidence for these particular configurations and task set.
