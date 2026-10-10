# MIPIC Training and TweetEval Evaluation

## Goal

Train the MIPIC sentence encoder and compare it with three controlled variants on TweetEval sentiment across nested representation widths.

## Run recorded in the notebook

The saved output for notebook cell 29 records a run on the TinyBERT General 6L 768D backbone. It used the 20,240-row training corpus (three thousand rows each for TweetEval sentiment, Banking77, WiC, MRPC, and DGEM_FORMAT; 2,893 for STS-B; and 2,347 for SICK). Training used five epochs, sequence length 256, learning rate 2e-5, global batch size 16, and two Tesla T4 GPUs with DDP.

The notebook log says training completed in 44.08 minutes, saved the epoch-5 checkpoint, and then completed TweetEval evaluation. The MIPIC held-out probe preferred epoch 1/5; the reported dimension table is from the evaluation selected by the run. The local notebook setup cell (cell 14) has stale output that disagrees with its current source, so the executed configuration above is taken from the later training log, not inferred from the cell's source text.

## TweetEval results

Accuracy and F1 are percentages. The table reproduces the saved notebook output.

| Dimensions | MRL F1 | MRL+SIA F1 | MRL+PIC F1 | MIPIC F1 | MIPIC accuracy |
|---:|---:|---:|---:|---:|---:|
| 16 | 56.46 | 53.26 | 54.27 | 58.60 | 58.71 |
| 32 | 59.99 | 55.16 | 59.85 | 60.51 | 60.58 |
| 64 | 62.82 | 56.16 | 61.62 | 61.83 | 61.95 |
| 128 | 65.57 | 57.30 | 62.55 | 62.65 | 62.67 |
| 256 | 66.82 | 58.37 | 63.21 | 62.92 | 62.91 |
| 512 | 67.61 | 58.99 | 64.27 | 64.01 | 63.90 |
| 768 | 67.24 | 59.53 | 64.82 | 64.64 | 64.54 |

MIPIC has the highest F1 among the four methods at 16D and 32D. At 64D and above, standard MRL has higher F1 in this run. MIPIC's F1 rises from 58.60 at 16D to 64.64 at 768D; its best listed F1 is 64.64 at 768D. These results show compact-width performance for this run, but do not establish statistical significance because the notebook reports a single run without seed-level variation.

## Artifacts and scope

The notebook log says it saved `tweeteval_results.json`, `tweeteval_results.csv`, and the MIPIC checkpoint under `/kaggle/working/mipic_results_all/`. Those external Kaggle artifacts are not included in this local workspace; the saved notebook output is the available source for the metrics. The subsequent POS probe experiments use the MIPIC encoder's token vectors and are summarized separately in [POS_EXPERIMENTS_README.md](POS_EXPERIMENTS_README.md).
