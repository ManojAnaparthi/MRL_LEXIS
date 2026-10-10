# NLP Experiments: MRL, MIPIC, and Dimension Routing

This repository contains three experiment groups. The linked reports summarize the question, setup, observed results, and limits of each result.

| Experiment | Report | Main question |
|---|---|---|
| FlowVec MRL on Indic sentiment | [Indic FlowVec MRL](indic/A2_FlowVec_MRL_Final/README.md) | How much sentiment performance remains as a 4096-dimensional FlowVec embedding is reduced? |
| MIPIC training on TweetEval | [MIPIC training and evaluation](MIPIC_LEXIS_A1/MIPIC_TRAINING_README.md) | How does MIPIC compare with the MRL, MRL+SIA, and MRL+PIC variants on compact TweetEval embeddings? |
| MIPIC POS dimension probes | [MIPIC POS probes](MIPIC_LEXIS_A1/POS_EXPERIMENTS_README.md) | Can a POS probe use selected coordinates better than a prefix of the frozen MIPIC token representation? |
| MRL dimension routing | [Routing and task specialization](routing/diff_dimensions/README.md) | Does assigning different dimension blocks to different tasks improve each task's scores? |

## Reading the reports

The Indic report records the FlowVec training and evaluation results. The MIPIC training report is based on the saved notebook log, while its checkpoint and result CSV/JSON are in the external Kaggle output path and are not in this workspace. The MIPIC POS report summarizes two fixed-budget selector protocols and the dense L2-gated probe; the latter's full report is [here](MIPIC_LEXIS_A1/POS_L2_README.md). The routing report reads the saved cross-task score matrices and compares each task at its assigned block and at the full 768 dimensions.

The notebook's setup-cell output differs from its current source, but the later saved training log records a five-epoch DDP run completing and saving a MIPIC checkpoint to Kaggle. The checkpoint is not included locally. The POS report treats its external MIPIC checkpoint as the representation source, as the probe scripts and existing results notes specify.
