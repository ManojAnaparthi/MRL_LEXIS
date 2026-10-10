# Assignment 2 — Matryoshka Representation Learning with FlowVec-v1

## 1. Project Overview

This project extends the Assignment 1 Matryoshka Representation Learning (MRL) experiment to **FlowVec-v1**, with additional evaluation on Indic languages.

The overall goal was to investigate whether a high-dimensional embedding model can be trained so that useful downstream information is preserved in progressively smaller embedding dimensions.

The final experiment evaluates three representation conditions:

1. **Frozen FlowVec-v1**
2. **English-MRL**
   - FlowVec-v1 fine-tuned using English unlabeled data
3. **Indic-MRL**
   - FlowVec-v1 fine-tuned using unlabeled Indic-language data

All representations are evaluated at the nested dimensions:

```text
16, 32, 64, 128, 256, 512, 768, 1024, 2048, 4096
```

The downstream task is sentiment classification using frozen embeddings and a Logistic Regression probe.

The final Indic experiment uses:

```text
Hindi
Bengali
Gujarati
Tamil
Telugu
```

---

# 2. Starting Point — Assignment 1

Assignment 2 was built on top of the MRL pipeline developed in Assignment 1.

The Assignment 1 backbone was:

```text
huawei-noah/TinyBERT_General_6L_768D
```

The encoder was trained using an unsupervised same-sentence positive-pair setup.

For each sentence, two views of the same sentence were passed through the encoder. Training-mode dropout provided stochasticity between the two views.

The representation used for MRL was the final-layer `[CLS]` representation.

The Assignment 1 MRL dimensions were:

```text
16, 32, 64, 128, 256, 512, 768
```

The active Assignment 1 training configuration used:

```text
Maximum sequence length : 256
Batch size              : 16
Epochs                  : 5
Learning rate           : 2e-5
Optimizer               : AdamW
Scheduler               : Cosine
Warmup ratio            : 0.10
Temperature             : 0.05
Random seed             : 42
```

The downstream evaluation used a frozen encoder and a Logistic Regression probe.

The labeled downstream data was not used to train the MRL encoder.

---

# 3. Moving from Assignment 1 to FlowVec-v1

For Assignment 2, the MRL approach was transferred from TinyBERT to:

```text
LingoIITGN/FlowVec-v1
```

FlowVec-v1 is based on Qwen3-Embedding-8B and produces:

```text
4096-dimensional embeddings
```

The FlowVec architecture uses:

```text
last-token pooling
```

followed by:

```text
L2 normalization
```

Therefore, the Assignment 2 extraction pipeline uses FlowVec's last-token representation rather than the `[CLS]` representation used in Assignment 1.

The final dimensions used for Assignment 2 were expanded to:

```text
16
32
64
128
256
512
768
1024
2048
4096
```

This is an adaptation of the MRL dimension ladder to FlowVec's 4096-dimensional representation.

---

# 4. Initial Frozen FlowVec Baseline

Before fine-tuning FlowVec, the original pretrained model was evaluated without any MRL training.

The frozen FlowVec English embeddings were extracted at the full 4096-dimensional representation and evaluated by slicing the representation to the required nested dimensions.

The frozen English baseline was:

| Dimension | Accuracy | Macro-F1 |
|----------:|---------:|---------:|
| 64   | 0.680070 | 0.684105 |
| 128  | 0.701340 | 0.705413 |
| 256  | 0.715326 | 0.720032 |
| 512  | 0.729312 | 0.733474 |
| 1024 | 0.736888 | 0.741390 |
| 2048 | 0.740676 | 0.745123 |
| 4096 | 0.743007 | 0.747546 |

The 16D and 32D representations can be obtained by slicing the same 4096-dimensional embeddings.

This frozen baseline became the reference point for evaluating whether MRL fine-tuning improved compact representations.

---

# 5. First Major Bottleneck — Training FlowVec-v1

FlowVec-v1 is dramatically larger than the TinyBERT model used in Assignment 1.

The initial goal was to perform **full-parameter fine-tuning**, without LoRA, so that the experiment would remain faithful to the intended MRL adaptation.

The FlowVec model has approximately:

```text
7.567 billion trainable parameters
```

The first major issue was GPU memory.

An NVIDIA L40S was available on the IITGN Slurm cluster.

However, using the standard AdamW optimizer caused an out-of-memory failure during the optimizer step.

The model parameters and gradients could fit, but the optimizer state pushed the memory requirement beyond the available GPU memory.

The observed memory requirement was approximately:

```text
~47.38 GiB
```

with only approximately:

```text
~15.81 MiB
```

free at the failure point.

The conclusion was that the optimizer state was the major additional memory bottleneck.

---

# 6. Solution — Adafactor

Instead of AdamW, the training was changed to PyTorch's native `torch.optim.Adafactor`.

The final optimizer configuration was:

```python
optimizer = torch.optim.Adafactor(
    model.parameters(),
    lr=2e-5,
    beta2_decay=-0.8,
    eps=(None, 0.001),
    d=1.0,
    weight_decay=0.0,
)
```

This substantially reduced optimizer-state memory.

An earlier attempt used Hugging Face-style Adafactor arguments such as:

```text
relative_step
scale_parameter
warmup_init
```

but the installed PyTorch Adafactor implementation did not accept those arguments.

The configuration was therefore changed to the native PyTorch API shown above.

---

# 7. GPU and Slurm Constraints

The IITGN cluster provided:

```text
GPU partition : gpu
GPU            : NVIDIA L40S
GPUs/node      : 4
GPU memory     : approximately 49 GB
CPU/node       : 128
RAM/node       : approximately 515 GB
```

There were two relevant nodes, each equipped with four L40S GPUs.

A multi-GPU allocation was attempted, but the Slurm QoS imposed a per-user GPU limit:

```text
QOSMaxGRESPerUser
```

Therefore, the experiment had to be designed to run on:

```text
1 × L40S
```

Attempts to work around the GPU requirement using CPU-only allocations also failed because the GPU partition/QoS required GPU resources.

For example, an attempted CPU-heavy allocation was rejected with:

```text
QOSMaxCpuPerUserLimit
```

and a CPU-only allocation was rejected with:

```text
QOSMinGRES
```

Therefore, CPU-only full-parameter FlowVec training was not a viable alternative.

---

# 8. Gradient Checkpointing

To further reduce activation memory, gradient checkpointing was enabled.

The final training configuration used non-reentrant gradient checkpointing:

```text
use_reentrant = False
```

and:

```text
use_cache = False
```

With:

```text
Batch size = 16
```

the successful training run fit on one L40S.

Peak GPU usage was approximately:

```text
~31.6 GB
```

during the successful English-MRL training run.

---

# 9. Batch Size Issue and InfoNCE Negatives

An important issue occurred while experimenting with memory-saving batch sizes.

An initial configuration used:

```text
batch size = 1
gradient accumulation = 16
```

Although gradient accumulation produced an effective optimization batch size of 16, this did **not** provide 16 in-batch negatives to the InfoNCE loss.

The actual microbatch contained only one example.

Consequently, the InfoNCE computation had no meaningful negative examples.

The batch size was therefore increased to provide real in-batch negatives.

The final configuration became:

```text
Batch size            : 16
Gradient accumulation : 1
```

This provides 15 in-batch negatives per example.

---

# 10. English-MRL Training

The final English-MRL training corpus contained:

```text
20,240 unlabeled English sentences
```

The training configuration was:

```text
Model                   : LingoIITGN/FlowVec-v1
Training                : Full-parameter fine-tuning
Maximum sequence length : 256
Batch size              : 16
Gradient accumulation   : 1
Epochs                  : 5
Learning rate           : 2e-5
Minimum LR              : 2e-6
Warmup ratio            : 0.10
Temperature             : 0.05
Random seed             : 42
Precision               : BF16
Gradient checkpointing  : Enabled
Optimizer               : Adafactor
```

The final checkpoint was:

```text
results/flowvec_mrl/FlowVec_MRL_final.pt
```

The English-MRL training completed all five epochs.

---

# 11. English-MRL Results

The final English-MRL results were:

| Dimension | Accuracy | Macro-F1 |
|----------:|---------:|---------:|
| 16   | 0.556235 | 0.553465 |
| 32   | 0.641900 | 0.645572 |
| 64   | 0.681235 | 0.685330 |
| 128  | 0.703963 | 0.707956 |
| 256  | 0.718823 | 0.723270 |
| 512  | 0.731061 | 0.735192 |
| 768  | 0.734266 | 0.738527 |
| 1024 | 0.734266 | 0.738740 |
| 2048 | 0.739802 | 0.744383 |
| 4096 | 0.742716 | 0.747339 |

Compared with frozen FlowVec, English-MRL preserves essentially the same full-dimensional performance while producing a useful compact representation.

For example:

```text
512D  : 73.11% accuracy
4096D : 74.27% accuracy
```

Thus, most of the downstream information is already available well below the original 4096-dimensional representation.

---

# 12. Moving to Indic Languages

The second major objective was to investigate whether MRL could be trained for Indic-language representations.

The selected languages were:

```text
Hindi
Bengali
Gujarati
Tamil
Telugu
```

The selection was fixed in advance and was intended to provide representation across multiple Indic scripts and both Indo-Aryan and Dravidian language groups.

The experiment was deliberately limited to five languages rather than the full set of languages available in IndicSentiment.

---

# 13. IndicSentiment Dataset

The downstream dataset used was:

```text
mteb/IndicSentiment
```

The selected five languages were:

```text
hi
bn
gu
ta
te
```

The prepared downstream data contained:

```text
156 training examples per language
998 test examples per language
```

The original dataset contains 1000 test examples per language, but two rows per language were removed during preparation because rows with missing `text` or `label` were dropped.

Therefore, the final prepared test split contains 998 examples per language.

This difference is intentional and should not be changed back to 1000.

---

# 14. Critical Distinction — Encoder Training vs Downstream Training

There are two completely different datasets in the Indic experiment.

## Encoder training

Indic-MRL was trained on:

```text
20,240 unlabeled sentences
```

from AI4Bharat IndicCorpV2.

## Downstream probe

The labeled IndicSentiment data was used only for:

```text
Logistic Regression
```

The 156 examples per language were therefore **not** used to train the Indic-MRL encoder.

The final pipeline is:

```text
IndicCorpV2
20,240 unlabeled sentences
        |
        v
Indic-MRL training
        |
        v
FlowVec_MRL_Indic_final.pt
        |
        v
IndicSentiment embeddings
        |
        +---- 156 train examples ---> Logistic Regression
        |
        +---- 998 test examples ----> Evaluation
```

This distinction is critical when interpreting the results.

---

# 15. IndicCorpV2 Data Preparation

A large unlabeled Indic corpus was required for Indic-MRL training.

AI4Bharat IndicCorpV2 was selected.

The complete corpus is very large, so using the entire corpus was not practical for this experiment.

The required raw files were located in the dataset's `data/` area, including files such as:

```text
hi-1.txt
hi-2.txt
hi-3.txt
bn.txt
gu.txt
ta.txt
te.txt
```

An initial attempt incorrectly assumed that language-specific directories existed on Hugging Face, using paths such as:

```text
data/hi/
```

This resulted in:

```text
EmptyDatasetError
```

because the dataset did not have that directory structure.

The data preparation script was therefore changed to access the actual raw files.

Because the complete files are extremely large, bounded prefixes were read and sentences were sampled from them.

A balanced corpus was constructed with:

| Language | Sentences |
|---|---:|
| Hindi | 4,048 |
| Bengali | 4,048 |
| Gujarati | 4,048 |
| Tamil | 4,048 |
| Telugu | 4,048 |
| **Total** | **20,240** |

The random seed was:

```text
42
```

This is a balanced practical sample from bounded portions of the source files. It should not be described as a uniformly random sample from the entire IndicCorpV2 corpus.

---

# 16. Indic-MRL Training

The final Indic-MRL training corpus was:

```text
20,240 unlabeled sentences
```

Training used:

```text
Model                   : LingoIITGN/FlowVec-v1
Training                : Full-parameter fine-tuning
Maximum sequence length : 256
Batch size              : 16
Gradient accumulation   : 1
Epochs                  : 5
Learning rate           : 2e-5
Minimum LR              : 2e-6
Warmup ratio            : 0.10
Temperature             : 0.05
Random seed             : 42
Precision               : BF16
Gradient checkpointing  : Enabled
Optimizer               : Adafactor
```

The final checkpoint is:

```text
results/flowvec_mrl_indic/FlowVec_MRL_Indic_final.pt
```

The checkpoint is approximately 15 GB and is intentionally excluded from the lightweight group-share package.

---

# 17. Indic-MRL Training Bottleneck — Slurm Wall Time

The first long Indic-MRL training allocation did not provide enough wall time to finish all five epochs.

The training completed:

```text
Epoch 1: complete
Epoch 2: approximately 91% complete
```

before Slurm terminated the job because the allocation reached its time limit.

The epoch-1 checkpoint had already been saved:

```text
flowvec_mrl_indic_epoch_1.pt
```

The resume script was then used to restore:

```text
model_state_dict
optimizer_state_dict
scheduler_state_dict
```

and continue from the beginning of Epoch 2.

The partial Epoch-2 progress was not treated as safely resumable because checkpoints were saved at epoch boundaries rather than continuously.

Training was therefore resumed from the completed Epoch-1 checkpoint.

The final five-epoch Indic-MRL training then completed successfully.

---

# 18. Indic-MRL Loss

The final Indic-MRL implementation uses a symmetric/bidirectional InfoNCE formulation.

For each nested dimension, the similarity matrix is computed as:

```text
q @ k.T / temperature
```

and the loss is evaluated in both directions:

```text
q -> k
k -> q
```

with the two losses averaged.

The nested MRL dimensions are:

```text
16, 32, 64, 128, 256, 512, 768, 1024, 2048, 4096
```

The final Indic training run used this formulation consistently; the objective was not changed midway through the run.

---

# 19. Final Indic-MRL Embedding Extraction

After training, the final checkpoint was used to extract embeddings for all five Indic languages.

Extraction configuration:

```text
Model                   : FlowVec-v1 + Indic-MRL checkpoint
Maximum sequence length : 256
Batch size              : 8
Precision               : BF16
Pooling                 : Last token
Normalization           : L2
Saved dtype             : float32
```

Example final extraction:

```text
Telugu train : (156, 4096)
Telugu test  : (998, 4096)
```

The resulting embeddings had L2 norms approximately equal to 1.0.

---

# 20. Final Indic-MRL Results

The final Indic-MRL results averaged over the five languages are:

| Dimension | Accuracy | Macro-F1 |
|----------:|---------:|---------:|
| 16   | 0.502605 | 0.351212 |
| 32   | 0.898196 | 0.897617 |
| 64   | 0.956713 | 0.956707 |
| 128  | 0.967134 | 0.967129 |
| 256  | 0.971944 | 0.971929 |
| 512  | 0.970341 | 0.970324 |
| 768  | 0.971944 | 0.971928 |
| 1024 | 0.971543 | 0.971525 |
| 2048 | 0.972345 | 0.972325 |
| 4096 | 0.972545 | 0.972526 |

The strongest observation is the rapid transition from 16D to 32D and the subsequent saturation:

```text
16D     ~50.3%
32D     ~89.8%
64D     ~95.7%
128D    ~96.7%
256D    ~97.2%
4096D   ~97.3%
```

Thus, the 256-dimensional representation retains almost all of the downstream performance of the full 4096-dimensional representation.

---

# 21. Frozen FlowVec vs English-MRL vs Indic-MRL on Indic Data

The five-language mean results provide the final cross-condition comparison.

## Frozen FlowVec

| Dimension | Accuracy | Macro-F1 |
|----------:|---------:|---------:|
| 16   | 0.502806 | 0.351626 |
| 32   | 0.905210 | 0.904761 |
| 64   | 0.958116 | 0.958112 |
| 128  | 0.967735 | 0.967731 |
| 256  | 0.971944 | 0.971930 |
| 512  | 0.970541 | 0.970525 |
| 768  | 0.971543 | 0.971527 |
| 1024 | 0.971543 | 0.971526 |
| 2048 | 0.972345 | 0.972327 |
| 4096 | 0.972144 | 0.972125 |

## English-MRL

| Dimension | Accuracy | Macro-F1 |
|----------:|---------:|---------:|
| 16   | 0.502204 | 0.350380 |
| 32   | 0.894790 | 0.894139 |
| 64   | 0.956713 | 0.956707 |
| 128  | 0.966733 | 0.966728 |
| 256  | 0.971343 | 0.971327 |
| 512  | 0.970341 | 0.970323 |
| 768  | 0.971543 | 0.971526 |
| 1024 | 0.970942 | 0.970922 |
| 2048 | 0.971944 | 0.971924 |
| 4096 | 0.971543 | 0.971522 |

## Indic-MRL

| Dimension | Accuracy | Macro-F1 |
|----------:|---------:|---------:|
| 16   | 0.502605 | 0.351212 |
| 32   | 0.898196 | 0.897617 |
| 64   | 0.956713 | 0.956707 |
| 128  | 0.967134 | 0.967129 |
| 256  | 0.971944 | 0.971929 |
| 512  | 0.970341 | 0.970324 |
| 768  | 0.971944 | 0.971928 |
| 1024 | 0.971543 | 0.971525 |
| 2048 | 0.972345 | 0.972325 |
| 4096 | 0.972545 | 0.972526 |

The frozen FlowVec baseline is slightly ahead at several intermediate dimensions, while the three conditions become extremely close at larger dimensions.

The important final finding is therefore not a large improvement over frozen FlowVec, but the preservation of downstream performance at much smaller dimensions.

---

# 22. What Did Not Work

Several approaches were attempted and rejected during development.

## 22.1 AdamW for full FlowVec fine-tuning

**Problem:** optimizer-state memory exceeded the available L40S memory.

**Solution:** use Adafactor.

---

## 22.2 CPU-only training

**Problem:** full-parameter FlowVec training was not practical on CPU, and the relevant Slurm QoS required GPU resources.

**Solution:** use one L40S with gradient checkpointing and memory-efficient optimization.

---

## 22.3 Multi-GPU allocation

**Problem:** a multi-GPU allocation was rejected by:

```text
QOSMaxGRESPerUser
```

**Solution:** optimize the experiment for one L40S.

---

## 22.4 Batch size 1 with gradient accumulation

**Problem:** gradient accumulation does not create in-batch negatives for the InfoNCE loss.

**Solution:**

```text
batch size = 16
gradient accumulation = 1
```

---

## 22.5 Incorrect IndicCorpV2 directory assumption

**Problem:** an initial Hugging Face dataset-loading approach expected language directories such as:

```text
data/hi/
```

which did not exist.

**Solution:** use the actual raw files and bounded-prefix sampling.

---

## 22.6 Insufficient Slurm wall time

**Problem:** the first Indic-MRL run was terminated after Epoch 1 and most of Epoch 2.

**Solution:** save epoch-level checkpoints and resume from the completed Epoch-1 checkpoint using a longer allocation.

---

## 22.7 Intermediate Indic-MRL experiment using the labeled IndicSentiment split

An earlier Indic-MRL experiment used the very small labeled IndicSentiment training split.

This was not retained as the final encoder-training experiment.

The final Indic-MRL encoder was trained correctly on:

```text
20,240 unlabeled IndicCorpV2 sentences
```

The IndicSentiment labels are used only for downstream Logistic Regression.

Any earlier results based on training the encoder directly from the 156-example-per-language dataset should be considered intermediate and should not be used as final A2 results.

---

# 23. Final Experimental Matrix

The final experiment can be summarized as:

| Representation | English evaluation | Indic evaluation |
|---|---:|---:|
| Frozen FlowVec | Yes | Yes |
| English-MRL | Yes | Yes |
| Indic-MRL | Not the primary final evaluation | Yes |

The primary final Indic comparison is:

```text
Frozen FlowVec
English-MRL
Indic-MRL
```

on the same five-language Indic downstream evaluation.

---

# 24. Final Repository Structure

The lightweight group-share package contains:

```text
A2_FlowVec_MRL_Final/
│
├── README.md
│
├── code/
│   ├── evaluate_frozen_indic.py
│   ├── evaluate_mrl_indic.py
│   ├── evaluate_mrl.py
│   ├── extract_flowvec_embeddings.py
│   ├── extract_frozen_indic_embeddings.py
│   ├── extract_indic_embeddings.py
│   ├── flowvec_smoke_test.py
│   ├── mrl_flowvec.py
│   ├── prepare_indic_mrl.py
│   ├── test_indic_dataset.py
│   ├── train_flowvec_mrl.py
│   ├── train_flowvec_mrl_indic.py
│   └── train_flowvec_mrl_indic_resume.py
│
├── slurm/
│   ├── flowvec_encode_english.slurm
│   ├── flowvec_encode_indic.slurm
│   ├── flowvec_mrl.slurm
│   └── flowvec_smoke.slurm
│
├── data/
│   ├── final_data.csv
│   ├── paper_train.csv
│   ├── tweet_train.csv
│   ├── tweet_test.csv
│   ├── indic/
│   └── indic_mrl/
│
└── results/
    ├── english_mrl_results.csv
    ├── indic_frozen_results.csv
    └── indic_mrl_results.csv
```

No notebooks are required in the final lightweight package.

---

# 25. Large Artifacts Intentionally Excluded

The following are intentionally excluded because they are very large or machine/environment-specific:

```text
*.pt
*.npy
IndicCorpV2 raw corpus
Hugging Face model cache
Conda/Miniconda installers
Slurm stdout/stderr logs
temporary files
```

The most important excluded artifact is:

```text
results/flowvec_mrl_indic/FlowVec_MRL_Indic_final.pt
```

which is approximately 15 GB.

It should be transferred separately if another group member needs to perform embedding extraction without retraining.

---

# 26. Final Status

The core Assignment 2 experiment is complete.

Completed:

- FlowVec-v1 integration
- Frozen FlowVec baseline
- English-MRL training
- English downstream evaluation
- Indic dataset preparation
- IndicCorpV2 unlabeled training corpus construction
- Indic-MRL full-parameter training
- Slurm checkpoint/resume workflow
- Indic-MRL embedding extraction
- Frozen Indic embedding extraction
- Indic downstream evaluation
- Frozen vs English-MRL vs Indic-MRL comparison
- Final result CSVs
- Reproducible training/evaluation scripts
- Final experiment documentation

No additional training run is required for the final experiment unless a separate controlled ablation is explicitly desired.

---

# 27. Key Final Takeaway

The central result of the experiment is that FlowVec-v1 can be adapted to the Matryoshka Representation Learning setting while retaining almost all of its downstream sentiment performance in substantially smaller representations.

For the final five-language Indic evaluation:

```text
Indic-MRL

256D  : 97.1944% accuracy
4096D : 97.2545% accuracy
```

The difference is only approximately:

```text
0.06 percentage points
```

Therefore, the 256-dimensional representation retains essentially the same performance as the original 4096-dimensional representation for this downstream task.

This demonstrates that the learned Matryoshka representation can provide substantial dimensionality reduction while preserving downstream task performance.
