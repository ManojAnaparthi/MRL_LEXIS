"""
Baseline MRL Experiment
=======================
This script trains a STANDARD MRL model where every dimension block
receives contrastive gradients from ALL samples (no task-specific masking).

Training:  All dimension blocks receive SimCSE contrastive gradients from
           every sample in the batch, regardless of task source.
Evaluation: Every block is evaluated on EVERY available task to produce
           a baseline cross-evaluation matrix for comparison with the
           task-specialization experiment.

Output:    A cross-evaluation matrix (7 blocks × N tasks) saved to
           results_baseline_mrl.json.
"""

import os
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler
from models.tinybert import load_tinybert_6l, load_tinybert_tokenizer, mrl_sentence_embedding
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.stats import spearmanr
import warnings
import numpy as np
import time
import json

# ============================================================
# Constants
# ============================================================
MRL_NESTED_DIMS = (16, 32, 64, 128, 256, 512, 768)
BLOCKS = [(0, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 512), (512, 768)]

# Ordered by difficulty: easiest task → smallest block, hardest → largest block
SOURCES = ["TWEET_EVAL_SENT", "BANKING77", "MRPC", "DEEHAN1866/WIC", "DGEM_FORMAT", "SICK", "STSB"]

# ============================================================
# Training
# ============================================================

class SpecializedDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.samples = [(str(row['text']), str(row['text']), SOURCES.index(row['source'])) for _, row in df.iterrows()]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def collate_fn(batch, tokenizer, max_len=256):
    texts1, texts2, source_ids = zip(*batch)
    enc1 = tokenizer(list(texts1), max_length=max_len, truncation=True, padding=True, return_tensors="pt")
    enc2 = tokenizer(list(texts2), max_length=max_len, truncation=True, padding=True, return_tensors="pt")
    return enc1, enc2, torch.tensor(source_ids)

def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for i, (enc1, enc2, _sources) in enumerate(loader):
        enc1 = {k: v.to(device) for k, v in enc1.items()}
        enc2 = {k: v.to(device) for k, v in enc2.items()}
        
        optimizer.zero_grad()
        out1 = model(**enc1, output_hidden_states=True, return_dict=True)
        out2 = model(**enc2, output_hidden_states=True, return_dict=True)
        
        z1 = mrl_sentence_embedding(out1)
        z2 = mrl_sentence_embedding(out2)
        
        batch_loss = 0
        for block_idx, (start, end) in enumerate(BLOCKS):
            # STANDARD MRL: No masking, all samples train all blocks
            q = z1[:, start:end]
            k = z2[:, start:end] 
            
            q_norm = F.normalize(q, dim=-1)
            k_norm = F.normalize(k, dim=-1)
            logits = torch.matmul(q_norm, k_norm.T) / 0.05
            
            labels = torch.arange(logits.size(0), device=device)
            loss_block = F.cross_entropy(logits, labels)
            batch_loss += loss_block
            
        if batch_loss > 0:
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += batch_loss.item()
            
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  Batch {i+1}/{len(loader)}, Loss: {batch_loss.item():.4f}, Elapsed: {elapsed:.0f}s")

    avg_loss = total_loss / max(1, len(loader))
    print(f"  Epoch finished in {time.time() - start_time:.1f}s, Avg Loss: {avg_loss:.4f}")

# ============================================================
# Embedding Extraction
# ============================================================

def extract_embeddings(model, texts, tokenizer, batch_size=64, max_length=256):
    """Extract full 768-dim embeddings for a list of texts."""
    model.eval()
    device = next(model.parameters()).device
    all_embs = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded = tokenizer(
            batch_texts, max_length=max_length, truncation=True,
            padding=True, return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        with torch.no_grad():
            output = model(**encoded, output_hidden_states=True, return_dict=True)
            emb = mrl_sentence_embedding(output)
            all_embs.append(emb.cpu().numpy())
    
    return np.concatenate(all_embs, axis=0)

# ============================================================
# Per-Task Evaluation Functions
# ============================================================

def eval_single_sentence(model, tokenizer, train_csv, test_csv):
    """Evaluate all blocks on a single-sentence classification task.
    Returns dict[block_idx -> macro-F1]."""
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    train_embs = extract_embeddings(model, train_df["text"].astype(str).tolist(), tokenizer)
    test_embs = extract_embeddings(model, test_df["text"].astype(str).tolist(), tokenizer)
    train_labels = train_df["label"].values
    test_labels = test_df["label"].values
    
    results = {}
    for block_idx, (start, end) in enumerate(BLOCKS):
        clf = LogisticRegression(random_state=42, max_iter=1000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(train_embs[:, start:end], train_labels)
        preds = clf.predict(test_embs[:, start:end])
        results[block_idx] = float(f1_score(test_labels, preds, average="macro"))
    
    return results


def eval_sentence_pair(model, tokenizer, train_csv, test_csv):
    """Evaluate all blocks on a sentence-pair classification task.
    Uses |u - v| feature representation.
    Returns dict[block_idx -> macro-F1]."""
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    train_embs1 = extract_embeddings(model, train_df["sentence1"].astype(str).tolist(), tokenizer)
    train_embs2 = extract_embeddings(model, train_df["sentence2"].astype(str).tolist(), tokenizer)
    test_embs1 = extract_embeddings(model, test_df["sentence1"].astype(str).tolist(), tokenizer)
    test_embs2 = extract_embeddings(model, test_df["sentence2"].astype(str).tolist(), tokenizer)
    train_labels = train_df["label"].values
    test_labels = test_df["label"].values
    
    results = {}
    for block_idx, (start, end) in enumerate(BLOCKS):
        # |u - v| is a standard sentence-pair feature for probing embeddings
        train_feat = np.abs(train_embs1[:, start:end] - train_embs2[:, start:end])
        test_feat = np.abs(test_embs1[:, start:end] - test_embs2[:, start:end])
        
        clf = LogisticRegression(random_state=42, max_iter=1000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(train_feat, train_labels)
        preds = clf.predict(test_feat)
        results[block_idx] = float(f1_score(test_labels, preds, average="macro"))
    
    return results


def eval_sts_regression(model, tokenizer, test_csv):
    """Evaluate all blocks on STS regression using cosine similarity + Spearman.
    No training required — directly measures embedding quality.
    Returns dict[block_idx -> Spearman correlation]."""
    test_df = pd.read_csv(test_csv)
    
    test_embs1 = extract_embeddings(model, test_df["sentence1"].astype(str).tolist(), tokenizer)
    test_embs2 = extract_embeddings(model, test_df["sentence2"].astype(str).tolist(), tokenizer)
    gold_scores = test_df["label"].values.astype(float)
    
    results = {}
    for block_idx, (start, end) in enumerate(BLOCKS):
        u = test_embs1[:, start:end]
        v = test_embs2[:, start:end]
        # Cosine similarity between sentence pairs
        norms = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-8
        cos_sim = np.sum(u * v, axis=1) / norms
        corr, _ = spearmanr(cos_sim, gold_scores)
        results[block_idx] = float(corr) if not np.isnan(corr) else 0.0
    
    return results

# ============================================================
# Dataset Download & Preparation
# ============================================================

def download_eval_datasets():
    """Download labeled evaluation datasets for all available tasks from HuggingFace.
    Returns dict of task configs with type, file paths, and metric info."""
    from datasets import load_dataset
    os.makedirs("data/eval", exist_ok=True)
    
    tasks = {}
    print("\n" + "="*60)
    print("STEP 1: Downloading Evaluation Datasets")
    print("="*60)
    
    # --- 1. TWEET_EVAL_SENT (local CSVs already exist) ---
    tasks["TWEET_EVAL_SENT"] = {
        "type": "single", "metric": "F1",
        "train": "data/tweet_train.csv",
        "test": "data/tweet_test.csv",
    }
    print("  [✓] TWEET_EVAL_SENT: using existing local CSVs")
    
    # --- 2. BANKING77 (single-sentence intent classification, 77 classes) ---
    try:
        ds = load_dataset("PolyAI/banking77", trust_remote_code=True)
        for split, name in [("train", "train"), ("test", "test")]:
            pd.DataFrame({"text": ds[split]["text"], "label": ds[split]["label"]}) \
              .to_csv(f"data/eval/banking77_{name}.csv", index=False)
        tasks["BANKING77"] = {
            "type": "single", "metric": "F1",
            "train": "data/eval/banking77_train.csv",
            "test": "data/eval/banking77_test.csv",
        }
        print(f"  [✓] BANKING77: {len(ds['train'])} train, {len(ds['test'])} test")
    except Exception as e:
        print(f"  [✗] BANKING77: skipped ({e})")
    
    # --- 3. MRPC (sentence-pair paraphrase detection, binary) ---
    try:
        ds = load_dataset("nyu-mll/glue", "mrpc", trust_remote_code=True)
        for split, name in [("train", "train"), ("validation", "test")]:
            pd.DataFrame({
                "sentence1": ds[split]["sentence1"],
                "sentence2": ds[split]["sentence2"],
                "label": ds[split]["label"],
            }).to_csv(f"data/eval/mrpc_{name}.csv", index=False)
        tasks["MRPC"] = {
            "type": "pair", "metric": "F1",
            "train": "data/eval/mrpc_train.csv",
            "test": "data/eval/mrpc_test.csv",
        }
        print(f"  [✓] MRPC: {len(ds['train'])} train, {len(ds['validation'])} test")
    except Exception as e:
        print(f"  [✗] MRPC: skipped ({e})")
    
    # --- 4. WIC (sentence-pair word-in-context, binary) ---
    try:
        ds = load_dataset("aps/super_glue", "wic", trust_remote_code=True)
        for split, name in [("train", "train"), ("validation", "test")]:
            pd.DataFrame({
                "sentence1": ds[split]["sentence1"],
                "sentence2": ds[split]["sentence2"],
                "label": ds[split]["label"],
            }).to_csv(f"data/eval/wic_{name}.csv", index=False)
        tasks["DEEHAN1866/WIC"] = {
            "type": "pair", "metric": "F1",
            "train": "data/eval/wic_train.csv",
            "test": "data/eval/wic_test.csv",
        }
        print(f"  [✓] WIC: {len(ds['train'])} train, {len(ds['validation'])} test")
    except Exception as e:
        print(f"  [✗] WIC: skipped ({e})")
    
    # --- 5. DGEM_FORMAT (SciTail dgem_format configuration) ---
    try:
        try:
            ds = load_dataset("allenai/scitail", "dgem_format", trust_remote_code=True)
        except Exception:
            ds = load_dataset("scitail", "dgem_format", trust_remote_code=True)
        for split, name in [("train", "train"), ("test", "test")]:
            labels = [1 if str(l).strip().lower() in ("entails", "entailment", "1") else 0 for l in ds[split]["label"]]
            pd.DataFrame({
                "sentence1": ds[split]["premise"],
                "sentence2": ds[split]["hypothesis"],
                "label": labels,
            }).to_csv(f"data/eval/dgem_{name}.csv", index=False)
        tasks["DGEM_FORMAT"] = {
            "type": "pair", "metric": "F1",
            "train": "data/eval/dgem_train.csv",
            "test": "data/eval/dgem_test.csv",
        }
        print(f"  [✓] DGEM_FORMAT (SciTail): {len(ds['train'])} train, {len(ds['test'])} test")
    except Exception as e:
        print(f"  [✗] DGEM_FORMAT: skipped ({e})")
    
    # --- 6. SICK (sentence-pair NLI, 3-class: entailment/neutral/contradiction) ---
    try:
        ds = load_dataset("RobZamp/sick", trust_remote_code=True)
        for split, name in [("train", "train"), ("test", "test")]:
            pd.DataFrame({
                "sentence1": ds[split]["sentence_A"],
                "sentence2": ds[split]["sentence_B"],
                "label": ds[split]["label"],
            }).to_csv(f"data/eval/sick_{name}.csv", index=False)
        tasks["SICK"] = {
            "type": "pair", "metric": "F1",
            "train": "data/eval/sick_train.csv",
            "test": "data/eval/sick_test.csv",
        }
        print(f"  [✓] SICK: {len(ds['train'])} train, {len(ds['test'])} test")
    except Exception as e:
        print(f"  [✗] SICK: skipped ({e})")
    
    # --- 7. STSB (sentence-pair similarity regression, scores 0.0-5.0) ---
    try:
        ds = load_dataset("nyu-mll/glue", "stsb", trust_remote_code=True)
        for split, name in [("train", "train"), ("validation", "test")]:
            pd.DataFrame({
                "sentence1": ds[split]["sentence1"],
                "sentence2": ds[split]["sentence2"],
                "label": ds[split]["label"],
            }).to_csv(f"data/eval/stsb_{name}.csv", index=False)
        tasks["STSB"] = {
            "type": "pair_regression", "metric": "Spearman",
            "train": "data/eval/stsb_train.csv",
            "test": "data/eval/stsb_test.csv",
        }
        print(f"  [✓] STSB: {len(ds['train'])} train, {len(ds['validation'])} test")
    except Exception as e:
        print(f"  [✗] STSB: skipped ({e})")
    
    print(f"\n  Total tasks available for evaluation: {len(tasks)}/7")
    return tasks

# ============================================================
# Cross-Evaluation Matrix
# ============================================================

def evaluate_all_tasks(model, tokenizer):
    """Evaluate ALL blocks on ALL available tasks.
    Produces a cross-evaluation matrix and saves results to JSON."""
    
    tasks = download_eval_datasets()
    
    print("\n" + "="*60)
    print("STEP 2: Cross-Evaluation (All Blocks × All Tasks)")
    print("="*60)
    
    matrix = {}  # matrix[task_name][block_idx] = score
    
    for task_name, cfg in tasks.items():
        print(f"\n  Evaluating: {task_name} (metric: {cfg['metric']})...")
        
        if cfg["type"] == "single":
            scores = eval_single_sentence(model, tokenizer, cfg["train"], cfg["test"])
        elif cfg["type"] == "pair":
            scores = eval_sentence_pair(model, tokenizer, cfg["train"], cfg["test"])
        elif cfg["type"] == "pair_regression":
            scores = eval_sts_regression(model, tokenizer, cfg["test"])
        else:
            print(f"    Unknown type: {cfg['type']}, skipping")
            continue
        
        matrix[task_name] = scores
        
        for block_idx, (start, end) in enumerate(BLOCKS):
            assigned_marker = " ◀ ASSIGNED" if SOURCES[block_idx] == task_name else ""
            print(f"    Block {block_idx} ({start:3d}-{end:3d}) | {cfg['metric']}: {scores[block_idx]:.4f}{assigned_marker}")
    
    # ---- Print Summary Table ----
    print("\n" + "="*80)
    print("CROSS-EVALUATION SUMMARY TABLE")
    print("="*80)
    
    task_names = list(matrix.keys())
    metrics = {t: tasks[t]["metric"] for t in task_names}
    
    # Header row
    header = f"{'Block':>14} |"
    for t in task_names:
        short = t[:12]
        header += f" {short:>12} |"
    header += "  Assigned Task"
    print(header)
    print("-" * len(header))
    
    # Data rows
    for block_idx, (start, end) in enumerate(BLOCKS):
        row = f"  {start:3d} - {end:3d}   |"
        for t in task_names:
            score = matrix[t][block_idx]
            marker = " *" if SOURCES[block_idx] == t else "  "
            row += f"    {score:.4f}{marker}  |"
        row += f"  {SOURCES[block_idx]}"
        print(row)
    
    # Metrics row
    metrics_row = f"{'Metric':>14} |"
    for t in task_names:
        metrics_row += f" {metrics[t]:>12} |"
    print("-" * len(header))
    print(metrics_row)
    
    print(f"\n  * = Block that would be assigned to this task in the specialization experiment")
    print(f"  BASELINE: All blocks were trained identically on all tasks.")
    print(f"  Compare these scores against results_task_specialization.json\n")
    
    # ---- Save results to JSON ----
    results_path = "results_baseline_mrl.json"
    results_json = {
        "blocks": [{"index": i, "start": s, "end": e, "assigned_task": SOURCES[i]} 
                   for i, (s, e) in enumerate(BLOCKS)],
        "tasks": {t: {"metric": tasks[t]["metric"], "type": tasks[t]["type"]} for t in task_names},
        "matrix": {t: {str(k): v for k, v in scores.items()} for t, scores in matrix.items()},
    }
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"  Results saved to: {results_path}")
    
    return matrix

# ============================================================
# Main
# ============================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model, _ = load_tinybert_6l()
    model.to(device)
    tokenizer = load_tinybert_tokenizer()
    
    # Training setup
    loader = DataLoader(
        SpecializedDataset('data/final_data.csv'),
        batch_size=16, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    epochs = 1
    scheduler = get_scheduler(
        "cosine_with_min_lr", optimizer=optimizer,
        num_warmup_steps=int(0.1 * epochs * len(loader)),
        num_training_steps=len(loader) * epochs,
        scheduler_specific_kwargs={"min_lr": 2e-6}
    )
    
    # Train
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"BASELINE MRL TRAINING: Epoch {epoch+1}/{epochs}")
        print(f"{'='*60}")
        train_one_epoch(model, loader, optimizer, scheduler, device)
    
    # Evaluate all blocks on all tasks
    print(f"\n{'='*60}")
    print("BASELINE EVALUATION: All Blocks × All Tasks")
    print(f"{'='*60}")
    evaluate_all_tasks(model, tokenizer)

if __name__ == "__main__":
    main()
