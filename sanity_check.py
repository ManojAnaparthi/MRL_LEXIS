"""Sanity check verifying all four models before full TweetEval training.

Verifies:
1. MRL Baseline (TinyBERT-6L + MatryoshkaInfoNCE)
2. MRL + SIA (TinyBERT-6L + MRL + SIA, alpha=0.4)
3. MRL + PIC (TinyBERT-6L + MRL + PIC, alpha=0.4, 5 PIC pairs)
4. MIPIC (TinyBERT-6L + MRL + SIA + PIC, alpha=0.4)

Each model undergoes:
- 1 step forward pass on real tokenized data
- Loss calculation and finite value verification
- Backward pass with gradient existence verification
- Optimizer step verification
- 7-dimension linear probe evaluation smoke test on all MRL dimensions:
  [16, 32, 64, 128, 256, 512, 768]
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import pandas as pd
import torch

from evaluate import MRLEvaluationConfig, evaluate_tweeteval
from losses.mrl_loss import MRL_NESTED_DIMS, MatryoshkaInfoNCELoss
from models.mipic import TinyBERT_MIPIC, create_mipic_optimizer
from models.mrl_pic import TinyBERT_MRL_PIC, create_mrl_pic_optimizer
from models.mrl_sia import TinyBERT_MRL_SIA, create_mrl_sia_optimizer
from models.tinybert import load_tinybert_6l
from train import MRLBaselineConfig, PairTokenizerCollate, TextPairRaw, create_mrl_optimizer_scheduler


def run_sanity_checks() -> bool:
    print("=" * 60)
    print("RUNNING STEP 6 PRE-FLIGHT SANITY CHECKS (4 MODELS)")
    print("=" * 60)

    device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Prepare sample batch from real tweet data
    train_csv = Path("data/paper_train.csv")
    assert train_csv.exists(), f"Missing {train_csv}"
    sample_df = pd.read_csv(train_csv).head(8)
    sample_texts = sample_df["text"].tolist()

    _, tokenizer = load_tinybert_6l(device)
    collate = PairTokenizerCollate(tokenizer, max_length=64)
    raw_ds = TextPairRaw(sample_texts)
    batch = collate([raw_ds[i] for i in range(len(raw_ds))])

    # -------------------------------------------------------------
    # 1. MRL Baseline Sanity Check
    # -------------------------------------------------------------
    print("\n[1/4] Checking MRL Baseline...")
    mrl_model, _ = load_tinybert_6l(device)
    mrl_model.train()
    mrl_loss_fn = MatryoshkaInfoNCELoss()
    mrl_opt = torch.optim.AdamW(mrl_model.parameters(), lr=2e-5)

    mrl_opt.zero_grad()
    o1 = mrl_model(batch["input_ids1_stu"], batch["attention_mask1_stu"], output_hidden_states=True, return_dict=True)
    o2 = mrl_model(batch["input_ids2_stu"], batch["attention_mask2_stu"], output_hidden_states=True, return_dict=True)
    mrl_loss, _ = mrl_loss_fn(o1.pooler_output, o2.pooler_output)

    assert torch.isfinite(mrl_loss), f"MRL loss is not finite: {mrl_loss.item()}"
    mrl_loss.backward()

    encoder_grads = [p.grad for p in mrl_model.parameters() if p.grad is not None]
    assert len(encoder_grads) > 0, "MRL parameters did not receive gradients."
    mrl_opt.step()
    print(f"  -> MRL Loss: {mrl_loss.item():.4f} (Finite, gradients verified, optimizer stepped)")

    # -------------------------------------------------------------
    # 2. MRL + SIA Sanity Check
    # -------------------------------------------------------------
    print("\n[2/4] Checking MRL + SIA Model...")
    mrl_sia = TinyBERT_MRL_SIA(device=device)
    mrl_sia.train()
    sia_opt = create_mrl_sia_optimizer(mrl_sia)

    sia_opt.zero_grad()
    sia_losses = mrl_sia.compute_losses(batch=batch)
    loss_sia_total = sia_losses["total_loss"]
    assert torch.isfinite(loss_sia_total), f"MRL+SIA loss not finite: {loss_sia_total.item()}"
    loss_sia_total.backward()

    sia_grads = [p.grad for p in mrl_sia.parameters() if p.grad is not None]
    assert len(sia_grads) > 0, "MRL+SIA parameters did not receive gradients."
    sia_opt.step()
    print(f"  -> MRL+SIA Loss: {loss_sia_total.item():.4f} (MRL: {sia_losses['mrl_loss'].item():.4f}, SIA: {sia_losses['sia_loss'].item():.4f})")

    # -------------------------------------------------------------
    # 3. MRL + PIC Sanity Check
    # -------------------------------------------------------------
    print("\n[3/4] Checking MRL + PIC Model...")
    mrl_pic = TinyBERT_MRL_PIC(device=device)
    mrl_pic.train()
    pic_opt = create_mrl_pic_optimizer(mrl_pic)

    pic_opt.zero_grad()
    pic_losses = mrl_pic.compute_losses(batch=batch)
    loss_pic_total = pic_losses["loss"]
    assert torch.isfinite(loss_pic_total), f"MRL+PIC loss not finite: {loss_pic_total.item()}"
    loss_pic_total.backward()

    pic_grads = [p.grad for p in mrl_pic.parameters() if p.grad is not None]
    assert len(pic_grads) > 0, "MRL+PIC parameters did not receive gradients."
    pic_opt.step()
    print(f"  -> MRL+PIC Loss: {loss_pic_total.item():.4f} (MRL: {pic_losses['mrl_loss'].item():.4f}, PIC: {pic_losses['pic_loss'].item():.4f})")

    # -------------------------------------------------------------
    # 4. MIPIC Sanity Check
    # -------------------------------------------------------------
    print("\n[4/4] Checking MIPIC Model...")
    mipic = TinyBERT_MIPIC(device=device)
    mipic.train()
    mipic_opt = create_mipic_optimizer(mipic)

    mipic_opt.zero_grad()
    mipic_losses = mipic.compute_losses(batch=batch)
    loss_mipic_total = mipic_losses["loss"]
    assert torch.isfinite(loss_mipic_total), f"MIPIC loss not finite: {loss_mipic_total.item()}"
    loss_mipic_total.backward()

    mipic_grads = [p.grad for p in mipic.parameters() if p.grad is not None]
    assert len(mipic_grads) > 0, "MIPIC parameters did not receive gradients."
    mipic_opt.step()
    print(f"  -> MIPIC Loss: {loss_mipic_total.item():.4f} (MRL: {mipic_losses['mrl_loss'].item():.4f}, SIA: {mipic_losses['sia_loss'].item():.4f}, PIC: {mipic_losses['pic_loss'].item():.4f})")

    # -------------------------------------------------------------
    # 5. Linear Probe Evaluation Smoke Test (all 7 dimensions)
    # -------------------------------------------------------------
    print("\n[5/5] Smoke testing 7-dimension Linear Probe evaluation...")
    smoke_train_csv = Path("data/smoke_tweet_train.csv")
    smoke_test_csv = Path("data/smoke_tweet_test.csv")

    full_train = pd.read_csv("data/tweet_train.csv")
    full_test = pd.read_csv("data/tweet_test.csv")

    smoke_train = full_train.head(60)
    smoke_test = full_test.head(30)
    smoke_train.to_csv(smoke_train_csv, index=False)
    smoke_test.to_csv(smoke_test_csv, index=False)

    eval_config = MRLEvaluationConfig(
        dimensions=MRL_NESTED_DIMS,
        batch_size=32,
        max_length=64,
    )
    results = evaluate_tweeteval(
        model=mipic.encoder,
        tokenizer=tokenizer,
        train_csv=smoke_train_csv,
        test_csv=smoke_test_csv,
        config=eval_config,
    )

    smoke_train_csv.unlink(missing_ok=True)
    smoke_test_csv.unlink(missing_ok=True)

    print("  -> Evaluated dimensions:", sorted(results.keys()))
    assert len(results) == len(MRL_NESTED_DIMS), f"Expected {len(MRL_NESTED_DIMS)} dims, got {len(results)}"
    assert 128 in results, "Dimension 128 missing from evaluation!"

    for dim in MRL_NESTED_DIMS:
        assert dim in results, f"Dimension {dim} missing"
        assert "accuracy" in results[dim], f"Accuracy missing for dim {dim}"
        assert "f1" in results[dim], f"F1 missing for dim {dim}"
        print(f"     Dim {dim:3d}: Acc = {results[dim]['accuracy']:.4f}, F1 = {results[dim]['f1']:.4f}")

    print("\n" + "=" * 60)
    print("ALL PRE-FLIGHT SANITY CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    run_sanity_checks()

