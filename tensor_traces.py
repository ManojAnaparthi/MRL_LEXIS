"""Tiny synthetic shape/gradient traces for the three MIPIC objectives.

Run with:  python tensor_traces.py

The traces use the sizes requested by the audit brief:
    MRL : B=2, D=8, nested dims = [2, 4, 8]
    SIA : B=2, L=5, D=8
    PIC : B=2, d_i=4, d_{i+1}=8
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from losses.mrl_loss import Matry_infonce
from losses.pic_loss import PipelineInfoNCELoss
from losses.sia_loss import (
    SIALoss,
    TokenImportanceAttention,
    _topk_indices,
    attention_kl,
    contextual_token_mask,
    linear_cka,
)


def _rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def trace_mrl() -> None:
    _rule("MRL trace  --  Eq. (1) x Eq. (19):  B=2, D=8, nested dims [2, 4, 8]")
    torch.manual_seed(0)
    B, D, dims = 2, 8, (2, 4, 8)

    view_a = torch.randn(B, D, requires_grad=True)   # dropout view z
    view_b = torch.randn(B, D, requires_grad=True)   # dropout view z'
    print(f"view_a (h_CLS of view 1)      shape {tuple(view_a.shape)}")
    print(f"view_b (h_CLS of view 2)      shape {tuple(view_b.shape)}")

    total = 0.0
    for d in dims:
        q_raw, k_raw = view_a[:, :d], view_b[:, :d]
        q, k = F.normalize(q_raw, dim=-1), F.normalize(k_raw, dim=-1)
        logits = q @ k.T / 0.05
        labels = torch.arange(B)
        term = F.cross_entropy(logits, labels)
        total = total + term
        print(
            f"  d={d:<2} prefix slice {tuple(q_raw.shape)} -> normalize {tuple(q.shape)}"
            f" -> logits {tuple(logits.shape)} (= q k^T / tau, tau=0.05)"
            f"  labels {labels.tolist()}  CE={term.item():.4f}"
        )
    loss, logits_map = Matry_infonce(view_a, view_b, 0.05, dims)
    print(f"  sum over prefixes (Eq. 1, unweighted) = {loss.item():.4f}   (manual {total.item():.4f})")
    print(f"  returned logit keys: {sorted(logits_map)}")
    print("  reduction: cross_entropy default 'mean' over the batch, then summed over d")

    loss.backward()
    print(f"  grad path -> view_a {tuple(view_a.grad.shape)} |g|={view_a.grad.norm():.4f}")
    print(f"  grad path -> view_b {tuple(view_b.grad.shape)} |g|={view_b.grad.norm():.4f}")
    print("  under DDP: keys are all_gathered (differentiable), labels become")
    print("             arange(rank*B, (rank+1)*B) so the diagonal stays the positive")


def trace_sia() -> None:
    _rule("SIA trace  --  Eq. (4)-(13):  B=2, L=5, D=8, prefix d_i=4")
    torch.manual_seed(1)
    B, L, D, d_i = 2, 5, 8, 4

    hidden = torch.randn(B, L, D, requires_grad=True)
    attention_mask = torch.ones(B, L, dtype=torch.long)
    special_tokens_mask = torch.zeros(B, L, dtype=torch.long)
    special_tokens_mask[:, 0] = 1    # [CLS]
    special_tokens_mask[:, -1] = 1   # [SEP]
    token_mask = contextual_token_mask(attention_mask, special_tokens_mask)
    print(f"hidden (layer k)              shape {tuple(hidden.shape)}")
    print(f"attention_mask                shape {tuple(attention_mask.shape)}  {attention_mask[0].tolist()}")
    print(f"special_tokens_mask           shape {tuple(special_tokens_mask.shape)}  {special_tokens_mask[0].tolist()}")
    print(f"contextual key mask (Eq. 4 j) shape {tuple(token_mask.shape)}  {token_mask[0].tolist()}")

    module = TokenImportanceAttention(d_small=d_i, d_full=D)
    h_cls = hidden[:, 0, :]
    print(f"h_CLS (query anchor, detached) shape {tuple(h_cls.shape)}")

    teacher_probs, teacher_scores = module.teacher_scores(hidden, token_mask, temperature=1.0)
    print(f"teacher scores s^(D)          shape {tuple(teacher_scores.shape)}   = h_CLS . h_j / sqrt({D}) = /{math.sqrt(D):.4f}")
    print(f"teacher probs a_D             shape {tuple(teacher_probs.shape)}   row0={teacher_probs[0].tolist()}")
    print(f"teacher requires_grad         {teacher_probs.requires_grad}  (fixed reference, Sec. 3.2.1)")

    small = hidden[..., :d_i]
    student_probs, student_scores = module.student_scores(small, h_cls.detach(), token_mask, temperature=1.0)
    print(f"student slice h_j^(i)         shape {tuple(small.shape)}")
    print(f"P_i^T (proj_small)            shape {tuple(module.proj_small.weight.shape)}  (Linear {d_i} -> {D}, no bias)")
    print(f"student scores s^(i)          shape {tuple(student_scores.shape)}   = h_CLS . P_i^T h_j^(i) / sqrt({D})")
    print(f"student probs a_i             shape {tuple(student_probs.shape)}   requires_grad={student_probs.requires_grad}")

    kl = attention_kl(student_probs, teacher_probs)
    print(f"L_att = KL(a_i || a_D)        scalar {kl.item():.6f}   (direction: student || teacher)")

    m = int(token_mask.sum(dim=1).max())
    sia = SIALoss(D, (d_i, D), (1,), temperature=1.0, topk_ratios={d_i: 0.5})
    k = sia._topk_count(d_i, m)
    idx = _topk_indices(teacher_scores, k, token_mask)
    print(f"sequence length m (contextual) {m}")
    print(f"k_i = max(8, ceil(0.5*m))     {k}  -> clamped to available tokens: {idx.shape[1]}")
    print(f"top-k indices (teacher-ranked) shape {tuple(idx.shape)}  row0={idx[0].tolist()}")

    student_sub = torch.gather(small, 1, idx.unsqueeze(-1).expand(-1, -1, d_i))
    teacher_sub = torch.gather(hidden.detach(), 1, idx.unsqueeze(-1).expand(-1, -1, D))
    print(f"CKA student h_i               shape {tuple(student_sub.shape)}  (k_i x d_i)")
    print(f"CKA teacher H_i               shape {tuple(teacher_sub.shape)}  (k_i x D)")
    cka = linear_cka(student_sub, teacher_sub)
    print(f"CKA(h_i, H_i)                 {cka.item():.6f}   L_CKA = 1 - CKA = {(1 - cka).item():.6f}")

    out = sia([hidden, hidden], attention_mask, special_tokens_mask=special_tokens_mask)
    print(f"aggregate (Eq. 13/14): sum over {len(sia.matryoshka_dims)} dims x "
          f"{len(sia.align_layers)} layer(s) = {out['n_terms']} (att, cka) pairs")
    print(f"  attn_loss={out['attn_loss'].item():.4f}  cka_loss={out['cka_loss'].item():.4f}  "
          f"sia_loss={out['sia_loss'].item():.4f}")
    out["sia_loss"].backward()
    print(f"  grad path -> hidden {tuple(hidden.grad.shape)} |g|={hidden.grad.norm():.4f}")
    print(f"  grad path -> P_i    {tuple(module.proj_small.weight.shape)} (trained jointly, Eq. 5)")
    print("  full TinyBERT-6L run: dims [512,256,128,64,32,16] x layers [1..6] = 36 pairs")


def trace_pic() -> None:
    _rule("PIC trace  --  Eq. (15)-(17):  B=2, d_i=4, d_{i+1}=8")
    torch.manual_seed(2)
    B, L, d_i, d_j = 2, 5, 4, 8

    src_layer = torch.randn(B, L, d_i, requires_grad=True)   # hidden_states[l_i][..., :d_i]
    tgt_layer = torch.randn(B, L, d_j, requires_grad=True)   # hidden_states[l_{i+1}][..., :d_{i+1}]
    module = PipelineInfoNCELoss(d_src=d_i, d_tgt=d_j, d_hidden=max(d_i, d_j) // 2)
    print(f"src hidden (layer l_i)        shape {tuple(src_layer.shape)}")
    print(f"tgt hidden (layer l_i+1)      shape {tuple(tgt_layer.shape)}")

    z_i = src_layer[:, 0, :]
    z_j = tgt_layer[:, 0, :]
    print(f"z_i  = truncated [CLS]        shape {tuple(z_i.shape)}")
    print(f"z_i+1= truncated [CLS]        shape {tuple(z_j.shape)}")
    print(f"phi_i: Linear({d_i}->{max(d_i, d_j) // 2}) -> ReLU -> Linear({max(d_i, d_j) // 2}->{d_j})")

    u = module.phi(z_i.float())
    print(f"phi_i(z_i)                    shape {tuple(u.shape)}")
    q, k = F.normalize(u, dim=-1), F.normalize(z_j.float(), dim=-1)
    print(f"query  normalize(phi(z_i))    shape {tuple(q.shape)}   (cosine sim)")
    print(f"key    normalize(z_i+1)       shape {tuple(k.shape)}")
    logits = q @ k.T / 0.05
    labels = torch.arange(B)
    print(f"logits = q k^T / tau          shape {tuple(logits.shape)}   tau=0.05")
    print(f"labels (diagonal positives)   {labels.tolist()}   B_i+1 = 1 positive + {B - 1} in-batch negatives")

    loss = module(src_layer, tgt_layer, temperature=0.05)
    print(f"detach_target                 {module.detach_target}   (deeper checkpoint is the teacher)")
    print(f"L_chain^(i)                   {loss.item():.6f}   (reduction: cross_entropy mean over batch)")
    loss.backward()

    def _grad(name, tensor):
        if tensor.grad is None:
            print(f"  grad path -> {name} None  (stop-gradient)")
        else:
            print(f"  grad path -> {name} {tuple(tensor.grad.shape)} |g|={tensor.grad.norm():.4f}")

    _grad("src_layer  ", src_layer)
    _grad("tgt_layer  ", tgt_layer)
    _grad("phi[0].weight", module.phi[0].weight)

    # The symmetric variant, for contrast.
    symmetric = PipelineInfoNCELoss(d_src=d_i, d_tgt=d_j, d_hidden=max(d_i, d_j) // 2,
                                    detach_target=False)
    tgt2 = torch.randn(B, L, d_j, requires_grad=True)
    symmetric(torch.randn(B, L, d_i), tgt2, temperature=0.05).backward()
    print(f"  detach_target=False -> tgt_layer |g|={tgt2.grad.norm():.4f}  (gradient reaches the target)")
    print("  full TinyBERT-6L chain: (1,16)->(2,32)->(3,64)->(4,256)->(5,512)->(6,768)")
    print("  = exactly 5 adjacent transitions, summed unweighted (Eq. 17)")
    print("  under DDP: keys are all_gathered, labels offset by rank*B")


def trace_objective() -> None:
    _rule("Total objective  --  Eq. (18)")
    mrl, sia, pic, alpha = torch.tensor(1.2), torch.tensor(0.7), torch.tensor(0.9), 0.4
    total = alpha * mrl + (1.0 - alpha) * (sia + pic)
    print(f"L_MIPIC = {alpha} * {mrl.item()} + {1 - alpha:.1f} * ({sia.item()} + {pic.item()}) = {total.item():.4f}")


if __name__ == "__main__":
    trace_mrl()
    trace_sia()
    trace_pic()
    trace_objective()
