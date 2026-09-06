import torch

from losses.pic_loss import PICModule
from losses.sia_loss import SIALoss
from models.mrl_sia_pic import TINYBERT_PIC_CHECKPOINTS
from models.mrl_sia import MRLSIAConfig


def test_tinybert_pic_checkpoints_intentionally_omit_128():
    assert TINYBERT_PIC_CHECKPOINTS == (
        (1, 16), (2, 32), (3, 64), (4, 256), (5, 512), (6, 768)
    )


def test_sia_projections_are_shared_by_dimension_across_layers():
    sia = SIALoss(768, [16, 32, 64, 128, 256, 512, 768], [1, 2, 3, 4, 5, 6])
    assert set(sia.attn_modules.keys()) == {
        "dim_16", "dim_32", "dim_64", "dim_128", "dim_256", "dim_512"
    }


def test_sia_temperature_is_paper_temperature():
    assert MRLSIAConfig().temperature == 0.05


def test_sia_loss_averages_terms():
    torch.manual_seed(7)
    sia = SIALoss(4, [2], [1], temperature=0.05, topk_ratios={2: 1.0})
    hidden = [torch.zeros(2, 4, 4), torch.randn(2, 4, 4)]
    mask = torch.ones(2, 4, dtype=torch.long)
    out = sia(hidden, mask)
    # sia_loss is still attn + cka; each of those is now a mean over terms.
    assert torch.allclose(out["sia_loss"], out["attn_loss"] + out["cka_loss"])


def test_pic_pairs_are_averaged_and_the_source_receives_gradient():
    torch.manual_seed(9)
    pic = PICModule(
        d_full=4,
        pipeline_pairs=[(1, 2, 2, 4, 3, 4)],
        temperature=0.05,
    )
    hidden = [torch.randn(3, 5, 4, requires_grad=True) for _ in range(4)]
    out = pic(hidden)
    total = out["total_loss"]
    pairs = out["pair_losses"]
    assert torch.allclose(total, sum(pairs.values()) / len(pairs))
    total.backward()
    # hidden[2] is the SOURCE of the second transition, so it gets a gradient
    # even though it is also the (detached) target of the first.
    assert hidden[2].grad is not None
    assert hidden[2].grad.abs().sum() > 0
