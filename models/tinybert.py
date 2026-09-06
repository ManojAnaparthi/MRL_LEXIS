"""TinyBERT-6L configuration used by the existing MRL baseline."""

from typing import Optional, Tuple

import torch


TINYBERT_6L_MODEL_NAME = "huawei-noah/TinyBERT_General_6L_768D"
TINYBERT_6L_HIDDEN_SIZE = 768
TINYBERT_6L_NUM_HIDDEN_LAYERS = 6


def load_tinybert_tokenizer() -> object:
    """Load only the TinyBERT tokenizer without materializing the 287 MB model."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TINYBERT_6L_MODEL_NAME)


# THE sentence embedding z. One value, used by BOTH the MRL loss and the
# evaluation probe -- there is deliberately no switch.
#
# Matryoshka nesting is a property of one specific vector: Eq. (1) supervises the
# prefixes z_{1:d_i} and inference truncates z. Enforcing nestedness on one
# vector and truncating another measures a structure that was never trained, so
# the two read-outs can never be allowed to drift apart.
#
# "pooler_output" is HuggingFace's Linear+Tanh head. Appendix A.3 instantiates
# the MRL task loss as L_SimCSE "adopted from (Gao et al., 2022)", whose head
# this is. Measured on 2xT4, MRL arm, pooled corpus, TweetEval macro-F1 at
# d=768: this read-out 65.59, the raw final [CLS] 61.15 -- and the [CLS] curve
# was non-monotonic in dimension (peaked at 128 and declined), which is the
# signature of an unsupervised high-dimensional tail.
#
# SIA and PIC always read layer hidden states instead (paper Sec. 3.2.1, 3.3).
MRL_READOUT: str = "pooler_output"


def mrl_sentence_embedding(output: object) -> "torch.Tensor":
    """Return z, the sentence embedding, from an encoder output.

    Used identically by the MRL loss and by ``evaluate.extract_embeddings``, so
    the trained and measured vectors cannot diverge.
    """
    pooled = getattr(output, "pooler_output", None)
    if pooled is not None:
        return pooled
    hidden = getattr(output, "hidden_states", None)
    if hidden is None:
        raise ValueError(
            "Encoder output exposes neither pooler_output nor hidden_states; "
            "load the backbone with its pooling layer."
        )
    # Lightweight test doubles without a pooler fall back to the final [CLS].
    return hidden[-1][:, 0, :]


def load_tinybert_6l(device: Optional[torch.device] = None) -> Tuple[object, object]:
    """Load the baseline TinyBERT encoder and tokenizer.

    ``output_hidden_states=True`` is retained from ``mrl-baseline-a1.ipynb`` so
    later SIA/PIC work can consume the same model outputs without replacing the
    baseline backbone.
    """
    from transformers import AutoModel

    tokenizer = load_tinybert_tokenizer()
    model = AutoModel.from_pretrained(
        TINYBERT_6L_MODEL_NAME,
        output_hidden_states=True,
    )
    if device is not None:
        model.to(device)
    return model, tokenizer
