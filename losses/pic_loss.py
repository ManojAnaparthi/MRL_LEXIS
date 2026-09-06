"""
PIC (Pipeline Information Chaining) Loss Module
Vertical information flow across layers and dimensions

This module implements the vertical chaining component of MIPIC:
- Chains representations from shallow layers (small dims) to deeper layers (larger dims)
- Uses InfoNCE loss with non-linear projection for alignment
- Works as complement to MRL's horizontal Matryoshka learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Union

from losses.distributed import gather_with_grad, global_batch_labels


class PipelineInfoNCELoss(nn.Module):
    """InfoNCE loss for vertical pipeline chaining across layers and dimensions.

    Chains information from shallow layers (small dimensions) to deeper layers (larger dimensions).
    Example: (layer_1, dim_16) --[InfoNCE]--> (layer_3, dim_128) --[InfoNCE]--> (layer_5, dim_768)

    Architecture:
        - Projects source (layer_i, dim_i) through non-linear f to target dimension (dim_j)
        - Uses InfoNCE (contrastive) loss with CLS tokens
        - The target z_{i+1} is detached by default. Eq. (15)-(16) frame PIC as
          "propagating task-relevant cues from deeper layers to earlier ones",
          i.e. the deeper checkpoint is the teacher; both the official notebook
          (`v_tgt = v_tgt.detach()`) and the repository's best-scoring run do
          this. Set detach_target=False for the symmetric variant.
    """

    def __init__(self, d_src: int, d_tgt: int, d_hidden: int = 256, detach_target: bool = True):
        """Initialize pipeline InfoNCE loss module.

        Args:
            d_src: Source dimension (smaller, from shallower layer)
            d_tgt: Target dimension (larger, from deeper layer)
            d_hidden: Hidden dimension for non-linear projector
        """
        super().__init__()
        self.d_src = d_src
        self.d_tgt = d_tgt
        self.detach_target = bool(detach_target)

        # Non-linear projector f: d_src -> d_tgt
        # Helps model learn complex mappings between different representation spaces
        self.phi = nn.Sequential(
            nn.Linear(d_src, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_tgt)
        )

    def forward(
        self,
        src_hidden: torch.Tensor,  # [B, L, d_src] - source layer representation
        tgt_hidden: torch.Tensor,  # [B, L, d_tgt] - target layer representation
        mask: Optional[torch.Tensor] = None,  # [B, L] - attention mask (not used, for compat)
        temperature: float = 0.05
    ) -> torch.Tensor:
        """Compute pipeline InfoNCE loss.

        Encourages source representation to align with target through contrastive learning.

        Args:
            src_hidden: [B, L, d_src] source hidden states
            tgt_hidden: [B, L, d_tgt] target hidden states
            mask: [B, L] attention mask (optional, not used)
            temperature: InfoNCE temperature scaling

        Returns:
            InfoNCE loss value (scalar)
        """
        # Extract CLS tokens [B, d_src] and [B, d_tgt]
        # Keep contrastive loss arithmetic in fp32 under CUDA AMP.
        u_src = src_hidden[:, 0, :].float()
        v_tgt = tgt_hidden[:, 0, :].float()

        if self.detach_target:
            # Deeper checkpoint acts as the teacher for the shallower one.
            v_tgt = v_tgt.detach()

        # Project source to target dimension [B, d_tgt]
        u_proj = self.phi(u_src)

        # Normalize for cosine similarity
        u_proj = F.normalize(u_proj, dim=-1)
        v_tgt = F.normalize(v_tgt, dim=-1)

        # Under DDP, keep local queries but gather all target keys so the
        # InfoNCE denominator uses the global effective batch.
        gathered_targets = torch.cat(gather_with_grad(v_tgt), dim=0)
        logits = torch.matmul(u_proj, gathered_targets.T) / temperature

        # Positive for local query i is the corresponding global target i.
        labels = global_batch_labels(u_proj.size(0), u_proj.device)

        # InfoNCE loss (cross entropy over in-batch negatives)
        loss = F.cross_entropy(logits, labels)

        return loss


class PICModule(nn.Module):
    """Pipeline Information Chaining (PIC) - Vertical alignment across layers.

    PIC creates a vertical information flow pipeline where representations are chained
    from shallow layers with small dimensions to deeper layers with larger dimensions.

    Pipeline example for TinyBERT (6 layers):
        (layer_1, dim_16) ? (layer_3, dim_128) ? (layer_5, dim_768)

    This complementary to MRL's horizontal Matryoshka learning:
        - MRL: Horizontal (within-layer Matryoshka dimensions)
        - PIC: Vertical (cross-layer pipeline chaining)
        - Together: Ensures both horizontal and vertical alignment
    """

    def __init__(
        self,
        d_full: int = 768,
        pipeline_pairs: List[Tuple[int, ...]] = None,
        gamma: float = 1.0,
        temperature: float = 0.05,
        detach_target: bool = True,
    ):
        """Initialize PIC module.

        Args:
            d_full: Full hidden dimension (e.g., 768 for BERT)
            pipeline_pairs: List of pipeline configurations
                Each config is tuple: (layer_1, dim_1, layer_2, dim_2, ...)
                Example: [(1, 16, 3, 128, 5, 768)]
            gamma: Weight for chain loss (in combined loss)
            temperature: InfoNCE temperature
        """
        super().__init__()

        self.d_full = d_full
        self.gamma = gamma
        self.temperature = temperature
        self.detach_target = bool(detach_target)

        # Parse pipeline configuration into consecutive pairs
        if pipeline_pairs is None:
            pipeline_pairs = [(1, 16, 2, 32, 3, 64, 4, 256, 5, 512, 6, 768)]

        self.parsed_pipeline_pairs = []

        for pipe_config in pipeline_pairs:
            if len(pipe_config) % 2 != 0:
                raise ValueError(f"Pipeline config must have even length (layer, dim pairs): {pipe_config}")

            # Extract (layer, dim) pairs from config
            stages = [(pipe_config[i], pipe_config[i+1]) for i in range(0, len(pipe_config), 2)]

            # Create consecutive pairs for chaining
            # E.g., [(1,16), (3,128), (5,768)] ? [(1,16)?(3,128), (3,128)?(5,768)]
            for i in range(len(stages) - 1):
                layer_i, dim_i = stages[i]
                layer_j, dim_j = stages[i + 1]
                self.parsed_pipeline_pairs.append((layer_i, dim_i, layer_j, dim_j))

        # Create InfoNCE modules for each pipeline pair
        self.infonce_modules = nn.ModuleDict()

        for layer_i, dim_i, layer_j, dim_j in self.parsed_pipeline_pairs:
            key = f"pipe_{layer_i}_{dim_i}_to_{layer_j}_{dim_j}"
            self.infonce_modules[key] = PipelineInfoNCELoss(
                d_src=dim_i,
                d_tgt=dim_j,
                d_hidden=max(dim_i, dim_j) // 2,
                detach_target=self.detach_target,
            )

        # Log configuration
        print(f" PICModule initialized with {len(self.parsed_pipeline_pairs)} pipeline pairs:")
        for layer_i, dim_i, layer_j, dim_j in self.parsed_pipeline_pairs:
            print(f"   (layer_{layer_i}, dim_{dim_i}) ? (layer_{layer_j}, dim_{dim_j})")

    def forward(
        self,
        hidden_states: List[torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute PIC loss (vertical chaining).

        Args:
            hidden_states: List of [B, L, d_full] tensors (one per layer)
            mask: [B, L] attention mask

        Returns:
            Dictionary containing:
                - 'total_loss': summed pipeline chaining loss
                - 'pair_losses': per-pair loss values
        """
        total_chain_loss = 0.0
        losses = {}

        # Apply InfoNCE loss for each pipeline pair
        for layer_i, dim_i, layer_j, dim_j in self.parsed_pipeline_pairs:
            # Extract and truncate hidden states to specified dimensions
            src_hidden = hidden_states[layer_i][..., :dim_i]  # [B, L, dim_i]
            tgt_hidden = hidden_states[layer_j][..., :dim_j]  # [B, L, dim_j]

            key = f"pipe_{layer_i}_{dim_i}_to_{layer_j}_{dim_j}"
            infonce_module = self.infonce_modules[key]

            # Compute pipeline InfoNCE loss
            chain_loss = infonce_module(
                src_hidden=src_hidden,
                tgt_hidden=tgt_hidden,
                mask=mask,
                temperature=self.temperature
            )

            losses[key] = chain_loss
            total_chain_loss += chain_loss

        # Mean, not sum, over the 5 adjacent transitions. Eq. (17) writes a sum;
        # see losses/sia_loss.py for why the sum is not used. The official
        # notebook divides by len(parsed_pipeline_pairs) and so does the
        # best-scoring run of this repository (logged pic_loss ~2.8 vs ~13.3).
        if self.parsed_pipeline_pairs:
            total_chain_loss = total_chain_loss / len(self.parsed_pipeline_pairs)

        return {
            'total_loss': total_chain_loss,
            'pair_losses': losses
        }


def create_pic_module(
    model_name: str,
    d_full: int = 768,
    pipeline_pairs: List[Tuple[int, ...]] = None,
    device: torch.device = None
) -> PICModule:
    """Factory function to create PIC module with model-specific defaults.

    Args:
        model_name: Model identifier (tinybert, bert, qwen, bgem3)
        d_full: Full dimension
        pipeline_pairs: Custom pipeline configuration (overrides defaults)
        device: Device to move module to

    Returns:
        Initialized PICModule
    """

    # Model-specific pipeline defaults
    defaults = {
        "tinybert": {
            "d_full": 768,
            "pipeline_pairs": [
                (1, 16, 2, 32, 3, 64, 4, 256, 5, 512, 6, 768)
            ],
        },
        "bert": {
            "d_full": 768,
            "pipeline_pairs": [
                (2, 16, 4, 32, 6, 64, 8, 128,
                9, 256, 10, 512, 12, 768)
            ],
        },
        "qwen": {
            "d_full": 1024,
            "pipeline_pairs": [
                (2, 16, 6, 32, 12, 64, 16, 128,
                20, 256, 24, 512, 28, 1024)
            ],
        },
        "bgem3": {
            "d_full": 1024,
            "pipeline_pairs": [
                (1, 16, 4, 32, 7, 64, 11, 128,
                15, 256, 19, 512, 24, 1024)
            ],
        },
    }

    if model_name in defaults:
        cfg = defaults[model_name]
        d_full = cfg["d_full"]
        if pipeline_pairs is None:
            pipeline_pairs = cfg["pipeline_pairs"]

    module = PICModule(
        d_full=d_full,
        pipeline_pairs=pipeline_pairs
    )

    if device is not None:
        module = module.to(device)

    return module


if __name__ == "__main__":
    # Example usage
    print("PIC Module Test")
    print("=" * 50)

    # Create module
    pic = PICModule(
        d_full=768,
        pipeline_pairs=[(1, 16, 3, 128, 5, 768)],
        temperature=0.07
    )

    # Simulate hidden states (6 layers, batch_size=2, seq_len=10)
    B, L, d = 2, 10, 768
    hidden_states = [torch.randn(B, L, d) for _ in range(6)]
    mask = torch.ones(B, L)

    # Forward pass
    output = pic(hidden_states, mask)
    print(f"\nOutput:")
    print(f"  total_loss: {output['total_loss'].item():.4f}")
    for key, loss in output['pair_losses'].items():
        print(f"  {key}: {loss.item():.4f}")

    print("\nPIC module working correctly!")

