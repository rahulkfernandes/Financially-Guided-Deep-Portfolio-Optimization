import torch
from torch import Tensor, nn


class FinalStrategyLayer(nn.Module):
    """
    This is a final layer for models to use long only or long-short strategy. Long
    only strategy uses a softmax function to output a portfolio weights vector which
    sums to 1. Long-Short uses a tanh function followed by normalization using absolute
    sum of the vector such that the vector sums to 1.
    """
    def __init__(self, allow_short: bool = False) -> None:
        """
        Initialize the constructor for the final strategy layer.

        Args:
            allow_short (bool): Toggle to allow long-short strategies. Default = False.
                If True, weights can be negative.
                If False, weights can only be positive.
        """
        self.allow_short = allow_short

    def forward(self, logits: Tensor) -> Tensor:
        """
        Forward pass method for the final layer.

        Args:
            logits (Tensor): Tensor containing logits from the previous layer.

        Returns:
            pf_weights (Tensor): Portfolio allocation weights.
        """
        if self.allow_short:
            raw_weights = torch.tanh(logits)
            # Normalize by sum of absolutes
            pf_weights = raw_weights / torch.sum(torch.abs(raw_weights), dim=-1, keepdim=True)
        else:
            pf_weights = torch.softmax(logits, dim=-1)

        return pf_weights
