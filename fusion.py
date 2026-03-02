import torch 
from typing import Literal

from utils.tensor import make_normalized, slerp

def fusion(
        image_features : torch.Tensor,
        text_features : torch.Tensor, 
        fusion_type : Literal["sum", "slerp", "cslerp"]="sum",
        alpha: float = None
) -> torch.Tensor:
    """
    Fuse image and text features using the specified fusion type.

    Args:
        image_features (torch.Tensor): Tensor of shape (batch_size, feature_dim) representing image features.
        text_features (torch.Tensor): Tensor of shape (batch_size, feature_dim) representing text features.
        fusion_type (str): Type of fusion to apply. Options are 'sum', 'slerp', 'cslerp'. cslerp is slerp with custom alpha.
        alpha (float): Interpolation factor for slerp fusion. Ignored if fusion_type is not 'slerp'.

    Returns:
        torch.Tensor: Fused features tensor.
    """

    image_features = make_normalized(image_features)
    text_features = make_normalized(text_features)

    if fusion_type == "sum":
        fused_features = image_features + text_features
    elif fusion_type == "slerp":
        # Spherical linear interpolation (slerp) with alpha=0.8
        fused_features = slerp(image_features, text_features, alpha=0.8)
    elif fusion_type == "cslerp":
        # Spherical linear interpolation (slerp) with custom alpha
        assert alpha is not None, "Alpha must be provided for cslerp fusion type."
        fused_features = slerp(image_features, text_features, alpha=alpha)
    else:
        raise ValueError(f"Unsupported fusion type: {fusion_type}. Supported types are 'sum', 'slerp', and 'cslerp'.")
    return fused_features

