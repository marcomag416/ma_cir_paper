import torch 
from typing import Literal

from utils.tensor import make_normalized, slerp

def fusion(
        image_features : torch.Tensor,
        text_features : torch.Tensor, 
        fusion_type : Literal["sum", "slerp"]="sum"
) -> torch.Tensor:
    """
    Fuse image and text features using the specified fusion type.

    Args:
        image_features (torch.Tensor): Tensor of shape (batch_size, feature_dim) representing image features.
        text_features (torch.Tensor): Tensor of shape (batch_size, feature_dim) representing text features.
        fusion_type (str): Type of fusion to apply. Options are 'sum', 'slerp'.

    Returns:
        torch.Tensor: Fused features tensor.
    """

    image_features = make_normalized(image_features)
    text_features = make_normalized(text_features)

    if fusion_type == "sum":
        fused_features = image_features + text_features
    elif fusion_type == "slerp":
        # Spherical linear interpolation (slerp) with alpha=0.8
        alpha = 0.8
        fused_features = slerp(image_features, text_features, alpha)
    else:
        raise ValueError(f"Unsupported fusion type: {fusion_type}. Supported types are 'sum' and 'slerp'.")
    return fused_features

