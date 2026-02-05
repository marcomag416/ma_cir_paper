import torch


def make_normalized(tensor: torch.Tensor, apply_bias: bool = False) -> torch.Tensor:
    """
    Normalize the input tensor along the last dimension.

    Args:
        tensor (torch.Tensor): Input tensor of shape (..., feature_dim).
        apply_bias (bool): Whether to apply a small bias to avoid division by zero.

    Returns:
        torch.Tensor: Normalized tensor of the same shape as input.
    """
    bias = 1e-10 if apply_bias else 0.0
    norm = torch.norm(tensor, p=2, dim=-1, keepdim=True)
    normalized_tensor = tensor / (norm + bias) 
    return normalized_tensor

def slerp(a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Perform Spherical Linear Interpolation (slerp) between two tensors.
    Args:
        a (torch.Tensor): Starting tensor of shape (..., feature_dim).
        b (torch.Tensor): Ending tensor of shape (..., feature_dim).
        alpha (float): Interpolation factor between 0 and 1. 0 corresponds to 'a', and 1.0 corresponds to 'b'. Use alpha > 1.0 for extrapolation.
    Returns:
        torch.Tensor: Interpolated tensor of the same shape as input tensors.
    """
    a = make_normalized(a)
    b = make_normalized(b)
    dot_product = torch.sum(a * b, dim=-1, keepdim=True)
    eps = 5e-8 if a.dtype == torch.float32 else 5e-4
    # Clamp dot_product to ensure theta is never exactly 0 or pi
    theta = torch.acos(torch.clamp(dot_product, -1.0 + eps, 1.0 - eps))
    sin_theta = torch.sin(theta)
    
    slerp_a = (torch.sin((1 - alpha) * theta) / sin_theta) * a
    slerp_b = (torch.sin(alpha * theta) / sin_theta) * b
    return  slerp_a + slerp_b