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
    theta = torch.acos(torch.clamp(dot_product, -1.0, 1.0))
    sin_theta = torch.sin(theta)
    
    #if angle is zero, return a to avoid division by zero
    identity_mask = (sin_theta == 0)


    # Avoid division by zero by replacing zeros with an arbitrary value (the affected values will be ignored due to the identity_mask)
    sin_theta = torch.where(identity_mask, torch.tensor(0.5, device=sin_theta.device), sin_theta)
    
    slerp_a = (torch.sin((1 - alpha) * theta) / sin_theta) * a
    slerp_b = (torch.sin(alpha * theta) / sin_theta) * b
    return  torch.where(identity_mask, a, slerp_a + slerp_b)