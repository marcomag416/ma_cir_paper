
def compute_modality_gap_metrics(image_features, text_features):
    """
    Compute modality gap metrics given image and text features.

    Args:
        image_features (torch.Tensor): Image features of shape (N, D).
        text_features (torch.Tensor): Text features of shape (N, D).
    Returns:
        dict: Dictionary containing 'modality_gap', 'alignment', and 'XSC-SR'.

    """

    assert image_features.shape == text_features.shape, "Image and text features must have the same shape."
    N, D = image_features.shape

    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # compute modality delta vectors
    mod_delta = image_features - text_features  # shape (N, D)

    # compute modality gap
    modality_gap = mod_delta.mean(dim=0).norm(p=2).item()

    # compute alignment score
    alignment = mod_delta.pow(2).sum(dim=1).mean().item()

    # compute XSC-SR
    xsc_sr = (2*N / (N-1)) * mod_delta.var(dim=0, unbiased=False).sum().item()  

    metrics = {
        "modality_gap": modality_gap,
        "alignment": alignment,
        "XSC-SR": xsc_sr,
    }

    return metrics
