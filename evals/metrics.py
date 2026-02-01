import torch
import matplotlib.pyplot as plt


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

def compute_statistic_metrics(image_features, text_features):
    """
    Compute basic statistics for image and text features.

    Args:
        image_features (torch.Tensor): Image features of shape (N, D).
        text_features (torch.Tensor): Text features of shape (N, D).
    Returns:
        dict: Dictionary containing variances of image and text features.
    """

    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    
    image_var = image_features.var(dim=0, unbiased=False).sum().item()
    text_var = text_features.var(dim=0, unbiased=False).sum().item()

    metrics = {
        "image_variance": image_var,
        "text_variance": text_var,
    }

    return metrics

def generate_derangement(N) -> torch.Tensor:
    """
    Generate a derangement of size N (Indices in [0, 1, ..., N-1]).
    A derangement is a permutation where no element appears in its original position.

    The probability that a random permutation is a derangement is 1/e (~36.8%) for N->infinity.
    Reference: https://mathworld.wolfram.com/Derangement.html
    """
    if N < 2:
        raise ValueError("Size must be at least 2 to create a derangement.")

    # we use a simple rejection sampling strategy
    # the dearangement is found with probability ~ 1/e for large N. This means that on average we need e ~ 2.718 attempts.
    indices = torch.randperm(N)
    while (indices == torch.arange(N)).any():
        indices = torch.randperm(N)
  
    return indices

def compute_sim_distributions(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute similarity distributions for image and text features.

    Args:
        image_features (torch.Tensor): Image features of shape (N, D).
        text_features (torch.Tensor): Text features of shape (N, D).
    Returns:
        tuple: A tuple containing:
            - pos_sims (torch.Tensor): Similarities of positive pairs.
            - rnd_sims (torch.Tensor): Similarities of random deranged pairs.
    """

    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    image_to_text_sim = image_features @ text_features.t()

    rnd_index = generate_derangement(image_features.shape[0])

    pos_sims = torch.diag(image_to_text_sim)
    rnd_sims = image_to_text_sim[torch.arange(image_features.shape[0]), rnd_index]

    return pos_sims, rnd_sims
    
def plot_sim_distributions(
    pos_similarities: torch.Tensor,
    rnd_similarities: torch.Tensor,
    save_path: str,
):
    pos_mean = pos_similarities.mean().item()
    rnd_mean = rnd_similarities.mean().item()


    plt.figure(figsize=(10,6))
    plt.hist(pos_similarities.detach().cpu().numpy(), bins=50, alpha=0.5, label='Positive pairs')
    plt.hist(rnd_similarities.detach().cpu().numpy(), bins=50, alpha=0.5, label='Random pairs')
    # plt.hist(worst_similarities.detach().cpu().numpy(), bins=50, alpha=0.3, label='Worst similarities')

    # plot vertical lines for means
    plt.axvline(pos_mean, color='blue', linestyle='dashed', linewidth=1)
    plt.axvline(rnd_mean, color='red', linestyle='dashed', linewidth=1)
    # plt.axvline(worst_mean, color='green', linestyle='dashed', linewidth=1)
    # add text for means
    plt.text(pos_mean + (plt.xlim()[1] - plt.xlim()[0]) * 0.01, plt.ylim()[1]*0.92, f'Mean: {pos_mean:.2f}', color='blue', va='bottom', ha='left', fontsize=14)
    plt.text(rnd_mean - (plt.xlim()[1] - plt.xlim()[0]) * 0.01, plt.ylim()[1]*0.92, f'Mean: {rnd_mean:.2f}', color='red', va='bottom', ha='right', fontsize=14)
    # plt.text(worst_mean - 0.001, plt.ylim()[1]*0.95, f'Mean: {worst_mean:.4f}', color='green', va='bottom', ha='right')
    plt.xlabel('Similarity', fontsize=12)
    # plt.ylabel('Frequency')
    plt.title(f'Similarity Distribution', fontsize=16)
    plt.legend(loc='best', fontsize=14)
    plt.savefig(save_path)
    plt.close()