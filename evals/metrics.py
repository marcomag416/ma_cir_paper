from sklearn.neighbors import KernelDensity
import torch
import matplotlib.pyplot as plt
import numpy as np


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

def generate_derangement(N, seed: int = 42) -> torch.Tensor:
    """
    Generate a derangement of size N (Indices in [0, 1, ..., N-1]).
    A derangement is a permutation where no element appears in its original position.

    The probability that a random permutation is a derangement is 1/e (~36.8%) for N->infinity.
    Reference: https://mathworld.wolfram.com/Derangement.html
    """
    if N < 2:
        raise ValueError("Size must be at least 2 to create a derangement.")
    
    generator = torch.Generator().manual_seed(seed)

    # we use a simple rejection sampling strategy
    # the dearangement is found with probability ~ 1/e for large N. This means that on average we need e ~ 2.718 attempts.
    indices = torch.randperm(N, generator=generator)
    while (indices == torch.arange(N)).any():
        indices = torch.randperm(N, generator=generator)
  
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
        seed (int): Random seed for generating derangements.
    Returns:
        tuple: A tuple containing:
            - pos_sims (torch.Tensor): Similarities of positive pairs.
            - rnd_sims (torch.Tensor): Similarities of random deranged pairs.
    """

    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    image_to_text_sim = image_features @ text_features.t()

    rnd_index = generate_derangement(image_features.shape[0], seed=seed)

    pos_sims = torch.diag(image_to_text_sim)
    rnd_sims = image_to_text_sim[torch.arange(image_features.shape[0]), rnd_index]

    return pos_sims, rnd_sims
    
def plot_sim_distributions(
    pos_similarities: torch.Tensor,
    rnd_similarities: torch.Tensor,
    save_path: str,
    pos_label: str = 'Positive pairs',
    rnd_label: str = 'Random pairs',
):
    pos_mean = pos_similarities.mean().item()
    rnd_mean = rnd_similarities.mean().item()


    plt.figure(figsize=(10,6))
    plt.hist(pos_similarities.detach().cpu().numpy(), bins=50, alpha=0.5, label=pos_label)
    plt.hist(rnd_similarities.detach().cpu().numpy(), bins=50, alpha=0.5, label=rnd_label)
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

def plot_sim_distribution_v2(
    true_trgt_sims: torch.Tensor,
    rnd_trgt_sims: torch.Tensor,
    save_path: str,
    true_trgt_label: str = 'True Target',
    rnd_trgt_label: str = 'Random Target',
    show_legend: bool = True,
):
    true_trgt_sims = true_trgt_sims.cpu()
    rnd_trgt_sims = rnd_trgt_sims.cpu()
    true_trgt_kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(true_trgt_sims.unsqueeze(1))
    rnd_trgt_kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(rnd_trgt_sims.unsqueeze(1))
    true_trgt_avg = true_trgt_sims.mean().item()
    rnd_trgt_avg = rnd_trgt_sims.mean().item()
    x_plot = torch.linspace(-1, 1, 1000).unsqueeze(1)
    true_trgt_log_dens = true_trgt_kde.score_samples(x_plot)
    rnd_trgt_log_dens = rnd_trgt_kde.score_samples(x_plot)
    intersection_log_dens = np.minimum(true_trgt_log_dens, rnd_trgt_log_dens)

    #compute intersection area
    intersection_area = np.trapezoid(np.exp(intersection_log_dens), x_plot.squeeze())
    union_area = np.trapezoid(np.exp(true_trgt_log_dens) + np.exp(rnd_trgt_log_dens) - np.exp(intersection_log_dens), x_plot.squeeze())
    jaccard_index = intersection_area / union_area

    gap = true_trgt_avg - rnd_trgt_avg

    plt.figure(figsize=(6, 4))
    #Positive trgts distribution
    plt.plot(x_plot, np.exp(true_trgt_log_dens), label=true_trgt_label, color='darkviolet')
    plt.fill_between(x_plot.squeeze(), np.exp(true_trgt_log_dens), alpha=0.5, color='darkviolet')
    plt.hist(true_trgt_sims.numpy(), bins=100, density=True, alpha=0.5, color='grey')
    plt.axvline(true_trgt_avg, color='darkviolet', linestyle='--')
    plt.text(true_trgt_avg, 0.2, f'Avg: {true_trgt_avg:.3f}', 
            color='black', ha='center', va='center', 
            bbox=dict(facecolor='white', edgecolor='darkviolet', alpha=0.8),
            transform=plt.gca().get_xaxis_transform())

    # Random trgts distribution
    plt.plot(x_plot, np.exp(rnd_trgt_log_dens), label=rnd_trgt_label, color='green')
    plt.fill_between(x_plot.squeeze(), np.exp(rnd_trgt_log_dens), alpha=0.5, color='green')
    plt.hist(rnd_trgt_sims.numpy(), bins=100, density=True, alpha=0.5, color='grey')
    plt.axvline(rnd_trgt_avg, color='green', linestyle='--')
    plt.text(rnd_trgt_avg, 0.2, f'Avg: {rnd_trgt_avg:.3f}', 
            color='black', ha='center', va='center', 
            bbox=dict(facecolor='white', edgecolor='green', alpha=0.8),
            transform=plt.gca().get_xaxis_transform())

    #intersection area
    plt.fill_between(x_plot.squeeze(), np.exp(intersection_log_dens), color='blue', alpha=0.3)

    # Gap arrow
    plt.annotate('', xy=(true_trgt_avg, 0.9), xytext=(rnd_trgt_avg, 0.9), 
                arrowprops=dict(arrowstyle='<->', color='red', lw=2),
                xycoords=plt.gca().get_xaxis_transform())
    plt.text((true_trgt_avg + rnd_trgt_avg)/2, 0.91, f'Gap: {gap:.3f}', color='black', ha='center', va='bottom', transform=plt.gca().get_xaxis_transform())

    #boxes with metrics
    plt.text(0.05, 0.9, f'IoU: {jaccard_index:.3f}', bbox=dict(facecolor='blue', edgecolor='blue', alpha=0.3), transform=plt.gca().transAxes)


    plt.xlim(-0.1, 0.5)
    # plt.xlabel('Cosine Similarity')
    # plt.ylabel('Density')
    if show_legend:
        plt.legend()
    # plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()