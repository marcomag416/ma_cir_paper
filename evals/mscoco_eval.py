from typing import Literal
from datasets.mscoco import build_mscoco_dataset, MSCOCOCaptions
import torch
from evals.metrics import compute_statistic_metrics, compute_modality_gap_metrics, compute_sim_distributions
from models import TwoEncoderVLM
from tqdm.auto import tqdm

@torch.no_grad()
def generate_mscoco_embeddings(
    model: TwoEncoderVLM,
    dataset: MSCOCOCaptions,
    batch_size: int = 64,
    num_workers: int = 4,
    use_tqdm : bool = False,
    accelerator=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate image and caption embeddings for the MSCOCO dataset.

    Args:
        model (TwoEncoderVLM): the model to use for generating embeddings
        dataset (MSCOCOCaptions): the MSCOCO dataset
        batch_size (int): batch size for data loading
        num_workers (int): number of workers for data loading
        tqdm: whether to use tqdm for progress bar
        accelerator: accelerator instance for distributed evaluation
    Returns:
        tuple: A tuple containing:
            - image_embeddings (torch.Tensor): Tensor of shape (N, D) where N is the number of images and D is the embedding dimension.
            - text_embeddings (torch.Tensor): Tensor of shape (N, D) where N is the number of captions and D is the embedding dimension.
    """

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
    )

    model.eval()
    all_img_embeddings = []
    all_text_embeddings = []

    for batch in tqdm(dataloader, desc="Generating MSCOCO embeddings", disable=not use_tqdm):
        pixel_values = batch['pixel_values'].to(model.device)
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)

        img_embeddings = model.vision(pixel_values).image_embeds
        text_embeddings = model.text(input_ids=input_ids, attention_mask=attention_mask).text_embeds

        all_img_embeddings.append(img_embeddings)
        all_text_embeddings.append(text_embeddings)

    image_embeddings = torch.vstack(all_img_embeddings)
    text_embeddings = torch.vstack(all_text_embeddings)

    return image_embeddings, text_embeddings



def eval_mscoco(
    model: TwoEncoderVLM,
    batch_size: int = 64,
    num_workers: int = 4,
    tqdm : bool = False,
    accelerator=None,
    split: Literal['train', 'val'] = 'val',
    return_embeddings: bool = False,
) -> dict[str, float] | tuple[dict[str, float], tuple[torch.Tensor, torch.Tensor]]:
    """
    Evaluate the model on MSCOCO image-caption retrieval task.

    Args:
        model (TwoEncoderVLM): the model to evaluate
        batch_size (int): batch size for evaluation
        num_workers (int): number of workers for data loading
        tqdm: whether to use tqdm for progress bar
        accelerator: accelerator instance for distributed evaluation
        split (str): dataset split, should be in ['train', 'val']
    Returns:
        If return_embeddings is True, returns a tuple containing: the dictionary of evaluation metrics, a tuple of similarity distributions, and a tuple of image and text embeddings.
        Otherwise, returns only the dictionary of evaluation metrics and the tuple of similarity distributions.
    """

    dataset = build_mscoco_dataset(
        split=split,
        image_transform=model.image_processor,
        caption_transform=model.tokenizer,
        cpi=1
    )

    image_embeddings, text_embeddings = generate_mscoco_embeddings(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        use_tqdm=tqdm,
        accelerator=accelerator,
    )

    metrics = {}
    metrics.update(compute_statistic_metrics(
        image_features=image_embeddings,
        text_features=text_embeddings,
    ))
    metrics.update(compute_modality_gap_metrics(
        image_features=image_embeddings,
        text_features=text_embeddings,
    ))

    sim_distributions = compute_sim_distributions(
        image_features=image_embeddings,
        text_features=text_embeddings,
    )

    if return_embeddings:
        return metrics, sim_distributions, (image_embeddings, text_embeddings)
    return metrics, sim_distributions
