# Adapted from: https://github.com/jaeseokbyun/MACIR/blob/main/eval.py


import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Literal
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from datasets.macir import build_macir_dataset
from models import TwoEncoderVLM
from tqdm.auto import tqdm
from contextlib import nullcontext

from utils.decorators import timed_metric
from utils.transformers import is_accelerator


def macir_compute_test_metrics(
    predicted_features: torch.Tensor,
    reference_names: List[str],
    target_names: List[str],
    composition_types: List[str],
    condition_types: List[str],
    index_features: torch.Tensor,
    index_names: List[str],
    eval_level: str,
    split: str,  
) -> Dict[str, float]:
    """
    Compute retrieval metrics on the MACIR test set given:
      - the query dataset
      - index database features + names
      - pseudo tokens (when using projection-based ZS-CIR)
      - reference names / composition types / condition types
    Returns:
        metrics: dict mapping metric name -> value
    """
    
    # Generate predicted query features and metadata from the dataset
    # predicted_features, reference_names, target_names, composition_types, condition_types = \
    #      macir_generate_test_predictions(clip_model, query_test_dataset, ref_names_list, pseudo_tokens, fusion_type)

    # Determine device from index_features (or predicted_features)
    device = index_features.device

    # Move tensors to the target device
    index_features = index_features.to(device)
    predicted_features = predicted_features.to(device)

    # Normalize features (index features; queries kept as they are)
    # index_features: (N, D)
    index_features = F.normalize(index_features, dim=-1).float()
    predicted_features = predicted_features.float()

    
    # Compute distances and sort
    # Let B = number of queries, N = size of the database
    # distances: (B, N)
    distances = 1 - predicted_features @ index_features.T
    # sorted_indices[b] = indices of index_features sorted by ascending distance
    sorted_indices = torch.argsort(distances, dim=-1).cpu()  # (B, N)
    # sorted_index_names_matrix[b, j] = index_names[ sorted_indices[b, j] ]
    sorted_index_names_matrix = np.array(index_names)[sorted_indices]  # (B, N)

    
    # Build reference prefixes for "restricted" evaluation cases
    reference_prefixes = [
        ref.split("_edit_")[0] if "_edit_" in ref else ref.split(".")[0]
        for ref in reference_names
    ]

    
    # Remove the reference (original) image itself from the ranking (we do not want the reference image to count as a retrieval candidate)
    # reference_names: (B,)
    # sorted_index_names_matrix: (B, N)
    # Build a mask that is False where the candidate equals the reference image,
    reference_mask = torch.tensor(
        sorted_index_names_matrix
        != np.repeat(np.array(reference_names), len(index_names)).reshape(
            len(target_names), -1
        )
    )
    # Apply the mask and reshape back to (B, N-1).
    # Now each row corresponds to candidates excluding the exact reference image.
    sorted_index_names_matrix = sorted_index_names_matrix[reference_mask].reshape(
        sorted_index_names_matrix.shape[0], sorted_index_names_matrix.shape[1] - 1
    )

    
    # Metric containers & type lists
    metrics = {}
    unique_composition_types = ['left_right', 'top_bottom', 'spatial_reasoning', 'size', 'action', 'color','object_reasoning','naive_object']
    unique_condition_types = ['remove', 'replace', 'add']
    # =========================================================================
    # CASE 1: eval_level == "full" (include all hard negatives)
    #   - Only group by composition_type
    # =========================================================================
    if eval_level == "full":
        for comp_type in unique_composition_types:
            # Indices of queries with the given composition_type
            type_indices = [
                i
                for i, ct in enumerate(composition_types)
                if ct == comp_type
            ]
            if not type_indices:
                # No queries with this composition_type
                continue

            # Select the rows corresponding to the chosen composition_type
            #   filtered_sorted_index_names_matrix: (B_c, N-1)
            filtered_sorted_index_names_matrix = sorted_index_names_matrix[type_indices]
            filtered_target_names = [target_names[i] for i in type_indices]

            # Build labels: labels[b, j] = True if candidate matches the target
            labels = torch.tensor(filtered_sorted_index_names_matrix== np.repeat(
                    np.array(filtered_target_names),
                    len(index_names) - 1,
                ).reshape(len(filtered_target_names), -1)
            )

            # Sanity check: For each query, exactly one candidate should match
            assert torch.equal(
                torch.sum(labels, dim=-1).int(),
                torch.ones(len(filtered_target_names)).int(),
            ), "Each query should have exactly one matching candidate."

            # Recall@1: fraction of queries where rank-1 candidate is correct
            recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100.0

            metrics[f"{comp_type}"] = {
                "recall_at1": recall_at1,
            }

        return metrics

    # =========================================================================
    # CASE 2: eval_level == "full_splits" (include all hard negatives, but split with condition_type)
    #   - Group by composition_type AND condition_type
    # =========================================================================
    elif eval_level == "full_splits":
        for comp_type in unique_composition_types:
            for cond_type in unique_condition_types:
                # Indices of queries matching both comp_type and cond_type
                type_indices = [
                    i
                    for i, (ct, cd) in enumerate(
                        zip(composition_types, condition_types)
                    )
                    if ct == comp_type and cd == cond_type
                ]

                if not type_indices:
                    # (e.g., spatial_reasoning+add, size+add)
                    metrics[f"{comp_type}_{cond_type}"] = {
                        "recall_at1": 0.0,
                    }
                    continue

                # Select the rows corresponding to this (comp_type, cond_type)
                filtered_sorted_index_names_matrix = sorted_index_names_matrix[
                    type_indices
                ]
                filtered_target_names = [target_names[i] for i in type_indices]

                # Build labels: labels[b, j] = True if candidate matches target
                labels = torch.tensor(
                    filtered_sorted_index_names_matrix
                    == np.repeat(
                        np.array(filtered_target_names),
                        len(index_names) - 1,
                    ).reshape(len(filtered_target_names), -1)
                )

                # Sanity check: one matching candidate per query
                assert torch.equal(
                    torch.sum(labels, dim=-1).int(),
                    torch.ones(len(filtered_target_names)).int(),
                ), "Each query should have exactly one matching candidate."

                # Recall@1 for this combination
                recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100.0

                metrics[f"{comp_type}_{cond_type}"] = {
                    "recall_at1": recall_at1,
                }

        return metrics

    # =========================================================================
    # CASE 3: eval_level is something else (e.g., "restricted"-style evaluation)
    #   - cond_type is given by `split` arg; we group by composition_type and
    #     restrict to candidates sharing the same "reference prefix"
    # =========================================================================
    elif eval_level == "restricted":
        cond_type = split 
        # -------------------------------------------------------------
        # For each query, keep only candidates that share the same
        #      prefix as the reference 
        # -------------------------------------------------------------
        reference_mask_split = torch.tensor(
            [
                [name.startswith(prefix) for name in sorted_index_names_row]
                for sorted_index_names_row, prefix in zip(
                    sorted_index_names_matrix, reference_prefixes
                )
            ],
            dtype=torch.bool,
        )

        # For each query, select only the candidates with the same prefix.
        # NOTE: This becomes a "ragged" list: each row can have different length. (some queries have multiple candidates (more than 2) )
        filtered_sorted_index_names_per_query = [
            np.array(sorted_index_names_row)[mask_row].tolist()
            for sorted_index_names_row, mask_row in zip(
                sorted_index_names_matrix, reference_mask_split
            )
        ]

        # -------------------------------------------------------------
        # Group by composition_type and compute recall@1
        #      using the filtered candidates
        # -------------------------------------------------------------
        for comp_type in unique_composition_types:
            # Indices of queries with the given composition_type
            type_indices = [
                i
                for i, ct in enumerate(composition_types)
                if ct == comp_type
            ]
            if not type_indices:
                # No queries for this composition type
                continue

            # Subset the filtered candidate lists and targets for this comp_type
            cur_rows = [filtered_sorted_index_names_per_query[i] for i in type_indices]
            cur_targets = [target_names[i] for i in type_indices]

            # Keep only queries that have at least 2 candidates
            # (we are interested in non-trivial cases)
            rows_targets = [
                (row, tgt)
                for row, tgt in zip(cur_rows, cur_targets)
                if len(row) > 1
            ]

            if not rows_targets:
                # No valid rows for this comp_type under the split setting
                continue

            cur_rows, cur_targets = zip(*rows_targets)  # unpack

            # Build labels: labels[q][j] = 1 if candidate j == target, else 0
            labels = [
                [int(name == target) for name in sorted_row]
                for sorted_row, target in zip(cur_rows, cur_targets)
            ]

            # Compute Recall@1: fraction of queries where the first candidate is correct
            num_queries = len(labels)
            num_correct_at1 = sum(1 for row in labels if row[0] == 1)
            recall_at1 = (num_correct_at1 / num_queries) * 100.0

            metrics[f"{comp_type}_{cond_type}"] = {
                "recall_at1": recall_at1,
            }

        return metrics

@torch.no_grad()
def macir_generate_test_predictions(
    clip_model: TwoEncoderVLM, 
    query_test_dataset: Dataset, 
    fusion_type: str, 
    batch_size: int = 64, 
    num_workers: int = 4, 
    use_tqdm: bool = False,
    accelerator = None
):
    pin_mem = (accelerator is None and torch.cuda.is_available())
    dataloader = DataLoader(query_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    
    clip_model.eval()
    vision_encoder = clip_model.vision
    text_encoder = clip_model.text

    if accelerator:
        dataloader = accelerator.prepare(dataloader)
        vision_encoder, text_encoder = accelerator.prepare(vision_encoder, text_encoder)
        device_ctx = accelerator.autocast
    else:
        device_ctx = nullcontext

    all_predicted_features = []
    all_reference_names = []
    all_target_names = []
    all_composition_types = []
    all_condition_types = []

    for batch in tqdm(dataloader, disable=not use_tqdm, desc="Generating MACIR test predictions"):
        if accelerator:
            reference_images = batch['reference_image']
            relative_captions = batch['relative_caption']
            attention_mask = batch['attention_mask']
        else:
            reference_images = batch['reference_image'].to(vision_encoder.device)
            relative_captions = batch['relative_caption'].to(text_encoder.device)
            attention_mask = batch['attention_mask'].to(text_encoder.device)

        reference_names = batch['reference_name']
        target_names = batch['target_name']
        composition_types = batch['composition_type']
        condition_types = batch['condition_type']

        with device_ctx():
            image_features = vision_encoder(reference_images).image_embeds
            text_features = text_encoder(input_ids=relative_captions, attention_mask=attention_mask).text_embeds
            if fusion_type == 'sum':
                predicted_features = image_features + text_features
            else:
                raise ValueError(f"Unknown fusion_type: {fusion_type}")
        
        all_predicted_features.append(predicted_features)
        all_reference_names.extend(reference_names)
        all_target_names.extend(target_names)
        all_composition_types.extend(composition_types)
        all_condition_types.extend(condition_types)

    all_predicted_features = torch.vstack(all_predicted_features)

    return (all_predicted_features, all_reference_names, all_target_names, all_composition_types, all_condition_types)    

@torch.no_grad()
def macir_generate_index_features(
    clip_model: TwoEncoderVLM, 
    index_dataset: Dataset, 
    batch_size: int = 64, 
    num_workers: int = 4, 
    use_tqdm: bool = False,
    accelerator = None
):
    pin_mem = (accelerator is None and torch.cuda.is_available())
    dataloader = DataLoader(index_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    
    clip_model.eval()
    vision_encoder = clip_model.vision

    if accelerator:
        dataloader = accelerator.prepare(dataloader)
        vision_encoder = accelerator.prepare(vision_encoder)
        device_ctx = accelerator.autocast
    else:
        device_ctx = nullcontext
    
    all_index_features = []
    all_index_names = []

    for batch in tqdm(dataloader, disable=not use_tqdm, desc="Generating MACIR index features"):
        if accelerator:
            images = batch['image']
        else:
            images = batch['image'].to(vision_encoder.device)
            
        image_names = batch['image_name']

        with device_ctx():
            image_features = vision_encoder(images).image_embeds

        all_index_features.append(image_features)
        all_index_names.extend(image_names)

    all_index_features = torch.vstack(all_index_features)

    return all_index_features, all_index_names  
    


@timed_metric
def evaluate_macir(
    model: TwoEncoderVLM, 
    eval_level: Literal["full", "full_splits", "restricted"],
    split: str, 
    fusion_type: str = "sum", 
    batch_size: int = 64, 
    num_workers: int = 4, 
    tqdm: bool = False,
    accelerator = None
) -> Dict[str, float]:
    # SAFETY CHECK
    if accelerator is not None and not is_accelerator(accelerator):
        accelerator = None

    macir_db = build_macir_dataset(
        split=split,
        mode="database",
        preprocess=model.image_processor,
        eval_level=eval_level,
        caption_transform=model.tokenizer,
        max_length_tokenizer=77,
    )
    ma_cir_query = build_macir_dataset(
        split=split,
        mode="query",
        preprocess=model.image_processor,
        eval_level=eval_level,
        caption_transform=model.tokenizer,
        max_length_tokenizer=77,
    )
    index_features, index_names = macir_generate_index_features(
        clip_model=model,
        index_dataset=macir_db,
        batch_size=batch_size,
        num_workers=num_workers,
        use_tqdm=tqdm,
        accelerator=accelerator
    )

    predicted_features, reference_names, target_names, composition_types, condition_types = macir_generate_test_predictions(
        clip_model=model,
        query_test_dataset=ma_cir_query,
        fusion_type=fusion_type,
        batch_size=batch_size,
        num_workers=num_workers,
        use_tqdm=tqdm,
        accelerator=accelerator
    )
    metrics = macir_compute_test_metrics(
        predicted_features=predicted_features,
        reference_names=reference_names,
        target_names=target_names,
        composition_types=composition_types,
        condition_types=condition_types,
        index_features=index_features,
        index_names=index_names,
        eval_level=eval_level,
        split=split,
    )
    return metrics