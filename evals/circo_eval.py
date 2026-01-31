# Adapted from https://github.com/miccunifi/CIRCO/blob/main/src/evaluation.py

# In order to reuse CIRCO compute_metrics function we use a slightly different architecture for this evaluation than with other datasets

import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from datasets.circo import CIRCODataset, build_circo_dataset
from models import TwoEncoderVLM
from torch.utils.data import DataLoader

from utils.decorators import timed_metric


base_path = Path(__file__).absolute().parents[1].absolute()  # Getting the path to the base directory


def compute_metrics(relative_val_dataset: CIRCODataset, predictions_dict: Dict[int, List[int]], ranks: List[int] = [5, 10, 25, 50]) -> Tuple[
    Dict[int, float], Dict[int, float], Dict[str, float]]:
    """Computes the Average Precision (AP) and Recall for a given set of predictions.

    Args:
        data_path (Path): Path where the CIRCO dataset is located
        predictions_dict (Dict[int, List[int]]): Predictions of image ids for each query id
        ranks (List[int], optional): Ranks to consider in the evaluation. Defaults to [5, 10, 25, 50].

    Returns:
        Tuple[Dict[int, float], Dict[int, float], Dict[str, float]]: Dictionaries with the AP and Recall for each rank,
            and the semantic mAP@10 for each semantic aspect
    """

    # relative_val_dataset = CIRCODataset(data_path, split='val', mode='relative', preprocess=None)

    semantic_aspects_list = ['cardinality', 'addition', 'negation', 'direct_addressing', 'compare_change',
                              'comparative_statement', 'statement_with_conjunction', 'spatial_relations_background',
                              'viewpoint']

    # Initialize empty dictionaries to store the AP and Recall values for each rank
    aps_atk = defaultdict(list)
    recalls_atk = defaultdict(list)
    semantic_aps_at10 = defaultdict(list)

    # Iterate through each query id and its corresponding predictions
    for query_id, predictions in predictions_dict.items():
        target = relative_val_dataset.get_target_img_ids(int(query_id))
        semantic_aspects = relative_val_dataset.get_semantic_aspects(int(query_id))
        gt_img_ids = target['gt_img_ids']
        target_img_id = target['target_img_id']

        # Check if the predictions are unique
        if len(set(predictions)) != len(predictions):
            raise ValueError(f"Query {query_id} has duplicate predictions. Please ensure to provide unique predictions"
                             f"for each query.")

        # gt_img_ids = np.trim_zeros(gt_img_ids)  # remove trailing zeros added for collate_fn (when using dataloader)

        predictions = np.array(predictions, dtype=int)
        ap_labels = np.isin(predictions, gt_img_ids)
        precisions = np.cumsum(ap_labels, axis=0) * ap_labels  # Consider only positions corresponding to GTs
        precisions = precisions / np.arange(1, ap_labels.shape[0] + 1)  # Compute precision for each position

        # Compute the AP and Recall for the given ranks
        for rank in ranks:
            aps_atk[rank].append(float(np.sum(precisions[:rank]) / min(len(gt_img_ids), rank)))

        recall_labels = (predictions == target_img_id)
        for rank in ranks:
            recalls_atk[rank].append(float(np.sum(recall_labels[:rank])))

        # Compute the AP@10 for each semantic aspect
        for aspect in semantic_aspects:
            semantic_aps_at10[aspect].append(float(np.sum(precisions[:10]) / min(len(gt_img_ids), 10)))

    # Compute the mean AP and Recall for each rank and store them in a dictionary
    map_atk = {}
    recall_atk = {}
    semantic_map_at10 = {}
    for rank in ranks:
        map_atk[rank] = float(np.mean(aps_atk[rank]))
        recall_atk[rank] = float(np.mean(recalls_atk[rank]))

    # Compute the mean AP@10 for each semantic aspect and store them in a dictionary
    for aspect in semantic_aspects_list:
        semantic_map_at10[aspect] = float(np.mean(semantic_aps_at10[aspect]))

    return map_atk, recall_atk, semantic_map_at10

@torch.no_grad()
def generate_circo_predictions(
    clip_model :TwoEncoderVLM,
    relative_dataset: CIRCODataset,
    fusion_type: str,
    batch_size: int = 64,
    num_workers: int = 4,
    use_tqdm: bool = False,
    accelerator=None,
) -> Dict[int, List[int]]:
    dataloader = DataLoader(
        relative_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_predictions = []
    all_query_ids = []
    all_reference_ids = []

    clip_model.eval()
    vision_encoder = clip_model.vision
    text_encoder = clip_model.text

    for batch in tqdm(dataloader, disable=not use_tqdm, desc="Generating CIRCO predictions"):
        reference_images = batch['reference_img'].to(vision_encoder.device)
        input_ids = batch['input_ids'].to(text_encoder.device)
        attention_mask = batch['attention_mask'].to(text_encoder.device)
        query_ids = batch['query_id']
        reference_id = batch['reference_imd_id']

        image_features = vision_encoder(reference_images).image_embeds
        text_features = text_encoder(input_ids=input_ids, attention_mask=attention_mask).text_embeds

        if fusion_type == 'sum':
            predicted_features = image_features + text_features
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")
        
        all_predictions.append(predicted_features)
        all_query_ids.extend(query_ids)
        all_reference_ids.extend(reference_id)

    all_predictions = torch.vstack(all_predictions)
    return all_predictions, all_query_ids, all_reference_ids

@torch.no_grad()
def generate_circo_index_features(
    clip_model :TwoEncoderVLM,
    database_dataset: CIRCODataset,
    batch_size: int = 64,
    num_workers: int = 4,
    use_tqdm: bool = False,
    accelerator=None,
) -> Dict[str, torch.Tensor]:
    dataloader = DataLoader(
        database_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_image_features = []
    all_image_ids = []

    clip_model.eval()
    vision_encoder = clip_model.vision

    for batch in tqdm(dataloader, disable=not use_tqdm, desc="Generating CIRCO index features"):
        images = batch['img'].to(vision_encoder.device)
        image_ids = batch['img_id']

        image_features = vision_encoder(images).image_embeds

        all_image_features.append(image_features)
        all_image_ids.extend(image_ids)

    all_image_features = torch.vstack(all_image_features)
    return all_image_features, all_image_ids

def compute_prediction_dict(
    predicted_features: torch.Tensor, #(N, D)
    query_ids: List[int],   #N
    reference_ids: List[int], #N
    index_features: torch.Tensor, #(M, D)
    index_ids: List[int], #M
    num_candidates: int = -1  #to reduce the number of candidates returned per query
) -> Dict[int, List[int]]:
    
    N = predicted_features.shape[0] #number of queries
    M = index_features.shape[0] #number of database images
    assert N == len(query_ids), f"Expected {N} query_ids, but got {len(query_ids)}"
    assert N == len(reference_ids), f"Expected {N} reference_ids, but got {len(reference_ids)}"
    assert M == len(index_ids), f"Expected {M} index_ids, but got {len(index_ids)}"
    assert predicted_features.shape[1] == index_features.shape[1], f"Feature dimension mismatch: {predicted_features.shape[1]} vs {index_features.shape[1]}"
    
    similarity_matrix = predicted_features @ index_features.T #(N, M)
    sorted_indeces = torch.argsort(similarity_matrix, dim=1, descending=True).cpu() #(N, M)
    sorted_index_ids = np.array(index_ids)[sorted_indeces] #(N, M)

    # remove reference image from candidates
    reference_mask = (np.array(reference_ids)[:, np.newaxis] != sorted_index_ids).reshape(N, M) #(N, M)
    sorted_index_ids = sorted_index_ids[reference_mask].reshape(N, M -1) #(N, M-1)

    #build prediction dict
    if num_candidates < 0:
        num_candidates = M -1
    predictions_dict = {}
    for i in range(N):
        query_id = query_ids[i]
        predictions = sorted_index_ids[i][:num_candidates].tolist()
        predictions_dict[query_id] = predictions

    return predictions_dict

@timed_metric
def evaluate_circo(
    model: TwoEncoderVLM,
    fusion_type: str = 'sum',
    batch_size: int = 64,
    num_workers: int = 4,
    tqdm: bool = True,
    accelerator=None,
    return_index_tuple: bool = False,
    index_tuple: Tuple[torch.Tensor, List[int]] = None,
) -> Tuple[Dict[int, float], Tuple[torch.Tensor, List[int]]] | Dict[int, float]:
    relative_dataset = build_circo_dataset(
        split='val',
        mode='relative',
        preprocess=model.image_processor,
        tokenizer=model.tokenizer,
        max_length_tokenizer=77,
    )
    if index_tuple is None:
        database_dataset = build_circo_dataset(
            split='val',
            mode='classic',
            preprocess=model.image_processor,
            tokenizer=model.tokenizer,
            max_length_tokenizer=77,
        )

    predicted_features, query_ids, reference_ids = generate_circo_predictions(
        model,
        relative_dataset,
        fusion_type,
        batch_size,
        num_workers,
        tqdm,
        accelerator
    )

    if index_tuple is None:
        index_features, index_ids = generate_circo_index_features(
            model,
            database_dataset,
            batch_size,
            num_workers,
            tqdm,
            accelerator
        )
    else:
        index_features, index_ids = index_tuple

    predictions_dict = compute_prediction_dict(
        predicted_features,
        query_ids,
        reference_ids,
        index_features,
        index_ids,
    )

    map_atk, recall_atk, semantic_map_at10 = compute_metrics(
        relative_dataset,
        predictions_dict
    )

    metrics = {}
    for rank, value in map_atk.items():
        metrics[f'mAP_at{rank}'] = value * 100
    for rank, value in recall_atk.items():
        metrics[f'recall_at{rank}'] = value * 100
    for aspect, value in semantic_map_at10.items():
        metrics[f'semantic_mAP_at10_{aspect}'] = value * 100

    if return_index_tuple:
        return metrics, (index_features, index_ids)
    return metrics
    
def generate_circo_test_submission(
    model: TwoEncoderVLM,
    fusion_type: str = 'sum',
    batch_size: int = 64,
    num_workers: int = 4,
    tqdm: bool = True,
    accelerator=None,
    return_index_tuple: bool = False,
    index_tuple: Tuple[torch.Tensor, List[int]] = None,
) -> Dict[int, List[int]] | Tuple[Dict[int, List[int]], Tuple[torch.Tensor, List[int]]]:
    relative_dataset = build_circo_dataset(
        split='test',
        mode='relative',
        preprocess=model.image_processor,
        tokenizer=model.tokenizer,
        max_length_tokenizer=77,
    )
    if index_tuple is None:
        database_dataset = build_circo_dataset(
            split='test',
            mode='classic',
            preprocess=model.image_processor,
            tokenizer=model.tokenizer,
            max_length_tokenizer=77,
        )

    predicted_features, query_ids, reference_ids = generate_circo_predictions(
        model,
        relative_dataset,
        fusion_type,
        batch_size,
        num_workers,
        tqdm,
        accelerator
    )

    if index_tuple is None:
        index_features, index_ids = generate_circo_index_features(
        model,
        database_dataset,
        batch_size,
        num_workers,
        tqdm,
        accelerator
        )
    else:
        index_features, index_ids = index_tuple

    predictions_dict = compute_prediction_dict(
        predicted_features,
        query_ids,
        reference_ids,
        index_features,
        index_ids,
        num_candidates=50
    )

    if return_index_tuple:
        return predictions_dict, (index_features, index_ids)
    return predictions_dict
        
