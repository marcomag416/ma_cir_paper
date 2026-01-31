from typing import Literal
import torch.nn as nn
from torchvision import datasets
import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
from datasets.simat import SIMATDataset, build_simat_dataset
from models import TwoEncoderVLM
from utils.decorators import timed_metric

DEBUG=False


def take_topk(candidates, target, top_k):
    tops = [r for r in candidates if r.item() != target]

    return tops[:top_k]

def make_normalized(x):
    return x/x.norm(dim=-1, keepdim=True)


def compute_simat_scores(
        simat_dataset : SIMATDataset,
        image_embeddings, 
        words_embeddings, 
        lbds=[1,], 
        top_k=[1,]
):
    # Adapted from https://github.com/facebookresearch/SIMAT/blob/main/eval.py

    output = {}
    transfos = simat_dataset.get_transfos_df()
    did2rid = simat_dataset.get_did2rid_map()
    rid2did = simat_dataset.get_rid2did_map()
        
    transfos_did = [rid2did[rid] for rid in transfos.region_id]

    if DEBUG:
        print(f"\nimg_embeds length: {len(image_embeddings)}. Type: {type(image_embeddings)}")
        print(f"words_embeds length: {len(words_embeddings)}")
        print(f"transfos length: {len(transfos)}")
        print(f"transfos_did length: {len(transfos_did)}")
        print(f"did2rid length: {len(did2rid)}. type: {type(did2rid)}")
        print(f"rid2did length: {len(rid2did)}. type: {type(rid2did)}")
    
    img_embs_stacked = torch.stack([image_embeddings[did2rid[i]] for i in range(len(image_embeddings))]).float()
    img_embs_stacked = make_normalized(img_embs_stacked)
    value_embs = torch.stack([img_embs_stacked[did] for did in transfos_did])
    
    w2v = {k:make_normalized(v.float()) for k, v in words_embeddings.items()}
    delta_vectors = torch.stack([w2v[x.target] - w2v[x.value] for i, x in transfos.iterrows()])
    
    oscar_scores = simat_dataset.get_oscar_scores()
    weights = 1/np.array(transfos.norm2)**.5
    weights = weights/sum(weights)

    for lbd in lbds:
        for k in top_k:
            target_embs = value_embs + lbd*delta_vectors

            nnb = (target_embs @ img_embs_stacked.T).topk(2*k).indices
            nnb_notself = [take_topk(r, t, k) for r, t in zip(nnb, transfos_did)]

            scores = []
            for ri, tc in zip(nnb_notself, transfos.target_ids):
                temp = []
                for rii in ri:
                    if oscar_scores[rii.item(), tc] > 0.5:
                        temp.append(True)
                    else:
                        temp.append(False)

                temp = any(temp)
                if temp:
                    scores.append(1)
                else:
                    scores.append(0)
            
            
            output[(lbd, k)] = float(100*np.average(scores, weights=weights))

    return output

@torch.no_grad()
def encode_simat_words(
        clip_model : TwoEncoderVLM,
        simat_words_dataset : SIMATDataset,
        batch_size : int = 64,
        num_workers : int = 4,
        use_tqdm : bool = False,
        accelerator = None,
):
    dataloader = torch.utils.data.DataLoader(
        simat_words_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_word_embeddings = {}
    clip_model.eval()
    text_encoder = clip_model.text

    for batch in tqdm(dataloader, disable=not use_tqdm, desc="Encoding SIMAT words"):
        input_ids = batch['input_ids'].to(text_encoder.device)
        attention_mask = batch['attention_mask'].to(text_encoder.device)
        words = batch['word']

        embeddings = text_encoder(input_ids=input_ids, attention_mask=attention_mask ).text_embeds.cpu()

        all_word_embeddings.update(dict(zip(words, embeddings)))
    
    return all_word_embeddings

@torch.no_grad()
def encode_simat_images(
        clip_model : TwoEncoderVLM,
        simat_images_dataset : SIMATDataset,
        batch_size : int = 64,
        num_workers : int = 4,
        use_tqdm : bool = False,
        accelerator = None,
):
    dataloader = torch.utils.data.DataLoader(
        simat_images_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_image_embeddings = {}
    clip_model.eval()
    image_encoder = clip_model.vision

    for batch in tqdm(dataloader, disable=not use_tqdm, desc="Encoding SIMAT images"):
        pixel_values = batch['image'].to(image_encoder.device)
        region_ids = batch['region_id'].tolist()

        embeddings = image_encoder(pixel_values=pixel_values).image_embeds.cpu()

        all_image_embeddings.update(dict(zip(region_ids, embeddings)))
    
    return all_image_embeddings

@timed_metric
def evaluate_simat(
    model : TwoEncoderVLM,
    batch_size : int = 64,
    num_workers : int = 4,
    tqdm: bool = False,
    accelerator = None,
    split: Literal['test', 'val'] = 'val',
):
    simat_words_dataset = build_simat_dataset(
        split=split,
        mode='words',
        tokenizer=model.tokenizer,
        max_length_tokenizer=77,
    )

    simat_images_dataset = build_simat_dataset(
        split=split,
        mode='images',
        image_transform=model.image_processor,
    )

    words_embeddings = encode_simat_words(
        clip_model=model,
        simat_words_dataset=simat_words_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        use_tqdm=tqdm,
        accelerator=accelerator,
    )

    image_embeddings = encode_simat_images(
        clip_model=model,
        simat_images_dataset=simat_images_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        use_tqdm=tqdm,
        accelerator=accelerator,
    )

    scores = compute_simat_scores(
        simat_dataset=simat_words_dataset,
        image_embeddings=image_embeddings,
        words_embeddings=words_embeddings,
        lbds=[1,],
        top_k=[1,]
    )

    metrics = {f"score_at{k}_lmbd{lbd}": value for (lbd, k), value in scores.items()}
        
    return metrics


DEFAULT_OUTPUT_DIR = os.path.join('results', 'simat', 'sim_distribution')
def plot_simat_sim_distribution(image_embeddings, words_embeddings, domain: Literal['test', 'val'] = 'test', lbd=1.0, output_dir=DEFAULT_OUTPUT_DIR):
    transfos = pd.read_csv('data/annotations/simat/transfos.csv', index_col=0)
    triplets = pd.read_csv('data/annotations/simat/triplets.csv', index_col=0)
    did2rid = dict(zip(triplets.dataset_id, triplets.index))
    rid2did = dict(zip(triplets.index, triplets.dataset_id))

    transfos = transfos[transfos.is_test == (domain == 'test')]
    transfos_did = [rid2did[rid] for rid in transfos.region_id]

    img_embs_stacked = torch.stack([image_embeddings[did2rid[i]] for i in range(len(image_embeddings))]).float()
    img_embs_stacked = make_normalized(img_embs_stacked)
    value_embs = torch.stack([img_embs_stacked[did] for did in transfos_did])

    w2v = {k:make_normalized(v.float()) for k, v in words_embeddings.items()}
    delta_vectors = torch.stack([w2v[x.target] - w2v[x.value] for i, x in transfos.iterrows()])


    oscar_scores = torch.load('data/annotations/simat/oscar_similarity_matrix.pt')
    # weights = 1/np.array(transfos.norm2)**.5
    # weights = weights/sum(weights)


    target_embs = value_embs + lbd*delta_vectors
    target_embs = make_normalized(target_embs)

    # top_target_ids = oscar_scores.max(axis=0).indices
    top_region_ids = oscar_scores.topk(2, dim=0).indices
    # display(top_region_ids.shape)
    gt_image_ids = torch.stack([top_region_ids[0,t] if top_region_ids[0,t] != r else top_region_ids[1,t] for t, r in zip(transfos.target_ids, transfos.region_id)])
    gt_images = img_embs_stacked[gt_image_ids]
    pos_similarities = (target_embs * gt_images).sum(dim=-1)
    # display(pos_similarities.shape)

    rnd_image_ids = torch.randint(0, img_embs_stacked.shape[0], (len(transfos.target_ids),))
    rnd_images = img_embs_stacked[rnd_image_ids]
    rnd_similarities = (target_embs * rnd_images).sum(dim=-1)
    # display(rnd_similarities.shape)

    # worst_region_ids = oscar_scores.topk(2, dim=0, largest=False).indices
    # # display(worst_region_ids.shape)
    # worst_images_ids = torch.stack([worst_region_ids[0,t] if worst_region_ids[0,t] != r else worst_region_ids[1,t] for t, r in zip(transfos.target_ids, transfos.region_id)])
    # worst_images = img_embs_stacked[worst_images_ids]
    # worst_similarities = (target_embs * worst_images).sum(dim=-1)
    # display(worst_similarities.shape) 

    pos_mean = pos_similarities.mean().item()
    rnd_mean = rnd_similarities.mean().item()


    plt.figure(figsize=(10,6))
    plt.hist(pos_similarities.detach().cpu().numpy(), bins=50, alpha=0.5, label='Targets')
    plt.hist(rnd_similarities.detach().cpu().numpy(), bins=50, alpha=0.5, label='Random images')
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
    plt.savefig(os.path.join(output_dir, f'simat_sim_distribution.svg'), format='svg')
    plt.close()