import PIL
from torch.utils.data import Dataset
from typing import Callable, Optional, Literal
import json
import os
import torch
import pandas as pd

class SIMATDataset(Dataset):
    '''
    Args:
        images_dirpath (str): Directory where images are stored.
        annotations_file (string): Path to annotation file.
        split (str): Dataset split, one of ['train', 'val', 'test'].
        image_transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        caption_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        max_length_tokenizer (int): The maximum length required by some text tokenizers,
    '''

    def __init__(
        self,
        dataset_path: str,
        split: Literal['test', 'val'] = 'val',
        mode: Literal['words', 'images', 'none'] = 'none',
        image_transform: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
        max_length_tokenizer: int = 77,
    ):
        super(SIMATDataset, self).__init__()

        self.name = 'SIMAT'
        self.split = split
        self.mode = mode
        self.dataset_path = dataset_path
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.max_length_tokenizer = max_length_tokenizer

        self.img_dir_path = os.path.join(self.dataset_path, 'images')
        self.ann_dir_path = os.path.join(self.dataset_path, 'annotations')

        if self.mode == "words":
            transfos = self.get_transfos_df()
            self.words_list = list(set(transfos.target) | set(transfos.value))

        if self.mode == "images":
            self.image_list = os.listdir(self.img_dir_path)

    def get_oscar_scores(self) -> torch.Tensor:
        return torch.load(os.path.join(self.ann_dir_path, 'oscar_similarity_matrix.pt'))
    
    def get_triplets_df(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.ann_dir_path, 'triplets.csv'), index_col=0)
    
    def get_transfos_df(self) -> pd.DataFrame:
        transfos_df =  pd.read_csv(os.path.join(self.ann_dir_path, 'transfos.csv'), index_col=0)
        return transfos_df[transfos_df.is_test == (self.split == 'test')]
    
    def get_did2rid_map(self) -> dict[int, int]:
        triplets = self.get_triplets_df()
        return dict(zip(triplets.dataset_id, triplets.index))
    
    def get_rid2did_map(self) -> dict[int, int]:
        triplets = self.get_triplets_df()
        return dict(zip(triplets.index, triplets.dataset_id))
    
    def __getitem__(self, index):
        if self.mode == "words":
            word = self.words_list[index]

            if self.tokenizer is not None:
                tokenized_word = self.tokenizer(
                    word,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length_tokenizer,
                    return_tensors='pt'
                )
            else:
                tokenized_word = {"input_ids": [None], "attention_mask": [None]}

            return {
                'word': word,
                'input_ids': tokenized_word['input_ids'][0],
                'attention_mask': tokenized_word['attention_mask'][0]
            }
        
        elif self.mode == "images":
            image_name = self.image_list[index]
            image_path = os.path.join(self.img_dir_path, image_name)
            #region id is the image file name without extension
            region_id = int(os.path.splitext(image_name)[0])

            image = PIL.Image.open(image_path).convert('RGB')
            
            if self.image_transform is not None:
                image = self.image_transform(
                    image,
                    return_tensors='pt'
                )['pixel_values'][0]

            return {
                'image': image,
                'region_id': region_id
            }
        
        else:
            return {}

    def __len__(self):
        if self.mode == "words":
            return len(self.words_list)
        elif self.mode == "images":
            return len(self.image_list)
        else:
            return 0
    
def build_simat_dataset(
    split: Literal['test', 'val'] = 'val',
    mode: Literal['words', 'images', 'none'] = 'none',
    image_transform: Optional[Callable] = None,
    tokenizer: Optional[Callable] = None,
    max_length_tokenizer: int = 77,
    dataset_path: str = 'data/simat',
) -> SIMATDataset:
    dataset = SIMATDataset(
        dataset_path=dataset_path,
        split=split,
        mode=mode,
        image_transform=image_transform,
        tokenizer=tokenizer,
        max_length_tokenizer=max_length_tokenizer,
    )
    return dataset