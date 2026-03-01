import os
import json
from collections import OrderedDict, defaultdict
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import itertools
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset


class MSCOCOCaptions(Dataset):
    '''
    Args:
        root (string): Root directory where images are downloaded to.
        annotations_file (string): Path to annotation file.
        image_transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        caption_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        max_length_tokenizer (int): The maximum length required by some text tokenizers,
        no_cap_per_img (int): The number of captions for an image. Could be between 1 and 5 
            since each image in MSCOCO has 5 captions.
        resize_dataset (bool): If True, resize the dataset to 10 samples for quick testing.
    '''

    def __init__(
        self,
        root: str,
        annotations_file: str,
        image_transform: Optional[Callable] = None,
        caption_transform: Optional[Callable] = None,
        max_length_tokenizer: int = 77,
        no_cap_per_img = 1,
        classified_ann_file: str = None,
        no_image_per_cls: int = 2,
        resize_dataset: bool = False,
    ):
        super(MSCOCOCaptions, self).__init__()

        self.name = 'MSCOCO'
        self.root = root
        self.image_transform = image_transform
        self.caption_transform = caption_transform
        self.max_length_tokenizer = max_length_tokenizer
        self.cpi = no_cap_per_img
        self.resize_dataset = resize_dataset

        f_name = Path(annotations_file)
        with f_name.open('rt') as handle:
            annotations = json.load(handle, object_hook=OrderedDict)

        self.img_id_to_file_name = {}
        for img_info in annotations['images']:
            img_id = img_info['id']
            file_name = img_info['file_name']
            self.img_id_to_file_name[img_id] = file_name
    
        self.img_id_to_captions = defaultdict(list)
        for caption_info in annotations['annotations']:
            img_id = caption_info['image_id']
            self.img_id_to_captions[img_id].append(caption_info['caption'])

        if classified_ann_file is None:
            self.img_ids = list(self.img_id_to_file_name.keys())
        else:
            df = pd.read_csv(classified_ann_file)
            df = df.groupby('categories').filter(lambda x: len(x) >= 2)
            df = df.groupby('categories')[['categories', 'image_id']].apply(lambda x: x.sample(n=no_image_per_cls)).reset_index(drop=True)
            class_img_id = df.groupby('categories')['image_id'].apply(list).to_dict()

            self.img_ids = [img_ids for img_ids in class_img_id.values()]
            self.img_ids = list(itertools.chain(*self.img_ids))
            

    def __getitem__(self, index: int):
        """
        Args:
            index (int): index in [0, self.__len__())

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """

        img_id = self.img_ids[index]
        filename = os.path.join(self.root, self.img_id_to_file_name[img_id])
        img = Image.open(filename).convert('RGB')

        if self.image_transform is not None:
            img = self.image_transform(img, return_tensors='pt')
            # print(f"Debug: img after transform type: {type(img)}")
            # print(img.keys())
            # print(img['pixel_values'].shape)

            img = img['pixel_values'].squeeze(0)

        captions = list(
            map(str, np.random.choice(self.img_id_to_captions[img_id], size=self.cpi, replace=False))
            )
        
        transformed_captions = captions
            
        if self.caption_transform is not None:
            transformed_captions = self.caption_transform(
                captions,
                padding='max_length',
                max_length=self.max_length_tokenizer,
                truncation=True,
                return_tensors='pt')

        return {
            'pixel_values': img,
            'input_ids': transformed_captions["input_ids"],
            'attention_mask': transformed_captions["attention_mask"],
            'labels': img_id,
        }


    def __len__(self) -> int:
        if self.resize_dataset:
            return 8
        return len(self.img_ids)
    
    def collate_fn(self, batch):
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        input_ids = torch.cat([item['input_ids'] for item in batch], dim=0)
        attention_mask = torch.cat([item['attention_mask'] for item in batch], dim=0)
        labels = [item['labels'] for item in batch]
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
def build_mscoco_dataset(split, image_transform, caption_transform, cpi=1):
    if split == 'train':
        annotations_file = 'data/mscoco/annotations/captions_train2017.json'
        image_root = 'data/mscoco/images/train2017'
    elif split == 'val':
        annotations_file = 'data/mscoco/annotations/captions_val2017.json'
        image_root = 'data/mscoco/images/val2017'
    else:
        raise ValueError(f"Invalid split name: {split}")

    dataset = MSCOCOCaptions(
        root=image_root,
        annotations_file=annotations_file,
        image_transform=image_transform,
        caption_transform=caption_transform,
        max_length_tokenizer=77,
        no_cap_per_img=1,
    )
    return dataset