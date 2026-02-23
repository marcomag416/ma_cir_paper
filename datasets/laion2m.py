import os
import json
from collections import OrderedDict
# import numpy as np
# import pandas as pd
# from pathlib import Path
from PIL import Image
from typing import Callable, Optional, Tuple
from tqdm.auto import tqdm
import torch

from torch.utils.data import Dataset


class Laion2MDataset(Dataset):
    '''
    Args:
        root (string): Root directory where images and annotations are downloaded to.
        image_transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        caption_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        max_length_tokenizer (int): The maximum length required by some text tokenizers,
        no_cap_per_img (int): The number of captions for an image. Could be between 1 and 5 
    '''

    def __init__(
        self,
        root: str,
        split: str = 'train',
        image_transform: Optional[Callable] = None,
        caption_transform: Optional[Callable] = None,
        max_length_tokenizer: int = 77,
        num_val_file: int = 5,
        verbose: bool = False,
    ):
        super(Laion2MDataset, self).__init__()

        assert split in ['train', 'val', 'all'], f"split must be one of ['train', 'val'], found {split} instead."

        self.name = 'Laion2M'
        self.root = root
        self.image_transform = image_transform
        self.caption_transform = caption_transform
        self.max_length_tokenizer = max_length_tokenizer


        #create a list with all json files in the directory
        json_files = sorted([f for f in os.listdir(root) if f.endswith('.json')])

        #split the json files into train and val
        if split == 'val':
            json_files = json_files[:num_val_file]
        elif split == 'train':
            json_files = json_files[num_val_file:]

        self.index_mapping = OrderedDict()
        accum = 0
        #prepare dataset 
        for json_file in tqdm(json_files):
            # index is the number before _
            stats_file = os.path.join(root, json_file)
            index = json_file.split('_')[0]
            image_tar = self.get_image_tarname(index)
            image_dir = self.get_dirname(index)
            # check all the files exists
            if not os.path.exists(image_dir):
                if os.path.exists(image_tar):
                    if verbose:
                        print(f"Extracting image file {image_tar} to {image_dir}...")
                    #untar the file and name the directory as the tar file without .tar
                    os.makedirs(image_dir, exist_ok=True)
                    try:
                        os.system(f"tar -xf {image_tar} -C {image_dir}")
                        os.remove(image_tar)
                    except Exception as e:
                        if verbose:
                            print(f"Error extracting {image_tar} to {image_dir}: {e}")
                else:
                    if verbose:
                        print(f"Image .tar file {image_tar} or directory {image_dir} does not exist. Skipping this sample")
                    continue

            #check if samples_id.csv exists, otherwise create it
            if not os.path.exists(os.path.join(image_dir, 'samples_id.csv')):
                if verbose:
                    print(f"Creating samples_id.csv in {image_dir}...")
                with open(os.path.join(image_dir, 'samples_id.csv'), 'x') as samples_file:
                    samples_file.write('sample_id,image,caption\n')
                    sample_id = 0
                    for f in sorted(os.listdir(image_dir)):
                        if f.endswith('.txt'):
                            image_name = f.replace('.txt', '.jpg')
                            samples_file.write(f"{sample_id},{image_name},{f}\n")
                            sample_id += 1

            #update the directory index mapping with the number of samples in the stats file
            if verbose:
                print(f"Loading stats file {stats_file}...")
            stats = json.load(open(stats_file, 'r'))
            self.index_mapping[accum] = index
            accum += stats["successes"]
        self.length = accum
    
    def get_image_tarname(self, index:int) -> str:
        '''
        Get the image tar filename for a given index
        Args:
            index (int): index of the image tar to get
        '''
        return os.path.join(self.root, index + '.tar')
    
    def get_dirname(self, index:int) -> str:
        '''
        Get the image directory name for a given index
        Args:
            index (int): index of the image directory to get
        '''
        return os.path.join(self.root, index)
    
    # def get_sample2dir_index(self, index:int) -> Tuple[int, int]:
    #     '''
    #     Get the directory index for a given sample index, and the accum value
    #     Args:
    #         index (int): index of the sample to get the directory index for
    #     '''
    #     keys = list(self.index_mapping.keys())
    #     for i in range(len(keys)):
    #         if index < keys[i]:
    #             return self.index_mapping[keys[i-1]], keys[i-1]
    #     return self.index_mapping[keys[-1]], keys[-1]

    #same as above but with binary search
    def get_sample2dir_index(self, index:int) -> Tuple[int, int]:
        '''
        Get the directory index for a given sample index, and the accum value
        Args:
            index (int): index of the sample to get the directory index for
        '''
        keys = list(self.index_mapping.keys())
        i_small = 0
        i_large = len(keys) -1
        if index >= keys[-1]:
            return self.index_mapping[keys[-1]], keys[-1]
        while True:
            i = (i_small + i_large) // 2
            if index < keys[i]:
                i_large = i
            elif index >= keys[i + 1]:
                i_small = i + 1
            else:
                return self.index_mapping[keys[i]], keys[i]

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

    def __getitem__(self, index: int):
        """
        Args:
            index (int): index in [0, self.__len__())

        Returns:
            tuple: Tuple (candidate, target, tranformed_query, text_query). target is a list of captions for the image.
        """
        dir_index, accum = self.get_sample2dir_index(index)
        dirname = self.get_dirname(dir_index)

        if(index > self.length):
            raise IndexError(f"Index {index} out of range for dataset of length {self.length}")
        
        #get image and caption files name from samples_id.csv
        with open(os.path.join(dirname, 'samples_id.csv'), 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    raise IndexError(f"Index {index} (sample_id:{index-accum}) not listed in {dirname}/samples_id.csv")
                if line.startswith('sample_id'):
                    continue
                sample_id, imagefile, captionfile = line.strip().split(',')
                if index - accum == int(sample_id):
                    break

        image_path = os.path.join(dirname, imagefile)
        caption_path = os.path.join(dirname, captionfile)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} does not exist.")
        if not os.path.exists(caption_path):
            raise FileNotFoundError(f"Caption file {caption_path} does not exist.")
        
        img = Image.open(image_path).convert('RGB')
        
        if self.image_transform is not None:
            img = self.image_transform(img, return_tensors='pt')
            img = img['pixel_values'].squeeze(0)

        caption = open(caption_path, 'r').read().strip()
        
        transformed_captions = caption
            
        if self.caption_transform is not None:
            transformed_captions = self.caption_transform(
                caption,
                padding='max_length',
                max_length=self.max_length_tokenizer,
                truncation=True,
                return_tensors='pt')

        return {
            'pixel_values': img,
            'input_ids': transformed_captions["input_ids"],
            'attention_mask': transformed_captions["attention_mask"],
            'labels': index,
        }

    def __len__(self) -> int:
        return self.length

def build_laion_dataset(split, image_transform, caption_transform, cpi=1):
    return Laion2MDataset(
        root='data/laion2m',
        split=split,
        image_transform=image_transform,
        caption_transform=caption_transform,
        max_length_tokenizer=77,
        num_val_file=5,
        verbose=False
    )