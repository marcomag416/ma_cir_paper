from typing import Literal, Optional, Callable

from datasets import load_dataset
import torch
from torch.utils.data import Dataset


class CC3Mdataset(Dataset):
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
        split: Literal["train", "val"] = "train",
        image_transform: Optional[Callable] = None,
        caption_transform: Optional[Callable] = None,
        max_length_tokenizer: int = 77,
        resize_dataset: bool = False,
        cache_path: str = "data/cc3m/cache",
    ):
        super(CC3Mdataset, self).__init__()

        self.name = 'CC3M'
        self.image_transform = image_transform
        self.caption_transform = caption_transform
        self.max_length_tokenizer = max_length_tokenizer
        self.resize_dataset = resize_dataset

        if split == "val":
            split = "validation"

        self.data = load_dataset("pixparse/cc3m-wds", cache_dir=cache_path, split=split)

    def __len__(self):
        if self.resize_dataset:
            return 10
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img = item["jpg"]
        caption = item["txt"]
        id = item["__key__"]

        if self.image_transform is not None:
            img = self.image_transform(img, return_tensors='pt')['pixel_values'].squeeze(0)

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
            'labels': id,
        }
    
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

def build_cc3m_dataset(split, image_transform, caption_transform, max_length_tokenizer=77, resize_dataset=False):
    dataset = CC3Mdataset(
        split=split,
        image_transform=image_transform,
        caption_transform=caption_transform,
        max_length_tokenizer=max_length_tokenizer,
        resize_dataset=resize_dataset,
    )
    return dataset
