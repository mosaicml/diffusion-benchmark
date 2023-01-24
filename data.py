# Copyright 2022 MosaicML
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
from typing import Callable, Optional

import streaming
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPTokenizer


class StreamingLAIONDataset(streaming.StreamingDataset):
    """
    Implementation of the LAION dataset as a streaming dataset.

    Args:
        remote (str): Remote directory (S3 or local filesystem) where dataset is stored.
        local (str): Local filesystem directory where dataset is cached during operation.
        split (str): The dataset split to use. Currently, only ``None`` is supported.
        shuffle (bool): Whether to shuffle the samples in this dataset
        tokenizer_name_or_path (str): The name or path of the tokenizer to use. Default: ``'stabilityai/stable-diffusion-2-base'``.
        transform (Optional[Callable]): The transforms to apply to the image. Default: ``None``.
        predownload (Optional[int]): The number of samples to prefetch. Default: ``100_000``.
        download_retry (Optional[int]): The number of times to retry a download. Default: ``2``.
        download_timeout (Optional[float]): The timeout for a download. Default: ``120``.
        batch_size (Optional[int]): Hint batch_size that will be used on each device's DataLoader. Default: ``None``.
    """

    def __init__(
            self,
            remote: str,
            local: str,
            split: str,
            shuffle: bool,
            tokenizer_name_or_path: str = 'stabilityai/stable-diffusion-2-base',
            transform: Optional[Callable] = None,
            predownload: Optional[int] = 100_000,
            download_retry: Optional[int] = 2,
            download_timeout: Optional[float] = 120,
            batch_size: Optional[int] = None) -> None:
        super().__init__(local, remote)

        self.transform = transform
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name_or_path,
                                                       subfolder='tokenizer')

        super().__init__(remote=remote,
                         local=local,
                         split=split,
                         shuffle=shuffle,
                         predownload=predownload,
                         keep_zip=False,
                         download_retry=download_retry,
                         download_timeout=download_timeout,
                         validate_hash=None,
                         batch_size=batch_size)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        img = Image.open(BytesIO(sample['jpg']))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        caption = sample['caption']
        tokenized_caption = self.tokenizer(
            caption,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True)['input_ids']  # TODO: do we need attention masks
        tokenized_caption = torch.tensor(tokenized_caption)
        return {'image': img, 'caption': tokenized_caption}

class SyntheticImageCaptionDataset(Dataset):
    """Synthetic dataset imitating a dataset of images plus captions."""

    def __init__(self,
                 image_size=512,
                 caption_length=77,
                 num_samples=100_000):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            image_size (int): Size of the images.
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.caption_length = caption_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(3, self.image_size, self.image_size)
        caption = torch.randint(0, 128, (self.caption_length,), dtype=torch.long)
        return {'image': image, 'caption': caption}
