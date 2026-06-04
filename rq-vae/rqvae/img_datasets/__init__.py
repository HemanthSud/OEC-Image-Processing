# Copyright (c) 2022-present, Kakao Brain Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch

from .transforms import create_transforms
from .eurosat import EuroSAT
from .flair import FLAIR

SMOKE_TEST = bool(os.environ.get("SMOKE_TEST", 0))


def _dataset_max_samples(config, split):
    key = f'max_{split}_samples'
    return config.dataset.get(key, config.dataset.get('max_samples', None))


def create_dataset_split(config, split='train', is_eval=False):
    transform_split = 'train' if split == 'train' else 'val'
    transforms_ = create_transforms(
        config.dataset, split=transform_split, is_eval=is_eval or split != 'train')

    root = config.dataset.get('root', None)

    if config.dataset.type == 'eurosat':
        root = root if root else '../EuroSAT_RGB'
        split_indices_path = config.dataset.get(
            'split_indices_path', '../eurosat_split_indices.pt')
        return EuroSAT(root, split=split, transform=transforms_,
                       split_indices_path=split_indices_path,
                       max_samples=_dataset_max_samples(config, split))

    if config.dataset.type in ('flair', 'flair1', 'flair_rgb'):
        csv_key = f'{split}_csv'
        csv_path = config.dataset.get(csv_key, None)
        if csv_path is None and split == 'test':
            csv_path = config.dataset.get('val_csv', None)
        if csv_path is None:
            raise ValueError(f"Missing dataset.{csv_key} for FLAIR split '{split}'")

        channels = list(config.dataset.get('channels', [1, 2, 3]))
        return FLAIR(csv_path=csv_path,
                     split=split,
                     transform=transforms_,
                     channels=channels,
                     data_root=config.dataset.get('data_root', None),
                     max_samples=_dataset_max_samples(config, split),
                     png_root=config.dataset.get('png_root', None))

    raise ValueError('%s not supported...' % config.dataset.type)


def create_dataset(config, is_eval=False, logger=None):
    dataset_trn = create_dataset_split(config, split='train', is_eval=is_eval)
    dataset_val = create_dataset_split(config, split='val', is_eval=True)

    if SMOKE_TEST:
        dataset_len = config.experiment.total_batch_size * 2
        dataset_trn = torch.utils.data.Subset(
            dataset_trn, torch.randperm(len(dataset_trn))[:dataset_len])
        dataset_val = torch.utils.data.Subset(
            dataset_val, torch.randperm(len(dataset_val))[:dataset_len])

    if logger is not None:
        logger.info(
            f'#train samples: {len(dataset_trn)}, #valid samples: {len(dataset_val)}')

    return dataset_trn, dataset_val
