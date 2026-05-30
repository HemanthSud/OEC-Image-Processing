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

import torchvision.transforms as transforms


def create_transforms(config, split='train', is_eval=False):
    if 'eurosat' in config.transforms.type:
        image_size = config.transforms.get('image_size', 64)
        if split == 'train' and not is_eval:
            transforms_ = [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        else:
            transforms_ = [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]

    elif 'flair' in config.transforms.type:
        image_size = config.transforms.get('image_size', 512)
        channels = config.transforms.get('channels', [1, 2, 3])
        means = [0.5] * len(channels)
        stds = [0.5] * len(channels)

        transforms_ = []
        if image_size:
            transforms_.append(
                transforms.Resize((image_size, image_size), antialias=True)
            )
        if split == 'train' and not is_eval:
            transforms_.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ])
        transforms_.append(transforms.Normalize(means, stds))

    elif config.transforms.type == 'none':
        transforms_ = []
    else:
        raise NotImplementedError('%s not implemented..' %
                                  config.transforms.type)

    transforms_ = transforms.Compose(transforms_)

    return transforms_
