import os

from torch.utils.data import Dataset
from torch.utils.data import Sampler
from PIL import Image
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized
from inversion_utils.data_utils import make_dataset
import cv2
import numpy as np
import torch
import random
import json
from tqdm import tqdm
import copy

from torchvision import transforms

class ImagesDataset(Dataset):

    def __init__(self, source_root, source_transform=None):
        self.source_paths = sorted(make_dataset(source_root))
        self.source_transform = source_transform

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        fname, from_path = self.source_paths[index]
        from_im = Image.open(from_path).convert('RGB')
        if self.source_transform:
            tf=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            from_im = tf(copy.deepcopy(np.asarray(from_im)))
        return fname, from_im

class GrayscaleImagesDataset(Dataset):

    def __init__(self, source_root, source_transform=None):
        self.source_paths = sorted(make_dataset(source_root))
        self.source_transform = source_transform

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        fname, from_path = self.source_paths[index]
        from_im = Image.open(from_path).convert('L')
        if self.source_transform:
            from_im = self.source_transform(from_im)

        return fname, from_im

class DECADataset(Dataset):
    def __init__(self, source_root, source_transform=None,
                 batch_sequential_pairs=False, res=(512, 512),
                 pose_path=None):
        self.source_paths = sorted(make_dataset(source_root))
        self.source_paths = [self.source_paths[i:i+4] + [idx,] for idx, i in enumerate(range(0, len(self.source_paths), 4))]
        self.source_transform = source_transform

        self.shuffled = False
        self.contains_w = False
        self.res = res
        self.image_name = None

        if pose_path is not None:
            base = os.path.basename(source_root)
            # self.image_name = base[:5] + '.jpg'

            if 'american' in base:
                self.image_name = 'american_gothic' + '.jpg'
            else:
                self.image_name = base.split('_')[0] + '.jpg'

            if os.path.basename(pose_path).split(".")[1] == "json":
                f = open(pose_path)
                self.pose = np.asarray(json.load(f)[self.image_name]['pose']).astype(np.float32)
                f.close() 
            else:
                self.pose = np.load(pose_path).astype(np.float32)

            #with open(pose_path, 'r') as f:
                #self.pose = np.asarray(json.load(f)[self.image_name]['pose']).astype(np.float32)
        else:
            self.pose = None

        self.load_data()

    def load_data(self):
        # initialize buffers
        self.depth = []
        self.face_mask = []
        self.face_bg_mask = []
        self.face_img = []
        self.face_bg_img = []
        self.w = []
        self.w_img = []

        print('loading data')
        for fnames in self.source_paths:

            depth, mask, face_img, img = [Image.open(fname[1]).convert('RGB') for fname in fnames[:-1]]
            w_img = np.zeros((1, 1, 3))
            w = -1 * torch.ones(1)

            # make mask for face & for face + bg
            face_mask = ((np.array(mask)[...,  0]/255 > 0.5) * 255).astype(np.uint8)
            face_bg_mask = face_mask.copy()
            cv2.floodFill(face_bg_mask, None, (0, 0), 255)

            face_mask = face_mask.astype(np.float32)[..., None] / 255.
            face_bg_mask = face_bg_mask.astype(np.float32)[..., None] / 255.

            depth = torch.from_numpy(np.array(depth).astype(np.float32)/255.)
            face_img = self.source_transform(face_img)
            face_bg_img = self.source_transform(img * face_bg_mask.astype(np.uint8))

            face_mask = face_mask.transpose(2, 0, 1)
            face_bg_mask = face_bg_mask.transpose(2, 0, 1)

            # add to buffer
            self.depth.append(depth)
            self.face_mask.append(face_mask)
            self.face_bg_mask.append(face_bg_mask)
            self.face_img.append(face_img)
            self.face_bg_img.append(face_bg_img)
            self.w.append(w)
            self.w_img.append(w_img)


    def __len__(self):
        return len(self.source_paths)

    def load_w(self, logdir):
        w = np.load(os.path.join(logdir, 'w.npy'))
        w = torch.from_numpy(w)

        w_img_paths = sorted(make_dataset(os.path.join(logdir, 'w_output')))

        assert not self.shuffled, "don't shuffle before assigning w"

        self.w = []
        self.w_img = []
        for i in range(len(self.source_paths)):
            w_img = Image.open(w_img_paths[i][1]).convert('RGB')

            self.w.append(w[i, ...])

            w_img = self.source_transform(w_img)
            self.w_img.append(w_img)
            
        self.contains_w = True

    def __getitem__(self, idx):

        return {'idx': idx,
                'depth': self.depth[idx],
                'face_mask': self.face_mask[idx],
                'face_bg_mask': self.face_bg_mask[idx],
                'face_img': self.face_img[idx],
                'face_bg_img': self.face_bg_img[idx],
                'w': self.w[idx],
                'w_img': self.w_img[idx]}


class RandomPairSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return 2 * len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        idx = torch.randperm(n-1, generator=generator)
        idx = torch.stack((idx, idx+1), dim=-1).reshape(-1).tolist()
        
        yield from idx

    def __len__(self) -> int:
        return 2 * len(self.data_source)
