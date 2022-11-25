"""Datasets"""

import os
from easydict import EasyDict as edict
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision
import glob
import PIL
import random
import math
import pickle
import numpy as np


class CelebA(Dataset):
    """CelelebA Dataset"""

    def __init__(self, dataset_path, img_size, **kwargs):
        super().__init__()

        self.data = glob.glob(dataset_path)
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
                    [
                        transforms.Resize(320),
                        transforms.CenterCrop(256),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5], [0.5]), 
                        transforms.RandomHorizontalFlip(p=0.5), 
                        transforms.Resize((img_size, img_size), interpolation=0)
                    ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)

        return X, 0


class CropCelebA(Dataset):
    """CelelebA Dataset"""

    def __init__(self, dataset_path, img_size, **kwargs):
        super().__init__()

        self.data = glob.glob(dataset_path)
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((img_size, img_size), interpolation=0)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)

        return X, 0


class MaskCelebA(Dataset):
    """CelelebA Dataset"""

    def __init__(self, depth_path, img_size, **kwargs):
        super().__init__()

        self.depth = glob.glob(depth_path)
        assert len(self.depth) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform_img = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.RandomHorizontalFlip(p=0.5), transforms.Resize((img_size, img_size), interpolation=0)])
        self.transform_depth = transforms.Compose(
                    [transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0.5), transforms.Resize((img_size, img_size), interpolation=0)])

    def __len__(self):
        return len(self.depth)

    def __getitem__(self, index):
        depth_path = self.depth[index]
        depth = np.load(depth_path)['depth']
        img_path = depth_path.replace('depth_','img_').replace('_depth.npz','.jpg')
        print(depth_path)
        print(img_path)
        img = PIL.Image.open(img_path)
        
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform_img(img)

        random.seed(seed)
        torch.manual_seed(seed)
        depth = self.transform_depth(depth)
        
        return img, depth, 0


class ParamCelebA(Dataset):
    """CelelebA Dataset"""

    def __init__(self, param_path, img_size, **kwargs):
        super().__init__()

        self.param = glob.glob(param_path)
        assert len(self.param) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform_img = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.Resize((img_size, img_size), interpolation=0)])

    def __len__(self):
        return len(self.param)

    def __getitem__(self, index):
        
        param_path = self.param[index]
        img_path = param_path.replace('param_','img_').replace('_params.npz','.jpg')
        # print(param_path)
        # print(img_path)
        
        img = PIL.Image.open(img_path)
        img = self.transform_img(img)
        
        param = dict(np.load(param_path))
        # xs, xe = param['xs'], param['xe']
        # yaw, pitch = param['yaw'], param['pitch']
        # R, RT, t3d = param['R'], param['RT'], param['t3d']
        for key in param.keys():
            param[key] = torch.from_numpy(param[key])
            if len(param[key].shape) == 0:
                param[key] = param[key].unsqueeze(0)
        param = edict(param)
        
        return img, param, 0


class TddfaCelebA(Dataset):
    """CelelebA Dataset"""

    def __init__(self, param_path, img_size, **kwargs):
        super().__init__()
        
        mean_std = pickle.load(open('models/bfm_models/param_mean_std_62d_120x120.pkl','rb'))
        mean = mean_std['mean']
        std = mean_std['std']
        self.xs_xe_mean = mean[12:]
        self.xs_xe_std = std[12:]
        
        self.param = glob.glob(param_path)
        assert len(self.param) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform_img = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.Resize((img_size, img_size), interpolation=0)])

    def __len__(self):
        return len(self.param)

    def __getitem__(self, index):
        
        param_path = self.param[index]
        img_path = param_path.replace('param_','img_').replace('_params.npz','.jpg')
        tddfa_path = param_path.replace('param_','tddfa_').replace('_params.npz','_2d_sparse.npz')
        
        img = PIL.Image.open(img_path)
        img = self.transform_img(img)
        
        param = dict(np.load(param_path))
        for key in param.keys():
            param[key] = torch.from_numpy(param[key])
            if len(param[key].shape) == 0:
                param[key] = param[key].unsqueeze(0)
        # param = edict(param)
        
        tddfa = {}
        np_tddfa = dict(np.load(tddfa_path))
        xs = np_tddfa['alpha_shp'].flatten()
        xe = np_tddfa['alpha_exp'].flatten()
        xs_xe = np.concatenate([xs,xe],axis=0)
        xs_xe = (xs_xe-self.xs_xe_mean)/self.xs_xe_std
        xs_xe = torch.from_numpy(xs_xe)
        
        # tddfa = edict(tddfa)
        
        return img, param, xs_xe, 0


class TddfaCelebAHD(Dataset):
    """CelelebA Dataset"""

    def __init__(self, param_path, img_size, hd_img_size, **kwargs):
        super().__init__()
        
        mean_std = pickle.load(open('models/bfm_models/param_mean_std_62d_120x120.pkl','rb'))
        mean = mean_std['mean']
        std = mean_std['std']
        self.xs_xe_mean = mean[12:]
        self.xs_xe_std = std[12:]
        
        self.param = glob.glob(param_path)
        assert len(self.param) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform_img = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.Resize((img_size, img_size), interpolation=0)])
        self.transform_img_hd = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.Resize((img_size, img_size), interpolation=0)])

    def __len__(self):
        return len(self.param)

    def __getitem__(self, index):
        
        param_path = self.param[index]
        img_path = param_path.replace('param_','img_').replace('_params.npz','.jpg')
        tddfa_path = param_path.replace('param_','tddfa_').replace('_params.npz','_2d_sparse.npz')
        
        img = PIL.Image.open(img_path)
        img_hd = self.transform_img_hd(img)
        img = self.transform_img(img)
        
        param = dict(np.load(param_path))
        for key in param.keys():
            param[key] = torch.from_numpy(param[key])
            if len(param[key].shape) == 0:
                param[key] = param[key].unsqueeze(0)
        # param = edict(param)
        
        tddfa = {}
        np_tddfa = dict(np.load(tddfa_path))
        xs = np_tddfa['alpha_shp'].flatten()
        xe = np_tddfa['alpha_exp'].flatten()
        xs_xe = np.concatenate([xs,xe],axis=0)
        xs_xe = (xs_xe-self.xs_xe_mean)/self.xs_xe_std
        xs_xe = torch.from_numpy(xs_xe)
        
        # tddfa = edict(tddfa)
        
        return img, param, xs_xe, img_hd, 0


class Cats(Dataset):
    """Cats Dataset"""

    def __init__(self, dataset_path, img_size, **kwargs):
        super().__init__()
        
        self.data = glob.glob(dataset_path)
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
                    [transforms.Resize((img_size, img_size), interpolation=0), transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.RandomHorizontalFlip(p=0.5)])
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)
        
        return X, 0


class Carla(Dataset):
    """Carla Dataset"""

    def __init__(self, dataset_path, img_size, **kwargs):
        super().__init__()
        
        self.data = glob.glob(dataset_path)
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
                    [transforms.Resize((img_size, img_size), interpolation=0), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)
        
        return X, 0


def get_dataset(name, subsample=None, batch_size=1, **kwargs):
    dataset = globals()[name](**kwargs)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=8
    )
    return dataloader, 3


def get_dataset_distributed(name, world_size, rank, batch_size, **kwargs):
    dataset = globals()[name](**kwargs)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=12,
        # num_workers=36,
    )

    return dataloader, 3


if __name__ == "__main__":
    from torchvision.utils import save_image, make_grid
    dataset = MaskCelebA('/media/SSD/kqsun/depth_crop_celeba_sample/*.npz', 64)
    for i in range(len(dataset)):
        img, depth, _ = dataset[i]
        img = img.mul(255).add_(0.5).clamp_(0, 255).permute(1,2,0).to('cpu', torch.uint8).numpy()#.permute(1, 2, 0)
        depth = make_grid(depth, nrow=1, normalize=True).mul(255).clamp_(0,255).permute(1, 2, 0).numpy()
        print(img.shape)
        print(depth.shape)
        concat = np.concatenate([img,depth],axis = 1).astype(np.uint8)
        print(concat.shape)
        PIL.Image.fromarray(concat).save(f'debug_img_{i}.png')
