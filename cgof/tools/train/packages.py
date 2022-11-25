import os
import sys
# add_path = os.path.realpath('Deep3DFaceRecon_pytorch')
# # print(add_path)
# sys.path.insert(0, add_path)

from losses.contrastive_id_loss import (
    Z_Manager, sample_yaw_pitch, ContrastiveIDLoss,
    ContrastiveGeoLoss, ContrastiveTexLoss, ContrastiveFaceLoss,
    ImitativeTexLoss, ContrastiveTexGramLoss
)

from util.preprocess import estimate_norm_torch
from options.test_options import TestOptions
from models import create_model
from util.visualizer import MyVisualizer
from util.preprocess import align_img
from util.load_mats import load_lm3d
from data.flist_dataset import default_flist_reader
from scipy.io import loadmat, savemat
# from test_msra19 import resize_n_crop_tensor, read_img, read_pigan
from test_grad import resize_n_crop_tensor, read_img, read_pigan
from deep3dfacerecon_opt import deep3dfacerecon_opt

from IPython import embed

import argparse
import numpy as np
import math
import logging

from pprint import pprint
from collections import deque

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import L1Loss as l1loss
from torch.nn.modules.loss import _WeightedLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

from generators import generators
from generators.volumetric_rendering import get_volume_mask
from discriminators import discriminators
from siren import siren
from tools.eval import fid_evaluation

from datasets import datasets
import curriculums
from tqdm import tqdm
from datetime import datetime
import copy
from easydict import EasyDict as edict
from torch_ema import ExponentialMovingAverage
import warnings
# from test_bfm_render import Renderer

from utils.utils import *
# from ldmk_model import get_ldmk_model
# from models.model_106.MobileNet_v29 import MobileNet
from IPython import embed
from losses.wing_loss import WingLoss
from losses.laplacian_loss import LaplacianLoss, EdgeAwareDepthSmoothLoss
from losses.rel_depth_consistency_loss import RelDConsistencyLoss
# from spade_networks.discriminator import MultiscaleDiscriminator
from losses.spade.loss import GANLoss, VGGLoss

from easydict import EasyDict
# from model.PerCostFormer.warpformer import WarpFormer
import kornia
