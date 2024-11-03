import sys
# sys.path.append("../../")
sys.path.insert(0, '/mnt/afs/kqsun/Tasks/eg3d_022/eg3d/eg3d')

import pickle
import functools
import torch
from configs import paths_config, global_config

import legacy
import dnnlib
# import training


def toogle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def load_tuned_G(run_id, type, full_path=None):
    if full_path is None:
        new_G_path = f'{paths_config.checkpoints_dir}/model_{run_id}_{type}.pt'
    else:
        new_G_path = full_path

    with open(new_G_path, 'rb') as f:
        new_G = torch.load(f).to(global_config.device).eval()
    new_G = new_G.float()
    toogle_grad(new_G, False)
    return new_G


def load_3dgan(reload_modules=False):
    with dnnlib.util.open_url(paths_config.eg3d_ffhq) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].to(global_config.device).eval()
        G = G.float()
    if reload_modules:
        print("Reloading Modules!")
        # from IPython import embed; embed(header='reload_modules')
        from training.triplane import TriPlaneGenerator
        from torch_utils import misc
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(global_config.device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new
    return G


def load_stylegan2d():
    with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
        old_G = pickle.load(f)['G_ema'].to(global_config.device).eval()
        old_G = old_G.float()
    return old_G
