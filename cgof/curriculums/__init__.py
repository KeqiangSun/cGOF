"""
To easily reproduce experiments, and avoid passing several command line
arguments, we implemented a curriculum utility.

Parameters can be set in a curriculum dictionary.

Curriculum Schema:

    Numerical keys in the curriculum specify an upsample step. When the current step matches the upsample step,
    the values in the corresponding dict be updated in the curriculum. Common curriculum values specified at upsamples:
        batch_size: Batch Size.
        num_steps: Number of samples along ray.
        img_size: Generated image resolution.
        batch_split: Integer number over which to divide batches and aggregate sequentially. (Used due to memory constraints)
        gen_lr: Generator learnig rate.
        disc_lr: Discriminator learning rate.

    fov: Camera field of view
    ray_start: Near clipping for camera rays.
    ray_end: Far clipping for camera rays.
    fade_steps: Number of steps to fade in new layer on discriminator after upsample.
    h_stddev: Stddev of camera yaw in radians.
    v_stddev: Stddev of camera pitch in radians.
    h_mean:  Mean of camera yaw in radians.
    v_mean: Mean of camera yaw in radians.
    sample_dist: Type of camera pose distribution. (gaussian | spherical_uniform | uniform)
    topk_interval: Interval over which to fade the top k ratio.
    topk_v: Minimum fraction of a batch to keep during top k training.
    betas: Beta parameters for Adam.
    unique_lr: Whether to use reduced LRs for mapping network.
    weight_decay: Weight decay parameter.
    r1_lambda: R1 regularization parameter.
    latent_dim: Latent dim for Siren network  in generator.
    grad_clip: Grad clipping parameter.
    model: Siren architecture used in generator. (SPATIALSIRENBASELINE | TALLSIREN)
    generator: Generator class. (ImplicitGenerator3d)
    discriminator: Discriminator class. (ProgressiveEncoderDiscriminator | ProgressiveDiscriminator)
    dataset: Training dataset. (CelebA | Carla | Cats)
    clamp_mode: Clamping function for Siren density output. (relu | softplus)
    z_dist: Latent vector distributiion. (gaussian | uniform)
    hierarchical_sample: Flag to enable hierarchical_sampling from NeRF algorithm. (Doubles the number of sampled points)
    z_labmda: Weight for experimental latent code positional consistency loss.
    pos_lambda: Weight parameter for experimental positional consistency loss.
    last_back: Flag to fill in background color with last sampled color on ray.
"""

import math
from re import A
from easydict import EasyDict

import glob
import os
import importlib.util

# spec.loader.exec_module(foo)
# foo.MyClass()

# spec = importlib.util.spec_from_file_location(
#     os.path.splitext(os.path.basename(f))[0], f)
# foo = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(foo)


def get(name):
    spec = importlib.util.spec_from_file_location(
        name, f"./curriculums/{name}.py")
    # os.path.splitext(os.path.basename(name))[0], name)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return getattr(foo, name)


config_ldmk = EasyDict(
    {
        "ckpt_path": "./model/model_106/ldmk_detector/exp_mobile_v17-ceph-long/ckpt_best_model.pth.tar",
        "model": {
            "conv_channels": [1, 10, 16, 24, 32, 64, 64, 96, 128, 128],
            "ip_channels": [128, 128],
            "init_type": 'init'
        },
        "num_landmarks": 106,
        "crop_size": 112
    }
)


def next_upsample_step(curriculum, current_step):
    # Return the epoch when it will next upsample
    current_metadata = extract_metadata(curriculum, current_step)
    current_size = current_metadata['img_size']
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        if curriculum_step > current_step and curriculum[curriculum_step].get('img_size', 512) > current_size:
            return curriculum_step
    return float('Inf')


def last_upsample_step(curriculum, current_step):
    # Returns the start epoch of the current stage,
    # i.e. the epoch it last upsampled
    current_metadata = extract_metadata(curriculum, current_step)
    current_size = current_metadata['img_size']
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        if curriculum_step <= current_step and curriculum[curriculum_step]['img_size'] == current_size:
            return curriculum_step
    return 0


def get_current_step(curriculum, epoch):
    step = 0
    for update_epoch in curriculum['update_epochs']:
        if epoch >= update_epoch:
            step += 1
    return step


def extract_metadata(curriculum, current_step):
    return_dict = {}
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int], reverse=True):
        if curriculum_step <= current_step:
            for key, value in curriculum[curriculum_step].items():
                return_dict[key] = value
            break
    for key in [k for k in curriculum.keys() if type(k) != int]:
        return_dict[key] = curriculum[key]
    return return_dict


def check_diff(a, b):
    for key in a.keys():
        val = a[key]
        if key in b.keys():
            if a[key] != b[key]:
                if isinstance(a[key], dict) and isinstance(b[key], dict):
                    check_diff(a[key], b[key])
                else:
                    print(f"{key} changed from {a[key]} to {b[key]}.")
        else:
            print(f"{key} not found.")


if __name__ == "__main__":
    check_diff(get("recon4_minv1e1_reld1e1"), get("recon4_minv1e1_reld1e2"))
    check_diff(get("recon4_minv1e1_reld1e2"), get("recon4_minv1e1_reld1e1"))
    # check_diff(
    #   recon4_norm_glbmean_minv1e1_reld1e2_newctex5,
    #   recon4_minv1e1_reld1e2_cgeo3e2mg1e_2)

    # check_diff(CelebA_pigan_spade_tddfa_gan, CelebA_pigan_spade_tddfa10)
