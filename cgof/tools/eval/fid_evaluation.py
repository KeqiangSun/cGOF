"""
Contains code for logging approximate FID scores during training.
If you want to output ground-truth images from the training dataset, you can
run this file as a script.
"""

import os
import shutil
import torch
import copy
import argparse

from torchvision.utils import save_image
from pytorch_fid import fid_score
from tqdm import tqdm

from datasets import datasets
# import datasets
import curriculums

from IPython import embed


def output_real_images(dataloader, num_imgs, real_dir):
    img_counter = 0
    batch_size = dataloader.batch_size
    dataloader = iter(dataloader)
    # embed()
    for i in range(num_imgs//batch_size):
        # embed()
        data = next(dataloader)
        if len(data) == 2:
            real_imgs, _ = data
        elif len(data) == 3:
            real_imgs, param, _ = data
        elif len(data) == 4:
            real_imgs, param, xs_xe, _ = data

        for img in real_imgs:
            save_path = os.path.join(
                real_dir,
                f'{img_counter:0>5}.jpg'
            )
            # print(f"img: {img}")
            print(f"output_images save_path: {save_path}")
            save_image(
                img, save_path,
                normalize=True, range=(-1, 1)
            )
            img_counter += 1


def setup_evaluation(dataset_name=None, dataset_path=None, generated_dir=None,
                     target_size=128, num_imgs=8000, **kwargs):
    # Only make real images if they haven't been made yet
    real_dir = os.path.join(
        'EvalImages', dataset_name + '_real_images_' + str(target_size))
    if not os.path.exists(real_dir):
        os.makedirs(real_dir)
        dataloader, CHANNELS = datasets.get_dataset(
            dataset_name, img_size=target_size,
            dataset_path=dataset_path, **kwargs)
        print('outputting real images...')
        output_real_images(dataloader, num_imgs, real_dir)
        print('...done')

    if generated_dir is not None:
        os.makedirs(generated_dir, exist_ok=True)
    return real_dir


def output_images(generator, input_metadata, rank, world_size, output_dir,
                  face_recon=None, num_imgs=2048):  # 2048
    metadata = copy.deepcopy(input_metadata)
    metadata['img_size'] = 128
    metadata['batch_size'] = 4

    metadata['h_stddev'] = metadata.get('h_stddev_eval', metadata['h_stddev'])
    metadata['v_stddev'] = metadata.get('v_stddev_eval', metadata['v_stddev'])
    metadata['sample_dist'] = metadata.get('sample_dist_eval',
                                           metadata['sample_dist'])
    metadata['psi'] = 1

    img_counter = rank
    generator.eval()
    img_counter = rank

    zs = []
    imgs = []

    if rank == 0: pbar = tqdm("generating images", total=num_imgs)
    with torch.no_grad():
        while img_counter < num_imgs:
            z = torch.randn((metadata['batch_size'], generator.module.z_dim), device=generator.module.device)
            gen_data = generator.module.staged_forward(z, face_recon=face_recon, **metadata)
            generated_imgs = gen_data[0]

            zs.append(z)
            imgs.append(generated_imgs)

            for img in generated_imgs:
                save_path = os.path.join(output_dir, f'{img_counter:0>5}.jpg')
                print(f"output_images save_path: {save_path}")
                save_image(img, save_path, normalize=True, range=(-1, 1))
                img_counter += world_size
                if rank == 0: pbar.update(world_size)
    if rank == 0: pbar.close()

    zs = torch.cat(zs, dim=0)
    imgs = torch.cat(imgs, dim=0)

    return zs, imgs


def calculate_fid(dataset_name, generated_dir, target_size=256):
    real_dir = os.path.join('EvalImages', dataset_name + '_real_images_' + str(target_size))
    print('-------------------------------')
    print(f"real_dir: {real_dir}")
    print(f"generated_dir: {generated_dir}")

    fid = fid_score.calculate_fid_given_paths(
        [real_dir, generated_dir], 128, 'cuda', 2048)  # 2048
    torch.cuda.empty_cache()

    return fid


def setup_imgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ParamCelebA')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--num_imgs', type=int, default=100)

    opt = parser.parse_args()

    real_images_dir = setup_evaluation(
        opt.dataset, None, target_size=opt.img_size, num_imgs=opt.num_imgs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_dir', type=str, default='ParamCelebA')
    parser.add_argument('--gen_dir', type=str, default='pigan')
    parser.add_argument('--bs', type=int, default=128)

    opt = parser.parse_args()

    fid = fid_score.calculate_fid_given_paths(
        [opt.real_dir, opt.gen_dir], opt.bs, 'cuda', 2048)  # 2048
    torch.cuda.empty_cache()

    print(f"fid: {fid}")

if __name__ == '__main__':
    main()