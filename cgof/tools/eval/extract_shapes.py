import plyfile
import argparse
import torch
import numpy as np
import skimage.measure
import scipy
import mrcfile
import os
import mcubes
from plyfile import PlyData, PlyElement

import sys
sys.path.insert(0, '/home/kqsun/Tasks/pigan')
# import open3d as o3d


def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size


def sample_generator(generator, z, max_batch=100000, voxel_resolution=256, voxel_origin=[0,0,0], cube_length=2.0, psi=0.5):
    head = 0
    samples, voxel_origin, voxel_size = create_samples(voxel_resolution, voxel_origin, cube_length)
    samples = samples.to(z.device)
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)

    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
    transformed_ray_directions_expanded[..., -1] = -1

    print('getting truncated frequencies and phase shifts...')
    generator.generate_avg_frequencies()
    with torch.no_grad():
        raw_frequencies, raw_phase_shifts = generator.siren.mapping_network(z)
        truncated_frequencies = generator.avg_frequencies + psi * (raw_frequencies - generator.avg_frequencies)
        truncated_phase_shifts = generator.avg_phase_shifts + psi * (raw_phase_shifts - generator.avg_phase_shifts)

    print('getting coarse_output...')
    with torch.no_grad():
        while head < samples.shape[1]:
            print(f'head/samples.shape[1]:{head}/{samples.shape[1]}')
            coarse_output = generator.siren.forward_with_frequencies_phase_shifts(samples[:, head:head+max_batch], truncated_frequencies, truncated_phase_shifts, ray_directions=transformed_ray_directions_expanded[:, :samples.shape[1]-head]).reshape(samples.shape[0], -1, 4)
            sigmas[:, head:head+max_batch] = coarse_output[:, :, -1:]
            head += max_batch

    sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()

    return sigmas

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator_path', type=str)
    parser.add_argument('--seeds', nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--cube_size', type=float, default=0.3)
    parser.add_argument('--voxel_resolution', type=int, default=256)
    parser.add_argument('--output_dir', type=str, default='shapes')
    parser.add_argument('--sigma_threshold',
                        type=float,
                        default=20.0,
                        help='threshold to consider a location is occupied')
    opt = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # from IPython import embed
    # embed()
    generator = torch.load(opt.generator_path, map_location=torch.device(device))
    print('loading generator: Done.')
    ema = torch.load(opt.generator_path.split('generator')[0] + 'ema.pth')
    print('loading ema: Done.')
    ema.copy_to(generator.parameters())
    generator.set_device(device)
    generator.eval()


    torch.manual_seed(31)
    z = torch.randn(1, 80+64+80+80+27+80, device=device)

    for seed in opt.seeds:
        import time
        a = time.time()
        print(f'using seed: {seed}.')
        torch.manual_seed(seed)

        z_exp = torch.randn(1,64)
        z[:, 80:80+64] = z_exp

        print('generator sampling...')
        voxel_grid = sample_generator(
            generator, z, max_batch=500000, cube_length=opt.cube_size,
            voxel_resolution=opt.voxel_resolution)

        # Save Volume MRC File Here! @skq
        os.makedirs(opt.output_dir, exist_ok=True)
        with mrcfile.new_mmap(os.path.join(opt.output_dir, f'{seed}.mrc'), overwrite=True, shape=voxel_grid.shape, mrc_mode=2) as mrc:
            mrc.data[:] = voxel_grid  # 0.075s

        voxel_grid = np.maximum(voxel_grid, 0)
        vertices, triangles = mcubes.marching_cubes(
            voxel_grid, opt.sigma_threshold)  # 0.59s

        b = time.time()
        print(f'using: {b-a}s')

        # vertices_ = vertices.astype(np.float32)
        # # invert x and y coordinates (WHY? maybe because of the marching cubes algo)
        # x_ = vertices_[:, 1]
        # y_ = vertices_[:, 0]
        # vertices_[:, 0] = x_
        # vertices_[:, 1] = y_
        # vertices_[:, 2] = vertices_[:, 2]
        # vertices_.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

        # face = np.empty(len(triangles), dtype=[('vertex_indices', 'i4', (3, ))])
        # face['vertex_indices'] = triangles

        # filename = os.path.join(opt.output_dir, f'{seed}.ply')
        # PlyData([
        #     PlyElement.describe(vertices_[:, 0], 'vertex'),
        #     PlyElement.describe(face, 'face')
        # ]).write(filename)

        # Save Obj File Here! @skq
        mcubes.export_obj(vertices, triangles,
                          os.path.join(opt.output_dir, f'{seed}.obj')) # 2.49s

        # mesh = o3d.io.read_triangle_mesh(filename)
        # idxs, count, _ = mesh.cluster_connected_triangles()
        # max_cluster_idx = np.argmax(count)
        # triangles_to_remove = [
        #     i for i in range(len(face)) if idxs[i] != max_cluster_idx
        # ]
        # mesh.remove_triangles_by_index(triangles_to_remove)
        # mesh.remove_unreferenced_vertices()
        # print(
        #     f'Mesh has {len(mesh.vertices)/1e6:.2f} M vertices and {len(mesh.triangles)/1e6:.2f} M faces.'
        # )

        # vertices_ = np.asarray(mesh.vertices).astype(np.float32)
        # triangles = np.asarray(mesh.triangles)

        # face = np.empty(len(triangles),
        #                 dtype=[('vertex_indices', 'i4', (3, ))])
        # face['vertex_indices'] = triangles

        # filename = os.path.join(opt.output_dir, f'{seed}_wo_noise.ply')
        # PlyData([
        #     PlyElement.describe(vertices_[:, 0], 'vertex'),
        #     PlyElement.describe(face, 'face')
        # ]).write(filename)
