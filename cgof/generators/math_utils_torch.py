"""
Utilities for geometry etc.
"""

import torch
from trimesh import available_formats


def transform_vectors(matrix: torch.Tensor, vectors4: torch.Tensor) -> torch.Tensor:
    """
    Left-multiplies MxM @ NxM. Returns NxM.
    """
    res = torch.matmul(vectors4, matrix.T)
    return res


def normalize_vecs(vectors: torch.Tensor, norm_mode='vec_len') -> torch.Tensor:
    """
    Normalize vector lengths.
    """

    available_modes = ['vec_len', 'z_val']
    if norm_mode not in available_modes:
        raise ValueError(f'norm_mode should be in {available_modes}')

    if norm_mode == 'vec_len':
        ret = vectors / (torch.norm(vectors, dim=-1, keepdim=True))
    elif norm_mode == 'z_val':
        ret = vectors / torch.abs(vectors[..., -1:])
    return ret

def torch_dot(x: torch.Tensor, y: torch.Tensor):
    """
    Dot product of two tensors.
    """
    return (x * y).sum(-1)
