import torch
import numpy as np
import mindspore as ms
from mindspore import Tensor

def compute_diff_for_torch(x_ms, x_torch):
    x_ms = x_ms.detach().cpu().numpy()
    x_torch = x_torch.detach().cpu().numpy()
    abs_diff = np.abs(x_ms - x_torch)
    eps = 1e-8
    rel_diff = abs_diff / (np.abs(x_torch) + eps)
    rel_diff2 = (abs_diff / (np.abs(x_torch) + np.abs(x_torch).mean()))
    print(f"abs diff {abs_diff.mean()}, relative diff1 {rel_diff}, relative diff2 {rel_diff2}")