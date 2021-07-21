import torch


def get_z(heat, clean, nb=3):
    B, C, H, W = clean.size()
    factor = 2 ** nb
    H, W = H // factor, W // factor
    C = C * factor * factor
    size = (B, C, H, W)
    z = torch.normal(mean=0, std=heat, size=size) if heat > 0 else torch.zeros(size)
    return z
