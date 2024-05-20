from typing import Dict
import torch


def dict2device(data: Dict, device: torch.device):
    for key, _ in data.items():
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].to(device)

    return data
