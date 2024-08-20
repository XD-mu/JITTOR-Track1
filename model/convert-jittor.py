from jittor.utils.pytorch_converter import convert
import os

current_file_name = "utils.py"

new_file_name = f"{os.path.splitext(current_file_name)[0]}-jittor.py"

pytorch_code="""
from collections import OrderedDict

import torch
from torch import Tensor


def compute_prototypes(support_features: Tensor, support_labels: Tensor) -> Tensor:
    n_way = len(torch.unique(support_labels))
    return torch.cat(
        [
            support_features[torch.nonzero(support_labels == label)].mean(0)
            for label in range(n_way)
        ]
    )

def entropy(logits: Tensor) -> Tensor:
    probabilities = logits.softmax(dim=1)
    return (-(probabilities * (probabilities + 1e-12).log()).sum(dim=1)).mean()


def k_nearest_neighbours(features: Tensor, k: int, p_norm: int = 2) -> Tensor:
    distances = torch.cdist(features, features, p_norm)

    return distances.topk(k, largest=False).indices[:, 1:]


def power_transform(features: Tensor, power_factor: float) -> Tensor:
    return (features.relu() + 1e-6).pow(power_factor)


def strip_prefix(state_dict: OrderedDict, prefix: str):
    return OrderedDict(
        [
            (k[len(prefix) :] if k.startswith(prefix) else k, v)
            for k, v in state_dict.items()
        ]
    )
"""
jittor_code = convert(pytorch_code)
print(jittor_code)
with open(new_file_name, 'w') as file:
    file.write(jittor_code)
