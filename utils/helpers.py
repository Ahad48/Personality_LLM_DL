import numpy as np
import torch
import transformers


def find_backend():
    # if you want to default to cuda first change order.
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print("Currently using: ", device)
    return device