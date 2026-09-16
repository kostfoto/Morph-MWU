import random
import os

import numpy as np
import torch

def generate_seed():
    random_data = os.urandom(4)
    seed = int.from_bytes(random_data, byteorder="big")
    return seed

def set_seed(seed: int):
    """Seed Python, NumPy and PyTorch (CPU + all CUDA devices)."""
    # ---- Python RNG ----
    random.seed(seed)

    # ---- NumPy RNG ----
    np.random.seed(seed)

    # ---- PyTorch CPU/GPU RNG ----
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    """DataLoader worker init fn so each worker is deterministically seeded."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)