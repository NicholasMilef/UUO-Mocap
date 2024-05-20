import random
import numpy as np
import torch

def set_random_seed(seed):
    torch.backends.cuda.benchmark = False
    torch.backends.cuda.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
