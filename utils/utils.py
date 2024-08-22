import os
import torch
from torch.autograd import Variable


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
