from __future__ import print_function, absolute_import
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# Data params
data_mean = 4
data_stddev = 1.25

# Model params

g_input_size = 1
g_hidden_size = 50
g_output_size = 1

d_input_size = 100
d_hidden_size = 50
d_output_size = 1

def get_distribution_sampler(mu, sigma):
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))


def get_generator_input_sampler():
    return lambda m, n: torch.rand(m, n)

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forword(self, x):
        x = F.elu(self.map1(x))
        x = F.sigmoid(self.map2(x))
        return self.map3(x)

class Discriminator(nn.Module):
