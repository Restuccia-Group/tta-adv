import torch

def softmax_entropy(x):
    softmax = x.softmax(1)
    return -(softmax * torch.log(softmax + 1e-6)).sum(1)

def softmax_entropy_rotta(x, x_ema):
    return - (x_ema.softmax(1) * x.log_softmax(1)).sum(1)