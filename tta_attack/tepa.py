import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import torchvision.transforms as transforms

def input_diversity(x):
    resize_rate = 0.8
    diversity_prob = 0.5
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    if resize_rate < 1:
        img_size = img_resize
        img_resize = x.shape[-1]
    
    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype = torch.int32)
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left

    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

    return padded if torch.rand(1) < diversity_prob else x

def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def DIM(image, surrogate_model):
    image = image.cuda()
    eps = 32/255
    alpha = 1/255
    mu = 1.0
    g = 0 
    decay = 1.0

    momentum = torch.zeros_like(image).detach().cuda()

    ori_image = image.data
    ori_image = ori_image.cuda()

    num_restart


    