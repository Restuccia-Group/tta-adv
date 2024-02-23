import torch
import torch.nn as nn

class DIA:

    def __init__(self, cfg):
        self.cfg = cfg


    def generate_attack(self, sur_model, x, y):

        num_iter = self.cfg.DIA.NUM_ITER
        epsilon = self.cfg.DIA.EPSILON
        alpha = self.cfg.DIA.ALPHA
        mal_num = self.cfg.DIA.MAL_NUM
        fixed = torch.zeros_like(x.clone()[:-mal_num], requires_grad=False)
        adv = (torch.zeros_like(x.clone()[-mal_num:])- x[-mal_num:] + 127.5/255).requires_grad_(True)
        adv_pad = torch.cat((fixed,adv), 0)

        for t in range(num_iter):
            x_adv = x + adv_pad
            out = sur_model(x_adv)
            loss = nn.CrossEntropyLoss()(out[:-mal_num], y[:-mal_num])
            loss.backward()

            adv.data = (adv + alpha*adv.grad.detach().sign()).clamp(-epsilon,epsilon)
            adv.data = (adv.data +x[-mal_num:]).clamp(0,1)-(x[-mal_num:])
            adv_pad.data = torch.cat((fixed, adv), 0) 
            adv.grad.zero_()

        x_adv = x + adv_pad
        return x_adv
    
    



