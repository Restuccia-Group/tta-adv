import torch
import numpy as np
import torch.nn as nn
from tta_algo.tta_base import TTA_BASE
from utils.loss_functions import softmax_entropy
from utils.bn_layers import *


class SAR(TTA_BASE):
    def __init__(self, cfg, model):
        super(SAR, self).__init__(cfg,model)
        self.cfg = cfg
        self.ema = None
    
    def forward_and_adapt(self, data_batch, model, optimizer):
        self.optimizer.zero_grad()
        outputs = model(self.normalize(data_batch))

        entropys = softmax_entropy(outputs)
        filter_ids_1 = torch.where(entropys < self.cfg.TTA.SAR.MARGIN_E0)
        entropys = entropys[filter_ids_1]
        loss = entropys.mean(0)
        loss.backward()

        optimizer.first_step(zero_grad = True)
        entropys2 = softmax_entropy(model(self.normalize(data_batch)))
        entropys2 = entropys2[filter_ids_1]
        #loss_second_value = entropys2.clone().detach().mean(0)
        filter_ids_2 = torch.where(entropys2 < self.cfg.TTA.SAR.MARGIN_E0)
        loss_second = entropys2[filter_ids_2].mean(0)
        if not np.isnan(loss_second.item()):
            self.ema = update_ema(self.ema, loss_second.item())
        
        loss_second.backward()
        optimizer.second_step(zero_grad = True)

        with torch.no_grad():
            outputs = model(self.normalize(data_batch))

        return outputs
    
    def configure_model(self, model, device):
        model = model.train()
        model.requires_grad_(False)

        for m in model.modules():
            if isinstance(m , nn.BatchNorm2d) or isinstance(m,nn.LayerNorm):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
        model = model.to(device)
        return model
    
    def convert_bn(self):
        normlayer_names = []
        for name, sub_module in self.model.named_modules():
                if isinstance(sub_module, nn.BatchNorm2d):
                    normlayer_names.append(name)

        for name in normlayer_names:
            bn_layer = self.get_named_submodule(self.model, name)

            if isinstance(bn_layer, nn.BatchNorm2d):
                # norm_layer = BatchNorm(num_features=bn_layer.num_features,
                #                      affine=True, track_running_stats=False,
                #                      use_tracked_mean=False,
                #                      use_tracked_var=False)
                norm_layer = MedBN2d(bn_layer, momentum = 1)
                    
            else:
                raise RuntimeError()

                # momentum_bn = NewBN(num_features=bn_layer.num_features,
                #                     momentum=bn_layer.momentum,
                #                     )
                # norm_layer.weight.requires_grad_(True)
                # norm_layer.bias.requires_grad_(True)
                # norm_layer = NewBN(bn_layer, momentum = 1)
            norm_layer.weight.requires_grad_(True)
            norm_layer.bias.requires_grad_(True)
                #norm_layer.training = True
            self.set_named_submodule(self.model, name, norm_layer)
        params, param_names = self.collect_params(self.model)
        if len(param_names) != 0:
            self.optimizer = self.setup_optimizer(params,self.cfg)

    def revert_bn(self):
        normlayer_names = []
        for name, sub_module in self.model.named_modules():
                if isinstance(sub_module,MedBN2d):
                    normlayer_names.append(name)

        for name in normlayer_names:
            bn_layer = self.get_named_submodule(self.model, name)

            if isinstance(bn_layer, MedBN2d):
                # norm_layer = BatchNorm(num_features=bn_layer.num_features,
                #                      affine=True, track_running_stats=False,
                #                      use_tracked_mean=False,
                #                      use_tracked_var=False)
                norm_layer = nn.BatchNorm2d(num_features=bn_layer.num_features)
                    
            else:
                raise RuntimeError()

                # momentum_bn = NewBN(num_features=bn_layer.num_features,
                #                     momentum=bn_layer.momentum,
                #                     )
                # norm_layer.weight.requires_grad_(True)
                # norm_layer.bias.requires_grad_(True)
                # norm_layer = NewBN(bn_layer, momentum = 1)
            norm_layer.weight.requires_grad_(True)
            norm_layer.bias.requires_grad_(True)
                #norm_layer.training = True
            self.set_named_submodule(self.model, name, norm_layer)
        params, param_names = self.collect_params(self.model)
        if len(param_names) != 0:
            self.optimizer = self.setup_optimizer(params,self.cfg)
    
    @staticmethod
    def get_named_submodule(model, sub_name):
        names = sub_name.split(".")
        module = model
        for name in names:
            module = getattr(module, name)

        return module

    
    @staticmethod
    def set_named_submodule(model, sub_name, value):
        names = sub_name.split(".")
        module = model
        for i in range(len(names)):
            if i != len(names) - 1:
                module = getattr(module, names[i])

            else:
                setattr(module, names[i], value)


def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data