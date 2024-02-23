import torch
import torch.nn as nn
from tta_base import TTA_BASE
from utils.bn_layers import RobustBN2d
from copy import deepcopy
from utils.memory import CSTU
from utils.custom_transforms import get_tta_transforms
from utils.loss_functions import softmax_entropy, softmax_entropy_rotta

class RoTTA(TTA_BASE):
    def __init__(self, cfg, model):
        super(RoTTA,self).__init__(cfg, model)
        self.mem = CSTU(capacity=self.cfg.TTA.RoTTA.MEMORY_SIZE, 
                        num_class=cfg.DATA.NUM_CLASS, 
                        lambda_t=cfg.TTA.RoTTA.LAMBDA_T, 
                        lambda_u=cfg.TTA.RoTTA.LAMBDA_U)
        self.model_ema = self.build_ema(self.model_ema)
        self.transform = get_tta_transforms(cfg)
        self.nu = cfg.TTA.RoTTA.NU
        self.update_frequency = cfg.TTA.RoTTA.UPDATE_FREQUENCY  # actually the same as the size of memory bank
        self.current_instance = 0
    
    
    @torch.enable_grad()
    def forward_and_adapt(self, data_batch, model, optimizer):
        with torch.no_grad():
            model.eval()
            self.model_ema.eval()
            ema_out = self.model_ema(data_batch)
            predict = torch.softmax(ema_out, dim=1)
            pseudo_label = torch.argmax(predict, dim=1)
            entropy = softmax_entropy(predict)

    #add into CSTU memory bank
        for i, data in enumerate(data_batch):
            p_l = pseudo_label[i].item()
            uncertainity = entropy[i].item()
            current_instance = (data, p_l, uncertainity)
            self.mem.add_instance(current_instance)
            self.current_instance += 1

            if self.current_instance % self.update_frequency == 0:
                self.update_model(model, optimizer)
        return ema_out
    

    def update_model(self, model, optimizer):
        model.train()
        self.model_ema.train()
        data, ages = self.mem.get_memory()
        loss = None
        data = torch.stack(data)
        strong_aug_data = self.transform(data)
        ema_out = self.model_ema(data)
        stu_out = model(strong_aug_data)
        isinstance_weight = self.timeliness_reweighting(ages)
        loss = (softmax_entropy_rotta(ema_out, stu_out)*isinstance_weight).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.update_ema_variables(self.model_ema, self.model, self.nu)



    def configure_model(self, model):

        model.requires_grad_(False)
        normlayer_names = []

        for name, sub_module in model.named_modules():
            if isinstance(sub_module, nn.BatchNorm2d):
                normlayer_names.append(name)

        for name in normlayer_names:
            bn_layer = self.get_named_submodule(model, name)

            if isinstance(bn_layer, nn.BatchNorm2d):
                NewBN = RobustBN2d
            else:
                raise RuntimeError()

            momentum_bn = NewBN(bn_layer,
                                self.cfg.ADAPTER.RoTTA.ALPHA)
            momentum_bn.requires_grad_(True)
            self.set_named_submodule(model, name, momentum_bn)
        return model
    

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

    @staticmethod
    def build_ema(model):
        ema_model = deepcopy(model)
        for param in ema_model.parameters():
            param.detach_()
        return ema_model
    

    @staticmethod
    def timeliness_reweighting(ages):
        if isinstance(ages, list):
            ages = torch.tensor(ages).float().cuda()
        return torch.exp(-ages) / (1 + torch.exp(-ages))


    @staticmethod
    def update_ema_variables(ema_model, model, nu):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1 - nu) * ema_param[:].data[:] + nu * param[:].data[:]
        return ema_model
