import torch
import torch.nn as nn
from tta_algo.tta_base import TTA_BASE
from utils.memory import HUS
from utils.bn_layers import *

class SoTTA(TTA_BASE):
    def __init__(self, cfg, model):
        self.cfg = cfg
        super(SoTTA, self).__init__(cfg, model)
        self.mem = HUS( capacity = cfg.TTA.SoTTA.MEM_SIZE,
                       num_class = cfg.DATA.NUM_CLASSES,
                       threshold= cfg.TTA.SoTTA.THRESH
                       )
        self.mem_state = self.mem.save_state_dict() # TODO: Put it inside the data loop

    @torch.enable_grad    
    def forward_and_adapt(self, data_batch, model, optimizer):
        prev_mem_state = self.mem.save_state_dict()
        self.construct_memory(data_batch, model)
        mem_state = self.mem.save_state_dict()
        feats, _ , _ = self.mem.get_memory()
        if len(feats) == 0:
            print("No data is available in memory")
        filtered_x = torch.stack(feats)
        outputs = model(self.normalize(filtered_x))
        loss = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.first_step(zero_grad = True)

        outputs = model(self.normalize(filtered_x))
        second_loss = (
            -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1).mean()
        )
        second_loss.backward()
        optimizer.second_step(zero_grad=True)

        with torch.no_grad():
            outputs = model(self.normalize(data_batch))
        return outputs
        
    def construct_memory(self, data_batch, model):
        domain = 0 # domain information is not relevant here
        training = model.training
        with torch.no_grad():
            model.eval()
            logits = model(self.normalize(data_batch))
            pseudo_cls = logits.max(dim=1, keepdim=False)[1].detach().cpu()
            pseudo_conf = (
                    nn.functional.softmax(logits, dim=1).max(1, keepdim=False)[0].detach().cpu()
                    )
            
            for i,x in enumerate(data_batch):
                self.mem.add_instance([x, pseudo_cls[i], domain, pseudo_conf[i]])
        if training:
            model.train()

    def configure_model(self, model, device):
        
        for param in model.parameters():
            param.requires_grad = False
        
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module,nn.LayerNorm):
                module.track_running_stats = False
                # module.running_mean = None
                # module.running_var = None
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)
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



