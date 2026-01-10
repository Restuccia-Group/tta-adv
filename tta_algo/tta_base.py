import torch
import torch.nn as nn
import torch.optim as optim
from utils.sam_optimizer import SAM
from utils.util import Normalize


class TTA_BASE(nn.Module):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        device = torch.device("cuda:{:d}".format(cfg.BASE.GPU_ID) if torch.cuda.is_available() else "cpu")
        self.model = self.configure_model(model,device)
        params, param_names = self.collect_params(self.model)
        if len(param_names) != 0:
            self.optimizer = self.setup_optimizer(params,cfg)
        self.steps = self.cfg.OPTIM.STEPS
        self.normalize = Normalize(device, cfg.CORRUPTION.DATASET)

    def forward(self, x):
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)
        return outputs
    
    def forward_and_adapt(self, *args):
        raise NotImplementedError("implement forward_and_adapt by yourself!")
    
    
    def configure_model(self, model):
        raise NotImplementedError("implement forward_and_adapt by yourself!")
    

    @staticmethod
    def collect_params(model):
        names = []
        params = []

        for n, p in model.named_parameters():
            if p.requires_grad:
                names.append(n)
                params.append(p)

        return params, names
    
    @staticmethod
    def setup_optimizer(params,cfg):

        lr_adapt = cfg.OPTIM.LR 
        
        if cfg.OPTIM.METHOD == 'Adam':
            if cfg.TTA.NAME in ['sar', 'sotta']:
                return SAM(params,
                           optim.Adam,
                           rho = 0.05,
                           lr = lr_adapt,
                           weight_decay = cfg.OPTIM.WD
                           )
            else:
                return optim.Adam(params,
                        lr=lr_adapt,
                        betas=(cfg.OPTIM.BETA, 0.999),
                        weight_decay=cfg.OPTIM.WD)
            
        elif cfg.OPTIM.METHOD == 'SGD':
            if cfg.TTA.NAME in ['sar', 'sotta']:
                return SAM(params,
                           optim.SGD,
                           rho = 0.05,
                           lr = lr_adapt,
                           weight_decay = cfg.OPTIM.WD
                           )
            else:

                return optim.SGD(params,
                    lr=lr_adapt,
                    momentum=cfg.OPTIM.MOMENTUM,
                    dampening=cfg.OPTIM.DAMPENING,
                    weight_decay=cfg.OPTIM.WD,
                    nesterov=cfg.OPTIM.NESTEROV)
        else:
            raise NotImplementedError        


