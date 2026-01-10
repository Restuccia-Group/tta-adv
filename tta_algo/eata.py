import torch
import torch.nn as nn
import math
from tta_algo.tta_base import TTA_BASE
from utils.loss_functions import softmax_entropy
from utils.bn_layers import *
from data.data import get_fisher_loader
from utils.loss_functions import softmax_entropy
import torch.nn.functional as F
from copy import deepcopy


class EATA(TTA_BASE):
    def __init__(self, cfg, model, steps=1, fisher_alpha= 2000, e_margin = math.log(1000)/2-1, d_margin=0.05):
        super().__init__(cfg, model)
        self.fisher_alpha = fisher_alpha
        self.d_margin=d_margin
        self.e_margin = e_margin
        self.steps = steps
        self.num_samples_update_1 = 0  # number of samples after First filtering, exclude unreliable samples
        self.num_samples_update_2 = 0 
        self.current_model_probs = None
        self.device = torch.device("cuda:{:d}".format(cfg.BASE.GPU_ID) if torch.cuda.is_available() else "cpu")
        self.episodic = False
        ### fisher dataset loader here
        data_loader = get_fisher_loader(cfg = cfg)

        params, param_names = self.collect_params(self.model)
        ewc_optimizer = torch.optim.SGD(params, 0.001)
        fishers = {}
        train_loss_fn = nn.CrossEntropyLoss().to(self.device)

        for iter, (inputs,labels) in enumerate(data_loader):

            inputs = inputs.to(self.device, dtype=torch.float)
            # if torch.cuda.is_available():
            #     targets = targets.cuda(conf.args.gpu_idx, non_blocking=True)
            outputs = self.model(self.normalize(inputs))
            _, targets = outputs.max(1)
            loss = train_loss_fn(outputs, targets)
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if iter > 1:
                        fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                    else:
                        fisher = param.grad.data.clone().detach() ** 2
                    if iter == len(data_loader):
                        fisher = fisher / iter
                    fishers.update({name: [fisher, param.data.clone().detach()]})
            ewc_optimizer.zero_grad()
        del ewc_optimizer
    
    def forward(self, x):
        if self.episodic:
            self.reset()
        if self.steps > 0:
            for _ in range(self.steps):
                outputs, num_counts_2, num_counts_1, updated_probs = self.forward_and_adapt(x)
                self.num_samples_update_2 += num_counts_2
                self.num_samples_update_1 += num_counts_1
                self.reset_model_probs(updated_probs)
        else:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(self.normalize(x))
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

    def reset_steps(self, new_steps):
        self.steps = new_steps

    def reset_model_probs(self, probs):
        self.current_model_probs = probs   
    
    def forward_and_adapt(self,x):

        outputs = self.model(x)
        entropys = softmax_entropy(outputs)
        filter_ids_1 = torch.where(entropys < self.e_margin)
        ids1 = filter_ids_1
        ids2 = torch.where(ids1[0] > -0.1)
        entropys = entropys[filter_ids_1]

        if self.current_model_probs is not None: 
            cosine_similarities = F.cosine_similarity(self.current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
            entropys = entropys[filter_ids_2]
            ids2 = filter_ids_2
            updated_probs = self.update_model_probs(self.current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
        else:
            updated_probs = self.update_model_probs(self.current_model_probs, outputs[filter_ids_1].softmax(1))
            coeff = 1 / (torch.exp(entropys.clone().detach() - self.e_margin))
            # implementation version 1, compute loss, all samples backward (some unselected are masked)
            entropys = entropys.mul(coeff) # reweight entropy losses for diff. samples
            loss = entropys.mean(0)
        outputs = self.model(x)
        return outputs, entropys.size(0), filter_ids_1[0].size(0), updated_probs
    
    def update_model_probs(self,current_model_probs, new_probs):
        if current_model_probs is None:
            if new_probs.size(0) == 0:
                return None
            else:
                with torch.no_grad():
                    return new_probs.mean(0)
        else:
            if new_probs.size(0) == 0:
                with torch.no_grad():
                    return current_model_probs
            else:
                with torch.no_grad():
                    return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)

    def configure_model(self, model, device):
        model.train()
        model.requires_grad_(False)

        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m,nn.LayerNorm):
                m.requires_grad_(True)
                m.track_running_stats = False
                # m.running_mean = None
                # m.running_var = None
        model = model.to(device)
        return model
    
    def collect_params(self,model):
        params = []
        names = []
        for nm, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names
    
    def convert_bn(self):
        normlayer_names = []
        for name, sub_module in self.model.named_modules():
                if isinstance(sub_module, nn.BatchNorm2d):
                    normlayer_names.append(name)

        for name in normlayer_names:
            bn_layer = self.get_named_submodule(self.model, name)

            if isinstance(bn_layer, nn.BatchNorm2d):

                norm_layer = MedBN2d(bn_layer, momentum = 1)
                    
            else:
                raise RuntimeError()

            norm_layer.weight.requires_grad_(True)
            norm_layer.bias.requires_grad_(True)
            self.set_named_submodule(self.model, name, norm_layer)

    def revert_bn(self):
        normlayer_names = []
        for name, sub_module in self.model.named_modules():
                if isinstance(sub_module,MedBN2d):
                    normlayer_names.append(name)

        for name in normlayer_names:
            bn_layer = self.get_named_submodule(self.model, name)

            if isinstance(bn_layer, MedBN2d):
   
                norm_layer = nn.BatchNorm2d(num_features=bn_layer.num_features)
                    
            else:
                raise RuntimeError()

            norm_layer.weight.requires_grad_(True)
            norm_layer.bias.requires_grad_(True)

            self.set_named_submodule(self.model, name, norm_layer)

    
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
    def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
    
    
