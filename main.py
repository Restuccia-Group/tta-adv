import torch
from torchvision.models import resnet50
import numpy as np
import pandas as pd
import random
import math
import argparse
from tqdm import tqdm
from config.conf import cfg
import torch.optim as optim
from data.data import get_loader
from copy import deepcopy
from tta_algo.build import build_tta_adapter
from tta_algo.tta_base import TTA_BASE
from tta_attack.dia import DIA
from tta_attack.fca import FCA
from utils.util import accuracy, AverageMeter
from torch.profiler import profile, record_function, ProfilerActivity 
from torch.func import functional_call, vmap, grad
from robustbench.utils import load_model
from timm import create_model
from utils.util import Normalize
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

def set_deterministic(seed=42, cudnn=True):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = cudnn

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def load_selective_model_params(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict, strict = False)

def test_tta(model,data_loader,cfg,device):
    model=model.to(device)
    tta_adapter_cls = build_tta_adapter(cfg)
    tta_adapter = tta_adapter_cls(cfg=cfg,model=model)

    mal_num = math.ceil(cfg.DATA.BATCH_SIZE * cfg.DIA.MAL_PORTION)
    acc = AverageMeter()

    for n_batch, (inputs,labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device)
        outputs = tta_adapter.forward(inputs).detach().cpu()

        if outputs[:-mal_num].size()[0]:
            acc.update(accuracy(outputs[:-mal_num].cpu(),labels[:-mal_num].cpu()))
    return acc.avg

def test_normal(model,data_loader,cfg,device):
    model=model.to(device)
    model = model.eval()

    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    acc = AverageMeter()
    mal_num = math.ceil(cfg.DATA.BATCH_SIZE * cfg.DIA.MAL_PORTION)
    
    for n_batch, (inputs,labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device)
        inputs = transform(inputs)
        outputs = model.forward(inputs).detach().cpu()
        if outputs[:-mal_num].size()[0]:
            acc.update(accuracy(outputs[:-mal_num],labels[:-mal_num].cpu()))
    return acc.avg


# def compare_batchnorm_layers(model1, model2):
#     # Get the named parameters of both models
#     params1 = dict(model1.named_parameters())
#     params2 = dict(model2.named_parameters())

#     # Iterate through the named parameters and compare BatchNorm layer weights
#     for name, param1 in params1.items():
#         if 'conv' in name and isinstance(dict(model1.named_modules())[name.split('.')[0]], nn.Conv2d):
#             param2 = params2.get(name, None)
#             if param2 is None or not torch.equal(param1, param2):
#                 return False

#     return True

def test_attack(model, data_loader, cfg,device):
    centroid = torch.tensor([])
    model = model.to(device)
    victim_model = deepcopy(model)
    victim_model = victim_model.to(device)
    sur_model = deepcopy(model)
    sur_model = sur_model.to(device)

    tta_adapter_class = build_tta_adapter(cfg)
    tta_adapter = tta_adapter_class(cfg, model) # Unattacked TTA Adapter
    tta_adapter_victim = tta_adapter_class(cfg, victim_model) # Victim TTA Adapter
    # configuring surrogate model for crafting white box attack
    # tta_adapter_sur = tta_adapter_class(cfg, sur_model)
    # print(f"Sorrogate TTA Adapter Built")
    sur_model = tta_adapter.configure_model(sur_model, device) # TODO: check
    # sur_model = sur_model.to(device)
    # for n,p in sur_model.named_parameters():
    #     if p.requires_grad:
    #         print(p)
    sur_params, _ = TTA_BASE.collect_params(sur_model)
    # print(sur_params)
    # raise Exception
    sur_optim = TTA_BASE.setup_optimizer(sur_params, cfg)

    if cfg.BASE.ATTACK == "dia":
        attack = DIA(cfg=cfg)
    elif cfg.BASE.ATTACK == "fca":
        attack = FCA(cfg=cfg, layers=['avgpool'])
        # attack = FCA(cfg=cfg, layers=[str([*model.named_modules()][-1][0])])
    else:
        raise NotImplementedError("Specify an attack Name that is implemeted")
    
    mal_num = math.ceil(cfg.DATA.BATCH_SIZE * cfg.DIA.MAL_PORTION)

    acc_normal = AverageMeter()
    acc_attacked = AverageMeter()
    acc_normal_mal = AverageMeter()
    acc_attacked_mal = AverageMeter()

    bar = tqdm(data_loader)
    for n_batch, (inputs,labels) in enumerate(bar):
        if n_batch>50:
            continue
        inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device)

        model_state, optim_state = copy_model_and_optimizer(tta_adapter.model, tta_adapter.optimizer)
        if cfg.DIA.MED:
            tta_adapter_victim.revert_bn()
            tta_adapter_victim = tta_adapter_class(cfg, victim_model) # Victim TTA Adapter

        # if not cfg.DIA.SRC_ONLY: # if malicious insider provides last updated model
        #     load_model_and_optimizer(sur_model, sur_optim, model_state, optim_state)
            
        # if not cfg.DIA.CONTINUAL:
        #     load_model_and_optimizer(tta_adapter_victim.model, tta_adapter_victim.optimizer, model_state, optim_state)

        if cfg.DIA.SRC_ONLY:
            load_model_and_optimizer(tta_adapter_victim.model, tta_adapter_victim.optimizer, model_state, optim_state)
        
        else:
            load_model_and_optimizer(tta_adapter_victim.model, tta_adapter_victim.optimizer, model_state, optim_state)
            load_model_and_optimizer(sur_model, sur_optim, model_state, optim_state)
  
        outputs_clean = tta_adapter.forward(inputs).detach().cpu()

        x_adv, centroid = attack.generate_attack(sur_model=sur_model,
                                       x = inputs, y=labels, centroid=centroid)
    

        if cfg.DIA.MED:
            tta_adapter_victim.convert_bn()
        outputs_mal = tta_adapter_victim.forward(x_adv).detach().cpu()
       
        if cfg.DIA.CONTINUAL:
            victim_model_state, victim_optim_state = copy_model_and_optimizer(tta_adapter_victim.model, tta_adapter_victim.optimizer)
            load_model_and_optimizer(tta_adapter.model, tta_adapter.optimizer, 
                                     victim_model_state, victim_optim_state)
        #outputs_mal = victim_model(x_adv).detach().cpu()
        # is_same = compare_batchnorm_layers(tta_adapter.model, sur_model)
        # print(f'Are the parmeters same ?? Answer : {is_same}')
        # break

        #load surrogate weights
        #load_selective_model_params(sur_model, model_state)
        
        
        
        if outputs_clean[:-mal_num].size()[0]:
            normal_accuracy = accuracy(outputs_clean[:-mal_num], labels[:-mal_num].cpu())
            attacked_accuracy = accuracy(outputs_mal[:-mal_num], labels[:-mal_num].cpu())
            normal_accuracy_mal = accuracy(outputs_clean[-mal_num:], labels[-mal_num:].cpu())
            attacked_accuracy_mal = accuracy(outputs_mal[-mal_num:], labels[-mal_num:].cpu())

            acc_normal.update(normal_accuracy.item())
            acc_attacked.update(attacked_accuracy.item())
            acc_normal_mal.update(normal_accuracy_mal.item())
            acc_attacked_mal.update(attacked_accuracy_mal.item())

        bar.set_description(f"Batch => [{n_batch}/{len(data_loader)}]")
        bar.set_postfix(normal = acc_normal.avg, attacked = acc_attacked.avg, 
                        normal_mal = acc_normal_mal.avg, attacked_mal = acc_attacked_mal.avg)

    return acc_normal.avg, acc_attacked.avg, acc_normal_mal.avg, acc_attacked_mal.avg

def log_gradients(model, data_loader, cfg, device):

    model = model.to(device)
    grad_model = deepcopy(model)
    grad_model = grad_model.to(device)
    tta_adapter_class = build_tta_adapter(cfg)
    tta_adapter = tta_adapter_class(cfg,model)

    ft_compute_grad = grad(compute_loss)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

    if cfg.BASE.ATTACK == "dia":
        attack = DIA(cfg=cfg)
    elif cfg.BASE.ATTACK == "fca":
        attack = FCA(cfg=cfg, layers=['avgpool'])
    else:
        raise NotImplementedError("Specify an attack Name that is implemeted")

    mal_num = math.ceil(cfg.DATA.BATCH_SIZE * cfg.DIA.MAL_PORTION)
    grad_benign = torch.tensor([])
    grad_mal = torch.tensor([])
    grad_all = torch.tensor([])

    for n_batch,data in enumerate(data_loader):

        inputs, labels = data['data'], data['label']
        inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device)
        params = {k:v.detach() for k,v in grad_model.named_parameters()}
        buffers = {k: v.detach() for k, v in grad_model.named_buffers()}
        outputs_clean = tta_adapter.forward(inputs).detach().cpu()

        x_adv = attack.generate_attack(sur_model=tta_adapter.model,
                                       x = inputs, y=labels)

        ft_per_sample_grads = ft_compute_sample_grad(params, buffers, inputs, labels)
        ft_per_sample_grads = ft_per_sample_grads['layer3.2.conv2.weight']

        grad_all = torch.cat(
            (grad_all, torch.linalg.matrix_norm(torch.squeeze(ft_per_sample_grads)
        ,ord='nuc').cpu()),dim=0
        )
        grad_mal = torch.cat(
            (grad_mal, torch.linalg.matrix_norm(torch.squeeze(ft_per_sample_grads[-mal_num:])
        ,ord='nuc').cpu()),dim=0
        )
        grad_benign = torch.cat(
            (grad_mal, torch.linalg.matrix_norm(torch.squeeze(ft_per_sample_grads[:-mal_num])
            ,ord='nuc').cpu()),dim=0)
        model_state, optimizer_state = copy_model_and_optimizer(tta_adapter.model, tta_adapter.optimizer)
    return grad_all.numpy(), grad_mal.numpy(), grad_benign.numpy()


    # return acc_normal.avg, acc_attacked.avg

    

def compute_loss(params, buffers, sample, target):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)

    predictions = functional_call(model, (params, buffers), (batch,))
    loss = torch.nn.functional.nll_loss(predictions, targets)
    return loss

class SaveOutputHook:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs[self.layer_id] = module_out.detach()
    
    def get_output(self):
        return self.outputs

class SaveInputHook:

    def __call__(self, module, module_in, module_out):
        module.in_feat = module_in[0].detach().clone().flatten(1)

        
def update_configs(cfg,args):
    #gpu_id
    if args.gpu_id:
        cfg.BASE.GPU_ID = args.gpu_id
    #tta method 
    if args.tta:
        cfg.TTA.NAME = args.tta
    # batch size
    if args.batch_size:
        cfg.DATA.BATCH_SIZE = args.batch_size
    #whether to use pseudo label or not for DIA
    if args.pseudo:
        cfg.DIA.PSEUDO = args.pseudo
    if args.attack:
        cfg.BASE.ATTACK=args.attack
    if args.dataset:
        cfg.CORRUPTION.DATASET = args.dataset
    if args.severity:
        cfg.DATA.SEVERITY = args.severity
    if args.model:
        cfg.MODEL.ARCH= args.model
    return cfg




if __name__ == "__main__":
    
    set_deterministic(seed=cfg.BASE.SEED, cudnn=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('-bs','--batch_size',type=int, help="Specify Batch Size",default=200)
    parser.add_argument('--gpu_id',type=int, help="Specify GPU ID")
    parser.add_argument('--tta',type=str, help="Specify the TTA algorithm to be used")
    parser.add_argument('--severity',type=int, help="The severity of corruption type")
    parser.add_argument('--pseudo',type=bool, help="Specify whether to use pseudo labels or not")
    parser.add_argument('--dataset', type=str,help="The dataset used for evaluation" )
    parser.add_argument('--attack', type=str, help="Specify the attack algorithm" )
    parser.add_argument('--model', type=str, help="Specify the model" )
    args = parser.parse_args()

    conf = update_configs(cfg,args)
    device = torch.device("cuda:{:d}".format(cfg.BASE.GPU_ID) if torch.cuda.is_available() else "cpu")
    results_df = pd.DataFrame(columns=["Corruption", "Normal Accuracy", "Adaptation Accuracy",
                                       "Malicious Normal Accuracy", "Malicious Adaptation Accuracy"])
    
   
    
    corruptions = conf.CORRUPTION.TEST
    #corruptions = ['frost','fog']
    for corruption in corruptions:
        if cfg.CORRUPTION.DATASET == 'cifar10c':
            cfg.DATA.NUM_CLASSES = 10
            if not cfg.DIA.ADV_MODEL:
                model = torch.hub.load("chenyaofo/pytorch-cifar-models", 
                            "cifar10_resnet20", pretrained=True)
            else:
                model = load_model(model_name='Hendrycks2020AugMix_WRN', dataset='cifar10', threat_model='common_corruptions')

            cfg.DATA.NUM_CLASSES = 10
        elif cfg.CORRUPTION.DATASET == 'cifar100c':
            cfg.DATA.NUM_CLASSES = 100
            if not cfg.DIA.ADV_MODEL:
                model = torch.hub.load("chenyaofo/pytorch-cifar-models", 
                            "cifar100_mobilenetv2_x0_5", pretrained=True)
                
                # model = create_model('vit_base_patch16_224',
                #         pretrained=True,
                #         num_classes=cfg.DATA.NUM_CLASSES,
                #         img_size=32,  # Adjusting the image size for CIFAR-100
                #         patch_size=4)  # Smaller patch size for smaller input images
                # checkpoint = torch.load('vit_b_cifar100_model_SA_best.pth.tar')
                # print(checkpoint['state_dict'].keys())
                # raise Exception
                model.load_state_dict(torch.load('vit_b_cifar100_model_SA_best.pth.tar')['state_dict'],strict=False)
            else:
                model = load_model(model_name='Wu2020Adversarial', dataset='cifar100', threat_model='Linf')
            

        elif cfg.CORRUPTION.DATASET == 'imagenetc':
            model = resnet50(weights="IMAGENET1K_V2")
            model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
            cfg.DATA.NUM_CLASSES = 1000
            # if cfg.MODEL.ARCH == 'ViT_b':
            #     model = timm.create_model('vit_base_patch16_224', pretrained=True)
            # elif cfg.MODEL.ARCH == 'ViT_l':
            #     model = timm.create_model('vit_large_patch16_224', pretrained=True)
            # elif cfg.MODEL.ARCH == 'SWin_b':
            #     model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
            # elif cfg.MODEL.ARCH == 'SWin_l':
            #     model = timm.create_model('swin_large_patch4_window7_224', pretrained=True)
            # else:
            #     print(f"Model {cfg.MODEL.ARCH} not supported")
            


        else:
            print("The dataset is not supported")
        # model = load_model(model_name='Standard',
        #            dataset='cifar10',
        #            threat_model = 'Linf')
        data_loader = get_loader(cfg=cfg,
                              corruption = corruption,
                              robustbench=True
                              )
        
        # acc = test_normal(model, data_loader,cfg,device)
        # print(f"Normal Accuracy in {corruption} --> {acc}")
        # acc_adapted = test_tta(model,data_loader,cfg,device)
        # print(f"Accuracy in {corruption} after adaptation --> {acc_adapted}")

        acc_normal, acc_attacked, acc_normal_mal, acc_attacked_mal = test_attack(model,data_loader,cfg,device)
        # acc_normal, acc_attacked, acc_normal_mal, acc_attacked_mal= acc_normal.item(), acc_attacked.item(), acc_normal_mal.item(), acc_attacked_mal.item()
        print(f"Accuracy after Adaptation for {corruption} --> {acc_normal}")
        print(f"Attacked Accuracy for  {corruption} --> {acc_attacked}")
        print(f"Malicious Accuracy after Adaptation for {corruption} --> {acc_normal_mal}")
        print(f"Attacked Malicious Accuracy for  {corruption} --> {acc_attacked_mal}")
        #results_df.loc[len(results_df)] = np.array([corruption, acc_normal,acc_attacked, acc_normal_mal, acc_attacked_mal])
        results_df.loc[len(results_df)] = np.array([corruption, acc_normal,acc_attacked, acc_normal_mal, acc_attacked_mal])
        results_df.to_csv('results_{}_{}_{}_{}_{}'.format(cfg.TTA.NAME, cfg.BASE.ATTACK, 
                cfg.CORRUPTION.DATASET, cfg.DATA.BATCH_SIZE, cfg.DATA.SEVERITY), sep='\t', encoding='utf-8')
        # all, mal, benign = log_gradients(model, data_loader,cfg, device)
        # np.save('all.npy',all)
        # np.save('mal.npy',mal)
        # np.save('benign.npy',benign)



        
