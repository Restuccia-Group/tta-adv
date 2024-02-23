import torch
from config.conf import cfg
import torch.optim as optim
from utils.data import get_loader
from copy import deepcopy
from tta_algo.build import build_tta_adapter
from tta_attack.dia import DIA

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")

""" def setup_optimizer(params):

    lr_adapt = cfg.OPTIM.LR_ADAPT 
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=lr_adapt,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=lr_adapt,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError """

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def test_attack(model, dataloader, cfg, tta_algo='rotta'):

    model = torch.hub.load("chenyaofo/pytorch-cifar-models", 
                           "cifar10_resnet20", pretrained=True)
    victim_model = deepcopy(model).to(device)
    model = model.to(device)

    tta_adapter_class = build_tta_adapter(cfg)
    tta_adapter = tta_adapter_class(cfg,model)
    tta_adapter_victim = tta_adapter_class(cfg, victim_model)
    attack = DIA(cfg=cfg)

    train_loader = get_loader(dataset = cfg.DATASET.NAME,
                             corruptions = cfg.DATSET.CORRUPTIONS,
                             n_examples = cfg.DATASET.N_EXAMPLES,
                             severity = cfg.DATASET.SEVERITY
                             )  
    ### test dia attack
    params_sur,_ = tta_adapter.collect_params(model)
    inner_opt_sur = tta_adapter.setup_optimizer(params_sur)

    for data in dataloader:
        x , y = data['img'].to(device), data['label'].to(device)
        #model_state, optimizer_state = copy_model_and_optimizer(tta_adapter.model, tta_adapter.optimizer)
        #load_model_and_optimizer(sur_model, inner_opt_sur, model_state, optimizer_state)
        #load_model_and_optimizer(victim_model, inner_opt_victim, model_state, optimizer_state)
        x_adv = attack.generate_attack(sur_model=tta_adapter.model,
                                       x = x, y=y)
        
        outputs_clean = tta_adapter.forward(x)
        
        outputs_mal = tta_adapter_victim.forward(x_adv)

        
        
        