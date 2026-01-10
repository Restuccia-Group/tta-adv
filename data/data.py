import os
import json
import random
from pathlib import Path
import pandas as pd
import torch
from config.conf import cfg
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from robustbench.data import load_cifar10c, load_cifar100c, load_imagenetc
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# resolve path issues for sanity check
#sys.path.append("..")

class CIFARDataset(Dataset):
    """
    A class for loading CIFAR Dataset from robustbench
    """
    def __init__(self,cfg, corruption_type) -> None:
        assert cfg.CORRUPTION.DATASET in ['cifar10c', 'cifar100c', 'imagenetc'], "Invalid dataset name in config"

        if cfg.CORRUPTION.DATASET == 'cifar10c':
            self.x, self.y = load_cifar10c(cfg.CORRUPTION.N_EXAMPLES,
                                           cfg.DATA.SEVERITY,
                                           cfg.DATA_DIR,
                                           False,
                                           [corruption_type]
                                           )
        
        elif cfg.CORRUPTION.DATASET == 'cifar100c':
            self.x, self.y = load_cifar100c(cfg.CORRUPTION.N_EXAMPLES,
                                           cfg.DATA.SEVERITY,
                                           cfg.DATA_DIR,
                                           False,
                                           [corruption_type]
                                           )
        elif cfg.CORRUPTION.DATASET == 'imagenetc':
            self.x, self.y = load_imagenetc(
                                           severity = cfg.DATA.SEVERITY,
                                           data_dir = cfg.DATA_DIR,
                                           shuffle = False,
                                           corruptions=[corruption_type]
                                           )
        else:
            raise ValueError(" cfg.CORRUPTION.DATASET should be cifar10c or cifar100c for this class")
        # self.transform = transforms.Compose([
        #     transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
        #     ])
    def __len__(self):
            return len(self.y)

    def __getitem__(self, index):
            input, label = self.x[index], self.y[index]
            # input = self.transform(input)
            return input, label        
        
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(cfg.BASE.SEED)
    random.seed(cfg.BASE.SEED)

def get_transform(mode='test'):
    transform = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
        ])
    }
    return transform[mode] 

imgnet_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

idx2label = []
# cls2label = {}
# with open("imagenet_class_index.json", "r") as read_file:
#     class_idx = json.load(read_file)
#     # idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
#     # cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
#     folder2label = { class_idx[str(k)][0]:k for k in range(len(class_idx))}

# def target_transform(target):
#         return folder2label[target]

def get_corrupted_dataset(cfg, corruption=None):
    data_dir = Path(cfg.DATA.PATH) / f'ImageNet-C' / corruption / f"{cfg.DATA.SEVERITY}"
    transform = imgnet_transform
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)#, target_transform=target_transform)
    return dataset 

def get_loader(cfg, corruption, robustbench=True):
    g = torch.Generator()
    g.manual_seed(cfg.BASE.SEED)
    if cfg.CORRUPTION.DATASET == 'imagenetc':
        print("Loading corrupted Imagenet")
        dataset = get_corrupted_dataset(cfg, corruption)
    else:
        print("Loading corrupted dataset from directory")
        dataset = CIFARDataset(cfg, corruption)

    dataloader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = cfg.DATA.BATCH_SIZE,
        generator = g,
        shuffle = True, # for test we also need shuffle to generate diversified batch of data
        num_workers = cfg.BASE.NUM_WORKERS,
        worker_init_fn = seed_worker
    )
    return dataloader

def get_fisher_loader(cfg, corruption = 'normal'):

    g = torch.Generator()
    g.manual_seed(cfg.BASE.SEED)   
    if cfg.CORRUPTION.DATASET == 'imagenetc':
        print("Loading ImageNet for Fisher Loader") 
        dataset = get_corrupted_dataset(cfg, corruption= "brightness")
    else:
        print("Loading corrupted dataset from directory")
        dataset = CIFARDataset(cfg,corruption_type = "brightness")

    dataset_size = len(dataset)
    dataset_indices = list(range(dataset_size))
    np.random.seed(0)
    np.random.shuffle(dataset_indices)
    fisher_split_index = int(np.floor(0.1*dataset_size))
    fisher_idx = dataset_indices[:fisher_split_index]
    fisher_sampler = SubsetRandomSampler(fisher_idx)

    fisher_dataloader = DataLoader(dataset=dataset, batch_size=cfg.DATA.BATCH_SIZE, 
                            sampler=fisher_sampler,worker_init_fn=seed_worker, shuffle = False,
                            generator=g, num_workers= cfg.BASE.NUM_WORKERS)
    return fisher_dataloader

def sanity_check(dataset='cifar10c'):

    dataloader = get_loader(cfg,corruption='fog', robustbench=False)
    for i, (x,y) in enumerate(dataloader):
        #x , y = data['img'], data['label']
        print(f'data shape: {x.size()}')
        print(f'label shape: {y.size()}')
        break

if __name__ == '__main__':

    sanity_check('cifar10c')
        
        


