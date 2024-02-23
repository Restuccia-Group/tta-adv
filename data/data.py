import torch
from torch.utils.data import Dataset, DataLoader
from robustbench.data import load_cifar10c, load_cifar100c
#from config.conf import cfg
#for sanity check
import matplotlib.pyplot as plt

"""
An example of robustbench dataloader import code
corruptions = ['fog']
x_test,y_test = load_cifar10c(n_examples=1000,
 corruptions=corruptions,
 severity=5)

"""
class CIFARDataset(Dataset):
    """
    Desc: "Creation of cifar dataset class"

    """
    def __init__(self,
                 dataset:str = "cifar10c",
                 corruptions:str = None,
                 n_examples:int = 10000,
                 severity:int = 5) -> None:
        
        self.n_examples = n_examples
        if dataset=="cifar10c":
            self.x, self.y = load_cifar10c(n_examples=self.n_examples,
                                 corruptions=corruptions,
                                 severity=severity)
        elif dataset=="cifar100c":
            self.x, self.y = load_cifar100c(n_examples=self.n_examples,
                                 corruptions=corruptions,
                                 severity=severity)
        else:
            raise ValueError("Invalid Name:-> Please enter cifar10c or cifar100c")


    
    def __len__(self):
        return self.n_examples
    
    def __getitem__(self, index):
        #data are not read here, the whole dataset is loaded during init

        image = self.x[index]
        label = self.y[index]
        data = {'img':image, 'label': label}
        return data


def get_loader(dataset='cifar10c',
                      corruptions=None,
                      n_examples=1000,
                      severity = 5,
                      batch_size=64,
                      num_workers=2):
    
    assert dataset in ['cifar10c','cifar100c', 'imagenetc'], "dataset should be ['cifar10c','cifar100c', 'imagenetc']" 
    
    if dataset == 'cifar10c':
        dataset = CIFARDataset(dataset=dataset,
                                 corruptions=corruptions,
                           n_examples=n_examples,
                           severity=severity)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers)
    return dataloader

def sanity_check(dataset='cifar10c'):

    dataloader = get_loader(dataset=dataset,corruptions=['fog'], batch_size=4)
    for i,data in enumerate(dataloader):
        x , y = data['img'], data['label']
        print(f'data shape: {x.size()}')
        print(f'label shape: {x.size()}')
        break

if __name__ == '__main__':

    sanity_check('cifar10c')

    
    


