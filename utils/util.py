import torch
from timm import create_model
class AverageMeter():
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0 
        self._avg = 0
        self._sum = 0
        self._count = 0

    def update(self, val, n=1):
        self.val = val
        self._sum += val*n
        self._count += n 
        self._avg = self._sum / self._count 
    
    @property
    def avg(self):
        return self._avg

def accuracy(output, target):
    acc = 0
    acc += (output.max(1)[1] == target).float().sum()
    batch_size = target.size()[0]
    return acc.mul_(100.0/batch_size)

class Normalize(torch.nn.Module):
    def __init__(self,device,dataset):
        super().__init__()
        if dataset in['cifar10c', 'cifar100c']:
            self.mu = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float).view(3,1,1).to(device)
            self.sigma = torch.tensor([0.2470, 0.2435, 0.2616], dtype=torch.float).view(3,1,1).to(device)
        else:
            self.mu = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float).view(3,1,1).to(device)
            self.sigma = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float).view(3,1,1).to(device)

    def forward(self, x):
        x = (x-self.mu)/self.sigma
        return x

def vit_b(num_classes, img_size=32, patch_size=4):
    # Create Vision Transformer model
    return create_model('vit_base_patch16_224',
                        pretrained=False,
                        num_classes=num_classes,
                        img_size=32,  # Adjusting the image size for CIFAR-100
                        patch_size=4)  # Smaller patch size for smaller input images