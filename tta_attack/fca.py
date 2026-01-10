import torch
import math
import torch.nn as nn
from config.conf import cfg 
from torchvision import transforms
from utils.loss_functions import *
from utils.util import Normalize

class FCA:

    def __init__(self, cfg, layers):
        self.cfg = cfg
        self.device = torch.device("cuda:{:d}".format(cfg.BASE.GPU_ID) if torch.cuda.is_available() else "cpu")
        self.normalize = Normalize(self.device, cfg.CORRUPTION.DATASET)
        self.layers=layers

        # Initialize feature dict to save features
        self._features = {layer_id:torch.empty(0) for layer_id in layers}
        self._features_adv = {layer_id: torch.empty(0) for layer_id in layers}
        # Initialize handles to attach and remove hooks
        self.handles = {layer_id:[] for layer_id in layers}

        self.activation_hook = Activation_Hook()
        
    def generate_attack(self, sur_model, x, y, centroid):
        #x = self.denorm(x, mean =[0.4914, 0.4822, 0.4465], std =[0.2470, 0.2435, 0.2616])
        self.model = sur_model
        for layer_id in self.layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            self.handles[layer_id] = layer.register_forward_hook(self.activation_hook)
        
        # self._features = layer.act
        # del layer.act

        x,y = x.to(self.device), y.to(self.device)

        # variables needed to calculate class feature centroids
        # out = sur_model(x_adv) if cfg.DIA.ADV_MODEL else out = sur_model(self.normalize(x_adv))
        if cfg.DIA.ADV_MODEL:
            prediction = self.model(x).detach().clone() 
        else: 
            prediction = self.model(self.normalize(x)).detach().clone()
        self.feat_for_centroid = layer.act.squeeze().detach().clone()
        if centroid.nelement() == 0:
            self.centroid = torch.zeros(self.cfg.DATA.NUM_CLASSES, self.feat_for_centroid.size()[1]).to(self.device)
        else:
            self.centroid = centroid
        del(layer.act)
        self.pred_cls = torch.argmax(prediction,dim=1)
        self._cls = torch.unique(self.pred_cls)#.tolist() # create a list
        self.calculate_centriod()
      

        num_iter = self.cfg.DIA.STEPS
        epsilon = self.cfg.DIA.EPS
        alpha = self.cfg.DIA.ALPHA
        mal_num = math.ceil(self.cfg.DIA.MAL_PORTION * self.cfg.DATA.BATCH_SIZE)
        fixed = torch.zeros_like(x.clone()[:-mal_num], requires_grad=False)
        if cfg.DIA.INIT == 'const':
            adv = (torch.zeros_like(x.clone()[-mal_num:])- x[-mal_num:] + 127.5/255).requires_grad_(True)
        else:
            adv = (torch.rand_like(x.clone()[-mal_num:])).requires_grad_(True)
        adv_pad = torch.cat((fixed,adv), 0)
        adv_pad = adv_pad.to(self.device)

        for t in range(num_iter):
             # to compute clean activations
            weight = None
            if cfg.DIA.ADV_MODEL:
                out_clean = sur_model(x)
            else:
                out_clean = sur_model(self.normalize(x))

            prob = out_clean[:-mal_num].softmax(1).max(dim=1,keepdim=False)[0]
            self._features = layer.act
            del layer.act
            x_adv = x + adv_pad
            if cfg.DIA.ADV_MODEL:
                out = sur_model(x_adv) 
            else:
                out = sur_model(self.normalize(x_adv))

            self._features_adv = layer.act
            del layer.act
            #predict = torch.softmax(out, dim=1)
            # entropy = softmax_entropy(out)[:-mal_num]
            # print(out_clean[:-mal_num].softmax(1).max(dim=1,keepdim=False)[0])
            # print(torch.argmax(out_clean[:-mal_num].clone().detach(), dim=1)==y[:-mal_num])
            # raise Exception
            # loss = nn.CrossEntropyLoss()(out[:-mal_num], y[:-mal_num])
            feat_col = self.feature_collapse()
            feat_disperse = self.feature_dispersion()
            feat_centroid = self.feat_centroid_loss()
            #weight = prob
            if weight is not None:
                #loss = torch.mul(loss, torch.exp(-weight))
                feat_disperse = torch.mul(feat_disperse, weight)

            #loss = feat_loss #+ entropy
            #weight = entropy
            loss = feat_disperse - feat_col + feat_centroid
            # print(f'Unweighted Loss ===> {loss[:20]}')
            
            # print(f'Loss ===> {loss[:20]}')
            # print(f'adv_out : {torch.argmax(out.clone().detach(),dim=1)[:20]}')
            # print(f'clean out: {torch.argmax(out_clean.clone().detach(),dim=1)[:20]}')
            # print(f'labels {y[:20]}')
            
            # print(f"weights ==> {weight[:20]}")
            #raise Exception
            loss = loss.mean()
            
            loss.backward()

            adv.data = (adv - alpha*adv.grad.detach().sign()).clamp(-epsilon,epsilon)
            adv.data = (adv.data +x[-mal_num:]).clamp(0,1) -(x[-mal_num:])
            adv_pad.data = torch.cat((fixed, adv), 0) 

        for layer_id in self.layers:
            self.handles[layer_id].remove()
        
        x_adv = x + adv_pad
        #x_adv = self.transform(x_adv)
        #x_adv = x_adv.detach()
        # print(f'adv_out : {torch.argmax(out.clone().detach(),dim=1)[:20]}')
        # print(f'clean out: {torch.argmax(out_clean.clone().detach(),dim=1)[:20]}')
        # print(f'labels {y[:20]}')
        # #print(f'Unweighted Loss ===> {loss[:20]}')
        # print(f"weights ==> {weight[:20]}")
        # raise Exception
        return x_adv, self.centroid
    
    def calculate_centriod(self):
        momentum = 0.6
        # self.centroid = torch.zeros(self.cfg.DATA.NUM_CLASSES, self.feat_for_centroid.size()[1]).to(self.device)
        for i in self._cls:
            self.centroid[i,:] = momentum*self.centroid [i,:] + (1-momentum)*torch.mean(self.feat_for_centroid[self.pred_cls == i], 0)
        
    
    def check_dissimilarity(self, X):
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2) 
        print(dist)
    
    def feat_centroid_loss(self):
        mal_num = math.ceil(self.cfg.DIA.MAL_PORTION * self.cfg.DATA.BATCH_SIZE)
        loss_func = nn.CosineSimilarity(dim=1, eps=1e-6)
        #loss_func = nn.MSELoss()
        feat_adv = self._features_adv[:-mal_num].squeeze()
        centroid_copy = torch.zeros_like(feat_adv)
        index_list = self.pred_cls[:-mal_num]
        #print(f"Printing Index List {index_list}")
        centroid_copy = self.centroid[index_list]
        #centroid_copy = centroid_copy[:-mal_num]
        loss = loss_func(feat_adv, centroid_copy)
        loss = loss
        return loss


        
    def feature_collapse(self, weight=None):
        mal_num = math.ceil(self.cfg.DIA.MAL_PORTION * self.cfg.DATA.BATCH_SIZE)
        feat = self._features_adv[:-mal_num].squeeze()
        mu = torch.mean(feat,0).squeeze()
        mu = mu.unsqueeze(dim=0)
        #temp2 = temp-mu
        #distance = torch.sqrt(torch.tensordot(temp2,temp2,dims=([1],[1])).diag())
        #loss_func = nn.MSELoss()
        #loss = nn.L1Loss()
        loss_func = nn.CosineSimilarity(dim=1, eps=1e-6)
        loss = loss_func(feat, mu)
        return loss
    
    def feature_dispersion(self, weight=None):
        mal_num = math.ceil(self.cfg.DIA.MAL_PORTION * self.cfg.DATA.BATCH_SIZE)
        feat_clean = self._features[:-mal_num].squeeze()
        feat_adv = self._features_adv[:-mal_num].squeeze()
        loss_func = nn.CosineSimilarity(dim=1, eps=1e-6)
        #loss_func = nn.MSELoss()
        loss = loss_func(feat_clean, feat_adv)
        return loss



    def save_outputs_hook(self, layer_id:str):
        def fn(module, input , output):
            #self._features[layer_id] = output.detach()
            self._features[layer_id] = output.clone()
        return fn

    def remove_handles(self):
        for layer_id in self.layers:
            self.handles[layer_id].remove()

    def denorm(self, batch, mean, std):

        if isinstance(mean, list):
            mean = torch.tensor(mean).to(batch.device)
        if isinstance(std, list):
            std = torch.tensor(std).to(batch.device)
        return torch.clamp(batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1), 0, 1)

class Activation_Hook():
    def __call__(self, module, input, output):
        # module.act = output.clone()
        # print(input[0].shape)
        module.act = input[0].clone() #detach().clone().cpu()
        


    
    

