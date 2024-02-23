import random
import numpy as np
import torch
import math

from config import conf

device = torch.device("cuda:{:d}.format(conf.args.gpu_idx)" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(conf.args.gpu_idx)

class FIFO:
    def __init__(self, capacity):
        self.data = [[], [], []]
        self.capacity = capacity
    
    def set_memory(self, state_dict):
        self.data = [ls[:] for ls in state_dict['data']]
        if 'capacity' in state_dict.keys():
            self.capacity = state_dict['capacity']
    
    def save_state_dict(self):
        dic = {}
        dic['data'] = [ls[:] for ls in self.data]
        dic['capacity'] = self.capacity
        return dic
    
    def get_memory(self):
        return self.data
    
    def get_occupancy(self):
        return len(self.data[0])
    
    def add_instance(self,instance):
        assert (len(instance)==3), "An instance should contain data, label and domain label"
        
        if self.get_occupancy() >= self.capacity:
            self.remove_instance()
        for i, d in enumerate(self.data):
            d.append(instance[i])

    def remove_instance(self):
        for d in self.data:
            d.pop(0)

class HUS:
    def __init__(self, capacity, threshold=None):
        self.data = [[[], [], [], []] 
                     for  _ in range(conf.args.opt['num_class'])]
        self.counter = [0] * conf.args.opt['num_class']
        self.marker = [''] * conf.args.opt['num_class']
        self.capacity = capacity
        self.threshold = threshold
    
    def set_memory(self, state_dict):
        self.data = [
            [[l[:] for l in ls] for ls in state_dict['data']]
        ]

        self.counter = state_dict['counter'][:]
        self.marker = state_dict['marker'][:]
        self.capacity = state_dict['capacity']
        self.threshold = state_dict['threshold']

    def save_state_dict(self):
        dic = {}
        dic['data'] = [[l[:] for l in ls] for ls in self.data]
        dic['counter'] = self.counter[:]
        dic['marker'] = self.marker[:]
        dic['threshold'] = self.threshold

        return dic
    
    def print_class_dist(self):
        print(self.get_occupancy_per_class())
    
    def print_real_class_distribution(self):
        occupancy_per_class = [0] * conf.args.opt['num_class']
        ## lets implement later

    def get_occupancy(self):
        occupancy=0
        for data_per_class in self.data:
            occupancy += len(data_per_class[0])
        return occupancy
    
    def get_occupancy_per_class(self):

        occupancy_per_cls = [0] * conf.args.opt['num_class']
        for i, data_per_class in enumerate(self.data):
            occupancy_per_cls[i] += len(data_per_class[0])
        return occupancy_per_cls

    def get_memory(self):
        data = self.data
        tmp_data = [[], [], []]
        for data_per_class in self.data:
            feats, cls, dls, _ = data_per_class
            tmp_data[0].extend(feats)
            tmp_data[1].extend(cls)
            tmp_data[2].extend(dls)

        return tmp_data
    
    def add_instance(self, instance):
        assert (len(instance)), "instance length should be 4"
        cls = instance[1]
        is_add = True
        if self.threshold is not None and instance[3] < self.threshold:
            is_add = False
        elif self.get_occupancy > self.capacity:
            is_add = self.remove_instance(cls)
    
    def get_largest_indices(self):
        occpancy_per_class = self.get_occupancy_per_class()
        max_value = max(occpancy_per_class)
        largest_indices = []
        for i,value in enumerate(occpancy_per_class):
            if value == max_value:
                largest_indices.append(i)
        return largest_indices
    
    def remove_instance(self,cls):
        largest_indices = self.get_largest_indices()
        if cls not in largest_indices:
            largest = random.choice(largest_indices)
            tgt_idx = random.randrange(0,len(self.data[largest][3]))
            for dim in self.data[largest]:
                dim.pop(tgt_idx)
        else: 
            tgt_idx = random.randrange(0,len(self.data[cls][3]))
            for dim in self.data[cls]:
                dim.pop(tgt_idx)
        return True
    def reset_value(self, feats, cls, aux):
        pass #Implement later

# Rotta CSTU

class MemoryItem:
    def __init__(self, data=None, uncertainity=0, age=0):
        self.data = data
        self.uncertainity = uncertainity
        self.age = age
    
    def increase_age(self):
        if not self.empty():
            self.age +=1
    
    def get_data(self):
        return self.data, self.uncertainity, self.age
    
    def empty(self):
        return self.data == "empty"

class CSTU:

    def __init__(self, capacity, num_class, 
                 lambda_t=1.0, lambda_u=1.0):
        self.capacity = capacity
        self.num_class = num_class
        self.per_class = self.capacity / self.num_class
        self.lambda_t = lambda_t
        self.lamda_u = lambda_u

        self.data = [[] for _ in range(self.num_class)]
    
    def set_memory(self, state_dict):
        self.capacity = state_dict['capacity']
        self.num_class = state_dict['num_class']
        self.per_class = state_dict['per_class']
        self.lambda_t = state_dict['lambda_t']
        self.lambda_u = state_dict['lambda_u']
        self.data = [ls[:] for ls in state_dict['data']]
    
    def save_state_dict(self):
        state_dict = {}
        state_dict['capacity'] = self.capacity
        state_dict['num_class'] = self.num_class
        state_dict['per_class'] = self.per_class
        state_dict['lambda_t'] = self.lambda_t
        state_dict['lambda_u'] = self.lambda_u
        state_dict['data'] = [ls[:] for ls in self.data]

        return state_dict
    
    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls)
        return occupancy

    def per_class_dist(self):
        per_class_occupied = [0] * self.num_class
        for i,data_per_cls in enumerate(self.data):
            per_class_occupied[i] += len(data_per_cls)
        return per_class_occupied
    
    def add_instance(self, instance):
        assert(len(instance)==3) 
        x, prediction, uncertainity = instance
        new_item = MemoryItem(data=x, uncertainity=uncertainity, age=0)
        new_score = self.heuristic_score(0, uncertainity)
        if self.remove_instance(prediction, new_score):
            self.data[prediction].append(new_item)
        self.add_age()
    
    def remove_instance(self, cls, score):
        class_list = self.data[cls]
        class_occupancy = len(class_list)
        all_occupancy = self.get_occupancy
        if all_occupancy < self.capacity:
            return True
        if class_occupancy < self.per_class:
            majority_classes = self.get_majority_classes()
            return self.remove_from_classes(majority_classes, score)
        else:
            pass

    def remove_from_classes(self, classes, score_base):
        max_class = None
        max_index = None
        max_score = None
        for cls in classes:
            for idx, item in enumerate(self.data[cls]):
                uncertainity = item.uncertainity
                age = item.age
                score = self.heuristic_score()
                if max_score is None or score >= max_score:
                    max_score = score
                    max_index = idx
                    max_class = cls
        if max_class is not None:
            if max_score > score_base:
                self.data[max_class].pop(max_index)
                return True
            else:
                return False
        else:
            return True
                

    def get_majority_classes(self):
        per_class_dist = self.per_class_dist()
        max_occupancy = max(per_class_dist)
        max_classes = []
        for i, num in enumerate(per_class_dist):
            if num == max_occupancy:
                max_classes.append(i)
        return max_classes

    def heuristic_score(self,age,uncertainity):
        return self.lambda_t * 1/(1+math.exp(-age/self.capacity)) + self.lambda_u*(uncertainity/math.log(
            self.num_class
        ))
    
    def add_age(self):
        for class_list in self.data:
            for item in class_list:
                item.increase_age()
        return
    
    def get_memory(self):
        tmp_data = []
        tmp_age = []

        for class_list in self.data:
            for item in class_list:
                tmp_data.append(item.data)
                tmp_age.append(item.age)
        tmp_age = [x/self.capacity for x in tmp_age]

        return tmp_data, tmp_age

    


