import h5py
import numpy as np
import torch
from transformers import ViltProcessor, ViltModel

'''
There is a training pipeline for behavioral cloning based dataset 
using transformer-based agent
'''

DATASET = '/home/renas/pythonprogv2/snake_DQN/datasets/dataset_2024_09_26_12:34.h5'

class Renas(torch.nn.Module):
    def __init__(self, device):
        super(Renas, self).__init__()
        self.device = device
        
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        self.processor.current_processor.do_rescale = False
        self.processor.current_processor.do_resize = False
        self.processor.current_processor.do_normalize = False

        self.vilt_model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        self.d_model = self.vilt_model.config.hidden_size
        for param in self.vilt_model.parameters():
            param.requires_grad = True
    
    def forward(self, batch, action_vocab_token):
        im, action, _, prompt = batch
        i1, i2, i3, i4, i5 = im.size()
        im = im.view(i1*i2, i3, i4, i5)

        prompt = [prompt for prompt in prompt for _ in range(i2)]

if __name__ == '__main__':
    #load dataset
    with h5py.File(DATASET, 'r') as hdf:
        states = torch.from_numpy(np.array(hdf['states']))
        next_states = torch.from_numpy(np.array(hdf['next_states']))
        actions = torch.from_numpy(np.array(hdf['actions']))
        rewards = torch.from_numpy(np.array(hdf['rewards']))
    #preprocessing
    

print(torch.from_numpy(states).shape)
print(torch.from_numpy(next_states).shape)
print(torch.from_numpy(actions).shape)
print(torch.from_numpy(rewards).shape)