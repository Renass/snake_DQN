import h5py
import numpy as np
import torch
from transformers import ViltProcessor, ViltModel
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR
from torch.utils.tensorboard import SummaryWriter
import os

'''
There is a training pipeline for behavioral cloning based dataset 
using transformer-based agent
'''

DATASET = '/home/renas/pythonprogv2/snake_DQN/datasets/dataset_2024_09_28_06:59.h5'
TEST_PART = 0.2
BATCH_SIZE = 1000
DEVICE = 'cuda:0'

LR = 10e-5
LR_WARMUP_EPOCHS = 5 
LR_DECAY_EPOCHS = 100

WEIGHTS_DIR = '/home/renas/pythonprogv2/snake_DQN/weights'
LOAD_WEIGHTS = 'renas.pt'
SAVE_WEIGHTS = 'renas.pt'

###
#Classes
###

class SnakeDataset(Dataset):
    def __init__(self, states, actions, rewards, next_states):
        self.states = states
        self.next_states = next_states
        self.actions = actions
        self.rewards = rewards

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]
        next_state = self.next_states[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        return state, action, reward, next_state

class Renas(torch.nn.Module):
    def __init__(self, device):
        super(Renas, self).__init__()
        self.device = device
        
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        self.processor.current_processor.do_rescale = False
        self.processor.current_processor.do_resize = True
        self.processor.current_processor.do_normalize = False

        self.vilt_model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        self.d_model = self.vilt_model.config.hidden_size
        for param in self.vilt_model.parameters():
            param.requires_grad = True
    
    def forward(self, batch):
        pass

###
#Functions
###

def train_loop(train_dataset, test_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)    
    model = Renas(DEVICE).to(DEVICE)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler1 = LinearLR(optimizer, start_factor=0.1, total_iters=LR_WARMUP_EPOCHS)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=LR_DECAY_EPOCHS, eta_min= LR/10)
    scheduler3 = ConstantLR(optimizer, factor=LR/10, total_iters= 100000)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2, scheduler3], milestones=[LR_WARMUP_EPOCHS, LR_WARMUP_EPOCHS+LR_DECAY_EPOCHS])
    criterion = torch.nn.CrossEntropyLoss()
    ten_board_writer = SummaryWriter()

    if os.path.isfile(os.path.join(WEIGHTS_DIR, LOAD_WEIGHTS)):
        model_dict = model.state_dict()
        pretrained_dict = torch.load(os.path.join(WEIGHTS_DIR, LOAD_WEIGHTS))
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        del model_dict, pretrained_dict
        print('weights loaded from file.')

if __name__ == '__main__':
    #load dataset
    with h5py.File(DATASET, 'r') as hdf:
        states = torch.from_numpy(np.array(hdf['states']))
        next_states = torch.from_numpy(np.array(hdf['next_states']))
        actions = torch.from_numpy(np.array(hdf['actions']))
        rewards = torch.from_numpy(np.array(hdf['rewards']))
    #preprocessing
    dataset = SnakeDataset(states, actions, rewards, next_states)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [1-TEST_PART, TEST_PART])
    train_loop(train_dataset, test_dataset)