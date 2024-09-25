import numpy as np
from PIL import Image
from collections import deque,namedtuple
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

BATCH_SIZE = 2000
GAMMA = 0.95
n_actions = 4
MEMORY_SIZE = 2000
LR = 10e-7
SCHEDULER = False

def get_state(snake,apple):
    state = torch.zeros((30, 30))
    state[apple.x, apple.y] = 0.5
        
    for body_segment in snake.body:
        state[body_segment[0], body_segment[1]]= 0.75
    
    state[snake.body[0][0], snake.body[0][1]] = 1
    state=state.unsqueeze(0)
    return state




def optimize_model(memory,device,policy_net,target_net,optimizer,snake, update, scheduler):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))
    state_batch = torch.stack(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    next_state_batch = torch.stack(batch.next_state).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    #print('here', torch.min(reward_batch), torch.max(reward_batch))
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values = target_net(next_state_batch).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.MSELoss()
    #print('here', state_action_values[0], expected_state_action_values.unsqueeze(1)[0])
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    loss1 = round(loss.item(),3)
    #print('loss = '+str(loss1))
    
    loss.backward()
    if update:
        #for param in policy_net.parameters():
        #    param.grad.data.clamp_(-1, 1)
        optimizer.step()
        if SCHEDULER:
            scheduler.step()
        optimizer.zero_grad()

    
    return loss1





Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory():

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 750)
        self.fc2 = nn.Linear(750, 4)
        self.gelu = nn.GELU()   

    def forward(self, x):
        # x shape [batch, channel, w, h]
        x = self.pool(self.gelu(self.conv1(x)))  # Conv + ReLU + Pooling
        x = self.pool(self.gelu(self.conv2(x)))  # Conv + ReLU + Pooling
        x = x.view(x.size(0), -1)  # Flatten
        x = self.gelu(self.fc1(x))  # Fully connected layer
        x = self.fc2(x)  # Output layer
        return x  # Q-values for each action
    
    
def create_agent(settings,device,MEMORY_SIZE):
   
    policy_net = DQN().to(device)
    target_net = DQN().to(device)

    
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_SIZE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99) 
    return policy_net, target_net, optimizer, memory, scheduler

#def select_action(state,policy_net):
#    action = F.softmax(policy_net(state), dim=1)
#    print('here', action)
#    return action