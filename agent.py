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

BATCH_SIZE = 100
GAMMA = 0.99
n_actions = 4
MEMORY_SIZE = 100000

def get_screen(settings,snake,apple):
    
    screen = torch.zeros(100)-1
    screen[0] = apple.x
    screen[1] = apple.y
        
    for index,i in enumerate(snake.body):
        screen[2*index+2] = i[0]
        screen[2*index+3] = i[1]
        if index==48:
            break
    
    #Resize, and add a batch dimension (BCHW)
    #resize = T.Compose([T.ToPILImage(),T.Resize((30,30), interpolation=T.InterpolationMode.BICUBIC),T.ToTensor()])
    #screen = resize(screen)
    
    screen=screen.unsqueeze(0)
    return screen




def optimize_model(memory,device,policy_net,target_net,optimizer,snake):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values = target_net(state_batch).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    loss1 = round(loss.item(),3)
    #print('loss = '+str(loss1))
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
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

    def __init__(self, inputs, outputs, device):
        super(DQN, self).__init__()
        self.head1 = nn.Linear(inputs, 512)
        self.head2 = nn.Linear(512, 512)
        self.head = nn.Linear(512, outputs)


    def forward(self, x):
        x = F.relu(self.head1(x))
        x = F.relu(self.head2(x))
        x = self.head(x)
        #print(x)
        return x
    
    
def create_agent(settings,device,MEMORY_SIZE):
   
    policy_net = DQN(100, n_actions,device).to(device)
    target_net = DQN(100, n_actions,device).to(device)

    
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-5)
    memory = ReplayMemory(MEMORY_SIZE)
    return policy_net, target_net, optimizer, memory

def select_action(state,policy_net):
    with torch.no_grad():
        w = policy_net(state)
        w = w - w.min()
        w = (w / w.sum())[0]
        #w = w**3
        #action = random.choices([0,1,2,3], weights=w, k=1)
        action = np.argmax(w)
        action = torch.tensor(action)
        action = action.view(1,1) 
        return action