import pygame
from settings import Settings
import game_functions as gf
import agent as ag
import keyboard as k

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import random

from torch.utils.tensorboard import SummaryWriter

LOAD_WEIGHTS = False
LOAD_WEIGHTS_NAME = 'snake.pt'
SAVE_WEIGHTS = True
SAVE_WEIGHTS_NAME = 'snake.pt' 
SCREEN_INIT = True
# Training epochs for no_gui mode, when (SCREEN_INIT==False)
EPOCHS = 10000
#How many steps goes untill optimizer make learning step for policy_net
GRAD_ACCUMULATION_RATE = 1
#How many training steps goes untill target_net copies policy_net
SYNC_POLICIES_RATE = 10

settings = Settings()
snake,apple = gf.new_game(settings)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
print(device)
policy_net, target_net, optimizer, memory, scheduler = ag.create_agent(settings,device,ag.MEMORY_SIZE)
writer = SummaryWriter()

#Optional loading weights of NN from file
if LOAD_WEIGHTS:
    checkpoint = torch.load(LOAD_WEIGHTS_NAME, map_location=torch.device('cpu'))
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    target_net.load_state_dict(policy_net.state_dict())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('Weights loaded.')

run=True
if SCREEN_INIT:
    screen = gf.run_window(settings)
    gf.update_screen(screen,settings,snake,apple)
epoch=0
while run:
    if (EPOCHS == epoch) and (SCREEN_INIT==False):
        run = False
    epoch+=1
    state = ag.get_screen(settings, snake, apple)
    
    if SCREEN_INIT:
        keys = pygame.key.get_pressed()
        if not keys[pygame.K_LCTRL] and not keys[pygame.K_RCTRL]:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    run=False
            q_prediction = policy_net(state)
            #action = F.softmax(q_prediction, dim=1)
            #action = random.choices([0,1,2,3], weights=q_prediction[0], k=1)
            #action = torch.tensor(action).view(1, 1)
            action = torch.argmax(q_prediction).unsqueeze(0).unsqueeze(0)

        else:
            press=0
            while press==0 and run:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        run=False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RIGHT:
                            action=torch.tensor([[0]])
                            #action = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32)
                            press=1
                        elif event.key == pygame.K_LEFT:
                            action=torch.tensor([[1]])
                            #action = torch.tensor([[0, 1, 0, 0]], dtype=torch.float32)
                            press=1
                        elif event.key == pygame.K_UP:
                            action=torch.tensor([[2]])
                            #action = torch.tensor([[0, 0, 1, 0]], dtype=torch.float32)
                            press=1
                        elif event.key == pygame.K_DOWN:
                            action=torch.tensor([[3]])
                            #action = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)
                            press=1
    else:
        q_prediction = policy_net(state)
        #action = F.softmax(q_rediction, dim=1)
        #action = random.choices([0,1,2,3], weights=q_prediction[0], k=1)
        #action = torch.tensor(action).view(1, 1)
        action=torch.argmax(q_prediction).unsqueeze(0).unsqueeze(0)
        
        

    
    if run:
        gf.check_events(settings,snake,apple,action)
        reward = torch.tensor([snake.reward],device=device)
        if snake.dead:
            writer.add_scalar('Length',len(snake.body),epoch)
            snake,apple = gf.new_game(settings)
        if SCREEN_INIT:
            gf.update_screen(screen,settings,snake,apple)
        next_state = ag.get_screen(settings, snake, apple)
        memory.push(state, action, next_state, reward)
        update = False
        if epoch % GRAD_ACCUMULATION_RATE == 0:
            update = True
        loss1 = ag.optimize_model(memory,device,policy_net,target_net,optimizer,snake, update, scheduler)
        if epoch % SYNC_POLICIES_RATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        if loss1 != None:
            writer.add_scalar('Loss',loss1,epoch)

#saving weights
if SAVE_WEIGHTS:
    torch.save({
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),

                }, SAVE_WEIGHTS_NAME)
    print('Weights saved.')