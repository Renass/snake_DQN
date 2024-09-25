import pygame
from settings import Settings
import game_functions as gf
import agent as ag

import torch
import numpy as np
import copy
import h5py
from datetime import datetime
import os

'''
Program for gathering (state-action-reward-next_state) dataset 
This script based on main.py 
'''

SCREEN_INIT = True
DATASET_LENGTH = 10000
#Perform actions by naive solver instead of ai
CHEAT_DEMO = True

if __name__ == '__main__':

    settings = Settings()
    snake,apple = gf.new_game(settings)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    print(device)
    _, _, _, memory, _ = ag.create_agent(settings, device, DATASET_LENGTH)

    run=True
    if SCREEN_INIT:
        screen = gf.run_window(settings)
        gf.update_screen(screen,settings,snake,apple)
    epoch=0
    while run:
        if (DATASET_LENGTH == epoch):
            run = False
        epoch+=1
        state = ag.get_state(snake, apple)
        
        if SCREEN_INIT:
            keys = pygame.key.get_pressed()
            if not keys[pygame.K_LCTRL] and not keys[pygame.K_RCTRL]:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        run=False

            else:
                press=0
                while press==0 and run:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            run=False
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_RIGHT:
                                action=torch.tensor([[0]]).to(device)
                                #action = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32)
                                press=1
                            elif event.key == pygame.K_LEFT:
                                action=torch.tensor([[1]]).to(device)
                                #action = torch.tensor([[0, 1, 0, 0]], dtype=torch.float32)
                                press=1
                            elif event.key == pygame.K_UP:
                                action=torch.tensor([[2]]).to(device)
                                #action = torch.tensor([[0, 0, 1, 0]], dtype=torch.float32)
                                press=1
                            elif event.key == pygame.K_DOWN:
                                action=torch.tensor([[3]]).to(device)
                                #action = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)
                                press=1
                        
        if run:
            #print('Imaginary reward:')
            m = -100
            action_by_cheat = 0
            for i in (0,2,1,3):
                #imaginary snake is simulated changes of snake for every imaginary step
                imaginary_snake = copy.deepcopy(snake)
                gf.check_events(settings,imaginary_snake,apple,torch.tensor([[i]]).to(device), imaginary=True)
                reward = torch.tensor([imaginary_snake.reward],device=device)
                if reward > m:
                    m = reward
                    action_by_cheat = i
                #print(reward)
                next_state = ag.get_state(imaginary_snake, apple)
                memory.push(state, torch.tensor([i]), next_state, reward)

            if CHEAT_DEMO:
                gf.check_events(settings,snake,apple, torch.tensor([[action_by_cheat]]).to(device), imaginary=False)
            else:
                gf.check_events(settings,snake,apple,action, imaginary=False)
            reward = torch.tensor([snake.reward],device=device)
            #print('Real reward: ', reward)
            if snake.dead:
                snake,apple = gf.new_game(settings)
            if SCREEN_INIT:
                gf.update_screen(screen,settings,snake,apple)
            next_state = ag.get_state(snake, apple)
            #memory.push(state, action, next_state, reward)

    #saving weights
    os.makedirs('datasets', exist_ok=True)
    current_time = datetime.now().strftime('%Y_%m_%d_%H:%M')
    filename = os.path.join('datasets', f"dataset_{current_time}.h5")
    states = np.stack([transition[0].numpy() for transition in memory.memory])
    actions = np.array([transition[1].numpy() for transition in memory.memory])
    next_states = np.stack([transition[2].numpy() for transition in memory.memory])
    rewards = np.array([transition[3].cpu().numpy() for transition in memory.memory])
    
    #print(torch.from_numpy(states).shape)
    #print(torch.from_numpy(actions).shape)
    #print(torch.from_numpy(next_states).shape)
    #print(torch.from_numpy(rewards).shape)
    
    
    with h5py.File(filename, 'w') as hdf:
        #Save the states, actions, rewards, and next_states
        hdf.create_dataset('states', data=states)
        hdf.create_dataset('actions', data=actions)
        hdf.create_dataset('rewards', data=rewards)
        hdf.create_dataset('next_states', data=next_states)
    print('Weights saved.')