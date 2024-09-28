import copy
import game_functions as gf
import settings
import torch

settings = settings.Settings()
device = 'cuda:0'

def imaginary_step(snake, apple, current_depth, current_return, action_trajectory):
    current_depth -=1
    m= current_return - 10
    for i in (0,2,1,3):
        imaginary_snake = copy.deepcopy(snake)
        gf.check_events(settings,imaginary_snake,apple,torch.tensor([[i]]).to(device), imaginary=True)
        current_return = current_return + torch.tensor([imaginary_snake.reward],device=device)

        if current_depth >0:
            imaginary_step(imaginary_snake, apple, current_depth, current_return, action_trajectory) 
        
        if current_return > m:
            m = current_return
            action_by_cheat = i 
    action_trajectory = action_trajectory.append(action_by_cheat)
    return action_trajectory
