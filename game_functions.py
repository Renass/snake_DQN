import pygame
import random
import torch
import copy
from collections import deque
import time

from settings import Settings
settings = Settings()


def run_window(settings):
    pygame.init()
    screen = pygame.display.set_mode((settings.width,settings.height))
    pygame.display.set_caption("Snake")
    return screen



def new_game(settings):
    snake = Snake(settings)
    apple = Apple(settings,snake)
    return snake,apple



    
def check_events(settings,snake,apple,action, imaginary):
    x = snake.body[0][0]
    y = snake.body[0][1]

    if snake.dead==False:
        if action[0][0] == 0:
            if [x+1,y] not in snake.body[:-1] and [x+1,y]!=[apple.x,apple.y] and x<settings.field_size[0]-1:
                snake.reward = ((x-apple.x)**2+(y-apple.y)**2)**0.5 - ((x+1-apple.x)**2+(y-apple.y)**2)**0.5
                #snake.reward = -1
                snake.body.insert(0,[x+1,y])
                snake.body.pop(-1)
            elif [x+1,y]==[apple.x,apple.y]:
                snake.body.insert(0,[x+1,y])
                if imaginary == False:
                    apple.new_apple(settings,snake)
                snake.reward=1.1
            elif [x+1,y] in snake.body[:-1] or x==settings.field_size[0]-1:
                snake.dead=True
                snake.reward=-10
                    
                
        elif action[0][0] == 1:
            if [x-1,y] not in snake.body[:-1] and [x-1,y]!=[apple.x,apple.y] and x>0:
                snake.reward = ((x-apple.x)**2+(y-apple.y)**2)**0.5 - ((x-1-apple.x)**2+(y-apple.y)**2)**0.5
                #snake.reward = -1 
                snake.body.insert(0,[x-1,y])
                snake.body.pop(-1)
            elif [x-1,y]==[apple.x,apple.y]:
                snake.body.insert(0,[x-1,y])
                if imaginary==False:
                    apple.new_apple(settings,snake)
                snake.reward=1.1
            elif [x-1,y] in snake.body[:-1] or x==0:
                snake.dead=True
                snake.reward=-10
                    
        elif action[0][0] == 2:
            if [x,y-1] not in snake.body[:-1] and [x,y-1]!=[apple.x,apple.y] and y>0:
                snake.reward = ((x-apple.x)**2+(y-apple.y)**2)**0.5 - ((x-apple.x)**2+(y-1-apple.y)**2)**0.5
                #snake.reward = -1
                snake.body.insert(0,[x,y-1])
                snake.body.pop(-1)
            elif [x,y-1]==[apple.x,apple.y]:
                snake.body.insert(0,[x,y-1])
                if imaginary==False:
                    apple.new_apple(settings,snake)
                snake.reward=1.1
            elif [x,y-1] in snake.body[:-1] or y==0:
                snake.dead=True
                snake.reward=-10
                    
        elif action[0][0] == 3:
            if [x,y+1] not in snake.body[:-1] and [x,y+1]!=[apple.x,apple.y] and y<settings.field_size[1]-1:
                snake.reward = ((x-apple.x)**2+(y-apple.y)**2)**0.5 - ((x-apple.x)**2+(y+1-apple.y)**2)**0.5
                #snake.reward = -1
                snake.body.insert(0,[x,y+1])
                snake.body.pop(-1)
            elif [x,y+1]==[apple.x,apple.y]:
                snake.body.insert(0,[x,y+1])
                if imaginary==False:
                    apple.new_apple(settings,snake)
                snake.reward=1.1
            elif [x,y+1] in snake.body[:-1] or y==settings.field_size[1]-1:
                snake.dead=True
                snake.reward=-10
        else:
            print('Check events: NO CORRECT ACTION!')
    
                    





def update_screen(screen,settings, snake,apple):
    screen.fill(settings.bg_color)
    pygame.draw.rect(screen,settings.bg_stat_color,(settings.kvadra*settings.field_size[0],0,settings.stat_width,settings.height))
    font = pygame.font.SysFont('serif', 48)
    text_score = font.render(str(len(snake.body)), False, (0, 180, 0))
    #text_dead = font.render('Game Over',False,(0,180,0))
    screen.blit(text_score, (settings.width-100,100))
    #if snake.dead==True:
        #screen.blit(text_dead, (settings.width-250,200))
    num_segments = len(snake.body)
    for i, body in enumerate(snake.body):
        intensity = 1.0 - (i / num_segments)
        color = (int(255 * intensity), int(242 * intensity), 0)
        #pygame.draw.rect(screen,(255,242,0),(settings.kvadra*body[0],settings.kvadra*body[1],settings.kvadra,settings.kvadra))
        pygame.draw.rect(screen, color, 
                         (settings.kvadra * body[0], 
                          settings.kvadra * body[1], 
                          settings.kvadra, settings.kvadra))
    pygame.draw.rect(screen,(0,0,255),(settings.kvadra*snake.body[0][0],settings.kvadra*snake.body[0][1],settings.kvadra,settings.kvadra))
    pygame.draw.rect(screen,(255,0,0),(settings.kvadra*apple.x,settings.kvadra*apple.y,settings.kvadra,settings.kvadra))
    pygame.display.update()
    #pygame.image.save(screen, 'screenshot.bmp')
    


    
class Snake():
    def __init__(self,settings):
        self.reward = 0
        self.dead = False
        self.body=[]
        self.x = random.randint(2,settings.field_size[0]-1)
        self.y = random.randint(2,settings.field_size[1]-1)
        self.body.append([self.x,self.y])
        self.body.append([self.x-1,self.y])
        self.body.append([self.x-2,self.y])
        

        
class Apple():
    def __init__(self,settings,snake):
        self.l=0
        while self.l==0:
            self.x = random.randint(0,settings.field_size[0]-1)
            self.y = random.randint(0,settings.field_size[1]-1)
            if [self.x,self.y] not in snake.body:
                self.l=1
                
    def new_apple(self,settings,snake):
        self.l=0
        while self.l==0:
            self.x = random.randint(0,settings.field_size[0]-1)
            self.y = random.randint(0,settings.field_size[1]-1)
            if [self.x,self.y] not in snake.body:
                self.l=1


def get_state(snake,apple):
    state = torch.zeros((settings.field_size[0], settings.field_size[1], 3))
    #Backgroud filed
    state[:, :, 1] = 100
    #Apple
    state[apple.x, apple.y] = torch.tensor([255, 0, 0]) 

    #Body with gradient intensity
    num_segments = len(snake.body)    
    for i, body_segment in enumerate(snake.body):
        intensity = 1.0 - (i / num_segments)
        color = torch.tensor([255 * intensity, 242 * intensity, 0])
        state[body_segment[0], body_segment[1]] = color
    
    #Snake head
    state[snake.body[0][0], snake.body[0][1]] = torch.tensor([0, 0, 255])
    state = state.permute(2,0,1)
    return state


def imaginary_step(snake, apple, current_depth, max_depth, current_return, settings, device):
    action_by_cheat = None
    current_depth -=1
    max_return = current_return - 100
    #m1= current_return - 100
    action_reward_acsess = []

    for i in (0,2,1,3):
        imaginary_snake = copy.deepcopy(snake)
        check_events(settings,imaginary_snake,apple,torch.tensor([[i]]).to(device), imaginary=True)
        if imaginary_snake.reward > -10:
            current_return1 = current_return + torch.tensor([imaginary_snake.reward],device=device)
            tale_acsess = check_head_to_tail_accessibility(imaginary_snake)
            field_accessible, acsess_part = check_field_acessibility(imaginary_snake)
            if (current_depth >0):
                _, current_return1, tale_acsess = imaginary_step(imaginary_snake, apple, current_depth, max_depth, current_return1, settings=settings, device=device) 
            action_reward_acsess.append((i, current_return1, acsess_part, tale_acsess))
            if tale_acsess:
                if current_return1 > max_return:
                    max_return = current_return1
                    action_by_cheat = i
    if action_by_cheat == None:
        tale_acsess = False
    else:
        tale_acsess = True    

    #if (action_by_cheat == None) and (current_depth==max_depth-1):
    #    print('additional search2: go where more field acsess')
    #    max_acsess=-1
    #    for item in action_reward_acsess:
    #        #print(item)
    #        action, reward, acsess, _ = item
    #        #print(acsess)
    #        if acsess > max_acsess:
    #            max_acsess = acsess
    #            action_by_cheat = action
    #    print(max_acsess)
        #pause_game()

    return action_by_cheat, max_return, tale_acsess


def check_field_acessibility(snake, target_acsess = 1.0):
    visited = [[False for _ in range(settings.field_size[0])] for _ in range(settings.field_size[1])]
    accessible_cells = 0
    total_free_cells = 900 - len(snake.body)
    required_accessible_cells = total_free_cells * target_acsess
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    #head_x, head_y = snake.body[0]
    queue = deque([snake.body[0]])
    #visited[snake.body[0][0]][snake.body[0][1]] = True

    for x,y in snake.body:
        visited[x][y] = True

    
    while queue:
        x, y = queue.popleft()
        accessible_cells += 1
        
        if accessible_cells >= required_accessible_cells:
            return True, target_acsess
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < settings.field_size[0] and 0 <= ny < settings.field_size[1] and not visited[nx][ny]:
                visited[nx][ny] = True
                queue.append((nx, ny))
    return accessible_cells >= required_accessible_cells, accessible_cells/total_free_cells
    

def pause_game():
    paused = True
    print("Game paused. Press SPACE to continue...")
    
    while paused:
        # Check for events
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                # If space bar is pressed, unpause the game
                if event.key == pygame.K_SPACE:
                    paused = False
                    print("Game resumed.")
            # Allow the user to quit the game
            elif event.type == pygame.QUIT:
                pygame.quit()
                quit()

def check_head_to_tail_accessibility(snake):
    """
    This function checks if the snake's head can reach the snake's tail 
    without hitting its own body or other obstacles.
    """
    visited = [[False for _ in range(settings.field_size[0])] for _ in range(settings.field_size[1])]  # 30x30 grid
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Possible directions (up, down, left, right)
    
    head = snake.body[0]  # Snake's head (starting point)
    #print('head', head)
    tail = snake.body[-1]  # Snake's tail (goal point)
    #print('tale', tail)
    
    queue = deque([head])  # Initialize BFS queue with the head position
    visited[head[0]][head[1]] = True  # Mark the head position as visited
    
    # Mark snake's body (excluding the head and tail) as visited to avoid crossing over itself
    for x, y in snake.body[:-1]:  # Exclude the head and tail from being marked as visited
        visited[x][y] = True

    # BFS loop to find if a path exists from head to tail
    while queue:
        x, y = queue.popleft()

        # If we reach the tail, return True
        if [x, y] == tail:
            return True
        
        # Explore the neighboring cells
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < settings.field_size[0] and 0 <= ny < settings.field_size[1] and not visited[nx][ny]:
                visited[nx][ny] = True
                queue.append([nx, ny])
    
    # If we finish BFS without reaching the tail, return False
    return False
