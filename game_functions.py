import pygame
import random
import torch

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
            if [x+1,y] not in snake.body and [x+1,y]!=[apple.x,apple.y] and x<settings.field_size[0]-1:
                snake.reward = ((x-apple.x)**2+(y-apple.y)**2)**0.5 - ((x+1-apple.x)**2+(y-apple.y)**2)**0.5
                #snake.reward = -1
                snake.body.insert(0,[x+1,y])
                snake.body.pop(-1)
            elif [x+1,y]==[apple.x,apple.y]:
                snake.body.insert(0,[x+1,y])
                if imaginary == False:
                    apple.new_apple(settings,snake)
                snake.reward=10
            elif [x+1,y] in snake.body or x==settings.field_size[0]-1:
                snake.dead=True
                snake.reward=-10
                    
                
        elif action[0][0] == 1:
            if [x-1,y] not in snake.body and [x-1,y]!=[apple.x,apple.y] and x>0:
                snake.reward = ((x-apple.x)**2+(y-apple.y)**2)**0.5 - ((x-1-apple.x)**2+(y-apple.y)**2)**0.5
                #snake.reward = -1 
                snake.body.insert(0,[x-1,y])
                snake.body.pop(-1)
            elif [x-1,y]==[apple.x,apple.y]:
                snake.body.insert(0,[x-1,y])
                if imaginary==False:
                    apple.new_apple(settings,snake)
                snake.reward=10
            elif [x-1,y] in snake.body or x==0:
                snake.dead=True
                snake.reward=-10
                    
        elif action[0][0] == 2:
            if [x,y-1] not in snake.body and [x,y-1]!=[apple.x,apple.y] and y>0:
                snake.reward = ((x-apple.x)**2+(y-apple.y)**2)**0.5 - ((x-apple.x)**2+(y-1-apple.y)**2)**0.5
                #snake.reward = -1
                snake.body.insert(0,[x,y-1])
                snake.body.pop(-1)
            elif [x,y-1]==[apple.x,apple.y]:
                snake.body.insert(0,[x,y-1])
                if imaginary==False:
                    apple.new_apple(settings,snake)
                snake.reward=10
            elif [x,y-1] in snake.body or y==0:
                snake.dead=True
                snake.reward=-10
                    
        elif action[0][0] == 3:
            if [x,y+1] not in snake.body and [x,y+1]!=[apple.x,apple.y] and y<settings.field_size[1]-1:
                snake.reward = ((x-apple.x)**2+(y-apple.y)**2)**0.5 - ((x-apple.x)**2+(y+1-apple.y)**2)**0.5
                #snake.reward = -1
                snake.body.insert(0,[x,y+1])
                snake.body.pop(-1)
            elif [x,y+1]==[apple.x,apple.y]:
                snake.body.insert(0,[x,y+1])
                if imaginary==False:
                    apple.new_apple(settings,snake)
                snake.reward=10
            elif [x,y+1] in snake.body or y==settings.field_size[1]-1:
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
    state = torch.zeros((30, 30, 3))
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
            
        