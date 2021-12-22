{"metadata":{"language_info":{"name":"python","version":"3.6.6","mimetype":"text/x-python","codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3","nbconvert_exporter":"python","file_extension":".py"},"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"# %% [code]\nimport random\n\ndef new_game(settings):\n    snake = Snake(settings)\n    apple = Apple(settings,snake)\n    return snake,apple\n\n\n\n    \ndef check_events(settings,snake,apple,action):\n    x = snake.body[0][0]\n    y = snake.body[0][1]\n\n    if snake.dead==False:\n        if action[0][0] == 0:\n            if [x+1,y] not in snake.body and [x+1,y]!=[apple.x,apple.y] and x<settings.field_size[0]-1:\n                snake.reward = ((x-apple.x)**2+(y-apple.y)**2)**0.5 - ((x+1-apple.x)**2+(y-apple.y)**2)**0.5\n                snake.body.insert(0,[x+1,y])\n                snake.body.pop(-1)\n            elif [x+1,y]==[apple.x,apple.y]:\n                snake.body.insert(0,[x+1,y])\n                apple.new_apple(settings,snake)\n                snake.reward=10\n            elif [x+1,y] in snake.body or x==settings.field_size[0]-1:\n                snake.dead=True\n                snake.reward=-10\n                    \n                \n        if action[0][0] == 1:\n            if [x-1,y] not in snake.body and [x-1,y]!=[apple.x,apple.y] and x>0:\n                snake.reward = ((x-apple.x)**2+(y-apple.y)**2)**0.5 - ((x-1-apple.x)**2+(y-apple.y)**2)**0.5\n                snake.body.insert(0,[x-1,y])\n                snake.body.pop(-1)\n            elif [x-1,y]==[apple.x,apple.y]:\n                snake.body.insert(0,[x-1,y])\n                apple.new_apple(settings,snake)\n                snake.reward=10\n            elif [x-1,y] in snake.body or x==0:\n                snake.dead=True\n                snake.reward=-10\n                    \n        if action[0][0] == 2:\n            if [x,y-1] not in snake.body and [x,y-1]!=[apple.x,apple.y] and y>0:\n                snake.reward = ((x-apple.x)**2+(y-apple.y)**2)**0.5 - ((x-apple.x)**2+(y-1-apple.y)**2)**0.5\n                snake.body.insert(0,[x,y-1])\n                snake.body.pop(-1)\n            elif [x,y-1]==[apple.x,apple.y]:\n                snake.body.insert(0,[x,y-1])\n                apple.new_apple(settings,snake)\n                snake.reward=10\n            elif [x,y-1] in snake.body or y==0:\n                snake.dead=True\n                snake.reward=-10\n                    \n        if action[0][0] == 3:\n            if [x,y+1] not in snake.body and [x,y+1]!=[apple.x,apple.y] and y<settings.field_size[1]-1:\n                snake.reward = ((x-apple.x)**2+(y-apple.y)**2)**0.5 - ((x-apple.x)**2+(y+1-apple.y)**2)**0.5\n                snake.body.insert(0,[x,y+1])\n                snake.body.pop(-1)\n            elif [x,y+1]==[apple.x,apple.y]:\n                snake.body.insert(0,[x,y+1])\n                apple.new_apple(settings,snake)\n                snake.reward=10\n            elif [x,y+1] in snake.body or y==settings.field_size[1]-1:\n                snake.dead=True\n                snake.reward=-10\n                    \n\n\n\n    \nclass Snake():\n    def __init__(self,settings):\n        self.reward = 0\n        self.dead = False\n        self.body=[]\n        self.x = random.randint(2,settings.field_size[0]-1)\n        self.y = random.randint(2,settings.field_size[1]-1)\n        self.body.append([self.x,self.y])\n        self.body.append([self.x-1,self.y])\n        self.body.append([self.x-2,self.y])\n        \n\n        \nclass Apple():\n    def __init__(self,settings,snake):\n        self.l=0\n        while self.l==0:\n            self.x = random.randint(0,settings.field_size[0]-1)\n            self.y = random.randint(0,settings.field_size[1]-1)\n            if [self.x,self.y] not in snake.body:\n                self.l=1\n                \n    def new_apple(self,settings,snake):\n        self.l=0\n        while self.l==0:\n            self.x = random.randint(0,settings.field_size[0]-1)\n            self.y = random.randint(0,settings.field_size[1]-1)\n            if [self.x,self.y] not in snake.body:\n                self.l=1\n            \n        ","metadata":{"collapsed":false,"_kg_hide-input":false,"jupyter":{"outputs_hidden":false}},"execution_count":null,"outputs":[]}]}