# %% [code]
class Settings():
    def __init__(self):
        self.field_size = (30,30)
        self.kvadra = 20                                                #How big single field square in pixels 
        self.stat_width = 300                                           # Width of statistics field in pixels
        self.width = self.field_size[0]*self.kvadra+self.stat_width     #total width in pixels
        self.height = self.field_size[1]*self.kvadra                    #total height in pixels
        self.bg_color = (0,100,0)
        self.bg_stat_color = (30,30,30)
