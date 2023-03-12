import tkinter as tk
from math import cos, sin, sqrt, radians

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hive")
        self.geometry("900x900")
        self.width = 800
        self.height = 800
        self.canvas = tk.Canvas(self, width=self.width, height=self.height)
        self.canvas.pack()
        self.Label = tk.Label(self, text="Hive")
        self.Label.pack()
        self.x = 0
        self.y = 0
        self.q = 0
        self.r = 0
        self.s = 0

        self.bind('<Motion>', self.motion)
        self.bind('<Button-1>', self.click)

        # set size of the hexagon
        self.size = 30
        self.angle = 60

        # self.hexagon = self.canvas.create_polygon(self.hexagon_corners(self.x, self.y), outline='gray', fill='white', width=2)


        # draw hexagons for q,r,s in range [-k,k]
        k = 3
        for q in range(-k,k+1):
            for r in range(-k,k+1):
                s=q+r
                if q**2 + r**2 + s**2 <= k**2:  
                    self.draw_hexagon(q,r,s)   

        self.draw_hexagon(0,0,0,tag='hover', outline='red', width=3, fill='red')


    
    def hexagon_corners(self, xc, yc):
        coords = []
        for i in range(6):
            x = xc + self.size * cos(radians(self.angle * i))
            y = yc + self.size * sin(radians(self.angle * i))
            coords.append([x,y])
        return coords

    def hex_to_pixel(self, q, r, s):
        x = (3/2)*self.size*q + self.width/2
        y = self.size*sqrt(3)*q/2 + self.size*sqrt(3)*r + self.height/2
        return x, y

    def draw_hexagon(self, q, r, s, tag='hexagon', outline='gray', width=2, fill='white'):
        x,y = self.hex_to_pixel(q,r,s)
        self.canvas.create_polygon(self.hexagon_corners(x,y), outline=outline, fill=fill, width=width, tags=tag)
        self.canvas.create_text(x, y, text=str(q)+","+str(r)+","+str(s), tags=tag)

    @staticmethod
    def cube_round(q,r,s):

        q_diff = abs(q-round(q))
        r_diff = abs(r-round(r))
        s_diff = abs(s-round(s))

        if q_diff > r_diff and q_diff > s_diff:
            q = -round(r)+round(s)
        elif r_diff > s_diff:
            r = -round(q)+round(s)
        else:
            s = round(q)+round(r)
        
        return round(q), round(r), round(s)

    def motion(self, event):
        self.x, self.y = event.x, event.y

        # get position in hex coordinates
        q = (2/3) * (self.x-self.width/2)/self.size
        r = -(1/3) * (self.x-self.width/2)/self.size + (sqrt(3)/3) * (self.y-self.height/2)/self.size
        s =q+r
        q, r, s = self.cube_round(q,r,s)
        if q != self.q or r != self.r or s != self.s:
            self.q, self.r, self.s = q, r, s
            self.canvas.delete('hover')
            self.draw_hexagon(q,r,s,tag='hover', outline='red', width=3, fill='red')

    def click(self, event):
        print(self.q, self.r, self.s)

if __name__ == "__main__":
    app = App()
    app.mainloop()


