import tkinter as tk
from math import cos, sin, sqrt, radians
import hive
from ComputerPlayers import AgressiveComputer, RandomComputer, AIPlayer
from tkinter import messagebox

class App(tk.Tk):
    """
    Not an expert in GUI with python, so I´m not sure if this is the best way to do this.
    View this more as a proof of concept.
    """

    def __init__(self):
        """Initialize the application."""
        super().__init__()
        self.title("Hive")
        self.geometry("700x800")
        self.width = 700
        self.height = 700
        self.images = []
        self.canvas = tk.Canvas(self,background="green" ,width=self.width, height=self.height)
        # add padiing to canvas
        self.canvas.pack(padx=10, pady=10)

        # initialize coordinates
        self.x, self.y = self.width/2, self.height/2
        self.q, self.s, self.r = 0, 0, 0

        # initialize the click counter
        self._clicknum = 0

        # set size of the hexagon
        self.size = 30
        self.angle = 60

        # draw initially highlighted hexagon
        self.draw_hexagon(0,0,0,tag='hover', outline='red', width=3, fill='red')

        # set up mouse click event
        self.canvas.bind('<Motion>', self.motion)

        # set up hive
        # self.setup_hive()
        self.setup_computer_game()

        # set up for movement
        self.setup_move()   

        # render the board
        self.rerender_board() 

    @property
    def clicknum(self):
        """Returns the number of clicks."""
        return self._clicknum
    
    @clicknum.setter
    def clicknum(self, value):
        """Sets the number of clicks."""
        self._clicknum = value
        if value == 0:
            self.canvas.delete('possible_moves')
        print("clicknum: {}".format(self._clicknum))

    def setup_hive(self):
        """Sets up a game. Eventually this would of course be done by the user and there would be some kind of lobby."""
        self.player1 = hive.HumanPlayer("Gregor", "w")
        self.player2 = hive.HumanPlayer("Wilke", "b")
        self.game = hive.Game(self.player1, self.player2)
        self.current_player = self.player1

    def setup_computer_game(self):
        self.player1 = hive.HumanPlayer("Gregor", "w")
        self.player2 = AIPlayer("HiveBot", "b")
        self.game = hive.Game(self.player1, self.player2)
        self.current_player = self.player1

    def setup_move(self):
        """Sets up initial values necessary for how moves are made.
        Also binds the left mouse click to the click function.

        #### Attributes
        - `clicknum`: 
        - `place_piece`: Boolean value that determines if a piece is being placed or moved.
        """
        self.clicknum = 0
        self.place_piece = False
        self.move_from = None
        self.move_to = None
        self.canvas.bind('<Button-1>', self.click)
        self.button_holder = tk.Frame(self)
        self.place_buttons = []

    def get_color(self):
        """Returns the color of the current player."""
        return "white" if self.current_player == self.player1 else "black"

    def placement_mode(self, piece_type):
        """Sets up the game for placing a piece."""
        self.place_piece = True
        self.piece = self.game.board.get_piece(None, self.current_player, piece_type)
        self.clicknum = 1
        self.highlight_possible_moves()

    def hex_to_pixel(self, q, s, r):
        """Converts hexagon coordinates to pixel coordinates."""
        x = (3/2)*self.size*q + self.width/2
        y = self.size*sqrt(3)*q/2 + self.size*sqrt(3)*r + self.height/2
        return x, y

    def pixel_to_hex(self, x, y):
        """Converts pixel coordinates to hexagon coordinates."""
        q = (x - self.width/2)*(2/3)/self.size
        r = -(1/3)*(x - self.width/2)/self.size + (y - self.height/2) * (sqrt(3)/3)/self.size
        s = q + r
        return q, s, r

    def hexagon_corners(self, xc, yc):
        """Returns the coordinates of the corners of a hexagon. Where the center is (xc, yc)."""
        coords = []
        for i in range(6):
            x = xc + self.size * cos(radians(self.angle * i))
            y = yc + self.size * sin(radians(self.angle * i))
            coords.append([x,y])
        return coords

    def draw_hexagon(self, q, s, r, tag='hexagon', outline='gray', width=2, fill='white', piece = None, mounting=False):
        """Draws a hexagon at the given coordinates."""
        x,y = self.hex_to_pixel(q,s,r)
        textfill = "green" if not mounting else "red"
        if mounting:
            outline = "red"
        txt = str(piece)
        self.canvas.create_polygon(self.hexagon_corners(x,y), outline=outline, fill=fill, width=width, tags=tag)
        if piece is None:
            self.canvas.create_text(x, y, text=str(q)+","+str(s)+","+str(r), fill="green", tags=tag)
        else:
            # self.canvas.create_text(x, y, text=txt, fill=textfill, tags=tag)
            image = tk.PhotoImage(file="Media/{}.png".format(piece))
            self.images.append(image)
            self.canvas.create_image(x, y, image=image, tags=tag)

    def highlight_possible_moves(self):
        possible_moves = self.game.all_legal_moves()
        for move in possible_moves:
            if move.piece == self.piece:
                q, s, r = move.position
                self.draw_hexagon(q, s, r, tag='possible_moves', outline='red', width=3, fill='')

    def motion(self, event):
        """Tracks mouse movement and updates the highlighted hexagon to show the user where they are hovering."""
        self.x, self.y = event.x, event.y

        # get q,r,s from pixel coordinates
        q, s, r = self.pixel_to_hex(self.x, self.y)
        # round q,r,s to correct hexagon
        q, s, r = self.cube_round(q, s, r)

        # if coordinates don´t match currently highlighted hexagon, change it
        if self.q != q or self.r != r or self.s != s:
            self.canvas.delete('hover')
            self.draw_hexagon(q, s, r, tag='hover', outline='red', width=3, fill='red')
            self.q, self.s, self.r = q, s, r

    def click(self, event):
        """Based on the number of clicks, this method determines what the user wants to do.
        
        on the first click (clicknum = 0) moving piece is selected
        on the second click (clicknum = 1) the piece is moved to the selected hexagon or placed there
        """
        if self.clicknum == 0:
            self.piece = self.game.board.get_piece([self.q, self.s, self.r])
            if self.piece is not None and self.piece.player == self.current_player:
                self.move_from = [self.q, self.s, self.r]
                self.clicknum += 1
                self.highlight_possible_moves()
        elif self.clicknum == 1:
            self.move_to = [self.q, self.s, self.r]
            self.clicknum = 0
            move = hive.Move(self.current_player, self.piece, self.move_to)
            print(self.piece)
            if self.game.make_move(move):
                q, s, r = self.move_to
                self.change_turn()
            else: 
                print("invalid move")

    def show_winner(self):
        """Show a message box announcing the winner."""
        messagebox.showinfo("Game Over", f"Winner: {self.game.winner}")

    def change_turn(self):
        """After a turn everything is set up for the next player."""
        
        self.current_player = self.game.turn

        self.clicknum = 0
        self.place_piece = False
        self.rerender_board()

        # Check if the game is over
        if isinstance(self.game.winner, hive.Player):
            self.after(100, self.show_winner)
            print(f"Game Over! Winner: {self.game.winner}")
            return

        # if the current player object is an instance of ComputerPlayer, make the computer move
        if self.current_player.computer_player:
            print("Computer to move")
            self.current_player.make_move()
            self.change_turn()

    def rerender_board(self):
        """Everything is deleted and the board is redrawn."""
        self.canvas.delete('all')
        self.images = []
        pieces = self.game.board.pieces
        for piece in pieces:
            if piece.position is not None:
                q, s, r = piece.position
                color = "white" if piece.player == self.player1 else "#8CC0ED"
                self.draw_hexagon(q, s, r, tag="{},{},{}".format(q,s,r), outline='gray', width=2, fill=color, piece=piece.__str__(), mounting=hasattr(piece, 'mounting') and piece.mounting)

        # delete the "place piece" buttons
        # for button in self.place_buttons:
        #     button.destroy()
        self.button_holder.destroy()
        self.button_holder = tk.Frame(self)
        
        piece_types = set([piece.__str__() for piece in self.game.board.pieces if piece.position is None and piece.mounted == False and piece.player == self.current_player])
        for piece_type in piece_types:
            button = tk.Button(self.button_holder, text=piece_type, command=lambda type=piece_type: self.placement_mode(type))
            button.pack(side=tk.LEFT)
            self.place_buttons.append(button)
        self.button_holder.pack()
        
    @staticmethod
    def cube_round(q,s,r):
        """Rounds a cube coordinate to the nearest hexagon."""
        q_round, r_round, s_round = int(round(q)), int(round(r)), int(round(s))

        q_diff = abs(q_round - q)
        r_diff = abs(r_round - r)
        s_diff = abs(s_round - s)

        if q_diff > r_diff and q_diff > s_diff:
            q_round = -r_round + s_round
        elif r_diff > s_diff:
            r_round = -q_round + s_round
        else:
            s_round = q_round + r_round

        return q_round, s_round, r_round


if __name__ == '__main__':
    app = App()
    app.mainloop()


    