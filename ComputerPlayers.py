from hive import Game, Player
import numpy as np
import random

class AgressiveComputer(Player):

    def make_random_move(self):

        possible_moves = self.active_game.legal_moves

        # choose a random move
        move = random.choice(possible_moves)

        # make the move
        super().make_move(move)
    
    def make_move(self):

        # matrices 
        A, M, firstPiece = self.active_game.board.get_matrices()

        mask = self.active_game.board.move_mask()

        # find col index of opponent queen
        if self.color == 'w':
            queen_col = self.active_game.board.labels.index('bQ1')
        else:
            queen_col = self.active_game.board.labels.index('wQ1')

        # find the first non zero entry in the queen's column in our 22x22x7 mask (XxYxZ)
        # queen_col is the Y index

        # get an array withs 1s for pieces not alread next to the queen
        non_attacking = np.array([1 if x == 0 else 0 for x in A[queen_col,:]]).reshape(-1,1)

        aggressive_mask = mask[:, queen_col, :]

        # multiply the mask by the non_attacking array
        aggressive_mask = aggressive_mask * non_attacking
        
        # Get the indices of the first non-zero entry
        non_zero_indices = np.nonzero(aggressive_mask)

        if non_zero_indices[0].size > 0:  # Check if there is any non-zero entry
            x, z = non_zero_indices[0][0], non_zero_indices[1][0]
            first_non_zero_entry = (x, queen_col, z)
            # get zeros matrix of same size as mask
            move_matrix = np.zeros_like(mask)
            # set the first non zero entry to 1
            move_matrix[first_non_zero_entry] = 1
            # get the move
            move = self.active_game.matrix_to_move(move_matrix)
            super().make_move(move)
        else:
            self.make_random_move()
            

