from hive import Game, Player
import numpy as np
import random
import torch
from torch_geometric.data import Data
from concurrent.futures import ProcessPoolExecutor, as_completed

def find_move_index(moves, target_move_repr):
    """
    Finds the index of a move in the list of moves based on its string representation.

    Parameters:
    moves (list): List of move objects.
    target_move_repr (str): The string representation of the target move.

    Returns:
    int: The index of the target move in the list, or None if not found.
    """
    for i, move in enumerate(moves):
        if repr(move) == target_move_repr:
            return i
    return None

class AgressiveComputer(Player):

    def __init__(self, name, color, auto=True):
        super().__init__(name, color, auto)

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


class RandomComputer(Player):

    def __init__(self, name, color, auto=True):
        super().__init__(name, color, auto)

    def __repr__(self):
        return self.name
    
    def make_move(self):

        possible_moves = self.active_game.legal_moves

        # choose a random move
        move = random.choice(possible_moves)

        # make the move
        super().make_move(move)

    def make_random_move(self):

        possible_moves = self.active_game.legal_moves

        # choose a random move
        move = random.choice(possible_moves)

        # make the move
        super().make_move(move)

class GNNPlayer(Player):

    def __init__(self, name, color, model):
        super().__init__(name, color)
        self.model = model

    def make_move(self):
        x, edge_index, edge_attr = self.game.get_graph()
        x = torch.tensor(x)
        edge_index = torch.tensor(edge_index)
        edge_attr = torch.tensor(edge_attr)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        v, p = self.model.predict(data)
        # Reshape p to (22, 22, 7)
        p = p.view(-1, 22, 22, 7)

        # run through mask
        p = p * self.game.get_mask()
        # renormalize
        p = p / p.sum()

        # get move
        move = self.game.matrix_to_move(p)
        super().make_move(move)
            

class AIPlayer(Player):
    def __init__(self, name, color, auto=True, depth=3):
        super().__init__(name, color, auto)
        self.depth = depth

    def evaluate(self, game):
        # Simple evaluation function for demonstration
        score = 0
        for piece in game.board.pieces:
            if piece.player.name == self.name and piece.position is not None:
                score += 1  # You can add more sophisticated heuristics here
            if game.turn_number > 2:
                if piece.position is not None:
                    for p in piece.edge:
                        if p is not None and p.__str__() == 'QueenBee' and piece.player.name != self.name:
                            score += 4
        return score

    def minimax(self, game, depth, alpha, beta, maximizing_player):
        if depth == 0 or game.winner is not None:
            return self.evaluate(game), None

        legal_moves = game.all_legal_moves()
        best_move = None

        if maximizing_player:
            max_eval = float('-inf')
            for move in legal_moves:
                clone = game.clone()
                # Adapt move attributes to link to the clone
                i = find_move_index(game.legal_moves, repr(move))
                clonned_move = clone.legal_moves[i]

                clone.make_move(clonned_move)
                eval = self.minimax(clone, depth - 1, alpha, beta, False)[0]
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in legal_moves:
                clone = game.clone()

                # Adapt move attributes to link to the clone
                i = find_move_index(game.legal_moves, repr(move))
                clonned_move = clone.legal_moves[i]

                clone.make_move(clonned_move)
                eval = self.minimax(clone, depth - 1, alpha, beta, True)[0]
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def make_move(self):
        _, best_move = self.minimax(self.active_game, self.depth, float('-inf'), float('inf'), True)
        if best_move:
            self.active_game.make_move(best_move)
        else:
            # Fallback to a random move if no best move found
            super().make_random_move()
