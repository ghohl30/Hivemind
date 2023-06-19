import random

class Game:

    def __init__(self, player1, player2):
        """Initialize a new game."""
        self.player1 = player1
        self.player2 = player2
        self.turn = player1
        self.winner = None
        self.turn_number = 1

        # initialize the board
        self.board = Board(self)
        self.moves = []

        # set active games
        player1.play_game(self)
        player2.play_game(self)

    def switch_turn(self):
        """Switch the turn to the other player."""
        if self.turn == self.player1:
            self.turn = self.player2
        else:
            self.turn = self.player1
        self.turn_number += 1

    def make_move(self, move):
        """Make a move on the board.

        Passes the move to the board, and then updates whos turn it is
        
        Parameters:
        ------------
        move : Move
            The move to be made.
        """
        if self.winner is not None:
            print(self.winner)
            raise ValueError("Game is over")
        
        # for the first for turns check:
        # 1. if the bee has not been placed the move is a placement move
        # 2. if the bee has not been placed within the first 3 moves decline any non-placement moves for the 4th turn
        if self.turn_number < 5:
            bee = self.board.get_bee(move.player)
            if move.piece.position is not None:
                # if bee.position is None:
                if not bee.placed:
                    print("You cannot move before placing the bee")
                    return False
            if self.turn_number == 4:
                if not bee.placed and move.piece.__str__() != "Bee":
                    print("You have to place your bee within the first 4 turns")
                    return False

        res = move.piece.try_move(move)
        if res:
            move.piece.make_move(move)
            self.moves.append(move)
            self.board.generate_board_state()
            # check if game is over
            result = self.board.check_game_over()
            if result is not False:
                self.winner = result
                print("Game Over")
            self.turn = self.player2 if self.turn == self.player1 else self.player1
            print(self.turn)
            if self.turn == self.player1:
                self.turn_number += 1
        
        # print(self.all_legal_moves())
        return res