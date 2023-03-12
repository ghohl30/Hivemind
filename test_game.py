import unittest
import hive

class TestGame(unittest.TestCase):

    def build_game(self): 

        Gregor = hive.HumanPlayer("Gregor")
        Wilke = hive.HumanPlayer("Wilke")
        game = hive.Game(Gregor, Wilke)

        return game
    
    def get_piece(self, game, piece_type, player, position=None):
        if position is None: 
            piece = game.board.get_piece(None, player, piece_type)
        else:
            piece = game.board.get_piece(position, player, piece_type)
        return piece
    
    def test_place_a_piece(self):
        game = self.build_game()
        piece = self.get_piece(game, 'Bee', game.player1)
        firstMove = hive.Move(game.player1, piece, [0,0,0])
        game.make_move(firstMove)

        self.assertEqual(game.board.board_state, [['Bee(Gregor, [0, 0, 0])', [0, 0, 0], [None for i in range(6)]]])

    def test_beetle_mount_unmount(self):
        """Test if the beetle can mount and unmount in the next move."""
        game = self.build_game()

        whiteBeetle1 = self.get_piece(game, 'Beetle', game.player1)
        move = hive.Move(game.player1, whiteBeetle1, [0,0,0])
        game.make_move(move)

        blackBeetle1 = self.get_piece(game, 'Beetle', game.player2)
        move = hive.Move(game.player2, blackBeetle1, [1,0,-1])
        game.make_move(move)

        whiteBee = self.get_piece(game, 'Bee', game.player1)
        move = hive.Move(game.player1, whiteBee, [-1,0,1])
        game.make_move(move)

        blackBee = self.get_piece(game, 'Bee', game.player2)
        move = hive.Move(game.player2, blackBee, [2,0,-2])
        game.make_move(move)

        whiteBeetle2 = self.get_piece(game, 'Beetle', game.player1)
        move = hive.Move(game.player1, whiteBeetle2, [-2,0,2])
        game.make_move(move)

        blackAnt1 = self.get_piece(game, 'Ant', game.player2)
        move = hive.Move(game.player2, blackAnt1, [3,0,-3])
        game.make_move(move)

        move = hive.Move(game.player1, whiteBeetle2, [-1,0,1])
        game.make_move(move)

        move = hive.Move(game.player1, blackAnt1, [2,-1,-3])
        game.make_move(move)

        move = hive.Move(game.player1, whiteBeetle2, [-2,0,2])
        game.make_move(move)

        state = []
        state.append(['Beetle(Gregor, [-2, 0, 2])', [-2, 0, 2], [None, None, None, None, None, whiteBee]])
        state.append(['Bee(Gregor, [-1, 0, 1])', [-1, 0, 1], [whiteBeetle1, None, None, None, None, whiteBeetle1]])
        state.append(['Beetle(Gregor, [0, 0, 0])', [0, 0, 0], [whiteBee, None, None, None, None, blackBeetle1]])
        state.append(['Bee(Wilke, [1, 0, -1])', [1, 0, -1], [whiteBeetle1, None, None, None, None, blackBee]])
        state.append(['Ant(Wilke, [2, -1, -3])', [2, -1, -3], [None, blackBee, None, None, None, None]])
        #self.assertEqual(game.board.board_state, state)


if __name__ == '__main__':
    unittest.main()