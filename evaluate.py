import argparse
import numpy as np
import random
import hive
import ComputerPlayers
import matplotlib.pyplot as plt


def evaluate_player(player, n=100):

    if n % 2 != 0:
        raise ValueError("Input must be an even integer")

    wins = np.zeros(n)
    num_moves = np.zeros(n)
    color = np.zeros(n)
    color[0:n//2] = 1

    # player is white
    for i in range(n//2):
        player2 = hive.ComputerPlayer("Random", "b")

        game = hive.Game(player, player2)
        game_over = False

        while not game_over:
            game.turn.make_move()
            if game.winner is not None:
                game_over = True
                print("The winner is {}!".format(game.winner))
                if game.winner.name == player.name:
                    wins[i] = 1
                    num_moves[i] = game.turn_number

    # player is black
    for i in range(n//2, n):
        player2 = hive.ComputerPlayer("Random", "w")
        player.color = "b"

        game = hive.Game(player2, player)
        game_over = False

        while not game_over:
            game.turn.make_move()
            if game.winner is not None:
                game_over = True
                print("The winner is {}!".format(game.winner))
                if game.winner.name == player.name:
                    wins[i] = 1
                    num_moves[i] = game.turn_number
    return wins, num_moves

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a Hive player.')
    parser.add_argument('player_type', type=str, help='Name of the player to evaluate')
    parser.add_argument('--num-games', type=int, default=100, help='Number of games to play')
    args = parser.parse_args()

    player = getattr(ComputerPlayers, args.player_type)
    player = player(args.player_type, "w")
    wins, num_moves = evaluate_player(player, args.num_games)


    print("Player {} won {} out of {} games".format(args.player_type, np.sum(wins), args.num_games))
    print("Average number of moves to win: {}".format(np.mean(num_moves)))
