import argparse
import numpy as np
import random
import hive
import ComputerPlayers
import matplotlib.pyplot as plt
import time
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

from multiprocessing import Pool

# deactivate warnings
import warnings
warnings.filterwarnings('ignore')

def simulate_game(args):
    player, i, color = args
    player2 = hive.ComputerPlayer("Random", color)
    player.color = "b" if color == "w" else "w"

    game = hive.Game(player, player2) if color == "b" else hive.Game(player2, player)
    game_over = False

    while not game_over:
        game.turn.make_move()
        if game.winner is not None:
            game_over = True
            # print("The winner is {}!".format(game.winner))
            if game.winner.name == player.name:
                return 1, game.turn_number
    return 0, 0

def evaluate_player_multicore(player, n=100):
    if n % 2 != 0:
        raise ValueError("Input must be an even integer")

    # with Pool() as p:
    #     args = [(player, i, "w") if i < n//2 else (player, i, "b") for i in range(n)]
    #     results = p.map(simulate_game, args)
    args = [(player, i, "w") if i < n//2 else (player, i, "b") for i in range(n)]
    results = process_map(simulate_game, args, max_workers=n)

    wins, num_moves = zip(*results)
    return wins, num_moves


def evaluate_player(player, n=100):

    if n % 2 != 0:
        raise ValueError("Input must be an even integer")

    wins = np.zeros(n)
    num_moves = np.zeros(n)
    color = np.zeros(n)
    color[0:n//2] = 1

    pbar = tqdm(total=n, desc="Simulating games", ncols=80)

    # player is white
    for i in range(n//2):
        player2 = hive.ComputerPlayer("Random", "b")

        game = hive.Game(player, player2)
        game_over = False

        while not game_over:
            game.turn.make_move()
            if game.winner is not None:
                game_over = True
                # print("The w inner is {}!".format(game.winner))
                if game.winner.name == player.name:
                    wins[i] = 1
                    num_moves[i] = game.turn_number
        pbar.update(1)

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
                # print("The winner is {}!".format(game.winner))
                if game.winner.name == player.name:
                    wins[i] = 1
                    num_moves[i] = game.turn_number
        pbar.update(1)

    return wins, num_moves

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Evaluate a Hive player.')
    # parser.add_argument('player_type', type=str, help='Name of the player to evaluate')
    # parser.add_argument('--num-games', type=int, default=100, help='Number of games to play')
    # args = parser.parse_args()

    # player = getattr(ComputerPlayers, args.player_type)
    # player = player(args.player_type, "w")
    # wins, num_moves = evaluate_player(player, args.num_games)

    # compare how long it takes to simulate 100 games with and without multiprocessing
    
    player = ComputerPlayers.AgressiveComputer("Agressive", "w")
    start = time.time()
    wins, num_moves = evaluate_player(player, 100)
    end = time.time()
    print("Time taken to simulate 100 games without multiprocessing: {}".format(end-start))

    print("Player {} won {} out of {} games".format("Aggressive", np.sum(wins), 100))
    print("Average number of moves to win: {}".format(np.mean(num_moves)))

    start = time.time()
    wins, num_moves = evaluate_player_multicore(player, 100)
    end = time.time()
    print("Time taken to simulate 100 games with multiprocessing: {}".format(end-start))


    print("Player {} won {} out of {} games".format("Aggressive", np.sum(wins), 100))
    print("Average number of moves to win: {}".format(np.mean(num_moves)))
