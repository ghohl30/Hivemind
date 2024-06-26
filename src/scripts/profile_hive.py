# Description: This script profiles the main function of the hive game.
# It is used to identify bottlenecks in the code and optimize it.
# It uses the cProfile module to profile the main function and saves the results to a file.

import sys
 
# setting path
# make sure your in the src/scripts directory (not so clean, open to suggestions)
sys.path.append('../../src')

import cProfile
import pstats
import hive
from ComputerPlayers import AIPlayer
import random

def main():
    R2_D2 = AIPlayer("R2_D2", 'w')
    Optimus_Prime = AIPlayer("Optimus_Prime", 'b')
    game = hive.Game(R2_D2, Optimus_Prime)

    # let the two bots play a game
    while game.winner is None and game.turn_number < 4:
        print(game.turn)
        print(game.board.board_state)
        game.turn.make_move()

if __name__ == "__main__":
    # Profile the main function
    cProfile.run('main()', 'profile_results')

    # Load and print the profiling results
    with open("profile_results.txt", "w") as f:
        p = pstats.Stats('profile_results', stream=f)
        p.sort_stats('cumulative').print_stats()
    
    # Optionally, print to console as well
    p = pstats.Stats('profile_results')
    p.sort_stats('cumulative').print_stats()