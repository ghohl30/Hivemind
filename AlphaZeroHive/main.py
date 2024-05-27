import torch

from hive import Game
from model import HiveGNN
from trainer import Trainer

import sys
sys.path.append('..')  # add parent directory to system path

from ComputerPlayers import GNNPlayer  # import function from parent directory file


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = {
    'batch_size': 64,
    'numIters': 500,                                # Total number of training iterations
    'num_simulations': 100,                         # Total number of MCTS simulations to run when deciding on a move to play
    'numEps': 100,                                  # Number of full games (episodes) to run during each iteration
    'numItersForTrainExamplesHistory': 20,
    'epochs': 2,                                    # Number of epochs of training per iteration
    'checkpoint_path': 'latest.pth'                 # location to save latest set of weights
}


# Create a neural network
model = HiveGNN(1, 22*22*7)

# Create a game
player1 = GNNPlayer("Player1", 'w', model)
player2 = GNNPlayer("Player2", 'b', model)
game = Game(player1, player2)

trainer = Trainer(game, model, args)
trainer.learn()