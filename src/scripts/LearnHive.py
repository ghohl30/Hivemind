import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool, NNConv
from torch.nn import Sequential, ReLU
from torch.nn import ReLU
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data
import numpy as np

from collections import defaultdict
from math import sqrt

import hive

from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

from multiprocessing import Pool

# deactivate warnings
import warnings

class GNNPlayer(hive.Player):

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



class HiveGNN(torch.nn.Module):
    def __init__(self, node_features, edge_features, num_actions):
        super(HiveGNN, self).__init__()

        # GNN layers
        self.conv1 = GCNConv(node_features, 128)
        self.conv2 = GCNConv(128, 128)

        # Activation function
        self.act = ReLU()

        # Output layers
        self.fc_v = Linear(128, 1)
        self.fc_p = Linear(128, num_actions)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # edge_attr = edge_attr.unsqueeze(-1)

        if edge_index.size(0) > 0:
            # Pass node features, edge index, and edge attributes through GNN layers
            x = ReLU()(self.conv1(x, edge_index, edge_attr))
            x = ReLU()(self.conv2(x, edge_index, edge_attr))

            # Pooling layer to get a graph-level representation
            x = global_mean_pool(x, data.batch)
        else:
            # If there are no edges, make x have the correct shape
            x = torch.zeros((1, 128)).to(data.x.device)

        # Pass the graph representation through the output layers
        v = self.fc_v(x)
        p = self.fc_p(x)

        return torch.tanh(v), torch.softmax(p, dim=-1)

    def predict(self, data):
        return self.forward(data)
    
    
class MCTS:
    def __init__(self, nnet):
        self.nnet = nnet
        self.Q = defaultdict(int)      # total value of each state-action pair
        self.N = defaultdict(int)      # total visit count of each state-action pair
        self.P = {}                    # initial policy
        self.c_puct = 1                # exploration constant

    def search(self, game):

        s = game.board.get_current_state()
        if game.winner is not None:
            return -1

        if s not in self.P:
            x, edge_index, edge_attr = game.board.get_graph()
            # get data object and add grah as tensors
            x = torch.tensor(x, dtype=torch.float32)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32) 
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            v, self.P[s] = self.nnet.predict(data)
            return -v

        max_u, best_a = -float("inf"), -1
        for a in game.get_valid_actions():
            u = self.Q[(s,a)] + self.c_puct * self.P[s][0,a] * sqrt(sum(self.N[(s,b)] for b in game.get_valid_actions())) / (1+self.N[(s,a)])
            if u > max_u:
                max_u = u
                best_a = a
        a = best_a

        game.act(a)
        v = self.search(game.clone())

        self.Q[(s,a)] = (self.N[(s,a)] * self.Q[(s,a)] + v) / (self.N[(s,a)]+1)
        self.N[(s,a)] += 1
        return -v

    

def compute_loss(v_pred, p_pred, v_true, p_true):
    # v_pred: predicted value
    # p_pred: predicted policy  
    # v_true: true value from MCTS
    # p_true: true policy from MCTS
    value_loss = ((v_pred - v_true)**2).mean()  # Mean squared error loss
    policy_loss = -(p_true * torch.log(p_pred)).sum(-1).mean()  # Negative log likelihood loss
    return value_loss + policy_loss
    
def policyIterSP(game, nnet, numIters=10, numEps=100, threshold=0.55):
    mcts = MCTS(nnet)
    for i in range(numIters):
        examples = []
        for e in range(numEps):
            examples += executeEpisode(game, mcts)
        new_nnet = trainNNet(examples)
        frac_win = pit(new_nnet, nnet)
        if frac_win > threshold:
            nnet = new_nnet
    return nnet

def executeEpisode(game, mcts, numMCTSSims=2):
    examples = []
    while True:
        for _ in range(numMCTSSims):
            mcts.search(game.clone())
        s = game.board.get_current_state()
        examples.append([s, mcts.P[s], None])

        # run through mask
        mask = game.board.move_mask()
        # flatten mask
        mask = mask.flatten()
        mcts.P[s] = mcts.P[s].detach().numpy()[0] * mask
        # renormalize
        mcts.P[s] = mcts.P[s] / mcts.P[s].sum()
        a = np.random.choice(len(mcts.P[s]), p=mcts.P[s])
        game.act(a)
        if game.winner is not None:
            print("The winner is {}!".format(game.winner))
            examples = assignRewards(examples, game.gameReward())
            return examples
        
def simulate_game(args, nnet, old_nnet):
    i, color = args
    player1 = GNNPlayer("Player1", 'w' , nnet)
    player2 = GNNPlayer("Player2", 'b', old_nnet)
    player1.color = color
    player2.color = "b" if color == "w" else "w"

    game = hive.Game(player1, player2) if color == "b" else hive.Game(player2, player1)
    game_over = False

    while not game_over:
        game.turn.make_move()
        if game.winner is not None:
            game_over = True
            # print("The winner is {}!".format(game.winner))
            if game.winner.name == player1.name:
                return 1, game.turn_number
    return 0, 0

def pit(new_nnet, old_nnet, n=100):
    if n % 2 != 0:
        raise ValueError("Input must be an even integer")

    args = [(i, "w") if i < n//2 else (i, "b") for i in range(n)]
    results = process_map(simulate_game, args, new_nnet, old_nnet, max_workers=n)

    wins, num_moves = zip(*results)
    return wins, num_moves

def assignRewards(examples, reward):
    for example in examples:
        example[-1] = reward
    return examples

def trainNNet(examples, nnet, learning_rate=0.001, batch_size=64, epochs=10):

    print("Training neural network...")
    # Create a data loader
    dataset = torch.utils.data.TensorDataset(
        torch.tensor([x[0] for x in examples], dtype=torch.float32),
        torch.tensor([x[1] for x in examples], dtype=torch.float32),
        torch.tensor([x[2] for x in examples], dtype=torch.float32)
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define the optimizer
    optimizer = torch.optim.Adam(nnet.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        for i, (s, p, v) in enumerate(data_loader):
            optimizer.zero_grad()
            
            # Forward pass
            p_pred, v_pred = nnet(s)
            
            # Compute the loss
            loss = compute_loss(v_pred, p_pred, v, p)
            
            # Backward pass
            loss.backward()
            
            # Update the weights
            optimizer.step()

    return nnet


if __name__ == "__main__":

    # Create a neural network
    nnet = HiveGNN(1, 1, 22*22*7)

    # Create a game
    player1 = GNNPlayer("Player1", 'w', nnet)
    player2 = GNNPlayer("Player2", 'b', nnet)
    game = hive.Game(player1, player2)

    # Train the neural network
    nnet = policyIterSP(game, nnet, numIters=10, numEps=100, threshold=0.55)

    # Save the neural network
    torch.save(nnet.state_dict(), "HiveGNN.pt")