import torch
from torch.nn import Linear, ReLU
from torch_geometric.nn import GCNConv, global_mean_pool

class HiveGNN(torch.nn.Module):
    def __init__(self, node_features, num_actions):
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
        x, edge_index = data.x, data.edge_index

        if edge_index.size(0) > 0:
            # Pass node features and edge index through GNN layers
            x = ReLU()(self.conv1(x, edge_index))
            x = ReLU()(self.conv2(x, edge_index))

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