import torch
from torch.nn import Linear
from torch_geometric.nn import SomeGNNLayer
from torch_geometric.nn import global_mean_pool

class HiveGNN(torch.nn.Module):
    def __init__(self, node_features, num_edge_features, num_actions):
        super(HiveGNN, self).__init__()

        # GNN layers
        self.conv1 = SomeGNNLayer(node_features, 128)  # Adjust the number of output features as needed
        self.conv2 = SomeGNNLayer(128, 128)  # Adjust the number of output features as needed

        # Output layers
        self.fc_v = Linear(128, 1)
        self.fc_p = Linear(128, num_actions)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Pass node features and edge index (and possibly edge attributes) through GNN layers
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)

        # Pooling layer to get a graph-level representation
        x = global_mean_pool(x, data.batch)

        # Pass the graph representation through the output layers
        v = self.fc_v(x)
        p = self.fc_p(x)

        return torch.tanh(v), torch.softmax(p, dim=-1)
    

