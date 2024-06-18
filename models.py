import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import mul
from torch_sparse import sum as sparsesum
from torch_sparse import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    
    
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, int(hidden_dim/2))
        self.conv2 = GCNConv(int(hidden_dim/2), output_dim)
        self.fc3_6_class = nn.Linear(output_dim, 6)
        self.fc3_5_class = nn.Linear(output_dim, 5)

    def forward(self, data, edge_SF_num):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x_SF = self.fc3_6_class(x)
        x_Ptx = self.fc3_5_class(x)
        return x_SF, x_Ptx
    
    
class DirGCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha):
        super(DirGCNConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lin_src_to_dst = nn.Linear(input_dim, output_dim)
        self.lin_dst_to_src = nn.Linear(input_dim, output_dim)
        self.alpha = alpha
        self.adj_norm, self.adj_t_norm = None, None
        
        self.fc3_6_class = nn.Linear(output_dim, 6)
        self.fc3_5_class = nn.Linear(output_dim, 5)

    def forward(self, data, edge_SF_num):
        x, edge_index = data.x, data.edge_index
        if self.adj_norm is None:
            row, col = edge_index
            num_nodes = x.shape[0]
            
            adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            self.adj_norm = get_norm_adj(adj, norm="dir")
            adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
            self.adj_t_norm = get_norm_adj(adj_t, norm="dir")
        
        y = self.alpha * self.lin_src_to_dst(self.adj_norm @ x) + (1 - self.alpha) * self.lin_dst_to_src(
            self.adj_t_norm @ x)
        x_SF = self.fc3_6_class(y)
        x_Ptx = self.fc3_5_class(y)
        
        return x_SF, x_Ptx
    
class GAT(torch.nn.Module):
    def __init__(self, features, hidden, classes, heads):
        super(GAT, self).__init__()
        self.gat1 = GATConv(features, hidden, heads)
        self.gat2 = GATConv(hidden*heads, classes)

    def forward(self, data, edge_SF_num):
        x, edge_index = data.x, data.edge_index

        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gat2(x, edge_index)
        
        x_SF = F.softmax(x[:, 0:edge_SF_num], dim=1)
        x_Ptx = F.softmax(x[:, edge_SF_num:], dim=1)
        return x_SF, x_Ptx
    
    
class GraphSAGE(torch.nn.Module):

    def __init__(self, feature, hidden, classes):
        super(GraphSAGE, self).__init__()
        self.sage1 = SAGEConv(feature, hidden)
        self.sage2 = SAGEConv(hidden, classes)

    def forward(self, data, edge_SF_num):
        x, edge_index = data.x, data.edge_index

        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.sage2(x, edge_index)
        
        x_SF = F.log_softmax(x[:, 0:edge_SF_num], dim=1)
        x_Ptx = F.log_softmax(x[:, edge_SF_num:], dim=1)

        return x_SF, x_Ptx

    
    
class DNN(nn.Module):
    def __init__(self, input_dim, output_dim_SF_class, output_dim_Ptx_class):
        super(DNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim_SF_class = output_dim_SF_class
        self.output_dim_Ptx_class = output_dim_Ptx_class
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        #self.fc3 = nn.Linear(128, 64)
        self.fc3_6_class = nn.Linear(64, input_dim*output_dim_SF_class)
        self.fc3_5_class = nn.Linear(64, input_dim*output_dim_Ptx_class)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        #x = torch.relu(self.fc3(x))
        
        output_6_class = torch.relu(self.fc3_6_class(x))
        output_5_class = torch.relu(self.fc3_5_class(x))
        
        output_6_class = output_6_class.view(-1, self.input_dim, 6)
        output_5_class = output_5_class.view(-1, self.input_dim, 5)
        
        return output_6_class, output_5_class
    
class DNN_single(nn.Module):
    def __init__(self, input_dim, output_dim_class):
        super(DNN_single, self).__init__()
        self.input_dim = input_dim
        self.output_dim_class = output_dim_class
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3_class = nn.Linear(64, input_dim*output_dim_class)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        output_class = torch.relu(self.fc3_class(x))
        output_class = output_class.view(-1, self.input_dim, self.output_dim_class)
        
        return output_class
    
class TCNN(nn.Module):
    def __init__(self, output_dim_SF_class, output_dim_Ptx_class):
        super(TCNN, self).__init__()
        self.output_dim_SF_class = output_dim_SF_class
        self.output_dim_Ptx_class = output_dim_Ptx_class
        self.fc1 = nn.Linear(1, 16)
        #self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 8)
        
        self.fc4 = nn.Linear(9, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 24)
        
        self.fc3_6_class = nn.Linear(32, output_dim_SF_class)
        self.fc3_5_class = nn.Linear(32, output_dim_Ptx_class)
    
    def forward(self, target, condition):
        x1 = torch.relu(self.fc1(target))
        #x1 = torch.relu(self.fc2(x1))
        x1 = torch.relu(self.fc3(x1))
        
        x2 = torch.relu(self.fc4(condition))
        x2 = torch.relu(self.fc5(x2))
        x2 = torch.relu(self.fc6(x2))
        
        combiner = torch.cat((x1, x2), dim=1)
        
        output_6_class = torch.relu(self.fc3_6_class(combiner))
        output_5_class = torch.relu(self.fc3_5_class(combiner))

        return output_6_class, output_5_class
    
def get_norm_adj(adj, norm):
    if norm == "sym":
        return gcn_norm(adj, add_self_loops=False)
    elif norm == "row":
        return row_norm(adj)
    elif norm == "dir":
        return directed_norm(adj)
    else:
        raise ValueError(f"{norm} normalization is not supported")
        
def row_norm(adj):
    """
    Applies the row-wise normalization:
        \mathbf{D}_{out}^{-1} \mathbf{A}
    """
    row_sum = sparsesum(adj, dim=1)
    return mul(adj, 1 / row_sum.view(-1, 1))


def directed_norm(adj):
    """
    Applies the normalization for directed graphs:
        \mathbf{D}_{out}^{-1/2} \mathbf{A} \mathbf{D}_{in}^{-1/2}.
    """
    in_deg = sparsesum(adj, dim=0)
    in_deg_inv_sqrt = in_deg.pow_(-0.5)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)

    out_deg = sparsesum(adj, dim=1)
    out_deg_inv_sqrt = out_deg.pow_(-0.5)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)

    adj = mul(adj, out_deg_inv_sqrt.view(-1, 1))
    adj = mul(adj, in_deg_inv_sqrt.view(1, -1))
    return adj










