import torch
import numpy as np
from scipy import sparse as sp
from torch_geometric.data import Data, Dataset
from joblib import Parallel, delayed

# def lap_eig_batch(dense_adj_batch, num_nodes_batch):
#     dense_adj_batch = dense_adj_batch.detach().cpu().float()
#     num_batches, num_samples, num_nodes, _ = dense_adj_batch.size()
    
#     in_degree_batch = dense_adj_batch.sum(dim=-1)
#     in_degree_batch = torch.clamp(in_degree_batch, min=1) ** -0.5
#     N = torch.eye(num_nodes, device=dense_adj_batch.device).repeat(num_batches, num_samples, 1, 1) * in_degree_batch.unsqueeze(-1)
    
#     L = torch.eye(num_nodes, device=dense_adj_batch.device).repeat(num_batches, num_samples, 1, 1) - N @ dense_adj_batch @ N
#     EigVal, EigVec = torch.linalg.eigh(L)
#     eigval = torch.sort(torch.abs(EigVal), dim=-1).values.float()
#     eigvec = EigVec.float()
#     return eigvec, eigval

# def get_pe_batch(ADJ, dense_subadj_batch, n_node, half_pos_enc_dim=128):
#     assert ADJ.size(-1) <= half_pos_enc_dim
#     device = ADJ.device
    
#     batch_size = dense_subadj_batch.size(0)
#     num_sample = dense_subadj_batch.size(1)
#     num_nodes = dense_subadj_batch.size(2)
    
#     batch_eigvecs, batch_eigvals = lap_eig_batch(dense_subadj_batch, num_nodes)
#     # print('batch_eigvecs',batch_eigvecs.size()) [150, 20, 7, 7]
    
#     node_pe = torch.zeros(batch_size, n_node, half_pos_enc_dim).to(device)
#     batch_eigvecs = batch_eigvecs.reshape(batch_size, num_sample * num_nodes, -1)
    
#     node_pe[:, :batch_eigvecs.size(1), :batch_eigvecs.size(2)] = batch_eigvecs
    
#     in_degree_ADJ = ADJ.sum(dim=2)
#     eigvec_ADJ, _ = lap_eig_batch(ADJ.unsqueeze(1), ADJ.size(-1))
    
#     node_pe[:, batch_eigvecs.size(1):batch_eigvecs.size(1) + eigvec_ADJ.size(2), :eigvec_ADJ.size(3)] = eigvec_ADJ.squeeze(1)
    
#     return node_pe, batch_eigvecs.view(batch_size, num_sample, num_nodes, -1)

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^0723
# def lap_eig(dense_adj, number_of_nodes, in_degree):
#     device = dense_adj.device
    
#     dense_adj = dense_adj.detach().float().to(device)
#     in_degree = in_degree.detach().float().to(device)

#     A = dense_adj.to(device)
#     N = torch.diag_embed(torch.clamp(in_degree, min=1.0) ** -0.5).to(device)
#     L = torch.eye(number_of_nodes).to(device) - torch.bmm(torch.bmm(N, A), N).to(device)

#     EigVal, EigVec = torch.linalg.eigh(L)
#     eigvec = EigVec.float().to(device)
#     eigval = torch.sort(torch.abs(EigVal)).values.float().to(device)
#     return eigvec, eigval

# def get_pe(t, subadj, ADJ, n_node):
#     device = subadj.device
    
#     batch_size = subadj.size(0)
#     sub_nodes = subadj.size(1)
#     max_nodes = ADJ.size(1)

#     in_degree_sub = subadj.sum(dim=2)
#     eigvec_sub, eigval_sub = lap_eig(subadj, sub_nodes, in_degree_sub)
    
#     node_pe = torch.zeros(batch_size, n_node, max_nodes).to(device)
#     node_pe[:, :sub_nodes, :sub_nodes] = eigvec_sub
    
#     if t > 0:
#         in_degree_ADJ = ADJ.sum(dim=2)
#         eigvec_ADJ, _ = lap_eig(ADJ, max_nodes, in_degree_ADJ)
#         node_pe[:, sub_nodes:, :] = eigvec_ADJ

#     return node_pe
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^0723


# def lap_eig(dense_adj, number_of_nodes, in_degree):
#     device = dense_adj.device
    
#     # Convert only once
#     # dense_adj = dense_adj.float()
#     # in_degree = in_degree.float()

#     A = dense_adj
#     N = torch.diag_embed(torch.clamp(in_degree, min=1.0) ** -0.5)
#     L = torch.eye(number_of_nodes, device=device) - torch.bmm(torch.bmm(N, A), N)

#     EigVal, EigVec = torch.linalg.eigh(L)
#     eigvec = EigVec#.float()
#     eigval = torch.sort(torch.abs(EigVal)).values#.float()
#     return eigvec, eigval

def lap_eig(dense_adj, number_of_nodes, in_degree):
    device = dense_adj.device
    
    dense_adj = dense_adj.detach().float().to(device)
    in_degree = in_degree.detach().float().to(device)

    # 防止度数为零的情况，添加一个小值
    in_degree = torch.clamp(in_degree, min=1e-6)

    A = dense_adj.to(device)
    N = torch.diag_embed(in_degree ** -0.5).to(device)
    
    # 正则化项 epsilon
    epsilon = 1e-5
    L = torch.eye(number_of_nodes).to(device) - torch.bmm(torch.bmm(N, A), N).to(device) + epsilon * torch.eye(number_of_nodes).to(device)

    try:
        EigVal, EigVec = torch.linalg.eigh(L)
    except torch._C._LinAlgError as e:
        print("Eigenvalue decomposition did not converge: ", e)
        # 设置 EigVal 为度数，EigVec 为度矩阵对角线元素的对角矩阵
        EigVal = in_degree
        EigVec = torch.diag(in_degree).to(device)
    
    eigvec = EigVec.float().to(device)
    eigval = torch.sort(torch.abs(EigVal)).values.float().to(device)
    return eigvec, eigval
    
def get_pe(t, subadj, ADJ, n_node):
    device = subadj.device
    
    batch_size = subadj.size(0)
    sub_nodes = subadj.size(1)
    max_nodes = ADJ.size(1)

    in_degree_sub = subadj.sum(dim=2)
    eigvec_sub, eigval_sub = lap_eig(subadj, sub_nodes, in_degree_sub)
    # print('eigvec_sub',eigvec_sub)
    # print('eigval_sub',eigval_sub)
    # exit()
    
    # print('subadj',subadj.size())
    sub_pe = torch.cat((eigvec_sub, subadj), dim=2)
    
    node_pe = torch.zeros(batch_size, n_node, max_nodes*2, device=device)
    node_pe[:, :sub_nodes, :sub_nodes*2] = sub_pe
    
    if t > 0:
        in_degree_ADJ = ADJ.sum(dim=2)
        eigvec_ADJ, _ = lap_eig(ADJ, max_nodes, in_degree_ADJ)
        # print('eigvec_ADJ',eigvec_ADJ.size())
        # print('ADJ',ADJ.size())
        # exit()
        ADJ_pe = torch.cat((eigvec_ADJ, ADJ), dim=2)
        
        node_pe[:, sub_nodes:, :] = ADJ_pe

    return node_pe




# def lap_eig(dense_adj, number_of_nodes, in_degree):
#     dense_adj = dense_adj.detach().cpu().float().numpy()
#     in_degree = in_degree.detach().cpu().float().numpy()

#     A = dense_adj
#     N = np.diag(in_degree.clip(1) ** -0.5)
#     L = np.eye(number_of_nodes) - N @ A @ N

#     EigVal, EigVec = np.linalg.eigh(L)
#     eigvec = torch.from_numpy(EigVec).float()
#     eigval = torch.from_numpy(np.sort(np.abs(np.real(EigVal)))).float()
#     return eigvec, eigval

# def get_pe(t, subadj, ADJ, n_node):
#     device = subadj.device
    
#     batch_size = ADJsubadj.size(0)
#     sub_nodes = subadj.size(1)
#     max_nodes = .size(1)

    
#     batch_eigvecs_sub = []
#     batch_eigvals_sub = []
#     batch_eigvecs_ADJ = []
#     batch_eigvals_ADJ = []

#     for g in range(batch_size):
#         adj = subadj[g]
#         in_degree = adj.sum(dim=1)
#         eigvec, eigval = lap_eig(adj, sub_nodes, in_degree)
#         batch_eigvecs_sub.append(eigvec)
#         batch_eigvals_sub.append(eigval)
#         if t > 0 :
#             adj_max = ADJ[g]
#             in_degree_ADJ = adj_max.sum(dim=1)
#             eigvec_ADJ, eigval_ADJ = lap_eig(adj_max, max_nodes, in_degree_ADJ)
#             batch_eigvecs_ADJ.append(eigvec_ADJ)
#             batch_eigvals_ADJ.append(eigval_ADJ)
    
#     EigVec_sub = torch.stack(batch_eigvecs_sub)#[150, 7, 7]
#     EigVal_sub = torch.stack(batch_eigvals_sub)#[150, 7]

#     if t > 0 :
#         EigVec = torch.stack(batch_eigvecs_ADJ)#[150, 28, 28]
#         EigVal = torch.stack(batch_eigvals_ADJ)#[150, 28]

#     node_pe = torch.zeros(batch_size, n_node, max_nodes).to(device)
#     node_pe[:, :EigVec_sub.size()[1], :EigVec_sub.size()[2]] = EigVec_sub
    
#     if t > 0 :
#         node_pe[:, EigVec_sub.size()[1]:, :] = EigVec
#     return node_pe



def get_pe_2d(ADJ, dense_subadj_batch, edge12_indices: torch.LongTensor, n_node: int, n_edge12: int,
              half_pos_enc_dim=128):
    assert ADJ.size()[1] <= half_pos_enc_dim
    device = edge12_indices.device
    
    batch_size = dense_subadj_batch.size(0)
    num_nodes = dense_subadj_batch.size(1)
    
    batch_eigvecs = []
    batch_eigvals = []

    for g in range(batch_size):
        adj = dense_subadj_batch[g]
        in_degree = adj.sum(dim=1)
        eigvec, eigval = lap_eig(adj, num_nodes, in_degree)
        batch_eigvecs.append(eigvec)
        batch_eigvals.append(eigval)
    
    EigVec = torch.cat(batch_eigvecs, dim=0)
    EigVal = torch.cat(batch_eigvals, dim=0)
    
    node_pe = torch.zeros(n_node, half_pos_enc_dim).to(device)
    node_pe[:EigVec.size()[0], :EigVec.size()[1]] = EigVec
    
    in_degree_ADJ = ADJ.sum(dim=1)
    eigvec_ADJ, eigval_ADJ = lap_eig(ADJ, ADJ.size()[1], in_degree_ADJ)
    node_pe[EigVec.size()[0]:, :eigvec_ADJ.size()[1]] = eigvec_ADJ
    return node_pe.unsqueeze(0), EigVec.unsqueeze(0)


def get_pe_2d_t(ADJ, EigVec, edge12_indices: torch.LongTensor, n_node: int, n_edge12: int,
              half_pos_enc_dim=128):
    assert ADJ.size()[1] <= half_pos_enc_dim
    device = edge12_indices.device
    
    node_pe = torch.zeros(n_node, half_pos_enc_dim).to(device)  # [N, half_pos_enc_dim]
    node_pe[:EigVec.size()[0], :EigVec.size()[1]] = EigVec
    
    in_degree_ADJ = ADJ.sum(dim=1)
    eigvec_ADJ, eigval_ADJ = lap_eig(ADJ, ADJ.size()[1], in_degree_ADJ)
    # print('eigvec_ADJ',eigvec_ADJ.size())#[28, 28]
    # print('eigval_ADJ',eigval_ADJ.size())
    # print('node_pe',node_pe.size())
    node_pe[EigVec.size()[0]:, :eigvec_ADJ.size()[1]] = eigvec_ADJ

    # print('node_pe',node_pe)
    
    # E = edge12_indices.shape[0]
    # print('E',E)
    
    # all_edges_pe = torch.zeros([E, 2 * half_pos_enc_dim]).to(device)
    # all_edges_pe[:n_edge12, :half_pos_enc_dim] = torch.index_select(node_pe, 0, edge12_indices[:n_edge12, 0])
    # all_edges_pe[:n_edge12, half_pos_enc_dim:] = torch.index_select(node_pe, 0, edge12_indices[:n_edge12, 1])

    return node_pe.unsqueeze(0)#all_edges_pe.unsqueeze(0)  # [1, E, 2*half_pos_enc_dim]


def create_self_loop_edge_index(num_nodes):
    # 生成自环边的索引
    arr1 = torch.arange(num_nodes)
    self_loop_edge_index = torch.stack([arr1, arr1], dim=0)
    return self_loop_edge_index

# def reshape_and_preprocess_self_loops(batch_size, num_graphs, num_nodes):
#     N = num_graphs * num_nodes
#     single_self_loop_edge_index = create_self_loop_edge_index(num_nodes)
#     # Adjust indices for all graphs in a batch
#     self_loop_edge_indices = []
#     for i in range(num_graphs):
#         offset = i * num_nodes
#         adjusted_edge_index = single_self_loop_edge_index + offset
#         self_loop_edge_indices.append(adjusted_edge_index)
#     # Concatenate self-loop edge indices for all graphs
#     self_loop_edge_index = torch.cat(self_loop_edge_indices, dim=1)
#     # print('self_loop_edge_index',self_loop_edge_index)
#     # print('self_loop_edge_index',self_loop_edge_index.size())
#     # exit()
#     # Create list of graphs
#     list_graphs = []
#     for i in range(batch_size):
#         graph_data = Data(dense_edge_index=self_loop_edge_index, num_nodes=N)
#         list_graphs.append(graph_data)
#     return list_graphs

# Create dense edge index for one graph [2, num_nodes * (num_nodes - 1)]
def create_dense_edge_index(num_nodes):
    arr1 = torch.arange(num_nodes).unsqueeze(-1).repeat(1, num_nodes)
    arr2 = torch.arange(num_nodes).unsqueeze(0).repeat(num_nodes, 1)
    arr3 = torch.cat((arr1.unsqueeze(-1), arr2.unsqueeze(-1)), dim=-1)
    arr4 = ~torch.eye(num_nodes, dtype=torch.bool)
    dense_edge_index = arr3[arr4].transpose(0, 1)
    return dense_edge_index

def create_max_edge_index(MaxNodeNum):
    # Create self-loop edges for additional nodes
    arr1 = torch.arange(MaxNodeNum)
    max_edge_index = torch.stack([arr1, arr1], dim=0)
    return max_edge_index
    
def reshape_and_preprocess(batch_size, num_graphs, num_nodes, MaxNodeNum):
    # batch_size, num_graphs, num_nodes, _ = adj_matrices.shape
    N = num_graphs * num_nodes + MaxNodeNum
    single_dense_edge_index = create_self_loop_edge_index(num_nodes)
    MaxNodeNum_dense_edge_index = create_max_edge_index(MaxNodeNum)
    
    # Adjust indices for all graphs in a batch
    dense_edge_indices = []
    for i in range(num_graphs):
        offset = i * num_nodes
        adjusted_edge_index = single_dense_edge_index + offset
        dense_edge_indices.append(adjusted_edge_index)
        
    dense_edge_index0 = torch.cat(dense_edge_indices, dim=1)
    # print('dense_edge_index0',dense_edge_index0)
    # print('dense_edge_index0',dense_edge_index0.size())
    
    offset = num_graphs * num_nodes
    adjusted_edge_index = MaxNodeNum_dense_edge_index + offset
    dense_edge_indices.append(adjusted_edge_index)
    # Concatenate dense edge indices for all graphs
    dense_edge_index = torch.cat(dense_edge_indices, dim=1)
    
    # print('dense_edge_index',dense_edge_index)
    # print('dense_edge_index',dense_edge_index.size())
    # exit()

    # Create list of graphs
    list_graphs = []
    for i in range(batch_size):
        graph_data = Data(dense_edge_index=dense_edge_index, num_nodes=N)
        list_graphs.append(graph_data)
    return list_graphs

class CustomSyntheticDataset(Dataset):
    def __init__(self, batch_size, num_graphs, num_nodes, MaxNodeNum):
        super().__init__()
        self.list_graphs = reshape_and_preprocess(batch_size, num_graphs, num_nodes, MaxNodeNum)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def len(self):
        return len(self.list_graphs)

    def get(self, idx):
        item = self.list_graphs[idx]
        item.idx = idx
        return item

class Batch:
    def __init__(self, idx, dense_edge_index, node_num, dense_edge_num):
        super(Batch, self).__init__()
        self.idx = idx
        self.dense_edge_index = dense_edge_index
        self.node_num = node_num
        self.dense_edge_num = dense_edge_num

    def to(self, device):
        self.idx = self.idx.to(device)
        self.dense_edge_index = self.dense_edge_index.to(device)
        return self

    def __len__(self):
        return len(self.node_num)
    
def collator(items):
    #  Data(num_nodes=num_nodes, edge_index=edge_index)
    items = [item for item in items if item is not None]
    node_num = [item.num_nodes for item in items]
    dense_edge_num = [item.dense_edge_index.shape[1] for item in items]
    idxs = tuple([item.idx for item in items])
    # dense
    dense_edge_indices = tuple([item.dense_edge_index for item in items])
    dense_edge_index = torch.cat(dense_edge_indices, dim=1)  # [2, dense_E]
    return Batch(idx=torch.LongTensor(idxs), dense_edge_index=dense_edge_index,
                 node_num=node_num, dense_edge_num=dense_edge_num)
    

# def make_batch_concatenated(node_feature: torch.Tensor, edge_index: torch.LongTensor, edge_feature: torch.Tensor,
#                             n_nodes: list, n_edges: list) -> Batch:
#     """
#     :param node_feature: Tensor([sum(n), Dv])
#     :param edge_index: LongTensor([2, sum(e)])
#     :param edge_feature: Tensor([sum(e), De])
#     :param n_nodes: list
#     :param n_edges: list
#     :parem null_params: dict
#     """
#     assert len(node_feature.size()) == len(edge_index.size()) == len(edge_feature.size()) == 2

#     bsize = len(n_nodes)
#     node_dim = node_feature.size(-1)  # 节点特征的维度
#     edge_dim = edge_feature.size(-1)  # 边特征的维度
#     assert node_dim == edge_dim
#     shared_dim = node_dim   # 共享维度
#     device = node_feature.device  # 张量所在的设备
#     dtype = node_feature.dtype   # 节点特征的数据类型
#     n = node_feature.size(0)  # sum(n)  # 所有图的节点总数
#     e = edge_feature.size(0)  # sum(e)  # 所有图的边总数
#     # unpack nodes
#     idx = torch.arange(max(n_nodes), device=device)
#     idx = idx[None, :].expand(bsize, max(n_nodes))  # [B, N]
#     node_index = torch.arange(max(n_nodes), device=device, dtype=torch.long)
#     node_index = node_index[None, :, None].expand(bsize, max(n_nodes), 2)  # [B, N, 2]
#     node_num_vec = torch.tensor(n_nodes, device=device)[:, None]  # [B, 1]
#     unpacked_node_index = node_index[idx < node_num_vec]  # [N, 2]
#     unpacked_node_feature = node_feature  # [sum(n), Dv]
#     # unpack edges
#     edge_num_vec = torch.tensor(n_edges, device=device)[:, None]  # [B, 1]
#     unpacked_edge_index = edge_index.t()  # [|E|, 2]
#     unpacked_edge_feature = edge_feature  # [sum(e), De]

#     # compose tensor
#     n_edges_ = [n + e for n, e in zip(n_nodes, n_edges)]
#     max_size = max(n_edges_)
#     edge_index_ = torch.zeros(bsize, max_size, 2, device=device, dtype=torch.long)  # [B, N + |E|, 2]
#     edge_feature_ = torch.zeros(bsize, max_size, shared_dim, device=device, dtype=dtype)  # [B, N + |E|, shared_dim]
#     full_index = torch.arange(max_size, device=device)[None, :].expand(bsize, max_size)  # [B, N + |E|]

#     node_mask = full_index < node_num_vec  # [B, N + |E|]
#     edge_mask = (node_num_vec <= full_index) & (full_index < node_num_vec + edge_num_vec)  # [B, N + |E|]
#     edge_index_[node_mask] = unpacked_node_index
#     edge_index_[edge_mask] = unpacked_edge_index
#     edge_feature_[node_mask] = unpacked_node_feature
#     edge_feature_[edge_mask] = unpacked_edge_feature
#     # setup batch
#     return Batch(edge_index_, edge_feature_, n_nodes, n_edges_)




# def lap_eig(dense_adj, number_of_nodes, in_degree):
#     dense_adj = dense_adj.detach().cpu().float().numpy()
#     in_degree = in_degree.detach().cpu().float().numpy()

#     A = dense_adj
#     N = np.diag(in_degree.clip(1) ** -0.5)
#     L = np.eye(number_of_nodes) - N @ A @ N

#     EigVal, EigVec = np.linalg.eigh(L)
#     eigvec = torch.from_numpy(EigVec).float()
#     eigval = torch.from_numpy(np.sort(np.abs(np.real(EigVal)))).float()
#     return eigvec, eigval


# def batch_lap_eig(batch_adj):
#     batch_size, num_graphs, num_nodes, _ = batch_adj.shape
#     batch_eigvecs = torch.zeros(batch_size, num_graphs, num_nodes, num_nodes)
#     batch_eigvals = torch.zeros(batch_size, num_graphs, num_nodes)

#     for i in range(batch_size):
#         for j in range(num_graphs):
#             adj = batch_adj[i, j]
#             in_degree = adj.sum(dim=1)
#             eigvec, eigval = lap_eig(adj, num_nodes, in_degree)
#             batch_eigvecs[i, j] = eigvec
#             batch_eigvals[i, j] = eigval

#     return batch_eigvecs, batch_eigvals


# def batch_get_pe_2d(batch_adj, edge_indices, n_node, half_pos_enc_dim=128):
#     assert n_node <= half_pos_enc_dim
#     batch_size, num_graphs, _, _ = batch_adj.shape
#     device = batch_adj.device

#     batch_eigvecs, batch_eigvals = batch_lap_eig(batch_adj)

#     batch_node_pe = torch.zeros(batch_size, num_graphs, n_node, half_pos_enc_dim).to(device)
#     batch_all_edges_pe = torch.zeros(batch_size, num_graphs, edge_indices.shape[2], 2 * half_pos_enc_dim).to(device)

#     for i in range(batch_size):
#         for j in range(num_graphs):
#             node_pe = torch.zeros(n_node, half_pos_enc_dim).to(device)
#             node_pe[:, :n_node] = batch_eigvecs[i, j]
#             batch_node_pe[i, j] = node_pe

#             edge12_indices = edge_indices[i, j]
#             E = edge12_indices.shape[0]
#             all_edges_pe = torch.zeros([E, 2 * half_pos_enc_dim]).to(device)
#             all_edges_pe[:, :half_pos_enc_dim] = torch.index_select(node_pe, 0, edge12_indices[:, 0])
#             all_edges_pe[:, half_pos_enc_dim:] = torch.index_select(node_pe, 0, edge12_indices[:, 1])
#             batch_all_edges_pe[i, j] = all_edges_pe

#     return batch_node_pe, batch_all_edges_pe
