# model_new_0320.py
import Aggregation
import dgl
from util import *
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm
import torch.nn.functional as F
import torch.nn.init as init
from sys import exit
from torch.nn.parameter import Parameter
import torch.nn as nn
from torch_geometric.nn.dense import DenseGCNConv    
import random
from a_utils import *
import math
import torch
from torch.nn.utils.rnn import pad_sequence


class GCNLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        x = self.linear(x)
        x = torch.bmm(adj, x)  # 批量矩阵乘法
        return F.relu(x)


class MultiLayerAttention(nn.Module):
    def __init__(self, dim_hidden, num_heads, num_layers):
        super(MultiLayerAttention, self).__init__()
        self.layers = nn.ModuleList([nn.MultiheadAttention(dim_hidden, num_heads) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim_hidden) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, B, C, D):
        # B, C, D 均要求形状为 [seq_len, batch, dim_hidden]
        for i in range(self.num_layers):
            attn_output, _ = self.layers[i](B, C, D)
            B = self.norms[i](B + attn_output)  # 残差连接 + LayerNorm
        return B, _


class EnhancedBondDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, sub_graph_node_num: int, dropout=0.1):
        super(EnhancedBondDecoder, self).__init__()
        self.sub_graph_node_num = sub_graph_node_num
        
        # 更深层次的解码器网络
        layers = [input_dim, hidden_dim, hidden_dim, hidden_dim, (sub_graph_node_num - 1) * sub_graph_node_num // 2]
        
        self.layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1], dtype=torch.float32) for i in range(len(layers) - 1)]
        )
        self.norm_layers = nn.ModuleList(
            [nn.LayerNorm(layers[i + 1], elementwise_affine=True) for i in range(len(layers) - 2)]
        )
        
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(len(layers) - 2)]
        )
        
        # 残差连接：当前后层尺寸相同时使用残差
        self.residual_connections = []
        for i in range(1, len(layers) - 1):
            self.residual_connections.append(layers[i-1] == layers[i])

        # 预计算下三角索引，避免重复计算
        self.register_buffer('tril_indices', torch.tril_indices(sub_graph_node_num, sub_graph_node_num, -1))

    def forward(self, in_tensor: torch.Tensor, activation=nn.LeakyReLU(0.01)) -> torch.Tensor:
        # 输入要求为 [batch, seq, feat]
        if in_tensor.dim() == 2:
            in_tensor = in_tensor.unsqueeze(0)  # 添加批次维度
        assert in_tensor.dim() == 3, "EnhancedBondDecoder: 输入张量维度必须为3"
        batch_size, seq_len, feat_dim = in_tensor.shape
        device = in_tensor.device
        
        # 展平处理以加速批处理
        x = in_tensor.reshape(-1, feat_dim)
        
        for i in range(len(self.layers)):
            if i > 0 and i < len(self.layers) - 1 and self.residual_connections[i-1]:
                residual = x
            x = self.layers[i](x)
            if i != len(self.layers) - 1:
                x = activation(x)
                x = self.norm_layers[i](x)
                x = self.dropouts[i](x)
                if i > 0 and self.residual_connections[i-1]:
                    x = x + residual
        
        # 恢复原始维度
        x = x.view(batch_size, seq_len, -1)
        
        # 创建邻接矩阵：期望输出形状 [batch, seq, sub_graph_node_num, sub_graph_node_num]
        temp_ADJ = torch.zeros((batch_size, seq_len, self.sub_graph_node_num, self.sub_graph_node_num), 
                                 device=device, dtype=x.dtype)
        temp_ADJ[:, :, self.tril_indices[0], self.tril_indices[1]] = x
        temp_ADJ = temp_ADJ + temp_ADJ.transpose(2, 3)
        return temp_ADJ


class TopicSpecificBondDecoder(nn.Module):
    """为每个主题创建专门的子结构解码器"""
    def __init__(self, input_dim: int, hidden_dim: int, sub_graph_node_num: int, n_topics: int):
        super(TopicSpecificBondDecoder, self).__init__()
        self.decoders = nn.ModuleList([
            EnhancedBondDecoder(input_dim, hidden_dim, sub_graph_node_num)
            for _ in range(n_topics)
        ])
        self.n_topics = n_topics
        self.sub_graph_node_num = sub_graph_node_num

    def forward(self, in_tensor: torch.Tensor, topic_indices: torch.Tensor = None):
        """
        Args:
            in_tensor: 输入张量，形状为 [batch, seq, feat_dim]
            topic_indices: 每个样本对应的主题索引 [batch, seq]，若为 None 则从最后 n_topics 维提取 one-hot 信息
        """
        assert in_tensor.dim() == 3, "TopicSpecificBondDecoder: 输入必须为 [batch, seq, feat_dim]"
        batch_size, seq_len, feat_dim = in_tensor.shape
        device = in_tensor.device

        if topic_indices is None:
            # 从最后 n_topics 维提取 one-hot 信息
            topic_onehot = in_tensor[:, :, -self.n_topics:]
            topic_indices = torch.argmax(topic_onehot, dim=2)
        
        # 简单情况：所有样本主题相同时直接批量调用对应解码器
        if torch.all(topic_indices == topic_indices[0, 0]):
            topic_idx = topic_indices[0, 0].item()
            topic_idx = min(topic_idx, self.n_topics - 1)
            return self.decoders[topic_idx](in_tensor)
        
        # 否则按主题分组处理
        output = torch.empty(batch_size, seq_len, self.sub_graph_node_num, self.sub_graph_node_num, 
                             device=device, dtype=in_tensor.dtype)
        for topic in range(self.n_topics):
            mask = (topic_indices == topic)  # [batch, seq]
            if mask.sum() == 0:
                continue
            # 选出所有属于该主题的位置
            selected_inputs = in_tensor[mask]  # [N, feat_dim]
            # 需要将其扩展为 [N, 1, feat_dim]，解码器要求3D输入
            out_i = self.decoders[topic](selected_inputs.unsqueeze(1))  # [N, 1, sub_graph_node_num, sub_graph_node_num]
            out_i = out_i.squeeze(1)  # [N, sub_graph_node_num, sub_graph_node_num]
            output[mask] = out_i
        return output


class NVGM(torch.nn.Module):
    def __init__(self, device, ker, n_topics, in_feature_dim, max_num_nodes, n_samples, sub_num_nodes, 
                 substructure_size, node_size_options, hiddenLayers=[256, 256, 256], GraphLatntDim=1024):
        super(NVGM, self).__init__()
        self.kernel = ker
        self.n_topics = n_topics
        self.n_samples = n_samples
        self.sub_num_nodes = sub_num_nodes
        self.max_num_nodes = max_num_nodes
        self.node_size_options = node_size_options
        self.dim_hidden = 256
        self.device = device
        self.epsilon = 1e-10  # 数值稳定性常数
        self.in_feature_dim = in_feature_dim
        # 构建图级特征网络（主题相关）
        hiddenLayers_topic = [in_feature_dim] + hiddenLayers + [GraphLatntDim // 2]
        self.normLayers = nn.ModuleList(
            [nn.LayerNorm(hiddenLayers_topic[i + 1], elementwise_affine=False) for i in range(len(hiddenLayers_topic) - 1)]
        )
        self.normLayers.append(nn.LayerNorm(hiddenLayers_topic[-1], elementwise_affine=False))
        self.GCNlayers = nn.ModuleList([
            dgl.nn.pytorch.conv.GraphConv(hiddenLayers_topic[i], hiddenLayers_topic[i + 1],
                                          activation=None, bias=True, weight=True)
            for i in range(len(hiddenLayers_topic) - 1)
        ])
        self.poolingLayer = Aggregation.AvePool()

        self.stochastic_mean_layer = nn.Sequential(
            nn.Linear(GraphLatntDim // 2, self.n_topics),
            nn.ReLU(),
            nn.BatchNorm1d(self.n_topics),
            nn.Linear(self.n_topics, self.n_topics)
        )
        self.stochastic_log_std_layer = nn.Sequential(
            nn.Linear(GraphLatntDim // 2, self.n_topics),
            nn.ReLU(),
            nn.BatchNorm1d(self.n_topics),
            nn.Linear(self.n_topics, self.n_topics)
        )

        # 构建图级特征网络（映射相关）
        hiddenLayers_2 = [in_feature_dim] + hiddenLayers + [GraphLatntDim]
        self.normLayers2 = nn.ModuleList(
            [nn.LayerNorm(hiddenLayers_2[i + 1], elementwise_affine=False) for i in range(len(hiddenLayers_2) - 1)]
        )
        self.normLayers2.append(nn.LayerNorm(hiddenLayers_2[-1], elementwise_affine=False))
        self.GCNlayers2 = nn.ModuleList([
            dgl.nn.pytorch.conv.GraphConv(hiddenLayers_2[i], hiddenLayers_2[i + 1],
                                          activation=None, bias=True, weight=True)
            for i in range(len(hiddenLayers_2) - 1)
        ])
        self.poolingLayer2 = Aggregation.AvePool()

        # 初始化每个主题的均值和对数标准差参数矩阵
        self.mu_mat = nn.Parameter(torch.randn(self.n_topics, self.dim_hidden)).to(self.device)
        self.log_sigma_mat = nn.Parameter(torch.randn(self.n_topics, self.dim_hidden)).to(self.device)
        
        # 初始化主题嵌入以增强正交性
        self._initialize_topic_embeddings()

        # 为每个主题构建独立的网络
        self.mean_layers = nn.ModuleList()
        self.log_std_layers = nn.ModuleList()
        self.prior_topic = nn.ModuleList()
        for i in range(self.n_topics):
            self.prior_topic.append(nn.Linear(GraphLatntDim + self.n_topics, GraphLatntDim // 2))
            self.mean_layers.append(nn.Sequential(
                nn.Linear(GraphLatntDim // 2, self.dim_hidden),
                nn.ReLU(),
                nn.BatchNorm1d(self.dim_hidden),
                nn.Linear(self.dim_hidden, self.dim_hidden)
            ))
            self.log_std_layers.append(nn.Sequential(
                nn.Linear(GraphLatntDim // 2, self.dim_hidden),
                nn.ReLU(),
                nn.BatchNorm1d(self.dim_hidden),
                nn.Linear(self.dim_hidden, self.dim_hidden)
            ))

        self.bond_decoder = TopicSpecificBondDecoder(self.dim_hidden + self.n_topics, 256, self.sub_num_nodes, self.n_topics)

        self.normLayers3 = nn.ModuleList(
            [nn.LayerNorm(hiddenLayers_topic[i + 1], elementwise_affine=False) for i in range(len(hiddenLayers_topic) - 1)]
        )
        self.normLayers3.append(nn.LayerNorm(hiddenLayers_topic[-1], elementwise_affine=False))
        self.GCNlayers3 = nn.ModuleList([
            dgl.nn.pytorch.conv.GraphConv(hiddenLayers_topic[i], hiddenLayers_topic[i + 1],
                                          activation=None, bias=True, weight=True)
            for i in range(len(hiddenLayers_topic) - 1)
        ])
        self.poolingLayer3 = Aggregation.AvePool()

        self.map_mean_layer = nn.Sequential(
            nn.Linear(GraphLatntDim // 2, self.dim_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(self.dim_hidden),
            nn.Linear(self.dim_hidden, self.dim_hidden)
        )
        self.map_log_std_layer = nn.Sequential(
            nn.Linear(GraphLatntDim // 2, self.dim_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(self.dim_hidden),
            nn.Linear(self.dim_hidden, self.dim_hidden)
        )

        self.mapping_decoder = MappingLayer(
            input_dim=self.dim_hidden, hidden_dim=self.dim_hidden,
            output_dim=self.max_num_nodes, n_samples=self.n_samples,
            sub_num_nodes=self.sub_num_nodes, max_num_nodes=self.max_num_nodes
        )

        self.laplacian_encoder = nn.Linear(self.max_num_nodes, self.dim_hidden)

        self.attn_BC = MultiLayerAttention(self.dim_hidden, num_heads=8, num_layers=3)
        self.attn_BD = MultiLayerAttention(self.dim_hidden, num_heads=8, num_layers=3)
        self.attn_BB = MultiLayerAttention(self.dim_hidden, num_heads=8, num_layers=3)
        
        # 融合门控
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.dim_hidden * 4, self.dim_hidden),
            nn.Sigmoid()
        )
        
        # 缓存变量初始化为 None
        self._all_indices = None

    def _initialize_topic_embeddings(self):
        n_topics = self.n_topics
        dim = self.dim_hidden
        if n_topics <= dim:
            rand_mat = torch.randn(dim, dim, device=self.device)
            q, _ = torch.linalg.qr(rand_mat)
            orthogonal_vecs = q[:n_topics, :]
            scales = torch.linspace(0.5, 2.0, n_topics).unsqueeze(1).expand(-1, dim).to(self.device)
            self.mu_mat.data = orthogonal_vecs * scales
            log_sigma_init = torch.ones(n_topics, dim, device=self.device) * -2.0
            for i in range(n_topics):
                log_sigma_init[i] = -2.0 - i * 0.2
            self.log_sigma_mat.data = log_sigma_init
        else:
            for i in range(n_topics):
                torch.manual_seed(i * 100)
                self.mu_mat.data[i] = torch.randn(dim, device=self.device) * math.sqrt(2.0 / dim) * (i % 5 * 0.2 + 0.5)
                self.log_sigma_mat.data[i] = torch.ones(dim, device=self.device) * (-2.0 - (i % 5) * 0.2)
            with torch.no_grad():
                for _ in range(5):
                    gram = torch.mm(self.mu_mat, self.mu_mat.t())
                    gram.fill_diagonal_(0)
                    update = torch.mm(gram, self.mu_mat)
                    self.mu_mat.data -= 0.1 * update

    def process_gcn_layers(self, graph, features, layers, norm_layers, activation=nn.LeakyReLU(0.01)):
        h = features
        for i in range(len(layers)):
            h = layers[i](graph, h)
            h = activation(h)
            h = norm_layers[i](h)
        return h

    def forward(self, graph, features, batchSize, real_adj, activation=nn.LeakyReLU(0.01)):
        # 图编码部分：获取图级特征
        h = self.process_gcn_layers(graph, features, self.GCNlayers, self.normLayers, activation)
        h = h.reshape(*batchSize, -1)
        h = self.poolingLayer(h)
        h = self.normLayers[-1](h)
        theta_mean = self.stochastic_mean_layer(h)
        theta_log_std = self.stochastic_log_std_layer(h)
        theta_z = self.reparameterization(theta_mean, theta_log_std, 1).to(self.device)
        self.theta = F.softmax(theta_z, dim=1)
        topic_entropy = -torch.sum(self.theta * torch.log(self.theta + self.epsilon), dim=1).mean()
        
        h_z = self.process_gcn_layers(graph, features, self.GCNlayers2, self.normLayers2, activation)
        h_z = h_z.reshape(*batchSize, -1)
        h_z = self.poolingLayer2(h_z)
        h_z = self.normLayers2[-1](h_z)

        h_m = self.process_gcn_layers(graph, features, self.GCNlayers3, self.normLayers3, activation)
        h_m = h_m.reshape(*batchSize, -1)
        h_m = self.poolingLayer3(h_m)
        h_m = self.normLayers3[-1](h_m)
        
        map_mean = self.map_mean_layer(h_m)
        map_log_std = self.map_log_std_layer(h_m)
        mapping_posterior = self.reparameterization(map_mean, map_log_std, 1).to(self.device)
        
        batch_size = self.theta.shape[0]
        topics = torch.multinomial(self.theta, self.n_samples, replacement=True)
        sorted_topics, _ = torch.sort(topics, dim=1)
        sorted_topics = torch.clamp(sorted_topics, 0, self.n_topics - 1)
        one_hot_n_topic = F.one_hot(sorted_topics, num_classes=self.n_topics).to(self.device)
        num_sample_topics = one_hot_n_topic.sum(dim=1)  # [batch, n_topics]

        # 使用独立网络计算各主题的均值和对数标准差
        mean_zqns_list = []
        log_std_zqns_list = []
        all_topic_labels = torch.eye(self.n_topics, device=self.device).repeat(batch_size, 1, 1)
        for i, (mean_layer, log_std_layer, prior_topic_layer) in enumerate(zip(self.mean_layers, self.log_std_layers, self.prior_topic)):
            labels = all_topic_labels[:, i]
            p_topic = torch.cat([h_z, labels], dim=1)
            p_topic = prior_topic_layer(p_topic)
            mean_zqn = mean_layer(p_topic)
            log_std_zqn = log_std_layer(p_topic)
            mean_zqns_list.append(mean_zqn)
            log_std_zqns_list.append(log_std_zqn)
        self.mean_zqns = torch.stack(mean_zqns_list, dim=1)  # [batch, n_topics, dim_hidden]
        self.log_std_zqns = torch.stack(log_std_zqns_list, dim=1)  # 同上

        diversity_loss = self.diversity_encouraging_regularization(self.mean_zqns)
        ortho_loss = self.orthogonal_regularization(self.mean_zqns)
        topic_entropy_regularization = diversity_loss - topic_entropy

        # 子结构采样：对每个批次中各主题的采样采用向量化实现（使用 pad_sequence 处理变长情况）
        substructure_size = self.mean_zqns.shape[-1]
        samples_list = []
        for b in range(batch_size):
            samples_b = []
            for j in range(self.n_topics):
                ns = int(num_sample_topics[b, j].item())
                if ns <= 0:
                    continue
                eps = torch.randn(ns, substructure_size, device=self.device)
                mean_j = self.mean_zqns[b, j].unsqueeze(0).expand(ns, -1)
                std_j = torch.exp(self.log_std_zqns[b, j]).unsqueeze(0).expand(ns, -1)
                samples = mean_j + std_j * eps
                topic_onehot = torch.zeros(ns, self.n_topics, device=self.device)
                topic_onehot[:, j] = 1.0
                samples_with_topic = torch.cat([samples, topic_onehot], dim=1)
                samples_b.append(samples_with_topic)
            if samples_b:
                samples_b = torch.cat(samples_b, dim=0)
            else:
                samples_b = torch.zeros(1, substructure_size + self.n_topics, device=self.device)
            samples_list.append(samples_b)
        # 将变长序列 pad 到相同长度，得到 [batch, seq, feature]
        z_qn = pad_sequence(samples_list, batch_first=True)
        
        # 使用主题特定解码器生成子结构
        z_qn_adj = self.bond_decoder(z_qn)
        
        node_num = self.sub_num_nodes + self.max_num_nodes
        current_G = torch.zeros((batch_size, self.max_num_nodes, self.max_num_nodes), device=self.device)
        mapping_matrix_list = []
        selected_indices = None

        # 逐步构建图（此处仍采用循环，批次数量一般较小）
        for i in range(self.n_samples):
            selected_substructures, _, selected_indices = self.select_random_substructure(z_qn_adj, selected_indices)
            node_pe = torch.zeros(batch_size, node_num, self.max_num_nodes, device=self.device)
            node_pe[:, :self.sub_num_nodes, :self.sub_num_nodes] = selected_substructures
            node_pe[:, self.sub_num_nodes:, :] = current_G
            node_pe = self.laplacian_encoder(node_pe)

            B = node_pe[:, :self.sub_num_nodes, :]  # 子结构
            C = node_pe[:, self.sub_num_nodes:, :]  # 当前图
            D = mapping_posterior.unsqueeze(1)      # 映射编码

            # 转置以适应注意力层：注意力层要求 [seq, batch, dim]
            B = B.permute(1, 0, 2)
            C = C.permute(1, 0, 2)
            D = D.permute(1, 0, 2)

            B2, _ = self.attn_BC(B, C, C)
            B3, _ = self.attn_BD(B, D, D)
            B4, _ = self.attn_BB(B, B, B)
            B_concat = torch.cat([B, B2, B3, B4], dim=-1)
            gates = self.fusion_gate(B_concat)
            B_update = B + gates * (B2 + B3 + B4 - B)
            B_update = B_update.permute(1, 0, 2)

            mapping_matrix = self.mapping_decoder(B_update)
            mapping_matrix = F.softmax(mapping_matrix, dim=-1)
            mapping_matrix_list.append(mapping_matrix)
            mapping_transpose = mapping_matrix.transpose(-1, -2)
            candidate_G = torch.bmm(
                torch.bmm(mapping_transpose, selected_substructures.view(batch_size, self.sub_num_nodes, self.sub_num_nodes)),
                mapping_matrix
            )
            current_G = candidate_G + current_G

        reconstructed_adj_logit = current_G
        reconstructed_adj = torch.sigmoid(reconstructed_adj_logit)
        mapping_matrix_list = torch.stack(mapping_matrix_list, dim=1)
        kernel_value = self.kernel(reconstructed_adj)

        kl_div = self.compute_kl_divergence(
            self.mean_zqns, 
            self.log_std_zqns,
            self.mu_mat.unsqueeze(0).expand(batch_size, -1, -1),
            self.log_sigma_mat.unsqueeze(0).expand(batch_size, -1, -1)
        )
        theta_expanded = self.theta.unsqueeze(-1)
        weighted_kl_div = kl_div * theta_expanded
        total_kl_div = weighted_kl_div.sum(dim=-1).mean()

        theta_std = torch.exp(theta_log_std)
        kl_theta = 0.5 * (theta_mean.pow(2) + theta_std.pow(2) - 1 - 2 * theta_log_std)
        kl_theta = kl_theta.sum(dim=-1).mean()

        map_std = torch.exp(map_log_std)
        kl_map = 0.5 * (map_mean.pow(2) + map_std.pow(2) - 1 - 2 * map_log_std)
        kl_map = kl_map.sum(dim=-1).mean()

        z_qn_adj_sigmoid = torch.sigmoid(z_qn_adj)
        
        return (reconstructed_adj, theta_mean, theta_log_std, kernel_value, reconstructed_adj_logit,
                total_kl_div, kl_theta, ortho_loss, self.theta, z_qn_adj_sigmoid, sorted_topics,
                mapping_matrix_list, kl_map, topic_entropy_regularization)

    def compute_kl_divergence(self, mean_q, log_std_q, mean_p, log_std_p):
        log_std_q_clamped = torch.clamp(log_std_q, min=-20, max=20)
        log_std_p_clamped = torch.clamp(log_std_p, min=-20, max=20)
        term1 = log_std_p_clamped - log_std_q_clamped
        sq_diff = (mean_q - mean_p).pow(2)
        term2_1 = torch.exp(2 * log_std_q_clamped - 2 * log_std_p_clamped)
        term2_2 = sq_diff * torch.exp(-2 * log_std_p_clamped)
        term2 = term2_1 + term2_2
        kl = 0.5 * (2 * term1 + term2 - 1)
        return kl

    def diversity_encouraging_regularization(self, topic_zqns):
        batch_size, n_topics, dim = topic_zqns.size()
        topic_means = topic_zqns.mean(dim=0)
        norm_topics = F.normalize(topic_means, p=2, dim=1)
        cos_sim = torch.mm(norm_topics, norm_topics.t())
        mask = ~torch.eye(n_topics, dtype=torch.bool, device=topic_means.device)
        cos_sim = cos_sim.masked_select(mask)
        diversity_loss = cos_sim.abs().mean()
        return diversity_loss

    def init_tensor(self, size, device):
        return torch.randn(size, device=device)

    def generate_topics(self):
        with torch.no_grad():
            self.eval()
            samples = self.mu_mat.unsqueeze(1).expand(-1, 30, -1).to(self.device)
            one_hot_vectors = torch.eye(self.n_topics, device=self.device).unsqueeze(1).repeat(1, 30, 1)
            z_qn = torch.cat([samples, one_hot_vectors], dim=2)
            z_qn_adj = self.bond_decoder(z_qn)
            return torch.sigmoid(z_qn_adj)

    def generate(self, num, eps):
        with torch.no_grad():
            self.eval()
            batch_size = num
            theta_z = self.init_tensor((num, self.n_topics), self.device)
            self.theta = F.softmax(theta_z, dim=1)
            mapping_posterior = self.init_tensor((num, self.dim_hidden), self.device)
            topics = torch.multinomial(self.theta, self.n_samples, replacement=True)
            sorted_topics, _ = torch.sort(topics, dim=1)
            sorted_topics = torch.clamp(sorted_topics, 0, self.n_topics - 1)
            flat_sorted_topics = sorted_topics.view(-1)
            mean_zk = self.mu_mat[flat_sorted_topics].view(batch_size, self.n_samples, self.dim_hidden)
            log_sigma_zk = self.log_sigma_mat[flat_sorted_topics].view(batch_size, self.n_samples, self.dim_hidden)
            z_qn = self.reparameterization(mean_zk, log_sigma_zk, 1)
            topics_one_hot = F.one_hot(sorted_topics, num_classes=self.n_topics).to(self.device)
            z_qn = torch.cat((z_qn, topics_one_hot), dim=2)
            z_qn_adj = self.bond_decoder(z_qn, topic_indices=sorted_topics)
            node_num = self.sub_num_nodes + self.max_num_nodes
            current_G = torch.zeros((batch_size, self.max_num_nodes, self.max_num_nodes), device=self.device)
            mapping_matrix_list = []
            selected_indices = None
            for i in range(self.n_samples):
                selected_substructures, _, selected_indices = self.select_random_substructure(z_qn_adj, selected_indices)
                node_pe = torch.zeros(batch_size, node_num, self.max_num_nodes, device=self.device)
                node_pe[:, :self.sub_num_nodes, :self.sub_num_nodes] = selected_substructures
                node_pe[:, self.sub_num_nodes:, :] = current_G
                node_pe = self.laplacian_encoder(node_pe)
                B = node_pe[:, :self.sub_num_nodes, :]
                C = node_pe[:, self.sub_num_nodes:, :]
                D = mapping_posterior.unsqueeze(1)
                B = B.permute(1, 0, 2)
                C = C.permute(1, 0, 2)
                D = D.permute(1, 0, 2)
                B2, _ = self.attn_BC(B, C, C)
                B3, _ = self.attn_BD(B, D, D)
                B4, _ = self.attn_BB(B, B, B)
                B_concat = torch.cat([B, B2, B3, B4], dim=-1)
                gates = self.fusion_gate(B_concat)
                B_update = B + gates * (B2 + B3 + B4 - B)
                B_update = B_update.permute(1, 0, 2)
                mapping_matrix = self.mapping_decoder(B_update)
                mapping_matrix = F.softmax(mapping_matrix, dim=-1)
                mapping_matrix_list.append(mapping_matrix)
                mapping_transpose = mapping_matrix.transpose(-1, -2)
                candidate_G = torch.bmm(
                    torch.bmm(mapping_transpose, selected_substructures.view(batch_size, self.sub_num_nodes, self.sub_num_nodes)),
                    mapping_matrix
                )
                current_G = candidate_G + current_G
            reconstructed_adj_logit = current_G
            z_qn_adj_sigmoid = torch.sigmoid(z_qn_adj)
            reconstructed_adj = torch.sigmoid(reconstructed_adj_logit)
            mapping_matrix_list = torch.stack(mapping_matrix_list, dim=1)
            return reconstructed_adj, self.theta, sorted_topics, z_qn_adj_sigmoid, mapping_matrix_list

    def select_random_substructure(self, z_qn_adj, selected_indices=None):
        """
        选择子结构，输入 z_qn_adj 形状 [batch, num_substructures, node_count, node_count]
        返回选中的子结构、对应索引，以及更新后的 selected_indices
        """
        batch_size, num_substructures, node_count, _ = z_qn_adj.shape
        device = z_qn_adj.device
        
        if self._all_indices is None or self._all_indices.shape != (batch_size, num_substructures):
            self._all_indices = torch.arange(num_substructures, device=device).unsqueeze(0).expand(batch_size, -1)
        if selected_indices is None:
            selected_indices = torch.empty((batch_size, 0), dtype=torch.long, device=device)
        
        if selected_indices.shape[1] > 0:
            selected_mask = (self._all_indices.unsqueeze(-1) == selected_indices.unsqueeze(1)).any(dim=-1)
            valid_mask = ~selected_mask
        else:
            valid_mask = torch.ones((batch_size, num_substructures), dtype=torch.bool, device=device)
        
        valid_indices = torch.where(valid_mask, self._all_indices, -torch.ones_like(self._all_indices))
        row_counts = valid_mask.sum(dim=1)
        max_count = int(row_counts.max().item())
        
        padded_indices = []
        for b in range(batch_size):
            valid_b = valid_indices[b][valid_mask[b]]
            if valid_b.numel() == 0:
                valid_b = self._all_indices[b]
            if valid_b.numel() < max_count:
                pad = valid_b.new_zeros(max_count - valid_b.numel())
                valid_b = torch.cat([valid_b, pad])
            padded_indices.append(valid_b.unsqueeze(0))
        padded_indices = torch.cat(padded_indices, dim=0)  # [batch, max_count]
        random_idx = torch.randint(0, max_count, (batch_size,), device=device)
        random_indices = padded_indices[torch.arange(batch_size, device=device), random_idx]
        selected_substructures = z_qn_adj[torch.arange(batch_size, device=device), random_indices]
        selected_indices = torch.cat((selected_indices, random_indices.unsqueeze(1)), dim=1)
        return selected_substructures, random_indices, selected_indices

    def orthogonal_regularization(self, mu, scale=1.0):
        batch_size, K, dim = mu.size()
        dot_product = torch.bmm(mu, mu.transpose(1, 2))
        identity = torch.eye(K, device=mu.device).unsqueeze(0).expand(batch_size, -1, -1)
        diff = dot_product - identity
        mask = torch.triu(torch.ones_like(diff), diagonal=1)
        ortho_loss = ((diff * mask) ** 2).sum(dim=[1, 2])
        return scale * ortho_loss.mean()

    def reparameterization(self, mu, logsig, noise_scale):
        std = torch.exp(logsig)
        eps = torch.randn_like(mu).to(mu.device)
        return mu + eps * noise_scale * std


class MappingLayer(nn.Module):
    """映射层，用于将子结构嵌入到完整图中"""
    def __init__(self, input_dim, hidden_dim, output_dim, n_samples, sub_num_nodes, max_num_nodes):
        super(MappingLayer, self).__init__()
        layers = [input_dim, hidden_dim, hidden_dim, max_num_nodes]
        self.layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1], dtype=torch.float32) for i in range(len(layers) - 1)]
        )
        self.norm_layers = nn.ModuleList(
            [nn.LayerNorm(layers[i + 1], elementwise_affine=False) for i in range(len(layers) - 2)]
        )

    def forward(self, in_tensor: torch.Tensor, activation=nn.LeakyReLU(0.001)):
        # 如果输入是2D，则添加批次维度
        if in_tensor.dim() == 2:
            in_tensor = in_tensor.unsqueeze(0)
        assert in_tensor.dim() == 3, "MappingLayer: 输入必须为 [batch, seq, feat]"
        batch_size, seq_len, feat_dim = in_tensor.shape
        in_tensor_flat = in_tensor.reshape(-1, feat_dim)
        x = in_tensor_flat
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers) - 1:
                x = activation(x)
                x = self.norm_layers[i](x)
        out_tensor = x.view(batch_size, seq_len, -1)
        return out_tensor
