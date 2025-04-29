# data_MM.py
import random
import networkx as nx
import numpy as np
import torch
import os
import pickle as pkl
import scipy.sparse as sp
from scipy.sparse import lil_matrix, csr_matrix
import warnings
from ogb.graphproppred import DglGraphPropPredDataset
import dgl

# -------------------- 辅助函数 --------------------
def parse_index_file(filename):
    index = []
    with open(filename, 'r') as f:
        for line in f:
            index.append(int(line.strip()))
    return index

# 加载 cora、citeseer 和 pubmed 数据集，此处只用到 citeseer
def Graph_load(dataset='cora'):
    data_path = "data/Kernel_dataset/ind.{}."
    names = ['x', 'tx', 'allx', 'graph']
    
    objects = {}
    for name in names:
        with open(data_path.format(dataset, name), 'rb') as f:
            objects[name] = pkl.load(f, encoding='latin1')
    
    x = objects['x']
    tx = objects['tx']
    allx = objects['allx']
    graph = objects['graph']
    
    test_idx_reorder = parse_index_file(data_path.format(dataset, 'test.index'))
    test_idx_range = np.sort(test_idx_reorder)
    
    if dataset == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
    
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)
    
    return adj, features, G

# 返回子集（打乱顺序）
def return_subset(A, X, Y, limited_to):
    indx = list(range(len(A)))
    random.shuffle(indx)
    A = [A[i] for i in indx]
    X = [X[i] for i in indx]
    if Y is not None and len(Y) != 0:
        Y = [Y[i] for i in indx]
    return A, X, Y


def visualize_substructures(substructure_adjs, substructure_types, save_dir='/home/data/zyx/A-NeurIPS/AAsubsturctureFIGURE'):
    """
    将提取的子结构按类型分组可视化，并保存到指定目录
    
    参数:
      - substructure_adjs: 子结构邻接矩阵列表
      - substructure_types: 对应的子结构类型列表
      - save_dir: 图像保存目录
    """
    import os
    import matplotlib.pyplot as plt
    import networkx as nx
    from collections import defaultdict
    import numpy as np
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 按类型分组子结构
    grouped_structures = defaultdict(list)
    for adj, struct_type in zip(substructure_adjs, substructure_types):
        grouped_structures[struct_type].append(adj)
    
    # 为每种类型绘制子结构
    for struct_type, adj_list in grouped_structures.items():
        # 确定网格大小
        n_structures = len(adj_list)
        grid_size = int(np.ceil(np.sqrt(n_structures)))
        
        # 创建大图和子图网格
        fig = plt.figure(figsize=(grid_size*3, grid_size*3))
        fig.suptitle(f'子结构类型: {struct_type}', fontsize=16, fontweight='bold')
        
        # 对该类型的每个子结构绘图
        for i, adj in enumerate(adj_list):
            if i >= grid_size * grid_size:  # 限制每个类型最多显示grid_size^2个子结构
                break
                
            # 创建子图
            ax = fig.add_subplot(grid_size, grid_size, i+1)
            
            # 转换为NetworkX图进行可视化
            G = nx.from_scipy_sparse_array(adj)
            
            # 按子结构类型选择合适的布局
            if 'cycle' in struct_type:
                pos = nx.circular_layout(G)
            elif 'star' in struct_type:
                pos = nx.kamada_kawai_layout(G)
            elif 'chain' in struct_type:
                pos = nx.spectral_layout(G)
            else:
                pos = nx.spring_layout(G, seed=42, k=0.5)  # 固定种子以获得一致布局
            
            # 自定义节点和边的样式
            node_size = 300 if G.number_of_nodes() <= 10 else 200
            node_color = get_color_for_type(struct_type)
            
            # 绘制图
            nx.draw(G, pos, ax=ax, with_labels=True, node_size=node_size, 
                   node_color=node_color, font_weight='bold', font_color='white',
                   edge_color='gray', width=2.0, alpha=0.8)
            
            ax.set_title(f'#{i+1}', fontsize=10)
            ax.axis('off')
        
        # 调整布局并保存
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # 为总标题留出空间
        
        # 安全的文件名
        safe_filename = struct_type.replace('/', '_').replace('\\', '_')
        save_path = os.path.join(save_dir, f"{safe_filename}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"已保存 {struct_type} 的子结构可视化到 {save_path}")

def get_color_for_type(struct_type):
    """根据子结构类型返回合适的节点颜色"""
    if 'triangle' in struct_type:
        return '#1f77b4'  # 蓝色
    elif 'clique' in struct_type:
        return '#ff7f0e'  # 橙色
    elif 'cycle' in struct_type:
        return '#2ca02c'  # 绿色
    elif 'star' in struct_type:
        return '#d62728'  # 红色
    elif 'chain' in struct_type:
        return '#9467bd'  # 紫色
    else:
        return '#8c564b'  # 棕色
    
# -------------------- 子结构检测与特征扩展 --------------------
def detect_substructures(G, substructure_types):
    """
    检测图 G 中预定义的子结构，返回一个字典，键为子结构名称，值为 (n x 1) 的 numpy 数组，n 为节点数。
    确保所有特征向量具有一致的维度。
    """
    G.remove_edges_from([(i, i) for i in G.nodes()])
    n = G.number_of_nodes()
    results = {}
    
    for st in substructure_types:
        try:
            if st == "triangle_count":
                try:
                    triangles = nx.triangles(G)
                    results[st] = np.array([triangles.get(node, 0) for node in range(n)]).reshape(n, 1)
                except Exception as e:
                    # print(f"Error in triangle_count: {e}")
                    results[st] = np.zeros((n, 1))
                    
            elif st == "5cycle_detection":
                try:
                    cycle_size = 5
                    cycle_indicator = np.zeros(n)
                    
                    # 仅对较小的图尝试计算循环基
                    if n <= 50:  # 设置一个阈值，避免大图的计算
                        try:
                            cycle_basis = nx.cycle_basis(G)
                            for cycle in cycle_basis:
                                if len(cycle) == cycle_size:
                                    for node in cycle:
                                        if node < n:  # 确保节点索引在有效范围内
                                            cycle_indicator[node] = 1
                        except:
                            pass
                            
                    results[st] = cycle_indicator.reshape(n, 1)
                except Exception as e:
                    # print(f"Error in 5cycle_detection: {e}")
                    results[st] = np.zeros((n, 1))
                    
            elif st == "6cycle_detection":
                try:
                    cycle_size = 6
                    cycle_indicator = np.zeros(n)
                    
                    if n <= 50:
                        try:
                            cycle_basis = nx.cycle_basis(G)
                            for cycle in cycle_basis:
                                if len(cycle) == cycle_size:
                                    for node in cycle:
                                        if node < n:
                                            cycle_indicator[node] = 1
                        except:
                            pass
                            
                    results[st] = cycle_indicator.reshape(n, 1)
                except Exception as e:
                    # print(f"Error in 6cycle_detection: {e}")
                    results[st] = np.zeros((n, 1))
                    
            elif st == "7cycle_detection":
                try:
                    cycle_size = 7
                    cycle_indicator = np.zeros(n)
                    
                    if n <= 50:
                        try:
                            cycle_basis = nx.cycle_basis(G)
                            for cycle in cycle_basis:
                                if len(cycle) == cycle_size:
                                    for node in cycle:
                                        if node < n:
                                            cycle_indicator[node] = 1
                        except:
                            pass
                            
                    results[st] = cycle_indicator.reshape(n, 1)
                except Exception as e:
                    # print(f"Error in 7cycle_detection: {e}")
                    results[st] = np.zeros((n, 1))
                    
            elif st == "laplacian_features":
                try:
                    # 固定特征值数量为5
                    k = 5
                    
                    # 计算拉普拉斯矩阵特征值
                    if nx.is_connected(G) and n > 1:
                        L = nx.normalized_laplacian_matrix(G).todense()
                        eigenvalues = np.linalg.eigvalsh(L)
                        
                        # 处理特征值数量不足k的情况
                        if len(eigenvalues) < k:
                            padded_eigenvalues = np.zeros(k)
                            padded_eigenvalues[:len(eigenvalues)] = eigenvalues
                            spectral_feat = np.tile(padded_eigenvalues.reshape(1, -1), (n, 1))
                        else:
                            spectral_feat = np.tile(eigenvalues[:k].reshape(1, -1), (n, 1))
                    else:
                        # 不连通或只有一个节点的图
                        spectral_feat = np.zeros((n, k))
                        
                    results[st] = spectral_feat
                except Exception as e:
                    # print(f"Error in laplacian_features: {e}")
                    results[st] = np.zeros((n, 5))  # 固定为5列
                    
            elif st == "diffusion_features":
                try:
                    # 固定步数为3
                    k_steps = 3
                    
                    # 获取邻接矩阵并归一化
                    adj_matrix = nx.adjacency_matrix(G).todense()
                    row_sums = adj_matrix.sum(axis=1)
                    # 避免除零错误
                    row_sums[row_sums == 0] = 1
                    norm_adj = np.array(adj_matrix) / row_sums
                    
                    diffusion_feats = np.zeros((n, k_steps))
                    
                    # 计算从每个节点开始的k步随机游走
                    for i in range(n):
                        start = np.zeros(n)
                        start[i] = 1
                        current = start.copy()
                        for step in range(k_steps):
                            current = np.dot(current, norm_adj)
                            diffusion_feats[i, step] = current[i] if i < len(current) else 0
                            
                    results[st] = diffusion_feats
                except Exception as e:
                    # print(f"Error in diffusion_features: {e}")
                    results[st] = np.zeros((n, 3))  # 固定为3列
                    
            elif st == "clustering_coef":
                try:
                    clustering = nx.clustering(G)
                    results[st] = np.array([clustering.get(node, 0) for node in range(n)]).reshape(n, 1)
                except Exception as e:
                    # print(f"Error in clustering_coef: {e}")
                    results[st] = np.zeros((n, 1))
                    
            elif st == "star_indicator":
                try:
                    degrees = dict(G.degree())
                    # 确保所有节点都有度数
                    degree_values = np.array([degrees.get(node, 0) for node in range(n)])
                    median_degree = np.median(degree_values)
                    indicator = (degree_values > median_degree).astype(np.float32)
                    results[st] = indicator.reshape(n, 1)
                except Exception as e:
                    # print(f"Error in star_indicator: {e}")
                    results[st] = np.zeros((n, 1))
                    
            elif st == "clique_indicator":
                try:
                    indicator = np.zeros(n)
                    
                    # 仅对较小的图尝试寻找团
                    if n <= 50:
                        cliques = list(nx.find_cliques(G))
                        for clique in cliques:
                            if len(clique) >= 3:
                                for node in clique:
                                    if node < n:  # 确保节点索引在有效范围内
                                        indicator[node] = 1
                                        
                    results[st] = indicator.reshape(n, 1)
                except Exception as e:
                    # print(f"Error in clique_indicator: {e}")
                    results[st] = np.zeros((n, 1))
                    
            elif st == "chain_indicator":
                try:
                    degrees = dict(G.degree())
                    triangles = nx.triangles(G)
                    indicator = np.array([1 if (degrees.get(node, 0) == 2 and triangles.get(node, 0) == 0) 
                                        else 0 for node in range(n)])
                    results[st] = indicator.reshape(n, 1)
                except Exception as e:
                    # print(f"Error in chain_indicator: {e}")
                    results[st] = np.zeros((n, 1))
                    
            elif st == "4cycle_count":
                try:
                    count = np.zeros(n)
                    
                    if n <= 30:  # 仅对较小的图进行计算
                        nodes = list(G.nodes())
                        for i in nodes:
                            if i >= n:  # 跳过超出范围的节点
                                continue
                            neighbors_i = list(G.neighbors(i))
                            for j in neighbors_i:
                                if j <= i or j >= n:
                                    continue
                                neighbors_j = list(G.neighbors(j))
                                for k in neighbors_j:
                                    if k in [i, j] or k >= n:
                                        continue
                                    neighbors_k = list(G.neighbors(k))
                                    for l in neighbors_k:
                                        if l in [i, j, k] or l >= n:
                                            continue
                                        if G.has_edge(l, i):
                                            count[i] += 1
                                            
                    results[st] = (count / 4).reshape(n, 1)
                except Exception as e:
                    # print(f"Error in 4cycle_count: {e}")
                    results[st] = np.zeros((n, 1))
                    
            # 新增: 计算节点的中心性度量
            elif st == "betweenness_centrality":
                try:
                    if n <= 100:  # 小图使用精确算法
                        centrality = nx.betweenness_centrality(G)
                    else:  # 大图使用近似算法
                        centrality = nx.betweenness_centrality(G, k=min(20, n))
                        
                    results[st] = np.array([centrality.get(node, 0) for node in range(n)]).reshape(n, 1)
                except Exception as e:
                    # print(f"Error in betweenness_centrality: {e}")
                    results[st] = np.zeros((n, 1))
                    
            # 新增: 计算节点的特征向量中心性
            elif st == "eigenvector_centrality":
                try:
                    if n <= 100 and nx.is_connected(G):  # 仅对连通的小图计算
                        try:
                            centrality = nx.eigenvector_centrality(G, max_iter=100)
                            results[st] = np.array([centrality.get(node, 0) for node in range(n)]).reshape(n, 1)
                        except:
                            results[st] = np.zeros((n, 1))
                    else:
                        results[st] = np.zeros((n, 1))
                except Exception as e:
                    # print(f"Error in eigenvector_centrality: {e}")
                    results[st] = np.zeros((n, 1))
                    
            # 新增: 计算节点的PageRank值
            elif st == "pagerank":
                try:
                    pr = nx.pagerank(G, alpha=0.85)
                    results[st] = np.array([pr.get(node, 0) for node in range(n)]).reshape(n, 1)
                except Exception as e:
                    # print(f"Error in pagerank: {e}")
                    results[st] = np.zeros((n, 1))
                    
            # 新增: 检测节点是否位于桥边上
            elif st == "bridge_nodes":
                try:
                    bridges = list(nx.bridges(G))
                    indicator = np.zeros(n)
                    for u, v in bridges:
                        if u < n:
                            indicator[u] = 1
                        if v < n:
                            indicator[v] = 1
                    results[st] = indicator.reshape(n, 1)
                except Exception as e:
                    # print(f"Error in bridge_nodes: {e}")
                    results[st] = np.zeros((n, 1))
                    
            # 新增: 节点在k-核中的最大k值
            elif st == "kcore_number":
                try:
                    core_numbers = nx.core_number(G)
                    results[st] = np.array([core_numbers.get(node, 0) for node in range(n)]).reshape(n, 1)
                except Exception as e:
                    # print(f"Error in kcore_number: {e}")
                    results[st] = np.zeros((n, 1))
                    
            # 新增: 节点是否为割点(移除后会增加连通分量数)
            elif st == "articulation_points":
                try:
                    if nx.is_connected(G):  # 仅对连通图计算
                        cut_nodes = list(nx.articulation_points(G))
                        indicator = np.zeros(n)
                        for node in cut_nodes:
                            if node < n:
                                indicator[node] = 1
                        results[st] = indicator.reshape(n, 1)
                    else:
                        results[st] = np.zeros((n, 1))
                except Exception as e:
                    # print(f"Error in articulation_points: {e}")
                    results[st] = np.zeros((n, 1))
                    
            # 新增: 节点的连通分量大小
            elif st == "component_size":
                try:
                    components = list(nx.connected_components(G))
                    component_map = {}
                    for comp in components:
                        size = len(comp)
                        for node in comp:
                            component_map[node] = size
                            
                    results[st] = np.array([component_map.get(node, 1) for node in range(n)]).reshape(n, 1)
                except Exception as e:
                    # print(f"Error in component_size: {e}")
                    results[st] = np.zeros((n, 1))
                    
            # 新增: 节点的局部效率 (邻居子图的效率)
            elif st == "local_efficiency":
                try:
                    efficiency = np.zeros(n)
                    
                    if n <= 30:  # 仅对小图计算
                        nodes = list(G.nodes())
                        for i, node in enumerate(nodes):
                            if i >= n:
                                continue
                                
                            neighbors = list(G.neighbors(node))
                            if len(neighbors) < 2:  # 少于2个邻居无法形成子图
                                efficiency[i] = 0
                                continue
                                
                            # 检查邻居节点是否存在于有效范围内
                            valid_neighbors = [neigh for neigh in neighbors if neigh < n]
                            if len(valid_neighbors) < 2:
                                efficiency[i] = 0
                                continue
                                
                            try:
                                subgraph = G.subgraph(valid_neighbors)
                                if nx.is_connected(subgraph) and len(subgraph) > 1:
                                    path_lengths = dict(nx.all_pairs_shortest_path_length(subgraph))
                                    n_sub = len(subgraph)
                                    eff = 0
                                    for u in subgraph:
                                        for v in subgraph:
                                            if u != v:
                                                if v in path_lengths.get(u, {}):
                                                    eff += 1.0 / path_lengths[u][v]
                                    efficiency[i] = eff / (n_sub * (n_sub - 1)) if n_sub > 1 else 0
                            except:
                                efficiency[i] = 0
                                
                    results[st] = efficiency.reshape(n, 1)
                except Exception as e:
                    # print(f"Error in local_efficiency: {e}")
                    results[st] = np.zeros((n, 1))
                    
            # 新增: 小世界指数 (仅对整个图计算一次，所有节点值相同)
            elif st == "small_world_index":
                try:
                    # 对于太小的图或非连通图，返回零向量
                    if n < 10 or not nx.is_connected(G):
                        results[st] = np.zeros((n, 1))
                        continue
                        
                    try:
                        # 计算平均路径长度
                        avg_path = nx.average_shortest_path_length(G)
                        # 计算平均聚类系数
                        avg_clust = nx.average_clustering(G)
                        
                        # 生成随机图作为参考
                        random_G = nx.gnm_random_graph(n, G.number_of_edges())
                        if nx.is_connected(random_G):
                            rand_avg_path = nx.average_shortest_path_length(random_G)
                            rand_avg_clust = nx.average_clustering(random_G)
                            
                            # 计算小世界指数
                            if rand_avg_clust > 0 and rand_avg_path > 0:
                                sw_index = (avg_clust / rand_avg_clust) / (avg_path / rand_avg_path)
                            else:
                                sw_index = 0
                        else:
                            sw_index = 0
                            
                        results[st] = np.ones((n, 1)) * sw_index
                    except:
                        results[st] = np.zeros((n, 1))
                except Exception as e:
                    # print(f"Error in small_world_index: {e}")
                    results[st] = np.zeros((n, 1))
            else:
                results[st] = np.zeros((n, 1))
            
            if st in results:
                # 检查并替换非法值
                if np.isnan(results[st]).any() or np.isinf(results[st]).any():
                    warnings.warn(f"发现非法值在特征 {st} 中，已替换为0")
                    results[st] = np.nan_to_num(results[st], nan=0.0, posinf=1.0, neginf=-1.0)
                    
                # 防止极端值 - 将值裁剪到合理范围
                if results[st].size > 0:
                    results[st] = np.clip(results[st], -10.0, 10.0)
        
        except Exception as e:
            warnings.warn(f"计算特征 {st} 时出错: {e}")
            # 提供安全的默认值
            if st == 'laplacian_features':
                results[st] = np.zeros((n, 5))
            elif st == 'diffusion_features':
                results[st] = np.zeros((n, 3))
            else:
                results[st] = np.zeros((n, 1))
    return results

def enrich_node_features(G, X, substructure_types=['triangle_count', 'clustering_coef', 'star_indicator', 'clique_indicator', 'chain_indicator', '4cycle_count']):
    """
    扩展节点特征。若原始特征 X 为空则使用单位矩阵作为初始特征，
    然后将预定义子结构检测得到的特征与原始特征拼接在一起。
    """
    n = G.number_of_nodes()
    if X is None:
        X = np.identity(n, dtype=np.float32)
    else:
        X = np.array(X)
    sub_features = detect_substructures(G, substructure_types)
    feature_list = [X]
    for st in substructure_types:
        feature_list.append(sub_features[st])
    enriched_X = np.concatenate(feature_list, axis=1)
    return enriched_X

# -------------------- Datasets 类（保留原有功能并扩展子结构特征） --------------------
class Datasets:
    def __init__(self, list_adjs, self_for_none, list_Xs, graphlabels=None, padding=True, max_num=None, set_diag_of_isol_zer=True):
        if max_num:
            list_adjs, graphlabels, list_Xs = self.remove_largergraphs(list_adjs, graphlabels, list_Xs, max_num)
        self.set_diag_of_isol_zer = set_diag_of_isol_zer
        self.padding = padding
        self.list_Xs = list_Xs
        self.labels = graphlabels
        self.list_adjs = list_adjs
        self.total_num_of_edges = 0
        self.max_num_nodes = 0
        self.min_num_nodes = float('inf')

        for i, adj in enumerate(list_adjs):
            adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
            adj += sp.eye(adj.shape[0])
            list_adjs[i] = adj
            self.max_num_nodes = max(self.max_num_nodes, adj.shape[0])
            self.min_num_nodes = min(self.min_num_nodes, adj.shape[0])
            self.total_num_of_edges += adj.sum()

        if max_num:
            self.max_num_nodes = max_num
        self.processed_Xs = []
        self.processed_adjs = []
        self.num_of_edges = []
        for i in range(self.__len__()):
            a, x, n, _ = self.process(i, self_for_none)
            self.processed_Xs.append(x)
            self.processed_adjs.append(a)
            self.num_of_edges.append(n)
        self.feature_size = self.processed_Xs[0].shape[-1]
        self.adj_s = []
        self.x_s = []
        self.num_nodes = []
        self.subgraph_indexes = []
        self.featureList = None

    def remove_largergraphs(self, adjs, labels, Xs, max_size):
        processed_adjs = []
        processed_labels = []
        processed_Xs = []
        for i in range(len(adjs)):
            if adjs[i].shape[0] <= max_size:
                processed_adjs.append(adjs[i])
                if labels is not None:
                    processed_labels.append(labels[i])
                if Xs is not None:
                    processed_Xs.append(Xs[i])
        return processed_adjs, processed_labels, processed_Xs

    def get(self):
        indices = list(range(self.__len__()))
        return [self.processed_adjs[i] for i in indices], [self.processed_Xs[i] for i in indices]

    def set_features(self, some_feature):
        self.featureList = some_feature

    def get_adj_list(self):
        return self.adj_s

    def get__(self, from_, to_, self_for_none, bfs=None, ignore_isolate_nodes=False):
        adj_s = []
        x_s = []
        num_nodes = []
        subgraph_indexes = []
        if bfs is None:
            graphfeatures = [element[from_:to_] for element in self.featureList]
            return (self.adj_s[from_:to_], self.x_s[from_:to_], 
                    self.num_nodes[from_:to_], self.subgraph_indexes[from_:to_], graphfeatures)

        for i in range(from_, to_):
            adj, x, num_node, indexes = self.process(i, self_for_none, None, bfs, ignore_isolate_nodes)
            adj_s.append(adj)
            x_s.append(x)
            num_nodes.append(num_node)
            subgraph_indexes.append(indexes)
            
        # 确保所有特征张量具有相同的最后一个维度
        x_s = self.ensure_consistent_feature_dims(x_s)
        
        return adj_s, x_s, num_nodes, subgraph_indexes
    
    def ensure_consistent_feature_dims(self, x_list):
        """
        确保列表中所有张量的最后一个维度相同
        """
        if not x_list:
            return x_list
            
        # 找出所有张量中最后一个维度的最大值
        max_dim = max([x.shape[-1] for x in x_list])
        
        # 确保所有张量的最后一个维度相同
        for i in range(len(x_list)):
            if x_list[i].shape[-1] < max_dim:
                # 使用零填充来匹配最大维度
                padding = torch.zeros(x_list[i].shape[0], max_dim - x_list[i].shape[-1], 
                                      device=x_list[i].device)
                x_list[i] = torch.cat([x_list[i], padding], dim=1)
                
        return x_list

    def get_max_degree(self):
        return np.max([adj.sum(-1) for adj in self.processed_adjs])

    def processALL(self, self_for_none, bfs=None, ignore_isolate_nodes=False):
        self.adj_s = []
        self.x_s = []
        self.num_nodes = []
        self.subgraph_indexes = []
        for i in range(len(self.list_adjs)):
            adj, x, num_node, indexes = self.process(i, self_for_none, None, bfs, ignore_isolate_nodes)
            self.adj_s.append(adj)
            self.x_s.append(x)
            self.num_nodes.append(num_node)
            self.subgraph_indexes.append(indexes)

    def __len__(self):
        return len(self.list_adjs)

    def process(self, index, self_for_none, padded_to=None, bfs_max_length=None, ignore_isolate_nodes=True, use_base_features=False):
        """
        处理单个图，提取节点特征和结构信息
        
        参数:
        - index: 图的索引
        - self_for_none: 是否为缺失值使用自环
        - padded_to: 填充大小（如果为None则使用默认值）
        - bfs_max_length: BFS最大长度
        - ignore_isolate_nodes: 是否忽略孤立节点
        - use_base_features: 是否使用基础特征(True表示同时使用基础特征和子结构特征，False表示仅使用子结构特征)
        """
        if bfs_max_length:
            bfs_max_length = min(bfs_max_length, self.max_num_nodes)
        num_nodes = self.list_adjs[index].shape[0]
        max_num_nodes = self.max_num_nodes if self.padding else num_nodes

        # 对邻接矩阵进行填充
        adj_padded = lil_matrix((max_num_nodes, max_num_nodes), dtype=np.float32)
        if max_num_nodes == num_nodes:
            adj_padded[:num_nodes, :num_nodes] = self.list_adjs[index].tolil()
        else:
            adj_padded[:num_nodes, :num_nodes] = self.list_adjs[index]
        adj_padded.setdiag(0)
        nodeDegree = adj_padded.sum(-1).A1
        if not ignore_isolate_nodes:
            nodeDegree += 1
        adj_padded.setdiag(1)

        # 获取原始节点数
        n = self.list_adjs[index].shape[0]
        
        # 基础特征部分 - 如果use_base_features为True
        if use_base_features:
            base_features = np.identity(max_num_nodes, dtype=np.float32)
            featureVec = adj_padded.sum(1).A1.reshape(-1, 1) / max_num_nodes
            X_base = np.concatenate([base_features, featureVec], axis=1)
        
        # 子结构特征部分 - 从图结构中提取
        G = nx.from_scipy_sparse_array(self.list_adjs[index])
        
        # 检测子结构特征
        sub_feats = detect_substructures(G, 
                substructure_types=[
                    'triangle_count',
                    '5cycle_detection',
                    '6cycle_detection',
                    '7cycle_detection',
                    'laplacian_features',
                    'diffusion_features',
                    'clustering_coef', 
                    'star_indicator',
                    'clique_indicator', 
                    'chain_indicator', 
                    '4cycle_count',
                    'betweenness_centrality',
                    'eigenvector_centrality',
                    'pagerank',
                    'bridge_nodes',
                    'kcore_number',
                    'articulation_points',
                    'component_size',
                    'small_world_index'
                ])

        # 构建子结构特征列表
        subfeature_list = []
        # 用于存储每种特征的最大值，用于归一化
        feature_max_values = {}
        
        # 第一次遍历: 找出每种特征的最大值
        for key in sub_feats.keys():
            feature_shape = sub_feats[key].shape
            
            if len(feature_shape) > 1 and feature_shape[1] > 1:
                # 多列特征，为每列单独计算最大值
                for col in range(feature_shape[1]):
                    column_key = f"{key}_col{col}"
                    values = sub_feats[key][:, col].flatten()
                    max_val = np.max(np.abs(values)) if len(values) > 0 else 1.0
                    feature_max_values[column_key] = max_val if max_val > 0 else 1.0
            else:
                # 单列特征
                values = sub_feats[key].flatten()
                max_val = np.max(np.abs(values)) if len(values) > 0 else 1.0
                feature_max_values[key] = max_val if max_val > 0 else 1.0
        
        # 第二次遍历: 归一化并添加到特征列表
        # 特征归一化优化 - 使用更稳健的方法
        def normalize_feature(values, max_val, eps=1e-8):
            """使用更稳定的方式进行特征归一化"""
            # 添加小值防止除零
            safe_max = max(abs(max_val), eps)
            # 使用tanh进行平滑缩放到[-1,1]区间
            return np.tanh(values / safe_max)

        # 在process函数中替换原有的归一化代码
        for key in sub_feats.keys():
            feature_shape = sub_feats[key].shape
            
            if len(feature_shape) > 1 and feature_shape[1] > 1:
                # 处理多列特征
                for col in range(feature_shape[1]):
                    column_key = f"{key}_col{col}"
                    feat = np.zeros((max_num_nodes, 1), dtype=np.float32)
                    values = sub_feats[key][:, col].flatten()
                    # 使用改进的归一化
                    normalized_values = normalize_feature(values, feature_max_values[column_key])
                    feat[:n, 0] = normalized_values
                    subfeature_list.append(feat)
            else:
                # 处理单列特征
                feat = np.zeros((max_num_nodes, 1), dtype=np.float32)
                values = sub_feats[key].flatten()
                # 使用改进的归一化
                normalized_values = normalize_feature(values, feature_max_values[key])
                feat[:n, 0] = normalized_values
                subfeature_list.append(feat)
        
        # 拼接所有子结构特征
        sub_features_padded = np.concatenate(subfeature_list, axis=1)

        # 最终节点特征：根据use_base_features决定是否包含基础特征
        if use_base_features:
            enriched_X = np.concatenate([X_base, sub_features_padded], axis=1)
        else:
            enriched_X = sub_features_padded
        
        X_tensor = torch.tensor(enriched_X).float()

        # BFS 索引（可选）
        bfs_indexes = set()
        if bfs_max_length is not None:
            while len(bfs_indexes) < bfs_max_length:
                indexes = set(range(adj_padded.shape[0])).difference(bfs_indexes).difference(np.where(nodeDegree == 0)[0])
                if not indexes:  # 如果没有剩余的非零度节点
                    break
                source_idx = list(indexes)[np.random.randint(len(indexes))]
                bfs_index = sp.csgraph.breadth_first_order(adj_padded, source_idx, return_predecessors=False)
                portion_size = min(len(bfs_index), bfs_max_length // 5)
                if portion_size + len(bfs_indexes) >= bfs_max_length:
                    bfs_indexes.update(bfs_index[:bfs_max_length - len(bfs_indexes)])
                else:
                    bfs_indexes.update(bfs_index[:portion_size])
            bfs_indexes = list(bfs_indexes)
        if len(bfs_indexes) == 0:
            bfs_indexes = list(range(max_num_nodes))
                
        return adj_padded, X_tensor, num_nodes, bfs_indexes

    def shuffle(self):
        indx = list(range(len(self.list_adjs)))
        np.random.shuffle(indx)
        if self.list_Xs is not None:
            self.list_Xs = [self.list_Xs[i] for i in indx]
        else:
            warnings.warn("X is empty")
        self.list_adjs = [self.list_adjs[i] for i in indx]
        if self.featureList is not None:
            for i, element in enumerate(self.featureList):
                self.featureList[i] = element[indx]
        else:
            warnings.warn("Graph structural feature is an empty set")
        if self.labels is not None:
            self.labels = [self.labels[i] for i in indx]
        else:
            warnings.warn("Label is an empty set")
        if len(self.subgraph_indexes) > 0:
            self.adj_s = [self.adj_s[i] for i in indx]
            self.x_s = [self.x_s[i] for i in indx]
            self.num_nodes = [self.num_nodes[i] for i in indx]
            self.subgraph_indexes = [self.subgraph_indexes[i] for i in indx]

    def __getitem__(self, index):
        return self.processed_adjs[index], self.processed_Xs[index]

def extract_common_substructures(graph_list, min_size=3, max_size=10, random_seed=42):
    """
    提取数据集中的所有不同子结构，使用严格的图同构检测进行去重
    """
    import networkx as nx
    from scipy.sparse import csr_matrix, spmatrix
    from collections import defaultdict
    import numpy as np
    import random
    
    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # 转换图为NetworkX格式
    nx_graphs = []
    for g in graph_list:
        try:
            if hasattr(g, 'toarray') or isinstance(g, spmatrix):
                nx_graphs.append(nx.from_scipy_sparse_array(g))
            elif hasattr(g, 'nodes'):
                nx_graphs.append(g)
            else:
                print(f"跳过无法处理的类型: {type(g)}")
        except Exception as e:
            print(f"转换图时出错: {e}")
            continue
    
    # 用于存储实例
    substructure_instances = defaultdict(list)
    
    # --- 辅助函数，与之前相同 ---
    def canonical_tuple(nodes):
        return tuple(sorted(nodes))
    
    def canonical_cycle(cycle):
        cycle = list(cycle)
        n = len(cycle)
        rotations = [tuple(cycle[i:] + cycle[:i]) for i in range(n)]
        rev = list(reversed(cycle))
        rev_rotations = [tuple(rev[i:] + rev[:i]) for i in range(n)]
        return min(rotations + rev_rotations)
    
    def find_simple_paths(G, min_size, max_size):
        # 代码与之前相同...
        paths = set()
        visited_paths = set()
        
        def dfs(path):
            if len(path) > max_size:
                return
            if len(path) >= min_size:
                candidate = tuple(path)
                candidate_rev = tuple(reversed(path))
                canonical = candidate if candidate <= candidate_rev else candidate_rev
                if canonical not in visited_paths:
                    paths.add(canonical)
                    visited_paths.add(canonical)
            
            last = path[-1]
            for neighbor in G.neighbors(last):
                if neighbor in path:
                    continue
                dfs(path + [neighbor])
                
        nodes_to_process = list(G.nodes())
        if len(nodes_to_process) > 100:
            random.shuffle(nodes_to_process)
            nodes_to_process = nodes_to_process[:100]
            
        for start in nodes_to_process:
            dfs([start])
            
        return list(paths)
    
    # 子结构提取部分 (与之前相同，保留所有5个提取模块)
    # --- 1. 提取三角形 ---
    print("正在提取三角形结构...")
    for g_idx, G in enumerate(nx_graphs):
        nodes_to_process = list(G.nodes())
        if len(nodes_to_process) > 1000:
            random.shuffle(nodes_to_process)
            nodes_to_process = nodes_to_process[:1000]
            
        for node in nodes_to_process:
            neighbors = list(G.neighbors(node))
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if G.has_edge(neighbors[i], neighbors[j]):
                        tri = canonical_tuple([node, neighbors[i], neighbors[j]])
                        substructure_instances["triangle"].append((g_idx, tri))

    # --- 2. 提取团结构（clique） ---
    print("正在提取团结构...")
    for g_idx, G in enumerate(nx_graphs):
        try:
            if G.number_of_nodes() > 200:
                subgraph_nodes = random.sample(list(G.nodes()), min(200, G.number_of_nodes()))
                G_sub = G.subgraph(subgraph_nodes).copy()
            else:
                G_sub = G
            for clique in nx.find_cliques(G_sub):
                if min_size <= len(clique) <= max_size:
                    clique_can = canonical_tuple(clique)
                    key = f"clique_{len(clique)}"
                    substructure_instances[key].append((g_idx, clique_can))
        except Exception as e:
            print(f"提取团结构时出错: {e}")
            continue

    # --- 3. 提取环结构 ---
    print("正在提取环结构...")
    for g_idx, G in enumerate(nx_graphs):
        try:
            if G.number_of_nodes() <= 150:
                try:
                    cycles = nx.cycle_basis(G)
                    for cycle in cycles:
                        if min_size <= len(cycle) <= max_size:
                            cycle_can = canonical_cycle(cycle)
                            key = f"cycle_{len(cycle)}"
                            substructure_instances[key].append((g_idx, cycle_can))
                except Exception as e:
                    print(f"计算环基时出错: {e}")
            elif G.number_of_nodes() <= 500:
                for _ in range(5):
                    try:
                        subgraph_nodes = random.sample(list(G.nodes()), min(100, G.number_of_nodes()))
                        G_sub = G.subgraph(subgraph_nodes).copy()
                        cycles = nx.cycle_basis(G_sub)
                        for cycle in cycles:
                            if min_size <= len(cycle) <= max_size:
                                cycle_can = canonical_cycle(cycle)
                                key = f"cycle_{len(cycle)}"
                                substructure_instances[key].append((g_idx, cycle_can))
                    except Exception:
                        continue
        except Exception:
            continue

    # --- 4. 提取星形结构 ---
    print("正在提取星形结构...")
    for g_idx, G in enumerate(nx_graphs):
        nodes_to_process = list(G.nodes())
        if len(nodes_to_process) > 1000:
            random.shuffle(nodes_to_process)
            nodes_to_process = nodes_to_process[:1000]
        for node in nodes_to_process:
            neighbors = list(G.neighbors(node))
            if len(neighbors) >= min_size - 1:  # 中心节点+邻居节点 >= min_size
                is_star = True
                for idx in range(len(neighbors)):
                    for jdx in range(idx + 1, len(neighbors)):
                        if G.has_edge(neighbors[idx], neighbors[jdx]):
                            is_star = False
                            break
                    if not is_star:
                        break
                if is_star:
                    star_nodes = canonical_tuple([node] + neighbors)
                    key = f"star_{len(star_nodes)}"
                    substructure_instances[key].append((g_idx, star_nodes))

    # --- 5. 提取链（路径）结构 ---
    print("正在提取链结构...")
    for g_idx, G in enumerate(nx_graphs):
        if G.number_of_nodes() <= 50:
            chains = find_simple_paths(G, min_size, max_size)
            for chain in chains:
                key = f"chain_{len(chain)}"
                substructure_instances[key].append((g_idx, chain))
        elif G.number_of_nodes() <= 200:
            start_nodes = random.sample(list(G.nodes()), min(30, G.number_of_nodes()))
            G_sub = G.copy()
            for start_node in start_nodes:
                nodes_in_radius = {start_node}
                frontier = {start_node}
                for _ in range(2):
                    new_frontier = set()
                    for node in frontier:
                        new_frontier.update(G_sub.neighbors(node))
                    frontier = new_frontier - nodes_in_radius
                    nodes_in_radius.update(frontier)
                if len(nodes_in_radius) > 50:
                    nodes_in_radius = set(random.sample(list(nodes_in_radius), 50))
                local_g = G_sub.subgraph(nodes_in_radius).copy()
                try:
                    local_chains = find_simple_paths(local_g, min_size, max_size)
                    for chain in local_chains:
                        key = f"chain_{len(chain)}"
                        substructure_instances[key].append((g_idx, chain))
                except Exception as e:
                    print(f"提取链结构时出错: {e}")
    
    # --- 使用严格的图同构检测进行去重 ---
    print("开始使用图同构检测进行去重...")
    
    # 存储不同同构类的代表图
    unique_representatives = []  # 存储代表图的列表
    unique_types = []  # 存储对应的子结构类型
    isomorphism_count = 0  # 记录同构数量
    
    # 为每种类型分别处理
    for subtype in sorted(substructure_instances.keys()):
        instances = substructure_instances[subtype]
        print(f"  - {subtype}: 发现 {len(instances)} 个实例，开始去重")
        
        # 防止实例过多导致处理时间过长
        max_to_process = 5000
        if len(instances) > max_to_process:
            random.shuffle(instances)
            instances = instances[:max_to_process]
            print(f"    由于实例过多，随机采样 {max_to_process} 个进行处理")
        
        # 处理每个实例
        for g_idx, nodes in instances:
            if not isinstance(nodes, (list, tuple)):
                continue
            
            # 创建子图
            G_sub = nx_graphs[g_idx].subgraph(list(nodes)).copy()
            
            # 确保子图是连通的
            if not nx.is_connected(G_sub):
                continue
            
            # 标准化节点标签
            G_canon = nx.convert_node_labels_to_integers(G_sub, ordering="sorted")
            
            # 检查是否与已有代表图同构
            is_isomorphic_to_existing = False
            for rep_idx, rep_graph in enumerate(unique_representatives):
                if nx.is_isomorphic(G_canon, rep_graph):
                    is_isomorphic_to_existing = True
                    isomorphism_count += 1
                    break
            
            # 如果不同构于任何已有代表图，则添加为新的代表图
            if not is_isomorphic_to_existing:
                adj_sub = nx.adjacency_matrix(G_sub, nodelist=list(nodes))
                unique_representatives.append(G_canon)
                unique_types.append(subtype)

    
    # 从去重后的代表图中提取邻接矩阵
    selected_substructures = []
    for i, G_rep in enumerate(unique_representatives):
        # 转换为邻接矩阵
        adj = nx.adjacency_matrix(G_rep)
        selected_substructures.append(adj)
    
    selected_types = unique_types
    
    print(f"去重后共有 {len(selected_substructures)} 个不同子结构，覆盖 {len(set(selected_types))} 种类型")
    print(f"去重过程中共发现 {isomorphism_count} 个同构(重复)实例")
    
    # 打印各类型数量统计
    type_count = defaultdict(int)
    for t in selected_types:
        type_count[t] += 1
    
    for t, count in sorted(type_count.items()):
        print(f"  - {t}: {count} 个不同实例")
    
    return selected_substructures, selected_types




# -------------------- 数据集加载函数 --------------------
def list_graph_loader(graph_type, _max_list_size=None, return_labels=False, limited_to=None, augment_with_substructures=True):
    list_adj = []
    list_x = []
    list_labels = []
    
    # 通过 GIN 数据集加载 NCI1、MUTAG、PTC、PROTEINS 数据集
    def load_gin_dataset(name):
        data = dgl.data.GINDataset(name=name, self_loop=False)
        graphs, labels = data.graphs, data.labels
        for i, graph in enumerate(graphs):
            list_adj.append(csr_matrix(graph.adjacency_matrix().to_dense().numpy()))
            list_x.append(None)
            list_labels.append(labels[i].item())
    
    def load_qm9_dataset():
        data = dgl.data.QM9Dataset(label_keys=['mu'])
        for graph in data:
            adj = dgl.to_homogeneous(graph[0]).adjacency_matrix().to_dense().numpy()
            list_adj.append(csr_matrix(adj))
            list_x.append(None)
    
    def load_ogbg_molbbbp():
        dataset = DglGraphPropPredDataset(name="ogbg-molbbbp")
        for graph, label in dataset:
            list_adj.append(csr_matrix(graph.adjacency_matrix().to_dense().numpy()))
            list_x.append(None)
            list_labels.append(label.item())
    
    def load_grid_graphs():
        for i in range(10, 20):
            for j in range(10, 20):
                list_adj.append(nx.adjacency_matrix(nx.grid_2d_graph(i, j)))
                list_x.append(None)
    
    def load_triangular_grid_graphs():
        for i in range(10, 20):
            for j in range(10, 20):
                list_adj.append(nx.adjacency_matrix(nx.triangular_lattice_graph(i, j)))
                list_x.append(None)
    
    def load_community_graphs():
        for i in range(30, 81):
            for j in range(30, 81):
                list_adj.append(nx.adjacency_matrix(nx.random_partition_graph([i, j], p_in=0.3, p_out=0.05)))
                list_x.append(None)
    
    def load_lobster_graphs():
        count = 0
        min_node = 10
        max_node = 100
        mean_node = 80
        num_graphs = 100
        seed = 1234
        while count < num_graphs:
            G = nx.random_lobster(mean_node, 0.7, 0.7, seed=seed)
            if min_node <= len(G.nodes) <= max_node:
                list_adj.append(nx.adjacency_matrix(G))
                list_x.append(None)
                count += 1
            seed += 1
    
    # 通过 Graph_load 加载 citeseer 数据集
    def load_citeseer():
        adj, features, _ = Graph_load(dataset='citeseer')
        return return_subset([adj], [features], None, limited_to)
    
    # 仅保留所需数据集对应的加载函数
    graph_loaders = {
        "citeseer": load_citeseer,
        "NCI1": lambda: load_gin_dataset('NCI1'),
        "MUTAG": lambda: load_gin_dataset('MUTAG'),
        "PTC": lambda: load_gin_dataset('PTC'),
        "PROTEINS": lambda: load_gin_dataset('PROTEINS'),
        "QM9": load_qm9_dataset,
        "ogbg-molbbbp": load_ogbg_molbbbp,
        "grid": load_grid_graphs,
        "triangular_grid": load_triangular_grid_graphs,
        "community": load_community_graphs,
        "lobster": load_lobster_graphs
    }
    
    if graph_type in graph_loaders:
        result = graph_loaders[graph_type]()
        if graph_type == "citeseer":
            return result
    else:
        raise ValueError(f"Unsupported graph type: {graph_type}")

    if return_labels and not list_labels:
        list_labels = None
    
    return return_subset(list_adj, list_x, list_labels, limited_to)

# -------------------- 数据划分函数 --------------------
def data_split(graph_list, list_x=None, list_label=None, train_ratio=0.8):
    random.seed(123)
    index = list(range(len(graph_list)))
    random.shuffle(index)
    graph_list = [graph_list[i] for i in index]
    if list_x is not None:
        list_x = [list_x[i] for i in index]
    if list_label is not None:
        list_label = [list_label[i] for i in index]
    train_size = int(train_ratio * len(graph_list))
    graph_train = graph_list[:train_size]
    graph_test = graph_list[train_size:]
    list_x_train = list_x_test = None
    if list_x is not None:
        list_x_train = list_x[:train_size]
        list_x_test = list_x[train_size:]
    list_label_train = list_label_test = None
    if list_label is not None:
        list_label_train = list_label[:train_size]
        list_label_test = list_label[train_size:]
    return graph_train, graph_test, list_x_train, list_x_test, list_label_train, list_label_test

# -------------------- 核心辅助函数 --------------------
def BFS(list_adj):
    for i in range(len(list_adj)):
        bfs_index = sp.csgraph.breadth_first_order(list_adj[i], 0, return_predecessors=False)
        list_adj[i] = list_adj[i][bfs_index, :][:, bfs_index]
    return list_adj

def BFSWithAug(list_adj, X_s, label_s, number_of_per=1):
    list_adj_, X_s_, label_s_ = [], [], []
    for _ in range(number_of_per):
        for i, adj in enumerate(list_adj):
            non_isolated_nodes = [x for x in range(adj.shape[0]) if adj[x].sum() >= 1]
            if not non_isolated_nodes:  # 如果没有非孤立节点
                node_i = 0  # 使用第一个节点
            else:
                node_i = random.choice(non_isolated_nodes)
            bfs_index = sp.csgraph.breadth_first_order(adj, node_i, return_predecessors=False)
            list_adj_.append(adj[bfs_index, :][:, bfs_index])
            X_s_.append(X_s[i])
            if label_s:
                label_s_.append(label_s[i])
    if not label_s_:
        label_s_ = label_s
    return list_adj_, X_s_, label_s_

def permute(list_adj, X):
    for i in range(len(list_adj)):
        p = np.random.permutation(list_adj[i].shape[0])
        list_adj[i] = list_adj[i][p, :][:, p]
        if X is not None:
            X[i] = X[i][p, :]
    return list_adj, X

def node_feature_creator(adj_in, steps=3, rand_dim=0, use_identity=False, norm=None, uniform_size=False):
    if norm is None:
        norm = adj_in.shape[0]
    if uniform_size:
        adj = csr_matrix((norm, norm))
        adj[:adj_in.shape[0], :adj_in.shape[1]] = adj_in
    else:
        adj = adj_in
    traverse_matrix = adj
    feature_vec = [np.array(adj.sum(1)) / norm]
    for i in range(steps):
        traverse_matrix = traverse_matrix.dot(adj.transpose())
        feature = traverse_matrix.diagonal().reshape(-1, 1)
        feature_vec.append(feature / norm**(i + 1))
    if rand_dim > 0:
        np.random.seed(0)
        feature_vec.append(np.random.rand(adj.shape[-1], rand_dim))
    if use_identity:
        feature_vec.append(np.identity(norm))
    return np.concatenate(feature_vec, axis=1)

def pad_adj_to(adj_list, size):
    uniformed_list = []
    for adj in adj_list:
        adj_padded = lil_matrix((size, size))
        adj_padded[:adj.shape[0], :adj.shape[1]] = adj
        adj_padded.setdiag(1)
        uniformed_list.append(adj_padded)
    return uniformed_list

def BFS_Permute(adj_s, x_s, target_kernel_val):
    for i in range(len(adj_s)):
        degree = np.array(adj_s[i].sum(0)).reshape(-1)
        connected_node = np.where(degree > 1)[0]
        unconnected_nodes = np.where(degree == 1)[0]
        
        if len(connected_node) > 0:  # 如果有连通节点
            bfs_index = sp.csgraph.breadth_first_order(adj_s[i], random.choice(connected_node), return_predecessors=False)
            bfs_index = list(np.unique(bfs_index)) + list(unconnected_nodes)
            adj_s[i] = adj_s[i][bfs_index, :][:, bfs_index]
            x_s[i] = x_s[i][bfs_index, :]
            for j in range(len(target_kernel_val) - 2):
                target_kernel_val[j][i] = target_kernel_val[j][i][bfs_index, :][:, bfs_index]
    
    return adj_s, x_s, target_kernel_val

# -------------------- 主函数 --------------------
if __name__ == '__main__':
    # 示例：依次加载每个目标数据集，并输出加载的图数量
    datasets = ["lobster", "citeseer", "community", "ogbg-molbbbp", "NCI1", 
                "triangular_grid", "PTC", "PROTEINS", "MUTAG", "QM9", "grid"]
    for ds in datasets:
        print(f"Loading dataset: {ds}")
        try:
            result = list_graph_loader(ds)
            list_adj, list_x, list_labels = result
            print(f"{ds}: {len(list_adj)} graphs loaded")
        except Exception as e:
            print(f"Error loading {ds}: {e}")