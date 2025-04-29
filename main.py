# main_new_0320.py
import argparse
import itertools
import json
import logging
import os
import pickle
import random
import time
import timeit
from pathlib import Path

import dgl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

import plotter
from data_MM import *
from evaluation.evaluator import Evaluator
from evaluation.gin_evaluation import load_feature_extractor, MMDEvaluation, prdcEvaluation
from model_new_0320 import *
from stat_rnn import mmd_eval



class kernel(torch.nn.Module):
    def __init__(self, device, kernel_type, step_num, bin_width, bin_center, degree_bin_center, degree_bin_width):
        super(kernel, self).__init__()
        self.device = device
        self.kernel_type = kernel_type
        kernel_set = set(kernel_type)

        if "in_degree_dist" in kernel_set or "out_degree_dist" in kernel_set:
            self.degree_hist = Histogram(self.device, degree_bin_width.to(self.device), degree_bin_center.to(self.device))

        if "trans_matrix" in kernel_set:
            self.num_of_steps = step_num

    def forward(self, adj):
        return self.kernel_function(adj)

    def kernel_function(self, adj):
        vec = []
        for kernel in self.kernel_type:
            if kernel == "TotalNumberOfTriangles":
                vec.append(self.TotalNumberOfTriangles(adj))
            elif kernel == "in_degree_dist":
                degree_hit = [self.degree_hist(adj[i].sum(1).view(1, -1).to(self.device)) for i in range(adj.shape[0])]
                vec.append(torch.cat(degree_hit))
            elif kernel == "out_degree_dist":
                degree_hit = [self.degree_hist(adj[i].sum(0).view(1, -1).to(self.device)) for i in range(adj.shape[0])]
                vec.append(torch.cat(degree_hit))
            elif kernel == "trans_matrix":
                vec.extend(self.S_step_trasition_probablity(adj, self.num_of_steps))
            elif kernel == "tri":
                tri, square = self.tri_square_count(adj)
                vec.append(tri)
                vec.append(square)
            elif kernel == "TrianglesOfEachNode":
                vec.append(self.TrianglesOfEachNode(adj))
            elif kernel == "ThreeStepPath":
                vec.append(self.TreeStepPathes(adj))
        return vec

    def tri_square_count(self, adj):
        ind = torch.eye(adj[0].shape[0], device=self.device)
        adj = adj - ind
        two_step = torch.matmul(adj, adj)
        tri = torch.matmul(two_step, adj)
        squares = torch.matmul(two_step, two_step)
        return torch.diagonal(tri, dim1=1, dim2=2), torch.diagonal(squares, dim1=1, dim2=2)

    def S_step_trasition_probablity(self, adj, s=4, dataset_scale=None):
        p1 = adj.to(self.device)
        TP_list = []

        if dataset_scale == "large":
            p1 = torch.stack([p1[i] * (p1[i].sum(1, keepdim=True).clamp(min=1).float().reciprocal()) for i in range(adj.shape[0])])
        else:
            p1 = p1 * p1.sum(2, keepdim=True).clamp(min=1).float().reciprocal().view(adj.shape[0], adj.shape[1], 1)

        if s > 0:
            TP_list.append(p1)
        for _ in range(s - 1):
            TP_list.append(torch.matmul(p1, TP_list[-1]))

        return TP_list

    def TrianglesOfEachNode(self, adj):
        p1 = adj.to(self.device)
        p1 = p1 * (1 - torch.eye(adj.shape[-1], device=self.device))
        tri = torch.diagonal(torch.matmul(p1, torch.matmul(p1, p1)), dim1=-2, dim2=-1) / 6
        return tri

    def TreeStepPathes(self, adj):
        p1 = adj.to(self.device)
        p1 = p1 * (1 - torch.eye(adj.shape[-1], device=self.device))
        tri = torch.matmul(p1, torch.matmul(p1, p1))
        return tri

    def TotalNumberOfTriangles(self, adj):
        p1 = adj.to(self.device)
        p1 = p1 * (1 - torch.eye(adj.shape[-1], device=self.device))
        tri = torch.diagonal(torch.matmul(p1, torch.matmul(p1, p1)), dim1=-2, dim2=-1) / 6
        return tri.sum(-1)

# Histogram 类定义
class Histogram(torch.nn.Module):
    def __init__(self, device, bin_width=None, bin_centers=None):
        super(Histogram, self).__init__()
        self.device = device
        self.bin_width = bin_width.to(self.device)
        self.bin_center = bin_centers.to(self.device)
        self.bin_num = self.bin_width.shape[0] if self.bin_width is not None else None

    def forward(self, vec):
        score_vec = vec.view(vec.shape[0], 1, vec.shape[1]) - self.bin_center
        score_vec = 1 - torch.abs(score_vec) * self.bin_width
        score_vec = torch.relu(score_vec)
        return score_vec.sum(2)

    def prism(self):
        pass

        


# 设置随机种子，确保结果可复现
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 全局设置
keepThebest = False


def test_(graph_save_path, epoch, num_samples, model, path_to_save, remove_self=True, save_graphs=True, picture=0, load_tag='Final20000'):
    """生成图，并保存部分可视化结果"""
    os.makedirs(path_to_save, exist_ok=True)
    generated_graph_list = []
    z = torch.randn(num_samples, model.n_topics, device=model.device)
    adj, theta, topics, z_qn_adj, test_mapping_matrix = model.generate(num_samples, z.float())
    reconstructed_adj = adj.cpu().detach().numpy()
    for j in range(num_samples):
        sample_graph = (reconstructed_adj[j] > 0.5).astype(int)
        G = nx.from_numpy_array(sample_graph)
        G.remove_edges_from(nx.selfloop_edges(G))
        G.remove_nodes_from(list(nx.isolates(G)))
        generated_graph_list.append(G)
        if picture == 1:
            plotter.plotG(G, f"test{load_tag}",
                          file_name=os.path.join(graph_save_path, f"{epoch}j{j}_{load_tag}.png"))
            plotter.plot_subgraph_test(z_qn_adj[j].cpu().detach().numpy(), graph_save_path, epoch, j, 1, topics[j], load_tag)
    return generated_graph_list


def get_subGraph_features(org_adj, kernel_model, device):
    """计算子图的核特征"""
    subgraphs = [torch.tensor(subGraph.todense(), dtype=torch.float32).to(device) for subGraph in org_adj]
    subgraphs = torch.stack(subgraphs)
    target_kernel_val = kernel_model(subgraphs) if kernel_model is not None else None
    return target_kernel_val, subgraphs


def softclip(tensor, min_val):
    """Soft-clips tensor的值，确保不低于 min_val"""
    return min_val + F.softplus(tensor - min_val)


def log_guss(mean, log_std, samples):
    log_std_exp = log_std.exp().to(mean.device)
    return 0.5 * torch.pow((samples - mean) / log_std_exp, 2) + log_std + 0.5 * torch.log(2 * torch.pi * torch.ones(1, device=mean.device))


def OptimizerVAE(reconstructed_adj, reconstructed_kernel_val, target_adj, target_kernel_val, reconstructed_adj_logit, pos_weight, device):
    loss = F.binary_cross_entropy_with_logits(reconstructed_adj_logit.float(), target_adj.float(), pos_weight=pos_weight)
    kernel_diff = 0
    for i in range(len(target_kernel_val)):
        log_sigma = ((reconstructed_kernel_val[i] - target_kernel_val[i]) ** 2).mean().sqrt().log().to(device)
        log_sigma = softclip(log_sigma, -6)
        step_loss = log_guss(target_kernel_val[i], log_sigma, reconstructed_kernel_val[i]).mean()
        kernel_diff += step_loss
    return loss, kernel_diff


def parse_args():
    parser = argparse.ArgumentParser(description='Kernel VGAE')
    parser.add_argument('-ALTER_TRAIN', action='store_true', default=False)
    parser.add_argument('--dec_train', type=int, default=10)
    parser.add_argument('-e', dest="epoch_number", default=20000, type=int, help="Number of Epochs to train the model")
    parser.add_argument('-v', dest="Vis_step", default=20000, type=int, help="Visualization update step")
    parser.add_argument('-save', dest="save_step", default=500, type=int, help="Save model every n epochs")
    parser.add_argument('-redraw', dest="redraw", action='store_true', default=False, help="Update log plot each step")
    parser.add_argument('-lr', dest="lr", type=float, default=0.0003)
    
    parser.add_argument('-dataset', dest="dataset", default="MUTAG",
                        help="Dataset name")#ogbg-molbbbp
    parser.add_argument('-graphEmDim', dest="graphEmDim", type=int, default=512, help="Graph embedding dimension")
    parser.add_argument('-graph_save_path', dest="graph_save_path", default=None, help="Directory to save generated graphs")
    parser.add_argument('-batchSize', dest="batchSize", type=int, default=300, help="Mini-batch size")
    parser.add_argument('-model', dest="model", default="NVGM", help="Model type (default: NVGM)")
    parser.add_argument('-ideal_Evalaution', dest="ideal_Evalaution", action='store_true', default=False,
                        help="Compare 50%/50% subset of dataset")
    parser.add_argument('--substructure_size', type=int, default=256, help="Substructure embedding size")
    parser.add_argument('--n_topics', type=int, default=10)
    parser.add_argument('--n_samples', type=int, default=20)
    parser.add_argument('--sub_num_nodes', type=int, default=10)
    parser.add_argument('--dim_hidden', type=int, default=256)
    
    parser.add_argument('-device', dest="device", default="cuda:6", help="Which device to use")
    parser.add_argument('-regularize', default="ortho")
    parser.add_argument('-BFS', dest="bfsOrdering", action='store_true', default=True, help="Use BFS for graph permutations")
    parser.add_argument('-lr_step', dest="lr_step", type=int, default=500)
    parser.add_argument('-lr_gamma', dest="lr_gamma", type=float, default=0.9)
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device

    learning_rates = [0.0003]
    n_topics_values = [5]
    n_samples_values = [20]
    sub_num_nodes_values = [10]
    lr_step_values = [0]
    lr_gamma_values = [0]
    
  
    re_a_values= [14000]
    kl1_a_values= [200]
    mkl_a_values= [55]
    ker_a_values= [20]
    or_a_values= [350]
    kl_m_a_values= [35]
    entropy_a_values= [120]



    all_combinations = list(itertools.product(
        learning_rates, re_a_values, kl1_a_values, mkl_a_values,
        ker_a_values, or_a_values, n_topics_values,
        n_samples_values, sub_num_nodes_values, lr_step_values, lr_gamma_values, kl_m_a_values, entropy_a_values))
    n_searches = 100
    selected_combinations = random.sample(all_combinations, min(n_searches, len(all_combinations)))

    for combination in selected_combinations:
        valid_combination = True
        (lr, re_a, kl1_a, mkl_a, ker_a, or_a, n_topics, n_samples,
        sub_num_nodes, lr_step, lr_gamma, kl_m_a, entropy_a) = combination

        # 使用 argparse 解析其它参数
        graphEmDim = args.graphEmDim
        vis_step = args.Vis_step
        redraw = args.redraw
        epoch_number = args.epoch_number
        dataset = args.dataset
        mini_batch_size = args.batchSize
        substructure_size = args.substructure_size
        self_for_none = True

        # 使用 f-string 格式化保存目录
        if args.graph_save_path is None:
            graph_save_path = (f"A/{lr_step}_{lr_gamma}_BFS{args.bfsOrdering}_"
                            f"{dataset}_{lr}_{re_a}_{kl1_a}_{mkl_a}_{ker_a}_{or_a}_{kl_m_a}_{entropy_a}_"
                            f"Topic{n_topics}_Samples{n_samples}_subnode{sub_num_nodes}_E{epoch_number}_{int(time.time())}/")
        else:
            graph_save_path = args.graph_save_path
        Path(graph_save_path).mkdir(parents=True, exist_ok=True)

        # 配置日志记录（每次重新设置日志handlers）
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=os.path.join(graph_save_path, 'alog.log'),
                            filemode='w', level=logging.INFO)

        print("KernelVGAE SETTING:", args)
        logging.info("KernelVGAE SETTING: %s", args)

        # 根据数据集设置核函数计算步数
        kernl_type = ["trans_matrix", "in_degree_dist", "out_degree_dist", "TotalNumberOfTriangles"]
        step_num = 2 if dataset == "QM9" else 5

        print("kernl_type:", kernl_type)
        print("Selected device:", device)
        logging.info("Selected device: %s", device)

        evaluations = ["20000-MMD_RBF-test-z", "20000-F1_PR-test-z"]
        pltr_eval = plotter.Plotter(save_to_filepath="kernelVGAE_Log", functions=evaluations)

        list_adj, list_x, list_label = list_graph_loader(dataset, return_labels=True)

        print('Train graphs:', len(list_adj))
        if args.bfsOrdering:
            list_adj = BFS(list_adj)


        list_adj, test_list_adj, list_x_train, list_x_test, _, list_label_test = data_split(list_adj, list_x, list_label)
        
        print('After split, train:', len(list_adj), "test:", len(test_list_adj))

        list_graphs = Datasets(list_adj, self_for_none, list_x_train, list_label, max_num=None, set_diag_of_isol_zer=False)
        list_test_graphs = Datasets(test_list_adj, self_for_none, list_x_test, list_label_test,
                                    max_num=list_graphs.max_num_nodes, set_diag_of_isol_zer=False)

        # 50%/50%数据集评估
        fifty_fifty_dataset = list_adj + test_list_adj
        fifty_fifty_dataset = [nx.from_numpy_array(graph.toarray()) for graph in fifty_fifty_dataset]
        random.shuffle(fifty_fifty_dataset)
        print("50%/50% Evaluation of dataset")
        logging.info("50%/50% Evaluation of dataset")
        mmd_res_fif = mmd_eval(fifty_fifty_dataset[:len(fifty_fifty_dataset)//2],
                            fifty_fifty_dataset[len(fifty_fifty_dataset)//2:], diam=True)
        logging.info(mmd_res_fif)
        with open(os.path.join(graph_save_path, '_MMD.log'), 'a') as f:
            f.write("\n@ 50%/50% Evaluation: " + mmd_res_fif + "\n\n")

        fifty1_dgl = [dgl.DGLGraph(g).to(device) for g in fifty_fifty_dataset[:len(fifty_fifty_dataset)//2]]
        fifty2_dgl = [dgl.DGLGraph(g).to(device) for g in fifty_fifty_dataset[len(fifty_fifty_dataset)//2:]]

        gin = load_feature_extractor(device=device)
        mmd_eval_GIN = MMDEvaluation(gin)
        result, time1 = mmd_eval_GIN.evaluate(generated_dataset=fifty1_dgl, reference_dataset=fifty2_dgl)
        with open(os.path.join(graph_save_path, '_MMD.log'), 'a') as f:
            f.write(f"\n@ 50%/50% Evaluation: result: {result}, time: {time1}\n")
        print('GIN eval: result: {}, time: {:.3f}s'.format(result, time1))

        f1_eval_GIN = prdcEvaluation(gin, use_pr=True)
        result, time1 = f1_eval_GIN.evaluate(generated_dataset=fifty1_dgl, reference_dataset=fifty2_dgl)
        with open(os.path.join(graph_save_path, '_MMD.log'), 'a') as f:
            f.write(f"\n@ 50%/50% Evaluation F1: result: {result}, time: {time1}\n\n")
        print('F1 eval: result: {}, time: {:.3f}s'.format(result, time1))
        del fifty_fifty_dataset

        # 构建核模型与 NVGM 模型
        MaxNodeNum = list_graphs.max_num_nodes
        in_feature_dim = list_graphs.feature_size
        print('MaxNodeNum:', MaxNodeNum)
        print('in_feature_dim:', in_feature_dim)

        degree_center = torch.tensor([[x] for x in range(MaxNodeNum)], device=device)
        degree_width = torch.tensor([[0.1] for _ in range(MaxNodeNum)], device=device)
        bin_center = torch.tensor([[x] for x in range(MaxNodeNum)], device=device)
        bin_width = torch.tensor([[1] for _ in range(MaxNodeNum)], device=device)

        kernel_model = kernel(device=device, kernel_type=kernl_type, step_num=step_num,
                            bin_width=bin_width, bin_center=bin_center,
                            degree_bin_center=degree_center, degree_bin_width=degree_width)

        model = NVGM(device, kernel_model, n_topics, in_feature_dim, MaxNodeNum, n_samples,
                    sub_num_nodes, substructure_size, [], hiddenLayers=[256, 256, 256], GraphLatntDim=graphEmDim)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr)

        print('Total parameters:', sum(param.numel() for param in model.parameters()))
        print('Trainable parameters:', sum(param.numel() for param in model.parameters() if param.requires_grad))
        logging.info(model.__str__())

        list_graphs.processALL(self_for_none=self_for_none)
        adj_list = list_graphs.get_adj_list()
        graphFeatures, _ = get_subGraph_features(adj_list, kernel_model, device)
        list_graphs.set_features(graphFeatures)

        # 保存初始主题向量以便后续分析
        with torch.no_grad():
            init_mu_mat = model.mu_mat.cpu().clone().numpy()
            np.save(os.path.join(graph_save_path, "init_topic_vectors.npy"), init_mu_mat)

        phase1 = int(epoch_number * 0.2)  # 阶段1: 0-30%, 只关注重建损失和核损失
        phase2 = int(epoch_number * 0.5)  # 阶段2: 30-60%, 加入KL损失
        phase3 = int(epoch_number * 0.7)  # 阶段3: 60-80%, 加入正交损失
        # 阶段4: 80-100%, 所有损失一起训练

        # phase1 = 0
        # phase2 = 0
        # phase3 = 0
        
        for epoch in range(epoch_number):
            start = timeit.default_timer()
            list_graphs.shuffle()
            batch = 0
            num_batches = max(int(len(list_graphs.list_adjs) / mini_batch_size), 1)
            epoch_losses = {"total": 0, "recon": 0, "kl_theta": 0, 
                            "mkl": 0, "kernel": 0, "ortho": 0, "kl_map": 0, 
                            "topic_entropy": 0}
            
            for iter in range(0, num_batches * mini_batch_size, mini_batch_size):
                from_ = iter
                to_ = mini_batch_size * (batch + 1)
                org_adj, x_s, node_num, _, target_kernel_val = list_graphs.get__(from_, to_, self_for_none)
                x_s = torch.cat(x_s).reshape(-1, x_s[0].shape[-1]).to(device)
                model.train()

                # 获得子图核特征（仅用于重构误差计算）
                _, subgraphs = get_subGraph_features(org_adj, None, device)
                
                batchSize = [len(org_adj), org_adj[0].shape[0]]
                org_adj_dgl = [dgl.from_scipy(graph) for graph in org_adj]
                org_adj_dgl = dgl.batch(org_adj_dgl).to(device)
                pos_weight = torch.true_divide(sum([x.shape[-1] ** 2 for x in subgraphs]) - subgraphs.sum(),
                                                subgraphs.sum()).to(device)
                
                # 注意这里将subgraphs作为real_adj参数传递
                (reconstructed_adj, post_mean, post_log_std, generated_kernel_val, reconstructed_adj_logit,
                mkl, kl_theta, ortho_loss, train_data_theta, train_data_z_qn_adj,
                z_qn_adj_topics, mapping_matrix, kl_map, topic_entropy_regularization) = model(
                    org_adj_dgl, x_s, batchSize, subgraphs)
                
                # 计算重建损失和核损失
                reconstruction_loss, kernel_cost = OptimizerVAE(reconstructed_adj, generated_kernel_val, 
                                                                subgraphs.to(device),
                                                                [val.to(device) for val in target_kernel_val],
                                                                reconstructed_adj_logit, pos_weight, device)
                
                # 根据训练阶段选择性加入各项损失
                if epoch < phase1:
                    loss = re_a * reconstruction_loss + ker_a * kernel_cost
                elif epoch < phase2:
                    loss = re_a * reconstruction_loss + ker_a * kernel_cost + kl1_a * kl_theta+ mkl_a * mkl + kl_m_a * kl_map
                elif epoch < phase3:
                    loss = re_a * reconstruction_loss + ker_a * kernel_cost + kl1_a * kl_theta + or_a * ortho_loss+ mkl_a * mkl + kl_m_a * kl_map
                else:
                    loss = (re_a * reconstruction_loss + kl1_a * kl_theta + mkl_a * mkl +
                            ker_a * kernel_cost + or_a * ortho_loss + kl_m_a * kl_map + 
                            entropy_a * topic_entropy_regularization)
                
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪，裁剪最大范数为10
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20)
                
                optimizer.step()
                batch += 1
                
                # 记录各损失项
                epoch_losses["total"] += loss.item()
                epoch_losses["recon"] += reconstruction_loss.item()
                epoch_losses["kl_theta"] += kl_theta.item()
                epoch_losses["mkl"] += mkl.item()
                epoch_losses["kernel"] += kernel_cost.item()
                epoch_losses["ortho"] += ortho_loss.item()
                epoch_losses["kl_map"] += kl_map.item()
                epoch_losses["topic_entropy"] += topic_entropy_regularization.item()
            
            # 计算平均损失
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
            
            stop = timeit.default_timer()
            print(f'Epoch: {epoch} | Training time: {stop - start:.3f}s | Loss: {epoch_losses["total"]:.4f}')
            logging.info("Epoch: {:03d} | Loss: {:.4f} | Recon: {:.4f} | KL_theta: {:.4f} | MKL: {:.4f} | Kernel: {:.4f} | Ortho: {:.4f} | KL_map: {:.4f} | Topic_entropy: {:.4f}".format(
                        epoch, epoch_losses["total"], epoch_losses["recon"], epoch_losses["kl_theta"], 
                        epoch_losses["mkl"], epoch_losses["kernel"], epoch_losses["ortho"], 
                        epoch_losses["kl_map"], epoch_losses["topic_entropy"]))
            
            print(f'Epoch: {epoch} | Training time: {stop - start:.3f}s')

            torch.save(model.state_dict(), os.path.join(graph_save_path, "model_final20000.pth"))

            if epoch % 2000 == 0 and epoch != 0:
                torch.save(model.state_dict(), os.path.join(graph_save_path, f"model_epoch{epoch}.pth"))


            # 可视化与中间评估
            if (epoch + 1) % vis_step == 0 or epoch == epoch_number - 1:
                model.eval()
                rnd_indx = random.randint(0, len(node_num) - 1)
                sample_graph = reconstructed_adj[rnd_indx].cpu().detach().numpy()
                sample_graph = np.where(sample_graph > 0.5, 1, 0)
                G = nx.from_numpy_array(sample_graph)
                G.remove_edges_from(nx.selfloop_edges(G))
                G.remove_nodes_from(list(nx.isolates(G)))
                if not nx.is_empty(G):
                    plotter.plotG(G, "generated" + dataset,
                                  file_name=os.path.join(graph_save_path, f"generatedSample_At_epoch{epoch}"))
                plotter.plot_subgraph(train_data_z_qn_adj[rnd_indx].cpu().detach().numpy(), graph_save_path, epoch, z_qn_adj_topics[rnd_indx])
                z_qn_adj_topic = model.generate_topics()
                plotter.plot_subgraph_topic(z_qn_adj_topic, graph_save_path, epoch)

            if epoch % args.save_step == 0 or epoch == epoch_number - 1:
                print("Epoch:", epoch, "Batch:", batch, "Loss:", loss.item())
                logging.info("Epoch: {:03d} | Batch: {:03d} | loss: {:.5f} | recon_loss: {:.5f} | z_kl: {:.5f} | Mixture_KL: {:.5f} | kernel_cost: {:.5f} | ortho_loss: {:.5f} | kl_map: {:.5f} | topic_entropy: {:.5f}".format(
                    epoch + 1, batch, loss.item(), reconstruction_loss.item(), kl_theta.item(), mkl.item(),
                    kernel_cost.item(), ortho_loss.item(), kl_map.item(), topic_entropy_regularization.item()))
                
                test_graphs = [nx.from_numpy_array(graph.toarray()) for graph in test_list_adj]
                test_graphs_dgl = [dgl.DGLGraph(g).to(device) for g in test_graphs]
                model.load_state_dict(torch.load(os.path.join(graph_save_path, "model_final20000.pth")))
                generated_graphs = test_(graph_save_path, epoch, len(test_list_adj), model, graph_save_path,
                                          save_graphs=True, picture=0, load_tag='Final20000')
                generated_graphs = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len)))
                                    for G in generated_graphs if not nx.is_empty(G)]
                if not generated_graphs or any(np.isnan(np.array(nx.adjacency_matrix(G).todense())).any() or
                                                np.isinf(np.array(nx.adjacency_matrix(G).todense())).any() for G in generated_graphs):
                    print("Generated graphs invalid at epoch:", epoch)
                    with open(os.path.join(graph_save_path, '_MMD.log'), 'a') as f:
                        f.write("\nGenerated graphs invalid at epoch: " + str(epoch) + "\n")
                    valid_combination = False
                    break

                mmd_res = mmd_eval(generated_graphs, [nx.from_numpy_array(graph.toarray()) for graph in test_list_adj], diam=True)
                if mmd_res in [1, 2, 3, 4, 5]:
                    print(f"mmd_res == {mmd_res} at epoch:", epoch)
                    with open(os.path.join(graph_save_path, '_MMD.log'), 'a') as f:
                        f.write(f"\nmmd_res == {mmd_res} at epoch: {epoch}\n")
                    valid_combination = False
                    break

                with open(os.path.join(graph_save_path, '_MMD.log'), 'a') as f:
                    f.write(f"\nEpoch: {epoch} @ Final Test: {mmd_res}\n")
                generated_graphs_dgl = [dgl.DGLGraph(g).to(device) for g in generated_graphs]
                result_mmd_20000, time1 = mmd_eval_GIN.evaluate(generated_dataset=generated_graphs_dgl,
                                                                reference_dataset=test_graphs_dgl)
                with open(os.path.join(graph_save_path, '_MMD.log'), 'a') as f:
                    f.write(f"\nEpoch: {epoch} @ Final Test_GIN: result_mmd: {result_mmd_20000}, time: {time1}\n")
                print('Final GIN eval: result_mmd: {}, time: {:.3f}s'.format(result_mmd_20000, time1))
                result_f1_20000, time2 = f1_eval_GIN.evaluate(generated_dataset=generated_graphs_dgl,
                                                              reference_dataset=test_graphs_dgl)
                with open(os.path.join(graph_save_path, '_MMD.log'), 'a') as f:
                    f.write(f"\nEpoch: {epoch} @ Final Test_GIN: result_f1: {result_f1_20000}, time: {time2}\n")
                print('Final F1 eval: result_f1: {}, time: {:.3f}s'.format(result_f1_20000, time2))
                pltr_eval.add_values(epoch, [result_mmd_20000['mmd_rbf'], result_f1_20000['f1_pr']], [None]*len(evaluations), redraw=redraw)
                pltr_eval.redraw()
                pltr_eval.save_plot(os.path.join(graph_save_path, "_Evaluation_log_plot"))

        if not valid_combination:
            continue

        print("Processing with combination:", combination)
        model.load_state_dict(torch.load(os.path.join(graph_save_path, "model_final20000.pth")))
        model.eval()
        generated_graphs = test_(graph_save_path, epoch, len(test_list_adj), model, graph_save_path, save_graphs=True, picture=1, load_tag='Final20000')
        generated_graphs = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in generated_graphs if not nx.is_empty(G)]
        mmd_res = mmd_eval(generated_graphs, [nx.from_numpy_array(graph.toarray()) for graph in test_list_adj], diam=True)
        with open(os.path.join(graph_save_path, '_MMD.log'), 'a') as f:
            f.write(f"\nFinal Test: Epoch: {epoch} @ {mmd_res}\n")
        generated_graphs_dgl = [dgl.DGLGraph(g).to(device) for g in generated_graphs]
        result_mmd, time1 = mmd_eval_GIN.evaluate(generated_dataset=generated_graphs_dgl, reference_dataset=test_graphs_dgl)
        with open(os.path.join(graph_save_path, '_MMD.log'), 'a') as f:
            f.write(f"\nFinal Test_GIN: Epoch: {epoch} @ result_mmd: {result_mmd}, time: {time1}\n")
        print('Final GIN eval: result_mmd: {}, time: {:.3f}s'.format(result_mmd, time1))
        result_f1, time2 = f1_eval_GIN.evaluate(generated_dataset=generated_graphs_dgl, reference_dataset=test_graphs_dgl)
        with open(os.path.join(graph_save_path, '_MMD.log'), 'a') as f:
            f.write(f"\nFinal Test_GIN: Epoch: {epoch} @ result_f1: {result_f1}, time: {time2}\n")
        print('Final F1 eval: result_f1: {}, time: {:.3f}s'.format(result_f1, time2))


if __name__ == '__main__':
    main()
