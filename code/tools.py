import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as sp

import world
import mult_vae.main_function as vae


# 即使用Mult-VAE的main函数，通过变分推断获取用户的隐特征，再通过相似度推断社交关联
def get_uu_graph(bi_graph: sp.csr_matrix):
    n_users = bi_graph.shape[0]
    dim_users = bi_graph.shape[1]

    if os.path.isfile('C:/codes/buffer/uu_graph_vae.pt'):
        uu_graph = torch.load('C:/codes/buffer/uu_graph_vae.pt')
    else:
        # 将n个维度的用户特征，通过vae转换为d个维度的隐特征，用于求后面的相似性
        user_emb_mu, user_emb_std = vae.mult_vae_inference(bi_graph)
        uu_graph = torch.zeros([n_users, n_users], dtype=torch.float32, device=world.device)

        # 目前只是可以获取到每一个用户的隐式分布，但还需要找到根据分布情况来求用户间相似性的方法
        for mu1, std1, idx1 in zip(user_emb_mu, user_emb_std, range(n_users)):
            for mu2, std2, idx2 in zip(user_emb_mu, user_emb_std,range(n_users)):
                # 获取两个单一正态分布的各个维度的巴氏距离
                para1 = (mu1 - mu2)**2 / (std1**2 + std2**2)
                para2 = torch.log((std1**2 + std2**2) / (2 * std1 * std2))
                bdistance = para1 + para2

                # 将各维度聚合为数字的相似度度量指标
                uu_graph[idx1][idx2] = bdistance
                uu_graph[idx2][idx1] = bdistance
        uu_graph = 1. / uu_graph # 将巴氏距离转化为相似度，巴氏距离越近，则相似度越高
        torch.save(user_emb_mu, 'C:/codes/buffer/user_emb_mu_vae.pt')
        torch.save(user_emb_std, 'C:/codes/buffer/user_emb_std_vae.pt')
        torch.save(uu_graph, 'C:/codes/buffer/uu_graph_vae.pt')

    # uu_adj = torch.zeros([n_users, n_users], dtype=torch.float32, device=world.device)
    # 设定阈值，相似度高于此阈值的，邻接矩阵设为1
    similarity_threshold = 0.1
    uu_adj = (uu_graph - similarity_threshold + 1).int()
    for i in range(uu_adj.shape[0]):
        uu_adj[i][i] = 0

    uu_adj_flatten = uu_adj.flatten().tolist()
    count_whole = len(uu_adj_flatten)
    count = uu_adj_flatten.count(1)

    uu_adj_csr = sp.csr_matrix(uu_adj.cpu().numpy())
    return uu_adj_csr