import os
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

    # user_array = bi_graph.toarray()
    # user_array = torch.Tensor(user_array).to(world.device)


    if os.path.isfile('C:/codes/buffer/uu_graph_vae.pt'):
        uu_graph = torch.load('C:/codes/buffer/uu_graph_vae.pt')
    else:
        # 将n个维度的用户特征，通过vae转换为d个维度的隐特征，用于求后面的相似性
        user_emb = vae.mult_vae_inference(bi_graph)
        # user_emb_norm = F.normalize(user_emb)
        uu_graph = torch.zeros([n_users, n_users], dtype=torch.float32, device=world.device)

        # 目前只是可以获取到每一个用户的隐式分布，但还需要找到根据分布情况来求用户间相似性的方法
        for idx1, u1 in enumerate(user_emb):
            for idx2, u2 in enumerate(user_emb):
                similarity: float = torch.sigmoid(torch.dot(u1, u2))
                uu_graph[idx1][idx2] = similarity
                uu_graph[idx2][idx1] = similarity
        torch.save(user_emb, 'C:/codes/buffer/user_emb_vae.pt')
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