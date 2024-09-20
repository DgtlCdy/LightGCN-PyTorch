import os
import numpy as np
import torch
import scipy.sparse as sp

import world


def get_uu_graph(bi_graph: sp.csr_matrix):
    numUsers = bi_graph.shape[0]
    dimUsers = bi_graph.shape[1]

    user_array = bi_graph.toarray()
    user_array = torch.Tensor(user_array).to(world.device)

    if os.path.isfile('C:/codes/buffer/uu_graph.pt'):
        uu_graph = torch.load('C:/codes/buffer/uu_graph.pt')
    else:
        uu_graph = torch.zeros([numUsers, numUsers], dtype=torch.float32, device=world.device)
        for idx1, u1 in enumerate(user_array):
            for idx2, u2 in enumerate(user_array):
                intersection = torch.logical_and(u1, u2).sum()
                union = torch.logical_or(u1, u2).sum()
                similarity: float = intersection / union
                uu_graph[idx1][idx2] = similarity
                uu_graph[idx2][idx1] = similarity
        torch.save(uu_graph, 'C:/codes/buffer/uu_graph.pt')

    # uu_adj = torch.zeros([numUsers, numUsers], dtype=torch.float32, device=world.device)
    # 设定阈值，相似度高于此阈值的，邻接矩阵设为1
    similarity_threshold = 0.09
    uu_adj = (uu_graph - similarity_threshold + 1).int()
    for i in range(uu_adj.shape[0]):
        uu_adj[i][i] = 0


    uu_adj_flatten = uu_adj.flatten().tolist()
    count_whole = len(uu_adj_flatten)
    count = uu_adj_flatten.count(1)

    uu_adj_csr = sp.csr_matrix(uu_adj.cpu().numpy())
    return uu_adj_csr