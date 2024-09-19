import os
import numpy as np
import torch
from scipy.sparse import csr_matrix

import world
import dataloader


def get_uu_graph(dataset: dataloader.Loader):
    bi_graph = dataset.getBipartiteGraph()
    numUsers = bi_graph.shape[0]
    dimUsers = bi_graph.shape[1]

    user_array = bi_graph.toarray()
    user_array = torch.Tensor(user_array).to(world.device)

    if os.path.isfile('C:/codes/buffer/uu_graph.pt'):
        uu_graph = torch.load('C:/codes/buffer/uu_graph.pt')
    else:
        uu_graph = torch.eye(numUsers, dtype=torch.float32, device=world.device)
        for idx1, u1 in enumerate(user_array):
            for idx2, u2 in enumerate(user_array):
                if idx1 == idx2:
                    continue
                intersection = torch.logical_and(u1, u2).sum()
                union = torch.logical_or(u1, u2).sum()
                similarity: float = intersection / union
                uu_graph[idx1][idx2] = similarity
                uu_graph[idx2][idx1] = similarity
                # if similarity >= 0.1:
                #     uu_graph[idx1][idx2] = 1
                #     uu_graph[idx2][idx1] = 1
        torch.save(uu_graph, 'C:/codes/buffer/uu_graph.pt')
    
    # 设定阈值，相似度高于此阈值的，邻接矩阵设为1
    similarity_threshold = 0.1
    uu_adj = (uu_graph - similarity_threshold + 1).int()

    uu_adj_flatten = uu_adj.flatten().tolist()
    count_whole = len(uu_adj_flatten)
    count = uu_adj_flatten.count(1)

    uu_graph = csr_matrix(uu_graph)
    return uu_graph