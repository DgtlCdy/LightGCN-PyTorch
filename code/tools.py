import numpy as np
import torch
from scipy.sparse import csr_matrix

import dataloader


def get_uu_graph(dataset: dataloader.Loader):
    bi_graph = dataset.getBipartiteGraph()
    numUsers = bi_graph.shape[0]
    dimUsers = bi_graph.shape[1]

    usersArray = bi_graph.toarray()

    uu_graph = np.eye(numUsers)
    for idx1, u1 in enumerate(bi_graph):
        for idx2, u2 in enumerate(bi_graph):
            if idx1 == idx2:
                continue
            intersection = np.logical_and(u1, u2).sum()
            union = np.logical_or(u1, u2).sum()
            similarity: float = intersection / union
            if similarity >= 0.1:
                uu_graph[u1][u2] = 1
                uu_graph[u2][u1] = 1
    uu_graph_flatten = uu_graph.flatten()
    count = 1
    a = 0
    uu_graph = csr_matrix(uu_graph)
    return uu_graph