"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time

# 函数输入：原邻接矩阵，规模为M*N
# 函数输出：规模为(M+N)*(M+N)的矩阵，元素为0代表不是噪声，为1代表是噪声
def denoise(GraphOrigin: torch.tensor):
    # 先获取用户和产品规模

    # 先对用户进行聚类
    
    # 聚类后，将每个簇投射到产品中，定义产品簇

    # 然后将边分类为对应的簇

    # 给边赋权值，然后根据概率定义噪声

    # 最后将噪声矩阵拼接为(M+N)*(M+N)的规模，然后用稀疏矩阵表示

    return