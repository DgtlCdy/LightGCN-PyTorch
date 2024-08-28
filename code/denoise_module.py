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

import random
from sklearn.cluster import KMeans


# 如何均匀地聚类是一个问题，k-means聚类肯定不行
def get_user_labels(graphArray, num_clusters):
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    predicted_labels = kmeans.fit_predict(graphArray)
    return predicted_labels


def get_most_similar(itemClusterCenters, item):
    similarity = []
    for ic in itemClusterCenters:
        similarity.append(np.dot(ic, item))
    minValue = max(similarity)
    minIndex = similarity.index(minValue)
    return minIndex


# 对于一条边，根据其user和item的id，获取其所在的边簇
def get_edge_cluster(userLabels, itemLabels, i, j, numClusters):
    userCluster = userLabels[i]
    itemCluster = itemLabels[j]
    if userCluster == itemCluster:
        return userCluster
    else:
        return userCluster + numClusters


# 边簇
class EdgeCluster:
    def __init__(self):
        self.indexUsers = []
        self.indexItems = []
        self.edges = []

    def add_user_index(self, userIndex):
        self.indexUsers.append(userIndex)

    def add_item_index(self, itemIndex):
        self.indexItems.append(itemIndex)

    def add_edge(self, i, j):
        self.edges.append((i, j))

    def get_density(self):
        return len(self.edges) / len(self.indexUsers) / len(self.indexItems)


def denoise_edge(value, i, j, p):
    if value == 0:
        return 0
    r = random.random()
    return 1 if r < p else 0


# 函数输入：原邻接矩阵，规模为M*N
# 函数输出：规模为(M+N)*(M+N)的矩阵，为1即噪声，要被清除掉
def denoise(graphOrigin: csr_matrix):
    # 先获取用户和产品信息
    numUsers = graphOrigin.shape[0]
    numItems = graphOrigin.shape[1]
    dimUsers = numItems
    dimItems = numUsers


    # getrow(i)[source] 返回矩阵第 i 行的副本，以 (1 x n) 稀疏矩阵（行向量）形式表示。
    # userList = [graphOrigin.getrow(i).toarray() for i in range(graphOrigin.shape[0])]
    usersArray = graphOrigin.toarray()
    itemsArray = usersArray.T


    # 先对用户进行聚类
    numClusters = 20
    userLabels = get_user_labels(usersArray, numClusters)


    # 聚类后，将每个簇投射到产品中，定义产品簇
    # 先确定产品的簇中心，itemClusterCenters: 簇个数行，item特征维度列；countClusterCenters：簇个数个
    itemClusterCenters = np.zeros((numClusters, dimItems),dtype=np.float32)
    countClusterCenters = np.zeros((numClusters), dtype=int)
    for i in range(numUsers):
        userLabel = userLabels[i]
        itemClusterCenters[userLabel] += itemsArray[i]
        countClusterCenters[userLabel] += 1
    # 均一化，每个itemClusterCenters的元素都除以总和数量
    for i in range(numClusters):
        itemClusterCenters[i] /= countClusterCenters[i]


    #然后根据产品与簇中心的相似度排序，对产品进行归类
    itemLabels = []
    for i in itemsArray:
        itemLabel = get_most_similar(itemClusterCenters, i)
        itemLabels.append(itemLabel)


    # 将边分类为对应的簇
    # 先建立边簇类
    numEdgeClusters = numClusters * 2
    edgeClusters = [EdgeCluster() for i in range(numEdgeClusters)]
    # 初始化边簇的节点信息
    for i in range(numUsers):         #添加全部用户节点到边簇
        userCluster = userLabels[i]
        edgeClusters[userCluster].add_user_index(i)
        edgeClusters[userCluster + numClusters].add_user_index(i)
    for i in range(numItems):        # 添加全部产品节点到边簇，每个产品节点入一个对应簇、numClusters-1个非对应簇
        itemCluster = itemLabels[i]
        edgeClusters[itemCluster].add_item_index(i)          # 入对应产品簇
        for j in range(numClusters):
            if j == itemCluster:
                continue
            edgeClusters[j + numClusters].add_item_index(i)  # 入非对应产品簇

    # 然后对边进行入簇
    for i, j in zip(*graphOrigin.nonzero()):
        # 获取对应的用户簇和产品簇
        userCluster = userLabels[i]
        itemCluster = itemLabels[j]
        if userCluster == itemCluster:
            edgeClusters[userCluster].add_edge(i, j) # 入对应簇
        else:
            edgeClusters[userCluster + numClusters].add_edge(i, j) # 入非对应簇


    # 对边簇进行降噪参数分析，确定每一个边簇内每一条边的去噪概率，然后对每一个边簇分配降噪数量指标
    # 首先确定总规模系数，然后密度调整系数
    # 每个边簇中降噪的边数指标：簇内边数 *（簇内全联通边数 / 全联通边数） * 稠密度系数——越稠密越小 * 归一化系数——使降噪指标之和调整为降噪总边数
    denoiseRate = 0.01
    edgeDenoiseTotal = graphOrigin.nnz * denoiseRate
    densityTotal = graphOrigin.nnz / (numUsers * numItems)
    for edgeCluster in edgeClusters:
        density = edgeCluster.get_density()
        weight = 
    for edgeCluster in edgeClusters:
        edgeCluster.set_denoise_mount()


    # 对每一条边进行概率性的降噪处理
    denoisedGraph = np.zeros([numUsers, numItems])
    for edgeCluster in edgeClusters:
        for edge in edgeCluster.edges:
            if random.random() < edgeCluster.get_denoise_probability() # 小于这个随机概率的时候，则认为是噪音，将其放置
            userIndex, itemIndex = edge
            denoisedGraph[userIndex][itemIndex] = 1


    # 最后将噪声矩阵拼接为(M+N)*(M+N)的规模，然后用稀疏矩阵表示
    adjMatrix = np.zeros((numUsers + numItems, numUsers + numItems))
    adjMatrix[:numUsers, :numItems] = denoisedGraph
    adjMatrix[numUsers:, :numUsers] = denoisedGraph.T


    # 返回带有噪音label的邻接矩阵，传回给神经网络用于加工
    return csr_matrix(adjMatrix)