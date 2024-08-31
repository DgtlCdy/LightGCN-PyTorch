import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset  # 到这里，dataset就是一个获取到的Loader(BasicDataset)对象，存放了数据集

# from denoise_module import denoise

import pymetis
import networkx as nx


# 用后即焚
graphOrigin = dataset.getAdjList()
print('test1')
n_cuts, membership = pymetis.part_graph(64, adjacency=graphOrigin)
print('test2')
# for i in range(64):
#     print(f'count of {i} is: {membership.count(i)}')
usersList = membership[:29858 - 1]
itemsList = membership[29859:]
for i in range(64):
    print(f'count of {i} is: {usersList.count(i)} and {itemsList.count(i)}, whole{membership.count(i)}')

# graphOrigin = dataset.getBipartiteGraph()
# denoisedGraph = denoise(graphOrigin)
# graphOrigin = dataset.getSparseGraph()
# n_cuts, membership = pymetis.part_graph(64, adjacency=graphOrigin)
# for i in range(64):
#     print(f'count of {i} is: {membership.count(i)}')
# denoiseGraph = denoise(graphOrigin)

# # Recmodel：即使用的协同过滤模型，现有MF和LightGCN
# Recmodel = register.MODELS[world.model_name](world.config, dataset)
# Recmodel = Recmodel.to(world.device)
# bpr = utils.BPRLoss(Recmodel, world.config)

# weight_file = utils.getFileName()
# print(f"load and save to {weight_file}")
# if world.LOAD:
#     try:
#         # Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
#         Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cuda')))
#         world.cprint(f"loaded model weights from {weight_file}")
#     except FileNotFoundError:
#         print(f"{weight_file} not exists, start from beginning")
# Neg_k = 1

# # init tensorboard
# if world.tensorboard:
#     w : SummaryWriter = SummaryWriter(
#                                     join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
#                                     )
# else:
#     w = None
#     world.cprint("not enable tensorflowboard")

# # 获取原交互图二部图，用于聚类
# graphOrigin = dataset.getBipartiteGraph()

# try:
#     for epoch in range(world.TRAIN_epochs):
#         start = time.time()
#         if epoch %10 == 0:
#             cprint("[TEST]")
#             Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
        
#         # todo
#         # 进行聚类，输出噪音判定矩阵Rn
#         denoisedGraph = denoise(graphOrigin)

#         # 图卷积的矩阵更新为：(R-Rn) 哈达玛积 R

#         output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
#         print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
#         torch.save(Recmodel.state_dict(), weight_file)
# finally:
#     if world.tensorboard:
#         w.close()