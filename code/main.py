# 外部包
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
from os.path import join

# 自身
import world
import utils
from world import cprint
import Procedure
import register
from register import dataset

from denoise_module import denoise

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================

# Recmodel：即使用的协同过滤模型，有纯MF和LightGCN
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        # Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cuda')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch %10 == 0:
            cprint("[TEST]")
            Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
        
        # 进行聚类，输出噪音判定矩阵Rn
        # 图卷积的矩阵更新为：(R-Rn) 哈达玛积 R
        RNoise = denoise(Recmodel.get_origin_graph())
        Recmodel.update_graph(RNoise)

        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        torch.save(Recmodel.state_dict(), weight_file)
finally:
    if world.tensorboard:
        w.close()