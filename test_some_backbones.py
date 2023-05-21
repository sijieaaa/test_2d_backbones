# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from pointnet2.pointnet2_modules import PointnetSAModule
from models_pointloc.sequential_layers import fc_dropout
# from models.self_attention_module import SelfAttentionModule



from pointnet2.pointnet2_utils import *
from pointnet2.pointnet2_modules import *

import numpy as np
from viz_lidar_mayavi_open3d import *


import torchvision



import network.swin_transformer as swin_transformer


import spconv.pytorch as spconv
from tqdm import tqdm

import torchvision.models as TVmodels


num_iters = 50
b = 16
h = 256
w = 256

print('convnext_tiny')
model = TVmodels.convnext_tiny(weights='IMAGENET1K_V1').cuda()
for i in tqdm(range(num_iters)):
    data = torch.rand([b,3,h,w]).cuda()
    output = model(data)
torch.cuda.empty_cache()



print('efficientnet_b2')
model = TVmodels.efficientnet_b2(weights='IMAGENET1K_V1').cuda()
for i in tqdm(range(num_iters)):
    data = torch.rand([b,3,h,w]).cuda()
    output = model(data)
torch.cuda.empty_cache()




print('efficientnet_b3')
model = TVmodels.efficientnet_b3(weights='IMAGENET1K_V1').cuda()
for i in tqdm(range(num_iters)):
    data = torch.rand([b,3,h,w]).cuda()
    output = model(data)
torch.cuda.empty_cache()



print('efficientnet_b4')
model = TVmodels.efficientnet_b4(weights='IMAGENET1K_V1').cuda()
for i in tqdm(range(num_iters)):
    data = torch.rand([b,3,h,w]).cuda()
    output = model(data)
torch.cuda.empty_cache()



print('efficientnet_v2_s')
model = TVmodels.efficientnet_v2_s(weights='IMAGENET1K_V1').cuda()
for i in tqdm(range(num_iters)):
    data = torch.rand([b,3,h,w]).cuda()
    output = model(data)
torch.cuda.empty_cache()



print('regnet_x_3_2gf')
model = TVmodels.regnet_x_3_2gf(weights='IMAGENET1K_V2').cuda()
for i in tqdm(range(num_iters)):
    data = torch.rand([b,3,h,w]).cuda()
    output = model(data)
torch.cuda.empty_cache()



print('regnet_y_1_6gf')
model = TVmodels.regnet_y_1_6gf(weights='IMAGENET1K_V2').cuda()
for i in tqdm(range(num_iters)):
    data = torch.rand([b,3,h,w]).cuda()
    output = model(data)
torch.cuda.empty_cache()



print('regnet_y_3_2gf')
model = TVmodels.regnet_y_3_2gf(weights='IMAGENET1K_V2').cuda()
for i in tqdm(range(num_iters)):
    data = torch.rand([b,3,h,w]).cuda()
    output = model(data)
torch.cuda.empty_cache()



print('resnet18')
model = TVmodels.resnet18(weights='IMAGENET1K_V1').cuda()
for i in tqdm(range(num_iters)):
    data = torch.rand([b,3,h,w]).cuda()
    output = model(data)
torch.cuda.empty_cache()



print('resnet34')
model = TVmodels.resnet34(weights='IMAGENET1K_V1').cuda()
for i in tqdm(range(num_iters)):
    data = torch.rand([b,3,h,w]).cuda()
    output = model(data)
torch.cuda.empty_cache()



print('resnet50')
model = TVmodels.resnet50(weights='IMAGENET1K_V2').cuda()
for i in tqdm(range(num_iters)):
    data = torch.rand([b,3,h,w]).cuda()
    output = model(data)
torch.cuda.empty_cache()



print('swin_t')
model = TVmodels.swin_t(weights='IMAGENET1K_V1').cuda()
for i in tqdm(range(num_iters)):
    data = torch.rand([b,3,h,w]).cuda()
    output = model(data)
torch.cuda.empty_cache()



print('swin_v2_t')
model = TVmodels.swin_v2_t(weights='IMAGENET1K_V1').cuda()
for i in tqdm(range(num_iters)):
    data = torch.rand([b,3,h,w]).cuda()
    output = model(data)
torch.cuda.empty_cache()













