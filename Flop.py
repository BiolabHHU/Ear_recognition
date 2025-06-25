import torch
import torch.nn as nn
# from ResNetSE34L import MainModel
from ECAPA_Model import ECAPA_TDNN
# from main_GE2E import XVectorModel
# from VGG_M_40 import XVectorModel
from main_S2P import AutoEncoderModel
from main import XVectorModel
# -- coding: utf-8 --
import torch
import torchvision
from thop import profile
#
# # Model
# print('==> Building model..')
# # model = ECAPA_TDNN(C=512)
# model = XVectorModel()
# dummy_input = torch.randn(128, 299, 13)
# flops, params = profile(model, (dummy_input,))
# print('flops: ', flops, 'params: ', params)
# print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))


# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 读取Excel文件
# file_path = 'F:/耳纹数据/L50.xlsx'  # 请根据您的文件名调整路径
# data = pd.read_excel(file_path)
#
# # 提取横坐标（第一列）
# x_values = data.iloc[:, 0]
#
# # 提取纵坐标（从第二列到第751列）
# y_values = data.iloc[:, 1:751]
#
# # # 将 y_values 转换为 PyTorch 张量
# # y_tensor = torch.tensor(y_values, dtype=torch.float32)
# # # 计算均值和标准差
# # means = torch.mean(y_tensor, dim=1)          # 逐行计算均值
# # std_devs = torch.std(y_tensor, dim=1)        # 逐行计算标准差
#
# # 计算均值和平方差
# means = y_values.mean(axis=1)          # 逐行计算均值
# std_devs = y_values.std(axis=1)        # 逐行计算标准差
#
# # 提取第一行数据（第二列到第751列）
# first_row_data = data.iloc[:, 1:16]  # 提取第一行数据
# means_1 = first_row_data.mean(axis=1)
# std_devs_1 = first_row_data.std(axis=1)
# # 画图
# plt.figure(figsize=(12, 6))
#
# # 均值图
# plt.subplot(1, 2, 1)
# plt.plot(x_values, means, label='Mean', color='b')
# plt.plot(x_values, means_1, label='First Row Data', color='g')  # 用不同颜色绘制第一行数据
# plt.title('Mean Values')
# plt.xlabel('X Axis (First Column)')
# plt.ylabel('Mean')
# plt.legend()
#
# # 平方差图
# plt.subplot(1, 2, 2)
# plt.plot(x_values, std_devs, label='Standard Deviation', color='r')
# plt.plot(x_values, std_devs_1, label='First Row Data Standard Deviation', color='b')
# plt.title('Standard Deviation Values')
# plt.xlabel('X Axis (First Column)')
# plt.ylabel('Standard Deviation')
# plt.legend()
#
# # 绘制带误差棒的图
# # plt.subplot(1, 2, 1)
# # plt.errorbar(x_values, means.numpy(), yerr=std_devs.numpy(), fmt='o',
# #              label='Mean with Standard Deviation', color='b', capsize=5)
# #
# # plt.title('Mean Values with Error Bars')
# # plt.xlabel('X Axis (First Column)')
# # plt.ylabel('Mean')
# # plt.legend()
#
# # 显示图形
# plt.tight_layout()
# plt.show()

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams["font.sans-serif"] = ["FangSong"]
mpl.rcParams["axes.unicode_minus"] = False

# 读取Excel文件
file_path = 'F:/耳纹数据/L50.xlsx'  # 请根据您的文件名调整路径
data = pd.read_excel(file_path)

# 提取横坐标（第一列）
x_values = data.iloc[:, 0]

# 提取纵坐标（从第二列到第751列）
y_values = data.iloc[:, 1:751]


error_attri = dict(elinewidth=2, ecolor="black", capsize=3)

bar_width = 0.01

# create bar with errorbar
plt.bar(x_values, y_values,
        bar_width,
        color="#87CEEB",
        align="center",
        yerr=y_values,
        error_kw=error_attri,
        label="2010")

# set x,y_axis label
plt.xlabel("Hz")
plt.ylabel("db")


# set yaxis grid
plt.grid(True, axis="y", ls=":", color="gray", alpha=0.2)

plt.legend()

plt.show()
