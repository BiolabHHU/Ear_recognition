

import math, torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(channels, 512, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(512, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        # out = self.sigmoid(out)
        # return x * out.expand_as(x)
        return self.sigmoid(out)

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    # def forward(self, x):
    #     avg_out = torch.mean(x, dim=1, keepdim=True)
    #     max_out, _ = torch.max(x, dim=1, keepdim=True)
    #     att_input = torch.cat([avg_out, max_out], dim=1)
    #     att_map = self.conv1(att_input)
    #     att_weights = self.sigmoid(att_map)
    #     # 扩展注意力权重的通道数，使其与输入特征图通道数一致
    #     att_weights = att_weights.expand_as(x)
    #     return att_weights * x

# CBAM 模块
class CBAM(nn.Module):
    def __init__(self, in_planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, input):
        x = input * self.ca(input)
        x = x * self.sa(x)
        return input * x

class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=512):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        # self.se = SEModule(planes)
        # 使用 CBAM 替换 SEModule
        self.cbam = CBAM(planes)
        # self.CA = ChannelAttention(planes)
        # self.SA = SpatialAttention()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        # out = self.se(out)
        out = self.cbam(out)
        # out = self.CA(out)
        # out = self.SA(out)
        out += residual
        return out


class ECAPA_TDNN(nn.Module):

    def __init__(self, C):
        super(ECAPA_TDNN, self).__init__()
        self.learnable_param = nn.Parameter(torch.ones(1))  # 初始化为 1
        self.conv1 = nn.Conv1d(299, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)

        self.attention = nn.Sequential(
            nn.Conv1d(4608, 512, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Conv1d(512, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )

        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 128)
        self.bn6 = nn.BatchNorm1d(128)

    def forward(self, x):
        # 输入数据乘以可学习参数
        # x = x * self.learnable_param
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        # x = x * self.learnable_param
        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)
        # x3 = x3 * self.learnable_param
        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        t = x.size()[-1]
        global_x = torch.cat((x, torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                               torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)), dim=1)
        w = self.attention(global_x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))
        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x