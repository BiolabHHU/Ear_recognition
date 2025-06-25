import torch
import torch.nn as nn
class TdnnLayer1(nn.Module):
    def __init__(self, input_size=13, output_size=512, context=[0], batch_norm=True, dropout_p=0.0):
        """
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf
        Structure inspired by https://github.com/cvqluu/TDNN/blob/master/tdnn.py
        """
        super(TdnnLayer1, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.context = context
        self.batch_norm = batch_norm
        self.dropout_p = dropout_p

        # 创建5个全连接层
        self.linear1 = nn.Linear(13, 12)
        self.linear2 = self.linear1  # 参数共享
        self.linear3 = self.linear1  # 参数共享
        self.linear4 = self.linear1  # 参数共享
        self.linear5 = nn.Linear(13, 16)

        self.relu = nn.ReLU()
        if (self.batch_norm):
            self.norm = nn.BatchNorm1d(output_size)
        if (self.dropout_p):
            self.drop = nn.Dropout(p=self.dropout_p)

    def forward(self, x):

        xc = get_time_context(x, self.context)
        x1 = self.linear1(xc[0])
        x2 = self.linear2(xc[1])
        x3 = self.linear3(xc[2])
        x4 = self.linear4(xc[3])
        x5 = self.linear5(xc[4])

        x = torch.cat((x1, x2, x3, x4, x5), dim=2)  # 合并5个全连接层的结果
        x = self.relu(x)

        if (self.dropout_p):
            x = self.drop(x)

        if (self.batch_norm):
            x = x.transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2)

        return x
class TdnnLayer2(nn.Module):
    def __init__(self, input_size=24, output_size=512, context=[0], shared_linear=None, batch_norm=True, dropout_p=0.0):
        """
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf
        Structure inspired by https://github.com/cvqluu/TDNN/blob/master/tdnn.py
        """
        super(TdnnLayer2, self).__init__()
        self.shared_linear = shared_linear
        self.input_size = input_size
        self.output_size = output_size
        self.context = context
        self.batch_norm = batch_norm
        self.dropout_p = dropout_p

        # 创建3个全连接层
        self.linear1 = nn.Linear(64, 16)
        self.linear2 = self.linear1  # 参数共享
        self.linear3 = self.linear1  # 参数共享
        self.linear4 = self.linear1  # 参数共享

        self.relu = nn.ReLU()
        if (self.batch_norm):
            self.norm = nn.BatchNorm1d(output_size)
        if (self.dropout_p):
            self.drop = nn.Dropout(p=self.dropout_p)

    def forward(self, x):

        xc = get_time_context(x, self.context)
        x4 = xc[0]
        x1 = self.linear1(xc[0])
        x2 = self.linear2(xc[1])
        x3 = self.linear3(xc[2])
        x4 = self.linear4(x4)
        x = torch.cat((x1, x2, x3, x4), dim=2)  # 合并3个全连接层的结果
        x = self.relu(x)

        if (self.dropout_p):
            x = self.drop(x)

        if (self.batch_norm):
            x = x.transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2)

        return x
class TdnnLayer3(nn.Module):
    def __init__(self, input_size=24, output_size=512, context=[0], shared_linear=None, batch_norm=True, dropout_p=0.0):
        """
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf
        Structure inspired by https://github.com/cvqluu/TDNN/blob/master/tdnn.py
        """
        super(TdnnLayer3, self).__init__()
        self.shared_linear = shared_linear
        self.input_size = input_size
        self.output_size = output_size
        self.context = context
        self.batch_norm = batch_norm
        self.dropout_p = dropout_p

        # 创建3个全连接层
        self.linear1 = nn.Linear(64, 16)
        self.linear2 = self.linear1  # 参数共享
        self.linear3 = self.linear1  # 参数共享
        self.linear4 = self.linear1  # 参数共享

        self.relu = nn.ReLU()
        if (self.batch_norm):
            self.norm = nn.BatchNorm1d(output_size)
        if (self.dropout_p):
            self.drop = nn.Dropout(p=self.dropout_p)

    def forward(self, x):

        xc = get_time_context(x, self.context)
        x4 = xc[1]
        x1 = self.linear1(xc[0])
        x2 = self.linear2(xc[1])
        x3 = self.linear3(xc[2])
        x4 = self.linear4(x4)
        x = torch.cat((x1, x2, x3, x4), dim=2)  # 合并3个全连接层的结果
        x = self.relu(x)

        if (self.dropout_p):
            x = self.drop(x)

        if (self.batch_norm):
            x = x.transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2)

        return x
def get_time_context(x, c=[0]):
    """
    Returns x with the applied time context. For this the surrounding time frames are concatenated together.
    For example an input of shape (100, 10) with context [-1,0,1] would become (98,30).
    Visual example:
    x=          c=          result=
    [[1,2],
    [3,4],                  [[1,2,3,4,5,6],
    [5,6],      [-1,0,1]    [3,4,5,6,7,8],
    [7,8],                  [5,6,7,8,9,0]]
    [9,0]]
    """
    l = len(c) - 1
    xc = [x[:, c[l] + cc:c[0] + cc, :]
          if cc != c[l] else
          x[:, c[l] + cc:, :]
          for cc in c]
    return xc