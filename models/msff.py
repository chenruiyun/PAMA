import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .coordatt import CoordAtt

class ChannelAtt(nn.Module):
    def __init__(self, inp, reduction=16):
        super(ChannelAtt, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(inp, inp // reduction, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp // reduction, inp, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return out

class MSFFBlock(nn.Module):
    def __init__(self, in_channel,out_channel):
        super(MSFFBlock, self).__init__()
        # 添加通道注意力模块
        self.channel_att = ChannelAtt(in_channel)

        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.attn = CoordAtt(in_channel, in_channel)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channel // 2, out_channel, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        # 添加通道注意力操作
        channel_w = self.channel_att(x)
        x=x*channel_w
        x_conv = self.conv1(x)
        x_att = self.attn(x)

        x = x_conv * x_att
        x = self.conv2(x)
        return x


class MSFF(nn.Module):
    def __init__(self):
        super(MSFF, self).__init__()
        self.blk1 = MSFFBlock(128+64,64)
        self.blk2 = MSFFBlock(256+128,128)
        self.blk3 = MSFFBlock(512+256,256)
        # self.blk1 = MSFFBlock(128 , 64)
        # self.blk2 = MSFFBlock(256 , 128)
        # self.blk3 = MSFFBlock(512 , 256)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.upconv32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        )
        self.upconv21 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        )


# ///////////////////////////////
        self.downconv12 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        )
        self.downconv23 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        )


    def forward(self, features):
        # features = [level1, level2, level3]
        f1, f2, f3 = features

        # MSFF Module
        f1_k = self.blk1(f1)
        f2_k = self.blk2(f2)
        f3_k = self.blk3(f3)

        f2_f = f2_k
        f1_f = f1_k
        #f3_f=f3_k+self.downconv23(f2_k)

        # f2_f = f2_k
        # f1_f = f1_k
        # spatial attention

        # mask
        m3 = f3[:,256:512,...].mean(dim=1, keepdim=True)
        m2 = f2[:,128:256,...].mean(dim=1, keepdim=True)
        m1 = f1[:,64:128,...].mean(dim=1, keepdim=True)

        # mask_c+
        m3_c = f3[:, 512:, ...].mean(dim=1, keepdim=True)
        m2_c = f2[:, 256:, ...].mean(dim=1, keepdim=True)
        m1_c = f1[:, 128:, ...].mean(dim=1, keepdim=True)
        #
        alpha= 0.3

        f1_out = f1_f * m1+f1_f*m1_c*alpha
        f2_out = f2_f * m2+f2_f*m2_c*alpha
        f3_out = f3_k * m3+f3_k*m3_c*alpha
        # f3_out = f3_f * m3 + f3_f * m3_c * alpha

        # f1_out = f1_f * m1
        # f2_out = f2_f * m2
        # f3_out = f3_k * m3

        return [f1_out, f2_out, f3_out]