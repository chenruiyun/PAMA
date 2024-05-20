import torch
import torch.nn as nn
import torch.nn.init as init

class UpConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpConvBlock, self).__init__()
        self.blk = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.blk(x)


class SelfAttention(nn.Module):
    def __init__(self, in_dim, attention_dim):
        super(SelfAttention, self).__init__()

        self.query_conv = nn.Conv2d(in_dim, attention_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, attention_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-2)

        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')

        # Bias set to zero (common practice)
        init.constant_(self.query_conv.bias, 0)
        init.constant_(self.key_conv.bias, 0)
        init.constant_(self.value_conv.bias, 0)

    def forward(self, x):
        query = self.query_conv(x).view(x.size(0), -1, x.size(-2) * x.size(-1)).permute(0, 2, 1)
        key = self.key_conv(x).view(x.size(0), -1, x.size(-2) * x.size(-1))
        value = self.value_conv(x).view(x.size(0), -1, x.size(-2) * x.size(-1)).permute(0, 2, 1)

        attention = torch.bmm(query, key)
        attention = self.softmax(attention)

        out = torch.bmm(attention, value).view(x.size(0), x.size(1), x.size(2), x.size(3))
        out = x + 0.1*out

        return out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv = nn.Conv2d(64, 48, kernel_size=3, stride=1, padding=1)

        self.upconv3 = UpConvBlock(512, 256)
        self.upconv2 = UpConvBlock(512, 128)
        self.upconv1 = UpConvBlock(256, 64)
        self.upconv0 = UpConvBlock(128, 48)
        self.upconv2mask = UpConvBlock(96, 48)

        self.final_conv = nn.Conv2d(48, 2, kernel_size=3, stride=1, padding=1)

        self.self_attention3 = SelfAttention(512, 256)
        self.self_attention2 = SelfAttention(256, 128)
        self.self_attention1 = SelfAttention(128, 64)

    # x_up3 = self.self_attention3(x_up3)
    # x_up2 = self.self_attention2(x_up2)
    # x_up1 = self.self_attention1(x_up1)
    def forward(self, encoder_output, concat_features):
        # concat_features = [level0, level1, level2, level3]
        f0, f1, f2, f3 = concat_features

        # 512 x 8 x 8 -> 512 x 16 x 16
        x_up3 = self.upconv3(encoder_output)
        x_up3 = torch.cat([x_up3, f3], dim=1)

        # 512 x 16 x 16 -> 256 x 32 x 32
        x_up2 = self.upconv2(x_up3)
        x_up2 = torch.cat([x_up2, f2], dim=1)

        # 256 x 32 x 32 -> 128 x 64 x 64
        x_up1 = self.upconv1(x_up2)
        x_up1 = torch.cat([x_up1, f1], dim=1)
        # x_up1 = self.self_attention1(x_up1)

        # 128 x 64 x 64 -> 96 x 128 x 128
        x_up0 = self.upconv0(x_up1)
        f0 = self.conv(f0)
        x_up2mask = torch.cat([x_up0, f0], dim=1)


        # 96 x 128 x 128 -> 48 x 256 x 256
        x_mask = self.upconv2mask(x_up2mask)

        # 48 x 256 x 256 -> 1 x 256 x 256
        x_mask = self.final_conv(x_mask)

        return x_mask