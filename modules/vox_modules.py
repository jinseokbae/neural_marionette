import torch
import torch.nn as nn
import torch.nn.functional as F

"""  
reference : https://github.com/zhan-xu/AnimSkelVolNet/blob/master/models3D/model3d_hg.py
"""
class Basic3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(Basic3DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding = ((kernel_size - 1) // 2)),
            # nn.BatchNorm3d(out_planes, track_running_stats=False),
            nn.GroupNorm(out_planes // 16, out_planes),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.block(x)


class Res3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Res3DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(out_planes, track_running_stats=False),
            nn.GroupNorm(out_planes // 16, out_planes),
            nn.LeakyReLU(),
            nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(out_planes, track_running_stats=False)
            nn.GroupNorm(out_planes // 16, out_planes),
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                # nn.BatchNorm3d(out_planes, track_running_stats=False)
                nn.GroupNorm(out_planes // 16, out_planes),
            )

    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.leaky_relu(res + skip, True)

class Pool3DBlock(nn.Module):
    def __init__(self, pool_size, input_plane):
        super(Pool3DBlock, self).__init__()
        self.stride_conv = nn.Sequential(
            nn.Conv3d(input_plane, input_plane, kernel_size=pool_size, stride=pool_size, padding=0),
            # nn.BatchNorm3d(input_plane, track_running_stats=False),
            nn.GroupNorm(input_plane // 16, input_plane),
            nn.LeakyReLU()
        )

    def forward(self, x):
        y = self.stride_conv(x)
        return y

class Upsample3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, output_padding=0):
        super(Upsample3DBlock, self).__init__()
        assert (stride == 2)
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, output_padding=output_padding),
            # nn.BatchNorm3d(out_planes, track_running_stats=False),
            nn.GroupNorm(out_planes // 16, out_planes),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.block(x)


class HG(nn.Module):
    def __init__(self, input_channels, output_channels, N=88):
        super(HG, self).__init__()
        outer_padding = [(N//4)%2, (N//2)%2, (N//1)%2]
        self.encoder_pool1 = Pool3DBlock(2, input_channels)
        self.encoder_res1 = Res3DBlock(input_channels, 32)
        self.encoder_pool2 = Pool3DBlock(2, 32)
        self.encoder_res2 = Res3DBlock(32, 48)
        self.encoder_pool3 = Pool3DBlock(2, 48)
        self.encoder_res3 = Res3DBlock(48, 72)

        self.decoder_res3 = Res3DBlock(72, 72)
        self.decoder_upsample3 = Upsample3DBlock(72, 48, 2, 2, outer_padding[0])
        self.decoder_res2 = Res3DBlock(48, 48)
        self.decoder_upsample2 = Upsample3DBlock(48, 32, 2, 2, outer_padding[1])
        self.decoder_res1 = Res3DBlock(32, 32)
        self.decoder_upsample1 = Upsample3DBlock(32, output_channels, 2, 2, outer_padding[2])

        self.skip_res1 = Res3DBlock(input_channels, output_channels)
        self.skip_res2 = Res3DBlock(32, 32)
        self.skip_res3 = Res3DBlock(48, 48)

    def forward(self, x):
        skip_x1 = self.skip_res1(x)
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x)
        skip_x2 = self.skip_res2(x)
        x = self.encoder_pool2(x)
        x = self.encoder_res2(x)
        skip_x3 = self.skip_res3(x)
        x = self.encoder_pool3(x)
        x = self.encoder_res3(x)

        x = self.decoder_res3(x)
        x = self.decoder_upsample3(x)
        x = x + skip_x3
        x = self.decoder_res2(x)
        x = self.decoder_upsample2(x)
        x = x + skip_x2
        x = self.decoder_res1(x)
        x = self.decoder_upsample1(x)
        x = x + skip_x1
        return x

