import torch
import torch.nn as nn


class DownSampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super(DownSampleLayer, self).__init__()
        padding_size = int((kernel_size - 1) / 2)
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=1, padding=padding_size),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size, stride=1, padding=padding_size),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size, stride=2, padding=padding_size),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        out = self.Conv_BN_ReLU_2(x)
        out = self.downsample(out)
        return out


class Gatedconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super(Gatedconv, self).__init__()
        padding_size = int((kernel_size - 1) / 2)
        self.Conv_BN_LReLU = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=1, padding=padding_size),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size, stride=1, padding=padding_size),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )
        self.dconv_mask = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=1, padding=padding_size),
            nn.BatchNorm2d(out_ch),
            nn.Sigmoid()
        )
        self.dconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_ch, out_channels=in_ch, kernel_size=kernel_size, stride=2,
                               padding=padding_size,
                               output_padding=1),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(0.2),
        )
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, dconv=False):
        if dconv:
            out = self.dconv(x)
        else:
            out = self.upsample(x)
        x = self.Conv_BN_LReLU(out)
        mask = self.dconv_mask(out)
        return x * mask


class AGGConv(nn.Module):
    def __init__(self, in_ch, out_ch, rgb_ch, h, w, kernel_size=3, ratio_linear=1):
        super().__init__()
        self.h = h
        self.w = w
        self.size = h * w
        padding_size = int((kernel_size - 1) / 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding_size, bias=False,
                      padding_mode='replicate'),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=2, padding=padding_size, bias=False,
                      padding_mode='replicate'),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )
        self.gate_conv = nn.Sequential(
            nn.Conv2d(out_ch + rgb_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding_size, bias=False,
                      padding_mode='replicate'),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )
        self.gate_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.size, self.size * ratio_linear, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.size * ratio_linear, self.size, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, depth, rgb):
        depth = self.conv(depth)
        mask = torch.cat([depth, rgb], dim=1)
        mask = self.gate_conv(mask)

        # V1
        # attention = torch.zeros_like(depth)
        # for layer in range(len(attention[0])):
        #     bs = mask.size(0)
        #     attention[:, layer, :, :] = self.gate_fc(mask[:, layer, :, :]).reshape([bs, self.h, self.w])

        # V2, shorter time cost
        bs, ch, *spatial = depth.shape
        attention = self.gate_conv(mask).reshape([bs * ch, *spatial])
        attention = self.gate_fc(attention).reshape([bs, ch, *spatial])

        return depth * attention


class AGSC(nn.Module):
    def __init__(self, in_channels_color, in_channels_depth, out_channels):
        super().__init__()
        self.conv_depth = nn.Sequential(
            nn.Conv2d(in_channels_depth, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels)
        )

        self.conv_color = nn.Sequential(
            nn.Conv2d(in_channels_color, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels)
        )

        self.w = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, color, depth):
        conv_c = self.conv_color(color)
        conv_d = self.conv_depth(depth)
        weight = self.relu(torch.cat([conv_c, conv_d], dim=1))
        weight = self.w(weight)

        return weight * color


class PreFilling(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=48, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=48, out_channels=96, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),
        )
        self.dconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=96, out_channels=48, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=48, out_channels=out_ch, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.Sigmoid(),
        )

    def forward(self, x, depth, mask):
        x = self.conv(x)
        x = self.dconv(x)
        out = x * mask + depth
        return out


class AGGNet(nn.Module):
    def __init__(self, h, w, num_layer=3, kernel_size=3, ratio_gate=1):
        super().__init__()

        self.num_layer = num_layer
        rgb_channels = [64, 128, 256, 384, 576]
        depth_channels = [64, 128, 256, 384, 576]
        out_channels = [576, 384, 256, 128, 64]
        padding_size = int((kernel_size - 1) / 2)
        self.fill_depth = PreFilling(3, 1)

        self.conv_rgb_0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=rgb_channels[0], kernel_size=kernel_size, stride=1,
                      padding=padding_size),
            nn.BatchNorm2d(rgb_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=rgb_channels[0], out_channels=rgb_channels[0], kernel_size=kernel_size, stride=2,
                      padding=padding_size),
            nn.BatchNorm2d(rgb_channels[0]),
            nn.ReLU(inplace=True)
        )
        self.conv_gate_0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=depth_channels[0], kernel_size=kernel_size, stride=1,
                      padding=padding_size),
            nn.BatchNorm2d(depth_channels[0]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=depth_channels[0], out_channels=depth_channels[0], kernel_size=kernel_size, stride=2,
                      padding=padding_size),
            nn.BatchNorm2d(depth_channels[0]),
            nn.LeakyReLU(0.2)
        )

        self.conv_rgb = nn.ModuleList([])
        self.conv_gate = nn.ModuleList([])
        self.dconv = nn.ModuleList([Gatedconv(depth_channels[num_layer] + rgb_channels[num_layer], out_channels[0], kernel_size)])
        self.att_r = nn.ModuleList([])
        for i in range(num_layer):
            self.conv_rgb.append(DownSampleLayer(rgb_channels[i], rgb_channels[i+1], kernel_size))
            self.conv_gate.append(AGGConv(depth_channels[i], depth_channels[i + 1], rgb_channels[i + 1],
                                          h[i+2], w[i+2], kernel_size, ratio_gate))
            self.dconv.append(Gatedconv(depth_channels[num_layer-1-i] + rgb_channels[num_layer-1-i] + out_channels[i],
                                        out_channels[i+1], kernel_size))
            self.att_r.append(AGSC(rgb_channels[num_layer - 1 - i], out_channels[i], rgb_channels[num_layer - 1 - i]))

        self.dconv_last = nn.Sequential(
            nn.Conv2d(out_channels[num_layer], 64, kernel_size=kernel_size, padding=padding_size, padding_mode='replicate'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, kernel_size=kernel_size, padding=padding_size, padding_mode='replicate'),
            nn.Sigmoid(),
        )

    def forward(self, rgb, depth_input, mask, replace=False):
        depth = self.fill_depth(rgb, depth_input, mask)
        conv_rgb = [self.conv_rgb_0(rgb)]
        conv_gate = [self.conv_gate_0(depth)]
        for i in range(self.num_layer):
            conv_rgb.append(self.conv_rgb[i](conv_rgb[i]))
            conv_gate.append(self.conv_gate[i](conv_gate[i], conv_rgb[i+1]))

        bottle = torch.cat([conv_rgb[self.num_layer], conv_gate[self.num_layer]], dim=1)

        x = self.dconv[0](bottle)
        att_r = [self.att_r[0](conv_rgb[self.num_layer-1], x)]
        x = torch.cat([x, att_r[0], conv_gate[self.num_layer-1]], dim=1)
        for i in range(1, self.num_layer):
            x = self.dconv[i](x)
            att_r.append(self.att_r[i](conv_rgb[self.num_layer-1-i], x))
            x = torch.cat([x, att_r[i], conv_gate[self.num_layer-1-i]], dim=1)

        x = self.dconv[self.num_layer](x)
        x = self.dconv_last(x)
        if replace:
            x = mask * x + depth_input
        return x
