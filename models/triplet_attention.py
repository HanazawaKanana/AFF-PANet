### For latest triplet_attention module code please refer to the corresponding file in root.

import torch
import torch.nn as nn
import math

class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(
        self,
        gate_channels,
        reduction_ratio=16,
        pool_types=["avg", "max"],
        no_spatial=False,
    ):
        super(TripletAttention, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
            x_out = (1 / 3) * (x_out + x_out11 + x_out21)
        else:
            x_out = (1 / 2) * (x_out11 + x_out21)
        return x_out
    
class FEM(nn.Module):
    def __init__(self,in_ch):
        super(FEM,self).__init__()
        self.in_ch=in_ch
        self.branch_0 = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, groups=128),##nn.Conv2d
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True)
                    )
        self.branch_1 = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128),##nn.Conv2d
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True)
                    )
        self.branch_2 = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1,3), stride=1, padding=(0,1), groups=128),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True)
                    )
        self.branch_3 = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,1), stride=1, padding=(1,0), groups=128),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True)
                    )
        self.dw3x3=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, dilation=3, padding=3, groups=128)
        self.att=TripletAttention(self.in_ch)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        ##print(x.size())
        f1,f2,f3,f4=x.chunk(4,dim=1)
        f1=self.att(self.dw3x3(self.branch_0(f1)))
        f2=self.att(self.dw3x3(self.branch_1(f2)))
        f3=self.att(self.dw3x3(self.branch_2(f3)))
        f4=self.att(self.dw3x3(self.branch_3(f4)))
        f=torch.cat([f1,f2,f3,f4],dim=1)
        x=x+f
        return x

if __name__ == '__main__':
    input_images=torch.randn(8,128,160,160)
    model = TripletAttention(512)

    det_out = model(input_images)
    print("PAN output size is:", det_out.shape)
    
    from thop import profile
    flops, params = profile(model, inputs=(input_images,))##这个函数可以计算FLOPs和参数量
    print("flops:{:.3f}G".format(flops / 1e9))
    print("params:{:.3f}M".format(params / 1e6))
