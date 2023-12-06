import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import math

# 空洞卷积
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

# 池化 -> 1*1 卷积 -> 上采样
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        # 上采样
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)  

##整个 ASPP 总体架构
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=512):
        super(ASPP, self).__init__()
        modules = []
        # 1*1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        # 多尺度空洞卷积
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 池化
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        
        # 拼接后的卷积
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

##MSFF模块
class MSFF(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(MSFF,self).__init__()
        self.app1=nn.AdaptiveAvgPool2d(4)
        self.app2=nn.AdaptiveAvgPool2d(8)
        self.app3=nn.AdaptiveAvgPool2d(12)
        self.attention=nn.Sequential(
            nn.Conv2d(3*in_channels, out_channels, 1, bias=False),
            nn.Conv2d(out_channels,3,3,stride=1,padding=1),
            nn.Sigmoid(),
        )
    def forward(self,x):
        size=x.shape[-2:]
        y1=self.app1(x)
        y2=self.app2(x)
        y3=self.app3(x)
        y1=F.interpolate(y1, size=size, mode='bilinear', align_corners=False)
        y2=F.interpolate(y2, size=size, mode='bilinear', align_corners=False)
        y3=F.interpolate(y3, size=size, mode='bilinear', align_corners=False)
        y=torch.cat([y1,y2,y3],dim=1)
        att_map=self.attention(y)##.unsqueeze(1)
        split_att_map=torch.chunk(att_map,3,dim=1)
        print("att_map:",att_map.size())
        print(split_att_map[0].size())
        y=y1*split_att_map[0]+y2*split_att_map[1]+y3*split_att_map[2]
        return y


class pyramidPooling(nn.Module):

    def __init__(self, in_channels, pool_sizes):
        super(pyramidPooling, self).__init__()

        self.paths = []
        for i in range(len(pool_sizes)):
            self.paths.append(conv2DBatchNormRelu(in_channels, int(in_channels / len(pool_sizes)), 1, 1, 0, bias=False))

        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes

    def forward(self, x):
        output_slices = [x]
        h, w = x.shape[2:]

        for module, pool_size in zip(self.path_module_list, self.pool_sizes): 
            out = F.avg_pool2d(x, int(h/pool_size), int(h/pool_size), 0)
            out = module(out)
            out = F.upsample(out, size=(h,w), mode='bilinear')
            output_slices.append(out)

        return torch.cat(output_slices, dim=1)

##CA注意力
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=8):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv1x1 = nn.Conv2d(inp, oup, kernel_size=1, stride=1)
    
    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h)##.sigmoid()
        a_w = self.conv_w(x_w)##.sigmoid()

        out = (a_w+a_h).sigmoid()

        return out

##MAFF模块
class ScaleSpatialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, num_features, init_weight=True):
        super(ScaleSpatialAttention, self).__init__()
        self.channel_wise = nn.Sequential(   ##64-->16-->64
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, out_planes , 1, bias=False),
            ##nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            nn.Conv2d(out_planes, in_planes, 1, bias=False),
            ##nn.Sigmoid() 
        )
        self.spatial_wise = nn.Sequential(    ##获得单层的空间注意力图
            #Nx1xHxW
            nn.Conv2d(1, 1, 3, bias=False, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Conv2d(1, 1, 1, bias=False),
            nn.Sigmoid() 
        )
        self.attention_wise = nn.Sequential(    ##通道数64-->4
            nn.Conv2d(in_planes, num_features, 1, bias=False),
            nn.Sigmoid()
        )
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        global_x = torch.mean(x, dim=1, keepdim=True)##通道维度上获得平均值组成的矩阵
        global_x = self.spatial_wise(global_x) + x
        global_x = self.attention_wise(global_x)
        return global_x
    
class Conv_BN_ReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0):
        super(Conv_BN_ReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
        
class MAFF(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(MAFF, self).__init__()
        self.squre = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=1)
        self.cross_ver = nn.Conv2d(in_planes, out_planes, kernel_size=(1, 3), padding=(0, 1), stride=1)
        self.cross_hor = nn.Conv2d(in_planes, out_planes, kernel_size=(3, 1), padding=(1, 0), stride=1)
        self.dwconv = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, dilation=3, padding=3, groups=out_planes)##add a dw convlayer
        self.bn = nn.BatchNorm2d(in_planes)
        self.ReLU = nn.ReLU(True)
        self.conv1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1)
        self.attention=ScaleSpatialAttention(512,64,4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x1 =   self.dwconv(self.squre(x))
        x2 =   self.dwconv(self.cross_ver(x))
        x3 =   self.dwconv(self.cross_hor(x))
        x4 =   self.dwconv(self.conv1x1(x))
        score=self.attention(torch.cat((x1,x2,x3,x4),1))##torch.cat((x1,x2,x3,x4),1)
        ##print("x1 size:",x1.size())
        return self.ReLU(self.bn(torch.cat((x1*score[:,0:1], x2*score[:,1:2], x3*score[:,2:3], x4*score[:,3:]), dim=1)))##self.bn(self.ReLU())

if __name__ == '__main__':
    import torch
    img1 = torch.rand([8, 128, 160, 160])
    img2 = torch.rand([8,128,80,80])
    ##model = ASPP(128,[6,12,18])
    x=torch.randn(8,512,20,20)
    ##model = pyramidPooling(512, [6, 3, 2, 1])
    model=MAFF(512,128)
    ##x = model(img1,img2)
    output=model(x)
    print(output.shape)
    
    from thop import profile
    flops, params = profile(model, inputs=(x,))##这个函数可以计算FLOPs和参数量
    print("flops:{:.3f}G".format(flops / 1e9))
    print("params:{:.3f}M".format(params / 1e6))
