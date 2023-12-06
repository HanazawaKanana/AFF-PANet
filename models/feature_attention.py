import torch
import torch.nn as nn
import torch.nn.functional as F
##from attention import *

##以下给了三种DBNet在特征融合模块上运用的注意力模块
class ScaleChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes, num_features, init_weight=True):##in_planes=64 out_planes=32 num_features=4
        super(ScaleChannelAttention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, out_planes, 1, bias=False)##64->32
        self.bn = nn.BatchNorm2d(out_planes)
        self.fc2 = nn.Conv2d(out_planes, num_features, 1, bias=False)##32->4
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
        global_x = self.avgpool(x)
        global_x = self.fc1(global_x)
        global_x = F.relu(self.bn(global_x))
        global_x = self.fc2(global_x)
        global_x = F.softmax(global_x, 1)
        print("输出的global_x:",global_x.size())
        return global_x

class ScaleChannelSpatialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, num_features, init_weight=True):
        super(ScaleChannelSpatialAttention, self).__init__()
        self.channel_wise = nn.Sequential(   ##64-->16-->64
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, out_planes , 1, bias=False),
            # nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            nn.Conv2d(out_planes, in_planes, 1, bias=False)
        )
        self.spatial_wise = nn.Sequential(
            #Nx1xHxW
            nn.Conv2d(1, 1, 3, bias=False, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.attention_wise = nn.Sequential(
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

    def forward(self, x):##这是采用的串联的方式
        # global_x = self.avgpool(x)
        #shape Nx4x1x1
        global_x = self.channel_wise(x).sigmoid()
        #shape: NxCxHxW
        global_x = global_x + x  ##加上通道级的
        #shape:Nx1xHxW
        x = torch.mean(global_x, dim=1, keepdim=True)
        global_x = self.spatial_wise(x) + global_x  ##每个通道都会加上该空间级的矩阵
        global_x = self.attention_wise(global_x)
        return global_x

class ScaleSpatialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, num_features, init_weight=True):
        super(ScaleSpatialAttention, self).__init__()
        self.spatial_wise = nn.Sequential(    ##获得单层的空间注意力图
            #Nx1xHxW
            nn.Conv2d(1, 1, 3, bias=False, padding=1),
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
        global_x = self.spatial_wise(global_x) + x##这里会有广播机制，直接相加，还是得到64通道的特征图
        global_x = self.attention_wise(global_x)
        return global_x
        
        
class ScaleEcaSpatialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, num_features, init_weight=True):
        super(ScaleEcaSpatialAttention, self).__init__()
        self.channel_wise = eca(64)
        self.spatial_wise = nn.Sequential(
            #Nx1xHxW
            nn.Conv2d(1, 1, 3, bias=False, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.attention_wise = nn.Sequential(
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

    def forward(self, x):##这是采用的串联的方式
        # global_x = self.avgpool(x)
        #shape Nx4x1x1
        global_x = self.channel_wise(x).sigmoid()
        #shape: NxCxHxW
        ##global_x = global_x + x  ##加上通道级的
        #shape:Nx1xHxW
        x = torch.mean(global_x, dim=1, keepdim=True)##在通道维度上取平均值
        global_x = self.spatial_wise(x) + global_x  ##每个通道都会加上该空间级的矩阵
        global_x = self.attention_wise(global_x)
        return global_x

class ScaleFeatureSelection(nn.Module):
    def __init__(self, in_channels, inter_channels , out_features_num=4, attention_type='scale_spatial'):
        super(ScaleFeatureSelection, self).__init__()
        self.in_channels=in_channels
        self.inter_channels = inter_channels
        self.out_features_num = out_features_num
        self.conv = nn.Conv2d(in_channels, inter_channels, 3, padding=1)
        self.type = attention_type
        if self.type == 'scale_spatial':##根据输入的type选择注意力类型
            self.enhanced_attention = ScaleSpatialAttention(inter_channels, inter_channels//4, out_features_num)
        elif self.type == 'scale_channel_spatial':
            self.enhanced_attention = ScaleChannelSpatialAttention(inter_channels, inter_channels // 4, out_features_num)
        elif self.type == 'scale_channel':
            self.enhanced_attention = ScaleChannelAttention(inter_channels, inter_channels//2, out_features_num)
        elif self.type == 'scale_eca_spatial':##加入这个新的，暂命名为这个
            self.enhanced_attention = ScaleEcaSpatialAttention(inter_channels, inter_channels//4, out_features_num)

    def _initialize_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
    def forward(self, concat_x, features_list):
        concat_x = self.conv(concat_x)
        score = self.enhanced_attention(concat_x)
        ##print("看看score：",score.size())
        assert len(features_list) == self.out_features_num##这里得保证score的通道数和特征列表的长度相等
        if self.type not in ['scale_channel_spatial', 'scale_spatial']:
            shape = features_list[0].shape[2:]
            score = F.interpolate(score, size=shape, mode='bilinear')
        x = []
        for i in range(self.out_features_num):
            x.append(score[:, i:i+1] * features_list[i])##这里先保存在张量列表中，然后拼接，另外，乘法也有广播机制
        return torch.cat(x, dim=1)
        
if __name__=="__main__":
    input = torch.randn(2,256,160,160)
    p5 = torch.randn(2,64,160,160)
    p4 = torch.randn(2,64,160,160)
    p3 = torch.randn(2,64,160,160)
    p2 = torch.randn(2,64,160,160)
    concat_attention = ScaleFeatureSelection(256,64,attention_type = 'scale_spatial')
    fuse = concat_attention(input, [p5, p4, p3, p2])
    print(fuse.size())
    
    from thop import profile
    flops, params = profile(concat_attention, inputs=(input, [p5, p4, p3, p2]))
    print("flops:{:.3f}G".format(flops / 1e9))
    print("params:{:.3f}M".format(params / 1e6))
