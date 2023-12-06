'''
This code is the integrted model for PAN.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

##from .backbone import resnet18
from .fpem import FPEM, Conv_BN_ReLU
from .ffm import FFM
from .head import PA_Head
from .resnet import deformable_resnet18,resnet18,deformable_resnet50
from .mobilenetv3 import mobilenet_v3_large
from .aspp_fpn_attention import pyramidPooling,ACBlock
from .crossatt import cross_attention
from .evc_blocks import MSF,EVCBlock
from .od_resnet import od_resnet18
from .triplet_attention import TripletAttention,FEM
import math

__all__ = ['PAN']

##feature selection module
class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan, norm="GN"):
        super(FeatureSelectionModule, self).__init__()
        self.conv_atten = nn.Conv2d(in_chan, in_chan, kernel_size=1, stride=1, bias=False )
        self.gn = nn.BatchNorm2d(in_chan)##nn.GroupNorm(8, in_chan, eps=1e-05)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        atten = self.sigmoid(self.gn(self.conv_atten(F.avg_pool2d(x, x.size()[2:])))) #fm 就是激活加卷积，我觉得这平均池化用的贼巧妙
        feat = torch.mul(x, atten) #相乘，得到重要特征
        x = x + feat #再加上
        feat = self.relu(self.bn(self.conv(x))) #最后一层 1*1 的卷积
        return feat

class PAN(nn.Module):
    def __init__(self, pretrained, neck_channel, pa_in_channels, hidden_dim, num_classes):
        super(PAN, self).__init__()
        self.backbone = deformable_resnet50(pretrained=pretrained)##deformable_resnet18
        in_channels = neck_channel
        self.reduce_layer1 = Conv_BN_ReLU(in_channels[0], 128)##FeatureSelectionModule(in_channels[0], 128)
        self.reduce_layer2 = Conv_BN_ReLU(in_channels[1], 128)##FeatureSelectionModule(in_channels[1], 128)
        self.reduce_layer3 = Conv_BN_ReLU(in_channels[2], 128)##FeatureSelectionModule(in_channels[2], 128)
        self.reduce_layer4 = Conv_BN_ReLU(in_channels[3], 128)##FeatureSelectionModule(in_channels[3], 128)##ACBlock(in_channels[3], 128)
        self.fpem1 = FPEM(128, 128)
        self.fpem2 = FPEM(128, 128)

        self.ffm = FFM()
        '''
        self.ffmv2 = FFMv2()
        self.pp=pyramidPooling(512,[6,3,2,1])
        
        self.fsm=FeatureSelectionModule(512,512)
        self.aspp=ASPP_module(512,128,[6,12,18])
        
        self.ulsam=ULSAM(512,512,20,20,4)
        self.croatt=cross_attention()
        '''
        self.acb=ACBlock(512,128)
        self.msf=MSF()
        self.evc=EVCBlock(128,128)
        self.rfb=RFBblock(128,True)

        self.det_head = PA_Head(pa_in_channels, hidden_dim, num_classes)
    
    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.interpolate(x, size=(H // scale, W // scale), mode='bilinear')
    
    def forward(self, imgs):
        # backbone
        f = self.backbone(imgs)
        ##f[0]=f[0]+self.tr0(f[0])
        ##f_agg=self.evc(self.msf(f[0],f[1],f[2],f[3]))
        ##f_agg=self.evc(f[3])

        # reduce channel
        f1 = self.reduce_layer1(f[0])
        f2 = self.reduce_layer2(f[1])
        f3 = self.reduce_layer3(f[2])
        f4 = self.reduce_layer4(self.acb(f[3]))##f[3]##self.acb()self.reduce_layer4(f[3]) self.fsm(f[3])self.aspp(f[3])
        ##f11,f22,f33,f44=f1,f2,f3,f4
        ##f_agg=self.evc(f4)
        '''
        f1=f1+F.interpolate(f22, size=f[0].shape[2:], mode='bilinear')
        f2=F.avg_pool2d(f11,kernel_size=2,stride=2)+f2+F.interpolate(f33, size=f[1].shape[2:], mode='bilinear')
        f3=F.avg_pool2d(f22,kernel_size=2,stride=2)+f3+F.interpolate(f44, size=f[2].shape[2:], mode='bilinear')
        f4=F.avg_pool2d(f33,kernel_size=2,stride=2)+f4
        '''
        
        ##f1=f1+F.interpolate(f_agg, size=f[0].shape[2:], mode='bilinear', align_corners=False)
        ##f2=f2+F.interpolate(f_agg, size=f[1].shape[2:], mode='bilinear', align_corners=False)
        ##f3=f3+F.interpolate(f_agg, size=f[2].shape[2:], mode='bilinear', align_corners=False)
        ##f4=f4+F.interpolate(f_agg, size=f[3].shape[2:], mode='bilinear', align_corners=False)
        ##f4=self.rfb(f_agg)
        
        # FPEM
        f1_1, f2_1, f3_1, f4_1 = self.fpem1(f1, f2, f3, f4)
        f1_2, f2_2, f3_2, f4_2 = self.fpem2(f1_1, f2_1, f3_1, f4_1)

        # FFM
        f = self.ffm(f1_1, f2_1, f3_1, f4_1, f1_2, f2_2, f3_2, f4_2)
        ##f=self.ffmv2(f1_1, f2_1, f3_1, f4_1)

        # detection
        det_out = self.det_head(f)

        return det_out

# unit testing
if __name__ == '__main__':
    
    batch_size = 1
    Height = 640
    Width = 640
    neck_channel = [64, 128, 256, 512]
    pa_in_channels = 512
    hidden_dim = 128
    Channel = 3

    input_images = torch.randn(batch_size,Channel,Height,Width)
    
    model = PAN(pretrained=False, neck_channel=neck_channel, pa_in_channels=pa_in_channels, hidden_dim=hidden_dim, num_classes=6)

    det_out = model(input_images)
    print("PAN output size is:", det_out.shape)
    
    from thop import profile
    flops, params = profile(model, inputs=(input_images,))##这个函数可以计算FLOPs和参数量
    print("flops:{:.3f}G".format(flops / 1e9))
    print("params:{:.3f}M".format(params / 1e6))
