'''
This code is for FFM model in PAN.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['FFM']

class attention(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=128, r=8):
        super(attention, self).__init__()
        inter_channels = int(channels // r)
        
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1,3), stride=1, padding=(0,1),groups=channels),
            nn.Conv2d(channels, channels, kernel_size=(3,1), stride=1, padding=(1,0),groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        
        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        
        '''
        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att2 = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        '''
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual): 
        xa = x + residual
        xl = self.local_att(xa)
        ##xg = self.global_att(xa)
        ##xl = self.global_att2(xa)##max pool
        ##xlg = xl + xg
        xlg=xl
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)
        return xi
        
class FFM(nn.Module):
    def __init__(self):
        super(FFM, self).__init__()
        self.attention=attention()

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.interpolate(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self, f1_1, f2_1, f3_1, f4_1, f1_2, f2_2, f3_2, f4_2):
        f1 = f1_1 + f1_2##self.attention(f1_1, f1_2)##f1_1 + f1_2
        f2 = f2_1 + f2_2##self.attention(f2_1, f2_2)##f2_1 + f2_2
        f3 = f3_1 + f3_2##self.attention(f3_1, f3_2)##f3_1 + f3_2
        f4 = f4_1 + f4_2##self.attention(f4_1, f4_2)##f4_1 + f4_2

        f2 = self._upsample(f2, f1.size())
        f3 = self._upsample(f3, f1.size())
        f4 = self._upsample(f4, f1.size())

        f = torch.cat((f1, f2, f3, f4), 1)
        ##
        ##fuse = self.sfs(f, [f4, f3, f2, f1])
        

        return f

# unit testing
if __name__ == '__main__':
    batch_size = 32
    Height = 512
    Width = 768
    Channel = 128
    f1_1 = torch.randn(batch_size,Channel,Height//4,Width//4)
    f2_1 = torch.randn(batch_size,Channel,Height//8,Width//8)
    f3_1 = torch.randn(batch_size,Channel,Height//16,Width//16)
    f4_1 = torch.randn(batch_size,Channel,Height//32,Width//32)

    f1_2 = torch.randn(batch_size,Channel,Height//4,Width//4)
    f2_2 = torch.randn(batch_size,Channel,Height//8,Width//8)
    f3_2 = torch.randn(batch_size,Channel,Height//16,Width//16)
    f4_2 = torch.randn(batch_size,Channel,Height//32,Width//32)

    ffm_model = FFM()
    f = ffm_model(f1_1, f2_1, f3_1, f4_1, f1_2, f2_2, f3_2, f4_2)
    print("FFM input layer 1 shape:", f1_1.shape)
    print("FFM input layer 2 shape:", f2_1.shape)
    print("FFM input layer 3 shape:", f3_1.shape)
    print("FFM input layer 4 shape:", f4_1.shape)
    print("FFM output shape:", f.shape)


    import profile
    flops, params = profile(ffm_model, inputs=(input, [f4_2, f3_2, f2_2, f1_2]))
    print("flops:",flops)
    print("params:",params)
