import torch.nn.functional as F
import torch.nn as nn
import torch

class DeConvBnRelu(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=4,stride=2,with_relu=False,padding=1,bias=False):
        """
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param bias:
        """
        super(DeConvBnRelu,self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding,bias=bias)  # Reduce channels
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.with_relu = with_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_relu:
            x = self.relu(x)
        return x

class ConvBnRelu(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,with_relu=True,bias=False):
        """
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param bias:
        """
        super(ConvBnRelu,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,bias=bias) 
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.with_relu = with_relu
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_relu:
            x = self.relu(x)
        return x

class DWBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,bias=False):
        super(DWBlock,self).__init__()
        self.dw_conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding=kernel_size//2,groups=out_channels,bias=bias)
        self.point_conv = nn.Conv2d(out_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=bias)
        self.point_bn = nn.BatchNorm2d(out_channels)
        self.point_relu = nn.ReLU()

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.point_relu(self.point_bn(self.point_conv(x)))
        return x


def upsample(x, y, scale=1):
    _, _, H, W = y.size()
    # return F.upsample(x, size=(H // scale, W // scale), mode='nearest')
    return F.interpolate(x, size=(H // scale, W // scale), mode='nearest')

def upsample_add(x, y):
    _, _, H, W = y.size()
    # return F.upsample(x, size=(H, W), mode='nearest') + y
    return F.interpolate(x, size=(H, W), mode='nearest') + y
    
class cross_attention(nn.Module):
    
    def __init__(self):
        super(cross_attention,self).__init__()
        self.conv_attention1 = ConvBnRelu(128,128,1,1,0)
        self.conv_attention2 = ConvBnRelu(128,128,1,1,0)
        self.conv_attention3 = ConvBnRelu(128,128,1,1,0)
        
        self.conv_attention4 = ConvBnRelu(128,128,1,1,0,with_relu=False)
        self.conv_attention5 = ConvBnRelu(128,128,1,1,0,with_relu=False)
        self.conv_attention6 = ConvBnRelu(128,128,1,1,0,with_relu=False)
        self.conv_attention7 = ConvBnRelu(128,128,1,1,0,with_relu=False)
        
        self.conv_attention8 = ConvBnRelu(256,128,1,1,0)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)

    def forward(self, x):
        f_shape = x.shape
        f_theta = self.conv_attention1(x)
        f_phi = self.conv_attention2(x)
        f_g = self.conv_attention3(x)
        fh_theta = f_theta
        fh_phi = f_phi
        fh_g = f_g
        # flatten
        fh_theta = fh_theta.permute((0, 2, 3, 1))
        fh_theta = torch.reshape(fh_theta, (f_shape[0] * f_shape[2], f_shape[3], 128))
        fh_phi = fh_phi.permute((0, 2, 3, 1))
        fh_phi = torch.reshape(fh_phi, (f_shape[0] * f_shape[2], f_shape[3], 128))
        fh_g = fh_g.permute((0, 2, 3, 1))
        fh_g = torch.reshape(fh_g, (f_shape[0] * f_shape[2], f_shape[3], 128))
        # correlation
        fh_attn = torch.matmul(fh_theta, fh_phi.permute((0, 2, 1)))
        # scale
        fh_attn = fh_attn / (128 ** 0.5)
        fh_attn = F.softmax(fh_attn,-1)
        # weighted sum
        fh_weight = torch.matmul(fh_attn, fh_g)
        fh_weight = torch.reshape(fh_weight, (f_shape[0], f_shape[2], f_shape[3], 128))
        # print("fh_weight: {}".format(fh_weight.shape))
        fh_weight = fh_weight.permute((0, 3, 1, 2))

        fh_weight = self.conv_attention4(fh_weight)
        fh_sc = self.conv_attention5(x)
        f_h = F.relu(fh_weight + fh_sc)

        # vertical
        fv_theta = f_theta.permute((0, 1, 3, 2))
        fv_phi = f_phi.permute((0, 1, 3, 2))
        fv_g = f_g.permute((0, 1, 3, 2))
        # flatten
        fv_theta = fv_theta.permute((0, 2, 3, 1))
        fv_theta = torch.reshape(fv_theta, (f_shape[0] * f_shape[3], f_shape[2], 128))
        fv_phi = fv_phi.permute((0, 2, 3, 1))
        fv_phi = torch.reshape(fv_phi, (f_shape[0] * f_shape[3], f_shape[2], 128))
        fv_g = fv_g.permute((0, 2, 3, 1))
        fv_g = torch.reshape(fv_g, (f_shape[0] * f_shape[3], f_shape[2], 128))
        # correlation
        fv_attn = torch.matmul(fv_theta, fv_phi.permute((0, 2, 1)))
        # scale
        fv_attn = fv_attn / (128 ** 0.5)
        fv_attn = F.softmax(fv_attn,-1)
        # weighted sum
        fv_weight = torch.matmul(fv_attn, fv_g)
        fv_weight = torch.reshape(fv_weight, (f_shape[0], f_shape[3], f_shape[2], 128))
        # print("fv_weight: {}".format(fv_weight.shape))
        fv_weight = fv_weight.permute((0, 3, 2, 1))
        fv_weight = self.conv_attention6(fv_weight)
        # short cut
        fv_sc = self.conv_attention7(x)
        f_v = F.relu(fv_weight + fv_sc)
        ######
        f_attn = torch.cat([f_h, f_v], 1)
        f_attn = self.conv_attention8(f_attn)
        return f_attn

if __name__=="__main__":
    import torch
    input_images = torch.randn(8,128,20,20)
    
    model = cross_attention()

    det_out = model(input_images)
    print("PAN output size is:", det_out.shape)
    
    from thop import profile
    flops, params = profile(model, inputs=(input_images,))##这个函数可以计算FLOPs和参数量
    print("flops:{:.3f}G".format(flops / 1e9))
    print("params:{:.3f}M".format(params / 1e6))
