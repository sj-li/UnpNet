import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1

def get_group_mask(xyz, threshold=0.6, kernel_size=3, dilation=1, padding=1, stride=1):
    N,C,H,W = xyz.size()

    center = xyz
    xyz = F.unfold(xyz, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride).view(N, C, kernel_size*kernel_size, H, W)
    group_xyz = xyz - center.unsqueeze(2)
    dists = torch.sqrt(torch.sum(group_xyz*group_xyz, 1))

    mask_valid = (torch.sum(center*center, 1)>0).unsqueeze(1).repeat(1, kernel_size*kernel_size, 1, 1).float()
    mask = (dists < threshold).float()

    dists = 1.0 / (dists + 1e-4)
    dists *= mask
    dists *= mask_valid
    norm = torch.sum(dists, dim=2, keepdim=True)+1e-4
    weight = dists / norm

    return weight, group_xyz


def PMaxpooling(feat, group_mask):
    feat = feat*(group_mask != 0).float()
    feat, _ = torch.max(feat, 2)

    return feat

def PAvgpooling(feat, group_mask):
    feat = feat*group_mask
    feat = torch.sum(feat, 2)

    return feat

def PSumpooling(feat, group_mask):
    feat = feat*(group_mask != 0).float()
    feat, _ = torch.sum(feat, 2)

    return feat

class PDownsample(nn.Module):
    def __init__(self, factor):
        super(PDownsample, self).__init__()
        self.factor = factor

    def forward(self, x):
        N, C, W, H = x.size()
        new = torch.zeros([N, C, int(W/self.factor), int(H/self.factor)]).cuda()
        #new = torch.zeros([N, C, int(W/self.factor), int(H/self.factor)])
        new = x[:, :, ::factor, ::factor]

        return x


class PUpsample(nn.Module):
    def __init__(self, factor):
        super(PUpsample, self).__init__()
        self.factor = factor

    def forward(self, x):
        N, C, W, H = x.size()
        new = torch.zeros([N, C, W*self.factor, H*self.factor]).cuda()
        #new = torch.zeros([N, C, W*self.factor, H*self.factor])
        new[:, :, ::self.factor, ::self.factor] = x

        return new

class WeightNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit = [8, 8]):
        super(WeightNet, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        
    def forward(self, localized_xyz):
        #xyz : BxCxKxN

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            weights =  F.relu(bn(conv(weights)))

        return weights

class PConv(nn.Module):
    def __init__(self, in_channel, mlp, kernel_size=3, dilation=1, padding=1, stride=1, group_operation=PMaxpooling):
        super(PConv, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        self.group_operation = group_operation
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.linear = nn.Conv2d(16 * mlp[-1], mlp[-1], 1)
        self.bn_linear = nn.BatchNorm2d(mlp[-1])

        self.weightnet = WeightNet(3, 16)

    def forward(self, x, group_mask, group_xyz):
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            x = F.relu(bn(conv(x)))


        B, C, N, H, W = group_xyz.shape
        group_xyz = self.weightnet(group_xyz.view(B, C, N, -1)).view(B, -1, N, H, W)

        B,C,H,W = x.size()
        
        x = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, dilation=self.dilation, stride=self.stride).view(B, C, self.kernel_size*self.kernel_size, group_mask.size(-2), group_mask.size(-1))

        x = torch.matmul(input=x.permute(0, 3, 4, 1, 2), other=group_xyz.permute(0, 3, 4, 2, 1)).view(B, group_mask.size(-2), group_mask.size(-1), -1)
        x = self.linear(x.permute(0, 3, 1, 2))
        x = self.bn_linear(x)
        x = F.relu(x)
        #x = self.group_operation(x, group_mask.unsqueeze(1))

        return x

class PDConv(nn.Module):
    def __init__(self, in_channel, mlp, kernel_size=3, padding=1, factor=1, group_operation=PMaxpooling):
        super(PDConv, self).__init__()
        self.kernel_size = kernel_size
        self.padding=padding
        self.up = PUpsample(factor)
        self.group_operation = group_operation
        
        #self.mlp_convs = nn.ModuleList()
        #self.mlp_bns = nn.ModuleList()
        #last_channel = in_channel
        #for out_channel in mlp:
        #    self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
        #    self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        #    last_channel = out_channel

    def forward(self, x_pre, x, group_mask):
        x = self.up(x)
        B, C, H, W = x.size()
        x = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding).view(B, C, self.kernel_size*self.kernel_size, group_mask.size(-2), group_mask.size(-1))
        x = self.group_operation(x, group_mask.unsqueeze(1))

        if isinstance(x_pre, torch.Tensor):
            x = torch.cat([x, x_pre], 1)
        #for i, conv in enumerate(self.mlp_convs):
        #    bn = self.mlp_bns[i]
        #    x = F.relu(bn(conv(x)))

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, padding=1, bias=False)
        # self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(ResContextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)


    def forward(self, x):

        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(shortcut)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        output = shortcut + resA2
        return output


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3,3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3,3),dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters*3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            #self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
            self.pconv_d = PConv(out_filters, [out_filters, out_filters, out_filters], kernel_size=5, dilation=1, padding=2, stride=2, group_operation=PMaxpooling)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)



    def forward(self, x, l_xyz, group_xyz):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        resA = self.conv4(resA2)
        resA = self.act4(resA)
        resA3 = self.bn3(resA)

        concat = torch.cat((resA1,resA2,resA3),dim=1)
        resA = self.conv5(concat)
        resA = self.act5(resA)
        resA = self.bn4(resA)
        resA = shortcut + resA


        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            #resB = self.pool(resB)
            resB = self.pconv_d(resB, l_xyz, group_xyz)

            return resB, resA
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, drop_out=True, in_filters_=512):
        super(UpBlock, self).__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.dropout2 = nn.Dropout2d(p=dropout_rate)
        

        #in_out_c = in_filters//4 + 2*out_filters
        #self.pconv_u = PDConv(in_filters, 512, [in_out_c, in_out_c, in_out_c], kernel_size=5, padding=2, factor=2, group_operation=PMaxpooling)
        self.pconv_u = PDConv(in_filters, [in_filters, in_filters//4 + 2*out_filters], kernel_size=5, padding=2, factor=2, group_operation=PMaxpooling)

        self.conv1 = nn.Conv2d(in_filters_, out_filters, (3,3), padding=1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (2,2), dilation=2,padding=1)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)


        self.conv4 = nn.Conv2d(out_filters*3,out_filters,kernel_size=(1,1))
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        self.dropout3 = nn.Dropout2d(p=dropout_rate)

        

    def forward(self, x, skip, l_xyz, group_xyz):
        upB = self.pconv_u(skip, x, l_xyz)
        #upA = nn.PixelShuffle(2)(x)
        #if self.drop_out:
        #    upA = self.dropout1(upA)

        #upB = torch.cat((upA,skip),dim=1)
        if self.drop_out:
            upB = self.dropout2(upB)

        upE = self.conv1(upB)
        upE = self.act1(upE)
        upE1 = self.bn1(upE)

        upE = self.conv2(upE1)
        upE = self.act2(upE)
        upE2 = self.bn2(upE)

        upE = self.conv3(upE2)
        upE = self.act3(upE)
        upE3 = self.bn3(upE)

        concat = torch.cat((upE1,upE2,upE3),dim=1)
        upE = self.conv4(concat)
        upE = self.act4(upE)
        upE = self.bn4(upE)
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE

class UnpNet(nn.Module):
    def __init__(self, in_channels, nclasses, drop=0, use_mps=True):
        super(UnpNet, self).__init__()
        self.nclasses = nclasses

        self.block_in = self._make_layer(BasicBlock, 5, 32, 8, 1)

        self.resBlock1 = ResBlock(32, 2 * 32, 0.2, pooling=True, drop_out=False)
        self.resBlock2 = ResBlock(2 * 32, 2 * 2 * 32, 0.2, pooling=True)
        self.resBlock3 = ResBlock(2 * 2 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock5 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=False)

        self.upBlock1 = UpBlock(2 * 4 * 32, 4 * 32, 0.2, in_filters_=512)
        self.upBlock2 = UpBlock(4 * 32, 4 * 32, 0.2, in_filters_=384)
        self.upBlock3 = UpBlock(4 * 32, 2 * 32, 0.2, in_filters_=256)
        self.upBlock4 = UpBlock(2 * 32, 32, 0.2, drop_out=False, in_filters_=128)

        self.logits = nn.Conv2d(32, nclasses, kernel_size=(1, 1))

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)



    def forward(self, x):

        xyz_0 = x[:, 1:4, :, :]
        
        l0_xyz, group0_xyz= get_group_mask(xyz_0, 0.2, kernel_size=5, padding=2)
        xyz_1 = F.interpolate(xyz_0, scale_factor=0.5, mode='nearest')
        l1_xyz, group1_xyz= get_group_mask(xyz_1, 0.2, kernel_size=5, padding=2)
        xyz_2 = F.interpolate(xyz_1, scale_factor=0.5, mode='nearest')
        l2_xyz, group2_xyz = get_group_mask(xyz_2, 0.4, kernel_size=5, padding=2)
        xyz_3 = F.interpolate(xyz_2, scale_factor=0.5, mode='nearest')
        l3_xyz, group3_xyz = get_group_mask(xyz_3, 0.6, kernel_size=5, padding=2)
        xyz_4 = F.interpolate(xyz_3, scale_factor=0.5, mode='nearest')
        l4_xyz, group4_xyz = get_group_mask(xyz_4, 0.8, kernel_size=5, padding=2)

        downCntx = self.block_in(x)

        down0c, down0b = self.resBlock1(downCntx, l1_xyz, group1_xyz)
        down1c, down1b = self.resBlock2(down0c, l2_xyz, group2_xyz)
        down2c, down2b = self.resBlock3(down1c, l3_xyz, group3_xyz)
        down3c, down3b = self.resBlock4(down2c, l4_xyz, group4_xyz)
        down5c = self.resBlock5(down3c, None, None)

        up4e = self.upBlock1(down5c,down3b, l3_xyz, group3_xyz)
        up3e = self.upBlock2(up4e, down2b, l2_xyz, group2_xyz)
        up2e = self.upBlock3(up3e, down1b, l1_xyz, group1_xyz)
        up1e = self.upBlock4(up2e, down0b, l0_xyz, group0_xyz)
        logits = self.logits(up1e)

        logits = logits
        logits = F.softmax(logits, dim=1)
        return logits, {}
