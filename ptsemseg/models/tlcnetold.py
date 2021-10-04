'''
nad, fwd, bwd network
refer to PSMN
'''

from __future__ import print_function
from ptsemseg.models.submodule import *
from ptsemseg.models.utils import unetUpsimple

class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes * 2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes))  # +x

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post


class TLCNet(nn.Module):
    def __init__(self, maxdisp):
        super(TLCNet, self).__init__()
        self.maxdisp = maxdisp # max floor

        self.feature_extraction = feature_extraction()
        # add
        # self.dropout = nn.Dropout2d(p=0.5, inplace=False)
        # 1/4 w x h
        self.up1 = unetUpsimple(32, 32, True)
        self.up2 = unetUpsimple(32, 32, True)
        self.up3 = nn.Conv2d(32, 1, 1)
        #old
        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, tlc):
        heights=tlc.size()[2]
        widths=tlc.size()[3]
        # weight sharing
        muximg_fea = self.feature_extraction(tlc[:, 0:3, :, :]) # need decoder N x C x 1/4 H x 1/4 W
        refimg_fea = self.feature_extraction(tlc[:, 3:6, :, :])
        targetimg_fea = self.feature_extraction(tlc[:, 6:9, :, :])
        bwdimg_fea = self.feature_extraction(tlc[:, 9:12, :, :]) # need cost fusion

        # learning fine border: predicting building boundary
        costmux = self.up1(muximg_fea)
        costmux = self.up2(costmux)
        costmux = self.up3(costmux)
        predmux = torch.squeeze(costmux, 1)
        # return predmux

        cost1 = self.costvolume(refimg_fea, targetimg_fea)
        cost2 = self.costvolume(refimg_fea, bwdimg_fea)
        # max pooling to aggregate different volumes
        cost = torch.max(cost1, cost2)
        # correlation layer: fail to compile
        # costa = correlate(refimg_fea, targetimg_fea) #
        # costb = correlate(refimg_fea, bwdimg_fea) #
        # cost = torch.cat((costa, costb, muximg_fea), 1)  # concatenate mux

        cost = cost.contiguous() # contigous in memory

        # 3D CNN
        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        # deep supervision
        if self.training:
            cost1 = F.interpolate(cost1, [self.maxdisp, heights, widths], mode='trilinear', align_corners=True)
            cost2 = F.interpolate(cost2, [self.maxdisp, heights, widths], mode='trilinear', align_corners=True)

            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparityregression(self.maxdisp)(pred1)

            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparityregression(self.maxdisp)(pred2)

        cost3 = F.interpolate(cost3, [self.maxdisp, heights, widths], mode='trilinear', align_corners=True)
        cost3 = torch.squeeze(cost3, 1)
        pred3 = F.softmax(cost3, dim=1)
        # For your information: This formulation 'softmax(c)' learned "similarity"
        # while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        # However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        pred3 = disparityregression(self.maxdisp)(pred3)

        # return three supervision only when training
        # predict the floor number
        if self.training:
            return pred1, pred2, pred3, predmux
        else:
            return pred3

    def costvolume(self, refimg_fea , targetimg_fea):
        # matching: aggregate cost by conjunction along the disparity dimension
        # Shape: N x 2C x D/4 x (1/4)H x (1/4)W
        cost = Variable(
            torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1] * 2, self.maxdisp // 4, refimg_fea.size()[2],
                              refimg_fea.size()[3]).zero_()).cuda()
        for i in range(self.maxdisp // 4):
            if i > 0:
                cost[:, :refimg_fea.size()[1], i, :, i:] = refimg_fea[:, :, :, i:]
                cost[:, refimg_fea.size()[1]:, i, :, i:] = targetimg_fea[:, :, :, :-i]
            else:
                cost[:, :refimg_fea.size()[1], i, :, :] = refimg_fea
                cost[:, refimg_fea.size()[1]:, i, :, :] = targetimg_fea
        return cost


class PSMNet(nn.Module):
    def __init__(self, maxdisp):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp

        self.feature_extraction = feature_extraction()

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right):

        refimg_fea = self.feature_extraction(left)
        targetimg_fea = self.feature_extraction(right)

        # matching: aggregate cost
        cost = Variable(
            torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1] * 2, self.maxdisp // 4, refimg_fea.size()[2],
                              refimg_fea.size()[3]).zero_()).cuda()

        for i in range(self.maxdisp // 4):
            if i > 0:
                cost[:, :refimg_fea.size()[1], i, :, i:] = refimg_fea[:, :, :, i:]
                cost[:, refimg_fea.size()[1]:, i, :, i:] = targetimg_fea[:, :, :, :-i]
            else:
                cost[:, :refimg_fea.size()[1], i, :, :] = refimg_fea
                cost[:, refimg_fea.size()[1]:, i, :, :] = targetimg_fea
        cost = cost.contiguous()

        # 3D CNN
        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        # deep supervision
        if self.training:
            cost1 = F.upsample(cost1, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')

            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparityregression(self.maxdisp)(pred1)

            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparityregression(self.maxdisp)(pred2)

        cost3 = F.upsample(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
        cost3 = torch.squeeze(cost3, 1)
        pred3 = F.softmax(cost3, dim=1)
        # For your information: This formulation 'softmax(c)' learned "similarity"
        # while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        # However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        pred3 = disparityregression(self.maxdisp)(pred3)

        # return three supervision only when training
        if self.training:
            return pred1, pred2, pred3
        else:
            return pred3