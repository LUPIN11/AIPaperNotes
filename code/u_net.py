import torch
import torch.nn as nn
import torch.utils.checkpoint as C


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class U_Net(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(U_Net, self).__init__()
        self.conv0 = Conv(n_channels, 64)
        self.conv1 = Conv(64, 128)
        self.conv2 = Conv(128, 256)
        self.conv3 = Conv(256, 512)
        self.conv4 = Conv(512, 1024)
        self.conv5 = Conv(1024, 512)
        self.conv6 = Conv(512, 256)
        self.conv7 = Conv(256, 128)
        self.conv8 = Conv(128, 64)
        self.maxpool = nn.MaxPool2d(2)
        self.convT0 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.convT1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.convT2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.convT3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.outconv = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        #         # contracting path
        #         x0 = self.conv0(x)
        #         x1 = self.conv1(self.maxpool(x0))
        #         x2 = self.conv2(self.maxpool(x1))
        #         x3 = self.conv3(self.maxpool(x2))
        #         x = self.conv4(self.maxpool(x3))
        #         # expanding path
        #         x = self.conv5(self.concat(self.convT0(x), x3))
        #         x = self.conv6(self.concat(self.convT1(x), x2))
        #         x = self.conv7(self.concat(self.convT0(x), x1))
        #         x = self.conv8(self.concat(self.convT0(x), x0))
        #         return self.outconv(x)
        # contracting path
        x0 = C.checkpoint(self.conv0, x)
        # print(x0.size())
        x1 = C.checkpoint(self.conv1, C.checkpoint(self.maxpool, x0))
        # print(x1.size())
        x2 = C.checkpoint(self.conv2, C.checkpoint(self.maxpool, x1))
        # print(x2.size())
        x3 = C.checkpoint(self.conv3, C.checkpoint(self.maxpool, x2))
        # print(x3.size())
        x = C.checkpoint(self.conv4, C.checkpoint(self.maxpool, x3))
        # print(x.size())
        # expanding path
        x = C.checkpoint(self.conv5, self.concat(C.checkpoint(self.convT0, x), x3))
        # print(x.size())
        x = C.checkpoint(self.conv6, self.concat(C.checkpoint(self.convT1, x), x2))
        # print(x.size())
        x = C.checkpoint(self.conv7, self.concat(C.checkpoint(self.convT2, x), x1))
        # print(x.size())
        x = C.checkpoint(self.conv8, self.concat(C.checkpoint(self.convT3, x), x0))
        # print(x.size())
        return C.checkpoint(self.outconv, x)

    @staticmethod
    def concat(x_e, x_c):
        diff_h = x_c.size()[2] - x_e.size()[2]
        diff_w = x_c.size()[3] - x_e.size()[3]
        x_c = x_c[:, :, diff_h // 2:-(diff_h - diff_h // 2), diff_w // 2:-(diff_w - diff_w // 2)]
        return torch.cat([x_c, x_e], dim=1)
