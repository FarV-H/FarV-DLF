from .utils.model_parts import *

class UNet_2D(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(UNet_2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv2D(in_channels, 64)
        self.down1 = Down(64, 128, '2d')
        self.down2 = Down(128, 256, '2d')
        self.down3 = Down(256, 512, '2d')
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, '2d')
        self.up1 = Up(1024, 512 // factor, bilinear, '2d')
        self.up2 = Up(512, 256 // factor, bilinear, '2d')
        self.up3 = Up(256, 128 // factor, bilinear, '2d')
        self.up4 = Up(128, 64, bilinear, '2d')
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x) + input
        return out

class UNet_3D(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(UNet_3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv3D(in_channels, 64)
        self.down1 = Down(64, 128, '3d')
        self.down2 = Down(128, 256, '3d')
        self.down3 = Down(256, 512, '3d')
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, '3d')
        self.up1 = Up(1024, 512 // factor, bilinear, '3d')
        self.up2 = Up(512, 256 // factor, bilinear, '3d')
        self.up3 = Up(256, 128 // factor, bilinear, '3d')
        self.up4 = Up(128, 64, bilinear, '3d')
        self.outc = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x) + input
        return out