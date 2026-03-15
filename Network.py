import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity 
        out = self.relu(out)
        return out

class Encoder(nn.Module):
  def __init__(self, in_channels=3):
    super().__init__()
    self.pool = nn.MaxPool2d(2)
    self.block0 = ResNetBlock(in_channels, 64)    
    self.block1 = ResNetBlock(64, 128)            
    self.block2 = ResNetBlock(128, 256)            
    self.block3 = ResNetBlock(256, 512)            
    self.block4 = ResNetBlock(512, 1024)      
         
  def forward(self, x):
        x_00 = self.block0(x)
        x_10 = self.block1(self.pool(x_00))
        x_20 = self.block2(self.pool(x_10))
        x_30 = self.block3(self.pool(x_20))
        x_40 = self.block4(self.pool(x_30))
        return x_00, x_10, x_20, x_30, x_40


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #1
        self.x_01 = ResNetBlock(64 + 128, 64)
        self.x_11 = ResNetBlock(128 + 256, 128)
        self.x_21 = ResNetBlock(256 + 512, 256)
        self.x_31 = ResNetBlock(512 + 1024, 512)
        #2
        self.x_02 = ResNetBlock(64 + 64 + 128, 64)
        self.x_12 = ResNetBlock(128 + 128 + 256, 128)
        self.x_22 = ResNetBlock(256 + 256 + 512, 256)
        #3
        self.x_03 = ResNetBlock(64 * 3 + 128, 64)
        self.x_13 = ResNetBlock(128 * 3 + 256, 128)
        # 4
        self.x_04 = ResNetBlock(64 * 4 + 128, 64)
    def forward(self, x_00, x_10, x_20, x_30, x_40):
        #  1
        x_01 = self.x_01(torch.cat([x_00, self.up(x_10)], dim=1))
        x_11 = self.x_11(torch.cat([x_10, self.up(x_20)], dim=1))
        x_21 = self.x_21(torch.cat([x_20, self.up(x_30)], dim=1))
        x_31 = self.x_31(torch.cat([x_30, self.up(x_40)], dim=1))
        # 2
        x_02 = self.x_02(torch.cat([x_00, x_01, self.up(x_11)], dim=1))
        x_12 = self.x_12(torch.cat([x_10, x_11, self.up(x_21)], dim=1))
        x_22 = self.x_22(torch.cat([x_20, x_21, self.up(x_31)], dim=1))
        # 3
        x_03 = self.x_03(torch.cat([x_00, x_01, x_02, self.up(x_12)], dim=1))
        x_13 = self.x_13(torch.cat([x_10, x_11, x_12, self.up(x_22)], dim=1))
        # 4
        x_04 = self.x_04(torch.cat([x_00, x_01, x_02, x_03, self.up(x_13)], dim=1))
    
        return x_01, x_02, x_03, x_04
        

class UNetPlusPlus(nn.Module):
    
    def __init__(self, in_channels=3, num_classes=3, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder()
        if self.deep_supervision:
            self.out1 = nn.Conv2d(64, num_classes, kernel_size=1)
            self.out2 = nn.Conv2d(64, num_classes, kernel_size=1)
            self.out3 = nn.Conv2d(64, num_classes, kernel_size=1)
            self.out4 = nn.Conv2d(64, num_classes, kernel_size=1)
        else:
            self.out = nn.Conv2d(64, num_classes, kernel_size=1)
            
    def forward(self, x):
        x_00, x_10, x_20, x_30, x_40 = self.encoder(x)
        x_01, x_02, x_03, x_04 = self.decoder(x_00, x_10, x_20, x_30, x_40)
        if self.deep_supervision:
            return [self.out1(x_01), self.out2(x_02), self.out3(x_03), self.out4(x_04)]
        else:
            return self.out(x_04)
