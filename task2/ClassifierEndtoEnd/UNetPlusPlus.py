import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ResNetBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
      
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        # Squeeze & Excitation
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Identity Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        
        # Apply SE scaling
        out = out * self.se(out)
        
        out += identity 
        return self.relu(out)


class UNetPlusPlus(nn.Module):
    def __init__(self, encoder_name='seresnet50', pretrained=True, num_classes=3, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        
        self.encoder = timm.create_model(
            encoder_name, 
            pretrained=pretrained, 
            features_only=True, 
            out_indices=(0, 1, 2, 3, 4)
        )

        nb = self.encoder.feature_info.channels()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Column 1
        self.x_01 = ResNetBlock(nb[0] + nb[1], nb[0])
        self.x_11 = ResNetBlock(nb[1] + nb[2], nb[1])
        self.x_21 = ResNetBlock(nb[2] + nb[3], nb[2])
        self.x_31 = ResNetBlock(nb[3] + nb[4], nb[3])

        # Column 2
        self.x_02 = ResNetBlock(nb[0]*2 + nb[1], nb[0])
        self.x_12 = ResNetBlock(nb[1]*2 + nb[2], nb[1])
        self.x_22 = ResNetBlock(nb[2]*2 + nb[3], nb[2])

        # Column 3
        self.x_03 = ResNetBlock(nb[0]*3 + nb[1], nb[0])
        self.x_13 = ResNetBlock(nb[1]*3 + nb[2], nb[1])

        # Column 4
        self.x_04 = ResNetBlock(nb[0]*4 + nb[1], nb[0])

        # Final Head
        if self.deep_supervision:
            self.out1 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
            self.out2 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
            self.out3 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
            self.out4 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
        else:
            self.out = nn.Conv2d(nb[0], num_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        # features is a list of tensors: [x00, x10, x20, x30, x40]
        features = self.encoder(x)
        x_00, x_10, x_20, x_30, x_40 = features

        # --- Decoder Nesting logic ---
        x_01 = self.x_01(torch.cat([x_00, self.up(x_10)], 1))
        x_11 = self.x_11(torch.cat([x_10, self.up(x_20)], 1))
        x_21 = self.x_21(torch.cat([x_20, self.up(x_30)], 1))
        x_31 = self.x_31(torch.cat([x_30, self.up(x_40)], 1))

        x_02 = self.x_02(torch.cat([x_00, x_01, self.up(x_11)], 1))
        x_12 = self.x_12(torch.cat([x_10, x_11, self.up(x_21)], 1))
        x_22 = self.x_22(torch.cat([x_20, x_21, self.up(x_31)], 1))

        x_03 = self.x_03(torch.cat([x_00, x_01, x_02, self.up(x_12)], 1))
        x_13 = self.x_13(torch.cat([x_10, x_11, x_12, self.up(x_22)], 1))

        x_04 = self.x_04(torch.cat([x_00, x_01, x_02, x_03, self.up(x_13)], 1))

        if self.deep_supervision:
            return [
                F.interpolate(self.out1(x_01), size=x.shape[2:], mode='bilinear', align_corners=True),
                F.interpolate(self.out2(x_02), size=x.shape[2:], mode='bilinear', align_corners=True),
                F.interpolate(self.out3(x_03), size=x.shape[2:], mode='bilinear', align_corners=True),
                F.interpolate(self.out4(x_04), size=x.shape[2:], mode='bilinear', align_corners=True)
            ]

        output = self.out(x_04)
        return F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=True)