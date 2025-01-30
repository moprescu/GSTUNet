""" UNet variants """

from .utils import *

# UNet with multiple channels
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 16))
        self.down1 = (Down(16, 32))
        self.down2 = (Down(32, 64))
        self.down3 = (Down(64, 128))
        factor = 2 if bilinear else 1
        self.down4 = (Down(128, 256 // factor))
        self.up1 = (Up(256, 128 // factor, bilinear))
        self.up2 = (Up(128, 64 // factor, bilinear))
        self.up3 = (Up(64, 32 // factor, bilinear))
        self.up4 = (Up(32, 16, bilinear))
        self.outc = (OutConv(16, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class BigUNet(nn.Module):
    """BigUNet: Like UNetConvLSTM but without ConvLSTM or static inputs."""

    def __init__(self, n_channels, n_classes, bilinear=False):
        """
        Args:
            in_channel (int): Number of input channels.
            n_classes (int): Number of output classes.
            bilinear (bool): Whether to use bilinear upsampling.
        """
        super(BigUNet, self).__init__()
        
        self.inc = DoubleConv(n_channels, 64)  # First DoubleConv block

        # Downsampling path
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Upsampling path
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Output convolution
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor with shape (batch_size, in_channel, H, W).
        """
        # Encoder path
        x1 = self.inc(x)     # (batch_size, 64, H, W)
        x2 = self.down1(x1)  # (batch_size, 128, H/2, W/2)
        x3 = self.down2(x2)  # (batch_size, 256, H/4, W/4)
        x4 = self.down3(x3)  # (batch_size, 512, H/8, W/8)
        x5 = self.down4(x4)  # (batch_size, 1024, H/16, W/16)

        # Decoder path
        x = self.up1(x5, x4)  # (batch_size, 512, H/8, W/8)
        x = self.up2(x, x3)   # (batch_size, 256, H/4, W/4)
        x = self.up3(x, x2)   # (batch_size, 128, H/2, W/2)
        x = self.up4(x, x1)   # (batch_size, 64, H, W)

        logits = self.outc(x)  # Final output: (batch_size, n_classes, H, W)
        return logits


# UNet with ConvLSTM layer
class BigUNetConvLSTM(nn.Module):
    def __init__(self, in_channel, n_classes, dim_static=0, bilinear=False):
        super(BigUNetConvLSTM, self).__init__()
        self.inc = ConvLSTM(input_dim=in_channel, hidden_dim=64, kernel_size=3, num_layers=1)
        self.inc2 = DoubleConv(64 + dim_static, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)


    def forward(self, x, s=None):
        _, (x0, _) = self.inc(x)
        if s is not None:
            x0 = torch.cat([x0, s], dim=1)
        x0 = self.inc2(x0)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        logits = self.outc(x)
        return logits # b x h_size x H x W
    

# UNet with ConvLSTM layer
class UNetConvLSTM(nn.Module):
    def __init__(self, in_channel, n_classes, dim_static=0, bilinear=False, attention=False):
        super(UNetConvLSTM, self).__init__()
        self.inc = ConvLSTM(input_dim=in_channel, hidden_dim=32, kernel_size=3, num_layers=1)
        self.inc2 = DoubleConv(32 + dim_static, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)
        if attention:
            self.up1 = UpAtt(256, 128 // factor, bilinear)
            self.up2 = UpAtt(128, 64 // factor, bilinear)
            self.up3 = UpAtt(64, 32 // factor, bilinear)
            self.up4 = UpAtt(32, 16, bilinear)
        else:
            self.up1 = Up(256, 128 // factor, bilinear)
            self.up2 = Up(128, 64 // factor, bilinear)
            self.up3 = Up(64, 32 // factor, bilinear)
            self.up4 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)

        # Initialize weights
        #self._initialize_weights()


    def forward(self, x, s=None):
        _, (x0, _) = self.inc(x)
        if s is not None:
            x0 = torch.cat([x0, s], dim=1)
        x0 = self.inc2(x0)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        logits = self.outc(x)
        return logits # b x h_size x H x W
    

    def _initialize_weights(self):
        """Apply Kaiming initialization to Conv2d and Xavier to Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)