import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *
from .unet import UNet

class LFM(nn.Module):
    """
    PyTorch version of LFM block, generalized to (B, C, H, W) with user-specified H, W.
    
    Steps:
      1) ConvLSTM2D -> final hidden state
      2) Conv2d(16) -> Dropout -> Conv2d(16, kernel=1)
      3) BatchNorm -> Flatten
      4) Dense(64) -> Dense(H*W)
      5) Reshape to (B, 1, H, W)
    """
    def __init__(
        self,
        in_channels=1,
        hidden_lstm=8,
        kernel_size_lstm=5,
        out_channels_conv=16,
        height=64,
        width=64,
        dropout=0.2
    ):
        """
        Args:
            in_channels: number of input channels
            hidden_lstm: hidden dim for ConvLSTM
            kernel_size_lstm: kernel size for ConvLSTM
            out_channels_conv: channels in the 2D conv layers (16 in your example)
            height, width: final H, W for the reshape
            dropout: dropout rate
        """
        super().__init__()
        self.height = height
        self.width = width
        
        # 1) ConvLSTM
        self.convlstm = ConvLSTM(
            input_dim=in_channels,
            hidden_dim=hidden_lstm,
            kernel_size=kernel_size_lstm,
            num_layers=1
        )
        
        # 2) Convolution chain
        self.conv1 = nn.Conv2d(hidden_lstm, out_channels_conv, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels_conv, out_channels_conv, kernel_size=1, padding=0)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm2d(out_channels_conv)
        
        # 3) Flatten -> Dense(64) -> Dense(H*W)
        #    At flatten, shape = [B, out_channels_conv * H * W]
        #    Then fc1 -> [B, 64], fc2 -> [B, H*W].
        self.flat_dim = out_channels_conv * self.height * self.width
        self.fc1 = nn.Linear(self.flat_dim, 64)
        self.fc2 = nn.Linear(64, self.height * self.width)
        
    def forward(self, x):
        """
        x: [B, C, H, W] or [T, B, C, H, W]. We only use final hidden state in either case.
        Returns:
            lfm_out: [B, 1, H, W]
        """
        # 1) ConvLSTM => final hidden state (h) only
        _, (h, _) = self.convlstm(x)   # [B, hidden_lstm, H, W]
        
        # 2) CNN pipeline
        h = F.relu(self.conv1(h))  # -> [B, out_channels_conv, H, W]
        h = self.dropout(h)
        h = F.relu(self.conv2(h))  # -> [B, out_channels_conv, H, W]
        h = self.bn(h)
        
        # 3) Flatten -> FC(64) -> FC(H*W) -> reshape
        B, C, Ht, Wt = h.shape
        # Safety check: if (Ht, Wt) != (self.height, self.width), we might adapt or up/downsample.
        # For simplicity, assume input was indeed (height, width).
        
        h = h.view(B, -1)          # [B, C*Ht*Wt]
        h = F.relu(self.fc1(h))    # [B, 64]
        h = self.fc2(h)            # [B, height*width]
        
        # Reshape to [B, 1, height, width]
        lfm_out = h.view(B, 1, self.height, self.width)
        return lfm_out

class STCINet(nn.Module):
    def __init__(self, 
                 in_channel, 
                 n_classes, 
                 dim_static=0, 
                 bilinear=False,
                 # LFM params:
                 lfm_hidden=8,
                 lfm_outconv=16,
                 height=64,
                 width=64
                ):
        """
        Args:
            in_channel:  channels for the main UNet input
            n_classes:   # of classes in final segmentation (or output channels)
            dim_static:  extra static channels appended in inc2 (if needed)
            bilinear:    whether U-Net upsampling is bilinear or transposed conv
            lfm_*:       parameters passed to LFM
            height,width: for LFM reshape
        """
        super().__init__()
        
        # ---- (A) The existing UNet + ConvLSTM + Attention pieces ----
        #     Reuse the attention-based UNet blocks from your previous code.

        self.inc = ConvLSTM(input_dim=in_channel, hidden_dim=32, kernel_size=3, num_layers=1)
        self.lfm = LFM(
            in_channels=in_channel,
            hidden_lstm=lfm_hidden,
            kernel_size_lstm=5,
            out_channels_conv=lfm_outconv,
            height=height,
            width=width,
            dropout=0.2
        )

        self.inc2 = DoubleConv(32 + dim_static + 1, 16)

        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)

        self.up1 = UpAtt(256, 128 // factor, bilinear)
        self.up2 = UpAtt(128, 64 // factor, bilinear)
        self.up3 = UpAtt(64, 32 // factor, bilinear)
        self.up4 = UpAtt(32, 16, bilinear)

        self.outc = OutConv(16, n_classes)


    def forward(self, x, s=None):
        """
        x should be [T, B, in_channel, H, W] if the inc(ConvLSTM) expects a sequence
        or [B, in_channel, H, W] if inc is single-step. 
        Below, we assume T=1 for simplicity or you adapt your code as needed.

        s: optional static features [B, dim_static, H, W]
        """
        # 1) pass input through the main inc(ConvLSTM)
        #    Suppose x is [T, B, C, H, W]. We want the final hidden state (x0).
        _, (x0, _) = self.inc(x)  # -> x0 shape [B, 32, H, W]
        lfm_out = self.lfm(x)           # -> [B, 1, H, W]

        # 3) Concatenate x0 (B,32,H,W), lfm_out (B,1,H,W), and static s (B,dim_static,H,W) if present
        if s is not None:
            x0 = torch.cat([x0, s, lfm_out], dim=1)  # -> [B, 32+dim_static+1, H, W]
        else:
            x0 = torch.cat([x0, lfm_out], dim=1)     # -> [B, 32+1, H, W]

        # 4) inc2
        x0 = self.inc2(x0)   # -> [B, 16, H, W]
        
        # 5) Down path
        x1 = self.down1(x0)  # -> [B, 32, H/2, W/2]
        x2 = self.down2(x1)  # -> [B, 64, H/4, W/4]
        x3 = self.down3(x2)  # -> [B, 128, H/8, W/8]
        x4 = self.down4(x3)  # -> [B, 256//factor, H/16, W/16]
        
        # 6) Up path with attention
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        
        # 7) Output
        logits = self.outc(x)  # [B, n_classes, H, W]
        return logits