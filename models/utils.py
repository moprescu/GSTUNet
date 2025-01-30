""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

##############
# UNet Utils #
##############
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, p=0.1):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #nn.Dropout(p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

######################
# UNetConvLSTM Utils #
######################
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)

    def forward(self, x):
        batch_size, seq_len, _, height, width = x.size()
        h, c = self.cell.init_hidden(batch_size, (height, width))

        outputs = []
        for t in range(seq_len):
            h, c = self.cell(x[:, t, :, :, :], (h, c))
            outputs.append(h)
        return torch.stack(outputs, dim=1), (h, c)
    
# Attention Gate
class AttentionBlock(nn.Module):
    """
    Attention block as proposed in 'Attention U-Net: Learning Where to Look for the Pancreas'
    (https://arxiv.org/pdf/1804.03999).
    This block takes:
      - gating signal g (decoder feature)
      - skip connection x (encoder feature)
    to produce a spatial attention map, which is then multiplied by the skip connection.
    """
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: Number of channels in the gating signal (decoder)
            F_l: Number of channels in the skip connection (encoder)
            F_int: Number of intermediate channels (usually smaller than F_g and F_l)
        """
        super(AttentionBlock, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g: gating signal (decoder feature)
        x: skip connection (encoder feature)
        Returns:
            scaled skip-connection: x * att_weight
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Combine gating & skip signals
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)  # attention coefficients in [0,1]

        # Multiply attention coefficients by skip connection
        out = x * psi
        return out
    
class UpAtt(nn.Module):
    """
    Up-sampling + Attention + DoubleConv
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpAtt, self).__init__()
        
        # If bilinear, reduce channels with a 1x1 convolution after upsampling
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv_reduce = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv_reduce = None
        
        # Attention block
        #   F_g = in_channels//2 (decoder after up-sampling)
        #   F_l = out_channels (skip connection channels from encoder)
        #   F_int = out_channels // 2 (arbitrary choice, can be changed)
        self.attention = AttentionBlock(F_g=in_channels // 2, 
                                        F_l=out_channels, 
                                        F_int=out_channels // 2)
        
        # Final convolution after concatenation
        #   We'll end up with (decoder + skip) = out_channels + in_channels//2
        #   That total feeds into DoubleConv(...).
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: decoder feature (from the previous up block or bottom of the UNet)
        x2: encoder feature (skip connection)
        """
        # Upsample
        x1 = self.up(x1)
        if self.conv_reduce is not None:
            x1 = self.conv_reduce(x1)

        # Attention gating on the skip connection
        x2 = self.attention(x1, x2)

        # Concatenate
        x = torch.cat([x2, x1], dim=1)

        # Convolve
        return self.conv(x)

#######################
# GSTUNet Utils #
#######################
class GHead(nn.Module):
    """Used by GUNet"""

    def __init__(self, hr_size, fc_layer_sizes, dim_treatments, dim_outcome):
        super().__init__()

        self.hr_size = hr_size
        self.fc_layer_sizes = fc_layer_sizes
        self.dim_treatments = dim_treatments
        self.dim_outcome = dim_outcome

        layers = []

        in_features = self.hr_size + self.dim_treatments
        for size in self.fc_layer_sizes:
            layers.append(nn.Linear(in_features, size))
            layers.append(nn.ELU())  # Activation function
            in_features = size
        
        # Add the final layer to produce the outcome
        layers.append(nn.Linear(in_features, self.dim_outcome))
        
        # Store as a sequential model
        self.network = nn.Sequential(*layers)

        # Initialize weights
        #self._initialize_weights()

    def forward(self, hr, current_treatment):
        x = torch.cat((hr, current_treatment), dim=-1)
        return self.network(x)
    
    def _initialize_weights(self):
        """Apply Kaiming initialization to linear layers."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                #nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    