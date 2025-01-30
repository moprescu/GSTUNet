""" UNet variants """

from .utils import *
from .unet import *
import math

# UNet with ConvLSTM layer and one G-head
# To be used only for validation purposes
class GSTUNet_SingleHead(nn.Module):
    def __init__(self, in_channel, h_size, fc_hidden_units=64, dim_treatments = 1, dim_outcome = 1):
        super().__init__()
        self.unet = UNetConvLSTM(in_channel = in_channel, n_classes = h_size)
        self.head = GHead(hr_size = h_size, fc_hidden_units = fc_hidden_units, 
                               dim_treatments = dim_treatments, dim_outcome = dim_outcome)
    
    def forward(self, x, A):
        """
        inputs
        ------
        x: b x seq_len x in_channel x H x W
        A: b x 1 x H x W
        
        outputs
        -------
        y: b x H*W
        """
        
        out = self.unet(x) # b x h_size x H x W
        b, h, _, _ = out.shape
        hidden = self.unet(x).permute(0, 2, 3, 1).reshape(b, -1, h) # b x H*W x h_size
        A = A.permute(0, 2, 3, 1).reshape(b, -1, 1) # b x H*W x 1
        y = self.head(hidden, A).squeeze(-1) # b x H*W
        return y


# UNet with ConvLSTM layer and multiple G-Heads
class GSTUNet(nn.Module):
    def __init__(self, in_channel, h_size, A_counter, fc_layer_sizes=[64], dim_treatments = 1, 
                 dim_outcome = 1, dim_horizon = 5, dim_static = 0, use_constant_feature = False,
                 attention=False):
        """
        inputs
        ------
        A_counter: dim_horizon x dim_treatments x H x W
        """
        super().__init__()
        self.unet = UNetConvLSTM(in_channel = in_channel, n_classes = h_size, 
                                 dim_static = (1 if use_constant_feature else 0) + dim_static,
                                 attention=attention)
        self.heads = nn.ModuleList([GHead(hr_size = h_size, 
                                 fc_layer_sizes = fc_layer_sizes, 
                                 dim_treatments = dim_treatments, 
                                 dim_outcome = dim_outcome) for _ in range(dim_horizon)])
        self.A_counter = A_counter
        self.dim_horizon = dim_horizon
        self.dim_treatments = dim_treatments
        self.dim_outcome = dim_outcome
        self.dim_static = dim_static
        self.use_constant_feature = use_constant_feature
        assert A_counter.shape[0] == dim_horizon
    
    def forward_unet(self, x, s=None):
        """
        inputs
        ------
        x: b x seq_len x in_channel x H x W
        s: b x dim_static x H x W or None - Time-invariant features
        
        outputs
        -------
        hidden: b x H*W x h_size
        """
        b, _, _, H, W = x.shape
        # Handle `s` based on whether `use_constant_feature` is True or False
        if s is None:
            s = torch.ones(b, 1, H, W).to(x.device) if self.use_constant_feature else None
        elif self.use_constant_feature:
            constant_feature = torch.ones(b, 1, H, W).to(x.device)
            s = torch.concat([s, constant_feature], dim=1)
        out = self.unet(x, s)  # Single call
        b, h, _, _ = out.shape
        hidden = out.permute(0, 2, 3, 1).reshape(b, -1, h)  # b x H*W x h_size
        return hidden

    def forward_head(self, hidden, A, head_idx = -1):
        """
        inputs
        ------
        hidden: b x H*W x h_size
        A: b x dim_treatments x H x W - Can either be counterfactual or factual
        head_idx: Index of the head to use
        
        outputs
        -------
        hidden: b x H*W
        """
        b, HW, _ = hidden.shape
        A_copy = A.permute(0, 2, 3, 1).reshape(b, HW, self.dim_treatments) # b x H*W x 1
        y = self.heads[head_idx](hidden, A_copy).squeeze(-1) # b x H*W x 1
        return y
    
    @torch.no_grad()
    def forward_nograd(self, x, head_idx, s=None):
        """
        inputs
        ------
        x: b x seq_len x in_channel x H x W
        head_idx: Index of the head to use
        s: b x dim_static x H x W or None - Time-invariant features

        outputs
        -------
        out: b x H*W
        """

        # Change the -2 channel of x up to dim_horizon with self.A_counter
        # and fill in the last element with zeros
        b, seq_len, in_channel, H, W = x.shape
        assert seq_len > head_idx 

        # Modify the treatment (-2) channel for each batch
        x_copy = x.clone()
        x_copy[:, seq_len-head_idx-1:, -1-self.dim_treatments:-1, :, :] = self.A_counter[:head_idx+1].unsqueeze(0).repeat(b, 1, 1, 1, 1)  # Apply self.A_counter
        #x_copy[:, -1, -2:-1, :, :] = 0  # Pad the last treatment with zeros

        # Create "A" by replicating the last element of self.A_counter for each batch
        last_A = self.A_counter[head_idx].unsqueeze(0)  # Shape 1 x dim_treatments x H x W
        A = last_A.repeat(b, 1, 1, 1)  # Replicate for batch size => b x dim_treatments x H x W

        # Call forward_unet and forward_head with modified x and A
        hidden = self.forward_unet(x_copy, s)
        out = self.forward_head(hidden, A, head_idx)
        return out
    
    def forward_grad(self, x, A, head_idx, s=None):
        """
        inputs
        ------
        x: b x seq_len x in_channel x H x W
        head_idx: Index of the head to use
        s: b x dim_static x H x W or None - Time-invariant features
        
        outputs
        -------
        out: b x H*W
        """
        hidden = self.forward_unet(x, s)
        out = self.forward_head(hidden, A, head_idx)
        return out
    
    def forward(self, x, s=None):
        """
        inputs
        ------
        x: b x seq_len x in_channel x H x W
        s: b x dim_static x H x W or None - Time-invariant features
        
        outputs
        -------
        out: b x H*W
        """
        
        hidden = self.forward_unet(x, s)
        b, seq_len, in_channel, H, W = x.shape
        last_A = self.A_counter[0].unsqueeze(0)  # Shape 1 x dim_treatments x H x W
        A = last_A.repeat(b, 1, 1, 1)  # Replicate for batch size
        out = self.forward_head(hidden, A, 0)
        return out

class GSTUNetList(nn.Module):
    def __init__(self, in_channel, h_size, A_counter, fc_layer_sizes=[64],
                 dim_treatments=1, dim_outcome=1, dim_horizon=5, attention=False):
        """
        inputs
        ------
        A_counter: shape (dim_horizon, 1, H, W)
        """
        super().__init__()
        
        # Instead of a single UNet, create one UNet per head
        self.unets = nn.ModuleList([
            UNetConvLSTM(in_channel=in_channel, n_classes=h_size, attention=attention)
            for _ in range(dim_horizon)
        ])
        
        # Keep a separate GHead for each horizon
        self.heads = nn.ModuleList([
            GHead(hr_size=h_size,
                  fc_layer_sizes=fc_layer_sizes,
                  dim_treatments=dim_treatments,
                  dim_outcome=dim_outcome)
            for _ in range(dim_horizon)
        ])
        
        self.A_counter = A_counter  # (dim_horizon, 1, H, W)
        self.dim_horizon = dim_horizon
        self.dim_treatments = dim_treatments
        self.dim_outcome = dim_outcome
        
        assert A_counter.shape[0] == dim_horizon, "A_counter must match dim_horizon"
    
    def forward_unet(self, x, head_idx):
        """
        inputs
        ------
        x: shape (b, seq_len, in_channel, H, W)
        
        outputs
        -------
        hidden: shape (b, H*W, h_size)
        """
        # Run the forward pass through the unet for this specific head_idx
        out = self.unets[head_idx](x)  # out: (b, h_size, H, W) or similar
        b, h, _, _ = out.shape
        # Flatten spatial dimension: (b, H*W, h_size)
        hidden = out.permute(0, 2, 3, 1).reshape(b, -1, h)
        return hidden

    def forward_head(self, hidden, A, head_idx):
        """
        inputs
        ------
        hidden: (b, H*W, h_size)    from forward_unet
        A:      (b, 1, H, W)        can be counterfactual or factual

        outputs
        -------
        y:      (b, H*W)           outcome predictions
        """
        b = hidden.size(0)
        # Reshape A to match hidden's second dimension:
        # A_copy: (b, H*W, 1)
        A_copy = A.permute(0, 2, 3, 1).reshape(b, -1, 1)
        
        # Pass the hidden + A_copy to the GHead for this horizon
        y = self.heads[head_idx](hidden, A_copy).squeeze(-1)  # (b, H*W, 1) -> (b, H*W)
        return y
    
    @torch.no_grad()
    def forward_nograd(self, x, head_idx):
        """
        Example of a no_grad pass for iterative G-computation at a particular horizon index.
        """
        b, seq_len, in_channel, H, W = x.shape
        assert seq_len > head_idx, "Sequence length must exceed head_idx"
        
        # Clone x so we can modify the 'treatment' channel for the next head_idx steps
        x_copy = x.clone()
        
        # Overwrite the last -2 channel with A_counter up to `head_idx + 1`
        # shape of self.A_counter[:head_idx+1] -> (head_idx+1, 1, H, W)
        x_copy[:, seq_len - head_idx - 1:, -2:-1, :, :] = (
            self.A_counter[:head_idx+1]
                .unsqueeze(0)              # -> (1, head_idx+1, 1, H, W)
                .repeat(b, 1, 1, 1, 1)     # -> (b, head_idx+1, 1, H, W)
        )
        
        # Build "A" for this horizon
        last_A = self.A_counter[head_idx].unsqueeze(0)  # shape: (1, 1, H, W)
        A = last_A.repeat(b, 1, 1, 1)                   # shape: (b, 1, H, W)
        
        # Run the specific unet for head_idx
        hidden = self.forward_unet(x_copy, head_idx)
        # Then pass to the head for that horizon
        out = self.forward_head(hidden, A, head_idx)
        
        return out
    
    def forward_grad(self, x, A, head_idx):
        """
        Same idea as forward_nograd, but allowing gradients.
        """
        hidden = self.forward_unet(x, head_idx)
        out = self.forward_head(hidden, A, head_idx)
        return out
    
    def forward(self, x, head_idx=0):
        """
        A 'default' forward pass if you want to explicitly pick one horizon.
        This function can be customized or replaced by an iterative loop over
        horizon indices if you want to do multi-step predictions in a single call.
        
        inputs
        ------
        x: (b, seq_len, in_channel, H, W)
        head_idx: which horizon to use
        
        outputs
        -------
        out: (b, H*W)
        """
        hidden = self.forward_unet(x, head_idx)
        
        # For illustration, just replicate the first element of A_counter for head_idx
        b, seq_len, in_channel, H, W = x.shape
        A = self.A_counter[head_idx].unsqueeze(0).repeat(b, 1, 1, 1)
        
        out = self.forward_head(hidden, A, head_idx)
        return out


class GSTMultiTaskUNet(nn.Module):
    def __init__(self, in_channel,  A_counter, dim_treatments=1, 
                 dim_outcome=1, dim_horizon=5, bilinear=False):
        """
        MultiTaskUNet with compact encoder and decoders acting as task-specific G-heads.
        
        Args:
        - in_channel: Number of input channels.
        - A_counter: Pre-defined counterfactual treatments (dim_horizon x 1 x H x W).
        - dim_treatments: Number of treatment channels.
        - dim_outcome: Number of output channels per task.
        - dim_horizon: Number of prediction horizons (number of task-specific decoders).
        - bilinear: Use bilinear upsampling in decoders.
        """
        super().__init__()

        factor = 2 if bilinear else 1
        
        # Compact shared encoder
        self.encoder = nn.ModuleList([
            ConvLSTM(input_dim=in_channel, hidden_dim=32, kernel_size=3, num_layers=1),
            DoubleConv(32, 16),
            Down(16, 32),
            Down(32, 64),
            Down(64, 128),
            Down(128, 256 // factor)
        ])

        # Task-specific decoders (acting as G-heads)
        self.heads = nn.ModuleList([
            nn.Sequential(
                Up(256, 128 // factor, bilinear),
                Up(128, 64 // factor, bilinear),
                Up(64, 32 // factor, bilinear),
                Up(32, 16, bilinear),
                OutConv(16, dim_outcome)) for i in range(dim_horizon)
            ])
        
        self.A_counter = A_counter
        self.dim_horizon = dim_horizon
        assert A_counter.shape[0] == dim_horizon, "A_counter must match dim_horizon"
    
    def forward_head(self, x, head_idx = -1):
        """
        inputs
        ------
        hidden: b x H*W x h_size
        A: b x 1 x H x W - Can either be counterfactual or factual
        
        outputs
        -------
        hidden: b x H*W
        """
        _, (x0, _) = self.encoder[0](x)
        x0 = self.encoder[1](x0)
        x1 = self.encoder[2](x0)
        x2 = self.encoder[3](x1)
        x3 = self.encoder[4](x2)
        x4 = self.encoder[5](x3)

        x = self.heads[head_idx][0](x4, x3)
        x = self.heads[head_idx][1](x, x2)
        x = self.heads[head_idx][2](x, x1)
        x = self.heads[head_idx][3](x, x0)
        y = self.heads[head_idx][4](x)
        return y
    
    @torch.no_grad()
    def forward_nograd(self, x, head_idx):
        # Change the -2 channel of x up to dim_horizon with self.A_counter
        # and fill in the last element with zeros
        b, seq_len, in_channel, H, W = x.shape
        assert seq_len > head_idx 

        # Modify the treatment (-2) channel for each batch
        x_copy = x.clone()
        x_copy[:, seq_len-head_idx-1:, -2:-1, :, :] = self.A_counter[:head_idx+1].unsqueeze(0).repeat(b, 1, 1, 1, 1)  # Apply self.A_counter
        #x_copy[:, -1, -2:-1, :, :] = 0  # Pad the last treatment with zeros

        # Call forward_unet and forward_head with modified x and A
        out = self.forward_head(x_copy, head_idx)
        return out
    
    def forward_grad(self, x, head_idx):
        """
        inputs
        ------
        x: b x seq_len x in_channel x H x W
        A: b x 1 x H x W
        
        outputs
        -------
        out: b x H*W
        """
        out = self.forward_head(x, head_idx)
        return out
    
    def forward(self, x):
        """
        inputs
        ------
        x: b x seq_len x in_channel x H x W
        A: b x 1 x H x W
        
        outputs
        -------
        hidden: b x H*W
        """
        out = self.forward_head(x, 0)
        return out

    
