import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Positional Encoding of the input coordinates.

    encodes x to [x, ..., sin(2^k x), cos(2^k x), ...]
    k takes "num_freqs" number of values equally spaced between [0, max_freq]
    """
    def __init__(self, max_freq, num_freqs):
        """
        Args:
            max_freq (float): maximum frequency in the positional encoding
            num_freqs (int): number of frequencies between [0, max_freq]
        """
        super().__init__()
        freqs = 2**torch.linspace(0, max_freq, num_freqs)
        self.register_buffer("freqs", freqs) # [num_freqs]

    def forward(self, x):
        """
        Inputs:
            x [batch_size, num_samples, in_dim]
        Outputs:
            out [batch_size, num_samples, in_dim + 2*num_freqs*in_dim]
        """
        x_proj = x[..., None, :] * self.freqs[..., None] # [batch_size, num_samples, num_freqs, in_dim]
        x_proj = x_proj.reshape(*x.shape[:-1], -1) # [batch_size, num_samples, num_freqs*in_dim]
        out = torch.cat([x, torch.sin(x_proj), torch.cos(x_proj)], dim=-1) # [batch_size, num_samples, in_dim + 2*num_freqs*in_dim]

        return out


class GeometryNet(nn.Module):
    """
    MLP for geometry prediction
    """
    def __init__(self,
                in_dim,
                hidden_dim,
                num_hidden,
                out_dim,
                max_freq,
                num_freqs,
                radius):
        """
        Args:
            in_dim (int): number of dimensions in the input
            hidden_dim (int): number of dimensions in the hidden layer
            num_hidden (int): number of hidden layers in the network
            out_dim (int): number of dimensions in the output
            max_freq (float): maximum frequency in the positional encoding
            num_freqs (int): number of frequencies between [0, max_freq]
            radius (float): radius of the initial sphere for geometric init
        """
        super().__init__()
        
        self.pos_enc = PositionalEncoding(max_freq, num_freqs)
        self.net = []

        # input layer
        in_layer = nn.Linear(in_dim + 2*num_freqs*in_dim, hidden_dim)
        torch.nn.init.constant_(in_layer.bias, 0.0)
        torch.nn.init.constant_(in_layer.weight[:, in_dim:], 0.0)
        torch.nn.init.normal_(in_layer.weight[:, :in_dim], 0.0,
                              math.sqrt(2 / hidden_dim))

        self.net.append(in_layer)
        self.net.append(nn.Softplus(beta=100))

        # hidden layers
        for i in range(num_hidden-1):
            hidden_layer = nn.Linear(hidden_dim, hidden_dim)
            torch.nn.init.constant_(hidden_layer.bias, 0.0)
            torch.nn.init.normal_(hidden_layer.weight, 0.0,
                                  math.sqrt(2 / hidden_dim))
            
            self.net.append(hidden_layer)
            self.net.append(nn.Softplus(beta=100))
        
        # output layer
        out_layer = nn.Linear(hidden_dim, out_dim)
        torch.nn.init.normal_(out_layer.weight,
                              mean=math.sqrt(math.pi / out_dim),
                              std=0.0001)
        torch.nn.init.constant_(out_layer.bias, -radius)

        self.net.append(out_layer)
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        """
        Inputs:
            x [batch_size, num_samples, in_dim]: input points
        Outputs:
            sdf [batch_size, num_samples]: SDF value at the input points
            geometric_feature [batch_size, num_samples, out_dim-1]: geometric feature
                                                                    at the input points
        """
        x.requires_grad_(True)
        x_encoded = self.pos_enc(x)
        output = self.net(x_encoded)

        sdf = output[..., 0]
        geometric_feature = output[..., 1:]
        
        return sdf, geometric_feature
    
    def gradient(self, sdf, x):
        """
        Derivative of the SDF
        Inputs:
            x [batch_size, num_samples, in_dim]: input points
            sdf [batch_size, num_samples]: SDF value at the input points
        Outputs:
            sdf_grad [batch_size, num_samples, in_dim]: gradient of the SDF at the input points
        """
        sdf_grad = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True)[0]

        return sdf_grad


class AppearanceNet(nn.Module):
    """
    MLP for appearance prediction
    """
    def __init__(self,
                in_dim,
                feature_dim,
                hidden_dim,
                num_hidden,
                out_dim,
                max_freq,
                num_freqs):
        """
        Args:
            in_dim (int): number of dimensions in the input
            feature_dim (int): number of dimensions in the geometric feature
            hidden_dim (int): number of dimensions in the hidden layer
            num_hidden (int): number of hidden layers in the network
            out_dim (int): number of dimensions in the output
            max_freq (float): maximum frequency for view direction encoding
            num_freqs (int): number of frequencies between [0, max_freq]
        """
        super().__init__()

        self.dir_enc = PositionalEncoding(max_freq, num_freqs)
        self.net = []

        # input layer
        in_layer = nn.Linear(in_dim + 6*num_freqs + feature_dim, hidden_dim)
        self.net.append(in_layer)
        self.net.append(nn.ReLU())

        # hidden_layers
        for i in range(num_hidden-1):
            hidden_layer = nn.Linear(hidden_dim, hidden_dim)
            self.net.append(hidden_layer)
            self.net.append(nn.ReLU())
        
        # output layer
        out_layer = nn.Linear(hidden_dim, out_dim)
        self.net.append(out_layer)
        self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

    def forward(self, point, direction, normal, feature):
        """
        Inputs:
            point [batch_size, num_samples, 3]: input points
            direction [batch_size, num_samples, 3]: viewing direction of the points
            normal [batch_size, num_samples, 3]: normal direction of the points
            feature [batch_size, num_samples, feature_dim]: features from the geometry network
        Outputs:
            rgb [batch_size, num_samples, out_dim]: RGB color at the input points
        """
        dir_encoded = self.dir_enc(direction)
        x = torch.cat([point, dir_encoded, normal, feature], dim=-1)
        rgb = self.net(x)

        return rgb


class SDensity(nn.Module):
    """
    Learning the "scale" parameter of the S-density

    "inv_s" (inverse of the "scale" parameter) is defined as an exponential
    function of a learnable variable, to make sure "scale" is always positive
    and has good gradient properites

    https://github.com/Totoro97/NeuS/issues/12
    """
    def __init__(self, init_val):
        """
        Args:
            init_val (float): initial value of the learnable variable
        """
        super().__init__()

        init_val = torch.as_tensor(init_val)
        self.variable = nn.Parameter(init_val)

    def forward(self):
        """
        Outputs:
            inv_s (float): inverse of the "scale" parameter
        """
        inv_s = torch.exp(10 * self.variable)
        return inv_s
