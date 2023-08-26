import torch
from torch import nn
import math
import tinycudann as tcnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AnnealedHash(nn.Module):
    def __init__(
        self, in_channels, annealed_step, annealed_begin_step=0, identity=True
    ):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(AnnealedHash, self).__init__()
        self.N_freqs = 16
        self.in_channels = in_channels
        self.annealed = True
        self.annealed_step = annealed_step
        self.annealed_begin_step = annealed_begin_step

        self.index = torch.linspace(0, self.N_freqs - 1, self.N_freqs)
        self.identity = identity

        self.index_2 = self.index.view(-1, 1).repeat(1, 2).view(-1)

    def forward(self, x_embed, step):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """

        if self.annealed_begin_step == 0:
            # calculate the w for each freq bands
            alpha = self.N_freqs * step / float(self.annealed_step)
        else:
            if step <= self.annealed_begin_step:
                alpha = 0
            else:
                alpha = (
                    self.N_freqs
                    * (step - self.annealed_begin_step)
                    / float(self.annealed_step)
                )

        w = (
            1
            - torch.cos(
                math.pi
                * torch.clamp(
                    alpha * torch.ones_like(self.index_2) - self.index_2, 0, 1
                )
            )
        ) / 2

        out = x_embed * w.to(x_embed.device)

        return out


class ImplicitVideo_Hash(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = tcnn.Encoding(n_input_dims=2, encoding_config=config["encoding"])
        self.decoder = tcnn.Network(
            n_input_dims=self.encoder.n_output_dims + 2,
            n_output_dims=3,
            network_config=config["network"],
        )

    def forward(self, x):
        input = x
        input = self.encoder(input)
        input = torch.cat([x, input], dim=-1)
        weight = torch.ones(input.shape[-1], device=input.device).cuda()
        x = self.decoder(weight * input)
        return x


class Deform_Hash3d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = tcnn.Encoding(
            n_input_dims=3, encoding_config=config["encoding_deform3d"]
        )
        self.decoder = tcnn.Network(
            n_input_dims=self.encoder.n_output_dims + 3,
            n_output_dims=2,
            network_config=config["network_deform"],
        )

    def forward(self, x, step=0, aneal_func=None):
        input = x
        input = self.encoder(input)
        if aneal_func is not None:
            input = torch.cat([x, aneal_func(input, step)], dim=-1)
        else:
            input = torch.cat([x, input], dim=-1)

        weight = torch.ones(input.shape[-1], device=input.device).cuda()
        x = self.decoder(weight * input) / 5

        return x


class Deform_Hash3d_Warp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Deform_Hash3d = Deform_Hash3d(config)

    def forward(self, xyt_norm, step=0, aneal_func=None):
        x = self.Deform_Hash3d(xyt_norm, step=step, aneal_func=aneal_func)
        return x


def positionalEncoding_vec(in_tensor, b):
    proj = torch.einsum(
        "ij, k -> ijk", in_tensor, b
    )  # shape (batch, in_tensor.size(1), freqNum)
    mapped_coords = torch.cat(
        (torch.sin(proj), torch.cos(proj)), dim=1
    )  # shape (batch, 2*in_tensor.size(1), freqNum)
    output = mapped_coords.transpose(2, 1).contiguous().view(mapped_coords.size(0), -1)
    return output


class IMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=256,
        use_positional=True,
        positional_dim=10,
        skip_layers=[4, 6],
        num_layers=8,  # includes the output layer
        use_tanh=True,
        apply_softmax=False,
    ):
        super(IMLP, self).__init__()
        self.use_tanh = use_tanh
        self.apply_softmax = apply_softmax
        if apply_softmax:
            self.softmax = nn.Softmax()
        if use_positional:
            encoding_dimensions = 2 * input_dim * positional_dim
            self.b = torch.tensor(
                [(2**j) * np.pi for j in range(positional_dim)], requires_grad=False
            )
        else:
            encoding_dimensions = input_dim

        self.hidden = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                input_dims = encoding_dimensions
            elif i in skip_layers:
                input_dims = hidden_dim + encoding_dimensions
            else:
                input_dims = hidden_dim

            if i == num_layers - 1:
                # last layer
                self.hidden.append(nn.Linear(input_dims, output_dim, bias=True))
            else:
                self.hidden.append(nn.Linear(input_dims, hidden_dim, bias=True))

        self.skip_layers = skip_layers
        self.num_layers = num_layers

        self.positional_dim = positional_dim
        self.use_positional = use_positional

    def forward(self, x):
        if self.use_positional:
            pos = positionalEncoding_vec(x, self.b.to(x.device))
            x = pos

        input = x.detach().clone()
        for i, layer in enumerate(self.hidden):
            if i > 0:
                x = F.relu(x)
            if i in self.skip_layers:
                x = torch.cat((x, input), 1)
            x = layer(x)
        if self.use_tanh:
            x = torch.tanh(x)

        if self.apply_softmax:
            x = self.softmax(x)
        return x

    def _initialize_weights(self):
        for layer in self.hidden:
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.uniform_(layer.bias, -0.1, 0.1)
