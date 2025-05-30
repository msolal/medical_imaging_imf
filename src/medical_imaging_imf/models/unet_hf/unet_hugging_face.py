from functools import partial

import torch
import torch.nn as nn

from medical_imaging_imf.models.unet_hf.layers import *

from typing import Optional


class Unet(nn.Module):
    def __init__(
        self,
        dim : int,
        init_dim : Optional[int] = None,
        out_dim : Optional[int] = None,
        dim_mults : tuple[int, int, int, int] = (1, 2, 4, 8),
        channels : int = 3,
        self_condition : bool = False,
        resnet_block_groups : int = 4,
        n_block_klass : int = 2
    ):
        super().__init__()
        # Determine the number of Resnet blocks
        self.n_block_klass = n_block_klass
        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0) # changed to 1 and 0 from 7,3

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:], strict=False))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        *[block_klass(dim_in, dim_in, time_emb_dim=time_dim) for i in range(self.n_block_klass)],
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        *[block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim) for i in range(self.n_block_klass)],
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, time, x, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for i in range(len(self.downs)):
            if self.n_block_klass>1:
                for block in self.downs[i][:self.n_block_klass-1]:
                    x = block(x, t)
                    h.append(x)

            for block, attn, downsample in [self.downs[i][self.n_block_klass-1:]]:

                x = block(x, t)
                x = attn(x)
                h.append(x)

                x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for i in range(len(self.ups)):
            if self.n_block_klass>1:
                for block in self.ups[i][:self.n_block_klass-1]:
                    x = torch.cat((x, h.pop()), dim=1)
                    x = block(x, t)

            for block, attn, upsample in [self.ups[i][self.n_block_klass-1:]]:
                x = torch.cat((x, h.pop()), dim=1)
                x = block(x, t)
                x = attn(x)

                x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
