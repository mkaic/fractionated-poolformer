import torch
import torch.nn as nn
import torch.nn.functional as F


def get_rotary_position_encoding(shape, num_frequencies, device):

    positions = torch.stack(
        torch.meshgrid(
            *[torch.arange(i, dtype=torch.float32, device=device) for i in shape],
            indexing="ij"
        ),
        dim=-1,
    )

    freq_bands = []

    for freq_idx in range(1, num_frequencies + 1):
        for pe_axis in range(len(shape)):
            pos = positions[..., pe_axis] * (
                1 / (10000 ** (freq_idx / num_frequencies))
            )
            cos = torch.cos(pos)
            sin = torch.sin(pos)
            freq_bands.append(torch.complex(cos, sin))

    positions = torch.stack(freq_bands, dim=-1)  # C, H, W

    return positions


def apply_rotary_encoding(x: torch.Tensor, pos_enc: torch.Tensor):
    b, h, w, c = x.shape
    x = x.view(b, h, w, c // 2, 2)
    x = torch.view_as_complex(x)
    x = x * pos_enc
    x = torch.view_as_real(x)
    x = x.view(b, h, w, c)
    return x


class FractionatedAvgPool2d(nn.Module):
    def __init__(self, levels=5):
        super().__init__()
        self.levels = levels
        self.padding_layers = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()
        for i in range(1, levels):
            kernel_size = 2**i
            # "same" padding
            padding_big = kernel_size // 2
            padding_small = padding_big - 1
            self.padding_layers.append(
                nn.CircularPad2d(
                    (padding_big, padding_small, padding_big, padding_small)
                )
            )
            self.pooling_layers.append(
                nn.AvgPool2d(kernel_size=kernel_size, stride=1)
            )  # 2, 4, 8, 16 etc

    def forward(self, x):
        x = torch.movedim(x, -1, 1)
        channel_chunks = torch.chunk(x, self.levels, dim=1)
        pooled = [channel_chunks[0]]
        for chunk, pad, pool in zip(
            channel_chunks[1:], self.padding_layers, self.pooling_layers
        ):
            chunk = pad(chunk)
            pooled.append(pool(chunk))

        x = torch.cat(pooled, dim=1)
        x = torch.movedim(x, 1, -1)
        return x


class FractionatedPoolFormerBlock(nn.Module):
    def __init__(self, channels, levels=5):
        super().__init__()
        self.channels = channels
        self.levels = levels

        self.norm_a = nn.LayerNorm(channels)
        self.pool = FractionatedAvgPool2d(levels)
        self.norm_b = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        residual = x

        x = self.norm_a(x)
        x = self.pool(x)

        x = x + residual
        residual = x

        x = self.norm_b(x)
        x = x + self.mlp(x)

        x = x + residual
        return x
