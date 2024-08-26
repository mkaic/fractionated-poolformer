import torch
import torch.nn as nn

from .layers import (
    get_rotary_position_encoding,
    apply_rotary_encoding,
    FractionatedPoolFormerBlock,
)


class FractionatedPoolFormer(nn.Module):
    def __init__(
        self,
        num_classes,
        blocks,
        channels,
        levels,
        input_channels=3,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_layers = blocks
        self.channels = channels
        self.levels = levels

        self.proj_in = nn.Linear(input_channels, channels, bias=False)

        self.layers = nn.ModuleList()

        for i in range(self.num_layers):
            self.layers.append(
                FractionatedPoolFormerBlock(channels=channels, levels=levels)
            )

        self.out_proj = nn.Linear(channels, num_classes, bias=True)

        self.pos_enc = None

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        b, c, h, w = x.shape

        x = torch.movedim(x, 1, -1)  # B, C, H, W -> B, H, W, C
        x = self.proj_in(x)  # increase channel count

        if self.pos_enc is None:
            self.pos_enc = get_rotary_position_encoding(
                shape=(h, w),
                num_frequencies=self.channels // 4,
                device=x.device,
            )

        x = apply_rotary_encoding(x, self.pos_enc)

        for layer in self.layers:
            x = layer(x)

        # Average all pixels and make final prediction
        x = x.mean(dim=(1, 2))
        x = self.out_proj(x)

        return x
