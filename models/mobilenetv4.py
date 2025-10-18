"""MIT License
Copyright (C) 2024 Face Anti-Spoofing Project
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom
the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_tools import *


class UniversalInvertedBottleneck(nn.Module):
    """
    Universal Inverted Bottleneck (UIB) block from MobileNetV4.
    Combines elements from Inverted Bottleneck, ConvNext, FFN, and ExtraDW.
    """

    def __init__(
        self,
        inp,
        hidden_dim,
        oup,
        kernel_size,
        stride,
        use_se,
        use_hs,
        prob_dropout,
        type_dropout,
        sigma,
        mu,
        block_type="uib",
    ):
        super().__init__()
        assert stride in [1, 2]
        self.identity = stride == 1 and inp == oup
        self.block_type = block_type
        self.dropout2d = Dropout(dist=type_dropout, mu=mu, sigma=sigma, p=prob_dropout)

        if block_type == "uib":
            self._build_uib_block(
                inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs
            )
        elif block_type == "extra_dw":
            self._build_extra_dw_block(
                inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs
            )
        elif block_type == "ffn":
            self._build_ffn_block(
                inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs
            )
        else:
            raise ValueError(f"Unknown block type: {block_type}")

    def _build_uib_block(
        self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs
    ):
        """Build Universal Inverted Bottleneck block"""
        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # Depthwise convolution
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # Pointwise convolution
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # Pointwise expansion
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Depthwise convolution
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Pointwise projection
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def _build_extra_dw_block(
        self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs
    ):
        """Build Extra Depthwise block"""
        self.conv = nn.Sequential(
            # First depthwise
            nn.Conv2d(
                inp,
                inp,
                kernel_size,
                stride,
                (kernel_size - 1) // 2,
                groups=inp,
                bias=False,
            ),
            nn.BatchNorm2d(inp),
            h_swish() if use_hs else nn.ReLU(inplace=True),
            # Pointwise expansion
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            h_swish() if use_hs else nn.ReLU(inplace=True),
            # Second depthwise
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size,
                1,
                (kernel_size - 1) // 2,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            SELayer(hidden_dim) if use_se else nn.Identity(),
            h_swish() if use_hs else nn.ReLU(inplace=True),
            # Pointwise projection
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def _build_ffn_block(
        self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs
    ):
        """Build Feed Forward Network block"""
        self.conv = nn.Sequential(
            # First pointwise
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            h_swish() if use_hs else nn.ReLU(inplace=True),
            # Depthwise
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size,
                stride,
                (kernel_size - 1) // 2,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            SELayer(hidden_dim) if use_se else nn.Identity(),
            h_swish() if use_hs else nn.ReLU(inplace=True),
            # Second pointwise
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.identity:
            return x + self.dropout2d(self.conv(x))
        else:
            return self.dropout2d(self.conv(x))


class MobileNetV4(MobileNet):
    """
    MobileNetV4 implementation with Universal Inverted Bottleneck blocks.
    """

    def __init__(self, cfgs, mode, **kwargs):
        super().__init__(**kwargs)
        self.cfgs = cfgs
        self.mode = mode

        # Build first layer
        input_channel = make_divisible(16 * self.width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2, theta=self.theta)]

        # Build UIB blocks
        block = UniversalInvertedBottleneck
        for k, t, c, use_se, use_hs, s, block_type in self.cfgs:
            output_channel = make_divisible(c * self.width_mult, 8)
            exp_size = make_divisible(input_channel * t, 8)
            layers.append(
                block(
                    input_channel,
                    exp_size,
                    output_channel,
                    k,
                    s,
                    use_se,
                    use_hs,
                    prob_dropout=self.prob_dropout,
                    mu=self.mu,
                    sigma=self.sigma,
                    type_dropout=self.type_dropout,
                    block_type=block_type,
                )
            )
            input_channel = output_channel

        self.features = nn.Sequential(*layers)
        self.conv_last = conv_1x1_bn(input_channel, self.embeding_dim)

        # Multi-head classification
        self.spoofer = nn.Sequential(
            Dropout(
                p=self.prob_dropout_linear,
                mu=self.mu,
                sigma=self.sigma,
                dist=self.type_dropout,
                linear=True,
            ),
            nn.BatchNorm1d(self.embeding_dim),
            h_swish(),
            nn.Linear(self.embeding_dim, 2),
        )

        if self.multi_heads:
            self.lightning = nn.Sequential(
                Dropout(
                    p=self.prob_dropout_linear,
                    mu=self.mu,
                    sigma=self.sigma,
                    dist=self.type_dropout,
                    linear=True,
                ),
                nn.BatchNorm1d(self.embeding_dim),
                h_swish(),
                nn.Linear(self.embeding_dim, 5),
            )
            self.spoof_type = nn.Sequential(
                Dropout(
                    p=self.prob_dropout_linear,
                    mu=self.mu,
                    sigma=self.sigma,
                    dist=self.type_dropout,
                    linear=True,
                ),
                nn.BatchNorm1d(self.embeding_dim),
                h_swish(),
                nn.Linear(self.embeding_dim, 11),
            )
            self.real_atr = nn.Sequential(
                Dropout(
                    p=self.prob_dropout_linear,
                    mu=self.mu,
                    sigma=self.sigma,
                    dist=self.type_dropout,
                    linear=True,
                ),
                nn.BatchNorm1d(self.embeding_dim),
                h_swish(),
                nn.Linear(self.embeding_dim, 40),
            )

    def forward(self, x):
        x = self.features(x)
        x = self.conv_last(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return x

    def make_logits(self, features, all=False):
        """Generate classification logits"""
        if all and self.multi_heads:
            return (
                self.spoofer(features),
                self.lightning(features),
                self.spoof_type(features),
                self.real_atr(features),
            )
        else:
            return self.spoofer(features)


def mobilenetv4_small(**kwargs):
    """
    Constructs a MobileNetV4-Small model
    """
    cfgs = [
        # k, t, c, SE, HS, s, block_type
        [3, 1, 16, 0, 0, 2, "uib"],
        [3, 4, 24, 0, 0, 2, "uib"],
        [3, 3, 24, 0, 0, 1, "uib"],
        [5, 3, 40, 1, 0, 2, "uib"],
        [5, 3, 40, 1, 0, 1, "uib"],
        [5, 3, 40, 1, 0, 1, "uib"],
        [3, 6, 80, 0, 1, 2, "extra_dw"],
        [3, 2.5, 80, 0, 1, 1, "extra_dw"],
        [3, 2.3, 80, 0, 1, 1, "extra_dw"],
        [3, 2.3, 80, 0, 1, 1, "extra_dw"],
        [3, 6, 112, 1, 1, 1, "ffn"],
        [3, 6, 112, 1, 1, 1, "ffn"],
        [5, 6, 160, 1, 1, 2, "uib"],
        [5, 6, 160, 1, 1, 1, "uib"],
        [5, 6, 160, 1, 1, 1, "uib"],
    ]
    return MobileNetV4(cfgs, mode="small", **kwargs)


def mobilenetv4_large(**kwargs):
    """
    Constructs a MobileNetV4-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s, block_type
        [3, 1, 16, 0, 0, 2, "uib"],
        [3, 4, 24, 0, 0, 2, "uib"],
        [3, 3, 24, 0, 0, 1, "uib"],
        [5, 3, 40, 1, 0, 2, "uib"],
        [5, 3, 40, 1, 0, 1, "uib"],
        [5, 3, 40, 1, 0, 1, "uib"],
        [3, 6, 80, 0, 1, 2, "extra_dw"],
        [3, 2.5, 80, 0, 1, 1, "extra_dw"],
        [3, 2.3, 80, 0, 1, 1, "extra_dw"],
        [3, 2.3, 80, 0, 1, 1, "extra_dw"],
        [3, 6, 112, 1, 1, 1, "ffn"],
        [3, 6, 112, 1, 1, 1, "ffn"],
        [5, 6, 160, 1, 1, 2, "uib"],
        [5, 6, 160, 1, 1, 1, "uib"],
        [5, 6, 160, 1, 1, 1, "uib"],
        [5, 6, 160, 1, 1, 1, "uib"],
        [5, 6, 160, 1, 1, 1, "uib"],
    ]
    return MobileNetV4(cfgs, mode="large", **kwargs)


def mobilenetv4_medium(**kwargs):
    """
    Constructs a MobileNetV4-Medium model (balanced between small and large)
    """
    cfgs = [
        # k, t, c, SE, HS, s, block_type
        [3, 1, 16, 0, 0, 2, "uib"],
        [3, 4, 24, 0, 0, 2, "uib"],
        [3, 3, 24, 0, 0, 1, "uib"],
        [5, 3, 40, 1, 0, 2, "uib"],
        [5, 3, 40, 1, 0, 1, "uib"],
        [5, 3, 40, 1, 0, 1, "uib"],
        [3, 6, 80, 0, 1, 2, "extra_dw"],
        [3, 2.5, 80, 0, 1, 1, "extra_dw"],
        [3, 2.3, 80, 0, 1, 1, "extra_dw"],
        [3, 6, 112, 1, 1, 1, "ffn"],
        [3, 6, 112, 1, 1, 1, "ffn"],
        [5, 6, 160, 1, 1, 2, "uib"],
        [5, 6, 160, 1, 1, 1, "uib"],
        [5, 6, 160, 1, 1, 1, "uib"],
    ]
    return MobileNetV4(cfgs, mode="medium", **kwargs)
