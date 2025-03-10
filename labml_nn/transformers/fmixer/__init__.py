"""
---
title: FNet - Mixing Tokens with Fourier Transforms
summary: >
  This is an annotated implementation/tutorial the FNet in PyTorch.
---

# FNet: Mixing Tokens with Fourier Transforms

This is a [PyTorch](https://pytorch.org) implementation of the paper
[FNet: Mixing Tokens with Fourier Transforms](https://arxiv.org/abs/2105.03824).

This paper replaces the [self-attention layer](../mha.html) with two
[Fourier transforms](https://en.wikipedia.org/wiki/Discrete_Fourier_transform) to
*mix* tokens.
This is a $7 \times$ more efficient than self-attention.
The accuracy loss of using this over self-attention is about 92% for
[BERT](https://paperswithcode.com/method/bert) on
[GLUE benchmark](https://paperswithcode.com/dataset/glue).

## Mixing tokens with two Fourier transforms

We apply Fourier transform along the hidden dimension (embedding dimension)
 and then along the sequence dimension.

$$
\mathcal{R}\big(\mathcal{F}_\text{seq} \big(\mathcal{F}_\text{hidden} (x) \big) \big)
$$

where $x$ is the embedding input, $\mathcal{F}$ stands for the fourier transform and
$\mathcal{R}$ stands for the real component in complex numbers.

This is very simple to implement on PyTorch - just 1 line of code.
The paper suggests using a precomputed DFT matrix and doing matrix multiplication to get the
Fourier transformation.

Here is [the training code](experiment.html) for using a FNet based model for classifying
[AG News](https://paperswithcode.com/dataset/ag-news).
"""

from typing import Optional

import torch
from torch import nn


class FFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(1, 512)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # $\text{query}$,$\text{key}$, and $\text{value}$ all should be equal to $x$ for token mixing
        assert query is key and key is value
        # Token mixing doesn't support masking. i.e. all tokens will see all other token embeddings.
        assert mask is None
        x = query
        # x = torch.fft.fft(x, dim=2).real

        x = x.permute((1, 2, 0)).contiguous()
        # print("x, ", x.shape)
        B, C, N = x.shape
        # print("x ", x.shape)
        # x = x.permute(0, 2, 1).contiguous()
        x = torch.fft.ifft(x)
        # print("x", x.shape)
        w = torch.fft.fft(self.w.weight).unsqueeze(0).expand(B, C, N)
        # print("w", w.shape)
        xw = x.mul(w)
        # print("xw: ", xw.shape)
        xw = torch.fft.fft(xw).real
        # x = xw.permute(0, 2, 1).contiguous()
        x = xw.view(B, C, N)
        x = x.permute((2, 0, 1)).contiguous()
        #
        fft_hidden = torch.fft.fft(x, dim=2).real
        x = torch.fft.fft(fft_hidden, dim=0).real
        return x


class MIXFFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_w = nn.Linear(1, 512)
        self.hidden_w = nn.Linear(1, 512)

    def channel_mixer(self, x):
        """

        :param x:
        :return:
        """
        x = x.permute((1, 0, 2)).contiguous()
        # print("x, ", x.shape)
        B, N, C = x.shape
        # print("x ", x.shape)
        # x = x.permute(0, 2, 1).contiguous()
        x = torch.fft.ifft(x)
        # print("x", x.shape)
        w = torch.fft.fft(self.hidden_w.weight).unsqueeze(0).expand(B, C, N)
        # print("hiddel w", self.hidden_w.weight[:10, :])

        # print("w", w.shape)
        xw = x.mul(w)
        # print("xw: ", xw.shape)
        xw = torch.fft.fft(xw)
        # x = xw.permute(0, 2, 1).contiguous()
        x = xw.view(B, N, C)
        x = x.permute((1, 0, 2)).contiguous()
        return x

    def token_mixer(self, x):
        """

        :param x:
        :return:
        """

        # toke mixer
        x = x.permute((1, 2, 0)).contiguous()
        # print("x, ", x.shape)
        B, C, N = x.shape
        # print("x ", x.shape)
        # x = x.permute(0, 2, 1).contiguous()
        x = torch.fft.ifft(x)
        # print("x", x.shape)
        w = torch.fft.fft(self.token_w.weight).unsqueeze(0).expand(B, C, N)
        # print("w", w.shape)
        xw = x.mul(w)
        # print("xw: ", xw.shape)
        xw = torch.fft.fft(xw).real
        # x = xw.permute(0, 2, 1).contiguous()
        x = xw.view(B, C, N)
        x = x.permute((2, 0, 1)).contiguous()
        return x

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # $\text{query}$,$\text{key}$, and $\text{value}$ all should be equal to $x$ for token mixing
        assert query is key and key is value
        # Token mixing doesn't support masking. i.e. all tokens will see all other token embeddings.
        assert mask is None
        x = query
        # channel mixer

        x = torch.fft.fft(x, dim=2)
        # Apply the Fourier transform along the sequence dimension
        # $$\mathcal{F}_\text{seq} \big(\mathcal{F}_\text{hidden} (x) \big)$$
        x = torch.fft.fft(x, dim=0)
        x = self.channel_mixer(x)
        x = self.token_mixer(x)
        return x


class MIXER(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_w = nn.Linear(512, 512)
        self.hidden_w = nn.Linear(512, 512)
        self.pfft_token_w = nn.Linear(1, 512)
        self.pfft_hidden_w = nn.Linear(1, 512)
        self.token_norm = nn.LayerNorm(normalized_shape=512)
        self.hidden_norm = nn.LayerNorm(normalized_shape=512)
        # self.pfft_token_w.requires_grad_(False)

    def channel_mixer(self, x):
        """

        :param x:
        :return:
        """
        x = self.hidden_w(x)
        # x = torch.softmax(x, dim=-1)
        # x = self.hidden_norm(x)

        return x

    def token_mixer(self, x):
        """

        :param x:
        :return:
        """
        x = x.permute((2, 1, 0))
        x = self.token_w(x)
        # x = self.token_norm(x)
        x = x.permute((2, 1, 0))
        return x

    def pfft_channel_mixer(self, x):
        """

        :param x:
        :return:
        """
        x = x.permute((1, 0, 2)).contiguous()
        # print("x, ", x.shape)
        B, N, C = x.shape
        # print("x ", x.shape)
        # x = x.permute(0, 2, 1).contiguous()
        x = torch.fft.ifft(x)
        # print("x", x.shape)
        w = torch.fft.fft(self.pfft_hidden_w.weight).unsqueeze(0).expand(B, C, N)
        # print("hiddel w", self.hidden_w.weight[:10, :])

        # print("w", w.shape)
        xw = x.mul(w)
        # print("xw: ", xw.shape)
        xw = torch.fft.fft(xw)
        # x = xw.permute(0, 2, 1).contiguous()
        x = xw.view(B, N, C)
        x = x.permute((1, 0, 2)).contiguous()
        return x

    def pfft_token_mixer(self, x):
        """

        :param x:
        :return:
        """

        # toke mixer
        x = x.permute((1, 2, 0)).contiguous()
        # print("x, ", x.shape)
        B, C, N = x.shape
        # print("x ", x.shape)
        # x = x.permute(0, 2, 1).contiguous()
        x = torch.fft.ifft(x)
        # print("x", x.shape)
        w = torch.fft.fft(self.pfft_token_w.weight).unsqueeze(0).expand(B, C, N).real
        # w = self.pfft_token_w.weight.unsqueeze(0).expand(B, C, N)
        # print("w", w.shape)
        xw = x.mul(w).real
        # print("xw: ", xw.shape)
        xw = torch.fft.fft(xw).real
        # x = xw.permute(0, 2, 1).contiguous()
        x = xw.view(B, C, N)
        x = x.permute((2, 0, 1)).contiguous()
        return x

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # $\text{query}$,$\text{key}$, and $\text{value}$ all should be equal to $x$ for token mixing
        assert query is key and key is value
        # Token mixing doesn't support masking. i.e. all tokens will see all other token embeddings.
        assert mask is None
        x = query
        # channel mixer fft
        # x = torch.fft.fft(x, dim=2)
        # Apply the Fourier transform along the sequence dimension
        # $$\mathcal{F}_\text{seq} \big(\mathcal{F}_\text{hidden} (x) \big)$$
        # token mixer fft
        # x = torch.fft.fft(x, dim=0)
        x = self.channel_mixer(x)

        x = self.token_mixer(x)
        # x = self.pfft_token_mixer(x)
        # x = torch.fft.fft(x, dim=0).real

        return x


class FNetMix(nn.Module):
    """
    ## FNet - Mix tokens

    This module simply implements
    $$
    \mathcal{R}\big(\mathcal{F}_\text{seq} \big(\mathcal{F}_\text{hidden} (x) \big) \big)
    $$

    The structure of this module is made similar to a [standard attention module](../mha.html) so that we can simply
    replace it.
    """


    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        The [normal attention module](../mha.html) can be fed with different token embeddings for
        $\text{query}$,$\text{key}$, and $\text{value}$ and a mask.

        We follow the same function signature so that we can replace it directly.

        For FNet mixing, $$x = \text{query} = \text{key} = \text{value}$$ and masking is not possible.
        Shape of `query` (and `key` and `value`) is `[seq_len, batch_size, d_model]`.
        """

        # $\text{query}$,$\text{key}$, and $\text{value}$ all should be equal to $x$ for token mixing
        assert query is key and key is value
        # Token mixing doesn't support masking. i.e. all tokens will see all other token embeddings.
        assert mask is None

        # Assign to `x` for clarity
        x = query
        # print("x shape: ", x.shape)

        # Apply the Fourier transform along the hidden (embedding) dimension
        # $$\mathcal{F}_\text{hidden} (x)$$
        #
        # The output of the Fourier transform is a tensor of
        # [complex numbers](https://pytorch.org/docs/stable/complex_numbers.html).

        fft_hidden = torch.fft.fft(x, dim=2)
        # Apply the Fourier transform along the sequence dimension
        # $$\mathcal{F}_\text{seq} \big(\mathcal{F}_\text{hidden} (x) \big)$$
        fft_seq = torch.fft.fft(fft_hidden, dim=0)
        # print("fft swq: ", fft_seq.shape)
        # Get the real component
        # $$\mathcal{R}\big(\mathcal{F}_\text{seq} \big(\mathcal{F}_\text{hidden} (x) \big) \big)$$
        return torch.real(fft_seq)
