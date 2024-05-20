import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from positional_encodings.torch_encodings import PositionalEncoding1D


class PermutationLearningBlock(nn.Module):
    def __init__(
        self,
        latent_dim,
        negative_slope=0.01,
    ):
        super().__init__()
        self.negative_slope = negative_slope

        self.linear0 = nn.Linear(latent_dim, latent_dim)
        self.linear1 = nn.Linear(latent_dim, latent_dim)


    def forward(self, x):
        x0 = x
        x1 = F.leaky_relu(self.linear0(x0), negative_slope=self.negative_slope)
        x2 = F.leaky_relu(self.linear1(x1), negative_slope=self.negative_slope)
        return x2


class PermutationLearningModel(nn.Module):
    """
    Based on Ghorbani et al. 2019
    """
    def __init__(
        self,
        num_markers=41,
        latent_dim=128,
        negative_slope=0.01,
    ):
        super().__init__()

        self.num_markers = num_markers

        self.embedding = nn.Linear(num_markers * 3, latent_dim)
        self.negative_slope = negative_slope

        self.block0 = PermutationLearningBlock(latent_dim, negative_slope=self.negative_slope)
        self.block1 = PermutationLearningBlock(latent_dim, negative_slope=self.negative_slope)
        self.block2 = PermutationLearningBlock(latent_dim, negative_slope=self.negative_slope)

        self.output = nn.Linear(latent_dim, num_markers * num_markers)


    def forward(self, x):
        x_flatten = torch.flatten(x, start_dim=2, end_dim=3)
        embedding = F.leaky_relu(self.embedding(x_flatten), negative_slope=self.negative_slope)
        x0 = F.leaky_relu(self.block0(embedding) + embedding, negative_slope=self.negative_slope)
        x1 = F.leaky_relu(self.block1(x0) + x0, negative_slope=self.negative_slope)
        x2 = F.leaky_relu(self.block2(x1) + x1, negative_slope=self.negative_slope)
        out = self.output(x2)
        out = torch.reshape(out, (out.shape[0], out.shape[1], self.num_markers, self.num_markers))
        return out


class MarkerTrackingAttention(nn.Module):
    def __init__(
        self,
        sequence_length,
        num_markers,
        latent_dim=128,
        num_heads=8,
        num_layers=3,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.embedding = nn.Embedding(sequence_length * num_markers, latent_dim)
        self.pos_encoder = PositionalEncoding1D(latent_dim)
        encoder_layers = nn.TransformerEncoderLayer(latent_dim, num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output = nn.Linear(latent_dim, sequence_length * num_markers)

    def forward(
        self,
        x,  # [N, F, M, 3]
    ):
        embedding = self.embedding(x) * math.sqrt(self.latent_dim)
        encoding = self.pos_encoder(embedding)
        encoding_transformer = self.transformer_encoder(encoding)
        output = self.output(encoding_transformer)
        return output
