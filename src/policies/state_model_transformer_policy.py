import torch
from torch import nn
import torch.nn.init as nn_init

from src.networks.sinusoidal_positional_encoding import SinusoidalPositionalEncoding


class StateModelTransformerPolicy(nn.Module):
    def __init__(self, in_dim, out_dim, d_model=64, nhead=4, dim_feedforward=128, num_layers=4, max_len=1024):
        super().__init__()


        self.in_dim = in_dim
        self.out_dim = out_dim

        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward

        self.num_layers = num_layers

        self.in_linear = nn.Linear(in_dim, d_model)

        self.pos_embedding = SinusoidalPositionalEncoding(d_model, max_len)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
            ),
            num_layers=num_layers
        )

        self.out_linear = nn.Linear(d_model, out_dim)
        nn_init.xavier_uniform_(self.out_linear.weight)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, sequence):
        mask = torch.triu(torch.ones(sequence.shape[0], sequence.shape[0]), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))

        embedded_sequence = self.in_linear(sequence)

        embedded_sequence += self.pos_embedding(torch.arange(len(embedded_sequence)))

        pred = self.transformer(embedded_sequence, mask=mask)
        pred = self.out_linear(pred)

        return pred

