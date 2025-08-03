import torch
import torch.nn as nn

from model.FeatureEncoder import PositionalEncoding


class FeatureDecoder(nn.Module):
    def __init__(self,
                 input_dim=21,
                 embedding_dim=64,
                 num_layers=2,
                 num_heads=3,
                 dropout=0.1,
                 ff_dim=256,
                 use_embedding=True,
                 ):
        super().__init__()

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim))

        if use_embedding:
            self.decoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dropout=dropout,
                dim_feedforward=ff_dim,
                batch_first=True,
            )
            input_dim = embedding_dim
        else:
            self.decoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dropout=dropout,
                dim_feedforward=ff_dim,
                batch_first=True,
            )
        self.decoder = nn.TransformerEncoder(encoder_layer=self.decoder_layer, num_layers=num_layers)
        self.positional_encoding = PositionalEncoding(input_dim)
        # self.linear = nn.Linear(embedding_dim, input_dim)

    def forward(self, x):
        x = self.positional_encoding(x)
        x = self.decoder(x)
        return x

class SpikeDecoder(nn.Module):
    def __init__(self,
                 input_dim=21,
                 embedding_dim=64,
                 num_layers=4,
                 num_heads=3,
                 dropout=0.1,
                 ff_dim=512,
                 use_embedding=True,
                 ):
        super().__init__()

        self.use_embedding = use_embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim))

        if use_embedding:
            self.decoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dropout=dropout,
                dim_feedforward=ff_dim,
                batch_first=True,
            )
            self.proj = nn.Linear(embedding_dim, input_dim)
            input_dim = embedding_dim
        else:
            self.decoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dropout=dropout,
                dim_feedforward=ff_dim,
                batch_first=True,
            )
        self.decoder = nn.TransformerEncoder(encoder_layer=self.decoder_layer, num_layers=num_layers)
        self.positional_encoding = PositionalEncoding(input_dim)

    def forward(self, x):
        # x = self.positional_encoding(x)
        x = self.decoder(x)

        if self.use_embedding:
            x = self.proj(x)
        return x
