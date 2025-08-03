import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self,
                 input_dim=5,
                 embedding_dim=64,
                 hidden_dim=256,
                 output_dim=5,
                 use_embedding=True,):
        super().__init__()
        self.output_dim = output_dim
        # if use_embedding:
        #     input_dim = embedding_dim
        self.projection = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, output_dim),
            # nn.BatchNorm1d(output_dim, affine=False),
        )

    def forward(self, x):
        return self.projection(x)


class Reduce(nn.Module):
    def __init__(self,
                 input_dim=21,
                 embedding_dim=64,
                 output_dim=5,
                 use_embedding=True,):
        super().__init__()

        if use_embedding:
            input_dim = embedding_dim

        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )


    def forward(self, x):
        return self.projection(x)