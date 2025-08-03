from torch import nn

class Prediction(nn.Module):
    def __init__(self,
                 input_dim=21,
                 embedding_dim=64,
                 hidden_dim=256,
                 use_embedding=True,):
        super().__init__()

        self.input_dim = input_dim
        # if use_embedding:
        #     self.input_dim = embedding_dim

        self.prediction = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.input_dim),
        )

    def forward(self, x):
        return self.prediction(x)
