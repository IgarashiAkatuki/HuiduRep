
import torch
import torch.nn as nn
import torch.nn.functional as F

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        if d_model % 2 != 0:
            d_model += 1

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))


    # x: [batch_size, samples, channels]
    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :x.size(2)]
        return x


def random_masking(x, mask_ratio=0.25, patch_size=4):
    B, T, C = x.shape
    pad_len = (patch_size - T % patch_size) % patch_size

    if pad_len > 0:
        x = F.pad(x, (0, 0, 0, pad_len))

    T_padded = x.shape[1]
    num_patches = T_padded // patch_size
    len_keep = int(num_patches * (1 - mask_ratio))

    x_reshaped = x.view(B, num_patches, patch_size, C)

    # if pad_len > 0:
    #     noise = torch.rand(B, num_patches - 1, device=x.device)
    # else:
    #     noise = torch.rand(B, num_patches, device=x.device)
    noise = torch.rand(B, num_patches, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    ids_mask = ids_shuffle[:, len_keep:]
    # if pad_len > 0:
    #     ids_mask = torch.cat([ids_mask, torch.zeros(B, 1, device=x.device, dtype=torch.int64) + num_patches - 1], dim=1)

    # ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep, _ = torch.sort(ids_keep, dim=-1)
    ids_restore = torch.argsort(torch.cat((ids_keep, ids_mask), dim=1), dim=-1)
    ids_keep_exp = ids_keep.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, patch_size, C)
    x_kept = torch.gather(x_reshaped, dim=1, index=ids_keep_exp)
    x_kept = x_kept.reshape(B, -1, C)[:, :num_patches * patch_size, :]

    # x_kept = x_kept.reshape(B, -1, C * patch_size)
    return x_kept, ids_keep, ids_mask, ids_restore, (pad_len, num_patches, patch_size)

def recover(masked_x, ids_keep, ids_restore, pad_len, num_patches, patch_size=10, original_size=121, cls=True):
    B, T, C = masked_x.shape
    # C = C // patch_size
    # masked_x = masked_x.reshape(B, -1, C)

    if cls:
        patch_tokens = masked_x[:, 1:, :]
        len_keep = masked_x.shape[1] - 1
    else:
        patch_tokens = masked_x
        len_keep = masked_x.shape[1]

    mask_token = torch.zeros(1, 1, C, device=masked_x.device, dtype=masked_x.dtype)
    mask_tokens = mask_token.expand(B, original_size - len_keep + pad_len, C)
    tokens = torch.cat((patch_tokens, mask_tokens), dim=1)
    tokens= tokens.view(B, num_patches, patch_size, C)

    x_recovered = torch.gather(tokens, 1, index=ids_restore[:, :, None, None].expand(-1, -1, patch_size, C))
    x_recovered = x_recovered.view(B, original_size + pad_len, C)[:, :original_size, :]

    if cls:
        cls_token = masked_x[:, 0:1, :]
        x_recovered = torch.cat((cls_token, x_recovered), dim=1)

    return x_recovered


class TransformerEncoder(nn.Module):
    def __init__(self,
                 input_dim=21,
                 embedding_dim=64, # max_channels
                 num_heads=4,
                 num_layers=6,
                 dropout=0.1,
                 ff_dim=256,
                 use_embedding=True,
                 use_avg_pool=True,):
        super().__init__()

        self.use_embedding = use_embedding
        self.use_avg_pool = use_avg_pool
        if use_embedding:
            self.embedding = nn.Linear(input_dim, embedding_dim)
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dropout=dropout,
                batch_first=True,
                dim_feedforward=ff_dim,
            )
            input_dim = embedding_dim
        else:
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dropout=dropout,
                batch_first=True,
                dim_feedforward=ff_dim,
            )

        if not self.use_avg_pool:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim))

        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.positional_encoding = PositionalEncoding(input_dim)


    def forward(self, x):
        # if self.use_embedding:
        #     x = self.embedding(x)

        batch_size = x.size(0)

        if not self.use_avg_pool:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.positional_encoding(x)
        x = self.transformer(x)
        # cls_output = x[:, 0, :]
        # return F.normalize(cls_output, dim=1) #[batch_size, 1, embed_dim]
        return x


class ConvEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=5, stride=1, padding=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size, stride, padding),
            # nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Conv1d(output_dim, output_dim, kernel_size - 2, stride, padding - 1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(output_dim, output_dim // 16, 1),
            nn.ReLU(),
            nn.Conv1d(output_dim // 16, output_dim, 1),
            nn.Sigmoid()
        )

        self.shortcut = nn.Conv1d(input_dim, input_dim, 1) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, C, T]
        out = self.block(x)
        out = out

        # weight = self.fc(out)
        # out = out * weight
        out = out.permute(0, 2, 1)  # [B, T, C]
        return out
