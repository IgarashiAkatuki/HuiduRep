import torch
from torch import nn


def load_model(model_class, checkpoint, device='cpu', **model_kwargs):
    model = model_class
    model.to(device)

    checkpoint = torch.load(checkpoint, map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=True)

    return model


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def batch_shuffle(x):
    """
    Shuffle batch for ShuffleBN.
    """
    batch_size = x.shape[0]
    idx_shuffle = torch.randperm(batch_size).to(x.device)
    idx_unshuffle = torch.argsort(idx_shuffle)
    return x[idx_shuffle], idx_unshuffle

def batch_unshuffle(x, idx_unshuffle):
    """
    Unshuffle back after passing through encoder_k.
    """
    return x[idx_unshuffle]