import math

import torch
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        max_epochs,
        warmup_start_lr=1e-6,
        eta_min=1e-5,
        last_epoch=-1
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_epoch = self.last_epoch + 1
        lrs = []

        for base_lr in self.base_lrs:
            if current_epoch < self.warmup_epochs:
                slope = (base_lr - self.warmup_start_lr) / self.warmup_epochs
                lr = self.warmup_start_lr + slope * current_epoch
            else:
                progress = (current_epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                lr = self.eta_min + (base_lr - self.eta_min) * cosine_decay
            lrs.append(lr)

        return lrs

def get_param_groups(model, weight_decay=5e-3):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # if any(pattern in name.lower() for pattern in ['bias', 'bn', 'ln', 'norm']):
        #     no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]

class LARS(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=0.0, eta=0.001, eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.eta = eta
        self.eps = eps
        self.state = {}

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                dp = p.grad

                # Weight decay
                if group['weight_decay'] != 0:
                    dp = dp.add(p, alpha=group['weight_decay'])

                # LARS scaling
                param_norm = torch.norm(p)
                update_norm = torch.norm(dp)
                if param_norm > 0 and update_norm > 0:
                    dp *= self.eta * param_norm / (update_norm + self.eps)

                # Momentum
                param_state = self.state.setdefault(p, {})
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(group['momentum']).add_(dp)

                # Update
                p.add_(mu, alpha=-group['lr'])
