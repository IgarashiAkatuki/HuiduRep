import copy
import math

import torch
from torch import nn
import torch.nn.functional as F

from model.FeatureDecoder import SpikeDecoder, FeatureDecoder
from model.FeatureEncoder import TransformerEncoder, random_masking, recover, ConvEmbedding
from model.Prediction import Prediction
from model.ProjectionHead import ProjectionHead, Reduce

class CMAES(nn.Module):
    def __init__(self,
                 samples=121,
                 channels=11,
                 embedding_dim=64,
                 K=4096, # queue_size
                 m=0.997,
                 T=0.07,
                 n_heads=4,
                 num_layers=3,
                 dropout=0.05,
                 max_channels=11,
                 mask_ratio=0.25,
                 alpha=0.15,
                 momentum_update_interval=1,
                 use_avg_pool=True,
                 ff_dim=256,
                 use_embedding=True,
                 moco_v3=True): # embed_dim sames to max_channels
        super().__init__()

        self.K = K
        self.m = m
        self.T = T
        self.alpha = alpha
        self.mask_ratio = mask_ratio
        self.max_channels = max_channels
        self.samples = samples
        self.use_avg_pool = use_avg_pool
        self.use_embedding = use_embedding
        self.momentum_update_interval = momentum_update_interval
        self.global_step = 0
        self.moco_v3 = moco_v3
        self.embedding_dim = embedding_dim

        self.conv_embedding = ConvEmbedding(channels, embedding_dim)

        self.encoder_q = TransformerEncoder(embedding_dim=embedding_dim,
                                            num_layers=num_layers + 12,
                                            dropout=dropout,
                                            num_heads=n_heads,
                                            use_embedding=use_embedding,
                                            input_dim=channels,
                                            ff_dim=ff_dim,
                                            use_avg_pool=use_avg_pool,)
        self.encoder_k = copy.deepcopy(self.encoder_q)

        self.spike_decoder = SpikeDecoder(input_dim=channels,
                                          dropout=dropout,
                                          num_heads=n_heads,
                                          num_layers=num_layers,
                                          use_embedding=use_embedding,
                                          embedding_dim=embedding_dim,
                                          ff_dim=ff_dim)
        self.feature_decoder = FeatureDecoder(input_dim=channels,
                                              dropout=dropout,
                                              num_heads=n_heads,
                                              num_layers=num_layers - 2,
                                              use_embedding=use_embedding,
                                              embedding_dim=embedding_dim,
                                              ff_dim=ff_dim)

        self.reduce_q = Reduce(input_dim=channels,
                               use_embedding=use_embedding,
                               output_dim=32,
                               embedding_dim=embedding_dim,)
        self.reduce_k = copy.deepcopy(self.reduce_q)

        self.proj_q = ProjectionHead(input_dim=32,
                                     use_embedding=use_embedding,
                                     embedding_dim=embedding_dim)
        self.proj_k = copy.deepcopy(self.proj_q)

        if self.moco_v3:
            self.prediction = Prediction(input_dim=5,
                                           use_embedding=use_embedding,
                                           embedding_dim=embedding_dim)

        for param in self.encoder_k.parameters():
            param.requires_grad = False

        for param in self.proj_k.parameters():
            param.requires_grad = False

        for param in self.reduce_k.parameters():
            param.requires_grad = False

        if not self.moco_v3:
            self.register_buffer("queue", torch.randn(K, int(self.proj_k.output_dim)))
            self.queue = F.normalize(self.queue, dim=1)
            self.register_buffer("queue_pointer", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update_key_encoder_and_proj(self, momentum):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1.0 - momentum)

        for param_q, param_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1.0 - momentum)

        for param_q, param_k in zip(self.reduce_q.parameters(), self.reduce_k.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1.0 - momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.size(0)
        pointer = int(self.queue_pointer)

        if self.K - pointer <= batch_size:
            first_part = self.K - pointer
            second_part = batch_size - first_part
            self.queue[pointer:] = keys[:first_part]
            self.queue[:second_part] = keys[first_part:]
        else:
            self.queue[pointer : pointer + batch_size] = keys

        pointer = (pointer + batch_size) % self.K
        self.queue_pointer[0] = torch.tensor(pointer)

    def forward(self, q, k, spike, tau=0):

        batch_size, samples, channels = q.size()
        q_copy = copy.deepcopy(spike)

        if channels < self.max_channels:
            pad = torch.zeros(batch_size, samples, self.max_channels - channels).to(q.device)
            q = torch.cat((q, pad), dim=-1)
            k = torch.cat((k, pad), dim=-1)

        elif channels > self.max_channels:
            q = q[:, :, :self.max_channels]
            k = k[:, :, :self.max_channels]

        if not self.moco_v3:

            q = self.conv_embedding(q)
            k = self.conv_embedding(k)

            q = self.encoder_q(q)

            with torch.no_grad():
                k = self.encoder_k(k)
                B, _, C = k.shape

            if not self.use_avg_pool:
                q_cls = q[:, 0, :]
                q_reduce = self.reduce_q(q_cls)
                q_proj = self.proj_q(q_reduce)
                q_proj = self.prediction(q_proj)

                with torch.no_grad():
                    k_cls = k[:, 0, :]
                    k_reduce = self.reduce_k(k_cls)
                    k_proj = self.proj_k(k_reduce)
            else:
                q_mean = torch.mean(q, dim=1, keepdim=True)
                q_mean = torch.squeeze(q_mean, dim=1)
                q_reduce = self.reduce_q(q_mean)
                q_proj = self.proj_q(q_reduce)

                k_mean = torch.mean(k, dim=1, keepdim=True)
                k_mean = torch.squeeze(k_mean, dim=1)
                with torch.no_grad():
                    k_reduce = self.reduce_k(k_mean)
                    k_proj = self.proj_k(k_reduce)

            q_proj = F.normalize(q_proj, dim=-1)
            k_proj = F.normalize(k_proj, dim=-1)

            if self.use_avg_pool:
                reconstructed_q = self.spike_decoder(k_reduce)
            else:
                reconstructed_q = self.spike_decoder(k_reduce)

            l_batch = q_proj @ q_proj.t()
            mask = ~torch.eye(q_proj.shape[0], dtype=torch.bool, device=q.device)
            l_batch = l_batch[mask].view(q_proj.size(0), -1)

            l_pos = torch.einsum('nc,nc->n', [q_proj, k_proj]).unsqueeze(-1)
            l_neg = torch.einsum('nc,kc->nk', [q_proj, self.queue.clone().detach()])
            l_neg = torch.cat([l_neg, l_batch], dim=1)

            alpha = 10.0
            mask = (l_neg >= (l_pos - 0.10)) & (l_neg >= 0.7)
            l_neg[mask] = (torch.softmax(-alpha * (l_neg[mask] - 1.0), dim=-1) * l_neg[mask]).to(l_neg.dtype)

            logits = torch.cat([l_pos, l_neg], dim=1) / self.T

            contrastive_loss = self.debiased_contrastive_loss(logits, self.T)

        else:
            q_1, feature_q, ids_mask = self.calculate_q(q)
            q_2, _, _ = self.calculate_q(k)

            k_1 = self.calculate_k(q)
            k_2 = self.calculate_k(k)

            q_proj = q_1
            k_proj = k_2

            if self.use_avg_pool:
                reconstructed_q = self.spike_decoder(feature_q)
            else:
                reconstructed_q = self.spike_decoder(feature_q[:, 1:, :])

            l_batch_1 = q_1 @ k_2.t()
            l_pos_1 = torch.einsum('nc,nc->n', [q_1, k_2]).unsqueeze(-1)
            mask = ~torch.eye(q_1.shape[0], dtype=torch.bool, device=q.device)
            l_batch_1 = l_batch_1[mask].view(q_1.size(0), -1)
            l_neg = l_batch_1
            l_batch_1 = torch.cat([l_pos_1, l_batch_1], dim=1) / self.T

            labels_1 = torch.zeros(l_batch_1.size(0), dtype=torch.long, device=l_batch_1.device)
            contrastive_loss_1 = 2 * self.T * self.contrastive_loss(l_batch_1, labels_1)

            l_batch_2 = q_2 @ k_1.t()
            l_pos_2 = torch.einsum('nc,nc->n', [q_2, k_1]).unsqueeze(-1)
            mask = ~torch.eye(q_2.shape[0], dtype=torch.bool, device=q.device)
            l_batch_2 = l_batch_2[mask].view(q_2.size(0), -1)
            l_batch_2 = torch.cat([l_pos_2, l_batch_2], dim=1) / self.T

            labels_2 = torch.zeros(l_batch_2.size(0), dtype=torch.long, device=l_batch_2.device)
            contrastive_loss_2 = 2 * self.T * self.contrastive_loss(l_batch_2, labels_2)

            contrastive_loss = contrastive_loss_1 + contrastive_loss_2

        # monitor
        q_std = q_proj.std(dim=0).mean().item()
        cos_sim_pos = F.cosine_similarity(q_proj, k_proj).mean().item()
        if not self.moco_v3:
            cos_sim_neg = F.cosine_similarity(q_proj.unsqueeze(1), self.queue.detach(), dim=-1).mean().item()
        else:
            cos_sim_neg = 0

        reconstruction_loss = self.reconstruction_loss(reconstructed_q, q_copy)

        loss =  reconstruction_loss * self.alpha + contrastive_loss

        if not self.moco_v3:
            self._dequeue_and_enqueue(k_proj)
            self.global_step += 1
        return loss, reconstruction_loss, contrastive_loss, (q_std, cos_sim_neg, cos_sim_pos, l_neg)

    @staticmethod
    def reconstruction_loss(reconstructed_q, q):
        return F.mse_loss(reconstructed_q, q)

    @staticmethod
    def contrastive_loss(logits, labels):
        return F.cross_entropy(logits, labels)

    @staticmethod
    def debiased_contrastive_loss(logits, T):
        bias = torch.exp(torch.tensor(-1.0 / T, device=logits.device))
        exp_logits = torch.exp(logits)
        pos = exp_logits[:, 0]
        neg = exp_logits[:, 1:]
        neg = torch.clamp(neg - bias, min=0.0)
        loss = -torch.log(pos / (pos + neg.sum(dim=1)))
        return loss.mean()

    @staticmethod
    def gather_by_mask(x, ids_mask):
        B, _, C = x.shape
        ids_expanded = ids_mask.unsqueeze(-1).expand(-1, -1, C)
        return torch.gather(x, dim=1, index=ids_expanded)

    def get_reconstruction_q(self, q):
        batch_size, samples, channels = q.size()
        q, ids_keep, ids_mask, ids_restore, (pad_len, num_patches, patch_size) = random_masking(q,
                                                                                                mask_ratio=self.mask_ratio)

        if channels < self.max_channels:
            pad = torch.zeros(batch_size, samples, self.max_channels - channels).to(q.device)
            q = torch.cat((q, pad), dim=-1)

        elif channels > self.max_channels:
            q = q[:, :, :self.max_channels]

        q = self.encoder_q(q)
        q = recover(q, ids_keep, ids_restore, pad_len, num_patches, patch_size, samples, cls=not self.use_avg_pool)

        if self.use_avg_pool:
            reconstructed_q = self.spike_decoder(q)
        else:
            reconstructed_q = self.spike_decoder(q[:, 1:, :])

        return reconstructed_q

    def transform(self, x, x_denoised):
        with torch.no_grad():
            self.eval()
            x = self.channel_truncation(x)
            x = self.conv_embedding(x)
            if x_denoised is not None:
                # x =  x + x_denoised
                # print(x.shape)
                # print(x_denoised.shape)
                x = torch.cat((x, x_denoised), dim=1)
            x = self.encoder_q(x)
            if self.use_avg_pool:
                x_mean = torch.mean(x, dim=1, keepdim=True)
                x_mean = torch.squeeze(x_mean, dim=1)
                x = self.reduce_q(x_mean)
            else:
                x_cls = x[:, 0, :]
                x_cls = F.normalize(x_cls, dim=1)
                x = self.reduce_q(x_cls)

            return x

    def denoise(self, x, num_samples=11):
        with torch.no_grad():
            self.eval()

            x = self.channel_truncation(x)
            x = self.conv_embedding(x)
            x = self.encoder_q(x)
            x = self.feature_decoder(x)

            return x

    def conv(self, x):
        with torch.no_grad():
            self.eval()
            x = self.channel_truncation(x)
            x = self.conv_embedding(x)
            return x

    def residual_transform(self, x):
        with torch.no_grad():
            self.eval()
            x_denoised = self.denoise(x)
            x_denoised = self.channel_truncation(x_denoised)
            x = self.channel_truncation(x)

            x =  x + x_denoised
            x = self.conv_embedding(x)
            x = self.encoder_q(x)

            if self.use_avg_pool:
                x_mean = torch.mean(x, dim=1, keepdim=True)
                x_mean = torch.squeeze(x_mean, dim=1)
                x = self.reduce_q(x_mean)
            else:
                x_cls = x[:, 0, :]
                x_cls = F.normalize(x_cls, dim=1)
                x = self.reduce_q(x_cls)

            return x

    def calculate_q(self, q):
        q = self.conv_embedding(q)
        q = self.encoder_q(q)

        if not self.use_avg_pool:
            q_cls = q[:, 0, :]
            q_cls = F.normalize(q_cls, dim=1)
            q_reduce = self.reduce_q(q_cls)
            q_proj = self.proj_q(q_reduce)

        else:
            q_mean = torch.mean(q, dim=1, keepdim=True)
            q_mean = torch.squeeze(q_mean, dim=1)
            q_reduce = self.reduce_q(q_mean)
            q_proj = self.proj_q(q_reduce)

        if self.moco_v3:
            q_proj = self.prediction(q_proj)
        q_proj = F.normalize(q_proj, dim=-1)

        return q_proj, q, []

    def calculate_k(self, k):
        k = self.conv_embedding(k)
        with torch.no_grad():
            k = self.encoder_k(k)

            if not self.use_avg_pool:
                k_cls = k[:, 0, :]
                k_cls = F.normalize(k_cls, dim=1)
                k_reduce = self.reduce_k(k_cls)
                k_proj = self.proj_k(k_reduce)
            else:
                k_mean = torch.mean(k, dim=1, keepdim=True)
                k_mean = torch.squeeze(k_mean, dim=1)
                k_reduce = self.reduce_k(k_mean)
                k_proj = self.proj_k(k_reduce)

            k_proj = F.normalize(k_proj, dim=-1)

            return k_proj


    def channel_truncation(self, x):
        B, T, C = x.shape
        truncation_size = self.max_channels // 2
        out = torch.zeros(B, T, self.max_channels).to(x.device)

        if C > self.max_channels:
            for i in range(0, B):
                spike = x[i, :, :].transpose(0, 1)
                max_channel = math.floor(C / 2)

                start = (max_channel - truncation_size)
                end = start + self.max_channels

                if start < 0:
                    start = 0
                    end = self.max_channels
                if end > C:
                    end = C
                    start = C - self.max_channels

                spike = spike[start : end, :].transpose(0, 1)
                out[i, :, :] = spike
            return out
        else:
            return x

    def byol_loss(self, p, z):
        return 2 - 2 * (p * z).sum(dim=-1).mean()

