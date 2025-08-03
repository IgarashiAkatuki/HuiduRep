import copy
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from train.spike_dataset import SpikeDataset
from utils.monitor_utils import gmm_monitor, knn_monitor, kmeans_monitor


class CMAESTrainer:
    def __init__(self,
                 model,
                 optimizer,
                 train_dataset,
                 val_dataset,
                 train_labels,
                 val_labels,
                 test_labels,
                 use_transform=True,
                 batch_size=4096,
                 num_workers=2,
                 epochs=16,
                 device='cuda',
                 save_path='./checkpoints',
                 scheduler=None,
                 print_details=True,
                 tensorboard=False,
                 tensorboard_steps=100,
                 eval_epochs=4,
                 save_epochs=4,
                 total_steps=0,
                 valid=True):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.save_path = save_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.device = device
        self.scheduler = scheduler
        self.print_details = print_details
        self.tensorboard = tensorboard
        self.tensorboard_steps = tensorboard_steps
        self.eval_epochs = eval_epochs
        self.save_epochs = save_epochs
        self.valid = valid
        self.steps = 0
        self.total_steps = total_steps

        self.tau = self.model.m

        if self.tensorboard:
            self.writer = SummaryWriter(log_dir=self.save_path + '/tensorboard/log/' + str(round(time.time())))

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels

        self.train_loader = DataLoader(SpikeDataset(train_dataset, use_transform=use_transform),
                                       batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
        self.val_loader = DataLoader(SpikeDataset(val_dataset, use_transform=use_transform),
                                     batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers) if val_dataset is not None else None
        self.save_path = save_path
        self.scaler = GradScaler()
        os.makedirs(self.save_path, exist_ok=True)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            epoch_reconstruction_loss = 0
            epoch_contrastive_loss = 0

            progress = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.epochs}', leave=False)
            for batch in progress:
                self.steps += 1

                x_q, x_k, spike = batch
                x_q = x_q.to(self.device).to(torch.float32)
                x_k = x_k.to(self.device).to(torch.float32)
                spike = spike.to(self.device).to(torch.float32)

                with autocast():
                    loss, reconstruction_loss, contrastive_loss, (q_std, cos_sim_pos, cos_sim_neg, l_neg) = self.model.forward(x_q, x_k, spike, self.tau)
                epoch_loss += loss.item()

                if self.print_details:
                    epoch_reconstruction_loss += reconstruction_loss.item()
                    epoch_contrastive_loss += contrastive_loss.item()

                self.optimizer.zero_grad()
                clip_grad_norm_(self.model.parameters(), max_norm=8.0)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.scheduler.step()

                self.model.momentum_update_key_encoder_and_proj(self.model.m)

                if self.tensorboard and self.steps % self.tensorboard_steps == 0:

                    l_neg, _ = torch.sort(l_neg, dim=1, descending=True)
                    max_neg_sim = l_neg[:, :20].mean(dim=1)
                    ratio = (l_neg > 0.5).float().mean()

                    self.writer.add_scalar('Train/Loss', loss, self.steps)
                    self.writer.add_scalar('Train/Reconstruction Loss', reconstruction_loss, self.steps)
                    self.writer.add_scalar('Train/Contrastive Loss', contrastive_loss, self.steps)
                    self.writer.add_scalar('Details/q_std', q_std, self.steps)
                    self.writer.add_scalar('Details/cos_sim_pos', cos_sim_pos, self.steps)
                    self.writer.add_scalar('Details/cos_sim_neg', cos_sim_neg, self.steps)
                    self.writer.add_scalar('Details/max_neg_sim', max_neg_sim.mean().item(), self.steps)
                    self.writer.add_scalar('Details/ratio', ratio, self.steps)


                    current_lr = 0
                    for group in self.optimizer.param_groups:
                        current_lr = group['lr']
                        break
                    self.writer.add_scalar('LearningRate', current_lr, self.steps)

                    total_norm = 0.0
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            param_norm = param.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    self.writer.add_scalar("GradNorm/Total", total_norm, self.steps)

                self.tau = self.update_tau(self.steps, self.total_steps, self.model.m)

                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        grad_mean = param.grad.mean().item()
                        grad_std = param.grad.std().item()
                        if torch.isnan(param.grad).any() or abs(grad_mean) > 10:
                            print(f"[!] {name}: grad_mean={grad_mean:.3e}, grad_std={grad_std:.3e}")

                progress.set_postfix({'loss': loss.item(), 'reconstruction_loss': reconstruction_loss.item(), 'contrastive_loss': contrastive_loss.item()})

            print(f"[Epoch {epoch}] Train Loss: {epoch_loss / len(self.train_loader):.4f}")


            if self.print_details:
                print(f"[Epoch {epoch}] Reconstruction Loss: {epoch_reconstruction_loss / len(self.train_loader):.4f}")
                print(f"[Epoch {epoch}] Contrastive Loss: {epoch_contrastive_loss / len(self.train_loader):.4f}")

            if epoch % self.save_epochs == 0:
                self.save_checkpoint(epoch)
                print(f"[Epoch {epoch}] Saved checkpoint for epoch {epoch}")

            if epoch % self.eval_epochs == 0:

                if self.valid:
                    scores, acc, kmeans = self.monitor(
                        torch.tensor(copy.deepcopy(self.train_dataset).transpose(0, 2, 1), dtype=torch.float),
                        torch.tensor(self.train_labels, dtype=torch.int),
                        torch.tensor(copy.deepcopy(self.val_dataset).transpose(0, 2, 1), dtype=torch.float),
                        torch.tensor(self.val_labels, dtype=torch.int),
                        self.test_labels,
                    )
                    self.writer.add_scalar('Validation/ARI', torch.tensor(np.mean(scores)), self.steps)
                    self.writer.add_scalar('Validation/Accuracy', acc, self.steps)
                    self.writer.add_scalar('Validation/KMeans', torch.tensor(kmeans[0]), self.steps)


    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        val_reconstruction_loss=0
        val_contrastive_loss=0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, leave=False):
                x_q, x_k = batch
                x_q = x_q.to(self.device)
                x_k = x_k.to(self.device)
                loss, reconstruction_loss, contrastive_loss = self.model(x_q, x_k)
                val_loss += loss.item()

                if self.print_details:
                    val_reconstruction_loss += reconstruction_loss.item()
                    val_contrastive_loss += contrastive_loss.item()

        if self.tensorboard:
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Reconstruction Loss/Validation', val_reconstruction_loss, epoch)
            self.writer.add_scalar('Contrastive Loss/Validation', val_contrastive_loss, epoch)

        print(f"[Epoch {epoch}] Val Loss: {val_loss / len(self.val_loader):.4f}")

        if self.print_details:
            print(f"[Epoch {epoch}] Reconstruction Loss: {val_reconstruction_loss / len(self.val_loader):.4f}")
            print(f"[Epoch {epoch}] Contrastive Loss: {val_contrastive_loss / len(self.val_loader):.4f}")

    def save_checkpoint(self, epoch):
        save_file = os.path.join(self.save_path, f'cmaes_epoch{epoch}.pt')
        torch.save(self.model.state_dict(), save_file)


    def monitor(self, train_data, train_label, val_data, val_label, test_labels):
        self.model.eval()
        scores, _, _ = gmm_monitor(self.model, train_data, train_label, val_data, val_label, test_labels, device=self.device)
        acc, _, _ = knn_monitor(self.model, train_data, train_label, val_data, val_label, test_labels, device=self.device)
        kmeans, _, _ = kmeans_monitor(self.model, train_data, train_label, val_data, val_label, test_labels, device=self.device)
        return scores, acc, kmeans

    def update_tau(self, curr_step, total_steps, base_tau):
        cosine = (1 + math.cos(math.pi * curr_step / total_steps)) / 2
        return 1 - (1 - base_tau) * cosine
