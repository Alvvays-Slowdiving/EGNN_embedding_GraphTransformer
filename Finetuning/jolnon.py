import ast
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class NodeEncoder(nn.Module):
    def __init__(self, vocab_size: int, model_dim: int, atom_embed_dim: int = 192, aux_embed_dim: int = 64):
        super().__init__()
        self.atom_emb = nn.Embedding(vocab_size, atom_embed_dim)
        self.aux_mlp = nn.Sequential(
            nn.Linear(2, aux_embed_dim),  # [charge, aromatic_flag]
            nn.SiLU(),
            nn.Linear(aux_embed_dim, aux_embed_dim)
        )
        self.proj = nn.Linear(atom_embed_dim + aux_embed_dim, model_dim)

    def forward(self, atom_ids: torch.Tensor, charges: torch.Tensor, aromatic: torch.Tensor) -> torch.Tensor:
        a = self.atom_emb(atom_ids)               # (B,N,atom_embed_dim)
        aux = self.aux_mlp(torch.cat([charges, aromatic], dim=-1))  # (B,N,aux_embed_dim)
        return self.proj(torch.cat([a, aux], dim=-1))               # (B,N,model_dim)
    
class EGCL(nn.Module):
    def __init__(self, feat_dim: int, m_dim: int = 128, hidden_dim: int = 256, update_coords: bool = False):
        super().__init__()
        self.update_coords = update_coords
        self.phi_e = nn.Sequential(
            nn.Linear(feat_dim * 2 + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, m_dim), nn.SiLU()
        )
        self.phi_x = nn.Sequential(
            nn.Linear(m_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.phi_h = nn.Sequential(
            nn.Linear(feat_dim + m_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, feat_dim)
        )
        self.h_norm = nn.LayerNorm(feat_dim)

    def forward(self, h: torch.Tensor, x: torch.Tensor, mask: torch.Tensor):
        B, N, F = h.shape
        r2 = pairwise_squared_dist(x, mask)
        adj = torch.isfinite(r2) & (~torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0))

        r2_valid = torch.where(torch.isfinite(r2), r2, torch.zeros_like(r2))
        hi = h.unsqueeze(2).expand(B, N, N, F)
        hj = h.unsqueeze(1).expand(B, N, N, F)
        e_in = torch.cat([hi, hj, r2_valid.unsqueeze(-1)], dim=-1)
        m_ij = self.phi_e(e_in) * adj.unsqueeze(-1)

        m_i = m_ij.sum(dim=2)

        if self.update_coords:
            s_ij = self.phi_x(m_ij).squeeze(-1)
            s_ij = torch.tanh(s_ij) * 0.1              # tamed step size
            s_ij = s_ij * adj.float()
            diff = x.unsqueeze(2) - x.unsqueeze(1)
            deg = adj.sum(dim=-1).clamp(min=1).unsqueeze(-1)
            delta_x = (diff * s_ij.unsqueeze(-1)).sum(dim=2) / deg
            x = x + delta_x

        dh = self.phi_h(torch.cat([h, m_i], dim=-1))
        h = self.h_norm(h + dh)
        return h, x
    

class EGNNEmbedding(nn.Module):
    def __init__(self, feat_dim: int, depth: int = 2, m_dim: int = 128, hidden_dim: int = 256, update_coords: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([EGCL(feat_dim=feat_dim, m_dim=m_dim, hidden_dim=hidden_dim, update_coords=update_coords)
                                     for _ in range(depth)])

    def forward(self, h: torch.Tensor, x: torch.Tensor, mask: torch.Tensor):
        for layer in self.layers:
            h, x = layer(h, x, mask)
        return h, x
    
class GraphTransformerLayer(nn.Module):
    def __init__(self, dim: int, heads: int = 8, attn_dropout: float = 0.1, resid_dropout: float = 0.1,
                 ff_mult: int = 4):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_dropout)
        self.resid_drop = nn.Dropout(resid_dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Dropout(resid_dropout),
            nn.Linear(ff_mult * dim, dim)
        )

    def forward(self, h: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor):
        B, N, C = h.shape
        eps = 1e-8

        # --- inverse-square, row-normalized prior ---
        r2 = pairwise_squared_dist(coords, mask)                         # (B,N,N)
        eye = torch.eye(N, device=h.device, dtype=torch.bool).unsqueeze(0)
        valid = torch.isfinite(r2)

        # per-row mean of finite off-diagonal r^2
        off = valid & (~eye)
        r2_off = r2.masked_fill(~off, 0.0)
        cnt_row = off.sum(dim=-1, keepdim=True).clamp_min(1)             # (B,N,1)
        r2_row_mean = (r2_off.sum(dim=-1, keepdim=True) / cnt_row).clamp_min(1e-4)  # (B,N,1)

        # avoid self singularity using row mean
        r2_eff = torch.where(eye, r2_row_mean.expand_as(r2), r2)
        r2_eff = torch.nan_to_num(r2_eff, posinf=1e8).clamp_(min=1e-6, max=1e8)

        w_raw = torch.where(valid, 1.0 / (r2_eff + eps), torch.zeros_like(r2_eff))   # (B,N,N)
        row_sum = w_raw.sum(dim=-1, keepdim=True).clamp_min(1e-12)                   # (B,N,1)
        w = w_raw / row_sum                                                          # row-normalized
        log_w = torch.log(w.clamp_min(1e-12)).unsqueeze(1)                           # (B,1,N,N)

        # --- attention ---
        x = self.norm1(h)
        q = self.to_q(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)  # (B,H,N,D)
        k = self.to_k(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)  # (B,H,N,D)
        v = self.to_v(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)  # (B,H,N,D)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_logits = attn_logits + log_w.expand(B, self.heads, N, N)

        # Robust masking
        mask_q = mask.unsqueeze(1).unsqueeze(-1).expand(B, self.heads, N, 1)
        mask_k = mask.unsqueeze(1).unsqueeze(2).expand(B, self.heads, 1, N)
        pair_valid = valid.unsqueeze(1) & mask_q & mask_k
        attn_logits = attn_logits.masked_fill(~pair_valid, -1e9)
        attn_logits = attn_logits.masked_fill(~mask_q, 0.0)

        attn = torch.softmax(attn_logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)     # final guard
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)                        # (B,H,N,D)
        out = out * mask_q.float()
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.to_out(out)
        h = h + self.resid_drop(out)

        y = self.norm2(h)
        y = self.ff(y)
        h = h + self.resid_drop(y)
        return h

class GraphTransformer(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 256, depth: int = 6, heads: int = 8,
                 attn_dropout: float = 0.1, resid_dropout: float = 0.1, ff_mult: int = 4,
                 egnn_depth: int = 2, egnn_update_coords: bool = False):
        super().__init__()
        self.encoder = NodeEncoder(vocab_size=vocab_size, model_dim=dim,
                                   atom_embed_dim=192, aux_embed_dim=64)
        self.egnn = EGNNEmbedding(feat_dim=dim, depth=egnn_depth, m_dim=128, hidden_dim=dim,
                                  update_coords=egnn_update_coords)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(dim=dim, heads=heads, attn_dropout=attn_dropout,
                                  resid_dropout=resid_dropout, ff_mult=ff_mult)
            for _ in range(depth)
        ])
        self.mlp_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, atom_ids, charges, aromatic, coords, mask):
        h = self.encoder(atom_ids, charges, aromatic)
        h, coords = self.egnn(h, coords, mask)   # EGNN embedding
        for layer in self.layers:
            h = layer(h, coords, mask)
        h_sum = (h * mask.unsqueeze(-1)).sum(dim=1)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        h_pool = h_sum / denom
        return self.mlp_out(h_pool)
    
class GraphTransformerPredictor(pl.LightningModule):
    def __init__(self,
                 vocab_size: int, dim: int = 256, depth: int = 6, heads: int = 8,
                 attn_dropout: float = 0.1, resid_dropout: float = 0.1, ff_mult: int = 4,
                 lr: float = 3e-4, weight_decay: float = 1e-6,
                 t_max: int = 100,                      # used by cosine if per-epoch
                 scaler: Optional[StandardScalerTorch] = None,
                 egnn_depth: int = 2, egnn_update_coords: bool = False,
                 # ==== NEW: scheduler & loss config ====
                 scheduler: str = "cosine",             # ["cosine", "onecycle"]
                 warmup_ratio: float = 0.1,             # for cosine
                 steps_per_epoch: Optional[int] = None, # required for onecycle/per-step cosine
                 max_epochs: Optional[int] = None,      # required for onecycle/per-step cosine
                 loss_name: str = "focal_huber",              # ["mse","mae","huber","focal_huber","logcosh"]
                 huber_delta: float = 1.0,              # in *scaled* units (≈1 std)
                 focal_gamma: float = 0.8,              # 0.5~1.0 recommended
                 focal_wmax: float = 5.0):              # cap for focal weights
        super().__init__()
        self.model = GraphTransformer(vocab_size=vocab_size, dim=dim, depth=depth, heads=heads,
                                      attn_dropout=attn_dropout, resid_dropout=resid_dropout, ff_mult=ff_mult,
                                      egnn_depth=egnn_depth, egnn_update_coords=egnn_update_coords)
        self.lr = lr
        self.weight_decay = weight_decay
        self.t_max = t_max
        self.scaler = scaler or StandardScalerTorch()
        # store
        self.scheduler_name = scheduler
        self.warmup_ratio = warmup_ratio
        self.steps_per_epoch = steps_per_epoch
        self.max_epochs_cfg = max_epochs
        self.loss_name = loss_name
        self.huber_delta = huber_delta
        self.focal_gamma = focal_gamma
        self.focal_wmax = focal_wmax

    # ---------- losses (computed on *scaled* targets) ----------
    def _huber_per_sample(self, err_abs: torch.Tensor, delta: float) -> torch.Tensor:
        # err_abs: (B,1)
        small = 0.5 * (err_abs ** 2)
        large = delta * (err_abs - 0.5 * delta)
        return torch.where(err_abs <= delta, small, large)

    def _loss(self, y_hat_scaled: torch.Tensor, y_scaled: torch.Tensor) -> torch.Tensor:
        err = y_hat_scaled - y_scaled
        err_abs = err.abs()
        if self.loss_name == "mse":
            return (err ** 2).mean()
        if self.loss_name == "mae":
            return err_abs.mean()
        if self.loss_name == "logcosh":
            return torch.log(torch.cosh(err)).mean()
        if self.loss_name == "huber":
            per = self._huber_per_sample(err_abs, self.huber_delta)
            return per.mean()
        if self.loss_name == "focal_huber":
            per = self._huber_per_sample(err_abs, self.huber_delta)       # bounded gradient
            # batch-median normalized focal weight
            med = err_abs.detach().median().clamp_min(1e-8)
            w = ((err_abs / med) ** self.focal_gamma).clamp_max(self.focal_wmax)
            return (w * per).mean()
        raise ValueError(f"Unknown loss_name: {self.loss_name}")

    def forward(self, batch):
        return self.model(batch["atom_ids"], batch["charges"], batch["aromatic"],
                          batch["coords"], batch["mask"])

    def _compute_loss_and_metrics(self, batch, stage: str):
        y = batch["y"]
        y_scaled = self.scaler.transform(y)
        y_hat_scaled = self.forward(batch)

        loss = self._loss(y_hat_scaled, y_scaled)
        with torch.no_grad():
            y_hat = self.scaler.inverse_transform(y_hat_scaled)
            mae = F.l1_loss(y_hat, y)
            rmse = torch.sqrt(F.mse_loss(y_hat, y))
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, batch_size=y.shape[0])
        self.log(f"{stage}_mae_K", mae,  prog_bar=True, on_epoch=True, batch_size=y.shape[0])
        self.log(f"{stage}_rmse_K", rmse, prog_bar=True, on_epoch=True, batch_size=y.shape[0])
        return loss

    def training_step(self, batch, batch_idx):
        return self._compute_loss_and_metrics(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._compute_loss_and_metrics(batch, "val")

    def test_step(self, batch, batch_idx):
        self._compute_loss_and_metrics(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # ---- Scheduler selection ----
        if self.scheduler_name == "cosine":
            # per-step linear warmup then cosine
            if (self.steps_per_epoch is not None) and (self.max_epochs_cfg is not None):
                total_steps = int(self.steps_per_epoch * self.max_epochs_cfg)
                warmup_steps = max(1, int(self.warmup_ratio * total_steps))
                from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
                warm = LinearLR(opt, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
                cos  = CosineAnnealingLR(opt, T_max=max(1, total_steps - warmup_steps))
                sched = SequentialLR(opt, schedulers=[warm, cos], milestones=[warmup_steps])
                return {
                    "optimizer": opt,
                    "lr_scheduler": {"scheduler": sched, "interval": "step"}
                }
            else:
                # epoch-level fallback (still fine with early stopping)
                cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.t_max)
                return {"optimizer": opt, "lr_scheduler": cos}

        elif self.scheduler_name == "onecycle":
            assert self.steps_per_epoch is not None and self.max_epochs_cfg is not None, \
                "OneCycleLR requires steps_per_epoch and max_epochs."
            total_steps = int(self.steps_per_epoch * self.max_epochs_cfg)
            # set optimizer initial lr lower than max_lr via div_factor
            div_factor = 25.0
            final_div = 1e4
            for g in opt.param_groups:
                g["lr"] = self.lr / div_factor
            one = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=self.lr, total_steps=total_steps,
                pct_start=0.3, anneal_strategy="cos",
                div_factor=div_factor, final_div_factor=final_div
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": one, "interval": "step"}
            }

        else:
            return {"optimizer": opt}
