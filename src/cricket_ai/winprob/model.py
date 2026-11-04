"""Hand-written PyTorch win-probability model.

Everything here is plain PyTorch: the network, the standardisation, the mini-batch
training loop with early stopping, and temperature scaling for calibration. No
training frameworks (Lightning, HF Trainer) or sklearn estimators are involved.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from cricket_ai.winprob.data import FEATURES, LABEL, feature_matrix

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    hidden: tuple[int, ...] = (64, 64, 32)
    dropout: float = 0.1
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 2048
    max_epochs: int = 60
    patience: int = 6
    seed: int = 42
    device: str = "auto"


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0
    temperature: float = 1.0


def resolve_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Standardizer(nn.Module):
    """Feature scaling stored as buffers, so it travels with the checkpoint."""

    def __init__(self, n_features: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("std", torch.ones(n_features))

    def fit(self, x: torch.Tensor) -> Standardizer:
        self.mean.copy_(x.mean(0))
        self.std.copy_(x.std(0).clamp_min(1e-6))
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class WinProbMLP(nn.Module):
    """Standardise -> [Linear -> LayerNorm -> GELU -> Dropout] x N -> Linear -> logit / T."""

    def __init__(self, n_features: int, hidden: tuple[int, ...] = (64, 64, 32), dropout=0.1):
        super().__init__()
        self.scaler = Standardizer(n_features)
        layers: list[nn.Module] = []
        d = n_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        # Temperature for post-hoc calibration; 1.0 until calibrate() is called.
        self.register_buffer("temperature", torch.ones(()))
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.scaler(x)).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x) / self.temperature


def _to_tensors(df: pd.DataFrame, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(feature_matrix(df)).to(device)
    y = torch.from_numpy(df[LABEL].to_numpy(dtype=np.float32)).to(device)
    return x, y


@torch.no_grad()
def _eval_loss(model: WinProbMLP, x: torch.Tensor, y: torch.Tensor, bs: int = 8192) -> float:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    total = sum(loss_fn(model(x[i : i + bs]), y[i : i + bs]).item() for i in range(0, len(x), bs))
    return total / len(x)


def train_mlp(
    train: pd.DataFrame, val: pd.DataFrame, cfg: TrainConfig | None = None
) -> tuple[WinProbMLP, TrainHistory]:
    """Mini-batch AdamW with cosine LR decay and early stopping on validation log loss."""
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = resolve_device(cfg.device)

    x_tr, y_tr = _to_tensors(train, device)
    x_va, y_va = _to_tensors(val, device)

    model = WinProbMLP(len(FEATURES), cfg.hidden, cfg.dropout).to(device)
    model.scaler.fit(x_tr)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(x_tr) / cfg.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs * steps_per_epoch)
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

    hist = TrainHistory()
    best_state, best_val, stale = None, float("inf"), 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        perm = torch.randperm(len(x_tr), generator=gen).to(device)
        running = 0.0
        for i in range(0, len(x_tr), cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_tr[idx]), y_tr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item() * len(idx)
        hist.train_loss.append(running / len(x_tr))
        hist.val_loss.append(_eval_loss(model, x_va, y_va))
        log.info("epoch %3d  train %.4f  val %.4f", epoch, hist.train_loss[-1], hist.val_loss[-1])

        if hist.val_loss[-1] < best_val - 1e-5:
            best_val, stale, hist.best_epoch = hist.val_loss[-1], 0, epoch
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    model.load_state_dict(best_state)
    hist.temperature = calibrate(model, x_va, y_va)
    return model.eval(), hist


def calibrate(model: WinProbMLP, x_val: torch.Tensor, y_val: torch.Tensor) -> float:
    """Temperature scaling: fit a single scalar T on validation logits by minimising NLL."""
    model.eval()
    with torch.no_grad():
        logits = model.logits(x_val)
    log_t = torch.zeros((), device=logits.device, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    loss_fn = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = loss_fn(logits / log_t.exp(), y_val)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.exp().clamp(0.25, 4.0))
    model.temperature.fill_(t)
    return t


@torch.no_grad()
def predict_proba(model: WinProbMLP, df: pd.DataFrame, bs: int = 8192) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    x = torch.from_numpy(feature_matrix(df)).to(device)
    out = [torch.sigmoid(model(x[i : i + bs])).cpu() for i in range(0, len(x), bs)]
    return torch.cat(out).numpy() if out else np.empty(0, dtype=np.float32)


def save(model: WinProbMLP, cfg: TrainConfig, hist: TrainHistory, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(cfg),
            "features": FEATURES,
            "history": asdict(hist),
        },
        path,
    )


def load(path: Path, device: str = "cpu") -> WinProbMLP:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt["features"] != FEATURES:
        raise ValueError("Checkpoint was trained on a different feature set")
    cfg = ckpt["config"]
    model = WinProbMLP(len(FEATURES), tuple(cfg["hidden"]), cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()
