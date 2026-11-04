"""Hand-written conditional tabular GAN in plain PyTorch.

Architecture follows the ideas behind CTGAN (Xu et al., 2019), written from scratch:

* **Generator**: noise + conditional vector -> residual blocks
  (Linear -> BatchNorm -> ReLU, concatenated with the block input) -> per-column heads:
  tanh for mode offsets, Gumbel-softmax for one-hot blocks.
* **Critic**: PacGAN (Lin et al., 2018). It scores ``pac`` rows at once, so a generator
  that keeps emitting the same row is easy to spot; this counters mode collapse.
* **Objective**: WGAN with gradient penalty (Gulrajani et al., 2017), plus a
  cross-entropy term that makes the generator honour the conditional vector.
* **Training-by-sampling**: see ``sampler.py``.

The only non-PyTorch piece is the Gaussian mixture used for preprocessing
(``transformer.py``).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from cricket_ai.synthetic.sampler import ConditionalSampler
from cricket_ai.synthetic.transformer import DataTransformer, Span
from cricket_ai.winprob.model import resolve_device

log = logging.getLogger(__name__)


@dataclass
class GANConfig:
    embedding_dim: int = 128
    generator_dims: tuple[int, ...] = (256, 256)
    critic_dims: tuple[int, ...] = (256, 256)
    lr_generator: float = 2e-4
    lr_critic: float = 2e-4
    weight_decay: float = 1e-6
    batch_size: int = 500
    pac: int = 10
    critic_steps: int = 1
    critic_dropout: float = 0.5
    epochs: int = 300
    gp_lambda: float = 10.0
    gumbel_tau: float = 0.2
    log_frequency: bool = True
    seed: int = 42
    device: str = "auto"
    log_every: int = 10


@dataclass
class GANHistory:
    generator_loss: list[float] = field(default_factory=list)
    critic_loss: list[float] = field(default_factory=list)
    seconds: float = 0.0
    device: str = "cpu"


# ----- networks -----
class Residual(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)
        self.bn = nn.BatchNorm1d(d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([F.relu(self.bn(self.fc(x))), x], dim=1)


class Generator(nn.Module):
    def __init__(self, d_in: int, dims: tuple[int, ...], d_data: int):
        super().__init__()
        layers: list[nn.Module] = []
        d = d_in
        for h in dims:
            layers.append(Residual(d, h))
            d += h
        layers.append(nn.Linear(d, d_data))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, d_in: int, dims: tuple[int, ...], pac: int, dropout: float = 0.5):
        super().__init__()
        self.pac = pac
        self.pac_dim = d_in * pac
        layers: list[nn.Module] = []
        d = self.pac_dim
        for h in dims:
            layers += [nn.Linear(d, h), nn.LeakyReLU(0.2), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(-1, self.pac_dim))


# ----- helpers -----
def apply_activations(raw: torch.Tensor, spans: list[Span], tau: float) -> torch.Tensor:
    out, st = [], 0
    for span in spans:
        block = raw[:, st : st + span.dim]
        if span.activation == "tanh":
            out.append(torch.tanh(block))
        else:
            out.append(F.gumbel_softmax(block, tau=tau, hard=False))
        st += span.dim
    return torch.cat(out, dim=1)


def condition_loss(
    raw: torch.Tensor, transformer: DataTransformer, cond: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy between each discrete head and its requested category (chosen column only)."""
    losses, st, st_c = [], 0, 0
    for info in transformer.info:
        if info.kind == "discrete":
            n = len(info.categories)
            target = cond[:, st_c : st_c + n].argmax(dim=1)
            losses.append(F.cross_entropy(raw[:, st : st + n], target, reduction="none"))
            st_c += n
        st += info.output_dim
    if not losses:
        return raw.new_zeros(())
    return (torch.stack(losses, dim=1) * mask).sum() / raw.size(0)


def gradient_penalty(
    critic: Critic, real: torch.Tensor, fake: torch.Tensor, lam: float
) -> torch.Tensor:
    pac = critic.pac
    alpha = torch.rand(real.size(0) // pac, 1, 1, device=real.device)
    alpha = alpha.repeat(1, pac, real.size(1)).reshape(-1, real.size(1))
    mixed = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    score = critic(mixed)
    grads = torch.autograd.grad(
        score, mixed, grad_outputs=torch.ones_like(score), create_graph=True, retain_graph=True
    )[0]
    norms = grads.reshape(-1, pac * real.size(1)).norm(2, dim=1)
    return lam * ((norms - 1) ** 2).mean()


# ----- model -----
class TabularGAN:
    name = "gan"

    def __init__(self, cfg: GANConfig | None = None):
        self.cfg = cfg or GANConfig()
        self.transformer: DataTransformer | None = None
        self.sampler: ConditionalSampler | None = None
        self.generator: Generator | None = None
        self.history = GANHistory()

    def _batch_size(self, n_rows: int) -> int:
        pac = self.cfg.pac
        batch = min(self.cfg.batch_size, n_rows)
        return max(pac, batch // pac * pac)

    def fit(self, df: pd.DataFrame, discrete_columns: list[str]) -> TabularGAN:
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        device = resolve_device(cfg.device)
        self.history = GANHistory(device=str(device))

        self.transformer = DataTransformer(seed=cfg.seed).fit(df, discrete_columns)
        data = self.transformer.transform(df, rng)
        self.sampler = ConditionalSampler(data, self.transformer, cfg.log_frequency)
        spans = self.transformer.output_info
        d_data, d_cond = data.shape[1], self.sampler.cond_dim

        self.generator = Generator(cfg.embedding_dim + d_cond, cfg.generator_dims, d_data)
        self.generator.to(device)
        critic = Critic(d_data + d_cond, cfg.critic_dims, cfg.pac, cfg.critic_dropout).to(device)
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=cfg.lr_generator,
            betas=(0.5, 0.9),
            weight_decay=cfg.weight_decay,
        )
        opt_c = torch.optim.Adam(
            critic.parameters(), lr=cfg.lr_critic, betas=(0.5, 0.9), weight_decay=cfg.weight_decay
        )

        data_t = torch.from_numpy(data).to(device)
        batch = self._batch_size(len(data))
        steps = max(1, len(data) // batch)
        start = time.perf_counter()
        for epoch in range(1, cfg.epochs + 1):
            g_sum = c_sum = 0.0
            for _ in range(steps):
                for _ in range(cfg.critic_steps):
                    cond, _, col, cat = self.sampler.sample_condvec(batch, rng)
                    c1 = torch.from_numpy(cond).to(device)
                    perm = rng.permutation(batch)
                    rows = self.sampler.sample_rows(col[perm], cat[perm], rng)
                    real = data_t[torch.from_numpy(rows).to(device)]
                    c2 = c1[torch.from_numpy(perm).to(device)]
                    z = torch.randn(batch, cfg.embedding_dim, device=device)
                    with torch.no_grad():
                        fake = apply_activations(
                            self.generator(torch.cat([z, c1], 1)), spans, cfg.gumbel_tau
                        )
                    real_in = torch.cat([real, c2], 1)
                    fake_in = torch.cat([fake, c1], 1)
                    loss_c = critic(fake_in).mean() - critic(real_in).mean()
                    pen = gradient_penalty(critic, real_in, fake_in, cfg.gp_lambda)
                    opt_c.zero_grad(set_to_none=True)
                    (loss_c + pen).backward()
                    opt_c.step()

                cond, mask, _, _ = self.sampler.sample_condvec(batch, rng)
                c1 = torch.from_numpy(cond).to(device)
                m1 = torch.from_numpy(mask).to(device)
                z = torch.randn(batch, cfg.embedding_dim, device=device)
                raw = self.generator(torch.cat([z, c1], 1))
                fake = apply_activations(raw, spans, cfg.gumbel_tau)
                loss_g = -critic(torch.cat([fake, c1], 1)).mean()
                loss_g = loss_g + condition_loss(raw, self.transformer, c1, m1)
                opt_g.zero_grad(set_to_none=True)
                loss_g.backward()
                opt_g.step()
                g_sum += loss_g.item()
                c_sum += loss_c.item()

            self.history.generator_loss.append(g_sum / steps)
            self.history.critic_loss.append(c_sum / steps)
            if epoch == 1 or epoch % cfg.log_every == 0 or epoch == cfg.epochs:
                log.info(
                    "epoch %4d/%d  generator %.3f  critic %.3f  (%.0fs)",
                    epoch,
                    cfg.epochs,
                    self.history.generator_loss[-1],
                    self.history.critic_loss[-1],
                    time.perf_counter() - start,
                )
        self.history.seconds = round(time.perf_counter() - start, 1)
        self.generator.eval()
        return self

    @torch.no_grad()
    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        if self.generator is None:
            raise RuntimeError("Call fit() or load() first")
        seed = self.cfg.seed if seed is None else seed
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        device = next(self.generator.parameters()).device
        self.generator.eval()
        spans = self.transformer.output_info
        batch = self._batch_size(max(n, self.cfg.pac))
        out = []
        for _ in range(math.ceil(n / batch)):
            z = torch.randn(batch, self.cfg.embedding_dim, device=device)
            cond = torch.from_numpy(self.sampler.sample_original_condvec(batch, rng)).to(device)
            raw = self.generator(torch.cat([z, cond], 1))
            out.append(apply_activations(raw, spans, self.cfg.gumbel_tau).cpu().numpy())
        return self.transformer.inverse_transform(np.concatenate(out)[:n])

    # ----- persistence -----
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.cfg),
                "generator": self.generator.state_dict(),
                "history": asdict(self.history),
            },
            directory / "generator.pt",
        )
        joblib.dump(
            {"transformer": self.transformer, "sampler": self.sampler},
            directory / "encoding.joblib",
        )

    @classmethod
    def load(cls, directory: Path, device: str = "cpu") -> TabularGAN:
        ckpt = torch.load(directory / "generator.pt", map_location=device, weights_only=False)
        cfg_d = ckpt["config"]
        for k in ("generator_dims", "critic_dims"):
            cfg_d[k] = tuple(cfg_d[k])
        model = cls(GANConfig(**cfg_d))
        enc = joblib.load(directory / "encoding.joblib")
        model.transformer, model.sampler = enc["transformer"], enc["sampler"]
        d_in = model.cfg.embedding_dim + model.sampler.cond_dim
        model.generator = Generator(d_in, model.cfg.generator_dims, model.transformer.output_dim)
        model.generator.load_state_dict(ckpt["generator"])
        model.generator.to(device).eval()
        model.history = GANHistory(**ckpt["history"])
        return model
