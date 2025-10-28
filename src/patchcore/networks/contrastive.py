"""Contrastive memory and adapter modules for PatchCore."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class ContrastiveMemoryStats:
    """Light-weight container tracking CMAM diagnostics."""

    bank_size: int
    info_nce: float
    temperature: float
    momentum: float


class ContrastiveMemoryAugmentation(torch.nn.Module):
    """Dynamic memory bank updated through InfoNCE-style interactions.

    The module keeps a compact queue of normal feature prototypes. For every
    update we construct a similarity distribution between incoming features and
    the current memory, and softly update the prototypes via a momentum rule.
    When the bank is empty the incoming features are simply normalised and
    stored. The design mimics the behaviour of reciprocal dual memory while
    avoiding the hard separation of normal/pseudo branches.
    """

    def __init__(
        self,
        feature_dim: int,
        max_items: int = 65536,
        temperature: float = 0.07,
        momentum: float = 0.2,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.max_items = max_items
        self.temperature = temperature
        self.momentum = momentum
        self.device = device or torch.device("cpu")

        self.register_buffer("_memory", torch.zeros(0, feature_dim), persistent=False)
        self._stats = ContrastiveMemoryStats(0, 0.0, temperature, momentum)

    @property
    def memory(self) -> torch.Tensor:
        return self._memory

    def reset(self) -> None:
        self._memory = torch.zeros(0, self.feature_dim, device=self.device)
        self._stats = ContrastiveMemoryStats(0, 0.0, self.temperature, self.momentum)

    @torch.no_grad()
    def fit(self, features: torch.Tensor) -> torch.Tensor:
        """Populate the memory bank with the provided features.

        Args:
            features: Tensor shaped ``(num_features, feature_dim)``.
        Returns:
            Tensor containing the contrastively updated prototypes.
        """

        if features.numel() == 0:
            self.reset()
            return self._memory

        feats = F.normalize(features.to(self.device), dim=-1)

        if self._memory.numel() == 0:
            self._memory = feats[-self.max_items :]
            self._stats = ContrastiveMemoryStats(
                bank_size=self._memory.shape[0],
                info_nce=0.0,
                temperature=self.temperature,
                momentum=self.momentum,
            )
            return self._memory

        memory = self._memory
        sim = feats @ memory.t() / max(self.temperature, 1e-6)
        log_prob = F.log_softmax(sim, dim=-1)
        probs = log_prob.exp()
        # InfoNCE with positive pairs being the diagonal soft-assignments.
        info_nce = -torch.mean(torch.diagonal(log_prob, dim1=-2, dim2=-1))

        # Momentum update for prototypes.
        updated = probs.t() @ feats
        updated = F.normalize(updated, dim=-1)
        memory = (1.0 - self.momentum) * memory + self.momentum * updated
        memory = F.normalize(memory, dim=-1)

        if memory.shape[0] < self.max_items and feats.shape[0] > memory.shape[0]:
            # Append residuals to increase diversity.
            residuals = feats - probs @ memory
            residuals = F.normalize(residuals, dim=-1)
            memory = torch.cat([memory, residuals], dim=0)

        if memory.shape[0] > self.max_items:
            memory = memory[: self.max_items]

        self._memory = memory
        self._stats = ContrastiveMemoryStats(
            bank_size=memory.shape[0],
            info_nce=float(info_nce.item()),
            temperature=self.temperature,
            momentum=self.momentum,
        )
        return self._memory

    def export_stats(self) -> ContrastiveMemoryStats:
        return self._stats


class DomainInvariantContrastiveAdapter(torch.nn.Module):
    """Light-weight transformer adapter performing domain alignment.

    The adapter consumes aggregated patch representations and refines them using
    a stack of transformer encoder layers. During training ``update_reference``
    is called to build a small reference bank of source-domain prototypes. At
    inference time the adapter computes a mean-shift towards the reference
    distribution scaled by the empirical Maximum Mean Discrepancy (MMD).
    """

    def __init__(
        self,
        feature_dim: int,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        temperature: float = 0.07,
        device: Optional[torch.device] = None,
        reference_max_items: int = 4096,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.device = device or torch.device("cpu")
        self.reference_max_items = reference_max_items

        hidden_dim = int(feature_dim * mlp_ratio)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=max(1, num_heads),
            dim_feedforward=max(feature_dim, hidden_dim),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=max(1, depth))

        self.register_buffer("_reference_bank", torch.zeros(0, feature_dim), persistent=False)

    @property
    def reference_bank(self) -> torch.Tensor:
        return self._reference_bank

    @torch.no_grad()
    def update_reference(self, features: torch.Tensor) -> None:
        if features.numel() == 0:
            return
        feats = F.normalize(features.to(self.device), dim=-1)
        if self._reference_bank.numel() == 0:
            self._reference_bank = feats[-self.reference_max_items :]
        else:
            bank = torch.cat([self._reference_bank, feats], dim=0)
            if bank.shape[0] > self.reference_max_items:
                bank = bank[-self.reference_max_items :]
            self._reference_bank = bank

    def _estimate_mmd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0 or y.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        bandwidth = max(self.temperature, 1e-6)
        xx = self._gaussian_kernel(x, x, bandwidth)
        yy = self._gaussian_kernel(y, y, bandwidth)
        xy = self._gaussian_kernel(x, y, bandwidth)
        return xx.mean() + yy.mean() - 2.0 * xy.mean()

    @staticmethod
    def _gaussian_kernel(x: torch.Tensor, y: torch.Tensor, bandwidth: float) -> torch.Tensor:
        x_norm = (x ** 2).sum(dim=1).view(-1, 1)
        y_norm = (y ** 2).sum(dim=1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * (x @ y.t())
        return torch.exp(-dist / (2.0 * bandwidth ** 2))

    def forward(
        self,
        features: torch.Tensor,
        *,
        reference_override: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        """Adapt features and return optional MMD diagnostic."""

        if features.ndim == 2:
            inputs = features.unsqueeze(1)
        else:
            inputs = features
        encoded = self.encoder(inputs)
        encoded = encoded.squeeze(1)

        reference = reference_override
        if reference is None:
            reference = self._reference_bank

        mmd_value = None
        if reference is not None and reference.numel() > 0:
            ref = reference
            mmd = self._estimate_mmd(encoded, ref)
            mmd_value = float(mmd.detach().cpu().item())
            mean_shift = (ref.mean(dim=0, keepdim=True) - encoded.mean(dim=0, keepdim=True))
            encoded = encoded + torch.tanh(mmd) * mean_shift

        return encoded, mmd_value