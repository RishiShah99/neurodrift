"""TopK sparse autoencoder over the frozen VAE content latent — Phase 5.

Interpretability head for the disentangled VAE. The VAE's content latent
`z = (B, C=16, d=32, d, d)` is dense and entangled; a TopK-SAE factors each
SPATIAL TOKEN (one C-vector per voxel of the latent grid) into a sparse,
overcomplete dictionary so individual features become human-readable and
steerable (the aging direction in scripts/sae_probe.py).

Architecture is the OpenAI recipe (Gao et al. 2024, "Scaling and evaluating
sparse autoencoders"):

  * a tied pre-bias `b_dec` subtracted before the encoder and added back after
    the decoder (centres the dictionary on the data mean);
  * a TopK activation that keeps EXACTLY `k` latents per token (hard sparsity, no
    L1 penalty — so the L0 is fixed and the recon/sparsity trade-off has no knob);
  * unit-norm decoder columns (so a feature's activation magnitude, not its
    column norm, carries its contribution) re-normalised after every step;
  * AuxK dead-latent revival: an auxiliary loss reconstructs the residual
    `e = x - x_hat` from the top-`aux_k` CURRENTLY-DEAD latents, which routes a
    gradient back into latents that TopK has stopped selecting and brings them
    back to life instead of letting the dictionary collapse.

References (cited; recipe + masking follow these):
  - openai/sparse_autoencoder — TopK + AuxK + tied pre-bias + unit-norm decoder.
  - jbloomAus/SAELens — encode/decode/forward contract, L0 == k bookkeeping.
  - saprmarks/dictionary_learning — resample/normalize dead-feature hooks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SAEOutput:
    """Forward bundle.

    x_hat:   (N, d_in) reconstruction of the input tokens.
    acts:    (N, d_hidden) sparse codes — exactly `k` nonzero entries per row.
    indices: (N, k) long indices of the kept latents per token.
    aux_loss: scalar AuxK dead-latent reconstruction loss (0 when no dead latents).
    """

    x_hat: torch.Tensor
    acts: torch.Tensor
    indices: torch.Tensor
    aux_loss: torch.Tensor


class TopKSAE(nn.Module):
    """TopK sparse autoencoder operating per spatial token of a VAE latent.

    Operates on TOKENS of dim `d_in` (one C-vector per latent voxel); the
    LitModule reshapes `z = (B, C, d, d, d)` into `(B*d^3, C)` before calling.
    """

    # Declared so mypy treats the registered buffer as a Tensor (register_buffer is
    # typed Tensor | Module in torch's stubs, which breaks .sum()/comparisons).
    steps_since_fired: torch.Tensor

    def __init__(
        self,
        d_in: int = 16,
        d_hidden: int = 8192,
        k: int = 32,
        aux_k: int = 256,
        aux_coef: float = 1.0 / 32.0,
        dead_steps_threshold: int = 1000,
    ) -> None:
        super().__init__()
        if k > d_hidden:
            raise ValueError(f"k={k} cannot exceed d_hidden={d_hidden}")
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.k = k
        # AuxK can revive at most as many latents as exist; never request more dead
        # latents than the dictionary holds (tiny-test dictionaries hit this).
        self.aux_k = min(aux_k, d_hidden)
        self.aux_coef = aux_coef
        self.dead_steps_threshold = dead_steps_threshold

        self.encoder = nn.Linear(d_in, d_hidden)
        # No decoder bias: the tied pre-bias `b_dec` IS the decoder bias (added back
        # in decode). A separate decoder bias would double-count the data mean.
        self.decoder = nn.Linear(d_hidden, d_in, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Steps since each latent last fired (> dead_steps_threshold == dead). Buffer
        # so it survives checkpoint/resume. Long to count steps exactly.
        self.register_buffer(
            "steps_since_fired", torch.zeros(d_hidden, dtype=torch.long), persistent=True
        )

        self._init_weights()

    # -- initialisation -----------------------------------------------------
    def _init_weights(self) -> None:
        # Tied init (the OpenAI recipe): the encoder is the decoder's transpose at
        # start, then they decouple under training. Encoder bias 0 so the first
        # pre-activation is purely the centred projection.
        nn.init.kaiming_uniform_(self.decoder.weight)
        with torch.no_grad():
            self.normalize_decoder()
            self.encoder.weight.copy_(self.decoder.weight.t())
            self.encoder.bias.zero_()

    # -- encode / decode ----------------------------------------------------
    def _topk(self, pre_acts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep the top-k entries per row, zero the rest. Returns (acts, indices).

        ReLU AFTER selection so a kept-but-negative pre-activation contributes
        zero (the standard TopK-SAE: nonnegative codes), while the per-row L0 stays
        <= k. With centred, trained features the kept set is overwhelmingly
        positive, so L0 == k holds in practice and the LitModule logs k directly.
        """
        topk = pre_acts.topk(self.k, dim=-1)
        values = topk.values.relu()
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, topk.indices, values)
        return acts, topk.indices

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokens (N, d_in) -> (sparse acts (N, d_hidden), topk indices (N, k))."""
        pre_acts = self.encoder(x - self.b_dec)
        return self._topk(pre_acts)

    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        """Dense pre-activation (N, d_hidden) before TopK — used by the AuxK path."""
        out: torch.Tensor = self.encoder(x - self.b_dec)
        return out

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """Sparse codes (N, d_hidden) -> reconstruction (N, d_in)."""
        decoded: torch.Tensor = self.decoder(acts)
        return decoded + self.b_dec

    # -- AuxK dead-latent revival ------------------------------------------
    @property
    def dead_mask(self) -> torch.Tensor:
        """Bool (d_hidden,): latents that have not fired in > dead_steps_threshold steps."""
        return self.steps_since_fired > self.dead_steps_threshold

    def _aux_loss(
        self, x: torch.Tensor, x_hat: torch.Tensor, pre_acts: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct the residual e = x - x_hat from the top-aux_k DEAD latents.

        The OpenAI AuxK term: route a gradient into latents TopK has abandoned by
        asking ONLY them to explain what the main reconstruction missed. Dead
        latents are masked IN (everything else set to -inf so topk can't pick a
        live latent), then their pre-activations reconstruct the residual. Zero
        when there are no dead latents (nothing to revive).
        """
        dead = self.dead_mask
        n_dead = int(dead.sum())
        if n_dead == 0:
            return x.new_zeros(())
        kth = min(self.aux_k, n_dead)
        # -inf on live latents so topk selects only dead ones; relu for nonneg codes.
        masked = pre_acts.masked_fill(~dead, float("-inf"))
        topk = masked.topk(kth, dim=-1)
        aux_acts = torch.zeros_like(pre_acts)
        aux_acts.scatter_(-1, topk.indices, topk.values.relu())
        # Decode WITHOUT the pre-bias: this models the residual e directly (the
        # bias already lives in x_hat), so e_hat = W_dec @ aux_acts.
        e_hat: torch.Tensor = self.decoder(aux_acts)
        residual = (x - x_hat).detach()
        return (e_hat - residual).pow(2).mean()

    # -- forward ------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> SAEOutput:
        pre_acts = self.pre_acts(x)
        acts, indices = self._topk(pre_acts)
        x_hat = self.decode(acts)
        aux_loss = self.aux_coef * self._aux_loss(x, x_hat, pre_acts)
        return SAEOutput(x_hat=x_hat, acts=acts, indices=indices, aux_loss=aux_loss)

    # -- maintenance hooks (LitModule calls these each step) ----------------
    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Rescale every decoder column (a feature's dictionary vector) to unit L2.

        Keeps the feature magnitude in `acts`, not in the column norm, so feature
        activations are comparable across latents and the AuxK residual decode is
        well-scaled. Called after every optimizer step.
        """
        norm = self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.decoder.weight.div_(norm)

    @torch.no_grad()
    def update_dead_tracker(self, indices: torch.Tensor) -> None:
        """Advance the dead-latent clock: reset fired latents to 0, increment the rest.

        `indices` are the TopK selections for the batch (N, k); any latent that
        appears fired this step. Mirrors the OpenAI bookkeeping (steps_since_fired).
        """
        fired = torch.zeros(self.d_hidden, dtype=torch.bool, device=indices.device)
        fired[indices.reshape(-1)] = True
        self.steps_since_fired += 1
        self.steps_since_fired[fired] = 0

    @torch.no_grad()
    def resample_dead(self, x: torch.Tensor) -> int:
        """Re-seed dead latents toward poorly-reconstructed inputs (resample hook).

        The dictionary_learning-style alternative to AuxK: point each dead latent's
        encoder/decoder row at a high-residual example so it has signal to learn
        from, and clear its dead clock. AuxK handles revival during normal training;
        this is an explicit escape hatch for a fully-collapsed dictionary. Returns
        the number of latents resampled.
        """
        dead = self.dead_mask
        n_dead = int(dead.sum())
        if n_dead == 0 or x.numel() == 0:
            return 0
        out = self.forward(x)
        residual = x - out.x_hat
        err = residual.pow(2).sum(dim=-1)
        n = min(n_dead, x.shape[0])
        worst = err.topk(n).indices
        dead_idx = dead.nonzero(as_tuple=False).flatten()[:n]
        directions = residual[worst]
        unit = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.decoder.weight[:, dead_idx] = unit.t()
        self.encoder.weight[dead_idx] = unit
        self.encoder.bias[dead_idx] = 0.0
        self.steps_since_fired[dead_idx] = 0
        return int(n)
