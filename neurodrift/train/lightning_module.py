"""LightningModule wrapping the v0 Phase 1 VAE training loop.

Loss = masked L1 (reconstruct every present-in-ground-truth modality)
     + β·KL (linear warmup so the encoder learns recon before regularizing)
     + λ_p · perceptual (tri-orthogonal VGG19 on axial / coronal / sagittal slices)
     + λ_c · cross-modal cycle (re-encoding recon with a different mask
                                 must land in the same latent neighbourhood)

Per-cohort PSNR is logged separately so a single weak cohort never gets buried
under the pooled average.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn

from neurodrift.models.vae3d import VAE3D, DisentangledVAE3D, PatchDiscriminator3D


def _masked_l1(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 averaged only over modality slots where `mask` is 1.

    `recon`, `target`: (B, M, D, H, W); `mask`: (B, M).
    """
    err = (recon - target).abs().mean(dim=(2, 3, 4))
    denom = mask.sum().clamp_min(1.0)
    return (err * mask).sum() / denom


def _kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def _skip_step_if_nonfinite(module: L.LightningModule, optimizer: Any) -> bool:
    """Zero the grads (making the optimizer step a no-op) if ANY are non-finite.

    A single NaN/Inf gradient on one rank is all-reduced into EVERY rank's grads and
    poisons all weights. By the time this runs, backward and the DDP all-reduce have
    already completed in lockstep, so every rank sees the same (post-reduce) grads
    and makes the same skip decision — no NCCL desync (unlike returning None from
    training_step, which would skip backward on only one rank and hang the others).
    One host sync (the finiteness check), not the per-step loss.item() it replaces.
    Returns True when the step was skipped.
    """
    sq = [
        p.grad.detach().pow(2).sum()
        for p in module.parameters()
        if p.requires_grad and p.grad is not None
    ]
    if not sq:
        return False
    if bool(torch.isfinite(torch.stack(sq).sum())):
        return False
    optimizer.zero_grad(set_to_none=True)
    return True


class _TriOrthoVGGPerceptual(nn.Module):
    """Tri-orthogonal mid-slice VGG19 feature distance.

    Cheap perceptual loss that works without a med-imaging-specific backbone.
    Swap in Med3D-VGG once weights are downloaded by setting `model` to
    something with the same `features` interface.

    `weights` is passed straight to `torchvision.models.vgg19`. It MUST be the
    pretrained tag ("DEFAULT"/"IMAGENET1K_V1") for this to be a real perceptual
    signal: an untrained VGG (`weights=None`) is ~a random projection, so the term
    silently degrades to noise — a high-impact bug for a model whose whole job is to
    beat a blur ceiling. `None` is retained only for tests (no 550 MB download).
    """

    def __init__(self, weights: str | None = "DEFAULT") -> None:
        super().__init__()
        try:
            from torchvision import models
        except ImportError as err:
            raise RuntimeError("torchvision required for perceptual loss") from err
        try:
            vgg = models.vgg19(weights=weights)
        except Exception as err:  # offline box: download of the pretrained tag failed
            if weights is None:
                raise
            import warnings

            warnings.warn(
                f"vgg19(weights={weights!r}) failed to load ({err}); falling back to "
                "RANDOM features. The perceptual loss is now ~meaningless — pre-download "
                'the weights (python -c "import torchvision; '
                "torchvision.models.vgg19(weights='DEFAULT')\") or set use_perceptual=false.",
                RuntimeWarning,
                stacklevel=2,
            )
            vgg = models.vgg19(weights=None)
        self.features = vgg.features[:16]
        for p in self.features.parameters():
            p.requires_grad = False
        self.features.eval()

    def _to_rgb_slice(self, vol: torch.Tensor, axis: int) -> torch.Tensor:
        # NOT @torch.no_grad(): this is the only path from `recon` into the VGG
        # features, so detaching it here makes the perceptual term a constant with
        # zero gradient. VGG weights are already frozen via requires_grad=False on
        # self.features.parameters(), so the gradient flows to recon but not VGG.
        mid = vol.shape[axis + 2] // 2
        if axis == 0:
            s = vol[:, :, mid, :, :]
        elif axis == 1:
            s = vol[:, :, :, mid, :]
        else:
            s = vol[:, :, :, :, mid]
        s = s.repeat(1, 3 // s.shape[1] + 1, 1, 1)[:, :3]
        s = (s - s.amin(dim=(2, 3), keepdim=True)) / (
            s.amax(dim=(2, 3), keepdim=True) - s.amin(dim=(2, 3), keepdim=True) + 1e-6
        )
        return s

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = recon.new_zeros(())
        for axis in (0, 1, 2):
            s_r = self._to_rgb_slice(recon, axis)
            s_t = self._to_rgb_slice(target, axis)
            f_r = self.features(s_r)
            f_t = self.features(s_t)
            loss = loss + F.l1_loss(f_r, f_t)
        return loss / 3.0


class VAELitModule(L.LightningModule):
    """v0 Phase 1: VAE with masked L1 + KL + perceptual + cross-modal cycle loss."""

    def __init__(
        self,
        model: VAE3D,
        optimizer_partial: Any,
        scheduler_partial: Any | None = None,
        kl_warmup_steps: int = 5000,
        perceptual_weight: float = 0.1,
        cycle_weight: float = 0.5,
        use_perceptual: bool = True,
        perceptual_weights: str | None = "DEFAULT",
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer_partial = optimizer_partial
        self.scheduler_partial = scheduler_partial
        self.kl_warmup_steps = kl_warmup_steps
        self.perceptual_weight = perceptual_weight
        self.cycle_weight = cycle_weight
        self.use_perceptual = use_perceptual
        self.perceptual: nn.Module | None = (
            _TriOrthoVGGPerceptual(weights=perceptual_weights) if use_perceptual else None
        )

    def forward(self, x: torch.Tensor, modality_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(x, modality_mask).recon

    def _kl_beta(self) -> float:
        if self.kl_warmup_steps <= 0:
            return 1.0
        frac = min(1.0, self.global_step / float(self.kl_warmup_steps))
        return frac

    def _cycle_loss(
        self,
        recon: torch.Tensor,
        mu: torch.Tensor,
        present_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode recon under a different mask; match its mu to original mu (detached)."""
        b, m = present_mask.shape
        alt_mask = present_mask.clone()
        if m > 1:
            for i in range(b):
                present_idx = present_mask[i].nonzero(as_tuple=False).flatten()
                if present_idx.numel() < 2:
                    continue
                drop = present_idx[torch.randint(present_idx.numel(), (1,))].item()
                alt_mask[i, drop] = 0.0
        if alt_mask.sum() == 0:
            return mu.new_zeros(())
        recon_masked = recon * alt_mask.view(b, m, 1, 1, 1)
        mu2, _ = self.model.encode(recon_masked, alt_mask)
        return F.l1_loss(mu2, mu.detach())

    def _log_per_cohort_psnr(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        present_mask: torch.Tensor,
        cohorts: list[str],
        stage: str,
    ) -> None:
        b, m = present_mask.shape
        # Vectorize PSNR per (sample, modality) and do a SINGLE device->host copy,
        # instead of B*M `.item()` syncs (each serializes the GPU step).
        mse = (recon - target).pow(2).mean(dim=(2, 3, 4))  # (B, M)
        data_range = target.abs().amax(dim=(2, 3, 4)).clamp_min(1e-6)  # (B, M)
        psnr = 10.0 * torch.log10(data_range.pow(2) / mse.clamp_min(1e-12))  # (B, M)
        psnr_cpu = psnr.detach().cpu()
        present_cpu = present_mask.detach().bool().cpu()
        per_cohort: dict[str, list[torch.Tensor]] = defaultdict(list)
        for i in range(b):
            for j in range(m):
                if present_cpu[i, j]:
                    per_cohort[cohorts[i]].append(psnr_cpu[i, j])
        for cohort, vals in per_cohort.items():
            mean_psnr = torch.stack(vals).mean()
            self.log(
                f"{stage}/psnr_{cohort}",
                mean_psnr,
                on_step=False,
                on_epoch=True,
                batch_size=b,
            )

    def _step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        x = batch["image"]
        # `target` is the clean, never-dropped volume. Reconstructing against it
        # (not the zero-filled input `x`) is what turns modality dropout into a
        # cross-modal synthesis objective. Falls back to `x` for any caller that
        # predates the target split.
        target = batch.get("target", x)
        modality_mask = batch["modality_mask"]
        present_mask = batch["present_mask"]
        cohorts = batch["cohort"]

        out = self.model(x, modality_mask)

        recon_loss = _masked_l1(out.recon, target, present_mask)
        kl = _kl_divergence(out.mu, out.logvar)
        beta = self._kl_beta()
        loss = recon_loss + beta * self.model.beta_kl * kl

        perc_loss = x.new_zeros(())
        if self.perceptual is not None and stage == "train":
            # Zero recon's never-acquired slots so they match `target` (also zero
            # there) and contribute no spurious perceptual gradient.
            recon_perc = out.recon * present_mask.view(*present_mask.shape, 1, 1, 1)
            perc_loss = self.perceptual(recon_perc, target)
            loss = loss + self.perceptual_weight * perc_loss

        cycle = x.new_zeros(())
        if self.cycle_weight > 0 and stage == "train":
            cycle = self._cycle_loss(out.recon, out.mu, present_mask)
            loss = loss + self.cycle_weight * cycle

        bs = x.shape[0]
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/recon_l1", recon_loss, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/kl", kl, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/kl_beta", beta, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/perceptual", perc_loss, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/cycle", cycle, on_epoch=True, batch_size=bs)

        # No NaN gate here: a non-finite step is caught in on_before_optimizer_step
        # (which zeroes the grads), and gating PSNR on loss.item() would add a
        # per-step host sync. A rare NaN batch only pollutes that epoch's PSNR mean.
        self._log_per_cohort_psnr(out.recon, target, present_mask, cohorts, stage)
        return loss

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        _skip_step_if_nonfinite(self, optimizer)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self) -> Any:
        optimizer = self.optimizer_partial(self.model.parameters())
        if self.scheduler_partial is None:
            return optimizer
        scheduler = self.scheduler_partial(optimizer)
        # The configured T_max is a guess; the true step count depends on corpus
        # size, devices, batch size and accumulation — all of which have churned
        # on this project. Pin the cosine half-period to the actual planned step
        # count so the LR genuinely anneals to eta_min instead of stalling halfway.
        total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if total_steps and hasattr(scheduler, "T_max"):
            scheduler.T_max = int(total_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


# ---------------------------------------------------------------------------
# Content/style disentangled VAE + adversarial training (v0 improvement E1+E2)
# ---------------------------------------------------------------------------


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def _masked_mean(per_sample: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average a per-sample (B,) loss over the samples where mask==1 (no host sync)."""
    return (per_sample * mask).sum() / mask.sum().clamp_min(1.0)


class DisentangledVAELitModule(L.LightningModule):
    """Cross-modal synthesis VAE with a content/style split + SN-PatchGAN.

    Trained from the CLEAN `target` (every acquired modality at full fidelity), so
    the cross-modal objective is explicit and dropout-independent: for each present
    source modality `j` we (a) self-reconstruct it from its own content+style and
    (b) translate it into every other present modality `k` using `j`'s content and
    `k`'s learned prototype style — which is exactly the inference path the eval
    scores. Disentanglement is held together by a content-invariance term (the
    per-modality content codes of one subject must agree) and a style-cycle term
    (the decoder must actually consume the style it was handed).

    Uses MANUAL optimization (two optimizers: generator = VAE, discriminator). The
    adversarial weight is zero for `adv_start_step` then linearly ramps over
    `adv_warmup_steps` — the generator needs a head start or the 3D discriminator
    wins early and the generator's gradient vanishes.
    """

    def __init__(
        self,
        model: DisentangledVAE3D,
        optimizer_partial: Any,
        scheduler_partial: Any | None = None,
        kl_warmup_steps: int = 5000,
        recon_weight: float = 10.0,
        cross_weight: float = 10.0,
        content_invariance_weight: float = 2.0,
        style_cycle_weight: float = 1.0,
        kl_style_weight: float = 0.01,
        perceptual_weight: float = 1.0,
        adversarial_weight: float = 0.5,
        adv_start_step: int = 8000,
        adv_warmup_steps: int = 8000,
        disc_lr: float = 2.0e-4,
        disc_base_channels: int = 32,
        use_perceptual: bool = True,
        use_adversarial: bool = True,
        perceptual_weights: str | None = "DEFAULT",
    ) -> None:
        super().__init__()
        # Automatic optimization (Lightning drives backward/step/clip/sched) when
        # there's no GAN — the rock-solid DDP path the v0 cook used. Manual
        # optimization ONLY for the adversarial case, where two optimizers must
        # alternate. The recon/disentangled objective needs neither, and forcing
        # manual-opt + a discriminator + find_unused on the no-GAN path was the
        # source of an intermittent DDP deadlock, so keep them decoupled.
        self.automatic_optimization = not use_adversarial
        self.model = model
        self.optimizer_partial = optimizer_partial
        self.scheduler_partial = scheduler_partial
        self.kl_warmup_steps = kl_warmup_steps
        self.recon_weight = recon_weight
        self.cross_weight = cross_weight
        self.content_invariance_weight = content_invariance_weight
        self.style_cycle_weight = style_cycle_weight
        self.kl_style_weight = kl_style_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
        self.adv_start_step = adv_start_step
        self.adv_warmup_steps = adv_warmup_steps
        self.disc_lr = disc_lr
        self.use_perceptual = use_perceptual
        self.use_adversarial = use_adversarial
        self.perceptual: nn.Module | None = (
            _TriOrthoVGGPerceptual(weights=perceptual_weights) if use_perceptual else None
        )
        self.discriminator: PatchDiscriminator3D | None = (
            PatchDiscriminator3D(model.num_modalities, base_channels=disc_base_channels)
            if use_adversarial
            else None
        )
        # Own step counter (manual opt steps two optimizers, so global_step is an
        # unreliable clock for the ramps). Buffered so it survives spot-resume.
        self.register_buffer("_step_count", torch.zeros((), dtype=torch.long), persistent=True)

    def forward(self, x: torch.Tensor, modality_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(x, modality_mask).recon

    # -- schedule helpers ---------------------------------------------------
    def _kl_beta(self) -> float:
        if self.kl_warmup_steps <= 0:
            return 1.0
        return min(1.0, float(self._step_count.item()) / float(self.kl_warmup_steps))

    def _adv_weight(self) -> float:
        if not self.use_adversarial:
            return 0.0
        step = float(self._step_count.item())
        if step < self.adv_start_step:
            return 0.0
        if self.adv_warmup_steps <= 0:
            return self.adversarial_weight
        frac = (step - self.adv_start_step) / float(self.adv_warmup_steps)
        return self.adversarial_weight * min(1.0, max(0.0, frac))

    # -- masked term helpers ------------------------------------------------
    def _vol_l1(self, rec: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        err = (rec - tgt).abs().flatten(1).mean(dim=1)  # (B,)
        return _masked_mean(err, mask)

    def _kl_content(
        self, mu: torch.Tensor, logvar: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).flatten(1).mean(dim=1)  # (B,)
        return _masked_mean(kl, mask)

    def _perc(self, rec: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.perceptual is None:
            return rec.new_zeros(())
        m = mask.view(-1, 1, 1, 1, 1)
        # Zeroing absent samples makes their (normalized) slices identical -> ~0
        # perceptual contribution, so we avoid a per-sample host sync to skip them.
        return self.perceptual(rec * m, tgt * m)

    def _hinge_g(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        per = -logits.flatten(1).mean(dim=1)  # G wants D(fake) high
        return _masked_mean(per, mask)

    def _hinge_d(
        self, real_logits: torch.Tensor, fake_logits: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        rl = F.relu(1.0 - real_logits).flatten(1).mean(dim=1)
        fl = F.relu(1.0 + fake_logits).flatten(1).mean(dim=1)
        return _masked_mean(rl + fl, mask)

    # -- generator forward: all disentanglement losses + cached fakes -------
    def _generator_losses(
        self, target: torch.Tensor, present: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor | float],
        list[tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]],
    ]:
        b, m = present.shape
        enc = self.model.encode_all(target, present)
        zs = [
            self.model.reparameterize(enc.content_mu[:, j], enc.content_logvar[:, j])
            for j in range(m)
        ]
        beta = self._kl_beta()
        adv_w = self._adv_weight()

        recon_l = target.new_zeros(())
        cross_l = target.new_zeros(())
        kl_c = target.new_zeros(())
        kl_s = target.new_zeros(())
        inv_l = target.new_zeros(())
        cyc_l = target.new_zeros(())
        perc_l = target.new_zeros(())
        g_adv = target.new_zeros(())
        # (fake_detached, real_target, modality_idx, sample_mask) for the D step
        fakes: list[tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]] = []

        for j in range(m):
            pj = present[:, j]
            tgt_j = target[:, j : j + 1]
            # self-reconstruction with the modality's own (encoded) style
            rec_jj = self.model.decode_one(zs[j], enc.style[:, j])
            recon_l = recon_l + self._vol_l1(rec_jj, tgt_j, pj)
            kl_c = kl_c + self._kl_content(enc.content_mu[:, j], enc.content_logvar[:, j], pj)
            kl_s = kl_s + _masked_mean(enc.style[:, j].pow(2).mean(dim=1), pj)
            if self.perceptual is not None:
                perc_l = perc_l + self._perc(rec_jj, tgt_j, pj)

        for j in range(m):
            for k in range(m):
                if k == j:
                    continue
                pjk = present[:, j] * present[:, k]
                tgt_k = target[:, k : k + 1]
                # translate j -> k using k's PROTOTYPE style (the inference path)
                proto_k = self.model.style_prototype(k, b, target.device)
                fake_k = self.model.decode_one(zs[j], proto_k)
                cross_l = cross_l + self._vol_l1(fake_k, tgt_k, pjk)
                if self.perceptual is not None:
                    perc_l = perc_l + self._perc(fake_k, tgt_k, pjk)
                # style-cycle: the synthesized k must read back as prototype-k style
                if self.style_cycle_weight > 0:
                    s_rec = self.model.encode_style_one(fake_k)
                    cyc = (s_rec - proto_k).abs().mean(dim=1)
                    cyc_l = cyc_l + _masked_mean(cyc, pjk)
                if adv_w > 0 and self.discriminator is not None:
                    g_adv = g_adv + self._hinge_g(self.discriminator(fake_k, k), pjk)
                fakes.append((fake_k.detach(), tgt_k, k, pjk))

        # content-invariance: per-modality content means must agree for a subject.
        # L2 in latent space, one branch detached (lower-variance pull to consensus).
        for j in range(m):
            for k in range(j + 1, m):
                pjk = present[:, j] * present[:, k]
                diff = (
                    (enc.content_mu[:, j] - enc.content_mu[:, k].detach())
                    .pow(2)
                    .flatten(1)
                    .mean(dim=1)
                )
                inv_l = inv_l + _masked_mean(diff, pjk)

        kl_term = beta * self.model.beta_kl * kl_c + self.kl_style_weight * kl_s
        g_total = (
            self.recon_weight * recon_l
            + self.cross_weight * cross_l
            + self.content_invariance_weight * inv_l
            + self.style_cycle_weight * cyc_l
            + self.perceptual_weight * perc_l
            + kl_term
            + adv_w * g_adv
        )
        logs: dict[str, torch.Tensor | float] = {
            "recon_l1": recon_l.detach(),
            "cross_l1": cross_l.detach(),
            "kl_content": kl_c.detach(),
            "kl_style": kl_s.detach(),
            "content_inv": inv_l.detach(),
            "style_cycle": cyc_l.detach(),
            "perceptual": perc_l.detach(),
            "g_adv": g_adv.detach(),
            # Plain Python floats, not torch.tensor(...): a bare CPU scalar tensor
            # logged with on_epoch reduction in a CUDA DDP run triggers a device
            # mismatch / needless sync. Lightning moves floats to the right device.
            "kl_beta": beta,
            "adv_weight": adv_w,
        }
        return g_total, logs, fakes

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor | None:
        target = batch.get("target", batch["image"])
        present = batch["present_mask"]

        # --- automatic optimization (no GAN): just return the loss ---
        if self.automatic_optimization:
            g_total, logs, _ = self._generator_losses(target, present)
            self._step_count += 1
            bs = target.shape[0]
            self.log(
                "train/g_total", g_total, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs
            )
            for name, val in logs.items():
                self.log(f"train/{name}", val, on_step=False, on_epoch=True, batch_size=bs)
            return g_total

        # --- manual optimization (GAN): two optimizers, alternating ---
        opts = self.optimizers()
        opt_g, opt_d = (opts[0], opts[1]) if isinstance(opts, (list, tuple)) else (opts, None)

        # --- generator step (discriminator frozen so DDP expects no D grads) ---
        if self.discriminator is not None:
            _set_requires_grad(self.discriminator, False)
        _set_requires_grad(self.model, True)
        g_total, logs, fakes = self._generator_losses(target, present)
        opt_g.zero_grad(set_to_none=True)
        self.manual_backward(g_total)
        # Manual opt: Lightning does NOT call on_before_optimizer_step, so guard the
        # step inline. Skip (grads already zeroed) if non-finite so a NaN can't poison
        # the generator. Check before clipping, which would mangle an inf norm to 0.
        if not _skip_step_if_nonfinite(self, opt_g):
            self.clip_gradients(opt_g, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            opt_g.step()

        # --- discriminator step (generator frozen; fakes already detached) ---
        d_loss = target.new_zeros(())
        if opt_d is not None and self.discriminator is not None and self._adv_weight() > 0:
            _set_requires_grad(self.model, False)
            _set_requires_grad(self.discriminator, True)
            for fake_k, real_k, k, pjk in fakes:
                real_logits = self.discriminator(real_k, k)
                fake_logits = self.discriminator(fake_k, k)
                d_loss = d_loss + self._hinge_d(real_logits, fake_logits, pjk)
            d_loss = d_loss / max(1, len(fakes))
            opt_d.zero_grad(set_to_none=True)
            self.manual_backward(d_loss)
            if not _skip_step_if_nonfinite(self, opt_d):
                self.clip_gradients(opt_d, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
                opt_d.step()
            _set_requires_grad(self.model, True)

        # step the generator LR schedule (per optimizer step)
        sched = self.lr_schedulers()
        if sched is not None:
            sched_g = sched[0] if isinstance(sched, (list, tuple)) else sched
            sched_g.step()

        self._step_count += 1
        bs = target.shape[0]
        self.log(
            "train/g_total", g_total, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs
        )
        self.log("train/d_loss", d_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
        for name, val in logs.items():
            self.log(f"train/{name}", val, on_step=False, on_epoch=True, batch_size=bs)

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        # Fires only in automatic optimization (the no-GAN flagship). The manual GAN
        # path guards its steps inline in training_step instead.
        _skip_step_if_nonfinite(self, optimizer)

    @torch.no_grad()
    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        target = batch.get("target", batch["image"])
        present = batch["present_mask"]
        # self-recon loss (all present modalities fed) as the monitored val signal
        out = self.model(target * present.view(*present.shape, 1, 1, 1), present)
        recon = _masked_l1(out.recon, target, present)
        self.log("val/loss", recon, prog_bar=True, on_epoch=True, batch_size=target.shape[0])
        self._log_xmodal_psnr(target, present)

    @torch.no_grad()
    def _log_xmodal_psnr(self, target: torch.Tensor, present: torch.Tensor) -> None:
        """Live cross-modal synthesis PSNR per (src->dst) — the headline capability.

        Validation-only (extra decodes are too costly per train step) and sync-light:
        one device->host copy at the end, no per-sample `.item()`. Per-rank-local
        under DDP; scripts/eval.py is the canonical single-process number.
        """
        b, m = present.shape
        for src in range(m):
            for dst in range(m):
                if src == dst:
                    continue
                fake = self.model.translate(target[:, src : src + 1], src, dst)
                mse = (fake[:, 0] - target[:, dst]).pow(2).flatten(1).mean(dim=1)
                dr = target[:, dst].flatten(1).amax(dim=1).clamp_min(1e-6)
                psnr = 10.0 * torch.log10(dr.pow(2) / mse.clamp_min(1e-12))  # (B,)
                sel = present[:, src] * present[:, dst]
                pooled = (psnr * sel).sum() / sel.sum().clamp_min(1.0)
                self.log(
                    f"val/xpsnr_{self.model.modalities[src]}_to_{self.model.modalities[dst]}",
                    pooled,
                    on_epoch=True,
                    batch_size=b,
                )

    def _build_scheduler(self, opt_g: Any) -> Any | None:
        if self.scheduler_partial is None:
            return None
        scheduler = self.scheduler_partial(opt_g)
        total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if total_steps and hasattr(scheduler, "T_max"):
            scheduler.T_max = int(total_steps)
        return scheduler

    def configure_optimizers(self) -> Any:
        opt_g = self.optimizer_partial(self.model.parameters())

        # --- automatic optimization (no GAN): standard optimizer + scheduler dict.
        # Lightning steps the optimizer/scheduler and clips grads (Trainer config). ---
        if self.automatic_optimization:
            scheduler = self._build_scheduler(opt_g)
            if scheduler is None:
                return opt_g
            return {
                "optimizer": opt_g,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

        # --- manual optimization (GAN): two optimizers; we step the scheduler
        # ourselves in training_step (Lightning won't auto-step in manual mode). ---
        optimizers: list[Any] = [opt_g]
        if self.discriminator is not None:
            opt_d = torch.optim.Adam(
                self.discriminator.parameters(), lr=self.disc_lr, betas=(0.5, 0.9)
            )
            optimizers.append(opt_d)
        scheduler = self._build_scheduler(opt_g)
        if scheduler is None:
            return optimizers
        return optimizers, [scheduler]
