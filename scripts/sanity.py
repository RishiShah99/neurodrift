"""Spot-resume sanity script.

Trains a tiny 3D CNN on dummy data for `--steps` iterations, checkpointing to
`~/out/sanity/latest.ckpt` every `--ckpt-every` steps. `--resume` picks up the
latest checkpoint and continues.

Run via fleet:

    fleet up h100
    fleet sync
    fleet train "python scripts/sanity.py --steps 100"
    fleet logs -f
    fleet pull
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG = logging.getLogger("neurodrift.sanity")


class TinyCNN3D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NeuroDrift fleet sanity script")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--out",
        type=Path,
        default=Path(os.environ.get("FLEET_OUT", str(Path.home() / "out"))) / "sanity",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out / "latest.ckpt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG.info("device=%s out=%s", device, args.out)

    model = TinyCNN3D().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_step = 0
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optim"])
        start_step = int(ck["step"])
        LOG.info("resumed from step=%d", start_step)

    torch.manual_seed(1337)
    t0 = time.perf_counter()
    for step in range(start_step, args.steps):
        x = torch.randn(args.batch, 1, args.size, args.size, args.size, device=device)
        y = x.flip(dims=(-1,))
        pred = model(x)
        loss = F.l1_loss(pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % 10 == 0:
            LOG.info("step=%d loss=%.4f elapsed=%.1fs", step, loss.item(), time.perf_counter() - t0)

        if (step + 1) % args.ckpt_every == 0:
            torch.save(
                {"model": model.state_dict(), "optim": optim.state_dict(), "step": step + 1},
                ckpt_path,
            )
            LOG.info("ckpt step=%d → %s", step + 1, ckpt_path)

    torch.save(
        {"model": model.state_dict(), "optim": optim.state_dict(), "step": args.steps},
        ckpt_path,
    )
    LOG.info("done. final loss=%.4f total=%.1fs", loss.item(), time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
