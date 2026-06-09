"""Modal-hosted NeuroDrift inference endpoint.

Phase 0 stub. The `infer` endpoint accepts a NIfTI blob plus conditioning vars
and returns a mock trajectory. Phase 4/6 will replace the body with the real
1-step student + 3DGS decoder forward pass.

Deploy:
    modal deploy apps/api/modal_app.py
"""

from __future__ import annotations

from typing import Any, Literal

import modal

# ---- image -------------------------------------------------------------------

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi>=0.115",
    "pydantic>=2.7",
    "numpy>=1.26",
    "nibabel>=5.2",
)

app = modal.App("neurodrift-api", image=image)

# A10G is fine for the stub; flip to H100 in Phase 4 once the real model lands.
GPU = "A10G"


# ---- schemas -----------------------------------------------------------------


Treatment = Literal["placebo", "lecanemab", "donanemab", "anti_tau", "anti_inflammatory", "glp1"]


# ---- functions ---------------------------------------------------------------


@app.function(gpu=GPU, timeout=180, secrets=[modal.Secret.from_name("neurodrift-secrets")])
def infer(
    nifti_bytes: bytes | None,
    age_target: float,
    apoe: str,
    treatment: Treatment,
    samples: int = 10,
) -> dict[str, Any]:
    """Return a mock per-region trajectory keyed off conditioning vars.

    The real Phase 4 implementation will:
    1. Decode `nifti_bytes` with nibabel, normalize, run the VAE encoder.
    2. Integrate the 1-step student from `age_now` → `age_target` under the
       conditioning vector.
    3. Decode latents via the hierarchical 3DGS decoder.
    4. Serialize Gaussians to a streaming `.ksplat` binary in R2.
    """
    import numpy as np

    age_now = 60.0
    n = 24
    ages = np.linspace(age_now, age_target, n)
    rate = {"E2E2": 0.4, "E2E3": 0.6, "E3E3": 1.0, "E3E4": 1.5, "E4E4": 2.2}.get(apoe, 1.0)
    bend = {
        "placebo": 0.0,
        "lecanemab": 0.35,
        "donanemab": 0.4,
        "anti_tau": 0.2,
        "anti_inflammatory": 0.1,
        "glp1": 0.15,
    }.get(treatment, 0.0)

    regions: list[dict[str, Any]] = []
    envelope: list[dict[str, Any]] = []
    for idx, region in enumerate(["hippocampus", "ventricles", "entorhinal_cortex", "wmh"]):
        base = [4.2, 20.0, 3.5, 5.0][idx]
        direction = 1 if region in {"ventricles", "wmh"} else -1
        values = (base + direction * rate * (1 - bend) * 0.015 * (ages - age_now)).tolist()
        regions.append({"region": region, "ages": ages.tolist(), "values": values})
        envelope.append(
            {
                "region": region,
                "ages": ages.tolist(),
                "lower": [v * 0.97 for v in values],
                "upper": [v * 1.03 for v in values],
            }
        )

    return {
        "gaussians_url": None,
        "trajectory": {"age_now": age_now, "age_target": age_target},
        "regions": regions,
        "envelope": envelope,
        "samples": samples,
    }


# ---- HTTP entry --------------------------------------------------------------


@app.function(image=image, timeout=180)
@modal.asgi_app()
def fastapi_app():  # type: ignore[no-untyped-def]
    """Mount a FastAPI app with CORS so the Next.js dev server can call it."""
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.middleware.cors import CORSMiddleware

    api = FastAPI(title="NeuroDrift inference (stub)")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "https://rishishah.me"],
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["*"],
    )

    @api.post("/infer")
    async def infer_endpoint(
        nifti: UploadFile | None = File(default=None),
        age: float = Form(75.0),
        apoe: str = Form("E3E3"),
        treatment: Treatment = Form("placebo"),
    ) -> dict[str, Any]:
        blob = await nifti.read() if nifti is not None else None
        return infer.remote(blob, age, apoe, treatment)

    @api.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return api
