import { NextResponse } from "next/server";
import type { InferResponse, Treatment } from "@/lib/types";

const REGIONS = [
  "hippocampus",
  "ventricles",
  "entorhinal_cortex",
  "wmh",
];

const TREATMENT_BEND: Record<Treatment, number> = {
  placebo: 0,
  lecanemab: 0.35,
  donanemab: 0.4,
  anti_tau: 0.2,
  anti_inflammatory: 0.1,
  glp1: 0.15,
};

const APOE_RATE: Record<string, number> = {
  E2E2: 0.4,
  E2E3: 0.6,
  E3E3: 1.0,
  E3E4: 1.5,
  E4E4: 2.2,
};

export async function POST(req: Request): Promise<Response> {
  const fd = await req.formData();
  const ageTarget = Number(fd.get("age") ?? 75);
  const apoe = String(fd.get("apoe") ?? "E3E3");
  const treatment = String(fd.get("treatment") ?? "placebo") as Treatment;

  // Keep this baseline in sync with the Modal stub (apps/api/modal_app.py: age_now).
  const ageNow = 60;
  const n = 24;
  const ages = Array.from({ length: n }, (_, i) => ageNow + ((ageTarget - ageNow) * i) / (n - 1));
  const rate = APOE_RATE[apoe] ?? 1.0;
  const bend = TREATMENT_BEND[treatment] ?? 0;

  const regions = REGIONS.map((region, idx) => {
    const base = idx === 0 ? 4.2 : idx === 1 ? 20.0 : idx === 2 ? 3.5 : 5.0;
    const direction = idx === 1 || idx === 3 ? 1 : -1;
    const values = ages.map(
      (a) => base + direction * rate * (1 - bend) * 0.015 * (a - ageNow),
    );
    return { region, ages, values };
  });

  const envelope = regions.map((r) => ({
    region: r.region,
    ages: r.ages,
    lower: r.values.map((v) => v * 0.97),
    upper: r.values.map((v) => v * 1.03),
  }));

  const response: InferResponse = {
    gaussians_url: null,
    trajectory: { age_now: ageNow, age_target: ageTarget },
    regions,
    envelope,
  };
  return NextResponse.json(response);
}
