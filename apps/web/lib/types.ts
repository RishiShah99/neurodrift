export type Treatment =
  | "placebo"
  | "lecanemab"
  | "donanemab"
  | "anti_tau"
  | "anti_inflammatory"
  | "glp1";

// Single source of truth for the ApoE genotype options. Both backends
// (route.ts, modal_app.py) accept all five; keep this in sync with them.
export type Apoe = "E2E2" | "E2E3" | "E3E3" | "E3E4" | "E4E4";

export interface RegionTrajectory {
  region: string;
  ages: number[];
  values: number[];
}

export interface RegionEnvelope {
  region: string;
  ages: number[];
  lower: number[];
  upper: number[];
}

export interface InferResponse {
  gaussians_url: string | null;
  trajectory: {
    age_now: number;
    age_target: number;
  };
  regions: RegionTrajectory[];
  envelope: RegionEnvelope[];
}
