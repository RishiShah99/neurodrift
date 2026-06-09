export type Treatment =
  | "placebo"
  | "lecanemab"
  | "donanemab"
  | "anti_tau"
  | "anti_inflammatory"
  | "glp1";

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
