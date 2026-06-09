"use client";

import type { RegionEnvelope, RegionTrajectory } from "@/lib/types";

interface Props {
  regions?: RegionTrajectory[];
  envelope?: RegionEnvelope[];
}

export function RegionPanel({ regions, envelope }: Props) {
  if (!regions?.length) {
    return (
      <div className="text-xs text-foreground/40">
        Per-region trajectories and 90% uncertainty envelopes appear here after the first run.
      </div>
    );
  }
  return (
    <div className="flex flex-col gap-3">
      <h2 className="text-xs uppercase tracking-wider text-foreground/50">Regions</h2>
      {regions.map((r) => {
        const env = envelope?.find((e) => e.region === r.region);
        const min = Math.min(...r.values);
        const max = Math.max(...r.values);
        return (
          <div key={r.region} className="rounded-md bg-muted/60 p-3">
            <div className="flex justify-between items-baseline mb-1">
              <span className="text-sm">{r.region}</span>
              <span className="text-xs font-mono text-foreground/60">
                {r.values[0]?.toFixed(2)} → {r.values.at(-1)?.toFixed(2)}
              </span>
            </div>
            <Sparkline values={r.values} envelopeLower={env?.lower} envelopeUpper={env?.upper} domain={[min, max]} />
          </div>
        );
      })}
    </div>
  );
}

function Sparkline({
  values,
  envelopeLower,
  envelopeUpper,
  domain,
}: {
  values: number[];
  envelopeLower?: number[];
  envelopeUpper?: number[];
  domain: [number, number];
}) {
  const w = 280;
  const h = 48;
  const [lo, hi] = domain;
  const span = hi - lo || 1;
  const x = (i: number) => (i / (values.length - 1 || 1)) * w;
  const y = (v: number) => h - ((v - lo) / span) * h;

  const linePath = values.map((v, i) => `${i === 0 ? "M" : "L"}${x(i)},${y(v)}`).join(" ");
  const bandPath =
    envelopeLower && envelopeUpper
      ? [
          ...envelopeUpper.map((v, i) => `${i === 0 ? "M" : "L"}${x(i)},${y(v)}`),
          ...envelopeLower
            .map((v, i) => `${i === 0 ? "L" : "L"}${x(envelopeLower.length - 1 - i)},${y(envelopeLower[envelopeLower.length - 1 - i]!)}`),
          "Z",
        ].join(" ")
      : null;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-12">
      {bandPath && <path d={bandPath} fill="hsl(var(--accent) / 0.15)" />}
      <path d={linePath} fill="none" stroke="hsl(var(--accent))" strokeWidth={1.5} />
    </svg>
  );
}
