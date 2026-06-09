"use client";

import * as Slider from "@radix-ui/react-slider";
import * as Select from "@radix-ui/react-select";
import { ChevronDown, Play, Loader2 } from "lucide-react";
import type { Treatment } from "@/lib/types";
import { cn } from "@/lib/utils";

const TREATMENTS: { value: Treatment; label: string }[] = [
  { value: "placebo", label: "Placebo" },
  { value: "lecanemab", label: "Lecanemab (Aβ-clearing)" },
  { value: "donanemab", label: "Donanemab (Aβ-clearing)" },
  { value: "anti_tau", label: "Anti-tau (class)" },
  { value: "anti_inflammatory", label: "Anti-inflammatory (class)" },
  { value: "glp1", label: "GLP-1 (class)" },
];

const APOE = ["E2E2", "E2E3", "E3E3", "E3E4", "E4E4"] as const;

interface Props {
  age: number;
  setAge: (v: number) => void;
  apoe: (typeof APOE)[number];
  setApoe: (v: (typeof APOE)[number]) => void;
  treatment: Treatment;
  setTreatment: (v: Treatment) => void;
  onRun: () => void;
  loading: boolean;
}

export function Controls({ age, setAge, apoe, setApoe, treatment, setTreatment, onRun, loading }: Props) {
  return (
    <div className="flex flex-col gap-5">
      <div>
        <label className="flex justify-between text-xs uppercase tracking-wider text-foreground/50 mb-2">
          <span>Target age</span>
          <span className="text-foreground font-mono">{age}</span>
        </label>
        <Slider.Root
          className="relative flex items-center select-none touch-none w-full h-6"
          value={[age]}
          min={9}
          max={90}
          step={1}
          onValueChange={(v) => setAge(v[0]!)}
        >
          <Slider.Track className="bg-border relative grow rounded-full h-1">
            <Slider.Range className="absolute bg-accent rounded-full h-full" />
          </Slider.Track>
          <Slider.Thumb className="block w-4 h-4 bg-accent rounded-full shadow focus:outline-none" />
        </Slider.Root>
      </div>

      <div>
        <label className="block text-xs uppercase tracking-wider text-foreground/50 mb-2">
          ApoE genotype
        </label>
        <div className="grid grid-cols-5 gap-1">
          {APOE.map((g) => (
            <button
              key={g}
              onClick={() => setApoe(g)}
              className={cn(
                "rounded-md py-2 text-xs font-mono",
                apoe === g
                  ? "bg-accent text-background"
                  : "bg-muted text-foreground/70 hover:bg-muted/70",
              )}
            >
              {g}
            </button>
          ))}
        </div>
      </div>

      <div>
        <label className="block text-xs uppercase tracking-wider text-foreground/50 mb-2">
          Treatment
        </label>
        <Select.Root value={treatment} onValueChange={(v) => setTreatment(v as Treatment)}>
          <Select.Trigger className="inline-flex items-center justify-between w-full rounded-md bg-muted px-3 py-2 text-sm">
            <Select.Value />
            <Select.Icon>
              <ChevronDown className="w-4 h-4" />
            </Select.Icon>
          </Select.Trigger>
          <Select.Portal>
            <Select.Content className="bg-background border border-border rounded-md shadow-xl">
              <Select.Viewport className="p-1">
                {TREATMENTS.map((t) => (
                  <Select.Item
                    key={t.value}
                    value={t.value}
                    className="px-3 py-2 text-sm rounded cursor-pointer data-[highlighted]:bg-muted outline-none"
                  >
                    <Select.ItemText>{t.label}</Select.ItemText>
                  </Select.Item>
                ))}
              </Select.Viewport>
            </Select.Content>
          </Select.Portal>
        </Select.Root>
      </div>

      <button
        onClick={onRun}
        disabled={loading}
        className={cn(
          "mt-2 inline-flex items-center justify-center gap-2 rounded-md py-2 text-sm font-medium transition-colors",
          loading ? "bg-muted text-foreground/40" : "bg-accent text-background hover:opacity-90",
        )}
      >
        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
        {loading ? "Inferring…" : "Run trajectory"}
      </button>
    </div>
  );
}
