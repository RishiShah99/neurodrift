"use client";

import { useState } from "react";
import { Controls } from "@/components/Controls";
import { DropZone } from "@/components/DropZone";
import { RegionPanel } from "@/components/RegionPanel";
import { Viewer } from "@/components/Viewer";
import type { Apoe, InferResponse, Treatment } from "@/lib/types";

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [age, setAge] = useState(60);
  const [apoe, setApoe] = useState<Apoe>("E3E3");
  const [treatment, setTreatment] = useState<Treatment>("placebo");
  const [response, setResponse] = useState<InferResponse | null>(null);
  const [loading, setLoading] = useState(false);

  async function infer() {
    setLoading(true);
    try {
      const fd = new FormData();
      if (file) fd.append("nifti", file);
      fd.append("age", String(age));
      fd.append("apoe", apoe);
      fd.append("treatment", treatment);
      const res = await fetch("/api/infer", { method: "POST", body: fd });
      const json = (await res.json()) as InferResponse;
      setResponse(json);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="grid grid-cols-1 lg:grid-cols-[1fr_360px] h-screen">
      <section className="relative">
        <Viewer trajectory={response?.trajectory} />
        <div className="absolute top-6 left-6 right-6 lg:right-auto lg:max-w-md">
          <header className="mb-4">
            <h1 className="text-2xl font-semibold tracking-tight">NeuroDrift</h1>
            <p className="text-sm text-foreground/60">
              Continuous-time multimodal brain trajectory playground.
            </p>
          </header>
          <DropZone file={file} onFile={setFile} />
        </div>
      </section>

      <aside className="border-l border-border bg-muted/30 p-6 flex flex-col gap-6 overflow-y-auto">
        <Controls
          age={age}
          setAge={setAge}
          apoe={apoe}
          setApoe={setApoe}
          treatment={treatment}
          setTreatment={setTreatment}
          onRun={infer}
          loading={loading}
        />
        <RegionPanel regions={response?.regions} envelope={response?.envelope} />
      </aside>
    </main>
  );
}
