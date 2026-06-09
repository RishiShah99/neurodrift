"use client";

import { useCallback, useState } from "react";
import { cn } from "@/lib/utils";

interface Props {
  file: File | null;
  onFile: (f: File | null) => void;
}

export function DropZone({ file, onFile }: Props) {
  const [over, setOver] = useState(false);

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setOver(false);
      const f = e.dataTransfer.files?.[0];
      if (f) onFile(f);
    },
    [onFile],
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={onDrop}
      className={cn(
        "rounded-xl border border-dashed p-4 text-sm transition-colors",
        "bg-background/60 backdrop-blur",
        over ? "border-accent" : "border-border",
      )}
    >
      <p className="text-foreground/80">
        {file ? (
          <>
            <span className="text-accent">{file.name}</span>
            <span className="text-foreground/40"> · {(file.size / 1e6).toFixed(1)} MB</span>
          </>
        ) : (
          "Drop a T1 NIfTI here or pick a preloaded subject."
        )}
      </p>
      <p className="text-xs text-foreground/40 mt-1">
        Accepts <code>.nii</code> / <code>.nii.gz</code>. Demo runs on Modal GPUs.
      </p>
    </div>
  );
}
