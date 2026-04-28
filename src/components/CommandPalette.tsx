"use client";

import { useCallback, useEffect, useState } from "react";
import dynamic from "next/dynamic";

// cmdk + the dialog component aren't loaded into the bundle until the user
// actually opens the palette (Cmd/Ctrl+K). That keeps ~10KB of JS off every
// page load.
const CommandPaletteDialog = dynamic(
  () => import("./CommandPaletteDialog"),
  { ssr: false },
);

export default function CommandPalette() {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const isModK = (e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k";
      if (isModK) {
        e.preventDefault();
        setOpen((v) => !v);
      } else if (e.key === "Escape" && open) {
        setOpen(false);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open]);

  const close = useCallback(() => setOpen(false), []);

  if (!open) return null;
  return <CommandPaletteDialog onClose={close} />;
}
