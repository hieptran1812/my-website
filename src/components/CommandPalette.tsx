"use client";

import { useCallback, useEffect, useState } from "react";
import dynamic from "next/dynamic";

// cmdk + the dialog component aren't loaded into the bundle until the user
// actually opens the palette (Cmd/Ctrl+K or "/"). That keeps ~10KB of JS off
// every page load.
const CommandPaletteDialog = dynamic(
  () => import("./CommandPaletteDialog"),
  { ssr: false },
);

/** Dispatch this on `window` to open the palette from any button. */
export const OPEN_COMMAND_PALETTE_EVENT = "open-command-palette";

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return (
    tag === "INPUT" ||
    tag === "TEXTAREA" ||
    tag === "SELECT" ||
    target.isContentEditable
  );
}

export default function CommandPalette() {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const isModK = (e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k";
      // "/" opens search like GitHub/YouTube, but never steals a keystroke
      // the reader is typing into a field.
      const isSlash =
        e.key === "/" &&
        !e.metaKey &&
        !e.ctrlKey &&
        !e.altKey &&
        !isTypingTarget(e.target);
      if (isModK) {
        e.preventDefault();
        setOpen((v) => !v);
      } else if (isSlash && !open) {
        e.preventDefault();
        setOpen(true);
      } else if (e.key === "Escape" && open) {
        setOpen(false);
      }
    };
    const onOpen = () => setOpen(true);
    document.addEventListener("keydown", onKey);
    window.addEventListener(OPEN_COMMAND_PALETTE_EVENT, onOpen);
    return () => {
      document.removeEventListener("keydown", onKey);
      window.removeEventListener(OPEN_COMMAND_PALETTE_EVENT, onOpen);
    };
  }, [open]);

  const close = useCallback(() => setOpen(false), []);

  if (!open) return null;
  return <CommandPaletteDialog onClose={close} />;
}
