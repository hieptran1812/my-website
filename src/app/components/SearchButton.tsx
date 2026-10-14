"use client";

import React, { useEffect, useState } from "react";
import { OPEN_COMMAND_PALETTE_EVENT } from "@/components/CommandPalette";

export default function SearchButton() {
  // Render "Ctrl" on the server, swap to "⌘" after mount on Apple devices.
  const [modKey, setModKey] = useState("Ctrl");

  useEffect(() => {
    if (/Mac|iPhone|iPad|iPod/.test(navigator.platform || navigator.userAgent)) {
      setModKey("⌘");
    }
  }, []);

  return (
    <button
      type="button"
      onClick={() => window.dispatchEvent(new Event(OPEN_COMMAND_PALETTE_EVENT))}
      aria-label="Search articles"
      aria-keyshortcuts="Meta+K Control+K /"
      title={`Search articles (${modKey}+K or /)`}
      className="inline-flex h-8 items-center gap-2 rounded-full border px-2.5 text-sm transition-colors duration-200 hover:border-[var(--accent)] hover:text-[var(--accent)]"
      style={{
        color: "var(--text-muted)",
        borderColor: "var(--border)",
        background: "var(--card-bg)",
      }}
    >
      <svg
        className="h-4 w-4"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        viewBox="0 0 24 24"
        aria-hidden="true"
      >
        <circle cx="11" cy="11" r="7" />
        <path strokeLinecap="round" d="M20 20l-3.5-3.5" />
      </svg>
      <span className="hidden sm:inline">Search</span>
      <kbd
        className="hidden sm:inline-flex items-center rounded border px-1.5 font-mono text-[0.7rem] leading-5"
        style={{ borderColor: "var(--border)" }}
      >
        {modKey} K
      </kbd>
    </button>
  );
}
