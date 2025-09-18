"use client";

import React from "react";
import { useTheme } from "@/app/ThemeProvider";

interface SectionSeparatorProps {
  variant?: "wave" | "curve" | "diagonal" | "subtle";
  position?: "top" | "bottom";
  flip?: boolean;
  className?: string;
}

export default function SectionSeparator({
  variant = "wave",
  position = "bottom",
  flip = false,
  className = "",
}: SectionSeparatorProps) {
  const { theme } = useTheme();

  const getColors = () => {
    return {
      primary: theme === "dark" ? "var(--surface)" : "var(--background)",
      secondary:
        theme === "dark" ? "var(--surface-secondary)" : "var(--surface)",
      accent: "var(--accent)",
    };
  };

  const colors = getColors();

  const waveVariants = {
    wave: (
      <path
        d="M0,160L48,144C96,128,192,96,288,96C384,96,480,128,576,144C672,160,768,160,864,144C960,128,1056,96,1152,96C1248,96,1344,128,1392,144L1440,160L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z"
        fill={colors.primary}
      />
    ),
    curve: (
      <path d="M0,160L1440,160L1440,320Q720,240,0,320Z" fill={colors.primary} />
    ),
    diagonal: (
      <path d="M0,160L1440,200L1440,320L0,320Z" fill={colors.primary} />
    ),
    subtle: (
      <path
        d="M0,160C240,140,480,140,720,160C960,180,1200,180,1440,160L1440,320L0,320Z"
        fill={colors.primary}
      />
    ),
  };

  return (
    <div
      className={`absolute left-0 right-0 w-full pointer-events-none z-10 ${
        position === "top" ? "top-0" : "bottom-0"
      } ${className}`}
      style={{
        height: "80px",
        transform: flip ? "scaleY(-1)" : undefined,
      }}
    >
      <svg
        viewBox="0 0 1440 320"
        className="w-full h-full"
        preserveAspectRatio="none"
        style={{
          display: "block",
          transform: position === "top" && !flip ? "scaleY(-1)" : undefined,
        }}
      >
        <defs>
          <linearGradient id="waveGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor={colors.primary} stopOpacity="0.8" />
            <stop offset="50%" stopColor={colors.secondary} stopOpacity="0.9" />
            <stop offset="100%" stopColor={colors.primary} stopOpacity="0.8" />
          </linearGradient>
        </defs>
        {React.cloneElement(waveVariants[variant], {
          fill: "url(#waveGradient)",
        })}
      </svg>
    </div>
  );
}
