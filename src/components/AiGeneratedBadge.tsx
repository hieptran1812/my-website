"use client";

import React, { useState, useEffect, useRef } from "react";

interface AiGeneratedBadgeProps {
  variant?: "default" | "compact" | "detailed";
  className?: string;
}

const FULL_TEXT = "AI Generated";

export default function AiGeneratedBadge({
  variant = "default",
  className = "",
}: AiGeneratedBadgeProps) {
  const [showTooltip, setShowTooltip] = useState(false);
  const [displayedText, setDisplayedText] = useState("");
  const [hasAnimated, setHasAnimated] = useState(false);
  const [showCursor, setShowCursor] = useState(true);
  const badgeRef = useRef<HTMLSpanElement>(null);

  const sizeClasses = {
    compact: "px-1.5 py-0.5 text-[10px] gap-1",
    default: "px-2 py-1 text-xs gap-1.5",
    detailed: "px-3 py-1.5 text-sm gap-2",
  };

  const iconSize = {
    compact: "w-3 h-3",
    default: "w-3.5 h-3.5",
    detailed: "w-4 h-4",
  };

  // Typewriter effect triggered by IntersectionObserver
  useEffect(() => {
    if (hasAnimated) return;

    const el = badgeRef.current;
    if (!el) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          observer.disconnect();
          let i = 0;
          const interval = setInterval(() => {
            i++;
            setDisplayedText(FULL_TEXT.slice(0, i));
            if (i >= FULL_TEXT.length) {
              clearInterval(interval);
              setHasAnimated(true);
              setTimeout(() => setShowCursor(false), 600);
            }
          }, 70);
        }
      },
      { threshold: 0.5 },
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, [hasAnimated]);

  return (
    <>
      <style>{`
        @keyframes gradient-shift {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        @keyframes border-glow {
          0%, 100% { border-color: rgba(139, 92, 246, 0.4); box-shadow: 0 0 6px rgba(139, 92, 246, 0.15); }
          50% { border-color: rgba(236, 72, 153, 0.4); box-shadow: 0 0 12px rgba(168, 85, 247, 0.25); }
        }
        @keyframes shimmer {
          0% { background-position: -200% center; }
          100% { background-position: 200% center; }
        }
      `}</style>
      <span
        ref={badgeRef}
        className={`relative inline-flex items-center font-semibold rounded-full cursor-default select-none
          border transition-all duration-200 hover:scale-105 overflow-hidden
          ${sizeClasses[variant]} ${className}`}
        style={{
          background:
            "linear-gradient(135deg, rgba(139, 92, 246, 0.12), rgba(236, 72, 153, 0.10), rgba(59, 130, 246, 0.10))",
          backgroundSize: "200% 200%",
          animation:
            "gradient-shift 4s ease infinite, border-glow 3s ease infinite",
          borderColor: "rgba(139, 92, 246, 0.35)",
          borderWidth: "1.5px",
        }}
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Shimmer overlay */}
        {hasAnimated && (
          <span
            className="absolute inset-0 rounded-full pointer-events-none"
            style={{
              background:
                "linear-gradient(90deg, transparent 25%, rgba(255,255,255,0.12) 50%, transparent 75%)",
              backgroundSize: "200% 100%",
              animation: "shimmer 3s ease-in-out infinite",
            }}
          />
        )}

        <svg
          className={`${iconSize[variant]} transition-opacity duration-300 relative z-10`}
          style={{
            opacity: displayedText ? 1 : 0,
            filter: "drop-shadow(0 0 2px rgba(168, 85, 247, 0.4))",
          }}
          viewBox="0 0 24 24"
          fill="url(#star-gradient)"
          stroke="url(#star-gradient)"
          strokeWidth="1"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <defs>
            <linearGradient
              id="star-gradient"
              x1="0%"
              y1="0%"
              x2="100%"
              y2="100%"
            >
              <stop offset="0%" stopColor="#8b5cf6" />
              <stop offset="50%" stopColor="#ec4899" />
              <stop offset="100%" stopColor="#6366f1" />
            </linearGradient>
          </defs>
          <path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" />
        </svg>

        <span
          className="relative z-10"
          style={{
            background: "linear-gradient(135deg, #8b5cf6, #ec4899, #6366f1)",
            backgroundSize: "200% 200%",
            animation: "gradient-shift 4s ease infinite",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            backgroundClip: "text",
          }}
        >
          {displayedText}
          {showCursor && (
            <span
              className="inline-block w-[2px] ml-[1px] align-middle animate-pulse"
              style={{
                height: "1em",
                background: "linear-gradient(180deg, #8b5cf6, #ec4899)",
                WebkitTextFillColor: "initial",
              }}
            />
          )}
        </span>

        {/* Tooltip */}
        {showTooltip && hasAnimated && (
          <span
            className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 text-xs font-normal rounded-lg shadow-lg whitespace-nowrap z-50 pointer-events-none"
            style={{
              backgroundColor: "var(--surface-hover, #374151)",
              color: "var(--text-primary, #f3f4f6)",
              border: "1px solid var(--border, #4b5563)",
            }}
          >
            This article was generated with the assistance of AI
            <span
              className="absolute top-full left-1/2 -translate-x-1/2 -mt-px border-4 border-transparent"
              style={{
                borderTopColor: "var(--surface-hover, #374151)",
              }}
            />
          </span>
        )}
      </span>
    </>
  );
}
