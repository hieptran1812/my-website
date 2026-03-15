import React, { useState } from "react";

interface AiGeneratedBadgeProps {
  variant?: "default" | "compact" | "detailed";
  className?: string;
}

export default function AiGeneratedBadge({
  variant = "default",
  className = "",
}: AiGeneratedBadgeProps) {
  const [showTooltip, setShowTooltip] = useState(false);

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

  return (
    <span
      className={`relative inline-flex items-center font-medium rounded-full cursor-default select-none
        border transition-all duration-200 hover:scale-105
        ${sizeClasses[variant]} ${className}`}
      style={{
        backgroundColor: "rgba(139, 92, 246, 0.1)",
        borderColor: "rgba(139, 92, 246, 0.3)",
        color: "rgb(139, 92, 246)",
      }}
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
      onClick={(e) => e.stopPropagation()}
    >
      <svg
        className={iconSize[variant]}
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" />
      </svg>
      <span>AI Generated</span>

      {/* Tooltip */}
      {showTooltip && (
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
  );
}
