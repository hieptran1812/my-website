"use client";

import React from "react";
import { useRouter } from "next/navigation";

interface TagBadgeProps {
  tag: string;
  count?: number;
  variant?: "default" | "compact" | "large" | "pill";
  clickable?: boolean;
  isActive?: boolean;
  onClick?: (tag: string) => void;
  showIcon?: boolean;
}

/**
 * TagBadge Component
 * A reusable tag badge with smooth animations and multiple variants
 * Can be clickable to navigate to tag filter page or trigger custom onClick
 */
export default function TagBadge({
  tag,
  count,
  variant = "default",
  clickable = true,
  isActive = false,
  onClick,
  showIcon = false,
}: TagBadgeProps) {
  const router = useRouter();

  // Normalize tag for URL (lowercase, replace spaces with hyphens)
  const tagSlug = tag.toLowerCase().replace(/\s+/g, "-");

  // Variant-specific styles
  const variantStyles = {
    default: {
      padding: "px-2.5 py-1",
      fontSize: "text-xs",
      rounded: "rounded-full",
    },
    compact: {
      padding: "px-2 py-0.5",
      fontSize: "text-xs",
      rounded: "rounded-full",
    },
    large: {
      padding: "px-4 py-2",
      fontSize: "text-sm",
      rounded: "rounded-full",
    },
    pill: {
      padding: "px-3 py-1.5",
      fontSize: "text-xs",
      rounded: "rounded-lg",
    },
  };

  const styles = variantStyles[variant];

  // Base classes for the badge
  const baseClasses = `
    ${styles.padding}
    ${styles.fontSize}
    ${styles.rounded}
    font-medium
    inline-flex items-center gap-1.5
    transition-all duration-300 ease-out
    ${
      clickable
        ? "cursor-pointer hover:scale-105 hover:shadow-md active:scale-95"
        : ""
    }
  `;

  // Style object for colors (using CSS variables for theme support)
  const getStyle = (): React.CSSProperties => {
    if (isActive) {
      return {
        backgroundColor: "var(--accent)",
        color: "white",
        boxShadow: "0 4px 12px rgba(var(--accent-rgb, 130, 170, 255), 0.35)",
      };
    }

    return {
      backgroundColor: "var(--accent-subtle)",
      color: "var(--accent)",
    };
  };

  // Hover style enhancement
  const hoverStyle = clickable
    ? {
        "--hover-bg": "var(--accent)",
        "--hover-color": "white",
      }
    : {};

  // Content of the badge
  const content = (
    <>
      {showIcon && (
        <svg
          className="w-3 h-3"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"
          />
        </svg>
      )}
      <span>{tag}</span>
      {count !== undefined && (
        <span
          className="px-1.5 py-0.5 rounded-full text-xs font-bold ml-0.5"
          style={{
            backgroundColor: isActive
              ? "rgba(255, 255, 255, 0.2)"
              : "var(--accent)",
            color: "white",
          }}
        >
          {count}
        </span>
      )}
    </>
  );

  // Handle click for custom onClick handler
  const handleClick = (e: React.MouseEvent) => {
    if (onClick) {
      e.preventDefault();
      e.stopPropagation();
      onClick(tag);
    }
  };

  // If clickable and no custom onClick, use router navigation
  if (clickable && !onClick) {
    const handleTagClick = (e: React.MouseEvent) => {
      // Prevent parent Link from firing
      e.preventDefault();
      e.stopPropagation();
      // Navigate programmatically
      router.push(`/blog/tags/${tagSlug}`);
    };

    return (
      <span
        className={baseClasses}
        style={{ ...getStyle(), ...hoverStyle } as React.CSSProperties}
        onClick={handleTagClick}
        onMouseDown={(e) => e.stopPropagation()}
        role="link"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            e.stopPropagation();
            router.push(`/blog/tags/${tagSlug}`);
          }
        }}
      >
        {content}
      </span>
    );
  }

  // If clickable with custom onClick, use button behavior
  if (clickable && onClick) {
    return (
      <span
        className={baseClasses}
        style={{ ...getStyle(), ...hoverStyle } as React.CSSProperties}
        onClick={handleClick}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onClick(tag);
          }
        }}
      >
        {content}
      </span>
    );
  }

  // Non-clickable badge
  return (
    <span className={baseClasses} style={getStyle()}>
      {content}
    </span>
  );
}

/**
 * TagList Component
 * A wrapper component for displaying multiple tags with proper spacing
 */
interface TagListProps {
  tags: string[];
  maxTags?: number;
  variant?: "default" | "compact" | "large" | "pill";
  clickable?: boolean;
  showMoreCount?: boolean;
  className?: string;
}

export function TagList({
  tags,
  maxTags,
  variant = "default",
  clickable = true,
  showMoreCount = true,
  className = "",
}: TagListProps) {
  if (!tags || tags.length === 0) return null;

  const displayTags = maxTags ? tags.slice(0, maxTags) : tags;
  const remainingCount = maxTags ? tags.length - maxTags : 0;

  return (
    <div className={`flex flex-wrap gap-1.5 ${className}`}>
      {displayTags.map((tag) => (
        <TagBadge key={tag} tag={tag} variant={variant} clickable={clickable} />
      ))}
      {showMoreCount && remainingCount > 0 && (
        <span
          className="px-2 py-1 text-xs rounded-full font-medium"
          style={{
            backgroundColor: "var(--surface)",
            color: "var(--text-muted)",
            border: "1px dashed var(--border)",
          }}
        >
          +{remainingCount} more
        </span>
      )}
    </div>
  );
}
