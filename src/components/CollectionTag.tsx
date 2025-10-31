import React from "react";
import Link from "next/link";

interface CollectionTagProps {
  collection: string;
  variant?: "default" | "compact" | "detailed";
  className?: string;
}

export default function CollectionTag({
  collection,
  variant = "default",
  className = "",
}: CollectionTagProps) {
  // Convert collection name to slug for URL
  const collectionSlug = collection
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, "")
    .replace(/\s+/g, "-")
    .trim();

  // Collection icons mapping (có thể customize thêm)
  const getCollectionIcon = (collection: string) => {
    if (
      collection.toLowerCase().includes("finance") ||
      collection.toLowerCase().includes("trading")
    ) {
      return "💰";
    }
    if (
      collection.toLowerCase().includes("machine learning") ||
      collection.toLowerCase().includes("ai")
    ) {
      return "🤖";
    }
    if (
      collection.toLowerCase().includes("software") ||
      collection.toLowerCase().includes("development")
    ) {
      return "💻";
    }
    if (
      collection.toLowerCase().includes("paper") ||
      collection.toLowerCase().includes("research")
    ) {
      return "📚";
    }
    if (collection.toLowerCase().includes("notes")) {
      return "📝";
    }
    return "📖"; // Default icon
  };

  // Base styling classes
  const baseClasses = `
    inline-flex items-center gap-2 rounded-full font-medium border
    transition-all duration-300 ease-in-out
    hover:scale-105 hover:shadow-md 
    active:scale-95
    focus:outline-none focus:ring-2 focus:ring-offset-2
    cursor-pointer
  `.trim();

  // Variant-specific classes
  const variantClasses = {
    default: "px-3 py-1.5 text-sm",
    compact: "px-2 py-1 text-xs",
    detailed: "px-4 py-2 text-sm font-semibold",
  };

  // Dynamic styling using CSS variables
  const style = {
    backgroundColor: "var(--surface)",
    color: "var(--text-primary)",
    borderColor: "var(--border)",
    "--collection-accent": "var(--accent)",
  } as React.CSSProperties;

  const hoverStyle = `
    hover:bg-[var(--surface-hover)]
    hover:border-[var(--accent)]
    hover:text-[var(--accent)]
  `;

  return (
    <Link
      href={`/blog/collections/${collectionSlug}`}
      className={`${baseClasses} ${variantClasses[variant]} ${hoverStyle} ${className}`}
      style={style}
      title={`View all articles in "${collection}" collection`}
    >
      <span
        className={variant === "compact" ? "text-sm" : "text-base"}
        role="img"
        aria-label="Collection icon"
      >
        {getCollectionIcon(collection)}
      </span>
      <span className="leading-tight">
        {variant === "compact" && collection.length > 20
          ? `${collection.substring(0, 20)}...`
          : collection}
      </span>
    </Link>
  );
}
