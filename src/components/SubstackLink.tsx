import React from "react";
import { PAYWALL_SUBSCRIBE_URL } from "@/lib/paywallConfig";

interface SubstackLinkProps {
  substackUrl?: string;
  className?: string;
}

/**
 * Link to the post's Substack page, falling back to the publication home page
 * until the post has been published there.
 */
export default function SubstackLink({
  substackUrl,
  className = "",
}: SubstackLinkProps) {
  const postUrl = substackUrl?.trim();
  const href = postUrl || PAYWALL_SUBSCRIBE_URL;

  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className={`inline-flex items-center gap-2 text-sm font-medium transition-colors hover:text-[var(--accent)] ${className}`}
      style={{ color: "var(--text-secondary)" }}
    >
      {postUrl ? "Read on Substack" : "Publish soon on Substack"}
      <svg
        className="h-4 w-4"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        aria-hidden="true"
      >
        <path d="M14 5h5v5" />
        <path d="m10 14 9-9" />
        <path d="M19 14v4a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h4" />
      </svg>
    </a>
  );
}
