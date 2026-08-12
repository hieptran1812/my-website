"use client";

import React from "react";
import { PAYWALL_SUBSCRIBE_URL } from "@/lib/paywallConfig";
import "./styles/PaywallGate.css";

interface PaywallGateProps {
  /** Post title, woven into the copy when available. */
  title?: string;
  /** Exact Substack post URL, when this article has already been published. */
  substackUrl?: string;
}

/**
 * Substack-style gate drawn under a truncated post: a fade that dissolves the
 * last visible paragraph, then a card whose only exit is the Substack CTA.
 *
 * The withheld body never reaches the browser — the server already cut it
 * (`src/lib/paywall.ts`). This component is purely the visible half.
 */
export default function PaywallGate({ title, substackUrl }: PaywallGateProps) {
  const destination = substackUrl?.trim() || PAYWALL_SUBSCRIBE_URL;
  const isPublished = Boolean(substackUrl?.trim());

  return (
    <div className="paywall-gate" data-testid="paywall-gate">
      <div className="paywall-fade" aria-hidden="true" />

      <section className="paywall-card" aria-label="Subscriber-only content">
        <span className="paywall-lock" aria-hidden="true">
          <svg
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
            <path d="M7 11V7a5 5 0 0 1 10 0v4" />
          </svg>
        </span>

        <h2 className="paywall-title">
          Keep reading with a Substack subscription
        </h2>

        <p className="paywall-body">
          {title ? `“${title}” is` : "This post is"} part of the trading
          library, published in full on Substack. Subscribe there to read the
          rest of this post and the whole archive on markets, trading and
          finance.
        </p>

        <a
          className="paywall-button"
          href={destination}
          target="_blank"
          rel="noopener noreferrer"
        >
          {isPublished
            ? "Read the full post on Substack"
            : "Publish soon on Substack"}
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="M5 12h14" />
            <path d="m12 5 7 7-7 7" />
          </svg>
        </a>

        <p className="paywall-note">
          Already a subscriber?{" "}
          <a
            href={destination}
            target="_blank"
            rel="noopener noreferrer"
          >
            Open it on Substack
          </a>
          .
        </p>
      </section>
    </div>
  );
}
