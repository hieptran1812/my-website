"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";

const DynamicSpeedInsights = dynamic(
  () => import("@vercel/speed-insights/next").then((mod) => mod.SpeedInsights),
  {
    ssr: false,
    loading: () => null,
  }
);

const DynamicAnalytics = dynamic(
  () => import("@vercel/analytics/next").then((mod) => mod.Analytics),
  {
    ssr: false,
    loading: () => null,
  }
);

export function VercelSpeedInsights() {
  const [shouldRender, setShouldRender] = useState(false);
  const [isBlocked, setIsBlocked] = useState(false);

  useEffect(() => {
    // Only try to load in production
    if (process.env.NODE_ENV === "production") {
      // Test if Speed Insights can be loaded
      const timer = setTimeout(() => {
        // If the script doesn't load within 3 seconds, assume it's blocked
        if (!shouldRender) {
          setIsBlocked(true);
          console.info(
            "Vercel Speed Insights may be blocked by content blocker"
          );
        }
      }, 3000);

      // Try to detect if script loads successfully
      if (typeof window !== "undefined") {
        setShouldRender(true);
      }

      return () => clearTimeout(timer);
    }
  }, [shouldRender]);

  if (isBlocked || process.env.NODE_ENV !== "production") {
    return null;
  }

  return <DynamicSpeedInsights />;
}

export function VercelAnalytics() {
  const [shouldRender, setShouldRender] = useState(false);

  useEffect(() => {
    // Analytics is generally less likely to be blocked
    if (
      process.env.NODE_ENV === "production" &&
      typeof window !== "undefined"
    ) {
      setShouldRender(true);
    }
  }, []);

  if (!shouldRender) {
    return null;
  }

  return <DynamicAnalytics />;
}
