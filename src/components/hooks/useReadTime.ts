/**
 * React hook for calculating read time on the client side
 * Useful for dynamic content or real-time preview scenarios
 */

import { useMemo } from "react";
import {
  calculateReadTimeWithTags,
  ReadTimeResult,
} from "../lib/readTimeCalculator";

interface UseReadTimeOptions {
  content: string;
  tags?: string[];
  category?: string;
  wordsPerMinute?: number;
}

/**
 * Hook to calculate read time for blog content
 * @param options Configuration for read time calculation
 * @returns ReadTimeResult object with analysis and formatted read time
 */
export function useReadTime({
  content,
  tags = [],
  category = "general",
  wordsPerMinute = 200,
}: UseReadTimeOptions): ReadTimeResult {
  return useMemo(() => {
    return calculateReadTimeWithTags(content, tags, category, {
      wordsPerMinute,
    });
  }, [content, tags, category, wordsPerMinute]);
}

/**
 * Hook to get just the formatted read time string
 * @param options Configuration for read time calculation
 * @returns Formatted read time string (e.g., "8 min read")
 */
export function useReadTimeString(options: UseReadTimeOptions): string {
  const result = useReadTime(options);
  return result.readTime;
}

/**
 * Hook for real-time read time calculation (useful for editors)
 * @param options Configuration for read time calculation
 * @returns Object with read time and detailed analysis
 */
export function useReadTimeAnalysis(options: UseReadTimeOptions) {
  const result = useReadTime(options);

  return {
    readTime: result.readTime,
    readTimeMinutes: result.readTimeMinutes,
    wordCount: result.analysis.wordCount,
    complexity: result.analysis.complexity,
    breakdown: result.breakdown,
    analysis: result.analysis,
  };
}
