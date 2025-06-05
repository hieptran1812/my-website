"use client";

import React, { useEffect, useRef, useCallback } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";

interface MathRendererProps {
  latex: string;
  displayMode?: boolean;
  className?: string;
}

export default function MathRenderer({
  latex,
  displayMode = false,
  className = "",
}: MathRendererProps) {
  const mathRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (mathRef.current && latex) {
      try {
        katex.render(latex, mathRef.current, {
          displayMode,
          throwOnError: false,
          errorColor: "#cc0000",
          strict: "warn",
          trust: false,
          macros: {
            "\\RR": "\\mathbb{R}",
            "\\NN": "\\mathbb{N}",
            "\\ZZ": "\\mathbb{Z}",
            "\\QQ": "\\mathbb{Q}",
            "\\CC": "\\mathbb{C}",
            "\\PP": "\\mathbb{P}",
            "\\EE": "\\mathbb{E}",
            "\\Var": "\\text{Var}",
            "\\Cov": "\\text{Cov}",
            "\\argmin": "\\text{argmin}",
            "\\argmax": "\\text{argmax}",
          },
        });
      } catch (error) {
        console.error("KaTeX rendering error:", error);
        if (mathRef.current) {
          mathRef.current.textContent = latex;
          mathRef.current.style.color = "#cc0000";
        }
      }
    }
  }, [latex, displayMode]);

  return (
    <span
      ref={mathRef}
      className={`math-renderer ${
        displayMode ? "display-math" : "inline-math"
      } ${className}`}
      style={{
        fontSize: displayMode ? "1.1em" : "inherit",
        lineHeight: displayMode ? "1.5" : "inherit",
      }}
    />
  );
}

// Hook để parse LaTeX trong text content
export function useMathParser() {
  const parseLatex = useCallback((content: string) => {
    // Pattern để tìm inline math: $...$
    const inlineMathPattern = /\$([^$]+)\$/g;
    // Pattern để tìm display math: $$...$$
    const displayMathPattern = /\$\$([^$]+)\$\$/g;

    const parts: Array<{
      type: "text" | "inline-math" | "display-math";
      content: string;
    }> = [];
    let lastIndex = 0;

    // Tìm display math trước ($$...$$)
    let match;
    const displayMatches: Array<{
      match: RegExpExecArray;
      type: "display-math";
    }> = [];
    while ((match = displayMathPattern.exec(content)) !== null) {
      displayMatches.push({ match, type: "display-math" });
    }

    // Tìm inline math ($...$)
    const inlineMatches: Array<{
      match: RegExpExecArray;
      type: "inline-math";
    }> = [];
    const inlinePattern = new RegExp(inlineMathPattern.source, "g");
    while ((match = inlinePattern.exec(content)) !== null) {
      // Kiểm tra xem match này có nằm trong display math không
      const isInDisplayMath = displayMatches.some(
        (dm) =>
          match!.index >= dm.match.index &&
          match!.index < dm.match.index + dm.match[0].length
      );
      if (!isInDisplayMath) {
        inlineMatches.push({ match, type: "inline-math" });
      }
    }

    // Combine và sort tất cả matches
    const allMatches = [...displayMatches, ...inlineMatches].sort(
      (a, b) => a.match.index - b.match.index
    );

    for (const { match, type } of allMatches) {
      // Thêm text trước math
      if (match.index > lastIndex) {
        parts.push({
          type: "text",
          content: content.slice(lastIndex, match.index),
        });
      }

      // Thêm math content
      parts.push({
        type,
        content: match[1],
      });

      lastIndex = match.index + match[0].length;
    }

    // Thêm text còn lại
    if (lastIndex < content.length) {
      parts.push({
        type: "text",
        content: content.slice(lastIndex),
      });
    }

    return parts;
  }, []);

  return { parseLatex };
}
