"use client";

import { useEffect, useRef } from "react";
import katex from "katex";

interface MathParserOptions {
  displayMathDelimiters?: [string, string][];
  inlineMathDelimiters?: [string, string][];
  processEscapes?: boolean;
}

export function useBlogMathParser(options: MathParserOptions = {}) {
  const {
    displayMathDelimiters = [
      ["$$", "$$"],
      ["\\[", "\\]"],
    ],
    inlineMathDelimiters = [
      ["$", "$"],
      ["\\(", "\\)"],
    ],
    processEscapes = true,
  } = options;

  const processElement = (element: HTMLElement) => {
    // Process display math first
    for (const [open, close] of displayMathDelimiters) {
      const regex = new RegExp(
        escapeRegExp(open) +
          "((?:[^" +
          escapeRegExp(close[0]) +
          "]|" +
          escapeRegExp(close[0]) +
          "(?!" +
          escapeRegExp(close.slice(1)) +
          "))*)" +
          escapeRegExp(close),
        "g"
      );

      element.innerHTML = element.innerHTML.replace(regex, (match, latex) => {
        try {
          return `<div class="math-expression display-math">${katex.renderToString(
            latex,
            {
              displayMode: true,
              throwOnError: false,
              errorColor: "#cc0000",
              strict: "warn",
              trust: false,
            }
          )}</div>`;
        } catch (error) {
          console.error("KaTeX display math error:", error);
          return `<div class="math-error" style="color: #cc0000;">${match}</div>`;
        }
      });
    }

    // Process inline math
    for (const [open, close] of inlineMathDelimiters) {
      const regex = new RegExp(
        escapeRegExp(open) +
          "((?:[^" +
          escapeRegExp(close[0]) +
          "]|" +
          escapeRegExp(close[0]) +
          "(?!" +
          escapeRegExp(close.slice(1)) +
          "))*)" +
          escapeRegExp(close),
        "g"
      );

      element.innerHTML = element.innerHTML.replace(regex, (match, latex) => {
        try {
          return `<span class="math-expression inline">${katex.renderToString(
            latex,
            {
              displayMode: false,
              throwOnError: false,
              errorColor: "#cc0000",
              strict: "warn",
              trust: false,
            }
          )}</span>`;
        } catch (error) {
          console.error("KaTeX inline math error:", error);
          return `<span class="math-error" style="color: #cc0000;">${match}</span>`;
        }
      });
    }
  };

  const processBlogContent = (contentRef: React.RefObject<HTMLElement>) => {
    if (!contentRef.current) return;

    // Find all text nodes and process them
    const walker = document.createTreeWalker(
      contentRef.current,
      NodeFilter.SHOW_TEXT,
      null
    );

    const textNodes: Text[] = [];
    let node;
    while ((node = walker.nextNode())) {
      textNodes.push(node as Text);
    }

    // Process each text node
    textNodes.forEach((textNode) => {
      const parent = textNode.parentElement;
      if (!parent || parent.classList.contains("math-expression")) return;

      const content = textNode.textContent || "";
      if (containsMath(content)) {
        const tempDiv = document.createElement("div");
        tempDiv.innerHTML = content;
        processElement(tempDiv);

        // Replace the text node with processed content
        const fragment = document.createDocumentFragment();
        while (tempDiv.firstChild) {
          fragment.appendChild(tempDiv.firstChild);
        }
        parent.replaceChild(fragment, textNode);
      }
    });
  };

  const containsMath = (text: string): boolean => {
    for (const [open, close] of [
      ...displayMathDelimiters,
      ...inlineMathDelimiters,
    ]) {
      if (text.includes(open) && text.includes(close)) {
        return true;
      }
    }
    return false;
  };

  return { processBlogContent };
}

function escapeRegExp(string: string): string {
  return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function useMathProcessor() {
  const mathProcessorRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!mathProcessorRef.current) return;

    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.type === "childList") {
          mutation.addedNodes.forEach((node) => {
            if (node.nodeType === Node.ELEMENT_NODE) {
              const element = node as HTMLElement;
              processMathInElement(element);
            }
          });
        }
      });
    });

    observer.observe(mathProcessorRef.current, {
      childList: true,
      subtree: true,
    });

    // Initial processing
    processMathInElement(mathProcessorRef.current);

    return () => observer.disconnect();
  }, []);

  return mathProcessorRef;
}

function processMathInElement(element: HTMLElement) {
  // Auto-render math using KaTeX auto-render
  if (typeof window !== "undefined" && element) {
    // Simple math processing for $...$ and $$...$$
    const textContent = element.textContent || "";

    // Replace display math $$...$$
    element.innerHTML = element.innerHTML.replace(
      /\$\$([^$]+)\$\$/g,
      (match, latex) => {
        try {
          return `<div class="math-expression display-math">${katex.renderToString(
            latex.trim(),
            {
              displayMode: true,
              throwOnError: false,
              errorColor: "#cc0000",
            }
          )}</div>`;
        } catch (error) {
          return `<div class="math-error" style="color: #cc0000;">${match}</div>`;
        }
      }
    );

    // Replace inline math $...$
    element.innerHTML = element.innerHTML.replace(
      /\$([^$]+)\$/g,
      (match, latex) => {
        try {
          return `<span class="math-expression inline">${katex.renderToString(
            latex.trim(),
            {
              displayMode: false,
              throwOnError: false,
              errorColor: "#cc0000",
            }
          )}</span>`;
        } catch (error) {
          return `<span class="math-error" style="color: #cc0000;">${match}</span>`;
        }
      }
    );
  }
}
