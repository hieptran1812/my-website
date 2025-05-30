"use client";

import React, { useEffect, useRef } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";

interface MathJaxProps {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
}

export default function MathJax({
  children,
  className = "",
  style,
}: MathJaxProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const processMath = () => {
      const element = containerRef.current;
      if (!element) return;

      // Find all text nodes
      const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT, {
        acceptNode: (node) => {
          const parent = node.parentNode as HTMLElement;
          // Skip already processed math or script elements
          if (
            parent?.classList?.contains("katex") ||
            parent?.classList?.contains("math-expression") ||
            parent?.tagName === "SCRIPT"
          ) {
            return NodeFilter.FILTER_REJECT;
          }
          return NodeFilter.FILTER_ACCEPT;
        },
      });

      const textNodes: Text[] = [];
      let node;
      while ((node = walker.nextNode())) {
        textNodes.push(node as Text);
      }

      textNodes.forEach((textNode) => {
        const content = textNode.textContent || "";
        if (!content.includes("$")) return;

        let processedContent = content;
        let hasChanges = false;

        // Process display math ($$...$$)
        processedContent = processedContent.replace(
          /\$\$([^$]*(?:\$(?!\$)[^$]*)*)\$\$/g,
          (match, latex) => {
            hasChanges = true;
            try {
              const rendered = katex.renderToString(latex.trim(), {
                displayMode: true,
                throwOnError: false,
                errorColor: "#dc2626",
                strict: "warn",
                trust: false,
                macros: {
                  "\\RR": "\\mathbb{R}",
                  "\\NN": "\\mathbb{N}",
                  "\\ZZ": "\\mathbb{Z}",
                  "\\QQ": "\\mathbb{Q}",
                  "\\CC": "\\mathbb{C}",
                },
              });
              return `<div class="math-expression display-math">${rendered}</div>`;
            } catch (error) {
              console.error("KaTeX display error:", error);
              return `<div class="math-error">${match}</div>`;
            }
          }
        );

        // Process inline math ($...$)
        processedContent = processedContent.replace(
          /(?<!\$)\$([^$\n]*(?:[^$\n]|\\\$)*)\$(?!\$)/g,
          (match, latex) => {
            hasChanges = true;
            try {
              const rendered = katex.renderToString(latex.trim(), {
                displayMode: false,
                throwOnError: false,
                errorColor: "#dc2626",
                strict: "warn",
                trust: false,
                macros: {
                  "\\RR": "\\mathbb{R}",
                  "\\NN": "\\mathbb{N}",
                  "\\ZZ": "\\mathbb{Z}",
                  "\\QQ": "\\mathbb{Q}",
                  "\\CC": "\\mathbb{C}",
                },
              });
              return `<span class="math-expression inline">${rendered}</span>`;
            } catch (error) {
              console.error("KaTeX inline error:", error);
              return `<span class="math-error">${match}</span>`;
            }
          }
        );

        if (hasChanges) {
          const tempDiv = document.createElement("div");
          tempDiv.innerHTML = processedContent;

          const fragment = document.createDocumentFragment();
          while (tempDiv.firstChild) {
            fragment.appendChild(tempDiv.firstChild);
          }

          textNode.parentNode?.replaceChild(fragment, textNode);
        }
      });
    };

    // Process math after a small delay to ensure DOM is ready
    const timer = setTimeout(processMath, 50);

    // Set up mutation observer for dynamic content
    const observer = new MutationObserver((mutations) => {
      let shouldProcess = false;
      mutations.forEach((mutation) => {
        if (mutation.type === "childList" && mutation.addedNodes.length > 0) {
          shouldProcess = true;
        }
      });
      if (shouldProcess) {
        setTimeout(processMath, 10);
      }
    });

    observer.observe(containerRef.current, {
      childList: true,
      subtree: true,
    });

    return () => {
      clearTimeout(timer);
      observer.disconnect();
    };
  }, [children]);

  return (
    <div ref={containerRef} className={className} style={style}>
      {children}
    </div>
  );
}
