"use client";

import { useEffect } from "react";

/**
 * Walks the rendered article DOM and decorates every <pre><code> with:
 *   - a language label (top-left)
 *   - a copy button (top-right, visible on hover)
 *
 * Idempotent: re-runs are safe; previously decorated blocks are skipped.
 */
export default function CodeBlockEnhancer({
  containerSelector = ".blog-content",
}: {
  containerSelector?: string;
}) {
  useEffect(() => {
    const root =
      document.querySelector<HTMLElement>(containerSelector) || document.body;

    const decorate = () => {
      const pres = root.querySelectorAll<HTMLPreElement>("pre");
      pres.forEach((pre) => {
        if (pre.dataset.enhanced === "1") return;
        const code = pre.querySelector("code");
        if (!code) return;
        pre.dataset.enhanced = "1";

        // Language label from rehype-highlight class (e.g. "language-python" or "hljs language-ts").
        const langClass = Array.from(code.classList).find((c) =>
          c.startsWith("language-"),
        );
        const lang = langClass ? langClass.replace("language-", "") : "";
        if (lang) {
          const label = document.createElement("span");
          label.className = "code-lang-label";
          label.textContent = lang;
          pre.appendChild(label);
        }

        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "code-copy-btn";
        btn.textContent = "Copy";
        btn.setAttribute("aria-label", "Copy code");
        btn.addEventListener("click", async () => {
          const text = code.innerText;
          try {
            await navigator.clipboard.writeText(text);
            btn.textContent = "Copied";
            btn.dataset.copied = "true";
            setTimeout(() => {
              btn.textContent = "Copy";
              delete btn.dataset.copied;
            }, 1600);
          } catch {
            btn.textContent = "Failed";
            setTimeout(() => (btn.textContent = "Copy"), 1600);
          }
        });
        pre.appendChild(btn);
      });
    };

    decorate();
    // Re-run if article content swaps (e.g. client-side nav).
    const observer = new MutationObserver(() => decorate());
    observer.observe(root, { childList: true, subtree: true });
    return () => observer.disconnect();
  }, [containerSelector]);

  return null;
}
