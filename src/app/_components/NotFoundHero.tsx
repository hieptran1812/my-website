"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

const QUIPS = [
  "This page took a sabbatical.",
  "404: page is on PTO.",
  "We reorganized. The page did not survive.",
  "Even the sitemap shrugged.",
  "The page got rebased into oblivion.",
  "Searched 47 routes, 3 dimensions, 1 fridge. Nothing.",
];

export default function NotFoundHero({ surpriseHref }: { surpriseHref?: string }) {
  const [quip, setQuip] = useState(QUIPS[0]);

  useEffect(() => {
    setQuip(QUIPS[Math.floor(Math.random() * QUIPS.length)]);
  }, []);

  return (
    <div
      className="relative w-full max-w-2xl mx-auto rounded-2xl border overflow-hidden"
      style={{
        backgroundColor: "var(--surface)",
        borderColor: "var(--border)",
        boxShadow: "0 10px 40px -20px rgba(0,0,0,0.25)",
      }}
    >
      <div
        className="absolute inset-0 pointer-events-none opacity-60"
        style={{
          background:
            "radial-gradient(circle at 20% 0%, var(--surface-accent), transparent 60%)",
        }}
      />

      <div className="relative px-6 sm:px-10 py-12 sm:py-14 text-center">
        <p
          className="text-xs font-mono mb-4 uppercase tracking-[0.25em]"
          style={{ color: "var(--text-secondary)" }}
        >
          Error · 404 · Page Not Found
        </p>

        <h1
          className="glitch-404 font-mono font-black leading-none mb-6 select-none"
          data-text="404"
          style={{
            fontSize: "clamp(5rem, 18vw, 10rem)",
            backgroundImage:
              "linear-gradient(135deg, var(--accent), var(--accent-light))",
            WebkitBackgroundClip: "text",
            backgroundClip: "text",
            color: "transparent",
            letterSpacing: "-0.05em",
          }}
        >
          404
        </h1>

        <p
          className="text-lg sm:text-xl font-medium mb-2"
          style={{ color: "var(--text-primary)" }}
        >
          {quip}
          <span className="caret-404" aria-hidden>
            ▍
          </span>
        </p>

        <p
          className="text-sm max-w-md mx-auto mb-8"
          style={{ color: "var(--text-secondary)" }}
        >
          The link may have moved when posts were reorganized into category
          folders, or it never existed. Try one of the routes below.
        </p>

        <div
          className="font-mono text-left text-xs sm:text-sm rounded-lg border mb-8 overflow-hidden"
          style={{
            backgroundColor: "var(--surface-accent)",
            borderColor: "var(--border)",
            color: "var(--text-secondary)",
          }}
        >
          <div
            className="flex items-center gap-1.5 px-3 py-2 border-b"
            style={{ borderColor: "var(--border)" }}
          >
            <span
              className="w-2.5 h-2.5 rounded-full"
              style={{ backgroundColor: "#ff5f57" }}
            />
            <span
              className="w-2.5 h-2.5 rounded-full"
              style={{ backgroundColor: "#febc2e" }}
            />
            <span
              className="w-2.5 h-2.5 rounded-full"
              style={{ backgroundColor: "#28c840" }}
            />
            <span className="ml-2 opacity-70">~/where-am-i</span>
          </div>
          <pre className="px-4 py-3 leading-relaxed whitespace-pre-wrap">
            <span style={{ color: "var(--accent)" }}>$</span> locate page{"\n"}
            <span className="opacity-70">→ searched 47 routes, 3 dimensions, 1 fridge</span>{"\n"}
            <span className="opacity-70">→ not found. last seen: blaming the intern</span>{"\n"}
            <span style={{ color: "var(--accent)" }}>$</span> exit 404
          </pre>
        </div>

        <div className="flex flex-wrap gap-3 justify-center">
          <Link
            href="/blog"
            className="px-5 py-2.5 rounded-lg font-medium transition-transform hover:-translate-y-0.5"
            style={{ backgroundColor: "var(--accent)", color: "white" }}
          >
            Browse the blog
          </Link>
          <Link
            href="/"
            className="px-5 py-2.5 rounded-lg font-medium border transition-transform hover:-translate-y-0.5"
            style={{
              borderColor: "var(--border)",
              color: "var(--text-primary)",
            }}
          >
            Go home
          </Link>
          {surpriseHref ? (
            <Link
              href={surpriseHref}
              className="px-5 py-2.5 rounded-lg font-medium border transition-transform hover:-translate-y-0.5"
              style={{
                borderColor: "var(--accent)",
                color: "var(--accent)",
              }}
            >
              🎲 Surprise me
            </Link>
          ) : null}
        </div>

        <p
          className="mt-6 text-xs"
          style={{ color: "var(--text-secondary)" }}
        >
          Tip: press <kbd className="cmdk-kbd">⌘</kbd>
          <kbd className="cmdk-kbd">K</kbd> to search anywhere on the site.
        </p>
      </div>
    </div>
  );
}
