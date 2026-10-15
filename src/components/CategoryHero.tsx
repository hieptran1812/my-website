import React from "react";

export interface HeroStat {
  value: string | number;
  label: string;
}

interface CategoryHeroProps {
  /** Small caps line above the title, e.g. "Blog · Category". */
  eyebrow: string;
  title: string;
  description: string;
  /** Category accent, used only for the eyebrow dot and the rule. */
  accent: string;
  stats?: HeroStat[];
}

/**
 * Editorial header shared by the blog listing pages: a quiet eyebrow, a large
 * tightly tracked title in the text colour, one short accent rule and a
 * balanced description, with an optional row of facts. Colour is used as a
 * mark, never as a fill, so it sits calmly on the grain canvas in both themes.
 */
export default function CategoryHero({
  eyebrow,
  title,
  description,
  accent,
  stats = [],
}: CategoryHeroProps) {
  const shown = stats.filter((s) => s.value !== "" && String(s.value) !== "0");
  return (
    <header className="text-center mb-16 md:mb-20 pt-2">
      <div
        className="inline-flex items-center gap-2 text-[11px] md:text-xs font-medium uppercase tracking-[0.2em]"
        style={{ color: "var(--text-secondary)" }}
      >
        <span
          aria-hidden
          className="w-1.5 h-1.5 rounded-full"
          style={{ backgroundColor: accent }}
        />
        {eyebrow}
      </div>

      <h1
        className="mt-5 text-[2.75rem] leading-[1.05] sm:text-6xl md:text-7xl font-semibold tracking-[-0.035em]"
        style={{ color: "var(--text-primary)" }}
      >
        {title}
      </h1>

      <div
        aria-hidden
        className="mx-auto mt-6 h-px w-12"
        style={{ backgroundColor: accent, opacity: 0.8 }}
      />

      <p
        className="mt-6 mx-auto max-w-2xl text-base sm:text-lg md:text-xl leading-relaxed"
        style={{ color: "var(--text-secondary)", textWrap: "balance" }}
      >
        {description}
      </p>

      {shown.length > 0 && (
        <div className="mt-8 flex flex-wrap items-center justify-center gap-x-6 gap-y-2 text-sm">
          {shown.map((s, i) => (
            <div key={s.label} className="flex items-center gap-6">
              {i > 0 && (
                <span
                  aria-hidden
                  className="hidden sm:block w-px h-4"
                  style={{ backgroundColor: "var(--border)" }}
                />
              )}
              <div className="flex items-baseline gap-1.5">
                <span
                  className="font-semibold tabular-nums"
                  style={{ color: "var(--text-primary)" }}
                >
                  {typeof s.value === "number"
                    ? s.value.toLocaleString("en-US")
                    : s.value}
                </span>
                <span style={{ color: "var(--text-secondary)" }}>
                  {s.label}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </header>
  );
}
