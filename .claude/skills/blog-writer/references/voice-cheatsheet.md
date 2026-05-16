# Voice cheatsheet

Read at the start of Phase D. This replaces "read 80 lines from the gold-standard post" — distilled actual moves to imitate.

## Core voice

- **Principal-engineer voice.** Opinionated, war-story-flavored, intuition-first. Not a textbook.
- **First-person plural (`we`)** for shared reasoning. **First-person singular (`I have personally debugged…`)** only for war stories.
- **Always English.** Title, frontmatter, body, code comments, captions — all English. Regardless of how the user invoked the skill.

## Opening moves

- Open with the **real problem** or a sharp mismatch — never a dictionary definition.
- Build intuition with a **concrete analogy** (library, restaurant, city map, kitchen line) before introducing math.
- Reference the first ("mental model") figure in the intro: *"The diagram above is the mental model: …"*

## Section moves

- `##` for top-level sections, `###` for sub-sections, `####` for sub-sub. **Never `#` in the body** — frontmatter `title` becomes the H1.
- Numbered `## 1. …` `## 2. …` is fine for deep-dives, not for explainers.
- Every section answers: *why does this work, when does it fail, what are the second-order consequences?*
- For every claim: name the mechanism, quantify the tradeoff, give at least one concrete number, benchmark, or failure mode.
- Each H2 should contain **at least one of**: comparison table, runnable code block (≥ 15 lines), measured benchmark with units, or worked numerical example. Pure-prose sections are a smell.

## Math & code

- Math in `$...$` / `$$...$$`; define each symbol on first use.
- Code blocks look **runnable**: real imports, real flags, real version numbers. Avoid pseudocode unless explicitly labeled.

## Tables

Use comparison tables aggressively:
- "naive vs optimized"
- "assumption vs reality"
- "strategy / when to use / tradeoff"

## Closing moves

- **Deep-dives end with case studies**: 6–12 named, numbered incidents (~250–400 words each), then a closing **"When to reach for X / when not to"** section.
- **Never write a generic "Conclusion".**

## Length floors (hard gates, not warnings)

- Deep-dive: **≥ 50 min read** (≥ 11,000 words; target 12k–16k)
- Explainer: ≥ 25 min (~5,500 words)
- Paper-reading: ≥ 30 min (~6,500 words)

If short, expand the weakest sections — more case studies, deeper internals, more code, more tables. Do not ship short.

## Diagram embedding

- `![alt text](/imgs/blogs/<slug>-<n>.png)` directly under the heading or paragraph that introduces the concept.
- Cross-links use relative paths without `content/` or `.md`: `[KV cache](/blog/machine-learning/large-language-model/kv-cache)`.

## Don'ts

- No generic conclusions, no "in summary" tables of contents, no AI throat-clearing.
- No emojis.
- No "as we discussed earlier" filler.
- No ASCII art or Unicode box-drawing diagrams. Real Excalidraw PNGs only.
