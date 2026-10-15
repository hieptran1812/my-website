# Voice cheatsheet

Read at the start of Phase D. This replaces "read 80 lines from the gold-standard post": distilled actual moves to imitate.

## Core voice

- **Principal-engineer voice.** Opinionated, war-story-flavored, intuition-first. Not a textbook.
- **First-person plural (`we`)** for shared reasoning. **First-person singular (`I have personally debugged…`)** only for war stories.
- **Always English.** Title, frontmatter, body, code comments and captions are all English, regardless of how the user invoked the skill.

## Opening moves

- Open with the **real problem** or a sharp mismatch, never a dictionary definition.
- Build intuition with a **concrete analogy** (library, restaurant, city map, kitchen line) before introducing math.
- Reference the first ("mental model") figure in the intro: *"The diagram above is the mental model: …"*

## Section moves

- `##` for top-level sections, `###` for sub-sections, `####` for sub-sub. **Never `#` in the body**: frontmatter `title` becomes the H1.
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

If short, expand the weakest sections: more case studies, deeper internals, more code, more tables. Do not ship short.

## Diagram embedding

- `![alt text](/imgs/blogs/<slug>-<n>.webp)` directly under the heading or paragraph that introduces the concept. Images are always `.webp`, never `.png`/`.jpg`/`.svg`.
- Cross-links use relative paths without `content/` or `.md`: `[KV cache](/blog/machine-learning/large-language-model/kv-cache)`.

## Punctuation: no em dashes (hard gate)

**No em dash ships in a post.** Not in the body, not in a heading, not in a caption, not in the frontmatter `title` or `description`. `verify-post.sh` fails the post if one survives, so this is a gate, not a preference.

What counts as one:

| Mark | Verdict |
| --- | --- |
| `—` em dash | **Banned**, everywhere in prose |
| ` – ` spaced en dash | **Banned**: the same mark wearing a disguise |
| `–` unspaced en dash | **Fine**, a range: `2018–2022`, `p90–p99`, `3–4×` |
| ` -- ` spaced double hyphen | **Banned**: the Substack converter turns it into an em dash |
| `--flag`, `-` hyphenated words | **Fine**, not punctuation |

Code fences, inline code and `$...$` math are exempt. `npm run x -- article` is a command line, not a sentence.

### Repair it, do not just delete it

Swapping every dash for a comma is how you turn one tell into a worse error. Pick by what sits on each side:

| Shape | Repair | Example |
| --- | --- | --- |
| Both sides are independent clauses | **Period.** The strongest default | `The gate passed — the figure was still empty.` → `The gate passed. The figure was still empty.` |
| Second half explains or names the first | **Colon** | `Only one number matters — median context.` → `Only one number matters: median context.` |
| Aside in the middle of a sentence | **Two commas**, or parentheses if the aside already has commas | `Cache read, the largest line item, is billed at 0.1×.` |
| Trailing appositive or list tail | **Comma** | `..., which is the whole scoreboard.` |

A comma between two independent clauses is a **comma splice**, and it reads worse than the dash you removed. If neither a comma nor a colon fits, the sentence was two sentences all along, which is usually why the dash was there. Write the two sentences.

## Don'ts

- No generic conclusions, no "in summary" tables of contents, no AI throat-clearing.
- No emojis.
- No em dashes. See **Punctuation** above; the verify gate enforces it.
- No "as we discussed earlier" filler.
- No ASCII art or Unicode box-drawing diagrams. Real Excalidraw PNGs only.
