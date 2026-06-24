#!/usr/bin/env bash
# Phase E gate runner for finance-writer.
# Usage: bash scripts/verify-finance-post.sh <post.md> <slug> [depth]
# depth ∈ deep-dive | explainer | concept   (default: deep-dive)
# Emits PASS/FAIL/WARN lines. Any FAIL means re-enter the named phase.
#
# Superset of blog-writer's gate: same structural/visual checks, plus the three
# finance promises — TL;DR at top, a foundations section, and worked examples.

set -u
path="${1:?usage: verify-finance-post.sh <post.md> <slug> [depth]}"
slug="${2:?usage: verify-finance-post.sh <post.md> <slug> [depth]}"
depth="${3:-deep-dive}"

fail=0
pass()  { echo "PASS  $1"; }
warn()  { echo "WARN  $1"; }
fail_() { echo "FAIL  $1"; fail=1; }

# Body starts after the second '---' (end of frontmatter).
fm_end=$(grep -nE '^---[[:space:]]*$' "$path" | sed -n '2p' | cut -d: -f1)
fm_end="${fm_end:-1}"

# ── 1. Word-count gate ──────────────────────────────────────────────────────
words=$(wc -w < "$path" | tr -d ' ')
read_time=$(( (words + 110) / 220 ))
case "$depth" in
  deep-dive) min=27; min_words=6000 ;;
  explainer) min=16; min_words=3500 ;;
  concept)   min=10; min_words=2000 ;;
  *)         min=16; min_words=3500 ;;
esac
if [ "$words" -ge "$min_words" ]; then
  pass "word-count: $words words, readTime=$read_time (floor=$min min)"
else
  fail_ "word-count: $words words / readTime=$read_time below floor $min min ($min_words words). Expand thinnest sections (more worked examples, more misconceptions, deeper mechanics)."
fi

# ── 2. Diagram-count gate ────────────────────────────────────────────────────
fig_count=$(grep -cE '^!\[' "$path" || true)
case "$depth" in
  deep-dive) fig_min=5 ;;
  explainer) fig_min=3 ;;
  concept)   fig_min=2 ;;
  *)         fig_min=3 ;;
esac
if [ "$fig_count" -ge "$fig_min" ]; then
  pass "diagram-count: $fig_count figures (floor=$fig_min)"
else
  fail_ "diagram-count: $fig_count figures below floor $fig_min for depth=$depth"
fi

# ── 3. TL;DR-at-top gate (finance) ───────────────────────────────────────────
# A supported callout ([!important]/[!note]/[!info]) OR a **TL;DR** blockquote,
# within the first ~25 body lines.
tldr_window=$(awk -v S="$fm_end" 'NR>S && NR<=S+25' "$path")
if grep -qiE '^> *\[!(important|note|info)\]|^>.*\*\*TL;?DR' <<< "$tldr_window"; then
  pass "tldr-at-top: TL;DR box found near the top"
else
  fail_ "tldr-at-top: no TL;DR callout/blockquote in the first 25 body lines. Add a '> [!important]\\n> **TL;DR** — …' box as the first thing in the body."
fi

# ── 4. Foundations-section gate (finance) ────────────────────────────────────
if grep -qiE '^#{2,3} .*(foundation|fundamental|basics|first principles|primer|how .* works?|background|building blocks|what (it|is) )' "$path"; then
  pass "foundations: a basics/foundations section is present"
else
  fail_ "foundations: no foundations/basics section heading found. Add a '## Foundations: …' (or 'The basics', 'How X works', 'Building blocks') section that defines every term from zero."
fi

# ── 5. Worked-examples gate (finance) ────────────────────────────────────────
case "$depth" in
  deep-dive) ex_min=4 ;;
  explainer) ex_min=3 ;;
  concept)   ex_min=2 ;;
  *)         ex_min=3 ;;
esac
ex_markers=$(grep -icE '(worked example|example:|^#{2,4} .*\bexample\b|\bexample [0-9])' "$path" || true)
dollar_nums=$(grep -oE '\$[0-9]' "$path" | wc -l | tr -d ' ')
if [ "$ex_markers" -ge "$ex_min" ] && [ "$dollar_nums" -ge "$ex_min" ]; then
  pass "worked-examples: $ex_markers example markers, $dollar_nums \$-figures (floor=$ex_min)"
else
  fail_ "worked-examples: $ex_markers example markers / $dollar_nums \$-figures below floor $ex_min. Add concrete numeric walkthroughs ('#### Worked example:' with \$ figures, every step shown)."
fi

# ── 5b. Sourcing gate (finance) ──────────────────────────────────────────────
# Provenance for real figures. Backstops Phase D2; does NOT replace the manual
# fact-check. A deep-dive needs a Sources/References/Further-reading section OR
# inline citation links. We also count "live-number" lines (price/index/rate/
# size) that carry no nearby citation, link, or as-of date and WARN with that
# count so they can be sourced or dated. Hard FAIL only when a deep-dive has
# zero sourcing of any kind.
has_sources_section=0
grep -qiE '^#{2,3} .*(sources?|references?|further reading|citations?|bibliography)' "$path" && has_sources_section=1
inline_cites=$(grep -oE '\]\(https?://[^)]+\)' "$path" 2>/dev/null | wc -l | tr -d ' ')
# Live-number lines lacking a nearby source marker (same heuristic as extract-claims.sh).
unsourced_live=$(awk -v S="$fm_end" '
  NR<=S { next }
  { line=$0
    live = (line ~ /\$[0-9]/ || line ~ /[0-9][0-9.,]*[ ]?%/ || line ~ /[0-9][0-9.,]*[ ]?(trillion|billion|million)/ || line ~ /[0-9],[0-9][0-9][0-9]/)
    if (!live) next
    if (line ~ /^[[:space:]]*!\[/) next
    if (line ~ /^[[:space:]]*(suppose|imagine|say |take a|consider)/) next
    sourced = (line ~ /\]\(http/ || line ~ /\]\(\/blog/ || tolower(line) ~ /as.?of/ || line ~ /[Dd]ecree|[Cc]ircular|HOSE|HNX|SBV|SSC|UBCKNN|GSO|Fed|ECB|BIS|IMF|World Bank|SEC|Bloomberg|Reuters|CafeF|Vietstock/ || line ~ /\([^)]*20[0-9][0-9][^)]*\)/)
    if (!sourced) c++ }
  END { print c+0 }' "$path")
if [ "$has_sources_section" -eq 1 ] || [ "$inline_cites" -gt 0 ]; then
  pass "sourcing: provenance present (sources-section=$has_sources_section, inline citation links=$inline_cites)"
  if [ "${unsourced_live:-0}" -gt 0 ]; then
    warn "sourcing: $unsourced_live live-number line(s) lack a nearby citation/link/as-of date — re-run extract-claims.sh and source or date them (Phase D2)."
  fi
else
  if [ "$depth" = "deep-dive" ]; then
    fail_ "sourcing: deep-dive has NO sources/references section and NO inline citation links. Add a '## Sources & further reading' section + cite live figures (Phase D2 / references/fact-check.md)."
  else
    warn "sourcing: no sources section or inline citation links found; $unsourced_live live-number line(s) appear unsourced — verify per Phase D2."
  fi
fi

# ── 6. Abstraction-coverage sub-gate ─────────────────────────────────────────
missing=$(grep -nE 'imagine|think of (it|this|a|the)|picture (a|this)|consider (the|a) case|the way (this|it) works|under the hood|conceptually|in essence|in plain terms|the intuition (is|here)|abstract(ly|ion)' "$path" 2>/dev/null \
  | while IFS=: read -r line _; do
      next30=$(awk -v L="$line" 'NR>=L && NR<=L+30' "$path")
      if ! grep -q '!\[[^]]*\](/imgs/blogs/' <<< "$next30"; then
        echo "  line $line: $(awk -v L=$line 'NR==L' "$path" | cut -c1-80)"
      fi
    done)
if [ -z "$missing" ]; then
  pass "abstraction-coverage: every prose abstraction has a nearby figure"
else
  fail_ "abstraction-coverage: prose abstractions without a figure within 30 lines:"
  echo "$missing"
fi

# ── 7. Sharpness sub-gate ────────────────────────────────────────────────────
sharp_fail=0
for f in public/imgs/blogs/${slug}-*.webp; do
  [ -e "$f" ] || continue
  # Skip the 800x450 card thumbnail that `npm run optimize-blog-images` derives
  # (<slug>-N.cover.webp) — it is intentionally small, not a figure.
  case "$f" in *.cover.webp) continue ;; esac
  if command -v sips >/dev/null 2>&1; then
    w=$(sips -g pixelWidth  "$f" 2>/dev/null | awk '/pixelWidth/  {print $2}')
    h=$(sips -g pixelHeight "$f" 2>/dev/null | awk '/pixelHeight/ {print $2}')
  else
    read w h < <(identify -format "%w %h" "$f" 2>/dev/null || echo "0 0")
  fi
  bytes=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f")
  if [ "${w:-0}" -lt 1600 ] || [ "${h:-0}" -lt 900 ] || [ "${bytes:-0}" -lt 40960 ]; then
    fail_ "sharpness: $f is ${w}x${h} ${bytes}B (need >=1600x900, >=40KB WebP)"
    sharp_fail=1
  fi
done
[ "$sharp_fail" -eq 0 ] && pass "sharpness: all WebP figures ≥ 1600×900, ≥ 40 KB"

# ── 8. Format gate — no stray non-webp renders for this slug ──────────────────
stray=$(ls public/imgs/blogs/${slug}-*.png public/imgs/blogs/${slug}-*.jpg \
           public/imgs/blogs/${slug}-*.jpeg public/imgs/blogs/${slug}-*.gif 2>/dev/null || true)
if [ -z "$stray" ]; then
  pass "format: no leftover non-webp renders for slug=$slug"
else
  fail_ "format: non-webp render artifacts left in public/imgs/blogs (delete or convert):"
  echo "$stray" | sed 's/^/  /'
fi

# ── 9. Forbidden text-diagram substitutes ────────────────────────────────────
sub_fail=0
grep -nE '^```text'         "$path" >/dev/null 2>&1 && { fail_ "forbidden: \`\`\`text fenced 'diagrams'"; sub_fail=1; }
grep -nE '[│┌┐└┘├┤┬┴┼─]'    "$path" >/dev/null 2>&1 && { fail_ "forbidden: Unicode box-drawing"; sub_fail=1; }
grep -nE '\+--+\+|--->|<---' "$path" >/dev/null 2>&1 && { fail_ "forbidden: ASCII-art arrows/boxes"; sub_fail=1; }
grep -nE '^```mermaid'       "$path" >/dev/null 2>&1 && { fail_ "forbidden: inline \`\`\`mermaid block (must be rendered to WebP)"; sub_fail=1; }
[ "$sub_fail" -eq 0 ] && pass "no-text-diagrams: no ASCII/Unicode/mermaid substitutes"

# ── 10. Slug-match gate ──────────────────────────────────────────────────────
foreign=$(grep -oE '!\[[^]]*\]\(/imgs/blogs/[^)]+\)' "$path" 2>/dev/null \
          | grep -v "/imgs/blogs/${slug}-" || true)
if [ -z "$foreign" ]; then
  pass "slug-match: all images use slug=$slug"
else
  fail_ "slug-match: foreign image paths (typo or stale reference):"
  echo "$foreign" | sed 's/^/  /'
fi

# ── 11. WebP-only gate ───────────────────────────────────────────────────────
nonwebp=$(grep -oE '!\[[^]]*\]\(/imgs/blogs/[^)]+\)' "$path" 2>/dev/null \
          | grep -ivE '\.webp\)$' || true)
if [ -z "$nonwebp" ]; then
  pass "webp-only: all embedded images are .webp"
else
  fail_ "webp-only: non-webp image embeds (every figure must ship as .webp):"
  echo "$nonwebp" | sed 's/^/  /'
fi

# ── 12. No-H1 gate ───────────────────────────────────────────────────────────
if grep -nE '^# [^#]' "$path" >/dev/null 2>&1; then
  fail_ "no-H1: body contains '# ' headings (must be ##):"
  grep -nE '^# [^#]' "$path" | sed 's/^/  /'
else
  pass "no-H1: body has no single-# headings"
fi

# ── 13. English-only gate ────────────────────────────────────────────────────
non_ascii=$(grep -nP '[\x{00C0}-\x{1EF9}\x{4E00}-\x{9FFF}]' "$path" 2>/dev/null | head -10 || true)
if [ -z "$non_ascii" ]; then
  pass "english-only: no Vietnamese/CJK characters in body"
else
  warn "english-only: non-ASCII letters detected (review — may be legitimate proper nouns):"
  echo "$non_ascii" | sed 's/^/  /'
fi

# ── 14. Frontmatter sanity ───────────────────────────────────────────────────
fm_block=$(awk '/^---$/{c++; next} c==1' "$path")
grep -q "^date:"        <<< "$fm_block" || fail_ "frontmatter: missing 'date'"
grep -q "^tags:"        <<< "$fm_block" || fail_ "frontmatter: missing 'tags'"
grep -q "^category:"    <<< "$fm_block" || fail_ "frontmatter: missing 'category'"
grep -qE "^category:[[:space:]]*[\"']?trading" <<< "$fm_block" || warn "frontmatter: category is not 'trading' (finance-writer routes under content/blog/trading/)"
declared_rt=$(grep -E "^readTime:" <<< "$fm_block" | sed -E 's/.*: *([0-9]+).*/\1/' | head -1 || true)
if [ -n "${declared_rt:-}" ] && [ "${declared_rt}" != "$read_time" ]; then
  warn "frontmatter: declared readTime=$declared_rt but recomputed=$read_time"
fi

echo ""
if [ "$fail" -eq 0 ]; then
  echo "RESULT: all gates passed (words=$words, readTime=$read_time, figures=$fig_count, examples=$ex_markers)"
  exit 0
else
  echo "RESULT: gates FAILED — re-enter the named phase and fix"
  exit 1
fi
