#!/usr/bin/env bash
# Phase E gate runner for paper-writer. Usage: verify-paper-post.sh <post.md> <slug>
# Emits PASS/FAIL/WARN lines. Any FAIL means re-enter the named phase.
#
# Same shape as blog-writer's verify-post.sh, with two differences:
#   - sharpness is split into two tiers: extracted originals (<slug>-fig<n>.webp,
#     relaxed floor) vs redrawn Excalidraw diagrams (<slug>-<n>.webp, strict floor)
#   - adds paper-specific gates: TL;DR, extracted+redrawn floors, display-math,
#     References + paper URL, "what would change my mind"
set -u
path="${1:?usage: verify-paper-post.sh <post.md> <slug>}"
slug="${2:?usage: verify-paper-post.sh <post.md> <slug>}"

fail=0
pass()  { echo "PASS  $1"; }
warn()  { echo "WARN  $1"; }
fail_() { echo "FAIL  $1"; fail=1; }
code_stripped() { awk '/^```/{f=!f; next} !f' "$path"; }
prose_only() {
  awk '/^```/{f=!f; next} f{next}
       /^<figure class="blog-anim"/{a=1} a{ if(/^<\/figure>/) a=0; next } 1' "$path"
}
anim_count=$(grep -cE '^<figure class="blog-anim"' "$path" || true)

# 1. Word count gate (paper analyses are deep — higher floor than a normal paper-reading post)
words=$(wc -w < "$path" | tr -d ' ')
read_time=$(( (words + 110) / 220 ))
min=30; min_words=6500
if [ "$words" -ge "$min_words" ]; then
  pass "word-count: $words words, readTime=$read_time (floor=$min)"
else
  fail_ "word-count: $words words / readTime=$read_time below floor $min ($min_words words). Expand the thinnest technique sections."
fi

# 2. Figure gate — total count + minimums
#    img_count counts markdown image embeds; anim_count counts inline animated figures.
img_count=$(grep -cE '^!\[' "$path" || true)
fig_count=$(( img_count + anim_count ))
fig_min=6
if [ "$fig_count" -ge "$fig_min" ]; then
  pass "figure-count: $fig_count figures ($img_count image + $anim_count animated, floor=$fig_min)"
else
  fail_ "figure-count: $fig_count figures ($img_count image + $anim_count animated) below floor $fig_min"
fi

# 2b. Two-track floor — extracted originals AND redrawn diagrams must both be present
extracted_embeds=$(grep -oE '!\[[^]]*\]\(/imgs/blogs/'"${slug}"'-fig[0-9]+\.webp\)' "$path" 2>/dev/null | wc -l | tr -d ' ')
redrawn_embeds=$(grep -oE '!\[[^]]*\]\(/imgs/blogs/'"${slug}"'-[0-9]+\.webp\)' "$path" 2>/dev/null | wc -l | tr -d ' ')
if [ "$extracted_embeds" -ge 2 ]; then
  pass "extracted-figures: $extracted_embeds original figure(s) cut from the paper (floor=2)"
else
  fail_ "extracted-figures: only $extracted_embeds <slug>-fig<n>.webp embedded (floor=2). Extract the paper's own figures in Phase C1."
fi
if [ "$redrawn_embeds" -ge 2 ]; then
  pass "redrawn-figures: $redrawn_embeds clarifying Excalidraw diagram(s) (floor=2)"
else
  fail_ "redrawn-figures: only $redrawn_embeds <slug>-<n>.webp embedded (floor=2). Author clarifying diagrams in Phase C2."
fi

# 3. Abstraction-coverage — prose abstractions without a nearby figure (extracted OR redrawn OR animated)
missing=$(grep -nE 'imagine|think of (it|this) as|consider (the|a) case|the way (this|it) works|under the hood|conceptually|in essence|intuitively|abstract(ly|ion)' "$path" 2>/dev/null \
  | while IFS=: read -r line _; do
      next30=$(awk -v L="$line" 'NR>=L && NR<=L+30' "$path")
      if ! grep -qE '!\[[^]]*\]\(/imgs/blogs/|<figure class="blog-anim"' <<< "$next30"; then
        echo "  line $line: $(awk -v L=$line 'NR==L' "$path" | cut -c1-80)"
      fi
    done)
if [ -z "$missing" ]; then
  pass "abstraction-coverage: every prose abstraction has a nearby figure"
else
  fail_ "abstraction-coverage: prose abstractions without figure within 30 lines:"
  echo "$missing"
fi

# 4. Sharpness — TWO tiers.
webp_dims() {
  if command -v sips >/dev/null 2>&1; then
    local w h
    w=$(sips -g pixelWidth  "$1" 2>/dev/null | awk '/pixelWidth/  {print $2}')
    h=$(sips -g pixelHeight "$1" 2>/dev/null | awk '/pixelHeight/ {print $2}')
    echo "${w:-0} ${h:-0}"
  else
    identify -format "%w %h" "$1" 2>/dev/null || echo "0 0"
  fi
}
# 4a. Extracted originals: relaxed floor (paper figures aren't 16:9). ≥900 px long side, ≥20 KB.
ex_fail=0; ex_seen=0
for f in public/imgs/blogs/${slug}-fig[0-9]*.webp; do
  [ -e "$f" ] || continue
  ex_seen=1
  read w h < <(webp_dims "$f")
  bytes=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f")
  long=$(( w > h ? w : h ))
  if [ "${long:-0}" -lt 900 ] || [ "${bytes:-0}" -lt 20480 ]; then
    fail_ "sharpness(extracted): $f is ${w:-0}×${h:-0} ${bytes:-0}B (need long side ≥900, ≥20KB — raise --dpi and re-extract)"
    ex_fail=1
  fi
done
[ "$ex_seen" -eq 1 ] && [ "$ex_fail" -eq 0 ] && pass "sharpness(extracted): all -fig<n> WebP ≥ 900 px long side, ≥ 20 KB"
# 4b. Redrawn diagrams: strict blog-writer floor. ≥1600×900, ≥40 KB.
rd_fail=0; rd_seen=0
for f in public/imgs/blogs/${slug}-[0-9]*.webp; do
  [ -e "$f" ] || continue
  rd_seen=1
  read w h < <(webp_dims "$f")
  bytes=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f")
  if [ "${w:-0}" -lt 1600 ] || [ "${h:-0}" -lt 900 ] || [ "${bytes:-0}" -lt 40960 ]; then
    fail_ "sharpness(redrawn): $f is ${w:-0}×${h:-0} ${bytes:-0}B (need ≥1600×900, ≥40KB)"
    rd_fail=1
  fi
done
[ "$rd_seen" -eq 1 ] && [ "$rd_fail" -eq 0 ] && pass "sharpness(redrawn): all -<n> WebP ≥ 1600×900, ≥ 40 KB"

# 4c. Format gate — no stray non-webp renders for this slug
stray=$(ls public/imgs/blogs/${slug}-*.png public/imgs/blogs/${slug}-*.jpg \
           public/imgs/blogs/${slug}-*.jpeg public/imgs/blogs/${slug}-*.gif 2>/dev/null || true)
if [ -z "$stray" ]; then
  pass "format: no leftover non-webp renders for slug=$slug"
else
  fail_ "format: non-webp render artifacts left in public/imgs/blogs (delete or convert):"
  echo "$stray" | sed 's/^/  /'
fi

# 5. Forbidden text-diagram substitutes
sub_fail=0
grep -nE '^```text'                      "$path" >/dev/null 2>&1 && { fail_ "forbidden: \`\`\`text fenced 'diagrams'"; sub_fail=1; }
prose_only | grep -qE '[│┌┐└┘├┤┬┴┼─]' && { fail_ "forbidden: Unicode box-drawing (in prose, outside code fences)"; sub_fail=1; }
prose_only | grep -qE '\+--+\+|--->|<---' && { fail_ "forbidden: ASCII-art arrows/boxes (in prose, outside code fences)"; sub_fail=1; }
grep -nE '^```mermaid'                   "$path" >/dev/null 2>&1 && { fail_ "forbidden: inline \`\`\`mermaid block (must be rendered to PNG)"; sub_fail=1; }
[ "$sub_fail" -eq 0 ] && pass "no-text-diagrams: no ASCII/Unicode/mermaid substitutes"

# 5b. Animated-figure safety (only if any inline animated figures are present)
if [ "$anim_count" -gt 0 ]; then
  anim_problems=$(awk '
    /^<figure class="blog-anim"/ {inb=1; start=NR; buf=""; blank=0}
    inb {
      buf = buf $0 "\n"
      if (NR>start && $0 ~ /^[[:space:]]*$/) blank=1
      if ($0 ~ /^<\/figure>/) {
        if (blank)                                          print "  block@" start ": blank line inside (CommonMark will break the SVG)"
        if (buf ~ /<script[ >]/ || buf ~ /[ \t]on[a-z]+[ \t]*=/) print "  block@" start ": <script> or on*= handler (must be declarative)"
        if (buf !~ /prefers-reduced-motion/)                print "  block@" start ": no prefers-reduced-motion guard"
        if (buf !~ /role="img"/ && buf !~ /<title>/)        print "  block@" start ": no role=\"img\"/aria-label or <title> (accessibility)"
        if (buf !~ /@keyframes/ && buf !~ /<animate/)       print "  block@" start ": no @keyframes / SMIL — figure does not animate"
        if (buf !~ /<figcaption>/)                          print "  block@" start ": no <figcaption>"
        inb=0
      }
    }' "$path")
  if [ -z "$anim_problems" ]; then
    pass "animated-figures: $anim_count inline figure(s) are safe, accessible, reduced-motion-aware"
  else
    fail_ "animated-figures: problems in inline <figure class=\"blog-anim\"> block(s):"
    echo "$anim_problems"
  fi
fi

# 6. Slug-match — every embedded image starts with this slug
foreign=$(grep -oE '!\[[^]]*\]\(/imgs/blogs/[^)]+\)' "$path" 2>/dev/null \
          | grep -v "/imgs/blogs/${slug}-" || true)
if [ -z "$foreign" ]; then
  pass "slug-match: all images use slug=$slug"
else
  fail_ "slug-match: foreign image paths (typo or stale reference):"
  echo "$foreign" | sed 's/^/  /'
fi

# 6b. WebP-only
nonwebp=$(grep -oE '!\[[^]]*\]\(/imgs/blogs/[^)]+\)' "$path" 2>/dev/null \
          | grep -ivE '\.webp\)$' || true)
if [ -z "$nonwebp" ]; then
  pass "webp-only: all embedded images are .webp"
else
  fail_ "webp-only: non-webp image embeds (every figure must ship as .webp):"
  echo "$nonwebp" | sed 's/^/  /'
fi

# 7. No-H1
if code_stripped | grep -qE '^# [^#]'; then
  fail_ "no-H1: body contains '# ' headings (must be ##):"
  code_stripped | grep -nE '^# [^#]' | sed 's/^/  /'
else
  pass "no-H1: body has no single-# headings"
fi

# 8. English-only
non_ascii=$(grep -nP '[\x{00C0}-\x{1EF9}\x{4E00}-\x{9FFF}]' "$path" 2>/dev/null | head -10 || true)
if [ -z "$non_ascii" ]; then
  pass "english-only: no Vietnamese/CJK characters in body"
else
  warn "english-only: non-ASCII letters detected (review — may be legitimate proper nouns):"
  echo "$non_ascii" | sed 's/^/  /'
fi

# 9. Frontmatter sanity (+ paper block)
fm_block=$(awk '/^---$/{c++; next} c==1' "$path")
grep -q "^date:"     <<< "$fm_block" || fail_ "frontmatter: missing 'date'"
grep -q "^tags:"     <<< "$fm_block" || fail_ "frontmatter: missing 'tags'"
grep -q "^category:" <<< "$fm_block" || fail_ "frontmatter: missing 'category'"
grep -qE "^paper:"   <<< "$fm_block" || warn  "frontmatter: no 'paper:' block (recommended: title/authors/venue/url)"
declared_rt=$(grep -E "^readTime:" <<< "$fm_block" | sed -E 's/.*: *([0-9]+).*/\1/' | head -1 || true)
if [ -n "${declared_rt:-}" ] && [ "${declared_rt}" != "$read_time" ]; then
  warn "frontmatter: declared readTime=$declared_rt but recomputed=$read_time"
fi

# 10. Paper-specific content gates
# 10a. TL;DR near the top (first 40 lines): a [!tldr] callout or a TL;DR heading/bold
if head -40 "$path" | grep -qiE '\[!tldr\]|^#{2,4}[[:space:]]*TL;?DR|\*\*TL;?DR'; then
  pass "tldr: TL;DR box present near the top"
else
  fail_ "tldr: no TL;DR box in the first 40 lines (open with a > [!tldr] callout)"
fi
# 10b. Math rigor: ≥ 3 display-math $$…$$ blocks (count opening $$ on their own or inline)
dd=$(grep -oE '\$\$' "$path" | wc -l | tr -d ' ')
dd_blocks=$(( dd / 2 ))
if [ "$dd_blocks" -ge 3 ]; then
  pass "math-rigor: $dd_blocks display-math block(s) (floor=3)"
else
  fail_ "math-rigor: only $dd_blocks display-math \$\$…\$\$ block(s) (floor=3). Show the key equations."
fi
# 10c. References section
if grep -qiE '^#{2,3}[[:space:]]*References' "$path"; then
  pass "references: References section present"
else
  fail_ "references: no '## References' section"
fi
# 10d. Paper URL somewhere in the post (frontmatter or References)
if grep -qE 'arxiv\.org|doi\.org|openreview\.net|aclanthology\.org|proceedings\.|https?://[^ )]+\.pdf' "$path"; then
  pass "paper-link: a paper/source URL is present"
else
  warn "paper-link: no arxiv/doi/openreview/pdf URL found — add the paper link"
fi
# 10e. "what would change my mind" line in the critique
if grep -qiE 'change my mind|what would change|falsif' "$path"; then
  pass "critique: an explicit 'what would change my mind' line is present"
else
  warn "critique: no 'what would change my mind' line — add one to the Critique section"
fi

echo ""
if [ "$fail" -eq 0 ]; then
  echo "RESULT: all gates passed (words=$words, readTime=$read_time, figures=$fig_count: $extracted_embeds extracted + $redrawn_embeds redrawn)"
  exit 0
else
  echo "RESULT: gates FAILED — re-enter the named phase and fix"
  exit 1
fi
