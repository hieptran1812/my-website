#!/usr/bin/env bash
# Phase E gate runner. Usage: bash scripts/verify-post.sh <post.md> <slug>
# Emits PASS/FAIL lines. Any FAIL means re-enter the named phase.

set -u
path="${1:?usage: verify-post.sh <post.md> <slug> [depth]}"
slug="${2:?usage: verify-post.sh <post.md> <slug> [depth]}"
depth="${3:-deep-dive}"  # deep-dive | explainer | paper-reading

fail=0
pass()  { echo "PASS  $1"; }
warn()  { echo "WARN  $1"; }
fail_() { echo "FAIL  $1"; fail=1; }
# Fence-aware view of the post: drop ``` code fences and their contents so the
# no-H1 check never trips on legitimate '#' code comments or sample console output.
code_stripped() { awk '/^```/{f=!f; next} !f' "$path"; }
# Prose-only view: also drop inline animated-figure blocks (<figure class="blog-anim">
# … </figure>) so SVG/CSS markup inside them never trips the ASCII/Unicode-art check.
prose_only() {
  awk '/^```/{f=!f; next} f{next}
       /^<figure class="blog-anim"/{a=1} a{ if(/^<\/figure>/) a=0; next } 1' "$path"
}
# Count inline animated figures (each is one figure for the floor + coverage gates).
anim_count=$(grep -cE '^<figure class="blog-anim"' "$path" || true)

# 1. Word count gate
words=$(wc -w < "$path" | tr -d ' ')
read_time=$(( (words + 110) / 220 ))
case "$depth" in
  deep-dive)     min=27; min_words=6000 ;;
  paper-reading) min=23; min_words=5000 ;;
  explainer)     min=18; min_words=4000 ;;
  *)             min=18; min_words=4000 ;;
esac
if [ "$words" -ge "$min_words" ]; then
  pass "word-count: $words words, readTime=$read_time (floor=$min)"
else
  fail_ "word-count: $words words / readTime=$read_time below floor $min ($min_words words). Expand thinnest sections."
fi

# 2. Diagram gate — count + minimums (static WebP embeds + inline animated figures)
img_count=$(grep -cE '^!\[' "$path" || true)
fig_count=$(( img_count + anim_count ))
case "$depth" in
  deep-dive)     fig_min=5 ;;
  paper-reading) fig_min=4 ;;
  explainer)     fig_min=3 ;;
esac
if [ "$fig_count" -ge "$fig_min" ]; then
  pass "diagram-count: $fig_count figures ($img_count WebP + $anim_count animated, floor=$fig_min)"
else
  fail_ "diagram-count: $fig_count figures ($img_count WebP + $anim_count animated) below floor $fig_min for depth=$depth"
fi

# 3. Abstraction-coverage sub-gate — prose abstractions without nearby figure
missing=$(grep -nE 'imagine|think of (it|this) as|consider (the|a) case|the way (this|it) works|under the hood|conceptually|in essence|abstract(ly|ion)' "$path" 2>/dev/null \
  | while IFS=: read -r line _; do
      next30=$(awk -v L="$line" 'NR>=L && NR<=L+30' "$path")
      # An abstraction is covered by a nearby static WebP OR an inline animated figure.
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

# 4. Sharpness sub-gate — every figure ships as .webp (lossless, cropped to bbox)
sharp_fail=0
for f in public/imgs/blogs/${slug}-[0-9]*.webp; do
  [ -e "$f" ] || continue
  if command -v sips >/dev/null 2>&1; then
    w=$(sips -g pixelWidth  "$f" 2>/dev/null | awk '/pixelWidth/  {print $2}')
    h=$(sips -g pixelHeight "$f" 2>/dev/null | awk '/pixelHeight/ {print $2}')
  else
    read w h < <(identify -format "%w %h" "$f" 2>/dev/null || echo "0 0")
  fi
  bytes=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f")
  # Byte floor is 40 KB for WebP: lossless WebP of a real diagram is ~¼–⅓ the
  # size of the source PNG, so the old 80 KB PNG floor would reject crisp figures.
  if [ "${w:-0}" -lt 1600 ] || [ "${h:-0}" -lt 900 ] || [ "${bytes:-0}" -lt 40960 ]; then
    fail_ "sharpness: $f is ${w:-0}×${h:-0} ${bytes:-0}B (need ≥1600×900, ≥40KB WebP)"
    sharp_fail=1
  fi
done
[ "$sharp_fail" -eq 0 ] && pass "sharpness: all WebP figures ≥ 1600×900, ≥ 40 KB"

# 4b. Format gate — stray non-webp renders for this slug should not exist on disk
stray=$(ls public/imgs/blogs/${slug}-[0-9]*.png public/imgs/blogs/${slug}-[0-9]*.jpg \
           public/imgs/blogs/${slug}-[0-9]*.jpeg public/imgs/blogs/${slug}-[0-9]*.gif 2>/dev/null || true)
if [ -z "$stray" ]; then
  pass "format: no leftover non-webp renders for slug=$slug"
else
  fail_ "format: non-webp render artifacts left in public/imgs/blogs (delete or convert):"
  echo "$stray" | sed 's/^/  /'
fi

# 5. Forbidden text-diagram substitutes (prose_only drops code fences AND inline
#    animated-figure blocks, so SVG path/CSS markup is never mistaken for ASCII art)
sub_fail=0
grep -nE '^```text'                      "$path" >/dev/null 2>&1 && { fail_ "forbidden: \`\`\`text fenced 'diagrams'"; sub_fail=1; }
prose_only | grep -qE '[│┌┐└┘├┤┬┴┼─]' && { fail_ "forbidden: Unicode box-drawing (in prose, outside code fences)"; sub_fail=1; }
prose_only | grep -qE '\+--+\+|--->|<---' && { fail_ "forbidden: ASCII-art arrows/boxes (in prose, outside code fences)"; sub_fail=1; }
grep -nE '^```mermaid'                   "$path" >/dev/null 2>&1 && { fail_ "forbidden: inline \`\`\`mermaid block (must be rendered to PNG)"; sub_fail=1; }
[ "$sub_fail" -eq 0 ] && pass "no-text-diagrams: no ASCII/Unicode/mermaid substitutes"

# 5b. Animated-figure safety — re-check every inline <figure class="blog-anim"> block
#     at ship time (independent of Phase C's check-anim.mjs). Each block must be a
#     contiguous raw-HTML block (no blank line), declarative (no <script>/on*=),
#     accessible (role=img+aria-label or <title>), and honor reduced-motion.
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

# 6. Slug-match gate — every embedded image starts with this slug
foreign=$(grep -oE '!\[[^]]*\]\(/imgs/blogs/[^)]+\)' "$path" 2>/dev/null \
          | grep -v "/imgs/blogs/${slug}-" || true)
if [ -z "$foreign" ]; then
  pass "slug-match: all images use slug=$slug"
else
  fail_ "slug-match: foreign image paths (typo or stale reference):"
  echo "$foreign" | sed 's/^/  /'
fi

# 6b. WebP-only gate — every embedded blog image must be .webp
nonwebp=$(grep -oE '!\[[^]]*\]\(/imgs/blogs/[^)]+\)' "$path" 2>/dev/null \
          | grep -ivE '\.webp\)$' || true)
if [ -z "$nonwebp" ]; then
  pass "webp-only: all embedded images are .webp"
else
  fail_ "webp-only: non-webp image embeds (every figure must ship as .webp):"
  echo "$nonwebp" | sed 's/^/  /'
fi

# 7. No-H1 gate (fence-aware: '#' comments inside ``` code blocks are not headings)
if code_stripped | grep -qE '^# [^#]'; then
  fail_ "no-H1: body contains '# ' headings (must be ##):"
  code_stripped | grep -nE '^# [^#]' | sed 's/^/  /'
else
  pass "no-H1: body has no single-# headings"
fi

# 8. English-only gate
non_ascii=$(grep -nP '[\x{00C0}-\x{1EF9}\x{4E00}-\x{9FFF}]' "$path" 2>/dev/null | head -10 || true)
if [ -z "$non_ascii" ]; then
  pass "english-only: no Vietnamese/CJK characters in body"
else
  warn "english-only: non-ASCII letters detected (review — may be legitimate proper nouns):"
  echo "$non_ascii" | sed 's/^/  /'
fi

# 9. Frontmatter sanity
fm_block=$(awk '/^---$/{c++; next} c==1' "$path")
grep -q "^date:"        <<< "$fm_block" || fail_ "frontmatter: missing 'date'"
grep -q "^tags:"        <<< "$fm_block" || fail_ "frontmatter: missing 'tags'"
grep -q "^category:"    <<< "$fm_block" || fail_ "frontmatter: missing 'category'"
declared_rt=$(grep -E "^readTime:" <<< "$fm_block" | sed -E 's/.*: *([0-9]+).*/\1/' | head -1 || true)
if [ -n "${declared_rt:-}" ] && [ "${declared_rt}" != "$read_time" ]; then
  warn "frontmatter: declared readTime=$declared_rt but recomputed=$read_time"
fi

echo ""
if [ "$fail" -eq 0 ]; then
  echo "RESULT: all gates passed (words=$words, readTime=$read_time, figures=$fig_count)"
  exit 0
else
  echo "RESULT: gates FAILED — re-enter the named phase and fix"
  exit 1
fi
