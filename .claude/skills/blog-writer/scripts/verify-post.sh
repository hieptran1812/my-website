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

# 1. Word count gate
words=$(wc -w < "$path" | tr -d ' ')
read_time=$(( (words + 110) / 220 ))
case "$depth" in
  deep-dive)     min=50; min_words=11000 ;;
  paper-reading) min=30; min_words=6500  ;;
  explainer)     min=25; min_words=5500  ;;
  *)             min=25; min_words=5500  ;;
esac
if [ "$words" -ge "$min_words" ]; then
  pass "word-count: $words words, readTime=$read_time (floor=$min)"
else
  fail_ "word-count: $words words / readTime=$read_time below floor $min ($min_words words). Expand thinnest sections."
fi

# 2. Diagram gate — count + minimums
fig_count=$(grep -cE '^!\[' "$path" || true)
case "$depth" in
  deep-dive)     fig_min=8 ;;
  paper-reading) fig_min=5 ;;
  explainer)     fig_min=4 ;;
esac
if [ "$fig_count" -ge "$fig_min" ]; then
  pass "diagram-count: $fig_count figures (floor=$fig_min)"
else
  fail_ "diagram-count: $fig_count figures below floor $fig_min for depth=$depth"
fi

# 3. Abstraction-coverage sub-gate — prose abstractions without nearby figure
missing=$(grep -nE 'imagine|think of (it|this) as|consider (the|a) case|the way (this|it) works|under the hood|conceptually|in essence|abstract(ly|ion)' "$path" 2>/dev/null \
  | while IFS=: read -r line _; do
      next30=$(awk -v L="$line" 'NR>=L && NR<=L+30' "$path")
      if ! grep -q '!\[[^]]*\](/imgs/blogs/' <<< "$next30"; then
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
for f in public/imgs/blogs/${slug}-*.webp; do
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
    fail_ "sharpness: $f is $w×$h ${bytes}B (need ≥1600×900, ≥40KB WebP)"
    sharp_fail=1
  fi
done
[ "$sharp_fail" -eq 0 ] && pass "sharpness: all WebP figures ≥ 1600×900, ≥ 40 KB"

# 4b. Format gate — stray non-webp renders for this slug should not exist on disk
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
grep -nE '[│┌┐└┘├┤┬┴┼─]'                 "$path" >/dev/null 2>&1 && { fail_ "forbidden: Unicode box-drawing"; sub_fail=1; }
grep -nE '\+--+\+|--->|<---'             "$path" >/dev/null 2>&1 && { fail_ "forbidden: ASCII-art arrows/boxes"; sub_fail=1; }
grep -nE '^```mermaid'                   "$path" >/dev/null 2>&1 && { fail_ "forbidden: inline \`\`\`mermaid block (must be rendered to PNG)"; sub_fail=1; }
[ "$sub_fail" -eq 0 ] && pass "no-text-diagrams: no ASCII/Unicode/mermaid substitutes"

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

# 7. No-H1 gate
if grep -nE '^# [^#]' "$path" >/dev/null 2>&1; then
  fail_ "no-H1: body contains '# ' headings (must be ##):"
  grep -nE '^# [^#]' "$path" | sed 's/^/  /'
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
grep -q "^aiGenerated:" <<< "$fm_block" || fail_ "frontmatter: missing 'aiGenerated'"
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
