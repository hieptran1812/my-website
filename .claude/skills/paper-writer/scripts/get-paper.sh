#!/usr/bin/env bash
# Resolve a paper to a local PDF, extract its text, and render page thumbnails for
# figure hunting. Phase A helper for paper-writer.
#
# Usage:
#   bash get-paper.sh <arxiv-id | arxiv-url | pdf-url | /path/to/local.pdf> <slug>
#
# Writes into .cache/paper-writer/<slug>/:
#   paper.pdf          the source PDF
#   paper.txt          pdftotext -layout dump (ground truth for the analysis)
#   pages/page-N.png   ~120 DPI page renders (Read these to locate figures)
#   info.txt           pdfinfo (page count, page size in points)
set -euo pipefail

src="${1:?usage: get-paper.sh <arxiv-id|url|local.pdf> <slug>}"
slug="${2:?usage: get-paper.sh <arxiv-id|url|local.pdf> <slug>}"

cache=".cache/paper-writer/${slug}"
mkdir -p "$cache/pages"
pdf="$cache/paper.pdf"

resolve_url() {
  local s="$1"
  # bare arxiv id (e.g. 1706.03762 or 2401.01234v2)
  if [[ "$s" =~ ^[0-9]{4}\.[0-9]{4,5}(v[0-9]+)?$ ]]; then
    echo "https://arxiv.org/pdf/$s"; return
  fi
  # arxiv abs/pdf URL -> pdf URL
  if [[ "$s" =~ arxiv\.org/(abs|pdf)/([^/?#]+) ]]; then
    echo "https://arxiv.org/pdf/${BASH_REMATCH[2]}"; return
  fi
  echo "$s"  # assume it's already a direct PDF URL
}

if [[ -f "$src" ]]; then
  echo "→ local PDF: $src"
  cp "$src" "$pdf"
else
  url="$(resolve_url "$src")"
  echo "→ downloading: $url"
  curl -fsSL --max-time 120 -o "$pdf" "$url" \
    || { echo "FAIL: could not download $url"; exit 1; }
fi

# Guard: is it actually a PDF?
if ! head -c 5 "$pdf" | grep -q '%PDF'; then
  echo "FAIL: $pdf is not a PDF (arxiv may have returned HTML — check the id/url)"; exit 1
fi

echo "→ pdfinfo"
pdfinfo "$pdf" | tee "$cache/info.txt" | grep -iE "Pages|Page size" || true

echo "→ pdftotext -layout → paper.txt"
pdftotext -layout "$pdf" "$cache/paper.txt"
words=$(wc -w < "$cache/paper.txt" | tr -d ' ')
echo "   extracted $words words of text"

npages=$(pdfinfo "$pdf" | awk '/^Pages:/{print $2}')
echo "→ rendering $npages page thumbnails at 120 DPI → pages/"
pdftoppm -png -r 120 "$pdf" "$cache/pages/page"
# pdftoppm zero-pads multi-page output (page-01.png); normalize the -NN suffix to -N
# so Read paths are predictable. (Single-digit page counts are already page-1.png.)
if ls "$cache/pages/"page-0*.png >/dev/null 2>&1; then
  for f in "$cache/pages/"page-0*.png; do
    n=$(basename "$f" .png | sed 's/^page-0*//')
    mv "$f" "$cache/pages/page-${n}.png"
  done
fi

echo ""
echo "READY: $cache"
echo "  paper.pdf   — source"
echo "  paper.txt   — Read this in full (Phase B ground truth)"
echo "  pages/page-N.png ($npages pages) — Read to hunt figures, note page + fractional box"
echo "  info.txt    — page count & page size"
