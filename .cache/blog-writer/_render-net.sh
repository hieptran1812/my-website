#!/usr/bin/env bash
# Render one networking-series slug:
#   numbered DSL/element inputs -> validated scenes -> cache PNGs -> lossless WebPs
#
# Usage: .cache/blog-writer/_render-net.sh <slug>

set -Eeuo pipefail

readonly ROOT="/Users/hieptran1812/Documents/my-website"
readonly BLOG_WRITER="$ROOT/.claude/skills/blog-writer"
readonly RENDERER="/Users/hieptran1812/Documents/mcp_excalidraw/scripts/render-scene-batch.mjs"
readonly RENDERER_ROOT="/Users/hieptran1812/Documents/mcp_excalidraw"
readonly OUTPUT_DIR="$ROOT/public/imgs/blogs"
readonly MIN_WIDTH=1600
readonly MIN_HEIGHT=900
readonly MIN_BYTES=40960

die() {
  printf 'render-net: %s\n' "$*" >&2
  exit 1
}

usage() {
  printf 'usage: %s <slug>\n' "${0##*/}" >&2
  exit 2
}

(( $# == 1 )) || usage
slug=$1

[[ $slug =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]] \
  || die "invalid slug '$slug' (expected lowercase kebab-case)"

readonly CACHE="$ROOT/.cache/blog-writer/$slug"
[[ -d $CACHE ]] || die "cache directory does not exist: $CACHE"
[[ ! -L $CACHE ]] || die "cache directory must not be a symlink: $CACHE"
[[ -r $CACHE && -w $CACHE ]] || die "cache directory must be readable and writable: $CACHE"

command -v node >/dev/null 2>&1 || die "node is required"
command -v jq >/dev/null 2>&1 || die "jq is required"
command -v cwebp >/dev/null 2>&1 || die "cwebp is required"
if ! command -v sips >/dev/null 2>&1 && ! command -v identify >/dev/null 2>&1; then
  die "sips or ImageMagick identify is required to inspect image dimensions"
fi

readonly LAYOUT="$BLOG_WRITER/scripts/layout-scene.mjs"
readonly AUTHOR="$BLOG_WRITER/scripts/author-scene.mjs"
[[ -f $LAYOUT ]] || die "layout tool not found: $LAYOUT"
[[ -f $AUTHOR ]] || die "author tool not found: $AUTHOR"
[[ -f $RENDERER ]] || die "batch renderer not found: $RENDERER"
[[ -f $RENDERER_ROOT/dist/frontend/headless.html ]] \
  || die "headless renderer build not found: $RENDERER_ROOT/dist/frontend/headless.html"
[[ -d $RENDERER_ROOT/node_modules/puppeteer ]] \
  || die "renderer dependency not found: $RENDERER_ROOT/node_modules/puppeteer"

temp_paths=()
cleanup() {
  local path
  for path in "${temp_paths[@]-}"; do
    [[ ! -e $path ]] || rm -f -- "$path"
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Record only exact <slug>-<positive integer>.(dsl|in).json inputs. Other cache
# files belong to the authoring loop and are deliberately ignored.
records=$(mktemp "$CACHE/.render-net-sources.XXXXXX")
temp_paths+=("$records")
shopt -s nullglob
for input in "$CACHE/$slug"-*.dsl.json "$CACHE/$slug"-*.in.json; do
  base=${input##*/}
  if [[ $base =~ ^${slug}-([1-9][0-9]*)\.(dsl|in)\.json$ ]]; then
    printf '%s\t%s\t%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "$input" >> "$records"
  fi
done
shopt -u nullglob

[[ -s $records ]] || die "no numbered .dsl.json or .in.json inputs found in $CACHE"
sort -t $'\t' -k1,1n -k2,2 "$records" -o "$records"

indices=()
kinds=()
inputs=()
previous_index=''
previous_input=''
while IFS=$'\t' read -r index kind input; do
  if [[ $index == "$previous_index" ]]; then
    die "figure $index is ambiguous: both $previous_input and $input exist"
  fi

  indices+=("$index")
  kinds+=("$kind")
  inputs+=("$input")
  previous_index=$index
  previous_input=$input
done < "$records"

# A committed same-slug render belongs to a shipped post and must never be
# overwritten. Untracked outputs are allowed because visual-review fix cycles
# intentionally re-render the current wave before it is committed.
for index in "${indices[@]}"; do
  tracked="public/imgs/blogs/$slug-$index.webp"
  if git -C "$ROOT" ls-files --error-unmatch -- "$tracked" >/dev/null 2>&1; then
    die "refusing to overwrite committed output: $ROOT/$tracked"
  fi
done

# Current posts ship WebP only. A legacy-format sibling is almost always stale
# output and would make the post gate ambiguous.
for extension in png jpg jpeg gif svg; do
  if compgen -G "$OUTPUT_DIR/$slug-[0-9]*.$extension" >/dev/null; then
    die "stray non-WebP output exists for $slug (*.$extension); remove it before rendering"
  fi
done

scenes=()
for position in "${!inputs[@]}"; do
  index=${indices[$position]}
  kind=${kinds[$position]}
  input=${inputs[$position]}
  scene="$CACHE/$slug-$index.scene.json"
  if [[ $kind == dsl ]]; then
    node "$LAYOUT" "$input" "$scene" >/dev/null \
      || die "layout failed for figure $index: $input"
  else
    node "$AUTHOR" "$input" "$scene" >/dev/null \
      || die "authoring failed for figure $index: $input"
  fi
  [[ -s $scene ]] || die "figure $index did not produce a scene: $scene"

  scenes+=("$scene")
done

(( ${#scenes[@]} > 0 )) || die "no scenes were produced"

# Build the standard per-slug manifest atomically from this run's scenes only,
# so stale or unrelated .scene.json files can never enter the render batch.
manifest="$CACHE/manifest.json"
manifest_tmp=$(mktemp "$CACHE/.render-net-manifest.XXXXXX")
temp_paths+=("$manifest_tmp")
jq -n --args \
  '$ARGS.positional | map({in: ., out: sub("\\.scene\\.json$"; ".png")})' \
  -- "${scenes[@]}" > "$manifest_tmp"
mv -f -- "$manifest_tmp" "$manifest"

node "$RENDERER" "$manifest"

pngs=()
for scene in "${scenes[@]}"; do
  png="${scene%.scene.json}.png"
  [[ -s $png ]] || die "renderer did not produce a non-empty PNG: $png"
  pngs+=("$png")
done

mkdir -p -- "$OUTPUT_DIR"

dimensions() {
  local file=$1 metadata width height
  if command -v sips >/dev/null 2>&1; then
    metadata=$(sips -g pixelWidth -g pixelHeight "$file" 2>/dev/null) || return 1
    width=$(awk '/pixelWidth/ { print $2; exit }' <<< "$metadata")
    height=$(awk '/pixelHeight/ { print $2; exit }' <<< "$metadata")
  else
    metadata=$(identify -format '%w %h' "$file" 2>/dev/null) || return 1
    read -r width height <<< "$metadata"
  fi
  [[ $width =~ ^[0-9]+$ && $height =~ ^[0-9]+$ ]] || return 1
  printf '%s %s\n' "$width" "$height"
}

# Encode and validate every figure under temporary names first. A bad final
# figure therefore cannot leave a partly published set of new WebPs.
staged=()
finals=()
widths=()
heights=()
sizes=()
for position in "${!pngs[@]}"; do
  index=${indices[$position]}
  png=${pngs[$position]}
  final="$OUTPUT_DIR/$slug-$index.webp"
  stage="$OUTPUT_DIR/.$slug-$index.webp.tmp.$$"
  [[ ! -e $stage ]] || die "temporary output already exists: $stage"
  temp_paths+=("$stage")

  cwebp -quiet -lossless -m 6 "$png" -o "$stage" \
    || die "WebP conversion failed for figure $index: $png"
  [[ -s $stage ]] || die "WebP conversion produced an empty file: $stage"

  dims=$(dimensions "$stage") || die "cannot read WebP dimensions for figure $index: $stage"
  read -r width height <<< "$dims"
  bytes=$(wc -c < "$stage" | tr -d '[:space:]')
  [[ $bytes =~ ^[0-9]+$ ]] || die "cannot read WebP size for figure $index: $stage"
  if (( width < MIN_WIDTH || height < MIN_HEIGHT || bytes < MIN_BYTES )); then
    die "figure $index failed sharpness gate: ${width}x${height} ${bytes}B (need >=${MIN_WIDTH}x${MIN_HEIGHT}, >=${MIN_BYTES}B)"
  fi

  staged+=("$stage")
  finals+=("$final")
  widths+=("$width")
  heights+=("$height")
  sizes+=("$bytes")
done

for position in "${!staged[@]}"; do
  mv -f -- "${staged[$position]}" "${finals[$position]}"
done

printf 'render-net: %d figure(s) published for %s\n' "${#finals[@]}" "$slug"
for position in "${!finals[@]}"; do
  printf '  fig %s (%s): %s  %sx%s  %sB\n' \
    "${indices[$position]}" "${kinds[$position]}" "${finals[$position]}" \
    "${widths[$position]}" "${heights[$position]}" "${sizes[$position]}"
done
