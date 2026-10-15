#!/usr/bin/env bash
# Phase D2 helper for finance-writer: extract every quantitative claim from a
# drafted post and surface internal-contradiction / unsourced-number leads.
#
# Usage: bash scripts/extract-claims.sh <post.md>
#
# Prints four blocks (all advisory — the model does the judgement):
#   1. CLAIM LEDGER                  — every body line carrying a number, w/ line no.
#   2. NUMBER INDEX                  — each distinct numeric token -> lines it appears on
#   3. POSSIBLE INTERNAL CONTRADICTIONS — anchor term near >=2 different numbers
#   4. UNSOURCED LIVE-NUMBER LINES   — moving figures with no nearby citation/as-of
#
# This script never edits the post and never fails the build; it is a worklist
# generator for the fact-check protocol in references/fact-check.md.

set -u
path="${1:?usage: extract-claims.sh <post.md>}"
[ -f "$path" ] || { echo "extract-claims: no such file: $path" >&2; exit 2; }

# Body starts after the second '---' (end of YAML frontmatter).
fm_end=$(grep -nE '^---[[:space:]]*$' "$path" | sed -n '2p' | cut -d: -f1)
fm_end="${fm_end:-0}"

# A line is a "numeric claim" if it carries a money/percent/bps token, a scale
# word (million/billion/trillion), a comma-grouped number, a basis-point/yield
# figure, or a 4-digit year. We work on body lines only and skip image embeds,
# fenced code, and pure math display lines (illustrative, not factual claims).
NUMTOK='(\$[0-9]|[0-9][0-9.,]*[ ]?%|[0-9][0-9.,]*[ ]?(bps?|basis points?)|[0-9][0-9.,]*[ ]?(trillion|billion|million|nghìn tỷ|tỷ|đồng|VND|USD)|[0-9],[0-9]{3}|\b(19|20)[0-9]{2}\b|[0-9]+\.[0-9]+)'

echo "=============================================================="
echo " CLAIM LEDGER  —  $path"
echo " (every body line with a number; turn each into a sourced row)"
echo "=============================================================="
ledger=$(grep -nE "$NUMTOK" "$path" \
  | awk -F: -v S="$fm_end" '$1>S' \
  | grep -vE '^[0-9]+:[[:space:]]*!\[' \
  | grep -vE '^[0-9]+:[[:space:]]*```' \
  | grep -vE '^[0-9]+:[[:space:]]*\$\$' \
  | sed -E 's/^([0-9]+):/  L\1 | /')
if [ -n "$ledger" ]; then echo "$ledger"; else echo "  (no numeric claim lines found — is this the right file?)"; fi

echo ""
echo "=============================================================="
echo " NUMBER INDEX  —  each distinct figure -> body lines using it"
echo " (a figure repeated across the post must stay identical)"
echo "=============================================================="
awk -v S="$fm_end" 'NR>S {print NR": "$0}' "$path" \
  | grep -vE ':\s*!\[' \
  | grep -oE '[0-9]+: |\$[0-9][0-9.,]*|[0-9][0-9.,]*%|[0-9][0-9.,]*[ ]?(bps?|trillion|billion|million)' \
  | awk '
      /^[0-9]+: $/ { ln=$1; sub(/:$/,"",ln); next }
      { key=$0; gsub(/^[ \t]+|[ \t]+$/,"",key);
        if (key=="") next;
        if (!(key in seen)) { order[++n]=key }
        seen[key]=seen[key] (seen[key]==""?"":", ") ln }
      END { for (i=1;i<=n;i++){ k=order[i];
              # only show figures that appear more than once (consistency risk)
              c=gsub(/,/,",",seen[k]);
              if (c>=1) printf "  %-14s -> lines %s\n", k, seen[k] } }'

echo ""
echo "=============================================================="
echo " POSSIBLE INTERNAL CONTRADICTIONS"
echo " (anchor term seen near >=2 DIFFERENT numbers — verify each)"
echo "=============================================================="
# For each anchor (a built-in finance-metric / ticker list), collect the numeric
# tokens on the same line and report anchors that co-occur with two or more
# distinct numbers — a lead to check, not a verdict.
python3 - "$path" "$fm_end" <<'PY' 2>/dev/null || echo "  (contradiction scan needs python3; scan NUMBER INDEX + ledger by hand)"
import re, sys
path, fm_end = sys.argv[1], int(sys.argv[2])
anchors = ["VN-Index","HNX-Index","S&P","Nasdaq","Dow","10-year","2-year",
           "GDP","CPI","inflation","market cap","policy rate","reserve ratio",
           "foreign ownership","ownership cap","margin","P/E","P/B","revenue",
           "net profit","gross margin","spread","premium","exchange rate",
           "reserves","deficit","surplus","debt","AUM","trading volume","yield"]
numpat = re.compile(r'\$?\d[\d.,]*\s?(?:%|bps?|trillion|billion|million|tỷ)?')
buckets = {}
with open(path, encoding="utf-8") as f:
    lines = f.readlines()
for i, raw in enumerate(lines[fm_end:], start=fm_end+1):
    line = raw.rstrip("\n")
    if line.lstrip().startswith("![") or line.lstrip().startswith("```"):
        continue
    low = line.lower()
    for a in anchors:
        if a.lower() in low:
            nums = [n.strip() for n in numpat.findall(line) if re.search(r'\d', n)]
            # keep only number-bearing, drop bare years used as dates
            nums = [n for n in nums if not re.fullmatch(r'(19|20)\d{2}', n.replace('$','').replace(',',''))]
            if nums:
                buckets.setdefault(a, {})
                for n in nums:
                    buckets[a].setdefault(n, []).append(i)
flagged = False
for a, vals in buckets.items():
    if len(vals) >= 2:
        flagged = True
        detail = "; ".join(f"{n} (L{','.join(map(str,ls))})" for n, ls in vals.items())
        print(f"  '{a}': {detail}")
if not flagged:
    print("  (no anchor term found near two different numbers — still scan by eye)")
PY

echo ""
echo "=============================================================="
echo " UNSOURCED LIVE-NUMBER LINES"
echo " (price/index/rate/size with no nearby link, as-of, or cite —"
echo "  prioritise these for the Step-3 source waterfall)"
echo "=============================================================="
# A "live" number: $, %, index level, or scale word. "Sourced" if the line (or a
# neighbour) carries a markdown link, an 'as of' / 'as-of' marker, a parenthetical
# source, a decree/circular number, or an exchange/agency name.
awk -v S="$fm_end" '
  NR<=S { next }
  {
    line=$0
    live = (line ~ /\$[0-9]/ || line ~ /[0-9][0-9.,]*[ ]?%/ || line ~ /[0-9][0-9.,]*[ ]?(trillion|billion|million)/ || line ~ /[0-9],[0-9][0-9][0-9]/)
    if (!live) next
    if (line ~ /^[[:space:]]*!\[/ ) next
    sourced = (line ~ /\]\(http/ || line ~ /\]\(\/blog/ || tolower(line) ~ /as.?of/ || line ~ /[Dd]ecree|[Cc]ircular|NĐ-CP|HOSE|HNX|SBV|SSC|UBCKNN|GSO|Fed|ECB|BIS|IMF|World Bank|SEC|Bloomberg|Reuters|CafeF|Vietstock/ || line ~ /\([^)]*20[0-9][0-9][^)]*\)/)
    if (!sourced) printf "  L%d | %s\n", NR, substr(line,1,110)
  }' "$path" \
  | head -40 \
  || echo "  (none — every live number has a nearby source marker)"
unsourced=$(awk -v S="$fm_end" '
  NR<=S { next }
  { line=$0
    live = (line ~ /\$[0-9]/ || line ~ /[0-9][0-9.,]*[ ]?%/ || line ~ /[0-9][0-9.,]*[ ]?(trillion|billion|million)/ || line ~ /[0-9],[0-9][0-9][0-9]/)
    if (!live) next
    if (line ~ /^[[:space:]]*!\[/) next
    sourced = (line ~ /\]\(http/ || line ~ /\]\(\/blog/ || tolower(line) ~ /as.?of/ || line ~ /[Dd]ecree|[Cc]ircular|NĐ-CP|HOSE|HNX|SBV|SSC|UBCKNN|GSO|Fed|ECB|BIS|IMF|World Bank|SEC|Bloomberg|Reuters|CafeF|Vietstock/ || line ~ /\([^)]*20[0-9][0-9][^)]*\)/)
    if (!sourced) c++ }
  END { print c+0 }' "$path")

echo ""
echo "--------------------------------------------------------------"
echo " SUMMARY: $unsourced live-number line(s) lack a nearby source marker."
echo " Next: build the claims table, resolve contradictions, then run"
echo " the Tier 1->4 waterfall (legal text -> securities press ->"
echo " mainstream/official press -> international). See references/fact-check.md."
echo "--------------------------------------------------------------"
