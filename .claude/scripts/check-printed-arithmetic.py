#!/usr/bin/env python3
"""Recompute every stated division, product, square and root in a post using ONLY the
numbers printed beside it, at the precision printed.

The verify gates never do this: a post can be green with arithmetic that does not
reproduce. The recurring defect is an intermediate rounded one step early and then
carried, which passes any audit that compares computed values to each other.

Expect false positives and hand-check every hit before editing. Known artifacts:
percent signs are dropped (`5% / 25% = 20%` reads as 5/25), a LaTeX thousands
separator needs stripping (`5{,}050`), and a match preceded by an operator is a
fragment of a longer chain (`12 x 0.05 / 1.95`).

Usage: check-printed-arithmetic.py <file.md> [more.md ...]
"""
import math, os, re, sys

N = r'(-?\d+(?:\.\d+)?)'
PATS = [
    (re.compile(rf'(?<![\d.\\]){N}\s*(?:/|÷|\\div)\s*{N}\s*(?:=|\\approx|≈)\s*{N}'), lambda a, b: a / b, 'div'),
    (re.compile(rf'(?<![\d.\\]){N}\s*(?:\\times|×)\s*{N}\s*(?:=|\\approx|≈)\s*{N}'), lambda a, b: a * b, 'mul'),
    (re.compile(rf'(?<![\d.\\]){N}\s*\^\s*\{{?2\}}?\s*(?:=|\\approx|≈)\s*{N}'), None, 'sq'),
    (re.compile(rf'\\sqrt\{{\s*{N}\s*\}}\s*(?:=|\\approx|≈)\s*{N}'), None, 'sqrt'),
]

def clean(line):
    return line.replace('{,}', '').replace('\\,', '').replace(',', '')

def check(path):
    bad = seen = 0
    for i, raw in enumerate(open(path), 1):
        line = clean(raw)
        for pat, fn, kind in PATS:
            for m in pat.finditer(line):
                before = line[:m.start()].rstrip()
                if before and (before[-1] in '*x×/+-^' or before.endswith('\\times') or before.endswith('\\cdot')):
                    continue
                g = [float(x) for x in m.groups()]
                if kind == 'sq':
                    want, got = g[0] ** 2, g[1]
                elif kind == 'sqrt':
                    if g[0] < 0:
                        continue
                    want, got = math.sqrt(g[0]), g[1]
                else:
                    if g[1] == 0:
                        continue
                    want, got = fn(g[0], g[1]), g[2]
                seen += 1
                shown = m.groups()[-1]
                dp = len(shown.split('.')[1]) if '.' in shown else 0
                if abs(round(want, dp) - got) > 10 ** (-dp) / 2 + 1e-9:
                    bad += 1
                    print(f"  {os.path.basename(path)[:38]:40} L{i}: {m.group(0).strip()[:46]:48} -> {round(want, dp)}")
    return seen, bad

total = wrong = 0
for p in sys.argv[1:]:
    s, b = check(p)
    total += s
    wrong += b
print(f"\n  checked {total} printed calculations, {wrong} suspect (hand-check each before editing)")
