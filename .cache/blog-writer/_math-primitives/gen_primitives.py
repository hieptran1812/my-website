#!/usr/bin/env python3
"""Generate the 4 recurring figure primitives of the Math for AI Research series.

Writes element-form .in.json files (author-scene.mjs input) into
.cache/blog-writer/math-for-ai-primitives/math-for-ai-primitives-<N>.in.json.
Every coordinate below is computed, never eyeballed.

Conventions (keep these when adapting a primitive):
  * canvas 2400x1600, title y=60 (fontSize 32), caption y=120 (fontSize 28)
  * free text is left-aligned; centred text gets x = centre - width/2
  * curves / closed outlines are `line` elements that START at their lowest
    on-screen point, so every point offset has dy <= 0 and height is negative
    (the overlap rule then ignores the curve's bbox; strokes still checked)
  * Virgil (1) for prose, Cascadia (3) for numbers and math symbols
"""
import json
import math
import os
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "math-for-ai-primitives")
os.makedirs(OUT, exist_ok=True)

BLUE, AMBER, RED, GREEN, LAV, GRAY = "#a5d8ff", "#ffec99", "#ffc9c9", "#b2f2bb", "#d0bfff", "#e9ecef"
INK, DBLUE, DAMBER, MUTED, GRID = "#1e1e1e", "#1971c2", "#f08c00", "#868e96", "#ced4da"


def tw(text, fs, fam):
    f = 0.62 if fam == 3 else 0.6
    return max(len(l) for l in text.split("\n")) * fs * f


def text(id_, x, y, s, fs=22, fam=1, w=None, center=False, color=None):
    width = w if w is not None else math.ceil((tw(s, fs, fam) + 12) / 20) * 20
    lines = s.count("\n") + 1
    height = math.ceil(lines * fs * 1.25 / 20) * 20
    if center:
        x = x - width / 2
    el = {"type": "text", "id": id_, "x": round(x), "y": round(y), "width": width,
          "height": height, "text": s, "fontSize": fs, "fontFamily": fam}
    if color:
        el["strokeColor"] = color
    return el


def header(title, caption):
    t = text("title", 1200, 60, title, 32, 1, center=True)
    c = text("caption", 1200, 120, caption, 28, 1, center=True)
    return [t, c]


def hline(id_, x, y, w, color=GRID, sw=1.5, dashed=False):
    el = {"type": "line", "id": id_, "x": round(x), "y": round(y), "width": round(w), "height": 0,
          "points": [[0, 0], [round(w), 0]], "strokeColor": color, "strokeWidth": sw}
    if dashed:
        el["strokeStyle"] = "dashed"
    return el


def vline(id_, x, y, h, color=GRID, sw=1.5, dashed=False):
    el = {"type": "line", "id": id_, "x": round(x), "y": round(y), "width": 0, "height": round(h),
          "points": [[0, 0], [0, round(h)]], "strokeColor": color, "strokeWidth": sw}
    if dashed:
        el["strokeStyle"] = "dashed"
    return el


def curve(id_, pts, color=INK, sw=2, bg=None, dashed=False, closed=False):
    """pts are absolute screen coords. Start at the lowest on-screen point."""
    if closed:
        if pts[0] == pts[-1]:
            pts = pts[:-1]
        k = max(range(len(pts)), key=lambda i: pts[i][1])
        pts = pts[k:] + pts[:k] + [pts[k]]
    x0, y0 = pts[0]
    rel = [[round(px - x0, 1), round(py - y0, 1)] for px, py in pts]
    xs = [p[0] for p in rel]
    ys = [p[1] for p in rel]
    el = {"type": "line", "id": id_, "x": round(x0, 1), "y": round(y0, 1),
          "width": round(max(xs) - min(xs), 1), "height": round(min(ys), 1),
          "points": rel, "strokeColor": color, "strokeWidth": sw, "roughness": 0}
    if bg:
        el["backgroundColor"] = bg
        el["fillStyle"] = "hachure"
    if dashed:
        el["strokeStyle"] = "dashed"
    return el


def arrow(id_, x0, y0, x1, y1, color="#343a40", sw=2, start=None, end=None, dashed=False, head="arrow"):
    el = {"type": "arrow", "id": id_, "x": round(x0, 1), "y": round(y0, 1),
          "width": round(x1 - x0, 1), "height": round(y1 - y0, 1),
          "points": [[0, 0], [round(x1 - x0, 1), round(y1 - y0, 1)]],
          "strokeColor": color, "strokeWidth": sw, "endArrowhead": head, "roughness": 0}
    if abs(x1 - x0) > 0.5 and abs(y1 - y0) > 0.5:
        el["diagonal"] = True
    if start:
        el["startBinding"] = {"elementId": start, "focus": 0, "gap": 8}
    if end:
        el["endBinding"] = {"elementId": end, "focus": 0, "gap": 8}
    if not start and not end:
        el["unbound"] = True
    if dashed:
        el["strokeStyle"] = "dashed"
    return el


def rect(id_, x, y, w, h, bg="transparent", label=None, sw=2, fs=22, fam=1, color=None):
    el = {"type": "rectangle", "id": id_, "x": round(x), "y": round(y), "width": round(w),
          "height": round(h), "backgroundColor": bg, "strokeWidth": sw}
    if bg != "transparent":
        el["fillStyle"] = "solid"
    if label:
        el["label"] = {"text": label, "fontSize": fs, "fontFamily": fam}
    if color:
        el["strokeColor"] = color
    return el


def dot(id_, cx, cy, bg, d=12, color=INK):
    return {"type": "ellipse", "id": id_, "x": round(cx - d / 2, 1), "y": round(cy - d / 2, 1),
            "width": d, "height": d, "backgroundColor": bg, "fillStyle": "solid",
            "strokeWidth": 1.5, "strokeColor": color, "roughness": 0}


def write(n, title, claim, caption, els):
    doc = {"title": title, "_claim": claim, "_caption": caption, "elements": els,
           "export": {"padding": 48, "minWidth": 1600, "minHeight": 900}}
    path = os.path.join(OUT, f"math-for-ai-primitives-{n}.in.json")
    with open(path, "w") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
    print("wrote", path, len(els), "elements")


# ---------------------------------------------------------------- 1 ladder map
def ladder():
    title = "The math ladder: climb a rung, unlock a paper family"
    caption = "Each rung is one track of the series; the lit rung is where you stand and its arrows show which papers it opens."
    els = header(title, caption)
    tracks = ["A  Language of papers", "B–C  Linear algebra", "D  Calculus", "R  Rigor",
              "E  Probability", "F  Information theory", "G  Optimization",
              "H  Stochastic processes", "I  Flows and OT", "J  Geometry and symmetry",
              "K  Architecture math", "L  Learning theory", "M  LLM practice",
              "N–T  Frontier theory"]
    lit = 7
    RX0, RX1 = 360, 1000          # ladder rails
    TOP, STRIDE, RH = 200, 100, 80
    n = len(tracks)
    ry = {i: TOP + (n - 1 - i) * STRIDE for i in range(n)}   # i=0 bottom
    els.append(vline("rail-l", RX0, TOP - 20, (n - 1) * STRIDE + RH + 40, INK, 4))
    els.append(vline("rail-r", RX1, TOP - 20, (n - 1) * STRIDE + RH + 40, INK, 4))
    for i, name in enumerate(tracks):
        bg = BLUE if i == lit else (GRAY if i < lit else "transparent")
        els.append(rect(f"rung{i}", RX0, ry[i], RX1 - RX0, RH, bg, name,
                        sw=4 if i == lit else 2))
    # left annotations: learned bracket, you-are-here, ahead bracket
    gx = 320
    els.append(vline("br-learned", gx, ry[lit - 1], ry[0] + RH - ry[lit - 1], MUTED, 3))
    els.append(text("t-learned", 60, (ry[lit - 1] + ry[0] + RH) / 2 - 50,
                    "learned:\n7 rungs\nbelow you", 24, 1, w=200))
    els.append(vline("br-ahead", gx, ry[n - 1], ry[lit + 1] + RH - ry[n - 1], MUTED, 3, dashed=True))
    els.append(text("t-ahead", 60, (ry[n - 1] + ry[lit + 1] + RH) / 2 - 50,
                    "ahead:\n6 rungs\nto climb", 24, 1, w=200))
    els.append(text("t-here", 60, ry[lit] + 20, "you are here", 24, 1, w=200, color=DBLUE))
    els.append(arrow("a-here", 262, ry[lit] + 40, RX0 - 4, ry[lit] + 40, DBLUE, 3, end=f"rung{lit}"))
    # destinations
    DX, DW, DH = 1700, 620, 140
    # edges are monotone (higher rung -> higher destination), so none cross
    dests = [("llm", "LLM and RLHF\nscaling, DPO, GRPO", [11, 12], 280),
             ("moe", "Mixture of experts\nrouting, load balance", [10], 480),
             ("ot", "Optimal transport\nSinkhorn, W2 couplings", [8], 680),
             ("fm", "Flow matching\nrectified flow, CNF", [7, 8], 880),
             ("diff", "Diffusion\nDDPM, score SDEs", [2, 4, 7], 1100)]
    dy = {}
    for did, label, _, y in dests:
        dy[did] = y
        els.append(rect(did, DX, y, DW, DH, LAV, label, sw=2))
    for did, _, srcs, _ in dests:
        m = len(srcs)
        for j, s in enumerate(sorted(srcs, reverse=True)):
            ey = dy[did] + DH * (j + 1) / (m + 1)
            sy = ry[s] + RH / 2
            hot = s == lit
            els.append(arrow(f"u-{s}-{did}", RX1 + 24, sy, DX - 2, ey,
                             DBLUE if hot else MUTED, 5 if hot else 2,
                             start=f"rung{s}", end=did))
    dbot = 1100 + DH
    els.append(text("t-unlock", DX, dbot + 40, "thick blue: what your rung H unlocks\ngrey: what other rungs unlock", 22, 1))
    write(1, title, "Each track is a rung, and every rung unlocks specific paper families such as diffusion or flow matching.",
          caption, els)


# ---------------------------------------------------------------- 2 the plane
def plane():
    title = "The 2D plane: what the matrix A does to the unit circle"
    caption = "A matrix is a map that turns circles into ellipses: its columns are where e₁ and e₂ land."
    els = header(title, caption)
    S = 200                      # 1 unit = 200 px
    X0, Y0 = 760, 760            # origin
    XR, YR = 3, 2                # grid extent in units
    P = lambda x, y: (X0 + S * x, Y0 - S * y)
    for gx in range(-XR, XR + 1):
        if gx:
            els.append(vline(f"gv{gx}", P(gx, 0)[0], P(0, YR)[1], 2 * YR * S, GRID, 1.5, dashed=True))
    for gy in range(-YR, YR + 1):
        if gy:
            els.append(hline(f"gh{gy}", P(-XR, 0)[0], P(0, gy)[1], 2 * XR * S, GRID, 1.5, dashed=True))
    els.append(arrow("ax-x", P(-XR - 0.3, 0)[0], Y0, P(XR + 0.35, 0)[0], Y0, INK, 2))
    els.append(arrow("ax-y", X0, P(0, -YR - 0.3)[1], X0, P(0, YR + 0.35)[1], INK, 2))
    # tick labels (below axis between gridlines would cross; put them in cells)
    # positive ticks sit right-below (x) / right-above (y) of the gridline,
    # negative ticks left-above (x) / left-below (y): those cells are stroke-free
    for gx in (-3, -2, -1, 1, 2, 3):
        s = f"{gx}".replace("-", "−")
        if gx > 0:
            els.append(text(f"tx{gx}", P(gx, 0)[0] + 8, Y0 + 8, s, 22, 3, w=40))
        else:
            els.append(text(f"tx{gx}", P(gx, 0)[0] - 48, Y0 - 48, s, 22, 3, w=40))
    for gy in (-2, -1, 1, 2):
        s = f"{gy}".replace("-", "−")
        if gy > 0:
            els.append(text(f"ty{gy}", X0 + 8, P(0, gy)[1] - 48, s, 22, 3, w=40))
        else:
            els.append(text(f"ty{gy}", X0 - 48, P(0, gy)[1] + 8, s, 22, 3, w=40))
    els.append(text("t-xlab", P(XR + 0.35, 0)[0] - 20, Y0 - 52, "x", 24, 3, w=40))
    els.append(text("t-ylab", X0 - 52, P(0, YR + 0.35)[1] - 6, "y", 24, 3, w=40))
    # unit circle and its exact image under A = [[2,1],[0,1]]
    N = 120
    circ = [P(math.cos(2 * math.pi * k / N), math.sin(2 * math.pi * k / N)) for k in range(N)]
    ell = [P(2 * math.cos(2 * math.pi * k / N) + math.sin(2 * math.pi * k / N),
             math.sin(2 * math.pi * k / N)) for k in range(N)]
    els.append(curve("image-ellipse", ell, DBLUE, 3.5, bg=BLUE, closed=True))
    els.append(curve("unit-circle", circ, INK, 2.5, closed=True))
    # singular axes of A: A^T A eigenpairs; sigma1 = 2.288, sigma2 = 0.874
    ata = [[4, 2], [2, 2]]
    tr, det = 6, 4
    l1 = (tr + math.sqrt(tr * tr - 4 * det)) / 2
    l2 = (tr - math.sqrt(tr * tr - 4 * det)) / 2
    s1, s2 = math.sqrt(l1), math.sqrt(l2)
    # left singular vectors u = A v / sigma
    def evec(lam):
        v = (ata[0][1], lam - ata[0][0])
        nrm = math.hypot(*v)
        return (v[0] / nrm, v[1] / nrm)
    v1, v2 = evec(l1), evec(l2)
    u1 = ((2 * v1[0] + v1[1]) / s1, v1[1] / s1)
    u2 = ((2 * v2[0] + v2[1]) / s2, v2[1] / s2)
    a1 = P(s1 * u1[0], s1 * u1[1])
    a2 = P(s2 * u2[0], s2 * u2[1])
    els.append(curve("sv1", [P(-s1 * u1[0], -s1 * u1[1]), a1] if a1[1] < P(-s1 * u1[0], -s1 * u1[1])[1]
                     else [a1, P(-s1 * u1[0], -s1 * u1[1])], DBLUE, 1.5, dashed=True))
    p2a, p2b = a2, P(-s2 * u2[0], -s2 * u2[1])
    els.append(curve("sv2", [p2a, p2b] if p2a[1] > p2b[1] else [p2b, p2a], DBLUE, 1.5, dashed=True))
    # correspondence arrows: point on circle -> its image (theta = 45 and 225 degrees)
    for th, nm in ((45, "c45"), (225, "c225")):
        c, s = math.cos(math.radians(th)), math.sin(math.radians(th))
        p, q = P(c, s), P(2 * c + s, s)
        els.append(arrow(nm, p[0], p[1], q[0], q[1], MUTED, 2, dashed=True))
        els.append(dot(nm + "-p", p[0], p[1], GRAY))
        els.append(dot(nm + "-q", q[0], q[1], BLUE, color=DBLUE))
    # e2 -> A e2 shear guide
    e2, ae2 = P(0, 1), P(1, 1)
    # basis vectors and their images
    els.append(arrow("Ae1", X0, Y0, P(2, 0)[0], Y0, DBLUE, 6))
    els.append(arrow("Ae2", X0, Y0, ae2[0], ae2[1], DBLUE, 6))
    els.append(arrow("e1", X0, Y0, P(1, 0)[0], Y0, INK, 4))
    els.append(arrow("e2", X0, Y0, e2[0], e2[1], INK, 4))
    # labels placed inside grid cells, clear of every stroke
    els.append(text("l-e1", P(0.35, 0)[0], Y0 + 40, "e₁", 26, 3, w=40))
    els.append(text("l-e2", P(0.1, 0)[0], P(0, 0.62)[1], "e₂", 26, 3, w=40))
    els.append(text("l-Ae1", P(2.04, 0)[0], P(0, -0.3)[1], "A e₁ = (2, 0)", 22, 3, w=180, color=DBLUE))
    els.append(text("l-Ae2", P(1.04, 0)[0], P(0, 1.32)[1], "A e₂ = (1, 1)", 22, 3, w=180, color=DBLUE))
    els.append(text("l-circ", P(-2.95, 0)[0], P(0, -1.2)[1], "unit circle\n‖x‖ = 1", 24, 3, w=180))
    els.append(text("l-ell", P(1.06, 0)[0], P(0, -1.2)[1], "A·circle:\nan ellipse", 24, 3, w=180, color=DBLUE))
    # right-hand panel
    PX = 1580
    els.append(rect("mat", PX, 220, 700, 200, BLUE, "A =  ( 2   1 )\n     ( 0   1 )", sw=2, fs=30, fam=3))
    els.append(text("cols", PX, 460,
                    "column 1 = A e₁ = (2, 0)\ncolumn 2 = A e₂ = (1, 1)", 24, 3, w=700))
    els.append(text("svals", PX, 600,
                    f"semi-axes σ₁ = {s1:.3f}\n          σ₂ = {s2:.3f}", 24, 3, w=700, color=DBLUE))
    els.append(text("area", PX, 740,
                    f"area: π·σ₁·σ₂ = π·det A = 2π\nthe circle's area doubles", 24, 3, w=700))
    els.append(text("pts", PX, 880,
                    "dashed grey: a point on the\ncircle and where A sends it,\n(0.71, 0.71) ↦ (2.12, 0.71)", 24, 1, w=700))
    els.append(rect("scale", PX, 1040, 700, 100, GRAY, "grid: 1 unit = 200 px", sw=2, fs=24, fam=3))
    write(2, title, "Multiplying by A sends the unit circle to an exact ellipse whose axes are the singular values.",
          caption, els)


# ---------------------------------------------------------------- 3 filmstrip
def npdf(x, m, s):
    return math.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def ncdf(x, m, s):
    return 0.5 * (1 + math.erf((x - m) / (s * math.sqrt(2))))


def pt(x, t):
    return (1 - t) * npdf(x, 0, 1) + t * 0.5 * (npdf(x, -2, 0.5) + npdf(x, 2, 0.5))


def Ft(x, t):
    return (1 - t) * ncdf(x, 0, 1) + t * 0.5 * (ncdf(x, -2, 0.5) + ncdf(x, 2, 0.5))


def filmstrip():
    title = "The density filmstrip: p₀ becomes p₁ as t runs from 0 to 1"
    caption = "Generative models move probability mass over time: the centre drains while two modes at ±2 fill up."
    els = header(title, caption)
    L, W, G = 80, 416, 40
    XMIN, XMAX = -3.5, 3.5
    BASE, YS = 700, 1100.0       # density 1.0 = 1100 px, same for every panel
    ts = [0, 0.25, 0.5, 0.75, 1]
    names = ["t = 0   p₀ = N(0, 1)", "t = 0.25", "t = 0.5", "t = 0.75", "t = 1   p₁ bimodal"]
    MB, MS, BW = 1260, 520.0, 100   # mass bars: baseline, px per unit mass, bar width
    for i, t in enumerate(ts):
        left = L + i * (W + G)
        X = lambda x: left + (x - XMIN) * W / (XMAX - XMIN)
        xs = [XMIN + k * 0.125 for k in range(int((XMAX - XMIN) / 0.125) + 1)]
        pts = [(X(XMIN), BASE)] + [(X(x), BASE - YS * pt(x, t)) for x in xs] + [(X(XMAX), BASE), (X(XMIN), BASE)]
        els.append(text(f"name{i}", left, 200, names[i], 24, 3, w=W))
        els.append(hline(f"base{i}", left, BASE, W, INK, 2))
        els.append(curve(f"dens{i}", pts, DBLUE, 3, bg=BLUE))
        for v in (-2, 0, 2):
            s = f"{v}".replace("-", "−")
            els.append(text(f"tick{i}{v}", X(v), BASE + 12, s, 22, 3, w=40, center=True))
        # mass in three bins
        lm = Ft(-1, t)
        cm = Ft(1, t) - Ft(-1, t)
        rm = 1 - Ft(1, t)
        for j, (xv, m, bg, nm) in enumerate(((-2, lm, BLUE, "x<−1"), (0, cm, GRAY, "|x|<1"), (2, rm, BLUE, "x>1"))):
            h = MS * m
            cx = X(xv)
            els.append(rect(f"bar{i}{j}", cx - BW / 2, MB - h, BW, h, bg, None, sw=2))
            els.append(text(f"pc{i}{j}", cx, MB - h - 40, f"{round(100 * m)}%", 22, 3, w=60, center=True))
            els.append(text(f"bin{i}{j}", cx, MB + 12, nm, 22, 3, w=100, center=True))
        els.append(hline(f"mbase{i}", left, MB, W, INK, 2))
    els.append(text("row2", L, 790, "probability mass in three bins, same scale in every frame", 24, 1, w=900))
    els.append(arrow("time", L, 1380, L + 5 * W + 4 * G, 1380, INK, 4))
    els.append(text("t0", L, 1410, "t = 0: noise", 24, 1, w=220))
    els.append(text("t1", L + 5 * W + 4 * G - 260, 1410, "t = 1: data", 24, 1, w=260))
    els.append(text("tmid", 1200, 1410, "p_t = (1−t)·N(0,1) + t·½(N(−2,0.5²) + N(2,0.5²))", 24, 3, w=760, center=True))
    write(3, title, "The same probability mass shifts from one central Gaussian into two modes at plus and minus two as t grows.",
          caption, els)


# ---------------------------------------------------------------- 4 landscape
def landscape():
    title = "The loss landscape: f(x, y) = ½(x² + 10y²)"
    caption = "The shape of the landscape decides the path: steep walls make plain gradient descent zig-zag."
    els = header(title, caption)
    S = 170
    X0, Y0 = 1200, 600
    XR, YR = 6.4, 2.2
    P = lambda x, y: (X0 + S * x, Y0 - S * y)
    # frame axes at the bottom and left of the plot
    bx, by = P(-XR, -YR)
    els.append(arrow("ax-x", bx, by, P(XR, 0)[0], by, INK, 2))
    els.append(arrow("ax-y", bx, by, bx, P(0, YR)[1] - 10, INK, 2))
    for gx in range(-6, 7, 2):
        s = f"{gx}".replace("-", "−")
        els.append(vline(f"tkx{gx}", P(gx, 0)[0], by, 12, INK, 2))
        els.append(text(f"tx{gx}", P(gx, 0)[0], by + 16, s, 22, 3, w=40, center=True))
    for gy in (-2, -1, 0, 1, 2):
        s = f"{gy}".replace("-", "−")
        els.append(hline(f"tky{gy}", bx - 12, P(0, gy)[1], 12, INK, 2))
        els.append(text(f"ty{gy}", bx - 56, P(0, gy)[1] - 14, s, 22, 3, w=40))
    # contours f = a^2/2 for a = 1..6: semi-axes a and a/sqrt(10)
    N = 120
    for a in range(1, 7):
        b = a / math.sqrt(10)
        pts = [P(a * math.cos(2 * math.pi * k / N), b * math.sin(2 * math.pi * k / N)) for k in range(N)]
        els.append(curve(f"c{a}", pts, MUTED, 1.5, closed=True))
        if a >= 2:
            lv = a * a / 2
            s = f"f = {lv:g}"
            els.append(text(f"lv{a}", P(a, 0)[0] + 6, Y0 + 4, s, 22, 3, w=round(tw(s, 22, 3) / 10) * 10 + 20))
    # gradient descent, step 0.18: x <- 0.82x, y <- -0.8y
    f = lambda x, y: 0.5 * (x * x + 10 * y * y)
    eta = 0.18
    gd = [(-4.0, 1.2)]
    for _ in range(10):
        x, y = gd[-1]
        gd.append((x - eta * x, y - eta * 10 * y))
    # preconditioned GD, P = H^-1 = diag(1, 1/10), step 0.5: straight line, halves each step
    pc = [(-4.0, 1.2)]
    for _ in range(4):
        x, y = pc[-1]
        pc.append((x - 0.5 * x, y - 0.5 * (10 * y) / 10))
    for k in range(len(gd) - 1):
        p, q = P(*gd[k]), P(*gd[k + 1])
        els.append(arrow(f"gd{k}", p[0], p[1], q[0], q[1], DAMBER, 3.5))
    for k in range(len(pc) - 1):
        p, q = P(*pc[k]), P(*pc[k + 1])
        els.append(arrow(f"pc{k}", p[0], p[1], q[0], q[1], DBLUE, 4))
    els.append(dot("start", *P(-4, 1.2), GRAY, d=12))
    els.append(dot("min", *P(0, 0), GREEN, d=12, color="#2f9e44"))
    els.append(text("l-start", P(-6.2, 0)[0], P(0, 2.0)[1], "start (−4, 1.2), f = 15.2", 22, 3, w=380))
    els.append(text("l-min", P(0.3, 0)[0], P(0, -1.94)[1], "minimum (0, 0)", 22, 3, w=200, color="#2f9e44"))
    els.append(arrow("a-min", P(0.32, 0)[0], P(0, -1.9)[1], P(0.03, 0)[0], P(0, -0.07)[1], "#2f9e44", 2, end="min"))
    fgd, fpc = f(*gd[-1]), f(*pc[-1])
    LY = 1080
    els.append(rect("leg-gd", 80, LY, 1060, 140, AMBER,
                    f"plain GD, step 0.18: y flips sign each step\n10 steps reach f = {fgd:.3f}", sw=2, fs=24, fam=1))
    els.append(rect("leg-pc", 1260, LY, 1060, 140, BLUE,
                    f"preconditioned GD, P = H⁻¹, step 0.5: straight\n4 steps reach f = {fpc:.3f}", sw=2, fs=24, fam=1))
    els.append(text("kappa", 80, LY + 180,
                    "curvature 1 along x, 10 along y: condition number κ = 10, contour axis ratio √10 ≈ 3.16; equal axes, 1 unit = 170 px",
                    22, 1, w=2240))
    write(4, title, "On an anisotropic bowl plain gradient descent zig-zags while a preconditioned step walks straight to the minimum.",
          caption, els)
    return gd, pc


ladder()
plane()
filmstrip()
gd, pc = landscape()
print("GD iterates:", [(round(x, 3), round(y, 3)) for x, y in gd])
print("PC iterates:", [(round(x, 3), round(y, 3)) for x, y in pc])
