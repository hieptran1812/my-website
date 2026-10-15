#!/usr/bin/env node
/*
 * author-scene.mjs — normalize a blog-writer diagram payload into a complete
 * Excalidraw scene JSON file, validating the same containment, overlap,
 * coverage, and font-family invariants Phase C used to enforce at runtime
 * against the live MCP canvas.
 *
 * Input shape (the same shape blog-writer used to pass to
 * mcp__excalidraw__batch_create_elements, plus optional `export` block):
 *
 *   {
 *     "title": "Optional figure thesis (used for filename only)",
 *     "elements": [
 *       { "type": "rectangle", "x": ..., "y": ..., "width": ..., "height": ...,
 *         "label": "Text inside the box" | { "text": "..." }, ... },
 *       { "type": "text", "x": ..., "y": ..., "text": "Caption", "fontSize": 22,
 *         "fontFamily": 1, ... },
 *       ...
 *     ],
 *     "export": { "padding": 48, "theme": "light", "minWidth": 1600, "minHeight": 900 }
 *   }
 *
 * Output: a JSON file that render-scene.mjs can consume directly. The CLI
 * exits non-zero on any validation failure with a precise message naming the
 * offending element id / index.
 *
 * Usage:
 *   node author-scene.mjs <input.json> <output.json>
 *
 * Pure Node, zero deps.
 */
import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'

const CANVAS_W = 2400
const CANVAS_H = 1600
const COVERAGE_FLOOR = 0.7
const MIN_GAP = 40
const MIN_FONT_BODY = 22
const MIN_FONT_TITLE = 32
const ALLOWED_FONTS = new Set([1, 3]) // Virgil prose, Cascadia code
const ALLOWED_BG = new Set([
  'transparent', '#a5d8ff', '#ffec99', '#ffc9c9', '#b2f2bb', '#d0bfff', '#e9ecef',
])

function die(msg, code = 1) {
  process.stderr.write(`author-scene: ${msg}\n`)
  process.exit(code)
}

function rngId(prefix = 'el') {
  return `${prefix}-${Math.random().toString(36).slice(2, 10)}`
}

// Excalidraw uses Math.random() seeds and version fields; we fill enough for
// convertToExcalidrawElements (called inside the headless harness) to do the
// rest. Don't over-specify — convertToExcalidrawElements normalizes the gaps.
function defaultsForElement(el) {
  // Arrows render with thinner strokes and zero roughness so the routing
  // reads as a clean diagram line, not a sketchy bold gesture. Containers
  // keep the sketchy roughness 1 for visual warmth.
  const isArrow = el.type === 'arrow' || el.type === 'line'
  return {
    id: el.id ?? rngId(el.type === 'text' ? 'tx' : 'sh'),
    angle: 0,
    strokeColor: el.strokeColor ?? (isArrow ? '#343a40' : '#1e1e1e'),
    backgroundColor: el.backgroundColor ?? 'transparent',
    fillStyle: el.fillStyle ?? 'hachure',
    strokeWidth: el.strokeWidth ?? (isArrow ? 1.25 : 2),
    strokeStyle: el.strokeStyle ?? 'solid',
    roughness: el.roughness ?? (isArrow ? 0 : 1),
    opacity: el.opacity ?? 100,
    roundness: el.roundness ?? (el.type === 'rectangle' ? { type: 3 } : null),
    seed: el.seed ?? Math.floor(Math.random() * 2 ** 31),
    version: el.version ?? 1,
    versionNonce: el.versionNonce ?? Math.floor(Math.random() * 2 ** 31),
    isDeleted: false,
    boundElements: el.boundElements ?? null,
    updated: Date.now(),
    link: el.link ?? null,
    locked: false,
  }
}

export function normalize(input) {
  const out = []
  const elements = Array.isArray(input.elements) ? input.elements : []

  for (const raw of elements) {
    if (!raw || typeof raw !== 'object') continue
    if (raw.type === 'text') {
      out.push({
        ...defaultsForElement(raw),
        type: 'text',
        x: raw.x ?? 0,
        y: raw.y ?? 0,
        width: raw.width ?? estTextWidth(raw.text || '', raw.fontSize ?? 22),
        height: raw.height ?? estTextHeight(raw.text || '', raw.fontSize ?? 22),
        text: raw.text ?? '',
        fontSize: raw.fontSize ?? 22,
        fontFamily: raw.fontFamily ?? 1,
        textAlign: raw.textAlign ?? 'left',
        verticalAlign: raw.verticalAlign ?? 'top',
        baseline: raw.baseline ?? 18,
        containerId: raw.containerId ?? null,
        originalText: raw.originalText ?? raw.text ?? '',
        lineHeight: raw.lineHeight ?? 1.25,
      })
    } else {
      // Shape-with-optional-label (the templates' shape).
      const shapeId = raw.id ?? rngId('sh')
      const shape = {
        ...defaultsForElement({ ...raw, id: shapeId }),
        type: raw.type,
        x: raw.x ?? 0,
        y: raw.y ?? 0,
        width: raw.width ?? 200,
        height: raw.height ?? 100,
      }
      if (raw.type === 'arrow' || raw.type === 'line') {
        shape.points = raw.points ?? [[0, 0], [raw.width ?? 0, raw.height ?? 0]]
        shape.lastCommittedPoint = raw.lastCommittedPoint ?? null
        shape.startBinding = raw.startBinding ?? null
        shape.endBinding = raw.endBinding ?? null
        shape.startArrowhead = raw.startArrowhead ?? null
        shape.endArrowhead = raw.endArrowhead ?? 'arrow'
      }
      const labelText = typeof raw.label === 'string' ? raw.label : raw.label?.text
      if (labelText) {
        const labelFontSize = raw.label?.fontSize ?? 22
        const labelFontFamily = raw.label?.fontFamily ?? 1
        const labelId = rngId('tx')
        shape.boundElements = [
          ...(shape.boundElements || []),
          { type: 'text', id: labelId },
        ]
        out.push(shape)
        const tw = estTextWidth(labelText, labelFontSize, labelFontFamily)
        const th = estTextHeight(labelText, labelFontSize)
        // Center the text element within the container; Excalidraw uses x/y as
        // the rendered position when the bound-text alignment hooks don't fire
        // headlessly, so we pre-center to be safe.
        const tx = shape.x + Math.max(0, (shape.width - tw) / 2)
        const ty = shape.y + Math.max(0, (shape.height - th) / 2)
        out.push({
          ...defaultsForElement({ id: labelId }),
          type: 'text',
          x: tx,
          y: ty,
          width: tw,
          height: th,
          text: labelText,
          fontSize: labelFontSize,
          fontFamily: labelFontFamily,
          textAlign: 'center',
          verticalAlign: 'middle',
          baseline: Math.round(labelFontSize * 0.8),
          containerId: shape.id,
          originalText: labelText,
          lineHeight: 1.25,
        })
      } else {
        out.push(shape)
      }
    }
  }
  return out
}

function estTextWidth(text, fontSize, family = 1) {
  const factor = family === 3 ? 0.62 : 0.6
  const lines = String(text).split('\n')
  const longest = Math.max(0, ...lines.map((l) => l.length))
  return Math.ceil(longest * fontSize * factor)
}
function estTextHeight(text, fontSize) {
  const lines = String(text).split('\n').length || 1
  return Math.ceil(lines * fontSize * 1.25)
}

// Liang-Barsky segment ↔ axis-aligned-rect intersection test. Returns true
// when the segment p0→p1 passes through (or into) the rectangle. Used to catch
// text labels drawn on top of arrow strokes — the bbox-overlap rule skips
// arrows, so a label sitting on a routing channel would otherwise slip through.
function segIntersectsRect(p0, p1, x0, y0, x1, y1) {
  const ax = p0[0], ay = p0[1]
  const dx = p1[0] - ax, dy = p1[1] - ay
  let t0 = 0, t1 = 1
  const edges = [
    [-dx, ax - x0], [dx, x1 - ax],
    [-dy, ay - y0], [dy, y1 - ay],
  ]
  for (const [p, q] of edges) {
    if (p === 0) {
      if (q < 0) return false // parallel and outside this slab
    } else {
      const r = q / p
      if (p < 0) { if (r > t1) return false; if (r > t0) t0 = r }
      else { if (r < t0) return false; if (r < t1) t1 = r }
    }
  }
  return t0 <= t1
}

// Shortest distance from point (px,py) to segment p0→p1. Used by the
// anti-dead-space gate to treat the corridor a few px either side of an arrow
// stroke as "occupied" — a connector crossing an otherwise-blank region is
// legitimate content, not empty space.
function ptSegDist(px, py, p0, p1) {
  const x0 = p0[0], y0 = p0[1]
  const dx = p1[0] - x0, dy = p1[1] - y0
  const len2 = dx * dx + dy * dy
  if (len2 === 0) return Math.hypot(px - x0, py - y0)
  let t = ((px - x0) * dx + (py - y0) * dy) / len2
  t = Math.max(0, Math.min(1, t))
  return Math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))
}

function bbox(els) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity
  for (const e of els) {
    x0 = Math.min(x0, e.x)
    y0 = Math.min(y0, e.y)
    x1 = Math.max(x1, e.x + (e.width || 0))
    y1 = Math.max(y1, e.y + (e.height || 0))
  }
  if (!isFinite(x0)) return { x0: 0, y0: 0, x1: 0, y1: 0, w: 0, h: 0 }
  return { x0, y0, x1, y1, w: x1 - x0, h: y1 - y0 }
}

// Stopwords for the information-density gate. Keep short — the goal is to
// reject placeholder labels (A/B/C, foo/bar/baz, "step 1"/"step 2"), not to
// score writing quality.
const STOPWORDS = new Set([
  'a', 'an', 'the', 'of', 'for', 'to', 'and', 'or', 'in', 'on', 'at', 'by',
  'is', 'be', 'as', 'with', 'from', 'into', 'this', 'that', 'these', 'those',
  'step', 'foo', 'bar', 'baz', 'quux',
])

function tokenize(text) {
  return String(text)
    .toLowerCase()
    .split(/[^a-z0-9_]+/)
    .filter((t) => t.length > 1 && !STOPWORDS.has(t))
}

function validateMetadata(input) {
  const errors = []
  // The layout-scene.mjs frontend stamps _claim and _caption onto its
  // expanded payloads. Element-level callers may also set them directly. If
  // they're absent we treat this as the legacy element-only path and skip
  // — the layout engine is the right place to enforce these for DSL inputs.
  const claim = String(input._claim || '').trim()
  if (claim) {
    const words = claim.split(/\s+/).filter(Boolean)
    if (words.length < 8) {
      errors.push(`claim too short (${words.length} words, need ≥ 8): "${claim}"`)
    }
  }
  return errors
}

export function validate(els, input = {}) {
  const errors = [...validateMetadata(input)]

  // 1. fontFamily must be set explicitly on every text.
  for (const e of els) {
    if (e.type !== 'text') continue
    if (!ALLOWED_FONTS.has(e.fontFamily)) {
      errors.push(`text element ${e.id}: fontFamily must be 1 (Virgil) or 3 (Cascadia), got ${e.fontFamily}`)
    }
    if (e.fontSize < MIN_FONT_BODY) {
      errors.push(`text element ${e.id}: fontSize ${e.fontSize} below body floor ${MIN_FONT_BODY}`)
    }
  }

  // 1b. Free-floating text: declared width must fit rendered text.
  // Excalidraw renders at the real character widths, so a too-narrow declared
  // bbox lets text overflow into adjacent nodes. This catches edge labels,
  // axis ticks, annotations — anywhere a text element doesn't have a containerId.
  for (const e of els) {
    if (e.type !== 'text' || e.containerId) continue
    const rendered = estTextWidth(e.text || '', e.fontSize, e.fontFamily)
    // Allow a small slack (8 px) for rounding; a textAlign:right element with
    // generous declared width is fine. Only fail when declared < rendered.
    if (e.width < rendered - 8) {
      errors.push(
        `text element ${e.id}: declared width ${e.width} below rendered ${rendered} — text "${oneLine(e.text)}" will overflow its bbox to the right and may cross adjacent shapes`,
      )
    }
  }

  // 2. Containment: bound text must fit its container.
  const byId = new Map(els.map((e) => [e.id, e]))
  for (const e of els) {
    if (e.type !== 'text' || !e.containerId) continue
    const c = byId.get(e.containerId)
    if (!c) {
      errors.push(`text element ${e.id}: containerId ${e.containerId} references missing element`)
      continue
    }
    const reqW = estTextWidth(e.text, e.fontSize, e.fontFamily) + 2 * 24
    const reqH = estTextHeight(e.text, e.fontSize) + 2 * 20
    if (c.width < reqW) {
      errors.push(
        `container ${c.id} too narrow for label "${oneLine(e.text)}": width ${c.width} < required ${reqW} (chars × fontSize × 0.6 + 48)`,
      )
    }
    if (c.height < reqH) {
      errors.push(
        `container ${c.id} too short for label "${oneLine(e.text)}": height ${c.height} < required ${reqH}`,
      )
    }
  }

  // 3. Overlap: any two non-parent/child shapes overlapping is a hard fail.
  //    Arrows are routes, not bodies — their bbox is the bounding rectangle
  //    of their points, which may legitimately cross node bboxes (especially
  //    for orthogonal jogs in dense graphs). Arrow correctness is enforced
  //    by their bindings (startBinding / endBinding) and verified visually
  //    on the rendered PNG; bbox-overlap is the wrong tool for them.
  const childOf = new Map()
  for (const e of els) {
    if (e.type === 'text' && e.containerId) childOf.set(e.id, e.containerId)
  }
  const isRelated = (a, b) => childOf.get(a.id) === b.id || childOf.get(b.id) === a.id
  for (let i = 0; i < els.length; i++) {
    for (let j = i + 1; j < els.length; j++) {
      const a = els[i], b = els[j]
      if (isRelated(a, b)) continue
      if (a.type === 'arrow' || b.type === 'arrow') continue
      // Free-floating text labels are allowed to occupy the same y-band as a
      // distant shape only if their bbox doesn't actually overlap.
      const ax2 = a.x + (a.width || 0), ay2 = a.y + (a.height || 0)
      const bx2 = b.x + (b.width || 0), by2 = b.y + (b.height || 0)
      const overlapX = Math.min(ax2, bx2) - Math.max(a.x, b.x)
      const overlapY = Math.min(ay2, by2) - Math.max(a.y, b.y)
      if (overlapX > 0 && overlapY > 0) {
        errors.push(`overlap between ${a.id} (${a.type}) and ${b.id} (${b.type}): ${overlapX.toFixed(0)}×${overlapY.toFixed(0)} px`)
      }
    }
  }

  // 3c. Arrow-through-node collision: every polyline segment of every arrow
  //     must clear every non-endpoint shape's bbox. Endpoints (source/target,
  //     identified by startBinding/endBinding.elementId) and parent/child
  //     containers are exempt — the arrow legitimately lands on those. A 6 px
  //     inset on the obstacle bbox prevents false positives for arrows that
  //     legally graze an adjacent node along the routing channel. This catches
  //     fan-out/fan-in tangles in `graph` figures (e.g. h_t → head_2 piercing
  //     head_1) and any mis-routed hand-authored `raw` arrow.
  for (const ar of els) {
    if (ar.type !== 'arrow') continue
    const srcId = ar.startBinding?.elementId
    const dstId = ar.endBinding?.elementId
    const pts = (ar.points || []).map(([px, py]) => [ar.x + px, ar.y + py])
    if (pts.length < 2) continue
    for (const s of els) {
      if (s.type === 'arrow' || s.type === 'line' || s.type === 'text') continue
      if (s.id === srcId || s.id === dstId) continue
      if (isRelated(ar, s)) continue
      const x0 = s.x + 6, y0 = s.y + 6
      const x1 = s.x + (s.width || 0) - 6, y1 = s.y + (s.height || 0) - 6
      if (x1 <= x0 || y1 <= y0) continue
      let hit = false
      for (let k = 0; k + 1 < pts.length && !hit; k++) {
        if (segIntersectsRect(pts[k], pts[k + 1], x0, y0, x1, y1)) hit = true
      }
      if (hit) {
        errors.push(
          `arrow ${ar.id} crosses non-endpoint shape ${s.id} (${s.type}) — re-route through a clear channel (see diagram-authoring.md §Arrows: every in/out leg must sit in empty space; push the channel past blocking siblings or detour below the block)`,
        )
        break
      }
    }
  }

  // 3b. Text-over-arrow collision: a label drawn across an arrow's stroke is
  //     illegible and the figure-quality rules forbid it ("edge labels live in
  //     the gap, not on the line"). Rule 3 skips arrows, so check every
  //     free-floating text label against each arrow's actual polyline segments.
  //     The label bbox is inset 4 px so a label that merely grazes an
  //     arrowhead corner isn't flagged.
  for (const t of els) {
    if (t.type !== 'text' || t.containerId) continue
    const tx0 = t.x + 4, ty0 = t.y + 4
    const tx1 = t.x + (t.width || 0) - 4, ty1 = t.y + (t.height || 0) - 4
    if (tx1 <= tx0 || ty1 <= ty0) continue
    for (const ar of els) {
      if (ar.type !== 'arrow' && ar.type !== 'line') continue
      const pts = (ar.points || []).map(([px, py]) => [ar.x + px, ar.y + py])
      let hit = false
      for (let k = 0; k + 1 < pts.length && !hit; k++) {
        if (segIntersectsRect(pts[k], pts[k + 1], tx0, ty0, tx1, ty1)) hit = true
      }
      if (hit) {
        errors.push(
          `text element ${t.id} ("${oneLine(t.text)}") is drawn over ${ar.type} ${ar.id} — edge labels must sit in the gap clear of the stroke (above a horizontal run, beside a vertical one)`,
        )
        break
      }
    }
  }

  // 4. Coverage: at least one axis must reach 70% (the dominant axis for the
  // figure shape — pipelines are wide-and-short, stacks are tall-and-narrow,
  // matrices fill both). The other axis must still reach 40% so the figure
  // doesn't look empty. The renderer crops to bbox via exportPadding, so
  // unused canvas in the minor axis isn't wasted on the rendered PNG.
  const bb = bbox(els.filter((e) => !(e.type === 'text' && e.containerId)))
  const covW = bb.w / CANVAS_W
  const covH = bb.h / CANVAS_H
  const covMax = Math.max(covW, covH)
  const covMin = Math.min(covW, covH)
  if (covMax < COVERAGE_FLOOR || covMin < 0.4) {
    errors.push(
      `bounding-box coverage too low: ${(covW * 100).toFixed(0)}% × ${(covH * 100).toFixed(0)}% (need dominant axis ≥ ${(COVERAGE_FLOOR * 100).toFixed(0)}% and minor axis ≥ 40%). Scale elements up or add information density.`,
    )
  }

  // 5. Color palette: backgroundColor on shapes must be in the semantic palette.
  for (const e of els) {
    if (e.type === 'text') continue
    const bg = (e.backgroundColor || 'transparent').toLowerCase()
    if (!ALLOWED_BG.has(bg)) {
      errors.push(`shape ${e.id}: backgroundColor ${bg} is not in the semantic palette (${[...ALLOWED_BG].join(', ')})`)
    }
  }

  // 6. Caption element: if the input declared a _caption, the scene must
  // contain a matching text element near the top of the canvas (within the
  // first 220 px) at a fontSize >= 26. The layout engine produces this
  // automatically; element-level callers can opt in by stamping _caption.
  const caption = String(input._caption || '').trim()
  if (caption) {
    const titleText = String(els.find((e) => e.id === 'title')?.text || '').trim()
    // Caption text may be word-wrapped (\n inserted between words) to fit
    // canvas width. Compare on normalized whitespace so the match survives.
    const norm = (s) => String(s).replace(/\s+/g, ' ').trim()
    const capNorm = norm(caption)
    const matchingCap = els.find(
      (e) => e.type === 'text' && norm(e.text) === capNorm,
    )
    if (!matchingCap) {
      errors.push(`scene declares _caption but no text element matches it: "${oneLine(caption)}"`)
    } else if (matchingCap.fontSize < 26) {
      errors.push(`caption element fontSize ${matchingCap.fontSize} below 26 — caption must read as a thesis line, not a body label`)
    } else if (matchingCap.y > 220) {
      errors.push(`caption element placed at y=${matchingCap.y}; must sit in the top header band (y ≤ 220)`)
    }
    if (titleText && norm(titleText).toLowerCase() === capNorm.toLowerCase()) {
      errors.push('caption duplicates the title verbatim — caption must add information, not restate the title')
    }
  }

  // 7. Information density: count unique non-stopword tokens across all
  // labels (text elements + bound-text labels). Reject placeholders.
  const tokens = new Set()
  for (const e of els) {
    if (e.type === 'text') for (const t of tokenize(e.text)) tokens.add(t)
    else if (typeof e.label === 'string') for (const t of tokenize(e.label)) tokens.add(t)
    else if (e.label && typeof e.label.text === 'string') for (const t of tokenize(e.label.text)) tokens.add(t)
  }
  if (tokens.size < 6) {
    errors.push(`information density too low: only ${tokens.size} unique non-stopword tokens across all labels (need ≥ 6). Replace placeholder labels with concrete component / data / quantity names.`)
  }

  // 8. Anti-dead-space: the renderer crops the export to the content bounding
  //    box, so any empty region INSIDE that bbox ships as a visible blank band
  //    — exactly the "meaningless empty space" we forbid. Sample a grid over
  //    the content bbox; a sample is "occupied" if it lands inside any
  //    node/text bbox or within TOL px of any arrow/line segment. Reject a
  //    figure whose bbox is mostly empty (global) or that has a blank quadrant
  //    (a dead corner/band). Thresholds are deliberately conservative: airy but
  //    legitimate DAGs spread nodes + connectors across every quadrant and clear
  //    these floors comfortably; only genuinely under-filled layouts fail.
  const bodyEls = els.filter((e) => !(e.type === 'text' && e.containerId))
  const solids = els.filter(
    (e) => e.type !== 'arrow' && e.type !== 'line' && (e.width || 0) > 0 && (e.height || 0) > 0,
  )
  const segs = []
  for (const ar of els) {
    if (ar.type !== 'arrow' && ar.type !== 'line') continue
    const pts = (ar.points || []).map(([px, py]) => [ar.x + px, ar.y + py])
    for (let k = 0; k + 1 < pts.length; k++) segs.push([pts[k], pts[k + 1]])
  }
  const dbb = bbox(bodyEls)
  if (dbb.w > 40 && dbb.h > 40) {
    const NX = 48, NY = 32, TOL = 16
    const occupied = (gx, gy) => {
      for (const s of solids) {
        if (gx >= s.x && gx <= s.x + (s.width || 0) && gy >= s.y && gy <= s.y + (s.height || 0)) return true
      }
      for (const [p0, p1] of segs) {
        if (ptSegDist(gx, gy, p0, p1) <= TOL) return true
      }
      return false
    }
    const cx = dbb.x0 + dbb.w / 2, cy = dbb.y0 + dbb.h / 2
    let total = 0, hit = 0
    const qHit = [0, 0, 0, 0], qTot = [0, 0, 0, 0]
    for (let i = 0; i < NX; i++) {
      const gx = dbb.x0 + (i + 0.5) * dbb.w / NX
      for (let j = 0; j < NY; j++) {
        const gy = dbb.y0 + (j + 0.5) * dbb.h / NY
        const qi = (gx < cx ? 0 : 1) + (gy < cy ? 0 : 2)
        total++; qTot[qi]++
        if (occupied(gx, gy)) { hit++; qHit[qi]++ }
      }
    }
    const frac = total ? hit / total : 1
    if (frac < 0.1) {
      errors.push(
        `dead space: only ${(frac * 100).toFixed(0)}% of the figure's bounding box is occupied by shapes/arrows — the rest crops to a blank frame. Add internal structure or tighten the layout so content fills the figure (see diagram-authoring.md §No meaningless empty space).`,
      )
    }
    const qName = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
    for (let k = 0; k < 4; k++) {
      const f = qTot[k] ? qHit[k] / qTot[k] : 1
      if (f < 0.03) {
        errors.push(
          `dead space: the ${qName[k]} quadrant is blank (${(f * 100).toFixed(0)}% filled) — a cropped export shows it as a meaningless empty band. Rebalance the layout (move a node/annotation/connector there) instead of stretching one card to cover it.`,
        )
      }
    }
  }

  return errors
}

function oneLine(s) {
  return String(s).replace(/\s+/g, ' ').slice(0, 60)
}

// In-process entrypoint for callers like layout-scene.mjs that already hold
// the input as an object — skips the disk round-trip and Node startup of
// spawning author-scene as a subprocess.
export function buildScene(input) {
  const elements = normalize(input)
  const errors = validate(elements, input)
  if (errors.length) {
    const e = new Error(`author-scene: validation failed\n  - ${errors.join('\n  - ')}`)
    e.errors = errors
    throw e
  }
  // Strip arrow bindings AFTER validation. The validator uses them to identify
  // each arrow's endpoint shapes (so rule 3c can exempt source/target from
  // the crosses-non-endpoint check). Excalidraw's headless renderer, however,
  // re-routes any arrow whose endpoints are bound to live element positions,
  // overriding our explicit polyline `points` and visually collapsing the
  // route. Drop the bindings now that we no longer need them.
  for (const ar of elements) {
    if (ar.type !== 'arrow') continue
    ar.startBinding = null
    ar.endBinding = null
  }
  return {
    type: 'excalidraw',
    version: 2,
    source: 'blog-writer',
    elements,
    appState: {
      viewBackgroundColor: '#ffffff',
      gridSize: null,
    },
    files: input.files || {},
    export: {
      padding: input.export?.padding ?? 48,
      theme: input.export?.theme ?? 'light',
      background: input.export?.background ?? true,
      minWidth: input.export?.minWidth ?? 1600,
      minHeight: input.export?.minHeight ?? 900,
    },
  }
}

async function main() {
  const [, , inPath, outPath] = process.argv
  if (!inPath || !outPath) die('usage: author-scene.mjs <input.json> <output.json>', 2)

  const raw = await readFile(inPath, 'utf8').catch((e) => die(`cannot read ${inPath}: ${e.message}`))
  let input
  try { input = JSON.parse(raw) } catch (e) { die(`${inPath} is not JSON: ${e.message}`) }

  let scene
  try { scene = buildScene(input) }
  catch (e) {
    process.stderr.write(`author-scene: validation failed for ${inPath}\n`)
    for (const err of (e.errors || [e.message])) process.stderr.write(`  - ${err}\n`)
    process.exit(3)
  }

  await mkdir(dirname(outPath), { recursive: true }).catch(() => {})
  await writeFile(outPath, JSON.stringify(scene, null, 2))
  process.stdout.write(`wrote ${outPath} (${scene.elements.length} elements)\n`)
}

// Only run main() when invoked as a CLI, not when imported as a module.
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((e) => die(e.stack || e.message || String(e)))
}
