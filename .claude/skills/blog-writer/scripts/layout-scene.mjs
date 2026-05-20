#!/usr/bin/env node
/*
 * layout-scene.mjs — high-level DSL → element-level Excalidraw scene JSON.
 *
 * The blog-writer skill emits one DSL JSON file per figure. This tool expands
 * it into a fully-positioned scene that author-scene.mjs accepts as input,
 * then it pipes through that validator and writes the final scene file.
 *
 * The DSL trades coordinate-level control for layout-by-construction: the
 * engines below size containers from labels, place nodes on a deterministic
 * grid for the figure's `type`, and pick colors from the semantic `kind`
 * field. This keeps every figure within the figure-quality mandates by
 * default — overlap, coverage, font, palette — without relying on the model
 * to remember every rule.
 *
 * Input shape (see diagrams/dsl-schema.json for the full spec):
 *
 *   {
 *     "type": "pipeline" | "stack" | "before-after" | "matrix" | "graph" | "raw",
 *     "title":   "<figure title>",
 *     "caption": "<one-sentence thesis under the title>",
 *     "claim":   "<≥ 8 word claim the figure proves>",
 *     "nodes":   [{ id, label, kind, anchor }],
 *     "edges":   [{ from, to, label }],
 *     ...type-specific fields...
 *     "raw":     { elements: [...] }   // escape hatch
 *   }
 *
 * Output: an Excalidraw scene JSON, ready for render-scene.mjs. Every
 * engine (including `graph`, which used to emit Mermaid) now produces
 * element-form scenes so the validator can enforce containment, overlap,
 * coverage, and palette invariants uniformly.
 *
 * Usage:
 *   node layout-scene.mjs <input.dsl.json> <output.scene.json>
 *
 * Pure Node, zero deps. Spawns author-scene.mjs as a subprocess for the
 * shared validator instead of importing it (keeps both files independently
 * runnable).
 */
import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { buildScene } from './author-scene.mjs'

const __dirname = dirname(fileURLToPath(import.meta.url))

const CANVAS_W = 2400
const CANVAS_H = 1600
const TITLE_FONT = 32
const CAPTION_FONT = 28
const BODY_FONT = 22
const EDGE_FONT = 22
const PAD_NODE = 24
const GAP = 80

const PALETTE = {
  primary: '#a5d8ff',
  caution: '#ffec99',
  danger: '#ffc9c9',
  success: '#b2f2bb',
  external: '#d0bfff',
  neutral: 'transparent',
}

function die(msg, code = 1) {
  process.stderr.write(`layout-scene: ${msg}\n`)
  process.exit(code)
}

function rngId(prefix = 'el') {
  return `${prefix}-${Math.random().toString(36).slice(2, 10)}`
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

// Compute the smallest container width that fits a label cleanly per the
// formula from author-scene.mjs (chars × fontSize × 0.6 + 48), rounded up
// to a 20-px grid for visual alignment across nodes.
function containerWidthFor(label, fontSize = BODY_FONT) {
  const tw = estTextWidth(label, fontSize)
  const w = tw + 2 * PAD_NODE * 2 // 24-px padding on each side, doubled for breathing room
  return Math.ceil(w / 20) * 20
}
function containerHeightFor(label, fontSize = BODY_FONT) {
  const th = estTextHeight(label, fontSize)
  const h = th + 2 * 28
  return Math.ceil(h / 20) * 20
}

// ── Common header (title + caption) ────────────────────────────────────────
function headerElements(title, caption) {
  if (!title) die('DSL missing required field: title')
  if (!caption) die('DSL missing required field: caption')
  // Center title and caption within the top band of the canvas.
  const titleW = estTextWidth(title, TITLE_FONT)
  const capW = estTextWidth(caption, CAPTION_FONT)
  return [
    {
      id: 'title',
      type: 'text',
      x: Math.round((CANVAS_W - titleW) / 2),
      y: 60,
      width: titleW,
      height: estTextHeight(title, TITLE_FONT),
      text: title,
      fontSize: TITLE_FONT,
      fontFamily: 1,
      textAlign: 'center',
    },
    {
      id: 'caption',
      type: 'text',
      x: Math.round((CANVAS_W - capW) / 2),
      y: 60 + estTextHeight(title, TITLE_FONT) + 16,
      width: capW,
      height: estTextHeight(caption, CAPTION_FONT),
      text: caption,
      fontSize: CAPTION_FONT,
      fontFamily: 1,
      textAlign: 'center',
    },
  ]
}

// Top of the body region. Title sits at y=60, caption ~16 px below the title's
// baseline, body ~28 px below the caption (was 80 — too airy). Gives every
// engine ~1380 px of vertical to play with on the 1600-tall canvas.
function bodyTopY() {
  return 60 + estTextHeight('X', TITLE_FONT) + 16 + estTextHeight('X', CAPTION_FONT) + 28
}
const BODY_BOTTOM_MARGIN = 60

function paletteFor(kind) {
  return PALETTE[kind ?? 'neutral'] ?? PALETTE.neutral
}

function assertOneAccent(nodes) {
  const accents = new Set()
  for (const n of nodes || []) {
    const k = n.kind ?? 'neutral'
    if (k !== 'neutral' && k !== 'external') accents.add(k)
  }
  if (accents.size > 3) {
    die(`figure uses too many strong accents (${[...accents].join(', ')}); use at most three distinct kinds (e.g. primary main path, caution bottleneck, success outcome) plus neutral/external for context`)
  }
}

// Re-wrap a single-line label across lines so the longest line is short —
// lets the pipeline pick a much bigger font without overflowing the card.
function wrapLabel(label, maxChars = 13) {
  const s = String(label)
  if (s.includes('\n')) return s
  const words = s.split(' ')
  if (words.length < 2) return s
  const lines = []
  let cur = ''
  for (const w of words) {
    if (cur && (cur + ' ' + w).length > maxChars) {
      lines.push(cur)
      cur = w
    } else {
      cur = cur ? cur + ' ' + w : w
    }
  }
  if (cur) lines.push(cur)
  return lines.join('\n')
}

// Greedy word-wrap to a maximum character width per line.
function wrapToChars(words, maxChars) {
  const lines = []
  let cur = ''
  for (const w of words) {
    if (cur && (cur + ' ' + w).length > maxChars) {
      lines.push(cur)
      cur = w
    } else {
      cur = cur ? cur + ' ' + w : w
    }
  }
  if (cur) lines.push(cur)
  return lines.join('\n')
}

// Pick the largest font (and matching line-wrap) that lets `rawLabel` fill a
// box of `boxW × boxH` without overflowing. Engines whose box size is driven by
// the layout grid (grid, matrix, stack, before-after, tree, timeline) used to
// emit plain-string labels, which author-scene.mjs renders at a fixed 22-px
// font — illegibly small inside a large cell. fitLabel returns a label object
// so the text scales WITH its container instead of floating tiny inside it.
//
// It sweeps every candidate line width, wraps to it, and computes the font the
// wrap allows on each axis (width-bound and height-bound); the wrap that yields
// the largest min(widthFont, heightFont) wins. Padding matches the validator's
// containment rule (estTextWidth + 48, estTextHeight + 40) so the result is
// always valid by construction.
function fitLabel(rawLabel, boxW, boxH, opts = {}) {
  const minFont = opts.minFont ?? BODY_FONT
  const maxFont = opts.maxFont ?? 44
  const family = opts.fontFamily ?? 1
  const factor = family === 3 ? 0.62 : 0.6
  const words = String(rawLabel ?? '')
    .replace(/\s+/g, ' ')
    .trim()
    .split(' ')
    .filter(Boolean)
  if (words.length === 0) return { text: '', fontSize: minFont, fontFamily: family }
  const availW = Math.max(20, boxW - 48)
  const availH = Math.max(20, boxH - 40)
  const longestWord = Math.max(1, ...words.map((w) => w.length))
  let best = null
  for (let maxChars = longestWord; maxChars <= Math.max(longestWord, 48); maxChars++) {
    const wrapped = wrapToChars(words, maxChars)
    const lines = wrapped.split('\n')
    const longestLine = Math.max(1, ...lines.map((l) => l.length))
    const fByW = availW / (longestLine * factor)
    const fByH = availH / (lines.length * 1.25)
    const f = Math.min(fByW, fByH)
    if (!best || f > best.f) best = { f, text: wrapped }
  }
  const fontSize = Math.max(minFont, Math.min(maxFont, Math.floor(best.f)))
  return { text: best.text, fontSize, fontFamily: family }
}

// Build an arrow element from absolute polyline points.
function arrowFromPoints(absPts, fromId, toId) {
  const xs = absPts.map((p) => p[0])
  const ys = absPts.map((p) => p[1])
  const ox = Math.min(...xs)
  const oy = Math.min(...ys)
  return {
    id: rngId('ar'),
    type: 'arrow',
    x: ox,
    y: oy,
    width: Math.max(...xs) - ox,
    height: Math.max(...ys) - oy,
    strokeWidth: 2,
    endArrowhead: 'arrow',
    points: absPts.map(([x, y]) => [x - ox, y - oy]),
    startBinding: fromId ? { elementId: fromId, focus: 0, gap: 8 } : null,
    endBinding: toId ? { elementId: toId, focus: 0, gap: 8 } : null,
  }
}

// ── Engine: pipeline ────────────────────────────────────────────────────────
// A long pipeline (≥ 6 stages) wraps into a serpentine of two rows so the
// cards stay legibly large instead of becoming thin pillars in one ultra-wide
// row. Row 0 runs left→right, row 1 right→left, joined by a vertical connector.
function layoutPipeline(dsl) {
  const nodes = dsl.nodes || []
  const edges = dsl.edges || []
  if (nodes.length < 2) die('pipeline needs ≥ 2 nodes')
  assertOneAccent(nodes)

  const top = bodyTopY()
  const bodyH = CANVAS_H - top - BODY_BOTTOM_MARGIN
  const n = nodes.length

  const rowsCount = n <= 5 ? 1 : 2
  const perRow = Math.ceil(n / rowsCount)
  const labels = nodes.map((nd) => wrapLabel(nd.label))

  const PIPE_GAP = 150
  const V_GAP = 150
  const nodeW = Math.max(...labels.map((l) => containerWidthFor(l)))
  const stride = nodeW + PIPE_GAP
  const rowW = stride * perRow - PIPE_GAP
  const targetRowW = Math.max(rowW, Math.round(CANVAS_W * 0.88))
  const scale = targetRowW / rowW
  const W = Math.round(nodeW * scale)
  const S = Math.round(stride * scale)
  // Card height fills the row's vertical share of the body band, capped so a
  // card never gets taller than 1.5× its width (which would read as a pillar).
  let H = Math.round(((bodyH - V_GAP * (rowsCount - 1)) / rowsCount) * 0.9)
  H = Math.min(H, Math.round(W * 1.5))

  const maxLine = Math.max(
    ...labels.map((l) => Math.max(...l.split('\n').map((x) => x.length))),
  )
  const labelFont = Math.max(24, Math.min(48, Math.floor((W - 56) / (maxLine * 0.6))))
  const pipeEdgeFont = 30

  const rowStartX = Math.round((CANVAS_W - (S * perRow - (S - W))) / 2)
  const blockH = rowsCount * H + V_GAP * (rowsCount - 1)
  const blockTop = top + Math.round((bodyH - blockH) / 2)

  // Physical placement for logical node index i (serpentine).
  function posOf(i) {
    const row = Math.floor(i / perRow)
    const inRow = i % perRow
    const col = row % 2 === 0 ? inRow : perRow - 1 - inRow
    return {
      x: rowStartX + col * S,
      y: blockTop + row * (H + V_GAP),
      row,
      col,
    }
  }

  const els = headerElements(dsl.title, dsl.caption)
  const positions = new Map()
  const ids = []
  for (let i = 0; i < n; i++) {
    const nd = nodes[i]
    const id = nd.id || `n${i}`
    ids.push(id)
    const p = posOf(i)
    positions.set(id, { x: p.x, y: p.y, w: W, h: H })
    els.push({
      id,
      type: 'rectangle',
      x: p.x,
      y: p.y,
      width: W,
      height: H,
      backgroundColor: paletteFor(nd.kind),
      strokeWidth: 2,
      label: { text: labels[i], fontSize: labelFont, fontFamily: 1 },
    })
  }

  // Arrows between consecutive stages — horizontal within a row, a vertical
  // connector at the serpentine turn.
  const edgeLabel = new Map()
  for (const e of edges) edgeLabel.set(`${e.from}>${e.to}`, e.label || '')
  for (let i = 0; i < n - 1; i++) {
    const a = positions.get(ids[i])
    const b = positions.get(ids[i + 1])
    const pa = posOf(i)
    const pb = posOf(i + 1)
    const aMidY = a.y + Math.round(H / 2)
    const lbl = edgeLabel.get(`${nodes[i].id}>${nodes[i + 1].id}`)

    if (pa.row === pb.row) {
      const leftToRight = pb.col > pa.col
      const sx = leftToRight ? a.x + W : a.x
      const ex = leftToRight ? b.x : b.x + W
      els.push(arrowFromPoints([[sx, aMidY], [ex, aMidY]], ids[i], ids[i + 1]))
      if (lbl) {
        const tw = estTextWidth(lbl, pipeEdgeFont)
        const gap = Math.abs(ex - sx)
        if (tw <= gap - 16) {
          const lh = estTextHeight(lbl, pipeEdgeFont)
          els.push({
            id: rngId('lb'),
            type: 'text',
            x: Math.round((sx + ex) / 2 - tw / 2),
            // Sit the whole label bbox 14 px above the arrow line — using the
            // real text height (not just one font-size) keeps multi-line edge
            // labels off the stroke too.
            y: Math.round(aMidY - lh - 14),
            width: tw,
            height: lh,
            text: lbl,
            fontSize: pipeEdgeFont,
            fontFamily: 1,
            textAlign: 'center',
          })
        }
      }
    } else {
      // Serpentine turn: vertical connector, same physical column.
      const cx = a.x + Math.round(W / 2)
      els.push(arrowFromPoints([[cx, a.y + H], [cx, b.y]], ids[i], ids[i + 1]))
    }
  }
  return els
}

// ── Engine: stack (nested rectangles, outermost first) ─────────────────────
function layoutStack(dsl) {
  const nodes = dsl.nodes || []
  if (nodes.length < 2) die('stack needs ≥ 2 nodes')
  assertOneAccent(nodes)

  const els = headerElements(dsl.title, dsl.caption)
  const top = bodyTopY()
  const bottomMargin = 80
  const totalH = CANVAS_H - top - bottomMargin
  const layerH = Math.floor(totalH / nodes.length)
  const baseW = Math.round(CANVAS_W * 0.85)
  const startX = Math.round((CANVAS_W - baseW) / 2)
  const stepX = 40 // each inner layer indents 40 px on both sides

  for (let i = 0; i < nodes.length; i++) {
    const n = nodes[i]
    // For a true nested-rectangles stack, set i=outermost; here we treat the
    // DSL order as outer→inner, with the innermost being the focused layer.
    const w = baseW - 2 * i * stepX
    const x = startX + i * stepX
    const y = top + i * layerH
    els.push({
      id: n.id || `lyr${i}`,
      type: 'rectangle',
      x, y, width: w, height: layerH - 24,
      backgroundColor: paletteFor(n.kind),
      strokeWidth: 2,
      label: fitLabel(n.label, w, layerH - 24),
    })
  }
  return els
}

// ── Engine: before/after (two columns) ─────────────────────────────────────
function layoutBeforeAfter(dsl) {
  const before = dsl.before || { nodes: [] }
  const after = dsl.after || { nodes: [] }
  if (before.nodes.length === 0 || after.nodes.length === 0) {
    die('before-after requires non-empty before.nodes and after.nodes')
  }
  if (before.nodes.length !== after.nodes.length) {
    die(`before-after rows must align: before has ${before.nodes.length} nodes, after has ${after.nodes.length}`)
  }

  // Force the palette: left column uses caution/danger, right column uses success/primary.
  const fixKind = (n, side) => {
    if (n.kind) return n.kind
    return side === 'before' ? 'caution' : 'success'
  }
  const beforeNodes = before.nodes.map((n) => ({ ...n, kind: fixKind(n, 'before') }))
  const afterNodes = after.nodes.map((n) => ({ ...n, kind: fixKind(n, 'after') }))

  const els = headerElements(dsl.title, dsl.caption)
  const top = bodyTopY()
  const colW = Math.round(CANVAS_W * 0.40)
  const gap = Math.round(CANVAS_W * 0.05)
  const leftX = Math.round((CANVAS_W - (2 * colW + gap)) / 2)
  const rightX = leftX + colW + gap
  const rows = beforeNodes.length
  const rowH = Math.floor((CANVAS_H - top - 100) / rows)
  const nodeH = rowH - 30

  // Column header labels
  const leftHdr = before.label || 'Before'
  const rightHdr = after.label || 'After'
  els.push({
    id: 'leftHdr',
    type: 'text',
    x: leftX + Math.round((colW - estTextWidth(leftHdr, BODY_FONT + 4)) / 2),
    y: top - 10,
    width: estTextWidth(leftHdr, BODY_FONT + 4),
    height: estTextHeight(leftHdr, BODY_FONT + 4),
    text: leftHdr,
    fontSize: BODY_FONT + 4,
    fontFamily: 1,
    textAlign: 'center',
  })
  els.push({
    id: 'rightHdr',
    type: 'text',
    x: rightX + Math.round((colW - estTextWidth(rightHdr, BODY_FONT + 4)) / 2),
    y: top - 10,
    width: estTextWidth(rightHdr, BODY_FONT + 4),
    height: estTextHeight(rightHdr, BODY_FONT + 4),
    text: rightHdr,
    fontSize: BODY_FONT + 4,
    fontFamily: 1,
    textAlign: 'center',
  })

  for (let i = 0; i < rows; i++) {
    const yi = top + 40 + i * rowH
    const bn = beforeNodes[i]
    const an = afterNodes[i]
    els.push({
      id: bn.id || `b${i}`, type: 'rectangle',
      x: leftX, y: yi, width: colW, height: nodeH,
      backgroundColor: paletteFor(bn.kind),
      strokeWidth: 2,
      label: fitLabel(bn.label, colW, nodeH),
    })
    els.push({
      id: an.id || `a${i}`, type: 'rectangle',
      x: rightX, y: yi, width: colW, height: nodeH,
      backgroundColor: paletteFor(an.kind),
      strokeWidth: 2,
      label: fitLabel(an.label, colW, nodeH),
    })
    // Arrow between rows
    {
      const aw = gap - 20
      els.push({
        id: rngId('ar'),
        type: 'arrow',
        x: leftX + colW + 10,
        y: yi + Math.round(nodeH / 2),
        width: aw,
        height: 0,
        strokeWidth: 2,
        endArrowhead: 'arrow',
        points: [[0, 0], [aw, 0]],
      })
    }
  }
  return els
}

// ── Engine: matrix ─────────────────────────────────────────────────────────
function layoutMatrix(dsl) {
  const rows = dsl.rows || []
  const cols = dsl.cols || []
  const cells = dsl.cells || []
  if (rows.length === 0 || cols.length === 0) die('matrix requires rows and cols')
  if (cells.length !== rows.length) die('matrix cells.length must equal rows.length')

  const els = headerElements(dsl.title, dsl.caption)
  const top = bodyTopY()
  const cellW = Math.floor((CANVAS_W * 0.85) / (cols.length + 1))
  const cellH = Math.floor((CANVAS_H - top - 100) / (rows.length + 1))
  const startX = Math.round((CANVAS_W - cellW * (cols.length + 1)) / 2)

  // Column headers
  for (let j = 0; j < cols.length; j++) {
    const text = cols[j]
    els.push({
      id: `col${j}`,
      type: 'text',
      x: startX + (j + 1) * cellW + Math.round((cellW - estTextWidth(text, BODY_FONT, 3)) / 2),
      y: top + Math.round((cellH - estTextHeight(text, BODY_FONT)) / 2),
      width: estTextWidth(text, BODY_FONT, 3),
      height: estTextHeight(text, BODY_FONT),
      text,
      fontSize: BODY_FONT,
      fontFamily: 3,
      textAlign: 'center',
    })
  }
  // Row headers + cells
  for (let i = 0; i < rows.length; i++) {
    const rowText = rows[i]
    els.push({
      id: `row${i}`,
      type: 'text',
      x: startX + Math.round((cellW - estTextWidth(rowText, BODY_FONT, 3)) / 2),
      y: top + (i + 1) * cellH + Math.round((cellH - estTextHeight(rowText, BODY_FONT)) / 2),
      width: estTextWidth(rowText, BODY_FONT, 3),
      height: estTextHeight(rowText, BODY_FONT),
      text: rowText,
      fontSize: BODY_FONT,
      fontFamily: 3,
      textAlign: 'center',
    })
    const row = cells[i] || []
    for (let j = 0; j < cols.length; j++) {
      const c = row[j] || {}
      els.push({
        id: `c${i}-${j}`,
        type: 'rectangle',
        x: startX + (j + 1) * cellW,
        y: top + (i + 1) * cellH,
        width: cellW - 6,
        height: cellH - 6,
        backgroundColor: paletteFor(c.kind),
        strokeWidth: 2,
        label: fitLabel(c.label || '', cellW - 6, cellH - 6),
      })
    }
  }
  return els
}

// ── Engine: graph (element-form layered DAG) ───────────────────────────────
//
// Replaces the old Mermaid emission. Mermaid's auto-layout produced nodes
// whose bounding boxes did not include actual text width, causing right-side
// clipping in the headless render. The new engine sizes containers from
// labels, lays nodes out into longest-path layers L→R, and routes orthogonal
// arrows through the gaps between layers. Title and caption are now rendered
// at the top like every other engine.
function layoutGraph(dsl) {
  const nodes = dsl.nodes || []
  const edges = dsl.edges || []
  if (nodes.length < 2) die('graph needs ≥ 2 nodes')
  assertOneAccent(nodes)

  // 1. Per-node sizes from labels.
  const sizes = new Map()
  for (const n of nodes) {
    sizes.set(n.id, {
      w: containerWidthFor(n.label),
      h: containerHeightFor(n.label),
    })
  }

  // 2. Build adjacency, then assign layers via longest-path from sources.
  //    Cycles are broken by a recursion-stack guard (back-edge → layer 0).
  const incoming = new Map(nodes.map((n) => [n.id, []]))
  const outgoing = new Map(nodes.map((n) => [n.id, []]))
  for (const e of edges) {
    if (incoming.has(e.to)) incoming.get(e.to).push(e.from)
    if (outgoing.has(e.from)) outgoing.get(e.from).push(e.to)
  }
  const layerOf = new Map()
  function computeLayer(id, stack) {
    if (layerOf.has(id)) return layerOf.get(id)
    if (stack.has(id)) return 0
    stack.add(id)
    let L = 0
    for (const p of incoming.get(id) || []) {
      L = Math.max(L, computeLayer(p, stack) + 1)
    }
    stack.delete(id)
    layerOf.set(id, L)
    return L
  }
  for (const n of nodes) computeLayer(n.id, new Set())

  // 3. Group nodes into layers (preserve DSL order for determinism).
  const numLayers = Math.max(...layerOf.values()) + 1
  const layers = Array.from({ length: numLayers }, () => [])
  for (const n of nodes) layers[layerOf.get(n.id)].push(n)

  // 3b. Barycenter ordering — reduce edge crossings so the DAG reads as a
  //     scientific layered graph rather than a tangle. Each node is pulled
  //     toward the mean row-index of its neighbours in the adjacent layer;
  //     three alternating sweeps (down, up, down) settle small DAGs without a
  //     full crossing-minimisation solver. Neighbour-less nodes keep their
  //     DSL order (Infinity barycenter sorts stably to the bottom).
  function orderSweep(dir) {
    const lo = dir > 0 ? 1 : numLayers - 2
    const cond = (li) => (dir > 0 ? li < numLayers : li >= 0)
    for (let li = lo; cond(li); li += dir) {
      const adj = layers[li - dir]
      const pos = new Map(adj.map((n, i) => [n.id, i]))
      const bary = new Map()
      for (const n of layers[li]) {
        const neigh = dir > 0 ? incoming.get(n.id) : outgoing.get(n.id)
        const idxs = (neigh || []).map((m) => pos.get(m)).filter((v) => v != null)
        bary.set(n.id, idxs.length ? idxs.reduce((a, b) => a + b, 0) / idxs.length : Infinity)
      }
      layers[li] = layers[li]
        .map((n, i) => [n, i])
        .sort(([a, ia], [b, ib]) => {
          const x = bary.get(a.id), y = bary.get(b.id)
          if (x === y) return ia - ib // stable: preserve prior order on ties
          return x - y
        })
        .map(([n]) => n)
    }
  }
  if (numLayers > 1) { orderSweep(1); orderSweep(-1); orderSweep(1) }

  // 4. Per-layer UNIFORM node dimensions. Every node in a layer gets that
  //    layer's max width and max height, so node edges line up on a clean
  //    grid and every arrow enters/leaves at a predictable coordinate.
  const top = bodyTopY()
  const bodyH = CANVAS_H - top - BODY_BOTTOM_MARGIN
  const layerNodeW = layers.map((L) => Math.max(...L.map((n) => sizes.get(n.id).w)))
  const layerNodeH = layers.map((L) => Math.max(...L.map((n) => sizes.get(n.id).h)))
  const layerW = layerNodeW

  // A near-linear graph (one node per layer everywhere) is really a pipeline —
  // routed as a graph it degrades into a single ultra-wide row that can never
  // fill the canvas vertically. Send the author to the right figure type.
  const maxPerLayer = Math.max(...layers.map((L) => L.length))
  if (maxPerLayer < 2) {
    die('graph has at most one node per layer — this is a linear flow; author it as type "pipeline" (which serpentines and fills the canvas) instead of "graph"')
  }

  // Vertical gap between sibling nodes in a layer. Spread the busiest layer's
  // siblings to fill ~82% of the body band so a sparse, wide DAG (few branches,
  // many layers) reads as a proper scientific layered graph rather than a thin
  // ribbon floating in dead space. Floor of 70 px keeps dense layers legible.
  const nodeHRep = Math.max(...layerNodeH)
  const desiredBlockH = bodyH * 0.82
  const V_GAP = Math.max(
    70,
    Math.round((desiredBlockH - maxPerLayer * nodeHRep) / (maxPerLayer - 1)),
  )
  const layerH = layers.map(
    (L, i) => L.length * layerNodeH[i] + V_GAP * Math.max(0, L.length - 1),
  )

  // 5. Horizontal stride: pack layers across the canvas. Shrink the gap if
  //    we'd otherwise overflow; expand it if we'd undercover the canvas.
  const bodyMaxW = CANVAS_W - 160
  const bodyMinW = Math.round(CANVAS_W * 0.74)
  const sumLayerW = layerW.reduce((s, w) => s + w, 0)
  let H_GAP = 140
  let totalW = sumLayerW + H_GAP * (numLayers - 1)
  while (totalW > bodyMaxW && H_GAP > 60) {
    H_GAP -= 10
    totalW = sumLayerW + H_GAP * (numLayers - 1)
  }
  while (totalW < bodyMinW && H_GAP < 320 && numLayers > 1) {
    H_GAP += 20
    totalW = sumLayerW + H_GAP * (numLayers - 1)
  }
  // If still wider than canvas, accept overflow but warn — author shrink labels.
  const startX = Math.max(80, Math.round((CANVAS_W - totalW) / 2))
  const layerX = []
  let cx = startX
  for (let i = 0; i < numLayers; i++) {
    layerX.push(cx)
    cx += layerW[i] + H_GAP
  }

  // 6. Place nodes: each layer is a column of equal-size boxes, the column
  //    vertically centered around the body midline.
  const bodyMid = top + Math.round(bodyH / 2)
  const els = headerElements(dsl.title, dsl.caption)
  const positions = new Map()
  for (let li = 0; li < numLayers; li++) {
    const L = layers[li]
    const nW = layerNodeW[li]
    const nH = layerNodeH[li]
    let ny = bodyMid - Math.round(layerH[li] / 2)
    for (const n of L) {
      const nx = layerX[li]
      positions.set(n.id, { x: nx, y: ny, w: nW, h: nH })
      els.push({
        id: n.id,
        type: 'rectangle',
        x: nx,
        y: ny,
        width: nW,
        height: nH,
        backgroundColor: paletteFor(n.kind),
        strokeWidth: 2,
        label: n.label,
      })
      ny += nH + V_GAP
    }
  }

  // 7. Arrows — clean orthogonal routing on the node grid.
  //    - Adjacent-layer forward edges jog at a SHARED channel x (the gap
  //      midpoint between the two layers), so fan-out/fan-in reads as one
  //      tidy routing bus instead of a tangle of crossing diagonals.
  //    - Multi-layer-jump forward edges and back-edges detour through a
  //      channel BELOW the node block, at two distinct depths so the two
  //      kinds never share a lane.
  //    - Same-layer edges loop on the right side.
  const ARROW_GAP = 10

  let blockBottom = -Infinity
  for (const p of positions.values()) blockBottom = Math.max(blockBottom, p.y + p.h)
  const backChannelY = blockBottom + 60
  const jumpChannelY = blockBottom + 150

  function pushArrow(absPts, fromId, toId, dashed) {
    const xs = absPts.map((p) => p[0])
    const ys = absPts.map((p) => p[1])
    const ox = Math.min(...xs)
    const oy = Math.min(...ys)
    els.push({
      id: rngId('ar'),
      type: 'arrow',
      x: ox,
      y: oy,
      width: Math.max(...xs) - ox,
      height: Math.max(...ys) - oy,
      strokeWidth: 2,
      strokeStyle: dashed ? 'dashed' : 'solid',
      endArrowhead: 'arrow',
      points: absPts.map(([x, y]) => [x - ox, y - oy]),
      startBinding: { elementId: fromId, focus: 0, gap: ARROW_GAP },
      endBinding: { elementId: toId, focus: 0, gap: ARROW_GAP },
    })
  }

  for (const e of edges) {
    const a = positions.get(e.from)
    const b = positions.get(e.to)
    if (!a || !b) continue
    const la = layerOf.get(e.from)
    const lb = layerOf.get(e.to)
    const aMidY = a.y + Math.round(a.h / 2)
    const bMidY = b.y + Math.round(b.h / 2)
    const aMidX = a.x + Math.round(a.w / 2)
    const bMidX = b.x + Math.round(b.w / 2)

    if (lb === la + 1) {
      // Adjacent forward: horizontal out, vertical at channel, horizontal in.
      const sx = a.x + a.w + ARROW_GAP
      const ex = b.x - ARROW_GAP
      const cx = layerX[la] + layerW[la] + Math.round(H_GAP / 2)
      if (Math.abs(bMidY - aMidY) < 6) {
        pushArrow([[sx, aMidY], [ex, aMidY]], e.from, e.to)
      } else {
        pushArrow(
          [[sx, aMidY], [cx, aMidY], [cx, bMidY], [ex, bMidY]],
          e.from, e.to,
        )
      }
    } else if (lb > la + 1) {
      // Multi-layer forward jump: detour below the block (dashed = skips
      // intermediate layers, which the prose should explain).
      pushArrow(
        [
          [aMidX, a.y + a.h + ARROW_GAP],
          [aMidX, jumpChannelY],
          [bMidX, jumpChannelY],
          [bMidX, b.y + b.h + ARROW_GAP],
        ],
        e.from, e.to, true,
      )
    } else if (lb === la) {
      // Same layer: loop out the right side and back in.
      const sx = Math.max(a.x + a.w, b.x + b.w) + ARROW_GAP + 60
      pushArrow(
        [
          [a.x + a.w + ARROW_GAP, aMidY],
          [sx, aMidY],
          [sx, bMidY],
          [b.x + b.w + ARROW_GAP, bMidY],
        ],
        e.from, e.to,
      )
    } else {
      // Back-edge: detour below the block.
      pushArrow(
        [
          [aMidX, a.y + a.h + ARROW_GAP],
          [aMidX, backChannelY],
          [bMidX, backChannelY],
          [bMidX, b.y + b.h + ARROW_GAP],
        ],
        e.from, e.to,
      )
    }

    // 8. Edge label — adjacent-layer forward edges only. The label sits ABOVE
    //    the source-side horizontal run of the arrow (from the source node's
    //    right edge to the routing channel at cx), never on the vertical
    //    channel itself. That keeps it provably clear of the arrow stroke: it
    //    is 14 px above its own horizontal run and stays left of cx, so it
    //    cannot touch the vertical jog. Distinct edges leave their sources at
    //    distinct y-bands (≥ one node height + V_GAP apart), so labels never
    //    collide with each other either — no dedup/stacking needed.
    if (e.label && lb === la + 1) {
      const labelText = String(e.label)
      const tw = estTextWidth(labelText, EDGE_FONT)
      const th = estTextHeight(labelText, EDGE_FONT)
      const sx = a.x + a.w + ARROW_GAP
      const cx = layerX[la] + layerW[la] + Math.round(H_GAP / 2)
      const runW = cx - sx
      if (tw <= runW - 16 && runW > 48) {
        els.push({
          id: rngId('lb'),
          type: 'text',
          x: Math.round((sx + cx) / 2 - tw / 2),
          y: Math.round(aMidY - th - 14),
          width: tw,
          height: th,
          text: labelText,
          fontSize: EDGE_FONT,
          fontFamily: 1,
          textAlign: 'center',
        })
      }
    }
  }

  // 9. Scale-to-fit: the natural layout is sized by label text, which leaves
  //    a short graph floating in a thin band with dead space above. Scale
  //    every body element (nodes, arrows, edge labels) so the figure fills
  //    the body region. X and Y scale independently — a wide-but-short DAG
  //    needs more vertical stretch than horizontal — but the aspect-ratio
  //    distortion is capped at 1.7× so boxes never look cartoonish. Header
  //    stays fixed; the body is re-centered in the body band.
  const HEADER_COUNT = 2 // title + caption
  const body = els.slice(HEADER_COUNT)
  if (body.length > 0) {
    // Scale + center on the NODE rectangles only. Arrows (especially
    // back-edge detours) extend past the node area; including them would
    // pull the node block off-center and reintroduce a dead band.
    const rects = body.filter((e) => e.type === 'rectangle')
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity
    for (const e of rects) {
      x0 = Math.min(x0, e.x)
      y0 = Math.min(y0, e.y)
      x1 = Math.max(x1, e.x + (e.width || 0))
      y1 = Math.max(y1, e.y + (e.height || 0))
    }
    const natW = x1 - x0
    const natH = y1 - y0
    const targetW = CANVAS_W * 0.92
    const targetH = bodyH * 0.92
    // Never scale below 1.0 (fonts would drop under the 22-px floor).
    let sx = Math.max(1.0, Math.min(targetW / natW, 2.4))
    let sy = Math.max(1.0, Math.min(targetH / natH, 3.0))
    // Cap aspect distortion: neither axis may exceed the other by > 1.6×.
    const DISTORT = 1.6
    sx = Math.min(sx, sy * DISTORT)
    sy = Math.min(sy, sx * DISTORT)
    // Font scales with the SMALLER axis factor. Container width grows by sx
    // and height by sy; scaling the font by min(sx, sy) guarantees the text
    // still fits its (independently scaled) container in both dimensions.
    const fontScale = Math.min(sx, sy)
    if (sx !== 1.0 || sy !== 1.0) {
      const newW = natW * sx
      const offX = Math.round((CANVAS_W - newW) / 2 - x0 * sx)
      // Top-align the node block just below the caption. Centering a short
      // graph in the full body band leaves a dead band above it; the export
      // crops to the content bbox, so a top-aligned wide-short graph simply
      // renders as a wide-short PNG with no wasted vertical space. The
      // headroom band scales with sy so edge labels (which sit ~40 px above
      // their target node, pre-scale) never collide with the caption.
      const LABEL_HEADROOM = Math.ceil(48 * sy)
      const offY = Math.round(top + LABEL_HEADROOM - y0 * sy)
      for (const e of body) {
        e.x = Math.round(e.x * sx + offX)
        e.y = Math.round(e.y * sy + offY)
        if (e.width != null) e.width = Math.round(e.width * sx)
        if (e.height != null) e.height = Math.round(e.height * sy)
        if (e.points) e.points = e.points.map(([px, py]) => [Math.round(px * sx), Math.round(py * sy)])
        if (e.type === 'text') e.fontSize = Math.max(BODY_FONT, Math.round(e.fontSize * fontScale))
        if (e.type === 'rectangle' && e.label != null) {
          const t = typeof e.label === 'string' ? e.label : e.label.text
          e.label = { text: t, fontSize: Math.round(BODY_FONT * fontScale), fontFamily: 1 }
        }
      }
      // Rebuild positions from the scaled rectangles so overlay callouts,
      // if any, still target the right coordinates.
      for (const e of body) {
        if (e.type === 'rectangle' && positions.has(e.id)) {
          positions.set(e.id, { x: e.x, y: e.y, w: e.width, h: e.height })
        }
      }
    }
  }

  return { elements: els, positions }
}

// ── Engine: tree (hierarchical, root at top) ───────────────────────────────
// nodes: [{id, label, kind, parent}], parent omitted/empty for root.
// Lays out by depth: y = top + depth × levelH; x within parent's slice.
function layoutTree(dsl) {
  const nodes = dsl.nodes || []
  if (nodes.length < 2) die('tree needs ≥ 2 nodes')
  assertOneAccent(nodes)

  // Build parent → children map and depth per node.
  const byId = new Map(nodes.map((n) => [n.id, n]))
  const childrenOf = new Map()
  let root = null
  for (const n of nodes) {
    if (!n.parent) {
      if (root) die(`tree has multiple roots: ${root.id} and ${n.id}; pick one`)
      root = n
    } else {
      if (!byId.has(n.parent)) die(`tree node ${n.id} has unknown parent ${n.parent}`)
      const arr = childrenOf.get(n.parent) || []
      arr.push(n)
      childrenOf.set(n.parent, arr)
    }
  }
  if (!root) die('tree needs exactly one root (a node without a parent field)')

  // Recursively assign each subtree a width = max(1, sum of children widths).
  const subtreeW = new Map()
  function computeW(id) {
    const kids = childrenOf.get(id) || []
    if (kids.length === 0) { subtreeW.set(id, 1); return 1 }
    const w = kids.reduce((a, k) => a + computeW(k.id), 0)
    subtreeW.set(id, w)
    return w
  }
  computeW(root.id)
  const totalLeaves = subtreeW.get(root.id)

  // Depth assignment via BFS.
  const depth = new Map([[root.id, 0]])
  let maxDepth = 0
  const queue = [root]
  while (queue.length) {
    const n = queue.shift()
    const d = depth.get(n.id)
    maxDepth = Math.max(maxDepth, d)
    for (const k of childrenOf.get(n.id) || []) {
      depth.set(k.id, d + 1)
      queue.push(k)
    }
  }

  const top = bodyTopY()
  const bodyH = CANVAS_H - top - BODY_BOTTOM_MARGIN
  const levelH = Math.floor(bodyH / (maxDepth + 1))
  const bodyW = Math.round(CANVAS_W * 0.92)
  const startX = Math.round((CANVAS_W - bodyW) / 2)
  const nodeH = Math.min(140, Math.floor(levelH * 0.55))

  // Place each node at the center of its allocated horizontal slice.
  const positions = new Map()
  function place(id, leftLeaf) {
    const n = byId.get(id)
    const w = subtreeW.get(id)
    const slice = (w / totalLeaves) * bodyW
    const x = startX + (leftLeaf / totalLeaves) * bodyW + slice / 2
    const y = top + depth.get(id) * levelH + Math.round((levelH - nodeH) / 2)
    const nodeW = Math.max(containerWidthFor(n.label), Math.round(slice * 0.85))
    positions.set(id, { x: Math.round(x - nodeW / 2), y, w: nodeW, h: nodeH, node: n })
    let cursor = leftLeaf
    for (const k of childrenOf.get(id) || []) {
      place(k.id, cursor)
      cursor += subtreeW.get(k.id)
    }
  }
  place(root.id, 0)

  const els = headerElements(dsl.title, dsl.caption)
  // Edges first so they sit under nodes visually. Fan out arrows from each
  // parent so siblings don't share a start point (which the overlap validator
  // would flag as bbox-collision and which reads as a single thick line on
  // the rendered PNG).
  for (const n of nodes) {
    const kids = childrenOf.get(n.id) || []
    if (kids.length === 0) continue
    const p = positions.get(n.id)
    const py = p.y + p.h
    // Distribute the start points along the bottom edge of the parent.
    const span = p.w * 0.7 // use 70% of parent's width for the fan
    const startStride = kids.length === 1 ? 0 : span / (kids.length - 1)
    const startLeft = p.x + p.w / 2 - span / 2
    for (let i = 0; i < kids.length; i++) {
      const k = kids[i]
      const c = positions.get(k.id)
      const px = kids.length === 1 ? p.x + p.w / 2 : startLeft + i * startStride
      const cx = c.x + Math.round(c.w / 2)
      const cy = c.y
      els.push({
        id: rngId('ar'),
        type: 'arrow',
        x: px,
        y: py,
        width: cx - px,
        height: cy - py,
        strokeWidth: 2,
        endArrowhead: 'arrow',
        points: [[0, 0], [cx - px, cy - py]],
      })
    }
  }
  for (const [id, p] of positions) {
    els.push({
      id,
      type: 'rectangle',
      x: p.x, y: p.y, width: p.w, height: p.h,
      backgroundColor: paletteFor(p.node.kind),
      strokeWidth: 2,
      label: fitLabel(p.node.label, p.w, p.h),
    })
  }
  return { elements: els, positions }
}

// ── Engine: timeline (horizontal events, alternating labels) ───────────────
// events: [{id, label, kind, date?}] — order is left-to-right as given.
function layoutTimeline(dsl) {
  const events = dsl.events || []
  if (events.length < 2) die('timeline needs ≥ 2 events')
  assertOneAccent(events)

  const top = bodyTopY()
  const bodyH = CANVAS_H - top - BODY_BOTTOM_MARGIN
  const bodyW = Math.round(CANVAS_W * 0.92)
  const startX = Math.round((CANVAS_W - bodyW) / 2)
  const axisY = top + Math.round(bodyH / 2)

  const els = headerElements(dsl.title, dsl.caption)
  // Axis line.
  els.push({
    id: 'axis',
    type: 'line',
    x: startX,
    y: axisY,
    width: bodyW,
    height: 0,
    strokeWidth: 3,
    points: [[0, 0], [bodyW, 0]],
  })

  const stride = bodyW / (events.length - 1)
  const positions = new Map()
  for (let i = 0; i < events.length; i++) {
    const e = events[i]
    const cx = Math.round(startX + i * stride)
    const cy = axisY
    const above = i % 2 === 0 // alternate above/below
    const cardW = containerWidthFor(e.label)
    const cardH = containerHeightFor(e.label) + (e.date ? 36 : 0)
    const cardX = Math.round(cx - cardW / 2)
    const cardY = above
      ? axisY - 80 - cardH
      : axisY + 80
    positions.set(e.id || `t${i}`, { x: cardX, y: cardY, w: cardW, h: cardH })

    // Marker dot on the axis.
    els.push({
      id: rngId('dot'),
      type: 'ellipse',
      x: cx - 14, y: axisY - 14, width: 28, height: 28,
      backgroundColor: paletteFor(e.kind ?? 'primary'),
      strokeWidth: 2,
    })
    // Connector line from card to marker.
    if (above) {
      els.push({
        id: rngId('cn'), type: 'line',
        x: cx, y: cardY + cardH, width: 0, height: axisY - 14 - (cardY + cardH),
        strokeWidth: 1.5,
        points: [[0, 0], [0, axisY - 14 - (cardY + cardH)]],
      })
    } else {
      els.push({
        id: rngId('cn'), type: 'line',
        x: cx, y: axisY + 14, width: 0, height: cardY - (axisY + 14),
        strokeWidth: 1.5,
        points: [[0, 0], [0, cardY - (axisY + 14)]],
      })
    }
    // Card.
    els.push({
      id: e.id || `t${i}`,
      type: 'rectangle',
      x: cardX, y: cardY, width: cardW, height: cardH,
      backgroundColor: paletteFor(e.kind),
      strokeWidth: 2,
      label: fitLabel(e.date ? `${e.date} ${e.label}` : e.label, cardW, cardH),
    })
  }
  return { elements: els, positions }
}

// ── Engine: grid (free placement on a logical N × M grid) ──────────────────
// Each node has {id, label, kind, row, col, rowspan?, colspan?}.
// gridRows / gridCols default to max+1 of node positions.
// edges: [{from, to, label}] — drawn between centers of source/target cells.
function layoutGrid(dsl) {
  const nodes = dsl.nodes || []
  if (nodes.length < 2) die('grid needs ≥ 2 nodes')
  for (const n of nodes) {
    if (typeof n.row !== 'number' || typeof n.col !== 'number') {
      die(`grid node ${n.id || '<anon>'} missing row/col (numbers required)`)
    }
  }
  assertOneAccent(nodes)

  const gridRows = Math.max(dsl.gridRows ?? 0, ...nodes.map((n) => n.row + (n.rowspan || 1)))
  const gridCols = Math.max(dsl.gridCols ?? 0, ...nodes.map((n) => n.col + (n.colspan || 1)))
  const top = bodyTopY()
  const bodyH = CANVAS_H - top - BODY_BOTTOM_MARGIN
  const bodyW = Math.round(CANVAS_W * 0.92)
  const startX = Math.round((CANVAS_W - bodyW) / 2)
  const cellW = Math.floor(bodyW / gridCols)
  const cellH = Math.floor(bodyH / gridRows)
  const cellPad = 16

  const els = headerElements(dsl.title, dsl.caption)
  const positions = new Map()
  for (const n of nodes) {
    const colspan = n.colspan || 1
    const rowspan = n.rowspan || 1
    const x = startX + n.col * cellW + cellPad
    const y = top + n.row * cellH + cellPad
    const w = colspan * cellW - 2 * cellPad
    const h = rowspan * cellH - 2 * cellPad
    positions.set(n.id, { x, y, w, h, node: n })
    els.push({
      id: n.id,
      type: 'rectangle',
      x, y, width: w, height: h,
      backgroundColor: paletteFor(n.kind),
      strokeWidth: 2,
      label: fitLabel(n.label, w, h),
    })
  }

  // Edges: route as orthogonal two-segment polylines from source center-edge
  // to target center-edge to avoid arrow-through-node collisions.
  for (const e of dsl.edges || []) {
    const a = positions.get(e.from)
    const b = positions.get(e.to)
    if (!a || !b) die(`grid edge references unknown node: ${e.from} → ${e.to}`)
    const ax = a.x + a.w / 2
    const ay = a.y + a.h / 2
    const bx = b.x + b.w / 2
    const by = b.y + b.h / 2
    const dx = bx - ax
    const dy = by - ay
    // Pick start point on the edge of A facing B; same for end on B's facing edge.
    let sx, sy, tx, ty
    if (Math.abs(dx) >= Math.abs(dy)) {
      sx = ax + Math.sign(dx) * a.w / 2
      sy = ay
      tx = bx - Math.sign(dx) * b.w / 2
      ty = by
    } else {
      sx = ax
      sy = ay + Math.sign(dy) * a.h / 2
      tx = bx
      ty = by - Math.sign(dy) * b.h / 2
    }
    els.push({
      id: rngId('ar'),
      type: 'arrow',
      x: sx, y: sy,
      width: tx - sx, height: ty - sy,
      strokeWidth: 2,
      endArrowhead: 'arrow',
      points: [[0, 0], [tx - sx, ty - sy]],
    })
    if (e.label) {
      const tw = estTextWidth(e.label, EDGE_FONT)
      const th = estTextHeight(e.label, EDGE_FONT)
      // Offset the label PERPENDICULAR to the arrow segment so it never lands
      // on the stroke, whatever the angle. The normal is forced to the upper
      // side (negative y) for horizontal/diagonal runs; for a purely vertical
      // run the normal is horizontal and the label sits to one side. The
      // clearance is half the label height plus 16 px so the bbox stays off
      // the line by construction.
      const segdx = tx - sx, segdy = ty - sy
      const len = Math.hypot(segdx, segdy) || 1
      let nx = -segdy / len, ny = segdx / len
      if (ny > 0 || (ny === 0 && nx < 0)) { nx = -nx; ny = -ny }
      const clear = th / 2 + 16
      const mx = sx + 0.5 * segdx
      const my = sy + 0.5 * segdy
      const lx = Math.round(mx + nx * clear - tw / 2)
      const ly = Math.round(my + ny * clear - th / 2)
      els.push({
        id: rngId('lb'),
        type: 'text',
        x: lx, y: ly, width: tw, height: th,
        text: e.label,
        fontSize: EDGE_FONT,
        fontFamily: 1,
        textAlign: 'center',
      })
    }
  }
  return { elements: els, positions }
}

// ── Modifier: overlay callouts (works on any layout that returns positions) ─
// dsl.overlay: [{target: nodeId, text, side: "top"|"right"|"bottom"|"left"}]
// Adds a small Cascadia text + thin arrow pointing at the target's nearest edge.
function applyOverlay(els, dsl, positions) {
  const overlays = dsl.overlay || []
  if (!overlays.length) return els
  const out = els.slice()
  for (const o of overlays) {
    const p = positions.get(o.target)
    if (!p) die(`overlay target "${o.target}" not found among layout nodes`)
    const side = o.side || 'right'
    const text = String(o.text || '')
    if (!text.trim()) die(`overlay on ${o.target} has empty text`)
    const fontSize = 22
    const tw = estTextWidth(text, fontSize, 3)
    const th = estTextHeight(text, fontSize)
    const gap = 60
    let tx, ty, ax1, ay1, ax2, ay2
    switch (side) {
      case 'top':
        tx = p.x + Math.round((p.w - tw) / 2)
        ty = p.y - gap - th
        ax1 = p.x + p.w / 2;  ay1 = ty + th
        ax2 = p.x + p.w / 2;  ay2 = p.y
        break
      case 'bottom':
        tx = p.x + Math.round((p.w - tw) / 2)
        ty = p.y + p.h + gap
        ax1 = p.x + p.w / 2;  ay1 = ty
        ax2 = p.x + p.w / 2;  ay2 = p.y + p.h
        break
      case 'left':
        tx = p.x - gap - tw
        ty = p.y + Math.round((p.h - th) / 2)
        ax1 = tx + tw;        ay1 = p.y + p.h / 2
        ax2 = p.x;            ay2 = p.y + p.h / 2
        break
      case 'right':
      default:
        tx = p.x + p.w + gap
        ty = p.y + Math.round((p.h - th) / 2)
        ax1 = tx;             ay1 = p.y + p.h / 2
        ax2 = p.x + p.w;      ay2 = p.y + p.h / 2
    }
    out.push({
      id: rngId('ov'),
      type: 'text',
      x: tx, y: ty, width: tw, height: th,
      text,
      fontSize, fontFamily: 3, // Cascadia for callouts so they read as annotation
      textAlign: 'left',
      strokeColor: '#666',
    })
    out.push({
      id: rngId('ar'),
      type: 'arrow',
      x: ax1, y: ay1,
      width: ax2 - ax1, height: ay2 - ay1,
      strokeWidth: 1.5,
      strokeStyle: 'dashed',
      strokeColor: '#666',
      endArrowhead: 'arrow',
      points: [[0, 0], [ax2 - ax1, ay2 - ay1]],
    })
  }
  return out
}

// ── Driver ─────────────────────────────────────────────────────────────────
function expand(dsl) {
  // Each layout function may now return either a flat element array (legacy)
  // OR { elements, positions } so applyOverlay can attach callouts. Normalize.
  function viaScene(result) {
    const els = Array.isArray(result) ? result : result.elements
    let positions = Array.isArray(result) ? new Map() : (result.positions || new Map())
    if (positions.size === 0) {
      // Auto-extract positions from any rectangle/ellipse with a stable id —
      // lets overlay callouts target nodes in pipeline/stack/before-after/matrix
      // without each engine threading a positions map through.
      for (const e of els) {
        if ((e.type === 'rectangle' || e.type === 'ellipse') && e.id && !e.id.startsWith('el-') && e.id !== 'title' && e.id !== 'caption') {
          positions.set(e.id, { x: e.x, y: e.y, w: e.width, h: e.height })
        }
      }
    }
    return { kind: 'scene', elements: applyOverlay(els, dsl, positions) }
  }
  switch (dsl.type) {
    case 'pipeline':       return viaScene(layoutPipeline(dsl))
    case 'stack':          return viaScene(layoutStack(dsl))
    case 'before-after':   return viaScene(layoutBeforeAfter(dsl))
    case 'matrix':         return viaScene(layoutMatrix(dsl))
    case 'tree':           return viaScene(layoutTree(dsl))
    case 'timeline':       return viaScene(layoutTimeline(dsl))
    case 'grid':           return viaScene(layoutGrid(dsl))
    case 'graph':          return viaScene(layoutGraph(dsl))
    case 'raw': {
      if (!dsl.raw || !Array.isArray(dsl.raw.elements)) die('raw type requires raw.elements array')
      const hasTitle = dsl.raw.elements.some((e) => e.id === 'title')
      const head = hasTitle ? [] : headerElements(dsl.title, dsl.caption)
      // raw figures can still attach overlays if they wire up positions in
      // dsl.raw.positions: { nodeId: {x, y, w, h} }.
      const positions = new Map(Object.entries(dsl.raw.positions || {}))
      return { kind: 'scene', elements: applyOverlay([...head, ...dsl.raw.elements], dsl, positions) }
    }
    default: die(`unknown DSL type: ${JSON.stringify(dsl.type)}`)
  }
}

function validateClaim(dsl) {
  const c = String(dsl.claim || '').trim()
  const words = c.split(/\s+/).filter(Boolean)
  if (words.length < 8) die(`claim too short (${words.length} words, need ≥ 8): "${c}"`)
  const cap = String(dsl.caption || '').trim()
  if (!cap) die('caption is required (≥ 1 sentence describing the figure thesis)')
  if (cap.toLowerCase() === String(dsl.title || '').toLowerCase()) {
    die('caption must add information, not restate the title')
  }
}

async function main() {
  const [, , inPath, outPath] = process.argv
  if (!inPath || !outPath) die('usage: layout-scene.mjs <input.dsl.json> <output.scene.json>', 2)

  const raw = await readFile(inPath, 'utf8').catch((e) => die(`cannot read ${inPath}: ${e.message}`))
  let dsl
  try { dsl = JSON.parse(raw) } catch (e) { die(`${inPath} is not JSON: ${e.message}`) }

  validateClaim(dsl)
  const expanded = expand(dsl)

  await mkdir(dirname(outPath), { recursive: true }).catch(() => {})

  // Element-level scenes go through author-scene's buildScene() in-process —
  // no subprocess, no temp file, no second JSON parse. Validation errors
  // throw with .errors attached.
  let scene
  try {
    scene = buildScene({
      title: dsl.title,
      elements: expanded.elements,
      export: { padding: 48, theme: 'light', minWidth: 1600, minHeight: 900 },
      _claim: dsl.claim,
      _caption: dsl.caption,
    })
  } catch (e) {
    process.stderr.write(`layout-scene: validation failed for ${inPath}\n`)
    for (const err of (e.errors || [e.message])) process.stderr.write(`  - ${err}\n`)
    process.exit(3)
  }
  await writeFile(outPath, JSON.stringify(scene, null, 2))
  process.stdout.write(`wrote ${outPath} (${scene.elements.length} elements)\n`)
}

main().catch((e) => die(e.stack || e.message || String(e)))
