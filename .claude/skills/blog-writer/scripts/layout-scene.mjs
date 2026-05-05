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
 * Output: an Excalidraw scene JSON, ready for render-scene.mjs.
 *
 * Usage:
 *   node layout-scene.mjs <input.dsl.json> <output.scene.json>
 *
 * For the `graph` type the output is `{ mermaid, export }` instead — the
 * renderer routes that to window.__renderMermaid in the harness.
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

// ── Engine: pipeline ────────────────────────────────────────────────────────
function layoutPipeline(dsl) {
  const nodes = dsl.nodes || []
  const edges = dsl.edges || []
  if (nodes.length < 2) die('pipeline needs ≥ 2 nodes')
  assertOneAccent(nodes)

  const nodeW = Math.max(...nodes.map((n) => containerWidthFor(n.label)))
  const nodeH = Math.max(...nodes.map((n) => containerHeightFor(n.label)))
  const stride = nodeW + GAP
  const totalRowW = stride * nodes.length - GAP

  // Scale width so the row fills ≥ 85 % of the canvas, then push height up so
  // each card's aspect ratio fills the body band — pipelines used to render as
  // a thin strip floating in a sea of whitespace; now they actually use the
  // canvas. Cap the aspect ratio at 1:1.4 so very-wide nodes don't become
  // square (which reads as "stack", not "pipeline").
  const top = bodyTopY()
  const bodyH = CANVAS_H - top - BODY_BOTTOM_MARGIN
  const targetRowW = Math.max(totalRowW, Math.round(CANVAS_W * 0.88))
  const scale = targetRowW / totalRowW
  const W = Math.round(nodeW * scale)
  const S = Math.round(stride * scale)
  // Fill the body band: H is a fixed fraction of the body height. With
  // horizontal arrows between cards, the row reads as a pipeline regardless
  // of card aspect ratio, so we don't cap by W — using the full vertical
  // budget removes the empty-canvas problem entirely.
  const H = Math.round(bodyH * 0.78)

  const startX = Math.round((CANVAS_W - (S * nodes.length - (S - W))) / 2)
  const yMid = top + Math.round((bodyH - H) / 2)

  const els = headerElements(dsl.title, dsl.caption)
  const positions = new Map()
  for (let i = 0; i < nodes.length; i++) {
    const n = nodes[i]
    const id = n.id || `n${i}`
    const x = startX + i * S
    const y = yMid
    positions.set(id, { x, y, w: W, h: H })
    els.push({
      id,
      type: 'rectangle',
      x, y, width: W, height: H,
      backgroundColor: paletteFor(n.kind),
      strokeWidth: 2,
      label: n.label,
    })
  }

  // Arrows between consecutive nodes (use edges only for labels).
  const edgeLabel = new Map()
  for (const e of edges) edgeLabel.set(`${e.from}>${e.to}`, e.label || '')
  for (let i = 0; i < nodes.length - 1; i++) {
    const a = positions.get(nodes[i].id || `n${i}`)
    const b = positions.get(nodes[i + 1].id || `n${i + 1}`)
    const arrowId = rngId('ar')
    const aw = b.x - (a.x + a.w)
    els.push({
      id: arrowId,
      type: 'arrow',
      x: a.x + a.w,
      y: a.y + Math.round(a.h / 2),
      width: aw,
      height: 0,
      strokeWidth: 2,
      endArrowhead: 'arrow',
      points: [[0, 0], [aw, 0]],
    })
    const lbl = edgeLabel.get(`${nodes[i].id}>${nodes[i + 1].id}`)
    if (lbl) {
      const tw = estTextWidth(lbl, EDGE_FONT)
      els.push({
        id: rngId('lb'),
        type: 'text',
        x: a.x + a.w + Math.round(((b.x - (a.x + a.w)) - tw) / 2),
        y: a.y + Math.round(a.h / 2) - EDGE_FONT - 12,
        width: tw,
        height: estTextHeight(lbl, EDGE_FONT),
        text: lbl,
        fontSize: EDGE_FONT,
        fontFamily: 1,
        textAlign: 'center',
      })
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
      label: n.label,
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
      label: bn.label,
    })
    els.push({
      id: an.id || `a${i}`, type: 'rectangle',
      x: rightX, y: yi, width: colW, height: nodeH,
      backgroundColor: paletteFor(an.kind),
      strokeWidth: 2,
      label: an.label,
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
        label: c.label || '',
      })
    }
  }
  return els
}

// ── Engine: graph (Mermaid output, not element JSON) ───────────────────────
function layoutGraph(dsl) {
  const nodes = dsl.nodes || []
  const edges = dsl.edges || []
  if (nodes.length < 2) die('graph needs ≥ 2 nodes')
  assertOneAccent(nodes)

  // Emit Mermaid `flowchart LR` source. Per-kind classDef carries the palette.
  const classDefs = Object.entries(PALETTE)
    .filter(([_, c]) => c !== 'transparent')
    .map(([k, c]) => `  classDef ${k} fill:${c},stroke:#1e1e1e,stroke-width:2px`)
    .join('\n')

  const lines = ['flowchart LR']
  for (const n of nodes) {
    const id = n.id
    const lbl = String(n.label).replace(/"/g, '\\"')
    lines.push(`  ${id}["${lbl}"]`)
    if (n.kind && n.kind !== 'neutral') lines.push(`  class ${id} ${n.kind};`)
  }
  for (const e of edges) {
    const lbl = e.label ? `|${String(e.label).replace(/\|/g, '/')}|` : ''
    lines.push(`  ${e.from} -->${lbl} ${e.to}`)
  }
  lines.push(classDefs)
  return {
    mermaid: lines.join('\n'),
    export: { padding: 48, theme: 'light', minWidth: 1600, minHeight: 900 },
    // Title and caption are not added in graph mode — Mermaid lays out nodes
    // edge-to-edge and adding title text would conflict with that layout.
    // The figure's prose context (heading and surrounding paragraph) carries
    // the title/caption role.
    _claim: dsl.claim,
    _title: dsl.title,
    _caption: dsl.caption,
  }
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
      label: p.node.label,
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
      label: e.date ? `${e.date}\n${e.label}` : e.label,
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
      label: n.label,
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
      const isVertical = Math.abs(ty - sy) > Math.abs(tx - sx)
      // For vertical edges (same column or near-same): use the line midpoint
      // and offset sideways by enough to clear the adjacent cells. For
      // horizontal/diagonal edges: bias toward source (35% along) and offset
      // upward — keeps the label out of long-rowspan target cells.
      const t = isVertical ? 0.5 : 0.35
      const px = sx + t * (tx - sx)
      const py = sy + t * (ty - sy)
      const lx = isVertical ? Math.round(px + 32) : Math.round(px - tw / 2)
      const ly = isVertical ? Math.round(py - th / 2) : Math.round(py - th - 22)
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
    case 'graph':          return { kind: 'mermaid', payload: layoutGraph(dsl) }
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

  if (expanded.kind === 'mermaid') {
    // Skip author-scene (it's element-level only). Caller's render-scene.mjs
    // detects { mermaid: ... } and routes through window.__renderMermaid.
    await writeFile(outPath, JSON.stringify({
      ...expanded.payload,
      claim: dsl.claim,
    }, null, 2))
    process.stdout.write(`wrote ${outPath} (mermaid)\n`)
    return
  }

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
