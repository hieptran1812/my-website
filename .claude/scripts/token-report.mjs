#!/usr/bin/env node
/*
 * token-report.mjs — where the tokens actually went.
 *
 * Reads this project's Claude Code transcripts and reports the numbers that
 * drive spend on series production: cache-read volume (the dominant line item),
 * per-turn context size, and how much of it is images lingering in context.
 *
 * The governing identity:
 *
 *     cache-read tokens  =  Σ over turns of (context size at that turn)
 *
 * So the scoreboard is the *median context per turn*, not the word count.
 * Run this after a wave and compare it against the previous wave.
 *
 * Usage:
 *   node .claude/scripts/token-report.mjs            # all transcripts
 *   node .claude/scripts/token-report.mjs --since 2026-07-20
 *   node .claude/scripts/token-report.mjs --top 20
 *
 * Pure Node, zero deps.
 */
import { readdirSync, createReadStream, statSync, existsSync } from 'node:fs'
import { createInterface } from 'node:readline'
import { join, resolve } from 'node:path'
import { homedir } from 'node:os'

// Claude Code stores transcripts under ~/.claude/projects/<cwd-with-slashes-as-dashes>/
const PROJECT_DIR = join(
  homedir(),
  '.claude',
  'projects',
  resolve(process.cwd()).replace(/\//g, '-'),
)

const args = process.argv.slice(2)
const argOf = (flag) => {
  const i = args.indexOf(flag)
  return i === -1 ? null : args[i + 1]
}
const SINCE = argOf('--since')
const TOP = Number(argOf('--top') ?? 12)

// A blog figure is >=1600x900; Claude bills images at ~(w*h)/750 after the
// 1568px long-edge resize, which lands around 1.8-2k tokens. 2000 is the
// estimate used for the image-carry line.
const IMG_TOK = 2000

// Opus 5 rates, $/MTok. Cache read is 0.1x input; cache write is 1.25x (5-min
// TTL) to 2x (1-hour TTL) — Claude Code sessions use the 1-hour TTL, so the
// upper bound is the realistic one.
const RATE = { input: 5, output: 25 }
const COST = {
  cacheRead: RATE.input * 0.1,
  cacheWriteLo: RATE.input * 1.25,
  cacheWriteHi: RATE.input * 2,
  output: RATE.output,
  input: RATE.input,
}

if (!existsSync(PROJECT_DIR)) {
  console.error(`No transcripts found at ${PROJECT_DIR}`)
  console.error('Run this from the project root whose sessions you want to measure.')
  process.exit(1)
}

const files = readdirSync(PROJECT_DIR).filter((f) => f.endsWith('.jsonl'))
if (files.length === 0) {
  console.error(`No .jsonl transcripts in ${PROJECT_DIR}`)
  process.exit(1)
}

const sessions = []
const toolCalls = new Map()
const grand = { in: 0, out: 0, cacheW: 0, cacheR: 0, turns: 0 }
let imageCount = 0
let imageCarry = 0

for (const f of files) {
  const path = join(PROJECT_DIR, f)
  const rl = createInterface({ input: createReadStream(path), crlfDelay: Infinity })
  const s = {
    file: f,
    turns: 0, in: 0, out: 0, cacheW: 0, cacheR: 0,
    images: 0, imgCarry: 0, ctx: [],
    firstTs: null, lastTs: null,
    bytes: statSync(path).size,
  }
  let liveImages = 0

  for await (const line of rl) {
    if (!line.trim()) continue
    let j
    try { j = JSON.parse(line) } catch { continue }

    if (j.timestamp) { s.firstTs ??= j.timestamp; s.lastTs = j.timestamp }

    const content = j.message?.content
    if (Array.isArray(content)) {
      for (const c of content) {
        if (c.type === 'tool_use') {
          const t = toolCalls.get(c.name) || { n: 0 }
          t.n++
          toolCalls.set(c.name, t)
        }
        // Images arrive as tool_result content arrays containing image blocks.
        if (c.type === 'tool_result' && Array.isArray(c.content)) {
          for (const sub of c.content) {
            if (sub.type === 'image') { liveImages++; s.images++ }
          }
        }
      }
    }

    const u = j.message?.usage
    if (!u) continue
    const ctx =
      (u.cache_read_input_tokens || 0) +
      (u.cache_creation_input_tokens || 0) +
      (u.input_tokens || 0)
    s.turns++
    s.in += u.input_tokens || 0
    s.out += u.output_tokens || 0
    s.cacheW += u.cache_creation_input_tokens || 0
    s.cacheR += u.cache_read_input_tokens || 0
    s.ctx.push(ctx)
    // Every image already in context is re-billed on this turn too.
    s.imgCarry += liveImages * IMG_TOK
  }

  if (SINCE && (s.firstTs || '') < SINCE) continue

  grand.in += s.in; grand.out += s.out
  grand.cacheW += s.cacheW; grand.cacheR += s.cacheR; grand.turns += s.turns
  imageCount += s.images
  imageCarry += s.imgCarry
  sessions.push(s)
}

if (sessions.length === 0) {
  console.error(SINCE ? `No sessions since ${SINCE}.` : 'No sessions with usage data.')
  process.exit(1)
}

const M = (n) => (n / 1e6).toFixed(1) + 'M'
const K = (n) => (n / 1e3).toFixed(0) + 'k'
const pct = (arr, q) => {
  const sorted = [...arr].sort((a, b) => a - b)
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * q))] || 0
}
const usd = (n) => '$' + n.toLocaleString('en-US', { maximumFractionDigits: 0 })

console.log(`\n═══ TOKEN REPORT — ${sessions.length} session(s)${SINCE ? ` since ${SINCE}` : ''} ═══\n`)

sessions.sort((a, b) => b.cacheR - a.cacheR)
console.log('TOP SESSIONS BY CACHE READ (cache read = Σ context over turns)')
console.log(
  'cacheR'.padEnd(8), 'out'.padEnd(7), 'turns'.padEnd(7),
  'medCtx'.padEnd(8), 'maxCtx'.padEnd(8), 'imgs'.padEnd(6),
  'span'.padEnd(8), 'started',
)
for (const s of sessions.slice(0, TOP)) {
  const span = s.firstTs && s.lastTs
    ? ((new Date(s.lastTs) - new Date(s.firstTs)) / 3.6e6).toFixed(0) + 'h'
    : '?'
  console.log(
    M(s.cacheR).padEnd(8),
    M(s.out).padEnd(7),
    String(s.turns).padEnd(7),
    K(pct(s.ctx, 0.5)).padEnd(8),
    K(Math.max(0, ...s.ctx)).padEnd(8),
    String(s.images).padEnd(6),
    span.padEnd(8),
    (s.firstTs || '').slice(0, 10),
  )
}

const allCtx = sessions.flatMap((s) => s.ctx)
console.log('\nCONTEXT PER API CALL — this is the scoreboard')
console.log(
  `  p10 ${K(pct(allCtx, 0.1))}   p25 ${K(pct(allCtx, 0.25))}   ` +
  `p50 ${K(pct(allCtx, 0.5))}   p75 ${K(pct(allCtx, 0.75))}   ` +
  `p90 ${K(pct(allCtx, 0.9))}   p99 ${K(pct(allCtx, 0.99))}`,
)
console.log(
  `  calls over 200k: ${allCtx.filter((c) => c > 200e3).length} / ${allCtx.length}` +
  `   over 400k: ${allCtx.filter((c) => c > 400e3).length}`,
)
console.log('  Target for series waves: p50 under 60k, p90 under 120k.')

console.log('\nTOTALS')
console.log(`  cache read   ${M(grand.cacheR).padStart(8)}   ${usd(grand.cacheR / 1e6 * COST.cacheRead).padStart(8)}`)
console.log(
  `  cache write  ${M(grand.cacheW).padStart(8)}   ` +
  `${usd(grand.cacheW / 1e6 * COST.cacheWriteLo)}–${usd(grand.cacheW / 1e6 * COST.cacheWriteHi)}`,
)
console.log(`  output       ${M(grand.out).padStart(8)}   ${usd(grand.out / 1e6 * COST.output).padStart(8)}`)
console.log(`  plain input  ${M(grand.in).padStart(8)}   ${usd(grand.in / 1e6 * COST.input).padStart(8)}`)
const lo = (grand.cacheR * COST.cacheRead + grand.cacheW * COST.cacheWriteLo +
  grand.out * COST.output + grand.in * COST.input) / 1e6
const hi = (grand.cacheR * COST.cacheRead + grand.cacheW * COST.cacheWriteHi +
  grand.out * COST.output + grand.in * COST.input) / 1e6
console.log(`  ── equivalent at Opus 5 API rates: ${usd(lo)}–${usd(hi)} (orchestrator sessions only)`)
console.log(`  ${grand.turns} API calls, mean context ${K(grand.cacheR / grand.turns)}`)

console.log('\nIMAGE CARRY — cost of images sitting in context after being judged')
console.log(
  `  ${imageCount} images read inline → ${M(imageCarry)} cache-read tokens ` +
  `(${usd(imageCarry / 1e6 * COST.cacheRead)})`,
)
console.log('  Should trend to ~0: the figure-reviewer subagent looks at pixels in a context that dies.')

const tools = [...toolCalls.entries()].sort((a, b) => b[1].n - a[1].n).slice(0, 10)
console.log('\nTOOL CALLS')
console.log('  ' + tools.map(([n, t]) => `${n} ${t.n}`).join('   '))
console.log()
