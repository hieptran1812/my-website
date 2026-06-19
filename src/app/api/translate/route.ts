import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

// Hard cap so a stray request can't fan out into many upstream calls.
const MAX_CHARS = 8000;
// We POST the text in the request body, so we're bounded by the upstream's
// per-request capacity, not URL length. Keep chunks large so a typical
// selection translates in ONE request — full surrounding context is what
// gives the best quality. Only very long selections get split, and only on
// real boundaries (never mid-word, which produced garbled output before).
const CHUNK_SIZE = 4000;

/**
 * Split text into chunks no larger than `size`, breaking ONLY on paragraph,
 * sentence, or (last resort) word boundaries — never mid-word. Each cut keeps
 * its delimiter, so joining the translated chunks with "" reproduces the
 * original spacing and paragraph structure.
 */
function chunkText(text: string, size: number): string[] {
  if (text.length <= size) return [text];
  const chunks: string[] = [];
  let i = 0;
  while (i < text.length) {
    if (text.length - i <= size) {
      chunks.push(text.slice(i));
      break;
    }
    const w = text.slice(i, i + size);
    // Candidate end positions (exclusive), preferring the latest one.
    const ends: number[] = [];
    const nl = w.lastIndexOf("\n");
    if (nl >= 0) ends.push(nl + 1);
    for (const p of [". ", "! ", "? "]) {
      const k = w.lastIndexOf(p);
      if (k >= 0) ends.push(k + 2);
    }
    for (const p of ["。", "！", "？", "…"]) {
      const k = w.lastIndexOf(p);
      if (k >= 0) ends.push(k + 1);
    }
    const sp = w.lastIndexOf(" ");
    if (sp >= 0) ends.push(sp + 1);
    // Only accept a boundary in the back ~60% of the window; otherwise the
    // chunk would be tiny. A token longer than that (rare: a long URL) gets a
    // hard cut as the fallback.
    const valid = ends.filter((e) => e >= size * 0.4);
    const cut = valid.length ? Math.max(...valid) : size;
    chunks.push(text.slice(i, i + cut));
    i += cut;
  }
  return chunks;
}

interface GoogleResult {
  translated: string;
  detected: string;
}

/**
 * Calls Google Translate's public (keyless) gtx endpoint via POST. The text
 * goes in the request body (not the URL), so the whole selection can be sent
 * in one request without hitting URL-length limits — more context per request
 * means better translations. The response is a nested array: data[0] holds
 * [translatedSegment, originalSegment, ...] tuples and data[2] holds the
 * detected source language.
 */
async function googleTranslate(
  text: string,
  target: string,
  source: string,
): Promise<GoogleResult> {
  const body = new URLSearchParams({
    client: "gtx",
    sl: source,
    tl: target,
    dt: "t",
    q: text,
  }).toString();

  const res = await fetch(
    "https://translate.googleapis.com/translate_a/single",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
        "User-Agent":
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
      },
      body,
      // Server-side fetch; no client CSP applies here.
      cache: "no-store",
    },
  );
  if (!res.ok) throw new Error(`upstream ${res.status}`);

  const data = (await res.json()) as unknown;
  if (!Array.isArray(data) || !Array.isArray(data[0])) {
    throw new Error("unexpected upstream shape");
  }
  const segments = data[0] as Array<[string, ...unknown[]]>;
  const translated = segments
    .map((seg) => (Array.isArray(seg) ? seg[0] : ""))
    .join("");
  const detected =
    typeof data[2] === "string" && data[2] ? (data[2] as string) : source;
  return { translated, detected };
}

export async function POST(request: NextRequest) {
  let body: { text?: unknown; target?: unknown; source?: unknown };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const rawText = typeof body.text === "string" ? body.text : "";
  const target = typeof body.target === "string" ? body.target.trim() : "";
  const source =
    typeof body.source === "string" && body.source.trim()
      ? body.source.trim()
      : "auto";

  const text = rawText.slice(0, MAX_CHARS).trim();
  if (!text) {
    return NextResponse.json({ error: "No text provided" }, { status: 400 });
  }
  if (!/^[a-zA-Z-]{2,8}$/.test(target)) {
    return NextResponse.json(
      { error: "Invalid target language" },
      { status: 400 },
    );
  }

  try {
    const chunks = chunkText(text, CHUNK_SIZE);
    const results: GoogleResult[] = [];
    // Detect the source language on the first chunk, then pin it for the rest
    // so a multi-chunk selection isn't re-detected (and possibly mis-detected)
    // chunk by chunk.
    let src = source;
    for (const chunk of chunks) {
      // One light retry to ride out transient upstream hiccups.
      let r: GoogleResult;
      try {
        r = await googleTranslate(chunk, target, src);
      } catch {
        r = await googleTranslate(chunk, target, src);
      }
      results.push(r);
      if (src === "auto" && r.detected) src = r.detected;
    }
    return NextResponse.json({
      translation: results.map((r) => r.translated).join(""),
      detected: results[0]?.detected ?? source,
      target,
    });
  } catch {
    return NextResponse.json(
      { error: "Translation service unavailable. Please try again." },
      { status: 502 },
    );
  }
}
