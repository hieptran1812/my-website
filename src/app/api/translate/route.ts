import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

// Hard cap so a stray request can't fan out into dozens of upstream calls.
const MAX_CHARS = 5000;
// Google's keyless endpoint is a GET, so each chunk must keep the encoded
// query string well under typical URL limits.
const CHUNK_SIZE = 1200;

/**
 * Split text into chunks no larger than `size`, preferring to break on
 * paragraph / sentence / word boundaries so translation quality holds up.
 */
function chunkText(text: string, size: number): string[] {
  if (text.length <= size) return [text];
  const chunks: string[] = [];
  let rest = text;
  while (rest.length > size) {
    let cut = rest.lastIndexOf("\n", size);
    if (cut < size * 0.5) cut = rest.lastIndexOf(". ", size) + 1;
    if (cut < size * 0.5) cut = rest.lastIndexOf(" ", size);
    if (cut <= 0) cut = size;
    chunks.push(rest.slice(0, cut));
    rest = rest.slice(cut);
  }
  if (rest) chunks.push(rest);
  return chunks;
}

interface GoogleResult {
  translated: string;
  detected: string;
}

/**
 * Calls Google Translate's public (keyless) gtx endpoint. The response is a
 * nested array: data[0] holds [translatedSegment, originalSegment, ...] tuples
 * and data[2] holds the detected source language.
 */
async function googleTranslate(
  text: string,
  target: string,
  source: string,
): Promise<GoogleResult> {
  const url =
    "https://translate.googleapis.com/translate_a/single" +
    `?client=gtx&sl=${encodeURIComponent(source)}` +
    `&tl=${encodeURIComponent(target)}&dt=t&q=${encodeURIComponent(text)}`;

  const res = await fetch(url, {
    headers: {
      "User-Agent":
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    },
    // Server-side fetch; no client CSP applies here.
    cache: "no-store",
  });
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
    for (const chunk of chunks) {
      // One light retry to ride out transient upstream hiccups.
      try {
        results.push(await googleTranslate(chunk, target, source));
      } catch {
        results.push(await googleTranslate(chunk, target, source));
      }
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
