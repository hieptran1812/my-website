import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

// Hard cap so a stray request can't fan out into many upstream calls.
const MAX_CHARS = 8000;

// --- Render / Hy-MT2 (primary engine when configured) -----------------------
// Set TRANSLATE_SERVER_URL to the Render service URL and TRANSLATE_SERVER_KEY
// to the matching TRANSLATE_API_KEY secret.  When both are present, translation
// goes through the on-server Hy-MT2 1.8B model (best quality, same model as
// the browser worker but running via onnxruntime-node at native speed).
// Falls through to Gemini → Google when the Render service is absent or errors.
const RENDER_URL = process.env.TRANSLATE_SERVER_URL ?? "";
const RENDER_KEY = process.env.TRANSLATE_SERVER_KEY ?? "";

async function renderTranslate(
  text: string,
  target: string,
): Promise<{ translation: string; detected: string } | null> {
  if (!RENDER_URL) return null;
  try {
    const res = await fetch(`${RENDER_URL}/translate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(RENDER_KEY ? { Authorization: `Bearer ${RENDER_KEY}` } : {}),
      },
      body: JSON.stringify({ text, target }),
      signal: AbortSignal.timeout(300_000), // 5 min — server CPU inference
    });
    if (!res.ok) return null;
    const data = (await res.json()) as { translation?: string; detected?: string };
    if (!data.translation) return null;
    return { translation: data.translation, detected: data.detected ?? "auto" };
  } catch {
    return null;
  }
}
// Google fallback POSTs the text in the request body, so we're bounded by the
// upstream's per-request capacity, not URL length. Keep chunks large so a
// typical selection translates in ONE request — full surrounding context is
// what gives the best quality. Only very long selections get split, and only on
// real boundaries (never mid-word, which produced garbled output before).
const CHUNK_SIZE = 4000;

// --- Gemini (primary engine) -------------------------------------------------
// A free-tier Google AI Studio key. When set, translation goes through Gemini
// (an LLM) which is dramatically better than Google Translate for our technical
// content — it keeps jargon, code, Markdown and inline formatting intact and
// uses full context. When the key is absent OR Gemini errors/rate-limits, we
// transparently fall back to the keyless Google endpoint so the feature never
// breaks.
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
// 2.5-flash-lite: cheapest and highest free-tier rate limits, still LLM-quality
// (better than Google Translate). Override via GEMINI_MODEL — e.g.
// "gemini-2.5-flash" — for higher translation quality.
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-2.5-flash-lite";

// Human names for the BCP-47 codes the picker can send, so the prompt is
// unambiguous. Gemini understands the bare codes too, but the name helps.
const LANG_NAMES: Record<string, string> = {
  vi: "Vietnamese",
  en: "English",
  "zh-CN": "Chinese (Simplified)",
  "zh-TW": "Chinese (Traditional)",
  ja: "Japanese",
  ko: "Korean",
  fr: "French",
  de: "German",
  es: "Spanish",
  pt: "Portuguese",
  it: "Italian",
  ru: "Russian",
  hi: "Hindi",
  ar: "Arabic",
  th: "Thai",
  id: "Indonesian",
};

interface TranslationResult {
  translation: string;
  detected: string;
}

// --- In-memory result cache --------------------------------------------------
// Cuts repeat upstream calls: re-translating the same selection, toggling a
// language back and forth, or two readers hitting the same article while the
// serverless instance stays warm. This is per-instance and ephemeral (Vercel
// recycles instances), so it's a best-effort speedup, not a durable store. A
// durable cache (Prisma table keyed by content hash + language) is an easy
// follow-up if call volume ever matters.
const CACHE_MAX = 500;
const cache = new Map<string, TranslationResult>();

function cacheKey(target: string, text: string): string {
  return `${target}\u0000${text}`;
}

function cacheGet(key: string): TranslationResult | undefined {
  const hit = cache.get(key);
  if (hit) {
    // Touch for LRU recency.
    cache.delete(key);
    cache.set(key, hit);
  }
  return hit;
}

function cacheSet(key: string, value: TranslationResult): void {
  cache.set(key, value);
  if (cache.size > CACHE_MAX) {
    // Evict the oldest (first-inserted) entry.
    const oldest = cache.keys().next().value;
    if (oldest !== undefined) cache.delete(oldest);
  }
}

/**
 * Split text into chunks no larger than `size`, breaking ONLY on paragraph,
 * sentence, or (last resort) word boundaries — never mid-word. Each cut keeps
 * its delimiter, so joining the translated chunks with "" reproduces the
 * original spacing and paragraph structure. (Only used by the Google fallback;
 * Gemini takes the whole selection in one call.)
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

// --- Gemini translation ------------------------------------------------------

/**
 * Translate via Gemini with structured JSON output. The caller (the full-article
 * translator) joins individual text nodes with a blank line ("\n\n"); we split
 * the selection back into those segments, translate them as a JSON array, and
 * re-join — so the segment count is preserved end to end and the client's
 * node-by-node mapping stays correct. We control the split/join and validate the
 * array length, which makes this MORE reliable than relying on an opaque MT
 * endpoint to happen to preserve blank lines.
 *
 * Throws on any failure (missing key, HTTP error, blocked content, shape
 * mismatch) so the POST handler can fall back to Google.
 */
async function geminiTranslate(
  text: string,
  target: string,
): Promise<TranslationResult> {
  if (!GEMINI_API_KEY) throw new Error("no gemini key");

  const segments = text.split("\n\n");
  const targetName = LANG_NAMES[target] ?? target;

  const system = [
    `You are a professional translator. Translate text into ${targetName} (language code "${target}").`,
    `You receive a JSON array of text segments. Return a JSON object with:`,
    `- "sourceLanguage": the detected source language as a short code (e.g. "en", "vi", "zh-CN", "ja").`,
    `- "translations": a JSON array of the translated segments, with EXACTLY the same number of elements, in the same order as the input.`,
    ``,
    `Rules:`,
    `- Output the SAME number of translation elements as input segments. Never merge, split, add, or drop elements.`,
    `- Translate naturally and fluently for a technical/educational audience; preserve meaning and tone.`,
    `- Keep technical terms, product/library names, code identifiers, inline code, URLs, math symbols, numbers, and any Markdown/HTML syntax unchanged.`,
    `- If a segment is empty, whitespace, pure punctuation, a number, or code with nothing to translate, return it unchanged.`,
    `- Do not add explanations, notes, or surrounding quotation marks. Return only the JSON object.`,
  ].join("\n");

  const body = {
    systemInstruction: { parts: [{ text: system }] },
    contents: [{ role: "user", parts: [{ text: JSON.stringify(segments) }] }],
    generationConfig: {
      responseMimeType: "application/json",
      responseSchema: {
        type: "OBJECT",
        properties: {
          sourceLanguage: { type: "STRING" },
          translations: { type: "ARRAY", items: { type: "STRING" } },
        },
        required: ["sourceLanguage", "translations"],
        propertyOrdering: ["sourceLanguage", "translations"],
      },
      temperature: 0.2,
      maxOutputTokens: 16384,
      // Translation needs no chain-of-thought; disabling it is faster and
      // spends far fewer free-tier tokens. Only valid on 2.5+ models.
      ...(GEMINI_MODEL.includes("2.5")
        ? { thinkingConfig: { thinkingBudget: 0 } }
        : {}),
    },
  };

  const url = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent`;

  // One light retry to ride out a transient 429/5xx before falling back.
  let res: Response | null = null;
  for (let attempt = 0; attempt < 2; attempt++) {
    res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
      },
      body: JSON.stringify(body),
      cache: "no-store",
      signal: AbortSignal.timeout(25000),
    });
    if (res.ok) break;
    if (attempt === 0 && (res.status === 429 || res.status >= 500)) {
      await new Promise((r) => setTimeout(r, 600));
      continue;
    }
    throw new Error(`gemini ${res.status}`);
  }
  if (!res || !res.ok) throw new Error("gemini failed");

  const data = (await res.json()) as {
    candidates?: Array<{
      content?: { parts?: Array<{ text?: string }> };
      finishReason?: string;
    }>;
    promptFeedback?: { blockReason?: string };
  };

  if (data.promptFeedback?.blockReason) {
    throw new Error(`gemini blocked: ${data.promptFeedback.blockReason}`);
  }
  const raw = data.candidates?.[0]?.content?.parts?.[0]?.text;
  if (typeof raw !== "string") throw new Error("gemini empty response");

  const parsed = JSON.parse(raw) as {
    sourceLanguage?: string;
    translations?: unknown;
  };
  const translations = parsed.translations;
  if (!Array.isArray(translations) || translations.length !== segments.length) {
    throw new Error("gemini segment count mismatch");
  }

  // Keep the original for any segment the model left blank or had nothing to
  // translate — never blank out content.
  const out = segments
    .map((seg, i) => {
      const t = translations[i];
      return typeof t === "string" && t.trim() ? t : seg;
    })
    .join("\n\n");

  const detected =
    typeof parsed.sourceLanguage === "string" && parsed.sourceLanguage
      ? parsed.sourceLanguage
      : "auto";

  return { translation: out, detected };
}

// --- Google fallback ---------------------------------------------------------

interface GoogleResult {
  translated: string;
  detected: string;
}

/**
 * Calls Google Translate's public (keyless) gtx endpoint via POST. Used only as
 * a fallback when Gemini is unavailable. The response is a nested array:
 * data[0] holds [translatedSegment, originalSegment, ...] tuples and data[2]
 * holds the detected source language.
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

/** Chunked Google translation — the resilient fallback path. */
async function googleTranslateAll(
  text: string,
  target: string,
  source: string,
): Promise<TranslationResult> {
  const chunks = chunkText(text, CHUNK_SIZE);
  const results: GoogleResult[] = [];
  // Detect the source language on the first chunk, then pin it for the rest so
  // a multi-chunk selection isn't re-detected (and possibly mis-detected)
  // chunk by chunk.
  let src = source;
  for (const chunk of chunks) {
    let r: GoogleResult;
    try {
      r = await googleTranslate(chunk, target, src);
    } catch {
      r = await googleTranslate(chunk, target, src);
    }
    results.push(r);
    if (src === "auto" && r.detected) src = r.detected;
  }
  return {
    translation: results.map((r) => r.translated).join(""),
    detected: results[0]?.detected ?? source,
  };
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

  const key = cacheKey(target, text);
  const cached = cacheGet(key);
  if (cached) {
    return NextResponse.json({ ...cached, target });
  }

  try {
    let result: TranslationResult;

    // 1. Render / Hy-MT2 server (highest quality; on when TRANSLATE_SERVER_URL is set)
    const renderResult = await renderTranslate(text, target);
    if (renderResult) {
      result = renderResult;
    } else {
      try {
        // 2. Gemini (LLM-quality fallback)
        result = await geminiTranslate(text, target);
      } catch {
        // 3. Google Translate (keyless last resort)
        result = await googleTranslateAll(text, target, source);
      }
    }

    cacheSet(key, result);
    return NextResponse.json({ ...result, target });
  } catch {
    return NextResponse.json(
      { error: "Translation service unavailable. Please try again." },
      { status: 502 },
    );
  }
}
