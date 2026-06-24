import express, { Request, Response, NextFunction } from "express";
import { pipeline, env } from "@huggingface/transformers";
import path from "path";
import os from "os";

// ── HuggingFace cache ──────────────────────────────────────────────────────
// On Render, mount a persistent disk at /data so the 1.37 GB model is only
// downloaded once. Locally, falls back to ~/.cache/huggingface.
const HF_CACHE =
  process.env.HF_CACHE_DIR ??
  path.join(os.homedir(), ".cache", "huggingface");
env.cacheDir = HF_CACHE;
env.allowLocalModels = false;

console.log(`[server] HF cache: ${HF_CACHE}`);

// ── Config ─────────────────────────────────────────────────────────────────
// Same model as the browser worker — single-file ONNX q4f16, ~1.37 GB.
// In Node.js @huggingface/transformers automatically uses onnxruntime-node
// (native binary) instead of onnxruntime-web WASM → much faster on CPU.
const MODEL_ID = "DavidLuong/Hy-MT2-1.8B-ONNX-q4f16";
const API_KEY = process.env.TRANSLATE_API_KEY ?? "";
const PORT = parseInt(process.env.PORT ?? "3001", 10);
const MAX_CHARS = 8000;

// ── Language map ───────────────────────────────────────────────────────────
const LANG_NAMES: Record<string, string> = {
  vi: "Vietnamese",
  en: "English",
  "zh-CN": "Simplified Chinese",
  "zh-TW": "Traditional Chinese",
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

function buildPrompt(text: string, targetCode: string): string {
  const lang = LANG_NAMES[targetCode] ?? targetCode;
  return (
    "<｜hy_begin▁of▁sentence｜>" +
    "<｜hy_User｜>" +
    `Translate the following text into ${lang}:\n${text}` +
    "<｜hy_Assistant｜>"
  );
}

function calcMaxNewTokens(inputChars: number): number {
  return Math.min(Math.max(150, Math.round(inputChars / 3.5)), 600);
}

// ── Model singleton ────────────────────────────────────────────────────────
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let modelPipeline: any | null = null;
let modelLoadPromise: Promise<void> | null = null;

function loadModel(): Promise<void> {
  if (modelLoadPromise) return modelLoadPromise;
  modelLoadPromise = (async () => {
    console.log(`[server] loading model: ${MODEL_ID}`);
    const t0 = Date.now();
    modelPipeline = await pipeline("text-generation", MODEL_ID, {
      dtype: "q4f16",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      progress_callback: (info: any) => {
        if (info.status === "progress") {
          const pct = Math.round(info.progress ?? 0);
          process.stdout.write(`\r[server] downloading ${info.file ?? ""} · ${pct}%   `);
        } else if (info.status === "done") {
          process.stdout.write(`\n[server] cached: ${info.file ?? ""}\n`);
        }
      },
    });
    console.log(`[server] model ready in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  })().catch((err) => {
    modelLoadPromise = null; // allow retry
    console.error("[server] model load failed:", err);
    throw err;
  });
  return modelLoadPromise;
}

// ── Express app ────────────────────────────────────────────────────────────
const app = express();
app.use(express.json({ limit: "1mb" }));

// Auth middleware — skip when no key is configured (local dev)
function requireAuth(req: Request, res: Response, next: NextFunction): void {
  if (!API_KEY) { next(); return; }
  const auth = req.headers.authorization ?? "";
  if (auth !== `Bearer ${API_KEY}`) {
    res.status(401).json({ error: "Unauthorized" });
    return;
  }
  next();
}

// Health check — used by Render's health-check probe and by the Vercel proxy
// to decide whether to fall through to Gemini/Google.
app.get("/health", (_req, res) => {
  res.json({ status: "ok", modelReady: modelPipeline !== null });
});

// Translation endpoint
app.post("/translate", requireAuth, async (req: Request, res: Response) => {
  const { text, target } = req.body as { text?: unknown; target?: unknown };

  if (typeof text !== "string" || !text.trim()) {
    res.status(400).json({ error: "Missing text" });
    return;
  }
  if (typeof target !== "string" || !/^[a-zA-Z-]{2,8}$/.test(target)) {
    res.status(400).json({ error: "Invalid target language" });
    return;
  }

  const safeText = text.slice(0, MAX_CHARS).trim();

  try {
    // Ensure model is loaded (no-op if already ready)
    await loadModel();

    const prompt = buildPrompt(safeText, target);
    const maxNewTokens = calcMaxNewTokens(safeText.length);
    console.log(`[server] translate → ${target}, chars=${safeText.length}, max_new_tokens=${maxNewTokens}`);

    const t0 = Date.now();
    const results = (await modelPipeline(prompt, {
      max_new_tokens: maxNewTokens,
      do_sample: false,
      return_full_text: false,
    })) as Array<{ generated_text: string }>;
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

    const raw = results[0].generated_text.trim();
    const translation = raw
      .replace(/^translate the following text into [\w\s()]+:\s*/i, "")
      .trim();

    console.log(`[server] done in ${elapsed}s — "${translation.slice(0, 60)}…"`);
    res.json({ translation, detected: "auto", target });
  } catch (err) {
    console.error("[server] translate error:", err);
    res.status(500).json({
      error: err instanceof Error ? err.message : "Translation failed",
    });
  }
});

// ── Start ──────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`[server] listening on port ${PORT}`);
  // Pre-warm: start loading the model immediately so the first real request
  // doesn't have to wait for the full download + initialisation.
  loadModel().catch(() => {
    // Logged inside loadModel(); server stays up and retries on next request.
  });
});
