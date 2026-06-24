/**
 * Hy-MT2-1.8B Translation Speed Test (via transformers.js)
 *
 * Usage: node scripts/testTranslation.mjs
 *
 * First run downloads the ONNX model (~1.37 GB) to the HuggingFace cache.
 * Subsequent runs use the cached model.
 */
import { pipeline, env } from "@huggingface/transformers";

const MODEL_ID = "DavidLuong/Hy-MT2-1.8B-ONNX-q4f16";

const LANG_NAMES = {
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
};

// Representative blog post excerpts for realistic speed testing
const TEST_CASES = [
  {
    text: "Hello, how are you?",
    target: "vi",
    note: "Short EN→VI",
  },
  {
    text: "This article explains how neural networks learn to represent information. The key insight is that layers transform the input through a series of non-linear activations.",
    target: "vi",
    note: "Medium EN→VI (technical ML)",
  },
  {
    text: "Xin chào! Hôm nay bạn có khỏe không?",
    target: "en",
    note: "Short VI→EN",
  },
  {
    text: "Mô hình ngôn ngữ lớn (LLM) là các hệ thống AI được huấn luyện trên lượng lớn dữ liệu văn bản để hiểu và tạo ra ngôn ngữ tự nhiên.",
    target: "en",
    note: "Medium VI→EN (technical ML)",
  },
  // Real blog post paragraph (from ai-interpretability)
  {
    text: "Representation engineering is a mechanistic interpretability technique that analyzes activations in neural networks to understand how concepts and knowledge are encoded. By probing the internal states of language models, researchers can identify which directions in activation space correspond to specific behaviors or concepts.",
    target: "vi",
    note: "Long EN→VI (real blog paragraph)",
  },
  // Real blog post paragraph (Vietnamese)
  {
    text: "Các mô hình khuếch tán (diffusion models) đã trở thành phương pháp hàng đầu trong sinh ảnh AI, vượt qua GANs về chất lượng và độ đa dạng. Quá trình huấn luyện bao gồm việc thêm nhiễu Gaussian vào ảnh gốc và học cách loại bỏ nhiễu đó.",
    target: "en",
    note: "Long VI→EN (real blog paragraph)",
  },
];

function buildPrompt(text, targetLang) {
  const langName = LANG_NAMES[targetLang] ?? targetLang;
  const instruction = `Translate the following text into ${langName}:\n${text}`;
  return (
    "<｜hy_begin▁of▁sentence｜>" +
    "<｜hy_User｜>" +
    instruction +
    "<｜hy_Assistant｜>"
  );
}

async function main() {
  env.allowLocalModels = false;
  env.useBrowserCache = false;

  console.log("=".repeat(70));
  console.log("  Hy-MT2-1.8B Translation Speed Test  (transformers.js)");
  console.log("=".repeat(70));
  console.log(`  Model : ${MODEL_ID}`);
  console.log(`  Date  : ${new Date().toISOString()}`);
  console.log();

  // ── Load model ──────────────────────────────────────────────────────────────
  console.log("Loading model (downloads ~1.37 GB on first run)…");
  const loadStart = performance.now();
  let lastFile = "";

  const generator = await pipeline("text-generation", MODEL_ID, {
    dtype: "q4f16",
    progress_callback: (info) => {
      const file = (info.file ?? info.name ?? "").split("/").pop() ?? "";
      if (
        (info.status === "progress" || info.status === "progress_total") &&
        typeof info.progress === "number"
      ) {
        if (file !== lastFile) { lastFile = file; process.stdout.write("\n"); }
        process.stdout.write(
          `\r  ${file.padEnd(35)} ${info.progress.toFixed(1).padStart(5)}%`
        );
      } else if (info.status === "done") {
        process.stdout.write(`\r  ${file.padEnd(35)} done      \n`);
      }
    },
  });

  const loadMs = performance.now() - loadStart;
  console.log(`\nModel ready in ${(loadMs / 1000).toFixed(2)} s`);
  console.log();

  // ── Run test cases ───────────────────────────────────────────────────────────
  const results = [];

  for (let i = 0; i < TEST_CASES.length; i++) {
    const { text, target, note } = TEST_CASES[i];
    const prompt = buildPrompt(text, target);

    console.log(`[${i + 1}/${TEST_CASES.length}] ${note}`);
    const preview = text.length > 90 ? text.slice(0, 87) + "…" : text;
    console.log(`  IN  : "${preview}"`);

    const start = performance.now();
    const output = await generator(prompt, {
      max_new_tokens: 400,
      do_sample: false,
      return_full_text: false,
      repetition_penalty: 1.05,
    });
    const ms = performance.now() - start;

    const translation =
      Array.isArray(output) && output[0]
        ? (output[0].generated_text ?? "").trim()
        : "";

    const outPreview =
      translation.length > 90 ? translation.slice(0, 87) + "…" : translation;
    console.log(`  OUT : "${outPreview}"`);
    console.log(
      `  TIME: ${ms.toFixed(0)} ms  |  in=${text.length} chars  out=${translation.length} chars`
    );
    console.log();

    results.push({ note, ms, inputLen: text.length, outputLen: translation.length, translation });
  }

  // ── Summary ──────────────────────────────────────────────────────────────────
  console.log("=".repeat(70));
  console.log("  SUMMARY");
  console.log("=".repeat(70));
  console.log(`  Model load : ${(loadMs / 1000).toFixed(2)} s`);
  console.log();
  let total = 0;
  for (const r of results) {
    total += r.ms;
    console.log(`  ${r.note}`);
    console.log(
      `    ${r.ms.toFixed(0).padStart(5)} ms  |  ${r.inputLen} in → ${r.outputLen} out chars`
    );
  }
  console.log();
  console.log(`  Average : ${(total / results.length).toFixed(0)} ms / request`);
  console.log(`  Total   : ${(total / 1000).toFixed(2)} s for ${results.length} requests`);
  console.log("=".repeat(70));

  // ── Quality spot-check ────────────────────────────────────────────────────────
  const shortEN = results.find((r) => r.note === "Short EN→VI");
  const shortVI = results.find((r) => r.note === "Short VI→EN");
  if (shortEN || shortVI) {
    console.log();
    console.log("  QUALITY SPOT-CHECK");
    if (shortEN) {
      const ok = shortEN.translation.toLowerCase().includes("khỏe") ||
                 shortEN.translation.toLowerCase().includes("chào");
      console.log(`  EN→VI "Hello, how are you?" → "${shortEN.translation}"  ${ok ? "✓ OK" : "? check"}`);
    }
    if (shortVI) {
      const ok = shortVI.translation.toLowerCase().includes("hello") ||
                 shortVI.translation.toLowerCase().includes("how");
      console.log(`  VI→EN "Xin chào…" → "${shortVI.translation}"  ${ok ? "✓ OK" : "? check"}`);
    }
  }
  console.log();
}

main().catch((err) => {
  console.error("Test failed:", err);
  process.exit(1);
});
