/**
 * Test script: justinchuby/Hy-MT2-1.8B-ONNX via @huggingface/transformers
 *
 * The justinchuby repo is in onnxruntime-genai format:
 *   - No root config.json  → we inject DavidLuong's config (same weights)
 *   - External data: model.onnx.data (dot) not model.onnx_data (underscore)
 *     → onnxruntime-node finds it automatically when given the model file path
 *   - KV cache shape uses "batch" symbol, not "batch_size"
 *     → model.generate() sets batch=0; we bypass it with a manual inference loop
 *
 * Usage: node scripts/testJustinchuby.mjs
 * First run downloads model.onnx.data (~2 GB). Subsequent runs use local cache.
 */
import { AutoModelForCausalLM, AutoTokenizer, AutoConfig, Tensor, env } from "@huggingface/transformers";

env.allowLocalModels = false;
env.useBrowserCache  = false;

const MODEL_ID  = "justinchuby/Hy-MT2-1.8B-ONNX";
const SUBFOLDER = "Q4_K_M/default";

// Model architecture constants (from normalized_config / ONNX metadata)
const NUM_LAYERS      = 32;
const NUM_KV_HEADS    = 4;
const HEAD_DIM        = 128;
const EOS_TOKEN_ID    = 120020;

const LANG_NAMES = {
  vi: "Vietnamese", en: "English",
  "zh-CN": "Simplified Chinese", ja: "Japanese", ko: "Korean",
};

function buildPrompt(text, lang) {
  const langName = LANG_NAMES[lang] ?? lang;
  return (
    "<｜hy_begin▁of▁sentence｜>" +
    "<｜hy_User｜>" +
    `Translate the following text into ${langName}:\n${text}` +
    "<｜hy_Assistant｜>"
  );
}

const TEST_CASES = [
  { text: "Hello, how are you?",
    target: "vi", note: "Short EN→VI" },
  { text: "This article explains how neural networks learn to represent information. The key insight is that layers transform the input through non-linear activations.",
    target: "vi", note: "Medium EN→VI (technical)" },
  { text: "Xin chào! Hôm nay bạn có khỏe không?",
    target: "en", note: "Short VI→EN" },
  { text: "Mô hình ngôn ngữ lớn (LLM) là các hệ thống AI được huấn luyện trên lượng lớn dữ liệu văn bản để hiểu và tạo ra ngôn ngữ tự nhiên.",
    target: "en", note: "Medium VI→EN (technical)" },
  { text: "Representation engineering is a mechanistic interpretability technique that analyzes activations in neural networks to understand how concepts and knowledge are encoded internally.",
    target: "vi", note: "Long EN→VI (real blog paragraph)" },
  { text: "Các mô hình khuếch tán (diffusion models) đã trở thành phương pháp hàng đầu trong sinh ảnh AI, vượt qua GANs về chất lượng và độ đa dạng.",
    target: "en", note: "Long VI→EN (real blog paragraph)" },
];

/**
 * Manual greedy decode loop that bypasses model.generate().
 *
 * transformers.js initialises KV cache using symbols from inputMetadata.
 * justinchuby uses "batch" (not "batch_size"), so the auto-init resolves to 0
 * and onnxruntime-node rejects the run.  We create the empty cache ourselves
 * with the correct shape [1, NUM_KV_HEADS, 0, HEAD_DIM] and call
 * model.sessions.model.run() directly.
 */
async function manualGenerate(model, inputIds, attentionMask, maxNewTokens) {
  const session = model.sessions["model"];

  // Build initial feeds with correct empty KV cache: [1, NUM_KV_HEADS, 0, HEAD_DIM]
  const emptyKvData = new Float32Array(0); // size = 1 * NUM_KV_HEADS * 0 * HEAD_DIM = 0

  function makeFeeds(ids, mask, kvCache) {
    const feeds = {
      input_ids: new Tensor("int64", ids.data, ids.dims),
      attention_mask: new Tensor("int64", mask.data, mask.dims),
    };
    for (let layer = 0; layer < NUM_LAYERS; layer++) {
      const keyTensor  = kvCache?.[layer]?.key   ?? new Tensor("float32", emptyKvData, [1, NUM_KV_HEADS, 0, HEAD_DIM]);
      const valTensor  = kvCache?.[layer]?.value ?? new Tensor("float32", emptyKvData, [1, NUM_KV_HEADS, 0, HEAD_DIM]);
      feeds[`past_key_values.${layer}.key`]   = keyTensor;
      feeds[`past_key_values.${layer}.value`] = valTensor;
    }
    return feeds;
  }

  const generatedIds = [];
  let currentIds = inputIds;
  let currentMask = attentionMask;
  let kvCache = null; // array of { key: Tensor, value: Tensor }

  for (let step = 0; step < maxNewTokens; step++) {
    const feeds = makeFeeds(currentIds, currentMask, kvCache);
    const outputs = await session.run(feeds);

    // logits shape: [1, seq_len, vocab_size]
    const logits = outputs.logits;
    const seqLen = logits.dims[1];
    const vocabSize = logits.dims[2];
    const lastLogits = logits.data.slice((seqLen - 1) * vocabSize, seqLen * vocabSize);

    // Greedy argmax
    let maxVal = -Infinity, nextToken = 0;
    for (let v = 0; v < vocabSize; v++) {
      if (lastLogits[v] > maxVal) { maxVal = lastLogits[v]; nextToken = v; }
    }

    generatedIds.push(nextToken);
    if (nextToken === EOS_TOKEN_ID) break;

    // Update KV cache with present.X.key/value from this step
    kvCache = Array.from({ length: NUM_LAYERS }, (_, layer) => ({
      key:   outputs[`present.${layer}.key`],
      value: outputs[`present.${layer}.value`],
    }));

    // Next step: feed only the new token with extended attention mask
    const newSeqLen = currentMask.dims[1] + 1;
    const newMaskData = new BigInt64Array(newSeqLen).fill(1n);
    currentIds  = new Tensor("int64", BigInt64Array.from([BigInt(nextToken)]), [1, 1]);
    currentMask = new Tensor("int64", newMaskData, [1, newSeqLen]);
  }

  return generatedIds;
}

async function main() {
  console.log("=".repeat(70));
  console.log("  justinchuby/Hy-MT2-1.8B-ONNX  via @huggingface/transformers");
  console.log("=".repeat(70));
  console.log(`  Model   : ${MODEL_ID}`);
  console.log(`  Variant : ${SUBFOLDER}`);
  console.log(`  Date    : ${new Date().toISOString()}`);
  console.log();

  // ── 1. Load & patch config ────────────────────────────────────────────────
  console.log("Loading base config from DavidLuong/Hy-MT2-1.8B-ONNX-q4f16…");
  const baseConfig = await AutoConfig.from_pretrained("DavidLuong/Hy-MT2-1.8B-ONNX-q4f16");
  if (baseConfig["transformers.js_config"]) {
    baseConfig["transformers.js_config"] = {
      ...baseConfig["transformers.js_config"],
      use_external_data_format: 0, // don't look for model.onnx_data (underscore)
    };
  }
  console.log(`  model_type: ${baseConfig.model_type}`);
  console.log();

  // ── 2. Load model from justinchuby ────────────────────────────────────────
  // onnxruntime-node receives the file PATH to model.onnx; it then automatically
  // loads model.onnx.data from the same directory (standard ONNX external-data).
  console.log(`Loading model from ${MODEL_ID}/${SUBFOLDER}…`);
  console.log("  (model.onnx.data ~2 GB — cached after first run)");
  const loadStart = performance.now();

  let lastFile = "";
  const progressCb = (info) => {
    const file = (info.file ?? info.name ?? "").split("/").pop() ?? "";
    if ((info.status === "progress" || info.status === "progress_total") &&
        typeof info.progress === "number") {
      if (file !== lastFile) { lastFile = file; process.stdout.write("\n"); }
      process.stdout.write(`\r  ${file.padEnd(35)} ${info.progress.toFixed(1).padStart(5)}%`);
    } else if (info.status === "done") {
      process.stdout.write(`\r  ${file.padEnd(35)} done      \n`);
    }
  };

  const model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
    config: baseConfig,
    subfolder: SUBFOLDER,
    model_file_name: "model",
    progress_callback: progressCb,
  });

  const loadMs = performance.now() - loadStart;
  console.log(`\nModel loaded in ${(loadMs / 1000).toFixed(2)} s`);
  console.log();

  // ── 3. Load tokenizer from DavidLuong ────────────────────────────────────
  // justinchuby subfolder has only model.onnx + model.onnx.data, no tokenizer.
  // DavidLuong has identical tokenizer (same base model).
  console.log("Loading tokenizer from DavidLuong/Hy-MT2-1.8B-ONNX-q4f16…");
  const tokenizer = await AutoTokenizer.from_pretrained("DavidLuong/Hy-MT2-1.8B-ONNX-q4f16");
  console.log(`  Tokenizer OK (EOS id: ${tokenizer.eos_token_id})`);
  console.log();

  // ── 4. Run test cases ─────────────────────────────────────────────────────
  const results = [];

  for (let i = 0; i < TEST_CASES.length; i++) {
    const { text, target, note } = TEST_CASES[i];
    const prompt = buildPrompt(text, target);

    console.log(`[${i + 1}/${TEST_CASES.length}] ${note}`);
    const preview = text.length > 90 ? text.slice(0, 87) + "…" : text;
    console.log(`  IN  : "${preview}"`);

    const start = performance.now();

    const tokenized = tokenizer(prompt, { return_tensors: "pt" });
    const inputIds  = tokenized.input_ids;
    const attnMask  = tokenized.attention_mask;
    const inputLen  = inputIds.dims[1];

    let generatedTokens;
    try {
      generatedTokens = await manualGenerate(model, inputIds, attnMask, 400);
    } catch (err) {
      console.error(`  GENERATE ERROR: ${err.message}`);
      console.error(err.stack);
      process.exit(1);
    }

    const ms = performance.now() - start;

    // Remove trailing EOS
    const cleanTokens = generatedTokens.filter(t => t !== EOS_TOKEN_ID);
    const translation = tokenizer.decode(cleanTokens, { skip_special_tokens: true }).trim();

    const outPreview = translation.length > 100 ? translation.slice(0, 97) + "…" : translation;
    console.log(`  OUT : "${outPreview}"`);
    console.log(
      `  TIME: ${ms.toFixed(0)} ms  |  in=${text.length} chars  out=${translation.length} chars  tokens=${generatedTokens.length}`
    );
    console.log();

    results.push({ note, ms, inputLen: text.length, outputLen: translation.length, translation });
  }

  // ── 5. Summary ────────────────────────────────────────────────────────────
  console.log("=".repeat(70));
  console.log("  SUMMARY — justinchuby/Hy-MT2-1.8B-ONNX");
  console.log("=".repeat(70));
  console.log(`  Model load : ${(loadMs / 1000).toFixed(2)} s`);
  console.log();
  let total = 0;
  for (const r of results) {
    total += r.ms;
    console.log(`  ${r.note}`);
    console.log(`    ${r.ms.toFixed(0).padStart(6)} ms  |  in=${r.inputLen}  out=${r.outputLen}`);
  }
  console.log();
  console.log(`  Average : ${(total / results.length).toFixed(0)} ms/request`);
  console.log("=".repeat(70));

  const shortEN = results.find(r => r.note === "Short EN→VI");
  if (shortEN) {
    const ok = shortEN.translation.toLowerCase().includes("khỏe") ||
               shortEN.translation.toLowerCase().includes("chào");
    console.log(`\n  QUALITY: "Hello, how are you?" → "${shortEN.translation}"  ${ok ? "✓ OK" : "? check"}`);
  }
}

main().catch(e => { console.error("FAILED:", e.message ?? e); process.exit(1); });
