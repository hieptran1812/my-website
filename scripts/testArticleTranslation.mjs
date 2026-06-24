/**
 * Speed test: translate real paragraphs from
 * content/blog/machine-learning/ai-agent/advanced-rag-financial-documents.md
 *
 * Simulates what the blog UI does: translate ~10 representative paragraphs
 * covering intro, body, technical details, and conclusion sections.
 *
 * Usage: node scripts/testArticleTranslation.mjs
 */
import { AutoModelForCausalLM, AutoTokenizer, AutoConfig, Tensor, env } from "@huggingface/transformers";

env.allowLocalModels = false;
env.useBrowserCache  = false;

const MODEL_ID  = "justinchuby/Hy-MT2-1.8B-ONNX";
const SUBFOLDER = "Q4_K_M/default";

const NUM_LAYERS   = 32;
const NUM_KV_HEADS = 4;
const HEAD_DIM     = 128;
const EOS_TOKEN_ID = 120020;
const TARGET_LANG  = "Vietnamese"; // translate EN → VI (most common use case)

// ── Real paragraphs extracted from the article ────────────────────────────
// These are unmodified verbatim paragraph from the .md file (markdown stripped).
const PARAGRAPHS = [
  // Introduction
  "Financial documents are among the hardest document types to process with RAG. Annual reports (10-K/10-Q filings), earnings call transcripts, prospectuses, balance sheets, and research reports are packed with sophisticated tables, diverse layouts, charts and figures, mixed modalities, and domain-specific conventions.",

  // Problem description paragraph
  "A naive RAG pipeline that treats documents as flat text will silently destroy the structural and numerical information that matters most in finance. This article covers battle-tested techniques for building advanced RAG pipelines that handle these challenges.",

  // Layout confusion paragraph
  "A typical 10-K filing has two-column layouts in risk factor sections, tables that span full pages, footnotes referenced across pages, and headers and page numbers interleaved with content. Standard PDF-to-text extraction merges columns incorrectly, producing interleaved gibberish.",

  // Architecture paragraph
  "A production-grade pipeline for financial documents requires specialized components at every stage. The ingestion layer must understand the visual structure of a document — not just extract raw text. Modern document AI models detect and classify regions on a page: text blocks, tables, figures, headers, footers, and page numbers.",

  // Table chunking technical paragraph
  "Tables require special treatment because they encode structured, relational information. Each table should be stored as a self-contained chunk with full context: the table serialized as natural text or Markdown, a header summary identifying what the table represents, the source section and page number, and all footnotes that qualify the data.",

  // Embedding strategy
  "Multi-vector indexing stores different representations of the same document element. For a financial table, you might store a text embedding of the serialized table for semantic search, a summary embedding for high-level matching, and BM25 tokens for exact keyword matching on ticker symbols, GAAP terms, and specific dollar amounts.",

  // Retrieval section
  "Financial queries often mix semantic intent with precise numerical filters. A query like 'What was the revenue growth in the Asia-Pacific segment for fiscal 2025?' requires both semantic understanding of 'revenue growth' and exact matching of 'Asia-Pacific' and '2025'. Pure vector search misses exact matches; pure keyword search misses paraphrase.",

  // Hybrid retrieval
  "Hybrid retrieval combines dense vector search with sparse BM25 to handle both semantic similarity and exact keyword matching. Reciprocal Rank Fusion merges the result lists from each retriever without requiring score normalization — simply sum the reciprocal ranks to produce a combined ranking.",

  // Generation section
  "When constructing the context for the language model, financial RAG requires structured formatting that preserves the semantic relationships between retrieved chunks. Tables should be rendered as Markdown and wrapped with source metadata. The prompt should explicitly instruct the model to cite sources and to decline answering when the retrieved context is insufficient.",

  // Conclusion paragraph
  "Building a production-quality RAG system for financial documents is substantially harder than general document RAG. The investment in proper layout parsing, table-aware chunking, multi-vector indexing, and hybrid retrieval pays off dramatically in answer quality — especially for quantitative questions that depend on exact numerical data from complex tables.",
];

// ── Prompt builder ─────────────────────────────────────────────────────────
function buildPrompt(text) {
  return (
    "<｜hy_begin▁of▁sentence｜>" +
    "<｜hy_User｜>" +
    `Translate the following text into ${TARGET_LANG}:\n${text}` +
    "<｜hy_Assistant｜>"
  );
}

// ── Manual greedy-decode (same approach as translateWorker.ts) ─────────────
async function manualGenerate(model, inputIds, attnMask, maxNewTokens) {
  const session = model.sessions["model"];
  const emptyKv = new Float32Array(0);

  function makeFeeds(ids, mask, kvCache) {
    const feeds = {
      input_ids: new Tensor("int64", ids.data, ids.dims),
      attention_mask: new Tensor("int64", mask.data, mask.dims),
    };
    for (let layer = 0; layer < NUM_LAYERS; layer++) {
      feeds[`past_key_values.${layer}.key`]   = kvCache?.[layer]?.key   ?? new Tensor("float32", emptyKv, [1, NUM_KV_HEADS, 0, HEAD_DIM]);
      feeds[`past_key_values.${layer}.value`] = kvCache?.[layer]?.value ?? new Tensor("float32", emptyKv, [1, NUM_KV_HEADS, 0, HEAD_DIM]);
    }
    return feeds;
  }

  const generatedIds = [];
  let currentIds  = inputIds;
  let currentMask = attnMask;
  let kvCache     = null;

  for (let step = 0; step < maxNewTokens; step++) {
    const outputs = await session.run(makeFeeds(currentIds, currentMask, kvCache));

    const logits    = outputs.logits;
    const seqLen    = logits.dims[1];
    const vocabSize = logits.dims[2];
    const lastLogits = logits.data.slice((seqLen - 1) * vocabSize, seqLen * vocabSize);

    let maxVal = -Infinity, nextToken = 0;
    for (let v = 0; v < vocabSize; v++) {
      if (lastLogits[v] > maxVal) { maxVal = lastLogits[v]; nextToken = v; }
    }

    generatedIds.push(nextToken);
    if (nextToken === EOS_TOKEN_ID) break;

    kvCache = Array.from({ length: NUM_LAYERS }, (_, l) => ({
      key:   outputs[`present.${l}.key`],
      value: outputs[`present.${l}.value`],
    }));

    const newSeqLen = currentMask.dims[1] + 1;
    const newMask = new BigInt64Array(newSeqLen).fill(BigInt(1));
    currentIds  = new Tensor("int64", BigInt64Array.from([BigInt(nextToken)]), [1, 1]);
    currentMask = new Tensor("int64", newMask, [1, newSeqLen]);
  }
  return generatedIds;
}

// ── Main ───────────────────────────────────────────────────────────────────
async function main() {
  console.log("=".repeat(72));
  console.log("  Article Translation Speed Test");
  console.log("  Source: advanced-rag-financial-documents.md");
  console.log(`  Model : ${MODEL_ID}  (${SUBFOLDER})`);
  console.log(`  Target: ${TARGET_LANG}`);
  console.log(`  Date  : ${new Date().toISOString()}`);
  console.log("=".repeat(72));
  console.log();

  // ── Load model ─────────────────────────────────────────────────────────
  console.log("Loading model (2 GB weights, cached after first run)…");
  const loadStart = performance.now();

  const baseConfig = await AutoConfig.from_pretrained("DavidLuong/Hy-MT2-1.8B-ONNX-q4f16");
  if (baseConfig["transformers.js_config"]) {
    baseConfig["transformers.js_config"] = {
      ...baseConfig["transformers.js_config"],
      use_external_data_format: 0,
    };
  }

  let lastFile = "";
  const model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
    config: baseConfig,
    subfolder: SUBFOLDER,
    model_file_name: "model",
    progress_callback: (info) => {
      const file = (info.file ?? info.name ?? "").split("/").pop() ?? "";
      if ((info.status === "progress" || info.status === "progress_total") && typeof info.progress === "number") {
        if (file !== lastFile) { lastFile = file; process.stdout.write("\n"); }
        process.stdout.write(`\r  ${file.padEnd(30)} ${info.progress.toFixed(1).padStart(5)}%`);
      } else if (info.status === "done") {
        process.stdout.write(`\r  ${file.padEnd(30)} done      \n`);
      }
    },
  });

  const loadMs = performance.now() - loadStart;
  console.log(`\nModel loaded in ${(loadMs / 1000).toFixed(2)} s`);

  const tokenizer = await AutoTokenizer.from_pretrained("DavidLuong/Hy-MT2-1.8B-ONNX-q4f16");
  console.log(`Tokenizer OK\n`);

  // ── Translate each paragraph ──────────────────────────────────────────
  const results = [];
  let totalChars = 0;
  let totalTokensGenerated = 0;

  for (let i = 0; i < PARAGRAPHS.length; i++) {
    const text = PARAGRAPHS[i];
    const prompt = buildPrompt(text);
    const inChars = text.length;

    process.stdout.write(`[${String(i + 1).padStart(2)}/${PARAGRAPHS.length}]  ${text.slice(0, 60).replace(/\n/g, " ")}…\n`);

    const t0 = performance.now();
    const enc = tokenizer(prompt, { return_tensors: "pt" });
    const generatedIds = await manualGenerate(model, enc.input_ids, enc.attention_mask, 512);
    const ms = performance.now() - t0;

    const cleanIds = generatedIds.filter(t => t !== EOS_TOKEN_ID);
    const translation = tokenizer.decode(cleanIds, { skip_special_tokens: true }).trim();

    const outChars = translation.length;
    totalChars += outChars;
    totalTokensGenerated += cleanIds.length;

    console.log(`       ${ms.toFixed(0).padStart(5)} ms  in=${inChars} → out=${outChars} chars  (${cleanIds.length} tokens)`);
    console.log(`       VI: ${translation.slice(0, 110)}${translation.length > 110 ? "…" : ""}`);
    console.log();

    results.push({ inChars, outChars, ms, tokens: cleanIds.length, translation });
  }

  // ── Summary ─────────────────────────────────────────────────────────
  const totalMs = results.reduce((s, r) => s + r.ms, 0);
  const avgMs   = totalMs / results.length;

  console.log("=".repeat(72));
  console.log("  RESULTS — advanced-rag-financial-documents.md  EN→VI");
  console.log("=".repeat(72));
  console.log(`  Model load  : ${(loadMs / 1000).toFixed(2)} s`);
  console.log();

  let n = 1;
  for (const r of results) {
    const bar = "█".repeat(Math.round(r.ms / 500));
    console.log(`  [${String(n++).padStart(2)}] ${String(r.ms.toFixed(0)).padStart(6)} ms  in=${r.inChars}  out=${r.outChars}  ${bar}`);
  }

  console.log();
  console.log(`  Total wall-clock : ${(totalMs / 1000).toFixed(2)} s  (${PARAGRAPHS.length} paragraphs)`);
  console.log(`  Average / para   : ${avgMs.toFixed(0)} ms`);
  console.log(`  Median           : ${results.map(r => r.ms).sort((a,b) => a-b)[Math.floor(results.length/2)].toFixed(0)} ms`);
  console.log(`  Fastest          : ${Math.min(...results.map(r => r.ms)).toFixed(0)} ms`);
  console.log(`  Slowest          : ${Math.max(...results.map(r => r.ms)).toFixed(0)} ms`);
  console.log(`  Total tokens out : ${totalTokensGenerated}`);
  console.log(`  Throughput       : ${(totalTokensGenerated / (totalMs / 1000)).toFixed(1)} tokens/s`);
  console.log("=".repeat(72));
}

main().catch(e => { console.error("FAILED:", e.message ?? e); process.exit(1); });
