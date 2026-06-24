/**
 * Speed test: translate real paragraphs from
 * anticipating-the-next-move-escalation-ladders-in-geopolitical-crises.md
 *
 * Usage: node scripts/testGeopoliticalArticle.mjs
 */
import { AutoModelForCausalLM, AutoTokenizer, AutoConfig, Tensor, env } from "@huggingface/transformers";

env.allowLocalModels = false;
env.useBrowserCache  = false;

const MODEL_ID  = "justinchuby/Hy-MT2-1.8B-ONNX";
const SUBFOLDER = "Q4_K_M/default";
const NUM_LAYERS = 32, NUM_KV_HEADS = 4, HEAD_DIM = 128, EOS = 120020;

// 10 đoạn đại diện trích nguyên văn từ bài
const PARAGRAPHS = [
  // 1. Narrative hook — đoạn mở đầu
  "The morning of February 23, 2022, anyone who had been watching the right signals for the preceding six weeks already knew that something catastrophic was coming. Russian troop deployments near the Ukrainian border had swelled to 190,000. Senior diplomat Sergei Lavrov had shifted his language from \"we have no plans\" to \"the situation is critical.\" The ruble had weakened 4% over two weeks without a news trigger.",

  // 2. Core insight — why actors rarely skip rungs
  "Each rung is costlier than the last — in economic terms (sanctions hurt the sender too), in political capital (domestic constituencies resist military action without exhausted alternatives), and in international legitimacy (allies require that lower-rung options be tried before committing to higher ones). A country that jumps straight to military action loses the ability to claim justification.",

  // 3. Rung 4 description — kỹ thuật + số liệu
  "Rung 4: Comprehensive sanctions and asset freezes. SWIFT exclusions, reserve asset freezes via IEEPA, oil price caps with shipping and insurance bans, full sector prohibitions. These target the entire economy rather than specific entities. The asset impact is severe: currency depreciates 20-50%, sovereign spreads blow out 500-2,000 basis points, equities circuit-break.",

  // 4. US toolkit — geopolitics technical
  "The critical tool in the US kit that other actors cannot replicate is secondary sanctions — the ability to threaten any third-country entity that deals with the sanctioned party. This extraterritorial reach is why US sanctions are uniquely powerful. When the US adds Russia's Sberbank to the SDN list with secondary sanctions, a Vietnamese steel company buying Russian iron ore faces a choice: deal with the US financial system or deal with Russia.",

  // 5. China toolkit — commodity leverage
  "China's key leverage tool is rare-earth and critical mineral controls. China processes 85% of the world's rare earths, 90% of graphite, 80% of gallium, and 65% of lithium. When China imposed germanium and gallium export controls in July 2023, semiconductor input prices spiked 40-60% for those materials within weeks. This tool is costly to use because it accelerates the diversification efforts China wants to prevent — but it produces immediate market impact.",

  // 6. Tit-for-tat mechanics — analytical
  "A retaliation that exceeds the original move risks re-escalating beyond what domestic audiences and allies will support. A retaliation far below the original move signals weakness and invites further pressure. The equilibrium is proportionality — matching the scale of the opponent's move to signal resolve without triggering a further ratchet. This is why China's response to US tariffs on $34 billion of goods was tariffs on exactly $34 billion of US goods.",

  // 7. Spike vs. regime change — key concept
  "Every geopolitical shock produces one of two market outcomes: a spike or a regime change. Getting this distinction wrong is the single most costly mistake in geopolitical trading. A spike looks terrifying in real-time but reverses in days to weeks. A regime change feels like a spike initially but never fully recovers — the asset permanently reprices to a new equilibrium. Approximately 80% of geopolitical events produce spikes. Approximately 20% produce regime changes.",

  // 8. De-escalation signal — Stage 2
  "Stage 2: Ambiguous language. A senior official replaces \"unacceptable\" with \"we are open to a framework that addresses our concerns.\" A press spokesperson says \"dialogue is always possible\" after months of \"there is nothing to discuss.\" These language shifts are measurable if you track the exact wording of official statements over time. They precede formal talks by 3-6 weeks on average.",

  // 9. Investor framework — step 4
  "Compute the probability-weighted expected move. Sum the probability-weighted asset impacts. If the expected move is less than the bid-ask spread plus the cost of carry on your hedge, stay flat — the expected value does not justify the position. Position size should be proportional to conviction probability times expected magnitude. Define in advance what observable event would invalidate your thesis.",

  // 10. Misconception 5 — conclusion-style
  "De-escalation rarely fully undoes the damage. After the US-China Phase One deal in January 2020, the US maintained tariffs at 19.3% — far above the pre-trade-war baseline of 3.1%. The Russian ruble partially recovered from its 2022 crash, but Russian banks remained on the SWIFT exclusion list, Russian assets remained largely uninvestable for Western capital, and the energy relationship with Europe did not restore. De-escalation moves assets from \"severe crisis discount\" to \"managed tension discount\" — rarely back to the pre-crisis equilibrium.",
];

function buildPrompt(text) {
  return "<｜hy_begin▁of▁sentence｜><｜hy_User｜>Translate the following text into Vietnamese:\n" + text + "<｜hy_Assistant｜>";
}

async function manualGenerate(model, inputIds, attnMask, maxNew) {
  const session = model.sessions["model"];
  const emptyKv = new Float32Array(0);

  function feeds(ids, mask, kv) {
    const f = {
      input_ids: new Tensor("int64", ids.data, ids.dims),
      attention_mask: new Tensor("int64", mask.data, mask.dims),
    };
    for (let l = 0; l < NUM_LAYERS; l++) {
      f[`past_key_values.${l}.key`]   = kv?.[l]?.key   ?? new Tensor("float32", emptyKv, [1, NUM_KV_HEADS, 0, HEAD_DIM]);
      f[`past_key_values.${l}.value`] = kv?.[l]?.value ?? new Tensor("float32", emptyKv, [1, NUM_KV_HEADS, 0, HEAD_DIM]);
    }
    return f;
  }

  const out = [];
  let ids = inputIds, mask = attnMask, kv = null;

  for (let s = 0; s < maxNew; s++) {
    const res = await session.run(feeds(ids, mask, kv));
    const logits = res.logits;
    const V = logits.dims[2];
    const last = logits.data.slice((logits.dims[1] - 1) * V, logits.dims[1] * V);
    let best = -Infinity, tok = 0;
    for (let v = 0; v < V; v++) if (last[v] > best) { best = last[v]; tok = v; }
    out.push(tok);
    if (tok === EOS) break;
    kv = Array.from({ length: NUM_LAYERS }, (_, l) => ({ key: res[`present.${l}.key`], value: res[`present.${l}.value`] }));
    const newLen = mask.dims[1] + 1;
    const newMask = new BigInt64Array(newLen); newMask.fill(BigInt(1));
    ids  = new Tensor("int64", BigInt64Array.from([BigInt(tok)]), [1, 1]);
    mask = new Tensor("int64", newMask, [1, newLen]);
  }
  return out;
}

async function main() {
  const ARTICLE = "anticipating-the-next-move-escalation-ladders-in-geopolitical-crises";

  console.log("=".repeat(72));
  console.log("  Article Translation Speed Test");
  console.log(`  Source: ${ARTICLE}.md`);
  console.log(`  Model : ${MODEL_ID}  (${SUBFOLDER})`);
  console.log(`  Target: Vietnamese`);
  console.log(`  Date  : ${new Date().toISOString()}`);
  console.log("=".repeat(72));
  console.log();

  console.log("Loading model (cached)…");
  const t0 = performance.now();

  const baseConfig = await AutoConfig.from_pretrained("DavidLuong/Hy-MT2-1.8B-ONNX-q4f16");
  if (baseConfig["transformers.js_config"])
    baseConfig["transformers.js_config"] = { ...baseConfig["transformers.js_config"], use_external_data_format: 0 };

  let lastFile = "";
  const model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
    config: baseConfig, subfolder: SUBFOLDER, model_file_name: "model",
    progress_callback: (info) => {
      const file = (info.file ?? info.name ?? "").split("/").pop();
      if ((info.status === "progress" || info.status === "progress_total") && typeof info.progress === "number") {
        if (file !== lastFile) { lastFile = file; process.stdout.write("\n"); }
        process.stdout.write(`\r  ${file.padEnd(30)} ${info.progress.toFixed(1).padStart(5)}%`);
      } else if (info.status === "done") {
        process.stdout.write(`\r  ${file.padEnd(30)} done\n`);
      }
    },
  });
  const loadMs = performance.now() - t0;
  console.log(`\nModel loaded in ${(loadMs / 1000).toFixed(2)} s`);

  const tokenizer = await AutoTokenizer.from_pretrained("DavidLuong/Hy-MT2-1.8B-ONNX-q4f16");
  console.log("Tokenizer OK\n");

  const results = [];
  let totalTokens = 0;

  const LABELS = [
    "Narrative hook (opening)",
    "Why actors skip rungs (analytical)",
    "Rung 4 description + figures",
    "US secondary-sanctions mechanism",
    "China rare-earth leverage",
    "Tit-for-tat proportionality",
    "Spike vs. regime-change concept",
    "De-escalation Stage 2 language",
    "Investor framework Step 4",
    "Misconception 5 / conclusion",
  ];

  for (let i = 0; i < PARAGRAPHS.length; i++) {
    const text = PARAGRAPHS[i];
    console.log(`[${String(i+1).padStart(2)}/10]  ${LABELS[i]}`);
    console.log(`       IN  (${text.length} chars): ${text.slice(0, 70).replace(/\n/g, " ")}…`);

    const enc = tokenizer(buildPrompt(text), { return_tensors: "pt" });
    const t1 = performance.now();
    const genIds = await manualGenerate(model, enc.input_ids, enc.attention_mask, 512);
    const ms = performance.now() - t1;

    const clean = genIds.filter(t => t !== EOS);
    const vi = tokenizer.decode(clean, { skip_special_tokens: true }).trim();
    totalTokens += clean.length;

    console.log(`       OUT (${vi.length} chars, ${clean.length} tok): ${vi.slice(0, 90)}${vi.length > 90 ? "…" : ""}`);
    console.log(`       TIME: ${ms.toFixed(0)} ms`);
    console.log();

    results.push({ label: LABELS[i], inChars: text.length, outChars: vi.length, tokens: clean.length, ms });
  }

  const totalMs = results.reduce((s, r) => s + r.ms, 0);
  const sorted  = [...results].map(r => r.ms).sort((a,b) => a - b);

  console.log("=".repeat(72));
  console.log(`  RESULTS — ${ARTICLE}`);
  console.log("=".repeat(72));
  console.log(`  Model load : ${(loadMs / 1000).toFixed(2)} s\n`);

  let n = 1;
  for (const r of results) {
    const bar = "█".repeat(Math.min(Math.round(r.ms / 500), 30));
    console.log(`  [${String(n++).padStart(2)}] ${String(r.ms.toFixed(0)).padStart(6)} ms  in=${String(r.inChars).padStart(3)}  out=${String(r.outChars).padStart(3)}  ${bar}`);
  }

  console.log();
  console.log(`  Total time   : ${(totalMs / 1000).toFixed(2)} s  (10 paragraphs)`);
  console.log(`  Average      : ${(totalMs / results.length).toFixed(0)} ms / paragraph`);
  console.log(`  Median       : ${sorted[5].toFixed(0)} ms`);
  console.log(`  Fastest      : ${sorted[0].toFixed(0)} ms`);
  console.log(`  Slowest      : ${sorted[sorted.length-1].toFixed(0)} ms`);
  console.log(`  Total tokens : ${totalTokens}  |  ${(totalTokens / (totalMs / 1000)).toFixed(1)} tokens/s`);
  console.log("=".repeat(72));
}

main().catch(e => { console.error("FAILED:", e.stack ?? e); process.exit(1); });
