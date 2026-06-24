# Optimizing AI Models for the Edge — From First Principles to the Device

Series of ~48 deep-dive posts in **`content/blog/machine-learning/edge-ai/`** (NEW folder, subcategory `Edge AI`).
**blog-writer** voice (principal-engineer, intuition→math→runnable code→tables→case studies), **English**,
≥ 8,000 words, ~7 figures each, `.png` embeds + `optimize-blog-images` (→ webp+cover+manifest; never rewrite
embeds to `.webp`). Commit + push **each wave** (explicit paths, never `git add -A`).

## Angle (non-negotiable)
"Make a model that fits, runs fast, and stays accurate — on hardware that fights you." Every post is **practical
and scientific at once**: the *why* (information theory / numerics / hardware physics behind the technique), the
*how* (a real flow in TFLite/ONNX/PyTorch/TensorRT/llama.cpp, with code that runs), and the *proof* (before→after
numbers on real targets, an accuracy–latency Pareto point, a profiler trace). No hand-waving: every non-obvious
claim gets a chart, a diagram, or a measured case study. The reader should be able to take their own model from
"trained" to "shipped on a phone/Jetson/MCU" with a defensible decision at each step.

The four levers and how they compose are the spine: **quantization · pruning/sparsity · distillation ·
architecture/NAS**, sitting on top of **compilers/runtimes** and validated by **profiling**. We build the
accuracy–efficiency Pareto frontier as the recurring mental model.

---

## Track A — Foundations: why & where the edge bites (Wave 1, 6)
A1 `why-optimize-ai-models-for-the-edge` — The case for on-device AI: latency, privacy, cost, offline, energy; what "edge" actually spans (MCU→phone→Jetson). **Series intro + the four-lever map.**
A2 `the-edge-hardware-landscape` — Hardware from Cortex-M to mobile NPUs to Jetson to FPGAs/ASICs: compute, memory hierarchy, bandwidth, what each is good at and why.
A3 `the-metrics-that-actually-matter-on-device` — Latency vs throughput, params vs activation vs peak memory, energy/power, model size, and **why FLOPs lie** about real latency.
A4 `the-roofline-model-where-your-bottleneck-lives` — Arithmetic intensity, compute-bound vs memory-bound, the roofline; how to read it for a layer and a whole model.
A5 `a-taxonomy-of-model-compression` — Quantization vs pruning vs distillation vs architecture vs runtime; how they compose and conflict; the accuracy–efficiency Pareto frontier as the master frame.
A6 `the-edge-deployment-lifecycle` — train → optimize → convert → deploy → monitor; where each lever slots in; the end-to-end mental model the rest of the series fills in.

## Track B — Quantization (Wave 2, 8) — the highest-ROI lever
B1 `quantization-from-first-principles` — Fixed vs floating point, scale & zero-point, symmetric vs affine, how int8 matmul actually works; the numerics that bound the error.
B2 `post-training-quantization-ptq` — Calibration sets, per-tensor vs per-channel, static vs dynamic, weight-only vs W+A; the cheapest win and its limits.
B3 `quantization-aware-training-qat` — Fake-quant nodes, the straight-through estimator, when QAT recovers the accuracy PTQ loses, and what it costs.
B4 `sub-8-bit-int4-ternary-and-binary-networks` — int4, ternary, BNN/XNOR-Net; the accuracy cliff, where it appears, and how to soften it.
B5 `mixed-precision-and-sensitivity-analysis` — Which layers tolerate low bits; Hessian-based sensitivity (HAWQ); automated bit allocation; the per-layer budget.
B6 `llm-quantization-weight-only-gptq-awq` — The outlier problem, group-wise quantization, GPTQ, AWQ, GGUF k-quants; why weight-only dominates for LLM memory.
B7 `llm-quantization-activations-smoothquant-kv-cache` — LLM.int8(), SmoothQuant, FP8/INT4 inference, KV-cache quantization; the activation-outlier fight.
B8 `quantization-in-practice-a-full-int8-pipeline` — A worked TFLite/ONNX/PyTorch int8 flow, debugging accuracy drops, common footguns, what to check before you ship.

## Track C — Pruning & sparsity (Wave 3, 5)
C1 `pruning-fundamentals` — What to prune (weights/neurons/channels/heads), saliency criteria, the prune→fine-tune loop, sparsity vs speedup.
C2 `unstructured-pruning-and-the-lottery-ticket` — Magnitude & iterative magnitude pruning, the Lottery Ticket Hypothesis, winning tickets; why high sparsity ≠ fast.
C3 `structured-pruning-that-actually-speeds-things-up` — Channel/filter/block pruning, rebuilding the graph, criteria; the kind of sparsity dense hardware rewards.
C4 `n-m-sparsity-and-sparse-tensor-cores` — 2:4 semi-structured sparsity, NVIDIA Sparse Tensor Cores, the hardware that makes sparsity pay.
C5 `pruning-llms-and-transformers` — Head pruning, SparseGPT, Wanda, depth/width & layer pruning; structured pruning of large models.

## Track D — Knowledge distillation (Wave 4, 4)
D1 `knowledge-distillation-fundamentals` — Teacher–student, soft targets, temperature, the dark-knowledge intuition, the loss.
D2 `what-to-distill-response-feature-relation` — Logit/response-based, feature-based, relation-based KD; choosing and combining signals.
D3 `distillation-case-studies-distilbert-to-cnns` — DistilBERT, TinyBERT, MobileBERT, distilling CNNs; the recipes that worked and why.
D4 `distilling-llms-and-reasoning` — Sequence-level & on-policy KD, MiniLLM, distilling CoT/reasoning, synthetic-data distillation; the compound recipe (KD+QAT+pruning) and the order that works.

## Track E — Efficient architectures & NAS (Wave 5, 7)
E1 `building-blocks-for-efficient-models` — Depthwise-separable conv, inverted residuals, grouped conv, squeeze-and-excite; the per-block FLOP/latency math.
E2 `the-mobilenet-family` — V1→V2→V3, width/resolution multipliers, latency-driven design decisions.
E3 `efficientnet-shufflenet-and-the-flops-latency-gap` — Compound scaling, ShuffleNet/GhostNet, and why low-FLOP models can still be slow on real hardware.
E4 `neural-architecture-search-basics` — Search space, strategy (RL/evolution/gradient), DARTS, the compute-cost problem.
E5 `hardware-aware-nas` — Latency lookup tables, MnasNet, FBNet, Once-for-All supernets, ProxylessNAS; searching for the device you'll ship on.
E6 `efficient-attention-and-vision-transformers-for-edge` — Linear/sparse attention, FlashAttention, MobileViT, EfficientViT; making transformers fit.
E7 `small-language-models-by-design` — Phi, Gemma-nano, MobileLLM, TinyLlama; architecture & data choices that make sub-3B models punch above their size.

## Track F — Compilers, runtimes & deployment (Wave 6, 7)
F1 `from-model-to-deployable-artifact` — Graph capture, ONNX as interchange, the conversion pipeline and its footguns (control flow, custom ops, dynamic shapes).
F2 `graph-level-optimization` — Operator fusion, constant folding, layout transforms (NCHW↔NHWC), dead-node elimination; what the optimizer does for free.
F3 `inference-runtimes-compared` — TFLite/LiteRT, ONNX Runtime, ExecuTorch, Core ML, NNAPI; what each is for and how delegates pick hardware.
F4 `tensorrt-and-gpu-edge-inference-on-jetson` — Builder, INT8 calibration, engine plans, dynamic-shape gotchas; getting the most out of a Jetson.
F5 `ml-compilers-and-autotuning-tvm-mlir-xla` — The schedule-search idea, Halide-style algorithm/schedule split, TVM/Ansor, MLIR, XLA, IREE.
F6 `memory-is-the-real-constraint` — Activation memory planning, in-place ops, tensor lifetime, arena allocation, streaming; why peak memory, not size, sinks deployments.
F7 `mobile-deployment-end-to-end` — A model on Android (LiteRT/NNAPI) and iOS (Core ML/ANE): delegate selection, fallback paths, the packaging.

## Track G — TinyML, on-device training & LLMs at the edge (Wave 7, 7)
G1 `tinyml-on-microcontrollers` — TFLite Micro, the no-malloc world, CMSIS-NN, keyword-spotting/anomaly detection on a Cortex-M.
G2 `squeezing-models-into-kilobytes` — Memory-aware design, MCUNet/TinyNAS, patch-based inference, the SRAM/Flash budget.
G3 `on-device-and-federated-learning` — Why train on-device, federated averaging, the privacy story, gradient compression & its limits.
G4 `running-llms-locally-llama-cpp-and-gguf` — llama.cpp & GGUF, k-quants, CPU/Metal/CUDA backends, mmap'd weights; the local-LLM stack explained.
G5 `running-llms-locally-mlc-and-mobile-stacks` — MLC-LLM, mobile LLM runtimes, the prefill/decode split, throughput vs latency on a phone.
G6 `making-on-device-llms-fast` — KV-cache management, speculative decoding, paged attention, continuous batching at the edge.
G7 `multimodal-and-speech-at-the-edge` — On-device VLMs, Whisper.cpp & streaming ASR, real-time constraints; squeezing multimodal into the budget.

## Track H — Profiling, MLOps & case studies (Wave 8, 6)
H1 `profiling-and-benchmarking-on-device` — Measuring latency/memory/energy honestly: per-layer profiling, warm-up, thermal throttling, the traps that fake your numbers.
H2 `the-accuracy-latency-pareto-frontier` — Building and reading the frontier; multi-objective tradeoffs; the decision framework for picking a config.
H3 `edge-mlops` — Model registry, OTA model updates, on-device A/B, drift monitoring, versioning compressed models.
H4 `case-study-real-time-vision-on-device` — A detection/segmentation model from trained→shipped on phone/Jetson: every lever applied, before→after numbers.
H5 `case-study-an-llm-assistant-on-a-laptop` — Quantize→convert→serve a local assistant; the tradeoffs that actually shipped, measured.
H6 `the-edge-optimization-playbook` — **Capstone:** a decision tree from "I have a model" to "it ships" — checklist, ordering of levers, anti-patterns.

**Total: 50 posts.** Flexible — Tracks C/D/H can be trimmed to land at ~45. Waves map to tracks (B/E/F/G split if a wave runs long).

## Cross-link policy
- Link OUT to existing ML posts instead of re-deriving: transformers/attention/KV-cache/decoding →
  `machine-learning/large-language-model/*`; serving/GPU/inference-ops → `machine-learning/mlops/*`;
  scaling tradeoffs → `machine-learning/scaling-laws/*`; LoRA/PEFT/finetuning → `machine-learning/training-techniques/*`.
- Within-series: every post links to A5 (the taxonomy/Pareto frame) and the capstone H6; technique posts cross-link
  where levers compose (e.g. B3 QAT ↔ D4 KD ↔ C5 pruning).

## Process notes (repo conventions — see memory)
- **Images:** author Excalidraw `.in.json` → render PNG → embed `.png` in markdown → run `optimize-blog-images`
  (makes webp + cover + manifest). NEVER rewrite embeds to `.webp`. webp-only / sharpness / `aiGenerated`
  verify-post.sh FAILs are **by-design accepted** under this repo's png+optimize convention.
- **Authoring:** parallel background agents draft prose + diagram DSL (one post each); the **main session** renders
  diagrams, runs `optimize-blog-images`, runs `verify-post.sh`, audits cross-links, and commits **only that wave's**
  md + webp files. Wait for ALL of a wave's agents before render/commit (late finishers re-author figures).
- **Diagram gotchas (carry over from prior series):** figure-kind matches content (loops/≥5-step flows = pipeline,
  not graph; matrices need row/col); chart curves = `arrow` with first point `[0,0]`; keep curves inside the plot
  band; no col-0 `#` comments in code blocks; escape `\$` in body prose only (plain `$` in YAML frontmatter);
  avoid abstraction-trigger phrases in figure alt text. ≥8,000 words, ~7 figures/post.
- **Per wave:** render → optimize-blog-images → verify-post.sh (deep-dive) on each → fix → commit explicit paths → push.
