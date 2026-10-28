# Google: The Gemma 3n Impact Challenge (2025): every prize winner

- **Slug / URL:** `google-gemma-3n-hackathon`, https://www.kaggle.com/competitions/google-gemma-3n-hackathon
- **Dates:** 2025-06-26 to 2025-08-06. Winners: https://www.kaggle.com/competitions/google-gemma-3n-hackathon/discussion/657756 (plus Google blog https://blog.google/technology/developers/developers-changing-lives-with-gemma-3n/).
- **Entrants:** 599 teams / 602 submissions / 862 competitors.
- **Prize pool:** $150,000. Overall 1st $50k, 2nd $25k, 3rd $15k, 4th $10k; five Special Technology Prizes of $10k (Unsloth, Ollama, Google AI Edge, Jetson, LeRobot).
- **Rubric:** Impact & Vision 40, Video Pitch & Storytelling 30, Technical Depth & Execution 30 (checked against a mandatory public repo). Special prizes judged on how central the sponsor technology is.
- **Bank coverage:** `kaggle-llm-writeup-bank-part4.md` H2 "## 2." covers 1st to 4th, Ollama (LENTERA) in depth and Unsloth in one line. It had NOTHING on the LeRobot and Jetson prizes; those two get full-depth sections below. Others get short sections with only facts the bank lacks.
- **Awards:** 9 awards, 8 distinct writeups (Gemma Vision won both 1st place and the Google AI Edge Prize). Host gives no honorable mentions on Kaggle (said they would be highlighted on @googleaidevs on X).

## Winners table

| Place / prize (USD) | Team | Project | Writeup | Video / demo / code (from writeup) |
|---|---|---|---|---|
| Overall 1st ($50k) + Google AI Edge Prize ($10k) | Tommaso Giovannini | Gemma Vision | [gemma-vision](https://www.kaggle.com/competitions/google-gemma-3n-hackathon/writeups/gemma-vision) | see section |
| Overall 2nd ($25k) | Guido Marangoni | Vite Vere Offline | [vite-vere-offline](https://www.kaggle.com/competitions/google-gemma-3n-hackathon/writeups/vite-vere-offline) | see section |
| Overall 3rd ($15k) | Changados Labs | 3VA | [3va](https://www.kaggle.com/competitions/google-gemma-3n-hackathon/writeups/3va) | repo github.com/libreearth/eva_voice_release |
| Overall 4th ($10k) | Sunny Osprey | Sixth Sense for Security Guards | [sixth-sense](https://www.kaggle.com/competitions/google-gemma-3n-hackathon/writeups/sixth-sense-for-security-guards-powered-by-googles) | see section |
| Unsloth Prize ($10k) | Brady Ali Medina | Dream's Assistant (host: "Dream Assistant") | [amoritas-assistant](https://www.kaggle.com/competitions/google-gemma-3n-hackathon/writeups/amoritas-assistant-ai-assistant-for-speech-impaire) | see section |
| Ollama Prize ($10k) | Lentera AI | LENTERA: Offline AI Microserver for Remote & Rural Schools | [lentera](https://www.kaggle.com/competitions/google-gemma-3n-hackathon/writeups/lentera-offline-ai-microserver-for-remote-and-rura) | see section |
| Jetson Prize ($10k) | Stephen Murphy | My (Jetson) Gemma | [my-jetson-gemma](https://www.kaggle.com/competitions/google-gemma-3n-hackathon/writeups/my-jetson-gemma) | video https://youtu.be/WZ93bcTmxw0, https://youtu.be/05_Kl6eQfjY; site https://www.my-gemma.com; repo https://github.com/Halsted312/Googalsted.git (404 / private as of 2026-10-28) |
| LeRobot Prize ($10k) | Raahim Lone | Graph-Based Cost Learning and Gemma 3n for Robotic Sensing | [graphbased-cost-learning](https://www.kaggle.com/competitions/google-gemma-3n-hackathon/writeups/graphbased-cost-learning-and-gemma-3n-for-sensing) | video https://youtu.be/VAvQdEn_PxI; demo https://raahim-lone.github.io/fusion-explorer-demo/; repo https://github.com/Raahim-Lone/gemma (with full PDF report) |

## Per-winner synthesis

### LeRobot Prize: Graph-Based Cost Learning and Gemma 3n for Robotic Sensing (Raahim Lone) [FULL DEPTH, bank gap]
Sources: the Kaggle writeup is THIN (about 200 words, an excerpt pointing to an attached report), so this section is built from the full PDF report in the repo (`Graph_Based_Cost_Learning_and_Gemma_3n_for_Robotic_Sensing___Gemma_3n_Impact_Challenge (1).pdf`, ~1,700 words) and the repo README, https://github.com/Raahim-Lone/gemma.

**Problem and user.** Disaster-response / search-and-rescue robots exploring unknown indoor zones. The author's thesis: movement planning is solved well enough, but *sensing time* (LiDAR, thermal, gas, audio scans) dominates mission time, and classic frontier planners treat scanning as free. The pipeline is "scanning-time first".

**Data (own, synthetic, from LeRobot simulation).** LeRobot simulates Q ≈ 100 synthetic rooms; in each it runs P0 ≈ 200 seed plans of random length 2 to 5 drawn from five primitives `{lidar_scan, thermal_snap, gas_sniff, audio_probe, wait}`. Each primitive logs time (ms), energy (mWh) and a safety flag. Outputs: a sparse latency matrix L (rooms × plans, observed only where run), a room-context matrix X (5 features: size, clutter, heat, gas, noise) and per-plan stats Z (6 features: nodes, parallelism, depth, watts, avg time, safety bit). The patched logger writes LeRobot's own `.parquet` schema so the same code would run on hardware "once a simulator-to-hardware swap is made" (no hardware run was done). README script sizes: `--rooms 20` logging (~3 min on a laptop CPU), `--rooms 40` execution, `--rooms 10 --per_room 3` runtime demo.

**Four-stage pipeline.**
1. *Simulation bootstrap* (above).
2. *Gemma plan generation:* seed sampling with optional duplicate/wait injection (to get both efficient and inefficient patterns), a few-shot grammar prompt, then parse-and-vet (strip Markdown, parse DSL/JSON, reject invalid or "spammy" plans), compute a 6-D feature vector and L2-row-normalise. Accepted plans are serialised straight into LeRobot's action-graph JSON. README: `rewrite_pipeline.py generate --n 200`. Expands to N ≈ 2,000 candidate plans.
3. *IGMC (Inductive Graph-based Matrix Completion):* rooms and plans are the two sides of a bipartite graph, each observed latency a labelled edge; for a query (room, plan) extract the 2-hop enclosing subgraph, apply relation-aware graph convolutions, regress log(1 + latency) with MAE in log space. Trained 160 epochs (`train_igmc_robot.py --epochs 160`), torch-geometric. k-center sampling picks diverse composites (k = 48, up to 120).
4. *Contextual fusion + safe bandit:* append the 5-D room context to the IGMC embedding, an 11-D vector into a two-layer MLP predicting standardised cost ("sub-millisecond"). Runtime loop: batch-score all ~2,000 plans with the MLP each decision step; if the top-2 gap < 10% or a heat/gas hazard is present, send the top-K IDs to Gemma for reranking, with cost-benefit gating from an EMA of Gemma latency; then Thompson-sample a bandit arm weighted by predictive uncertainty and execute via LeRobot. An `llm_rerank_cache.json` in the repo caches reranks.

**Model / runtime.** Gemma via **Ollama**, fully local. Inconsistency to flag: the report says the grammar prompt goes to `gemma3:4b` and "we upload a Gemma 3:4b model locally through Ollama", while the README quick start uses `gemma3n:e4b`. No fine-tuning, no quantisation stated (Ollama default tags are 4-bit). Modalities: text only (sensor data enters as structured features, not audio/images).

**Evaluation.** Simulated time, single agent, 80 × 80 grid, obstacle probability 0.08, speed 1.0 cell/s, sensing radius 4, target 95% free-space coverage, cap 10,000 steps. Baseline: nearest-frontier with a fixed 1 s scan per frontier. Result: baseline reaches 95% at 1,750 time units vs 1,450 for the fusion agent, a **17% speedup** (writeup subtitle rounds to "~20%"); "perfect hazard detection in all trials". Number of seeds/trials and variance are not reported; the result is one coverage-vs-time curve (Figure 1). Live-demo variant (smaller grid with a divider wall and doorway, obstacle p = 0.12, fire cells as no-touch obstacles) "virtually always" beats the baseline (qualitative). No wall-clock latency or memory numbers.

**Lessons stated.** Calling Gemma every step was "prohibitive prompt overhead", which is why gating exists; early fusion predictions were unstable until stratified room sampling, consistent normalisation and held-out validation of embeddings.

**Why it likely won the LeRobot prize.** LeRobot is structural, not decorative: data logged in LeRobot's schema, plans emitted as LeRobot action graphs, execution through the LeRobot simulator. It is also the only winner using the LLM as a selectively-invoked planner inside a learned cost model rather than as the user-facing interface.

### Jetson Prize: My (Jetson) Gemma (Stephen Murphy) [FULL DEPTH, bank gap]
Sources: Kaggle writeup (~600 words, mostly figure captions), the project site https://www.my-gemma.com. The GitHub link in the writeup (github.com/Halsted312/Googalsted) returns 404 and no matching public repo exists under that account, so code could not be checked.

**Problem and user.** A fully offline, privacy-first voice companion for children (homework, learning), elderly people (companionship) and off-grid users (camping, schools without internet). Battery-powered, zero running cost.

**Hardware.** **NVIDIA Jetson Orin NX 16GB** (8-core ARM Cortex-A78AE), CUDA 12.6 (nvidia-smi screenshot), PyTorch 2.6.0 with CUDA. Site: ~$499 hardware, ~15 W under load; writeup: 15 to 25 W.

**Pipeline (all local).**
- *Wake word:* two-stage "Gemma" activation with custom fuzzy matching (site: to handle "7-year-old diction"). The first activation silently pre-warms the model, hiding the cold-start load; subsequent wake words respond in < 100 ms.
- *VAD + STT:* Silero VAD with adaptive thresholds per conversation state; OpenAI **Whisper small** ("95%+ accurate", claimed, no test set), with smart buffering. STT 0.5 to 0.8 s.
- *LLM:* **Gemma 3n via Ollama**, **Q4_K_XL GGUF** quantisation. The site specifies "Gemma 3n 4B quantized to Q4_K_XL via Unsloth" (Unsloth's dynamic GGUF), ollama tag `gemma3n:4b` in the site's code sample (i.e. E4B), "5.4GB on disk, 2.3GB on the GPU", ~12 tok/s. LLM response 1 to 4 s.
- *TTS:* Piper neural TTS, "Amy" voice (kid-friendly), streamed sentence by sentence (< 0.1 s synthesis) so speech starts before generation finishes.
- *Systems engineering:* CPU-GPU split (STT/VAD/audio on the ARM cores, GPU memory kept for the LLM), whole system under 3 GB RAM, conversation context persisted across 20+ exchanges, child-safe content filtering, customisable response length and personality, automatic fallbacks.

**Numbers (all writeup/site claims, no test protocol).** End-to-end < 5 s (writeup) vs "2-3 seconds from end-of-speech to start-of-reply" (site); wake < 100 ms; STT 0.5 to 0.8 s; LLM 1 to 4 s; TTS < 0.1 s; ~12 tok/s; 2.3 GB GPU; < 3 GB RAM; 15 to 25 W.

**Fine-tuning:** none (uses Unsloth's pre-quantised GGUF, not an Unsloth fine-tune). **Modalities:** audio in (via Whisper, not Gemma's own audio encoder), text to Gemma, audio out. **Evaluation:** none beyond the per-stage latency table. **Video:** 3:47 demo running end-to-end on the Jetson.

**Why it likely won the Jetson prize.** It is the cleanest "whole product on one Jetson" story: a measured per-stage latency budget, a GPU/CPU split designed around the Orin's memory, and a power figure that makes battery operation credible. Note the site states the entry "ended up placing 1st"; per the host thread it won the Jetson Prize ($10k), not overall 1st.

### Overall 1st + Google AI Edge Prize: Gemma Vision (Tommaso Giovannini)
Bank covers the core (Flutter + `flutter_gemma` on MediaPipe GenAI v0.10.24, background HF download with the AI Edge Gallery licence flow, camera-preview 10x slowdown, TTS-pause loading music, chest harness, blind brother as tester). New facts from the writeup:
- **Model:** "the 2B version of the 4 bit quantised model" (Gemma 3n E2B int4 via MediaPipe/LiteRT); optional GPU backend in settings for high-end phones. Modalities: image + text in, speech out via platform TTS; dictation in.
- **Negative result on OCR:** Gemma's text recognition from photos "was questionable at best" and hallucinated even on clear photos, so the app runs **Google ML Kit text recognition** on device (usually < 1 s) and passes the detected text to Gemma as context. A hybrid "specialist model + LLM" fix for an LLM weakness.
- **Controller:** an 8BitDo Micro in keyboard mode with keys mapped to F1 to F10 (F13+ not registered on some Android phones; letters would type into the text field): start/stop dictation, send text with photo, send text only, canned "What is this?". A TikTok scroll ring failed because HID events could not be intercepted before the OS.
- **Screen-reader integration:** TalkBack read "button" without labels, so icons were replaced with text buttons; Enter key tracked against focus because iOS VoiceOver would not treat Enter as double-tap; TalkBack interruptions of TTS handled by recording the sentence position and retrying every 6 s up to 5 times (average interruption measured at ~5 s).
- Single-screen UI plus distinct sound cues (dictation start/end chimes, send whoosh, spoken "Sending text with photo").
- Links: site https://gemmavision.com/, code https://github.com/TGTech06/gemma-vision/ (with releases), video https://youtu.be/Fx6IuEggeac, short https://youtube.com/shorts/uY4y0Mu_LI0.
- **Eval:** none quantitative beyond the 50 s to 5 s "hey" latency; qualitative testing with one blind user.

### Overall 2nd: Vite Vere Offline (Guido Marangoni)
Bank covers the product. New facts: **Gemma 3n 2B IT** (E2B) run on **CPU via gemma.cpp** with **int4 quantisation**, streaming inference and "aggressive memory optimization" to stay under RAM crash thresholds on mid-range phones/tablets. Structured output is a fixed three-step instruction format, refined by "targeted prompt engineering, iterative testing with caregivers". Modalities: image (photo of the scene) + text in, text + audio out (offline TTS "Audio Guide"). No fine-tuning, no quantitative evaluation, no latency numbers. Links: code https://github.com/guidomarangoni/vite-vere-offline, page https://guido.link/vitevereoff, video https://youtu.be/yUEzuNDw7-c.

### Overall 3rd: 3VA (Changados Labs: Juan Francisco Pérez, tierralibre)
Bank covers it (809 co-written examples, LoRA via MLX, ~30 min local training, Spanish-native, "29 years of silence"). New facts: the repo ships `models/adapters/eva_voice/` (the 809-example adapter) and `training_scripts/run_eva_voice_finetuning.sh` (MLX pipeline); writeup invokes mix'n'match as "2B for speed, 4B for complexity". Input is pictograms + time + location + personal history (pictograms enter as tokens; no evidence the vision encoder is used), output is text then voice synthesis. LoRA rank/alpha/lr/epochs are not in the writeup. No held-out evaluation; the evidence is the user's own testimony in the video https://www.youtube.com/watch?v=AN9BYKK6pak. Not Unsloth (MLX on Apple silicon).

### Overall 4th: Sixth Sense for Security Guards (Sunny Osprey: Abylay Ospan, Alsu Ospan)
Bank covers the pipeline (16 RTSP cams, NVDEC, Frigate + YOLO-NAS prefilter, 10 frames per incident, JSON verdict, pytest regression clips, RTX 5060 Ti 16GB). New facts:
- **Model/runtime:** `gemma-3n-E2B-it` in **PyTorch / Hugging Face transformers**, unquantised: logged model footprint 9,068.95 MB on GPU + 1,305.95 MB on CPU, 5,439.44 M total parameters (the per-layer embedding table alone is 3,840 MB on GPU).
- **Memory trick:** the **`audio_tower` is kept in CPU memory** because the task is vision-only, saving ~**1.3 GB VRAM** (with a per-parameter device/memory log to prove it).
- **Measured latency:** PyTorch Profiler, **~483 ms per frame (~2 fps)**, which is why prefiltering is essential; 24 h Grafana trace shows GPU 25 to 60% utilisation following the day-night cycle, mostly YOLO-NAS + decoder load. The team adds a disclaimer that performance may be below optimal due to misconfiguration.
- **Cost table** for the "GPUter": RTX 5060 Ti 16GB $429, Ryzen 5 5500 $60, 32 GB DDR4 $52 (plus parts); Sbnb Linux boots into an Ubuntu 24.04 VM with the containers. Alerts go to Telegram.
- **Eval:** a curated clip set (alien-abduction fun test, package theft, a fall, a car pass, a sign) checked by pytest on every prompt/config change; "correctly classified the provided scenarios" (writeup). Small, but the only regression-test discipline among the winners. Live trial in the team's Miami residential community.
- Links: https://github.com/sbnb-io/sunny-osprey/ (prompt in `prompt.txt`), video https://youtu.be/h3UoQgH4F0M.

### Unsloth Prize: Dream's Assistant (Brady Ali Medina)
The Kaggle writeup is THIN (~400 words of features, no technical detail), so this section uses the repo https://github.com/Bra-AM/dream-assistant-android (README, training notebook `Sister_Dream_Assistant_Gemma_3n_Training_Uns.ipynb`, `deployment_info.json`).
- **User:** the author's sister, who has a speech impairment and runs WhatsApp disability communities, a YouTube channel and a small business.
- **Data:** ~200 (README "~200 audio samples", deployment file "202+ voice samples") common phrases recorded by the sister, stored as `sister_speech_recognition.json` + `sister_conversation.json` and formatted into chat-template text.
- **Fine-tune recipe (notebook):** Unsloth `FastLanguageModel`, `load_in_4bit=True`, `max_seq_length=2048`, LoRA **r=16, alpha=16** on q/k/v/o (+ more target modules, truncated), TRL `SFTTrainer`, batch 2 × grad-accum 4, **max_steps=150, lr=2e-4**, exported with `save_pretrained_gguf(quantization_method="q4_k_m")`; 1.59 GB GGUF; claimed loss 7.638 → 0.190.
- **IMPORTANT discrepancy:** the notebook's `model_name` is **`unsloth/Gemma-2-2b-it`** and `deployment_info.json` lists `base_model: Gemma-2-2b-it`, although cells and prints call it "Gemma 3n". The training is text-only (the LLM sees transcripts, not audio), so the "understands her unique speech patterns" claim rests on Android's platform `SpeechRecognitionService` for ASR plus a text model tuned on her phrasing. Judges awarded it anyway; for the series, treat it as evidence that the prize rewarded a personal-data Unsloth fine-tune story rather than a verified Gemma 3n audio fine-tune.
- **Deployment:** Android, Jetpack Compose, llama.cpp via JNI (`llama-android.cpp`, `LlamaEngine.kt`), personalised GGUF with fallback to a base GGUF, listen → LLM → platform TTS loop. Video https://youtu.be/4MTa6NzVsYw. **Eval:** none.

### Ollama Prize: LENTERA (Lentera AI: Fengkie Junis)
Bank covers the product and stack. New facts: **hardware-tier benchmark table** (writeup), Gemma 3n E2B quantised under Ollama: Helio G99 + 12 GB, $120 to 150, 2 to 4 tok/s; Core i7 7th gen + 8 GB, $200 to 300, 5 to 8 tok/s (frequent OOM); **Ryzen 7 7840HS + 16 GB mini-PC, $350 to 400, 10 to 12 tok/s (chosen deployment target)**; i3 12th gen + 16 GB, $500 to 700, 12 to 16 tok/s. Requirements: 16 GB RAM minimum, 128 GB SSD + ZIM storage. **Latency mitigation:** generating summaries/mind maps/quizzes takes 30 to 90 s on this hardware, so a pre-generated cache gives a < 1 s fast path, with "90% of common educational queries served instantly" (writeup claim, no measurement protocol). Text-only. Links: https://github.com/fengkiej/lentera, Notion write-up (in writeup links), live homelab demo.

## Feature matrix

Columns: GV = Gemma Vision (1st + AI Edge), VV = Vite Vere (2nd), 3VA (3rd), 6S = Sixth Sense (4th), DA = Dream's Assistant (Unsloth), LEN = LENTERA (Ollama), JET = My (Jetson) Gemma, LR = Graph-Based Cost Learning (LeRobot).

| Row | GV | VV | 3VA | 6S | DA | LEN | JET | LR |
|---|---|---|---|---|---|---|---|---|
| Model | 3n E2B int4 | 3n E2B ("2B IT") int4 | 3n 2B/4B + LoRA | 3n E2B-it | Gemma-2-2b-it per notebook (called "3n") | 3n E2B quantised | 3n E4B ("4B") Q4_K_XL | `gemma3:4b` (report) / `gemma3n:e4b` (README) |
| Runtime | MediaPipe/LiteRT (`flutter_gemma`) | gemma.cpp (CPU) | MLX | transformers/PyTorch (CUDA) | llama.cpp (Android JNI) | Ollama | Ollama on Jetson Orin NX | Ollama |
| Quantisation | int4 | int4 | ? | none (bf16-size footprint) | Q4_K_M | ? (quantised, level not stated) | Q4_K_XL (Unsloth GGUF) | ? (Ollama default) |
| Modalities into Gemma | image + text | image + text | text (pictogram tokens + context) | image (10 frames) + text | text | text | text (Whisper does ASR) | text |
| Fine-tuned | N | N | Y (LoRA, MLX) | N | Y (LoRA, Unsloth) | N | N | N |
| Unsloth used | N | N | N | N | Y (fine-tune) | N | Y (pre-quantised GGUF only) | N |
| On-device / offline | Y | Y | Y | Y (local GPU box) | Y | Y (local server) | Y | Y |
| Agentic (tools / multi-step) | N (+ ML Kit OCR pipeline) | N | N | N (pipeline + JSON) | partial (phone actions) | N (tools are fixed features) | N | Y (LLM as gated planner/reranker in a bandit loop) |
| Own dataset built | N | N | Y (809) | Y (small clip set) | Y (~200) | N (curated ZIM content) | N | Y (simulated logs) |
| Measured latency / memory | Y (50 s → 5 s) | N | N | Y (483 ms/frame, 9.07 GB GPU) | N | Y (tok/s by hardware) | Y (per-stage, 2.3 GB GPU, 15 to 25 W) | N (sim time only) |
| Eval rigour | qualitative | qualitative | qualitative | small quantitative (regression clips) | none | none | none (latency only) | small quantitative (sim, baseline, no variance) |
| Real named user | Y (blind brother) | Y (caregivers) | Y (Eva) | Y (own community) | Y (sister) | N | N (family use case) | N |
| Video | Y | Y | Y | Y | Y | ? (not in links list shown) | Y | Y |
| Live demo | Y (app release) | ? | N | N | N | Y (homelab) | N | Y (web demo) |
| Open code | Y | Y | Y | Y | Y | Y | N (repo 404) | Y |

**Counts (8 distinct projects):** 8/8 offline/on-device; 2/8 fine-tuned (3VA MLX, DA Unsloth); 3/8 run on llama.cpp-family GGUF or gemma.cpp, 4/8 via Ollama, 1 via MediaPipe/LiteRT, 1 via transformers, 1 via MLX training; 3/8 put images into Gemma (GV, VV, 6S); **0/8 use Gemma 3n's native audio input** (audio is always handled by Whisper, platform ASR, or ignored, and 6S explicitly offloads the audio tower); 5/8 report a latency or memory number; 2/8 have any baseline comparison (LR vs nearest-frontier; GV's before/after camera fix); 6/8 centre a real named user or community; 7/8 open code.

## Consensus vs divergence

- **Table stakes:** offline-first for privacy or connectivity (8/8); a specific vulnerable user group (blind, cognitively disabled, AAC, speech-impaired, rural students, children/elderly, disaster responders); video; public repo; E2B-class model at 4-bit.
- **What separated the podium from the sponsor prizes:** the four places all have a deeply specified *human* (a blind brother, caregivers, Eva, a residential community) and a polished end-to-end product; sponsor prizes went to entries where one technology carries the build (LeRobot schema end to end, a whole Jetson power/latency budget, Ollama on $350 school hardware, an Unsloth personal fine-tune). Technical rigour did not order the podium: 4th (Sixth Sense) has the most engineering measurement of the four; 1st has the most accessibility detail.
- **Contradictions:** *Let Gemma read text, or not?* Gemma Vision found E2B int4 OCR unreliable and offloaded it to ML Kit (https://www.kaggle.com/competitions/google-gemma-3n-hackathon/writeups/gemma-vision), whereas Vite Vere relies on Gemma's own image understanding for task steps (https://www.kaggle.com/competitions/google-gemma-3n-hackathon/writeups/vite-vere-offline). *Fine-tune vs prompt:* 3VA and DA personalise with LoRA on a few hundred user examples; Vite Vere and Sixth Sense explicitly iterate prompts against caregivers or a regression set instead. *Quantise vs not:* Sixth Sense runs E2B unquantised in transformers with ~9 GB VRAM, everyone else runs 4-bit.
- **Gemma 3n's headline feature (audio) went unused by every winner**, and the mix'n'match nested model was mentioned only by 3VA, without evidence of runtime switching.

## UNVERIFIED
- Dream's Assistant: whether any Gemma 3n weights (vs the notebook's Gemma-2-2b-it) were ever fine-tuned; the claim the model "understands her speech patterns".
- LeRobot entry: which model actually ran (`gemma3:4b` vs `gemma3n:e4b`); number of trials behind the 17% figure; whether it ever ran on hardware (report says no).
- Jetson entry: code not inspectable (repo 404); end-to-end latency (< 5 s writeup vs 2 to 3 s site); site's "placed 1st" claim conflicts with the host thread (Jetson Prize).
- 3VA LoRA hyperparameters; "first MLX + Gemma 3n training" claim.
- LENTERA's "90% of queries served instantly"; quantisation level.
- API tracks data orders Jetson as index 1 and LeRobot as index 0 within Special Technology Prizes; all special prizes are $10k and unranked, so the index is not a placing.
- Judges' per-entry comments: the host thread gives none.
