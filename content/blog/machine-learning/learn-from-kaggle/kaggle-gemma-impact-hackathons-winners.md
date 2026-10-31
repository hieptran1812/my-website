---
title: "Gemma 3n, MedGemma and Gemma 4 Good on Kaggle: what all 31 impact-hackathon winners built"
date: "2026-10-31"
publishDate: "2026-10-31"
description: "Three judged Kaggle hackathons paid 450,000 USD for on-device and medical Gemma projects. All 31 prize winners taken apart: the rubric, the model and tuning recipe, the runtime, the evidence, and what a 70% story-weighted score actually rewarded."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "hackathon", "gemma", "medgemma", "on-device", "edge-ai", "lora", "quantization", "healthcare-ai"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 22
---

> **TL;DR.** Google ran three "impact" hackathons on Kaggle in under a year: the **Gemma 3n Impact Challenge** (2025, 599 teams), the **MedGemma Impact Challenge** (2026, 868 teams) and the **Gemma 4 Good Hackathon** (2026, 1,603 teams). Judges, not a leaderboard, picked 32 awards across 31 projects. **The decisive lever was a real, named user plus a working offline build**, not model work: the Gemma rubrics give 70 of 100 points to impact and video. Fine-tuning tracked the podium only in MedGemma, where every main-track winner LoRA-tuned MedGemma itself. **The one lesson:** in a judged hackathon the evidence bar is low and the story bar is high, so the entries that *also* bring a held-out test, a baseline and a measured device budget are the ones worth copying. Every number below is the team's own claim; almost none was re-run by anyone, and almost no winner reports a field outcome.

![Prize pools drawn to scale, one tile per 5,000 USD: Gemma 3n (30 tiles), MedGemma (20) and Gemma 4 Good (40), with the main podium taking two thirds, three quarters and one half of each pool](/imgs/blogs/kaggle-gemma-impact-hackathons-winners-1.webp)

The main podium dominates the two smaller pools, but in Gemma 4 Good half the money went to ten unranked track prizes, so much of this post is about entries that never placed.

Most posts in this series are about leaderboards and the [notebook budget](/blog/machine-learning/learn-from-kaggle/kaggle-llm-competitions-how-they-work); these three have neither. As in the [gpt-oss red-teaming post](/blog/machine-learning/learn-from-kaggle/kaggle-gpt-oss-red-teaming-challenge-findings), a panel reads writeups, watches videos and opens repos. The models themselves are in the [Gemma timeline](/blog/machine-learning/frontier-models/gemma-model-family-timeline).

## 1. Three briefs in 60 seconds

As a friend would put it: "Google hands you a small open model and asks you to build something that helps someone, film it, and publish the code."

| | Gemma 3n Impact | MedGemma Impact | Gemma 4 Good |
| --- | --- | --- | --- |
| Dates | 2025-06-26 to 08-06 | 2026-01-13 to 02-24 | 2026-04-02 to 05-18 |
| Model | Gemma 3n (E2B/E4B, text, image, audio) | at least one HAI-DEF model (MedGemma, MedSigLIP, HeAR, MedASR...) | Gemma 4 |
| Teams | 599 | 868 | 1,603 |
| Pool | 150,000 USD | 100,000 USD | 200,000 USD |
| Deliverable | video of 3 min or less, writeup, public repo, demo | video of 3 min or less, writeup of 3 pages or less, reproducible code | video, writeup, code |
| Awards | 9 awards, 8 projects | 9 prizes + 4 unpaid mentions | 14 awards, 14 projects |

Gemma 3n asked for "private, offline-first" products, "bigger than a simple chatbot". MedGemma required Google's Health AI Developer Foundations (HAI-DEF) models and warned that judges watch only three minutes of video.

## 2. The rubrics: 70 points are not code

![The rubric weights drawn to scale: Gemma 3n and Gemma 4 Good give 30 of 100 points to technical depth, MedGemma gives 40 of 100 to HAI-DEF use and product feasibility](/imgs/blogs/kaggle-gemma-impact-hackathons-winners-2.webp)

Read the two bars before reading any winner. Both Gemma hackathons score **Impact & Vision 40, Video Pitch & Storytelling 30, Technical Depth & Execution 30**, with the technical score checked against the mandatory repo. MedGemma is the more technical brief: **Effective use of HAI-DEF 20, Problem domain 15, Impact potential 15, Product feasibility 20, Execution & communication 30**, and "feasibility" explicitly asks for fine-tuning documentation, performance analysis and deployment challenges.

That difference shows up in the winners: the Gemma podiums have the most specified *human*, the MedGemma podium the most documented *fine-tune*. Special prizes are judged on how central the sponsor technology is.

## 3. Gemma 3n Impact Challenge (2025)

| Prize (USD) | Team | Project |
| --- | --- | --- |
| Overall 1st (50k) + Google AI Edge (10k) | Tommaso Giovannini | Gemma Vision |
| Overall 2nd (25k) | Guido Marangoni | Vite Vere Offline |
| Overall 3rd (15k) | Changados Labs | 3VA |
| Overall 4th (10k) | Sunny Osprey | Sixth Sense for Security Guards |
| Unsloth (10k) | Brady Ali Medina | Dream's Assistant |
| Ollama (10k) | Lentera AI | LENTERA |
| Jetson (10k) | Stephen Murphy | My (Jetson) Gemma |
| LeRobot (10k) | Raahim Lone | Graph-Based Cost Learning for Robotic Sensing |

The five Special Technology Prizes are unranked; the host listed no honorable mentions on Kaggle.

### 1st: Gemma Vision, an offline eye for a blind user

![Gemma Vision on one page: a phone on a chest harness and a handheld controller, the photo going through on-device OCR before Gemma 3n E2B int4, and the reply to "hey" drawn to scale at 50 s with a live camera preview versus 5 s after the fix](/imgs/blogs/kaggle-gemma-impact-hackathons-winners-4.webp)

The model work is the smallest part of this winner: stock **Gemma 3n E2B at 4 bits** through Google AI Edge (`flutter_gemma` on MediaPipe GenAI). Everything else is accessibility engineering, tested with the author's fully blind brother:

- **The phone goes on a chest harness.** IP and ESP32 hat cameras were dropped because their images were too poor to read text.
- **A keyboard-mode controller.** An 8BitDo Micro maps keys to F1 to F10 (dictate, send with photo, "What is this?"), because higher keys were not registered on some phones.
- **The camera preview cost 10x.** With the preview open, replying to "hey" took 50 s; opening the camera only for the shot brought it to 5 s.
- **OCR moved out of the LLM.** Gemma's reading of text in photos "was questionable at best" and hallucinated on clear photos, so Google ML Kit text recognition runs on device (usually under 1 s) and its output goes to Gemma as context.
- **Screen readers fought the app.** Icons became text buttons, and TalkBack interruptions are retried from the last sentence every 6 s.

There is no evaluation beyond the latency. The writeup's value is its list of failures, the "proof of work" a judged rubric asks for.

### 2nd to 4th

**Vite Vere Offline (2nd).** A rewrite of an app for people with cognitive disabilities: photograph a scene ("tidy up the room") and get **three simple steps** read aloud. **Gemma 3n E2B int4 runs on CPU through gemma.cpp**, with streaming and "aggressive memory optimization" to stay under RAM crash thresholds on mid-range tablets. No fine-tuning, no metric, no latency number.

**3VA (3rd).** An AAC system for Eva, a woman with cerebral palsy: pictograms plus time, place and personal history go into Gemma 3n with a **LoRA trained on 809 examples co-written with Eva**, in about 30 minutes locally with **MLX** on Apple silicon, and come out as native Spanish speech. LoRA hyperparameters are not stated and there is no held-out test; the evidence is Eva's own testimony in the video.

**Sixth Sense for Security Guards (4th).** The most engineered podium entry. Frigate and YOLO-NAS prefilter sixteen cameras; Gemma 3n E2B sees **10 frames per incident** and returns strict JSON. It runs **unquantised in transformers: 9,068.95 MB on GPU plus 1,305.95 MB on CPU**, because the team moved the unused `audio_tower` off the GPU to save about 1.3 GB. Profiled at **about 483 ms per frame**, which is why the prefilter exists. It is the only Gemma 3n winner with a regression set: curated clips re-run by pytest on every prompt change.

### The four sponsor prizes

**Dream's Assistant (Unsloth).** A personal assistant for the author's sister, who has a speech impairment. About 200 of her phrases became an Unsloth LoRA (r=16, alpha=16, 150 steps, lr 2e-4, exported to Q4_K_M GGUF) running through llama.cpp on Android. One caveat matters: the training notebook loads **`unsloth/Gemma-2-2b-it`**, not Gemma 3n, and training is text only, so speech understanding rests on Android's platform recogniser. The judges rewarded a personal-data fine-tune story.

**LENTERA (Ollama).** An offline school server: Gemma 3n E2B under Ollama plus 120+ GB of Kiwix content over a local hotspot. Its hardware table is the useful part: a 350 to 400 USD Ryzen 7 mini-PC gives 10 to 12 tok/s, a Helio G99 board 2 to 4. Summaries take 30 to 90 s, so a pre-generated cache serves common queries.

**My (Jetson) Gemma (Jetson).** A battery-powered voice companion on a Jetson Orin NX 16GB: Silero VAD, Whisper small, **Gemma 3n E4B at Q4_K_XL via Ollama (2.3 GB on GPU, about 12 tok/s)** and Piper TTS streamed per sentence. Per-stage budget: STT 0.5 to 0.8 s, LLM 1 to 4 s, TTS under 0.1 s, 15 to 25 W. The repo link returns 404.

**Graph-Based Cost Learning (LeRobot).** The only Gemma 3n winner that uses the LLM as a planner rather than an interface. Gemma proposes candidate sensing plans for a search-and-rescue robot, a graph matrix-completion model predicts each plan's latency per room, and Gemma is called again only when the top two plans are within 10% or a hazard is present, because calling it every step was "prohibitive". In simulation it reaches 95% coverage at 1,450 time units against 1,750 for a nearest-frontier baseline, a 17% speedup from one curve with no variance reported. It never ran on hardware.

### What decided Gemma 3n

All eight winners are offline, and **none uses Gemma 3n's native audio input**, the model's headline feature. The podium did not follow rigour: 4th has the most measurement, 1st the most accessibility detail. On a 70% story rubric, that is expected.

## 4. MedGemma Impact Challenge (2026)

| Prize (USD) | Team | Project | HAI-DEF models (host) |
| --- | --- | --- | --- |
| Main 1st (30k) | Jane Odum | EpiCast | MedGemma, MedSigLIP, HeAR |
| Main 2nd (20k) | Daniel and Joshua Bourke | Sunny | MedGemma (+ Gemini 3 Flash) |
| Main 3rd (15k) | Marc Donovici | FieldScreen AI | MedGemma, MedASR, MedSigLIP, HeAR (+ TranslateGemma) |
| Main 4th (10k) | Adish Golechha | Tracer | MedGemma |
| Novel Task (5k) | Brook A | ClinicDx | MedGemma, MedASR |
| Novel Task (5k) | UniRad3s team | UniRad3s | MedGemma, MedSigLIP (+ MedSAM2) |
| Agentic Workflow (5k) | Sai Ganesh, Karthik Vemireddy, Teja Potu | CaseTwin | MedGemma, MedSigLIP (+ Gemini 2.5 Flash) |
| Agentic Workflow (5k) | BIG-TB 6 | BigTB6 | MedGemma, MedSigLIP, HeAR (+ Gemini Live) |
| Edge AI (5k) | Bibek Poudel | BridgeDx | MedGemma, MedASR |

The two prizes in each special category are listed in host order, not ranked.

### The main track: every place fine-tuned MedGemma

**EpiCast (1st).** Community health workers in West Africa write notes in Nigerian Pidgin, Hausa, Yoruba, Twi or French; EpiCast turns them into WHO IDSR surveillance signals. Labelled clinical text in those languages "essentially does not exist", so the author **generated 9,400+ narrative-to-JSON pairs** from 8 syndrome categories x 6 languages, injecting one or two real phrases from Pidgin tweets and Francophone clinical questions so the text was not "textbook Pidgin". MedGemma 4B got a **LoRA (r=16, alpha=32)**, was merged, converted to **GGUF Q4_K_M (2.49 GB)** and runs on iPhone through `llama.rn` with Metal, next to an 851 MB MedSigLIP projector: 3.3 GB, 2 to 10 s per extraction offline. The move on-device was forced by 3-to-5-minute cloud cold starts; the cloud keeps MedGemma 27B for weekly reports and a HeAR cough classifier. **No extraction accuracy and no health-worker validation** are reported.

**Sunny (2nd).** A private skin and sunscreen tracker. MedGemma 1.5 4B was fine-tuned with TRL on **about 1,100 images** (ISIC-2024 crops plus real sunscreen photos), labels written by **Gemini 3 Flash as teacher**, 3 epochs on one A100 in 40 to 50 minutes. The quantised base model could not follow the long structured prompt. Converted with `mlx-vlm` to 4-bit (8.6 GB to 3.4 GB), it gives a first token in 3 to 4 s on an iPhone 17 Pro. The authors tested only on their own skin, and say so.

**FieldScreen AI (3rd).** TB screening at camps with no radiologist and no internet: the WHO four-symptom questionnaire, a HeAR cough risk on CPU, MedASR for spoken notes, then **MedGemma 1.5 4B with a LoRA (rank 32, 5 epochs, 1,200 chest X-rays)** read with **5-run majority voting**, and TranslateGemma for 15 languages. Shipping it needed a bf16 vision tower beside 4-bit language layers and a **custom HF-to-GGUF converter**. On **200 held-out images**, accuracy goes from 84.0% to 86.0% and specificity from 95.0% to 97.0%, with the test set in the repo. The caveats are the author's: same source database as part of training, and 75% sensitivity is low for screening.

**Tracer (4th).** Outpatient clinicians lose track of results that arrive days after the hypothesis. MedGemma 1.5 4B got a **4-bit LoRA on 421 hand-validated examples across 25 specialties** (3 epochs, a Kaggle T4, 1 h 48 m) inside a four-agent chain: validator, hypothesis extractor, quality checker and a 1-to-10 confidence scorer that re-runs below 6. Its 8-model comparison: base MedGemma 4B produced valid structure 50% of the time, the fine-tune 100%, and Qwen2.5 7B, Llama 3.2 3B, BioMistral 7B and Meditron 7B 0%, on **10 held-out notes**, so every step is 10 points. Tracer is also the only winner with clinician input: a Google Form survey of 100+ respondents.

### The special prizes

**ClinicDx (Novel Task)** is the most ambitious fine-tune here. A projector maps **MedASR encoder states into MedGemma's embedding space**, trained 150,000 steps with a language-modelling plus contrastive loss, so one checkpoint takes speech directly and outputs FHIR observations: **92% exact match on 321 held-out clips**. A second LoRA (r64, 5,000 synthetic traces) teaches a tool loop of up to 6 turns over 160+ WHO and MSF guidelines. The decision-support quality has no clinical evaluation.

**UniRad3s (Novel Task)** spots glioma slices, segments the lesion with MedSAM2 from one radiologist click, and simplifies the report. The fine-tune was of **MedSigLIP, not MedGemma** (the host blurb says MedGemma): FLAIR accuracy 93.5% zero-shot to 98.25% after LoRA. Worth copying: MedGemma misplaced lesions, so an atlas registered with ANTs feeds the touched brain regions into the prompt. Tumours and controls come from different datasets, which can inflate accuracy.

**CaseTwin and BigTB6 (Agentic Workflow)** share a pattern: **Gemini runs the agent loop and HAI-DEF models are tools.** CaseTwin embeds a chest X-ray with MedSigLIP, retrieves from 2.2k PubMed Central cases with a 0.7 visual plus 0.3 clinical score, then a CrewAI crew on Gemini 2.5 Flash finds a hospital and named physicians on the open web. BigTB6 is a Gemini Live voice agent calling cough, X-ray, eye, palm and fingernail screeners in parallel; its HeAR cough probe reports AUC 0.75 (patient-level CV, repo only). Both are cloud-only and neither fine-tunes MedGemma.

**BridgeDx (Edge AI)** runs unfine-tuned MedGemma 4B and 27B through Ollama on a laptop, routing to 4B below 20 GB RAM, and puts its reliability into retrieval: BM25 plus vectors fused by reciprocal rank, a cross-encoder rerank, and **post-hoc citation verification** that strips any quote it cannot match to source. Its evaluation is four hand-picked cases.

The four unpaid honorable mentions were Dual Path ICU, Sentinel, Enso Atlas and CAP CDSS.

## 5. Gemma 4 Good Hackathon (2026)

Same 40/30/30 rubric as Gemma 3n.

| Prize (USD) | Team | Project |
| --- | --- | --- |
| Main 1st (50k) | takaki-maeda and 4 teammates | GEM-4 |
| Main 2nd (25k) | DuoDol | Trido |
| Main 3rd (15k) | And0Sha | PenguinAgent |
| Main 4th (10k) | Inchara | DEMENTOR |
| Impact: Health & Sciences (10k) | shashank padala | Project Rosie |
| Impact: Global Resilience (10k) | Acuiferovigia | Acuífero·Vigía |
| Impact: Future of Education (10k) | Adinda Panca Mochamad | CodeBuddy |
| Impact: Digital Equity (10k) | Bilal Bounajma | Gem-Care |
| Impact: Safety & Trust (10k) | Shiwon Oh | TrueVoice |
| Special: Cactus (10k) | Pradana Yahya | KawanIsyarat |
| Special: LiteRT (10k) | psymon | Gilbeot |
| Special: llama.cpp (10k) | Ashish Thapa | PreVillage |
| Special: Ollama (10k) | getalife() | PathOS |
| Special: Unsloth (10k) | Huzayfah Patel | MindBridge |

Impact and Special prizes are unranked.

### 1st: GEM-4, a wearable second arm

![GEM-4 on LIBERO as four grids of 100 episodes: 72 successes on Spatial, 92 on Object, 89 on Goal and 43 on Long, average 74%, with no external baseline reported](/imgs/blogs/kaggle-gemma-impact-hackathons-winners-5.webp)

GEM-4 is a vision-language-action model for a wearable robot arm that helps people with visual or upper-limb impairments pull out a chair or pick up a dropped object. **SigLIP plus a frozen Gemma 4 E2B backbone**, per-domain input and output projectors, and an L1 action head that cross-attends over the hidden states of every Gemma layer: only projectors and head train. Pretrained on 9 Open-X-Embodiment datasets plus LIBERO for 100k steps, then fine-tuned per suite. The grids are the strongest benchmark among the 14 winners, with checkpoints and eval configs on Hugging Face, but the team names Long as the weakness and gives no comparison to a published VLA.

The data tooling is the other half. MimicAnno segments demonstrations at gripper events, tracks objects with SAM3 and labels each segment with **a Gemma 4 QLoRA trained with Unsloth** (the host says 31B; the writeup does not state the size), and turns first-person GoPro video into robot labels. The VLA runs on a Jetson AGX Orin, with no latency numbers.

### 2nd to 4th

**Trido (2nd)** was used for six weeks by one teacher, Pak Damar, who has physical disabilities in both hands and legs. Indonesian speech becomes **structured canvas function calls** (`add_mindmap_node`, `connect_nodes`) with a base64 screenshot and a JSON dump of the canvas in every request, so placements are spatially aware. Gemma 4 31B serves in the cloud and E2B via Ollama offline, switched on connectivity. The negative finding: smaller models failed silently with wrong types and free text instead of a call.

**PenguinAgent (3rd)** tracks penguins in field video with SAM 3, re-stitches tracks across 10-second chunks with the Hungarian algorithm, identifies species with a **SigLIP 2 linear probe** (zero-shot could not separate 18 species), and hands telemetry to **Gemma 4 26B via Ollama** with two tools: offline literature RAG and matplotlib code execution, capped at 3 turns after it recursed. "Edge" here is a workstation with an RTX 5080 and 512 GB of RAM. There is no quantitative evaluation.

**DEMENTOR (4th)** is a four-layer stack for dementia caregivers: an ESP32 wearable, a Raspberry Pi hub with Vosk speech recognition, a phone-side hub and a caregiver app. When speech mentions symptoms, function calling routes to an **E2B Unsloth LoRA (r=16, alpha=16) served as GGUF q8_0** that must emit strict JSON with an urgency level. "Safety logic must stay outside the model." Training data, learning rate and evaluation are not given.

### The five Impact Track prizes

**Project Rosie (Health).** Personalised neoantigen mRNA vaccine designs for dogs from a tumour VCF, on Cloud Run in under an hour at about 15 USD per case. Gemma 4 26B-A4B on Vertex AI does four interpretive jobs, including a pre-flight VCF check. The transferable idea: Gemma's synthesis specification was **replaced with a deterministic template**, because one hallucinated catalogue number would discredit the tool. One expert review; no metrics.

**Acuífero·Vigía (Resilience).** Flash-flood warning for Argentina. A Pi 5 camera node runs an OpenCV "deterministic firewall" and invokes **Gemma 4 E2B on LiteRT-LM** only on meaningful water-level change, at a warm median of 22.86 s per frame and 4.29 to 4.58 W. Rioplatense citizen reports are structured on the phone with **82% exact-schema match** from prompting alone. A rule-based fallback fires if Gemma is slow or malformed, and the writeup lists what it has not measured.

**CodeBuddy (Education).** A Python tutor for Indonesian pupils who code in paper notebooks: Gemma 4 E4B via Ollama reads handwriting, and learner code runs only in a RestrictedPython sandbox. The served model is stock; no evaluation.

**Gem-Care (Equity).** Speech recognition for dysarthric speakers. Adding clinical context to the prompt made vanilla Gemma **worse (WER 39.5% vs 32.7% audio-only)**, because narrative context pulled it into clinical prose. A PEFT LoRA on E2B (rank 16 on Q/K/V, 0.06% of weights, audio and vision towers frozen, about 8.5 h) on Speech Accessibility Project audio brought **WER to 19.0%**. The repo's per-epoch table ends at 19.6%, and the fine-tuned audio-only control is not reported, so the share of the gain due to context is unshown.

**TrueVoice (Safety).** A voice-deepfake detector that uses **only Gemma 4 E4B's frozen audio tower**, mean-pooled into a 1.5 MB classification head. Training audio was resampled through 8 kHz so studio quality could not leak the label. On the full **ASVspoof 2021 LA eval (148,176 clips), EER 5.20%**. It runs on an A100 as a web app; on-device is future work.

### The five Special Technology prizes

**KawanIsyarat (Cactus).** Two-way BISINDO sign language for the author's Deaf brother on a Pixel 6a: a 1D CNN on MediaPipe landmarks reads 16 signs (86.2% leave-one-signer-out), and **Gemma 4 E2B INT4 via Cactus** handles audio, image and text in 1.7 to 1.9 GB. Its best gotcha: Cactus enabled thinking by default, so simple replies took 60 to 153 s; turning it off gave 4 to 9 s.

**Gilbeot (LiteRT).** Navigation for elderly Korean pedestrians by visible landmarks. An Unsloth rank-8 LoRA on the audio encoder and LM, trained on 45,823 elderly utterances, cut Korean **CER from 13.14% to 5.00% on 134 held-out utterances**. No tool could put an audio LoRA into an Android `.litertlm` bundle, so the author wrote a patcher using **same-grid requantisation**:

$$
q' = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{W + \Delta W}{s}\right),\ -8,\ 7\right)
$$

Keeping the original scale $s$ keeps the bundle a drop-in file. The toy shows the cost.

```python
# Illustrative toy of "same-grid requantisation" as Gilbeot's writeup describes it:
#   patched_int4 = clip(round((base_fp32 + lora_delta) / scale), -8, 7)
# Random matrices, NOT the team's code or weights.
import numpy as np

rng = np.random.default_rng(0)
d_out, d_in, r = 256, 256, 8
W = rng.normal(0, 0.02, (d_out, d_in)).astype(np.float32)

# Per-row symmetric int4 quantisation of the base weights (the "grid").
scale = np.abs(W).max(axis=1, keepdims=True) / 7.0
q_base = np.clip(np.round(W / scale), -8, 7)

# A rank-8 LoRA delta, B @ A, scaled by alpha / r.
A = rng.normal(0, 0.02, (r, d_in))
B = rng.normal(0, 0.02, (d_out, r))
alpha = 8
for mult in (1, 10, 50):
    delta = (alpha / r) * (B @ A) * mult
    q_new = np.clip(np.round((W + delta) / scale), -8, 7)   # same scale, same grid
    changed = (q_new != q_base).mean()
    # Does the change that lands on the grid point the same way as the LoRA delta?
    realised = (q_new - q_base) * scale
    cos = (realised * delta).sum() / (np.linalg.norm(realised) * np.linalg.norm(delta))
    step = (np.abs(delta) / scale).mean()      # delta size in grid steps
    print(f"delta x{mult:>2}: mean |delta| = {step:4.2f} grid steps, "
          f"codes changed {changed:5.1%}, cosine(realised, intended) = {cos:.2f}")
```

```console
delta x 1: mean |delta| = 0.10 grid steps, codes changed 10.4%, cosine(realised, intended) = 0.41
delta x10: mean |delta| = 1.02 grid steps, codes changed 68.9%, cosine(realised, intended) = 0.95
delta x50: mean |delta| = 5.10 grid steps, codes changed 93.1%, cosine(realised, intended) = 0.93
```

A delta a tenth of a grid step mostly rounds away, and what survives points only loosely the right way: one reason the riskier audio adapter shipped as a runtime sidecar. Gilbeot also found Cactus more accurate but about 3.4x slower than LiteRT, used [MTP speculative decoding](/blog/machine-learning/large-language-model/speculative-decoding) on GPU (decode 17.8 s to 11.5 s) but not on CPU, and had the VLM emit arrow coordinates so code decides left from right. [GGUF quant types](/blog/machine-learning/large-language-model/how-quantization-works-gguf-quant-types-decoded) explains the grids.

**PreVillage (llama.cpp).** WhatsApp voice help for rural Nepali families navigating government offices, built on resolver-first RAG over **272,718 chunks from 46,051 documents**. Its fine-tuning ladder is the most candid in the field: an E4B version had sharply lower validation loss but **invented a phone number, so it was not deployed**; the shipped design splits planner, answerability and composition. On a 48-case gold set: URL recall 0.94, wrong refusals 2 of 48. The llama.cpp lane runs quantised E2B on a Pi 5 at about 7.5 tok/s. "Evals became the product spec."

**PathOS (Ollama).** A first-pass histopathology screener on a CPU laptop. An Unsloth multimodal LoRA on E2B with vision layers tuned scored 26% overall, and **82% of its yes/no errors were hedges** inherited from RLHF. [GRPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-grpo) with four rewards, including a minus-one penalty on hedge language, fixed that ("SFT teaches what to say. GRPO teaches what not to say."). Then their self-consistency inference engine was distilled back into the weights: **88% yes/no on 100 in-distribution PathVQA samples** in one pass, shipped via `ollama pull`.

**MindBridge (Unsloth).** Hindi depression and anxiety screening for community health workers in India: Gemma 4 E2B scores each spoken PHQ-9 or GAD-7 answer directly from audio. 144 hand-authored seeds were expanded with Gemma 4 26B-A4B to **2,883 examples**, then a QLoRA (r=16, alpha=32, 2 epochs, about 30 minutes on an A100) with audio modules frozen. It is the only winner that **fixed its pass criteria before training**:

```python
# Illustrative toy of a pre-registered, hierarchical "kill gate" in the style
# MindBridge describes: thresholds fixed BEFORE training, checked in order,
# first failure stops the release. Inputs are the team-reported numbers on
# their 224-example held-out set; this is NOT the team's code.
GATES = [  # (name, metric, threshold, kind)
    ("Format",  "json_valid",       0.95, "abs"),    # JSON validity >= 95%
    ("Safety",  "item9_sensitivity", 0.90, "abs"),   # self-harm item recall >= 90%
    ("Utility", "likert_acc",       0.10, "delta"),  # >= +10 pp over base
]
base = {"json_valid": 1.000, "item9_sensitivity": 0.958, "likert_acc": 0.625}
tuned = {"json_valid": 1.000, "item9_sensitivity": 0.917, "likert_acc": 0.875}

for name, key, thr, kind in GATES:
    value = tuned[key] - base[key] if kind == "delta" else tuned[key]
    ok = value >= thr - 1e-9
    print(f"{name:<8} {key:<18} {value:.3f} vs {thr:.2f}  {'PASS' if ok else 'KILL'}")
    if not ok:
        break
# 22 of 24 adversarial items caught -> 0.917; base caught 23 of 24 -> 0.958
print(f"Item-9 cases: tuned {round(0.917*24)}/24, base {round(0.958*24)}/24")
```

```console
Format   json_valid         1.000 vs 0.95  PASS
Safety   item9_sensitivity  0.917 vs 0.90  PASS
Utility  likert_acc         0.250 vs 0.10  PASS
Item-9 cases: tuned 22/24, base 23/24
```

The fine-tune passed, with its one-case safety regression disclosed. Then deployment bit: E4B INT4 decoded at 0.1 tok/s on an iPhone, E2B INT8 reached 9.35 tok/s, and **the shipped iOS demo runs the base model**.

## 6. All 31 winners, side by side

![Feature matrix of all 31 winners grouped by hackathon: model, whether the LLM was fine-tuned, on-device, agentic, own dataset, evaluation level and video, with counts per column](/imgs/blogs/kaggle-gemma-impact-hackathons-winners-3.webp)

Read the matrix by column. Video is near-universal (30 of 31); on-device or offline holds for 23 of 31. Fine-tuning the LLM itself (13 of 31) clusters in MedGemma's main track and Gemma 4's special prizes. Agentic designs grow over time: 1 of 8 in Gemma 3n, 7 of 14 in Gemma 4 Good (the top-20 file counts 8; recounting its own cells gives 7). Evaluation is the weakest column: 11 of 31 compare against a baseline or public benchmark, and 8 report nothing beyond latency.

The podium is not where the evidence lives: 3 of the 12 main-track places have a baseline or benchmark (FieldScreen, Tracer, GEM-4), against 8 of the 19 unranked winners.

![Memory footprint of nine on-device winners in GB as the teams reported it, from Sixth Sense's unquantised 9.07 GB down to KawanIsyarat's 1.7 to 1.9 GB on a Pixel 6a](/imgs/blogs/kaggle-gemma-impact-hackathons-winners-6.webp)

"On-device" converged on one number: **about 2 to 3.5 GB** for a 4-bit 2B-to-4B model, whether a GGUF on an iPhone, a `.litertlm` bundle, or Ollama on a Jetson. The outlier is unquantised and on a 16 GB desktop GPU. Acuífero·Vigía and MindBridge both fell back to E2B after E4B failed on their devices. The [on-device speed post](/blog/machine-learning/edge-ai/making-on-device-llms-fast) covers the mechanics.

![Before-and-after pairs from six winners on their own test sets: CER 13.14 to 5.00%, WER 32.7 to 19.0%, Likert accuracy 62.5 to 87.5%, PathVQA 50 to 88%, TB accuracy 84 to 86%, valid JSON 50 to 100%, with test sizes from 10 to 224](/imgs/blogs/kaggle-gemma-impact-hackathons-winners-7.webp)

These are the strongest claims in the field, and the test sizes are the honest part: 10 to 224 examples, mostly in-distribution, none independently re-run. **No winner reports a field outcome.** Trido's six weeks of classroom use, Sixth Sense's community trial and Tracer's survey come closest, and all three are observation.

### Where the winners contradict each other

- **Let the model read, or not?** Gemma Vision moved OCR to ML Kit; Vite Vere trusts Gemma's image understanding; Gilbeot moved left-right decisions into code.
- **Fine-tune or retrieve?** Tracer shows base MedGemma at 50% valid structure; BridgeDx ships it unfine-tuned and relies on retrieval and citation checks. ClinicDx does both.
- **Context in the prompt?** Gem-Care's clinical context hurt until fine-tuned; Acuífero·Vigía reached 82% schema match from dialect exemplars alone.
- **Cloud or edge for agents?** Both MedGemma agentic winners are cloud systems with Gemini driving; EpiCast left the cloud over 3-to-5-minute cold starts.

## 7. What did not work

- **Gemma reading text in photos** hallucinated; **a live camera preview** cost 10x (Gemma Vision).
- **Calling the LLM every step** was "prohibitive" (LeRobot entry).
- **Clinical context in a vanilla prompt** raised WER from 32.7% to 39.5% (Gem-Care).
- **Continued pretraining on an instruct model** damaged chat; **a better validation loss** shipped a model that invented phone numbers (PreVillage).
- **SFT alone could not remove hedging**: 82% of yes/no errors were refusals (PathOS).
- **E4B on edge**: WebGPU buffer overflows on a Pi 5 (Acuífero·Vigía), 0.1 tok/s on an iPhone (MindBridge).
- **Default thinking mode** turned 4-second replies into 60 to 153 s (KawanIsyarat).
- **Smaller models for function calling** failed silently with wrong types (Trido).
- **An LLM-written compliance document** (Project Rosie).
- **A float32 batch into a bf16 audio tower** produced NaNs (TrueVoice).

## 8. Lessons that transfer

1. **Name one person and build for them.** Eva, a blind brother, Pak Damar: the Gemma podiums are built around a specific human, which 70 points of Impact and Video reward.
2. **Match the rubric's technical line exactly.** MedGemma's Feasibility criterion asked for fine-tuning documentation, and all four places LoRA-tuned MedGemma and wrote down rank, epochs and hardware.
3. **Keep safety in deterministic code.** DEMENTOR, Rosie, Acuífero·Vigía, CodeBuddy, Gilbeot and MindBridge each put the decision that must not be wrong outside the model.
4. **Measure the device, not the GPU.** Credible on-device entries report tok/s, RAM and watts on target hardware.
5. **Ship evidence anyway, and write down what failed.** Evidence did not order the podium, but it makes a writeup reusable.

### If you enter a hackathon like this

- Read the rubric weights first and spend your time in the same ratio.
- Put the user in the first 30 seconds of the video; judges may watch only three minutes.
- Use the model's headline feature on purpose. No Gemma 3n winner used native audio.
- Pick E2B-class at 4 bits unless you have measured something bigger on the target device.
- Fix a held-out set and a pass criterion before training, even if it is 100 examples.
- Report the baseline, the test size and what you have not measured.

Sibling hackathons: [OpenAI to Z](/blog/machine-learning/learn-from-kaggle/kaggle-openai-to-z-challenge-winners), [Tunix Hack](/blog/machine-learning/learn-from-kaggle/kaggle-tunix-hack-reasoning-winners), [2024 Gemma and Gemini](/blog/machine-learning/learn-from-kaggle/kaggle-gemma-gemini-early-hackathons-winners), [BigQuery AI](/blog/machine-learning/learn-from-kaggle/kaggle-bigquery-ai-hackathon-winners), [Measuring AGI](/blog/machine-learning/learn-from-kaggle/kaggle-measuring-agi-cognitive-abilities-winners). Background: [LoRA and QLoRA](/blog/machine-learning/reinforcement-learning/lora-qlora-for-rlhf), [GGUF locally](/blog/machine-learning/inference-frameworks/running-llms-locally-llama-cpp-and-gguf).

## 9. Sources

Gemma 3n Impact Challenge:
- Winners: https://www.kaggle.com/competitions/google-gemma-3n-hackathon/discussion/657756
- Gemma Vision: https://www.kaggle.com/competitions/google-gemma-3n-hackathon/writeups/gemma-vision
- Vite Vere Offline: https://www.kaggle.com/competitions/google-gemma-3n-hackathon/writeups/vite-vere-offline
- 3VA: https://www.kaggle.com/competitions/google-gemma-3n-hackathon/writeups/3va
- Sixth Sense: https://www.kaggle.com/competitions/google-gemma-3n-hackathon/writeups/sixth-sense-for-security-guards-powered-by-googles
- Dream's Assistant: https://www.kaggle.com/competitions/google-gemma-3n-hackathon/writeups/amoritas-assistant-ai-assistant-for-speech-impaire and https://github.com/Bra-AM/dream-assistant-android
- LENTERA: https://www.kaggle.com/competitions/google-gemma-3n-hackathon/writeups/lentera-offline-ai-microserver-for-remote-and-rura
- My (Jetson) Gemma: https://www.kaggle.com/competitions/google-gemma-3n-hackathon/writeups/my-jetson-gemma
- Graph-Based Cost Learning: https://www.kaggle.com/competitions/google-gemma-3n-hackathon/writeups/graphbased-cost-learning-and-gemma-3n-for-sensing and https://github.com/Raahim-Lone/gemma

MedGemma Impact Challenge:
- Winners: https://www.kaggle.com/competitions/med-gemma-impact-challenge/discussion/685002
- EpiCast: https://www.kaggle.com/competitions/med-gemma-impact-challenge/writeups/new-writeup-1770830823676
- Sunny: https://www.kaggle.com/competitions/med-gemma-impact-challenge/writeups/sunny-private-skin-health-tracker
- FieldScreen AI: https://www.kaggle.com/competitions/med-gemma-impact-challenge/writeups/fieldscreen-ai
- Tracer: https://www.kaggle.com/competitions/med-gemma-impact-challenge/writeups/new-writeup-1771923230850
- ClinicDx: https://www.kaggle.com/competitions/med-gemma-impact-challenge/writeups/new-writeup-1771522062957
- UniRad3s: https://www.kaggle.com/competitions/med-gemma-impact-challenge/writeups/unirad3s-reimagining-radiology-with-spot-segment
- CaseTwin: https://www.kaggle.com/competitions/med-gemma-impact-challenge/writeups/new-writeup-1771703368723
- BigTB6: https://www.kaggle.com/competitions/med-gemma-impact-challenge/writeups/bigtb6 and https://github.com/Sidharth1743/BigTb6
- BridgeDx: https://www.kaggle.com/competitions/med-gemma-impact-challenge/writeups/new-writeup-1771945306600

Gemma 4 Good Hackathon:
- Winners: https://www.kaggle.com/competitions/gemma-4-good-hackathon/discussion/736681
- GEM-4: https://www.kaggle.com/competitions/gemma-4-good-hackathon/writeups/new-writeup-1778618527713 and https://github.com/takaki-maeda-99/GEM-4
- Trido: https://www.kaggle.com/competitions/gemma-4-good-hackathon/writeups/trido
- PenguinAgent: https://www.kaggle.com/competitions/gemma-4-good-hackathon/writeups/PenguinAgent
- DEMENTOR: https://www.kaggle.com/competitions/gemma-4-good-hackathon/writeups/dementor-edge-ai-triage-for-dementia-care
- Project Rosie: https://www.kaggle.com/competitions/gemma-4-good-hackathon/writeups/new-writeup-1777596440505
- Acuífero·Vigía: https://www.kaggle.com/competitions/gemma-4-good-hackathon/writeups/acuifero4vigia
- CodeBuddy: https://www.kaggle.com/competitions/gemma-4-good-hackathon/writeups/new-writeup-1778344276798
- Gem-Care: https://www.kaggle.com/competitions/gemma-4-good-hackathon/writeups/gem-care-gemma-4-good-hackathon and https://github.com/BBounajma/Gemma4-hackathon
- TrueVoice: https://www.kaggle.com/competitions/gemma-4-good-hackathon/writeups/new-writeup-1776369081947
- KawanIsyarat: https://www.kaggle.com/competitions/gemma-4-good-hackathon/writeups/kawanisyarat-offline-gemma-4-for-bisindo
- Gilbeot: https://www.kaggle.com/competitions/gemma-4-good-hackathon/writeups/gilbeot and https://github.com/psymon-ai/gilbeot-public
- PreVillage: https://www.kaggle.com/competitions/gemma-4-good-hackathon/writeups/previllage-a-govspeak-platform
- PathOS: https://www.kaggle.com/competitions/gemma-4-good-hackathon/writeups/pathos-getalife
- MindBridge: https://www.kaggle.com/competitions/gemma-4-good-hackathon/writeups/new-writeup-1779141852948 and https://github.com/HP-00/MindBridge-Gemma-4
