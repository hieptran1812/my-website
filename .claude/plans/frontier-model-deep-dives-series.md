# Series: Frontier Model Deep-Dives (gap-fill for the family timelines)

**Folder:** `content/blog/machine-learning/frontier-models/` (subcategory "Frontier Models")
**Why:** the 14 `*-model-family-timeline` posts catalog every release, but many major releases have no deep-dive. Gap analysis run 2026-10-23 against all 2,900 post titles. Kimi is fully covered; Llama, Gemma, gpt-oss, INTELLECT, Hermes have zero model deep-dives.
**Size:** 12 new family timelines (Waves T, T2) + 37 deep-dives (Waves 1–6). Open-weight families only. Status: PLANNED (Wave 0 Fish Audio timeline shipped 2026-10-23; Wave T next).

## Per-post contract

- **Skill:** `paper-writer` when the release has an arXiv paper / technical report PDF, else `blog-writer`. Column `Skill` below: P / B.
- **Length:** house rule 2,000–3,000 prose words, ≥½ pictorial figures, no em dashes.
- **Frontmatter:** `category: "machine-learning"`, `subcategory: "Frontier Models"`, `author: "Hiep Tran"`. Note: paper-writer defaults to `paper-reading`; override to `machine-learning` + this folder (same as the 2026-10-14 consolidation).
- **Sources:** start from the links already in the family timeline's section for that release (`grep -n` the heading, read only that section with `sed -n`). Every number from a fetched primary source.
- **Before the figure pipeline:** `ls` the slug, and grep titles corpus-wide; a same-slug re-render clobbers committed webps.
- **Angle:** what this release changed versus its predecessor and why (architecture, data, training recipe, post-training, serving cost). Not a re-catalog. Link back to the family timeline in the intro.

## THE LINK-BACK RULE (mandatory, part of every post's definition of done)

After a post passes `post-verifier`, edit its family timeline:
1. Find the release's section (`grep -n '^##\|^###' <family>-model-family-timeline.md`).
2. Append one sentence at the end of that section's first prose paragraph, in the timeline's existing style, e.g.
   `The full technique breakdown is in [<Post title>](/blog/machine-learning/frontier-models/<slug>).`
   Do not add it to the glance table or the references list only; it must sit in the release's own section.
3. If the post spans several releases (e.g. Gemma 3 + 3n), link it from each section.
4. Commit the timeline `.md` in the same wave commit as the post (explicit paths).
5. Wave check: `grep -c <slug> <timeline>` ≥ 1 for every post in the wave before committing.

Timeline file per family: `deepseek-`, `qwen-`, `glm-`, `minimax-`, `nemotron-`, `bytedance-seed-`, `llama-`, `gemma-`, `olmo-`, `smollm-`, `nous-hermes-`, `prime-intellect-intellect-model-family-timeline.md`, and `openai-gpt-oss-open-weight-models-timeline.md`, `fish-audio-model-family-timeline.md`.

## Wave 0 (in progress)

| # | Slug | Family | Skill | Status |
|---|---|---|---|---|
| 0 | `fish-audio-model-family-timeline` | Fish Audio | B | SHIPPED 2026-10-23 (4,486 words, 5 figs; latest = Fish Audio Drama 3, 2026-10-13); links existing `paper-reading/speech-processing/fish-audio-s2-controllable-text-to-speech` |

Backfill done 2026-10-23: deepseek timeline now links Janus/Janus-Pro and DeepInfer; all other existing deep-dives were already linked.

## Scope rule: open-weight families only (user decision 2026-10-23)

Timelines and deep-dives are only for families that release weights. No GPT/o-series, Gemini, Claude, Grok timelines. For mixed labs (FLUX, Stability, Hunyuan, Mistral) catalog the open-weight lines; mention hosted-only siblings in one line each, as the Fish Audio timeline does.

## Wave T: new family timelines (run BEFORE Wave 1, so deep-dives have a hub to link into)

Same format as the existing `*-model-family-timeline` posts (glance table, one section per release, variant tables, license, primary-source link, "which to pick today"). blog-writer, WORD_MIN=2500 WORD_MAX=4500. After each ships: link any existing deep-dives of that family from the right section, and add the new timeline to the list of timeline files in the link-back rule.

| # | Slug | Family | Corpus mentions (2026-10-23) |
|---|---|---|---|
| T1 | `mistral-model-family-timeline` | Mistral 7B, Mixtral, Codestral, Pixtral, Mistral Small/Medium open releases, Magistral, Devstral, Voxtral | 108 |
| T2 | `stability-ai-model-family-timeline` | Stable Diffusion 1.x → SDXL → SD3/3.5, SVD, Stable Audio Open | 78 |
| T3 | `whisper-model-family-timeline` | Whisper v1 → large-v3 → turbo, distil-whisper | 52 |
| T4 | `flux-model-family-timeline` | Black Forest Labs FLUX.1 dev/schnell, Kontext, FLUX.2 open lines | 46 |
| T5 | `hunyuan-model-family-timeline` | Tencent Hunyuan LLM (Large, A13B, Turbo-S open), Image, Video, 3D, OCR | 44 |
| T6 | `meta-fair-vision-audio-model-family-timeline` | DINO/DINOv2/v3, SAM/SAM 2/3, AudioCraft/MusicGen, Seamless | 32 + 27 |

## Wave T2: second-tier open families

| # | Slug | Family | Mentions |
|---|---|---|---|
| T7 | `cohere-command-model-family-timeline` | Command R/R+, Aya, Command A (open weights, CC-BY-NC) | 28 |
| T8 | `wan-model-family-timeline` | Alibaba Wan 2.1 → 2.2 → later video releases | 22 |
| T9 | `falcon-model-family-timeline` | TII Falcon 1 → 2 → 3 → Falcon-H1, Mamba | 21 |
| T10 | `microsoft-phi-model-family-timeline` | Phi-1 → Phi-4 (mini/multimodal/reasoning), VibeVoice | 17 |
| T11 | `funaudiollm-model-family-timeline` | CosyVoice 1/2/3, SenseVoice, Fun-ASR | 13 |
| T12 | `internlm-model-family-timeline` | InternLM 1–3, InternVL 1–3.5, Intern-S1 | 12 |

Candidates for a later T3 wave (few mentions, rising open labs): Kyutai (Moshi/Unmute), AI21 Jamba, IBM Granite, StepFun, Xiaomi MiMo, Meituan LongCat, Ant Ling/Ring, MiniCPM, Baidu ERNIE 4.5, Liquid LFM.

## Wave 1: the open-weight pillars with zero coverage

| # | Slug | Family / timeline section | Skill |
|---|---|---|---|
| 1 | `gpt-oss-120b-alternating-attention-mxfp4` | gpt-oss / "gpt-oss-120b and gpt-oss-20b" | P (model card arXiv 2508.10925) |
| 2 | `llama-3-herd-of-models-405b-recipe` | Llama / "Llama 3" + "Llama 3.1" | P (arXiv 2407.21783) |
| 3 | `llama-4-moe-native-multimodality` | Llama / "Llama 4" | B |
| 4 | `gemma-3-local-global-attention-multimodal` | Gemma / "Gemma 3" | P (arXiv 2503.19786) |
| 5 | `gemma-4-thinking-moe-apache` | Gemma / "Gemma 4" | B (P if report exists) |
| 6 | `deepseek-v2-mla-deepseekmoe-economics` | DeepSeek / "Generation 2" | P (arXiv 2405.04434) |

## Wave 2: Qwen + DeepSeek gaps

| # | Slug | Family / section | Skill |
|---|---|---|---|
| 7 | `deepseek-v3-2-sparse-attention-dsa` | DeepSeek / "Generation 5" | P |
| 8 | `deepseek-v4-1-flash-encoder-decoder-moe` | DeepSeek / "Generation 7" | P (tech report PDF on HF) |
| 9 | `qwen2-5-technical-report-18t-tokens` | Qwen / "Qwen2.5" (cover Qwen2 lineage briefly) | P (arXiv 2412.15115) |
| 10 | `qwq-32b-reasoning-rl` | Qwen / "Qwen2.5" reasoning | B |
| 11 | `qwen3-vl-technical-report` | Qwen / "Qwen3 extensions" | P |
| 12 | `qwen3-5-hybrid-mainline` | Qwen / "Qwen3.5" | B/P |

## Wave 3: GLM + MiniMax

| # | Slug | Family / section | Skill |
|---|---|---|---|
| 13 | `cogvlm-cogagent-visual-expert` | GLM / "CogVLM and CogAgent" | P (arXiv 2311.03079) |
| 14 | `glm-4-6-glm-4-7-coding-point-releases` | GLM / sections 11 + 12 | B |
| 15 | `glm-5-sparse-attention-double-size` | GLM / "GLM-5" | B/P |
| 16 | `minimax-m2-5-rl-at-scale` | MiniMax / "M2.5" (+ M2.1 context) | B |
| 17 | `minimax-m3-sparse-attention-multimodal` | MiniMax / "M3" | B/P |
| 18 | `minimax-m2-7-self-evolution` | MiniMax / "M2.7" | B |

## Wave 4: Nemotron + Seed

| # | Slug | Family / section | Skill |
|---|---|---|---|
| 19 | `nemotron-nano-2-hybrid-reasoning` | Nemotron / "Nemotron Nano 2" | P (arXiv 2508.14444) |
| 20 | `nemotron-3-nano-super-ultra-hybrid-moe` | Nemotron / sections 10, 11, 13 | P/B |
| 21 | `hymba-hybrid-head-small-lm` | Nemotron / "late-2024 side releases" | P (arXiv 2411.13676) |
| 22 | `seed1-6-adaptive-thinking` | Seed / "Seed1.6" | B |
| 23 | `seed2-0-generalized-agent` | Seed / "Seed1.8" + "Seed2.0" | B/P |
| 24 | `seed-tts-and-seed-music` | Seed / "media lines" | P (Seed-TTS arXiv 2406.02430) |

## Wave 5: the fully-open labs (OLMo, SmolLM, INTELLECT)

| # | Slug | Family / section | Skill |
|---|---|---|---|
| 25 | `olmoe-open-mixture-of-experts` | OLMo / "OLMoE" | P (arXiv 2409.02060) |
| 26 | `tulu-3-open-post-training-rlvr` | OLMo / "Tülu 3" | P (arXiv 2411.15124) |
| 27 | `olmo-2-stability-and-mid-training` | OLMo / "OLMo 2" | P (arXiv 2501.00656) |
| 28 | `smollm3-3b-long-context-dual-mode` | SmolLM / "SmolLM3" (link `smol-training-playbook`) | B |
| 29 | `smolvlm-tiny-vision-language` | SmolLM / "SmolVLM" family | P |
| 30 | `intellect-3-large-scale-rl-open-moe` | INTELLECT / "INTELLECT-3" (+ 1/2 context) | P |

## Wave 6: Llama/Gemma long tail + Hermes + Fish Audio

| # | Slug | Family / section | Skill |
|---|---|---|---|
| 31 | `llama-2-rlhf-ghost-attention` | Llama / "Llama 2" | P (arXiv 2307.09288) |
| 32 | `llama-guard-safety-classifier-lineage` | Llama / "Llama Guard" | P (arXiv 2312.06674) |
| 33 | `gemma-2-distillation-local-global` | Gemma / "Gemma 2" | P (arXiv 2408.00118) |
| 34 | `gemma-3n-matformer-per-layer-embeddings` | Gemma / "Gemma 3n" | B |
| 35 | `hermes-4-hybrid-reasoning-open-post-training` | Hermes / "Hermes 4" (+ Hermes 3) | P (arXiv 2508.18255) |
| 36 | `fish-speech-vqgan-dual-ar-lineage` | Fish Audio / Fish Speech 1.x + OpenAudio S1 | P (arXiv 2411.01156) |
| 37 | `fish-audio-drama-3-moe-tts` | Fish Audio / "Drama 3" (35B-A3B MoE, license mismatch at source) | P (arXiv 2610.18010) |

## Wave procedure (CLAUDE.md token discipline)

One fresh session per wave; carry forward only this file's path + wave number. 6 drafting agents, each its own scratch subfolder, figure-author + post-verifier inside the drafter. Orchestrator then: verify on disk (`ls` md + webp count), apply the link-back rule (or check the drafter did), `grep -c` check, commit only that wave's `.md` + `.webp` + edited timeline files, `git pull --rebase --autostash`, push. Update this file's status line + per-wave status.
