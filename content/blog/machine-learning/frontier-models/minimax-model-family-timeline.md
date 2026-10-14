---
title: "The MiniMax Model Family, Release by Release: From abab6 to MiniMax-M3 and H3"
date: "2026-10-14"
publishDate: "2026-10-14"
category: "machine-learning"
subcategory: "Frontier Models"
tags:
  - minimax
  - model-timeline
  - mixture-of-experts
  - lightning-attention
  - sparse-attention
  - agentic-llm
  - text-to-speech
  - video-generation
  - open-weights
  - reference
description: "A chronological reference to every MiniMax release from the closed abab chat models to MiniMax-M3 and the open H3 video model: dates, variants, what changed each generation, licenses, and which one to pick today."
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/minimax-model-family-timeline-1.webp"
readTime: 33
---

MiniMax is the lab that changed its mind in public. In January 2025 it shipped a 456B-parameter model built on linear "lightning" attention and told everyone the quadratic wall was finally beaten. Ten months later it shipped MiniMax-M2 with full softmax attention on every layer and wrote a blog post explaining why. Seven months after that it shipped MiniMax-M3 with a third answer, a learned sparse attention. Few families give you a cleaner view of how a frontier lab learns, because MiniMax published the reasoning for each turn alongside the weights.

This post is a reference catalog, not a deep-dive. It walks every MiniMax generation in the order it shipped, from the API-only abab chat models of 2024 through the open MiniMax-01, M1, the M2 line, M3 and the H3 video model of July 2026. For each one you get the release date, what changed versus the previous generation, a variant table, and a primary-source link. Where this blog already has a deep read of a release, it is linked inline, so you can treat this page as the index to those posts.

The figure below is the map for the whole post. MiniMax ships on four tracks at once (text, speech, video, music), and the story of the text track is the story of a closed chat-model vendor becoming one of the most aggressive open-weight labs in the world.

![MiniMax release timeline from 2024 to 2026 across text, speech, video and music tracks, with open-weight releases highlighted](/imgs/blogs/minimax-model-family-timeline-1.webp)

Read it lane by lane. The top lane is the language models, and the color change from gray to blue around January 2025 is the single most important event in the family: that is MiniMax-01, the moment weights started shipping. The speech and video lanes stay API-only for almost the whole window, until H3 in July 2026 opens the video lane too. Notice also the cluster in late October 2025, when M2, Hailuo 2.3, Speech 2.6 and Music 2.0 all shipped within four days. MiniMax releases in waves, and a wave usually means a new flagship plus refreshed media models riding the same infrastructure push.

## How to read this catalog

A few conventions hold throughout, because MiniMax's naming and dating are less tidy than the ones you may know from Qwen or Llama.

**Dates.** Where MiniMax's own news post or platform release notes give a date, I use that. Hugging Face repository creation dates are often a few days earlier than the announcement (repos get created before they go public), so when the two disagree I give the announcement date and mention the repo date. For the 2024 abab models MiniMax has no surviving English announcement page I could locate, so those dates come from contemporaneous press reports and are marked as such.

**Parameters.** "Total / active" is the MoE convention: total parameters stored, active parameters used per token. Numbers come from model cards, and I cross-checked them against the `config.json` and safetensors metadata on Hugging Face where weights exist.

**Context.** I give the number the model card states, and where the config's `max_position_embeddings` differs, I note it. MiniMax quotes context in rounded marketing figures ("196K", "1M"), and the config value is the one your serving stack actually sees.

**License.** This matters more for MiniMax than for most families, because the license has moved in both directions. MiniMax-01 shipped under a custom model agreement, M1 under Apache 2.0, the M2 line under a modified MIT, M2.7 under a non-commercial license, and M3 under a community license. Do not assume the license of one generation carries to the next.

**What is excluded.** I do not include anything I could not tie to a MiniMax source, and I do not extrapolate version numbers. Where a release is reported only by the press, it is listed in a separate box at the end of its section and flagged.

## Before the open era: the abab chat models (2023 to 2024)

MiniMax was founded in Shanghai in 2021, and for its first three years it was a consumer AI company first and a model lab second. Its chat apps (Glow, then Talkie for international markets and Xingye in China) ran on an in-house model line called **abab**. These models were never released as weights; they were served through the MiniMax Open Platform API.

### abab5.5 (2023)

The abab5 and abab5.5 models were the 2023 production chat models behind the consumer apps and the first developer API. I could not find a primary MiniMax source that gives a release date or specifications for them, and the secondary listings disagree, so I am not giving numbers here. Treat abab5.5 as "the pre-MoE chat model MiniMax served in 2023" and nothing more precise.

### abab6 (January 2024): the first MoE

**Release:** full launch reported on January 16, 2024, after roughly two weeks of testing with selected customers (press reports; no primary MiniMax page located).

**What changed:** abab6 moved the line to a Mixture-of-Experts architecture, and Chinese press widely reported it as the first MoE large language model released in China. That is the first appearance of the design choice that every later MiniMax language model keeps: sparse MoE feed-forward layers, with the open question each generation being what to do about attention.

| Variant | Architecture | Context | Modality | Weights |
| --- | --- | --- | --- | --- |
| abab6 | MoE (size not disclosed) | not verified | text chat | API only |

### abab6.5 and abab6.5s (April 2024)

**Release:** reported on April 17, 2024 (press reports).

**What changed:** a much larger MoE (reported as roughly a trillion total parameters) and a long-context push to about 200k tokens, plus a cheaper sibling, abab6.5s, trained with the same data and techniques but tuned for speed. The press figure that circulated was that abab6.5s could process about 30,000 Chinese characters in a second.

| Variant | Architecture | Context | Modality | Weights |
| --- | --- | --- | --- | --- |
| abab6.5 | MoE, about 1T total (press-reported) | about 200k (press-reported) | text chat | API only |
| abab6.5s | MoE, efficiency variant | about 200k at launch (press-reported) | text chat | API only |

Two warnings about this generation. First, the specs are secondhand: MiniMax published no technical report, and later third-party API listings give abab6.5s a smaller context window than the launch figure. Second, abab6.5 is the last MiniMax language model whose architecture we know almost nothing about. Everything after it ships with weights and a paper.

## Video-01 and the Hailuo brand (August 2024)

**Release:** August 31, 2024, per MiniMax's news post ([MiniMax officially releases the Video-01 video generation model](https://www.minimax.io/news/video-01)).

**What changed:** this was MiniMax's first generative video model and the start of the **Hailuo** brand, which became the name of MiniMax's consumer video app and, later, of the model line itself. At launch, video-01 was text-to-video only and generated clips up to 6 seconds, with MiniMax promising 10-second clips in the next major version. Press coverage at the time described the output as 720p. The model went viral precisely because it was free and web-based while competitors were waitlisted.

| Variant | Task | Max duration | Resolution | Weights |
| --- | --- | --- | --- | --- |
| video-01 | text-to-video | 6 s | 720p (press-reported) | API and app only |
| I2V-01 (later in 2024) | image-to-video | 6 s | not verified | API and app only |

Video-01 matters to the language-model story for a structural reason: from here on, MiniMax runs a video track with its own release rhythm, and the late-2025 wave shows the tracks being shipped together.

## MiniMax-01 (January 2025): lightning attention and open weights

**Release:** January 15, 2025 ([MiniMax-01 is Now Open-Source](https://www.minimax.io/news/minimax-01-series-2)). Technical report: [MiniMax-01: Scaling Foundation Models with Lightning Attention](https://arxiv.org/abs/2501.08313), arXiv 2501.08313, dated January 14, 2025. Weights: [MiniMaxAI/MiniMax-Text-01](https://huggingface.co/MiniMaxAI/MiniMax-Text-01) and [MiniMaxAI/MiniMax-VL-01](https://huggingface.co/MiniMaxAI/MiniMax-VL-01).

**What changed versus abab6.5:** almost everything that can change.

1. **Open weights.** For the first time, MiniMax published the full weights, under the MiniMax Model Agreement for the weights and MIT for the code. The news post gives two reasons: to accelerate long-context research and to make MiniMax visible to developers outside its own platform.
2. **A hybrid attention stack.** This is the headline. MiniMax-01 replaces most softmax attention layers with **lightning attention**, an I/O-aware implementation of linear attention whose cost grows linearly with sequence length instead of quadratically. The layout is a fixed rhythm: seven lightning-attention layers, then one ordinary softmax attention layer, repeated through the 80-layer stack. The softmax layer is there because pure linear attention is weak at precise retrieval: it compresses the past into a fixed-size state, and a single softmax layer every eight restores exact lookups.
3. **A fat-expert MoE.** 32 experts with top-2 routing, so each expert is large and few fire per token. Total 456B, active 45.9B per token.
4. **Long context as the product.** One million tokens of training context and up to four million at inference, which was far beyond any open model at the time.
5. **A vision sibling.** MiniMax-VL-01 attaches a vision encoder to the same text backbone for image understanding.

The full mechanism (why 7:1, how the LASP+ sequence parallelism trains it, and what the recall tax looks like) is in the [MiniMax-01 deep-dive](/blog/machine-learning/frontier-models/minimax-01-lightning-attention-hybrid-moe), and the three MiniMax LLM papers are read side by side in [Reading the MiniMax Papers](/blog/machine-learning/frontier-models/minimax-papers-lightning-attention-cispo).

| Variant | Total / active | Layers, experts | Context | Modality | Weights |
| --- | --- | --- | --- | --- | --- |
| MiniMax-Text-01 | 456B / 45.9B | 80 layers, 32 experts, top-2 | 1M train, up to 4M inference | text | open (MiniMax Model Agreement) |
| MiniMax-VL-01 | 456B text backbone plus vision encoder | same backbone | long context | image and text in, text out | open (MiniMax Model Agreement) |
| MiniMax-Text-01-hf (June 2025) | same | same | same | text | open, Transformers-native port |

The `config.json` for Text-01 lists `max_position_embeddings` of 10,240,000, which is a ceiling for position handling, not a claim of tested quality at ten million tokens. Use the four-million figure from the card as the real upper bound, and expect quality to degrade well before it.

## The 2025 media wave: Director models, Image-01 and Speech-02

Between MiniMax-01 and M1, the media tracks moved fast. None of these ship weights, but they shape what MiniMax is as a company, and one of them has a paper.

**T2V-01-Director and I2V-01-Director (February 11, 2025).** Video models trained for camera-movement instructions and cinematic shot language, per MiniMax's [model release notes](https://platform.minimax.io/docs/release-notes/models). The point was control: you could ask for a dolly-in or a pan and get it.

**Image-01 (April 2, 2025).** MiniMax's first text-to-image model, generating in multiple aspect ratios (release notes).

**Speech-02 (April 2, 2025).** Announced in [the Speech-02 series post](https://www.minimax.io/news/speech-02-series) as the successor to Speech-01, in two tiers: Speech-02-HD for high-fidelity work like voiceovers and audiobooks, and Speech-02-Turbo for low latency. The launch post claims 30+ languages and zero-shot voice cloning from about 10 seconds of audio. On May 12, 2025, MiniMax published the research behind it: [MiniMax-Speech: Intrinsic Zero-Shot Text-to-Speech with a Learnable Speaker Encoder](https://arxiv.org/abs/2505.07916), arXiv 2505.07916. Its central idea is that the speaker encoder is trained jointly with the autoregressive TTS model instead of being a frozen speaker-verification network, and the acoustic side uses a Flow-VAE. The [MiniMax-Speech deep-dive](/blog/machine-learning/frontier-models/minimax-speech-speaker-encoder-flow-vae) covers both.

| Release | Date | Track | Key specs | Weights |
| --- | --- | --- | --- | --- |
| T2V-01-Director, I2V-01-Director | Feb 11, 2025 | video | camera-movement control | API only |
| Image-01 | Apr 2, 2025 | image | text-to-image, multiple sizes | API only |
| Speech-02-HD | Apr 2, 2025 | speech | high fidelity, 30+ languages (launch claim) | API only |
| Speech-02-Turbo | Apr 2, 2025 | speech | low latency tier | API only |

**SynLogic (May 2025).** A side release worth knowing about because it feeds M1: [SynLogic](https://arxiv.org/abs/2505.19641) (arXiv 2505.19641, May 26, 2025) is a framework for synthesizing verifiable logical-reasoning data for RL, released with [SynLogic-7B and SynLogic-32B](https://huggingface.co/MiniMaxAI/SynLogic-32B) checkpoints under MIT. These are research models trained on a Qwen2.5 base, not MiniMax architectures, and I list them only so you recognize them in the Hugging Face org.

## MiniMax-M1 (June 2025): the reasoning model

**Release:** June 16, 2025 ([MiniMax-M1 announcement](https://www.minimax.io/news/minimaxm1)). Technical report: [MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585), arXiv 2506.13585. Weights: [MiniMax-M1-40k](https://huggingface.co/MiniMaxAI/MiniMax-M1-40k) and [MiniMax-M1-80k](https://huggingface.co/MiniMaxAI/MiniMax-M1-80k).

**What changed versus MiniMax-01:** the architecture stayed, the purpose changed. M1 is MiniMax-01's backbone (same 456B / 45.9B, same 7:1 lightning hybrid) turned into a long-chain reasoning model by continued pretraining and large-scale reinforcement learning. MiniMax marketed it as the first open-weight, large-scale hybrid-attention reasoning model.

The argument for the hybrid is strongest exactly here. Reasoning models spend most of their compute generating long chains of thought, and in a full-attention model each new token costs more than the last. The M1 report says that at a 100k-token generation length, M1 uses about 25% of the FLOPs of DeepSeek-R1. Linear attention turns the cost of thinking longer from quadratic into roughly linear.

M1 also introduced **CISPO**, the RL objective that later generations keep. Instead of clipping the token-level policy update the way PPO and GRPO do (which silently drops gradient from rare, high-ratio tokens such as "wait" or "recheck"), CISPO clips the importance-sampling weight and keeps every token's gradient. The report gives a rare concrete compute figure: full RL training on 512 H800 GPUs for three weeks at a rental cost of $534,700. The [M1 deep-dive](/blog/machine-learning/frontier-models/minimax-m1-cispo-test-time-compute) works through CISPO and the economics.

| Variant | Total / active | Thinking budget | Input context | Modality | Weights |
| --- | --- | --- | --- | --- | --- |
| MiniMax-M1-40k | 456B / 45.9B | 40k tokens | 1M | text | open, Apache 2.0 |
| MiniMax-M1-80k | 456B / 45.9B | 80k tokens | 1M | text | open, Apache 2.0 |
| M1-40k-hf, M1-80k-hf (July 2025) | same | same | same | text | open, Transformers-native ports |

Note the license: Apache 2.0, the most permissive license MiniMax has ever used for a flagship. It did not last.

## Hailuo 02 (June 2025)

**Release:** June 18, 2025 ([MiniMax Hailuo 02 announcement](https://www.minimax.io/news/minimax-hailuo-02)).

**What changed versus video-01:** native 1080p and 10-second clips, the upgrade video-01's announcement had promised. MiniMax credits a new architecture it calls **Noise-aware Compute Redistribution (NCR)**, which it says improves training and inference efficiency by 2.5x at comparable scale. That headroom went into a model with three times the parameters, trained on four times the data, with better adherence to complex instructions and physics. Parameter counts are not disclosed.

| Variant | Resolution | Duration | Weights |
| --- | --- | --- | --- |
| Hailuo 02, 768p-6s | 768p | 6 s | API and app only |
| Hailuo 02, 768p-10s | 768p | 10 s | API and app only |
| Hailuo 02, 1080p-6s | 1080p | 6 s | API and app only |

## Speech 2.5 and Music 1.5 (summer 2025)

Two smaller media releases fill the gap before M2.

**Music-1.5.** A beta on June 20, 2025 that generated music from an inspiration audio clip plus lyrics, then general release on September 11, 2025 with tracks up to four minutes (release notes).

**Speech 2.5 (August 7, 2025).** Per [the Speech 2.5 post](https://www.minimax.io/news/minimax-speech-25), language support expanded to over 40 languages, with cross-lingual cloning that preserves a speaker's accent when switching languages. The release notes date it August 6.

| Release | Date | Key change | Weights |
| --- | --- | --- | --- |
| Music-1.5 (beta) | Jun 20, 2025 | audio-inspired music plus lyrics | API only |
| Speech 2.5 | Aug 7, 2025 | 40+ languages, cross-lingual cloning | API only |
| Music-1.5 (GA) | Sep 11, 2025 | tracks up to 4 minutes | API only |

## MiniMax-M2 (October 2025): back to full attention

**Release:** October 27, 2025 ([MiniMax M2 announcement](https://www.minimax.io/news/minimax-m2)). Weights: [MiniMaxAI/MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2) (repo created October 22). No arXiv report; the technical reasoning lives in MiniMax's blog posts and the model card.

**What changed versus M1:** this is the reversal, and it changed three things at once.

1. **Full softmax attention on every layer.** The lightning hybrid is gone. MiniMax's explanation, covered at length in the [MiniMax-M2 deep-dive](/blog/machine-learning/frontier-models/minimax-m2-full-attention-agentic), is that at larger scale the hybrid showed clear deficits on complex multi-hop reasoning, that the retrieval patterns softmax attention learns early in pretraining cannot be recovered later on a cheaper architecture. Agent workloads (long tool traces, repeated retrieval of exact earlier content) are exactly where the recall tax hurts.
2. **A much smaller active footprint.** M2 is 230B total with 10B active, down from 45.9B active. The MoE went from 32 large experts with top-2 routing to 256 small experts with top-8 routing across 62 layers. Fewer active parameters means lower latency and lower cost per token, which is what the agent use case needs.
3. **A product thesis: models for agents and code.** The launch post frames M2 as solving an "impossible triangle" of performance, price and speed for MiniMax's own internal agents, priced at 8% of Claude Sonnet's price at about twice the speed, served at roughly 100 tokens per second, with free API access until November 7, 2025.

M2 is also an **interleaved-thinking** model: it emits reasoning between tool calls inside `<think>...</think>` blocks, and the model card marks it IMPORTANT that clients retain that thinking content in the historical messages. Strip them, as many OpenAI-compatible clients do by default, and the model loses its own plan between turns. That serving contract is the single most common way M2-line deployments go wrong.

| Variant | Total / active | Layers, experts | Context | Modality | Weights |
| --- | --- | --- | --- | --- | --- |
| MiniMax-M2 | 230B / 10B | 62 layers, 256 experts, top-8 | config 196,608 tokens | text | open, modified MIT |

The model card's license header reads modified MIT, and the body says MIT; the modification is an attribution requirement. Either way, this is a permissive license for commercial use.

## The late-October 2025 media wave

M2 shipped as part of a four-day wave.

**Hailuo 2.3 (October 28, 2025).** [Announcement](https://www.minimax.io/news/minimax-hailuo-23): improved body movement, physics and instruction following, with stronger response to motion commands for objects.

**Speech 2.6 (October 29, 2025).** Better sound quality and prosody, faster and more natural, per the release notes.

**Music 2.0 (October 30, 2025).** Songs up to five minutes with verse, chorus and bridge structure, and controllable instruments and vocals ([announcement](https://www.minimax.io/news/minimax-music-20), dated October 31 on the news page).

| Release | Date | Key change | Weights |
| --- | --- | --- | --- |
| Hailuo 2.3 | Oct 28, 2025 | body motion, physics, instruction following | API only |
| Speech 2.6 | Oct 29, 2025 | quality, prosody, latency | API only |
| Music 2.0 | Oct 30, 2025 | 5-minute songs, instrument and vocal control | API only |

## MiniMax-M2.1 (December 2025): multilingual coding

**Release:** December 23, 2025 ([MiniMax M2.1 announcement](https://www.minimax.io/news/minimax-m21)); the release notes list December 22 and the Hugging Face repo was created December 20. Weights: [MiniMaxAI/MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1).

**What changed versus M2:** the architecture did not change at all. The config is identical: 230B total, 10B active, 62 layers, 256 experts with top-8, the same 196,608-token position limit. M2.1 is a post-training release. The announcement says M2 solved cost and accessibility, and M2.1 targets real-world complex tasks, specifically usability across more programming languages: Rust, Java, Go, C++, Kotlin, Objective-C, TypeScript and JavaScript. The card claims it beats Claude Sonnet 4.5 on multilingual software engineering benchmarks and approaches Claude Opus 4.5.

M2.1 also introduced **VIBE** (Visual and Interactive Benchmark for Execution in Application Development), an internal benchmark that has an agent build full applications from scratch and verifies their interactive behavior and visual output. M2.1 scores an average of 88.6 on it. Be careful with the number: it is MiniMax's own benchmark, run with Claude Code as scaffolding on MiniMax's infrastructure.

The other notable claim is framework generalization: MiniMax reports stable SWE-bench Verified results across Claude Code, Droid, Cline, Kilo Code, Roo Code and other scaffolds. That matters because a model that only performs inside its vendor's harness is a much weaker product than the leaderboard suggests.

| Variant | Total / active | Context | Modality | Weights |
| --- | --- | --- | --- | --- |
| MiniMax-M2.1 | 230B / 10B | config 196,608 (card: 196K) | text | open, modified MIT |
| MiniMax-M2.1-highspeed | same capability, faster serving | same | text | API only |

## Music 2.5 and Speech 2.8 (January 2026)

**Music-2.5 (January 16, 2026)** improved quality across dimensions MiniMax did not quantify. **Speech 2.8 (January 23, 2026)**, per [its announcement](https://www.minimax.io/news/minimax-speech-28), adds natural filler sounds, improves voice-cloning similarity from a 10-second sample, and fixes tone and pronunciation drift starting with the Mandarin and Japanese pair.

## MiniMax-M2.5 (February 2026): RL at scale

**Release:** February 12, 2026 ([MiniMax M2.5 announcement](https://www.minimax.io/news/minimax-m25)). Weights: [MiniMaxAI/MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5).

**What changed versus M2.1:** again, not the architecture (the safetensors total is within a rounding error of M2, and the config is the same shape), and again, a lot of training. The announcement and card describe:

- **RL in hundreds of thousands of real-world environments.** Over 200,000 environments across more than ten programming languages (Go, C, C++, TypeScript, Rust, Kotlin, Python, Java, JavaScript, PHP, Lua, Dart, Ruby), with most of MiniMax's own internal tasks and workspaces turned into RL environments.
- **CISPO, still.** MiniMax says it kept the CISPO objective from M1 to keep MoE training stable at this scale, plus work on credit assignment over long agent trajectories.
- **Speed through better decomposition.** M2.5 finishes the SWE-bench Verified evaluation 37% faster than M2.1, using 3.52M tokens per task on average versus 3.72M for M2.1, helped by parallel tool calling. MiniMax describes a "spec-writing" tendency that emerged in training: the model plans features and structure before writing code.
- **Beyond code.** Search (MiniMax built a benchmark called RISE for deep exploration of information-dense pages) and office work designed with finance, law and social-science professionals.

Headline numbers from the card: 80.2% on SWE-bench Verified. The announcement also frames cost: at 100 output tokens per second, an hour of continuous generation costs about $1, so four instances running for a year cost about $10,000.

| Variant | Total / active | Context | Modality | Weights |
| --- | --- | --- | --- | --- |
| MiniMax-M2.5 | 230B / 10B | config 196,608 (about 196K) | text | open, modified MIT, native FP8 checkpoint per the vLLM recipe |
| MiniMax-M2.5-Lightning (API name: M2.5-highspeed) | same capability, faster serving | same | text | API only |

The announcement makes a point worth quoting in substance: M2, M2.1 and M2.5 shipped in three and a half months, and the rate of improvement on SWE-bench Verified exceeded the team's expectations. The architecture froze and the RL machinery became the product. That is the M2 line in one sentence.

## Music 2.5+ (March 2026)

A point release adding instrumental generation (no vocals) and a wider style range (release notes, March 2026; no day given).

## MiniMax-M2.7 (March 2026): self-evolution and a license turn

**Release:** API on March 18, 2026 ([MiniMax M2.7: Early Echoes of Self-Evolution](https://www.minimax.io/news/minimax-m27-en)). Weights: [MiniMaxAI/MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7), repository created April 9, 2026, so open weights followed the API by about three weeks.

**What changed versus M2.5:**

1. **Same architecture, slightly longer context.** Still 230B / 10B, 62 layers, 256 experts top-8, but the config's `max_position_embeddings` rises to 204,800.
2. **"Self-evolution" as a training story.** The announcement describes an internal workflow in which an agent built on M2-series models helps MiniMax's own RL team: literature review, tracking experiment specs, running pipelines, and launching experiments. During M2.7's development, the model updated its own memory, built dozens of skills for RL experiments, and iteratively optimized a scaffold by looping through failure analysis, code changes, evaluation and comparison. Read this as an engineering workflow claim, not a claim that the model trained itself.
3. **Professional and agentic benchmarks.** From the card: 56.22% on SWE-Pro, 76.5 on SWE Multilingual, 52.7 on Multi-SWE-Bench, 55.6% on VIBE-Pro, and a GDPval-AA Elo of 1495. It also emphasizes character consistency and emotional intelligence, with an open-sourced demo called OpenRoom.
4. **A restrictive license.** This is the change that most affects you. The M2.7 license is non-commercial: personal use, self-hosted personal development, and non-profit or academic research are free, but any commercial use, including serving it as an API or deploying a fine-tuned derivative commercially, requires prior written authorization from MiniMax, and authorized commercial users must display "Built with MiniMax M2.7". That is a sharp reversal from M2.5's modified MIT, only five weeks earlier.

| Variant | Total / active | Context | Modality | Weights |
| --- | --- | --- | --- | --- |
| MiniMax-M2.7 | 230B / 10B | config 204,800 | text | open, non-commercial license |
| MiniMax-M2.7-highspeed | same capability, faster serving | same | text | API only |

## Music-2.6 (April 2026)

Covers and instrumental generation, with more emotional expression in vocals (release notes, April 2026).

## MiniMax-M3 (June 2026): sparse attention and native multimodality

**Release:** June 1, 2026 ([MiniMax M3 model page](https://www.minimax.io/models/text/m3)). Weights: [MiniMaxAI/MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) and an MXFP8 checkpoint, [MiniMaxAI/MiniMax-M3-MXFP8](https://huggingface.co/MiniMaxAI/MiniMax-M3-MXFP8), both repositories created June 2. The M3 launch page said full open-sourcing would follow "soon", and press reports put the public weights in the first week of June. Technical report for its attention: [MiniMax Sparse Attention](https://arxiv.org/abs/2606.13392), arXiv 2606.13392, dated June 11, 2026.

**What changed versus M2.7:** M3 is the first new MiniMax architecture since MiniMax-01, and it changes three things.

1. **MiniMax Sparse Attention (MSA).** The third answer to the attention question. M2 proved full attention was needed for agent-grade recall, and full attention made a million-token context too expensive. MSA keeps a standard grouped-query attention layer and adds a tiny learned selector that picks which blocks of the past each query attends to. Unlike lightning attention, it does not compress the past into a state; it reads exact tokens, but only the ones the selector chooses. The M3 card claims 9x faster prefill and 15x faster decode than M2 at 1M context, and per-token compute cut to about 1/20. The MSA paper validates the mechanism on a separate 109B MoE trained from scratch on 3T tokens, where it matches full GQA on downstream benchmarks with 28.4x less attention compute at 1M context, and 14.2x prefill and 7.6x decode wall-clock speedups on an H800. The [MiniMax Sparse Attention deep-dive](/blog/machine-learning/frontier-models/minimax-sparse-attention) covers the selector, its training trick, and the kernels.
2. **Native multimodality.** M3 is trained on mixed text, image and video data from the first step of pretraining, not with a vision encoder bolted on afterwards as in VL-01. The Hugging Face pipeline tag is image-text-to-text.
3. **A bigger model, still sparse.** About 428B total and about 23B active (the safetensors total is 427.0B). The config shows 60 layers, 128 routed experts with top-4 routing plus one shared expert, sigmoid router scoring with a routing bias, QK-norm, grouped-query attention with 64 query heads and 4 KV heads, and multi-token prediction modules. The position limit is 1,048,576 tokens, and the API guarantees a minimum of 512K.

M3 also exposes three reasoning modes through a `thinking` parameter: `enabled`, `adaptive` (the model decides when reasoning helps), and `disabled` for latency. The launch page's showcase is a long-horizon agent run: M3 reproduced an ICLR 2025 outstanding paper over nearly 12 hours, producing 18 commits and 23 experimental figures.

| Variant | Total / active | Layers, experts | Context | Modality | Weights |
| --- | --- | --- | --- | --- | --- |
| MiniMax-M3 | about 428B / about 23B | 60 layers, 128 experts top-4 plus 1 shared | 1M (API minimum 512K) | text, image, video in; text out | open, MiniMax Community License |
| MiniMax-M3-MXFP8 | same | same | same | same | open, MXFP8 quantized checkpoint |

The license moved again: the M3 card uses a MiniMax Community License rather than M2.7's non-commercial terms or M2.5's modified MIT. Read the license file in the repo before you build a product on it.

![Total versus active parameters across MiniMax open-weight generations, MiniMax-01 through M3](/imgs/blogs/minimax-model-family-timeline-3.webp)

The chart above shows the parameter story across the open generations. Total parameters went down from MiniMax-01 to the M2 line and back up for M3, but the number that sets serving cost, active parameters per token, fell from 45.9B to 10B and only climbed back to about 23B when M3 needed capacity for vision and video. Relative to M2, M3 grows total parameters about 1.9x (428B versus 230B) while keeping active parameters near 23B. The M2 line runs at about 4.3% active, the leanest ratio MiniMax has shipped.

## Music 3.0 (July 2026): the first open music model

**Release:** API on July 16, 2026 (release notes); open weights at [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3), repository created August 7, 2026.

**What changed versus Music-2.6:** open weights, and for the first time a documented architecture. The card describes a hierarchical autoregressive design: an **8B Global LLM** (initialized from Qwen3-8B) predicts the first RVQ codebook frame by frame to model long-range song structure, a **0.6B Local LLM** predicts the remaining codebooks within each frame for acoustic detail, and a **2.4B Flow Matching** module with a Flow-VAE decodes from the fused hidden states of both LLMs rather than from discrete tokens alone. Output is 32 kHz, 16-bit stereo, for complete songs up to five minutes, conditioned on lyrics with section tags and a structured music description.

| Variant | Components | Output | Max length | Weights |
| --- | --- | --- | --- | --- |
| MiniMax Music 3 | 8B global LLM, 0.6B local LLM, 2.4B flow matching, Flow-VAE | 32 kHz stereo | 5 minutes | open (see repo license) |

## MiniMax H3 (July 2026): open video with sound

**Release:** July 31, 2026 ([MiniMax H3 announcement](https://www.minimax.io/blog/minimax-h3)). Weights: [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), repository created July 28, under the MiniMax H3 Community License Agreement. Code: [MiniMax-AI/MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3).

**What changed versus Hailuo 2.3:** three things. It is **open**: MiniMax's announcement argues that closed models have dominated video generation and positions H3 as its contribution to an open ecosystem. It is **omni-modal**: H3 understands a unified context of text, images, video and audio, and generates video with native stereo audio, so one model covers text-to-video, image-to-video, video-to-video, audio-to-video and reference-driven generation. And it is a **documented architecture**: the card describes H3-Omni-Transformer, a 33B-parameter dense single-stream transformer with about 13B of those parameters in AdaLN branches, a new H3-VisualVAE with a ViT-based decoder for cheaper decoding, and native sparse-attention training (though the initial open release does inference with full attention only).

| Spec | Value |
| --- | --- |
| Generator | H3-Omni-Transformer, 33B dense (about 13B in AdaLN branches) |
| Inputs | text, image, video, audio context |
| Output duration | 4 to 15 seconds |
| Frame rate | 24 fps |
| Resolution | short side 768 by default; 2K via H3-Regenerate-2K |
| Audio | 32 kHz stereo, dialogue in 11 languages with stable support |
| Weights | open, MiniMax H3 Community License; the In-Context Regeneration workflow is API-only |

H3 is the end of a long arc on the video track: from a free 6-second app in August 2024 to an open 33B audio-video model two years later. For where it sits against Veo, Sora, Kling, Wan and HunyuanVideo, see [the 2026 video model landscape](/blog/machine-learning/video-generation/the-2026-video-model-landscape), which predates H3 and lists Hailuo as closed.

## Research side releases

Two smaller releases sit in the Hugging Face org without being product generations:

| Release | Date | What it is | Weights |
| --- | --- | --- | --- |
| SynLogic-7B / 32B / Mix-3-32B | May to June 2025 | RL reasoning data synthesis research, [arXiv 2505.19641](https://arxiv.org/abs/2505.19641) | open, MIT |
| VTP (Small, Base, Large f16d64) | Dec 16, 2025 | visual tokenizer pretraining research, [arXiv 2512.13687](https://arxiv.org/abs/2512.13687) | open, modified MIT |

> **Reported but not verified: M3.1-Flash-Preview (September 2026).** Several press outlets report that MiniMax launched an M3.1-Flash-Preview text model on September 27, 2026, available only inside its MiniMax Code product and subscription plan, with a 1M context and selectable effort levels. It does not appear on MiniMax's official model release notes as of this writing, there is no model card, and no weights. I list it so you are not surprised by the name, not as a confirmed generation.

## The full timeline in one table

| Date | Release | Track | Total / active | Context | Weights, license |
| --- | --- | --- | --- | --- | --- |
| 2023 | abab5.5 | text | not disclosed | not verified | API only |
| Jan 16, 2024 (press) | abab6 | text | MoE, not disclosed | not verified | API only |
| Apr 17, 2024 (press) | abab6.5, abab6.5s | text | about 1T MoE (press) | about 200k (press) | API only |
| Aug 31, 2024 | video-01 (Hailuo) | video | not disclosed | 6 s clips | API and app |
| Jan 15, 2025 | MiniMax-Text-01, VL-01 | text, vision | 456B / 45.9B | 1M train, 4M inference | open, MiniMax Model Agreement |
| Feb 11, 2025 | T2V-01 / I2V-01-Director | video | not disclosed | | API only |
| Apr 2, 2025 | Speech-02 (HD, Turbo), Image-01 | speech, image | not disclosed | | API only |
| Jun 16, 2025 | MiniMax-M1 (40k, 80k) | text, reasoning | 456B / 45.9B | 1M input | open, Apache 2.0 |
| Jun 18, 2025 | Hailuo 02 | video | not disclosed | 10 s, 1080p | API and app |
| Aug 7, 2025 | Speech 2.5 | speech | not disclosed | | API only |
| Sep 11, 2025 | Music-1.5 | music | not disclosed | 4 min | API only |
| Oct 27, 2025 | MiniMax-M2 | text, agents | 230B / 10B | 196,608 | open, modified MIT |
| Oct 28 to 30, 2025 | Hailuo 2.3, Speech 2.6, Music 2.0 | media | not disclosed | | API only |
| Dec 23, 2025 | MiniMax-M2.1 | text, coding | 230B / 10B | 196,608 | open, modified MIT |
| Jan 2026 | Music-2.5, Speech 2.8 | media | not disclosed | | API only |
| Feb 12, 2026 | MiniMax-M2.5 | text, agents | 230B / 10B | 196,608 | open, modified MIT |
| Mar 18, 2026 | MiniMax-M2.7 | text, agents | 230B / 10B | 204,800 | open, non-commercial |
| Jun 1, 2026 | MiniMax-M3 | multimodal | about 428B / 23B | 1M | open, MiniMax Community License |
| Jul 16, 2026 | Music 3.0 | music | 8B + 0.6B LLMs, 2.4B flow matching | 5 min | open (Aug 2026) |
| Jul 31, 2026 | MiniMax H3 | video plus audio | 33B dense | 15 s, 2K | open, H3 Community License |

## How the family evolved

Lay the generations side by side and four through-lines stand out.

![Four attention regimes in MiniMax language models: softmax MoE, lightning hybrid, full attention, learned sparse attention](/imgs/blogs/minimax-model-family-timeline-2.webp)

### 1. Attention swung three times, and each swing was a response to a measured failure

The figure above is the spine of the family. abab used ordinary softmax attention and was limited to about 200k context. MiniMax-01 and M1 bet on the lightning hybrid to make million-token context and long reasoning cheap, and for long generation it worked: a quarter of DeepSeek-R1's FLOPs at 100k tokens. But the 7:1 ratio meant one exact-retrieval layer per eight, and when MiniMax scaled the hybrid toward agent workloads it found multi-hop reasoning deficits it could not train away. So M2 went back to full attention and accepted a roughly 200k context. M3 then found a way to have both: exact token reads like full attention, but only over the blocks a learned selector picks, which brings back the million-token context at about 1/20 of M2's per-token compute. The lesson that transfers: linear attention's cost advantage is real, but its recall deficit compounds in exactly the workloads that matter most for agents, and a sparse method that keeps exact reads is a better trade.

### 2. Active parameters fell, and the per-token budget became the design target

MiniMax-01 activated 45.9B parameters per token. The M2 line activates 10B, and even M3, with vision and video on board, activates about 23B. The MoE recipe moved from few fat experts (32, top-2) to many thin ones (256, top-8) and then to a middle point with a shared expert (128, top-4 plus 1). Behind this is a product decision MiniMax stated outright with M2: agents call models many times per task, so price and latency per call dominate, and a model that is slightly weaker but five times cheaper per token wins.

### 3. From architecture releases to RL releases

MiniMax-01 and M1 were architecture papers. M2, M2.1, M2.5 and M2.7 share one config, and the improvements between them came from post-training: more programming languages, hundreds of thousands of real-world RL environments, CISPO kept for stability, better task decomposition, and, by M2.7, the model helping run the RL experiments that train its successor. The cadence shows it: four releases of one architecture in under five months. When the architecture is stable, the RL environment pipeline becomes the thing that ships.

### 4. Openness spread across tracks, but the license is a moving target

In 2024 nothing MiniMax made had public weights. By August 2026 the text, video and music tracks all had open models. But the licenses have zigzagged: a custom agreement for 01, Apache 2.0 for M1, modified MIT for M2 to M2.5, non-commercial for M2.7, community licenses for M3 and H3. Speech remains API-only to this day. Openness is a strategy MiniMax applies per release, not a policy.

### 5. Everything is built for agents now

The abab models were chat models for companion apps. MiniMax-01's launch post already said 2025 would be the year of agents, and every generation since has been tuned against agent benchmarks: tool-use traces, SWE-bench across third-party scaffolds, long-horizon office and research tasks, interleaved thinking, and in M3, adaptive thinking modes and a 12-hour autonomous research demo. Even the infrastructure story follows: interleaved thinking forces a new serving contract, and long agent traces are why prefix caching and [hybrid KV-cache management](/blog/machine-learning/model-serving/hybrid-kv-cache-manager) matter so much for this family. For the broader argument about hybrid models breaking the KV-cache assumption, see [hybrid models and the end of the KV-cache assumption](/blog/machine-learning/inference-engineering/hybrid-models-and-the-end-of-the-kv-cache-assumption).

## Which variant to pick today

As of October 2026, this is the short guide.

| You need | Pick | Why |
| --- | --- | --- |
| The strongest open MiniMax model, long context, image or video input | **MiniMax-M3** | 1M context through MSA, native multimodal, about 23B active. Check the Community License for your use. |
| A lean, permissive open model for coding agents | **MiniMax-M2.5** | 10B active, about 196K context, modified MIT, 80.2% SWE-bench Verified per the card. The most commercially simple choice in the family. |
| The best M2-line quality for personal use or research | **MiniMax-M2.7** | Strongest M2-line agent benchmarks, but the license forbids commercial use without authorization. |
| Studying long-reasoning RL or linear attention | **MiniMax-M1-80k** | Apache 2.0, the reference implementation of lightning attention at scale and of CISPO. |
| Studying hybrid linear attention or very long input | **MiniMax-Text-01** | The 7:1 lightning hybrid at 456B, with a 4M-token inference ceiling. Superseded for production. |
| Open video generation with synchronized audio | **MiniMax H3** | 33B dense, up to 15 s with stereo audio, open weights under the H3 Community License. |
| Open song generation | **MiniMax Music 3** | Five-minute songs, documented architecture, open weights. |
| Text-to-speech or voice cloning | **Speech 2.8 (API)** | No open weights on the speech track. The MiniMax-Speech paper describes the Speech-02 architecture if you want to build your own. |

Two practical rules for any M2-line or M3 deployment. First, preserve the `<think>` blocks in the conversation history; interleaved thinking depends on it. Second, if you were on M2.5 and are considering M2.7, read the license before the benchmark table: it is the difference between a drop-in upgrade and a legal review.

## References

**Language models**

- MiniMax. "MiniMax-01: Scaling Foundation Models with Lightning Attention." [arXiv 2501.08313](https://arxiv.org/abs/2501.08313), January 2025. Announcement: [minimax.io/news/minimax-01-series-2](https://www.minimax.io/news/minimax-01-series-2). Weights: [MiniMax-Text-01](https://huggingface.co/MiniMaxAI/MiniMax-Text-01), [MiniMax-VL-01](https://huggingface.co/MiniMaxAI/MiniMax-VL-01).
- MiniMax. "MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention." [arXiv 2506.13585](https://arxiv.org/abs/2506.13585), June 2025. Announcement: [minimax.io/news/minimaxm1](https://www.minimax.io/news/minimaxm1). Weights: [MiniMax-M1-40k](https://huggingface.co/MiniMaxAI/MiniMax-M1-40k), [MiniMax-M1-80k](https://huggingface.co/MiniMaxAI/MiniMax-M1-80k).
- MiniMax-M2 announcement: [minimax.io/news/minimax-m2](https://www.minimax.io/news/minimax-m2). Weights and card: [MiniMaxAI/MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2).
- MiniMax-M2.1 announcement: [minimax.io/news/minimax-m21](https://www.minimax.io/news/minimax-m21). Weights: [MiniMaxAI/MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1).
- MiniMax-M2.5 announcement: [minimax.io/news/minimax-m25](https://www.minimax.io/news/minimax-m25). Weights: [MiniMaxAI/MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5).
- MiniMax-M2.7 announcement: [minimax.io/news/minimax-m27-en](https://www.minimax.io/news/minimax-m27-en). Weights: [MiniMaxAI/MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7). License: [github.com/MiniMax-AI/MiniMax-M2.7/blob/main/LICENSE](https://github.com/MiniMax-AI/MiniMax-M2.7/blob/main/LICENSE).
- MiniMax-M3 model page: [minimax.io/models/text/m3](https://www.minimax.io/models/text/m3). Weights: [MiniMaxAI/MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3), [MiniMax-M3-MXFP8](https://huggingface.co/MiniMaxAI/MiniMax-M3-MXFP8).
- MiniMax. "MiniMax Sparse Attention." [arXiv 2606.13392](https://arxiv.org/abs/2606.13392), June 2026.
- abab6 launch (press): [Sohu, "MiniMax's first MoE large language model fully launched"](https://www.sohu.com/a/752415592_355140), January 2024.
- abab6.5 launch (press): [36Kr Japan, "MiniMax releases MoE model abab 6.5"](https://36kr.jp/286571/), April 2024.

**Speech, video, music**

- MiniMax. "MiniMax-Speech: Intrinsic Zero-Shot Text-to-Speech with a Learnable Speaker Encoder." [arXiv 2505.07916](https://arxiv.org/abs/2505.07916), May 2025. Speech-02 announcement: [minimax.io/news/speech-02-series](https://www.minimax.io/news/speech-02-series).
- Speech 2.5: [minimax.io/news/minimax-speech-25](https://www.minimax.io/news/minimax-speech-25). Speech 2.8: [minimax.io/news/minimax-speech-28](https://www.minimax.io/news/minimax-speech-28).
- Video-01: [minimax.io/news/video-01](https://www.minimax.io/news/video-01). Hailuo 02: [minimax.io/news/minimax-hailuo-02](https://www.minimax.io/news/minimax-hailuo-02). Hailuo 2.3: [minimax.io/news/minimax-hailuo-23](https://www.minimax.io/news/minimax-hailuo-23).
- MiniMax H3: [minimax.io/blog/minimax-h3](https://www.minimax.io/blog/minimax-h3), weights [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3).
- Music 2.0: [minimax.io/news/minimax-music-20](https://www.minimax.io/news/minimax-music-20). Music 3: [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3).

**Platform and research**

- MiniMax model release notes (dates for Director models, Image-01, Music and Speech point releases, M2.x highspeed variants): [platform.minimax.io/docs/release-notes/models](https://platform.minimax.io/docs/release-notes/models).
- SynLogic: [arXiv 2505.19641](https://arxiv.org/abs/2505.19641). VTP: [arXiv 2512.13687](https://arxiv.org/abs/2512.13687).
- MiniMax Hugging Face organization: [huggingface.co/MiniMaxAI](https://huggingface.co/MiniMaxAI).

**Deep-dives on this blog**

- [MiniMax-01: Lightning Attention, the 7:1 Hybrid, and a Million-Token Context](/blog/machine-learning/frontier-models/minimax-01-lightning-attention-hybrid-moe)
- [MiniMax-M1: CISPO and the Economics of Test-Time Compute](/blog/machine-learning/frontier-models/minimax-m1-cispo-test-time-compute)
- [MiniMax-M2: Why They Walked Back Linear Attention for Agents](/blog/machine-learning/frontier-models/minimax-m2-full-attention-agentic)
- [Reading the MiniMax Papers](/blog/machine-learning/frontier-models/minimax-papers-lightning-attention-cispo)
- [MiniMax Sparse Attention](/blog/machine-learning/frontier-models/minimax-sparse-attention)
- [MiniMax-Speech](/blog/machine-learning/frontier-models/minimax-speech-speaker-encoder-flow-vae)
