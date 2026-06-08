---
title: "Kimi-Audio: An Open Audio Foundation Model"
publishDate: "2026-06-05"
date: "2026-06-05"
category: "paper-reading"
subcategory: "Speech Processing"
tags:
  - audio-foundation-model
  - speech-recognition
  - text-to-speech
  - audio-llm
  - flow-matching
  - streaming-tts
  - multimodal
  - kimi
description: "A principal-engineer walkthrough of Kimi-Audio: how one Qwen2.5-7B-based model unifies ASR, audio understanding, TTS, and speech conversation using a 12.5 Hz hybrid tokenizer and a chunk-wise flow-matching detokenizer."
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/kimi-audio-1.png"
readTime: 31
---

If you have ever shipped a voice product, you know the dirty secret of the audio stack: it is not one model, it is a junk drawer of them. You run a dedicated ASR model to turn speech into text. You run a separate text-to-speech model to turn text back into speech. You bolt on an emotion classifier when product asks for "sentiment," a sound-event classifier when someone wants "detect the doorbell," and an audio-captioning model when marketing wants alt-text for clips. Each of those is a different checkpoint, a different training pipeline, a different latency budget, and a different on-call rotation. The glue between them — the transcript that ASR hands to your LLM, the text the LLM hands to TTS — is lossy at every hop. By the time a user's frustrated tone has been flattened into the string "I already told you this," your model has thrown away the exact signal that should have changed its answer.

Kimi-Audio, from Moonshot AI's KimiTeam, is an attempt to collapse that junk drawer into a single audio foundation model. One model handles speech recognition (ASR), audio question answering (AQA), audio captioning (AAC), speech emotion recognition (SER), sound event and scene classification (SEC/ASC), text-to-speech, voice conversion, and full end-to-end speech conversation. It is built on the Qwen2.5 7B language model, it ingests audio through a hybrid tokenizer that runs at 12.5 Hz, and it speaks back through a streaming flow-matching detokenizer that emits waveform one second at a time. The whole thing is pre-trained on more than 13 million hours of audio, and — the part that matters for the rest of us — the code, checkpoints, and a standardized evaluation toolkit are open.

![Kimi-Audio's three components and where the dataflow branches](/imgs/blogs/kimi-audio-1.png)

The diagram above is the mental model: raw audio enters and immediately splits into two parallel representations — discrete semantic tokens and continuous Whisper features — which are summed into a single embedding sequence. That sequence flows through transformer layers shared from Qwen2.5 7B, and then the network forks into two autoregressive heads, one predicting text tokens and one predicting audio tokens. The audio tokens are handed to a flow-matching detokenizer plus a BigVGAN vocoder that produces streaming waveform. Three components, two input paths, two output heads. Everything in this article is an elaboration of that one picture.

> [!tldr] TL;DR
> - **What it claims:** A single open model unifies audio understanding, generation, and speech conversation, built on Qwen2.5 7B with a 12.5 Hz hybrid tokenizer (discrete semantic tokens + continuous Whisper features) and a chunk-wise streaming flow-matching detokenizer, pre-trained on >13M hours of audio.
> - **Why it matters:** It reports state-of-the-art on the bulk of ASR and audio-understanding benchmarks (LibriSpeech test-other 2.42 WER, AISHELL-1 0.60 WER, VoiceBench average 76.93) while shipping reproducible checkpoints and a standardized eval toolkit, which the field has badly lacked.
> - **Most surprising finding:** A *training-free* look-ahead of ~4 future tokens fixes the audio quality that otherwise degrades at streaming chunk boundaries — no retraining, just feed the detokenizer a glimpse of the next chunk.
> - **Where it fails:** Pipelines still lean on ASR transcription and neglect paralinguistic detail; semantic tokens miss acoustic richness; and on open-ended speech conversation GPT-4o still leads the overall average (4.06 vs 3.90) and wins on empathy and style.

## Context: what came before

The audio modeling world has been on a long march from task-specific systems toward generalists, and Kimi-Audio sits at a very specific point on that arc. To see why its design choices are the way they are, we need to walk the lineage.

The oldest layer is the **specialist era**: a Conformer or Whisper-style encoder trained purely for ASR, a Tacotron/FastSpeech-then-VITS lineage trained purely for TTS, a wav2vec or AST backbone fine-tuned for audio classification. Each of these is excellent at one thing and useless at the others. The interface between them is text, and text is a brutally lossy bottleneck for anything paralinguistic — pitch, energy, rate, timbre, laughter, hesitation. If your downstream model only ever sees the transcript, it is structurally blind to *how* something was said.

The next layer is the **audio-LLM era**, where the field realized that if you could turn audio into a sequence of tokens, you could feed it to a transformer language model and let it do everything autoregressively. Two sub-traditions formed here. One uses **continuous** representations: you keep a Whisper-style encoder's float features and project them into the LLM's embedding space (this is roughly the Qwen-Audio and Qwen2.5-Omni recipe). Continuous features preserve acoustic detail beautifully but are not easy to *generate* — you cannot autoregressively sample a float vector the way you sample a token. The other tradition uses **discrete** audio tokens from a vector-quantized (VQ) codebook: GLM-4-Voice, for example, runs a supervised speech tokenizer that emits discrete semantic tokens at a low frame rate. Discrete tokens are generation-friendly (the LLM just predicts the next code, exactly like text) and compact, but a single semantic codebook throws away a lot of acoustic nuance.

![From a rack of task-specific models to one audio foundation model](/imgs/blogs/kimi-audio-5.png)

The gap Kimi-Audio fills is the seam between those two traditions. Prior universal audio models tended to be good at *understanding* (continuous input, text output) **or** good at *generation* (discrete input and output) but rarely state-of-the-art at both, and the conversational ones often lacked low-latency streaming generation. Kimi-Audio's bet is that you do not have to choose. You can feed the LLM **both** a discrete semantic stream and a continuous acoustic stream summed together, get the generation-friendliness of discrete tokens *and* the acoustic richness of continuous features, and then handle the latency problem separately with a chunk-wise detokenizer. The before/after figure above is the thesis in one image: a rack of single-purpose boxes becomes one shared LLM that does all of it.

It is worth being precise about what is *not* novel here, because the paper is honest about it. The discrete semantic tokenizer is borrowed from GLM-4-Voice. The continuous features come from a pre-trained Whisper encoder. The base language model is Qwen2.5 7B. The detokenizer architecture is "the same as MoonCast." Kimi-Audio's contribution is not a brand-new component; it is the *integration* — the hybrid input, the shared-layers-plus-parallel-heads LLM, the training recipe over 13M hours, and the evaluation infrastructure that makes the whole thing reproducible.

There is a second, easy-to-miss part of the gap this fills, and it is about *measurement* rather than modeling. Audio-LLM research has been notoriously hard to compare across papers. WER is deceptively sensitive to text normalization (do you lowercase? strip punctuation? expand "twenty" to "20"?), and two papers reporting "3.1 WER on LibriSpeech" can be using normalization rules that differ by a full point. Inference settings — temperature, beam width, prompt phrasing — vary silently from paper to paper. Open-ended audio QA has been scored by whatever judge each team happened to use. The result is a literature where leaderboard numbers are not actually commensurable. Kimi-Audio-Evalkit is the paper's attempt to fix that by fixing the harness: one WER normalizer, one inference platform with shared parameters, one judge (GPT-4o-mini), applied identically to every model under comparison. Whether or not you adopt the model, that standardization is a contribution to the field's hygiene.

## Contributions

Stripped to the load-bearing claims, the paper contributes the following.

1. **A hybrid audio input that fuses discrete and continuous representations.** Discrete semantic tokens at 12.5 Hz (single VQ codebook, from GLM-4-Voice's supervised tokenizer) carry compact, transcription-aligned content. Continuous Whisper features, natively 50 Hz and downsampled to 12.5 Hz by an adaptor, carry acoustic detail. The two are *added* at the embedding level so the LLM sees one fused sequence at one frame rate.

2. **A unified audio LLM with shared bottom layers and parallel text/audio heads.** Initialized from Qwen2.5 7B, the first several transformer layers are shared to process the multimodal input, after which the network diverges into two autoregressive heads — a text head and an audio head — each predicting its own token stream. The vocabulary is extended with audio semantic tokens and special tokens.

3. **A chunk-wise streaming flow-matching detokenizer with training-free look-ahead.** The detokenizer maps 12.5 Hz discrete tokens to 50 Hz mel-spectrograms via flow matching, then to waveform via a BigVGAN vocoder, running chunk-wise (~1 s) with causal masking. A training-free look-ahead borrows ~4 future tokens from the next chunk to smooth boundary artifacts.

4. **A unified multi-task pretraining recipe with deliberate task weighting.** The pretraining mixture spans unimodal text-only (weight 7) and audio-only (weight 1), audio-text mapping (ASR and TTS, weight 1 each), and interleaving tasks (weight ~1-2). The heavy text-only weight is a deliberate move to preserve the base LLM's language and knowledge abilities.

5. **Instruction diversification for SFT.** Instead of a fixed prompt per task, they auto-generate 200 instructions for ASR and 30 for each other task, in both audio and text versions, sampling one per example.

6. **An open standardized evaluation toolkit (Kimi-Audio-Evalkit).** Standardized WER computation, a unified inference harness with fixed parameters across diverse models, GPT-4o-mini as an LLM judge for open-ended audio QA, and a new speech-conversation benchmark for emotion, speed, accent, empathy, and style.

## Method

The method has three components, and the cleanest way to understand it is to follow a single utterance through the whole machine: how audio becomes a sequence the LLM can read, how the LLM processes it and produces both text and audio tokens, and how those audio tokens become a waveform you can actually hear. We will take them in that order.

### The audio tokenizer: why hybrid, and what "hybrid" actually means

The first design decision Kimi-Audio makes is the one everything else hangs on: how do you represent audio for a transformer? The paper's answer is "do not pick one representation; use two and add them."

![What the hybrid audio representation stacks together](/imgs/blogs/kimi-audio-2.png)

The stack above reads bottom to top. At the bottom is the raw 16 kHz waveform — the full signal, but far too long to model token by token (a single second is 16,000 samples). One layer up are **discrete semantic tokens** at 12.5 Hz from a single VQ codebook. These come from GLM-4-Voice's supervised speech tokenizer, which is itself derived from a Whisper encoder with a vector-quantization bottleneck. At 12.5 Hz, one second of audio is just 12.5 tokens, which is what makes autoregressive generation tractable. The catch is that a single semantic codebook is tuned to preserve *content* — what was said — and discards much of the acoustic texture of *how* it was said.

The next layer up is **continuous acoustic features** from a pre-trained Whisper encoder. Whisper natively produces features at 50 Hz, so an adaptor downsamples them by 4x to match the 12.5 Hz token rate. These float features carry the timbre, prosody, and fine acoustic detail that the discrete codebook throws away. The top layer is the **fused input embedding**: the continuous features are *added* to the embeddings of the discrete semantic tokens, producing one 12.5 Hz sequence that carries both content and acoustics.

Why add rather than concatenate? Addition keeps the sequence length and the embedding dimension fixed, so the fused stream drops into the LLM's existing embedding pipeline with zero architectural surgery beyond the adaptor. Here is the shape of it in PyTorch-flavored pseudocode — note the comments live inside the function body so nothing starts a line with a hash at column zero.

```python
import torch
import torch.nn as nn

class HybridAudioTokenizer(nn.Module):
    def __init__(self, d_model, codebook_size):
        super().__init__()
        # discrete path: a single VQ codebook (GLM-4-Voice style),
        #   each id maps to a learned d_model embedding.
        self.semantic_embed = nn.Embedding(codebook_size, d_model)
        # continuous path: a frozen-then-unfrozen Whisper encoder,
        #   plus an adaptor that downsamples 50 Hz -> 12.5 Hz (factor 4).
        self.whisper = WhisperEncoder()          # pretrained, native 50 Hz
        self.adaptor = nn.Sequential(
            nn.Conv1d(self.whisper.dim, d_model, kernel_size=4, stride=4),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, waveform, semantic_ids, train_step_frac):
        # discrete: ids -> embeddings, already at 12.5 Hz.
        z_disc = self.semantic_embed(semantic_ids)        # (B, T, d_model)

        # continuous: 50 Hz features -> adaptor -> 12.5 Hz.
        feats_50hz = self.whisper(waveform)               # (B, 4T, w_dim)
        z_cont = self.adaptor(feats_50hz.transpose(1, 2)) # (B, T, d_model)

        # staged training: the continuous path is frozen for the first
        #   ~20% of pretraining tokens, then jointly trained.
        if train_step_frac < 0.20:
            z_cont = z_cont.detach()

        # fuse by addition, not concatenation: same length, same width.
        return z_disc + z_cont                            # (B, T, d_model)
```

The `train_step_frac < 0.20` branch is not decoration. The continuous Whisper features are deliberately **frozen for the first ~20% of pretraining tokens** and only unfrozen afterward. The intuition is stability: early in training, the LLM is still learning to read the discrete stream, and letting gradients flow into the Whisper adaptor too soon risks the continuous path collapsing or destabilizing the joint representation. Freeze it until the model has its footing, then let the acoustic path adapt.

One honest gap: the paper states the tokenizer uses a *single* codebook but **does not report the codebook size**. If you are reproducing this, that is a number you will have to recover from the released checkpoints rather than the report.

It helps to do the frame-rate arithmetic explicitly, because the 12.5 Hz choice is doing a lot of quiet work. A 30-second utterance at 12.5 Hz is $30 \times 12.5 = 375$ discrete tokens. The same utterance fed to the LLM as raw 50 Hz Whisper frames would be 1,500 positions — a 4x longer sequence, with quadratic attention cost making it roughly 16x more expensive per layer. And the raw 16 kHz waveform would be 480,000 samples, which is a non-starter for any token-by-token transformer. So the tokenizer is not just a representation choice; it is a 4x-to-1200x sequence-length compression that is what makes a 7B LLM able to read and *generate* audio at all. The continuous path is the clever escape hatch: because the Whisper features are *added* at 12.5 Hz rather than fed as their own 50 Hz stream, you recover the acoustic detail without paying the 4x length penalty. You pay instead with a slightly heavier per-position embedding and the cost of running the Whisper encoder, both of which are cheap relative to autoregressive decoding over a 4x-longer sequence.

The other thing the arithmetic clarifies is why the discrete stream has to come first. The LLM generates *discrete* tokens autoregressively — sampling a codebook index is well-defined; sampling a 1024-dimensional float vector is not. So the discrete semantic tokens are load-bearing for the generation half of the model, and the continuous features are a pure *input-side* enrichment: they make understanding sharper without changing how generation works. That asymmetry is why the audio head predicts discrete tokens and the detokenizer exists to turn those discrete tokens back into sound — there is no continuous output stream to decode.

### The audio LLM: shared trunk, two heads

Once you have a fused 12.5 Hz embedding sequence, the LLM is almost ordinary — almost. It is initialized from **Qwen2.5 7B**, with the vocabulary extended to include the audio semantic tokens and a set of special tokens. The twist is the output structure.

The bottom "first several" transformer layers are **shared**: they process the multimodal input sequence as one stream. Above them, the network **diverges into two parallel autoregressive heads**. The **text head** predicts the next text token; the **audio head** predicts the next discrete audio semantic token. This is what lets a single forward pass produce both a transcript and a spoken response, and it is why the model can do ASR (read audio, emit text), TTS (read text, emit audio), and conversation (read audio, emit both) without changing architecture — only the task framing and the loss masking change.

The paper is careful, and so should we be: it says "first several" layers are shared and then the heads diverge, but it **does not state the exact count of shared layers or the per-head layer counts**. Nor does it state the total parameter count beyond the 7B base, the context length, or the attention variant (we assume a standard transformer, consistent with Qwen2.5). Those are genuine gaps in the report. Here is the structure at the level of detail the paper actually supports.

```python
class KimiAudioLLM(nn.Module):
    def __init__(self, base="Qwen2.5-7B", n_shared="several"):
        super().__init__()
        # initialized from Qwen2.5 7B; vocab extended with audio tokens.
        self.shared = TransformerStack.from_pretrained(base, n_shared)
        # two autoregressive heads on top of the shared trunk.
        self.text_branch  = TransformerStack(depth="rest")
        self.audio_branch = TransformerStack(depth="rest")
        self.text_head  = nn.Linear(d_model, text_vocab)
        self.audio_head = nn.Linear(d_model, audio_vocab)

    def forward(self, fused_embeds, attn_mask):
        # the shared trunk reads the fused multimodal sequence.
        h = self.shared(fused_embeds, attn_mask)
        # then the network forks: each branch is autoregressive.
        h_text  = self.text_branch(h, attn_mask)
        h_audio = self.audio_branch(h, attn_mask)
        return self.text_head(h_text), self.audio_head(h_audio)
```

The reason this matters in practice is the **interleaving** behavior. Because both heads run off the same trunk, the model can alternate between emitting audio and text segments within a single response — speak a sentence, then "reason" in text, then speak again. The pretraining mixture explicitly includes interleaving tasks (alternating semantic-audio and text segments) precisely to teach this. That is the structural foundation for end-to-end speech conversation, where the model has to listen, reason, and speak in one continuous loop rather than three separate models passing strings.

### The detokenizer: turning tokens back into sound, in a stream

The audio head produces a sequence of 12.5 Hz discrete tokens. Those are not sound — they are codebook indices. The detokenizer's job is to turn them back into a waveform, and to do it *fast enough to stream*, which is where most of the engineering lives.

![The streaming detokenizer: tokens to waveform, one chunk at a time](/imgs/blogs/kimi-audio-3.png)

The pipeline above has two stages. First, a **flow-matching** module maps the 12.5 Hz discrete tokens up to a 50 Hz mel-spectrogram. Flow matching is a continuous-generative technique (a cousin of diffusion) that learns a velocity field transporting a simple prior to the target mel distribution; it is a good fit here because it produces high-fidelity continuous output and integrates in a small number of steps. Second, a **BigVGAN** vocoder converts the 50 Hz mel-spectrogram into the final waveform. This is the same architecture as MoonCast — Kimi-Audio reuses it rather than reinventing it.

The hard part is streaming. If you wait for the entire token sequence before detokenizing, you cannot have a real-time conversation. So the detokenizer runs **chunk-wise**: it processes roughly one second of tokens at a time, with **causal masking** so that generating a chunk only depends on the current and past tokens, never the full future. Causality is what makes it streamable — you can emit chunk $c_i$ as soon as its tokens arrive, without knowing $c_{i+1}$.

But causal chunking introduces a defect: **quality degrades at chunk boundaries**. The flow-matching model, asked to generate the last few mel frames of a chunk, has no idea what comes next, so the audio can crack or warble right at the seam. The fix is the paper's most quietly clever idea — a **training-free look-ahead**. When generating chunk $c_i$, the detokenizer is also given the first $n$ (e.g. 4) semantic tokens of the *next* chunk $c_{i+1}$. Those future tokens give the flow-matching model enough context to make the boundary smooth. Crucially, this requires **no retraining**: the model was already trained to condition on tokens, and you are simply giving it a few extra ones at inference. You pay a tiny latency cost (you must wait for ~4 tokens of the next chunk, about 0.3 s at 12.5 Hz) in exchange for clean seams.

```python
def stream_detokenize(token_stream, flow_match, vocoder,
                      chunk_tokens=13, look_ahead=4):
    # ~13 tokens per chunk at 12.5 Hz is roughly one second.
    buffer = []
    for chunk in chunked(token_stream, size=chunk_tokens):
        buffer.extend(chunk)
        # training-free look-ahead: peek at the next chunk's first
        #   `look_ahead` tokens so the boundary does not crack.
        peek = peek_next(token_stream, n=look_ahead)
        cond = buffer[-chunk_tokens - look_ahead:] + peek

        # flow matching: 12.5 Hz tokens -> 50 Hz mel (causal mask).
        mel = flow_match.generate(cond, causal=True)
        # BigVGAN: mel -> waveform, emit immediately.
        wav = vocoder(mel[:chunk_tokens * 4])   # 12.5 Hz -> 50 Hz is 4x
        yield wav
```

Let me put a concrete latency budget on the look-ahead, because the tradeoff is easy to hand-wave. At 12.5 Hz, one token spans 80 ms of audio. A look-ahead of $n = 4$ tokens therefore means the detokenizer must wait for $4 \times 80 = 320$ ms of *future* tokens before it can finalize the current chunk's boundary. For a real-time conversation that target an end-to-end latency under, say, 1 second, spending ~320 ms on boundary smoothing is a meaningful but acceptable slice. If you needed lower latency you could drop to $n = 2$ (160 ms) at the cost of slightly rougher seams, or $n = 0$ (the unfixed baseline) for the lowest latency and the cracked boundaries. The look-ahead is a *knob*, not a fixed cost, which is part of why it is attractive: you can tune it per deployment without retraining anything.

The detokenizer is trained separately from the LLM, in **three stages**: (1) pretrain both the flow-matching model and the BigVGAN vocoder on roughly **1M hours** of audio drawn from the pretraining data; (2) chunk-wise fine-tuning to teach the streaming behavior; (3) speaker-specific tuning. Splitting it out from the LLM is a sensible modularity choice — you can improve the voice without retraining the 7B reasoning core.

### The data: where 13 million hours actually goes

It is worth pausing on the scale, because the data story is half the paper's moat. Pretraining uses **more than 13 million hours** of diverse audio — speech, sound, and music — though the report does **not** break that down by category, so we cannot say what fraction is speech versus music versus environmental sound. That audio is tokenized and mixed with a high-quality text corpus (the one associated with MoonLight) to reach the **585B audio + 585B text token** pretraining budget over one epoch. The deliberate 1:1 token ratio between audio and text, combined with the 7:1 *weighting* toward text-only examples, is the lever that keeps the language brain intact while the audio brain is being grown.

Behind those 13M hours sits a data-processing pipeline that is itself an engineering artifact. It runs at roughly **200,000 hours of audio per day** on a cluster of **30 cloud instances totaling 240 NVIDIA L20 GPUs**. The pipeline is automated — segmentation, filtering, transcription, and tokenization — which is what makes 13M hours tractable in the first place; you are not going to hand-curate that volume. The table below collects the data footprint across the whole system so the numbers are in one place.

| Stage | Data volume | Notes |
|---|---|---|
| LLM pretraining (audio) | >13M hours / 585B tokens | Speech + sound + music, 1 epoch |
| LLM pretraining (text) | 585B tokens | MoonLight-associated corpus, weight 7 |
| SFT (total) | ~300K hours | 2-4 epochs per source |
| SFT — audio understanding | ~190K hours, 26 datasets | ASR, AQA, AAC, SER, SEC, ASC |
| SFT — audio-to-text | 13.8M samples | Text converted to speech via TTS |
| SFT — speech conversation | synthesized | Generated with Kimi-TTS |
| Detokenizer pretraining | ~1M hours | Subset of pretraining audio |
| Data-processing throughput | ~200K hours/day | 240 L20 GPUs / 30 instances |

The thing to internalize from this table is the dependency structure. The conversation SFT data is *synthesized* by Kimi-TTS, and the 13.8M audio-to-text samples are *text-to-speech conversions*. That means a non-trivial fraction of the model's training signal for the hardest, most human task — open conversation — is itself machine-generated. This is the structural ceiling the authors flag in their limitations: the model's expressiveness is bounded by the TTS that made its training data. It is also why the objective benchmarks (real recorded ASR and classification data) are the ones I trust most.

### The training recipe: how the pieces are actually fit

We have three components; now we have to train them. The recipe is where the "preserve the LLM's brain" philosophy shows up most clearly.

![The training recipe, stage by stage](/imgs/blogs/kimi-audio-6.png)

Pretraining is **multi-task next-token prediction** over a weighted mixture, run for **585B audio tokens and 585B text tokens, one epoch**. The weighting is the interesting part and the table below lays it out. Text-only data gets weight **7**, audio-only gets **1**, ASR and TTS each get **1**, and interleaving tasks get **~1-2**. The lopsided text weight is intentional: it keeps the model's language and world-knowledge abilities from eroding as it learns audio. An audio model that has forgotten how to reason is useless for conversation.

| Pretraining task group | Specific task | Relative weight | What it teaches |
|---|---|---|---|
| Unimodal | Text-only | 7 | Preserve LLM language and knowledge |
| Unimodal | Audio-only | 1 | Acoustic/audio modeling |
| Audio-text mapping | ASR (speech to text) | 1 | Ground audio in transcription |
| Audio-text mapping | TTS (text to speech) | 1 | Generate audio from text |
| Interleaving | Audio/text alternation | ~1-2 | Conversational turn-taking |

The optimizer is **AdamW**, with learning rate decaying **2e-5 to 2e-6 on a cosine schedule**, and **1%** of tokens used for warmup. The continuous Whisper features are frozen for the first **~20%** of pretraining tokens, then unfrozen (Stage 1 in the timeline above).

Supervised fine-tuning (SFT) runs on roughly **300K hours** of data. Each data source is fine-tuned for **2-4 epochs**, with learning rate decaying **1e-5 to 1e-6** on cosine and **10%** of tokens for warmup. The SFT data breaks down into audio understanding (~**190K hours** across **26 datasets** spanning ASR, AQA, AAC, SER, SEC, ASC), synthetically generated speech conversation (produced with Kimi-TTS), and **13.8M** audio-to-text samples (text converted to speech via TTS). The **instruction diversification** trick lives here: rather than a single fixed prompt, the team auto-generated **200 instructions for ASR and 30 for each other task**, in both audio and text forms, and samples one per training example. The point is robustness — a model that has only ever seen "Transcribe this audio:" will choke when a user says "what did they say?"

The detokenizer's three-stage training (the right side of the timeline) runs independently on ~1M hours, as described above.

A few numbers worth pinning down because they are easy to misattribute. The **data-processing** pipeline ran at ~**200,000 hours/day** on **30 cloud instances / 240 NVIDIA L20 GPUs**. That is the *preprocessing* cluster, not the model-training cluster — the report **does not state** the GPU count, GPU-hours, or wall-clock for actually training the audio LLM. Do not cite the 240 L20s as the training hardware; that would be wrong.

## Experiments

The headline claim is "SOTA across audio understanding, generation, and conversation," and the evaluation is broad. Let me start with the picture, then put the exact numbers in tables, then say what I believe is load-bearing versus fragile.

![Headline numbers: Kimi-Audio versus the strongest reported competitor](/imgs/blogs/kimi-audio-4.png)

The matrix above is the one-glance summary: on ASR error rate (lower is better) and on most understanding tasks, Kimi-Audio beats the strongest reported competitor; on the open-ended speech-conversation average, GPT-4o still leads. Now the details.

### Speech recognition (WER, lower is better)

Kimi-Audio sets a low water mark across a wide spread of ASR benchmarks. Every cell here is word error rate, and "best competitor" is the strongest number the report cites for that benchmark.

| Benchmark | Kimi-Audio WER | Best competitor |
|---|---|---|
| LibriSpeech test-clean | **1.28** | Qwen2-Audio 1.74 |
| LibriSpeech test-other | **2.42** | Qwen2-Audio 4.04 |
| Fleurs zh / en | **2.69 / 4.44** | Qwen2.5-Omni 2.92 / 4.17 (en lower for Omni) |
| AISHELL-1 | **0.60** | Qwen2.5-Omni 1.13 |
| AISHELL-2 ios | **2.56** | Qwen2.5-Omni 2.56 (tie) |
| WenetSpeech test-meeting / test-net | **6.28 / 5.37** | Qwen2.5-Omni 7.71 / 6.04 |
| Kimi-ASR internal subset1 / subset2 | **1.42 / 2.44** | Qwen2.5-Omni 1.53 / 2.68 |

The AISHELL-1 number (0.60 WER) is the eye-catcher — that is roughly half the error of the next-best system on a major Mandarin benchmark. LibriSpeech test-other at 2.42 against Qwen2-Audio's 4.04 is a similarly large relative gap on the harder English split. The two places it does not clearly win are English Fleurs (Qwen2.5-Omni's 4.17 edges Kimi's 4.44) and AISHELL-2 ios (a tie at 2.56). The pattern is consistent: Kimi-Audio is a genuinely strong recognizer, with its biggest leads on Mandarin and on noisier conditions.

### Audio understanding (accuracy, higher is better)

Understanding spans audio QA, captioning, emotion, and sound classification. Again, the comparison is against the best reported competitor per row.

| Benchmark | Kimi-Audio Acc | Best competitor |
|---|---|---|
| MMAU — Sound | **73.27** | Qwen2.5-Omni 67.57 |
| MMAU — Music | **61.68** | Qwen2-Audio 58.98 |
| MMAU — Speech | **60.66** | Qwen2.5-Omni 53.92 |
| ClothoAQA test / dev | **71.24 / 73.18** | Qwen2.5-Omni 72.86 / 73.12 (test higher for Omni) |
| VocalSound | **94.85** | Qwen2-Audio 93.82 |
| Nonspeech7k | **93.93** | Qwen2-Audio 87.17 |
| MELD (emotion) | **59.13** | Qwen2-Audio 51.23 |
| TUT2017 | **65.25** | Qwen2.5-Omni 43.27 |
| CochlScene test / dev | **79.84 / 80.99** | Qwen2.5-Omni 63.82 / 63.82 |

The sound-scene results are where the margins get dramatic: TUT2017 at 65.25 versus 43.27, and CochlScene at ~80 versus ~64. These are acoustic-scene classification tasks where the *how it sounds* matters more than *what was said* — exactly the kind of task where the continuous Whisper features in the hybrid input should help, and the numbers are consistent with that story. MELD emotion at 59.13 versus 51.23 tells the same tale: paralinguistic tasks benefit from the acoustic path. The one soft spot is ClothoAQA test, where Qwen2.5-Omni edges it by a hair (72.86 vs 71.24).

### Speech conversation and voice chat

This is the hardest category to measure and the one where Kimi-Audio's lead is most qualified. VoiceBench (a voice-assistant benchmark suite) is mostly a clean sweep:

| VoiceBench task | Kimi-Audio | Qwen2.5-Omni |
|---|---|---|
| AlpacaEval (score) | **4.46** | 4.33 |
| CommonEval (score) | **3.97** | 3.84 |
| SD-QA (acc) | **63.12** | 57.41 |
| MMSU (acc) | **62.17** | 56.38 |
| OpenBookQA (acc) | **83.52** | 79.12 |
| IFEval (acc) | **61.10** | 53.88 |
| AdvBench (acc) | **100.00** | 99.62 |
| **Average** | **76.93** | (highest reported among compared audio LLMs) |

But the **open-ended speech-conversation benchmark** — the one Kimi-Audio introduces, scoring emotion, speed, accent, empathy, and style control on a 1-5 scale judged by GPT-4o-mini — is where the honest limits show. Here Kimi-Audio leads open-source models but **GPT-4o still wins the overall average**.

| Speech-conversation dimension | Kimi-Audio (1-5) | Best competitor |
|---|---|---|
| Speed control | **4.30** | GLM-4-Voice 3.83 |
| Accent control | 3.45 | GLM-4-Voice 3.51 (GLM higher) |
| Emotion control | **4.27** | GPT-4o-mini 4.24 |
| Empathy | 3.39 | GPT-4o 3.87 (GPT higher) |
| Style control | 4.09 | GPT-4o 4.54 (GPT higher) |
| **Average** | 3.90 | GPT-4o 4.06 (GPT higher) |

Kimi-Audio tops **speed control** (4.30) and **emotion control** (4.27), which are the dimensions most tied to controllable generation. It loses on **empathy** (3.39 vs GPT-4o's 3.87) and **style** (4.09 vs 4.54), and GLM-4-Voice narrowly beats it on **accent** (3.51 vs 3.45). The aggregate is 3.90 versus GPT-4o's 4.06. For reference, GLM-4-Voice averages 3.65 and Step-Audio-chat 3.33, so Kimi-Audio is clearly the open-source leader here. The precise, defensible claim is: **Kimi-Audio leads the open-source field on conversation and matches or beats GPT-4o-mini on the more mechanical control dimensions, but GPT-4o retains an edge on the soft, human dimensions (empathy, style) and on the overall average.**

### What is load-bearing and what might not transfer

A few things in this evaluation deserve scrutiny before you take the numbers to your own product.

The ASR and sound-classification wins are the most robust, because WER and classification accuracy are objective and the benchmarks are standard. I would trust those to transfer to similar domains (read speech, common acoustic scenes). The *internal* Kimi-ASR subsets are not reproducible by definition, so treat those two cells as directional rather than as evidence you can audit.

The speech-conversation scores are the least transferable, for a structural reason: they are judged by **GPT-4o-mini as an LLM judge**, and Kimi-Audio introduced the benchmark. An LLM judge introduces its own biases, and a self-introduced benchmark is, however well-intentioned, home turf. The fact that GPT-4o still wins the average under Kimi's own benchmark is actually *reassuring* — if the benchmark were rigged, you would not expect the authors' model to lose the headline number to a competitor. But you should anchor on the open, objective benchmarks (VoiceBench, MMAU, the ASR suite) when deciding whether this model fits your use case, and treat the 1-5 conversation scores as a tie-breaker rather than a primary signal.

## Critique

**What is strong.** The integration thesis is validated by the numbers: a hybrid discrete-plus-continuous input really does seem to give you both generation-friendliness and acoustic richness, and the sound-scene and emotion results (TUT2017 65.25 vs 43.27; MELD 59.13 vs 51.23) are exactly where you would predict the acoustic path to pay off. The training-free look-ahead is the kind of cheap, high-leverage trick that should outlive this specific model — it costs ~4 tokens of latency and zero retraining to fix streaming-boundary artifacts, and I expect it to be copied. And the open-sourcing of checkpoints *plus* a standardized eval toolkit addresses a real, embarrassing gap in the field: audio-LLM papers have historically been near-impossible to compare because everyone used different WER normalization and different inference settings.

**What is weak or unfalsifiable.** The biggest problem is the **complete absence of isolated ablations**. The report justifies its core design choices — hybrid input over either representation alone, look-ahead, shared-layers-then-heads, staged unfreezing — entirely *qualitatively*. There is not a single table that says "with continuous features: X WER; without: Y WER." So when I claim the acoustic path explains the sound-scene wins, that is an inference from the *story*, not a measured delta the paper provides. It is plausible, but it is not proven, and a skeptic could argue the wins come from the 13M-hour data scale alone. The look-ahead's benefit is described as fixing "quality degradation" with no MOS or quantitative boundary-artifact number attached. The 7:1 text-to-audio weighting is asserted to preserve language ability with no before/after on a language benchmark.

**What ablation is missing.** The one I most want: a 2x2 over {discrete-only, continuous-only, hybrid} input crossed with a couple of representative tasks (one ASR, one sound-scene). That single table would convert the central thesis from "told" to "shown." Secondary wishes: a look-ahead sweep over $n \in \{0, 2, 4, 8\}$ with a quality metric, and an ablation removing the text-only weight to quantify the language-preservation claim.

**What would change my mind.** If someone ran the discrete-only and continuous-only ablations and found the hybrid input gave less than ~1 WER point and less than a few accuracy points over the better single representation, I would downgrade the "hybrid is the key idea" framing and conclude the real driver is data scale plus a strong Qwen2.5 base. Conversely, if the hybrid ablation showed a large gap specifically on paralinguistic tasks, I would upgrade my confidence that this is the right architecture for emotion- and style-aware voice systems.

There are also honest gaps the authors themselves flag in Section 8: pipelines still lean on **ASR transcription and neglect paralinguistic information**; **no single audio representation is ideal** (semantic tokens miss acoustic detail, acoustic tokens lack abstract semantics); and overall performance is **bounded by the ASR accuracy and TTS quality** baked into the data pipeline. That last one is the sneaky structural ceiling: the speech-conversation SFT data is synthesized with Kimi-TTS and the audio-to-text data is TTS-converted, so the model can only be as expressive and diverse as the TTS that generated its training signal.

## What I'd build with this

1. **A paralinguistic-aware support agent.** The MELD and emotion-control numbers say the model can hear frustration, not just transcribe it. I would build a customer-support voice agent that routes or de-escalates based on the *acoustic* emotion signal flowing through the continuous path, rather than re-deriving sentiment from the (lossy) transcript. The whole point of the hybrid input is that the emotion never has to be flattened to text.

2. **A low-latency live captioner with smart endpointing.** Given the streaming detokenizer and the strong ASR, I would wrap the audio head's interleaving ability into a live captioning service that emits text incrementally and uses chunk-wise processing on the *input* side to balance latency against accuracy at utterance boundaries.

3. **A controllable TTS service exposing the speed/emotion axes.** Kimi-Audio tops speed control (4.30) and emotion control (4.27). I would expose those as first-class API parameters for a TTS product — "say this faster," "say this warmly" — since the model already does this better than GPT-4o-mini on those specific dimensions.

4. **A reproducible audio-LLM leaderboard for an internal model zoo.** Adopt Kimi-Audio-Evalkit as the harness — standardized WER, fixed inference parameters, GPT-4o-mini judging — so that every audio model we train or fine-tune is scored identically. The reproducibility infrastructure is arguably as valuable to an org as the checkpoint.

5. **A domain-adapted recognizer via the 26-dataset SFT recipe.** For a vertical with weird acoustics (medical dictation, call-center audio), I would replicate the instruction-diversification SFT (200 ASR prompts, audio + text) on domain data, starting from the released Kimi-Audio-7B, rather than training a specialist ASR model from scratch.

## When to reach for Kimi-Audio (and when not to)

Reach for Kimi-Audio when you want **one model for many audio tasks** and you value **open weights, reproducible evaluation, and strong ASR plus understanding** — especially if your workload is Mandarin-heavy, involves acoustic-scene or emotion classification, or needs end-to-end speech conversation with low-latency streaming output. It is genuinely state-of-the-art on the bulk of objective audio benchmarks, and the fact that you can download the checkpoints and the eval toolkit makes it a far better foundation to build on than a closed API you cannot inspect or fine-tune. The 12.5 Hz hybrid tokenizer and the training-free look-ahead are the two ideas most worth stealing even if you do not adopt the full model.

Do **not** reach for it as a drop-in upgrade if your product lives or dies on the *soft* conversational dimensions — deep empathy, nuanced persona and style control — because on those specific axes GPT-4o still leads (empathy 3.87 vs 3.39, style 4.54 vs 4.09, overall 4.06 vs 3.90), and the conversation scores come from a self-introduced, LLM-judged benchmark you should weight cautiously. Be equally cautious if your domain demands fidelity to fine acoustic or expressive detail that the semantic tokens are known to drop, or if your expressiveness ceiling is set by the TTS used to synthesize the conversation data — the authors are explicit that performance is bounded by the ASR accuracy and TTS quality in the pipeline. And if you need exact reproduction of the architecture, budget for the genuinely undisclosed pieces: shared-layer count, total parameters, context length, codebook size, and the LLM training hardware are all "not stated" in the report and will have to be recovered empirically from the release.

## References

- **arXiv abstract:** [Kimi-Audio Technical Report (arXiv:2504.18425)](https://arxiv.org/abs/2504.18425)
- **Code + checkpoints:** [MoonshotAI/Kimi-Audio (GitHub)](https://github.com/MoonshotAI/Kimi-Audio)
- **Evaluation toolkit:** [MoonshotAI/Kimi-Audio-Evalkit (GitHub)](https://github.com/MoonshotAI/Kimi-Audio-Evalkit)

Related reading from the Kimi/Moonshot line of work:

- [Kimi-VL: A Mixture-of-Experts Vision-Language Model](/blog/paper-reading/multimodal/kimi-vl)
- [Kimi K2: Open Agentic Intelligence](/blog/paper-reading/large-language-model/kimi-k2)
- [Muon is Scalable for LLM Training: Inside Moonlight](/blog/paper-reading/large-language-model/muon-moonlight)
- [Kimi k1.5: Scaling Reinforcement Learning with LLMs](/blog/paper-reading/reinforcement-learning/kimi-k1-5)
