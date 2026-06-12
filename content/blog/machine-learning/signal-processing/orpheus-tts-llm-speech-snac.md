---
title: "Orpheus-TTS Deep Dive: Teaching Llama to Speak with SNAC Tokens"
date: "2026-06-03"
publishDate: "2026-06-03"
description: "How Orpheus turns a 3B Llama into a streaming, emotive text-to-speech model by treating audio as just more tokens — the architecture, the SNAC codec, the 7-token frame trick, training, and how to finetune your own voice."
tags: ["tts", "speech-synthesis", "llama", "snac", "audio-codec", "streaming", "vllm", "voice-cloning", "neural-codec", "autoregressive", "low-latency", "finetuning"]
category: "machine-learning"
subcategory: "Signal Processing"
author: "Hiep Tran"
featured: true
readTime: 50
---

Most text-to-speech systems are a small zoo of specialized models bolted together: a text frontend that predicts phonemes and durations, an acoustic model that turns those into a mel-spectrogram, and a vocoder that turns the spectrogram into a waveform. Each stage has its own loss, its own failure modes, and its own opinions about prosody. If you have ever debugged why a name gets mispronounced, or why the energy collapses at the end of a long sentence, you have felt the seams between those stages.

[Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS) from Canopy Labs makes a different bet. It throws away the bespoke acoustic stack and asks a blunt question: *what if speech synthesis is just next-token prediction?* Take a pretrained Llama-3.2-3B, give it a vocabulary that contains both text tokens and audio tokens, and train it to continue a sequence. Feed it text, let it autoregressively emit audio tokens, and hand those tokens to a neural codec decoder to get a waveform. No duration predictor, no mel-spectrogram, no separate vocoder loss. One transformer, one cross-entropy loss, one `generate()` call.

That bet pays off in places the classical stack struggles: Orpheus produces genuinely expressive prosody, takes inline emotion tags like `<laugh>` and `<sigh>`, clones a voice zero-shot from a few seconds of reference audio, and streams its first audio byte in roughly 100–200 ms. It does all of this under an Apache-2.0 license with a model small enough to run on a single consumer GPU.

![How Orpheus turns text into speech](/imgs/blogs/orpheus-tts-llm-speech-snac-1.png)

The diagram above is the mental model for the entire article: text goes into a Llama tokenizer, the Llama-3B backbone autoregressively predicts a stream of *audio* tokens (7 SNAC codes per frame), and a small convolutional SNAC decoder turns those codes into a 24 kHz waveform. Everything interesting in Orpheus is an answer to one of two sub-questions — *how do you make audio look like tokens a language model can predict?* and *how do you turn those tokens back into sound fast enough to stream?* The rest of this post is a tour of that diagram, layer by layer, with the actual token arithmetic, the training recipe, the streaming decoder, and the production gotchas that bite people the first week they deploy it.

## Why Orpheus is different

If you come from a Tacotron/FastSpeech/VITS background, almost every instinct you have about TTS is slightly wrong for Orpheus. The table below is the assumption-versus-reality map I wish I had on day one.

| Assumption from classical TTS | The naive view | The Orpheus reality |
|---|---|---|
| You need a duration model to align text to audio | Phoneme durations must be predicted explicitly | Alignment is learned implicitly by autoregression; there is no duration head at all |
| Audio is a spectrogram | The model outputs continuous mel frames | The model outputs *discrete* tokens from a neural codec, exactly like words |
| The vocoder is a separate trained network with its own loss | GAN vocoder, adversarial training, careful tuning | The "vocoder" is a frozen SNAC decoder; the LM never sees waveform loss |
| Prosody and emotion need explicit conditioning | Reference encoders, global style tokens, pitch predictors | Emotion is in-context: literally type `<laugh>` in the text |
| Voice cloning needs a speaker embedding network | Train a d-vector / x-vector extractor | Prepend a reference (transcript, audio-token) pair; zero gradient updates |
| TTS models are small (5–30M params) | Bigger is wasteful for speech | A 3B-parameter LLM backbone is the *point* — it brings language understanding |
| Latency is dominated by the vocoder | Optimize the GAN | Latency is dominated by LLM decode throughput; the codec is ~5 ms |

The single most important row is the second one. **The entire design hinges on the idea that you can represent a waveform as a short sequence of integers that a language model can predict with a softmax.** That representation is the neural audio codec, and Orpheus uses [SNAC](https://github.com/hubertsiuzdak/snac). Before we can talk about the transformer at all, we have to talk about how sound becomes tokens — and that is where most of the subtlety lives. If you want the broader landscape of how speech becomes discrete tokens, our [speech tokenizers deep dive](/blog/machine-learning/signal-processing/speech-tokenizers-encodec-soundstream-mimi) covers EnCodec, SoundStream, and Mimi; SNAC is a close cousin with one extra trick.

> A neural codec is to an audio LLM what BPE is to a text LLM: it is the unglamorous component that decides what the model can possibly say.

## 1. The merged vocabulary

**Senior rule of thumb: in an audio LLM, the vocabulary is the architecture. Get the token layout right and the transformer is boring; get it wrong and nothing else matters.**

A standard Llama-3 tokenizer has roughly 128k entries. Orpheus keeps all of them and then *extends* the embedding table so that a single integer space holds three disjoint kinds of token: ordinary text subwords, a handful of control tokens, and a large block of audio tokens.

![The merged token vocabulary](/imgs/blogs/orpheus-tts-llm-speech-snac-2.png)

The figure shows the three ranges. Text tokens occupy the original `0 – 128 255` range — these are the BPE merges Llama already knows. Just above them sits a small band of **control tokens**: start-of-human, end-of-text, start-of-speech, and end-of-speech markers (in the repo these are named constants in the low `128256`–`128265` region). And above *those*, starting at offset `128266`, sits the audio block: one entry for every possible SNAC code, encoded as a textual `custom_token_N` form.

> Audio is not a special modality to the model; it is just a region of the vocabulary that happens to sound like something when you decode it.

Why does this matter? Because it means **one embedding matrix and one output softmax cover both modalities**. The model does not switch heads when it transitions from "reading text" to "emitting audio." It just keeps predicting the next integer. At a start-of-speech token, the distribution over the next token simply shifts most of its mass into the audio range. This is the same trick that lets a multimodal LLM emit image tokens, and it is why Orpheus can inline an emotion tag: `<laugh>` is a *text* token sitting in the same stream as the audio it modulates, so the model learns the correlation between the tag and the acoustic realization during training.

Here is the offset arithmetic in code. Every audio token is stored as a string like `<custom_token_12345>`, and decoding it back to a raw SNAC code is a fixed transformation:

```python
def turn_token_into_id(last_token: str, index: int) -> int | None:
    """decoder.py: map a `<custom_token_N>` string to a SNAC index in [0, 4095]."""
    token_string = last_token.strip()
    if not token_string.startswith("<custom_token_"):
        return None
    # Slice out the integer N from "<custom_token_N>"
    number_str = token_string[14:-1]
    # Undo the +10 padding and the per-slot codebook offset.
    return int(number_str) - 10 - ((index % 7) * 4096)
```

Two constants are doing the heavy lifting here and both are worth internalizing. The `- 10` removes a small reserved padding offset. The `- (index % 7) * 4096` is the interesting one: each audio token's *position within its 7-token frame* tells you which of the three SNAC codebooks it belongs to, and each codebook is given its own non-overlapping 4096-wide slice of the integer space. We will unpack that `% 7` in Section 3 — it is the single most Orpheus-specific piece of the whole design.

### Second-order optimization: why offset, not a separate head?

You could imagine an alternative where audio tokens get their own small output projection, separate from the text vocabulary. Orpheus deliberately does not do this. By packing audio codes into the *same* softmax via integer offsets, it inherits Llama's tied embeddings, its training infrastructure, its sampling code (temperature, top-p, repetition penalty), and — critically — every inference engine that already serves Llama. The cost is a wider softmax (≈128k + a few thousand audio entries) which is negligible on modern hardware. The payoff is that you can serve Orpheus on **stock vLLM with zero custom kernels**, which is exactly what the project does. That single decision is why the latency story later is so good.

## 2. SNAC: the hierarchical audio codec

**Senior rule of thumb: the codec's frame rate sets the LLM's token budget. Halve the frame rate and you double the audio per generated token — but you also blur the high frequencies the model can ever reproduce.**

SNAC — the *Multi-Scale Neural Audio Codec* — is a residual-vector-quantization (RVQ) autoencoder, in the same family as SoundStream, EnCodec, and DAC. A convolutional encoder compresses a 24 kHz waveform into a latent sequence; a stack of vector quantizers turns each latent into a small set of discrete codes; a convolutional decoder reconstructs the waveform from those codes. The 24 kHz SNAC model that Orpheus uses runs at roughly **0.98 kbps with about 19.8M parameters** — tiny next to the 3B LLM that drives it.

![SNAC: a hierarchical audio codec](/imgs/blogs/orpheus-tts-llm-speech-snac-3.png)

What makes SNAC *multi-scale* — and what makes it a perfect fit for an autoregressive LM — is the part the figure emphasizes: **the codebooks operate at different time resolutions.** Where a vanilla RVQ codec applies every quantizer at the same frame rate, SNAC samples its coarse codebook less frequently than its fine codebook. Concretely, for the 3-level configuration Orpheus relies on:

- **Codebook 1 (coarse)** runs at the lowest rate — roughly one code per frame group. It captures the slow-moving structure: pitch contour, broad phonetic identity, energy envelope.
- **Codebook 2 (mid)** runs at roughly double that rate — two codes per frame group — refining the residual the coarse level left behind.
- **Codebook 3 (fine)** runs at roughly double again — four codes per frame group — capturing the fast detail: fricative noise, transients, the crispness that makes speech sound un-muffled.

Each codebook holds **4096 entries**, which is exactly why the offset arithmetic in Section 1 strides by 4096. The "residual" framing matters: codebook 2 does not re-encode the audio from scratch, it quantizes *what codebook 1 got wrong*, and codebook 3 quantizes what 1 and 2 together still got wrong. This is the standard RVQ insight, and it is why you can drop the fine codebook and still get intelligible (if duller) audio — the coarse levels carry the load.

The reason hierarchical rates matter so much for an LLM is bandwidth. A flat 3-codebook codec at the fine rate would force the LLM to emit three tokens per fine frame — a brutal token budget. SNAC's pyramid means that across one "frame group" the model emits **1 + 2 + 4 = 7 tokens** total, and those 7 tokens cover a relatively long span of audio. That single number, 7, is the bridge between the codec and the transformer.

It is worth making the residual quantization concrete, because it is the mechanism that makes the pyramid coherent rather than three unrelated streams. Let $z$ be the continuous latent the encoder produces for a frame. The first quantizer finds the nearest entry $q_1$ in codebook 1 and records its index $c_0$; the residual $r_1 = z - q_1$ is what it could not represent. The second quantizer quantizes that residual, $q_2 \approx r_1$, recording $c_1$, leaving $r_2 = r_1 - q_2$; the third does the same on $r_2$. The reconstruction is the *sum* $\hat{z} = q_1 + q_2 + q_3$, and the decoder maps $\hat{z}$ back to a waveform. Two consequences follow directly. First, the codebooks are ordered by importance: $q_1$ carries the most energy, $q_3$ the least, which is exactly why predicting coarse-before-fine (Section 3) aligns with the data's own structure. Second, you can truncate — decode with only $q_1 + q_2$ — and still get intelligible audio, because you are dropping the smallest residual. This graceful degradation is a property of RVQ that flat codecs lack, and it is part of why codec-token TTS is robust to the occasional dropped fine token.

The *multi-scale* twist on top of plain RVQ is that the three quantizers do not operate on the same temporal grid. The coarse quantizer sees a downsampled latent (fewer, longer frames), while the fine quantizer sees the latent at full rate. This is what makes the 1-2-4 ratio fall out naturally: one coarse code spans the same wall-clock time as two mid codes and four fine codes. The codec is, in effect, deciding to spend its bit budget where the ear needs it — slowly-varying structure gets few bits, fast detail gets many — and handing the LLM a token layout that already reflects that decision.

### Second-order optimization: the reconstruction ceiling

There is a hard ceiling you inherit from the codec and can never exceed: **the LLM can only sound as good as SNAC can reconstruct.** If you take a clean studio recording, encode it to SNAC codes, and decode it straight back without any LLM in the loop, that resynthesized audio is the absolute upper bound on Orpheus output quality. In practice the codec is the source of the slightly "neural" timbre you can sometimes hear. If you are evaluating Orpheus and the audio sounds subtly compressed, do not blame the transformer first — run the codec round-trip on ground-truth audio and listen to *that*. It separates codec error from model error, and it is the first diagnostic I reach for. The same logic applies to any codec-based TTS; for the GAN-vocoder contrast, see our [HiFi-GAN post](/blog/machine-learning/signal-processing/hifi-gan).

## 3. Flattening 3 codebooks into a 7-token frame

**Senior rule of thumb: an autoregressive model can only predict a flat sequence. The art of an audio LLM is choosing the flattening order so that "easy, structural" tokens come before "hard, detail" tokens.**

Here is the crux of the whole design. SNAC gives you a *hierarchy* — three parallel streams at different rates. But a Llama decoder predicts a single flat sequence, one token at a time, left to right. So you must serialize the hierarchy into a 1-D order. Orpheus uses a specific, fixed interleave for each frame group of 7 tokens.

![Flattening 3 codebooks into a 7-token frame](/imgs/blogs/orpheus-tts-llm-speech-snac-4.png)

Reading the figure as an assignment grid: across the 7 slots of one frame, the codes are laid out as

$$\text{slot order} = [\,c_0,\ c_1^{a},\ c_2^{a},\ c_2^{b},\ c_1^{b},\ c_2^{c},\ c_2^{d}\,]$$

where $c_0$ is the single coarse code, $c_1^a, c_1^b$ are the two mid codes, and $c_2^{a..d}$ are the four fine codes. In code-position terms (matching the `% 7` from Section 1): slot 0 is codebook 0; slots 1 and 4 are codebook 1; slots 2, 3, 5, and 6 are codebook 2.

Why this exact order rather than the obvious `[c0, c1, c1, c2, c2, c2, c2]`? Two reasons, both about giving the autoregressive model an easier prediction problem:

1. **Coarse-before-fine within the frame.** The coarse code $c_0$ comes first, so when the model predicts the fine codes later in the same frame it has already committed to the structural content. The fine detail is *conditioned on* the coarse decision, which mirrors the residual dependency in the codec itself. You are aligning the prediction order with the causal structure of the data.
2. **Interleaving mid and fine.** The mid codes are split (slots 1 and 4) and interleaved with the fine codes rather than dumped consecutively. This keeps the two halves of the frame structurally similar — each half is "one mid code followed by two fine codes" — which empirically makes the sequence statistics more regular and easier to model.

The flattening is the reason a single number — 7 — shows up everywhere in the codebase: the frame size, the modulus in `index % 7`, the chunk granularity in streaming. Internalize "7 tokens = one SNAC frame group" and the rest of the code reads cleanly.

Here is the encode-side flattening you would write when preparing training data — the exact inverse of the decode path we will see next:

```python
import torch
from snac import SNAC

snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()

def audio_to_orpheus_tokens(wav_24k: torch.Tensor) -> list[int]:
    """wav_24k: (1, 1, T) float tensor at 24 kHz. Returns flat Orpheus audio token ids."""
    with torch.inference_mode():
        codes = snac.encode(wav_24k.cuda())  # list of 3 tensors: [ (1,F), (1,2F), (1,4F) ]
    c0, c1, c2 = codes[0][0], codes[1][0], codes[2][0]
    n_frames = c0.shape[0]
    flat = []
    OFFSET = 128266  # start of the audio token block
    for i in range(n_frames):
        # Interleave per the fixed 7-slot order, applying the per-slot 4096 stride.
        flat.append(OFFSET + 0 * 4096 + c0[i].item())          # slot 0  -> codebook 0
        flat.append(OFFSET + 1 * 4096 + c1[2 * i].item())      # slot 1  -> codebook 1
        flat.append(OFFSET + 2 * 4096 + c2[4 * i].item())      # slot 2  -> codebook 2
        flat.append(OFFSET + 3 * 4096 + c2[4 * i + 1].item())  # slot 3  -> codebook 2
        flat.append(OFFSET + 4 * 4096 + c1[2 * i + 1].item())  # slot 4  -> codebook 1
        flat.append(OFFSET + 5 * 4096 + c2[4 * i + 2].item())  # slot 5  -> codebook 2
        flat.append(OFFSET + 6 * 4096 + c2[4 * i + 3].item())  # slot 6  -> codebook 2
    return flat
```

Notice the indexing: `c1` is indexed `2*i` and `2*i+1` because it has two codes per frame; `c2` is indexed `4*i .. 4*i+3` because it has four. The stride `slot * 4096` is what guarantees that each codebook lives in its own non-overlapping window of the integer space, so the model never confuses a coarse code for a fine one.

### Second-order optimization: flattening order is a hyperparameter, not a law

It is tempting to treat the 7-slot order as sacred. It is not — it is a design choice, and other audio LLMs choose differently (some emit all coarse codes first across the whole utterance, then all fine codes in a second pass). Orpheus's *frame-interleaved* order is what makes single-pass streaming possible: because each 7-token frame is self-contained, you can decode audio as soon as you have one complete frame, rather than waiting for the end of the utterance. If you ever fork the model to use a different codec, the flattening order is the first thing you must redesign, and "can I decode incrementally?" is the constraint that should drive it.

## 4. Decoding: redistribute_codes

**Senior rule of thumb: decoding is encoding run backwards. If the encode flattening is correct, the decode is mechanical — but the off-by-one in the offset is the single most common bug when people port this.**

At inference the model emits a flat stream of `<custom_token_N>` strings. To make sound, you reverse the flattening: parse each token to a 0–4095 code, regroup the codes into the three SNAC layers, and call the decoder.

![Decoding a 7-token frame back to audio](/imgs/blogs/orpheus-tts-llm-speech-snac-5.png)

The figure traces the path: a flat 7-token frame → parse the integer out of the string → subtract `10 + (index % 7) * 4096` to recover the raw code → rebuild `codes_0 / codes_1 / codes_2` in the 1-2-4 split → `SNAC.decode` → 2048 samples of 24 kHz PCM. Here is the regrouping in full:

```python
import torch

def redistribute_codes(frame_ids: list[int]):
    """frame_ids: a flat list whose length is a multiple of 7. Returns 3 SNAC layers."""
    codes_0, codes_1, codes_2 = [], [], []
    n_frames = len(frame_ids) // 7
    for i in range(n_frames):
        base = 7 * i
        # Slot 0 -> codebook 0 (1 per frame)
        codes_0.append(frame_ids[base + 0])
        # Slots 1, 4 -> codebook 1 (2 per frame)
        codes_1.append(frame_ids[base + 1])
        codes_1.append(frame_ids[base + 4])
        # Slots 2, 3, 5, 6 -> codebook 2 (4 per frame)
        codes_2.append(frame_ids[base + 2])
        codes_2.append(frame_ids[base + 3])
        codes_2.append(frame_ids[base + 5])
        codes_2.append(frame_ids[base + 6])

    layers = [
        torch.tensor(codes_0).unsqueeze(0),
        torch.tensor(codes_1).unsqueeze(0),
        torch.tensor(codes_2).unsqueeze(0),
    ]
    return layers  # ready for snac.decode(layers)
```

A defensive check the real decoder performs and you should keep: every recovered code must satisfy $0 \le c \le 4095$. If a code lands outside that range, the model emitted a token from the wrong codebook slot — usually because your frame boundary drifted and the `index % 7` is misaligned. Orpheus validates this with `torch.any(codes < 0) or torch.any(codes > 4096)` per layer and drops the bad frame rather than feeding garbage to the decoder. Treat an out-of-range code as a loud alarm that your buffering is off by some non-multiple of 7, not as something to clamp and ignore.

It is worth dwelling on *why* the `index` argument to `turn_token_into_id` is a global running count rather than a per-frame position. The function computes `index % 7` to recover the slot, which means the caller must pass a monotonically increasing counter that started at the very first audio token. If you ever reset that counter mid-stream — say, because you restarted the buffer after an error — the modulus desynchronizes from the actual frame structure and *every subsequent token* decodes to the wrong codebook. This is the single nastiest class of bug in reimplementations, because it produces audio that is not silent and not obviously broken: it is plausible-sounding gibberish, the audio equivalent of a text model with a shifted positional encoding. The defensive posture is to make the counter the single source of truth, never reset it within a stream, and assert `len(buffer) % 7 == 0` before any decode. If you must recover from a mid-stream error, the only safe move is to discard tokens forward to the next multiple-of-7 boundary and resume counting from there, preserving the global alignment.

The `+10` padding deserves a one-line explanation too, since it looks like a magic number. The audio token block does not start exactly at the codebook-zero offset; a small reserved gap (the `10`) sits between the control tokens and the first audio code, leaving room for special-purpose tokens without renumbering the entire audio range. When you port the decoder, copy the `-10` exactly — it is not optional rounding, it is an offset into a deliberately laid-out integer space, and dropping it shifts every code by ten, which (because of the modulus interaction) corrupts the layer assignment rather than producing a uniform pitch shift.

### Second-order optimization: the `[2048:4096]` slice

There is a non-obvious detail in the decode that trips up everyone who reimplements streaming. When you decode a window of frames, the SNAC convolutional decoder produces samples whose *edges* are unreliable — the convolutions near the boundary of the window do not have full receptive-field context, so they contain artifacts. The Orpheus streaming decoder therefore decodes a sliding window of frames but only *keeps the middle slice*, `audio_hat[:, :, 2048:4096]` — the 2048 samples (~85 ms at 24 kHz) in the center, where the receptive field is fully populated. The edge samples are recomputed on the next window. This is the streaming equivalent of overlap-and-discard, and it is why naive single-frame decoding produces audible clicks while the real implementation does not.

## 5. Training: pretrain vs finetune

**Senior rule of thumb: pretraining teaches the model that audio tokens exist and how they relate to text; finetuning teaches it to be one specific, consistent voice. Do not confuse the two — they want different data and different sequence formats.**

Orpheus is released as a *pair* of model types, and understanding the split is essential to using it well:

- A **pretrained** base model, trained on **100k+ hours of English speech** plus text data, with a sequence length of **8192 tokens**. This model is a general audio continuation engine — excellent for zero-shot cloning, not tied to any single speaker.
- A **finetuned production** model, optimized for everyday TTS with a stable set of named voices (`tara`, `leah`, `jess`, `leo`, `dan`, `mia`, `zac`, `zoe`) and reliable emotion-tag behavior.

There is also a multilingual family of pretrained/finetuned pairs covering several languages, and a roadmap of smaller backbones (1B, 400M, 150M) for edge deployment — the same idea our [low-latency edge TTS post](/blog/machine-learning/signal-processing/low-latency-tts-edge-devices) is about.

![Pretrain vs finetune sequence format](/imgs/blogs/orpheus-tts-llm-speech-snac-6.png)

The figure contrasts the two sequence formats. During **pretraining**, the model sees text spans interleaved with `[SOA] audio [EOA]` (start/end-of-speech) blocks, *and* it sees plenty of text-only spans. That text-only data is not incidental — it is there deliberately to **preserve the language model's semantic reasoning**. A model that only ever saw "text → audio" pairs would slowly forget how to understand language, and its prosody would degrade because good prosody requires understanding what the sentence *means* (where the emphasis goes, whether it is a question, where the clause boundaries are). Keeping text in the mix is how Orpheus stays smart about language while learning to speak.

During **finetuning**, the format tightens. Each example wraps a single speaker's turn in control tokens: a start-of-human marker, the text (often prefixed with the speaker name, e.g. `tara: ...`), an end-of-text marker, then the `[SOA] audio [EOA]` block carrying the flattened SNAC tokens, all under the 8192-token context. The remarkable part is the data efficiency: the docs recommend a **minimum of ~50 examples** per speaker and call **~300 examples/speaker** the sweet spot for high quality. Because the heavy lifting (acoustic modeling, language understanding) is already done by pretraining, finetuning only has to pin down *identity and consistency*, which needs surprisingly little data.

A minimal training loop, using the offset-aware flattening from Section 3, looks like ordinary causal LM training:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "canopylabs/orpheus-3b-0.1-pretrained", torch_dtype=torch.bfloat16
).cuda()
tok = AutoTokenizer.from_pretrained("canopylabs/orpheus-3b-0.1-pretrained")

SOH, EOT, SOA, EOA = 128259, 128009, 128257, 128258  # control token ids (illustrative)

def build_example(speaker: str, text: str, wav_24k) -> list[int]:
    text_ids  = tok.encode(f"{speaker}: {text}", add_special_tokens=False)
    audio_ids = audio_to_orpheus_tokens(wav_24k)         # from Section 3
    return [SOH] + text_ids + [EOT, SOA] + audio_ids + [EOA]

batch = torch.tensor([build_example("tara", "Hello there.", wav)]).cuda()
loss = model(input_ids=batch, labels=batch).loss   # next-token CE over audio tokens too
loss.backward()                                     # no alignment loss, no duration loss
```

The only thing that distinguishes this from finetuning a chat model is the contents of the vocabulary. The loss is plain cross-entropy over the next token, including the audio tokens. The model learns alignment for free because predicting the right audio token at each step *is* the alignment problem, solved implicitly.

### Second-order optimization: catastrophic forgetting on tiny finetunes

The flip side of "50 examples is enough" is that **50 examples is also enough to overfit and forget.** If you finetune the 3B model on a handful of clips for too many epochs at too high a learning rate, it will nail your speaker's timbre and simultaneously lose its grip on emotion tags, pronunciation of rare words, and prosody on out-of-distribution sentences. The defenses are the usual ones, applied with more care than you would for text: keep the learning rate low (LoRA or a small full-finetune LR), cap epochs, hold out a few sentences with emotion tags as a validation probe, and *listen* to that probe every checkpoint. The failure mode is not a rising loss curve — the loss looks great — it is the model getting narrower. We saw the same dynamic across LoRA/PEFT runs generally; the audio case just makes the regression audible.

## 6. Streaming inference

**Senior rule of thumb: you do not wait for the whole sequence. You decode audio in a sliding window the moment you have enough frames, and the size of that window is the single knob that trades latency against artifacts.**

The reason Orpheus feels responsive is that it never waits for the full token sequence before producing sound. It rides vLLM's token-streaming API and decodes audio incrementally.

![Streaming: from token to first audio chunk](/imgs/blogs/orpheus-tts-llm-speech-snac-7.png)

The pipeline in the figure: vLLM generates tokens with `stream=True`, emitting each one as soon as it is sampled. A buffer accumulates tokens. The decoder fires whenever two conditions hold — the running count is a multiple of 7 (so we are on a frame boundary) *and* the count exceeds 27 (so we have a full sliding window). It then decodes `buffer[-28:]` — the trailing **28 tokens = 4 SNAC frames** — through SNAC, keeps the stable center slice `[2048:4096]`, applies a tiny fade at the seam, and emits a PCM chunk. The next fire reuses the overlapping context, which is what makes the seams gapless.

Here is the streaming consumer, close to the real `decoder.py` logic:

```python
import torch

def stream_audio(token_strings, snac):
    """token_strings: an async/sync generator of '<custom_token_N>' from vLLM."""
    buffer = []
    count = 0
    for tok_str in token_strings:
        code = turn_token_into_id(tok_str, count)   # Section 1
        if code is None:
            continue
        buffer.append(code)
        count += 1
        # Fire on every frame boundary once we have a full 28-token window.
        if count % 7 == 0 and count > 27:
            window = buffer[-28:]                    # last 4 frames
            layers = redistribute_codes(window)      # Section 4
            with torch.inference_mode():
                audio_hat = snac.decode([l.cuda() for l in layers])
            chunk = audio_hat[:, :, 2048:4096]       # stable center slice
            yield (chunk.squeeze().cpu().numpy() * 32767).astype("int16").tobytes()
```

The dual-window detail that real servers add: the *first* chunk uses a smaller window so the very first audio comes out fast (lower time-to-first-byte), then subsequent chunks use a larger batch (e.g. 210 tokens = 30 frames at a time) for throughput. This is the classic latency-versus-throughput trade made concrete — small first bite for responsiveness, big subsequent bites for efficiency. The bitbasti writeup of running this on a single RTX 3090 is a good worked example of the engineering; it also notes that Canopy's own sliding-window detokenizer eliminates the popping that a naive fade-in leaves behind.

There is also a backpressure subtlety that bites people who decode synchronously inside the token loop. If your SNAC decode runs on the same CUDA stream as the LLM generation, every decode call competes with the next token's forward pass for the GPU, and you can accidentally serialize the two so that generation stalls while audio is decoding. The clean architecture decouples them: let vLLM generate tokens onto a queue, and run the SNAC decoder as a separate consumer (a different thread, process, or even a second small GPU). Because the codec is so cheap (~5 ms per window) and runs at a fraction of the LLM's compute, this consumer never falls behind a single realtime stream — but the decoupling matters under concurrency, where many streams' decodes would otherwise interleave with many streams' forward passes and thrash the scheduler. Treat token generation and audio decoding as a producer/consumer pair connected by a bounded queue, and size the queue to a few frames so that a slow consumer applies backpressure rather than letting unbounded buffering inflate latency.

One more practical note on chunk sizing: the larger you make the steady-state window, the better your throughput but the coarser your ability to *interrupt*. In a conversational agent, the user may barge in mid-utterance, and you want to stop speaking promptly. A 30-frame steady-state chunk means up to ~2.5 seconds of already-decoded audio is queued ahead of the playhead and must be discarded on interruption. If barge-in latency matters for your product, keep the steady-state window smaller (say 8–12 frames) and accept the modest throughput cost — responsiveness to interruption is part of perceived latency too, and it does not show up in time-to-first-byte at all.

A note on serving: the project pins **`vllm==0.7.3`** because a later vLLM release introduced a regression for this model. This is the kind of pin you must respect — see the case studies. To stand up a server:

```bash
pip install "vllm==0.7.3" snac orpheus-speech    # required vLLM pin (see case study 3)
python -m vllm.entrypoints.openai.api_server \
  --model canopylabs/orpheus-3b-0.1-ft \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90    # fp8 weights fit LLM + SNAC on one 24 GB card
```

### Second-order optimization: generation throughput sets the realtime floor

Do the arithmetic. One 7-token frame decodes to ~85 ms of audio (the 2048-sample center slice at 24 kHz). For playback to keep up with generation — i.e. for *realtime* — the LLM must produce at least one frame's worth of tokens in 85 ms, which is $7 / 0.085 \approx 82$ audio tokens per second. Any modern serving setup runs a 3B model at hundreds of tokens per second, so a single stream is comfortably realtime with headroom to spare. The interesting constraint is *concurrency*: your realtime stream count is roughly `(tokens/sec your GPU sustains) / 82`. That is the number to put in your capacity planning spreadsheet, and it is why people reach for fp8 — not to make one stream faster, but to fit more concurrent streams per card.

## 7. The latency budget

**Senior rule of thumb: time-to-first-audio-byte is not "how long the utterance takes." It is prefill + the time to fill exactly one decode window + one SNAC decode. Optimize those three, ignore the rest.**

Orpheus advertises **~200 ms streaming latency, reducible to ~100 ms with input streaming.** Where does that number come from? The timeline figure breaks it into its actual components.

![Anatomy of first-audio-byte latency](/imgs/blogs/orpheus-tts-llm-speech-snac-8.png)

Walk it left to right:

1. **Prompt prefill** (~20–40 ms): the one-shot forward pass over the input text. This is FLOP-bound and scales with prompt length, which is why "input streaming" — feeding text as it arrives rather than waiting for the full prompt — shaves the latency down toward 100 ms. If you are piping the output of *another* LLM into Orpheus, start sending Orpheus the partial text immediately.
2. **First-frame fill** (~7 decode steps): the model must emit at least one complete 7-token frame before anything can be decoded.
3. **Window fill** (to 28 tokens / 4 frames): the sliding-window decoder waits for a full window before its first fire. This is the single biggest lever you control — a smaller first window cuts latency directly, at the cost of slightly more edge artifact in the very first chunk.
4. **First SNAC decode** (~5 ms): the convolutional decoder is tiny and fast. The codec is essentially never your bottleneck.
5. **First PCM byte to client** (~100–200 ms total): plus whatever your transport adds. Sending the WAV header *before* the first audio (so the client can start its decoder immediately) is a cheap trick worth doing.

The headline insight: **the codec is ~5 ms; everything else is the LLM and the window policy.** If you want lower latency, you tune prefill (input streaming), the first-window size, and your serving stack — not SNAC. This is the inverse of classical TTS, where the vocoder was often the long pole. Our general treatment of this trade lives in [Real-Time TTS: First-Audio-Byte Latency](/blog/machine-learning/signal-processing/real-time-tts-first-audio-byte-latency); Orpheus is a clean instance of those principles.

### Second-order optimization: latency is bimodal under load

The ~100–200 ms figure is for an unloaded or lightly loaded server. Under concurrency, vLLM batches requests, and your time-to-first-byte becomes sensitive to scheduling: a request that arrives just after a batch starts decoding waits for the next scheduler tick. The practical consequence is that p50 latency stays near the headline number while p99 can balloon if you oversubscribe the GPU past its realtime stream count. Size your fleet to the concurrency math from Section 6, set a hard cap on in-flight streams per replica, and shed load rather than letting every stream degrade. A queue that rejects fast is kinder than a server that makes everyone stutter.

## 8. Expressivity and zero-shot voice cloning

**Senior rule of thumb: everything expressive in Orpheus is in-context, not in-weights. Emotion is a text token; voice identity is a prefix. You steer the model by what you put in the prompt, not by retraining.**

Two of Orpheus's most-loved features fall directly out of the merged-vocabulary design.

**Emotion tags.** Because tags like `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, and `<gasp>` are ordinary text tokens sitting in the same stream as audio, the model learns during training to associate the tag with its acoustic realization. At inference you simply write them inline:

```python
prompt = "tara: I can't believe it actually worked <laugh> on the first try."
```

The model emits the audio tokens for a laugh at that position. No separate emotion encoder, no style embedding — the correlation is baked in by the training data, which contained those tags next to the corresponding sounds.

The depth of this mechanism is easy to underrate. Because the tag is a real token in the same autoregressive stream, its influence is not limited to the instant it appears — it conditions everything that follows until the context moves on. A `<sigh>` mid-sentence does not just insert a sigh; it nudges the prosody of the surrounding words toward a wearier delivery, because the model has learned that sighs co-occur with a particular emotional register. This is qualitatively different from a classical system where an emotion label is a fixed conditioning vector applied uniformly to an utterance. Here the emotion is *local and contextual*, exactly because it lives in the sequence. The practical upshot is that tag *placement* matters: the same `<laugh>` reads differently at the start, middle, or end of a clause, and you can shape delivery quite finely by where you drop tags. The limitation is the flip side — you only have the tags the model was trained on, and inventing a new tag like `<whisper>` does nothing unless that token appeared in training with whispered audio beside it. Expressivity is broad but not open-ended; it is the union of what the training data labeled.

**Zero-shot voice cloning.** This is the feature that most surprises people, because there is no speaker-embedding network anywhere.

![Zero-shot voice cloning by conditioning](/imgs/blogs/orpheus-tts-llm-speech-snac-9.png)

The recipe in the figure: take a few seconds of reference audio, SNAC-encode it into audio tokens, pair it with its transcript, and **prepend that (transcript, audio-token) pair to your target text** before generation. The model, being an autoregressive continuation engine, sees a completed example of "this voice saying these words" and continues the pattern — emitting your target text *in that voice*. There are zero gradient updates; it is pure in-context learning, the audio analogue of few-shot prompting. The pretrained model is better at this than the finetuned one, because finetuning narrows the model toward its fixed voices.

```python
def clone_prompt(ref_text, ref_wav_24k, target_text, speaker="custom"):
    ref_audio_ids = audio_to_orpheus_tokens(ref_wav_24k)   # Section 3
    ref_text_ids  = tok.encode(f"{speaker}: {ref_text}", add_special_tokens=False)
    tgt_text_ids  = tok.encode(f"{speaker}: {target_text}", add_special_tokens=False)
    # Reference turn (text + its audio), then the new text the model must voice.
    return ([SOH] + ref_text_ids + [EOT, SOA] + ref_audio_ids + [EOA]
            + [SOH] + tgt_text_ids + [EOT, SOA])  # model continues from here
```

### Second-order optimization: sampling parameters are part of the voice

Orpheus requires `repetition_penalty >= 1.1` for stable generation — without it the model can fall into a degenerate loop, repeating an audio token until the frame structure collapses. But repetition penalty and temperature do more than prevent collapse: **they perceptibly change the delivery.** Higher temperature and higher repetition penalty increase the *speech rate* and add variation; lower values produce flatter, slower, more deterministic speech. This means your sampling config is not a neutral knob — it is part of the voice's personality. Pin it per voice, treat it as a release artifact, and re-tune it whenever you finetune, because the right temperature for one speaker can sound rushed or robotic for another.

## 9. Finetuning your own speaker

**Senior rule of thumb: building the dataset is the whole job. Encoding audio to SNAC and flattening it is the exact mirror of inference — if you can decode, you can encode, and the training script is almost an afterthought.**

Putting Sections 3, 5, and 8 together, here is the end-to-end pipeline to make Orpheus speak in a new voice.

![Building a finetune dataset](/imgs/blogs/orpheus-tts-llm-speech-snac-10.png)

The stages in the figure: collect speaker WAVs with accurate transcripts → resample to 24 kHz mono (SNAC's expected input) → `SNAC.encode` each clip into its 3 layers → flatten to 7-token frames with the per-slot 4096 offset → wrap with the control markers and the audio offset `128266` → emit one training sequence per clip, capped at the 8192-token context. The mirror symmetry with inference is the whole point: the encode flattening here is byte-for-byte the inverse of the `redistribute_codes` you use to decode.

A few hard-won data rules:

| Knob | Cheap default | Why it matters |
|---|---|---|
| Examples per speaker | 50 minimum, 300 ideal | Below 50, identity is unstable; above 300, diminishing returns |
| Clip length | 3–20 s | Long clips blow the 8192-token budget; ~85 ms of audio ≈ 7 tokens, so 20 s ≈ 1650 audio tokens |
| Transcript accuracy | exact, with punctuation | The model learns prosody from punctuation; sloppy transcripts teach sloppy prosody |
| Sample rate | 24 kHz mono | SNAC is a 24 kHz model; resample first or you encode garbage |
| Emotion coverage | include tagged clips | If you want `<laugh>` to work in your voice, include laughs in the data |

Because the audio-token budget is roughly $7 \text{ tokens} \times 11.7 \text{ frames/s} \approx 82$ tokens per second of audio, a 20-second clip consumes about 1650 audio tokens — well within the 8192 context, but a 90-second monologue would not fit. Segment long recordings at sentence boundaries; do not truncate mid-word, because a half-frame at the end will misalign your `% 7` indexing for the whole example.

```bash
pip install "transformers>=4.40" peft datasets snac accelerate   # LoRA finetune deps
accelerate launch finetune_orpheus.py \
  --base canopylabs/orpheus-3b-0.1-pretrained \
  --data ./tara_dataset.jsonl \
  --use-lora --lora-r 16 --lora-alpha 32 \
  --lr 1e-4 --epochs 3 --max-seq-len 8192 \
  --bf16 --gradient-checkpointing
```

### Second-order optimization: the resample trap

The most common silent failure in Orpheus finetuning is feeding the codec audio at the wrong sample rate. SNAC's encoder was trained on 24 kHz; if you hand it 44.1 kHz or 16 kHz audio without resampling, it does not error — it just encodes a pitch-shifted, temporally-distorted version, and your model learns to speak in a chipmunk or a baritone that nobody asked for. Always resample to exactly 24 kHz with a real resampler (`torchaudio.functional.resample` or `soxr`), convert to mono, and normalize amplitude *before* `SNAC.encode`. Verify by decoding one encoded clip straight back and listening — if the round-trip sounds wrong, your data is wrong, and no amount of training will fix it.

## 10. Why a 3B language model at all

**Senior rule of thumb: the backbone size in an audio LLM does not buy you better audio fidelity — the codec caps that. It buys you better understanding of the text, which is where prosody actually comes from.**

The instinct from classical TTS is that a 3B-parameter model is absurd overkill for synthesizing speech — FastSpeech2 does respectable work with tens of millions of parameters. So why does Orpheus reach for a backbone two orders of magnitude larger? The answer is that the parameters are not spent on acoustics; they are spent on *language*. The acoustic ceiling is set entirely by SNAC, and you would hit that ceiling with a much smaller decoder. What the 3B Llama brings is everything the classical text frontend struggled with: knowing that "lead the team" and "lead pipe" are pronounced differently, that a sentence ending in a question mark rises, that "Dr. Smith lives on Main St." expands two abbreviations differently, that an em-dash signals a prosodic break, and that the emotional valence of a sentence should color its delivery.

This is the deepest reason the design works at all. Prosody is a *semantic* problem disguised as an acoustic one. The reason classical TTS sounds flat is not that its vocoder is bad — modern GAN vocoders are excellent — it is that its acoustic model has no idea what the sentence *means*, so it cannot place emphasis intelligently. By making the synthesizer a language model that was pretrained on text, Orpheus inherits a rich semantic prior for free, and then the text-interleaved pretraining (Section 5) keeps that prior alive while teaching the model to speak. The 3B size is the price of that semantic competence.

This framing also explains the roadmap. Canopy lists 1B, 400M, and 150M variants as planned releases. The bet behind the smaller models is that *most* of the semantic competence needed for everyday TTS survives distillation to a smaller backbone — you do not need full 3B language understanding to read a weather forecast naturally. The smaller models trade some robustness on hard, ambiguous, or out-of-distribution text for a dramatically smaller footprint, which is exactly the trade you want on an edge device. The 3B model is the quality anchor; the smaller ones are the deployment story. Expect the 150M model to stumble on the gnarly pronunciation cases the 3B handles effortlessly, and to need more finetuning data to pin down a voice — but to fit comfortably where a 3B never could.

There is a second, subtler payoff to a large backbone: **robustness to messy input.** Real production text is full of URLs, emoji, code snippets, mixed languages, and typos. A tiny acoustic model handed "see https://x.com 😀" produces garbage; a language-model backbone has seen enough of the web to make a reasonable decision about how (or whether) to vocalize it. You are buying graceful degradation on the long tail of inputs, which in a real product is worth more than another decibel of fidelity on clean text.

### Second-order optimization: do not over-index on parameter count

The flip side: because audio fidelity is codec-bound, throwing a bigger backbone at a quality complaint is usually the wrong move. If a specific phoneme sounds wrong, that is a codec or data problem, not a capacity problem, and a larger model will not fix it. The right diagnostic ladder is: (1) codec round-trip on ground truth — is the ceiling itself the problem? (2) sampling parameters — is it a stability/rate issue? (3) training data coverage — did the model ever see this kind of text? Parameter count comes last, and usually not at all. Spend your effort on data and codec, not on scaling the backbone past what your latency and memory budgets allow.

## Case studies from production

These are the failure modes that actually show up when teams put Orpheus into a product. Each is a real pattern with a concrete fix.

### 1. The popping seam

**Symptom.** Streaming output has a faint click or pop every ~85 ms, like a vinyl record with regular scratches. The transcript is perfect and the voice is right, but the artifact is maddening on headphones.

**Wrong first hypothesis.** "The model is emitting bad audio tokens" — engineers go hunting for malformed frames and find none.

**Root cause.** Naive streaming decodes each frame independently and concatenates the raw output. The SNAC decoder's convolutions produce unreliable samples at window edges, and stitching those edges together creates a discontinuity at every frame boundary. The pop is a decode artifact, not a model artifact.

**Fix.** Decode a sliding window (`buffer[-28:]`) and keep only the stable center slice `[2048:4096]`, exactly as in Section 6. The overlapping context means each emitted chunk comes from the middle of a window where the receptive field is fully populated. Canopy's detokenizer takes this further with a proper sliding-window overlap that is gapless; a simple fade-in at the seam (what the bitbasti writeup does) is a serviceable approximation. **Lesson:** in codec streaming, *never* decode and concatenate frame-by-frame — overlap-and-discard is mandatory, not optional.

### 2. Repetition collapse

**Symptom.** Generation starts fine, then degenerates into a stuck sound — a held vowel, a repeated syllable, or a buzz that never ends. The token stream shows the same audio token repeating.

**Wrong first hypothesis.** "The model is broken" or "the finetune corrupted it."

**Root cause.** Greedy or low-penalty sampling lets the autoregressive model fall into a fixed point where the most-likely next token reproduces the current state, looping forever. Audio tokens are especially prone to this because adjacent frames are genuinely similar, so the model's self-reinforcement is strong.

**Fix.** Set `repetition_penalty >= 1.1` — this is documented as a hard requirement, not a suggestion. The penalty breaks the loop by down-weighting recently emitted tokens. Combine with a modest temperature (e.g. 0.6–0.8) and top-p. **Lesson:** sampling parameters in an audio LLM are stability mechanisms first and aesthetic knobs second; ship them as part of the model config, never leave them to caller defaults.

### 3. The vLLM 0.7.3 pin

**Symptom.** Team upgrades vLLM to the latest release for an unrelated feature; Orpheus output becomes garbled or the server throws shape errors on the extended vocabulary.

**Wrong first hypothesis.** "Our model files are corrupted, re-download them."

**Root cause.** A post-0.7.3 vLLM release changed behavior in a way that breaks Orpheus's serving path (the project explicitly pins `vllm==0.7.3` and references a March-2024-era regression). The extended-vocabulary model exercises code paths that the upstream change did not account for.

**Fix.** Respect the pin: `pip install "vllm==0.7.3"`, and isolate Orpheus in its own environment or container so an unrelated dependency upgrade cannot drag vLLM forward. Track the upstream issue if you need a newer vLLM for other reasons, and re-test audio quality byte-for-byte before bumping. **Lesson:** for models that ride a fast-moving inference engine, the engine version is part of the model's contract. Pin it, containerize it, and treat an upgrade as a model change requiring re-validation.

### 4. fp8 to fit one card

**Symptom.** A 3B LLM plus the SNAC codec in bf16 spills past 24 GB on an RTX 3090, forcing a second GPU and doubling deployment cost.

**Wrong first hypothesis.** "We need an A100" — the team provisions expensive hardware for a model that should fit on a gaming card.

**Root cause.** bf16 weights for a 3B model are ~6 GB, but KV cache for 8192-context streams plus activations plus the codec push total VRAM over the 24 GB line under any real concurrency.

**Fix.** Serve fp8 weights (Baseten and several providers offer fp8 Orpheus). fp8 roughly halves weight memory and frees enough headroom to fit both the LLM and SNAC on a single 24 GB card with room for several concurrent streams, at negligible quality loss for this model. **Lesson:** for TTS the perceptual tolerance for weight quantization is high — the codec already band-limits quality — so fp8 is close to free. Measure with a codec round-trip baseline (Section 2) to confirm the quantization sits below the codec's own error floor.

### 5. Speech-rate drift from temperature

**Symptom.** After tuning sampling for variety, the voice starts talking noticeably faster, sometimes rushing to the point of clipping word boundaries.

**Wrong first hypothesis.** "The finetune changed the speaking rate" — but the same weights sounded fine yesterday.

**Root cause.** Higher temperature and higher repetition penalty increase speech rate in Orpheus, a documented interaction. A change made to add expressiveness silently sped the voice up.

**Fix.** Treat temperature and repetition penalty as a coupled pair tuned *per voice*, and validate speaking rate (words per minute on a fixed test sentence) as a regression metric whenever you touch them. Pin the values in the voice's config. **Lesson:** sampling parameters are part of the voice identity, not global defaults — a config that sounds great for `dan` may sound frantic for `tara`.

### 6. The 16 kHz dataset

**Symptom.** A freshly finetuned voice sounds subtly pitched-up and "off," like the speaker inhaled helium, even though the source recordings sounded normal.

**Wrong first hypothesis.** "The model can't capture this speaker" — the team collects more data, which does not help.

**Root cause.** The training audio was 16 kHz (typical of phone-quality corpora) and was fed to SNAC without resampling. SNAC expects 24 kHz; handed 16 kHz samples interpreted as 24 kHz, it encodes a time-compressed, pitch-shifted signal, and the model faithfully learns that distortion.

**Fix.** Resample every clip to exactly 24 kHz mono before encoding, with a proper anti-aliasing resampler. Add a guard in the data pipeline that asserts the sample rate and rejects anything that is not 24 kHz. **Lesson:** the codec's expected input format is a hard contract; a sample-rate mismatch fails silently and poisons the entire dataset. Always round-trip one clip and listen before launching a training run.

### 7. p99 latency under concurrency

**Symptom.** Demo latency is a crisp 150 ms, but in production under load the p99 time-to-first-byte jumps to over a second and users complain about lag.

**Wrong first hypothesis.** "The model got slower" — but p50 is unchanged.

**Root cause.** The GPU is oversubscribed past its realtime stream count (Section 6). vLLM batches requests, so a stream arriving mid-batch waits for the next scheduler tick, and when in-flight streams exceed `sustained_tokens_per_sec / 82`, the tail latency explodes even while the median looks healthy.

**Fix.** Compute the realtime stream capacity per replica from measured token throughput, cap in-flight streams below it, and shed or queue excess load rather than admitting it. Scale horizontally on stream count, not on average utilization. **Lesson:** for realtime audio, the right capacity metric is concurrent realtime streams, not GPU percent-busy; a card at 70% average utilization can still miss deadlines on the tail.

### 8. The truncated final frame

**Symptom.** Generated clips end with a tiny click or a clipped final syllable, as if the audio were cut a few milliseconds short.

**Wrong first hypothesis.** "The model is stopping too early" — engineers raise the max-token limit, which does nothing.

**Root cause.** The token count at the end of generation was not a multiple of 7, so the final partial frame was either dropped or fed to `redistribute_codes` with a misaligned boundary. A partial frame cannot be regrouped into the 1-2-4 layer structure, and the streaming decoder's `count % 7 == 0` gate never fired for the last few tokens.

**Fix.** When generation ends, pad or trim the buffer to the nearest lower multiple of 7 before the final decode, and make sure the model's stop token (`[EOA]`) is emitted on a frame boundary during training so it learns to end cleanly. On the consumer side, always flush a final aligned window rather than whatever is left over. **Lesson:** the frame size 7 is an invariant that must hold at *both* ends of the stream — the start (window fill) and the end (final flush). Off-by-non-7 at either boundary produces audible defects.

### 9. Emotion tags ignored after finetuning

**Symptom.** The base model laughs on `<laugh>`, but after finetuning on a custom voice, the tag is read as silence or has no effect.

**Wrong first hypothesis.** "The tag token got remapped" — the vocabulary is unchanged.

**Root cause.** The finetuning dataset contained no clips with emotion tags, so the model — narrowing toward the new voice — forgot the tag-to-acoustic association it learned in pretraining. This is the catastrophic-forgetting dynamic from Section 5, localized to a specific capability.

**Fix.** Include tagged clips in the finetuning data (a handful of genuine laughs, sighs, etc. in the target voice), keep the learning rate low, and use a held-out tagged sentence as a per-checkpoint probe. If you cannot record tagged audio, mix a small fraction of the original pretraining-style data back in to anchor the capability. **Lesson:** finetuning preserves only what the data exercises; any capability you want to keep must appear in the finetune set, or it will quietly erode.

### 10. Multilingual mispronunciation

**Symptom.** Using an English Orpheus model on Spanish or French text produces confident, fluent-sounding speech that is pronounced with an English accent and occasionally wrong.

**Wrong first hypothesis.** "TTS is language-agnostic, the codec just makes sound" — but the language model prior is very much English.

**Root cause.** The English model's text understanding and learned grapheme-to-acoustic mappings are English. SNAC can represent the sounds, but the backbone does not know how the target language maps text to those sounds, so it improvises from its English prior.

**Fix.** Use the appropriate model from the multilingual family — Canopy ships pretrained/finetuned pairs for several languages — or finetune on target-language data. Do not expect cross-lingual transfer from the English model for anything beyond loanwords. **Lesson:** in an audio LLM the language competence lives in the backbone, not the codec; the codec is language-agnostic but the model is not. Match the model to the language.

### 11. Transport buffering eats the latency win

**Symptom.** The server measures ~150 ms time-to-first-byte, but the browser client does not start playing for nearly a second.

**Wrong first hypothesis.** "The model or GPU is slow" — but server-side metrics are fine.

**Root cause.** The client-side audio stack buffers. A `MediaSource` or Web Audio pipeline that waits to accumulate a few hundred milliseconds before starting playback, or an HTTP layer that buffers until a chunk threshold, silently erases the server's latency advantage. The WAV header arriving late compounds it.

**Fix.** Send the WAV header immediately, before the first audio bytes, so the client can initialize its decoder without waiting. Use a streaming-friendly transport (chunked HTTP, WebSocket, or WebRTC) and configure the client for minimal jitter buffer. Measure end-to-end (mouth-to-ear), not just server-side. **Lesson:** the ~100–200 ms figure is a *server* number; the user's experience is dominated by transport and client buffering, which you must tune with the same discipline you applied to the model.

## Comparing Orpheus to the alternatives

It helps to place Orpheus on the map of modern open TTS. The systems differ less in raw quality on clean text — most are good now — and more in *what they make easy*. The table below is the comparison I use when advising a team on which to adopt.

| System | Backbone | Audio representation | Cloning | Streaming | Expressivity | Footprint |
|---|---|---|---|---|---|---|
| Orpheus | Llama-3B LLM | SNAC discrete tokens | Zero-shot, in-context | Native, ~100–200 ms | Inline emotion tags, strong prosody | Heavy (3B; fp8 fits 24 GB) |
| [VITS / VITS2](/blog/machine-learning/signal-processing/vits-vits2-end-to-end-tts) | Conditional VAE + flow | Continuous latent | Limited (speaker id) | Possible, non-trivial | Decent, no tag control | Light (tens of M) |
| [FastSpeech2](/blog/machine-learning/signal-processing/fastspeech2-vs-tacotron2) + vocoder | Non-autoregressive | Mel-spectrogram | Via speaker embedding | Fast, chunkable | Flat without extra conditioning | Very light |
| XTTS-style | GPT-style LM | Discrete codec tokens | Zero-shot from clip | Yes | Good | Medium |
| Bark-style | GPT-style LM | EnCodec tokens | Preset voices | Limited | Very expressive, less controllable | Medium-heavy |

The pattern is clear: the **LM-backbone codec-token** family (Orpheus, XTTS, Bark) wins on expressiveness and zero-shot cloning because in-context conditioning is natural for a language model, while the **classical non-autoregressive** family (FastSpeech2, lighter VITS configs) wins on footprint and determinism. Orpheus's distinguishing moves within the LM family are (1) the *hierarchical* SNAC codec with its 7-token frame, which makes single-pass streaming clean, (2) the explicit text-interleaved pretraining that protects semantic competence, and (3) the deliberate decision to ride stock vLLM, which gives it the best latency-and-ops story of the group. If your team already operates an LLM serving stack, that third point is decisive — Orpheus is the option that does not require a parallel, bespoke audio-serving system.

The honest caveat is that "best at X" is a moving target in open TTS; new models ship monthly. The durable insight is architectural: once you represent audio as codec tokens, TTS becomes a sequence-modeling problem, and every advance in LLM training and serving — better optimizers, longer context, faster kernels, cheaper quantization — flows into speech automatically. That compounding is why the LM-backbone approach is likely to keep gaining ground regardless of which specific model is on top this quarter.

## Evaluating an audio LLM

**Senior rule of thumb: you cannot eyeball audio quality, and a single MOS number hides the failures that matter. Evaluate intelligibility, identity, and latency as three separate axes, each with a metric you can regression-test in CI.**

Subjective listening is essential but insufficient — it does not scale and it does not catch regressions. A real Orpheus deployment needs an automated evaluation harness across three axes:

**1. Intelligibility (does it say the right words?).** Run the generated audio back through a strong ASR model — [Whisper](/blog/machine-learning/signal-processing/whisper-under-the-hood) is the standard choice — and compute Word Error Rate against the input text. A rising WER on a fixed test set is the earliest, most objective signal that a finetune or config change broke something. This catches repetition collapse, dropped words, and mispronunciations that a quick listen might miss. Keep a frozen set of ~100 sentences spanning easy text, abbreviations, numbers, and emotion tags, and gate releases on WER not regressing.

```python
import whisper
asr = whisper.load_model("large-v3")

def wer_check(text: str, wav_path: str) -> float:
    hyp = asr.transcribe(wav_path)["text"]
    ref_words, hyp_words = text.lower().split(), hyp.lower().split()
    # Levenshtein over words; use jiwer in practice.
    from jiwer import wer
    return wer(text.lower(), hyp.lower())
```

**2. Speaker identity (does it sound like the right voice?).** For cloned or finetuned voices, extract a speaker embedding (a pretrained verification model like ECAPA-TDNN) from both a reference clip and the generated audio, and compute cosine similarity. A drop in similarity flags identity drift — the voice wandering away from its target — which is the most common finetuning regression and one that WER completely misses. Track it per voice as a release metric.

**3. Latency (does it stream fast enough?).** Measure time-to-first-audio-byte and the per-chunk cadence under realistic concurrency, reporting p50 and p99 (Section 7 explained why the tail is the number that matters). A latency regression often comes not from the model but from a dependency bump or a config change — exactly the vLLM-pin and concurrency cases above — so this axis catches operational regressions the quality metrics never would.

The fourth, unavoidable axis is the **codec round-trip baseline** from Section 2: periodically encode and decode ground-truth audio with SNAC alone, with no model in the loop, and measure *its* WER and similarity. This tells you the ceiling. If your model's WER is close to the codec's round-trip WER, the model is doing as well as it possibly can, and further effort should go into the codec or the data, not the model. Without this baseline you will waste weeks trying to push past a ceiling that is not the model's fault.

### Second-order optimization: evaluate the tags and the seams, not just the words

Two failure modes hide from the standard metrics. First, emotion tags: WER does not penalize a model for ignoring `<laugh>`, because the laugh is non-lexical. Add a small set of tagged probes and check (by ASR confidence dips, or by a lightweight audio-event classifier) that the non-speech sound actually appears. Second, the streaming seams: a model can score perfectly on offline-generated audio while the *streamed* version pops at every chunk boundary because of a decoder bug. Always run the evaluation harness on the real streaming path, not just on a single batch decode, or you will ship artifacts your offline metrics swore were not there.

## When to reach for Orpheus / when not to

### Reach for Orpheus when…

- You need **genuinely expressive, emotive speech** — laughs, sighs, natural prosody — and the robotic flatness of a classical pipeline is a dealbreaker.
- You want **zero-shot or few-shot voice cloning** without training a speaker-embedding network or collecting hours of data per voice.
- You are building a **streaming, conversational** application where ~100–200 ms time-to-first-byte matters and you can ride vLLM.
- You want **inline control** over delivery via text tags rather than out-of-band conditioning signals.
- You already run an LLM serving stack and want TTS that **slots into the same infrastructure** (same engine, same quantization, same ops).
- Your licensing requires **Apache-2.0** and self-hostability.

### Skip Orpheus when…

- You need **sub-50 ms latency on a tiny CPU/edge device** — a 3B autoregressive model is heavy; a small non-autoregressive model like [FastSpeech2](/blog/machine-learning/signal-processing/fastspeech2-vs-tacotron2) or a compact [VITS](/blog/machine-learning/signal-processing/vits-vits2-end-to-end-tts) will win on raw footprint (though the promised 150M/400M Orpheus variants may change this).
- You need **deterministic, identical output** every run for a regulated or contractual reason — autoregressive sampling is inherently variable, and even greedy decoding is brittle for audio.
- Your domain is **far outside the training distribution** (heavy code-switching, singing, non-speech audio) and you cannot finetune — the language-model prior that helps on normal speech does not transfer for free.
- You are **GPU-constrained to the point** that you cannot fit even an fp8 3B model plus codec, and a 30M-parameter classical model would do.
- You need **fine-grained, frame-level prosody control** (exact pitch contours, precise timing) — Orpheus gives you expressive but not surgically controllable prosody.

The honest summary: Orpheus is the right tool when *expressiveness, cloning, and conversational latency* are the priorities and you have a GPU; the classical non-autoregressive stack is still the right tool when *footprint, determinism, and surgical control* dominate. Pick by what your product actually needs to be good at, not by which architecture is fashionable — the failure mode I see most often is teams adopting an LLM-backbone TTS for a use case that wanted a tiny deterministic model, then fighting latency and variance for months. The fact that "treat audio as tokens" is now competitive with bespoke acoustic models at all is the real headline — it means the relentless progress in LLM serving (paged attention, fp8, speculative decoding) now flows directly into TTS, for free.

## Further reading

- [Orpheus-TTS repository](https://github.com/canopyai/Orpheus-TTS) — the source, model cards, and finetuning notebooks.
- [SNAC: Multi-Scale Neural Audio Codec](https://github.com/hubertsiuzdak/snac) and the [SNAC paper](https://arxiv.org/abs/2410.14411) — the codec that makes the whole thing possible.
- [Speech Tokenizers: EnCodec, SoundStream, Mimi](/blog/machine-learning/signal-processing/speech-tokenizers-encodec-soundstream-mimi) — the family SNAC belongs to.
- [Real-Time TTS: First-Audio-Byte Latency](/blog/machine-learning/signal-processing/real-time-tts-first-audio-byte-latency) — the latency principles Orpheus instantiates.
- [VITS and VITS2: End-to-End TTS](/blog/machine-learning/signal-processing/vits-vits2-end-to-end-tts) — the strong non-LLM baseline to compare against.
- [HiFi-GAN](/blog/machine-learning/signal-processing/hifi-gan) — the GAN-vocoder world Orpheus replaces with a frozen codec decoder.
