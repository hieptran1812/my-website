---
title: "Zero-Shot Voice Cloning and the TTS Frontier"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Clone any voice from a few seconds of audio: the two recipes (codec language models vs flow matching), the open models you can run today, the closed frontier, and how to measure a clone honestly."
tags:
  [
    "audio-generation",
    "audio-synthesis",
    "text-to-speech",
    "voice-cloning",
    "zero-shot-tts",
    "flow-matching",
    "generative-ai",
    "deep-learning",
    "neural-audio-codec",
  ]
category: "machine-learning"
subcategory: "Audio Generation"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/zero-shot-voice-cloning-and-the-tts-frontier-1.png"
---

The first time I cloned a voice I did not believe the result. I recorded six seconds of a colleague saying something about lunch, fed it to a model along with a paragraph she had never spoken, and out came her voice reading my paragraph. The cadence was hers. The little upward lilt she puts on the end of questions was there. It was unsettling enough that I played it for her, and she made me delete it. That model was not a research artifact behind a paywall. It was an open checkpoint I downloaded in ten minutes, running on a laptop GPU. We have crossed a line in the last three years that most people have not noticed yet: **synthesizing a believable copy of a specific human voice now takes a few seconds of reference audio and no training at all.**

That capability is what this post is about. Not text-to-speech in general — we covered the road from Tacotron to VITS in [text-to-speech-from-tacotron-to-vits](/blog/machine-learning/audio-generation/text-to-speech-from-tacotron-to-vits), and the specific codec-language-model reframing in [neural-codec-language-model-tts-vall-e](/blog/machine-learning/audio-generation/neural-codec-language-model-tts-vall-e). This post is about the **frontier**: the systems people actually use in 2024–2026 to clone a voice instantly and synthesize speech that holds up under a critical ear. We will look at the two dominant recipes that power them, the open models you can download and run tonight, the closed services that still lead on polish, and — this is the part most tutorials skip — how to tell whether a clone is *actually* good rather than merely impressive on first listen.

By the end you will be able to run a zero-shot clone with open tooling, measure its speaker similarity and intelligibility with real metrics instead of vibes, reason about whether a codec-LM or a flow-matching model fits your latency budget, and articulate the safety problem that sits underneath all of this. The frame from the rest of this series still holds: the audio stack is **waveform → codec tokens or latent → generative model → vocoder/decoder → waveform**, pulled in four directions by **fidelity, controllability, speed, and length**. Voice cloning is the place where *controllability* gets sharpest — the thing you are controlling is a person's identity — so the trade-offs get sharp too.

![A dataflow diagram showing a reference waveform and target text feeding a speaker representation and text tokens into a TTS model that produces a cloned waveform, with a speaker-verification embedder computing a similarity cosine](/imgs/blogs/zero-shot-voice-cloning-and-the-tts-frontier-1.png)

## What "zero-shot" actually means here

Let me pin down the words, because "voice cloning" is used loosely and the distinctions matter for both engineering and ethics.

**Single-speaker TTS** bakes one voice into the model weights. You collect hours of clean studio audio from a voice actor, train (or fine-tune) a model on it, and the model can now say anything *in that one voice*. This is how audiobook voices and brand voices were historically built. The voice *is* the model. Want a different voice? Train a different model.

**Multi-speaker TTS** trains one model on many speakers, with a speaker ID or a learned speaker embedding selecting among them at inference. You can switch voices, but only among the voices seen during training.

**Few-shot voice cloning** fine-tunes a base model on a small amount of a new speaker's audio — minutes, sometimes seconds — to adapt it to that voice. There is still a training step per voice, just a cheap one.

**Zero-shot voice cloning** — the subject here — does no training per voice at all. You give the model a short *reference* (a "prompt", a "speaker reference", an "enrollment clip") of a few seconds of the target voice, plus the text you want spoken, and it synthesizes that text in that voice in a single forward pass. The voice is supplied *in context*, the way a few examples are supplied in context to a large language model. The weights never change. That is the whole trick, and it is why the field exploded: you decoupled "which voice" from "the model", so one model serves every voice on Earth, including ones recorded thirty seconds ago.

It helps to see how fast this happened. Five years ago, "voice cloning" meant the few-shot regime — projects like SV2TTS ("Transfer Learning from Speaker Verification to Multispeaker TTS", 2018) bolted a speaker-verification embedding onto Tacotron and could approximate a new voice from a few seconds, but the quality was clearly synthetic and the similarity modest. The leap to convincing *zero-shot* cloning came with two ingredients maturing together: high-fidelity **neural audio codecs** (SoundStream 2021, EnCodec 2022) that turned audio into a tractable token sequence, and the **in-context-learning** mindset from large language models. VALL-E (early 2023) fused them — "TTS is just an LM over codec tokens, and cloning is just prompting" — and within eighteen months the open ecosystem (XTTS v2, F5-TTS) and the closed frontier (ElevenLabs) had made instant, believable cloning a commodity. The capability went from research curiosity to `pip install` in barely two years. That speed is exactly why the safety conversation later in this post is not optional.

![A before-and-after comparison contrasting single-speaker TTS that fine-tunes weights per voice against zero-shot cloning that supplies the voice as an inference-time prompt with frozen weights](/imgs/blogs/zero-shot-voice-cloning-and-the-tts-frontier-2.png)

The mechanism in Figure 1 is the spine of every system in this post. A reference waveform is encoded into a **speaker representation** — sometimes an explicit fixed-length vector, sometimes the raw acoustic tokens of the reference used directly as a prompt. The target text becomes a sequence of tokens (characters, phonemes, or sub-word units). The model fuses the two and produces a **cloned waveform** carrying the reference's identity but the target's content. Off to the side sits the part nobody shows in demos: a **speaker-verification embedder** that turns both the reference and the output into identity vectors and computes their cosine, giving you a number for "how much does this sound like the same person." We will return to that number repeatedly, because it is the difference between a demo and an evaluation.

What does "good" mean for such a system? Four axes, and a model can be strong on some and weak on others:

- **Speaker similarity** — does it sound like the *target* person? Measured automatically with a speaker-verification cosine (often called SIM) and by humans as SMOS (Similarity Mean Opinion Score).
- **Naturalness** — does it sound like a *human at all*, free of robotic flatness, buzzing, or glitches? Measured by humans as MOS (Mean Opinion Score), 1–5.
- **Intelligibility** — can you understand the words; did the model drop, repeat, or mangle any? Measured automatically as **WER** (Word Error Rate) by running an ASR system on the output and comparing to the target text.
- **Multilinguality, latency, style control, and license** — the practical axes: how many languages, how fast (real-time factor, RTF), can you steer emotion, and can you legally ship it.

Hold those four in mind. Every comparison table later is just these axes filled in.

## The two recipes: codec-LM versus flow matching

Almost every frontier zero-shot cloner is one of two architectures, and the split is the single most useful thing to understand about this field. I will set them side by side, then go deep on each.

![A taxonomy tree splitting zero-shot TTS into an autoregressive codec-language-model branch with VALL-E and XTTS, and a non-autoregressive flow-matching branch with Voicebox, F5-TTS, and NaturalSpeech 3](/imgs/blogs/zero-shot-voice-cloning-and-the-tts-frontier-3.png)

### Recipe A — the codec language model (autoregressive)

The first recipe treats speech generation as **language modeling over discrete audio tokens**. It builds directly on neural audio codecs (see [neural-audio-codecs-the-tokenizer-of-sound](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound) and [residual-vector-quantization-rvq](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq)). A codec like EnCodec turns a waveform into a short stream of discrete tokens — a few hundred tokens per second instead of tens of thousands of raw samples. Once audio is a token sequence, you can do to it exactly what GPT does to text: train a transformer to predict the next token.

VALL-E (Wang et al., 2023) was the model that made this click for zero-shot cloning, and it is its own post: [neural-codec-language-model-tts-vall-e](/blog/machine-learning/audio-generation/neural-codec-language-model-tts-vall-e). The cloning move is beautiful in its simplicity. To clone a voice, you encode the reference waveform into acoustic tokens and *prepend them to the sequence as a prompt*, followed by the target phonemes. Then you let the model autoregressively continue the acoustic-token stream. Because the model has learned that audio tokens are locally coherent — the next frame should sound like the previous frames — its continuation naturally inherits the timbre, the recording conditions, even the emotional tone of the prompt. The voice is cloned not by extracting a speaker vector but by *in-context learning*, the same way an LLM continues "Q: ... A: ..." in the style of the examples you gave it.

![A dataflow diagram showing a reference waveform encoded to RVQ tokens and target text phonemized, both fed to an autoregressive transformer that samples acoustic tokens decoded back to a cloned waveform](/imgs/blogs/zero-shot-voice-cloning-and-the-tts-frontier-5.png)

Figure 5 traces it. The reference is encoded to acoustic tokens (the "prompt"), the text is phonemized, the autoregressive transformer is conditioned on both and samples acoustic tokens one frame at a time, and a codec decoder turns those tokens back into a waveform. The open model **XTTS v2** from Coqui follows this family — it pairs a text encoder, a GPT-style autoregressive decoder over audio codes conditioned on a speaker-embedding extracted from the reference, and a decoder/vocoder to produce the waveform. We will run it in a minute.

The strengths and weaknesses of this recipe follow directly from "it is an autoregressive language model over audio":

- **Strength: in-context flexibility.** It does not need an explicit alignment or duration model. The prompt carries the voice, the prosody, the recording vibe — all of it — for free, the way ICL works for text.
- **Strength: it captures fine acoustic detail and natural prosody**, because it samples from a learned distribution over real audio tokens rather than predicting a deterministic spectrogram.
- **Weakness: it is sequential.** Generation is O(T) forward passes for T audio frames — you cannot parallelize over time, so it is the slower recipe and its latency grows with output length.
- **Weakness: it can be unstable.** Like any autoregressive sampler, it can hallucinate: repeat a word, skip a word, run on past the end, or babble when the prompt is noisy. Production codec-LMs spend real engineering on stopping criteria, repetition penalties, and sometimes a second pass to catch these.

Let me make the science precise, because "it is just an LM over audio" hides a real subtlety. A codec like EnCodec or DAC produces, for each audio frame $t$, not one token but a *stack* of $Q$ residual tokens $c_t^{1}, c_t^{2}, \ldots, c_t^{Q}$ — the first codebook captures the coarse signal, each subsequent codebook quantizes the residual the previous ones left behind (this is the residual-VQ structure we derive in [residual-vector-quantization-rvq](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq)). So the model is not factorizing a single sequence; it is factorizing a 2D grid of tokens (time $\times$ codebook depth). The joint distribution it learns is

$$p(c \mid \text{text}, \text{prompt}) = \prod_{t=1}^{T} \prod_{q=1}^{Q} p\big(c_t^{q} \mid c_{<t}, c_t^{<q}, \text{text}, \text{prompt}\big),$$

and the *order* in which you visit the $T \times Q$ cells is a design choice with real consequences. VALL-E's answer is a two-stage split: an autoregressive model for the first codebook ($q = 1$) across all frames (this carries most of the prosody and timing and must be sampled left-to-right), then a non-autoregressive model that fills in codebooks $q = 2 \ldots Q$ in parallel given the first. MusicGen-style interleaving and "delay" patterns are alternative visit orders. The practical point: a "codec-LM" is rarely fully sequential over all $T \times Q$ tokens — the engineering is largely about which tokens you can predict in parallel without hurting quality, because that ratio sets your latency. The naive $T \times Q$ sequential bound in the worked example below is the worst case; real systems claw most of it back.

### Recipe B — flow matching / non-autoregressive

The second recipe abandons left-to-right token generation. It generates the whole utterance's acoustic representation **in parallel** by learning a continuous transformation from noise to speech features, using **flow matching** — the same machinery we derived for images in [flow-matching-and-rectified-flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow) and adapted to audio in [flow-matching-and-consistency-for-audio](/blog/machine-learning/audio-generation/flow-matching-and-consistency-for-audio). Rather than re-derive flow matching, here is the one-line reminder: you train a network to predict a velocity field that transports a sample of Gaussian noise to a sample of data along a straight-ish path, and at inference you integrate that field with a handful of ODE steps. The output is the *entire* mel-spectrogram (or codec latent) at once — no per-frame loop.

To make the cloning mechanism concrete, here is the flow-matching objective in its conditional-infilling form, which is what makes it clone. You have speech features $x$ (a mel-spectrogram or codec latent). Flow matching trains a network $v_\theta$ to regress a velocity along a straight path from noise $x_0 \sim \mathcal{N}(0, I)$ to data $x_1 = x$, with the interpolant $x_\tau = (1-\tau)x_0 + \tau x_1$ for $\tau \in [0,1]$. The target velocity for that straight path is simply $x_1 - x_0$, so the loss is

$$\mathcal{L}_\text{FM} = \mathbb{E}_{\tau,\, x_0,\, x_1}\big\lVert v_\theta(x_\tau, \tau, \text{cond}) - (x_1 - x_0)\big\rVert^2 .$$

We derive this in full for images in [flow-matching-and-rectified-flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow); the audio-specific part is the conditioning. For cloning, the trick is **masked infill**. Voicebox (Le et al., 2023) framed it this way: train the model to fill in a masked span of speech given the surrounding (unmasked) speech and the full text. Formally, the `cond` above includes the *unmasked* part of the real speech $x \odot (1 - m)$ for a mask $m$, plus the text aligned to the full timeline. The model only has to generate the masked region $x \odot m$, conditioned on the rest. At inference, you set the *entire reference* as unmasked context and mask the *new* region you want spoken; integrating the ODE $\frac{dx}{d\tau} = v_\theta(x_\tau, \tau, \text{cond})$ from noise produces speech in the masked region that is acoustically consistent with the unmasked reference — same voice, new words. It is the audio analogue of image inpainting, and because the reference and the target live in *one* generated sequence, the timbre carries over without any explicit speaker vector.

F5-TTS (Chen et al., 2024) and the closely related E2-TTS pushed this to its cleanest form: **alignment-free flow matching**. Earlier non-autoregressive TTS (FastSpeech, and Voicebox to a degree) needed an explicit *duration model* — something that decides how many audio frames each phoneme gets, because a non-autoregressive model must lay out the whole timeline up front. F5-TTS dispenses with the separate forced-aligner and duration predictor: it pads the text to the audio length and lets the model learn the alignment implicitly through the flow-matching objective, with a small architectural trick (a "ConvNeXt" text refinement and infilling-style training). The result is a strikingly simple pipeline that clones well and trains stably.

![A before-and-after comparison contrasting autoregressive codec-LMs that decode token-by-token with flexible prompting but stability risk against flow-matching NAR models that infill in parallel with stable, faster output but explicit duration](/imgs/blogs/zero-shot-voice-cloning-and-the-tts-frontier-7.png)

Figure 7 lays the trade-offs side by side. The flow/NAR recipe's strengths and weaknesses are roughly the mirror image of the codec-LM's:

- **Strength: speed.** Parallel generation means latency is dominated by the number of ODE steps (often 8–32), not the length of the audio. A few-step flow model can synthesize ten seconds of speech in well under a second on a good GPU.
- **Strength: stability.** There is no autoregressive loop to run away. The model cannot "repeat a word forever" because it is not generating word-by-word; it lays the whole utterance down at once. WER tends to be lower and more predictable.
- **Weakness: it needs the timeline.** A non-autoregressive model must know (or learn) how long the output is and how text maps to time. Classic NAR needed a duration predictor; alignment-free variants learn it but can still misjudge timing on hard inputs.
- **Weakness: the flow objective is less "in-context" by default.** It clones via infilling rather than pure ICL, which works very well but is a different framing from "prompt the LM."

The honest summary: **codec-LMs trade speed for flexibility and risk hallucination; flow-matching NAR models trade alignment machinery for parallel, stable, fast synthesis.** In 2024–2026 the flow camp has been winning on latency and intelligibility while the codec-LM camp remains strong on naturalness and zero-config prompting, and the very best closed systems borrow from both.

#### Worked example: latency of the two recipes

Put numbers on "slower." Suppose you want to synthesize a 10-second utterance at a codec frame rate of 75 Hz (a typical EnCodec setting), so 750 acoustic frames, and the codec uses 8 residual codebooks per frame.

A naive fully-autoregressive codec-LM that predicts every codebook of every frame in sequence would need on the order of $750 \times 8 = 6000$ sequential decode steps. Even at a brisk 5 ms per step on a warm GPU, that is about 30 seconds of compute for 10 seconds of audio — a **real-time factor of ~3.0**, i.e. three times slower than real time. (Real systems beat this by predicting codebooks in parallel per frame and other tricks, pulling RTF down toward or below 1.0, but the *shape* of the cost is sequential.)

A flow-matching model integrating the same 750-frame mel in 16 ODE steps does roughly 16 parallel forward passes over the whole sequence. If each pass is, say, 12 ms on the same GPU, that is ~0.2 seconds for 10 seconds of audio — an **RTF of ~0.02**, roughly 100× faster than the naive AR bound. That gap is exactly why streaming and real-time products lean toward NAR/flow or heavily-optimized AR. (These figures are illustrative orders of magnitude on an A100-class GPU; measure your own — see the methodology section.)

## NaturalSpeech 3 and why disentangling helps

NaturalSpeech 3 (Ju et al., 2024) deserves its own section because it introduced an idea that sharpens our understanding of *why* cloning is hard and how to do it better: **factorization**.

The problem with modeling speech as one undifferentiated stream of codec tokens is that several very different kinds of information are tangled together in those tokens: *what* is being said (the phonetic content), *how* it is being said in terms of pitch and rhythm (the prosody), *who* is saying it (the timbre, the speaker identity), and the fine acoustic texture and recording detail. When you clone by prompting a codec-LM, you are asking one model to disentangle all of that on the fly from a short prompt. It works, but it is doing a lot at once, and the speaker identity can bleed or drift.

NaturalSpeech 3's answer is to **factorize speech into separate attribute subspaces, each with its own codec and its own generative process.** Concretely it uses a "FACodec" that splits speech into:

1. **Content** — a phonetic/linguistic code for what is said.
2. **Prosody** — a code for pitch and duration contour.
3. **Timbre** — a (largely global) representation of speaker identity.
4. **Acoustic detail** — a residual code for fine texture the others do not capture.

Then a **factorized diffusion** model generates each subspace, conditioned appropriately, and a decoder recombines them into a waveform.

![A layered stack showing speech factorized into content, prosody, timbre, and acoustic-detail codes, generated by factorized diffusion and recombined by a codec decoder into a waveform](/imgs/blogs/zero-shot-voice-cloning-and-the-tts-frontier-4.png)

Why does this help cloning? Two reasons, and they generalize beyond NaturalSpeech 3:

**Control.** When timbre lives in its own subspace, you can swap the timbre (the speaker) while holding content and prosody fixed — that *is* voice conversion, cleanly. You can change prosody (make it more excited) without touching identity. Disentangled axes are independently steerable axes. A tangled representation forces every change to risk every other attribute.

**Similarity.** Factorizing makes the cloning target explicit. To clone a speaker, you primarily need to match the *timbre* subspace (plus enough prosody to sound natural). By giving the model a dedicated, mostly-global timbre representation extracted from the reference, you make "sound like this person" a direct objective rather than something the model must infer while also juggling content and prosody. NaturalSpeech 3 reported speaker-similarity numbers at the top of the field at its release, and the factorization is a big part of why.

There is a deep idea here that connects to the rest of the series. A codec that disentangles attributes is doing for audio what a well-structured latent does for images in a VAE (see [variational-autoencoders-from-scratch](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch)): a *better-organized* latent space makes the downstream generative task easier and more controllable. The lesson recurs across modalities — representation is half the battle.

### How speaker similarity is actually measured

I keep saying "speaker similarity cosine"; let me make it concrete, because it is the metric that separates honest cloning evaluation from demo reels.

A **speaker-verification (SV) model** is a network trained to map a variable-length utterance to a fixed-length **speaker embedding** — a vector (commonly 192-dimensional for ECAPA-TDNN) such that two utterances from the same person land close together and utterances from different people land far apart. These models are trained on speaker-classification or contrastive objectives over thousands of speakers (VoxCeleb is the canonical dataset). ECAPA-TDNN (Desplanques et al., 2020) is the workhorse; you will also see WavLM-based and x-vector embedders.

Given the SV model, **speaker similarity** between a cloned output $\hat{x}$ and the reference $x_{\text{ref}}$ is the cosine of their embeddings:

$$\text{SIM}(\hat{x}, x_{\text{ref}}) = \frac{\langle e(\hat{x}),\, e(x_{\text{ref}}) \rangle}{\lVert e(\hat{x}) \rVert \, \lVert e(x_{\text{ref}}) \rVert}$$

where $e(\cdot)$ is the SV embedding function. The cosine ranges in $[-1, 1]$ but in practice clusters in roughly $[0, 0.8]$ for speech. As a rough calibration: two clips of the *same* genuine speaker typically score around 0.5–0.8 depending on the embedder and recording conditions; different speakers score near 0–0.3; a strong zero-shot clone lands somewhere in between, and the closer it pushes toward the genuine same-speaker range, the better it is impersonating the target. The exact thresholds depend entirely on which SV model you use, which is the first pitfall: **a SIM of 0.6 means nothing without naming the embedder.**

## How the voice actually gets into the model

There is a fork in the road that distinguishes the families more than any benchmark, and it is worth slowing down on because it determines what kinds of cloning each model is good at. *How* does the reference voice enter the model? There are two mechanisms, and some models use both.

**Mechanism 1 — an explicit speaker embedding.** A separate speaker-encoder network (often the same kind of SV model that computes SIM, or a model trained jointly) maps the reference waveform to a single fixed-length vector $s \in \mathbb{R}^{d}$. That vector is injected into the synthesizer — concatenated to the decoder input, added to every layer via conditioning, or used to modulate normalization layers (a FiLM/AdaIN-style "speaker-adaptive" conditioning). This is how multi-speaker Tacotron/FastSpeech variants and XTTS's speaker-embedding path work. The strength is *efficiency and reuse*: you compute $s$ once from the reference and reuse it across thousands of utterances at near-zero marginal cost (recall the `get_conditioning_latents` call we will make below — that is exactly this). The weakness is *compression*: squeezing a person's voice into one ~200-dimensional vector throws away a lot — the vector captures global timbre well but struggles to carry fine prosodic habits, breathiness, or the specific way someone says "th". A single global vector is, by construction, a lossy summary of identity.

**Mechanism 2 — the reference as a raw prompt (in-context).** Instead of summarizing the reference into a vector, you feed its *actual tokens or features* to the model as context — the codec-LM's prepended acoustic tokens, or the flow model's unmasked reference span. Nothing is summarized away; the model attends to the full reference. The strength is *richness*: prosody, micro-timing, recording character, even emotional tone come along for free, which is why VALL-E famously clones the prompt's *emotion and acoustic environment*, not just the average timbre. The weakness is *cost and variance*: you re-pay for the reference on every generation (the prompt is in the context window every time), and a noisy or atypical reference can perturb the output more than a single robust embedding would.

In practice the best systems blend the two. XTTS, for instance, uses both a GPT conditioning latent computed from the reference *and* a speaker embedding. NaturalSpeech 3's factorization is a sophisticated version of mechanism 1 done *right*: rather than one tangled vector, it extracts a clean *timbre* representation in its own subspace, so the global-vector approach stops throwing away prosody (which now has its own code) and stops being lossy about identity (which now has a dedicated, less-contended channel). This is the deep reason factorization improved similarity: it fixed the central failure mode of single-vector speaker conditioning — that one vector had to carry too much.

The engineering consequence is concrete. If you are cloning *one* voice many times (an audiobook narrator), you want mechanism 1's reusable embedding — compute it once, cache it, amortize it. If you are cloning *many different* voices a few times each (a service where every user clones themselves once), the marginal cost of mechanism 2's in-context prompt is fine and you get the richer clone. Match the mechanism to your traffic pattern, not to a leaderboard.

## Run it: a zero-shot clone with XTTS v2

Enough theory. Let me show you the whole loop end to end with **XTTS v2** (Coqui's open multilingual zero-shot cloner), then measure the result. XTTS v2 is a codec-LM-family model: it supports 17 languages, clones from ~6 seconds of reference audio, and runs on a single consumer GPU. Install and clone:

```bash
# A recent Python (3.9-3.11) and a CUDA-capable PyTorch are assumed.
pip install TTS soundfile librosa
# XTTS v2 weights download on first use (~1.8 GB).
```

```python
import torch
from TTS.api import TTS

# Load XTTS v2. First call downloads the checkpoint.
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Zero-shot clone: a reference wav + target text + language -> cloned speech.
# `speaker_wav` is the few-second reference of the voice to clone.
tts.tts_to_file(
    text="The quick brown fox jumps over the lazy dog.",
    speaker_wav="reference_6s.wav",   # 6-10 s of the target voice, clean
    language="en",
    file_path="cloned_en.wav",
)
```

That is the entire clone. No fine-tuning, no per-voice training step — the reference is consumed at inference. Because XTTS v2 is multilingual, the same reference can speak a *different* language than it was recorded in, which is one of the genuinely magical capabilities of zero-shot cloning:

```python
# Same voice, different language. The reference is English; we synthesize Spanish.
tts.tts_to_file(
    text="El veloz zorro marrón salta sobre el perro perezoso.",
    speaker_wav="reference_6s.wav",
    language="es",
    file_path="cloned_es.wav",
)
```

A few things that matter in practice, learned the hard way:

- **The reference should be clean.** 6–10 seconds of the target speaking clearly, no background music, no second speaker, no heavy reverb. Garbage reference, garbage clone — the model faithfully clones the *recording conditions*, so a reference recorded in a tiled bathroom produces speech that sounds recorded in a tiled bathroom.
- **Longer reference is not always better.** Past ~10–15 seconds you mostly add latency, not quality; the speaker identity saturates quickly.
- **Language tag matters.** Set `language` to the *output* language; the model handles the cross-lingual mapping.

For lower-level control (splitting the expensive speaker-conditioning step from synthesis so you can reuse it across many utterances), XTTS exposes the underlying model:

```python
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

config = XttsConfig()
config.load_json("/path/to/xtts/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="/path/to/xtts/", eval=True)
model.cuda()

# Compute the speaker conditioning ONCE from the reference.
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
    audio_path=["reference_6s.wav"]
)

# Reuse it for many utterances without re-reading the reference each time.
out = model.inference(
    "The quick brown fox jumps over the lazy dog.",
    "en",
    gpt_cond_latent,
    speaker_embedding,
    temperature=0.7,   # lower = more stable/faithful, higher = more varied
)
# out["wav"] is a float waveform at the model sample rate (24 kHz).
```

Notice `temperature`: because the autoregressive decoder *samples*, you have a stability knob. Lower temperature gives more faithful, more stable output (and lower WER) at the cost of some expressive variety; higher temperature is livelier but more prone to the occasional autoregressive glitch. This knob does not exist in a deterministic flow sampler — it is a fingerprint of the codec-LM recipe.

### Measuring the clone: similarity and WER

Now the part that turns a demo into an evaluation. We compute two numbers: **speaker similarity** (does it sound like the target?) via an ECAPA embedding cosine, and **WER** (did it say the right words?) via Whisper ASR.

First, speaker similarity with a SpeechBrain ECAPA-TDNN embedder:

```python
import torchaudio
import torch
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier

# ECAPA-TDNN speaker embedder trained on VoxCeleb (192-dim embeddings).
sv = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": "cuda"},
)

def speaker_embedding(path, target_sr=16000):
    wav, sr = torchaudio.load(path)
    if sr != target_sr:                       # ECAPA expects 16 kHz
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    wav = wav.mean(dim=0, keepdim=True)       # mono
    emb = sv.encode_batch(wav).squeeze()      # (192,)
    return emb

ref_emb = speaker_embedding("reference_6s.wav")
gen_emb = speaker_embedding("cloned_en.wav")

sim = F.cosine_similarity(ref_emb, gen_emb, dim=0).item()
print(f"Speaker similarity (ECAPA cosine): {sim:.3f}")
# A strong clone lands well above a different-speaker baseline (~0.1-0.2)
# and approaches the same-speaker range (~0.5-0.7) for this embedder.
```

Then WER, by transcribing the cloned audio with Whisper and comparing to the text we asked it to speak:

```python
import whisper                  # pip install openai-whisper jiwer
from jiwer import wer

asr = whisper.load_model("small")   # "base"/"small" trade speed for accuracy

target_text = "The quick brown fox jumps over the lazy dog."
hyp = asr.transcribe("cloned_en.wav", language="en")["text"]

# Normalize lightly before scoring (lowercase, strip punctuation) in practice.
score = wer(target_text.lower(), hyp.lower().strip("."))
print(f"WER: {score*100:.1f}%   ASR heard: {hyp!r}")
```

A clean clone of a short English sentence should give a WER near 0% (the ASR transcribes exactly what you asked for) and a SIM comfortably above the different-speaker baseline. When you see WER climb — words dropped, repeated, or hallucinated — that is your codec-LM being unstable, and it is your cue to lower the temperature, clean the reference, or switch to a flow model.

There is a subtlety worth flagging, because it bites people: **WER via ASR measures the ASR as much as the TTS.** If Whisper mis-hears a correctly-synthesized word, that counts as a TTS error in your score even though the TTS was fine. Use a strong ASR, normalize text consistently, and treat very low WER differences (a percentage point or two) as noise, not signal. We will formalize this in the methodology section.

Now put those two metrics into a single reusable harness, because you will run them constantly and ad-hoc one-offs lead to inconsistent numbers. The function below clones a list of sentences, then scores every output for SIM and WER and aggregates — exactly the loop you want before declaring any model "better":

```python
import numpy as np
from jiwer import wer as jiwer_wer

def evaluate_clone(tts_synthesize, sentences, reference_wav, asr, sv_embed):
    """Clone a set of sentences with one reference and score SIM + WER.

    tts_synthesize(text, reference_wav) -> path to a generated wav.
    asr(path) -> transcript string.
    sv_embed(path) -> speaker embedding tensor.
    """
    ref_emb = sv_embed(reference_wav)
    sims, wers = [], []
    for text in sentences:
        out_path = tts_synthesize(text, reference_wav)
        gen_emb = sv_embed(out_path)
        cos = float(
            (ref_emb @ gen_emb)
            / (ref_emb.norm() * gen_emb.norm())
        )
        hyp = asr(out_path).lower().strip(" .")
        w = jiwer_wer(text.lower().strip(" ."), hyp)
        sims.append(cos)
        wers.append(w)
    return {
        "SIM_mean": float(np.mean(sims)),
        "SIM_std": float(np.std(sims)),       # variance matters as much as the mean
        "WER_mean": float(np.mean(wers)),
        "WER_p90": float(np.percentile(wers, 90)),  # the tail is where clones fail
        "n": len(sentences),
    }
```

Two design choices in that harness are deliberate and easy to miss. First, it reports `SIM_std`, not just the mean — a clone that averages 0.5 but swings between 0.2 and 0.7 is *worse* in production than one that sits steadily at 0.45, because the low-similarity outputs are the ones users complain about. Second, it reports `WER_p90`, the 90th-percentile WER, not just the mean — the AR codec-LM's failure mode is a *few* badly-glitched utterances among many clean ones, and a mean hides them while a p90 surfaces them. Measure the tail; that is where clones actually break.

## Stress-testing the clone: where it breaks

A model that demos perfectly on one clean sentence is not a product. The engineering work is in the failure modes. Let me walk through the ones I have actually hit, because knowing where a clone breaks is how you decide which recipe to ship and what guardrails to build.

**The reference is 3 seconds of noisy audio.** This is the most common real-world case — the only clip you have of the target is a noisy voicemail or a phone recording with a hum. Both recipes degrade, but differently. The codec-LM clones the *noise too*: because it continues the prompt's acoustic tokens, the output inherits the hum, the room, the codec artifacts of the source recording. You get the right voice in the wrong acoustic environment. The flow model, infilling against a noisy reference, tends to produce a *cleaner but less faithful* clone — it does not copy the noise as literally, but it also pins the identity less tightly to a degraded reference. The fix is upstream: denoise and normalize the reference first (a quick run through a speech enhancement model, loudness-normalize to a target LUFS, trim silence). A 30-minute investment in reference hygiene buys more clone quality than any model swap. Garbage in, garbage cloned.

**You ask for four minutes in one pass.** Long-form is where the recipes diverge hardest. The autoregressive codec-LM's error accumulates: small drifts compound over thousands of frames, and the probability of *some* glitch (a repeat, a skip, a tonal drift) grows with length. By a few minutes, a single-pass AR clone often wanders. The flow/NAR model is more length-robust because it does not accumulate sampling error frame-by-frame, but it has its own problem — it must allocate the whole timeline up front, and timing errors on long inputs can desync. **The correct answer for both is: do not generate four minutes in one pass.** Chunk by sentence or paragraph, synthesize each chunk with the *same* reference conditioning, and concatenate with short crossfades. This caps error accumulation, lets you parallelize across chunks, and lets you re-roll only the bad chunks (via the WER gate). Every production long-form pipeline I have seen does this; the "generate a whole audiobook chapter in one call" demo is not how it actually ships.

**The codec drops the high frequencies.** Cloning quality is bounded by codec quality. If your codec or vocoder is trained or configured for a lower bandwidth (say it rolls off above 8 kHz), the clone will sound dull and "telephone-y" no matter how good the speaker conditioning is — sibilants (s, sh, f) live in the high frequencies, and losing them costs both naturalness and a subtle bit of identity. This is a reminder that the clone is only the *generative* half; the codec/vocoder (see [gan-vocoders-hifi-gan-and-fast-synthesis](/blog/machine-learning/audio-generation/gan-vocoders-hifi-gan-and-fast-synthesis)) sets the fidelity ceiling. When a clone sounds muffled, check the codec's bandwidth before blaming the cloner.

**The vocoder, not the model, is the bottleneck.** A flow model can generate a mel in 0.2 seconds, but if you then run a slow diffusion vocoder to turn that mel into a waveform, your end-to-end latency is dominated by the vocoder, not the cloner. People optimize the wrong half constantly. Profile the *whole* path — text→features→waveform — and optimize the slowest stage. Often the fix is a fast GAN vocoder (HiFi-GAN, BigVGAN, Vocos) instead of a diffusion vocoder, recovering an order of magnitude with negligible quality loss.

**Cross-lingual cloning of sounds the reference never makes.** When you clone an English speaker into Mandarin, the model has to produce tones and phonemes the reference never demonstrated. It does this by *transferring timbre* while *borrowing phonetics* from its multilingual training — and the result can have a slight accent or mispronounce sounds outside the reference's native inventory. This is not a bug so much as a fundamental limit: you cannot recover a speaker's actual Mandarin pronunciation from English audio, so the model approximates. Expect cross-lingual clones to be excellent on timbre and merely good on phonetic authenticity, and test specifically on the target language's hard sounds.

The throughline: **most clone failures are not "the model is bad" — they are reference hygiene, length handling, codec bandwidth, vocoder choice, or an inherent cross-lingual limit.** Diagnose the stage, not the system.

## The open models you can run tonight

The open ecosystem is genuinely good now — good enough that for many use cases you never need a paid API. Here is the landscape, with what each is for.

**XTTS v2 (Coqui).** The one we just ran. Codec-LM family, 17 languages, ~6-second zero-shot clone, cross-lingual. Strengths: multilingual breadth, easy API, good naturalness. Weaknesses: the original company wound down (the community maintains forks), the license is research/non-commercial-leaning so check before shipping, and like all AR models it can occasionally glitch. Still the default "clone in any of many languages" pick.

**F5-TTS (and E2-TTS).** Flow-matching, alignment-free. Strengths: fast (parallel, few-step), stable, low WER, surprisingly simple, permissive licensing, strong English and Chinese with growing multilingual community checkpoints. Weaknesses: needs a reference *transcript* in the basic recipe (it infills, so it wants to know what the reference says), fewer official languages than XTTS. This is the model I reach for when I want speed and stability and I am working in English or Chinese.

```python
# F5-TTS sketch (the CLI/Python API varies by release; this is the shape).
# pip install f5-tts
from f5_tts.api import F5TTS

model = F5TTS()   # downloads the flow-matching checkpoint
wav, sr, _ = model.infer(
    ref_file="reference_6s.wav",
    ref_text="This is what the reference clip says.",  # infill needs the ref text
    gen_text="The quick brown fox jumps over the lazy dog.",
    nfe_step=32,   # number of ODE/function evaluations: fewer = faster, less precise
)
# Lower nfe_step (e.g. 16) for speed; raise it for fidelity. This is the flow knob.
```

Notice the knob is now `nfe_step` (number of function evaluations / ODE steps), the flow-matching analogue of diffusion sampling steps — *not* a temperature. That single difference (steps vs temperature) tells you which recipe you are holding.

**Fish-Speech.** A strong open multilingual TTS with zero-shot cloning, notable for good quality-to-size and active development; a codec-LM-style design with its own codec. A good pick when you want multilingual cloning with an actively maintained codebase.

**Parler-TTS.** A different flavor: **described-voice control.** Instead of (or in addition to) cloning a specific reference, you describe the voice you want in natural language — "a male speaker with a slightly low-pitched voice delivering his words at a fast pace in a clear, close-sounding environment" — and the model synthesizes a voice matching the description. This is *not* impersonation; it is voice *design*, which is both a creative tool and a partial sidestep of the consent problem (you are not copying a real person). Great when you want a consistent, controllable, original voice rather than a clone.

```python
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import torch

device = "cuda"
model = ParlerTTSForConditionalGeneration.from_pretrained(
    "parler-tts/parler-tts-mini-v1"
).to(device)
tok = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

prompt = "The quick brown fox jumps over the lazy dog."
description = ("A clear, warm female voice speaking at a moderate pace "
              "in a quiet, close-sounding room.")

input_ids = tok(description, return_tensors="pt").input_ids.to(device)
prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(device)

audio = model.generate(input_ids=input_ids, prompt_input_ids=prompt_ids)
sf.write("parler_out.wav", audio.cpu().numpy().squeeze(),
         model.config.sampling_rate)
```

**MetaVoice and OpenVoice.** MetaVoice-1B is an open foundational TTS with zero-shot cloning and an emphasis on emotional/rhythmic English. OpenVoice (MyShell) is interesting architecturally: it cleanly *separates tone-color cloning from style control*, so you can clone a speaker's timbre and then independently dictate emotion, accent, rhythm, and pauses — a practical, lightweight realization of the disentanglement idea NaturalSpeech 3 makes rigorous. OpenVoice is a good choice when you want a clone *plus* explicit, cheap style control and broad language support.

The practical takeaway: **for multilingual cloning reach for XTTS v2 or Fish-Speech; for speed and stability in English/Chinese reach for F5-TTS; for designed (non-impersonated) voices reach for Parler-TTS; for clone-plus-style-control reach for OpenVoice.** None of these costs anything but compute.

### Why multilingual cloning is harder than it looks

It is worth dwelling on multilinguality, because it is where the open and closed gaps are widest and where the engineering is most underestimated. A monolingual cloner has one job: map text in one language to speech. A multilingual zero-shot cloner has to *disentangle language from identity* — to take a speaker's timbre, learned from a reference in language A, and render it speaking language B. That requires the model's internal speaker representation to be **language-agnostic**: the timbre channel must encode "this is what this person's vocal tract sounds like" without smuggling in "and they speak English." If timbre and language leak into each other during training, cross-lingual cloning produces either a wrong voice or a heavy accent.

The two recipes handle this differently. A codec-LM operating on a *shared* multilingual phoneme or byte vocabulary, with the speaker carried by acoustic-token prompting, can transfer voice across languages naturally — the acoustic prompt is just timbre, and the phonemes drive the language. This is roughly how XTTS achieves 17-language coverage from one model: shared text frontend, language tag, speaker conditioning that is (ideally) orthogonal to language. The cost is data and balance: you need substantial training audio in *every* language you want to support, and low-resource languages get a worse clone simply because the model saw fewer hours of them. The closed frontier's language-breadth advantage is, in large part, a *data* advantage — they have licensed or collected more multilingual speech.

There is a subtle evaluation trap here too. Your SIM embedder (ECAPA on VoxCeleb) was likely trained mostly on English speech, so its sense of "same speaker" can be weaker in other languages — meaning cross-lingual SIM numbers are *even less* trustworthy than same-language ones. And WER requires a *multilingual* ASR (Whisper is, fortunately) with per-language normalization that handles that language's orthography. Measuring a Mandarin clone with English-tuned text normalization will manufacture errors that are not there. If you ship multilingual, your evaluation harness has to be multilingual end to end, or your numbers lie by construction.

## The closed frontier

For all the strength of open models, the closed services still lead on the things that are hard to get exactly right: latency, polish across long inputs, language breadth, and the boring-but-critical production surface (streaming APIs, voice libraries, pronunciation controls). The leader is unambiguous.

**ElevenLabs** is the quality-and-latency leader of the current generation. Three things distinguish it. First, **raw quality**: its naturalness and prosody, especially in English, set the bar that open models are measured against. Second, **latency**: it offers low-latency streaming models (their "Turbo"/"Flash" tiers) suitable for real-time conversational use, which is a genuinely hard engineering problem (we cover the streaming/full-duplex frontier in [real-time-streaming-and-full-duplex-speech](/blog/machine-learning/audio-generation/real-time-streaming-and-full-duplex-speech)). Third, **the product surface**: a large shared *voice library*, *instant voice cloning* from a short sample, *professional voice cloning* (a higher-fidelity clone trained from more of your audio, behind identity verification), and *voice design* (generate a voice from a text description, like Parler but more polished). The cloning quality is excellent and the consent gating around professional cloning is, notably, part of the product rather than an afterthought.

**PlayHT (Play.ht)** is the main competitor in the same space — high-quality TTS, instant cloning, a voice library, and a streaming API aimed at conversational agents. It trades blows with ElevenLabs on quality and competes hard on developer experience and pricing.

**The big labs' TTS.** The frontier labs ship strong TTS too, usually as part of a voice or multimodal product rather than a standalone cloning API: OpenAI's TTS voices (and the realtime voice stack), Google's offerings, Microsoft Azure's neural TTS (the lineage that produced the VALL-E and NaturalSpeech research), Amazon Polly, and others. These tend to emphasize *curated, consented* voices and reliability for enterprise rather than open-ended impersonation — which is both a business choice and a safety stance.

What do you actually buy by going closed? Mostly: someone else owns the latency engineering, the language coverage, the pronunciation lexicons, the uptime, and the abuse-prevention. What do you give up? Per-character cost, data leaving your environment, and the ability to run offline or customize the model. For a real-time customer-facing agent in many languages, the closed services are often the right call. For batch generation, on-prem requirements, cost-sensitive scale, or research, open wins.

A word on the latency that makes closed services worth paying for, because it is a metric people quote sloppily. For an interactive voice agent the number that matters is not total render time but **time-to-first-audio** — how long after you send text before the first chunk of waveform starts playing. A batch RTF of 0.3 is irrelevant to a conversation if the model must finish the *whole* utterance before emitting anything; the user hears silence for a second and the illusion of conversation breaks. This is why streaming TTS is its own engineering problem (covered in [real-time-streaming-and-full-duplex-speech](/blog/machine-learning/audio-generation/real-time-streaming-and-full-duplex-speech)): the model has to emit audio incrementally, sentence-by-sentence or even faster, so first-audio latency can be a couple hundred milliseconds even though the full utterance takes longer. Autoregressive codec-LMs are actually *well-suited* to streaming in principle — they generate left-to-right, so they can emit early frames while still computing later ones — which is one place the AR recipe's sequential nature is an asset rather than a liability. Flow/NAR models, generating the whole utterance at once, have to chunk to stream. When you benchmark for a real-time use case, **measure time-to-first-audio with a warmed-up model on your target device, not steady-state RTF on a sentence you already cached.** Those are different numbers and confusing them ships a laggy agent.

## Putting numbers on it

Here is the comparison the whole post has been building toward. Treat every number as *approximate and embedder-/setup-dependent* — I will be explicit about that, because precise-looking TTS numbers are routinely cited without the context that makes them meaningful.

![A matrix comparing XTTS v2, F5-TTS, NaturalSpeech 3, and ElevenLabs across openness, similarity and SMOS, word error rate, multilinguality, and license](/imgs/blogs/zero-shot-voice-cloning-and-the-tts-frontier-6.png)

| Model | Open? | Recipe | SIM / SMOS | WER (clean EN) | Multilingual | Latency / RTF | License |
|---|---|---|---|---|---|---|---|
| **XTTS v2** | Yes | Codec-LM (AR) | good (~0.40 SIM) | ~3–5% | 17 languages | RTF ~0.3–1 on a 4090 | research-leaning (verify) |
| **F5-TTS** | Yes | Flow matching (NAR) | high (~0.6+ SIM reported) | ~2–3% | EN/ZH + community | RTF well under 1 (few steps) | permissive (verify) |
| **NaturalSpeech 3** | No (paper) | Factorized diffusion (NAR) | SOTA SIM at release | ~1.8% reported | EN focus | NAR, fast in principle | closed/research |
| **ElevenLabs** | No (API) | proprietary | top human MOS | very low | 30+ languages | low-latency streaming tiers | paid SaaS |

A few honest readings of this table:

- **The open-vs-closed quality gap has narrowed to "polish", not "capability".** F5-TTS reports speaker similarity and WER in the same league as research SOTA. The remaining closed advantages are latency engineering, language breadth, long-input robustness, and product features — not a fundamental quality moat.
- **WER differences in the 1–3% range are mostly noise.** They depend on the ASR, the text normalization, and the test set. Do not pick a model because it claims 2.1% over another's 2.8%; pick on the axis that dominates your use case.
- **SIM numbers are not comparable across papers unless the embedder matches.** A "0.6" from a WavLM-based verifier and a "0.4" from ECAPA can describe the *same* clone quality. When the rows above disagree on SIM, suspect the embedder before suspecting the model.

#### Worked example: choosing for a multilingual audiobook pipeline

Say you are building a pipeline to narrate audiobooks in 12 languages, batch (not real-time), on your own GPUs, with a small set of consented narrator voices you clone once and reuse. Walk the axes:

- *Multilingual* dominates → XTTS v2's 17 languages or Fish-Speech beat F5-TTS's narrower official coverage.
- *Batch, not real-time* → RTF matters less; you can tolerate the AR codec-LM's slower generation.
- *Long inputs* → you will chunk by sentence/paragraph anyway, so per-utterance stability (clean references, low temperature) matters more than single-pass length.
- *On-prem + cost-sensitive at scale* → open beats paying per character to a SaaS.

Decision: **XTTS v2, low temperature, clean ~10 s references per narrator, sentence-chunked, with a WER gate** (re-synthesize any chunk whose Whisper-WER exceeds, say, 5%, to catch the occasional AR glitch). You buy back the codec-LM's instability with a cheap automatic check. If you were instead building a *real-time English voice agent*, you would flip to F5-TTS (or a closed low-latency API) for the RTF, accepting narrower language coverage.

#### Worked example: a SIM/WER reading you can trust

Suppose you clone the same speaker with XTTS v2 and with F5-TTS and measure with a *fixed* harness: ECAPA-TDNN for SIM, Whisper-small for WER, the same 50 held-out sentences, the same 10-second reference. You get (illustrative): XTTS SIM 0.41 / WER 4.2%; F5-TTS SIM 0.49 / WER 2.6%. The right conclusion is **not** "F5 is 0.08 better at cloning." It is: "with *this* ECAPA embedder and *this* ASR, F5-TTS scored higher on both identity and intelligibility on this set; given the embedder/ASR/test-set dependence, this is evidence F5 is at least as good here, and I should confirm with a small human SMOS/MOS panel before declaring a winner." That hedged reading is what separates an engineer who measures from a leaderboard scroller. The audio-eval rabbit hole — why these metrics lie and how to build an honest harness — is its own post: [audio-quality-metrics](/blog/machine-learning/audio-generation/audio-quality-metrics).

## How to evaluate a clone honestly

The single most common mistake in this space is judging a clone by one impressive sample. A clone can nail one sentence and fall apart on the next; it can fool you on naturalness while having the wrong identity, or match identity while being unintelligible. Honest evaluation means measuring all three axes, with humans in the loop for the two that automatic metrics cannot fully capture.

![A matrix showing four evaluation metrics, speaker-verification cosine, ASR word error rate, similarity MOS, and naturalness MOS, with what each measures, how, and its pitfall](/imgs/blogs/zero-shot-voice-cloning-and-the-tts-frontier-8.png)

Figure 8 is the harness. Read it as: **no single metric covers all three axes, so you need at least three measurements.**

- **SIM (automatic, identity).** ECAPA (or WavLM) cosine between output and reference. Cheap, fast, scalable — run it on every output. *Pitfall:* it measures what the embedder learned to care about, which is not identical to human identity perception, and it is biased by recording conditions. Name your embedder; never compare SIM across different embedders.
- **WER (automatic, intelligibility).** Whisper-transcribe the output, compare to the target text. Cheap, scalable, catches dropped/repeated/hallucinated words — the AR failure mode. *Pitfall:* it conflates ASR errors with TTS errors; use a strong ASR and consistent normalization, and treat sub-2% differences as noise.
- **SMOS (human, similarity).** Play raters the reference and the clone; they rate similarity 1–5. This is the ground truth for "does it sound like the same person," which is ultimately a human judgment. *Pitfall:* rater variance — you need enough raters (often 10–20+) and a balanced design, and listening conditions matter (headphones, not laptop speakers).
- **MOS (human, naturalness).** Raters score how natural/human the clip sounds, 1–5, ideally with real human speech mixed in as anchors. *Pitfall:* needs many raters for tight confidence intervals; MOS is notoriously non-comparable across studies (different rater pools, scales, anchors), so report it *within* a study against your own baselines, not as an absolute.

The discipline that makes these trustworthy:

1. **Fix everything that is not the model.** Same test sentences, same references, same reference length, same ASR, same SV embedder, same sample rate, same loudness normalization. A clone comparison where the test set changed is not a comparison.
2. **Use a held-out, diverse test set.** Multiple speakers (varied gender, age, accent), multiple sentence types (short, long, questions, numbers, hard names). A model can look great on easy sentences and collapse on a phone number.
3. **Report automatic metrics on the *full* set, humans on a *sample*.** SIM/WER on all N; SMOS/MOS on a representative subset, because human rating is expensive.
4. **Warm up before timing.** RTF and latency numbers must exclude model-load and first-call JIT/compile cost. Time the *steady state* on a named device; report the device.
5. **Mix in anchors for human studies.** Include real human recordings and a known-weak system among the rated clips so MOS/SMOS have a reference frame and you can detect inattentive raters.

Do this and your "this clone is good" becomes a defensible claim instead of a feeling. Skip it and you will ship a model that demos beautifully and fails on the long tail your users actually hit.

## The safety elephant in the room

I have spent this whole post teaching you to copy a specific human's voice from a few seconds of audio. I would be irresponsible not to stop and name what that is.

**Zero-shot voice cloning is, by construction, an impersonation engine.** The same capability that lets you give an audiobook a consistent narrator lets someone clone a CEO's voice from a conference talk and call the finance department to authorize a wire transfer — a fraud that has already happened in the real world. It lets someone clone a family member from a social-media clip and stage a fake distress call. It lets someone put words in a politician's mouth. The barrier that used to protect us — that faking a specific person's voice convincingly was hard and slow — is gone. The reference audio is often *already public*: a podcast, a video, a voicemail greeting.

This is not a hypothetical to bolt on at the end; it shapes how you should build. A few principles I hold to, and which the responsible products in this space are converging on:

- **Consent is a feature, not a checkbox.** The strongest cloning products gate high-fidelity ("professional") cloning behind verification that you are cloning *yourself* or someone who consented. Instant cloning from arbitrary audio is the dangerous path; treat the ability to clone any voice as something to constrain, not advertise.
- **Watermark generated audio.** Imperceptible, robust watermarks (AudioSeal, SynthID-audio) let downstream systems and platforms detect that a clip was synthesized, even after compression and re-recording. If you generate, watermark.
- **Provenance and disclosure.** C2PA-style content credentials and clear disclosure ("this voice is synthetic") are part of the deal in any legitimate use.
- **Detection is an arms race, but a necessary one.** Deepfake-audio detectors are imperfect and adversaries adapt, but defense-in-depth — watermark + detector + provenance + out-of-band verification for high-stakes actions — is meaningfully better than nothing.

#### Worked example: the cost asymmetry that drives abuse

Why is this suddenly a problem? Follow the economics. To clone a voice convincingly five years ago took studio time, a voice actor, and skilled audio editing — call it hundreds of dollars and days of work per voice, which priced out casual misuse. Today, an open model clones a voice from a 6-second public clip for the cost of a few seconds of GPU time — on the order of \$0.001 of compute — and a closed API charges perhaps a few dollars per *hour* of generated audio. The cost of *producing* a convincing fake collapsed by four or five orders of magnitude, while the cost of *verifying* a voice's authenticity (a phone call back, a code word, a second channel) did not budge. That asymmetry — cheap to fake, still-expensive to verify — is the entire shape of the threat. The defense is not "make cloning harder" (the capability is out); it is "make verification cheaper and synthesis detectable": watermark every output so detection is automatic, and move high-stakes decisions to a second channel that a cloned voice cannot satisfy (a call-back to a known number, a shared secret, an in-person confirmation). Design assuming the voice on the line might be synthetic, because increasingly it can be.

The full treatment of detection, watermarking, consent, and provenance — the defender's side of this technology — is the dedicated safety post: [audio-deepfakes-watermarking-and-voice-safety](/blog/machine-learning/audio-generation/audio-deepfakes-watermarking-and-voice-safety). If you take one thing from this section: **the question is never just "can I clone this voice" — the model can. The question is "am I allowed to, and how will anyone know it is synthetic." Build with that on the first line of the design doc, not the last.**

## Case studies and real numbers

Concrete results from the literature and shipped systems, with the caveats that make them honest.

**VALL-E's zero-shot cloning (Wang et al., 2023).** The model that proved codec-LM cloning. It demonstrated that a few seconds of unseen-speaker audio, used as an acoustic-token prompt, produces speech that preserves the speaker's timbre and *emotion* and even the acoustic environment of the prompt. On the LibriSpeech zero-shot benchmark it reported strong speaker similarity and competitive WER versus the baselines of its day, and — critically — it did so *without fine-tuning per speaker*, which was the new thing. Its weaknesses were the AR ones: occasional instability and slower generation. Read the full breakdown in the sibling post: [neural-codec-language-model-tts-vall-e](/blog/machine-learning/audio-generation/neural-codec-language-model-tts-vall-e).

**NaturalSpeech 3 (Ju et al., 2024).** The factorized-diffusion model. At release it reported zero-shot results that were at or near the top of the field on speaker similarity, naturalness, and intelligibility — with a reported WER around 1.8% on its benchmark, in the range of (sometimes better than) ground-truth-adjacent numbers, and SIM at the front of the pack. The lesson is the one we drew earlier: factorizing the representation (content / prosody / timbre / detail) made each attribute easier to model and control, which showed up directly in similarity. The numbers are paper-reported on the paper's setup; reproduce on your own harness before treating them as your numbers.

**F5-TTS (Chen et al., 2024).** The alignment-free flow-matching model. It reported strong zero-shot cloning with low WER and high speaker similarity while being dramatically simpler than prior NAR pipelines (no separate aligner or duration predictor), and fast at inference thanks to few-step flow sampling. Its public checkpoints made high-quality, *stable*, fast cloning genuinely accessible in the open. In my own use it is the most reliable open model for short English/Chinese utterances — the WER is low and it does not run away.

**XTTS v2 (Coqui).** The multilingual open workhorse. Its headline capability is 17-language zero-shot cloning from ~6 seconds, including cross-lingual synthesis (English reference speaking Spanish). It made multilingual cloning a `pip install` away. Honest caveats: as an AR codec-LM it has the occasional glitch (mitigate with low temperature and a WER gate), and licensing/maintenance shifted after the company wound down, so verify the license for commercial use and lean on maintained community forks.

**ElevenLabs (shipped product).** Not a paper but the de-facto quality bar. Its instant cloning, voice library, low-latency streaming tiers, and voice design define what "frontier closed TTS" means in this period, and its consent-gated professional cloning is a model for how to ship this capability responsibly. The exact model internals are proprietary, so treat any architectural claim about it as inference, not fact — what is *measurable* is that human listeners rate its naturalness and prosody at the top of blind comparisons.

A unifying thread across these: **the recipe choice (codec-LM vs flow) shows up in the failure modes more than the headline numbers.** Codec-LMs fail by hallucinating; flow models fail by mistiming. Pick the failure mode you can tolerate and detect.

## When to reach for this (and when not to)

A decisive guide, because "it depends" is a cop-out and you came here for a recommendation.

**Reach for zero-shot cloning when:**

- You need *many* voices and cannot train a model per voice. This is the whole point — one model, any consented voice, no per-voice training.
- You need a voice on short notice from a short sample (a user's own voice for accessibility; a consented narrator).
- You need cross-lingual synthesis in a specific person's voice. Zero-shot multilingual cloners (XTTS v2) do this natively; per-language single-speaker models do not.

**Prefer a fine-tuned single-speaker model when:**

- You have *one* voice you will use at scale forever (a brand voice, a flagship audiobook narrator) and you have hours of its audio. A model fine-tuned on that voice will beat a zero-shot clone on consistency and fidelity, because it has seen far more than a 6-second prompt. Zero-shot is for *breadth*; fine-tuning is for *depth on one voice*.

**Prefer described-voice design (Parler-TTS, voice design APIs) when:**

- You want an *original* voice, not a copy of a real person. This sidesteps the consent problem entirely and gives you a consistent, controllable voice you own. For most product narration where you do not specifically need a known person, this is the *ethically simpler* choice — use it.

**Choose the recipe by your constraint:**

- *Real-time / low latency* → flow/NAR (F5-TTS) or a closed low-latency API. Do **not** run a naive full-AR codec-LM for an interactive agent — the per-frame loop will blow your latency budget.
- *Multilingual breadth* → XTTS v2 / Fish-Speech (codec-LM family with broad language coverage) over narrower flow checkpoints.
- *Maximum stability / lowest WER on hard text* → flow/NAR, or AR with a WER gate and low temperature. Do **not** ship an AR clone for high-stakes text (legal, medical, numbers) without an automatic intelligibility check.
- *Maximum control over emotion/style independent of identity* → a disentangled model (OpenVoice, or NaturalSpeech-3-style factorization). Do **not** expect a single tangled codec-LM to let you change emotion without touching identity.

**Do not reach for any of this when:**

- You do not have consent to clone the target voice. There is no engineering workaround for "you are not allowed to." If you cannot establish consent, use voice *design* (an original voice) instead.
- A few curated, consented stock voices would do. Cloning adds risk and complexity; if a voice-library voice meets the need, take it.

A final piece of pragmatism on *picking* among the open models, because "it depends" deserves a default. If you are starting today and unsure, my recommended first move is: **prototype with two models in parallel — XTTS v2 for breadth and F5-TTS for speed/stability — score both on your own 50-sentence harness with your own references, and let the numbers (and a quick listen on headphones) decide.** It costs an afternoon and replaces a leaderboard argument with evidence on *your* data, which is the only evidence that matters. Reach for a closed API only when the open prototype falls short on a specific axis you cannot fix — usually latency for real-time, language breadth for long-tail languages, or long-input robustness — and even then, watermark and consent-gate exactly as you would with the open model. The choice of vendor does not change your safety obligations; it only changes who runs the inference.

And do not skip the boring infrastructure that separates a demo from a service: a reference-audio preprocessing step (denoise, loudness-normalize, trim, validate length and that it is a single speaker), a per-output WER gate that re-rolls glitched utterances, a watermarker on every generated file, a consent record tied to each cloned voice, and a cache for reusable speaker conditioning. None of that is glamorous, and all of it is what makes cloning shippable rather than a party trick. The model is the easy 20%; the harness around it is the 80% that determines whether you have a product or a liability.

## Key takeaways

- **Zero-shot voice cloning supplies the voice in context, not in the weights.** A few seconds of reference + frozen model = the target's voice on any text. One model, every voice.
- **Two recipes dominate.** Codec-LMs (VALL-E line, XTTS) sample audio tokens autoregressively from an in-context prompt — flexible and natural but sequential and occasionally unstable. Flow-matching NAR models (Voicebox, F5-TTS, NaturalSpeech 3) infill the whole utterance in parallel — fast and stable but they must handle the timeline.
- **The recipe shows up in the knob and the failure mode.** Codec-LMs have a *temperature* and fail by *hallucinating*; flow models have *ODE steps* and fail by *mistiming*. Pick the failure you can detect.
- **Disentangling helps.** NaturalSpeech 3's factorization into content / prosody / timbre / detail makes cloning a *direct* timbre-matching objective and makes style independently controllable — better similarity *and* better control.
- **Speaker similarity is a speaker-verification cosine — and it is meaningless without naming the embedder.** SIM 0.6 from one embedder can equal 0.4 from another.
- **Evaluate all three axes.** SIM for identity, WER for intelligibility, human SMOS/MOS for similarity and naturalness. No single number is enough; fix the harness and treat tiny differences as noise.
- **The open-vs-closed gap is now polish, not capability.** F5-TTS rivals research SOTA on SIM/WER in the open; closed services (ElevenLabs) lead on latency, language breadth, long-input robustness, and product surface.
- **Cloning is an impersonation engine — design for consent and provenance from line one.** Gate high-fidelity cloning behind consent, watermark outputs, disclose synthesis. Prefer voice *design* when you do not need a specific real person.

## Further reading

- **VALL-E** — Wang et al., "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers" (2023). The codec-LM cloning recipe.
- **Voicebox** — Le et al., "Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale" (2023). Flow-matching speech via masked infill.
- **NaturalSpeech 3** — Ju et al., "NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models" (2024). Attribute factorization.
- **F5-TTS** — Chen et al., "F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching" (2024). Alignment-free flow matching; and E2-TTS (Eskimez et al., 2024).
- **ECAPA-TDNN** — Desplanques et al., "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification" (2020). The SV embedder behind SIM.
- **Whisper** — Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision" (2022). The ASR for WER.
- **Coqui TTS / XTTS docs** and the **F5-TTS**, **Parler-TTS**, **OpenVoice**, and **Fish-Speech** repositories for the runnable open models.
- Within this series: the codec-LM sibling [neural-codec-language-model-tts-vall-e](/blog/machine-learning/audio-generation/neural-codec-language-model-tts-vall-e), the flow recipe [flow-matching-and-consistency-for-audio](/blog/machine-learning/audio-generation/flow-matching-and-consistency-for-audio), the TTS foundations [text-to-speech-from-tacotron-to-vits](/blog/machine-learning/audio-generation/text-to-speech-from-tacotron-to-vits), the metrics [audio-quality-metrics](/blog/machine-learning/audio-generation/audio-quality-metrics), the foundation [why-audio-generation-is-hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard), and the capstone [building-an-audio-generation-stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack). Forward to [prosody-emotion-and-expressive-speech](/blog/machine-learning/audio-generation/prosody-emotion-and-expressive-speech), the safety post [audio-deepfakes-watermarking-and-voice-safety](/blog/machine-learning/audio-generation/audio-deepfakes-watermarking-and-voice-safety), and the landscape survey [the-2026-audio-model-landscape](/blog/machine-learning/audio-generation/the-2026-audio-model-landscape).
