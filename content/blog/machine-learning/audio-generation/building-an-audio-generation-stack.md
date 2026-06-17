---
title: "Building an Audio Generation Stack: The End-to-End Playbook"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Assemble the whole audio series into real product decisions: choose a model by task, build the pipeline, fine-tune a voice, hit a latency target, and ship it with evaluation and safety."
tags:
  [
    "audio-generation",
    "audio-synthesis",
    "text-to-speech",
    "music-generation",
    "neural-audio-codec",
    "voice-cloning",
    "generative-ai",
    "deep-learning",
    "mlops",
  ]
category: "machine-learning"
subcategory: "Audio Generation"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/building-an-audio-generation-stack-1.png"
---

A product manager drops a one-line request on your desk: "We want our app to talk to users in real time, in five languages, and it has to sound human." A week later a different team asks for a button that turns a text prompt into a thirty-second background track for their videos. A third wants foley — footsteps, door creaks, rain — generated on demand for a game engine. Three requests, three completely different stacks, and a dozen ways to get each one wrong: a voice agent that takes four seconds to answer, a music tool that costs more per song than you charge for the whole subscription, a TTS service that nails English and mangles Vietnamese tone, a voice clone that sounds like the speaker on a good day and a stranger on a bad one.

This post is the map. Across this series we built every component from first principles — codecs, vocoders, autoregressive audio language models, diffusion and flow for audio, text-to-speech from Tacotron to VITS, zero-shot voice cloning, streaming and full-duplex speech, music generation, latent diffusion for music, honest evaluation, and the safety layer. Here we assemble all of it into decisions. By the end you will be able to look at any audio request, pick the right approach in two questions, sketch the pipeline, decide whether to fine-tune or prompt, hit a concrete latency and cost target, and ship it behind an evaluation gate and a watermark. The diagram below is the decision we start every project with: what are you generating, and what is the binding constraint?

![Decision tree showing the first fork on task type speech music or sound effects and the second fork on quality versus speed and cost](/imgs/blogs/building-an-audio-generation-stack-1.png)

Everything here ties back to the spine of the whole series: the **audio stack** — conditioning, then optional codec encode, then a generative model (an autoregressive token language model, a diffusion model, or a flow-matching model), then a vocoder or codec decode, then a waveform — pulled in four directions by **fidelity, controllability, speed, and length**. If you have not read it, [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard) sets up that tension and why our ears are so unforgiving. This is its bookend: the playbook that turns the tension into a shipping checklist.

## 1. Choosing the approach: two questions, not twenty

The single most common mistake I see is starting from a model ("let's use MusicGen") instead of a constraint ("we need fewer than 300 milliseconds to first audio at under one cent per minute"). Pick the model last. The decision is genuinely only two questions deep.

**Question one: what are you generating?** Speech, music, or sound effects and foley. These are different problems with different state-of-the-art and different licenses. Speech wants intelligibility (low word error rate) and speaker identity; music wants long-range structure and timbre; sound effects want sharp transients and the right envelope. A model tuned for one is mediocre at the others. MusicGen will not read you a paragraph; XTTS will not write you a chorus.

**Question two: what is the binding constraint?** Almost always exactly one of these dominates and the rest are negotiable:

- **Latency** — a live voice agent that has to answer before the human notices a pause. The bar is *time to first audio* (TTFA) under roughly 300 milliseconds, and a real-time factor (RTF) at or below 1.0 so you never fall behind the conversation.
- **Quality** — a hero voiceover, a marketing jingle, a flagship demo. Here a few hundred milliseconds and a few cents per minute do not matter; mean opinion score (MOS) and the absence of artifacts do.
- **Control** — you need a specific speaker, a specific genre, a specific BPM, a specific emotional read. Open models you can fine-tune give you the most; hosted APIs give you a prompt box and hope.
- **Cost** — you are generating millions of minutes and a tenth of a cent per minute is the difference between profit and bankruptcy.
- **License and data residency** — you cannot send user audio to a third party, or you cannot ship a model whose weights forbid commercial use.
- **On-device** — it has to run on a phone with no network, which rules out anything you cannot quantize under a few hundred megabytes.

Notice that **quality**, **control**, **cost**, and **latency** pull against each other. The best-sounding hosted TTS will not give you a fine-tunable speaker; the cheapest self-hosted model will not match a frontier API on a hard accent; the fastest streaming model trades a little fidelity for the speed. The job is to identify which one constraint is non-negotiable, satisfy it first, and spend the slack on the others.

Why only two questions? Because the audio field has converged enough that, once you fix the task and the binding constraint, the set of reasonable models is small — usually two or three — and the remaining choice is taste plus benchmarking. Five years ago you also had to choose a representation (raw waveform vs spectrogram vs codec tokens), a vocoder architecture, and a training recipe from scratch. Today those are largely settled: codecs are the standard intermediate for token models, mel-plus-GAN-vocoder is the standard for streaming TTS, latent diffusion is the standard for long-form music. The series you just read is the derivation of *why* those converged; this playbook is the consequence — you get to start from the converged answer and spend your effort on the product.

A note on the two constraints that quietly kill projects: **license** and **on-device**. License is the one people discover the week before launch. Many "open" audio models ship under research-only or non-commercial licenses; some codecs and vocoders have their own terms; some hosted APIs forbid using their output to train a competing model. Read every license in your stack *before* you build on it, because discovering a non-commercial clause after you have shipped is a rewrite, not a patch. On-device is the constraint that reshapes everything: if it must run offline on a phone, you are limited to models you can quantize under a few hundred megabytes and run at an acceptable RTF on a mobile NPU or CPU, which today means a small TTS model (a quantized VITS or a compact streaming model) and rules out the large music and frontier-clone models entirely. Decide both of these in question two, not in the launch retro.

#### Worked example: routing three real requests

Take the three requests from the intro and run them through the two questions.

- **Real-time voice agent.** Task: speech. Binding constraint: latency (a human is waiting). Therefore a streaming-capable model — F5-TTS with a streaming vocoder, or a full-duplex model like Moshi if you need barge-in — on your own GPU so you control the tail latency. Quality is "good enough to be natural," not "best in the world."
- **Hosted music tool.** Task: music with vocals and lyrics. Binding constraint: quality (users will judge it against Spotify). Therefore a hosted API — Suno or Udio — because the open music models do not yet match them on full songs with coherent vocals, and you do not want to operate a 4-minute-render GPU farm on day one. Cost per minute is high but you pass it through.
- **Multilingual TTS API.** Task: speech. Binding constraint: control plus cost (specific languages, specific voices, high volume, possibly data residency). Therefore self-hosted XTTS v2 or a fine-tuned VITS per language, on your own GPUs, where you control voices, latency, and the per-minute cost.

Three requests, three different answers, and the model fell out of the constraint every time. That is the whole discipline. The branching logic above is exactly what the decision tree in the intro encodes; the rest of this post fills in the pipeline, the customization, the latency math, the serving, and the gates.

## 2. The pipeline, recapped: five stages, five knobs

Every audio system in this series — TTS, music, SFX — is the same five-stage spine. If you internalize the spine and the one dominant knob at each stage, you can debug any audio system by asking "which stage is the bottleneck?" Here is the stack.

![Layered stack of the five audio pipeline stages from conditioning through codec model vocoder to waveform with the knob at each stage](/imgs/blogs/building-an-audio-generation-stack-2.png)

**Stage 1 — conditioning.** The thing you want the model to obey: text (tokenized and run through a text encoder like a byte-pair tokenizer feeding T5 or a phonemizer), a speaker reference clip, a melody, a genre tag, a BPM, an emotion label. The knob is *what* and *how strongly* you condition, controlled at inference by classifier-free guidance (CFG). Higher guidance pulls the output toward the prompt at some cost to diversity and naturalness. We covered the mechanics in [conditioning and control in audio generation](/blog/machine-learning/audio-generation/conditioning-and-control-in-audio-generation); the guidance derivation itself is shared with images, so I link out to [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance) rather than re-derive it. The practical tuning lesson from across the series: guidance has a knee, not a slope. Too low and the output ignores your prompt; too high and speech turns clipped and over-articulated, music turns harsh, and diversity collapses to a single mode. For speech a CFG around 2–3 is typical; for music around 3–4. Sweep it on a held-out set once and lock it, rather than leaving it at a library default that was tuned for a different model. For TTS the conditioning also carries *prosody and emotion* when the model supports it, via a reference encoder or a style description — the lever covered in [prosody, emotion, and expressive speech](/blog/machine-learning/audio-generation/prosody-emotion-and-expressive-speech) — which is the difference between a correct read and a *good* read.

**Stage 2 — codec encode (optional).** If the model works over discrete tokens (an autoregressive audio language model like VALL-E or MusicGen) or a continuous latent (latent diffusion like Stable Audio), you first compress the waveform with a neural codec — EnCodec, DAC, or Mimi — into a short sequence of residual-vector-quantization (RVQ) tokens or a latent. The knob is *bitrate*: more codebooks or higher kbps means higher fidelity and a longer token sequence (slower model). The codec is the audio analogue of the image VAE; we built it in [neural audio codecs, the tokenizer of sound](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound). Diffusion-on-mel and direct-waveform models skip this stage.

**Stage 3 — the generative model.** The engine that turns conditioning into audio tokens or a latent: an autoregressive token language model ([WaveNet to AudioLM](/blog/machine-learning/audio-generation/autoregressive-audio-models-wavenet-to-audiolm)), a diffusion model ([diffusion for audio](/blog/machine-learning/audio-generation/diffusion-for-audio)), or a flow-matching model ([flow matching and consistency for audio](/blog/machine-learning/audio-generation/flow-matching-and-consistency-for-audio)). The knobs are *model size* and, for diffusion and flow, *number of sampling steps*. This is almost always the latency bottleneck.

**Stage 4 — vocoder or codec decode.** Turn the model's output (a mel-spectrogram, RVQ tokens, or a latent) back into a waveform. A GAN vocoder like HiFi-GAN, BigVGAN, or Vocos does this in essentially one forward pass; a codec decoder does it from tokens. The knob is *vocoder choice and steps* — a one-step GAN vocoder is roughly 100× faster than a diffusion vocoder at a barely perceptible quality cost. We made the case in [GAN vocoders, HiFi-GAN and fast synthesis](/blog/machine-learning/audio-generation/gan-vocoders-hifi-gan-and-fast-synthesis).

**Stage 5 — waveform and post-processing.** The final 24 kHz or 44.1 kHz waveform, then loudness normalization (target around -16 LUFS for speech, -14 for music), optional de-essing or a high-pass to kill subsonic rumble, fade-in/out, and the safety watermark we add in Section 9. The knob is *post-processing chain*.

That is the whole machine. A TTS system is `text → (no codec) → VITS/F5 → HiFi-GAN → wav`. MusicGen is `text → EnCodec → AR LM → EnCodec decode → wav`. Stable Audio is `text+timing → VAE encode → diffusion → VAE decode → wav`. Same five stages every time. When something is slow, you profile per stage; when something sounds wrong, you bisect per stage. The image series uses the identical mental frame for its pipeline in [building an image generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack) — conditioning, latent, model, decode — which is no accident: generative media is generative media, and the audio-specific parts are the codec and the vocoder.

### A first runnable pipeline

Here is the speech spine end to end with 🤗 `transformers`, loading a VITS model, synthesizing, and writing a wav. This is the smallest thing that actually runs.

```python
import torch
import soundfile as sf
from transformers import VitsModel, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# VITS bundles the generative model AND the vocoder end to end (text -> wav).
model = VitsModel.from_pretrained("facebook/mms-tts-eng").to(device)
tok = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

text = "the quick brown fox jumps over the lazy dog"
inputs = tok(text, return_tensors="pt").to(device)

with torch.no_grad():
    out = model(**inputs).waveform  # (1, n_samples) at model.config.sampling_rate

wav = out.squeeze().cpu().numpy()
sr = model.config.sampling_rate
sf.write("fox.wav", wav, sr)
print(f"wrote {len(wav)/sr:.2f}s of audio at {sr} Hz")
```

VITS is end-to-end (it fuses stages 3 and 4 into one VAE-plus-flow-plus-adversarial network, which is exactly why it is fast). The moment you move to a token model like MusicGen, you see all five stages explicitly:

```python
import torch, soundfile as sf
from transformers import AutoProcessor, MusicgenForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
proc = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained(
    "facebook/musicgen-small"
).to(device)

inputs = proc(
    text=["upbeat lo-fi hip hop with a warm vinyl crackle"],
    padding=True, return_tensors="pt",
).to(device)

# max_new_tokens ~ 50 tokens/sec of audio for the 50 Hz EnCodec frame rate.
with torch.no_grad():
    tokens = model.generate(**inputs, do_sample=True, guidance_scale=3.0,
                            max_new_tokens=256)  # ~5 seconds

audio = tokens[0, 0].cpu().numpy()
sr = model.config.audio_encoder.sampling_rate
sf.write("lofi.wav", audio, sr)
```

Internally MusicGen encodes nothing at inference (you give it text), runs the autoregressive LM over EnCodec's 50 Hz token grid, and EnCodec's decoder vocodes the tokens to a 32 kHz waveform. Five stages, two lines of API.

### The codec-bitrate knob, made concrete

Stage 2's bitrate knob deserves its own moment, because it is the lever most people leave on the default and pay for in latency. A neural codec like EnCodec or DAC turns the waveform into a stack of RVQ codebooks; each codebook adds bits per frame and a token per frame to the sequence the model must emit. At a 50 Hz frame rate with 8 codebooks of 10 bits each, you are at $50 \times 8 \times 10 = 4000$ bits per second — 4 kbps — and the AR model emits $50 \times 8 = 400$ tokens per second of audio. Drop to 4 codebooks and you halve both the bitrate and the token count: the model runs roughly twice as fast and the audio loses a little high-frequency detail and stereo precision. The rate-distortion curve is steep at the bottom (going from 1.5 to 3 kbps is a large audible improvement) and flat at the top (going from 6 to 12 kbps is barely audible on speech), so the engineering move is to find the knee and sit just above it. We derived this curve in [residual vector quantization](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq); the practical consequence here is that *the codec bitrate is a latency knob you set deliberately per task* — speech tolerates a lower bitrate than music, and a lower-frame-rate codec (Mimi at 12.5 Hz) is a free 4× speedup for any AR model that can use it.

```python
import torch
from transformers import EncodecModel, AutoProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
codec = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device)
proc = AutoProcessor.from_pretrained("facebook/encodec_24khz")

# Encode at two bandwidths and compare token counts (the latency proxy).
import torchaudio
wav, sr = torchaudio.load("speaker_clean.wav")
inp = proc(raw_audio=wav.squeeze().numpy(), sampling_rate=24000,
           return_tensors="pt").to(device)

for bw in (1.5, 6.0):                       # target kbps
    enc = codec.encode(inp["input_values"], inp["padding_mask"], bandwidth=bw)
    codes = enc.audio_codes                  # (1, 1, n_codebooks, n_frames)
    n_cb = codes.shape[-2]
    print(f"{bw} kbps -> {n_cb} codebooks, "
          f"{codes.shape[-1]} frames, {n_cb * codes.shape[-1]} tokens")
```

The token count this prints is, to first order, the AR model's work — halve it and you halve the model-stage latency. That is the bitrate knob doing its job.

## 3. The trade-off table: approach by quality, latency, control, cost

Once you know your task and your binding constraint, the deployment *approach* — self-host open weights, call a hosted API, use a streaming open model, or use a voice-clone API — is a four-way trade. There is no globally best choice; there is the choice that satisfies your one constraint without blowing the others. Here is how the approaches actually compare.

![Matrix comparing open self-host hosted API streaming open and voice-clone API across quality latency control and cost per minute](/imgs/blogs/building-an-audio-generation-stack-3.png)

| Approach | Quality | Latency | Control | Cost/min (approx) | License | Reach for it when |
|---|---|---|---|---|---|---|
| Open, self-hosted (XTTS, MusicGen, F5) | Good–very good | 0.3–1.0× RTF | Full: fine-tune, LoRA, custom voices | \$0.001–0.005 (GPU amortized) | Mostly permissive (check each) | High volume, data residency, custom voices, cost-sensitive |
| Hosted API (ElevenLabs, Suno, Udio) | Best | 0.5–2 s TTFA | Prompt + a few params only | \$0.05–0.30 | Vendor terms | You need the absolute top quality and zero ops |
| Streaming open (F5, Moshi, Kokoro) | Good | <300 ms TTFA | Speaker ref, some prosody | \$0.003–0.006 (GPU) | Permissive | Real-time agents, conversational, barge-in |
| Voice-clone API (ElevenLabs, PlayHT) | Best clone | 1–3 s TTFA | Clone + style, consent-gated | \$0.10–0.30 | Vendor terms | One-off premium clones, no infra appetite |

A few honest notes on the numbers. The self-hosted "cost per minute" is the *amortized GPU cost* assuming reasonable utilization — an RTX 4090 at roughly \$0.30/hour amortized, generating audio at 5–10× real time, lands around a tenth of a cent of GPU time per audio-minute, and the rest is your ops overhead. The hosted API prices are list prices for character- or second-based billing converted to per-minute; they are an order of magnitude higher because you are paying for their model, their margin, and their uptime. "Best" quality for hosted speech is real today: ElevenLabs-class TTS still beats open models on the hardest accents and the most expressive reads, and Suno and Udio still beat open music models on full songs with coherent vocals. The gap is closing — F5-TTS and XTTS are genuinely good — but on 2026-06-17 it has not closed. Be honest with your PM about that. We surveyed exactly where the gap sits in [the 2026 audio model landscape](/blog/machine-learning/audio-generation/the-2026-audio-model-landscape).

The decision rule that falls out: **if your binding constraint is quality and volume is low, pay for the API.** If it is cost, control, latency, or data residency, self-host. If it is latency specifically, self-host a streaming model. The middle column — control — is the one people underweight: the day your PM asks for "the same narrator across 10,000 episodes," a prompt-only API cannot guarantee it and a fine-tuned open model can.

## 4. Customization: prompt, clone, fine-tune, or LoRA

"Customization" is four different operations with four different costs, and choosing the wrong one wastes either a week of GPU time or a month of mediocre output. Here is the ladder, cheapest to most expensive.

**Zero-shot prompt (no training).** You hand the model a 3–10 second reference clip (a voice, a melody) at inference time and it imitates it in-context. This is what VALL-E and XTTS made famous — the model learned in-context cloning during pretraining, so at deploy time you pay nothing. It is instant and free and gets you roughly 80% of the way to a target voice. It drifts on long generations, struggles with accents far from the training distribution, and the speaker identity wobbles between calls. We covered the in-context-learning paradigm in [the neural-codec language-model TTS, VALL-E](/blog/machine-learning/audio-generation/neural-codec-language-model-tts-vall-e).

**Voice clone (light fine-tune or speaker embedding).** A step up: you fit a small speaker representation, or do a brief fine-tune on 5–30 minutes of one speaker, to lock in identity. This is the [zero-shot voice cloning and TTS frontier](/blog/machine-learning/audio-generation/zero-shot-voice-cloning-and-the-tts-frontier) territory. Cost: minutes to a couple of GPU-hours.

**LoRA (parameter-efficient fine-tune).** You freeze the base model and train a small low-rank adapter — typically a few million parameters — on a target voice, style, or genre. This is the sweet spot for "a consistent custom voice" or "always generate in this artist's style" because the adapter is tiny (megabytes), you can hot-swap adapters per customer, and it does not catastrophically forget the base model. The PEFT machinery is shared with the LLM and image worlds; the method is identical, only the data is audio.

**Full fine-tune.** You update all the weights on a substantial dataset (tens of hours). You reach for this only when you are adding a whole new language, a whole new domain (medical narration, a constructed conlang), or fixing a systematic base-model weakness. It is expensive, it risks catastrophic forgetting, and it is rarely the right first move.

![Before-after comparison of zero-shot prompting which is instant and cheap against fine-tuning which buys consistency and speaker fidelity](/imgs/blogs/building-an-audio-generation-stack-4.png)

The decision is a function of how much consistency you need and how much clean data you have:

| Need | Method | Data | Compute | Buys you |
|---|---|---|---|---|
| One-off imitation | Zero-shot prompt | 3–10 s ref | None | Instant, free, ~80% there |
| One consistent voice | Voice clone / light FT | 5–30 min | Minutes–2 GPU-hr | Locked identity |
| Custom voice/style at scale | LoRA | 10–30 min | 1–4 GPU-hr | Hot-swappable, tiny, stable |
| New language/domain | Full fine-tune | 10+ hours | Many GPU-hr | New capability, risk of forgetting |

A word on **LoRA for audio specifically**, because it is underused. The adapter is a pair of low-rank matrices injected into the attention and feed-forward projections; with rank 16 on a model with hidden size 1024 you are training on the order of a few million parameters instead of hundreds of millions, the optimizer state fits comfortably on a single consumer GPU, and the trained adapter is a few megabytes on disk. The operational payoff is what makes it the production sweet spot: you can keep one frozen base model resident in GPU memory and hot-swap a per-customer adapter into it on each request, so a thousand custom voices cost you one base model plus a thousand tiny adapters, not a thousand full models. The catch is that LoRA captures *style and identity* well but does not add genuinely new *capability* — it will lock a voice or a genre, but it will not teach a model a language it never saw. For new capability you need a full fine-tune and a real dataset, which is why "new language" sits at the top of the ladder. The fine-tuning machinery itself is shared with the LLM and image worlds; only the data modality changes.

The single biggest predictor of success is not the method — it is the *cleanliness of the data*. Three seconds of studio-clean reference beats thirty seconds of laptop-mic audio with HVAC hum every time. A noisy reference does not just sound noisy; the model learns the noise as part of the speaker identity, so the clone *always* has a faint hum or room signature baked in, on every utterance, forever. This is the failure mode behind most disappointing clones, and it is entirely a data problem, not a model problem. Which brings us to the actual workflow.

## 5. A real voice-clone / fine-tune workflow

I have shipped enough voice clones to know the failure mode is never the model — it is the data and the gate. Here is the workflow I run, end to end. The shape is a curate-train-evaluate loop with a hard quality gate before anything ships.

![Graph of the voice-clone fine-tune loop from curation splitting into denoise and transcribe both feeding training then an evaluation gate before shipping](/imgs/blogs/building-an-audio-generation-stack-5.png)

**Step 1 — curate clean audio.** Collect 10–30 minutes of the target speaker. Single speaker, no background music, no overlapping speech, consistent mic and room. Then clean it: trim silence, high-pass at 80 Hz to kill rumble, denoise to a noise floor below roughly -60 dB, normalize loudness, and *resample to the model's rate* (24 kHz for XTTS). This step matters more than every later step combined.

```python
import torchaudio
import torchaudio.functional as F
import torch

wav, sr = torchaudio.load("raw_speaker.wav")
wav = wav.mean(0, keepdim=True)             # mono
if sr != 24000:
    wav = torchaudio.transforms.Resample(sr, 24000)(wav)
    sr = 24000
wav = F.highpass_biquad(wav, sr, cutoff_freq=80.0)   # kill subsonic rumble
# loudness normalize to ~ -23 dBFS RMS (cheap proxy; use pyloudnorm for true LUFS)
rms = wav.pow(2).mean().sqrt()
wav = wav * (10 ** (-23 / 20) / (rms + 1e-9))
wav = wav.clamp(-1.0, 1.0)
torchaudio.save("speaker_clean.wav", wav, sr)
```

**Step 2 — transcribe and segment.** Run an ASR model (Whisper) to get transcripts and word-level timestamps, then cut the audio into 3–15 second clips on sentence boundaries. Clean transcripts are what let the model align text to acoustics during fine-tuning; bad transcripts inject noise into the loss.

```python
import whisper
asr = whisper.load_model("base")
result = asr.transcribe("speaker_clean.wav", word_timestamps=True)
# Each segment becomes a (clip.wav, transcript) training pair; cut on segment["start"]/["end"].
pairs = [(seg["start"], seg["end"], seg["text"].strip()) for seg in result["segments"]]
print(f"{len(pairs)} training pairs, total "
      f"{sum(e - s for s, e, _ in pairs):.0f}s")
```

**Step 3 — clone or train.** For an instant clone, you simply pass the reference at inference. For a stronger result, fine-tune. With Coqui XTTS the instant path is one call:

```python
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

# Zero-shot: condition on the cleaned reference, synthesize new text.
tts.tts_to_file(
    text="Here is a sentence in the cloned voice.",
    speaker_wav="speaker_clean.wav",
    language="en",
    file_path="cloned.wav",
)
```

For a LoRA or light fine-tune you would instead launch a training run on the (clip, transcript) pairs from Step 2; the data pipeline above is 90% of the work and the trainer config is the easy 10%.

**Step 4 — evaluate before you ship.** This is the step everyone skips and everyone regrets. Two metrics, both cheap and both decisive:

- **WER (word error rate)** — synthesize a held-out set of sentences, run them back through Whisper, and compare to the reference text. WER measures intelligibility. Anything above roughly 5–8% means the clone is mumbling and you are not shipping it.
- **SMOS (speaker similarity MOS)** — a small human study, or a speaker-verification model's cosine similarity as a cheap proxy, asking "does this sound like the target speaker?" A light fine-tune typically buys +0.3 to +0.6 SMOS over zero-shot on a hard speaker.

```python
import whisper, jiwer
asr = whisper.load_model("base")
refs = ["the quick brown fox jumps over the lazy dog", "she sells sea shells"]
# Synthesize each ref with the clone, save to gen_i.wav, then:
hyps = [asr.transcribe(f"gen_{i}.wav")["text"].strip().lower() for i in range(len(refs))]
wer = jiwer.wer(refs, hyps)
print(f"clone WER = {wer:.1%}  (ship gate: < 5%)")
```

If WER is over the gate, the culprit is almost always Step 1: noisy data or a sample-rate mismatch. Go back and clean harder. The whole loop is the figure above — and the gate is the part that is non-optional.

#### Worked example: cloning a narrator from 12 minutes

A real run. Input: 12 minutes of a podcast narrator, laptop mic, mild room reverb, occasional keyboard clicks. Zero-shot XTTS gave WER 9.2% and a speaker-similarity cosine of 0.71 — intelligible but the identity wandered and the keyboard clicks bled into the timbre. After Step 1 cleaning (highpass, denoise, click removal) and a light LoRA fine-tune (rank 16, ~4M trainable params, 1.5 GPU-hours on a 4090), WER dropped to 4.1% and similarity rose to 0.86. The cost: about \$0.50 of GPU time and 20 minutes of human attention curating clips. The lesson: the +0.15 similarity came half from the fine-tune and half from the cleaning — and the cleaning was free.

## 6. The science: a latency and cost model you can compute

You cannot hit a latency target you cannot compute. Here is the model. For a single non-streaming request, end-to-end latency is the sum of the three stages that take real time:

$$L_\text{total} = L_\text{encode} + L_\text{model} + L_\text{vocode}$$

The **real-time factor** is the generation wall-clock divided by the duration of the audio produced:

$$\text{RTF} = \frac{L_\text{total}}{T_\text{audio}}$$

RTF below 1.0 means you generate faster than playback — mandatory for streaming. RTF of 0.1 means a 10-second clip renders in 1 second. For an autoregressive model the dominant term is the model: it emits tokens one frame at a time, so $L_\text{model} \approx N_\text{frames} \times t_\text{step}$, where $N_\text{frames} = T_\text{audio} \times f_\text{frame}$ (the codec frame rate, e.g. 50 Hz for EnCodec, 12.5 Hz for Mimi) and $t_\text{step}$ is the per-token forward time. This is why a *lower frame-rate codec* (Mimi at 12.5 Hz versus EnCodec at 50 Hz) is a 4× latency win for an AR model before you touch anything else — fewer tokens to emit. For a diffusion or flow model the model term is instead $L_\text{model} \approx S \times t_\text{NFE}$, the number of sampling steps times the per-step network evaluation, which is why few-step flow and consistency models are the big lever there.

For **streaming**, the metric that matters is not total latency but **time to first audio**:

$$\text{TTFA} = L_\text{encode} + L_\text{first chunk} + L_\text{vocode chunk}$$

You start playing the first chunk while the rest generates. As long as RTF < 1, the buffer never underruns and the human hears continuous speech after a short initial delay. This is the entire trick behind real-time voice, and we built it in [real-time streaming and full-duplex speech](/blog/machine-learning/audio-generation/real-time-streaming-and-full-duplex-speech).

There is a chunk-size trade buried here that is worth stating plainly. Smaller chunks lower TTFA (you emit sooner) but raise overhead — more vocoder invocations, more network frames, more boundary artifacts to crossfade — and they give the model less lookahead, which can hurt prosody at chunk boundaries. Larger chunks are more efficient and sound smoother but raise TTFA and the granularity at which a full-duplex system can be interrupted. The sweet spot for conversational speech is around 150–250 ms chunks: short enough that TTFA feels instant and barge-in is responsive, long enough that the vocoder overhead and boundary handling stay cheap. You also crossfade a few milliseconds across each chunk boundary so the waveform is continuous and no click leaks in. For a full-duplex agent the chunk size also bounds your *interruption latency* — how fast the model can stop talking when the human starts — so there it is a UX parameter, not just an efficiency one.

There is a subtlety worth making precise because it changes how you optimize. AR audio decoding is *memory-bandwidth bound*, not compute bound: at batch 1 the model reads its entire weight matrix from GPU memory for every single token, and the audio frame rate means tens of thousands of tokens for a few seconds of audio. The per-token time $t_\text{step}$ is therefore roughly the model's weight bytes divided by the GPU's memory bandwidth, not its FLOPs divided by its FLOP rate. This is why two interventions help so much: **quantization** (int8 halves the weight bytes you read, int4 quarters them, so $t_\text{step}$ drops nearly linearly) and **batching** (you amortize one weight read across many requests, which is free throughput). It is also why a smaller model helps more than its FLOP count suggests — fewer weight bytes to stream per token. Diffusion and flow models are the opposite, compute bound per step, so there the lever is *fewer steps* (few-step flow, consistency distillation), and quantization helps less.

Putting the model together, the total for an AR system is approximately

$$L_\text{total} \approx \underbrace{c_\text{enc}}_{\text{fixed}} + \underbrace{T_\text{audio} \cdot f_\text{frame} \cdot n_\text{cb} \cdot \frac{B_\text{weights}}{\text{bw}_\text{GPU}}}_{\text{model, bandwidth-bound}} + \underbrace{T_\text{audio} \cdot c_\text{voc}}_{\text{vocoder}}$$

which makes every lever legible: lower the frame rate $f_\text{frame}$ (Mimi vs EnCodec) or the codebook count $n_\text{cb}$ (Section 2's bitrate knob), shrink the weight bytes $B_\text{weights}$ (smaller model, int8), or raise the effective bandwidth via batching. The four levers, and roughly what each buys:

- **A smaller model** cuts $t_\text{step}$ and $t_\text{NFE}$ — often 2–4× — at some quality cost. It shrinks $B_\text{weights}$ directly.
- **Quantization** (int8 or int4 weights) cuts memory bandwidth, the real bottleneck for AR decoding, for another ~1.5–2× with negligible quality loss at int8.
- **A faster vocoder** (one-step HiFi-GAN/Vocos instead of a diffusion vocoder) collapses $L_\text{vocode}$ from hundreds of milliseconds to ~10 ms — up to 100×.
- **Streaming** does not lower total work but it hides it: TTFA replaces $L_\text{total}$ as the number the user feels.

Here is the budget for a concrete 10-second TTS request, before and after applying all four levers, on a single RTX 4090.

![Matrix of the latency budget per pipeline stage showing baseline optimized and the lever that moves each from encode to model to vocode to total](/imgs/blogs/building-an-audio-generation-stack-6.png)

| Stage | Baseline | Optimized | Lever applied |
|---|---|---|---|
| Encode / condition text | ~40 ms | ~20 ms | Cache text-encoder outputs |
| Model generate (10 s audio) | ~6.0 s | ~1.2 s | Smaller model + int8 |
| Vocode / decode | ~0.8 s | ~0.1 s | One-step HiFi-GAN/Vocos |
| **Total / RTF** | **~6.8 s · 0.68×** | **~1.3 s · 0.13×** | Stream chunks for TTFA |

The baseline already has RTF below 1, so it would technically stream — but at 0.68× the buffer is dangerously thin and any GPU contention causes an audible underrun. Optimized to 0.13× you have a 7× safety margin and a TTFA, with streaming, of well under 300 ms. These numbers are representative of a small-to-medium TTS model on a 4090; your mileage varies, so *measure your own RTF* with a warm GPU (discard the first run — CUDA kernels and `torch.compile` cache on first call) and a fixed text and seed.

```python
import time, torch
torch.cuda.synchronize()
# warm-up (compile + cache kernels) — do NOT count this run
_ = model(**inputs)
torch.cuda.synchronize()

t0 = time.perf_counter()
out = model(**inputs).waveform
torch.cuda.synchronize()
gen_s = time.perf_counter() - t0

audio_s = out.shape[-1] / model.config.sampling_rate
print(f"RTF = {gen_s / audio_s:.3f}  ({gen_s:.2f}s to make {audio_s:.2f}s)")
```

#### Worked example: hitting a 250 ms TTFA budget

A voice agent needs TTFA under 250 ms. Start: an AR TTS model on a 4090, EnCodec 50 Hz, full precision, batch 1, no streaming — measured TTFA ~1.9 s (it generates the whole utterance before the vocoder runs). Apply the levers in order. (1) Switch to a 12.5 Hz codec frame rate: 4× fewer tokens, TTFA drops to ~700 ms. (2) int8 weights: ~1.6× faster decode, ~440 ms. (3) Stream the first 200 ms chunk and play it while the rest generates: now the user hears audio after the first chunk's worth of model steps plus one vocoder pass — ~210 ms. Under budget, with the remaining audio rendered comfortably faster than playback. The order matters: the codec frame-rate change was free quality-wise and bought the most.

## 7. Serving: a real endpoint, batching, and the cost per minute

A model that runs in a notebook is not a product. Production audio serving has three concerns: **streaming** the output, **batching** across concurrent requests to keep the GPU busy, and **right-sizing** the hardware so the cost per minute pencils out. Here is the topology.

![Graph of a streaming serving topology with a FastAPI gateway batching across two GPU workers feeding one watermark stage then streaming chunks back](/imgs/blogs/building-an-audio-generation-stack-7.png)

The gateway accepts WebSocket or chunked-HTTP requests, batches them dynamically (collect requests for a few milliseconds, run them as one batch through the GPU — this is where throughput comes from), dispatches to a pool of GPU workers running the model in `bfloat16` with `torch.compile`, watermarks every chunk (Section 9), and streams the audio back. Here is a minimal FastAPI streaming endpoint — the sketch, not the whole framework.

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import io, soundfile as sf, numpy as np

app = FastAPI()

def synth_stream(text: str, chunk_s: float = 0.2):
    """Yield ~200 ms WAV chunks as the model produces them."""
    sr = model.config.sampling_rate
    for audio_chunk in model.stream_generate(text):     # model-specific generator
        buf = io.BytesIO()
        sf.write(buf, audio_chunk.astype(np.float32), sr, format="WAV")
        yield buf.getvalue()

@app.post("/tts")
def tts(text: str):
    return StreamingResponse(synth_stream(text), media_type="audio/wav")
```

`model.stream_generate` is the model-specific streaming generator (XTTS exposes `inference_stream`; a custom AR model yields after every K tokens once a chunk's worth of frames is ready). The pattern is the same regardless: produce a chunk, vocode it, watermark it, flush it, repeat.

**Batching.** Audio batching is trickier than text because utterances have different lengths and AR models finish at different times. The pragmatic approach is *continuous batching* — as one request in the batch finishes, slot in a waiting one — the same idea LLM servers use. For diffusion and flow models, all requests run the same fixed number of steps, so static batching by similar-length requests is simple and effective. Batch size is a throughput-vs-latency dial: bigger batches raise GPU utilization and lower cost per minute but raise the tail latency of any single request.

The non-obvious tension is between batching and streaming. Batching wants to *wait* — collect requests, run them together — and streaming wants to *start immediately* and emit the first chunk fast. For a real-time agent where TTFA is sacred, you keep the batch window tiny (a few milliseconds) and accept lower GPU utilization; for a non-interactive bulk job (narrate 10,000 articles overnight) you make the batch window large and the batch deep, because nobody is waiting and throughput per dollar is all that matters. The same model serves both, with a different batching policy per endpoint. A common production layout is two pools behind the gateway: a low-latency pool with tiny batches for interactive traffic and a high-throughput pool with deep batches for bulk jobs, routed by a header on the request.

A second production reality: **the tail, not the mean, is what users feel.** A model that averages RTF 0.15 but occasionally spikes to 0.9 under GPU contention will produce audible underruns on a streaming endpoint exactly when it is busiest. You size for the p99, not the mean — which usually means running the GPU at 60–70% average utilization so the spare headroom absorbs bursts, and keeping a small queue with a hard timeout that sheds load gracefully rather than letting every request degrade. This is why the cost-per-minute math above uses a utilization assumption, not a peak — you deliberately leave headroom, and that headroom is part of the cost.

**GPU vs CPU.** Vocoders (HiFi-GAN, Vocos) are small and run acceptably on CPU — useful for an on-device or cost-floor deployment. The generative model almost always wants a GPU; an AR audio LM on CPU is typically 10–50× slower and blows any real-time budget. The exception is a tiny quantized model (a few hundred MB, int4) for fully offline on-device speech, where you accept a higher RTF for zero network and zero server cost. On-device quantization is the same toolkit as [edge AI](/blog/machine-learning/edge-ai) for any modality.

The on-device path deserves a moment because its constraints are genuinely different. You are no longer optimizing cost per minute on a server you control; you are fitting under a fixed memory ceiling (a few hundred megabytes), running on a mobile NPU or CPU with a fraction of a desktop GPU's bandwidth, and you cannot stream from a backend because there is no backend. That pushes you to the smallest viable model (a quantized VITS or a compact streaming TTS), an aggressive int4 or int8 quantization, and a vocoder small enough to run on the same chip. You give up the frontier voices and the large music models entirely — they do not fit and would not run in real time if they did. What you gain is zero latency from the network (no round trip), perfect privacy (audio never leaves the device), and zero per-minute server cost. The decision rule: go on-device only when privacy or offline operation is a hard product requirement, because you are trading away a real amount of quality and capability to get there. For most products a server with a streaming endpoint is the better engineering, and on-device is the exception you reach for when the constraint forces it.

**Cost per minute.** The arithmetic that decides whether you self-host. Take an RTX 4090 at roughly \$0.30/hour (cloud spot) or amortized owned. At RTF 0.15 with batching that keeps it busy, one GPU produces on the order of 6–7 minutes of audio per wall-clock minute, so:

$$\text{cost/min} \approx \frac{\$0.30/\text{hr}}{60 \times 6.5} \approx \$0.00077/\text{audio-min}$$

Under a tenth of a cent per minute of GPU time. Add ops, storage, and idle overhead and call it \$0.002–0.005 in practice. Compare to \$0.10–0.30 for a hosted voice API. The crossover is volume: below a few thousand minutes a month, the API's zero-ops convenience wins; above it, self-hosting saves real money and the savings compound. That is the whole self-host-versus-API economic argument in one division.

#### Worked example: the self-host break-even

Suppose you will generate 100,000 minutes of TTS a month. A hosted API at \$0.15/min costs \$15,000/month. Self-hosting on two 4090s (one for redundancy) at, say, \$500/month all-in (GPU amortization, a small CPU box, storage, monitoring) plus an engineer's part-time attention produces the same 100,000 minutes at well under one cent each — call it \$800/month of compute and ops. The self-host break-even against the API arrives around 5,000–10,000 minutes/month; at 100,000 you are saving roughly \$14,000 every month, which pays for the engineering many times over. Below break-even, do not build the GPU farm — call the API and spend your engineers elsewhere.

## 8. Choosing the engine within a task: AR vs diffusion vs flow

Inside a single task you still pick a generative family, and the choice has real consequences. The four families — autoregressive token LM, diffusion, flow matching, and GAN vocoder — trade fidelity, control, speed, and length differently. Quick reference, the way I actually decide:

| Family | Fidelity | Control | Speed (RTF) | Long form | Reach for it when |
|---|---|---|---|---|---|
| Autoregressive LM | High | Strong text + in-context clone | Slow (sequential) | Drifts late | Zero-shot cloning, music with structure (MusicGen) |
| Diffusion | Very high | Strong CFG | Many steps (slow) | Fixed window | Top-fidelity music/audio (Stable Audio) |
| Flow matching | Very high | Strong CFG | Few steps (fast) | Window-based | Fast high-quality TTS (F5, Voicebox) |
| GAN vocoder | High (mel in) | Mel only | 100×+ real time | Frame-local | Always — as the final vocoder stage |

The non-obvious point: **these are not exclusive — production stacks compose them.** A flow-matching model generates a mel-spectrogram, and a GAN vocoder turns that mel into a waveform 100× faster than a diffusion vocoder would. An AR LM generates EnCodec tokens, and EnCodec's decoder vocodes them. The generative engine and the vocoder cover each other's weaknesses: the engine gives you fidelity and control, the vocoder gives you speed at the last mile. When someone says "diffusion is too slow for real time," they usually mean the *generative* diffusion is slow — the vocoder should never be a diffusion model in a real-time system. The decision is not "which one family" but "which engine, which vocoder," and the answer for streaming TTS in 2026 is overwhelmingly "flow or a small AR LM, plus a GAN vocoder."

For the shared diffusion and flow mathematics I keep linking out rather than re-deriving — the [diffusion-from-first-principles](/blog/machine-learning/image-generation/diffusion-from-first-principles) and [flow-matching-and-rectified-flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow) posts in the image series carry the derivations, and the audio-specific parts (the codec, the vocoder, the 1D high-rate signal) are what this series added.

#### Worked example: picking the engine for a podcast-narration product

A product narrates blog articles in a chosen voice, long form (5–15 minutes per article), batch overnight, quality matters more than latency. Walk the families. AR LM: strong cloning and control, but it *drifts* over a 15-minute horizon — the voice can subtly change timbre or the prosody can flatten, and you would have to chunk and re-condition at paragraph boundaries to keep it stable. Diffusion: top fidelity but a fixed generation window, so again you generate in segments and stitch. Flow matching: fast and high-quality, segment-based like diffusion. The decision: because latency is *not* binding (it is an overnight batch), the latency disadvantage of AR or diffusion does not matter, and the deciding factor becomes *which gives the most stable long-form voice with the least stitching*. In practice a flow-matching TTS (F5-class) with paragraph-level chunking plus a short overlap-and-crossfade between chunks, all vocoded by a single fast GAN vocoder, gives the cleanest result: stable identity per chunk, no audible seams, and it finishes the overnight batch with room to spare. The engine choice fell out of the constraint (length stability, latency irrelevant), exactly as the table predicts.

One more honest caveat on the families: the benchmark you read in a paper was measured on *that paper's* data, vocoder, and device. The only RTF and MOS that matter for your decision are the ones you measure on *your* hardware with *your* text and *your* voices. Use the table to narrow to two candidates, then run your own bake-off with a fixed seed and a held-out set. The families have stable *shapes* (AR is sequential and slow, diffusion is many-step, flow is few-step, GAN-vocoder is one-step), and those shapes are what the table captures; the exact numbers are yours to measure.

## 9. Evaluation and safety: the two gates before you ship

Two non-negotiable gates stand between "it works in the demo" and "it is in production." Skip either and you will learn why the hard way.

### Evaluation: measure, do not vibe

A demo that sounds good on three cherry-picked examples tells you nothing. Before shipping, run a small honest evaluation harness — the full version is in [evaluating audio generation honestly](/blog/machine-learning/audio-generation/evaluating-audio-generation-honestly), but the production-minimum is three numbers:

- **FAD (Fréchet Audio Distance)** for music and general audio — the distributional distance between your generated set and a reference set in an embedding space. State *which embedding* (VGGish, or the more modern CLAP/PANNs) and *how many samples* (a few hundred minimum; FAD is biased and noisy on small sets). Lower is better; report the delta against a baseline, not an absolute you cannot calibrate.
- **WER (word error rate)** for speech — run a frozen ASR (Whisper) over the synthesized speech and compare to reference text. This is intelligibility, and it is cheap and decisive.
- **CLAP-score** for text-to-audio — the cosine similarity between the text prompt and the generated audio in CLAP's joint space, measuring "did it generate what I asked for."

And for anything voice-facing, a **small MOS study**: 15–20 raters, a handful of held-out utterances, a 1–5 naturalness scale, with anchors (a real recording at 5, a deliberately bad sample at 1) so the scale is calibrated. You do not need a 200-person panel to catch "this voice sounds robotic" — you need enough raters to separate "ship" from "do not ship."

```python
# Sketch: FAD between a generated set and a reference set.
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance(model_name="vggish", sample_rate=16000,
                           use_pca=False, use_activation=False)
score = fad.score("ref_audio_dir/", "generated_audio_dir/")
print(f"FAD (VGGish) = {score:.2f}  (report delta vs baseline, n>=200)")
```

The discipline: fixed text, fixed seeds, a warm GPU, a named device, and a reported sample size. Anything else is a vibe, and vibes do not survive contact with production.

A caution on FAD that bit me once and bites everyone: **FAD is not comparable across embeddings or sample sizes.** A FAD of 3.1 computed with VGGish on 200 samples means nothing next to a FAD of 1.8 computed with CLAP on 2,000 samples — they are different distances in different spaces with different small-sample bias. So FAD is only useful as a *relative* number within one fixed harness: same embedding, same reference set, same sample size, varying only the thing you are testing (the model, the bitrate, the step count). Treat it as "is variant B closer to the reference distribution than variant A, holding everything else fixed," never as an absolute quality score you can quote out of context. The same warning applies to MOS: a 4.1 from your raters is not comparable to a 4.1 from a paper's raters, because the rater pool, the anchors, and the instructions differ. Calibrate within your own study and compare within it.

The cheapest honest harness, then, is three relative numbers and one absolute gate: FAD-delta vs a fixed baseline (relative, music), CLAP-score (relative, text-audio alignment), MOS within a calibrated study (relative, naturalness), and WER (absolute — an ASR either transcribes your speech correctly or it does not, so a WER gate at 5% is a real, portable threshold). Lead with WER for speech because it is the one number that travels.

### Safety: watermark, consent, provenance

Generated audio — especially cloned voices — is a fraud and misinformation vector, and the safety layer is not optional. Three pieces, covered in full in [audio deepfakes, watermarking, and voice safety](/blog/machine-learning/audio-generation/audio-deepfakes-watermarking-and-voice-safety):

- **Watermark every output.** Embed an inaudible, robust watermark (AudioSeal, or SynthID-audio) into every generated waveform so it can later be detected as synthetic even after re-encoding or mild editing. This is the watermark stage in the serving topology above — it runs on every chunk before it leaves your server. The image-and-video analogue is [safety, watermarking, and provenance](/blog/machine-learning/image-generation/safety-watermarking-and-provenance); the principle is identical, the signal is audio.
- **Gate voice cloning on consent.** Never clone a voice without verifiable consent from its owner. In practice: a consent record tied to the reference audio, a verification step (a spoken consent phrase), and a refusal path for public-figure voices. This is a product and policy gate, not just a model gate.
- **Attach provenance.** Sign outputs with C2PA-style provenance metadata (who generated it, with which model, when) so downstream consumers can verify the chain. Metadata can be stripped, which is exactly why you also watermark — the two are complementary, not redundant.

The watermark adds negligible latency (a few milliseconds per chunk) and is the cheapest insurance you will ever buy against your product showing up in a deepfake news story.

A subtle but important property: watermarking and provenance are *complementary defenses against different attacks*, not redundant. Provenance metadata answers "what does the file claim about itself" and is trivially stripped by re-encoding or screen-recording; the watermark answers "what is actually embedded in the signal" and survives mild editing, compression, and format changes because it lives in the waveform, not the container. An attacker who strips the metadata still trips the watermark detector; an attacker who heavily distorts the audio to defeat the watermark has degraded it enough to be obviously synthetic. You ship both because each covers the other's gap. And the consent gate is the *upstream* defense — it prevents the harmful generation from happening at all, which is strictly better than detecting it after the fact. Defense in depth here is three layers: refuse without consent, watermark what you do generate, and attach provenance so the chain is auditable. None of the three is expensive, and the combination is what lets you ship a voice-cloning feature responsibly instead of shipping a fraud tool.

## 10. Three scenarios, three stacks, three cost lines

Now assemble everything into the three concrete products from the intro. Each is a complete stack — model, vocoder, serving, and a defensible cost per minute.

![Matrix mapping three target scenarios real-time agent hosted music tool and multilingual TTS API to a recommended model vocoder serve approach and cost per minute](/imgs/blogs/building-an-audio-generation-stack-8.png)

| Scenario | Model | Vocoder / decode | Serving | Cost/min (approx) | Why |
|---|---|---|---|---|---|
| Real-time voice agent | F5-TTS or Moshi (full-duplex) | Vocos, streaming | Self-host GPU, continuous batch, WebSocket | ~\$0.004 | Latency is binding; stream + small model + fast vocoder |
| Hosted music tool | Suno / Udio API | (vendor codec decode) | Managed API, async job + webhook | ~\$0.10–0.30 | Quality is binding; open music not yet at parity on full songs |
| Self-hosted multilingual TTS API | XTTS v2 (per-language LoRA) | Built-in HiFi-GAN | Self-host GPU pool, batched, REST | ~\$0.002 | Control + cost + data residency binding; full ownership |

Walk each one.

**The real-time voice agent.** Binding constraint: latency. Stack: F5-TTS (flow-matching, fast, good zero-shot voices) or Moshi if you need true full-duplex barge-in, with a Vocos vocoder, streaming in ~200 ms chunks, on a self-hosted 4090 pool with continuous batching behind a WebSocket gateway. Watermark every chunk with AudioSeal. TTFA under 250 ms, RTF ~0.13, cost ~\$0.004/min of GPU plus ops. You self-host because you need to control the tail latency and the per-minute cost at conversational volume, and because sending live user audio to a third party may be a privacy non-starter.

**The hosted music tool.** Binding constraint: quality. Stack: call Suno or Udio's API as an async job (a 3-minute song takes tens of seconds to render, so you use a job-plus-webhook pattern, not a blocking request), store the result, attach provenance, and pass the cost through to the user. You do *not* self-host a music farm on day one — the open music models (MusicGen, [Stable Audio](/blog/machine-learning/audio-generation/latent-diffusion-for-music-stable-audio)) are excellent and a great fit for instrumental beds and SFX, but on 2026-06-17 they do not match Suno/Udio on full songs with coherent vocals and lyrics. We made that comparison in [music generation, MusicLM and MusicGen](/blog/machine-learning/audio-generation/music-generation-musiclm-and-musicgen) and [Suno, Udio, and the commercial music frontier](/blog/machine-learning/audio-generation/suno-udio-and-the-commercial-music-frontier). Revisit self-hosting when the open models close the vocal gap — and watch them, because they are closing it fast.

**The self-hosted multilingual TTS API.** Binding constraint: control, cost, and data residency. Stack: XTTS v2 with a per-language LoRA adapter (hot-swapped per request), the built-in HiFi-GAN decoder, a self-hosted GPU pool with batching behind a REST API. You own the voices, the languages, the latency, and the ~\$0.002/min cost. This is the stack you build when an API's per-minute price times your volume exceeds the cost of a small GPU fleet plus an engineer — which, at API-scale volume, it always does. The customization ladder from Section 4 is exactly how you add the next language: curate, fine-tune a LoRA, gate on WER, ship.

Three products, three stacks, and every choice traces back to the one binding constraint. That is the whole playbook in one table.

#### Worked example: the multilingual TTS API cost line in detail

Make the third row's economics concrete. You serve five languages, each with a fine-tuned XTTS LoRA, and you project 200,000 minutes of synthesis per month at launch. On a self-hosted pair of 4090s (one active, one for redundancy and burst), the active GPU at RTF 0.15 with batching produces roughly 6–7 audio-minutes per wall-minute, so 200,000 minutes is about 30,000 GPU-minutes — around 500 GPU-hours, comfortably inside one GPU running two-thirds utilized over a month. At an amortized \$0.30/GPU-hour that is roughly \$150 of compute; add the redundant GPU, a small CPU control-plane box, object storage for generated audio, and monitoring, and you land near \$500–800/month all-in. The same 200,000 minutes on a hosted voice API at \$0.12/min would be \$24,000/month. The self-hosted stack is therefore about 30× cheaper at this volume, and the gap *widens* as you grow because the GPU fleet scales sub-linearly with batching while the API bills strictly per minute. The one cost the table hides is the human: you need an engineer who can keep the fleet healthy and add the next language's LoRA. At 200,000 minutes that engineer is trivially paid for; at 2,000 minutes they are not, and you should be calling the API. The cost line is real, but so is the break-even — respect both.

The thread through all three scenarios: you never started from a model. You started from "what is generated" and "what is the one constraint that cannot move," and the model, the vocoder, the serving topology, and the cost line all fell out of those two answers. A new request next quarter — a singing-voice feature, a video-to-audio foley tool, a dubbing pipeline — runs through the identical two questions and lands on its own row. The series gave you the components; this section is the assembly.

## When to reach for this (and when not to)

A decisive recommendation section, because the most valuable thing a playbook does is tell you what *not* to build.

- **Do not self-host below your break-even.** Under a few thousand minutes a month, a hosted API's zero-ops convenience beats a GPU farm you have to babysit. Build the farm when the math (Section 7) says so, not before.
- **Do not autoregress a 4-minute song in one pass.** AR models drift over long horizons — the beat wanders, the key slips. Use a model designed for length (Stable Audio's timing conditioning) or generate in structured segments with continuation, and never expect a single AR pass to hold a 4-minute groove.
- **Do not put a diffusion model in the vocoder slot of a real-time system.** A GAN vocoder hits the quality bar 100× faster. Diffusion belongs in the *generative* stage where its fidelity earns the steps, never at the last-mile vocoding stage where speed is everything.
- **Do not fine-tune when a 5-second reference will do.** Zero-shot prompting gets you 80% of the way for free. Fine-tune only when you have measured a real consistency or fidelity gap that the prompt cannot close — and you have the clean data to fix it.
- **Do not ship without the two gates.** No WER/FAD/MOS gate means you are shipping on vibes; no watermark and consent gate means you are shipping a fraud vector. Both are cheap. Skipping either is the expensive choice.
- **Do not optimize the wrong stage.** Profile first. If the vocoder is 0.1 s and the model is 6 s, quantizing the vocoder is wasted effort. The latency model in Section 6 tells you the model stage is almost always the term to attack.
- **Do not chase the frontier when "good enough" ships.** The newest model in a benchmark table is rarely the right production choice; a slightly older model with a permissive license, a stable API, and a year of community tooling will cost you far less time. Pick the boring, well-supported option unless the binding constraint genuinely demands the frontier.
- **Do not skip the warm-up when you measure.** The first forward pass compiles kernels and caches; counting it inflates your RTF and sends you optimizing a number that does not exist in steady state. Warm up, discard, then measure.

## Case studies: real numbers from shipped systems

A few grounding data points from the literature and shipped models, to calibrate the tables above. Treat exact figures as approximate and check the source papers.

- **EnCodec / DAC bitrate vs quality.** Défossez et al.'s EnCodec (2022) and the Descript Audio Codec (Kumar et al., 2023) both show the rate-distortion curve bending sharply: DAC reconstructs 44.1 kHz audio at roughly 8 kbps with quality competitive to much higher-bitrate classical codecs, and the quality climbs fast from 1.5 kbps to ~6 kbps then flattens. The practical takeaway for the pipeline: pick the lowest bitrate that clears your quality bar, because every extra codebook is more tokens for the model to emit.
- **HiFi-GAN real-time factor.** Kong et al.'s HiFi-GAN (2020) reported synthesis far faster than real time on a single GPU — on the order of hundreds of times real time for the smaller variants — which is exactly why it (and its successors BigVGAN and Vocos) became the default vocoder. The vocoder is essentially free; the generative model is where your time goes.
- **VALL-E zero-shot cloning.** Wang et al.'s VALL-E (2023) demonstrated voice cloning from a 3-second enrolled prompt by reframing TTS as codec-token language modeling, with speaker similarity and naturalness that surprised the field. It also exposed the failure modes — identity drift and occasional unintelligible runs — that the evaluation gate in Section 5 exists to catch.
- **MusicGen sizes and conditioning.** Copet et al.'s MusicGen (2023) ships in roughly 300M, 1.5B, and 3.3B sizes, with melody conditioning, on a single-stage codec language model with codebook interleaving — the practical open music workhorse, and the reason "MusicGen-small on a 4090" is a real, cheap option for instrumental generation.
- **Stable Audio length.** Evans et al.'s Stable Audio (2024) used latent diffusion with explicit timing conditioning to generate variable-length, high-fidelity audio up to minutes long — the answer to the "AR drifts on long form" problem, and the reason latent diffusion is the right family when length is the binding constraint.

- **Mimi and the low-frame-rate codec.** The Moshi system (Défossez et al., 2024) introduced Mimi, a codec running at a 12.5 Hz frame rate while retaining high reconstruction quality, specifically to make a real-time full-duplex speech model tractable — fewer tokens per second is directly fewer model steps per second of audio. This is the concrete realization of the frame-rate lever in Section 6: dropping from 50 Hz to 12.5 Hz is a 4× reduction in the AR model's token budget, which is most of what makes sub-second conversational latency achievable on a single GPU.

Every one of these is a data point you can drop into the decision tables. They are not marketing claims; they are the measured results that make the playbook's recommendations defensible. And they all reinforce the same structural truth: the codec and the vocoder set the floor on latency and the ceiling on fidelity, the generative model sets the controllability, and the deployment choice sets the cost — four levers, and you tune each one against the constraint that binds.

One synthesis worth stating before the takeaways: notice that every number in every table traces to a stage in the five-stage pipeline. Codec bitrate sets the token count; the model size and step count set the model latency; the vocoder choice sets the last-mile cost; the deployment approach sets the dollars. There is no magic and no marketing here — there is a pipeline, a small set of knobs, and a binding constraint that tells you which knob to turn. That is the entire discipline, and it is why the same playbook handles a voice agent, a music tool, and a multilingual API without changing shape.

## Key takeaways

1. **Pick the model last.** Decide on task (speech / music / SFX) and the one binding constraint (latency / quality / control / cost / license / on-device) first; the model falls out.
2. **Every audio system is the same five stages** — conditioning, codec encode, generative model, vocoder/decode, waveform — each with one dominant knob. Debug by stage; optimize the bottleneck stage.
3. **Self-host vs API is a volume calculation.** Below a few thousand minutes a month, call the API; above it, self-hosting saves an order of magnitude per minute and the savings compound.
4. **Customization is a ladder** — zero-shot prompt, voice clone, LoRA, full fine-tune — and clean data matters more than the method you pick.
5. **Latency is encode + model + vocode**, the model dominates, and the four levers (smaller model, quantization, fast vocoder, streaming) are how you hit an RTF or TTFA budget. Measure with a warm GPU.
6. **Compose families, do not pick one.** A flow or AR engine for fidelity and control, a GAN vocoder for last-mile speed; never a diffusion vocoder in a real-time system.
7. **Two gates are non-negotiable**: an evaluation gate (WER / FAD / a small MOS study, with named device and sample size) and a safety gate (watermark + consent + provenance).
8. **Stream to hide latency.** TTFA, not total latency, is what the user feels; streaming chunks plus RTF < 1 gives continuous audio after a short initial delay.
9. **Cost per minute is arithmetic, not magic** — GPU hourly rate over (60 × minutes-of-audio-per-wall-minute) — and self-hosting lands near a tenth of a cent of GPU time at good utilization.
10. **Be honest about the open-vs-closed gap.** On 2026-06-17 hosted APIs still lead on the hardest speech and on full songs with vocals; open models lead on cost, control, and data residency, and they are closing the quality gap fast.

If you remember one thing from this whole series, make it this: audio generation is not a model, it is a *pipeline with knobs and a binding constraint*. The papers gave us the components — codecs that tokenize sound, vocoders that synthesize it fast, AR and diffusion and flow engines that generate it, and the metrics and watermarks that gate it. Shipping is the act of assembling those components against the one constraint that cannot move, measuring honestly, and refusing to skip the gates. Do that, and the request on your desk next quarter — whatever modality, whatever language, whatever latency — runs through the same two questions and lands on its own stack. That is the playbook, and now it is yours.

## Further reading

- van den Oord et al., **WaveNet: A Generative Model for Raw Audio** (2016) — the autoregressive raw-waveform foundation.
- Zeghidour et al., **SoundStream: An End-to-End Neural Audio Codec** (2021) and Défossez et al., **High Fidelity Neural Audio Compression** (EnCodec, 2022) — the neural codec that made token-based audio generation practical.
- Kumar et al., **High-Fidelity Audio Compression with Improved RVQGAN** (Descript Audio Codec, 2023) — the modern high-fidelity codec and its rate-distortion curve.
- Borsos et al., **AudioLM** (2022) and Wang et al., **VALL-E: Neural Codec Language Models are Zero-Shot Text-to-Speech Synthesizers** (2023) — audio language modeling and 3-second zero-shot cloning.
- Kong et al., **HiFi-GAN** (2020) — the fast GAN vocoder that became the default last-mile stage.
- Copet et al., **Simple and Controllable Music Generation** (MusicGen, 2023) and Evans et al., **Stable Audio** (2024) — the open music workhorse and the length-conditioned latent-diffusion alternative.
- 🤗 `transformers` and `diffusers` audio docs, `audiocraft`, and Coqui `TTS` — the toolchains used throughout this series.
- Within this series: start at [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard), then the engines ([autoregressive](/blog/machine-learning/audio-generation/autoregressive-audio-models-wavenet-to-audiolm), [diffusion](/blog/machine-learning/audio-generation/diffusion-for-audio), [flow](/blog/machine-learning/audio-generation/flow-matching-and-consistency-for-audio), [GAN vocoders](/blog/machine-learning/audio-generation/gan-vocoders-hifi-gan-and-fast-synthesis)), [TTS](/blog/machine-learning/audio-generation/text-to-speech-from-tacotron-to-vits), [streaming speech](/blog/machine-learning/audio-generation/real-time-streaming-and-full-duplex-speech), evaluation, safety, and [the 2026 landscape](/blog/machine-learning/audio-generation/the-2026-audio-model-landscape). For the parallel media-stack playbook, see [building an image generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack).
