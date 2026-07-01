---
title: "Data for Text-to-Speech: Cleaning and Enhancing Audio the Model Will Imitate"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "TTS data is the opposite of ASR data: a synthesizer imitates whatever acoustics live in the corpus, so denoising, dereverb, resampling, loudness normalization, trimming, and a hard quality gate are the difference between a clean voice and one that hums."
tags: ["training-data", "text-to-speech", "tts", "audio-enhancement", "denoising", "dereverberation", "loudness-normalization", "dnsmos", "snr", "libritts", "ljspeech", "speech-synthesis"]
category: "machine-learning"
subcategory: "Training Data"
author: "Hiep Tran"
featured: true
readTime: 31
---

The first time I shipped a text-to-speech voice that sounded *wrong*, nothing was wrong with the model. The architecture was a stock VITS clone that had produced a perfectly clean voice on LJSpeech the week before. This time I had trained it on a few hundred hours of a public podcast, and the synthesized speech came out with a faint, steady hiss under every word, plus a hollow, bathroom-tiled quality on the vowels. I spent two days staring at attention alignments and learning-rate schedules before I finally opened the training clips and listened. The hiss was in the data. The reverb was in the data. The model had done its job flawlessly: it had learned to reproduce the podcast's room and the podcast's microphone, and now it reproduced them on words the podcast host never said.

That is the one idea this entire post is built around, and it is the idea that separates TTS data work from every other kind of data curation: **a speech synthesizer reproduces whatever acoustics live in its training data.** An automatic speech recognition (ASR) model only has to *read through* noise to recover the words. A TTS model has to *imitate* the signal, so every defect in the audio, such as background noise, room echo, MP3 compression, clipping, inconsistent loudness, is a feature the model dutifully learns and re-synthesizes. This is why ASR-grade data and TTS-grade data are not the same thing, and why "we already have a big speech corpus" is almost never the end of the conversation when someone wants to build a voice.

![TTS data is the opposite of ASR data: a recognizer reads through noise while a synthesizer imitates it](/imgs/blogs/data-for-text-to-speech-cleaning-and-enhancement-1.webp)

The diagram above is the mental model for the whole article. On the left, ASR: the goal is to transcribe words, so the model wants *variety*, and more noisy hours usually beat fewer clean ones. On the right, TTS: the goal is to imitate a voice, so the model wants *cleanliness and consistency*, and any artifact in the data becomes an artifact in the output. This post is a tour of the enhancement pipeline that turns raw, "found" audio into something a synthesizer can safely imitate, the quality gates that decide what to throw away, and the hard limits of enhancement, because you cannot fully fix bad audio, and the ways over-processing backfires teach the deepest lesson in the field. It sits downstream of the recognizer's data problem covered in [speech data for ASR](/blog/machine-learning/training-data/speech-data-for-asr) and upstream of the text-side work in [TTS transcription, normalization, and speakers](/blog/machine-learning/training-data/tts-transcription-normalization-and-speakers); together those three are the data spine of any synthesis project.

## Why TTS data is the opposite of ASR data

The senior rule of thumb: **for ASR you optimize the data distribution to match the world; for TTS you optimize the data distribution to be the target you want the model to become.** Those are opposite objectives, and they lead to opposite curation instincts.

An ASR model is a function from audio to text. During training it learns to be *invariant* to everything that is not the transcript: speaker identity, channel, noise, room, codec. Throwing a far-field clip recorded in a cafeteria at an ASR model is not a bug, it is regularization, because at inference time the model will hear exactly that. The classic result from the field, seen everywhere from Deep Speech to Whisper, is that ASR robustness comes from *scale and diversity* of realistic, messy audio.

A TTS model is a function from text (plus a speaker or style condition) to audio. During training it learns to be *equivariant*: it learns the exact mapping from phonemes and prosody to a waveform, and it will reproduce the acoustic distribution of the training set as faithfully as it can. If 30 percent of your clips have a 60 Hz mains hum, the model will learn that a fraction of utterances should have a 60 Hz hum, and it will sprinkle them into synthesis. There is no "invariance" to hide behind because there is nothing to be invariant *to*: the audio is the label.

| The assumption | The naive view | The reality for TTS |
| --- | --- | --- |
| "We have a big speech corpus, so we can build a voice." | ASR data and TTS data are interchangeable audio. | ASR-grade data is downsampled, noisy, and text-normalized. It fails almost every TTS quality bar until re-processed. |
| "Denoising can only help." | Push every clip through the strongest denoiser. | Over-denoising smears speech and introduces artifacts the model then learns to reproduce. |
| "Louder is clearer." | Peak-normalize everything to 0 dBFS. | Peak normalization ignores perceived loudness and can worsen clipping. Loudness (LUFS) normalization is what stabilizes the voice. |
| "More hours is always better." | Add every clip you can scrape. | One noisy speaker or one bad room, mixed into a single-speaker set, destabilizes the timbre more than the extra hours help. |
| "The model will average out the noise." | Trust large-N to wash out defects. | The model reproduces the *distribution* of defects, including the tails. It does not average them away. |

The practical consequence is that a TTS data pipeline spends most of its effort on two activities that an ASR pipeline barely touches: **enhancing** the audio (making it cleaner and more consistent than it arrived) and **gating** the audio (discarding what cannot be made clean enough). The rest of this article is those two activities in depth.

## The enhancement pipeline: what to fix, and in what order

Enhancement is not one operation, it is a fixed-order sequence, and **the order is load-bearing because each stage assumes the previous stage already ran.** Denoise before you resample, because a denoiser trained at 48 kHz behaves differently on a 16 kHz signal. Normalize loudness *after* denoising and dereverb, because those stages change the energy of the signal and would invalidate an earlier measurement. Trim silence near the end, once the noise floor is low enough that a silence detector can actually find the silence.

![The TTS audio-enhancement pipeline: denoise and dereverb before resampling, normalize loudness late, then trim and gate](/imgs/blogs/data-for-text-to-speech-cleaning-and-enhancement-2.webp)

The pipeline above is the canonical order. Below is the same thing as a process you can watch: the highlight moves stage by stage while the clip's level view goes from jagged, with a raised noise floor and a clipped peak, to smooth, normalized, and trimmed at the silent edges. It is the single most useful picture to keep in your head, because it makes the "the model imitates the end state" point visceral, whatever the clip looks like when the sweep finishes is what the voice will sound like.

<figure class="blog-anim">
<svg viewBox="0 0 760 300" role="img" aria-label="A noisy clip is cleaned stage by stage: as a highlight sweeps denoise, resample, loudness and trim, the jagged level view becomes smooth and its silent edges collapse" style="width:100%;height:auto;max-width:820px">
<style>
.a7-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.a7-sweep{fill:var(--accent,#6366f1);opacity:.18}
.a7-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.a7-cap{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.a7-base{stroke:var(--border,#d1d5db);stroke-width:1.5}
.a7-noisy{fill:var(--text-secondary,#6b7280)}
.a7-clean{fill:var(--accent,#6366f1)}
@keyframes a7-move{0%{transform:translateX(0)}100%{transform:translateX(540px)}}
@keyframes a7-out{0%,30%{opacity:1}55%,100%{opacity:0}}
@keyframes a7-in{0%,30%{opacity:0}55%,100%{opacity:1}}
.a7-sweep{animation:a7-move 8s steps(4,end) infinite}
.a7-ng{animation:a7-out 8s ease-in-out infinite}
.a7-cg{animation:a7-in 8s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.a7-sweep{animation:none;transform:translateX(405px)}.a7-ng{animation:none;opacity:0}.a7-cg{animation:none;opacity:1}}
</style>
<rect class="a7-box" x="20"  y="30" width="120" height="56" rx="8"/>
<rect class="a7-box" x="155" y="30" width="120" height="56" rx="8"/>
<rect class="a7-box" x="290" y="30" width="120" height="56" rx="8"/>
<rect class="a7-box" x="425" y="30" width="120" height="56" rx="8"/>
<rect class="a7-sweep" x="20" y="30" width="120" height="56" rx="8"/>
<text class="a7-lbl" x="80"  y="63">denoise</text>
<text class="a7-lbl" x="215" y="63">resample</text>
<text class="a7-lbl" x="350" y="63">loudness</text>
<text class="a7-lbl" x="485" y="63">trim</text>
<text class="a7-cap" x="382" y="112">the same clip, one stage at a time</text>
<line class="a7-base" x1="36" y1="260" x2="728" y2="260"/>
<g class="a7-ng">
<rect class="a7-noisy" x="40"  y="226" width="46" height="34"/>
<rect class="a7-noisy" x="98"  y="212" width="46" height="48"/>
<rect class="a7-noisy" x="156" y="174" width="46" height="86"/>
<rect class="a7-noisy" x="214" y="196" width="46" height="64"/>
<rect class="a7-noisy" x="272" y="140" width="46" height="120"/>
<rect class="a7-noisy" x="330" y="168" width="46" height="92"/>
<rect class="a7-noisy" x="388" y="186" width="46" height="74"/>
<rect class="a7-noisy" x="446" y="208" width="46" height="52"/>
<rect class="a7-noisy" x="504" y="164" width="46" height="96"/>
<rect class="a7-noisy" x="562" y="194" width="46" height="66"/>
<rect class="a7-noisy" x="620" y="216" width="46" height="44"/>
<rect class="a7-noisy" x="678" y="226" width="46" height="34"/>
</g>
<g class="a7-cg">
<rect class="a7-clean" x="40"  y="252" width="46" height="8"/>
<rect class="a7-clean" x="98"  y="186" width="46" height="74"/>
<rect class="a7-clean" x="156" y="172" width="46" height="88"/>
<rect class="a7-clean" x="214" y="182" width="46" height="78"/>
<rect class="a7-clean" x="272" y="168" width="46" height="92"/>
<rect class="a7-clean" x="330" y="176" width="46" height="84"/>
<rect class="a7-clean" x="388" y="180" width="46" height="80"/>
<rect class="a7-clean" x="446" y="172" width="46" height="88"/>
<rect class="a7-clean" x="504" y="186" width="46" height="74"/>
<rect class="a7-clean" x="562" y="176" width="46" height="84"/>
<rect class="a7-clean" x="620" y="190" width="46" height="70"/>
<rect class="a7-clean" x="678" y="252" width="46" height="8"/>
</g>
<text class="a7-cap" x="382" y="284">jagged, raised floor, one clipped peak   ->   smooth, normalized, edges trimmed</text>
</svg>
<figcaption>Each enhancement stage lights up in turn while the level view transforms from jagged (a raised noise floor and a clipped peak) to smooth, loudness-normalized, and trimmed at the silent edges.</figcaption>
</figure>

Here is the whole pipeline as runnable code, before we take each stage apart. It uses [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) for denoising (a fast, real-time deep noise suppressor), `torchaudio` for I/O and resampling, `pyloudnorm` for loudness, and `librosa` for silence trimming. Every import is a real package on PyPI.

```python
# pip install deepfilternet torchaudio pyloudnorm librosa numpy soundfile
import numpy as np
import torch
import torchaudio
import pyloudnorm as pyln
import librosa
from df.enhance import enhance, init_df

TARGET_SR = 24_000          # 24 kHz is the modern TTS default
TARGET_LUFS = -23.0         # EBU R128 integrated loudness target
DF_MODEL, DF_STATE, _ = init_df()   # DeepFilterNet runs at 48 kHz internally

def enhance_clip(path: str) -> tuple[np.ndarray, int]:
    # 1. Load. Force mono; keep native sample rate for now.
    wav, sr = torchaudio.load(path)          # wav: (channels, samples)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # downmix to mono

    # 2. Denoise. DeepFilterNet expects 48 kHz, so resample up, enhance, done.
    wav48 = torchaudio.functional.resample(wav, sr, 48_000)
    wav48 = enhance(DF_MODEL, DF_STATE, wav48)   # background-noise removal

    # 3. Resample to the target rate AFTER denoising.
    wav = torchaudio.functional.resample(wav48, 48_000, TARGET_SR)
    x = wav.squeeze(0).numpy().astype(np.float64)

    # 4. Loudness-normalize to a fixed LUFS target (perceived loudness).
    meter = pyln.Meter(TARGET_SR)            # BS.1770 meter
    loudness = meter.integrated_loudness(x)
    if np.isfinite(loudness):
        x = pyln.normalize.loudness(x, loudness, TARGET_LUFS)

    # 5. Guard against post-normalization clipping.
    peak = np.max(np.abs(x)) + 1e-9
    if peak > 0.99:
        x = x * (0.99 / peak)                # scale down to leave headroom

    # 6. Trim leading/trailing silence with a 30 dB-below-peak threshold.
    x, _ = librosa.effects.trim(x, top_db=30)
    return x.astype(np.float32), TARGET_SR
```

Notice what this function does *not* do: dereverberation (a separate, heavier stage covered below), internal-silence handling (utterance splitting, which belongs to the text-alignment stage), and speaker consistency checks (which are a corpus-level gate, not a per-clip transform). We will get to each. First, the stages that are in the code.

### Denoising: remove the background, keep the voice

**The senior rule: use the gentlest denoiser that clears your SNR floor, never the strongest.** Denoising is the highest-leverage and highest-risk stage. Background noise, HVAC hum, traffic, keyboard clicks, a second person breathing, all of it will be reproduced by the model if you leave it in, and none of it can be removed later. But denoisers are lossy: they estimate a mask over the spectrogram and suppress what they think is noise, and when they are wrong they eat consonants, dull the high frequencies that carry intelligibility, and leave a smeared, "underwater" residue that is *itself* a learnable artifact.

The practical menu, from lightest to heaviest:

| Denoiser | Mechanism | When to reach for it |
| --- | --- | --- |
| [RNNoise](https://github.com/xiph/rnnoise) | Small RNN, spectral gain | Real-time, mild stationary noise, minimal artifacts |
| Spectral gating (`noisereduce`) | Classic noise-profile subtraction | A known, stationary noise floor you can sample from silence |
| [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) | Deep filtering, 48 kHz | The default: strong, fast, preserves speech well |
| [Demucs / `denoiser`](https://github.com/facebookresearch/denoiser) | Waveform U-Net (Facebook) | Heavier noise, willing to trade some speed |
| [Resemble Enhance](https://github.com/resemble-ai/resemble-enhance) | Denoise + generative restoration | Damaged audio where you also want bandwidth extension |

The trap is the last two. Generative restoration models (Resemble Enhance, VoiceFixer, Google's Miipher) do not just suppress noise, they *resynthesize* the speech from a learned prior. That can produce gorgeous, studio-clean audio from a phone recording, but it subtly changes the timbre, and if you apply it inconsistently across a single speaker's clips, you introduce a new inconsistency worse than the noise you removed. Use them deliberately, uniformly, and only when a lighter tool cannot hit your floor.

### Dereverberation: get the room out of the voice

Reverb, the room's echo tail, is the artifact people forget until they hear it in synthesis, where it manifests as a hollow, distant, "recorded in a hallway" quality on sustained sounds. Denoisers do not remove it, because reverb is correlated with the speech, not additive noise. The standard tool is **WPE (weighted prediction error)**, a linear-prediction dereverberation method available in the `nara_wpe` package:

```python
# pip install nara_wpe
import numpy as np
from nara_wpe.wpe import wpe
from nara_wpe.utils import stft, istft

def dereverb(x: np.ndarray, sr: int) -> np.ndarray:
    # WPE operates in the STFT domain on (freq, frames) with a leading channel axis.
    Y = stft(x[None, :], size=512, shift=128)          # (chan, frames, freq)
    Y = Y.transpose(2, 0, 1)                            # (freq, chan, frames)
    Z = wpe(Y, taps=10, delay=3, iterations=3)          # linear dereverb
    z = istft(Z.transpose(1, 2, 0), size=512, shift=128)
    return z[0]
```

Reverb removal is genuinely hard and imperfect, which is why studio single-speaker corpora, recorded in treated rooms, are so valued: there is nothing to remove. For found data, WPE takes the edge off, but a heavily reverberant clip is a clip you should consider *dropping* rather than fixing, and that decision belongs to the quality gate.

### Bandwidth and sample rate: 24 kHz is the modern floor

**The senior rule: pick your target sample rate first, and never let a low-bandwidth clip masquerade as a high-bandwidth one.** Speech synthesis quality is bounded by bandwidth. The old Tacotron/WaveNet era standardized on 22.05 kHz (LJSpeech's rate), which captures audio up to about 11 kHz and sounds slightly muffled on sibilants. Modern systems target 24 kHz (up to 12 kHz audio) as the baseline and 48 kHz for premium voices. If you want a survey of how the acoustic models and vocoders that consume this data evolved, see [text-to-speech from Tacotron to VITS](/blog/machine-learning/audio-generation/text-to-speech-from-tacotron-to-vits).

The subtle failure mode is *fake bandwidth*. A clip that was recorded at 8 kHz (telephone), upsampled to 24 kHz, and handed to you looks like a 24 kHz file but has no energy above 4 kHz. Mix a handful of these into a 24 kHz corpus and the model learns that some utterances should be dull, producing intermittent muffling. Detect it by measuring the actual spectral rolloff, not the file's declared sample rate:

```python
import numpy as np
import librosa

def effective_bandwidth(x: np.ndarray, sr: int) -> float:
    """Return the frequency (Hz) below which 99% of energy lives."""
    S = np.abs(librosa.stft(x, n_fft=2048)) ** 2
    power = S.mean(axis=1)                       # average power per freq bin
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    cumulative = np.cumsum(power) / power.sum()
    idx = np.searchsorted(cumulative, 0.99)
    return float(freqs[min(idx, len(freqs) - 1)])

# A true 24 kHz clip returns ~10-12 kHz; an upsampled telephone clip returns ~3.5 kHz.
```

Any clip whose effective bandwidth is far below what its sample rate implies gets dropped or routed to a bandwidth-extension model, never silently mixed in.

### Loudness normalization: stabilize the level, not the peak

**The senior rule: normalize to a loudness target (LUFS), not a peak target (dBFS).** New engineers reach for peak normalization, scaling every clip so its loudest sample hits some ceiling. It is the wrong tool. Two clips can share a peak of 0 dBFS while one is a quiet whisper with one loud cough and the other is a consistent, full voice. The model, trained on peak-normalized audio, learns a wildly inconsistent relationship between text and level.

Loudness normalization uses the ITU-R BS.1770 / EBU R128 integrated-loudness measurement, which models human perception, and scales each clip to a fixed LUFS target. Broadcast standard is -23 LUFS; many TTS pipelines use -23 to -20 LUFS for a slightly hotter voice. The effect is that every training clip has the *same perceived loudness*, so the model learns a stable text-to-level mapping and the synthesized voice does not swell and shrink between sentences.

```python
import numpy as np
import pyloudnorm as pyln

def normalize_loudness(x: np.ndarray, sr: int, target_lufs: float = -23.0):
    meter = pyln.Meter(sr)                          # BS.1770-4 meter
    loudness = meter.integrated_loudness(x)         # LUFS (negative number)
    if not np.isfinite(loudness):                   # silent clip
        return x, loudness
    normalized = pyln.normalize.loudness(x, loudness, target_lufs)
    # Loudness normalization can push peaks past 1.0 -> clip. Re-check downstream.
    return normalized, loudness
```

The one gotcha, visible in the code, is that raising a clip's loudness can push its peaks past full scale, which introduces clipping, the exact defect we are trying to avoid. Always follow loudness normalization with a peak guard, and if the required gain would clip badly, that clip is too quiet to use safely.

### Silence trimming: cut the dead air, keep the breaths

Leading and trailing silence teaches the model to emit dead air at utterance boundaries, and inconsistent trailing silence confuses duration modeling. Trim it. The simple, robust approach is an energy threshold relative to the clip's peak, which is exactly what `librosa.effects.trim` does with `top_db`. A `top_db` of 30 means "cut everything more than 30 dB below the loudest point," which is aggressive enough to remove room tone but gentle enough to keep the natural onset of speech.

Internal silence, long pauses *within* a clip, is a different problem. You usually do not want to delete internal pauses (they are prosody), but you do want to *split* over-long clips at long pauses into separate utterances, which is a job for the alignment and segmentation stage covered in the [transcription and speaker post](/blog/machine-learning/training-data/tts-transcription-normalization-and-speakers). For enhancement, trim the edges and leave the middle alone.

### Clipping detection: the artifact you cannot remove

Clipping, where the waveform hit the recording ceiling and got flattened, is the one defect on this list that is genuinely *unrecoverable*: the information is gone. A declipper can interpolate, but it is guessing. So the right move is detection and *removal from the training set*, not repair. A clip is clipped if a meaningful fraction of its samples sit at or very near full scale:

```python
import numpy as np

def clipping_ratio(x: np.ndarray, thresh: float = 0.99) -> float:
    """Fraction of samples at or above |thresh| of full scale."""
    return float(np.mean(np.abs(x) >= thresh))

# Rule of thumb: drop clips whose clipping_ratio exceeds ~0.001 (0.1% of samples).
```

Do this detection on the *original* clip, before any normalization, because loudness normalization changes absolute amplitudes and can hide or fake clipping.

## A worked scenario: enhancing 5,000 clips of found podcast audio

Numbers make this concrete. Suppose you have scraped and segmented 5,000 single-speaker clips from a public-domain interview podcast, totaling about 9 hours, and you want a clean single-speaker voice. Here is what a real enhancement-and-gate run looks like.

**Before enhancement**, measured across the 5,000 clips:

- Median segmental SNR: **14 dB** (spread 6 to 24 dB). Podcast room tone plus a low broadband hiss.
- Integrated loudness: ranges from **-31 LUFS to -12 LUFS**, a 19 dB spread. The host leans in and out of the mic.
- Median DNSMOS P.808 (predicted MOS): **2.9**. Below the 3.0 "acceptable" line.
- Clipping: **4 percent** of clips have a clipping ratio above 0.001, from the host's loud laughs.
- Effective bandwidth: mostly 11 kHz (good), but **6 percent** roll off below 5 kHz (a few phone-in segments).

Run the pipeline: DeepFilterNet denoise, resample to 24 kHz, loudness-normalize to -23 LUFS, peak-guard, trim. **After enhancement**:

- Median segmental SNR: **26 dB** (spread 16 to 34 dB). The denoiser bought about 12 dB.
- Integrated loudness: **-23 LUFS plus or minus 0.4 dB**. Now uniform, the biggest single win for voice stability.
- Median DNSMOS: **3.4**. The denoise lifted the whole distribution.

Enhancement improved the corpus but did not save every clip. Now the gate. Apply two floors, an SNR floor of 20 dB (post-enhancement) and a DNSMOS floor of 3.0:

- Clips passing **DNSMOS >= 3.0**: about 57 percent (the picture below).
- Clips also passing **SNR >= 20 dB**: about 48 percent.
- Also dropping the 4 percent still clipping (detected on originals) and the 6 percent low-bandwidth phone-ins: final keep-rate about **44 percent**, or roughly **2,200 clips / 4 hours**.

![Auto-filtering found audio by DNSMOS: a 3.0 floor drops the noisy left tail and keeps about 57 percent](/imgs/blogs/data-for-text-to-speech-cleaning-and-enhancement-3.webp)

The histogram is the DNSMOS view on a 2,205-clip validation slice: the red bars left of the floor are dropped, the green bars right of it are kept, and the keep-rate is just the fraction of area on the green side. The number that matters is not "we had 9 hours," it is "we had 4 hours good enough to imitate." A voice trained on those 4 clean hours will sound dramatically better than one trained on all 9, because the 5 discarded hours would have taught the model to hiss. **When you gate for TTS, you are not throwing data away, you are refusing to teach the model your defects.**

> The most important number in a TTS data pipeline is the keep-rate, and the second most important is your honesty about why the discarded clips failed.

## Quality gates: SNR floors and DNSMOS auto-filtering

You cannot listen to 5,000 clips, let alone 500,000. Gating has to be automatic, which means you need machine-computable proxies for "clean enough." The two workhorses are a signal-level metric (SNR) and a learned quality predictor (DNSMOS).

**SNR (signal-to-noise ratio)** is the ratio of speech power to noise power. The honest way to estimate it without a clean reference is to use a voice-activity detector to separate speech frames from non-speech frames, then compare their energies:

$$\text{SNR}_\text{dB} = 10 \log_{10}\!\left(\frac{P_\text{speech} - P_\text{noise}}{P_\text{noise}}\right)$$

where $P_\text{speech}$ is the mean power over frames the VAD marked as speech and $P_\text{noise}$ is the mean power over non-speech frames. Here is a runnable estimate using Silero VAD (loaded from `torch.hub`):

```python
import numpy as np
import torch

# Silero VAD: a small, fast voice-activity detector.
model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
(get_speech_timestamps, _, _, _, _) = utils

def estimate_snr(x: np.ndarray, sr: int = 16_000) -> float:
    t = torch.from_numpy(x).float()
    segments = get_speech_timestamps(t, model, sampling_rate=sr)
    if not segments:
        return -np.inf                              # no speech detected
    mask = np.zeros(len(x), dtype=bool)
    for s in segments:
        mask[s["start"]:s["end"]] = True
    speech_power = np.mean(x[mask] ** 2) + 1e-12
    noise_power = np.mean(x[~mask] ** 2) + 1e-12    # energy in the gaps
    return 10.0 * np.log10(max(speech_power - noise_power, 1e-12) / noise_power)
```

SNR is cheap and interpretable but blind to distortion: a clip that a denoiser turned into artifact soup can still post a high SNR because the "noise" between words is now silent. That is where a *perceptual* predictor earns its keep.

**DNSMOS**, from Microsoft's Deep Noise Suppression Challenge, is a neural network that predicts the mean opinion score (MOS) a human panel would give a clip, on a 1-to-5 scale, from the audio alone, no reference needed. It was trained on tens of thousands of human ratings and correlates well with listener judgments of overall quality, including the smeared, artifact-y output that fools SNR. The `speechmos` package wraps it:

```python
# pip install speechmos
from speechmos import dnsmos

def dnsmos_score(x, sr=16_000) -> dict:
    # Returns overall MOS (ovrl_mos) plus signal (sig) and background (bak) sub-scores.
    return dnsmos.run(x, sr)   # e.g. {"ovrl_mos": 3.42, "sig_mos": 3.6, "bak_mos": 4.0}
```

The gate combines them. A clip must clear an SNR floor *and* a DNSMOS floor, because they catch different failures, SNR catches additive noise, DNSMOS catches distortion and artifacts:

```python
def passes_quality_gate(x, sr,
                        snr_floor: float = 20.0,
                        dnsmos_floor: float = 3.0,
                        clip_ratio_ceiling: float = 0.001) -> bool:
    if clipping_ratio(x) > clip_ratio_ceiling:      # unrecoverable, drop
        return False
    if estimate_snr(x, sr) < snr_floor:             # too much residual noise
        return False
    if dnsmos_score(x, sr)["ovrl_mos"] < dnsmos_floor:  # perceptually poor / artifacty
        return False
    return True
```

Where to set the floors is a judgment call, and it is the same clean-proxy-ablation loop that governs every other curation decision, covered in [measuring data quality](/blog/machine-learning/training-data/measuring-data-quality): train a small model at floor 3.0, another at 3.3, listen to both, and let the *output* voice decide. As a starting point: DNSMOS 3.0 and SNR 20 dB for a "good enough" voice, DNSMOS 3.3 and SNR 25 dB for a premium single-speaker voice, and looser floors (DNSMOS 2.8) only when you have a downstream model robust to some noise and you need the hours.

## The limits of enhancement: you cannot fully fix bad audio

Here is the failure mode that took me two days to find, drawn as a causal chain, because it is the single most important thing to internalize about TTS data.

![The model reproduces whatever is in the data: over-denoising leaves an artifact that propagates into synthesis on new text](/imgs/blogs/data-for-text-to-speech-cleaning-and-enhancement-5.webp)

Read the top lane left to right. A found clip with hiss and faint music goes through an *over-aggressive* denoiser, which smears the speech to kill the noise. That smearing is now baked into the "cleaned" training data. The model trains on it and learns a voice-plus-artifact. At inference, on words that were never in the training set, the model synthesizes the same watery, smeared artifact, because it learned that "clean speech looks like this." The bottom lane is the same source through a *tuned* denoiser with a modest noise floor: clean data, natural voice model, natural output.

The lesson is uncomfortable: **enhancement can make audio worse for a synthesizer even while making it sound better to a casual listener.** A generative restorer that produces a crisp clip you would happily listen to may have altered the timbre just enough that, applied across a corpus, the model learns an averaged, slightly synthetic-sounding voice. The floor on quality is set by the *worst irreversible defect* in your kept data, not by how hard you can process. There are three hard limits worth stating plainly:

- **Clipping is information loss.** No declipper recovers what was flattened; it interpolates. Detect and drop.
- **Reverb is entangled with speech.** WPE helps, but a badly reverberant clip cannot be made to sound close-mic'd. Drop it.
- **Over-denoising is a new artifact.** The residue of aggressive suppression is as learnable as the noise it removed. The right denoiser is the *gentlest* one that clears your floor.

This is why the entire field leans on studio single-speaker data when it can: the cheapest enhancement is the enhancement you never had to do, because the room was treated and the mic never moved.

## Case studies: studio, found, and the corpora that taught the field

Before the individual corpora, the framing that connects them: every TTS dataset lives somewhere on the studio-to-found spectrum, and where it sits determines how much enhancement it needs. The matrix below is the tradeoff in one picture.

![Studio vs found data: studio buys quality and consistency but not scale; found buys scale but fails every quality bar by default](/imgs/blogs/data-for-text-to-speech-cleaning-and-enhancement-4.webp)

Studio single-speaker audio wins on everything a synthesizer cares about, recording control, high SNR, a treated room, one consistent mic, except the one thing that increasingly matters at the frontier: hours at scale. Found audio (audiobooks, podcasts) is the mirror image: effectively unlimited hours, but it fails every quality bar until the enhancement-and-gate pipeline drags it up. There is no free lunch, only a choice about which end of the spectrum you start from and how much cleaning you are willing to do. The corpora below are the canonical points on that spectrum.

### 1. LJSpeech: the single-speaker studio baseline

[LJSpeech](https://keithito.com/LJ-Speech-Dataset/) is the "hello world" of TTS for a reason. It is 13,100 short clips, about 24 hours, of a single female speaker (Linda Johnson) reading seven non-fiction books, recorded on a MacBook Pro's built-in mic in a home setting, and released into the public domain. At 22.05 kHz it is not high-fidelity by modern standards, but it is *consistent*: one speaker, one mic, one room, one reading style, clean transcripts. That consistency is precisely why a modest model trained on LJSpeech produces a clean, stable voice with almost no data engineering, and why nearly every TTS paper reports an LJSpeech number. It is the control variable of the field. The lesson: **consistency beats fidelity for a first voice.** Twenty-four hours of one speaker in one room will get you further than a hundred hours of mixed sources.

The catch is that LJSpeech's very cleanliness makes it a poor stress test. A pipeline that works on LJSpeech has proven almost nothing about its robustness to found data, which is where the real work lives, and where the next case study comes in.

### 2. LibriTTS: ASR-grade data is not TTS-grade data

This is the case study that proves the thesis of the whole post. [LibriSpeech](https://www.openslr.org/12) is the canonical ASR corpus: about 960 hours of read English audiobooks derived from LibriVox, used to train and benchmark countless recognizers. When Google's Heiga Zen and colleagues wanted a large multi-speaker *TTS* corpus in 2019, the obvious move was "just use LibriSpeech." It did not work, and *why* it did not work is the lesson.

![LibriSpeech to LibriTTS: the same LibriVox source re-derived as wideband, punctuation-restored, quality-filtered TTS data](/imgs/blogs/data-for-text-to-speech-cleaning-and-enhancement-6.webp)

They built **LibriTTS** by going back to the original LibriVox source (not the processed LibriSpeech release) and re-deriving TTS-grade audio, as the figure shows. LibriSpeech had been downsampled to 16 kHz (too low for good synthesis), had its text normalized with the punctuation and casing stripped (a synthesizer needs punctuation to model prosody), and included utterances with significant background noise and long internal silences (fine for a robust recognizer, poison for a voice). LibriTTS restored 24 kHz audio, restored the original punctuation and casing from the source texts, split utterances at sentence boundaries using long-pause detection, and filtered out clips with poor signal-to-noise ratio. The result was roughly 585 hours across 2,456 speakers, meaningfully *less* than the 960 hours it started from, because a large fraction of ASR-grade audio simply did not clear the TTS bar.

The 2023 follow-up, **LibriTTS-R**, went further: the team applied Google's Miipher speech-restoration model to the *entire* LibriTTS corpus, cleaning noise and improving fidelity while keeping the transcripts and speaker labels. It is the corpus-scale version of the enhancement pipeline in this post, and it measurably improved the naturalness of models trained on it, direct evidence that enhancing the data, not the model, was the bottleneck.

Two lessons compound here. First: **you cannot reuse ASR data for TTS without re-processing it,** and the re-processing throws away a lot. Second: **enhancement is a corpus-level lever with real payoff,** LibriTTS-R beat LibriTTS on the same architecture purely by cleaning the audio.

### 3. VCTK: multi-speaker studio, and the mic-consistency trap

[VCTK](https://datashare.ed.ac.uk/handle/10283/3443) is about 44 hours across 110 English speakers with various accents, recorded in a hemi-anechoic chamber at high sample rate, roughly the multi-speaker analogue of LJSpeech. It is a staple for multi-speaker and voice-cloning work. But VCTK carries a famous, subtle gotcha: across its recording sessions the level and channel are *not* perfectly uniform, and some speakers were captured with different effective energy or a leading silence quirk. Teams that trained speaker-conditioned models on raw VCTK sometimes found the model entangling "speaker identity" with "recording level," so switching speakers also switched loudness. The fix is exactly the loudness-normalization stage above, applied per clip to a common LUFS target, plus consistent trimming, before training. The lesson: **even studio corpora need loudness and silence normalization,** because "studio" guarantees a quiet room, not a uniform pipeline.

### 4. Mined audiobooks and podcasts: found data at scale

The frontier of TTS data is *found* data at massive scale: audiobooks (LibriVox and commercial), podcasts, YouTube, parliamentary recordings. This is how you get to the tens of thousands of hours that modern large TTS and voice-cloning models want, far beyond what any studio can produce. Corpora like the People's Speech, GigaSpeech, and the audiobook-derived sets are ASR-first, so the entire enhancement-and-gate pipeline in this post is the price of admission: denoise, dereverb, resample, loudness-normalize, trim, and gate hard on SNR and DNSMOS. The keep-rates are brutal, single-digit to low-double-digit percentages are normal, but the *absolute* number of clean hours is still large. The lesson: **found data is a keep-rate game, not a collection game.** The skill is discarding well.

## Troubleshooting: symptom to cause to fix

This is the section I wish I had had two days earlier. Each entry is a real failure I or teammates have hit, in the shape symptom, cause, detection, fix.

### The synthesized voice has a steady hiss on every word

- **Symptom:** every utterance the model produces, on any text, carries a faint broadband hiss or hum.
- **Cause:** the training clips share a noise floor (room tone, mains hum, mic self-noise) that the model learned as part of the voice.
- **Detection:** estimate SNR on the training clips; a median below ~20 dB predicts this. Listen to a silent gap in a training clip and you will hear the same hiss.
- **Fix:** denoise before training (DeepFilterNet), and raise the SNR floor in the gate so the noisiest clips are dropped rather than cleaned. Do not lower the floor to keep hours, an SNR floor set too low lets noisy clips poison the whole voice, because the model reproduces the *distribution* of noise, tails included.

### The voice sounds hollow, distant, or "in a hallway"

- **Symptom:** vowels and sustained sounds have a boxy, reverberant tail, as if recorded far from the mic.
- **Cause:** room reverberation in the training data, which denoisers do not remove.
- **Detection:** listen to onsets and decays; measure the reverberation tail (or just trust your ears, reverb is obvious once you know to listen for it).
- **Fix:** apply WPE dereverberation, and drop the worst-reverberant clips (they cannot be fixed). Prefer close-mic'd or treated-room sources. If the whole corpus is reverberant, no amount of processing will produce a dry voice.

### The voice has a watery, smeared, "underwater" quality

- **Symptom:** consonants are mushy and the voice has a phasey, artifact-y texture, worse than the original noise.
- **Cause:** over-aggressive denoising or generative restoration smeared the speech, and the model learned the artifact. This is the failure in the causal-chain figure above.
- **Detection:** DNSMOS catches it where SNR does not, an over-denoised clip can post high SNR but low DNSMOS. Compare DNSMOS before and after your denoise step; if it *drops*, you are over-processing.
- **Fix:** use a gentler denoiser or dial back its strength; the right denoiser is the weakest one that clears your floor. Gate on DNSMOS, not just SNR, so smeared clips are rejected.

### The voice swells and shrinks in loudness between sentences

- **Symptom:** synthesized speech is inconsistent in level, some sentences loud, some quiet, with no relation to content.
- **Cause:** the training clips have inconsistent loudness (a common found-data and even studio-data problem, see VCTK), so the model learned an unstable text-to-level mapping.
- **Detection:** measure integrated loudness (LUFS) across the corpus; a spread wider than ~3 dB predicts this.
- **Fix:** loudness-normalize every clip to a single LUFS target (-23 LUFS) with `pyloudnorm`, then peak-guard. This is the single highest-value stage for perceived voice stability.

### The timbre drifts or the voice sounds like "two people"

- **Symptom:** in a single-speaker voice, the timbre subtly shifts between utterances, or sounds like an average of slightly different voices.
- **Cause:** the "single speaker" corpus actually mixes recording conditions (two mics, two sessions months apart, a phone segment) or, worse, two similar speakers. The model learns the blend. Inconsistent mic and channel across one speaker produces an unstable timbre.
- **Detection:** cluster speaker embeddings (for example from a `speechbrain` or `pyannote` model) over the corpus; a supposedly single-speaker set should form one tight cluster, not two.
- **Fix:** enforce channel consistency, pick one recording condition and drop the rest, or if you must keep multiple, model them as distinct speakers. Normalize loudness and bandwidth so channel does not leak into timbre.

### Faint background music bleeds into the synthesized speech

- **Symptom:** the model occasionally emits musical tones or a rhythmic bed under the voice.
- **Cause:** the source (podcast, YouTube, audiobook with a musical intro) had background music that a speech denoiser does not fully remove, because music overlaps speech spectrally.
- **Detection:** run a music/speech detector (or a source-separation model like Demucs and check the "other" stem's energy); flag clips with significant non-speech harmonic content.
- **Fix:** use a music-aware source separator (Demucs) to isolate vocals, or, more reliably, *drop* segments with music bleed, they are rarely worth the risk. Music that overlaps speech is the hardest artifact to remove cleanly, and a partially-removed music bed is itself a learnable artifact.

### The corpus is clean but the model still sounds slightly artificial

- **Symptom:** every clip passes the gate, SNR and DNSMOS are high, yet the voice has a faint synthetic sheen.
- **Cause:** uniform over-processing. If every clip went through a generative restorer, the model learned the restorer's prior, not the speaker's true voice.
- **Detection:** A/B a model trained on lightly-processed data against one on heavily-processed data; if the lightly-processed one sounds more natural despite lower DNSMOS, you were over-enhancing.
- **Fix:** back off the enhancement. The goal is the gentlest processing that clears your floor, not the highest DNSMOS number. Enhancement is a means to a natural voice, not an end in itself.

## When to reach for heavy enhancement, and when not to

Reach for the full denoise-dereverb-restore pipeline when:

- You are working with **found data** (podcasts, audiobooks, scraped audio) that is your only path to the hours you need.
- The defects are **additive and stationary** (hiss, hum, steady room tone), which denoisers handle well without smearing.
- You have a **quality gate** downstream to catch clips the enhancement could not save, so enhancement plus gating work together.
- You can apply the enhancement **uniformly** across a speaker, so you do not introduce new inconsistency.

Skip or minimize enhancement when:

- You have **clean studio single-speaker data** (LJSpeech-style). The cheapest enhancement is the one you never run; do not over-process a corpus that is already clean.
- The defect is **irreversible** (clipping, severe reverb, heavy music bleed). Drop those clips instead of pretending processing fixed them.
- You would have to apply a **generative restorer inconsistently**, which trades a known defect for a subtler, worse one.
- You are building an **ASR** system, not TTS, in which case the messy audio is a feature and this entire pipeline is counterproductive (see [speech data for ASR](/blog/machine-learning/training-data/speech-data-for-asr)).

The through-line: TTS data work is the discipline of deciding what the model is allowed to imitate. Enhancement expands what you can safely keep; the quality gate enforces the line; and the humility to *drop* audio you cannot clean is what separates a voice that sounds like a person from one that sounds like the room it was recorded in.

## Further reading

- [Speech data for ASR](/blog/machine-learning/training-data/speech-data-for-asr) — the opposite discipline: why recognizers want the messy audio a synthesizer must reject.
- [TTS transcription, normalization, and speakers](/blog/machine-learning/training-data/tts-transcription-normalization-and-speakers) — the text-side companion to this audio-side post: punctuation, normalization, alignment, and speaker labeling.
- [Text-to-speech from Tacotron to VITS](/blog/machine-learning/audio-generation/text-to-speech-from-tacotron-to-vits) — the acoustic models and vocoders that consume this data.
- [Kokoro-82M](/blog/machine-learning/signal-processing/kokoro-82m) — a small, high-quality open TTS model, and a case study in what clean data buys you.
- [Measuring data quality](/blog/machine-learning/training-data/measuring-data-quality) — the clean-proxy-ablation loop for setting gate floors honestly.
- LibriTTS (Zen et al., 2019) and LibriTTS-R (2023) — the corpora that prove ASR data is not TTS data.
