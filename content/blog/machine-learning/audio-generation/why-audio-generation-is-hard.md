---
title: "Why Audio Generation Is Hard: Sample Rates, Phase, and the Ear"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "A first-principles map of why turning text or noise into believable sound is brutally hard, and the four model families and audio stack that make it work."
tags:
  [
    "audio-generation",
    "audio-synthesis",
    "text-to-speech",
    "music-generation",
    "neural-audio-codec",
    "diffusion-models",
    "generative-ai",
    "deep-learning",
    "signal-processing",
  ]
category: "machine-learning"
subcategory: "Audio Generation"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/why-audio-generation-is-hard-1.png"
---

The first time I shipped a text-to-speech model, it passed every offline metric and still sounded wrong. The words were correct. The speaker identity was correct. The mel-spectrogram, plotted side by side with ground truth, was nearly indistinguishable. And yet a faint, gritty buzz rode underneath every vowel, like the voice was being played through a cheap intercom. Nothing in my loss curves predicted it. The problem was not the words or the timing or the speaker. It was phase: the vocoder was reconstructing the right frequencies at slightly the wrong moments, and the human ear, which forgives almost nothing, heard it instantly.

That experience is the whole reason this series exists, and it is the thing that makes audio generation genuinely different from generating images or video. An image model that gets a pixel slightly wrong produces an image that is slightly wrong, and you usually cannot tell. An audio model that gets a sample slightly wrong produces an artifact your ear flags as unnatural in milliseconds. We have evolved a perceptual system that is exquisitely tuned to small temporal and spectral errors, because for most of human history those errors carried information about danger, speech, and the world. You cannot fool it casually.

This post is the foundation of the series **Audio Generation, From First Principles to the Frontier**, the third pillar of generative media after the completed [image-generation](/blog/machine-learning/image-generation/why-generating-images-is-hard) and video-generation series. My goal here is not to teach you any single model. It is to frame the whole problem: why audio is a uniquely punishing thing to synthesize, what the four generative families are and what each is good and bad at, what the recurring **audio stack** is that ties the series together, and where the 2024 to 2026 frontier sits. By the end you will understand the shape of the entire field, you will have generated a few seconds of sound yourself with about a dozen lines of Python, and you will know which of the series' six tracks to read next for any question you actually have.

![The five-stage audio generation stack from waveform to latent to generative core to vocoder back to waveform](/imgs/blogs/why-audio-generation-is-hard-1.png)

The frame above is the spine of everything that follows. Audio almost never gets generated as raw samples directly anymore. Instead a waveform is compressed into a compact latent (neural-codec tokens or a mel-spectrogram), a generative core (an autoregressive language model, a diffusion model, or a flow-matching model) produces that latent, and a decoder or vocoder turns it back into a waveform. The tension that runs through every design decision is **fidelity versus controllability versus speed versus length**. Push one and you usually pay somewhere else. Hold that picture in your head; we will return to it in every section.

## What a sound actually is, and why that is already a problem

Start with the physical object. Sound is a pressure wave traveling through air. A microphone measures that pressure many thousands of times per second and writes down a single number each time: how far the membrane was displaced. The result is a one-dimensional signal, a long list of numbers indexed by time. That is it. There is no inherent two-dimensional structure, no color channels, no spatial layout. Just amplitude over time.

The number of measurements per second is the **sample rate**, measured in hertz (Hz). This is the first place audio bites you. Telephone-quality speech is often 8 kHz (8,000 samples per second). Modern speech systems work at 16 kHz or 24 kHz. Music and high-quality audio are 44.1 kHz (the CD standard) or 48 kHz (the standard for video and professional audio). At 48 kHz, one second of mono audio is 48,000 numbers. A three-minute song in stereo is roughly 48,000 × 180 × 2, about 17 million samples. A model that generates audio one sample at a time, the way the earliest neural audio models did, has to produce all 17 million of them, in order, each one consistent with everything before it.

Compare that to images. A 1024 × 1024 RGB image is about 3 million values, and you can generate them in parallel because there is no strict ordering. A single second of CD-quality stereo audio (88,200 samples) is already a third of that, and the samples are deeply ordered in time. This is the **sequence-length problem**, and it is the single biggest structural difference between audio and the other media we generate. Audio is short in wall-clock time but enormous in token count, and the tokens are sequential.

Why is the sample rate so high in the first place? Because of a hard mathematical fact called the **Nyquist sampling theorem**, which we will derive properly in [the mathematics of audio signals](/blog/machine-learning/audio-generation/the-mathematics-of-audio-signals). In one sentence: to represent a frequency of $f$ hertz without aliasing it into a different, wrong frequency, you must sample at a rate strictly greater than $2f$. Human hearing tops out around 20 kHz, so to capture everything we can hear you need a sample rate above 40 kHz, which is exactly why CD audio chose 44.1 kHz. You cannot cheat this. Drop the sample rate and you do not just lose quality, you fold high frequencies down into audible garbage. The high rate is not a luxury; it is the price of admission for full-band audio.

There is a second axis to the raw signal beyond the sample rate: **bit depth**, how many bits encode each sample's amplitude. CD audio uses 16-bit integers (65,536 possible amplitude levels); professional audio uses 24-bit; and almost every model works internally with 32-bit floating-point samples normalized to the range $[-1, 1]$. Bit depth sets the noise floor: too few bits and you hear quantization noise, a gritty hiss on quiet passages. For generation this matters less than sample rate (models work in float and let the codec handle the final quantization), but it is the other half of "how much data is a waveform," and you will see it whenever you load a file and find samples that are already floats between minus one and one.

So the raw object a model must ultimately produce is a long float array, tens of thousands of values per second, each consistent with its neighbors down to a fraction of a sample. Stated that baldly, it sounds hopeless, and for raw-sample modeling it nearly is. The rest of this post is the story of how the field made it tractable: change the representation so the model predicts far fewer, far friendlier numbers, then decode back to the waveform with a specialized network that handles the punishing last step.

#### Worked example: the token budget of one minute of music

Suppose you want to generate one minute of stereo music at 44.1 kHz and you naively model raw samples. That is $44{,}100 \times 60 \times 2 = 5{,}292{,}000$ samples. If your model is autoregressive and runs at, optimistically, 50,000 samples per second of generation on a single modern GPU (WaveNet-class models were far slower than this), you would wait roughly 106 seconds to generate 60 seconds of audio, a real-time factor (RTF) worse than 1.0. RTF is generation-time divided by audio-duration; below 1.0 is faster than real time, above 1.0 is slower. Now contrast a neural codec that compresses the same minute to about 75 tokens per second per codebook with, say, 4 codebooks: that is $75 \times 60 \times 4 = 18{,}000$ tokens, a roughly 290× reduction in the number of things the model must predict. That single number, the compression ratio of the latent, is why the entire modern field is built on neural codecs and mel-spectrograms rather than raw samples. We devote all of Track B to it.

## The mel-spectrogram: how audio became a picture

If raw samples are too many and too sequential, the obvious move is to compress. The oldest and still most important compression for generation is to turn the waveform into a **spectrogram**, and then a **mel-spectrogram**.

Here is the idea, which we will make rigorous in [representing sound](/blog/machine-learning/audio-generation/representing-sound-waveforms-spectrograms-and-perception). Take the waveform and chop it into short overlapping windows, say 25 milliseconds each, stepping forward by 10 milliseconds (the **hop length**). For each window, compute the Fourier transform, which decomposes that slice of sound into how much energy sits at each frequency. Stack those per-window spectra side by side and you get a two-dimensional image: time on the horizontal axis, frequency on the vertical, brightness equal to energy. That is the **short-time Fourier transform**, or STFT. It turns a 1D signal into a 2D picture that a convolutional or transformer model can chew on the same way it chews on images.

The **mel-spectrogram** adds one more step that matters enormously: it warps the frequency axis to match human hearing. We do not perceive pitch linearly. The difference between 200 Hz and 400 Hz sounds like a big jump (an octave), but the difference between 5,000 Hz and 5,200 Hz is barely noticeable, even though both gaps are 200 Hz. The **mel scale** is a frequency warping, roughly logarithmic above 1 kHz, that spaces frequencies the way the ear spaces them, so that equal distances on the mel axis sound like equal pitch distances. Then the linear-frequency STFT is collapsed onto a small number of mel bins, typically 80 or 128. The result is compact (80 numbers per frame instead of 1,025) and perceptually weighted (it spends its resolution where the ear cares).

![Comparison of a raw waveform with millions of samples against a compact mel-spectrogram with phase discarded](/imgs/blogs/why-audio-generation-is-hard-2.png)

The figure above is the trade laid bare. One second of 48 kHz audio is 48,000 raw samples but only about 3,000 mel cells (roughly 100 frames per second times 80 bins, but laid out as a small image). The mel-spectrogram is dramatically smaller and it is perceptually aligned. This is why mel-spectrograms have been the workhorse intermediate representation for text-to-speech for years: you predict the picture, not the wiggle, and a separate model called a **vocoder** turns the picture back into a waveform.

But notice the line in the figure that says "phase discarded." That is not a small footnote. It is the second great difficulty of audio, and it deserves its own section.

## The phase problem: why the ear catches what the eye never would

When you compute the Fourier transform of a window of audio, each frequency component has two pieces of information: a **magnitude** (how loud that frequency is) and a **phase** (where in its cycle that frequency sits at the start of the window). A spectrogram, as usually plotted and as usually fed to a model, keeps only the magnitude. It throws phase away. The mel-spectrogram throws it away too.

Why throw away half the information? Because magnitude is the part that is stable, smooth, and predictable, the part that correlates with what we perceive as timbre and pitch and loudness. Phase is jumpy, hard to model, and, for a single static sound, perceptually weak. A pure tone sounds the same whether its phase is 0 or 90 degrees. So for a long time the field reasoned: predict the magnitude spectrogram, and reconstruct a plausible phase afterward.

The trouble is that "reconstruct a plausible phase" is exactly where audio generation goes to die. Phase is not perceptually weak when the sound is changing, which is to say, always. The relative phase between overlapping windows, and between harmonics, is what makes a voice sound like a continuous, living thing rather than a sequence of stitched-together frames. Get it slightly wrong and you get the buzz I described in the opening, or a metallic ring, or a "robotic" warble, or a smearing that makes consonants mushy. The classic algorithm for guessing phase from magnitude is **Griffin-Lim**, an iterative method that alternates between the magnitude you want and the phase consistency that a real signal must have. It works, sort of, and it always sounds a little artificial. The entire reason neural vocoders like HiFi-GAN exist is that Griffin-Lim's phase reconstruction is not good enough for the ear.

Here is the deep reason the ear is so unforgiving and the eye is not. Vision integrates over space and time; your visual system is happy to average, smooth, and fill in. A single wrong pixel vanishes into the scene. Hearing, by contrast, has extraordinary **temporal acuity**. The auditory system can resolve timing differences on the order of microseconds, which is how you localize a sound source from the tiny delay between your two ears. A phase error that shifts energy by a fraction of a millisecond is, to the ear, a real and audible event. The eye has nothing like this. There is no visual equivalent of binaural timing. So audio carries a perceptual constraint that images simply do not: it must be correct not just in content but in fine temporal alignment, sample by sample, and the listener will notice when it is not.

This is why audio is unforgiving, and it is the central technical reason the field spent years on vocoders and then on neural codecs that learn to reconstruct phase implicitly. We will return to this every time we talk about a decoder.

#### Worked example: how small a phase error the ear catches

Consider a 1 kHz tone. One full cycle is 1 millisecond. A phase error of 90 degrees is a quarter cycle, 0.25 ms. That sounds tiny. But the ear's interaural time difference sensitivity is roughly 10 microseconds, which on a 1 kHz tone is a phase error of about 3.6 degrees. When a vocoder reconstructs a vowel made of dozens of harmonics, each harmonic getting its phase wrong by even a few degrees relative to the others, the errors do not average out, they beat against each other and produce the audible buzz. The eye, asked to detect a 3.6-degree rotation of a single edge in a busy image, would never see it. The ear, asked to detect a 3.6-degree phase shift in a tone, hears a different sound. That asymmetry is the whole difficulty in one comparison.

## Seeing the representation in code: waveform to mel and back

Before we leave representation, let us make it concrete with the actual toolchain, because the phase problem is something you can hear yourself in about twenty lines. We will load a waveform, compute a mel-spectrogram with `torchaudio`, then try to reconstruct a waveform from the mel using **Griffin-Lim**, the classic magnitude-only phase-guessing algorithm. The reconstruction will sound noticeably worse than the original, and that audible gap is exactly the gap a neural vocoder was invented to close.

```python
import torch
import torchaudio
import torchaudio.transforms as T

# Load any wav; torchaudio returns (channels, samples) and the sample rate.
waveform, sr = torchaudio.load("speech.wav")
waveform = waveform.mean(dim=0, keepdim=True)  # mono

# STFT -> mel-spectrogram. n_fft sets frequency resolution, hop_length the
# time step, n_mels the number of perceptual frequency bins.
mel_transform = T.MelSpectrogram(
    sample_rate=sr,
    n_fft=1024,        # 1024-sample analysis window
    hop_length=256,    # step 256 samples (~10.7 ms at 24 kHz)
    n_mels=80,         # 80 mel bins, the TTS standard
)
mel = mel_transform(waveform)        # (1, 80, num_frames) -- the "picture"
log_mel = torch.log(mel + 1e-5)      # models train on log-mel
print("waveform:", waveform.shape, "  log-mel:", log_mel.shape)
```

Run that on a one-second 24 kHz clip and you will see the waveform is `(1, 24000)` while the log-mel is roughly `(1, 80, 94)`, about 7,500 numbers instead of 24,000, and laid out as a smooth image a convolutional or transformer model can predict frame by frame. That is the compression and the perceptual warping in one transform.

Now the painful half. To get a waveform back from a magnitude mel you must invent a phase. Griffin-Lim does it by iterating: assume a random phase, reconstruct a signal, recompute its STFT, keep the new (more self-consistent) phase, repeat.

```python
# Invert the mel back to a linear-frequency magnitude spectrogram,
# then guess phase with Griffin-Lim. This is the "no neural vocoder" baseline.
inv_mel = T.InverseMelScale(n_stft=1024 // 2 + 1, n_mels=80, sample_rate=sr)
griffin_lim = T.GriffinLim(n_fft=1024, hop_length=256, n_iter=64)

linear_mag = inv_mel(mel)            # (1, 513, num_frames)
recon = griffin_lim(linear_mag)      # (1, num_samples) -- phase was guessed
torchaudio.save("recon_griffinlim.wav", recon, sr)
```

Listen to `recon_griffinlim.wav` next to the original and the difference is obvious: a watery, slightly metallic quality, smeared consonants, a faint buzz. Nothing is wrong with the content, the words and pitch are all there, but the phase was guessed and the ear knows. This single experiment is the most direct way to feel why the field moved from Griffin-Lim to learned vocoders like HiFi-GAN, which we cover in Track C. The mel-spectrogram is a wonderful representation for a model to predict; turning it back into a believable waveform is a hard, separate learned problem, and that separation is a recurring shape in the audio stack.

## Perception: we do not hear linearly, and that changes the metrics

If you remember one idea from this section, make it this: **the right loss function for audio is not the one that matches the waveform, it is the one that matches what a human perceives.** And human perception of sound is wildly nonlinear in three ways that every audio system has to respect.

First, **pitch is logarithmic**, which is the mel scale we already met. Equal ratios of frequency sound like equal musical intervals. Doubling frequency is always an octave, whether from 100 to 200 Hz or 4,000 to 8,000 Hz.

Second, **loudness is logarithmic and frequency-dependent**. We perceive sound intensity on roughly a logarithmic scale, which is why audio levels are measured in decibels (dB), a log unit. And our sensitivity is not flat across frequencies: we hear best in the 2 to 5 kHz range (where speech consonants live) and poorly at the very low and very high ends. The equal-loudness contours that describe this are why a model that minimizes raw waveform error wastes capacity on inaudible low-frequency energy and underweights the midrange that actually carries intelligibility.

Third, and most subtly, **masking**. A loud sound makes nearby quieter sounds inaudible, both in frequency (a loud tone hides quieter tones near it) and in time (a loud transient hides quieter sounds just before and after it). This is the entire basis of perceptual audio compression, the reason an MP3 can throw away most of the bits and still sound fine. For generation, masking is a gift: it means a model does not have to get the masked parts right because nobody can hear them. The hard part is knowing which parts are masked, which is what a good perceptual loss or a learned codec captures implicitly.

There is one more perceptual quantity worth naming now: the **just-noticeable difference** (JND), the smallest change in a stimulus a listener can detect. JNDs are why audio evaluation leans on human listening tests. The gold-standard metric is **MOS** (Mean Opinion Score), where human raters score samples from 1 (bad) to 5 (excellent), and **CMOS** (Comparative MOS), where they score one sample against another. Because human tests are slow and expensive, the field also uses **FAD** (Fréchet Audio Distance), an automatic metric that embeds real and generated audio with a pretrained network (originally VGGish, increasingly CLAP or other learned embeddings) and measures the distance between the two distributions, lower is better. For speech specifically there is **WER** (Word Error Rate), computed by running an ASR model on the generated speech and checking how many words it gets wrong, a direct proxy for intelligibility. We cover all of these honestly, with their failure modes, in [audio quality metrics](/blog/machine-learning/audio-generation/audio-quality-metrics). The reason these metrics exist at all is the perceptual nonlinearity above: a metric that ignores perception (like raw sample MSE) does not predict whether a human will like the sound.

The practical consequence for training is that every good audio model bakes perception into its loss in some way. A vocoder does not minimize raw waveform MSE; it minimizes a **multi-resolution STFT loss** (compare spectrograms at several window sizes, so errors are weighted in the perceptual frequency domain) plus an adversarial loss (let a discriminator learn what "real" sounds like, which implicitly captures masking and phase). A codec adds a similar spectral reconstruction loss to its quantization objective. The thread running through all of them is the same: define the loss on something close to what the ear computes, not on the raw sample values, because two waveforms with identical perceptual content can differ enormously sample by sample (shift the phase, and every sample changes while the sound does not), and two waveforms that differ trivially in MSE can sound completely different (introduce a small buzz). Sample-space distance and perceptual distance are nearly unrelated, and that disconnect is the single most important fact a newcomer to audio modeling has to internalize.

#### Worked example: why sample MSE is a liar

Take a clean 200 Hz tone and make two corrupted copies. Copy A shifts the entire waveform by half a sample (a tiny sub-sample time delay). Copy B adds a faint 4 kHz buzz at one percent amplitude. Measure mean-squared error against the original. Copy A, the time shift, can have a large MSE because nearly every sample moved, yet it sounds **identical** to the original (a half-sample delay is inaudible). Copy B, the buzz, has a tiny MSE because only a quiet high tone was added, yet it sounds **clearly worse** (the ear hears the buzz in the sensitive 4 kHz region, and it is not masked by the low tone). MSE ranked them backwards. This is not a contrived edge case; it is the normal situation, and it is why a model trained to minimize waveform MSE produces oversmoothed, lifeless audio, and why every metric and loss that matters in this series is perceptual.

## Your first sound: hello world with 🤗 transformers

Enough framing. Let us make a noise. The fastest way to generate audio today is with the Hugging Face `transformers` library, which wraps several audio models behind a common interface. Here is a complete, runnable example that generates a few seconds of music with **MusicGen**, Meta's text-to-music model, and writes a WAV file. It is about a dozen lines.

```python
import torch
import soundfile as sf
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# Load the small MusicGen model and its processor (downloads on first run).
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

# Turn a text prompt into model inputs.
inputs = processor(
    text=["warm lo-fi hip hop with a mellow piano and vinyl crackle"],
    padding=True,
    return_tensors="pt",
)

# Generate ~5 seconds of audio. MusicGen runs at 50 Hz on a 4-codebook codec,
# so 256 tokens is roughly 5 seconds of audio.
audio_values = model.generate(**inputs, max_new_tokens=256)

# The model exposes its output sample rate; write a wav.
sr = model.config.audio_encoder.sampling_rate  # 32000 for MusicGen
sf.write("lofi.wav", audio_values[0, 0].cpu().numpy(), samplerate=sr)
print(f"Wrote lofi.wav at {sr} Hz, {audio_values.shape[-1] / sr:.1f} seconds")
```

Run that and you get a `lofi.wav` you can play. Notice how much of the audio stack is hidden inside `generate`. MusicGen is an autoregressive language model over **EnCodec** codec tokens; the processor turns text into a conditioning tensor, the model predicts a sequence of codec tokens with interleaved codebooks, and EnCodec's decoder turns those tokens back into a 32 kHz waveform. You did not touch a spectrogram, a vocoder, or a phase reconstruction directly, but all of it ran. That is the whole point of the stack.

If you would rather make speech, the same library gives you a tiny TTS model, **VITS** (here via the MMS or the `kakao-enterprise/vits-ljs` checkpoints), in even fewer lines:

```python
import torch
import soundfile as sf
from transformers import VitsModel, AutoTokenizer

model = VitsModel.from_pretrained("kakao-enterprise/vits-ljs")
tokenizer = AutoTokenizer.from_pretrained("kakao-enterprise/vits-ljs")

inputs = tokenizer("the quick brown fox jumps over the lazy dog", return_tensors="pt")
with torch.no_grad():
    output = model(**inputs).waveform  # (1, num_samples)

sf.write("fox.wav", output[0].cpu().numpy(), samplerate=model.config.sampling_rate)
```

VITS is a fully end-to-end TTS model (text straight to waveform, no separate vocoder step) built from a variational autoencoder, a normalizing flow, and an adversarial decoder, which is why it is a single forward pass with no `generate` loop. Our running sentence for the speech side of this series is exactly that one, "the quick brown fox," and we will keep returning to it.

With two short snippets you have driven an autoregressive codec-LM (MusicGen) and a flow-plus-GAN end-to-end TTS (VITS). Those are two of the four families. Let us name all four properly.

## The codec, hands on: encode and decode with EnCodec

Since the neural codec is the enabler that the rest of the series stands on, it is worth seeing it run before we even define the families in detail. Here is a round-trip with Meta's **EnCodec** through `transformers`: load a waveform, encode it to discrete tokens, inspect how few tokens there are, and decode straight back to audio. This is the "latent" stage of the audio stack made literal.

```python
import torch
import torchaudio
from transformers import EncodecModel, AutoProcessor

model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

# Load and resample audio to the codec's native 24 kHz.
wav, sr = torchaudio.load("speech.wav")
wav = torchaudio.functional.resample(wav.mean(0, keepdim=True), sr, 24000)

inputs = processor(raw_audio=wav.squeeze().numpy(), sampling_rate=24000,
                   return_tensors="pt")

with torch.no_grad():
    encoded = model.encode(inputs["input_values"], inputs["padding_mask"])
    codes = encoded.audio_codes        # (1, 1, num_codebooks, num_frames)
    decoded = model.decode(codes, encoded.audio_scales,
                           inputs["padding_mask"])[0]

print("input samples:", wav.shape[-1])
print("codec frames :", codes.shape[-1], "x", codes.shape[-2], "codebooks")
torchaudio.save("encodec_roundtrip.wav", decoded.squeeze(0), 24000)
```

For a one-second clip you will see roughly 24,000 input samples collapse to about 75 frames per codebook. With the default bandwidth EnCodec uses a handful of codebooks (you can ask for more codebooks to raise the bitrate and fidelity, or fewer to compress harder). That `codes` tensor of small integers is what a codec-LM like MusicGen or VALL-E actually models: not the waveform, not the mel, but these discrete tokens. And critically, `encodec_roundtrip.wav` sounds nearly identical to the original even though it has been crushed to a few kbps, because the codec's decoder is a learned neural network that reconstructs phase well, exactly where Griffin-Lim failed. That is the difference between a hand-designed inversion and a learned decoder, and it is the reason codec tokens, not mel-spectrograms, are the substrate of most 2026 systems.

## Measuring whether it worked: a metric in code

You cannot reason about audio generation without measuring it, so here is the third practical piece: computing a metric. The most common automatic metric for generative audio is **Fréchet Audio Distance (FAD)**, which embeds two sets of clips (real and generated) with a pretrained audio network and measures the Fréchet distance between the resulting Gaussians; lower means the generated distribution is closer to the real one. The `frechet-audio-distance` package wraps this.

```python
from frechet_audio_distance import FrechetAudioDistance

# Use a learned embedding; CLAP and VGGish are both common choices.
# Different embeddings give different (incomparable) FAD numbers, so always
# report which one you used.
fad = FrechetAudioDistance(model_name="clap", sample_rate=48000,
                           use_pca=False, use_activation=False, verbose=False)

score = fad.score(
    background_dir="real_clips/",     # a set of real reference clips
    eval_dir="generated_clips/",      # the model's outputs
)
print(f"FAD (CLAP): {score:.3f}  (lower is better)")
```

Three honesty notes that the [audio quality metrics](/blog/machine-learning/audio-generation/audio-quality-metrics) post expands on. First, FAD depends entirely on the embedding network, so a FAD computed with VGGish is not comparable to one computed with CLAP; always state the embedding. Second, FAD is unstable at small sample sizes; you want hundreds to thousands of clips per side before the number means anything. Third, FAD measures distributional similarity, not per-clip quality, so a model can have a great FAD and still produce occasional disasters. For speech you would pair this with **WER** (run an ASR like Whisper on the generated speech and compare to the target text) for intelligibility and a small **MOS** listening panel for naturalness. Five snippets in, you have now touched every stage of the stack: representation, codec, generation, and evaluation. Now let us name the four families that sit at the generative core.

## The four generative families

Almost every audio generation system in 2026 is built from one or more of four model families. They differ in what they model (discrete tokens versus continuous latents), how they generate (one step at a time versus iterative refinement versus a single learned map), and what they are good at. This is the most important taxonomy in the series, so let us be precise.

![Tree of the four audio generative families split into sequence models and continuous models](/imgs/blogs/why-audio-generation-is-hard-3.png)

The tree above groups them sensibly. On one side are **sequence models** that emit discrete tokens or samples in order; on the other are **continuous models** that refine a latent. Here is each family, with its core idea, its strength, and its weakness.

### 1. Autoregressive audio language models (WaveNet to AudioLM to MusicGen)

The core idea is the same as a text language model: factor the probability of the whole signal into a product of conditional probabilities, one per step, and predict each step from everything before it. The chain rule, applied to audio:

$$p(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_1, \ldots, x_{t-1})$$

**WaveNet** (2016) did this at the level of raw samples, using dilated causal convolutions to reach far back in time without an explosion of parameters. It sounded astonishing for its day and was hopelessly slow, because $T$ is the full sample count, millions per song. The modern version, pioneered by **AudioLM** (2022) and **MusicGen** (2023), does autoregression not over raw samples but over **neural-codec tokens**, where $T$ is the few-thousand-token latent rather than the millions-of-samples waveform. That single change, modeling tokens instead of samples, is what made AR audio practical.

Strength: autoregressive models are superb at long-range structure and at conditioning. Because each step sees all of history, they capture a melody's development, a sentence's prosody, a song's verse-chorus structure. They take text conditioning naturally (it is just more context). They also inherit the entire toolbox of the text-LLM world: the same transformer architectures, the same sampling tricks (temperature, top-k, top-p), the same in-context-learning behavior that lets VALL-E clone a voice from a prompt with no fine-tuning. Weakness: they are sequential by construction, so they are slow (you cannot generate token $t+1$ until you have token $t$), and they can **drift** over long generations, gradually losing the beat or wandering off-key, because errors accumulate with no global view. One subtlety unique to audio codec-LMs is the **multi-codebook** problem: RVQ produces several codebook tokens per frame, and naively flattening them multiplies the sequence length, so MusicGen and others use clever interleaving or delay patterns to model the codebooks without paying the full sequence-length cost. That detail, which has no analogue in text, is the kind of audio-specific machinery Track C exists to explain.

### 2. Diffusion models (DiffWave, AudioLDM, Stable Audio)

The core idea is to learn to reverse a gradual noising process. You take clean audio (or a clean latent), add Gaussian noise in many small steps until it is pure noise, and train a network to undo one step of noising. At generation time you start from noise and denoise repeatedly. We derived this fully for images in [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles), and the math transfers directly; what changes for audio is the data. **DiffWave** and **WaveGrad** apply diffusion to raw waveforms. **AudioLDM** and **Stable Audio** apply it in a latent space (a codec or VAE latent), exactly as latent diffusion does for images, which is far more efficient.

Strength: diffusion produces the highest raw fidelity of any family and takes guidance beautifully (classifier-free guidance, the same trick from images, gives strong text control). It is naturally parallel across the sequence within each denoising step, so it does not have the per-sample sequential bottleneck of AR, and it does not drift the way AR does because every step sees the whole window at once, giving it a global view of structure. Weakness: it takes many denoising steps (each a full network pass), so it is compute-heavy, and it generates a fixed window, which makes truly variable-length and streaming generation awkward (Stable Audio's timing conditioning is a clever fix we cover in Track E). The training objective is the same denoising score-matching loss you saw for images, so we do not re-derive it; we reuse [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles) and spend our pages on the audio-specific parts: what you diffuse (a raw waveform, a mel, or a codec latent) and how you condition it.

### 3. GAN vocoders (HiFi-GAN, BigVGAN, Vocos)

The core idea is narrower and incredibly useful: train a generator network to turn a mel-spectrogram into a waveform in a single forward pass, and train it adversarially against discriminators that judge whether the output sounds real. **HiFi-GAN** (2020) is the canonical example; its key insight was using multiple discriminators that look at the signal at different periods and scales (the multi-period and multi-scale discriminators) to catch exactly the phase and periodicity errors the ear hates.

Strength: GAN vocoders are blisteringly fast, often well over 100× real time, and high fidelity. They are the standard "last mile" that turns a predicted mel-spectrogram into actual sound. Weakness: a vocoder is not a full generator; it does not invent content, it only renders a mel-spectrogram you already have. And as a GAN it inherits training instability and a tendency to occasional artifacts. So a vocoder is almost always a component inside a larger stack, not the whole system.

### 4. Flow matching (Voicebox, Audiobox, F5-TTS)

The newest family. Flow matching learns a continuous-time **velocity field** that transports a simple noise distribution to the data distribution, and you generate by integrating an ordinary differential equation from noise to data. We derived it in [flow matching and rectified flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow); again the math carries over and only the data changes. **Voicebox** and **Audiobox** (Meta) and **F5-TTS** brought it to speech, where it has become a leading recipe.

Strength: flow matching reaches diffusion-level fidelity in far fewer steps (the learned paths are straighter), so it is faster than diffusion while keeping strong controllability and the same classifier-free-guidance machinery. Weakness: like diffusion it works in windows rather than streaming naturally, and it is newer, so the tooling and the body of tricks are less mature than for AR or GAN vocoders.

### How they compare, and why systems combine them

Here is the comparison that should anchor your intuition. No family wins on every axis.

![Matrix comparing the four families on fidelity, controllability, speed, and long-form generation](/imgs/blogs/why-audio-generation-is-hard-5.png)

The same data as a table you can act on:

| Family | Fidelity | Controllability | Speed (RTF) | Long-form | Typical role |
|---|---|---|---|---|---|
| Autoregressive LM | High | Strong (text, in-context) | Slow (sequential) | Good but drifts | Tokens for TTS, music |
| Diffusion | Very high | Strong (CFG) | Slow (many steps) | Fixed window | High-fidelity music, vocoding |
| Flow matching | Very high | Strong (CFG) | Fast (few steps) | Window-based | Modern TTS, fast synthesis |
| GAN vocoder | High (mel in) | Mel only | Very fast (100x+) | Frame-local | Mel-to-waveform last mile |

Notice the structure. Autoregressive models own long-range structure and conditioning but are slow. Diffusion and flow own fidelity but work in windows. GAN vocoders own speed but cannot invent content. So the obvious engineering move, and the one nearly every shipped system makes, is to **combine** them: use one family for the part it is best at and another for the rest.

![Graph of a combined production stack with a codec encoder, generative core, codec tokens, and decoder](/imgs/blogs/why-audio-generation-is-hard-6.png)

The combined stack above is how production audio actually works. A neural codec compresses audio into a short token sequence. A generative core, often an autoregressive LM or a diffusion or flow model, produces those tokens (conditioned on text or a speaker prompt). A decoder, the codec's own decoder or a separate vocoder, turns tokens back into a waveform. Concretely: MusicGen is an AR LM over EnCodec tokens with the codec doing the decoding. Modern VALL-E-style TTS is an AR LM over codec tokens with a 3-second speaker prompt for cloning. A Tacotron-plus-HiFi-GAN pipeline is a sequence model predicting a mel-spectrogram and a GAN vocoder rendering it. Stable Audio is latent diffusion over a codec latent. In every case it is a stack, not a monolith, and the stack is the frame that organizes this entire series.

## Stress-testing the stack: where it breaks

Naming the families is easy; the engineering is in knowing how they fail. Let me pose the real problems you hit and reason through what breaks, because this is the kind of thinking the rest of the series operationalizes.

**What happens at a lower bitrate?** Push a codec to fewer codebooks or a lower kbps and reconstruction degrades, but not uniformly. The codec spends its remaining bits on the loudest, most-energetic parts and starves the quiet, high-frequency detail first, so the symptom is dull, muffled high end, then a watery artifact, then, at very low bitrate, an audible "underwater" warble. For a generator built on that codec, the ceiling is the codec: a codec-LM can never sound better than its decoder, so if you hear artifacts in the EnCodec round-trip above, no amount of better language modeling fixes them. The fix is to raise the codec bitrate, which costs you tokens (the sequence-length problem returns), which is the rate-distortion trade made physical.

**What happens when the speaker prompt is three seconds of noisy audio?** Zero-shot voice cloning conditions on a short reference clip. Feed it a clean studio sample and VALL-E-class models clone strikingly well. Feed it three seconds of phone audio with background hum and the model clones the **hum** too, because it has no way to separate "this is the speaker's timbre" from "this is the room and the codec of the reference." The symptom is a clone that captures the right voice riding on the wrong acoustic. The fix is reference-audio denoising and longer, cleaner prompts, which Track D covers.

**What happens when you ask for four minutes and it loses the beat?** Autoregressive music models generate left to right and accumulate error. Over a few seconds they are coherent; over minutes they drift, gradually wandering off-key or letting the tempo slip, because each step only sees its own past, never a global plan, and there is no mechanism pulling it back to the original groove. The symptom is a song that starts strong and dissolves. The fix is either a model designed for length (Stable Audio's timing-conditioned diffusion fills a known-length window all at once) or segment-and-condition generation that re-anchors the model periodically. Do not autoregress four minutes in one pass and expect structure.

**What happens when the vocoder is the bottleneck, not the model?** A surprising number of "the TTS sounds bad" bugs are vocoder bugs. The acoustic model predicts a perfect mel-spectrogram and the vocoder, mismatched in sample rate or trained on a different mel configuration, renders it with a buzz or a pitch error. The symptom is good content with a consistent sonic defect, and the diagnosis is to vocode the **ground-truth** mel: if that also buzzes, the vocoder is the problem, not the model. This is the single most useful debugging move in a TTS stack and it falls straight out of treating the stack as separable stages.

The meta-lesson of all four: because audio is a stack, a defect lives in exactly one stage, and you find it by isolating stages, not by retraining the whole thing. Bisect the stack the way you would bisect a commit history.

## The audio stack, formalized

Let us name the stages once, carefully, because they recur in every later post.

1. **Waveform.** The raw 1D signal at the target sample rate (16/24 kHz for speech, 44.1/48 kHz for music). This is what goes in during training and comes out during inference.

2. **Latent / token representation.** A compressed intermediate. Two dominant kinds: a **mel-spectrogram** (a perceptually warped magnitude image, the classic TTS intermediate) or **neural-codec tokens** (discrete codes from a learned encoder-quantizer, the modern intermediate that enables language-model-style generation). The codec is, by analogy, the audio version of the [VAE](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) that latent image diffusion runs in: a learned compressor that gives the generator a smaller, friendlier space to work in. Track B is entirely about this.

3. **Generative core.** One (or more) of the four families, operating on the latent. AR LM over tokens, diffusion or flow over a continuous latent, conditioned on text, a speaker, a melody, or a description. Track C covers the engines; Tracks D and E cover their application to speech and music.

4. **Decoder / vocoder.** Turns the latent back into a waveform. If the latent is codec tokens, the codec's decoder does it. If the latent is a mel-spectrogram, a vocoder (HiFi-GAN, BigVGAN, Vocos) does it. This is where phase gets reconstructed and where the ear's verdict is decided.

5. **Waveform out.** The result, evaluated on fidelity (FAD, MOS), controllability (did it follow the prompt, the speaker, the melody), speed (RTF, seconds-to-generate on a named device), and length (did it hold structure over the full duration).

Every design choice in audio generation is a bet about where to spend in this stack and which family to use at the generative core. A streaming voice assistant spends on speed and accepts slightly lower fidelity (small codec, fast core, fast vocoder). A music-generation product spends on fidelity and length and accepts slow generation (diffusion or a large AR LM, high-quality codec). A voice-cloning tool spends on controllability (a strong AR LM that does in-context cloning from a short prompt). The stack is fixed; the bets are what differ.

It is worth saying explicitly why this layered structure is such a good idea, because it is not obvious and it took the field years to settle on it. A single end-to-end model that maps text straight to a 48 kHz waveform has to solve four hard problems at once: understand the prompt, plan long-range structure, render fine acoustic detail, and reconstruct phase, all in one set of weights and one loss. That is a brutally entangled optimization, and early monolithic models paid for it in quality and training instability. Splitting the job into a stack lets each stage be trained and improved independently, with the loss best suited to it: a reconstruction loss for the codec, a denoising or next-token loss for the generator, an adversarial-plus-spectral loss for the vocoder. You can swap a better codec under a fixed generator, or a better vocoder under a fixed acoustic model, and get a free quality bump without retraining everything. That modularity is why progress in audio has been so fast: improvements compound across the stack instead of being locked inside a monolith. It is the same lesson the rest of deep learning keeps relearning, and audio is one of its cleanest examples.

## The three sub-domains: TTS, music, and sound effects

Audio generation is not one problem, it is three, and they have genuinely different constraints. The series gives each its own track.

**Text-to-speech (TTS).** The most mature sub-domain. You have text and you want intelligible, natural, correctly-pronounced speech, often in a specific voice. The dominant constraints are intelligibility (measured by WER), naturalness (MOS), speaker similarity for cloning, and, increasingly, latency for real-time and full-duplex conversation. The frontier is **zero-shot voice cloning** from a few seconds of reference audio (VALL-E, XTTS, F5-TTS) and **full-duplex** conversational speech (Moshi). Track D is TTS, from Tacotron and VITS through codec-LM TTS and the cloning frontier.

**Music generation.** Harder than speech in some ways: longer (songs are minutes), more structured (rhythm, harmony, song form), and more subjective (there is no "word error rate" for a melody). The constraints are musicality, adherence to a text or melody prompt, length, and fidelity. The frontier is open codec-LMs (MusicGen), latent diffusion (Stable Audio, AudioLDM 2), and commercial full-song systems with vocals and lyrics (Suno, Udio). Track E is music and sound.

**Sound and sound effects (SFX / foley).** Generating non-speech, non-music audio: a door slam, rain, a sci-fi laser, environmental ambience, or audio to match a video. The constraints are realism, controllability from a text description, and synchronization with video. The frontier is AudioGen and text-to-audio models, plus video-to-audio, which links directly to the video series' [audio and joint AV generation](/blog/machine-learning/video-generation/audio-and-joint-av-generation) post. Track E covers SFX and the video link.

These three share machinery (codecs, the four families, the stack) but diverge in evaluation and in what "good" means. A TTS model is judged on whether you understand the words; a music model on whether the groove holds for two minutes; an SFX model on whether the rain sounds like rain. Keep the sub-domain in mind whenever you read a result, because a number that is great for one can be irrelevant for another. The contrast is worth a table:

| Sub-domain | Typical length | Primary metric | What "good" means | Frontier system |
|---|---|---|---|---|
| Text-to-speech | Seconds to minutes | WER, MOS, speaker similarity | Intelligible, natural, right voice | VALL-E, XTTS, F5-TTS, Moshi |
| Music | Tens of seconds to minutes | FAD, MOS, prompt adherence | Musical, on-prompt, holds structure | MusicGen, Stable Audio, Suno |
| Sound / SFX | Sub-second to seconds | FAD, CLAP-score, sync | Realistic, controllable, in-sync | AudioGen, video-to-audio |

Notice that even the length scales differ by orders of magnitude (a door slam is a fraction of a second; a song is minutes), which changes the right architecture: a fixed-window diffusion model is natural for short SFX, awkward for a full song; an autoregressive LM is natural for streaming speech, risky for long music. The sub-domain quietly dictates the stack.

## From WaveNet to Suno: how we got here

It helps to see the path, because each landmark added exactly one idea that the next one built on.

![Timeline of audio generation landmarks from WaveNet in 2016 to Suno in 2024](/imgs/blogs/why-audio-generation-is-hard-4.png)

The timeline above is the short version; here is the story.

**WaveNet (van den Oord et al., 2016)** proved you could generate raw audio with a neural network at all, autoregressively, sample by sample, using dilated causal convolutions to see far into the past cheaply. It was a quality breakthrough and a speed disaster; generating a second of audio took minutes. Everything since has, in one way or another, been about keeping WaveNet's quality while escaping its speed.

**HiFi-GAN (Kong et al., 2020)** solved the last mile. Instead of generating samples autoregressively, generate a mel-spectrogram with some other model and then convert mel to waveform in a single fast pass with an adversarially-trained generator, judged by multi-period and multi-scale discriminators that catch the phase errors the ear hates. This made the mel-spectrogram pipeline (text to mel to waveform) fast enough to ship, and it is still the dominant vocoder pattern.

**SoundStream (Zeghidour et al., 2021) and EnCodec (Défossez et al., 2022)** introduced the neural audio codec with **residual vector quantization** (RVQ): a learned encoder-decoder that compresses a waveform to a short sequence of discrete tokens at a few kbps with high fidelity. This is the enabler of everything that followed, because it turned audio into a sequence of tokens that a language model could model directly.

**AudioLM (Borsos et al., 2022)** combined two kinds of tokens: **semantic tokens** (from a self-supervised model like w2v-BERT, capturing content and structure) and **acoustic tokens** (from the codec, capturing fidelity and speaker identity), and modeled them hierarchically with a language model. This semantic-plus-acoustic two-token paradigm (Track B4) is the architectural backbone of modern audio LMs.

**VALL-E (Wang et al., 2023)** reframed text-to-speech as codec-token language modeling and discovered that it could clone a voice from a **3-second** prompt with no fine-tuning, purely by in-context learning, the same way a text LM does few-shot tasks. Zero-shot voice cloning went from a research dream to a checkpoint you could run.

**MusicGen (Copet et al., 2023)** brought the codec-LM recipe to music: a single-stage transformer over EnCodec tokens with a clever codebook-interleaving pattern and optional melody conditioning. It is the open music-generation workhorse and a model we will dissect in Track E.

**Stable Audio (Evans et al., 2024)** showed that latent diffusion, the image recipe, scales to long, high-fidelity, variable-length music using timing conditioning so the model knows where in a fixed-length window the actual content should sit.

**Suno and Udio (around 2024)** are the commercial frontier: full songs with coherent vocals, lyrics, and production from a text prompt. Their exact recipes are not public (we treat them as report-style in Track E, with honest uncertainty), but they clearly stand on this whole stack.

Here is the same set as a reference table. Years for closed commercial systems are approximate.

![Matrix of landmark audio systems with year and core idea from WaveNet to Suno](/imgs/blogs/why-audio-generation-is-hard-7.png)

| System | Year | Core idea |
|---|---|---|
| WaveNet | 2016 | Autoregressive raw samples via dilated causal convolutions |
| HiFi-GAN | 2020 | Fast adversarial mel-to-waveform vocoder, multi-period/scale discriminators |
| EnCodec | 2022 | Neural audio codec with residual vector quantization, low-bitrate tokens |
| AudioLM | 2022 | Semantic plus acoustic tokens modeled hierarchically by an LM |
| VALL-E | 2023 | TTS as codec-token LM, 3-second zero-shot voice cloning |
| MusicGen | 2023 | Single-stage codec LM for music with melody conditioning |
| Stable Audio | 2024 | Latent diffusion with timing conditioning for variable-length music |
| Suno / Udio | ~2024 (approx.) | Full songs with vocals and lyrics, commercial frontier |

## Case studies: real numbers from shipped systems

Orientation posts can float free of reality, so let us ground the families in a few named results from the literature and from models you can run. These are approximate and depend heavily on hardware, configuration, and measurement protocol; treat them as order-of-magnitude anchors, not benchmarks, and check the original papers for exact figures.

**EnCodec's bitrate-versus-quality curve.** EnCodec (Défossez et al., 2022) reconstructs 24 kHz audio at bitrates from roughly 1.5 to 24 kbps by varying the number of RVQ codebooks. At the low end the audio is intelligible but visibly compressed; at 6 kbps it is high quality for speech and good for music; the gains flatten as you add codebooks. The Descript Audio Codec (DAC) later pushed comparable or better quality to lower bitrates, which is why DAC became a popular generation substrate. The actionable takeaway: more codebooks means better reconstruction and a longer token sequence for any downstream generator, so codec bitrate is a knob you tune against your generator's sequence-length budget, not a free quality dial.

**HiFi-GAN's real-time factor.** HiFi-GAN (Kong et al., 2020) reported synthesis far faster than real time on GPU, with its smaller variants exceeding 100× real time while staying near ground-truth MOS for vocoding. This is the number that made the mel-plus-vocoder pipeline shippable: the vocoder is essentially free relative to the acoustic model, so the design question becomes how good a mel you can predict, not how fast you can render it.

**VALL-E's zero-shot cloning.** VALL-E (Wang et al., 2023) demonstrated that conditioning a codec-token LM on a roughly 3-second enrolled speaker prompt yields zero-shot voice cloning with strong speaker similarity and competitive naturalness, with no per-speaker fine-tuning. The headline is not a single metric but a capability shift: voice cloning went from "collect minutes of data and fine-tune" to "paste in three seconds," which is exactly why the safety track later in this series is not optional.

**MusicGen's size and conditioning.** MusicGen (Copet et al., 2023) ships in several sizes (roughly 300M, 1.5B, and 3.3B parameters) and models EnCodec tokens with a single-stage transformer, optionally conditioned on a reference melody. Bigger models sound better and cost more to run; melody conditioning lets you hum a tune and get an arrangement. It is the open workhorse for short clips, and its limits (it is not built for multi-minute songs and will drift) are exactly the AR weaknesses we discussed.

| System | Family | Headline result (approximate) | Source |
|---|---|---|---|
| EnCodec | Neural codec | High quality at ~6 kbps, 24 kHz, RVQ codebooks tunable 1.5-24 kbps | Défossez et al. 2022 |
| HiFi-GAN | GAN vocoder | >100x real-time vocoding, near ground-truth MOS | Kong et al. 2020 |
| VALL-E | AR codec-LM | Zero-shot voice clone from a ~3-second prompt | Wang et al. 2023 |
| MusicGen | AR codec-LM | ~300M-3.3B params, melody-conditioned music | Copet et al. 2023 |

The pattern across these is the thesis of the whole series in miniature: a codec makes the sequence short, a generative core (here mostly AR) produces the latent, a decoder renders it fast, and the interesting capabilities (cloning, melody control, length) live in how those pieces are wired, not in any one of them alone.

## A short science block: why the sequence-length argument is real

Let me make the central difficulty quantitative, because it is the reason for the whole stack rather than a slogan. Take an autoregressive model. Its compute and latency scale with the number of generation steps $T$, the sequence length. For raw 24 kHz speech, one second is $T = 24{,}000$. For a transformer with full self-attention, the cost of attention is $O(T^2)$ per layer, so a one-second clip costs on the order of $24{,}000^2 \approx 5.8 \times 10^8$ attention interactions per layer, and a ten-second clip is a hundred times that. This is simply intractable for raw samples, which is why WaveNet used convolutions, not attention, and was still slow.

Now introduce a neural codec at 24 kHz that downsamples by a factor of 320 (a typical EnCodec stride) to a token rate of 75 Hz, with, say, 8 codebooks. One second is now $T = 75$ frames; even counting the 8 codebooks per frame, that is 600 tokens, not 24,000 samples. The attention cost drops from $24{,}000^2$ to something on the order of $600^2 \approx 3.6 \times 10^5$, a reduction of more than three orders of magnitude. Suddenly a transformer is the right tool, and the entire codec-LM paradigm (AudioLM, VALL-E, MusicGen) follows directly from this arithmetic. The codec is not a convenience; it is the thing that moves audio from the $O(T^2)$-on-millions-of-samples regime into the $O(T^2)$-on-hundreds-of-tokens regime where modern generative models live.

The flip side, and the reason this is a genuine trade rather than a free lunch, is **rate-distortion**: the more you compress (fewer tokens, lower bitrate), the more the codec must discard, and at some point it discards things the ear can hear. The art of a neural codec, which we spend Track B on, is pushing the bitrate down while keeping the distortion below the ear's just-noticeable threshold. RVQ's stacked-codebook design is exactly a mechanism for trading bitrate against distortion in a controllable way. When you read later that EnCodec hits high quality at 6 kbps or that DAC pushes it lower, that number is the codec winning the rate-distortion fight that this whole paradigm depends on.

We can make the bitrate arithmetic concrete, because it is just counting bits. A codec with $N_q$ codebooks, each of size $K$ (so $\log_2 K$ bits per code), running at a frame rate of $f_r$ frames per second, produces a bitrate of

$$R = N_q \cdot \log_2(K) \cdot f_r \ \text{bits per second.}$$

For a typical EnCodec configuration with $K = 1024$ (so 10 bits per code), $f_r = 75$ Hz, and $N_q = 8$ codebooks, that is $8 \times 10 \times 75 = 6{,}000$ bits per second, exactly the 6 kbps you keep hearing about. Halve the codebooks to $N_q = 4$ and you halve the bitrate to 3 kbps and halve the token count the generator must produce, but you also climb the distortion curve and start to hear it. This formula is the lever: every term is a design choice ($K$, $f_r$, $N_q$), and each one trades reconstruction quality against sequence length and generation cost. The whole reason RVQ uses a *stack* of codebooks rather than one giant codebook is that adding a codebook adds $\log_2 K$ bits per frame and refines the residual, giving a smooth, tunable rate-distortion curve instead of an all-or-nothing jump. Track B2 derives why the stacked design bends the curve the way it does; for now, hold onto the fact that $R = N_q \log_2(K) f_r$ is the equation linking the ear (distortion) to the model (sequence length).

## The frontier at a glance (2024 to 2026)

Where does the field stand as of 2026? A quick orientation, expanded across the series.

**Codecs** are the quiet revolution. EnCodec, the **Descript Audio Codec (DAC)**, and **Mimi** (the streaming codec inside Moshi) deliver high fidelity at low bitrate and have become the standard substrate. The trend is lower bitrate, lower latency (streaming-capable), and better reconstruction, and a growing split between codecs optimized for reconstruction and codecs optimized as a generation substrate. Track B.

**Text-to-speech** has effectively solved naturalness for read speech in major languages and moved to zero-shot cloning (XTTS, F5-TTS, NaturalSpeech 3), expressive and instruction-controlled style, and real-time full-duplex conversation (Moshi, GPT-4o-voice-style systems). The open-versus-closed gap is real: ElevenLabs-class quality and reliability still lead the open checkpoints, but the gap is closing fast. Track D.

**Music** is split between strong open systems (MusicGen, Stable Audio, AudioLDM 2) that give you control and reproducibility, and commercial full-song systems (Suno, Udio) that produce strikingly complete songs with vocals but are closed and raise live questions about training data and rights. Track E.

**Sound and foley** (AudioGen and text-to-audio) and **video-to-audio** are less mature but moving fast, and they connect generative audio to the video stack. Track E.

**Safety** is now a first-class concern, not an afterthought. Voice cloning from three seconds is a deepfake vector, so watermarking (AudioSeal, SynthID-audio), deepfake detection, and provenance (C2PA-for-audio) are active and necessary. We treat this with a detection-and-defense framing in Track F.

A few cross-cutting trends are worth naming because they shape every sub-domain at once. The first is **convergence on the codec-token substrate**: speech, music, and sound increasingly run through the same kind of neural codec, which means a better codec lifts all three, and codec research has outsized leverage. The second is **streaming and latency** becoming the dominant axis: as audio generation moves from offline rendering into live assistants and conversational agents, the question shifts from "how good can it sound given unlimited time" to "how good can it sound at an interactive real-time factor," which favors flow matching and fast vocoders over many-step diffusion and reshapes which family wins. The third is the **open-versus-closed gap**, which is real and uneven: open checkpoints (MusicGen, XTTS, F5-TTS, the open codecs) are strong and reproducible and what you should reach for to learn and to build on, while the closed commercial frontier (ElevenLabs for voice, Suno and Udio for song) still leads on polish, reliability, and end-to-end song coherence, at the cost of being a black box you cannot inspect, fine-tune, or fully trust on provenance. Knowing which side of that gap a given result comes from is part of reading the field honestly, and it is a thread we pull through to the 2026-landscape post in Track F.

![Tree of the six series tracks grouped into building blocks and generation systems](/imgs/blogs/why-audio-generation-is-hard-8.png)

## The series map: six tracks

The figure above is the plan. Here is what each track fills in, and which one to read next depending on what you want.

**Track A, Foundations (you are here).** This post frames the problem. The siblings go deep on representation and theory: [representing sound: waveforms, spectrograms, and perception](/blog/machine-learning/audio-generation/representing-sound-waveforms-spectrograms-and-perception) (the mel-spectrogram, the phase problem, psychoacoustics in detail), [the mathematics of audio signals](/blog/machine-learning/audio-generation/the-mathematics-of-audio-signals) (Fourier, STFT, the time-frequency uncertainty trade, Nyquist, windowing), and [audio quality metrics](/blog/machine-learning/audio-generation/audio-quality-metrics) (FAD, MOS, PESQ, STOI, SI-SDR, CLAP-score, and where each lies). Read these next if you want the DSP and evaluation grounding before any model.

**Track B, Neural audio codecs.** The tokenizer of sound: how a learned encoder-quantizer-decoder turns a waveform into a short token sequence (the audio analogue of the image VAE), residual vector quantization and its rate-distortion math, EnCodec/DAC/Mimi, and the semantic-versus-acoustic two-token paradigm. Read this if you want to understand the substrate that makes modern generation possible.

**Track C, Generative engines.** The four families in depth: autoregressive audio models (WaveNet to AudioLM), diffusion for audio, flow matching and consistency for audio, GAN vocoders, and how conditioning and control (text, speaker, melody, classifier-free guidance) get injected. Read this if you know the families by name now and want the mechanisms.

**Track D, Text-to-speech.** The full TTS arc: Tacotron to VITS, codec-LM TTS (VALL-E), zero-shot cloning (XTTS, F5-TTS), prosody and expressive control, and real-time full-duplex speech (Moshi). Read this if you build voice products.

**Track E, Music and sound.** MusicLM and MusicGen, latent diffusion for music (Stable Audio), the commercial frontier (Suno, Udio), text-to-audio and sound effects, and singing-voice synthesis and music editing. Read this if you build music or audio products.

**Track F, Frontier, safety, and the capstone.** Honest evaluation, audio deepfakes and watermarking and voice safety, the 2026 model landscape as a buyer's and builder's guide, and the [capstone that builds a full audio generation stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack) end to end (choose a model, assemble codec-to-model-to-vocoder, fine-tune or clone, serve with latency and cost budgets, evaluate and watermark). Read the capstone when you are ready to ship.

## When to reach for what (a first decision sketch)

You will get full decision trees in each track, but here is the orientation-level version so you leave this post able to make a first call.

**If you need speech and want the easiest path:** start with an off-the-shelf TTS like VITS or XTTS through `transformers` or Coqui `TTS`. Do not build a vocoder yourself. A mel-plus-HiFi-GAN or end-to-end VITS pipeline beats a raw-waveform model for nearly all TTS, because the mel intermediate is compact and the GAN vocoder is fast and good. Reach for codec-LM TTS (VALL-E-style) only when you specifically need zero-shot cloning from a short prompt.

**If you need music:** start with MusicGen (open, controllable, reproducible) for short clips and melody conditioning, or Stable Audio for longer, higher-fidelity, variable-length pieces. Do not autoregress a four-minute song in one pass and expect it to hold structure; it will drift. Use a model designed for length, or generate in segments with conditioning.

**If you need sound effects or foley:** reach for AudioGen-class text-to-audio, and for video-synced audio look at the video-to-audio frontier (and the AV link to the video series).

**If you care about speed above all (streaming, real time):** prefer a fast vocoder (HiFi-GAN/Vocos) and a flow-matching or small AR core over a many-step diffusion model. Do not use a diffusion vocoder when HiFi-GAN hits your quality bar 100× faster; the extra fidelity rarely justifies the latency.

**If you care about fidelity above all (offline music, mastering):** diffusion or flow at the core with a high-quality codec, and accept the slower generation.

**If you are doing anything user-facing with voice:** plan for watermarking and consent from day one, not as an afterthought. The same technology that clones a voice for an audiobook clones it for fraud.

#### Worked example: choosing a stack for a meditation-app narration feature

Say you are adding spoken meditations to an app: a few hundred scripts, one calm narrator voice, generated offline and cached, English only, and you want it to sound human. Walk the stack. Latent: a mel-spectrogram is fine; you do not need codec tokens because you are not doing zero-shot cloning, you have one fixed voice. Generative core: a non-autoregressive TTS like FastSpeech-style or a flow-matching TTS gives you fast, stable, controllable prosody; pure AR is unnecessary risk of drift for long passages. Vocoder: HiFi-GAN or Vocos, fast and high quality. Evaluation: MOS on a small rater panel plus WER via an ASR to catch mispronunciations, on a fixed held-out set of scripts. Speed: offline and cached, so RTF barely matters, optimize for quality. Safety: it is one consented narrator voice you control, but still watermark the output so it cannot be lifted and misused. Total system: a mel-TTS plus HiFi-GAN, which is a decades-proven, cheap, high-quality stack, and nothing more exotic is warranted. The lesson: match the stack to the job, and do not reach for the frontier when the workhorse clears the bar.

## Key takeaways

- **Audio is a 1D signal sampled at a high rate** (16/24 kHz speech, 44.1/48 kHz music), so a few minutes is millions of deeply-ordered samples. This sequence-length problem, not raw difficulty of content, is what makes audio structurally hard, and it is why the field generates compact latents (codec tokens, mel-spectrograms), not raw samples.
- **The ear is unforgiving in a way the eye is not.** Its microsecond-scale temporal acuity means tiny phase and alignment errors become audible buzz, ring, or warble. Phase reconstruction is where audio generation goes to die, and it is why neural vocoders and codecs exist.
- **We hear nonlinearly:** pitch is logarithmic (the mel scale), loudness is logarithmic and frequency-dependent, and masking hides quiet sounds near loud ones. Metrics like FAD, MOS, WER, and CLAP-score exist because waveform-matching losses do not predict perception.
- **There are four generative families:** autoregressive LMs (long-range, controllable, slow), diffusion (highest fidelity, many steps), flow matching (diffusion-quality in fewer steps), and GAN vocoders (the fast mel-to-waveform last mile). None wins on every axis.
- **Production systems combine families on the audio stack:** waveform to codec/mel latent to a generative core to a decoder/vocoder back to waveform, trading fidelity, controllability, speed, and length. This stack is the spine of the whole series.
- **The neural codec is the enabler.** Modeling a 75 Hz token sequence instead of a 24 kHz waveform drops the cost by orders of magnitude and is what moved audio into the modern generative-model regime.
- **There are three sub-domains** (TTS, music, sound/SFX) with genuinely different constraints and definitions of "good"; always read a result in the context of its sub-domain.
- **The frontier (2024 to 2026)** is low-bitrate streaming codecs, zero-shot voice cloning and full-duplex speech, open and commercial music generation, video-to-audio, and a now-mandatory safety layer of watermarking and detection.

## Further reading

- van den Oord et al., **WaveNet: A Generative Model for Raw Audio** (2016), the paper that proved neural raw-audio generation.
- Kong, Kim, Bae, **HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis** (2020), the fast vocoder pattern.
- Zeghidour et al., **SoundStream: An End-to-End Neural Audio Codec** (2021), and Défossez et al., **High Fidelity Neural Audio Compression (EnCodec)** (2022), the neural codec foundation.
- Borsos et al., **AudioLM: a Language Modeling Approach to Audio Generation** (2022), the semantic-plus-acoustic token paradigm.
- Wang et al., **Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)** (2023), and Copet et al., **Simple and Controllable Music Generation (MusicGen)** (2023), the codec-LM recipe for speech and music.
- Evans et al., **Fast Timing-Conditioned Latent Audio Diffusion (Stable Audio)** (2024), latent diffusion for variable-length audio.
- Hugging Face `transformers` and `diffusers` audio docs, and the `audiocraft` repository, for the runnable toolchain used throughout this series.
- Within this series, read next: [representing sound: waveforms, spectrograms, and perception](/blog/machine-learning/audio-generation/representing-sound-waveforms-spectrograms-and-perception), [the mathematics of audio signals](/blog/machine-learning/audio-generation/the-mathematics-of-audio-signals), [audio quality metrics](/blog/machine-learning/audio-generation/audio-quality-metrics), and the capstone [building an audio generation stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack). For the shared diffusion and flow machinery this series reuses, see [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles) and [why generating images is hard](/blog/machine-learning/image-generation/why-generating-images-is-hard).
