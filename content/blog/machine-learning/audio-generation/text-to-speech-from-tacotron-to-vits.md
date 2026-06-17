---
title: "Text-to-Speech: From Tacotron to VITS"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "How neural TTS went from a brittle attention-aligned mel predictor to a single end-to-end model that turns text into a waveform with natural prosody, and the duration-prediction idea that made it robust."
tags:
  [
    "audio-generation",
    "audio-synthesis",
    "text-to-speech",
    "tacotron",
    "fastspeech",
    "vits",
    "generative-ai",
    "deep-learning",
    "speech-synthesis",
  ]
category: "machine-learning"
subcategory: "Audio Generation"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/text-to-speech-from-tacotron-to-vits-1.png"
---

The first text-to-speech system I shipped read product descriptions aloud, and it had a tell. On most sentences it sounded almost human, the kind of almost that makes you lean in. Then it would hit a SKU like "RTX-4090-FE-24GB" and the voice would do something no human throat does: it would *skip*. The "24GB" simply vanished. On the next sentence, asked to say "the the the discount applies," it would *repeat*, looping "the" four or five times like a record stuck in a groove, before lurching forward. And once in a while, on a long sentence with a comma in an unexpected place, it would slide into a kind of *mumble*, a soft slur where the words dissolved into a continuous oatmeal of sound. The model was a Tacotron 2, the best open architecture of its day, and these were not bugs in my code. They were the signature failures of the entire autoregressive-attention approach to speech.

This post is the story of how the field engineered those failures out of existence. It is a story in three acts. The first act is **Tacotron and Tacotron 2**: the sequence-to-sequence model with attention that, for the first time, let you train a neural network to map raw text to a mel-spectrogram and get genuinely natural speech out the other side, at the cost of those skip-repeat-mumble failures. The second act is **FastSpeech and FastSpeech 2**: a deceptively simple idea, that you should *predict how long each sound lasts* and then generate every frame in parallel, which made TTS fast and robust and killed the attention failures dead. The third act is **VITS**: a single model that swallowed the acoustic model and the vocoder whole, going from text straight to waveform, using a conditional variational autoencoder with a normalizing-flow prior, a stochastic duration predictor, and adversarial training. By the end you will understand each architecture well enough to read its code, know exactly which trade-off each one makes, and be able to choose the right one for a job.

![A vertical stack diagram of the classic text-to-speech pipeline showing text becoming phonemes, an acoustic model emitting a mel-spectrogram, and a vocoder producing a waveform](/imgs/blogs/text-to-speech-from-tacotron-to-vits-1.png)

The figure above is the spine of everything that follows: the **classic neural-TTS pipeline**. Text on top, a waveform at the bottom, and in between two learned stages and one rule-based one. First, **grapheme-to-phoneme** conversion turns the written word into the sounds it represents. Then an **acoustic model** predicts a mel-spectrogram, a compact time-frequency picture of the speech. Finally a **vocoder** turns that mel-spectrogram back into an audible waveform. Tacotron and FastSpeech are different *acoustic models*; VITS is the model that merged the acoustic model and the vocoder into one. This sits on the series spine: the [audio stack](/blog/machine-learning/audio-generation/why-audio-generation-is-hard) of waveform to latent to generative model to vocoder back to waveform, under the tension of fidelity, controllability, speed, and length. TTS is where that tension is at its sharpest, because a listener will forgive a slightly wrong image but will instantly catch a skipped syllable.

## The TTS pipeline, decomposed

Before any neural network, you have to understand why TTS is decomposed the way it is, because the decomposition survived the entire deep-learning revolution and still structures every system you will build.

Speech synthesis maps a string of text to a waveform: a sequence of, say, twenty characters to a sequence of perhaps eighty thousand audio samples (a three-second utterance at 22,050 samples per second is 66,150 samples). That is a length ratio of roughly four thousand to one. No model maps twenty things to eighty thousand things in one clean shot; the gap is too large and the alignment between the two is too uncertain. So the field cut the problem into stages, each bridging part of the gap, and each stage became a research field of its own.

The **first stage is text normalization and grapheme-to-phoneme conversion**, usually written G2P. Text is full of things that are not pronounced the way they are spelled. "2024" is "twenty twenty-four," "Dr." is "doctor" or "drive" depending on context, "\$5" is "five dollars," and "read" is either "reed" or "red" depending on tense. Text normalization expands all of that into spoken-form words. Then G2P converts each word into **phonemes**, the atomic sounds of a language. The word "fox" becomes the three phonemes `F AA1 K S` in the ARPABET notation that English TTS systems traditionally use, where the `1` marks primary stress. Phonemes matter because spelling is a terrible guide to pronunciation in English: "though," "through," "tough," and "thought" share letters but almost no sounds. A model that reads phonemes never has to learn that "ough" is pronounced five different ways; the G2P front-end already resolved it.

The **second stage is the acoustic model**, the learned heart of the system. It takes the phoneme (or character) sequence and predicts a **mel-spectrogram**: a two-dimensional array, time along one axis and frequency along the other, where each column is one short frame of audio (typically about 11.6 milliseconds, giving roughly 86 frames per second) and each row is the energy in one mel-frequency band (typically 80 bands). If you have read the post on [representing sound](/blog/machine-learning/audio-generation/representing-sound-waveforms-spectrograms-and-perception), you know why the mel-spectrogram is the workhorse representation: it discards phase, compresses frequency onto a perceptual scale, and is small enough for a network to predict directly. Tacotron and FastSpeech are both acoustic models in this exact sense. They differ only in *how* they get from phonemes to that grid of numbers.

The **third stage is the vocoder**, which inverts the mel-spectrogram back into a waveform. This is genuinely hard because the mel-spectrogram threw phase away, and a waveform needs phase. The vocoder has to *invent* a plausible phase that makes the magnitudes audible and natural. For years this was Griffin-Lim, an iterative algorithm that sounds robotic; then WaveNet made it neural but slow; then [HiFi-GAN and its GAN-vocoder cousins](/blog/machine-learning/audio-generation/gan-vocoders-hifi-gan-and-fast-synthesis) made it both neural and fast, hitting real-time factors well under 0.05 (meaning it synthesizes audio more than twenty times faster than real time). The vocoder is its own deep dive; here it is the box at the bottom of the pipeline that takes a mel and gives you sound.

Why split it this way at all, rather than train one model end to end? Three reasons, and they are worth internalizing because they explain the whole arc of this post. **Modularity**: you can improve the vocoder without retraining the acoustic model, and a good vocoder can be shared across many voices. **Supervision**: the mel-spectrogram is a clean, low-variance target that is cheap to compute from any audio file, so the acoustic model has a well-behaved thing to regress toward, whereas a raw waveform is high-variance and brutal to predict directly. **Compute**: predicting 86 mel frames per second is far cheaper than predicting 22,050 waveform samples per second, so you put the expensive generative work where it is small (the mel) and the cheap deterministic-ish work where it is big (the vocoder). The genius of VITS, as we will see, is that it found a way to collapse the acoustic model and vocoder back into one model *without* paying the supervision and compute costs that the split was designed to avoid.

### Grapheme-to-phoneme in practice

Let me make the front-end concrete, because it is the stage most newcomers skip and then spend a week debugging when their model mispronounces half their domain vocabulary. The job of G2P is to turn a normalized word into its phoneme string, and there are two ways to do it: a **lookup dictionary** for known words and a **predictive model** for the rest. The CMU Pronouncing Dictionary gives you about 134,000 English words with ARPABET pronunciations; for anything not in it, a small sequence model (or a rule-based fallback) predicts the phonemes from the letters. Here is the standard flow with the `g2p_en` library, which combines a dictionary lookup with a neural fallback.

```python
from g2p_en import G2p

g2p = G2p()
for word in ["fox", "RTX", "2024", "Zotac", "read"]:
    print(f"{word:7s} -> {g2p(word)}")

# fox     -> ['F', 'AA1', 'K', 'S']
# RTX     -> ['AA1', 'R', 'T', 'IY1', 'EH1', 'K', 'S']   # spelled out
# 2024    -> ['T', 'UW1', 'Z', 'IH0', 'R', 'OW0', ...]    # naive: needs norm first
# Zotac   -> ['Z', 'OW1', 'T', 'AE0', 'K']                # neural fallback guesses
# read    -> ['R', 'EH1', 'D']                            # past tense guessed here
```

Two of those outputs are wrong in ways that matter, and they show exactly where TTS pipelines break *before* the neural model ever runs. The "2024" is mishandled because G2P expects *text normalization* to have already expanded it to "twenty twenty four"; feed raw digits and you get garbage. The "read" is rendered as the past-tense "red" when the sentence might mean the present-tense "reed"; G2P alone cannot disambiguate homographs, that needs context the front-end usually does not have. And "Zotac," a brand name absent from the dictionary, is *guessed* by the neural fallback, which may or may not match how the company says it. The fix for all three is the same lesson from the skip-hunting worked example: **harden the front-end**. Normalize numbers and symbols before G2P, add domain words (brands, units, acronyms) to a custom pronunciation dictionary, and disambiguate homographs with part-of-speech tags where you can. A robust acoustic model on top of a sloppy G2P still mispronounces; the phonemes are the contract, and the contract is only as good as the front-end that writes it.

### Computing the mel-spectrogram target

The acoustic model regresses toward a mel-spectrogram, so you need to compute one from your training audio, and the parameters you choose here, the same `n_fft`, `hop_length`, `n_mels` that every TTS config exposes, fix the time and frequency resolution of everything downstream. Here is the canonical extraction with `torchaudio`, matching the LJSpeech-style settings most open TTS models use.

```python
import torchaudio, torch

wav, sr = torchaudio.load("ljspeech_sample.wav")   # (1, num_samples), sr=22050
mel_tf = torchaudio.transforms.MelSpectrogram(
    sample_rate=22050,
    n_fft=1024,        # FFT window -> frequency resolution
    win_length=1024,
    hop_length=256,    # frame stride -> 256/22050 = 11.6 ms per frame -> ~86 fps
    n_mels=80,         # 80 mel bands, the TTS standard
    f_min=0.0,
    f_max=8000.0,      # speech energy is mostly below 8 kHz
    power=1.0,
)
mel = mel_tf(wav)                       # (1, 80, num_frames)
mel = torch.log(torch.clamp(mel, min=1e-5))   # log-mel: the actual training target
print(mel.shape, "frames:", mel.shape[-1], "= duration", wav.shape[-1] / sr, "s")
```

The `hop_length` of 256 at 22,050 Hz is where the "about 86 frames per second" number comes from: $22050 / 256 \approx 86$. The `torch.log` is not cosmetic, the acoustic model regresses log-mel because energy spans many orders of magnitude and the log makes the target distribution well-behaved for an L2 loss, exactly as the log-duration trick does for durations. These parameters are a contract too: your vocoder must be trained on mels computed with the *identical* `n_fft`, `hop_length`, `n_mels`, and `f_max`, or it will hallucinate artifacts. A surprising fraction of "my TTS sounds buzzy" bugs trace to a mel-parameter mismatch between the acoustic model and the vocoder, not to either model itself.

### Why alignment is the central problem

There is one more thing the decomposition hides, and it is the villain of the first act. The acoustic model has to decide **how long each phoneme lasts**. The word "fox" is three phonemes but maybe thirty mel frames; the `AA1` vowel might stretch across twenty of them while the `F` and `S` consonants take five each. There is no fixed ratio. A stressed vowel at the end of a sentence might last 300 milliseconds; the same vowel in fast casual speech might last 60. The mapping from "which phoneme" to "how many frames" is the **alignment**, and every TTS architecture is, at its core, a different answer to the question: *how do we decide the alignment?*

Tacotron answers it with **attention**: a learned, soft, differentiable pointer that, at each output frame, decides which input phonemes to look at. FastSpeech answers it with an **explicit duration predictor**: a small network that says "this phoneme gets seven frames," and a length regulator that simply copies the phoneme's representation that many times. VITS answers it with **monotonic alignment search**, a dynamic-programming routine that finds the single best monotonic alignment during training and a stochastic duration predictor that samples a fresh rhythm at inference. Keep the alignment question in mind. It is the thread that ties all three acts together.

## Act one: Tacotron and Tacotron 2

In 2017, Tacotron showed that you could train a single sequence-to-sequence network to map characters directly to a spectrogram, with no hand-engineered linguistic features, and get speech that was clearly intelligible. A few months later Tacotron 2 refined the recipe, paired it with a WaveNet vocoder, and reached a mean opinion score (MOS) of about 4.53 on the LJSpeech-style internal benchmark in the paper, statistically close to the 4.58 of professionally recorded human speech. That number is the reason every TTS system for the next several years was a variation on Tacotron. It was the first time a neural model sounded, on a good sentence, genuinely human.

The architecture is an **encoder-decoder with attention**, the same family as a neural machine translation model, repurposed for speech. Let me walk through it in the order the data flows.

![A graph diagram of the Tacotron 2 architecture showing an encoder feeding a location-sensitive attention that a decoder LSTM queries to emit one mel frame per step](/imgs/blogs/text-to-speech-from-tacotron-to-vits-2.png)

The **encoder** reads the input characters or phonemes. Each symbol is embedded into a vector, the sequence passes through a stack of convolutions (to give each symbol some local context from its neighbors), and then through a bidirectional LSTM, which lets every position see the whole sentence in both directions. The output is a sequence of encoder states, one per input symbol, each a rich summary of that symbol in context. If the input is "fox," you get three encoder states, one each for the contextualized `F`, `AA1`, and `K`, `S`.

The **decoder** is autoregressive: it produces the mel-spectrogram one frame at a time, left to right, and each step's input includes the frame it produced last step (during training, the *true* previous frame, a trick called teacher forcing). At each step the decoder LSTM has a query, and it needs to decide which encoder states to attend to. That decision is the attention.

The **attention mechanism** is where the magic and the misery both live. At decoder step $t$, the model computes a set of attention weights $\alpha_{t,i}$ over the encoder positions $i$, with $\sum_i \alpha_{t,i} = 1$, and forms a context vector $c_t = \sum_i \alpha_{t,i} h_i$ where $h_i$ is the $i$-th encoder state. The context vector is the "which sound am I making right now" signal, and the decoder uses it, plus its recurrent state, to emit the next mel frame. Tacotron 2 uses **location-sensitive attention**, which adds a crucial inductive bias: the attention at step $t$ is conditioned not just on the content match between the query and each encoder state but also on the *attention weights from the previous step*. This nudges the model to move its focus forward smoothly rather than jumping around, because in speech the alignment is almost always **monotonic**: you say the sounds in order, you do not say the last phoneme before the first.

Finally a small **stop-token** network predicts, at each step, the probability that this is the last frame. At inference the decoder runs until the stop probability crosses a threshold, then halts. There is no other signal telling the model the utterance is over; it has to learn to stop.

### The science of attention alignment, and why it breaks

Here is the heart of the matter, the part worth slowing down for. The attention weights $\alpha_{t,i}$, plotted as a matrix with output frames on one axis and input phonemes on the other, *are* the alignment. In a healthy Tacotron, that matrix is a clean diagonal ridge: early frames attend to early phonemes, later frames to later phonemes, monotonically marching from corner to corner. When you train Tacotron and watch the attention plot, the moment the diagonal "snaps in" is the moment the model starts to produce intelligible speech. Before that, it is noise.

The trouble is that nothing in the architecture *guarantees* the diagonal. The attention is a soft, learned, content-based pointer with only a gentle locality nudge. On clean, in-distribution sentences, the nudge is enough and the diagonal holds. But on hard inputs, the three failure modes appear, and each one is a specific way the diagonal breaks:

- **Skipping**: the attention jumps forward over one or more phonemes without ever putting enough weight on them, so those sounds are never produced. My vanished "24GB" was a skip. It happens on long words, repeated tokens, and unusual character sequences where the content-based match is weak.
- **Repeating**: the attention gets stuck on a phoneme (or jumps *backward*), so the model produces the same sound twice or loops. My "the the the the" was a repeat. It happens when the locality signal is too strong relative to the forward pressure, or on repeated words where the model loses track of which instance it is on.
- **Mumbling and babbling**: the attention diffuses, spreading weight across many phonemes at once so the context vector is an average of several sounds, and the decoder produces a slurry that resembles speech rhythm but no actual words. This is the worst failure because it can run away: once the attention loses the plot, the stop token may never fire, and the model babbles until you cut it off at a max-length limit.

There is a fourth failure that is really a compound of the others but deserves its own name: the **failure to stop**. The stop-token network learns when an utterance is over, but it learns it from the *attention reaching the end of the input*. When the attention loses the plot and starts babbling, it never cleanly reaches the end, so the stop token never fires, and the decoder runs until your max-length cutoff, emitting many seconds of garbage from a short sentence. This is the worst possible failure for a user, a five-word prompt that produces a thirty-second clip of slurred noise, and it is why every production Tacotron deployment hard-caps the decoder steps as a backstop. The cap prevents the catastrophe but does not prevent the glitch; it just bounds how long the glitch lasts.

These are not rare curiosities. On a production stream of arbitrary text, a vanilla Tacotron 2 will produce an audible failure on a meaningful fraction of utterances, often quoted around one in twenty to one in a hundred depending on the domain and how clean your G2P is. For a demo, that is fine. For a system reading ten thousand product descriptions a day, that is a hundred to five hundred broken clips a day, each one a customer hearing a glitch. I spent weeks adding guardrails: forcing the attention to be monotonic at inference, capping the per-phoneme dwell time, adding a windowed attention that only allows the pointer to move forward. They helped. None of them made the problem *go away*, because they were patching a model whose fundamental alignment mechanism was a soft pointer that could, under pressure, point anywhere.

#### Worked example: hunting a skip

Picture the concrete debugging loop, because it is instructive. I have a Tacotron 2 trained on a single speaker, 24 hours of audio, and it produces a clean MOS around 4.3 on held-out sentences. I feed it 5,000 product titles and listen to a sample. Three of the first hundred have audible glitches. I pull the attention matrices for those three. All three show the same pathology: a clean diagonal that, at one point, jumps two phonemes forward in a single step, leaving a gap. In two of the three the gap is on a digit string, "4090" and "2024"; in the third it is on the proper noun "Zotac." The common thread is rare tokens: digits and uncommon names are underrepresented in 24 hours of training audio, so their encoder states are weak, the content match is weak, and the attention's forward momentum carries it right past them. The fix that actually worked was not a model change at all. It was the G2P front-end: spell digits out into words ("forty ninety," "twenty twenty-four") and add a pronunciation dictionary entry for the brand names. With strong, common phonemes in place of rare tokens, the attention had something to lock onto and the skips dropped by an order of magnitude. The lesson stuck with me: in an attention-based TTS, *the alignment is only as robust as the weakest token in the sequence*. That fragility is exactly what the next act removes.

### The guardrails people bolted onto attention

Before the field moved to explicit duration prediction, a great deal of engineering went into making soft attention behave, and the techniques are worth knowing because they reveal exactly what the attention lacks. **Forced monotonic attention at inference**: track the attention's center of mass and forbid it from moving backward, so a repeat becomes impossible; this is a hard constraint applied only at inference and it kills repeats but can cause skips if the model wanted to dwell. **Windowed attention**: restrict the attention at step $t$ to a small window of input positions around where it was at step $t-1$, so it cannot jump far forward (no skip) or far back (no repeat); the window width is a hyperparameter you tune, too narrow and natural pauses break, too wide and the guardrail does nothing. **Guided attention loss during training**: add a loss term that penalizes the attention matrix for deviating from a diagonal, literally pushing the model toward monotonic alignment from the start of training; this speeds up the "diagonal snap-in" and reduces failures but does not eliminate them. **Stepwise monotonic attention** and **forward attention** are architectural variants that bake monotonicity into the attention mechanism itself rather than bolting it on.

Every one of these is a patch that makes the soft pointer *more* like a hard, monotonic, left-to-right alignment, which tells you something: the destination of all this engineering was exactly the alignment that FastSpeech and VITS get by construction. The guardrails were the field discovering, the hard way, that it wanted monotonic alignment all along, and they are why FastSpeech's "just predict the duration" landed as such an obvious-in-hindsight fix. If you find yourself adding the third attention guardrail to a Tacotron, that is the signal to switch architectures rather than keep patching.

### Running and timing a Tacotron-class model

Let me ground this in code. The cleanest way to run a Tacotron-2-class acoustic model with a matched vocoder today is through NVIDIA's released checkpoints, but the most *convenient* end-to-end neural TTS in the 🤗 ecosystem is VITS, which we will use heavily below. For Tacotron specifically, here is the canonical two-stage flow using the published Tacotron 2 plus a HiFi-GAN vocoder, with timing instrumentation so you can compute a real-time factor.

```python
import torch, time, soundfile as sf

device = "cuda" if torch.cuda.is_available() else "cpu"

# Tacotron 2 acoustic model (text -> mel) and a HiFi-GAN vocoder (mel -> wav),
# both via torch.hub from NVIDIA's released checkpoints.
tacotron2 = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub",
                           "nvidia_tacotron2", model_math="fp16").to(device).eval()
hifigan, _, _ = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub",
                               "nvidia_hifigan")
hifigan = hifigan.to(device).eval()

utils = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub",
                       "nvidia_tts_utils")
text = "The quick brown fox jumps over the lazy dog."
seq, lengths = utils.prepare_input_sequence([text])
seq, lengths = seq.to(device), lengths.to(device)

torch.cuda.synchronize() if device == "cuda" else None
t0 = time.perf_counter()
with torch.no_grad():
    mel, mel_lengths, alignments = tacotron2.infer(seq, lengths)  # autoregressive
    audio = hifigan(mel).float()                                  # parallel vocode
torch.cuda.synchronize() if device == "cuda" else None
elapsed = time.perf_counter() - t0

wav = audio[0].cpu().numpy()
sr = 22050
duration = len(wav) / sr
print(f"audio {duration:.2f}s, gen {elapsed:.2f}s, RTF = {elapsed/duration:.3f}")
sf.write("tacotron2_fox.wav", wav, sr)
```

Two things to notice. First, `alignments` is the attention matrix; if you ever debug a glitch, plot it and look for the diagonal. Second, the timing splits cleanly: `tacotron2.infer` is autoregressive, so its cost grows with the *length* of the audio (it runs one decoder step per mel frame), while `hifigan` is fully parallel and fast. On an RTX 4090, a short sentence like this runs at an RTF of roughly 0.1 to 0.2 for the Tacotron stage and well under 0.02 for the vocoder. The autoregressive decoder is the bottleneck, and it is *sequential*: you cannot parallelize across the 86 frames per second because frame $t$ depends on frame $t-1$. That sequential dependency is the second thing FastSpeech attacks.

## Act two: FastSpeech and the duration predictor

FastSpeech, in 2019, made two observations that, once stated, seem obvious. **First**, the reason Tacotron is slow is that it generates frames one at a time; if you knew the alignment in advance, you could generate every frame in parallel. **Second**, the reason Tacotron is fragile is that it relies on a soft attention to find the alignment on the fly; if you predicted the alignment with a dedicated module, the skip-repeat-mumble failures could not happen, because there is no soft pointer to wander. Both observations point to the same fix: **align once, explicitly, then decode in parallel**.

![A before-and-after diagram contrasting autoregressive attention that can skip or repeat words against an explicit duration predictor that forces one monotonic pass](/imgs/blogs/text-to-speech-from-tacotron-to-vits-3.png)

The figure above is the whole idea in one picture. On the left, the autoregressive attention path, with its skip, repeat, and stall failure modes. On the right, the duration-predictor path: a small network predicts, for each phoneme, **how many mel frames it should occupy**, and a **length regulator** simply expands the phoneme sequence by copying each phoneme's representation that many times. If the phoneme `AA1` is assigned a duration of 12 frames, its encoder state is repeated 12 times in a row. After length regulation, the expanded sequence has exactly one entry per output frame, the alignment is *fixed and monotonic by construction*, and a decoder can now map that frame-rate sequence to a mel-spectrogram with no recurrence and no attention over the input at all. Every frame is computed in parallel.

The robustness win falls out for free. A length regulator that copies phoneme states in order *cannot* skip a phoneme (each one gets at least its predicted duration, and you can floor it at 1) and *cannot* repeat out of order (the expansion is strictly left to right). The pathologies that haunted Tacotron are simply not expressible in this architecture. You can break it in other ways, getting a duration badly wrong so a sound is too long or too short, but you cannot get the babbling runaway that an attention model can. In production that distinction is everything: a slightly-too-long vowel is a minor naturalness issue, while a babbling 30-second clip from a 3-second sentence is a catastrophic, customer-facing failure.

### Where do the target durations come from?

The natural question: to *train* a duration predictor you need ground-truth durations, "this phoneme really lasted 12 frames in this recording," and your dataset does not come with those labels. FastSpeech's original answer was to **distill** them from a teacher: train a Tacotron-style attention model first, extract its attention matrix, and read the alignment off it (for each phoneme, count how many output frames attended to it most strongly). This is clever but awkward, because it makes the robust model depend on the fragile one. FastSpeech 2 dropped the distillation and instead took durations from **forced alignment**: run a separate alignment tool (the Montreal Forced Aligner, or an internal one) that, given the audio and the transcript, finds where each phoneme starts and ends. Convert those boundaries to frame counts and you have clean duration targets. Most modern non-autoregressive TTS systems get durations this way, and the quality of your aligner matters: a bad alignment puts a vowel's frames on the wrong phoneme and the model learns the wrong rhythm.

FastSpeech 2 went further and made the model predict not just duration but **pitch** and **energy** as well, the three together called the **variance adaptor**. There is a nice detail in *how* FastSpeech 2 handles pitch that is worth knowing, because it is a small idea with a big quality effect. Rather than predict the raw fundamental-frequency contour (which is noisy and hard to regress), FastSpeech 2 decomposes the pitch into a **continuous wavelet transform**, predicts the wavelet coefficients, and reconstructs the contour, which gives a smoother, easier-to-learn target. Energy is the frame-level magnitude (the L2 norm of each STFT frame), quantized into a few hundred bins and predicted per frame, then added back into the hidden sequence as an embedding. Both pitch and energy are added to the expanded, frame-rate sequence *before* the mel decoder, so the decoder sees, for every output frame, not just "which phoneme" but "at what pitch and what loudness," and those two extra channels of information are what let it predict a sharp mel instead of a blurry average.

The insight here is about a fundamental difficulty in non-autoregressive TTS: speech is **one-to-many**. The same sentence can be said with countless valid prosodies, fast or slow, high or low, emphatic or flat. An autoregressive model handles this implicitly, because each frame is conditioned on the previous frames, so once it commits to a prosody it stays consistent. A parallel model that predicts all frames independently has no such anchor, and if you just regress to the mel-spectrogram with an L2 loss, the model averages over all the valid prosodies and produces a flat, over-smoothed, muffled result, the dreaded **over-smoothing** problem.

The math of why averaging hurts is worth stating because it generalizes far beyond TTS. An L2-trained regressor learns the *conditional mean* of its target: given the input, it outputs the average over all valid outputs. When the target distribution is unimodal and tight, the mean is a fine prediction. But speech prosody is multimodal: a question can rise sharply or rise gently, a phrase can be fast-then-slow or slow-then-fast, and these are genuinely different valid renderings, not noise around one true answer. The average of two valid pitch contours is a contour that is neither, often a flattened compromise that sounds lifeless, and the average of two valid mel-spectrograms is a blurry one with the sharp harmonic structure smeared out, which is exactly what "muffled" sounds like. This is the same reason an L2-trained image model produces blurry images and why diffusion and GANs, which model the full distribution rather than its mean, produce sharp ones. FastSpeech 2's pitch and energy conditioning sidesteps the problem not by modeling the distribution but by *narrowing* it: once you tell the decoder the exact pitch and energy for every frame, there is essentially one mel left to predict, the multimodality collapses, and the conditional mean becomes sharp because the conditional distribution became a spike. VITS attacks the same problem from the other direction, by modeling the full distribution with its flow prior and adversarial decoder so it can produce a *sample* rather than a mean. FastSpeech 2's fix is to feed the model the pitch and energy explicitly so the one-to-many mapping becomes more nearly one-to-one: *given this pitch contour and this energy curve and these durations, there is roughly one mel-spectrogram*, and the model can predict it sharply.

![A graph diagram of the FastSpeech 2 variance adaptor showing a phoneme encoder feeding duration, pitch, and energy predictors into a length regulator and a parallel mel decoder](/imgs/blogs/text-to-speech-from-tacotron-to-vits-4.png)

The figure shows the full FastSpeech 2 forward pass. The phoneme encoder (a stack of self-attention blocks, since there is no recurrence anywhere) produces a representation per phoneme. The variance adaptor predicts duration, pitch, and energy from that representation. The length regulator expands to frame rate using the durations. Then the mel decoder, another stack of self-attention blocks, maps the expanded, pitch-and-energy-augmented sequence to the mel-spectrogram, all frames at once. At training time you use the *ground-truth* durations, pitch, and energy (so the predictors are trained with their own losses but the decoder sees the real values); at inference you use the *predicted* ones. This train-inference split is standard and important: it stops errors in the predictors from compounding during training.

### The science: parallel versus sequential, made quantitative

Let me make the speed argument precise, because it is the cleanest quantitative law in this post. An autoregressive decoder produces $N$ mel frames in $N$ sequential steps; the wall-clock time is $T_\text{AR} \approx N \cdot t_\text{step}$, where $t_\text{step}$ is the time for one decoder step and the steps *cannot* overlap because of the data dependency. A non-autoregressive decoder produces all $N$ frames in a single forward pass whose cost is dominated by the self-attention over $N$ positions; on a GPU with enough parallelism, the wall-clock time is roughly $T_\text{NAR} \approx t_\text{pass}$, nearly independent of $N$ until you saturate the hardware. For a 10-second utterance, $N \approx 860$ frames. Tacotron runs 860 sequential decoder steps; FastSpeech runs one parallel pass. The original FastSpeech paper measured a **mel-spectrogram generation speedup of about 270 times** over an autoregressive Transformer-TTS baseline on a GPU, and an overall end-to-end speedup (including the vocoder) of around 38 times. The exact multiplier depends on hardware and sequence length, but the *shape* of the result is robust: parallel decoding turns a per-frame sequential cost into a near-constant cost, and the longer the utterance, the bigger the win.

There is a subtle cost on the other side of the ledger, and an honest post names it. The autoregressive model's frame-to-frame conditioning gives it a coherence that pure parallel prediction lacks; this is why FastSpeech 2 needs the pitch and energy predictors to claw back naturalness, and why, on the most expressive sentences, a well-trained autoregressive Tacotron can still edge out FastSpeech on a careful MOS test. The trade is the running theme of this whole series: you are buying speed and robustness with a little bit of the prosodic richness that sequential conditioning gives for free. For most products, that trade is a screaming bargain. For a flagship audiobook narrator where every ounce of expressiveness matters, you might think harder, and that is exactly where VITS and the codec-LM frontier come in.

### A duration predictor and length regulator in PyTorch

Here is the core machinery, stripped to its essence, so the idea is concrete rather than abstract. This is the FastSpeech-style duration predictor and length regulator; it is not a full model but it is runnable and it is the part that matters.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DurationPredictor(nn.Module):
    """Predict log-duration (frames per phoneme) from encoder states."""
    def __init__(self, d_model=256, d_hidden=256, kernel=3, dropout=0.1):
        super().__init__()
        pad = (kernel - 1) // 2
        self.conv1 = nn.Conv1d(d_model, d_hidden, kernel, padding=pad)
        self.conv2 = nn.Conv1d(d_hidden, d_hidden, kernel, padding=pad)
        self.norm1 = nn.LayerNorm(d_hidden)
        self.norm2 = nn.LayerNorm(d_hidden)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(d_hidden, 1)  # one log-duration per phoneme

    def forward(self, x):                  # x: (B, T_phon, d_model)
        h = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        h = self.drop(self.norm1(F.relu(h)))
        h = self.conv2(h.transpose(1, 2)).transpose(1, 2)
        h = self.drop(self.norm2(F.relu(h)))
        return self.proj(h).squeeze(-1)    # (B, T_phon) log-durations


def length_regulate(phon_states, durations):
    """Expand each phoneme's state by its (integer) frame count.
    phon_states: (T_phon, d_model)   durations: (T_phon,) ints >= 0
    returns: (T_frames, d_model) where T_frames = durations.sum()
    """
    return phon_states.repeat_interleave(durations, dim=0)


# --- usage at inference: predict log-durations, round, regulate ---
enc = torch.randn(1, 5, 256)                  # 5 phonemes, d_model=256
dp = DurationPredictor(d_model=256)
log_dur = dp(enc)[0]                           # (5,)
dur = torch.clamp(torch.round(torch.exp(log_dur)), min=1).long()
print("frames per phoneme:", dur.tolist())     # e.g. [3, 12, 5, 4, 6]
frames = length_regulate(enc[0], dur)          # (sum(dur), 256)
print("expanded length:", frames.shape[0])     # one row per output mel frame
```

The duration predictor regresses **log-duration**, not raw duration, because durations are positive and span a wide range (a plosive `T` might be 2 frames, a stressed final vowel 30), and the log compresses that range into something an L2 loss handles gracefully. The `repeat_interleave` *is* the length regulator: it copies phoneme state $i$ exactly `durations[i]` times, building a frame-rate sequence whose alignment is monotonic by construction. The `clamp(min=1)` is the small but vital guardrail that guarantees no phoneme is ever assigned zero frames, which is the closest a length-regulated model can come to the "skip" failure, and we forbid it outright. Train this predictor with an L2 loss against forced-alignment durations in log space, and you have the robustness half of FastSpeech in about forty lines.

## Act three: VITS, the end-to-end model

By 2021 the field had a robust, fast acoustic model (FastSpeech 2) and a fast, high-quality vocoder (HiFi-GAN). The obvious next move was to glue them and train end to end, but the obvious move has a trap. If you train an acoustic model to predict mel-spectrograms and a vocoder to invert mels, then chain them, the vocoder sees *predicted* mels at inference but was trained on *real* mels. Predicted mels are slightly over-smoothed and have their own statistical quirks, and the vocoder, trained on a cleaner distribution, amplifies those quirks into artifacts. This **train-inference mismatch** at the mel boundary was a real, audible problem; people fine-tuned vocoders on predicted mels to patch it, but the seam was always there.

**VITS** (Variational Inference with adversarial learning for end-to-end Text-to-Speech) removed the seam by removing the mel entirely. It is a single model that maps text to a waveform, trained end to end, with no intermediate mel-spectrogram target at all. The mel-spectrogram, the explicit acoustic-model output that had organized TTS for years, simply disappears as an interface. Instead VITS learns its *own* latent representation, optimized jointly for the whole task. The result was a model that matched or beat the two-stage Tacotron-plus-vocoder pipeline on naturalness (the paper reports a MOS around 4.43 versus a comparable two-stage system) while being a single artifact you train and serve as one piece, and with a striking bonus: genuinely natural *variation* in rhythm and prosody, because of a design choice we will get to.

![A graph diagram of the VITS model showing a text encoder feeding a normalizing-flow prior and a stochastic duration predictor into a latent that a HiFi-GAN decoder turns into a waveform](/imgs/blogs/text-to-speech-from-tacotron-to-vits-5.png)

VITS has four ideas working together, and the figure shows how they connect. There is a **conditional variational autoencoder** providing the overall probabilistic frame. There is a **normalizing-flow prior** that makes the latent distribution expressive enough to model real speech. There is **monotonic alignment search** plus a **stochastic duration predictor** that handle the alignment without attention failures and inject natural rhythm variation. And there is an **adversarial HiFi-GAN decoder** that turns the latent into a waveform at high fidelity. Let me take them one at a time, because each is a genuinely interesting piece of machinery and together they are the template for a lot of what came after.

### The conditional VAE backbone

At its core VITS is a [conditional variational autoencoder](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch). If the VAE is unfamiliar, the image-generation post derives it from scratch; here is the speech-specific version. A VAE models data $x$ (here, the waveform, or a high-resolution spectrogram of it) through a latent variable $z$, with an encoder $q(z \mid x)$ that maps data to a latent distribution and a decoder $p(x \mid z)$ that maps the latent back to data. You train it by maximizing the **evidence lower bound** (ELBO):

$$\log p(x) \ge \mathbb{E}_{q(z \mid x)}\big[\log p(x \mid z)\big] - D_\text{KL}\big(q(z \mid x) \,\|\, p(z)\big).$$

The first term is **reconstruction**: the decoder should rebuild the data from the latent. The second term is a **regularizer**: the encoder's latent distribution should stay close to a prior $p(z)$. In a vanilla VAE the prior is a standard Gaussian. VITS makes two speech-specific changes. First, it is **conditional**: everything is conditioned on the text $c$, so the prior becomes $p(z \mid c)$, a distribution over latents *given what should be said*. Second, the decoder $p(x \mid z)$ is not a probabilistic image decoder but the **HiFi-GAN waveform generator**, so the reconstruction term, instead of a pixel likelihood, becomes a combination of a mel-spectrogram reconstruction loss and an adversarial loss. The waveform comes out of the same decoder that a GAN vocoder uses, but now it is fed the VAE's learned latent $z$ rather than a mel-spectrogram. That single substitution is what fuses the acoustic model and the vocoder into one network.

### The normalizing-flow prior: why a Gaussian is not enough

Here is the subtlety that makes VITS work, and the reason it links out to the [normalizing-flows](/blog/machine-learning/image-generation/normalizing-flows-and-change-of-variables) post in the image series. The text-conditioned prior $p(z \mid c)$ has to describe the distribution of latents that correspond to a given sentence, and that distribution is *not* a simple Gaussian. The same text can map to many latents (different prosodies again), and the shape of that set in latent space is complicated and multimodal. If you force the prior to be a plain conditioned Gaussian, it is too rigid: the encoder's posterior $q(z \mid x)$, which sees the actual audio, will be richer than the prior can match, the KL term will stay large, and the model will be forced to either ignore detail or push everything into the decoder.

VITS fixes this by putting a **normalizing flow** on the prior. A normalizing flow is an invertible neural network $f$ that transforms a simple distribution into a complex one while tracking exactly how the probability density changes, via the change-of-variables formula:

$$p_z(z) = p_u\big(f^{-1}(z)\big) \, \left| \det \frac{\partial f^{-1}}{\partial z} \right|.$$

In VITS the text encoder produces a simple Gaussian over a base variable; the flow $f$ then warps that Gaussian into a much more expressive distribution that can actually match the audio encoder's posterior. Because the flow is invertible and its Jacobian determinant is tractable (VITS uses affine coupling layers, the same workhorse as the image-side flows), you can compute the exact likelihood and train it inside the ELBO with no approximation. The payoff is concrete: the flow-shaped prior is expressive enough that the latent $z$ can carry fine prosodic and timbral detail, which is a big part of why VITS sounds natural rather than smoothed. This is the single most important place where VITS reuses generative machinery from the image series rather than reinventing it, and it is worth reading that flow post if you want the change-of-variables math in full.

### Monotonic Alignment Search: alignment without attention

VITS still has to solve the alignment problem, decide how the text maps to the latent frames, and it does so without a soft attention and without a separate forced aligner. The trick is **Monotonic Alignment Search** (MAS). During training, VITS has both the text (through the flow-shaped prior) and the audio (through the posterior encoder), so it can *search* for the best alignment directly. It looks for the monotonic, surjective alignment between phonemes and latent frames that **maximizes the likelihood** of the data under the model, and because the alignment is constrained to be monotonic (no going backward, every frame assigned to exactly one phoneme, phonemes consumed in order), this search is a clean dynamic-programming problem, solvable in one pass over an alignment grid much like the forward algorithm in an HMM or the DP behind CTC.

Concretely, the search builds a table over (phoneme index, frame index) pairs and fills it the way the Viterbi or CTC-forward recursion does: the best alignment ending at frame $t$ on phoneme $i$ is the better of two predecessors, staying on the same phoneme (frame $t-1$, phoneme $i$) or having just advanced from the previous one (frame $t-1$, phoneme $i-1$), plus the log-likelihood of frame $t$ under phoneme $i$. Those two allowed transitions, *stay* or *advance*, are precisely what enforce monotonicity: you can never go back to an earlier phoneme, and every phoneme is consumed in order. The whole table is filled in $O(\text{phonemes} \times \text{frames})$ time, a single cheap dynamic-programming pass per training utterance, and backtracking from the final cell recovers the alignment. It is the same algorithmic shape as forced alignment, but the costs come from VITS's own model likelihood rather than an external acoustic model, which is what makes it self-consistent.

This is elegant for two reasons. First, MAS is *guaranteed* monotonic, so the skip-repeat-mumble failures cannot occur, just as with FastSpeech's length regulator, but VITS gets the alignment for free from its own likelihood instead of from an external aligner. Second, once MAS has found the alignment during training, VITS reads off the per-phoneme durations from it and trains a **duration predictor** to reproduce them, so at inference, with no audio to align against, the model can still predict how long each phoneme should last. So MAS plays the role that forced alignment played in FastSpeech 2, but it is internal, jointly optimized, and self-consistent with the rest of the model.

### The stochastic duration predictor: why randomness sounds natural

FastSpeech's duration predictor is deterministic: a given phoneme in a given context always gets the same predicted duration. That is robust, but it has a tell of its own, a slight mechanical regularity in the rhythm, because real human speakers do not say the same sentence with identical timing twice. VITS replaces the deterministic predictor with a **stochastic duration predictor**, and this is the piece that gives VITS its lifelike rhythm.

The stochastic duration predictor is itself a small **normalizing flow** (a flow-based generative model over durations), conditioned on the text. At inference, you sample a noise vector and push it through the flow to get a *sampled* set of durations, so two runs of the same sentence produce subtly different rhythms, one a hair faster here, a touch slower there, exactly the way two takes from a human narrator differ. The science behind why this matters: duration is a genuinely *random* variable in natural speech, with a real conditional distribution given the text, not a single right answer. Modeling it as a distribution and *sampling* from it produces rhythm that lands inside the natural range, whereas predicting the conditional *mean* (what a deterministic L2-trained predictor does) produces rhythm that is always at the boring center of the distribution. Sampling buys you variety and naturalness; the mean buys you a slightly robotic regularity. VITS chose variety, and it is a big part of why the model sounds alive.

There is an honest cost: because the durations are sampled, VITS's output is **non-deterministic** by default. The same text gives a slightly different rendition each call. For most uses that is a feature; for a regression test or a use case that demands bit-exact reproducibility, you fix the random seed, or you swap in a deterministic duration predictor (some VITS variants and the 🤗 `VitsModel` expose exactly this choice via a noise-scale parameter you can set to zero).

### The adversarial decoder and the full loss

The last piece is the **decoder**, which is the HiFi-GAN generator: a stack of transposed convolutions with multi-receptive-field fusion blocks that upsamples the frame-rate latent $z$ all the way to the waveform sample rate. It is trained adversarially against the same discriminators HiFi-GAN uses, the multi-period discriminator (MPD) and the multi-scale discriminator (MSD), which between them catch both the periodic structure of voiced speech and artifacts at multiple time scales. The full VITS training objective is a sum of several terms: the **reconstruction loss** (a mel-spectrogram L1 between the generated and target waveforms, computed by running an STFT on both), the **KL divergence** between the flow-shaped prior and the posterior (the VAE regularizer), the **duration loss** for the stochastic duration predictor, and the **adversarial losses** (the GAN generator and discriminator losses plus a feature-matching loss that stabilizes training). It is a genuinely multi-objective model, and getting the weights to balance is part of the craft of training one, but the payoff is a single network that you feed text and that emits a waveform, with state-of-the-art naturalness and no mel seam.

### Running VITS end to end and measuring RTF

The practical payoff of an end-to-end model is that the code is short. Here is VITS through 🤗 `transformers`, text straight to waveform, timed for a real-time factor.

```python
import torch, time, scipy.io.wavfile
from transformers import VitsModel, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model = VitsModel.from_pretrained("facebook/mms-tts-eng").to(device).eval()
tok = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

text = "The quick brown fox jumps over the lazy dog."
inputs = tok(text, return_tensors="pt").to(device)

# Warm up once so we time steady-state, not kernel compilation.
with torch.no_grad():
    _ = model(**inputs).waveform

torch.cuda.synchronize() if device == "cuda" else None
t0 = time.perf_counter()
with torch.no_grad():
    out = model(**inputs).waveform          # text -> waveform, ONE model
torch.cuda.synchronize() if device == "cuda" else None
elapsed = time.perf_counter() - t0

wav = out[0].cpu().numpy()
sr = model.config.sampling_rate              # 16000 for mms-tts-eng
duration = len(wav) / sr
print(f"audio {duration:.2f}s, gen {elapsed:.3f}s, RTF = {elapsed/duration:.4f}")
scipy.io.wavfile.write("vits_fox.wav", sr, wav)
```

Notice what is *not* there: no separate vocoder load, no mel hand-off, no two-stage timing split. One `from_pretrained`, one forward pass, a waveform. The `model.speaking_rate` and `model.noise_scale` attributes (on the 🤗 VITS config) let you trade rhythm and variation; set `noise_scale` to zero and the stochastic duration predictor collapses to its mean, making the output deterministic at the cost of some naturalness. On an RTX 4090, a sentence like this runs at an RTF well under 0.05, comfortably faster than real time, because the whole model is parallel except for the tiny duration sampling step. To measure RTF *honestly*: always warm up first (the first call pays for kernel compilation and would inflate your number), synchronize the GPU before and after timing (or you measure async launch time, not compute), use a fixed text and seed, and average over several runs.

#### Worked example: when VITS's non-determinism bites

Picture a regression suite for a voice assistant. You synthesize a fixed set of 200 prompts nightly and compare each waveform to a golden reference with a tight numerical tolerance, to catch any model or dependency regression. You upgrade to VITS, and overnight every single test fails. Nothing regressed; the stochastic duration predictor sampled different rhythms, so every waveform differs from its golden by more than the tolerance, even though all 200 sound perfect. The naive fix, comparing waveforms bit-for-bit, is simply the wrong test for a deterministically-non-deterministic model. The right fix is twofold: set `noise_scale=0.0` (and the duration noise scale to zero) to make synthesis deterministic for the regression run, *and* change the assertion from a waveform diff to a perceptual one, transcribe the output with an ASR model and assert the word error rate is below a threshold, plus assert the duration is within a few percent of golden. That perceptual harness is the right test for *any* TTS model, deterministic or not, because what you actually care about is "does it say the right words clearly," not "are the samples bit-identical." VITS just forced the issue. I keep that lesson handy: when you move to a stochastic model, your tests have to become perceptual or they become noise.

## The three architectures, side by side

We now have the three acoustic strategies in hand. Time to put them in one table and be precise about the trade.

![A matrix diagram comparing Tacotron 2, FastSpeech 2, and VITS across autoregression, end-to-end design, naturalness, and robustness](/imgs/blogs/text-to-speech-from-tacotron-to-vits-6.png)

The matrix above captures the headline differences; here it is in prose with numbers attached.

| Model | AR? | End-to-end? | MOS (approx) | RTF (4090, est.) | Robustness | Determinism |
|---|---|---|---|---|---|---|
| Tacotron 2 + HiFi-GAN | Yes (mel) | No (2-stage) | ~4.5 | ~0.1–0.2 | Fragile: skips/repeats | Deterministic |
| FastSpeech 2 + HiFi-GAN | No (parallel) | No (2-stage) | ~4.4 | ~0.02–0.05 | Robust (monotonic) | Deterministic |
| VITS | No (parallel) | Yes (text→wav) | ~4.4 | ~0.03–0.05 | Robust (MAS) | Stochastic by default |

A few readings of this table. The **MOS numbers are all close**, in the low-to-mid 4s, and you should treat the differences as within the noise of a typical MOS study unless they come from a single controlled paper; all three are "good," and which one wins a head-to-head depends heavily on the dataset, the vocoder, and the listening panel. The **robustness column is the one that should drive a production decision**: Tacotron's fragility is not a tuning issue you can fully fix, it is structural, while FastSpeech and VITS are robust by construction. The **RTF column** shows the parallel models clearing real time with huge margin while autoregressive Tacotron is slower and, crucially, gets *relatively* slower on longer utterances because its cost is per-frame-sequential. The **determinism column** is the surprise that bites people: VITS trades reproducibility for naturalness, and you opt back into determinism only by zeroing its noise scales.

If I had to compress the trade into one sentence: **autoregressive attention (Tacotron) buys you the richest implicit prosody at the price of fragility and speed; explicit duration prediction (FastSpeech, VITS) buys you robustness and speed at the price of needing extra machinery (variance predictors, or a flow prior and stochastic durations) to recover naturalness.** The field, almost universally, decided that robustness and speed were worth the extra machinery, and every modern system, VITS and everything after it, predicts duration explicitly rather than discovering it through soft attention.

## Where the field went next

VITS was not the end; it was a hinge. It established three things that the 2023–2026 frontier built on directly. **End-to-end is viable**: you do not need a mel interface, and removing it removes a whole class of train-inference mismatch bugs. **Flows make latents expressive**: the normalizing-flow prior showed that a richer latent distribution is worth the complexity, an idea that recurs in flow-matching TTS. And **stochastic, sampled prosody sounds natural**: modeling rhythm as a distribution you draw from, not a number you predict, became the default expectation for lifelike speech.

![A timeline diagram of the neural TTS lineage from WaveNet through Tacotron and FastSpeech to VITS and the codec-LM and flow-matching frontier](/imgs/blogs/text-to-speech-from-tacotron-to-vits-7.png)

The timeline shows the lineage. From VITS, two roads opened, and the rest of this series walks down both. One road is the **neural codec language model**: instead of predicting a mel or a learned VAE latent, predict the discrete tokens of a [neural audio codec](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound) with a language model, which is exactly what [VALL-E does](/blog/machine-learning/audio-generation/neural-codec-language-model-tts-vall-e). This reframes TTS as in-context learning: feed the model a 3-second sample of a target voice as a prompt, and it continues in that voice, giving [zero-shot voice cloning](/blog/machine-learning/audio-generation/zero-shot-voice-cloning-and-the-tts-frontier) that VITS could not do out of the box. The other road is **flow matching**, where models like F5-TTS and Voicebox apply the same flow-matching objective the image and video series cover to speech, getting non-autoregressive, high-quality synthesis with a small number of sampling steps. Both roads inherit VITS's core lessons; both are covered in their own posts.

It is worth being precise about *what* each road kept from VITS, because the inheritance is not vague. The codec-LM road kept VITS's commitment to discrete-ish, learned representations over the hand-built mel, but pushed it further: where VITS learns a continuous VAE latent decoded by HiFi-GAN, VALL-E predicts the *discrete tokens* of a separate neural codec with an autoregressive language model, which means it can use the entire transformer-language-model toolkit (in-context learning, prompting, scaling laws) for speech. The price is that it brings autoregression *back* at the token level, so it reintroduces some of the latency and a little of the instability that FastSpeech and VITS had banished, which is the central tension of that post. The flow-matching road kept VITS's insight that prosody and fine detail want a *generative* model, not a regressor, but replaced the VAE-plus-flow-plus-GAN stack with a single clean flow-matching objective borrowed straight from the image and video series, getting non-autoregressive generation that is both high quality and few-step. Both roads, in other words, are VITS's two big bets, go generative and go beyond the mel, taken to their separate logical conclusions.

What VITS gave the field, and what every successor kept, is the **architectural confidence to go end to end and to model prosody as something you sample rather than something you average**. The robustness lesson from FastSpeech, predict duration explicitly, and the naturalness lesson from VITS, sample the rhythm, are now so standard that they are invisible. They are just how you build TTS. The skip-repeat-mumble failures that ate my product descriptions in 2019 are, in any modern system, simply gone, and this post is the story of the three ideas that killed them.

## Case studies and real numbers

Let me anchor the architectures in concrete, citable results, with the usual honesty about which numbers are exact and which are order-of-magnitude.

**Tacotron 2 (Shen et al., 2018).** The original paper reports a MOS of $4.53 \pm 0.07$ for Tacotron 2 with a WaveNet vocoder on their internal US-English dataset, against $4.58 \pm 0.05$ for the ground-truth professional recordings, an essentially human-parity result that is the reason the architecture dominated. The catch the paper does not foreground, but that every practitioner hit, is the attention robustness on out-of-domain text; the paper's MOS is on clean, in-domain sentences, and the skip-repeat-mumble rate climbs sharply on digits, acronyms, and long or unusual inputs. Treat the 4.53 as a *ceiling* under favorable conditions, not a production expectation.

**FastSpeech (Ren et al., 2019) and FastSpeech 2 (Ren et al., 2021).** FastSpeech reported roughly a **270 times mel-generation speedup** and about **38 times end-to-end speedup** over an autoregressive Transformer-TTS baseline, with MOS competitive (slightly below the autoregressive teacher on the original distillation-based version). FastSpeech 2, by adding pitch and energy variance prediction and training durations from forced alignment rather than distillation, *closed and in some tests exceeded* the autoregressive MOS while keeping the speed and robustness. The robustness claim is the durable one: because the length regulator is monotonic by construction, the word-skipping and repeating that plague attention TTS do not occur, which is exactly the failure-mode elimination this post is about.

**VITS (Kim et al., 2021).** The VITS paper reports a MOS of about $4.43$ on the single-speaker LJSpeech benchmark, beating its two-stage Tacotron-2-plus-HiFi-GAN comparison and approaching the ground truth, while being a single end-to-end model. The paper's ablations are the interesting part for an engineer: removing the normalizing-flow prior drops quality noticeably (the Gaussian prior is too weak), and replacing the stochastic duration predictor with a deterministic one measurably reduces the naturalness of the rhythm, the two design choices this post emphasized, each shown to matter in the ablation table.

**A practical robustness data point.** On an internal stress test I ran years ago, feeding ten thousand e-commerce product titles (heavy with digits, units, and brand names) through a tuned Tacotron 2, the audible-glitch rate sat around one to two percent even after G2P hardening, a few hundred broken clips. Swapping to a FastSpeech-2-style model with a forced-alignment duration predictor dropped the audible-glitch rate to essentially zero on the same set; the remaining naturalness gap (FastSpeech was very slightly flatter) was invisible to customers but the glitch elimination was night-and-day. That single migration is the most concrete argument I can give for why the field abandoned soft-attention TTS, and the numbers, two percent down to roughly zero, are approximate but the *direction* and *magnitude* are exactly what the architecture predicts.

#### Worked example: budgeting an end-to-end migration

Picture a team running a two-stage Tacotron-2-plus-HiFi-GAN service and deciding whether to move to VITS. The relevant numbers: the two-stage system serves at an RTF around 0.15 (Tacotron dominated), so one A10G GPU at, say, \$1.00 per hour handles maybe 6 to 7 concurrent real-time streams. VITS at an RTF around 0.04 handles roughly 20 to 25 concurrent streams on the same GPU, a 3 to 4 times throughput gain, which at constant load cuts the GPU bill proportionally, from \$1.00 per hour per 6 streams to per 20 streams, roughly a 70% cost reduction on the synthesis tier. The migration cost is real, retrain or fine-tune VITS on your speaker, rebuild the serving path, and fix the determinism-breaks-regression-tests problem from the worked example above, but the operational savings plus the elimination of attention glitches usually pay it back fast. The decision rule I use: if you are serving any meaningful volume and still on autoregressive TTS, the parallel-or-end-to-end migration is almost always positive-ROI, and the only reason to stay autoregressive is a flagship single-voice product where you have squeezed every drop of expressiveness out of the AR model and a listening panel can tell the difference.

## When to reach for each (and when not to)

A decisive section, because the whole point of knowing three architectures is choosing among them.

![A decision-tree diagram routing a TTS need to FastSpeech, VITS, or a codec language model based on whether robustness, single-model serving, or voice cloning matters most](/imgs/blogs/text-to-speech-from-tacotron-to-vits-8.png)

The tree above is my actual decision routine, and here it is spelled out.

**Reach for VITS** when you want one model, trained and served as a single artifact, with state-of-the-art naturalness and natural prosodic variation, for a fixed voice or a modest set of voices you can train on. It is the sweet spot for most single-language, single-or-few-speaker production TTS today: no mel seam, no separate vocoder to maintain, robust alignment, and lifelike rhythm. The 🤗 `VitsModel` and the MMS-TTS checkpoints make it a few lines to run. The only real friction is non-determinism (handle it with noise-scale and perceptual tests) and that adding a *new* voice means training, not prompting.

**Reach for FastSpeech 2** (or a modern non-autoregressive mel model) when you want maximum control over the intermediate and a clean two-stage pipeline you can debug stage by stage, or when you specifically want to swap vocoders, the explicit mel interface is a *feature* for research and for systems where you mix and match components, and the explicit pitch and energy predictors give you direct, interpretable control knobs over prosody. It is also a fine choice when your serving stack already has a battle-tested vocoder you do not want to give up.

**Reach for a codec language model (VALL-E class) or a flow-matching model (F5-TTS class)** when you need **zero-shot voice cloning**, synthesize in a new voice from a few seconds of reference audio with no training, or maximum expressiveness and in-context controllability. This is the frontier, covered in the [VALL-E](/blog/machine-learning/audio-generation/neural-codec-language-model-tts-vall-e) and [voice-cloning](/blog/machine-learning/audio-generation/zero-shot-voice-cloning-and-the-tts-frontier) posts; it costs more compute and (for the AR codec-LM variants) reintroduces some sequential-generation latency and a touch of the old instability, which is why flow-matching and better decoding strategies are active areas.

**Do not reach for Tacotron 2** for a new production system. It is the architecture that taught the field, and its attention mechanism is a beautiful idea, but its skip-repeat-mumble fragility is structural and the parallel models dominate it on speed and robustness at equal naturalness. Run Tacotron to understand attention alignment, to read its attention plots, to feel why the diagonal matters; build your product on something monotonic. **Do not reach for a stochastic-duration model in a context that demands bit-exact reproducibility** without first zeroing the noise scales and moving to perceptual tests, or you will spend a frustrating week debugging "failures" that are just natural variation. And **do not over-index on small MOS differences** between these architectures, they are mostly within study noise, and the robustness, speed, determinism, and cloning axes will drive your real decision far more than a tenth of a MOS point.

## Key takeaways

- **The TTS pipeline decomposes into G2P, an acoustic model, and a vocoder**, and that decomposition survives every architecture: text to phonemes, phonemes to a mel-spectrogram (or a learned latent), latent to waveform. The mel-spectrogram is the classic interface; VITS's contribution was to dissolve it.
- **Alignment is the central problem of TTS**: how many output frames each phoneme occupies. Every architecture is a different answer, soft attention (Tacotron), an explicit duration predictor (FastSpeech), or monotonic alignment search plus a stochastic predictor (VITS).
- **Tacotron's soft attention is powerful but fragile.** The skip, repeat, and mumble failures are structural consequences of a learned soft pointer that can wander on hard inputs; you can patch them but not fully fix them, and they are worst exactly where production text is hardest (digits, acronyms, long sentences).
- **FastSpeech's explicit duration prediction makes TTS fast and robust.** A length regulator that copies phoneme states in order cannot skip or repeat, and predicting all frames in parallel is roughly two orders of magnitude faster than autoregressive decoding; the cost is needing pitch and energy predictors to recover naturalness.
- **VITS goes end to end** by replacing the mel interface with a conditional VAE whose latent is decoded straight to a waveform by a HiFi-GAN generator, removing the train-inference mismatch at the mel boundary and serving as one artifact.
- **VITS's two killer pieces are the normalizing-flow prior** (an expressive latent distribution that a plain Gaussian cannot match) **and the stochastic duration predictor** (rhythm sampled from a distribution rather than predicted as a mean, which is why it sounds alive). Both are validated in the paper's ablations.
- **Stochastic prosody trades reproducibility for naturalness.** VITS is non-deterministic by default; make it deterministic by zeroing the noise scales, and test it perceptually (ASR-based WER, duration tolerance) rather than by waveform diff.
- **The field universally chose explicit duration prediction over soft attention**, and VITS's end-to-end, flow-prior, sampled-prosody template became the foundation that codec-LM TTS (VALL-E) and flow-matching TTS (F5-TTS) built upon. For a new production system, start at VITS or the codec-LM frontier, not at Tacotron.

## Further reading

- **Shen et al., "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions" (Tacotron 2), 2018** — the attention seq2seq acoustic model plus WaveNet vocoder that reached human-parity MOS; the canonical reference for location-sensitive attention in TTS.
- **Wang et al., "Tacotron: Towards End-to-End Speech Synthesis," 2017** — the original character-to-spectrogram seq2seq model that started the neural-TTS era.
- **Ren et al., "FastSpeech: Fast, Robust and Controllable Text to Speech," 2019** and **Ren et al., "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech," 2021** — the non-autoregressive duration-predictor architecture and its variance-adaptor refinement; the source of the parallel-decoding speedup and the robustness-by-construction argument.
- **Kim, Kong, and Son, "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech" (VITS), 2021** — the end-to-end model with the flow prior, monotonic alignment search, and stochastic duration predictor; read the ablations for why each piece earns its place.
- **Kong, Kim, and Bae, "HiFi-GAN," 2020** — the adversarial mel-to-waveform vocoder whose generator and MPD/MSD discriminators VITS reuses as its decoder; covered in depth in the [GAN vocoders post](/blog/machine-learning/audio-generation/gan-vocoders-hifi-gan-and-fast-synthesis).
- **Within this series**: the foundation [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard), the representation post on [waveforms, spectrograms, and perception](/blog/machine-learning/audio-generation/representing-sound-waveforms-spectrograms-and-perception), [conditioning and control](/blog/machine-learning/audio-generation/conditioning-and-control-in-audio-generation), the codec-LM successor [VALL-E](/blog/machine-learning/audio-generation/neural-codec-language-model-tts-vall-e), [zero-shot voice cloning](/blog/machine-learning/audio-generation/zero-shot-voice-cloning-and-the-tts-frontier), [prosody and expressive speech](/blog/machine-learning/audio-generation/prosody-emotion-and-expressive-speech), and the [capstone stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack).
- **Out of series**: the [variational autoencoder](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) and [normalizing-flows](/blog/machine-learning/image-generation/normalizing-flows-and-change-of-variables) posts in the image series for the VAE ELBO and change-of-variables math that VITS's prior reuses, and the 🤗 `transformers` audio docs for the `VitsModel` and `AutoTokenizer` APIs used above.
