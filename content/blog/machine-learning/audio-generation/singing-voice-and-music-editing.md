---
title: "Singing Voice and Music Editing: Synthesis, Conversion, and Stems"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "How to synthesize a singing voice from a score, swap who is singing while keeping the melody, split a mix into stems, and inpaint or extend a track, with the disentanglement that makes all of it possible."
tags:
  [
    "audio-generation",
    "audio-synthesis",
    "singing-voice-synthesis",
    "voice-conversion",
    "stem-separation",
    "music-generation",
    "music-editing",
    "generative-ai",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Audio Generation"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/singing-voice-and-music-editing-1.png"
---

The first time I heard a convincing "AI cover," it was a famous pop singer's voice delivering a song they had never recorded, in tune, with their characteristic rasp on the held notes, over the original backing track. My first reaction was the same as everyone's: how. My second reaction, the engineer's reaction, was more useful: that recording was not generated from a text prompt. Somebody started with a real recording of a *different* singer performing that song, and a machine pulled the performance apart, kept the melody and the words, threw away the original singer's voice, and rebuilt the audio in the target voice. Nothing about the notes changed. Only *who* was singing changed. That is a fundamentally different operation from the text-to-song systems we have spent most of this series on, and it sits at the heart of a whole family of techniques: **editing** audio that already exists, rather than conjuring it from nothing.

This post is about that family. Most of generative audio, including everything from [MusicGen](/blog/machine-learning/audio-generation/conditioning-and-control-in-audio-generation) to [Stable Audio](/blog/machine-learning/audio-generation/diffusion-for-audio), starts from a prompt and a noise seed and runs forward. Editing starts from a *signal*. You hand the model a real recording and ask it to change exactly one thing while leaving everything else intact: change who is singing but keep the song (singing voice **conversion**); create a singer from lyrics and sheet music (singing voice **synthesis**); pull a finished mix apart into its instruments so you can remix or make a karaoke track (**stem separation**); or extend a clip, fill a gap, or restyle a section (music **inpainting** and **continuation**). The reason these belong together is that they all rely on the same trick: factoring a sound into the parts a listener perceives separately, so you can hold most of them fixed and swap one.

![A vertical stack diagram showing four ways to edit existing audio: singing synthesis from a score, singing conversion that swaps the singer, stem separation into vocals drums and bass, and music inpainting, all resting on shared content pitch and timbre machinery with a consent caution band](/imgs/blogs/singing-voice-and-music-editing-1.png)

The figure above is the map for the whole post. Four operations sit on top of one shared idea, the factoring of a signal into **content** (the phonemes, the words), **pitch** (the F0 contour, the melody), and **timbre** (the voice or instrument that colors it). Singing synthesis builds a voice from a score. Conversion swaps timbre while holding content and pitch fixed. Separation inverts a mix back into its sources. Inpainting and continuation fill or extend audio conditioned on the audio around it. By the end of this post you will be able to run a stem separation with `demucs`, sketch a conversion pipeline that wires a content encoder, an F0 extractor, and a target-singer embedding into a pitch-conditioned vocoder, and call a music-continuation API; you will also understand *why* each one works at the signal level, and the consent and intellectual-property questions that the conversion case forced the whole field to confront. We will preview the [voice-safety and watermarking post](/blog/machine-learning/audio-generation/audio-deepfakes-watermarking-and-voice-safety) for the ethics and return to the series spine, the audio stack and its tension of **fidelity, controllability, speed, and length**, at every turn. If you have not read [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard), the short version is that audio is a 1D signal at tens of thousands of samples per second where the ear catches artifacts the eye would never notice, and editing inherits every bit of that difficulty.

## Why editing is a different problem from generation

It is worth being precise about the difference, because it changes the engineering.

When you generate from a prompt, you have total freedom. The model can choose any plausible waveform that satisfies the prompt; there is no ground truth it must match sample for sample. That freedom is forgiving. If a text-to-music model renders a slightly different chord voicing than another run, both are correct. Evaluation is loose precisely because there is no reference.

Editing removes that freedom along one or more axes. When you convert a singer, the output is *wrong* if the melody drifts, even slightly, because the listener has the original melody in their head and the new vocal must lock to it. When you separate stems, the output is wrong if drums leak into the vocal track, because the goal is defined by the original mix you are trying to invert. When you inpaint a four-second gap, the fill is wrong if it does not match the tempo, key, and timbre of the audio on *both* sides of the hole. Editing is generation under a hard constraint that part of the output is dictated by an existing signal. That constraint is what makes the problem both more useful and, in a specific sense, harder: you cannot hide a mistake behind "well, that is a valid alternative," because there is a reference and the ear will hear the seam.

The unifying solution across all four tasks is **disentanglement**: represent the signal as a set of factors that vary independently, so an edit becomes "change factor X, keep the others." For a singing voice the natural factors are content, pitch, and timbre. For a mix the natural factors are the source instruments. The whole game is finding representations where those factors actually *are* separable, because in the raw waveform they are hopelessly entangled, every sample carries all of them at once.

There is a clean way to see *why* the raw waveform is the wrong place to edit. A single audio sample, a number between $-1$ and $1$ at one instant of time, tells you the instantaneous air pressure and nothing else. Pitch is not in any one sample; it is in the *rate* at which a pattern repeats across hundreds of samples. Timbre is not in any one sample either; it is in the *shape* of the spectrum across a window. Content, the phonemes, lives in how that spectral shape *changes* over tens of milliseconds. So all three factors are properties of *patterns over time*, not of individual samples, and they are superimposed in the same numbers. To edit one factor you must first move to a representation where that factor becomes an explicit, addressable quantity, an F0 value per frame, a speaker vector, a phoneme sequence, rather than an emergent property of a million entangled samples. Every system in this post is, at bottom, a machine for producing such a representation and then resynthesizing from it. That is the deep reason editing and codecs and self-supervised encoders all show up together: they are all in the business of turning a waveform into separable factors.

A second framing helps here, borrowed from the way we think about the [VAE as the image codec](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch). An editable representation is one where the axes *mean something*. A raw waveform's "axes" (the samples) are meaningless to edit individually, nudging sample 40,012 does nothing perceptible. A good factored representation has axes a human would name: this axis is the singer, that one is the note, those are the words. The research history of this whole area is the search for encoders whose axes line up with the factors people actually want to change. When they line up well, editing is a one-line swap; when they leak into each other, editing produces artifacts. Disentanglement quality is editing quality, restated.

#### Worked example: why you cannot just "lower the pitch" of a recording

Suppose a singer recorded a song a whole step too high and you want it down two semitones without changing who they sound like. The naive fix, resample the audio to play slower, lowers the pitch but also lowers the *formants*, the resonant peaks of the vocal tract, so the singer now sounds physically larger, the "chipmunk in reverse" effect. A proper pitch shift has to move the harmonic series (the pitch, set by vocal-fold rate) while holding the spectral envelope (the formants, set by vocal-tract shape) fixed. That single example is the entire thesis of this post in miniature: pitch and timbre are *different factors* of the same signal, and useful audio editing depends on separating them. Classic DSP does this with a phase vocoder; modern systems do it by learning representations where pitch (an F0 number per frame) and timbre (a speaker embedding) are literally different tensors. A typical phase-vocoder pitch shift of two semitones is essentially transparent; a learned conversion can do the same shift *and* change the singer, which DSP cannot.

## Singing voice synthesis: hitting the notes on purpose

Start with the "from scratch" case, because it sharpens what makes singing special. **Singing voice synthesis** (SVS) takes *lyrics* plus a *musical score*, the notes with their pitches and durations, and produces a sung performance. It is the singing cousin of [text-to-speech](/blog/machine-learning/audio-generation/text-to-speech-from-tacotron-to-vits), and at first glance you might think you can reuse a TTS model. You cannot, and the reason is illuminating.

In TTS, prosody is *free*. The model decides the pitch contour, the rhythm, the emphasis, and as long as the result sounds natural and intelligible, any reasonable choice is correct. There is no externally specified melody. Singing is the opposite. The score tells you exactly which pitch to sing on each syllable and exactly how long to hold it. If the score says a held A4 (440 Hz) for two beats, the model must produce an F0 that sits at 440 Hz, with the small expressive deviations a real singer adds, vibrato, a slight scoop into the note, a release at the end, but *anchored* to 440 Hz. A TTS model that invents its own pitch contour would sing the wrong tune. So the central technical fact of SVS is this: **it is conditioned on exact F0 targets, and it must respect them.** Free-prosody generation becomes constrained-prosody generation.

![A graph showing score-conditioned singing synthesis where lyrics and a musical score feed a phoneme aligner that maps notes to frames, then a diffusion acoustic model produces a mel and F0 with prescribed pitch, which a neural-source-filter vocoder turns into an in-tune singing waveform with vibrato](/imgs/blogs/singing-voice-and-music-editing-6.png)

The figure shows the canonical SVS pipeline, which is what systems like **DiffSinger** implement. Walk through it. The lyrics become a phoneme string. The score, a sequence of notes with MIDI pitch and duration, gets aligned to those phonemes so each phoneme knows its target pitch and how many frames it spans (a held note stretches one vowel over many frames). That aligned representation feeds an **acoustic model** that predicts a mel-spectrogram and an F0 curve, conditioned on the note pitches so the predicted F0 tracks the score. Finally a **vocoder** turns the mel and F0 into a waveform.

Two design choices in that pipeline carry the weight. First, the acoustic model is where DiffSinger earned its name: it uses a **diffusion** process to generate the mel-spectrogram, the same denoising idea from the [image-diffusion post](/blog/machine-learning/image-generation/diffusion-from-first-principles), refining noise into a mel over a number of steps. Diffusion mattered here because earlier SVS models, which regressed the mel directly, produced over-smoothed spectrograms, the predict-the-mean problem, and over-smoothing in singing sounds like a flat, lifeless voice with no breathiness or texture. Diffusion samples from the distribution instead of averaging it, restoring the high-frequency detail the ear reads as a real voice. Second, the vocoder is usually a **neural source-filter (NSF)** vocoder or an NSF-HiFiGAN hybrid, which takes the F0 explicitly as input and uses it to drive a harmonic excitation signal. That explicit F0 input is what keeps the synthesized voice locked to the score; a vocoder that ignored F0 and tried to infer pitch from the mel alone would wander off the note on long sustains.

### Why the F0 conditioning has to be explicit

Here is the science of why singing forces explicit F0 where speech can get away without it. The fundamental frequency of a voiced sound is set by how fast the vocal folds vibrate; the harmonics sit at integer multiples of F0. On a sustained sung note of two seconds, that is a long stretch where the harmonic structure must stay precisely at, say, 440 Hz and its multiples (880, 1320, ...). A mel-spectrogram, by design, has *coarse* frequency resolution in its lower bands relative to the precision a tuned note demands, and it discards phase entirely. If you ask a vocoder to reconstruct pitch purely from a mel, small errors in the predicted mel translate into audible pitch drift, the note goes flat or sharp over the sustain, which on a melody is a glaring error. By feeding F0 directly to the vocoder as a separate, high-precision channel, you decouple "what frequencies, roughly, and what timbre" (the mel) from "exactly what pitch" (the F0). The mel handles timbre and articulation; the F0 handles tuning. This is the same disentanglement theme, surfacing inside the synthesis pipeline itself.

#### Worked example: vibrato is not noise, it is signal

A real singer's vibrato is roughly a 5 to 7 Hz oscillation of F0 with a depth of half a semitone to a full semitone, often growing on longer notes. Early SVS that predicted a flat F0 sounded robotic precisely because it lacked this. So good SVS models either predict the vibrato as part of the F0 curve or add a parametric vibrato whose rate and depth can be controlled. Concretely: a note nominally at A4 (440 Hz) with a 6 Hz vibrato of one-semitone depth oscillates roughly between 427 and 453 Hz six times per second. Get the rate wrong (say 10 Hz) and it sounds like a nervous warble; get the depth too large and it sounds out of tune. The point for an engineer: the F0 channel is not just "the note," it carries the expressive micro-structure, and the quality difference between a passable SVS and a moving one lives largely in that channel. This is also why SVS evaluation leans on **F0 RMSE** (how far the realized pitch sits from the target) and voicing-decision error, metrics we will return to; a model can have a clean timbre and still fail if its pitch tracking is loose.

A minimal sketch of the score-conditioned forward pass, in PyTorch-flavored pseudocode that mirrors how DiffSinger-style repos are wired, makes the conditioning concrete.

```python
import torch

# Inputs already prepared by the front-end:
#   phonemes:  LongTensor [T_frames]  -- phoneme id per frame (expanded by duration)
#   f0:        FloatTensor [T_frames] -- target pitch in Hz, 0.0 where unvoiced
#   note_pitch:LongTensor [T_frames]  -- MIDI note per frame from the score
# The acoustic model predicts a mel; it is CONDITIONED on the note pitch so the
# predicted F0 tracks the score rather than being invented.

mel = acoustic_model(
    phonemes=phonemes,
    note_pitch=note_pitch,     # score conditioning -- the crucial extra input vs TTS
    f0=f0,                     # target pitch curve (incl. vibrato)
    speaker_id=torch.tensor([3]),   # which singer's timbre
    diffusion_steps=50,        # denoising steps for the mel
)

# The vocoder takes the mel AND the explicit F0 so long sustains stay in tune.
wav = nsf_hifigan(mel=mel, f0=f0)          # FloatTensor [T_samples] at 44.1 kHz
torchaudio_save = __import__("torchaudio").save
torchaudio_save("sung.wav", wav.unsqueeze(0).cpu(), 44100)
```

The thing to notice is `note_pitch` and `f0` flowing in as first-class inputs. That is the entire difference from a TTS forward pass, where the model would *predict* a duration and a pitch contour instead of *receiving* them. SVS is generation on rails the composer laid down.

## Singing voice conversion: keep the song, change the singer

Now the case that started this post. **Singing voice conversion** (SVC) takes an existing sung recording and changes *who* is singing while preserving the melody and the lyrics. This is the "AI cover" engine, and the open-source systems that made it a phenomenon, **so-vits-svc** and **RVC** (Retrieval-based Voice Conversion), are built almost exactly around the disentanglement picture we keep returning to.

![A graph of singing voice conversion where source singing flows into a content encoder using HuBERT or ContentVec and into an F0 extractor using RMVPE or CREPE, while a target singer embedding joins both at a pitch-conditioned NSF vocoder decoder that produces converted singing with the same song but a new voice](/imgs/blogs/singing-voice-and-music-editing-2.png)

Read the figure as a recipe. Take the source singing, the recording of singer A performing the song, and run it through two analyzers in parallel. The first is a **content encoder**, typically a self-supervised speech model like **HuBERT** or, better for this task, **ContentVec**, which produces a frame-by-frame representation of *what is being said*, the phonetic content, with as little of the speaker's identity as possible. The second is an **F0 extractor** like **RMVPE** or **CREPE** that pulls out the pitch contour, *what melody is being sung*. Those two streams, content and pitch, capture the parts of the performance you want to keep. The part you want to discard is singer A's timbre, and you replace it with a **target singer embedding** for singer B, learned during training on B's voice. A **pitch-conditioned decoder**, the same NSF-style vocoder family, takes content plus F0 plus the target embedding and resynthesizes a waveform: the same words, the same melody, in B's voice.

That is it. The cleanliness of the conversion is exactly the cleanliness of the disentanglement. If the content encoder leaks singer A's timbre into its output, you will hear a ghost of A in the result, a "leaked source" artifact. If the F0 extractor stumbles on a breathy passage and reports an octave error, the converted note jumps an octave. The whole quality of an SVC system can be predicted from how well it separates these factors, which is why the field's progress reads as a history of better encoders: ContentVec was adopted over plain HuBERT precisely because it was trained to be more speaker-*invariant*, leaking less identity, giving a cleaner content stream to convert.

![A before and after comparison showing the original recording with an F0 contour the phonemes and singer A bright nasal formants, versus the converted recording keeping the same F0 contour the same phonemes but singer B dark rounded formants](/imgs/blogs/singing-voice-and-music-editing-4.png)

The before-and-after framing in this figure is the cleanest way to hold what conversion does and does not touch. Two of the three factors are *invariant*: the F0 contour (the melody) and the phonemes (the lyrics) are carried through unchanged. Only the third factor, the **formant structure** that the ear reads as a particular voice, is replaced. Singer A's bright, nasal formants become singer B's darker, rounder ones, and because formants live in the spectral envelope rather than the harmonic positions, you can swap them without disturbing the pitch. This is the visual statement of "content and pitch preserved, timbre swapped."

### The science: why a pitch-conditioned decoder is non-negotiable

It is tempting to think of SVC as "extract content, run it through a target-voice TTS." That fails, and the failure teaches the principle. A TTS decoder would *invent* a pitch contour appropriate to speech, throwing away the melody. SVC must instead *condition the decoder on the source F0* so the realized pitch follows the original singing. So the decoder takes three conditioning streams, content (what), pitch (which note), and speaker (whose voice), and only the speaker stream is swapped at conversion time. Formally, if we write the synthesis as a function $\hat{x} = D(c, f_0, s)$ where $c$ is content, $f_0$ is the pitch curve, and $s$ is the speaker embedding, then conversion is simply

$$
\hat{x}_{A \to B} = D\big(c(x_A),\; f_0(x_A),\; s_B\big),
$$

holding the content and pitch of source A fixed and substituting B's embedding. The art is making $c(\cdot)$ and $f_0(\cdot)$ as speaker-independent as possible, so that nothing about A survives except the two factors you meant to keep. When $c$ leaks speaker identity, the equation is "lying": the output secretly still depends on A through $c$, and you hear it.

There is one practical subtlety the equation hides. The source and target singers may have different pitch *ranges*, a bass converting to a soprano part. So real systems include a **key shift**: before feeding $f_0(x_A)$ to the decoder, transpose it by a number of semitones into the target's comfortable range. RVC exposes this as a transpose parameter, and getting it wrong, leaving a bass's F0 unshifted while using a soprano embedding, produces a strained, unnatural result because the timbre and the pitch register disagree about who is singing. The transpose is multiplicative in frequency, $f_0 \mapsto f_0 \cdot 2^{n/12}$ for $n$ semitones, which is the line `f0 = f0 * 2 ** (key_shift / 12)` you will see in the code below; an octave up is $n = 12$, doubling every pitch.

### Why a self-supervised content encoder, specifically

It is worth dwelling on the content encoder, because its choice is the single biggest lever on conversion quality and it is the least obvious part. Why use a self-supervised speech model like HuBERT or ContentVec at all, rather than, say, an ASR model that outputs phonemes directly?

The reason is granularity and robustness. An ASR model outputs *discrete text*, which throws away everything about *how* a phoneme was sung, the exact articulation, the coarticulation between neighboring sounds, the timing within a syllable. Resynthesizing from text alone would lose the performance. A self-supervised encoder instead outputs a *continuous frame-level feature* (typically one vector every 20 milliseconds) that captures the phonetic content *and* its fine articulation while having been trained, through masked prediction on huge unlabeled speech corpora, to be a good general representation of "what is being said." HuBERT is trained to predict masked cluster assignments; ContentVec adds an explicit objective to make those features *invariant to the speaker*, by training so that the same utterance from different voices maps to similar features. That speaker-invariance is exactly the property conversion needs: you want a content stream that says "the singer is on the vowel /a/, mid-articulation" without also smuggling in "and the singer has a bright nasal timbre," because the timbre is what you are about to replace. When the content features still carry speaker information, the decoder receives contradictory instructions, the target embedding says "singer B" while the content secretly says "singer A", and the contradiction surfaces as the leaked-source artifact. The progression from HuBERT to ContentVec in the SVC community is precisely the field buying more speaker-invariance to reduce that leak.

There is a subtle trade here, the same kind that runs through the whole series as fidelity versus controllability. A *more* speaker-invariant content encoder gives cleaner identity swaps but can also strip away some articulatory detail that made the source performance expressive, because aggressively removing speaker information sometimes removes correlated content nuance with it. Push invariance too hard and conversions sound generic; too little and they leak. The sweet spot is an encoder that removes timbre while keeping articulation, and finding it is an empirical, per-encoder matter.

A pipeline sketch shows the wiring without hiding the moving parts. This mirrors the structure of an RVC or so-vits-svc inference path.

```python
import torch, torchaudio

# 1. Load the source singing (resample to the content encoder's rate, 16 kHz).
wav, sr = torchaudio.load("source_singing.wav")
wav16 = torchaudio.functional.resample(wav.mean(0, keepdim=True), sr, 16000)

# 2. Content: a self-supervised encoder, frame-level phonetic features.
#    ContentVec is preferred over plain HuBERT for being speaker-invariant.
content = content_encoder(wav16)          # [T, D_content], the WHAT, speaker-stripped

# 3. Pitch: an F0 extractor. RMVPE is robust on singing with accompaniment bleed.
f0 = extract_f0_rmvpe(wav16, sr=16000)    # [T], Hz, 0 where unvoiced
f0 = f0 * 2 ** (key_shift_semitones / 12) # transpose into the target's range

# 4. Identity: the target singer's learned embedding (the ONLY thing we swap).
spk_emb = target_speaker_table["singer_B"]   # [D_spk]

# 5. Resynthesize: a pitch-conditioned NSF vocoder consumes all three streams.
wav_out = nsf_decoder(content=content, f0=f0, spk=spk_emb)   # [T_samples] @ 40 kHz
torchaudio.save("converted_to_B.wav", wav_out.unsqueeze(0).cpu(), 40000)
```

Every line maps to a factor: `content` is what, `f0` is which note, `spk_emb` is whose voice, and only the last is the conversion. RVC adds one more trick worth naming, a **retrieval** step (the R in its name): it keeps an index of the target singer's training feature vectors and, at inference, nudges each frame's content feature toward its nearest neighbors in that index. This reduces timbre leakage further and sharpens articulation, at the cost of sometimes over-smoothing unusual phonemes. It is a small, pragmatic patch on the disentanglement, buying cleaner identity by anchoring content to the target's observed distribution.

### The F0 extractor is the other half of the battle

If the content encoder owns the "what," the **F0 extractor** owns the "which note," and a conversion is only as good as its pitch track. This is worth its own attention because pitch extraction on *singing* is genuinely harder than on speech, and the failure modes are specific.

The core difficulty is the **octave error**. F0 estimation works by finding the period of the quasi-periodic vocal signal, the time after which the waveform roughly repeats, and inverting it. The trap is that a signal repeating with period $T$ *also* repeats with period $2T$ (every other cycle looks the same), so a naive estimator can confidently report a pitch one octave too low (or, by the mirror error, one octave too high). On singing this happens most at the *onset and release* of notes, in breathy passages, and over instrumental bleed, exactly the moments where the periodicity is weakest. An octave error in a conversion is catastrophic: a single note jumps up or down by twelve semitones, an unmistakable glitch.

This is why the field moved from autocorrelation-style estimators to learned ones. **CREPE** is a convolutional network trained to classify pitch directly from the waveform, far more robust to octave errors than classical methods. **RMVPE** (Robust Model for Vocal Pitch Estimation) goes further: it is trained specifically to extract a *vocal* pitch track even in the presence of *accompaniment*, which is precisely the SVC setting where you may not have a perfectly isolated vocal. The practical guidance the community converged on, and that I would follow, is RMVPE as the default for singing conversion, CREPE as a strong alternative, and the classical estimators only when speed dominates and the input is a clean solo voice.

There is one more piece: **voiced/unvoiced decision**. Consonants like /s/ and /f/ have no pitch, they are noise, and the F0 track must report "unvoiced" (conventionally $f_0 = 0$) there rather than hallucinating a pitch. If the extractor marks an unvoiced consonant as voiced, the decoder will try to sing the consonant, producing a buzzy artifact; if it marks a voiced vowel as unvoiced, the note drops out. Getting the voicing decision right is half of what separates a clean conversion from a noisy one, and it is why **voicing decision error** sits alongside F0 RMSE in the evaluation.

```python
import numpy as np

def extract_f0_rmvpe(wav16, sr=16000):
    # Sketch of the interface; the real RMVPE model returns a per-frame pitch.
    f0 = rmvpe_model.infer_from_audio(wav16, sample_rate=sr)   # [T], Hz
    # Median-filter a few frames to suppress isolated octave-error spikes.
    f0 = median_filter(f0, kernel=3)
    # Unvoiced frames (consonants, silence) must stay at 0.0, not a guessed pitch.
    f0[f0 < 50.0] = 0.0          # 50 Hz floor: below any sung note
    return f0.astype(np.float32)
```

The median filter on the pitch track is a small, telling detail: it cheaply kills the isolated single-frame octave spikes that would otherwise produce audible clicks, without smearing the genuine pitch movement of a vibrato (which spans many frames). Little robustness patches like this are most of the distance between a research demo and a tool people use.

#### Worked example: a three-second-of-noise speaker prompt

A recurring failure: someone tries to clone a voice from a three-second clip recorded on a phone in a cafe. The content and F0 paths are fine, they come from a clean source recording, but the *target* embedding is estimated from three noisy seconds, so $s_B$ is a poor estimate of B's true timbre. The result sounds like B "through a wall," with the cafe's room tone baked into the voice, because the embedding absorbed the noise as if it were part of B's identity. The fix is not a better decoder; it is more and cleaner reference audio for the target, ideally tens of seconds to minutes of clean solo singing. This is the same lesson as in [zero-shot voice cloning](/blog/machine-learning/audio-generation/zero-shot-voice-cloning-and-the-tts-frontier): the speaker representation is only as good as the reference, and garbage reference makes garbage identity no matter how good the rest of the stack is.

### The consent and IP problem this created

I have described SVC purely mechanically, and mechanically it is a beautiful piece of engineering. But the "AI cover" phenomenon it enabled created a genuine harm that the field cannot wave away, and an engineer building this should understand it before shipping anything. The same pipeline that lets a hobbyist make their own voice sing a song they wrote lets anyone make a *specific real artist's* voice sing words that artist never sang, with no consent, monetize it, or use it to deceive. A singer's voice is, for many of them, their livelihood and their identity, and a high-quality conversion appropriates exactly that.

This is not a hypothetical. The wave of viral AI covers in 2023 led to takedowns, label statements, and an active policy debate about whether a voice is a protected attribute. The technical community's response, which we cover in depth in the [audio deepfakes, watermarking, and voice-safety post](/blog/machine-learning/audio-generation/audio-deepfakes-watermarking-and-voice-safety), centers on three things: **consent** (only clone voices you have permission to clone), **provenance** (watermark generated audio so it can be identified downstream, as AudioSeal and SynthID-audio do), and **detection** (classifiers that flag synthetic singing). My own rule, and I would make it yours: treat a target voice the way you would treat a copyrighted sample. If you do not have the right to use it, do not, regardless of what the tool will technically let you do. The capability is real and useful for consented and original voices; the same capability is a serious harm when pointed at a non-consenting person.

## Stem separation: the inverse problem that makes editing possible

Step back from voices to the whole mix. Almost everything you might want to *do* to a finished song, make a karaoke track, sample the drums, re-sing over the instrumental, remaster the bass, requires first pulling the mix apart into its component sources. That is **stem separation** (also called source separation or music demixing), and it is the quiet enabler underneath most music editing. Without it, a finished stereo file is a single tangled signal; with it, you have independent vocals, drums, bass, and "other" tracks you can edit in isolation.

![A vertical stack diagram showing stem separation splitting a mixed stereo track into a vocals stem with lead and harmony, a drums stem with kick snare and hats, a bass stem, and an other stem with guitars keys and pads, enabling karaoke and remix](/imgs/blogs/singing-voice-and-music-editing-3.png)

The figure shows the standard four-stem decomposition that **Demucs**, **Spleeter**, and the **MDX** family of models all target: a mix splits into vocals, drums, bass, and other. (Some models offer finer splits, separating guitar and piano from "other," but the four-stem layout is the workhorse because it matches how mixes are built and how people want to edit them.) Once you have the stems, karaoke is "mute the vocals," a remix is "keep the drums and bass, replace the rest," and re-instrumentation is "keep the vocals, regenerate the backing," each of which is now trivial because the separation already did the hard part.

### The science: separation as masking in a time-frequency domain

Why is this even possible? Two instruments playing at once sum into one waveform; how can a model un-sum them? The key insight is that, in a **time-frequency representation** like the [STFT spectrogram](/blog/machine-learning/audio-generation/representing-sound-waveforms-spectrograms-and-perception), different sources tend to occupy *different* time-frequency cells, or at least dominate different ones. A kick drum is a brief, broadband, low-frequency burst; a vocal is a sustained harmonic stack in the mid-range; a hi-hat is a short high-frequency hiss. At any single (time, frequency) cell, usually *one* source dominates the energy. So separation can be cast as **masking**: for each source, predict a mask, a value between 0 and 1, for every cell of the mixture spectrogram, multiply the mixture by the mask, and you recover that source.

Formally, if $X(t, f)$ is the complex STFT of the mixture and we want source $i$, we learn a mask $M_i(t, f)$ and estimate

$$
\hat{S}_i(t, f) = M_i(t, f) \cdot X(t, f),
$$

then invert the STFT to get the waveform. The classic training target is the **ideal ratio mask**, $M_i = |S_i| / \sum_j |S_j|$, the fraction of the cell's magnitude that belongs to source $i$. A network is trained to predict these masks from the mixture. Spleeter is essentially this: a U-Net that predicts magnitude masks on the spectrogram. The limitation is phase, the mask is applied to magnitude, and the phase is borrowed from the mixture, which causes artifacts when sources overlap heavily, the same phase problem that haunts all spectrogram-domain audio work.

There is a subtlety in the masking math worth making explicit, because it is the seed of the phase problem. The mixture STFT $X(t,f)$ is *complex*: it has a magnitude and a phase. A magnitude mask $M_i(t,f) \in [0,1]$ scales only the magnitude and, by construction, reuses the mixture's phase for the recovered source. When one source clearly dominates a cell, the mixture's phase is approximately that source's phase, so reusing it is fine. But when two sources have comparable energy in a cell, the mixture phase is a vector sum of their phases and matches *neither*; applying a real-valued mask then produces a magnitude that is right but a phase that is wrong, and the inverse STFT renders that as the characteristic "watery" or "phasey" smear. This is the same phase problem that motivated [Griffin-Lim and neural vocoders](/blog/machine-learning/audio-generation/representing-sound-waveforms-spectrograms-and-perception) elsewhere in the series, here it caps how clean a magnitude-mask separator can ever be. One response is the **complex ideal ratio mask**, a complex-valued mask that also corrects phase; another, more thorough, is to leave the magnitude-only domain entirely.

That is exactly what the modern systems do, in two ways. **Demucs** (in its current "hybrid" form, HT-Demucs) works in *both* the spectrogram domain and the *waveform* domain at once, with a U-Net that has a spectrogram branch and a time-domain branch whose outputs are combined. Operating partly in the waveform domain lets it model phase implicitly, the time-domain branch outputs samples directly, with no borrowed phase, so the network *learns* the correct phase for contested cells instead of reusing the mixture's. The **MDX** models (the winners of the Music Demixing Challenge lineage) push spectrogram-domain separation hard with large frequency-transformer architectures that model long-range spectral structure. The trend is the same as everywhere in audio: learned representations beat hand-designed ones, and modeling the waveform (with its phase) beats magnitude-only when you can afford it. The cost is compute, the hybrid and transformer models are heavier than a single spectrogram U-Net, which is the fidelity-versus-speed tension showing up yet again, this time on the separation axis.

Running a separation is genuinely a one-liner, which is part of why this technique is everywhere. Demucs ships a CLI:

```bash
# Install and split a song into 4 stems (vocals / drums / bass / other).
pip install demucs
demucs path/to/song.mp3
# Writes separated/htdemucs/song/{vocals,drums,bass,other}.wav at 44.1 kHz.

# Two-stem mode: just isolate (or remove) vocals, the karaoke shortcut.
demucs --two-stems=vocals path/to/song.mp3
```

And from Python, when you want the stems as tensors to feed into the next stage of an editing pipeline:

```python
import torch, torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model

model = get_model("htdemucs")            # hybrid transformer Demucs, 4 stems
model.eval()

wav, sr = torchaudio.load("song.wav")    # [2, T], stereo
# Demucs expects a batch dim and its own sample rate (44.1 kHz).
sources = apply_model(model, wav[None], device="cuda")[0]   # [4, 2, T]

stem_names = model.sources                # ["drums", "bass", "other", "vocals"]
for name, audio in zip(stem_names, sources):
    torchaudio.save(f"{name}.wav", audio.cpu(), sr)
```

The output stem order, `["drums", "bass", "other", "vocals"]`, is worth memorizing because the index ordering trips people up; index by `model.sources`, never by a hard-coded position.

#### Worked example: separation quality in SI-SDR, and where it leaks

Separation is measured in **SI-SDR** (scale-invariant signal-to-distortion ratio), in decibels: how much of the recovered stem is the true source versus interference and artifacts, with the scale-invariance removing trivial loudness differences. Higher is better. On the standard **MUSDB18** benchmark, the original Spleeter sits around 5 to 6 dB SI-SDR averaged across stems; HT-Demucs reaches roughly 9 dB and above on the same benchmark, a large perceptual jump, the difference between an obviously-processed karaoke track and one a casual listener might not flag. These are approximate, benchmark-dependent figures, treat them as the right order of magnitude rather than exact, and the gap is real and consistent across reports. Where does separation still leak? Two reliable failure modes: **vocal harmonies and reverb tails** smear into "other" because they share time-frequency cells with sustained instruments, and **transient overlap** (a cymbal crash exactly on a vocal consonant) confuses the mask because two sources genuinely co-occupy the same cells. When you hear a "watery" or "phasey" quality in a separated stem, that is the borrowed-phase artifact and the mask hedging its bets on contested cells.

## Music editing, inpainting, and continuation

The last operation is editing the music itself: not changing the singer or pulling apart the mix, but *modifying the content*, extending a clip forward in time, filling a removed section so it matches its surroundings, or restyling a passage from, say, piano to strings. These are the audio analogues of image **outpainting** (extend the canvas) and **inpainting** (fill a masked hole), and the machinery is the same idea, condition the generative model on the existing audio so its output is coherent with it, rather than starting from pure noise with only a prompt.

![A before and after comparison showing a given clip of eight seconds of lo-fi with an established groove, a masked four second gap, and a hard cut at the end, versus an edited version that keeps the same intro refills the gap to match both edges and extends the track to twelve seconds with the tempo held](/imgs/blogs/singing-voice-and-music-editing-7.png)

The figure makes the two main edits concrete. On the left is a clip with two problems: a four-second hole where a section was removed, and an abrupt ending where you want the music to keep going. On the right, both are fixed: the gap is filled with material that matches the groove on *both* sides of the hole, and the track is extended past its original end with the tempo and key held. Filling the gap is **inpainting**; extending past the end is **continuation** (outpainting). The crucial property in both is **boundary coherence**: the generated audio must agree with the existing audio at the seams, in tempo, key, timbre, and energy, or you hear a splice.

### Continuation: MusicGen conditioned on a prefix

The simplest form is continuation, and [MusicGen](/blog/machine-learning/audio-generation/conditioning-and-control-in-audio-generation) does it almost for free because it is autoregressive over codec tokens. To continue a clip, you encode the existing audio into [EnCodec tokens](/blog/machine-learning/audio-generation/encodec-dac-and-the-modern-codec), feed those tokens as the *prefix* of the generation, and let the model autoregress forward. Because the model is predicting the next token given all previous tokens, and the previous tokens *are* the real audio, the continuation naturally inherits the groove, key, and instrumentation of the prefix. It is the audio equivalent of giving a language model the start of a paragraph and asking it to keep writing.

```python
import torchaudio
from audiocraft.models import MusicGen

model = MusicGen.get_pretrained("facebook/musicgen-melody")
model.set_generation_params(duration=12)     # total length including the prompt

prompt_wav, sr = torchaudio.load("lofi_8s.wav")   # the existing 8 s to continue
# Continuation: the prompt audio becomes the prefix; text steers the new part.
out = model.generate_continuation(
    prompt=prompt_wav[None],          # [B, C, T]
    prompt_sample_rate=sr,
    descriptions=["warm lo-fi hip hop, same groove"],
    progress=True,
)
torchaudio.save("lofi_continued.wav", out[0].cpu(), model.sample_rate)
```

The continuation inherits its coherence from the prefix tokens, while the `descriptions` text steers the *new* material, the same conditioning machinery from the earlier post. The honest limitation, and the next stress test, is **length**: an autoregressive model continued far past its training window drifts. Ask MusicGen for four minutes in one pass and somewhere around the 30-second mark the beat can wander, the key can slip, or the texture can collapse into a loop, because the model was trained on roughly 30-second contexts and has no real long-range memory of the bar structure. The practical workaround is to generate in overlapping windows and crossfade, or to use a model designed for length, which is where latent diffusion comes in.

### Inpainting and audio-to-audio: diffusion's natural edit

Diffusion models are *naturally* good at inpainting because the denoising process can be constrained: at each denoising step, force the known region (the audio around the hole) to its true value and let the model only fill the masked region, so the fill is generated *conditioned on* the surrounding context at every step. This is exactly how image inpainting with diffusion works, and [Stable Audio](/blog/machine-learning/audio-generation/diffusion-for-audio) and AudioLDM-class models inherit it. Stable Audio's timing conditioning, the fact that it knows the absolute start and end times of the segment, also helps it produce material that fits a target duration cleanly, which matters when filling a gap of a specific length.

The related and very practical edit is **audio-to-audio** style transfer, Stable Audio's `audio2audio` and the general "init audio" pattern in diffusion. Instead of starting the reverse diffusion from pure noise, you start from a *noised version of an input clip*: add noise to the input up to some intermediate timestep (controlled by a strength parameter), then denoise from there with a new prompt. Low strength keeps most of the input's structure and only nudges its style; high strength keeps only the broad outline and lets the prompt dominate. It is the audio twin of image-to-image, and it is how you restyle a section, take the piano take, add noise to 60 percent, prompt "string quartet," and denoise, keeping the melody and timing while changing the instrument.

```python
import torch, soundfile as sf
from diffusers import StableAudioPipeline

pipe = StableAudioPipeline.from_pretrained(
    "stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16
).to("cuda")

init_audio, sr = sf.read("piano_phrase.wav")
init = torch.tensor(init_audio).T[None].float()    # [B, C, T]

# audio-to-audio: keep timing + melody, change the instrument via the prompt.
audio = pipe(
    prompt="a warm string quartet playing the same phrase",
    initial_audio_waveforms=init,
    initial_audio_sampling_rate=sr,
    strength=0.6,                 # 0 = keep input, 1 = ignore input
    num_inference_steps=100,
    guidance_scale=7.0,
).audios[0]
sf.write("restyled_strings.wav", audio.T.float().cpu().numpy(), pipe.vae.sampling_rate)
```

The `strength` knob is the whole game: it is the single number that trades *fidelity to the input* against *adherence to the prompt*, and finding the right value (often 0.5 to 0.7 for a restyle that keeps the tune) is the main thing you tune. This is the [classifier-free-guidance](/blog/machine-learning/image-generation/classifier-free-guidance) and image-to-image intuition carried straight over to audio.

#### Worked example: the seam test for an inpaint

When you fill a four-second gap, the only thing that matters is whether the listener can hear *where* the fill starts and ends. A useful, honest evaluation: render the inpainted track, then ask listeners (or yourself, blind) to mark the timestamps where they think edits occurred, and measure how often they land within a second of the true seams, call it the **seam-detection rate**. A perfect inpaint drives that rate to chance. In my experience the failures are rarely in the *middle* of the fill, which usually sounds fine, but at the *boundaries*, a tiny energy mismatch or a key that is right but a phase that is wrong at the splice point, producing a click or a momentary "swell." This is why diffusion inpainting, which conditions on the boundary at every denoising step, tends to beat naive "generate a clip and crossfade it in," which only matches the boundary approximately. Pair the seam test with **FAD** against the original track's distribution to catch global drift, and you have an honest, cheap evaluation of an edit.

## A unified view: the same factors, four edits

It is worth collecting the four operations into one comparison, because the symmetry is the point. Every one of them is an instance of "represent the audio as separable factors, hold some fixed, change one."

![A matrix comparing four editing tasks by input what changes and method, where singing synthesis takes lyrics plus score and creates a voice with DiffSinger, singing conversion takes a sung clip and changes singer timbre with RVC or so-vits-svc, stem separation takes a full mix and splits to stems with Demucs or MDX, and inpainting takes audio plus a gap and refills the section with MusicGen or Stable Audio](/imgs/blogs/singing-voice-and-music-editing-5.png)

The matrix lays out the symmetry. Read down the "What changes" column and you see the unifying logic: synthesis *creates* the voice factor (from a score that fixes content and pitch); conversion *changes* the timbre factor (holding content and pitch); separation *splits* the source factors (holding none, recovering all); inpainting *regenerates* a content section (holding the surrounding audio). Different tasks, one organizing principle. And the "Method" column shows the practical division of labor: autoregressive token models (MusicGen) for continuation, diffusion (DiffSinger, Stable Audio) for synthesis and inpainting, dedicated separators (Demucs) for demixing, and the content-plus-F0-plus-speaker decoder (RVC, so-vits-svc) for conversion.

Here is the same content as a table you can act on, with the factor each task holds fixed made explicit, the single most useful thing to internalize.

| Task | Input | Held fixed | Changed | Open tool | Engine type |
|---|---|---|---|---|---|
| Singing synthesis (SVS) | lyrics + score | content, pitch (from score) | creates the voice | DiffSinger | diffusion + NSF vocoder |
| Singing conversion (SVC) | a sung clip | content, pitch (F0) | singer timbre | RVC, so-vits-svc | content/F0/speaker decoder |
| Stem separation | a full mix | (recovers all sources) | splits to stems | Demucs, Spleeter, MDX | masking / hybrid U-Net |
| Inpainting / continuation | audio + a gap/edge | surrounding audio | a section or extension | MusicGen, Stable Audio | AR tokens or diffusion |

The "Held fixed" column is the discipline of editing. Generation has no held-fixed column, anything goes. Editing is defined by what it refuses to change.

The operations also **compose**, which is where the real power shows up and where the discipline matters most. A full "AI cover" of a song with a band is not one operation; it is a pipeline of them. You first *separate* the original mix to isolate the lead vocal (so the conversion is not fighting accompaniment), then *convert* that isolated vocal to the target voice, then *remix* the converted vocal back over the instrumental stems. Three edits chained, each touching one factor, each preserving the others. The order is not arbitrary: separate before convert, because the content encoder and F0 extractor are cleaner on an isolated vocal; convert before remix, because you want the new voice sitting in the same instrumental bed. Get the order wrong, try to convert the full mix without separating first, and the F0 extractor locks onto the bass line instead of the vocal, and the content encoder hears guitars as phonemes, and the result is mush. Composition is where a clear factor model pays off most, because it tells you not just *which* tool but in *which order*.

There is also a quiet asymmetry between the tasks worth naming. Separation is *lossy and irreversible* in a way the others are not: once two sources overlapped in a time-frequency cell, no separator perfectly recovers both, so a separated-then-remixed track is never bit-identical to the original even if you change nothing. Synthesis, conversion, and inpainting are *constructive*, they build new audio, so their "loss" is measured against an intent, not against a ground-truth original. This is why separation has a ground-truth metric (SI-SDR against the true stems on a benchmark) while the constructive tasks lean on reference-free or human metrics. When you chain separation into a pipeline, you inherit its irreversible loss, which is a good reason to separate as *few* times as the pipeline allows and to keep the original mix around as a fallback.

## Evaluation: no single number covers editing

Because the four tasks hold different things fixed, they need different metrics, and using the wrong one hides exactly the failure that matters. This is a recurring theme in the [evaluate-audio-generation-honestly](/blog/machine-learning/audio-generation/audio-quality-metrics) post, but editing makes it especially sharp.

![A matrix showing how to measure each editing task where singing synthesis uses F0 RMSE and voicing decision error for in-tune-ness watching for buzzy vibrato, singing conversion uses speaker embedding cosine similarity for singer match watching for leaked source, stem separation uses SI-SDR in decibels for stem cleanliness watching for bleed and holes, and inpainting uses FAD plus a seam check for coherence watching for an audible splice](/imgs/blogs/singing-voice-and-music-editing-8.png)

The figure matches each task to its primary metric and its characteristic failure. For **SVS**, the metric is **F0 RMSE** (root-mean-square error between realized and target pitch, in cents or Hz) plus **voicing decision error** (did the model sing where it should and stay silent where it should); these directly measure "is it in tune and in time," and the failure to watch for is a buzzy or wrong-rate vibrato that F0 RMSE alone might not catch, so pair it with listening. For **SVC**, the metric is **speaker similarity**, the cosine similarity (often called SECS) between a speaker-verification embedding of the output and of real target-singer audio; this measures "does it actually sound like singer B," and the failure to watch for is *leaked source*, where the output is intelligible and in tune but secretly still carries singer A's color, which similarity-to-B catches and naturalness alone does not. For **separation**, the metric is **SI-SDR** as discussed, with the failures being bleed (other sources in the stem) and holes (parts of the true source missing). For **inpainting and continuation**, there is no perfect single number, so use **FAD** against the original's distribution to catch global drift plus the **seam-detection** test to catch boundary artifacts.

A small but important honesty point: for SVC and SVS, **MOS** (mean opinion score, human naturalness ratings) is still the gold standard for "does it sound good," but it is *insufficient alone* because a conversion can sound perfectly natural while being the wrong singer, or an SVS can sound natural while being subtly out of tune. Always pair a naturalness score with a *fidelity-to-the-constraint* score: similarity for conversion, pitch error for synthesis, SI-SDR for separation, seam rate for inpainting. The constraint is the whole point of editing; do not let a naturalness number paper over a constraint violation.

| Task | Naturalness | Constraint fidelity | The trap |
|---|---|---|---|
| SVS | MOS | F0 RMSE, voicing error | sounds nice, slightly flat |
| SVC | MOS / CMOS | speaker sim (SECS) | natural but wrong singer |
| Separation | listening | SI-SDR (dB) | clean-sounding but leaky |
| Inpaint / extend | MOS | FAD + seam rate | great middle, audible seam |

## Case studies and real numbers

A few concrete anchors from shipped systems and the literature, with the usual caveat: treat exact figures as approximate and benchmark-dependent unless you re-measure on your own data with a fixed protocol.

**DiffSinger.** The diffusion-acoustic-model approach to SVS (Liu et al., 2022) was introduced specifically to fix the over-smoothing of regression-based SVS. Its reported contribution is sharper mel-spectrograms and better naturalness MOS than the FastSpeech-style regression baseline it compared against, at the cost of the usual diffusion expense, many denoising steps per mel, which is why later work added shallow-diffusion and few-step tricks to bring the step count (and the real-time factor) down. The lesson that generalizes: diffusion buys spectral detail (and thus a livelier voice) but costs steps, and the SVS field has spent considerable effort buying that detail back at fewer steps.

**so-vits-svc and RVC.** These are community open-source systems rather than papers with formal benchmarks, but their design is the textbook disentanglement: a self-supervised content encoder (HuBERT, then ContentVec), an F0 extractor (CREPE, then the more robust RMVPE), a speaker embedding, and an NSF-style pitch-conditioned decoder. RVC's distinguishing **retrieval** index measurably reduces timbre leakage in practice. The honest characterization: with tens of minutes of clean target audio and a good F0 extractor, conversions are convincing enough to have driven a cultural moment; with a noisy three-second target or a bad F0 track, they degrade exactly as the disentanglement story predicts.

**Demucs (HT-Demucs).** Défossez and colleagues' hybrid transformer Demucs is the open separation workhorse. On the MUSDB18 benchmark it reaches roughly 9 dB SI-SDR averaged across stems and higher, a clear improvement over the ~5 to 6 dB of the older Spleeter (Hennequin et al., 2020), which itself was a milestone for being fast and free. The practical takeaway: for a karaoke or remix pipeline today, HT-Demucs is the default, with the waveform-plus-spectrogram hybrid design being the reason it avoids the borrowed-phase "watery" artifact that magnitude-mask models exhibit. Spleeter still has a place when you need separation that runs very fast on CPU and can tolerate lower quality.

**MusicGen continuation and Stable Audio inpainting.** MusicGen (Copet et al., 2023) does continuation natively via token prefixing, and its melody-conditioned variant can hold a melody while regenerating the arrangement, a form of musical editing. Its limit is length: trained around 30-second contexts, single-pass generation drifts well before song length, which is the canonical "ask for four minutes and lose the beat" failure. Stable Audio (Evans et al., 2024) was built around timing conditioning precisely to generate longer, variable-length, duration-accurate audio, which is also what makes its inpainting and audio-to-audio edits fit a target gap cleanly. The division of labor is consistent with the rest of the series: autoregressive token models are natural for *continuation*, latent diffusion is natural for *length and inpainting*.

**The melody-conditioned editing path.** MusicGen-melody deserves a closer look as an editing tool, because it is a *different* kind of edit from continuation. You give it a reference audio clip and a text prompt; it extracts a chromagram (a 12-bin pitch-class profile per frame, "which of the twelve notes are sounding") from the reference and conditions on it, then renders new audio that follows that melodic contour but in the instrumentation the text describes. That is melody-preserving *re-instrumentation*: keep the tune, change the timbre of the whole arrangement, the orchestral analogue of singing-voice conversion. The honest caveat from the literature is that chromagram conditioning captures *pitch classes* but not octave or precise rhythm, so the regenerated arrangement follows the melody loosely rather than note-for-note; it is a strong stylistic steer, not a transcription-faithful copy. For a creator who wants "this melody, but as a synthwave track," it is exactly right; for someone who needs the exact notes reproduced, it is not, and you would reach for an explicit score and SVS-style synthesis instead.

**ContentVec's measured disentanglement.** The ContentVec paper (Qian et al., 2022) is worth citing not just as the encoder of choice but for *why* it works, because it quantified the thing this whole post hinges on. It showed that adding explicit speaker-disentanglement objectives to a HuBERT-style encoder measurably *reduced* the speaker information leaking through the content features (probed by how well a classifier could recover speaker identity from them) while *preserving* the content information (probed by phone-recognition accuracy). That is the disentanglement made quantitative: less speaker, same content. The practical consequence is the cleaner voice conversions the community observed when it switched, a rare case where a representation-learning improvement translated directly and visibly into a downstream editing quality gain.

#### Worked example: building a one-line karaoke maker, and its RTF

Suppose you want a karaoke pipeline: input a song, output the instrumental. The whole thing is `demucs --two-stems=vocals song.mp3` followed by keeping the `no_vocals` stem. On a single modern GPU (think an RTX 4090 class card), HT-Demucs runs comfortably faster than real time, separating a three-to-four-minute song in well under a minute, a real-time factor below roughly 0.3 (generation time divided by audio duration), so a 200-second song takes on the order of tens of seconds. Approximate, and it depends on the GPU and the `--shifts` augmentation setting (more shifts means better quality and slower), but the order of magnitude holds: separation is *cheap* relative to generation, which is exactly why it is the universal first step in editing pipelines. If you raised `--shifts` for quality, the RTF climbs proportionally because the model is run multiple times on shifted copies and averaged.

## A real pipeline, and where it breaks

Let me put the pieces together into a concrete engineering scenario, because that is where the trade-offs become real decisions rather than general principles. The task: a creator hands you a four-minute live recording, one singer plus a band, and wants two deliverables, a karaoke version (instrumental only) and a "studio re-sing" where the live vocal is replaced by a cleaner take in the *same* singer's consented studio voice. Walk the pipeline and watch where it strains.

**Step one, separate.** Run HT-Demucs to split the live mix into vocals, drums, bass, and other. The instrumental for the karaoke track is the sum of everything except vocals, which Demucs's two-stem mode gives directly. This step is cheap (sub-real-time on a GPU) and high quality, but here is the first stress point: live recordings have **heavy reverb and crowd noise**, and reverb tails on the vocal smear into "other," so the karaoke track may carry a faint vocal-reverb ghost. The honest mitigation is not a better separator alone but accepting that live separation is intrinsically harder than studio separation, and possibly a light de-reverb pass first. Measure it with SI-SDR if you have a reference; trust your ears if you do not.

**Step two, convert the vocal.** For the re-sing, you take the *separated* live vocal (not the full mix, separation first means the conversion's content encoder and F0 extractor see a clean-ish vocal, not one fighting the band) and run it through SVC with the singer's consented studio embedding. Now the stress points stack up. If the separated vocal still has accompaniment bleed, RMVPE's accompaniment-robustness earns its keep, but bleed still degrades the F0 track on quiet passages. If the live performance had pitchy moments (live singers drift), the conversion *faithfully reproduces the drift* because it preserves F0, the conversion does not auto-tune. That is correct behavior but may surprise the client, who hears "the AI sang it flat" when in fact the AI faithfully copied a flat live note. The fix, if desired, is an explicit pitch-correction pass on the F0 track *before* the decoder, snapping it toward the nearest scale degree, which is a deliberate edit on the pitch factor, exactly the kind of single-factor edit the disentanglement makes possible.

**Step three, the length stress test.** Suppose instead the client wanted the band's *outro* extended by another sixteen bars in the same style. Now you are in continuation territory, and the four-minute scale is the enemy. MusicGen continuation can extend the outro, but only for a short window before it drifts; you cannot continue four minutes of context in one pass. The realistic approach is to take the last ten to fifteen seconds of the instrumental as the prefix, generate a continuation of the new sixteen bars, and crossfade the seam, then, if the client wants still more, repeat from the new end. Each hop risks a small tempo or key slip, so you keep the hops short and check the beat grid between them. If instead you needed to *fill* a damaged four-second dropout in the middle of the recording, you would not use continuation at all, you would use diffusion inpainting, which conditions on *both* sides of the hole and so cannot drift away from the surrounding key and tempo.

Notice the pattern across all three steps: every stress point is a place where one factor's estimate degrades (reverb corrupts separation, bleed corrupts F0, length corrupts continuation), and every fix is either a better estimator for that one factor or an explicit edit on that one factor. The disentanglement is not just the conceptual frame; it is the *debugging* frame. When an edit sounds wrong, ask which factor is corrupted, content, pitch, timbre, or boundary, and the fault localizes immediately.

#### Worked example: the cost and latency budget of the full pipeline

Put rough numbers on the re-sing pipeline for a four-minute song on a single RTX 4090-class GPU. Separation with HT-Demucs: on the order of tens of seconds (RTF well under 1). F0 extraction with RMVPE over the four-minute vocal: a few seconds. The SVC decode, frame-by-frame through a small NSF vocoder: real-time or faster, so a minute or two at most. The total is a few minutes of compute for a four-minute song, dominated by whichever stage you run at high quality settings. At a rough cloud-GPU rate of around \$0.50 to \$2 per hour for a consumer-class GPU, that is a few cents of compute per song, the cost is not the constraint; the *consent and quality* are. These figures are approximate and setup-dependent; the point is the order of magnitude and the shape, separation and conversion are cheap, so engineering effort should go to the reference quality and the ethics, not to shaving compute.

## Editing in the codec-token domain

One more lens ties this post to the [neural codec](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound) backbone of the series, because increasingly editing happens not on waveforms or spectrograms but on **codec tokens**. Recall that a neural codec turns a waveform into a short sequence of discrete tokens (via [residual vector quantization](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq)) and back. Once audio is a token sequence, several edits become *sequence operations*, the same way text editing is a sequence operation.

Continuation is the cleanest example: as we saw, MusicGen continues a clip by taking its EnCodec tokens as a prefix and predicting more tokens, no waveform manipulation at all until the final decode. Inpainting in the token domain is the analogue of masked-language-model infilling: mask a span of tokens and predict them conditioned on the tokens before *and* after, which is exactly boundary-coherent gap filling expressed as sequence infilling. And because the tokens are organized in [semantic and acoustic levels](/blog/machine-learning/audio-generation/semantic-vs-acoustic-tokens), you can edit at different granularities, change the high-level semantic tokens to alter *what* is played while keeping fine acoustic tokens for texture, or vice versa.

The appeal of token-domain editing is that it reuses the *exact same model* that does generation, an autoregressive or masked transformer, so an edit is just a different way of prompting it, no separate edit architecture. The limitation is resolution: codec tokens are quantized, so a token-domain edit cannot be more precise than the codec's time and rate resolution, and a low-bitrate codec that has thrown away high-frequency detail will produce edits that inherit that loss. This is the bitrate-quality trade from the codec posts surfacing in editing: edit on a richer codec (more RVQ levels, higher bitrate) and your edits are higher fidelity but the sequences are longer and the model slower. The same four-way tension, fidelity, controllability, speed, length, governs the choice of *where* in the stack to edit, raw waveform, spectrogram, codec tokens, or latent, just as it governs everything else in the series.

#### Worked example: why a low-bitrate codec ruins a vocal edit

Take a vocal you intend to edit in the token domain and encode it with a codec at, say, 1.5 kbps versus 6 kbps. At 1.5 kbps the codec keeps only a couple of RVQ levels and discards much of the high-frequency air and sibilance; the *reconstruction* may still be intelligible, but the *editable* representation has lost the sibilant /s/ texture and the breathy top end. Now inpaint a word: the model fills the gap using the impoverished token vocabulary, and the filled word sounds duller and breathier-deficient than its neighbors, an audible seam that has nothing to do with the inpainting model and everything to do with the codec's bitrate. The same inpaint at 6 kbps, where the tokens retained the high-frequency detail, blends in cleanly. The lesson: for editing, choose the codec bitrate by the *detail you need to preserve and regenerate*, not just by the reconstruction quality, because an edit must regenerate from the tokens, and you cannot regenerate detail the tokens never stored.

## When to reach for this (and when not to)

A decisive section, because the editing tools are easy to misuse.

**Reach for stem separation** whenever you need to operate on one part of a finished mix: karaoke, sampling, remixing, re-instrumentation, or as the *first stage* of any other edit (you often separate the vocal before converting the singer, so the conversion is not confused by accompaniment bleed). It is cheap, fast, and high-quality enough now that there is rarely a reason not to. Default to HT-Demucs; drop to Spleeter only when you need CPU speed over quality.

**Reach for singing voice synthesis** when you need a *new* sung performance from a score you control, a composed melody with lyrics, and you have (or can train) a target voice. It is the right tool when you own the composition and the voice. It is the *wrong* tool when you actually have a recording you want to modify; modifying a recording is conversion or editing, not synthesis.

**Reach for singing voice conversion** when you have a real performance and want it in a *different, consented* voice: your own voice, a voice you have licensed, or a synthetic voice with no real-person claim. Do **not** reach for it to clone a non-consenting real artist; that is the harm the field is actively fighting, and beyond the ethics it is increasingly a legal and platform-policy risk. Technically, do not reach for SVC when a clean reference of the target voice is unavailable; the result will be poor and you will blame the model when the problem is the reference.

**Reach for continuation (MusicGen-style)** for *short* extensions where inheriting the prefix's groove matters and a few extra seconds is enough. Do **not** autoregress a full four-minute song in one pass expecting it to hold the beat; it will drift. For long-form or for filling a fixed-length gap, reach for **diffusion inpainting (Stable Audio-style)** instead, which conditions on both boundaries and respects target duration. And do not use a heavy diffusion edit when a trivial DSP operation suffices, if you only need a two-semitone pitch shift with the same voice, a phase vocoder is transparent, instant, and free; a learned conversion is overkill and riskier.

The meta-rule: **match the tool to which factor you are changing.** Changing the singer is conversion; creating a singer is synthesis; pulling apart sources is separation; changing the content is editing. Using the wrong tool, e.g. trying to "convert" when you really need to "separate then convert", is the most common pipeline mistake I see.

## Key takeaways

- **Editing is generation under a constraint**: part of the output is dictated by an existing signal, so the ear has a reference and will hear any seam or drift. That constraint, not freedom, defines every task here.
- **Everything rests on disentanglement**: factor a sound into content, pitch, and timbre (for voices) or into sources (for mixes), hold most fixed, change one. The quality of an edit is the quality of that factoring.
- **SVS is conditioned on exact F0 targets**, unlike free-prosody TTS. The score sets the pitch; the model must hit it, with vibrato as expressive signal carried in the F0 channel and an explicit-F0 vocoder keeping long notes in tune.
- **SVC swaps timbre while preserving content and F0** via a content encoder (ContentVec/HuBERT), an F0 extractor (RMVPE/CREPE), a target-speaker embedding, and a pitch-conditioned decoder. Speaker-invariant content is what prevents source leakage.
- **The SVC pipeline created a real consent and IP problem**: the same machinery that empowers consented and original voices appropriates non-consenting artists' voices. Treat a target voice like a copyrighted sample; lean on consent, watermarking, and detection.
- **Stem separation is masking in a time-frequency (or learned waveform) domain**: predict a per-cell mask per source, multiply, invert. Hybrid waveform-plus-spectrogram models (HT-Demucs) beat magnitude-only by modeling phase, reaching roughly 9 dB SI-SDR on MUSDB18 versus ~5 to 6 for Spleeter.
- **Continuation is autoregressive prefixing; inpainting is constrained diffusion.** AR token models inherit a prefix's groove but drift over length; diffusion conditions on both boundaries and respects target duration, so it fills gaps and restyles sections cleanly.
- **Audio-to-audio style transfer is image-to-image carried over**: start denoising from a noised input, and the `strength` knob trades fidelity-to-input against adherence-to-prompt.
- **No single metric covers editing**: F0 RMSE for synthesis, speaker similarity for conversion, SI-SDR for separation, FAD plus a seam test for inpainting. Always pair a naturalness score with a fidelity-to-the-constraint score, because a natural-sounding output can still violate the constraint that was the entire point.

## Further reading

- **Liu et al., "DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism" (2022)** — the diffusion-acoustic-model approach to SVS and the over-smoothing fix.
- **Défossez et al., "Hybrid Transformers for Music Source Separation" (HT-Demucs, 2023)** and **Hennequin et al., "Spleeter: a fast and efficient music source separation tool" (2020)** — the separation workhorses and their masking-versus-hybrid designs.
- **Copet et al., "Simple and Controllable Music Generation" (MusicGen, 2023)** and **Evans et al., "Stable Audio: Fast Timing-Conditioned Latent Audio Diffusion" (2024)** — continuation, melody conditioning, and timing-conditioned inpainting.
- **Qian et al., "ContentVec: An Improved Self-Supervised Speech Representation by Disentangling Speakers" (2022)** and the **HuBERT** paper (Hsu et al., 2021) — the content encoders that make conversion possible.
- **Official tools**: the [`demucs`](https://github.com/adefossez/demucs) repository, the 🤗 [`transformers` audio docs](https://huggingface.co/docs/transformers/index) (MusicGen, EnCodec), and the 🤗 [`diffusers` Stable Audio pipeline](https://huggingface.co/docs/diffusers/index).
- **Within this series**: the foundation [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard), [conditioning and control](/blog/machine-learning/audio-generation/conditioning-and-control-in-audio-generation), [diffusion for audio](/blog/machine-learning/audio-generation/diffusion-for-audio), [prosody and expressive speech](/blog/machine-learning/audio-generation/prosody-emotion-and-expressive-speech), the forward-looking [audio deepfakes, watermarking, and voice safety](/blog/machine-learning/audio-generation/audio-deepfakes-watermarking-and-voice-safety) and [the 2026 audio model landscape](/blog/machine-learning/audio-generation/the-2026-audio-model-landscape), and the capstone [building an audio generation stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack).
