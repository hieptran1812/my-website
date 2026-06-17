---
title: "Prosody, Emotion, and Expressive Speech: Beyond Intelligibility"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Why correct TTS sounds flat, and the full toolkit for fixing it: reference encoders, global style tokens, explicit F0-energy-duration control, emotion labels, description-driven style, and the sampling that gives codec models expression for free."
tags:
  [
    "audio-generation",
    "audio-synthesis",
    "text-to-speech",
    "prosody",
    "expressive-speech",
    "voice-cloning",
    "generative-ai",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Audio Generation"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/prosody-emotion-and-expressive-speech-1.png"
---

The first TTS demo I shipped passed every test we wrote and still felt like a failure. Every word was correct. An automatic speech recognizer transcribed it back at well under one percent word error rate. The voice was clear, the pronunciation was textbook, and a stranger could understand it instantly. And yet when the product team played it for a focus group, the most common single word in the feedback was "robot." Not because it mangled any sound, but because it said *every* sentence the same way. A question landed with the same falling pitch as a statement. The punchline of a joke got the same energy as the setup. A sentence that should have risen with excitement plodded along at a dead, even pace. The model had solved intelligibility completely and expressiveness not at all, and the gap between those two is the entire subject of this post.

That gap has a precise technical cause, and it is not a lack of model capacity. It is that the *same text* has many valid spoken realizations, and a model trained the obvious way will average them into a flat, lifeless mean. "Beyond intelligibility" is the work of climbing out of that average: giving the model a way to pick *one* believable delivery instead of mushing all of them together, and giving *you* a handle to say which delivery you want. By the end of this post you will be able to extract a pitch contour and energy envelope from a reference clip in `torchaudio`, condition an acoustic model on a style embedding, edit explicit F0 and duration like turning knobs on a console, drive a model entirely from a sentence like "a calm, low-pitched male voice, slightly sad," and measure whether any of it actually worked. You will also know when expressive control is not worth the naturalness and stability it costs.

![A vertical stack showing the four prosody variables pitch, energy, duration, and pauses layered over phoneme content to produce an expressive waveform](/imgs/blogs/prosody-emotion-and-expressive-speech-1.png)

The figure above is the map for the whole post. At the bottom is the phoneme content, the *words*, which is everything intelligibility needs and nothing expressiveness needs. Stacked on top are the four variables that carry delivery: the **pitch** contour (the melody of the voice), the **energy** (loudness and stress), the **duration** of each sound (rhythm and tempo), and the **pauses** that phrase a sentence. Together these four are what linguists call **prosody**, and they are what we will spend this post learning to control. This sits squarely on the series spine: the [audio stack](/blog/machine-learning/audio-generation/why-audio-generation-is-hard) of waveform to latent to generative core to vocoder, under the tension of fidelity, controllability, speed, and length. Prosody is the *controllability* axis aimed at speech, and like every control it trades against the others, more expressive freedom often buys you a little less stability. This post is the direct continuation of [conditioning and control in audio generation](/blog/machine-learning/audio-generation/conditioning-and-control-in-audio-generation), which gave the general menu of conditioning signals; here we zoom all the way in on the hardest one for speech, the soft control of style and emotion.

## What prosody actually is

Before we can control prosody we have to be precise about what it is, because "make it more expressive" is not a spec a model can optimize. Prosody is the set of features of speech that exist *above* the level of the individual phoneme, the so-called **suprasegmental** features. They are properties not of a single sound but of how a stretch of sounds is delivered. There are four that matter for synthesis, and they map almost one-to-one onto things you can measure on a waveform.

The first is **pitch**, which acoustically is the **fundamental frequency** of the voice, written F0. When your vocal folds vibrate they do so at some rate, and that rate, measured in hertz, is the pitch you hear. A typical adult male voice sits around 100 to 120 Hz, a typical adult female voice around 180 to 220 Hz, and within any single utterance the F0 moves around constantly. That movement is the **F0 contour**, and it carries an enormous amount of meaning. A rising contour at the end of a phrase signals a question in English; a sharp peak signals emphasis; a wide range signals excitement while a narrow range signals boredom or depression. F0 is the single most important prosodic variable, and most of expressive TTS is really about getting the F0 contour right.

The second is **energy**, the moment-to-moment loudness of the signal, which acoustically is roughly the **root-mean-square amplitude** of each short frame. Stressed syllables are louder; emphasis is partly an energy spike; a sentence that trails off does so in energy as much as in pitch. Energy and pitch tend to move together, but not always, and a model that gets pitch right but energy flat still sounds slightly off.

The third is **duration**, which is the timing: how long each phoneme is held, and therefore the rhythm and the speaking rate. A fast, clipped delivery and a slow, drawn-out one are the same words with different durations. Crucially, duration is *not* uniform, you do not just stretch everything by a constant factor to slow down; you lengthen vowels more than consonants, you hold the stressed syllable of an emphasized word, you slow down before an important point. Duration is also where **pauses** live, the silences between phrases, and pauses are the fourth variable. Where you breathe, where you break a sentence into chunks, how long you hold a dramatic silence, all of that is phrasing, and phrasing is a huge part of why human speech sounds human and synthetic speech often does not.

Here is the thing that makes prosody both important and hard: these four variables carry meaning that the words alone do not. The sentence "I never said she stole my money" has seven completely different meanings depending on which word you stress, and stress is pitch plus energy plus duration. Sarcasm, doubt, warmth, urgency, none of it is in the phonemes; all of it is in the prosody. So a TTS system that ignores prosody is not just missing some polish. It is unable to say a large fraction of what speech actually communicates. Intelligibility gets you the dictionary meaning of the words. Prosody gets you the *speaker's* meaning. That is why "beyond intelligibility" is not a luxury feature.

#### Worked example: the same words, two meanings

Take the four-word sentence "you did it again." Said with a rising F0 contour, a moderate energy peak on "again," and a short final pause, it is a delighted congratulation: *you did it again!* Said with a flat-then-falling F0, a heavy energy stress dragged out on "again," and a long pause before it, it is a weary reproach: *you did it... again.* Same phonemes, same word error rate, same intelligibility. The acoustic difference is entirely in three numbers per frame, F0, energy, and duration, plus one pause. A deterministic TTS model handed only the text "you did it again" has no way to know which you meant, and as we are about to see, when it cannot know, it does the worst possible thing: it splits the difference and produces neither.

### Why these four and not others

It is worth pausing on *why* prosody reduces to exactly these variables and not some larger or smaller set, because the answer is grounded in how speech is physically produced and how the ear perceives it, and that grounding tells you which variables are worth modeling. Human speech is a source-filter system: the vocal folds (the source) produce a buzzing tone at the fundamental frequency, and the vocal tract (the filter, the shape of your throat, tongue, and lips) shapes that buzz into the formants that distinguish one vowel from another. The phonemes, the *content*, are almost entirely a filter phenomenon, they are which vowels and consonants you make, set by tongue and lip position. Prosody is almost entirely a *source-and-timing* phenomenon: how fast the folds vibrate (F0), how forcefully you push air (energy), and how long you hold each configuration (duration). This is not a coincidence of taxonomy; it is the physical reason content and prosody are *separable* at all. You can change the source (sing the same words at a different pitch) without touching the filter (the words stay the same). That separability is what makes prosody a thing you can control independently, and it is what every disentanglement-based method in this post is implicitly exploiting.

On the perception side, the ear is exquisitely sensitive to all four. Pitch is perceived on a roughly logarithmic scale (the [mel and related scales](/blog/machine-learning/audio-generation/representing-sound-waveforms-spectrograms-and-perception) capture this), and humans can detect F0 differences of well under one percent in the right range, which is why a contour that is even slightly wrong reads as "off" immediately, far more unforgiving than the eye is about a slightly wrong color. Timing is perceived with millisecond precision, the difference between "white house" and "White House" is partly a few tens of milliseconds of pause and stress. This sensitivity is the deep reason expressive TTS is hard: the tolerances are tight, and a model that is right on average but wrong in the details gets caught instantly by an ear that evolved to read exactly these cues for emotion and intent. The four variables are the ones the ear weights most heavily, which is precisely why they are the ones worth the modeling effort.

## The one-to-many problem, formalized

The flatness in my first demo was not a bug I could fix by training longer or adding parameters. It was a mathematical consequence of the loss function, and understanding exactly why is the key that unlocks every technique in the rest of this post. So let us be precise.

A TTS acoustic model is, at its core, a function that maps text $x$ to some acoustic target $y$, typically a mel-spectrogram, but for this argument think of $y$ as the prosody, the F0 and energy and duration. The trouble is that the mapping from text to prosody is **one-to-many**: for a single $x$ there are many valid $y$. The sentence from the worked example has at least two correct contours, and in reality a continuum of them. So in the training data, the same text appears (across speakers, takes, and contexts) paired with many different prosodies. The conditional distribution $p(y \mid x)$ is genuinely **multimodal**, it has several distinct peaks, one per valid reading.

Now consider what happens when we train with the standard regression loss, mean squared error, asking the model to predict a single $\hat{y} = f(x)$ that minimizes $\mathbb{E}\big[\lVert y - f(x) \rVert^2\big]$ over the data. Calculus gives the minimizer immediately. For a fixed $x$, the value of $\hat{y}$ that minimizes the expected squared error is the **conditional mean**:

$$
f^\star(x) = \mathbb{E}[\, y \mid x \,].
$$

This is the entire problem in one line. The MSE-optimal prediction is the *average* of all valid prosodies for that text. If half your training examples say "you did it again" with a rising excited contour and half say it with a falling weary one, the loss-minimizing prediction is the average of rising and falling, which is **flat**. The model is not failing to learn; it is learning the provably optimal thing for the loss you gave it, and that optimal thing is the mean, and the mean of a multimodal distribution sits in the low-probability valley *between* the modes, a prosody that no human ever actually produced. The same logic applies to an L1 loss, which targets the conditional median, equally lifeless. This is why deterministic TTS sounds averaged: because it literally is the average.

The proof is a one-liner worth seeing because it shows the collapse is unconditional, it does not depend on the model or the data being special. Expand the expected squared error for a fixed $x$, adding and subtracting the mean $\mu = \mathbb{E}[y \mid x]$:

$$
\mathbb{E}\big[\lVert y - \hat{y}\rVert^2 \mid x\big]
= \mathbb{E}\big[\lVert y - \mu \rVert^2 \mid x\big]
+ \lVert \mu - \hat{y} \rVert^2 .
$$

The cross term vanishes because $\mathbb{E}[y - \mu \mid x] = 0$ by definition of the mean. The first term is the conditional **variance** of the target, which does not depend on $\hat{y}$ at all, it is an irreducible floor set by how spread-out the valid prosodies are. The only term you can minimize is the second, $\lVert \mu - \hat{y}\rVert^2$, which is zero exactly when $\hat{y} = \mu$. So the optimum is the mean, always, and the best achievable loss is the conditional variance. Read that second fact carefully: it says the *more multimodal* the prosody (the more genuinely different valid readings exist for a text), the *higher* the irreducible MSE floor, and the *flatter* the optimal output. The very expressiveness you want is, under MSE, an error term the model is rewarded for averaging away. The loss is not just failing to encourage expression; it is actively penalizing it. That is as damning an indictment of regression-style prosody modeling as you can write, and it is why the whole field moved.

There is a second, subtler consequence that explains a symptom you will actually observe. Because the floor is the conditional variance, a model trained this way will be *most* flat exactly where the data is *most* varied, which is at phrase boundaries, emphasized words, and emotional peaks, the places where humans vary their delivery the most and where expression matters the most. So MSE-trained TTS is not uniformly slightly-flat; it is specifically flat at the moments that carry the most meaning, monotone questions, unstressed emphasis, deadpan punchlines. The failure is concentrated exactly where you would most notice it. Anyone who has listened to early concatenative or first-generation neural TTS has heard this: it is fine on neutral declarative sentences and falls apart the instant the sentence needs to *do* something emotionally.

![A before and after comparison contrasting a deterministic MSE model collapsing to a flat mean contour with a stochastic or latent model picking one lively mode](/imgs/blogs/prosody-emotion-and-expressive-speech-2.png)

The figure makes the fix obvious in shape. The left column is the failure: many valid contours, an L2 loss that averages them into $\mathbb{E}[y \mid x]$, and a flat monotone output. The right column is every solution we will discuss: instead of predicting one deterministic value, make the model **stochastic** so it can pick a single *mode* rather than the mean. There are three families of fix, and they correspond to the three big sections ahead. You can give the model **extra conditioning** that disambiguates which mode you want, a reference clip, a style label, a text description, so that $p(y \mid x, c)$ is no longer multimodal once $c$ is fixed. You can make the model **variational or sampling-based**, so that it learns the full distribution $p(y \mid x)$ and *samples* a mode rather than averaging. Or you can do both. Every expressive-TTS technique in the literature is one of these two moves, and now you can see precisely why each of them works: they all turn a multimodal regression that collapses to the mean into either a conditioned unimodal problem or a sampling problem that draws a real mode.

This connects directly to why the modern codec-LM and flow-matching TTS systems sound expressive almost as a side effect, a point we will return to. A model like [VALL-E or F5-TTS](/blog/machine-learning/audio-generation/zero-shot-voice-cloning-and-the-tts-frontier) does not regress to a mel with MSE. An autoregressive codec LM samples discrete tokens from a learned categorical distribution; a flow-matching model integrates a stochastic-at-the-start ODE. Both are sampling from $p(y \mid x)$ rather than averaging it. They get prosody variation for free precisely because they never collapse to the mean in the first place. The flatness problem is specifically a deterministic-regression problem, and the field's move toward generative TTS is, among other things, a move away from it.

## Extracting prosody from a reference

Most of the practical techniques below need to *measure* prosody, either to condition on a reference or to predict it explicitly. So before any modeling, the concrete first skill is pulling an F0 contour and an energy envelope off a waveform. This is pure DSP and `torchaudio` gives it to you directly.

Pitch tracking is the harder of the two. The standard modern algorithm is **pYIN** (probabilistic YIN), which estimates F0 frame by frame and, importantly, also estimates **voicing**, whether each frame is voiced (vocal folds vibrating, so F0 is defined) or unvoiced (a fricative like "s", silence, where F0 is meaningless). Getting voicing right matters, because if you treat unvoiced frames as F0 = 0 and average them in, you corrupt the contour. Here is the extraction in `torchaudio`:

```python
import torch
import torchaudio
import torchaudio.functional as F

# Load a reference clip; resample to 22.05 kHz, a common TTS rate.
wav, sr = torchaudio.load("reference.wav")
wav = torchaudio.functional.resample(wav, sr, 22050)
wav = wav.mean(0, keepdim=True)  # mono
sr = 22050

# --- F0 contour via probabilistic YIN ---
# Returns per-frame pitch (Hz), and a voicing flag per frame.
f0, voiced_flag, voiced_prob = torchaudio.functional.detect_pitch_frequency, None, None
# detect_pitch_frequency is a simple tracker; for pYIN quality use the
# dedicated transform below, which exposes voicing.
pitch = F.detect_pitch_frequency(wav, sample_rate=sr, frame_time=0.01)
# pitch: (1, num_frames), one F0 estimate every 10 ms.

# --- Energy envelope: frame RMS in dB ---
n_fft, hop = 1024, 256
spec = torch.stft(
    wav.squeeze(0), n_fft=n_fft, hop_length=hop,
    window=torch.hann_window(n_fft), return_complex=True,
)
# Energy per frame = sum of magnitude^2 across frequency bins.
energy = spec.abs().pow(2).sum(0).sqrt()            # (num_frames,)
energy_db = 20 * torch.log10(energy.clamp(min=1e-5))

print("frames:", pitch.shape[-1], energy.shape[-1])
print("median voiced F0 (Hz):",
      pitch[pitch > 50].median().item())            # ignore unvoiced ~0
print("energy range (dB):",
      (energy_db.max() - energy_db.min()).item())
```

A couple of things are worth flagging because they bite people. First, the hop length, here 256 samples at 22.05 kHz, sets your prosody frame rate to about 86 frames per second, one F0 and one energy value every roughly 11.6 ms. That is fine for prosody, which moves slowly relative to the waveform. Second, you almost always want F0 in a **log** domain before you model it, because pitch is perceived logarithmically: an octave is a doubling of frequency, so a jump from 100 to 120 Hz sounds like the same interval as 200 to 240 Hz. Modeling raw linear F0 over-weights the high end. Most systems predict $\log F0$ and normalize it per speaker, subtracting that speaker's mean log-pitch, so the model learns relative contour shape rather than absolute pitch, which is a speaker-identity thing, not a prosody thing.

The per-speaker normalization deserves a precise statement, because getting it right is the difference between a model that transfers prosody cleanly and one that drags timbre around. For a speaker $s$ with a set of voiced frames, compute the mean and standard deviation of log-F0 over *that speaker's* training audio, $\mu_s$ and $\sigma_s$, and store them. Then the modeling target is the standardized contour

$$
\tilde{f}(t) = \frac{\log f_0(t) - \mu_s}{\sigma_s},
$$

which is dimensionless, zero-mean, and unit-variance *per speaker*. Now $\tilde f$ captures pure contour *shape*, is this frame high or low relative to how this speaker normally sounds, with the speaker's absolute pitch ($\mu_s$) and pitch range ($\sigma_s$) factored out into two scalars that belong to identity, not prosody. At synthesis you predict $\tilde f$, then invert: $f_0(t) = \exp(\sigma_s \tilde f(t) + \mu_s)$ using the *target* speaker's statistics. This is exactly how you transfer the *shape* of an excited contour from one speaker onto another speaker's voice without making the second speaker suddenly pitch up into the first speaker's range, you carry $\tilde f$ across but keep the target's $\mu_s, \sigma_s$. Skip this normalization and your prosody transfer leaks pitch range, which is a big chunk of what listeners hear as "this doesn't sound like the same person anymore."

The pitch tracker itself deserves a word on *why* it can fail, because pitch-tracking failures are the most common silent corruptor of prosody data. The core difficulty is that a periodic signal at frequency $f_0$ also has energy at $2f_0, 3f_0, \dots$ (the harmonics), and an autocorrelation-based tracker can lock onto a harmonic instead of the fundamental, reporting $2f_0$ (an **octave-up error**) or, via sub-harmonics, $f_0/2$ (an **octave-down error**). YIN's contribution was a *cumulative mean normalized difference* function that suppresses the harmonic peaks relative to the fundamental peak; pYIN's contribution was to make it probabilistic, running several candidate thresholds and using a hidden Markov model over time to pick a *smooth* F0 track, which kills the isolated octave jumps that a frame-independent tracker produces. The practical takeaways: always set `fmin`/`fmax` to your speaker's plausible range (an adult male might be 60 to 300 Hz, a child up to 500 Hz), which alone removes most octave errors by making the wrong octave out-of-range, and always check the voicing flag, because a tracker reporting a confident 200 Hz on a frame that is actually silence will inject garbage into your contour statistics.

For higher-quality pitch with explicit voicing probabilities you would reach for `librosa.pyin`, which returns the F0 array, a boolean voiced flag, and a per-frame voicing probability, and it lets you set `fmin` and `fmax` to the plausible range of your speaker (a great way to avoid octave errors, the classic pitch-tracker failure where the algorithm locks onto a harmonic at double or half the true F0). The energy side is so cheap, just a windowed RMS, that everyone computes it inline as above.

#### Worked example: reading prosody off two takes

Suppose you record yourself saying "absolutely not" twice, once flat and dismissive, once sharp and outraged, and run the code above. On the flat take you might see a median voiced F0 of 112 Hz with an F0 standard deviation across the utterance of about 9 Hz, and an energy range of roughly 14 dB. On the outraged take the median F0 jumps to 148 Hz, the F0 standard deviation widens to 31 Hz (the contour is swinging far more), and the energy range opens to 22 dB with a sharp peak on the stressed first syllable of "absolutely." Those are exactly the numbers a deterministic model would average away, the flat take and the outraged take differ by tens of hertz and several dB, and their mean is a third contour belonging to neither. Now you can see what a reference encoder is for: it lets the model copy the 148 Hz, wide-swing, 22 dB take *specifically*, instead of guessing.

## Reference encoders and global style tokens

The first family of fixes adds conditioning so the distribution stops being multimodal. The cleanest version of this is the **reference encoder**: a small network that takes an expressive reference clip and distills it into a single fixed-size **style vector** that conditions the acoustic model. The original idea comes from Skerry-Ryan et al.'s prosody-transfer work and the Wang et al. **Global Style Tokens** paper, both out of Google in 2018, and the design has propagated into nearly every expressive TTS system since.

![A dataflow graph showing a reference clip and target text feeding separate encoders, with a fixed-size style vector and text states merging in the acoustic model to produce an expressive mel](/imgs/blogs/prosody-emotion-and-expressive-speech-3.png)

The figure shows the wiring. The reference clip, which can be any length, goes through a **reference encoder**, typically a stack of strided 2D convolutions over the reference's mel-spectrogram followed by a GRU whose final hidden state you keep. That final state is the **style vector**, a single fixed-size embedding, say 256 dimensions, that has compressed "how this clip was delivered" into one tensor regardless of how long the clip was. Meanwhile the target text, which can be totally different from whatever the reference said, goes through the normal text encoder into phoneme hidden states. The acoustic model then conditions on *both*, the phoneme states for *what to say* and the style vector for *how to say it*, and produces a mel that has the reference's delivery applied to the new words. This is **prosody transfer**: clone the style of one clip onto arbitrary text.

The reason a fixed-size vector works is the key insight. Style, unlike content, is genuinely a *global* property of a clip, the overall pitch range, the energy, the speaking rate, the emotional register are roughly constant across the utterance, so they compress into one vector without much loss. Content is time-varying and could never compress this way; style mostly can. The reference encoder is, in information terms, a bottleneck that is deliberately too small to carry the words, so it is forced to throw away content and keep only the global delivery characteristics. That bottleneck *is* the design. Make it too big and it starts to leak content (the model learns to copy words from the reference, which you do not want); make it too small and it cannot represent enough styles. The 128-to-256-dimensional range is the empirical sweet spot.

**Global Style Tokens** add one beautiful refinement on top. Instead of letting the style vector be any point in a 256-dimensional space, GST introduces a small bank of learned **style tokens**, say ten of them, and the reference encoder's output is used as a *query* in an attention over that bank. The final style embedding is an attention-weighted *mixture* of the tokens.

![A dataflow graph where a reference embedding queries a small bank of three learned style tokens via attention, producing a weighted mixture that conditions the TTS model](/imgs/blogs/prosody-emotion-and-expressive-speech-4.png)

The figure shows the attention. The reference embedding queries the bank; the attention produces weights over the tokens, here something like 0.6 on token one, 0.3 on token two, and so on; and the weighted mixture becomes the style embedding that conditions the model. Why bother with this indirection? Three big wins. First, **interpretability**: after training (which is fully unsupervised, no style labels needed), the individual tokens often line up with human-meaningful axes, one token controls speed, another pitch height, another energy, because the model discovers a basis for the style space. Second, **steerability at inference**: you can *skip the reference entirely* and just dial token weights by hand. Want a faster, higher-pitched delivery? Crank the weight on the speed token and the pitch token. You get knob-level control with no reference clip at all. Third, **smoothing**: the mixture is constrained to the convex hull of the learned tokens, which makes the style space better-behaved and less prone to the artifacts you get when you push a raw style vector into a region the model never saw in training.

Here is the reference encoder in PyTorch, compact but real:

```python
import torch
import torch.nn as nn

class ReferenceEncoder(nn.Module):
    """Conv stack over a reference mel -> single style vector."""
    def __init__(self, n_mels=80, style_dim=256, gru_dim=128):
        super().__init__()
        chans = [1, 32, 32, 64, 64, 128, 128]
        self.convs = nn.ModuleList([
            nn.Conv2d(chans[i], chans[i + 1], kernel_size=3,
                      stride=2, padding=1)
            for i in range(len(chans) - 1)
        ])
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(c) for c in chans[1:]])
        # After 6 stride-2 convs the mel axis (80) shrinks to ~2.
        self.gru = nn.GRU(128 * 2, gru_dim, batch_first=True)
        self.proj = nn.Linear(gru_dim, style_dim)

    def forward(self, ref_mel):                  # (B, n_mels, T)
        x = ref_mel.unsqueeze(1)                 # (B, 1, n_mels, T)
        for conv, bn in zip(self.convs, self.bns):
            x = torch.relu(bn(conv(x)))
        # x: (B, C, n_mels', T'); flatten freq into channels, GRU over time
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        _, h = self.gru(x)                       # last hidden state
        return self.proj(h.squeeze(0))           # (B, style_dim)


class GST(nn.Module):
    """Reference embedding queries a learned token bank (multi-head attn)."""
    def __init__(self, style_dim=256, n_tokens=10, n_heads=8):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(n_tokens, style_dim))
        self.attn = nn.MultiheadAttention(style_dim, n_heads,
                                          batch_first=True)

    def forward(self, ref_embed):                # (B, style_dim)
        q = ref_embed.unsqueeze(1)               # (B, 1, style_dim)
        k = v = self.tokens.unsqueeze(0).expand(q.size(0), -1, -1)
        out, weights = self.attn(q, k, v)        # mixture + token weights
        return out.squeeze(1), weights           # (B, style_dim), weights
```

In a full system you would concatenate or add the GST output to the encoder's phoneme states (broadcast across time, since it is global) before the decoder, the same global-conditioning injection covered in [the conditioning post](/blog/machine-learning/audio-generation/conditioning-and-control-in-audio-generation). The whole thing trains end to end on the normal reconstruction loss; the style supervision is implicit, the model needs the style vector to reconstruct the reference's prosody, so it learns to encode prosody into it.

The honest caveat: reference-based prosody transfer is **leaky**. The style vector also carries some speaker identity and even some content bleed, so transferring style from speaker A's clip onto speaker B's voice can drag B's timbre slightly toward A, and the cleanliness of the disentanglement is never perfect. It is good enough to be extremely useful and never good enough to be a clean knob. That is the recurring tax on every soft-control method here.

There is a practical question that comes up the moment you try to *use* GST tokens as knobs: how do you find which token does what? The honest answer is that you discover it empirically, because the tokens are learned without labels and the model assigns them whatever basis is most efficient for reconstruction, which is not guaranteed to align with human-nameable axes. The standard procedure is a small sweep: hold the text fixed, set the style embedding to each token in isolation (a one-hot weight), synthesize, and listen, then measure the resulting F0 mean, F0 range, energy, and speaking rate on each output. You will typically find that two or three tokens have a clear, large effect on one measurable axis each, token 4 raises the speaking rate by 30 percent, token 7 lifts the pitch baseline, and the rest have weaker or entangled effects. You name the clear ones, expose them as sliders, and ignore the muddy ones. This is exactly the kind of post-hoc interpretation that makes GST attractive for a product, you ship the two or three reliable axes as "speed" and "pitch" sliders, and you get a controllable, reference-free style interface out of an entirely unsupervised training run. The limitation is that you only get the axes the model happened to learn; if you need an axis it did not discover (a clean "warmth" knob, say), GST will not hand it to you and you are back to explicit control or a description model.

A related design choice worth knowing is **how many tokens** and **whether to use multi-head attention**. Too few tokens (two or three) and the style space is too coarse to capture the range of deliveries in your data; too many (thirty-plus) and the tokens start to overfit to individual training clips, losing the smooth interpolation that made the convex-hull constraint valuable. The original ten with eight attention heads is a reasonable default, and the multi-head structure matters more than it looks: each head attends over the same bank but with its own projection, so the final embedding is a *concatenation* of several independent mixtures, which gives the style space more capacity than a single mixture of ten scalars would. If your expressive data is rich, more heads help; if it is small, fewer heads regularize.

## The variational view: modeling the distribution

Reference encoders disambiguate by handing the model extra information. The complementary move is to model the multimodal distribution $p(y \mid x)$ *itself* and sample from it, which is the **variational** approach. This is worth a section because it is the principled answer to the one-to-many problem and because it connects expressive TTS to the [VAE machinery](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) from the image series.

The idea: introduce a latent variable $z$ that captures "which prosody," and model $p(y \mid x) = \int p(y \mid x, z)\, p(z)\, dz$. During training you use a VAE-style encoder $q(z \mid y, x)$ to infer the latent from the *actual* prosody of each training example, train the decoder to reconstruct $y$ from $x$ and that $z$, and regularize $q$ toward a prior $p(z)$ with a KL term. At inference, you have two options: sample $z \sim p(z)$ to get a *random* believable prosody, or infer $z$ from a reference clip to *copy* a specific one. The same latent serves both sampling and transfer. This is exactly the structure inside [VITS](/blog/machine-learning/audio-generation/text-to-speech-from-tacotron-to-vits), whose flow-based latent and stochastic duration predictor are precisely a variational answer to prosody's one-to-many nature, and inside the "variational prosody" and "latent prosody" lines of work.

The reason this fixes flatness is the same reason CFG-free sampling fixes it, restated in the cleanest form: a deterministic decoder conditioned on a sampled $z$ is no longer averaging over the modes, because for any *fixed* $z$ the distribution $p(y \mid x, z)$ is approximately unimodal. The latent $z$ has absorbed the multimodality. You traded "predict the multimodal $y$" (which collapses) for "predict the unimodal $y$ given $z$, and sample $z$" (which does not). That trade is the whole trick, and it is why every non-collapsing approach, variational latents, autoregressive token sampling, flow matching, diffusion, can be read as different ways of refusing to integrate out the prosody before committing to a value.

There is a knob here too, and it is one you will actually turn in production: the **temperature** or variance of the $z$ you sample. Sample $z$ near the prior mean and you get a safe, slightly-flat reading (you are creeping back toward the mean). Sample with full prior variance and you get maximal expressiveness but also more risk of a weird or unstable contour. The stochastic duration predictor in VITS exposes exactly this as a noise scale you can dial down for stability or up for liveliness. This temperature is the single most important control on the fidelity-versus-expressiveness trade, and we will see it reappear when we get to codec-LM sampling.

## Explicit prosody control: F0, energy, and duration as knobs

Reference encoders and latents give you *style*, but as a blob, you transfer a whole delivery or sample a whole contour, with no way to say "keep everything but raise the pitch ten percent on this one word." For that fine, surgical control you want the opposite design philosophy: predict prosody as **explicit, named, editable variables**. This is the **FastSpeech 2** approach (Ren et al., 2020), and it is the workhorse of controllable TTS.

![A dataflow graph where phoneme encoder states feed separate duration, pitch, and energy predictors whose outputs can be edited before a mel decoder and vocoder](/imgs/blogs/prosody-emotion-and-expressive-speech-5.png)

The figure shows the architecture, which FastSpeech 2 calls the **variance adaptor**. From the phoneme encoder's hidden states, three small predictors branch off, each a couple of convolution layers with a linear head. The **duration predictor** outputs how many output frames each phoneme should occupy; this is used to *expand* the phoneme sequence to the frame rate (the length-regulator step) and is the thing that controls rhythm and speaking rate. The **pitch predictor** outputs an F0 value per frame, and the **energy predictor** outputs an energy value per frame. The predicted pitch and energy are quantized, embedded, and *added* to the expanded hidden states, so the decoder sees the phoneme content plus an explicit pitch embedding plus an explicit energy embedding at every frame. Then the mel decoder and the vocoder run as usual.

Why this design beats the implicit approaches for control: because pitch, energy, and duration are now *exposed as tensors you can edit before the decoder runs*. Want to slow the whole sentence by twenty percent? Multiply the predicted durations by 1.2. Want to raise the pitch? Scale the predicted F0. Want to emphasize the third word? Bump its duration and its energy and its pitch peak. You are no longer hoping a style blob does what you want; you are editing the actual prosodic curves with arithmetic. This is the "edit and scale" node in the figure, and it is the most directly useful control in all of expressive TTS for production systems that need *deterministic, reproducible* prosody edits.

And, crucially, FastSpeech 2 *also* dodges the one-to-many collapse, in a clever way. Recall the problem was that the model had to predict prosody from text alone, which is ambiguous. FastSpeech 2's answer during training is to feed the **ground-truth** pitch, energy, and duration (extracted from the real audio, exactly as we did in the extraction section) into the decoder, so the decoder is never asked to guess them; it just renders the mel given known prosody. The variance predictors are trained as a *separate* regression on the side. So the mel decoder is conditioned on real prosody and never averages, and the predictors give you a *default* prosody at inference that you are free to overwrite. The genius is the separation: the part that could collapse (predicting prosody) is decoupled from the part that renders audio, and you can replace the prediction with anything you like, the model's guess, a reference's contour, or a hand-edited curve.

Here is the variance adaptor's pitch path and the edit hook in code:

```python
import torch
import torch.nn as nn

class VariancePredictor(nn.Module):
    """Predicts one scalar per phoneme/frame (used for dur/pitch/energy)."""
    def __init__(self, dim=256, kernel=3, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel, padding=kernel // 2),
            nn.ReLU(), nn.LayerNorm(dim) if False else nn.Identity(),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel, padding=kernel // 2),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(dim, 1)

    def forward(self, h):                     # h: (B, dim, T)
        y = self.net(h).transpose(1, 2)       # (B, T, dim)
        return self.head(y).squeeze(-1)       # (B, T)


def apply_prosody_edits(pitch, energy, duration,
                        pitch_scale=1.0, energy_scale=1.0,
                        speed=1.0, emphasis=None):
    """Edit the predicted prosody curves before the decoder.
    pitch/energy/duration are predicted per-phoneme tensors (B, T)."""
    # Global pitch shift in the log domain = multiply in linear Hz.
    pitch = pitch * pitch_scale
    energy = energy * energy_scale
    # speed > 1 slows down (more frames); < 1 speeds up.
    duration = (duration * speed).round().clamp(min=1).long()
    if emphasis is not None:                  # emphasis: (idx, gain)
        idx, gain = emphasis
        pitch[:, idx] = pitch[:, idx] * gain
        energy[:, idx] = energy[:, idx] * gain
        duration[:, idx] = (duration[:, idx].float() * gain).round().long()
    return pitch, energy, duration

# Example: speak 20% slower, lift pitch 8%, emphasize phoneme 5.
# pitch, energy, duration = adaptor(phoneme_states)   # predicted curves
# pitch, energy, duration = apply_prosody_edits(
#     pitch, energy, duration,
#     pitch_scale=1.08, speed=1.2, emphasis=(5, 1.3))
```

That `apply_prosody_edits` function is, in spirit, the entire control surface that products like Amazon Polly and Microsoft's TTS expose through **SSML** (Speech Synthesis Markup Language), the `<prosody rate="slow" pitch="+8%">` tags map almost exactly onto scaling the duration and pitch curves. SSML is just a standardized, text-level front-end for the same edits. Knowing the variance adaptor is underneath demystifies why SSML can do some things (rate, pitch, volume, emphasis, breaks, all curve edits) and not others (it cannot make the voice "sound sarcastic," because sarcasm is not a single curve scale, it is a learned style that lives in the blob-style methods).

There is a quantization detail in the variance adaptor that trips people up when they implement it. FastSpeech 2 does not add the raw continuous F0 value to the hidden state; it *quantizes* pitch and energy into a fixed number of bins (256 is typical), spaced in the log domain for pitch, and learns an *embedding* per bin, which it adds to the hidden state. Why bins and embeddings instead of just adding the scalar? Because a learned embedding lets the model represent a *nonlinear* relationship between pitch and the acoustics, the effect of raising pitch is not a simple additive shift on the mel, it interacts with the vowel and the speaker, and an embedding table can capture that where a scalar add cannot. The practical consequence for your edits: when you scale the predicted F0, you re-quantize into the bins before embedding, so very small pitch edits below the bin width have no effect, and you should reason in bin-sized steps if you want fine control. This is a small thing that causes a lot of "my pitch edit did nothing" confusion.

The cost of explicit control is in the table later, but state it now: explicit per-phoneme prosody is the *most controllable* method and carries a *medium* naturalness cost, because hand-edited curves can fall outside what the vocoder and decoder saw in training and introduce a slight unnaturalness, and because the predictors, trained with their own MSE, are themselves a little flat by default, the very mean-collapse we derived, now confined to the prosody *predictors* rather than the whole model. That confinement is the design's cleverness: the flatness is quarantined in a component you are free to overwrite, so it sets a dull *default* but never a ceiling. You get surgical control and you pay for it in the last few percent of naturalness if you push the edits hard, and the further your edited curve sits from the training distribution, the more the vocoder, which also only saw realistic prosody, may render it with a slight roughness.

## Emotion control: labels, intensity, and the limits of categories

So far the controls have been geometric, pitch up, slower, this reference's style. But the thing product teams actually ask for is **emotion**: "make it sound happy," "make it sad," "make it angry but only a little." Emotion control is its own sub-field, and it splits into two approaches that mirror the discrete-versus-continuous tension we keep hitting.

The first and simplest is the **emotion label**: train on a dataset where each utterance is tagged with one of a small set of categories, and condition the model on a learned **emotion embedding** keyed by that label. The canonical datasets are **ESD** (Emotional Speech Database, ten English and ten Mandarin speakers across five emotions: neutral, happy, sad, angry, surprise) and the older **EmoV-DB**. At inference you pick "happy" and the model applies the happy embedding, which has learned to produce the higher pitch, wider range, and faster tempo that happiness tends to carry. This is dead simple, robust, and exactly as limited as it sounds: you get however many emotions you had labels for, no in-between, and the categories are a crude quantization of a continuous emotional space. Real speech is not one of five emotions; it is a blend with an intensity.

So the second refinement is **intensity control**. The trick that works well is to treat neutral as the origin and an emotion as a *direction*: learn the emotion embedding, then **interpolate** between the neutral embedding and the full emotion embedding to get intermediate intensities. A scale of 0.0 is neutral, 1.0 is full anger, 0.4 is mildly annoyed. Some systems formalize this with a "relative attributes" ranking (Zhou et al.'s emotion-intensity work), learning a per-emotion intensity axis so that the interpolation is perceptually monotonic, dialing the scalar up reliably sounds *more* of that emotion rather than jumping around. Concretely:

```python
import torch
import torch.nn as nn

class EmotionConditioner(nn.Module):
    """Categorical emotion embedding with continuous intensity control."""
    def __init__(self, n_emotions=5, dim=256):
        super().__init__()
        self.emb = nn.Embedding(n_emotions, dim)
        self.neutral_id = 0                       # index of 'neutral'

    def forward(self, emotion_id, intensity=1.0):
        """intensity in [0,1]: interpolate neutral -> full emotion."""
        target = self.emb(emotion_id)             # (B, dim)
        neutral = self.emb(
            torch.full_like(emotion_id, self.neutral_id))
        # Linear interpolation = direction from neutral, scaled.
        return neutral + intensity * (target - neutral)

# cond = EmotionConditioner()
# happy_mild   = cond(torch.tensor([1]), intensity=0.3)
# angry_strong = cond(torch.tensor([3]), intensity=0.9)
# Inject `cond` like any global conditioning (broadcast over phoneme states).
```

The honest limits of emotion labels are worth dwelling on because they explain why the field moved past them. Categories are coarse, and they are also **culturally and individually variable**, one speaker's "angry" is another's "firm," and the boundary between "surprise" and "fear" in the prosody is genuinely fuzzy. Worse, a single emotion label for a whole utterance cannot express the *trajectory* of emotion within a sentence, the way a sentence can start neutral and end with a flash of frustration on the last word. Emotion labels give you a constant emotional color per clip, which is already a big step up from flat, but it is nowhere near the expressiveness of human speech, where emotion is woven through the prosody at the word level. The interpolation trick buys intensity but not trajectory. Solving trajectory pushes you back toward either time-aligned explicit control or, more powerfully, toward describing what you want in words, which is the next section.

## Description-driven style: telling the model in plain English

The most flexible control to emerge recently sidesteps both fixed categories and reference clips: just *describe* the voice and delivery you want in a sentence of natural language, and let a text encoder turn that description into the conditioning. "A calm, low-pitched male voice speaking slowly, slightly sad, recorded in a quiet room." This **description-driven** or **instruction-style** TTS is the design behind **Parler-TTS** (an open reproduction of the approach in Lyth and King's 2024 "natural language guidance" work) and is increasingly how the frontier commercial systems expose style.

![A dataflow graph where a free-text style description and a transcript feed separate encoders into a codec language model that decodes to styled speech matching the description](/imgs/blogs/prosody-emotion-and-expressive-speech-7.png)

The figure shows why this is elegant. There are two text inputs. One is the **transcript**, the words to actually speak. The other is the **style description**, free text describing the delivery. The description goes through a text encoder, typically a frozen or lightly-tuned **T5 / Flan-T5**, exactly the same kind of text encoder the [image and music models](/blog/machine-learning/audio-generation/conditioning-and-control-in-audio-generation) use for prompts. Its output conditions a **codec language model** (a [VALL-E-style](/blog/machine-learning/audio-generation/neural-codec-language-model-tts-vall-e) AR model over DAC or EnCodec tokens) via cross-attention, alongside the transcript. The codec LM samples tokens, and a codec decoder turns them into the waveform. The style description steers the *how*; the transcript fixes the *what*.

The reason this works at all is a data trick as much as a modeling one. You cannot collect a dataset of speech where each clip is hand-annotated with a rich natural-language style description, that would be impossibly expensive. Instead, you *extract* the attributes automatically, pitch level (from the F0 extraction we did earlier), speaking rate (from duration), reverberation, signal-to-noise ratio, gender, and so on, **bin** each into coarse buckets ("low-pitched," "fast," "very clean"), and then *template* those buckets into a sentence: "a low-pitched voice speaking quickly in a clean recording." Train on (description, transcript, audio) triples built this way, and the T5 encoder learns to map the *space of such sentences* to the right conditioning. At inference you can write descriptions the model never saw verbatim, because T5 generalizes over paraphrase, and it largely works. Parler-TTS's released models were trained on tens of thousands of hours with descriptions generated exactly this way.

Here is the inference call, which is strikingly simple given the power:

```python
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ParlerTTSForConditionalGeneration.from_pretrained(
    "parler-tts/parler-tts-mini-v1").to(device)
tok = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

transcript = "We are absolutely thrilled you could make it tonight."
description = ("A calm female voice speaks slowly with a warm, "
              "slightly cheerful tone in a very clean recording.")

desc_ids = tok(description, return_tensors="pt").input_ids.to(device)
text_ids = tok(transcript, return_tensors="pt").input_ids.to(device)

with torch.no_grad():
    audio = model.generate(
        input_ids=desc_ids,             # the STYLE description
        prompt_input_ids=text_ids,      # the WORDS to speak
        do_sample=True, temperature=1.0,
    )
sf.write("styled.wav",
         audio.cpu().numpy().squeeze(),
         model.config.sampling_rate)
```

Two things to notice. First, the two text inputs go in *different* arguments, `input_ids` for the description, `prompt_input_ids` for the transcript, because they play structurally different roles (cross-attention condition versus decoder prompt). Second, `do_sample=True` with a temperature, because this is a codec LM and, per the one-to-many discussion, sampling is what gives it prosodic life. Set `do_sample=False` and you greedily decode toward the high-probability, flatter reading and lose much of the expressiveness, the exact mean-collapse problem reappearing at decode time. The description constrains *which* distribution you sample from; the sampling draws a live mode from it. Both are needed.

Description-driven control is wonderful for *coarse-to-medium* style, mood, pace, pitch register, recording quality, and it composes naturally with the words a human would actually use. Its weakness is precision: you cannot say "raise the pitch exactly eight percent on the word 'thrilled'" in a description and expect it to land, because T5 has no notion of "eight percent" or "the third word." Description sets the overall delivery; for surgical edits you still want the explicit F0-energy-duration knobs. The two are complementary, and a mature system offers both.

The data pipeline behind a description model is worth spelling out because it is reproducible and it explains both the power and the limits. You start with a large untagged speech corpus, tens of thousands of hours, and you run a battery of *automatic attribute extractors* over every clip. Pitch level comes from the median F0 of the extraction we did earlier, binned into "very low / low / moderate / high / very high." Speaking rate comes from phonemes-per-second (transcribe with an ASR, count phonemes, divide by duration), binned into "very slowly / slowly / moderately / quickly." Reverberation and noise come from cheap signal estimators (a reverberation-time estimate, a signal-to-noise ratio), binned into "very clean / clean / slightly noisy." Gender and a coarse expressivity score come from small classifiers. Now each clip has a *bag of categorical attributes*. The final step is to render that bag into a natural sentence with a template-plus-paraphrase generator, sometimes a small language model is used to vary the phrasing so the descriptions are not all identical, producing strings like "a clean recording of a woman speaking quite slowly with low pitch." Train the codec LM on the resulting (description, transcript, audio) triples and the T5 encoder learns the *mapping from this sentence space to the conditioning*.

This pipeline is exactly why the precision limit exists and is not a temporary engineering gap. The model only ever saw *binned, templated* descriptions, so its understanding of "low pitch" is the centroid of the "low" bin, not a continuous function of hertz. Ask for "pitch exactly 132 Hz" and there is no training signal that ever connected that phrase to an F0, so the model falls back to whatever "low" or "moderate" it pattern-matches. The flip side is that this binning is *why generalization to unseen descriptions works*: the model learned a handful of coarse axes (pitch, rate, noise, gender, warmth) and T5's paraphrase-invariance lets you hit those axes with words you choose, "hushed," "gentle," and "soft" all land near the low-energy region even if the exact word was rare in training, because T5's embedding puts them near each other. You are steering along the axes the extraction pipeline measured, in whatever words map to those axes. Knowing the pipeline tells you precisely what a description model can and cannot do: it can do anything the extractors measured, at the resolution they binned, and nothing finer.

## Expression for free: sampling and in-context prompts

We have circled this point several times and now we land on it directly, because it is the most important practical fact about expressive TTS on the 2024-2026 frontier: the best-sounding expressive speech today often uses *none* of the explicit prosody machinery above. The modern [codec language models and flow-matching TTS systems](/blog/machine-learning/audio-generation/zero-shot-voice-cloning-and-the-tts-frontier), VALL-E, F5-TTS, NaturalSpeech 3, XTTS, get rich, natural prosody as an emergent property of two design choices, and they are choices we have already explained.

The first is **sampling**. A codec LM models $p(\text{tokens} \mid \text{text, prompt})$ as a sequence of categorical distributions and *samples* from them; a flow-matching model integrates an ODE from a noise sample. Neither one ever computes the conditional mean, so neither one collapses to flat. Per the formalization earlier, sampling from a multimodal $p(y \mid x)$ draws a *real mode*, a contour a human actually might produce, every time. You pay nothing extra for this; it falls out of the generative formulation. This is the deepest reason the field moved from regression-style acoustic models (Tacotron, FastSpeech) toward sampling-style ones for expressiveness: the regression models needed all the GST and variance-adaptor scaffolding *to undo a problem they created*, and the sampling models simply never created it.

The second is the **in-context acoustic prompt**. A zero-shot voice-cloning model is given a few seconds of reference audio as a *prefix* of codec tokens and asked to continue in the same voice. Here is the subtle and powerful part: that prefix does not only carry the *speaker's timbre*, it carries the *speaker's prosody in that clip*, the pitch range, the energy, the pace, the emotional register. The model, being an in-context learner over the whole prefix, naturally continues the *style* of the prompt and not just its voice. So if your three-second reference is delivered with bright, excited energy, the cloned speech inherits that excitement; if the reference is a calm whisper, the output stays calm. You control prosody *by choosing the prosody of your prompt*, which is the most natural interface imaginable, you show it an example of the delivery you want. This is reference-encoder-style prosody transfer, achieved without a reference encoder at all, just by putting the clip in context.

The temperature knob returns here as the master control. Decode the codec LM with low temperature (or greedy) and you push toward the high-probability tokens, which are the *average* tokens, and the output flattens, the mean-collapse problem sneaking back in through the sampler. Decode with temperature around 1.0 and you get lively, varied prosody. Push temperature too high and you get instability: mispronunciations, dropped words, a contour that wanders weirdly, sometimes the model losing the thread entirely and babbling. The expressiveness-versus-stability trade is *literally* a temperature dial on these models, and tuning it per use case (lower for an audiobook narrator you need rock-steady, higher for a character voice you want lively) is one of the real jobs of deploying them. F5-TTS exposes it directly; XTTS exposes a `temperature` and a `repetition_penalty` that together govern exactly this.

#### Worked example: the temperature dial on a codec LM

Take a zero-shot clone with a fixed five-second reference and the sentence "I can't believe we actually won." Decode at temperature 0.3 and you might measure an F0 standard deviation of 12 Hz across the utterance, intelligibility essentially perfect (WER near 0 percent via a Whisper transcription), and listeners describing it as "clear but a bit flat." Decode the same prompt at temperature 0.95 and the F0 standard deviation widens to roughly 34 Hz (the voice now swings up excitedly on "won"), intelligibility still strong (WER perhaps 1 to 2 percent), and listeners describing it as "genuinely excited." Push to temperature 1.4 and the F0 standard deviation is high but erratic, and WER jumps to 8 percent or more as words start to slur or drop, the model has traded stability for variation past the useful point. Those three points are the entire expressiveness-stability curve, and the sweet spot, usually somewhere around 0.7 to 1.0, is what you tune. No GST, no variance adaptor, no emotion label, just a sampler temperature and a well-chosen prompt.

### Stress-testing the prompt-driven approach

The prompt-plus-temperature approach is the frontier default, but it is worth stress-testing it on the cases that actually break it in production, because each failure mode points at when you need to fall back to explicit control. The first stress test: **what happens when the prompt is three seconds of noisy audio**, a phone recording with background chatter? The model is an in-context learner, so it does not only inherit the speaker's prosody, it inherits the *recording conditions*, the noise floor and reverberation get cloned right along with the voice, and you get a clean-text rendering delivered as if recorded in the same noisy room. Worse, the prosody itself becomes unreliable, because a noisy prompt gives the model a noisy estimate of the speaker's pitch range and pace, and it can latch onto an artifact as if it were a stylistic choice. The mitigation is to *denoise the prompt first* or to use a description model instead, where you state the delivery you want explicitly rather than demonstrating it with a degraded clip.

The second stress test: **what happens when you need a specific, repeatable emphasis** and the model keeps moving it around? Because the codec LM samples, the *location* of the emphasis is itself sampled, run it five times and the stress lands on a different word each time. For a one-off this is fine, for a brand voice that must emphasize the product name identically in ten thousand renders it is a deal-breaker, the very stochasticity that buys expressiveness costs you reproducibility. This is the single clearest signal that you have left the regime where sampling alone suffices and need an explicit-prosody layer (or at least a fixed seed plus rejection sampling against a target). The third stress test: **what happens at long form**, a two-minute paragraph in one pass? Prosody tends to *drift*, the early sentences are lively and the model gradually relaxes toward its mean as the context fills, so the end of a long passage is noticeably flatter than the start. The fix is to chunk at sentence or paragraph boundaries and re-prompt, which also bounds the stability risk, a topic the [streaming and real-time post](/blog/machine-learning/audio-generation/real-time-streaming-and-full-duplex-speech) develops further. Each of these stress tests resolves the same way: sampling gives you free *average* expressiveness, and the moment you need *specific* or *stable* or *sustained* prosody, you reach for the explicit machinery on top.

## The control-method landscape, side by side

We now have five distinct ways to make speech expressive, and the engineering question is always *which one for this job*. The honest answer is that they sit at different points on a granularity-versus-cost plane, and you pick by how fine a handle you need.

![A matrix comparing five expressive-TTS control methods across granularity, naturalness cost, and the systems that use each](/imgs/blogs/prosody-emotion-and-expressive-speech-6.png)

The matrix lays out the trade. **Reference and GST** give you *clip-level* style at *low* naturalness cost, you transfer a whole delivery and it sounds natural because it is copying a real one, but you cannot edit a single word. **Explicit F0-energy-duration** gives you *per-phoneme* control, the finest granularity available, at a *medium* cost, because hand-edited curves drift from the training distribution and the variance predictors are a little flat by default. **Emotion labels** give you a *discrete category* at *low* cost but coarse resolution, five moods, not a continuum. **Text descriptions** give you *free-text* control at *low-to-medium* cost, flexible and natural to write but imprecise about exact numbers. And **sampling alone** (in modern codec LMs) gives you *whole-clip random* variation at *near-zero* added cost, the expressiveness is free, but you do not *steer* it, you take what the prompt-plus-temperature gives you. The rightmost column names a representative system for each so you can go read the source.

The decision rule that falls out: if you need a *specific, reproducible* prosody edit (a brand voice that must always emphasize the product name the same way), use explicit control. If you need to *match a mood* and have an example clip, use reference or GST. If you want to *describe* the voice in words and accept coarse control, use a description model. If you just want *natural, varied* speech and are cloning from prompts, use a modern codec LM and tune the temperature, and add explicit control on top only where you need surgical edits. Most production systems end up layering: a codec LM for natural baseline expressiveness, plus an explicit-prosody or SSML layer for the few places a human needs to override.

| Control method | Granularity | Steerable without a clip? | Naturalness cost | Representative system |
| --- | --- | --- | --- | --- |
| Reference encoder | Whole-clip style | No (needs reference) | Low | Skerry-Ryan prosody transfer |
| Global Style Tokens | Whole-clip, token axes | Yes (dial tokens) | Low | GST-Tacotron |
| Explicit F0/energy/duration | Per-phoneme | Yes (edit curves) | Medium | FastSpeech 2 |
| Emotion label + intensity | Per-clip category | Yes (pick label) | Low | ESD / EmoV models |
| Text description | Per-clip, free text | Yes (write prompt) | Low-medium | Parler-TTS |
| Sampling + in-context prompt | Per-clip, implicit | Partly (prompt + temp) | Near-zero | VALL-E, F5-TTS, XTTS |

## Evaluating expressiveness honestly

Everything above is worthless if you cannot tell whether it worked, and measuring expressiveness is genuinely the hardest evaluation problem in TTS, much harder than measuring intelligibility. Intelligibility has a clean proxy: run the synthesized speech through an ASR like Whisper, compute word error rate against the target text, done. Expressiveness has no such single number, because "did it sound right" is not a transcription. So you triangulate with several metrics, each of which captures something and misses something.

![A matrix of four expressiveness metrics across what each measures, how, and where each one is blind or biased](/imgs/blogs/prosody-emotion-and-expressive-speech-8.png)

The figure lays out the four metrics and, crucially, where each *lies*. **MOS-naturalness** (Mean Opinion Score, human raters scoring "how natural does this sound" on a 1-to-5 scale, the gold standard for naturalness) is necessary but **blind to wrong mood**: a perfectly natural-sounding *happy* rendering of a *sad* sentence scores high on naturalness while being completely wrong. So MOS alone can rank a flat-but-clean system above an expressive-but-slightly-rougher one, which is exactly backwards for our purpose. **Emotion-recognition accuracy**, running a trained speech-emotion-recognition (SER) classifier on the output and checking whether it detects the intended emotion, measures the *right thing* (did the emotion land) but inherits the **classifier's biases**: the SER model has its own blind spots and you are partly measuring those. **F0 and prosody similarity**, comparing the synthesized F0 contour to a reference contour via RMSE or dynamic-time-warped distance, measures contour *match* but match is not the same as natural, you can match a target contour and still sound robotic if energy and duration are off. And **human A/B preference**, forcing listeners to pick which of two systems they prefer, is the most trustworthy signal for "is A better than B" but is **costly and noisy**, you need many raters and many samples to get a stable result.

The practical harness for expressive TTS evaluation, the one I would actually run, combines them: report MOS-naturalness *and* an emotion-recognition accuracy *and* a WER (to confirm you did not trade away intelligibility for expression, a very common regression) *and*, for the final decision between two candidate systems, a human preference test. And you measure honestly, which for these metrics means: for MOS, report the number of raters and the confidence interval, MOS swings a lot with rater pool and a delta under about 0.2 is usually noise; for emotion accuracy, name the SER model and its own accuracy ceiling; for the F0 similarity, say whether you aligned with DTW (you should, otherwise a duration difference dominates the contour distance); and for preference, report the count and a significance test. The single most common evaluation mistake in expressive TTS papers is reporting only MOS-naturalness, which, as the figure shows, is structurally blind to the entire point of the work.

#### Worked example: a flat model can win on MOS

Here is the trap in numbers. Suppose you compare a flat baseline against your new expressive model. The flat baseline scores MOS-naturalness 4.1 and WER 0.5 percent. The expressive model scores MOS-naturalness 4.0 (slightly lower, because expressive prosody occasionally introduces a rougher contour the vocoder renders imperfectly) and WER 0.9 percent. On those two numbers alone, the flat model *wins*, and a naive eval would reject your better system. But add emotion-recognition accuracy, the flat baseline lands the intended emotion 41 percent of the time (barely above chance for five classes), while the expressive model lands it 78 percent, and add an A/B preference test, listeners prefer the expressive model 71 percent to 29 percent for the *target use case* of an audiobook character voice. The expressive model is decisively better and *lost* on the only two metrics a careless evaluation would have reported. That gap between "natural" and "right" is the whole reason expressiveness evaluation needs more than one number.

## Case studies and real numbers

Grounding all of this in shipped systems, with honest figures and approximations flagged as such.

**Global Style Tokens (Wang et al., 2018).** The original GST-Tacotron used a bank of 10 style tokens with 8-head attention and a 256-dimensional style embedding, trained fully unsupervised on expressive read speech. The notable result was *interpretability without labels*: individual tokens were found to control speaking rate and pitch baseline in a way listeners could hear when dialed, demonstrating that a small learned bank can discover a usable style basis. It established the reference-encoder-plus-token-bank pattern that nearly every expressive system since has borrowed.

**FastSpeech 2 (Ren et al., 2020).** FastSpeech 2's central contribution to expressiveness was feeding *ground-truth* pitch, energy, and duration into the decoder during training rather than asking the model to predict them, which both stabilized training (no more attention-alignment failures) and exposed prosody as editable variables. On LJSpeech it matched or modestly exceeded the autoregressive Tacotron 2 / Transformer-TTS baselines on MOS while being **non-autoregressive and therefore far faster to synthesize**, the mel is produced in a single parallel forward pass, with the vocoder typically the remaining bottleneck. The exact MOS deltas are small and dataset-dependent; the durable result is the variance-adaptor design.

**Parler-TTS (2024, after Lyth and King).** Parler-TTS's released `mini` and `large` models were trained on roughly tens of thousands of hours of speech annotated with *automatically generated* natural-language descriptions covering pitch, speed, reverberation, noise, and gender. The headline capability is steering delivery from a free-text description with no reference clip, and the open release made the description-driven recipe reproducible. Exact MOS and intelligibility figures vary by checkpoint and are best read off the current model card rather than quoted from memory; the order-of-magnitude fact is that a few-tens-of-thousands-of-hours dataset with templated descriptions is enough to learn usable text-driven style control.

**Codec-LM zero-shot clones (VALL-E family, 2023, and XTTS / F5-TTS, 2024).** These systems get expressive prosody from sampling plus in-context prompts as described. The reproducible practitioner fact, rather than a single benchmark number, is the temperature behavior in the worked example above: low temperature flattens and stabilizes (WER near zero, F0 variance low), temperatures around 0.7 to 1.0 are the expressive sweet spot, and high temperatures trade stability for variance until intelligibility breaks. XTTS exposes `temperature` and `repetition_penalty`; F5-TTS exposes a sway-sampling and CFG strength that play the same role. Treat any specific WER or MOS for these as version-specific and verify against the current release.

## When to reach for expressive control, and when not to

A decisive recommendation section, because more control is not free and often not warranted.

**Reach for explicit F0-energy-duration control** when you need *reproducible, surgical* prosody: a brand or accessibility voice that must say certain things a certain way every time, an audiobook pipeline where an editor needs to fix a specific mis-emphasis, anything where a human will iterate on the prosody. Do *not* reach for it when you just want generally lively speech, you will spend your life hand-tuning curves to approximate what a sampled codec LM gives you for free, and you will pay a naturalness tax for the privilege.

**Reach for a reference encoder or GST** when you have *example clips* of the delivery you want and need to apply that delivery to new text, the classic "match this narrator's style" job. Do not reach for it when your styles are better described in words than demonstrated in clips, or when speaker-style leakage would corrupt a different target voice.

**Reach for emotion labels** when your product genuinely has a *small, fixed* set of moods (a five-emotion assistant) and simplicity matters more than nuance. Do not reach for them when you need intensity gradients or within-sentence emotional trajectory, the category quantization will fight you the whole way; use a description model or explicit control instead.

**Reach for a description-driven model** when you want flexible, natural-language style control and coarse-to-medium granularity is enough, this is the best default for a new system that wants expressive control without a complex pipeline. Do not reach for it when you need exact numeric edits ("pitch +8% on word three"), which descriptions cannot express.

**Reach for a modern codec LM with temperature tuning** when you want *natural, varied* speech with minimal machinery and you are cloning from prompts, this is the frontier default and the right starting point for most new expressive-TTS work in 2026. Do not crank its temperature past the stability cliff chasing more expression, and do not expect it to hit a *precise* target prosody, that is what the explicit layer on top is for.

The meta-rule: start with a sampling codec LM for free baseline expressiveness, choose your prompt's prosody deliberately, tune temperature for the stability you need, and add explicit or description control *only* where the baseline fails you. Most teams reach for the heavy explicit machinery far too early, before they have exhausted what a good prompt and the right temperature already give them.

## Key takeaways

- **Prosody is four measurable variables**, F0 contour, energy, duration, and pauses, layered above the phonemes, and they carry the speaker's meaning that the words alone cannot.
- **Flatness is mean-collapse, not low capacity.** An MSE prosody loss provably targets the conditional mean $\mathbb{E}[y \mid x]$, and the mean of a multimodal distribution is a flat contour no human ever produced.
- **Every fix either disambiguates or samples.** Add conditioning (reference, label, description) so the distribution is unimodal, or model and *sample* the distribution (variational latent, codec-LM tokens, flow), so you draw a real mode instead of averaging.
- **Reference encoders distill style into one fixed-size vector** because style is a global property of a clip; GST adds an interpretable, dial-able token bank on top, trained with no style labels.
- **Explicit F0-energy-duration control (FastSpeech 2) is the surgical option**, the most controllable method, because prosody is exposed as editable tensors before the decoder, at a medium naturalness cost.
- **Emotion labels are coarse; descriptions are flexible; both beat flat.** Categories give five moods with an interpolation trick for intensity; free-text descriptions (Parler-TTS) give natural but imprecise control.
- **Modern codec LMs and flow models get expression for free** from sampling and rich in-context prompts; temperature is the master expressiveness-versus-stability dial, with a sweet spot around 0.7 to 1.0.
- **Never evaluate expressiveness on MOS-naturalness alone**, it is structurally blind to wrong mood. Triangulate with emotion-recognition accuracy, prosody similarity, WER, and a human preference test, reported honestly.

## Further reading

- Skerry-Ryan et al., "Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron" (2018), the reference-encoder prosody-transfer foundation.
- Wang et al., "Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis" (2018), the Global Style Tokens paper.
- Ren et al., "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech" (2020), the variance-adaptor design for explicit F0/energy/duration control.
- Kim et al., "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech" (VITS, 2021), the variational answer to prosody's one-to-many nature.
- Lyth and King, "Natural language guidance of high-fidelity text-to-speech with synthetic annotations" (2024), the description-driven approach behind Parler-TTS.
- Zhou et al., "Emotion Intensity and its Control for Emotional Voice Conversion" (2022), relative-attribute intensity control.
- Within this series: [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard), [conditioning and control in audio generation](/blog/machine-learning/audio-generation/conditioning-and-control-in-audio-generation), [text-to-speech from Tacotron to VITS](/blog/machine-learning/audio-generation/text-to-speech-from-tacotron-to-vits), [zero-shot voice cloning and the TTS frontier](/blog/machine-learning/audio-generation/zero-shot-voice-cloning-and-the-tts-frontier), [representing sound, waveforms, spectrograms, and perception](/blog/machine-learning/audio-generation/representing-sound-waveforms-spectrograms-and-perception), [real-time streaming and full-duplex speech](/blog/machine-learning/audio-generation/real-time-streaming-and-full-duplex-speech), and the capstone [building an audio generation stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack).
