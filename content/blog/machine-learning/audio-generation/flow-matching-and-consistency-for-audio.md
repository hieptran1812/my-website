---
title: "Flow Matching and Consistency for Audio: Fast, High-Fidelity Synthesis"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Why flow matching became the default engine for state-of-the-art TTS, how Voicebox, Audiobox, and F5-TTS use it, and how consistency distillation pushes audio synthesis down to a few steps — with runnable PyTorch and honest numbers."
tags:
  [
    "audio-generation",
    "audio-synthesis",
    "flow-matching",
    "text-to-speech",
    "consistency-models",
    "voicebox",
    "f5-tts",
    "generative-ai",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Audio Generation"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/flow-matching-and-consistency-for-audio-1.png"
---

The first time I watched a diffusion text-to-speech model render a sentence, I counted the steps the way you count seconds during a power outage. Fifty network evaluations, each one a full forward pass over a mel-spectrogram that was a couple thousand frames long, to produce four seconds of speech. On the GPU I had that day it ran slower than real time, which is a strange and humbling thing for a speech model — the audio was shorter than the wait to hear it. The output sounded great. It also sounded like it cost too much. I remember thinking that the model was spending most of its compute correcting a path it never needed to take, walking a long curved road from noise to speech when a straight one was sitting right there.

That straight road is what this post is about. **Flow matching** trains a model to follow a nearly straight line from noise to data, so a handful of integration steps reach the same place that diffusion needed dozens to find. Over 2023 to 2026 it quietly became the default engine for state-of-the-art speech: Meta's **Voicebox** and **Audiobox** built a single flow-matching model that does TTS, denoising, editing, and style transfer through masked infilling; **F5-TTS** and **E2-TTS** threw away the duration predictor and the phoneme aligner entirely and still hit zero-shot quality; and **consistency** and adversarial **distillation** pushed the whole thing down toward one or two steps. By the end of this post you will be able to write a flow-matching training step for a mel-spectrogram in PyTorch, run a low-step Euler sampler, reason about exactly how many steps you can cut before quality breaks, and decide when few-step audio is honest and when it is wishful thinking.

![A side by side comparison showing diffusion taking a curved many-step path from noise to a mel-spectrogram while flow matching takes a near-straight path with far fewer Euler steps](/imgs/blogs/flow-matching-and-consistency-for-audio-1.png)

This sits squarely on the series' recurring spine, the **audio stack**: waveform to neural-codec tokens or a mel latent, then a generative model, then a vocoder or decoder back to a waveform, all under the tension of **fidelity, controllability, speed, and length**. Flow matching is a generative-model choice that buys you *speed* and stable *training* without giving up *fidelity*, which is why it spread so fast. It is the natural sequel to [diffusion for audio](/blog/machine-learning/audio-generation/diffusion-for-audio): same conditional generative framing, straighter paths. For the mathematical derivation of flow matching itself I will lean hard on the image series — the math is modality-agnostic and I would rather spend our pages on what is genuinely audio-specific. Read [flow matching and rectified flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow) for the full derivation and [consistency models and few-step generation](/blog/machine-learning/image-generation/consistency-models-and-few-step-generation) for the distillation theory; I will recap both briefly here and then build on them.

## 1. The thirty-second recap: regress, don't simulate

Diffusion teaches a model to reverse a gradual noising process. You take clean data, add Gaussian noise in many small increments until it is pure noise, then train a network to undo one increment at a time. At inference you start from noise and crawl back, step by step, and because the reverse process is curved and partly stochastic you need many small steps to stay on the manifold of real data. That is why a diffusion sampler over a long mel-spectrogram is slow: each of the fifty-plus steps is a full pass over thousands of frames.

Flow matching reframes the goal. Instead of learning to denoise, you learn a **velocity field** — a function that, at any point between noise and data, tells you which direction to move and how fast. Concretely, pick a clean sample $x_1$ (a real mel) and a noise sample $x_0 \sim \mathcal{N}(0, I)$, and define a straight-line path between them:

$$
x_t = (1 - t)\, x_0 + t\, x_1, \qquad t \in [0, 1].
$$

The velocity along that straight line is just its time derivative, which is constant:

$$
\frac{d x_t}{dt} = x_1 - x_0.
$$

That constant $x_1 - x_0$ is the **target**. We train a network $v_\theta(x_t, t)$ to predict it. The **conditional flow matching** loss is a plain regression:

$$
\mathcal{L}_\text{CFM} = \mathbb{E}_{t,\, x_0,\, x_1}\,\big\lVert v_\theta(x_t, t) - (x_1 - x_0) \big\rVert^2.
$$

That is the whole training objective. Sample a random time $t$, interpolate, regress the velocity onto $x_1 - x_0$, done. There is no noise schedule to tune, no signal-to-noise weighting to get wrong, no stochastic differential equation to discretize during training. You **regress, you do not simulate**. The remarkable fact — proven in the [flow matching post](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow), and I will not re-derive it — is that the network you get from this per-sample objective recovers a velocity field whose ordinary differential equation transports the *whole* noise distribution to the *whole* data distribution. You optimize a conditional, per-pair target; you obtain an unconditional, distribution-level transport map.

There is one subtlety the conditional objective hides, and it is worth a sentence because it explains why training is so stable. For a *single* pair $(x_0, x_1)$ the target velocity is the constant $x_1 - x_0$, but many different $(x_0, x_1)$ pairs pass through the same point $x_t$ at the same time $t$, each demanding a different velocity. The network cannot satisfy all of them, so under the mean-squared-error loss it converges to the *average* of those competing targets — the conditional expectation $\mathbb{E}[x_1 - x_0 \mid x_t, t]$. That conditional average is precisely the **marginal velocity field** whose ODE transports the whole noise distribution to the whole data distribution. The magic is that you never have to compute that average yourself; minimizing the per-pair regression gets you there automatically, because the optimum of an MSE regression *is* the conditional mean. This is the same reason diffusion's denoising objective works, restated for velocities, and it is why neither objective needs the intractable marginal score or velocity written out explicitly. It also explains the stability: an MSE regression with a bounded target is about the best-conditioned optimization problem in deep learning, with none of the adversarial seesawing or schedule sensitivity that makes other generative objectives temperamental.

At inference you integrate the learned ODE from $t = 0$ (noise) to $t = 1$ (data):

$$
\frac{dx}{dt} = v_\theta(x_t, t), \qquad x_{t + \Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t).
$$

That last expression is a single Euler step. Because the path the model learned is close to straight — the linear-interpolation, or "rectified," construction deliberately encourages straightness — Euler integration with a coarse step size stays accurate. Where diffusion needs fifty to two hundred steps, flow matching with a straight-path target routinely produces clean audio in eight to thirty-two. Each saved step is a saved forward pass over the entire mel, and that is the entire reason this matters for audio.

#### Worked example: counting the saved passes

Take a four-second utterance at a 100-frame-per-second mel resolution — 400 frames. A diffusion-TTS model running 64 sampling steps does 64 forward passes over those 400 frames. A flow-matching model that holds quality at 16 steps does 16. If a single forward pass costs roughly the same in both (same backbone size, which is the apples-to-apples case), flow matching is doing one-quarter of the inference compute for the same audio. On an RTX 4090 where the diffusion pass clocked a real-time factor (RTF, generation-time divided by audio-duration) of about 1.0 — meaning four seconds of compute for four seconds of speech — the flow-matching model lands near 0.25, four times faster than the audio plays. That is the difference between "wait for it" and "streams comfortably." Same network, same data, just a straighter path and fewer evaluations.

## 2. What is actually different about audio

If the math is modality-agnostic, why does audio get its own post? Three reasons, and they compound.

**First, the per-step cost is brutal and it scales with length.** An image is a fixed grid — say $64 \times 64$ latent pixels — no matter how "long" the content is. Audio is a 1D signal whose representation grows linearly with duration. A 10-second clip at 100 mel frames per second is 1,000 frames; a 4-minute song is 24,000. Every sampling step is a forward pass over that entire sequence, and a transformer's attention cost grows worse than linearly in sequence length. So the number of steps multiplies an already-expensive, length-dependent pass. Cutting steps from 64 to 16 on an image is nice; on a long mel it is the difference between a model you can serve and one you cannot. **Fewer steps matters more when each step is costly, and on audio each step is costly precisely because the signal is long.** This is the single most important audio-specific fact in this post, and it is why flow matching, which is fundamentally a way to need fewer steps, took over speech faster than it took over images.

**Second, audio flow matching operates on a mel or a codec latent, not on raw waveform.** Almost nobody runs flow matching directly on 24,000 samples per second of waveform — the sequence is too long and the high-frequency detail is too unforgiving. Instead the model generates a compact intermediate representation: a mel-spectrogram (the workhorse; see [representing sound](/blog/machine-learning/audio-generation/representing-sound-waveforms-spectrograms-and-perception)) or the continuous latent of a [neural audio codec](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound). A vocoder or codec decoder then turns that into waveform. So an "audio flow-matching model" is really a flow-matching model over mel frames, and the velocity field lives in mel space. That changes what the network sees and how you condition it.

The choice between a mel and a codec latent is a real engineering decision with consequences. A mel-spectrogram is a fixed, deterministic transform — no learned encoder, fully understood, and any of the fast GAN vocoders (HiFi-GAN, BigVGAN, Vocos) can turn it back into waveform. That predictability is why F5-TTS and most modern TTS flow over mels: you get a stable target the model can hit and a decoder you trust. A continuous codec latent is more compact and can carry information a mel throws away (some phase structure, fine texture), but it ties your generator to a specific learned codec, and if the codec has artifacts your generator will faithfully reproduce them. For *music* — where stereo, wide bandwidth, and texture matter more — codec latents win; for *speech*, the mel-plus-vocoder path is simpler and usually sufficient. One important practical note: flow matching needs a *continuous* target, so you flow over the codec's *pre-quantization* continuous latent (or the mel), not over discrete codebook indices. Discrete tokens are the domain of autoregressive models ([autoregressive audio models](/blog/machine-learning/audio-generation/autoregressive-audio-models-wavenet-to-audiolm)); continuous latents are the domain of flow and diffusion. That split — discrete-to-AR, continuous-to-flow — is one of the cleanest organizing facts in the whole audio stack.

Here is the front of the pipeline in code: load audio and compute the mel that a flow-matching model will learn to generate. This is the target side of the training loop above.

```python
import torch, torchaudio

# Load and resample to the model's rate (24 kHz is common for TTS).
wav, sr = torchaudio.load("speech.wav")              # (channels, samples)
wav = torchaudio.functional.resample(wav, sr, 24000)
wav = wav.mean(0, keepdim=True)                       # mono

# The mel transform: this exact config must match the vocoder you decode with.
mel_tf = torchaudio.transforms.MelSpectrogram(
    sample_rate=24000, n_fft=1024, hop_length=256,
    win_length=1024, n_mels=100, power=1.0,          # magnitude mel
)
mel = mel_tf(wav)                                     # (1, n_mels, T)
mel = torch.log(mel.clamp(min=1e-5))                 # log-mel: the model's x1
mel = mel.squeeze(0).transpose(0, 1)                 # (T, n_mels) frame-major
print(mel.shape)   # e.g. (T, 100): T frames, the flow model's target tensor
```

The `hop_length=256` at 24 kHz means about 94 mel frames per second, so a 4-second clip is roughly 375 frames — that `T` is the sequence length every sampling step pays for, which is the whole point of section 2's first argument. And note the warning in the comment: the mel config must match the vocoder, because the vocoder learned to invert *that specific* time-frequency representation. Mismatch the `n_fft` or `n_mels` and the vocoder produces garbage from a perfectly good mel.

**Third, the conditioning is the whole game in speech.** For images, "a velocity field from noise to data" with a text prompt is most of the story. For speech you need the synthesized mel to say a *specific* text, in a *specific* voice, with the right *timing*. The cleverest audio-specific ideas — Voicebox's masked infilling, F5-TTS dropping the aligner — are all about *how text and a voice prompt get injected into the flow-matching model*. The flow part is borrowed; the conditioning is invented. That is where we will spend the next sections.

![A dataflow graph of Voicebox style masked mel infilling where context frames, a mask span, aligned phonemes, and noise all feed a flow-matching network that solves an ODE to produce an infilled mel for a vocoder](/imgs/blogs/flow-matching-and-consistency-for-audio-2.png)

## 3. A flow-matching training step over a mel, in PyTorch

Let us make this concrete before going further. Here is a self-contained, runnable training step for a flow-matching model over mel-spectrograms, with masked conditioning of the Voicebox kind. I am keeping the backbone abstract (any transformer or U-Net that maps a mel-shaped tensor plus a time embedding plus conditioning to a mel-shaped velocity) so the *flow* logic is unmissable.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowMatchingTTS(nn.Module):
    """Flow matching over mel frames with Voicebox-style masked conditioning.
    The backbone maps (noisy mel, time, context mel, mask, text) -> velocity."""
    def __init__(self, backbone, n_mels=100):
        super().__init__()
        self.backbone = backbone   # e.g. a DiT-style transformer over frames
        self.n_mels = n_mels

    def forward(self, mel, text_emb):
        # mel:      (B, T, n_mels)  clean target mel (x1)
        # text_emb: (B, T, d_text)  frame-aligned text features (the condition)
        B, T, _ = mel.shape

        # 1) Sample a random time t per item, uniform in (0, 1).
        t = torch.rand(B, device=mel.device)                      # (B,)

        # 2) Sample noise x0 and build the straight-path point x_t.
        x0 = torch.randn_like(mel)                                # x0 ~ N(0, I)
        x1 = mel                                                  # clean target
        t_ = t[:, None, None]                                     # (B,1,1) broadcast
        xt = (1.0 - t_) * x0 + t_ * x1                            # x_t on the line

        # 3) The flow-matching target is the constant straight-path velocity.
        target_v = x1 - x0                                        # d x_t / dt

        # 4) Voicebox-style masking: randomly mask a span of frames; the model
        #    must infill it from the *unmasked* context plus the text.
        mask = self._random_span_mask(B, T, mel.device)          # (B, T) 1 = masked
        context = x1 * (~mask)[..., None]                         # known frames revealed
        model_in = torch.where(mask[..., None], xt, x1)          # noisy only inside mask

        # 5) Predict velocity. Conditioning = context mel + text + time.
        v_pred = self.backbone(model_in, t, context, mask, text_emb)

        # 6) Regress velocity, but ONLY where we masked (that is what we generate).
        loss = F.mse_loss(v_pred[mask], target_v[mask])
        return loss

    @staticmethod
    def _random_span_mask(B, T, device, min_frac=0.2, max_frac=0.9):
        mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        for b in range(B):
            frac = torch.empty(1).uniform_(min_frac, max_frac).item()
            span = max(1, int(frac * T))
            start = torch.randint(0, max(1, T - span + 1), (1,)).item()
            mask[b, start:start + span] = True
        return mask
```

![A dataflow graph of one flow-matching training step showing clean mel and noise and a sampled time feeding an interpolation, which feeds both a velocity prediction and a straight-path target that meet at a mean-squared-error loss](/imgs/blogs/flow-matching-and-consistency-for-audio-5.png)

Five lines carry the whole idea: sample `t`, build `xt` by linear interpolation, set `target_v = x1 - x0`, predict, regress. The figure above traces exactly those five lines as a dataflow: the clean mel and the noise sample and the random time meet at the interpolation node, the interpolation feeds both the network's velocity prediction and the constant straight-path target, and the two meet at a mean-squared-error loss. There is nothing hidden — no schedule, no reweighting, no second network. Everything else in the code is the audio-specific conditioning. Notice the masking: by training the model to fill a *masked span* of mel frames given the surrounding context and the text, you teach one network to do many tasks. Mask the whole clip and you get pure TTS (generate everything from text). Mask the end and you get continuation. Mask a word in the middle and condition on a *different* text there and you get **editing**. Mask nothing but feed a noisy input and you get **denoising**. Mask the content but keep a voice prompt unmasked and you get **voice conversion / style transfer**. This is the masked-infilling generalist recipe, and flow matching is what makes the infilled region high quality.

One subtlety worth stating: we only compute the loss on masked frames. The unmasked context is given, not generated, so regressing velocity there would be wasted signal and would bias the model toward copying. Restricting the loss to the masked region is the analogue of masked language modeling's "only predict the masked tokens," and it is exactly what Voicebox does.

#### The mask schedule is a real knob

How much you mask during training shapes what the model is good at. Mask too little and the model leans on context and never learns to generate from scratch, so cold-start TTS (mask everything) is weak. Mask too much and you starve it of the in-context conditioning that makes voice cloning work. Voicebox samples the masked fraction broadly — anywhere from a small span to nearly the whole clip — so the same weights handle both "fill one word" and "speak this whole sentence in this voice." When I have trained models like this, the masked-fraction distribution mattered more to downstream cloning quality than another few million parameters in the backbone. It is the kind of detail that does not appear in the headline architecture diagram but decides whether the thing works.

## 4. Voicebox and Audiobox: one model, many jobs

Voicebox (Le et al., Meta, 2023) is the clearest statement of the generalist idea. It is a **non-autoregressive** (it generates all frames in parallel, not left-to-right), flow-matching model trained on the masked-infilling task above, conditioned on frame-aligned phonemes. Because the task is "infill masked audio given context and text," a single set of weights performs:

- **Zero-shot TTS**: give a 3-second voice prompt as unmasked context, mask the region for the new sentence, condition on its phonemes, and the model speaks the new text in the prompt's voice.
- **Speech editing**: mask the words you want to change, supply the edited transcript for that span, and the model regenerates only those words, splicing seamlessly into the surrounding real audio.
- **Denoising / enhancement**: treat the noisy signal as the thing to be "infilled" toward clean speech.
- **Style and accent transfer**: condition on a prompt in a target style and let the model carry that style into new content.
- **Diverse sampling**: because it is generative, you can sample multiple plausible realizations of the same text, useful for data augmentation.

Audiobox (Meta, 2023) generalizes the same recipe beyond speech to general audio and music, adding description-based control ("a dog barking over rain") on top of the infilling backbone. The architectural point is identical: **flow matching plus masked in-context conditioning yields a generalist**. You do not train a TTS model and an editing model and a denoiser; you train one infilling model and ask it different questions by choosing what to mask.

The editing case is worth seeing in code, because it shows the infilling framing concretely. To change a word in an existing recording, you mask only that word's frames, supply the edited transcript, and let the model regenerate just that span while the rest of the real audio stays untouched.

```python
# Speech editing via masked infilling: replace "Monday" with "Friday"
# in an existing recording, regenerating only that word's frames.
def edit_span(model, real_mel, full_text_emb, edit_start, edit_end,
              n_steps=32, device="cuda"):
    T = real_mel.shape[0]
    mask = torch.zeros(T, dtype=torch.bool, device=device)
    mask[edit_start:edit_end] = True              # only the edited word's frames

    context = real_mel.clone()                    # all real frames are context...
    context[mask] = 0.0                           # ...except the masked span

    new_mel = euler_sample(
        model,
        shape=real_mel[None].shape,               # (1, T, n_mels)
        text_emb=full_text_emb[None],             # transcript with the EDITED word
        context=context[None], mask=mask[None],
        n_steps=n_steps, device=device,
    )[0]
    out = real_mel.clone()
    out[mask] = new_mel[mask]                      # splice the regenerated span back
    return out                                     # vocode this to hear the edit
```

The model only ever generates the masked frames; everything else is the original recording, so the edit splices in seamlessly with the original voice, room tone, and prosody on either side. No re-recording, no audible seam — that is the payoff of treating editing as infilling, and it is the same code path as TTS with a different mask.

There is a duration wrinkle. Voicebox is non-autoregressive, which means it needs to know *how long* the output mel should be before it generates — you cannot infill a span without knowing the span's length. So Voicebox trains a **separate flow-matching duration model** that predicts per-phoneme durations, and uses those to size the masked region and align the phonemes to frames. This is a genuine moving part: a forced aligner to get training-time phoneme-to-frame alignment, plus a duration predictor at inference. It works extremely well, but it is machinery. That machinery is exactly what the next model deletes.

![A vertical stack showing the alignment-free F5-TTS recipe where characters are padded to mel length, concatenated with noise, fed to a DiT flow-matching transformer that learns alignment implicitly, then sampled and vocoded](/imgs/blogs/flow-matching-and-consistency-for-audio-3.png)

## 5. F5-TTS and E2-TTS: throw away the aligner

Here is the move that made me sit up. E2-TTS (Eskimez et al., 2024) and then F5-TTS (Chen et al., 2024) asked: do we actually need the duration model and the phoneme aligner at all? Their answer was no, and the simplification is almost embarrassing in its directness.

The F5/E2 recipe is:

1. Take the input **characters** (not even phonemes — raw text characters) and pad the character sequence with filler tokens until it is the **same length as the target mel**. No duration prediction; you just need the *total* length, which at inference you estimate crudely (for example, scale the reference audio's duration by the ratio of new-text length to reference-text length).
2. Concatenate that character-filled sequence with the noisy mel input along the feature dimension.
3. Feed the whole thing to a flow-matching **Diffusion Transformer (DiT)** and train with the plain CFM loss. The transformer's self-attention learns, on its own, which characters correspond to which frames. No forced alignment, no monotonic attention constraint, no duration predictor — alignment is **implicit**, an emergent property of training a big enough flow-matching transformer on enough data.

That is it. The diagram above shows the entire pipeline: characters in, pad to mel length, concatenate with noise, flow-matching DiT, ODE sample, vocoder out. F5-TTS adds a couple of refinements over E2 — a ConvNeXt block to pre-process the text stream before it meets the audio, and a "Sway Sampling" tweak to the time schedule that improves low-step inference — but the spine is "text-filled mel plus flow matching, no alignment." The result is simple, fast, high-quality **zero-shot TTS**: give it a few seconds of reference audio and its transcript, plus the text you want, and it clones the voice.

Why does dropping the aligner work? Two things. First, flow matching is a *stable, well-conditioned* training objective — a plain MSE regression on a velocity, with no adversarial term and no brittle alignment loss to balance — so you can scale the model and data without the training falling over, and a large model on large data is exactly what learns alignment implicitly. Second, the masked/filled-text framing gives the model the context it needs: it sees the reference audio (unmasked) and the full text, so the in-context examples teach it both the voice and the text-to-sound mapping in one shot. The aligner was solving a problem that a big enough in-context model solves for free.

It helps to be precise about *what* the transformer learns instead of an explicit aligner. Self-attention lets every output mel frame attend over the entire padded character sequence. During training, the only way to minimize the velocity loss is for each frame to attend to the *right* characters — the ones whose sounds it must produce at that moment. So the attention pattern that emerges is, in effect, a soft alignment: a learned, data-driven mapping from text positions to frame positions, discovered because it is the loss-minimizing thing to do, not because anyone wrote an alignment objective. You can actually visualize it — pull out an attention head and you will often see a roughly monotonic diagonal band linking characters to the frames that voice them, the same diagonal a forced aligner would have produced, except nobody supervised it. This is why scale matters so much: a small model on little data cannot afford the capacity to discover alignment on its own, so it benefits from the explicit prior; a large model on large data discovers a *better* alignment than the hand-built one, because it is free to be non-monotonic where real speech is (coarticulation, liaison across word boundaries) instead of being forced into a rigid monotonic constraint.

The flip side is the failure mode. Because the length is estimated crudely (scale the reference duration by the text-length ratio), a bad length estimate hurts. Ask for far more or far less audio than the text needs and the implicit alignment has to stretch or compress unnaturally, producing speech that is too slow and draggy or too fast and clipped. The fix is a better length heuristic — estimate from a phoneme-count proxy rather than raw character count, or run a tiny duration regressor purely to set the total length (not the per-phoneme alignment) — but the headline simplification holds: you need a *length*, you do not need an *alignment*. That distinction is the entire content of the E2/F5 insight.

#### Worked example: zero-shot cloning with F5-TTS

In practice the F5-TTS inference loop is short. You provide a reference clip and its text, the target text, and a step count.

```python
# pip install f5-tts
from f5_tts.api import F5TTS

tts = F5TTS()   # loads the pretrained F5-TTS checkpoint + a Vocos vocoder

wav, sr, _ = tts.infer(
    ref_file="prompt.wav",                       # ~3-10s of the target voice
    ref_text="This is the reference speaker.",   # its transcript (or auto-ASR it)
    gen_text="The quick brown fox jumps over the lazy dog.",
    nfe_step=32,        # number of function evaluations = ODE steps
    cfg_strength=2.0,   # classifier-free guidance strength
)
# wav: a numpy waveform at sr (24 kHz). Save it:
import soundfile as sf
sf.write("cloned.wav", wav, sr)
```

`nfe_step` is the lever this whole post is about — it is the number of Euler steps the ODE solver takes. Set it to 32 and you get the reference-quality output. Drop it to 16 and on most prompts you will not hear a difference while halving the latency. Drop it to 8 and you start hearing it on hard prompts (fast speech, unusual names), which is the knee we will quantify in section 8. The point is that the *same checkpoint* spans a quality-speed curve you control at inference, with no retraining — that flexibility is a direct gift of the flow-matching formulation, because the model learned a velocity field you can integrate as finely or coarsely as you like.

## 6. The Euler ODE sampler, and why few steps is enough

The sampler that turns a trained velocity field into audio is short enough to write in full. Starting from noise at $t = 0$, take uniform Euler steps to $t = 1$.

```python
import torch

@torch.no_grad()
def euler_sample(model, shape, text_emb, context, mask,
                 n_steps=16, cfg_scale=2.0, device="cuda"):
    """Integrate dx/dt = v_theta from noise (t=0) to data (t=1) with Euler."""
    x = torch.randn(shape, device=device)            # x0 ~ N(0, I), shape (B,T,n_mels)
    dt = 1.0 / n_steps
    ts = torch.linspace(0.0, 1.0 - dt, n_steps, device=device)

    for t in ts:
        t_batch = t.expand(shape[0])                 # (B,) same t for the batch
        # Classifier-free guidance: blend conditional and unconditional velocity.
        v_cond   = model(x, t_batch, context, mask, text_emb)
        v_uncond = model(x, t_batch, context, mask, text_emb=None)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)
        x = x + dt * v                               # one Euler step along the flow
    return x                                          # an estimate of x1 (clean mel)
```

A few things are worth pulling out. The loop body is one network evaluation (here two, because of guidance) and one addition — that is the entire per-step cost, and it is why "number of steps" is the dominant latency term. **Classifier-free guidance** appears exactly as in the image series ([classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance)): you run the model with and without the text condition and extrapolate, which sharpens adherence to the text at the cost of a second forward pass per step. For speech, guidance strengths around 1.5 to 3 are typical; too high and the prosody gets unnatural and clipped.

The reason 16 Euler steps suffice where diffusion wanted 64+ is the **straightness** of the path. Euler integration's error per step grows with the *curvature* of the trajectory — a straight line has zero curvature, so Euler is exact on it. The linear-interpolation construction of conditional flow matching, and especially the rectified-flow refinement, deliberately produces paths that are close to straight, so coarse Euler steps barely drift off course. Diffusion's reverse SDE, by contrast, follows a curved, noise-injected path that a coarse step would fall off, which is why it needs many fine steps to stay on the data manifold. If you want a higher-order solver you can swap in midpoint (a.k.a. Heun / RK2) integration to squeeze a few more steps out, but on a well-rectified flow the gains from a fancier solver are smaller than the gains from the straightness itself.

#### Why streaming likes this

Low-latency and streaming TTS care about *time to first audio* and *real-time factor*. Flow matching helps both. Because it is non-autoregressive, it generates all frames in parallel rather than waiting on a left-to-right loop, so the latency is "one batched ODE solve" rather than "one forward pass per frame." And because the ODE solve is few-step, that batched solve is cheap. For full-duplex and streaming systems (covered in [real-time and full-duplex speech](/blog/machine-learning/audio-generation/real-time-streaming-and-full-duplex-speech)) you can chunk the mel and run a few-step flow solve per chunk, keeping the pipeline fed without the long-tail latency of a 100-step diffusion solve. The fewer steps you need, the smaller and more frequent your chunks can be, which is exactly the streaming sweet spot.

There is a real tension to name here, though, because non-autoregressive flow matching is not a pure win for streaming. An autoregressive model emits frames left to right, so it can start producing audio before it has decided the *end* of the utterance — natural for streaming. A non-autoregressive flow model sizes the *whole* output mel up front and solves it all at once, so in its naive form it cannot emit the first frame until it has solved the last. The chunking trick is what reconciles this: you commit to a chunk's length, solve that chunk with a few-step flow (using the previous chunk as infilling context for continuity), emit it, and move on. The chunk size sets the latency floor — a 0.5-second chunk means you cannot respond faster than 0.5 seconds plus the solve time — so you trade chunk granularity against cross-chunk prosody smoothness. This is precisely the kind of decision where the *step count* and the *chunk size* interact: a model that needs 32 steps forces larger chunks (to amortize the solve overhead) than one distilled to 4 steps, so distillation does not just lower RTF, it lets you shrink the chunk and the latency floor with it. The streaming story and the few-step story are the same story.

## 7. The science of "few steps is enough": curvature and Euler error

I claimed straight paths let you take coarse steps. Let me make that quantitative, because it is the single most important *why* in this post and it deserves more than a hand-wave.

Euler integration approximates the true solution of an ODE by pretending the velocity is constant over each step. The error it makes on a single step is governed by how much the velocity actually changes over that interval — formally, the *local truncation error* of one Euler step of size $\Delta t$ is

$$
\text{error} \approx \tfrac{1}{2}\,\Delta t^2 \,\big\lVert \ddot{x}_t \big\rVert,
$$

where $\ddot{x}_t$ is the acceleration along the trajectory — the second derivative, i.e. the *curvature*. Read that formula slowly: the per-step error is proportional to the curvature of the path. If the trajectory is perfectly straight, $\ddot{x}_t = 0$, and a single Euler step is *exact* no matter how big you make it. If the trajectory curves sharply, you need small steps to keep $\Delta t^2 \lVert \ddot{x}_t \rVert$ small, and "small steps" means "many steps."

This is the whole ballgame. Diffusion's reverse trajectory is highly curved — it has to wind its way back onto the data manifold through a noise-injected, score-following path — so its $\lVert \ddot{x}_t \rVert$ is large and you pay for it in step count. The conditional-flow-matching construction builds each *conditional* path as a literal straight line, and the rectified-flow refinement (re-training on the model's own samples, the "reflow" procedure) straightens the *marginal* paths too, driving the curvature toward zero. The closer to straight you get, the larger the steps you can take, and the total error over $N$ steps of an order-1 method like Euler accumulates to roughly $O(\Delta t) = O(1/N)$ — but with a constant that *scales with the path's curvature*. Straighten the path and you shrink the constant, so the same target error needs fewer steps. That is not a heuristic; it is the truncation-error formula, and it is why "flow matching = fewer steps" is a mathematical consequence of "flow matching = straighter paths," not a lucky empirical finding.

#### Worked example: the error budget in steps

Suppose your tolerance is a fixed reconstruction error $\epsilon$ on the generated mel. With Euler, total error scales like $C / N$ where $C$ absorbs the path curvature. If diffusion's path has curvature constant $C_\text{diff}$ and a well-rectified flow has $C_\text{flow} \approx C_\text{diff} / 4$ (a plausible ratio after reflow), then to hit the *same* $\epsilon$ the flow model needs $N_\text{flow} = C_\text{flow} / \epsilon \approx N_\text{diff} / 4$ steps. If diffusion needed 64 steps, the flow needs about 16 — which is exactly the ratio you see in practice between a diffusion-TTS sampler and an F5-TTS default. The factor-of-four is illustrative, not a law, but the *mechanism* is exact: cut the curvature constant by some factor and you cut the steps needed for a given quality by the same factor. This is why people invest so much in straightening (rectified flow, optimal-transport couplings, consistency distillation): every bit of curvature you remove is a proportional cut in inference cost.

A practical corollary: a higher-order ODE solver (midpoint/Heun, RK4) has local error $O(\Delta t^3)$ or better, so on a *curved* path it helps a lot. But on an *already-straight* path there is little curvature left for a fancy solver to exploit, so the gain shrinks. That is why F5-TTS's default Euler sampler is competitive with fancier integrators — the path is straight enough that the solver order stops mattering, and the simplest solver wins on speed per step. When I have benchmarked this, swapping Euler for a second-order solver on a well-trained flow model bought maybe one or two steps of equivalent quality, not a halving — the straightness had already done the work the solver would otherwise do.

## 8. How conditioning gets injected: a comparison

Everything audio-specific is in the conditioning, so it is worth laying the mechanisms side by side. The table below compares how the four approaches in this post inject text, voice, and timing into the flow-matching model.

| Approach | Text input | Alignment / timing | Voice / style | Sampling |
|---|---|---|---|---|
| Diffusion-TTS (e.g. Grad-TTS) | Phonemes + encoder | Explicit duration + monotonic align | Speaker embedding | 50–200 step SDE/ODE |
| Voicebox | Frame-aligned phonemes | Separate FM duration model + aligner | Unmasked audio prompt (in-context) | ~32–64 step ODE |
| F5-TTS / E2-TTS | Raw characters, padded | **Implicit** (learned by the DiT) | Unmasked reference audio + its text | ~16–32 step ODE |
| Consistency-distilled | Inherited from teacher | Inherited from teacher | Inherited from teacher | 1–4 step |

Reading across the rows traces the trajectory of the field. Early diffusion-TTS bolted an explicit duration model and a monotonic alignment mechanism onto the generative core — a lot of moving parts. Voicebox kept the duration model (it needs to size the masked region) but replaced the alignment-and-generation machinery with one flow-matching infilling network and conditioned the *voice* purely in-context, by revealing unmasked prompt audio. F5-TTS then deleted even the duration model, reducing "timing" to a single crude length estimate and letting the transformer's attention discover the character-to-frame mapping on its own. Each step down the table removes a hand-engineered component and leans harder on the flow-matching model's capacity — which only works because flow matching trains stably enough to absorb the extra burden. The conditioning mechanism, not the flow math, is what distinguishes these systems, and it is where the next few years of TTS research will keep moving.

One more conditioning subtlety specific to audio: **classifier-free guidance interacts with prosody**. In images, cranking guidance sharpens adherence to the prompt with mostly cosmetic side effects. In speech, over-guiding flattens and clips the prosody — the model adheres so hard to the *text* that it loses the natural pitch and rhythm variation that makes speech sound human, producing a tense, over-articulated delivery. So audio guidance strengths live in a narrower, lower band (roughly 1.5–3) than image guidance (often 5–15), and the sweet spot is voice- and content-dependent. This is a small but real way that the same CFG machinery behaves differently when the data is a mel instead of an image, and it is the kind of thing you only learn by listening to a hundred over-guided samples.

## 9. Consistency and distillation: from few steps to one

Flow matching gets you to roughly 16 steps honestly. Getting to **one to four** steps takes one more idea: **distillation**, and specifically **consistency** training. The full theory is in the [consistency models post](/blog/machine-learning/image-generation/consistency-models-and-few-step-generation); here is the audio-relevant version.

A consistency model is trained so that *every point along the flow's trajectory maps directly to the same endpoint*. Formally, if $\Phi(x_t, t)$ is the true trajectory of the ODE (where you end up if you integrate from $x_t$ at time $t$ all the way to $t = 1$), a consistency function $f_\theta$ is trained to satisfy

$$
f_\theta(x_t, t) = f_\theta(x_{t'}, t') \quad \text{for all } t, t' \text{ on the same trajectory},
$$

with the boundary condition $f_\theta(x_1, 1) = x_1$. If the model truly has this property, then evaluating it *once* at any point — including straight from noise at $t = 0$ — jumps directly to the clean endpoint. That is one-step generation. In practice you enforce consistency by a distillation loss: take a teacher (the trained flow-matching model), run it a small step along the ODE to get a slightly-cleaner point, and train the student so its output at the current point matches its output (under a stop-gradient, exponential-moving-average copy) at the next point. The teacher's many-step trajectory becomes the student's single jump.

![A side by side figure contrasting a teacher many-step ODE solve from noise to clean mel with a consistency student that maps any point on the path straight to the endpoint in one to four evaluations](/imgs/blogs/flow-matching-and-consistency-for-audio-6.png)

Here is a compact consistency-distillation step over a mel, building on the flow-matching teacher from section 3.

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def teacher_ode_step(teacher, x, t, dt, cond):
    """One small Euler step of the teacher flow toward t=1 (cleaner)."""
    v = teacher(x, t, cond)
    return x + dt * v

def consistency_distill_step(student, student_ema, teacher, mel, cond):
    """Train the student so f(x_t, t) == f(x_{t+dt}, t+dt) on the teacher path."""
    B, T, M = mel.shape
    t = torch.rand(B, device=mel.device) * (1.0 - 1e-3)   # avoid t=1 exactly
    dt = 0.05

    x0 = torch.randn_like(mel)
    t_ = t[:, None, None]
    xt = (1.0 - t_) * x0 + mel * t_                        # point on the line at t

    # Teacher advances xt one small step toward the data end.
    xt_next = teacher_ode_step(teacher, xt, t, dt, cond)

    # Student maps both points to the endpoint; EMA target gives the next point.
    pred      = student(xt, t, cond)                       # f_theta(x_t, t)
    with torch.no_grad():
        target = student_ema(xt_next, t + dt, cond)        # stop-grad EMA target

    loss = F.mse_loss(pred, target)                        # the consistency loss
    return loss
```

The student learns that wherever it is on the trajectory, it should already predict the endpoint — so at inference you can call it once from noise and get clean audio, or call it two-to-four times in a short ping-pong (predict endpoint, re-noise to an intermediate time, predict again) to recover the bit of quality a single step loses.

There is a second distillation family worth naming: **adversarial** distillation. Instead of (or alongside) the consistency loss, you add a GAN discriminator that judges whether the few-step student's output is real audio, which sharpens the high-frequency detail that a pure regression student tends to blur. This is the audio analogue of adversarial diffusion distillation, and it is closely related to how GAN vocoders work — the discriminator's job is to kill the dullness and buzz that show up when you ask a model to produce clean audio in one shot. For mel/vocoder stacks specifically, distillation often targets the **vocoder** too: a one-step distilled vocoder plus a few-step flow-matching mel generator gives you an end-to-end pipeline that is fast at both stages. We cover the vocoder side in [GAN vocoders and fast synthesis](/blog/machine-learning/audio-generation/gan-vocoders-hifi-gan-and-fast-synthesis).

There is an audio-specific reason one-step is *harder* here than for images, and it is worth dwelling on because it sets realistic expectations. Human hearing is unforgiving of certain error structures that the eye shrugs off. A slightly blurry image looks "soft" and acceptable; slightly blurry audio sounds *wrong* — over-smoothing in the spectral domain collapses the fine harmonic structure that distinguishes a voice from a kazoo, and the ear flags it instantly as synthetic. Worse, the artifacts that one-step generation introduces tend to live in exactly the frequency bands the ear is most sensitive to (the 2–5 kHz range where consonant energy and formant transitions sit), so a distortion that would be invisible in an image is glaring in speech. This is the same perceptual asymmetry that makes audio generation hard in the first place ([why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard)): our ears evolved to catch tiny irregularities in sound, and a one-step model has the fewest opportunities to correct them. That is precisely why the adversarial term matters so much for few-step *audio* specifically — the discriminator's whole job is to police those perceptually-loud high-frequency bands that a pure regression student smooths over.

The honest caveat, then: one-step audio is not free. Distillation transfers most of the teacher's quality but rarely all of it, and the failure modes are audible — slight over-smoothing of timbre, a touch of metallic buzz on fricatives, reduced prosodic variety. The two-to-four step regime is a much safer place to live than strict one-step: a single extra evaluation gives the model a chance to correct the worst of the one-step error, and the quality usually comes back close to the teacher while still being dramatically faster than the full solve. When I have shipped distilled audio, I have almost always chosen two-to-four steps over one — the marginal latency is tiny and the quality recovery is large. Whether the residual tax is acceptable depends entirely on the application, which is the subject of section 11.

## 10. The numbers: steps, RTF, and quality

Time for the measurement angle. The table below aggregates representative figures for the four regimes we have discussed. I want to be explicit about provenance: the step counts and the non-autoregressive column are architectural facts from the papers; the RTF and MOS/WER figures are *order-of-magnitude, hardware-dependent* values synthesized from the Voicebox, F5-TTS, and consistency-model literature and from my own measurements on similar models. **Treat the RTF and quality numbers as approximate** — RTF in particular swings by 3-5x with GPU, batch size, mel length, and whether you count the vocoder. The *relationships* between rows are robust; the absolute cells are illustrative.

Before the table, a word on *which* metric to trust for *which* task, because the right number depends on what you are generating. For **speech**, intelligibility (WER via a fixed ASR) and speaker similarity are the load-bearing metrics, with MOS as the human check — FAD is a weak signal for speech because the distributional distance it measures is dominated by acoustic texture, not by whether the words are right. For **music and general audio**, FAD is the primary automatic metric precisely because there is no transcript to score against, and the distributional match to real audio is what you care about. So when you read a flow-matching audio paper, check that the metric matches the task: a TTS paper leaning on FAD instead of WER is hiding something, and a music paper reporting WER is measuring the wrong thing. The reason this matters for *this* post is that the few-step trade looks different under each metric — cutting steps tends to raise WER (a content error, audible as a slurred word) faster than it raises FAD (a texture error, audible as slight dullness), so a music model can often tolerate more aggressive step-cutting than a speech model before the relevant metric complains. The metric you choose decides where your honest knee sits. See [audio quality metrics](/blog/machine-learning/audio-generation/audio-quality-metrics) for the full treatment of what each metric does and does not capture.

![A four by four matrix comparing diffusion-TTS, Voicebox flow matching, F5-TTS, and consistency distillation across sampling steps, real-time factor, quality, and whether the method is non-autoregressive](/imgs/blogs/flow-matching-and-consistency-for-audio-4.png)

| Method | Family | Steps (NFE) | RTF on a 4090 (approx) | Quality (MOS / WER) | Non-AR? |
|---|---|---|---|---|---|
| DiffWave / diffusion-TTS | Waveform/mel diffusion | 50–200 | ~1.0–5.0 | High MOS; WER low | Yes |
| Voicebox | Flow matching + infilling | ~32–64 | ~0.3–0.6 | High MOS; WER ≈ ground truth | Yes |
| F5-TTS / E2-TTS | Flow matching, alignment-free | ~16–32 | ~0.1–0.3 | High MOS (~4.0+); low WER | Yes |
| Consistency / adversarial-distilled | Distilled flow | 1–4 | <0.1 (near real time) | Slight MOS drop; WER stable | Yes |

Read it as a ladder. Diffusion-TTS sets the quality bar but pays many steps. Voicebox keeps the quality and cuts steps by roughly half to a third while adding the generalist infilling capability. F5-TTS cuts steps again *and* deletes the duration/alignment machinery, so it is both faster and simpler. Consistency distillation pushes to a handful of steps and near-real-time RTF, at a small and measurable quality cost. The trend down each column — fewer steps, lower RTF — is the story of the last three years of fast TTS in one table.

Two measurement honesties to internalize. First, **RTF must include the vocoder** or it is a lie: if your flow-matching mel generator hits RTF 0.1 but your vocoder runs at RTF 0.4, your *system* RTF is 0.5, and the vocoder is now your bottleneck (a recurring theme — see section 9's stress test). Second, **measure WER with a fixed ASR model on fixed text**: F5-TTS and Voicebox both report near-ground-truth WER on standard sets, which is the honest way to claim intelligibility — run an ASR (Whisper, say) on the synthesized audio and compare its transcript to the input text. MOS needs a real listening panel (typically 15–30 raters, each scoring 1–5) with confidence intervals; a single-number MOS with no interval and no rater count is marketing, not measurement.

#### Worked example: where does the time actually go?

Say you are serving F5-TTS at `nfe_step=16` on a 4090 and you measure end-to-end RTF 0.30 for a 5-second utterance — 1.5 seconds of wall-clock to make 5 seconds of speech. You want it faster. Where is the time? Profiling reveals: 16 DiT forward passes at ~70 ms each = 1.12 s for the mel, and the Vocos vocoder at ~0.35 s. The mel generator dominates. Cutting `nfe_step` to 8 nearly halves the mel time to ~0.56 s, dropping system RTF to ~0.18 — *if* quality holds. On clean, slow speech it does. On a tongue-twister with three proper nouns it does not: Whisper-measured WER jumps from 3% to 7% and you can hear a smeared consonant. The honest decision: serve at 16 for general text, and gate down to 8 only for content you have validated. The lesson is that the steps knob is real and continuous, and you tune it against *your* content, not a benchmark average.

## 11. The problem-solving narrative: choosing a step count under pressure

Let me walk a real decision. You are shipping a voice assistant. The product constraint is a **system RTF under 0.5** on your serving GPU (so responses feel instant) and **WER under 5%** on your evaluation set (so it is intelligible). You have an F5-TTS-class flow-matching model and a Vocos vocoder. How many steps?

Start at the top of the ladder. At `nfe_step=32` you measure system RTF 0.55 — over budget — and WER 2.8%, comfortably under. You are paying for quality you do not need at the cost of latency you cannot afford. Step down. At 16 steps, RTF 0.30 and WER 3.1% — both green, with margin. At 8 steps, RTF 0.18 and WER 4.6% — still technically passing but the WER margin is thin and you can hear occasional roughness on fricatives. You ship **16**, banking the latency headroom for traffic spikes and keeping the WER margin wide.

Now stress-test the decision, because that is where engineering lives.

**What happens when the speaker prompt is 3 seconds of noisy audio?** Zero-shot cloning quality degrades with prompt quality, independent of step count. The flow-matching model conditions on whatever voice it is shown; a noisy prompt teaches it a noisy voice, and at low step counts the smearing compounds. Mitigation: run the prompt through the *same* model's denoising mode (the infilling generalist can clean it) before using it as a conditioning prompt, or raise the step count *just for noisy prompts*. The knob is content-dependent, so make it dynamic.

**What happens when you ask for a very long output?** A non-autoregressive flow model must size the whole mel up front, and very long single passes strain both memory and the implicit alignment (the model can lose track of which text goes where over thousands of frames). Mitigation: chunk the text into sentence-sized pieces, generate each as its own few-step solve with the previous chunk's tail as context (the infilling framing makes this natural), and crossfade. You trade a tiny bit of cross-chunk prosody continuity for bounded memory and stable alignment.

**What happens when the vocoder becomes the bottleneck?** This is the classic trap. You optimize the mel generator to RTF 0.1 and declare victory, but your system RTF is still 0.5 because the vocoder is 0.4. At that point cutting *more* flow steps is pointless — you are optimizing the wrong stage. The fix is to distill or swap the vocoder (a one-step GAN vocoder like Vocos or BigVGAN is already fast; a diffusion vocoder would be your bottleneck and you should not use one here). **Always profile the whole stack before cutting steps**, or you will spend effort where it does not move the system number.

**What happens at the very bottom, one step?** Raw flow matching at 1 step does not work — the single Euler step from noise overshoots wildly because it has no chance to correct curvature, and you get buzz and smearing. One-step audio *requires* a distilled consistency or adversarial student, and even then it carries the small quality tax from section 7. So "1 step" is not a setting on your F5 checkpoint; it is a *different model* you produce by distillation. Conflating the two is a common and costly mistake.

![A timeline tracing audio synthesis from DiffWave and WaveGrad waveform diffusion through Voicebox and Audiobox masked flow matching to F5 and E2 alignment-free flow matching and finally one to four step consistency distillation](/imgs/blogs/flow-matching-and-consistency-for-audio-7.png)

## 12. Case studies: real systems and real numbers

Four named results to ground the theory, with sources and honest uncertainty flags.

**Voicebox (Le et al., Meta, 2023).** The paper reports that Voicebox's zero-shot TTS achieves *better* word error rate and speaker similarity than VALL-E on English zero-shot synthesis, while being roughly **20x faster** at inference because it is non-autoregressive flow matching rather than autoregressive codec-token generation. The 20x is the headline gap between "generate all frames in parallel with a few-step ODE" and "generate codec tokens one autoregressive step at a time," and it is the most direct evidence that the non-AR flow approach is a genuine speed win, not just a quality one. The model also demonstrates the generalist claim concretely: the *same* weights do TTS, editing, denoising, and sampling. (Speed multiples are paper-reported and hardware-dependent; treat as order-of-magnitude.)

**F5-TTS (Chen et al., 2024).** F5-TTS reports zero-shot TTS quality competitive with the best open systems while being *simpler* — no duration predictor, no phoneme alignment, characters straight into a flow-matching DiT. On standard zero-shot benchmarks it reports WER near ground-truth levels and strong speaker similarity, with inference RTF well under 1 on a single modern GPU at its default ~32 NFE, and the "Sway Sampling" schedule lets it hold quality at reduced steps. The key takeaway is not a single number but the *simplification*: a flow-matching transformer absorbed the alignment problem that previously needed a dedicated module. (Exact benchmark cells vary by eval set and ASR; the paper's tables are the source of truth.)

**Audiobox (Meta, 2023).** Audiobox extends the Voicebox flow-matching-plus-infilling recipe to general audio and adds natural-language description control. The relevant lesson for this post is that the *same* generative engine — flow matching over a latent, conditioned by in-context masking and description — scales past speech to music and sound, which is strong evidence that flow matching is a general audio engine and not a TTS trick. It is the bridge from this post to the music posts ([latent diffusion for music](/blog/machine-learning/audio-generation/diffusion-for-audio) and the music-generation track).

**Consistency / adversarial-distilled TTS and vocoders (2024 onward).** A growing line of work distills flow-matching or diffusion TTS down to one-to-four steps, and consistency-model and adversarial-distillation techniques applied to vocoders (and to mel generators) report near-real-time RTF with small, measurable MOS drops versus the many-step teacher. The honest framing in the literature is consistent: distillation recovers *most* of the quality at a fraction of the steps, the residual gap is small but real, and adversarial terms help close the high-frequency part of it. I am deliberately not quoting a single precise MOS delta here because it varies widely by teacher, distillation method, and eval; the *direction* — small tax for a large speedup — is the reliable claim.

The common thread across all four: flow matching gave audio a *stable, scalable* training objective, and that stability is what let researchers both simplify the architecture (F5 deletes the aligner) and aggressively distill the sampler (consistency hits a few steps) without the training collapsing. Diffusion could be distilled too, but flow matching's straighter paths gave distillation an easier starting point.

#### Worked example: measuring the step trade honestly

Suppose you want to *prove*, not assert, that you can drop F5-TTS from 32 steps to 16 without hurting intelligibility. Here is the honest harness. Fix a held-out set of 500 sentences with known transcripts. For each step count, synthesize all 500 with a *fixed seed* and a *fixed voice prompt*, run a *fixed ASR model* (Whisper-large, say) over the outputs, and compute word error rate against the ground-truth transcripts. Crucially, warm up the GPU before timing — the first few inferences are slow due to kernel compilation and cache effects — and measure RTF as a median over the 500, not a single run. You will get something like: 32 steps, WER 2.8%, RTF 0.30; 16 steps, WER 3.0%, RTF 0.16; 8 steps, WER 4.7%, RTF 0.09. The 32-to-16 move costs 0.2 points of WER (within noise) for nearly half the latency — a clear win you can defend. The 16-to-8 move costs 1.7 points, which is real and audible. Now you have *measured* the knee instead of guessing it, and you can ship 16 with evidence. The discipline here — fixed seed, fixed text, fixed ASR, warmed-up median RTF — is what separates a defensible claim from a vibe. Report the FAD too if you are doing music or general audio: compute it on a fixed embedding (VGGish or a CLAP encoder, *state which*) over a fixed sample size (smaller samples inflate FAD), and report the embedding and N alongside the number, because an FAD with no stated embedding is uninterpretable.

The same harness exposes a trap worth naming: the *vocoder* must be held fixed across all step counts, or you are measuring two things at once. If you accidentally change the vocoder between runs, a WER difference you attribute to step count might actually be the vocoder, and you will draw the wrong conclusion and cut the wrong knob. Hold everything fixed except the variable you are studying — the oldest rule in measurement, and the one most often broken in audio benchmarks because the pipeline has so many stages.

## 13. When to reach for this, and when not to

A decisive recommendation section, because "it depends" is not advice.

**Reach for flow matching when** you are building non-autoregressive TTS or audio synthesis and you care about inference speed — which is almost always. It is the current default for state-of-the-art open TTS for good reasons: stable training (plain MSE, no adversarial balancing at the generative stage), fewer sampling steps than diffusion at equal quality, natural in-context conditioning via masking, and a clean path to distillation. If you are starting a new TTS or audio-generation project in 2026 and you do not have a specific reason to do otherwise, a flow-matching mel generator plus a fast GAN vocoder is the recipe to beat.

**Reach for the Voicebox/Audiobox masked-infilling framing when** you want *one* model to do TTS *and* editing *and* denoising *and* style transfer. If you only ever need plain TTS, the full infilling generality is overkill — but it costs little to train it in and buys you optionality, so I usually take it.

**Reach for F5/E2 alignment-free when** you want the simplest possible high-quality zero-shot TTS and you have enough data to let the transformer learn alignment implicitly. With limited data, the explicit duration model and aligner can actually help (they inject a strong prior), so on small datasets the older VITS-style or Voicebox-with-duration approach may train more reliably. Alignment-free is a *scale* play.

**Reach for consistency/adversarial distillation when** you have a working flow-matching model and a hard latency target that few-step-but-not-one-step cannot meet — interactive assistants, on-device synthesis, full-duplex dialogue. Distill *after* you have a good teacher, never instead of one.

**Do not** reach for one-step generation on your base flow-matching checkpoint and expect quality — it requires distillation, full stop. **Do not** cut sampling steps before profiling the whole stack — if the vocoder is the bottleneck, fewer mel steps change nothing. **Do not** assume the same step count works for all content — fast speech and proper nouns need more steps than slow, common words; validate on *your* hard cases. **Do not** use a diffusion vocoder in a latency-critical path when a GAN vocoder hits the quality bar at a fraction of the cost — that is the audio analogue of using a 100-step sampler when a 10-step one was fine. And **do not** report RTF without the vocoder, or MOS without a rater count and a confidence interval; both are how fast-audio claims get inflated.

![A vertical ladder showing how mel quality holds as flow-matching steps drop from sixty-four to thirty-two to sixteen, hits a knee around eight steps, and collapses into buzz and smearing at four and one steps without distillation](/imgs/blogs/flow-matching-and-consistency-for-audio-8.png)

The ladder above is the honest summary of the speed-quality trade. Steps are nearly free to cut from the top — 64 to 32 to 16 with no audible loss on most content — until you reach a knee around eight steps where word error rate starts to rise and fricatives smear. Below the knee, raw flow matching degrades fast: four steps buzzes, one step collapses. The knee is where few-step audio is *honest*. Pushing past it to one or two steps is honest only with a distilled model, and even then you accept a small, measured quality tax. Knowing exactly where your knee is — by measuring WER and listening, on your content, on your hardware — is the difference between a fast system and a broken one.

## 14. Connecting back to the stack and the other series

Step back to the spine. The audio stack is waveform to mel/codec latent to generative model to vocoder to waveform, under fidelity-controllability-speed-length. Flow matching is a *generative-model* choice that wins on **speed** (fewer steps), holds **fidelity** (straight paths reach diffusion quality), and improves **controllability** through masked in-context conditioning — and it composes with everything else in the series. It eats a mel or codec latent ([neural audio codecs](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound)), it is judged by the same metrics ([audio quality metrics](/blog/machine-learning/audio-generation/audio-quality-metrics)), it sits next to autoregressive and diffusion engines as a third option, and it feeds a fast vocoder downstream. In the capstone, [building an audio generation stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack), a flow-matching mel generator plus a distilled vocoder is one of the strongest default recipes you can assemble for TTS.

The cross-series links are not decoration — they are how I kept this post focused. The *derivation* of conditional flow matching, the proof that the per-pair velocity target recovers a distribution-level transport, and the rectified-flow straightening trick all live in the image series' [flow matching post](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow), and the consistency-distillation theory lives in [consistency models and few-step generation](/blog/machine-learning/image-generation/consistency-models-and-few-step-generation). The math is the same; what is audio-specific is everything in the sections above — operating on a mel, masked infilling, alignment-free TTS, the brutal length-dependent per-step cost, and where the few-step knee honestly sits. If you read this post and one of those two, you have the whole picture: the universal generative machinery and the audio-specific engineering that makes it sing.

It is worth ending on why this matters beyond TTS. Flow matching is, more than anything, a *bet that straightening the generative path is worth more than any single architectural trick*, and audio is the modality where that bet pays off hardest because the per-step cost is so punishing. The same logic is now spreading into the music and sound posts ahead — latent flow matching for long-form music, where a song is tens of thousands of frames and every saved step is a saved minute of render — and into the streaming and full-duplex frontier, where the few-step ODE solve is what makes sub-second response times achievable at all. When you assemble your own stack, the question is not "should I use flow matching" — for non-autoregressive audio in 2026 the answer is almost always yes — but "how far down the step ladder can I honestly go for *my* content, *my* metric, and *my* latency budget." Answer that with measurement, not faith, and you will have a system that is both fast and good, which for most of the history of audio generation you could not have at the same time. That is the quiet revolution flow matching brought to sound.

## Key takeaways

- **Flow matching is regression, not simulation.** Interpolate noise and data on a straight line, regress the network's velocity onto the constant target $x_1 - x_0$, and you get a distribution-level transport map you can integrate in few steps.
- **Fewer steps matters more for audio than for images**, because each sampling step is a forward pass over a long, length-dependent mel sequence — cutting 64 steps to 16 is the difference between unservable and streaming.
- **Audio flow matching lives in mel or codec-latent space**, not raw waveform, and a vocoder/decoder turns the generated latent into sound — so it is really a flow-matching model over mel frames.
- **Voicebox/Audiobox make one generalist** via masked-mel infilling: TTS, editing, denoising, and style transfer are all "fill the masked span given context and text," and flow matching is what makes the fill high quality.
- **F5-TTS/E2-TTS delete the aligner and duration model** — characters padded to mel length plus noise into a flow-matching DiT, with alignment learned implicitly — giving simple, fast, high-quality zero-shot cloning.
- **The Euler ODE sampler is one network eval plus one addition per step**; straight paths let coarse steps stay accurate, and classifier-free guidance adds a second eval per step to sharpen text adherence.
- **Consistency and adversarial distillation reach one-to-four steps**, but one-step audio is a *distilled* model with a small, measurable quality tax — never expect it from your base flow checkpoint.
- **Profile the whole stack before cutting steps**: if the vocoder is the bottleneck, fewer mel steps change nothing, and RTF reported without the vocoder is not a real number.
- **The honest few-step knee is around eight to sixteen steps**: free to cut from the top, but below the knee WER rises and fricatives smear, so tune the step count against *your* content and hardware.
- **Report RTF on a named device with the vocoder included, and MOS with a rater count and interval, WER with a fixed ASR** — that is the difference between a measurement and a marketing claim.

## Further reading

- Lipman, Chen, Ben-Hamu, Nickel, Le. *Flow Matching for Generative Modeling* (2023) — the foundational conditional flow matching paper.
- Liu, Gong, Liu. *Flow Straight and Fast: Rectified Flow* (2022) — the straightening construction that makes few-step Euler accurate.
- Le et al. (Meta). *Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale* (2023) — flow matching + masked infilling as a speech generalist.
- Meta. *Audiobox: Unified Audio Generation with Natural Language Prompts* (2023) — the Voicebox recipe extended to general audio.
- Eskimez et al. *E2-TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS* (2024) — the alignment-free, fill-text-to-mel idea.
- Chen et al. *F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching* (2024) — the practical alignment-free flow-matching TTS with Sway Sampling.
- Song, Dhariwal, Chen, Sutskever. *Consistency Models* (2023) — the one/few-step distillation theory adapted here to audio.
- Within series: [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard), [diffusion for audio](/blog/machine-learning/audio-generation/diffusion-for-audio), [GAN vocoders and fast synthesis](/blog/machine-learning/audio-generation/gan-vocoders-hifi-gan-and-fast-synthesis), [zero-shot voice cloning and the TTS frontier](/blog/machine-learning/audio-generation/zero-shot-voice-cloning-and-the-tts-frontier), and the capstone [building an audio generation stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack).
- Out of series: [flow matching and rectified flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow) and [consistency models and few-step generation](/blog/machine-learning/image-generation/consistency-models-and-few-step-generation) for the full generative-math derivations.
