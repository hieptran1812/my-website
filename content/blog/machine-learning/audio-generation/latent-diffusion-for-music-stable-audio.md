---
title: "Latent Diffusion for Music: Stable Audio and Long-Form Generation"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Build a working mental model of latent music diffusion — the timing trick that makes one model emit a full three-minute stereo song, with runnable diffusers code and honest numbers."
tags:
  [
    "audio-generation",
    "audio-synthesis",
    "music-generation",
    "latent-diffusion",
    "stable-audio",
    "diffusion-transformer",
    "generative-ai",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Audio Generation"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/latent-diffusion-for-music-stable-audio-1.png"
---

I once asked an autoregressive music model for a four-minute lo-fi track to fill a long render queue. The first thirty seconds were lovely: a warm Rhodes chord, a soft kick, a little vinyl crackle. By ninety seconds the drums had quietly wandered a few BPM off the original tempo. By two minutes the key had shifted, not through any modulation a musician would write, but through accumulated error — each token conditioned on a slightly-wrong past, the wrongness compounding. The track did not end so much as fray. That failure has a name in this field: autoregressive drift, and it is the single hardest thing about making music that lasts longer than a loop.

The fix that the field converged on is not "a bigger autoregressive model." It is a different generative engine entirely. Instead of emitting one codec token after another, left to right, you take the diffusion recipe that conquered images — denoise a latent from pure noise in a handful of parallel passes — and port it to a heavily compressed *audio* latent. The model sees the whole track at once, both directions, with full context. There is no "past" to drift away from because there is no serial generation. And then comes the trick that turns this from "a fixed ten-second clip generator" into "a full-song generator": you *tell the model how long the track should be* and *where in that track this latent sits*. That single piece of conditioning — timing conditioning — is why Stable Audio can hand you a coherent three-minute stereo song with an intro, a body, and an actual ending, rather than ninety seconds of texture that stops when the buffer runs out.

![A layered stack showing text and a desired duration entering a compressed audio latent that a diffusion transformer denoises before a VAE decoder reconstructs a full stereo waveform](/imgs/blogs/latent-diffusion-for-music-stable-audio-1.png)

By the end of this post you will be able to: explain *why* diffusion beats autoregression for long-form music (and where it loses); describe the Stable Audio architecture concretely — the high-compression audio VAE, the diffusion-transformer denoiser, the T5 and CLAP text conditioning, and the timing-conditioning mechanism; run a `StableAudioPipeline` and an `AudioLDM2Pipeline` in 🤗 `diffusers` with the right knobs (`audio_end_in_s`, `num_inference_steps`, `guidance_scale`, stereo export, timed); read a results table of FAD, length, stereo, and steps across AudioLDM 2 / Stable Audio / Stable Audio 2; and make the diffusion-versus-MusicGen call for a real job. This is the diffusion branch of the audio stack — the same `waveform → latent → generative model → decoder → waveform` spine from [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard), with the tension `fidelity × controllability × speed × length` tilted hard toward *length and fidelity*. Its autoregressive sibling, the codec language model, lives in [music generation with MusicLM and MusicGen](/blog/machine-learning/audio-generation/music-generation-musiclm-and-musicgen); read them together to see the trade in full.

This post assumes you have met diffusion before. I will not re-derive the forward and reverse processes here — for that, read [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles) and, for the latent variant specifically, [latent diffusion and Stable Diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion). What I *will* do is derive the parts that are *audio*-specific: why minutes of 44.1 kHz stereo would be intractable without aggressive compression, how timing embeddings actually enter the network, and why a non-causal denoiser is the right tool for global musical structure.

## 1. Why diffusion, not autoregression, for long music

Let me make the drift problem precise, because it is the load-bearing motivation for everything that follows.

An autoregressive music model — MusicGen is the canonical open one — factorizes the joint distribution of a sequence of codec tokens $x_1, \dots, x_N$ as a product of conditionals:

$$
p(x_1, \dots, x_N) = \prod_{i=1}^{N} p(x_i \mid x_1, \dots, x_{i-1}).
$$

Each token is sampled given the ones before it. This is exactly the language-model recipe applied to the discrete tokens of a neural audio codec (see [neural audio codecs, the tokenizer of sound](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound) for what those tokens are). It has two structural consequences that hurt long-form music.

**Causality.** When the model decides token $x_i$, it has seen $x_1 \dots x_{i-1}$ but *nothing* after. Music is full of long-range, *forward-looking* structure — a verse implies a chorus, an eight-bar phrase resolves on its last beat, a build-up only makes sense because of the drop it sets up. A causal model has to *gamble* on that future and hope its earlier choices remain consistent with where it ends up. A non-causal model, by contrast, can shape the whole arc at once.

**Drift.** Because each conditional depends on a *sampled* (and therefore slightly imperfect) past, errors do not cancel — they accumulate. If the per-step probability of a small tempo or pitch error is tiny but nonzero, the probability of having drifted *somewhere* in $N$ steps grows roughly like $1 - (1-\epsilon)^N$, which for the millions-of-samples / thousands-of-tokens horizon of a three-minute song is uncomfortably close to one. This is why autoregressive music tends to be demoed in ten-to-thirty-second clips: that is the horizon over which drift stays inaudible.

Diffusion sidesteps both. A diffusion model does not factorize along time. It learns to reverse a noising process on the *entire* latent tensor at once:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar\alpha_t}}\, \epsilon_\theta(x_t, t, c) \right) + \sigma_t z,
$$

where $\epsilon_\theta$ is the trained denoiser, $c$ is the conditioning (text, timing), and the same update is applied to *every* position of the latent simultaneously. Every denoising step touches the intro, the body, and the outro together, with full bidirectional context. There is no serial "past" to drift from. (If that update looks unfamiliar, it is the standard DDPM reverse step; [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles) derives it.)

![A two-column comparison contrasting serial causal autoregressive generation that drifts over minutes with parallel full-context diffusion that holds global structure](/imgs/blogs/latent-diffusion-for-music-stable-audio-2.png)

The figure above is the whole argument in one picture. Three properties fall out of "denoise the whole thing in parallel":

1. **Parallel sampling.** A diffusion track of any length is produced in a *fixed* number of denoising steps — say fifty — each of which is one forward pass over the full latent. Tripling the song length does not triple the number of *sequential* operations; it just makes each pass operate on a longer tensor (cheaper to scale on a GPU than $3\times$ more serial steps). Autoregression, by construction, needs roughly three times as many *sequential* token decodes for a three-times-longer clip.

2. **Global structure.** Bidirectional context means the model can commit to a key, a tempo, and an arrangement once and enforce them everywhere. This is the structural reason diffusion music holds together over minutes where autoregressive music wanders.

3. **High stereo fidelity.** Because there is no autoregressive bottleneck forcing the signal through a narrow discrete token stream one step at a time, diffusion models comfortably operate on a *continuous* latent and decode to full 44.1 kHz stereo. Stereo is two correlated channels; a non-causal model that sees both channels of the whole clip at once can keep them phase-coherent in a way a token-by-token model finds awkward.

The cost — and there is always a cost — is **streaming**. Because diffusion needs the whole latent in memory and denoises it jointly, you cannot emit the first second before the last second exists. Autoregression streams naturally: token one is done before token two starts. Hold that thought; it is the decisive factor in the trade-off section.

There is a deeper way to see why the non-causal property matters, and it is worth a paragraph because it explains a real perceptual effect. Music has *bidirectional* dependencies that a strictly left-to-right model cannot honor. The classic example is an anacrusis — a pickup note that only makes sense because of the downbeat that follows it. A human composer writes the pickup *knowing* the downbeat is coming; a causal model writing the pickup has not yet decided the downbeat, so it must gamble. More pervasively, the *energy envelope* of a phrase — the way a four-bar build crescendos into a drop — is a forward-looking structure: the loudness at bar one is chosen *because* of the climax at bar four. A non-causal denoiser, which shapes bars one through four jointly in every denoising step, can make the build and the drop mutually consistent. This is not a subtle theoretical nicety; it is audible. Autoregressive music often has builds that do not quite "land" because the model committed to the build before it knew the payoff. Diffusion music lands its builds more reliably because the build and the payoff are decided together.

One more structural difference deserves naming, because it changes how you *debug* these models. In autoregression, a single bad sample early in the sequence poisons everything downstream — you can literally point to the token where the track went wrong, and everything after it inherited the mistake. In diffusion, there is no such "first wrong token." A bad generation is bad *globally and gradually* — the whole latent converged to a slightly-off solution. Practically, this means autoregressive failures are often *localizable* (and sometimes fixable by re-sampling from a point), while diffusion failures are *holistic* (you re-roll the whole thing with a different seed). When you are staring at a generation that is almost-but-not-quite right, knowing which failure mode you are in tells you what lever to pull.

#### Worked example: the drift budget for a three-minute song

Suppose an autoregressive model has a per-token probability $\epsilon = 0.0005$ of introducing an inaudibly small tempo error, and a three-minute stereo track at the codec's frame rate is about $N = 13{,}000$ token steps per codebook. The chance the track drifts *somewhere* is $1 - (1 - 0.0005)^{13000} \approx 1 - e^{-6.5} \approx 0.998$. Even at a tenth that error rate, $\epsilon = 0.00005$, you get $1 - e^{-0.65} \approx 0.48$ — a coin flip per render. These numbers are illustrative, not measured, but they capture *why* drift is a length problem: the failure probability is exponential in horizon. A diffusion model's quality, by contrast, does not degrade with length in this compounding way; it degrades (if at all) because the *latent gets longer than the contexts the model trained on*, which is a flat ceiling, not an exponential one.

## 2. The recipe in one paragraph: latent image diffusion, ported to sound

Here is the entire Stable Audio family in a sentence, and then we spend the rest of the post unpacking it: **take latent diffusion — the exact recipe behind Stable Diffusion, where you compress the signal with an autoencoder, run the diffusion in the small latent space, and decode back — and apply it to audio, with two audio-specific additions: a very high-compression audio autoencoder, and timing conditioning.**

If you have read [latent diffusion and Stable Diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion), the skeleton is familiar. Stable Diffusion does not diffuse on $512\times512\times3$ pixels; it diffuses on a $64\times64\times4$ VAE latent, roughly $48\times$ smaller, and only decodes to pixels at the end. The win is that diffusion's expensive part — the denoiser's many forward passes — runs on a tensor a fraction of the size. Latent audio diffusion is the same bargain. You do not diffuse on raw samples; you diffuse on a heavily compressed audio latent, and only decode to a waveform at the end.

![A layered stack showing a three-minute stereo waveform of millions of samples compressed by an oobleck encoder into a few thousand latent frames that fit a single diffusion context before decoding back](/imgs/blogs/latent-diffusion-for-music-stable-audio-6.png)

Why is the compression *even more* essential for audio than for images? Do the arithmetic. A three-minute stereo song at 44.1 kHz is

$$
180 \text{ s} \times 44{,}100 \text{ samples/s} \times 2 \text{ channels} \approx 1.59 \times 10^{7} \text{ samples}.
$$

That is roughly *sixteen million* numbers. No diffusion transformer is going to run fifty denoising passes over a sixteen-million-element sequence — the attention alone is quadratic in sequence length and would be astronomically expensive. Now suppose the audio autoencoder compresses by a factor of around $64\times$ along time (Stable Audio's "oobleck" VAE is in this regime — a fully convolutional variational autoencoder with strided convolutions that downsample heavily). The latent sequence becomes a few thousand frames, each a modest-dimensional vector. *That* fits a diffusion transformer's context. The high-compression VAE is not an optimization; it is the thing that makes minutes tractable at all.

The audio autoencoder here plays exactly the role the VAE plays in Stable Diffusion, and exactly the role a neural codec plays for autoregressive audio — it is the "tokenizer" of the signal, except its output is a *continuous* latent rather than discrete codes. If you want the codec-as-VAE analogy spelled out, [neural audio codecs, the tokenizer of sound](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound) and the image series' [variational autoencoders from scratch](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) are the two halves of it. The crucial property for diffusion is that the latent be *smooth and continuous* — diffusion adds and removes Gaussian noise, which is natural in a continuous space and awkward over discrete codes. That is one reason Stable Audio uses a VAE-style continuous latent rather than the discrete RVQ tokens an autoregressive model wants.

#### Worked example: how much does the latent actually shrink the problem?

Take the same three-minute stereo clip, about $1.59 \times 10^7$ samples. At an aggressive but realistic $\sim 64\times$ temporal compression with a latent channel dimension of 64, the latent has on the order of $180 \times (44100 / 64) \approx 124{,}000$ frame-steps times 64 channels — but the *sequence length the transformer attends over* is the frame count, on the order of $10^5$ before any patchifying and well under that after. The key number is the ratio: the denoiser operates on a sequence two orders of magnitude shorter than the raw waveform. Self-attention cost scales with the square of sequence length, so a $64\times$ shorter sequence is roughly $64^2 = 4096\times$ cheaper in the attention term. That is the difference between "impossible" and "runs on one A100." (Exact internal dimensions vary by Stable Audio version; treat these as order-of-magnitude.)

### The rate-distortion floor: why you cannot just compress harder

It is tempting to read "more compression buys more length" and conclude you should compress as hard as possible. The autoencoder stops you. A VAE — like any lossy codec — sits on a *rate-distortion curve*: at a given model capacity, the reconstruction error rises as you push the bitrate (here, the latent's information content per second) down. Formally, the rate-distortion function $R(D)$ gives the minimum bits per second needed to reconstruct within distortion $D$; as you demand lower rate (more compression, more length), the achievable distortion $D$ rises. For audio the distortion you care about is *perceptual*, not mean-squared — a small numerical error in a high-frequency band where the ear is sensitive is far worse than a large error in a band the ear masks. (The residual-VQ version of this exact curve, for discrete codecs, is derived in [residual vector quantization (RVQ)](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq); the continuous-VAE version here obeys the same shape.)

What this means in practice: the oobleck VAE's compression ratio is not a free parameter you crank for length. It is chosen at the point where the decoder still reconstructs *perceptually transparent* audio — where a careful listener cannot reliably tell the VAE round-trip (encode then decode, no diffusion) from the original. Stable Audio 2's jump to three minutes was not "compress 2× harder and accept the quality hit"; it was a *better* autoencoder that moved the rate-distortion curve outward, buying more compression at the *same* perceptual fidelity. That distinction — moving the curve versus sliding along it — is the difference between a real architecture improvement and a quality regression dressed up as a length feature.

### The diffusion objective, in the audio latent

The training objective is the standard denoising one, applied to the VAE latent rather than to pixels or samples. You encode an audio clip to a latent $z_0 = \mathcal{E}(x)$, sample a timestep $t$ and Gaussian noise $\epsilon$, form the noised latent $z_t = \sqrt{\bar\alpha_t}\, z_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon$, and train the network to predict the noise:

$$
\mathcal{L} = \mathbb{E}_{z_0, \epsilon, t, c}\left[\; \big\| \epsilon - \epsilon_\theta(z_t, t, c) \big\|^2 \;\right],
$$

where $c$ bundles the text and timing conditioning. This is *identical* to the latent image diffusion loss — the only audio-specific parts are (a) $\mathcal{E}$ is an audio VAE, not an image VAE, and (b) $c$ contains the timing embeddings. (Some implementations predict a velocity $v$ or use a flow-matching objective rather than $\epsilon$; the parameterization changes the target but not the picture. The flow-matching variant for audio is the subject of [flow matching and consistency for audio](/blog/machine-learning/audio-generation/flow-matching-and-consistency-for-audio), and the image-series derivation is [flow matching and rectified flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow).) The point worth internalizing is how *little* is audio-specific in the generative math — almost the entire derivation is inherited from the image series. The audio engineering is in the VAE and the conditioning, which is exactly where this post spends its depth.

## 3. The denoiser: a diffusion transformer, not a U-Net

Early latent audio diffusion (AudioLDM, AudioLDM 2) inherited the convolutional U-Net denoiser from image diffusion. Stable Audio's later versions moved to a **diffusion transformer (DiT)** — the same architectural shift that image generation made, and for the same reasons: transformers scale more predictably with data and compute, and they handle long sequences and rich conditioning more gracefully than a U-Net's fixed convolutional inductive bias. If you want the DiT architecture in full — patchify, transformer blocks, adaptive layer norm conditioning (adaLN-Zero) — read [diffusion transformers (DiT)](/blog/machine-learning/image-generation/diffusion-transformers-dit). I will only summarize the audio-relevant parts here.

A DiT denoiser for audio does this, per denoising step:

1. **Patchify the latent.** The continuous audio latent (a `[channels, frames]` tensor) is split into patches and linearly projected into a sequence of tokens — the transformer's working units. For audio the "patches" are along the time axis.
2. **Add the conditioning.** The timestep $t$, the text embedding, and — the audio-specific part — the timing embeddings are combined and injected into every transformer block, typically through adaptive layer norm: the block's normalization gain and bias are *predicted* from the conditioning vector. This is the adaLN-Zero mechanism, and it is how DiT conditions without burning sequence positions on the conditioning.
3. **Run the transformer blocks.** Self-attention over the latent tokens (every position sees every other — this is the non-causal, full-context property that matters for music) plus cross-attention into the text tokens.
4. **Unpatchify** back to a latent-shaped noise prediction $\epsilon_\theta$.

The non-causal self-attention in step 3 is the architectural embodiment of "global structure." Every latent frame attends to every other latent frame, so a denoising step at the two-minute mark is informed by the latent at the ten-second mark and vice versa. That is precisely what an autoregressive model cannot do.

There is a real cost to "every frame attends to every frame," and it is worth being honest about: self-attention is quadratic in the latent length. For a three-minute track the latent is the longest the model handles, so attention is the dominant compute term, and it is the practical reason the VAE compression ratio matters so much — every factor of 2 you compress the latent is a factor of 4 you save in attention. Stable Audio's design lives in the tension between "compress enough that attention over three minutes is affordable" and "do not compress so hard the decoder loses fidelity" (the rate-distortion floor from Section 2). Some long-form audio diffusion systems mitigate the quadratic cost with windowed or sparse attention, trading a little global context for cheaper compute; the cleanest design keeps full attention and pays for it with an aggressive-but-transparent VAE.

### The sampler: turning the trained denoiser into audio

A trained $\epsilon_\theta$ is not yet a generator — you need a *sampler* (a scheduler, in diffusers terms) that runs the reverse process. The sampler is where the step-count budget lives, and it is a knob entirely separate from the trained weights. The original Stable Audio and AudioLDM lineage used many-step samplers (DDPM/DPM-style) in the low hundreds of steps; the DiT-era models pair with faster ODE samplers that reach good quality in roughly 50 steps. The relationship is the same one image diffusion taught us: each sampler defines a trajectory from noise to data, and better samplers reach the data manifold in fewer steps.

```python
from diffusers import StableAudioPipeline, DPMSolverMultistepScheduler

pipe = StableAudioPipeline.from_pretrained(
    "stabilityai/stable-audio-open-1.0", torch_dtype="auto"
)

# swap the scheduler to a faster multistep solver; fewer steps for the same quality
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# now you can often drop num_inference_steps and keep quality
audio = pipe(
    prompt="ambient drone, slow evolving pads",
    num_inference_steps=50,      # down from ~100 thanks to the better sampler
    audio_end_in_s=20.0,
    guidance_scale=7.0,
).audios
```

The lesson — identical to image diffusion, so I will state it once and move on — is that *sampler choice and step count are a quality-versus-speed dial you tune separately from the model*. Few-step distillation (consistency models and friends) pushes this even further; the audio version is in [flow matching and consistency for audio](/blog/machine-learning/audio-generation/flow-matching-and-consistency-for-audio), and the image-series derivation is [consistency models and few-step generation](/blog/machine-learning/image-generation/consistency-models-and-few-step-generation).

Here is a stripped-down, runnable sketch of how the conditioning is assembled and a DiT-style block consumes it. This is illustrative PyTorch, not the exact Stable Audio code, but it shows the real mechanism — how the timing embedding becomes part of the per-block modulation.

```python
import torch
import torch.nn as nn

class TimingEmbedding(nn.Module):
    """Turn (seconds_start, seconds_total) into a learned conditioning vector."""
    def __init__(self, dim=768, max_seconds=512):
        super().__init__()
        # one learned table per timing scalar, looked up by integer second
        self.start_emb = nn.Embedding(max_seconds, dim)
        self.total_emb = nn.Embedding(max_seconds, dim)

    def forward(self, seconds_start, seconds_total):
        s = self.start_emb(seconds_start.clamp(max=511))
        t = self.total_emb(seconds_total.clamp(max=511))
        return s + t                      # [batch, dim]

class DiTBlockWithAdaLN(nn.Module):
    """A DiT block whose norm gain/shift come from the conditioning vector."""
    def __init__(self, dim=768, heads=12):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        # adaLN: predict scale and shift for the norm from conditioning
        self.modulate = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, cond):           # x: [batch, frames, dim]
        scale, shift = self.modulate(cond).chunk(2, dim=-1)   # [batch, dim]
        h = self.norm1(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        attn_out, _ = self.attn(h, h, h)  # non-causal: every frame sees every frame
        return x + attn_out

# assemble conditioning: timestep + text + timing all live in one vector space
dim = 768
timing = TimingEmbedding(dim)
seconds_start = torch.tensor([0])         # this latent starts at t=0 s
seconds_total = torch.tensor([180])       # we want a 180 s track
timing_vec = timing(seconds_start, seconds_total)   # [1, 768]

text_vec = torch.randn(1, dim)            # stand-in for a pooled T5/CLAP embedding
t_vec = torch.randn(1, dim)               # stand-in for the sinusoidal timestep embed
cond = timing_vec + text_vec + t_vec      # the global conditioning vector

block = DiTBlockWithAdaLN(dim)
latent = torch.randn(1, 2000, dim)        # ~2000 latent frames for the clip
out = block(latent, cond)
print(out.shape)                          # torch.Size([1, 2000, 768])
```

The single most important line is `cond = timing_vec + text_vec + t_vec`. The timing information is not a side input bolted onto the decoder; it lives in the *same conditioning vector* that drives every transformer block's modulation, right alongside the text and the diffusion timestep. That is what lets a *single* model produce variable-length, structured output — the next section is entirely about why.

## 4. The timing trick: how one model makes variable-length, structured tracks

This is the heart of the post, so let me build it carefully.

The naive way to make a fixed-architecture diffusion model is to pick a single output length — say, a ten-second window — train it on ten-second clips, and always emit ten seconds. If you want something shorter, you generate ten seconds and truncate. If you want something longer, you are out of luck (or you stitch clips and pray they line up). Worse, a model trained only on ten-second windows has *no notion of a beginning or an end*. It learned to fill the window with plausible texture. Ask it for a song and it gives you a loop that could start or stop anywhere — no intro, no outro, no sense of arrival.

![A two-column before-and-after contrasting a fixed window that pads with silence and has no ending against a timing-conditioned model that places an intro body and outro and resolves on time](/imgs/blogs/latent-diffusion-for-music-stable-audio-4.png)

Timing conditioning solves this with a beautifully simple idea: during training, **tell the model two numbers** — `seconds_start`, where in the original full-length track this training chunk begins, and `seconds_total`, how long the full track is — and embed both as conditioning. Now the model is not learning "fill a window with texture." It is learning "produce the audio that belongs at *this* position in a track of *that* total length." A chunk with `seconds_start = 0` and `seconds_total = 180` should sound like the *opening* of a three-minute piece. A chunk with `seconds_start = 170, seconds_total = 180` should sound like an *ending* — resolving, fading, landing on the tonic.

At inference, you set these numbers to whatever you want:

- `seconds_start = 0, seconds_total = 30` → the model generates a self-contained thirty-second piece, with an opening and a close, because it learned what "a thirty-second whole" sounds like.
- `seconds_start = 0, seconds_total = 180` → a three-minute piece that paces itself across three minutes: it does not blow its whole arrangement in the first thirty seconds, because it "knows" it has three minutes to fill.

The variable length comes from a second lever: the latent the model denoises is *sized to the requested duration*. Because the VAE compression ratio is fixed, a requested 30 seconds maps to a latent of one length and 180 seconds maps to a latent roughly six times longer. The DiT denoises whatever length it is handed. The timing embedding tells it *what that length means musically*; the latent size tells it *how many frames to fill*. Together they give you "any length up to the trained maximum, structured appropriately."

![A branching dataflow showing total-duration and start-offset timing embeddings joining the text and timestep signals into one conditioning sum that drives every DiT block](/imgs/blogs/latent-diffusion-for-music-stable-audio-3.png)

The figure traces the injection path. The two timing scalars (`seconds_start`, `seconds_total`) hit a small learned embedding table, producing timing vectors. Those are *summed* with the text embedding (from T5 or CLAP) and the sinusoidal timestep embedding into a single global conditioning vector, which then modulates every DiT block through adaptive layer norm. The reason this is hard for a *fixed-window* model to replicate is subtle: a fixed-window model has no parameter that represents "where am I in the larger whole," so it physically cannot distinguish "the start of a long song" from "a standalone short clip." Timing conditioning gives the model that parameter, learned, continuous, and present in every layer.

Why is variable length genuinely hard, scientifically, and not just an engineering convenience? Because a diffusion model's denoiser has a *fixed* architecture but must operate on *variable-length* latents and still produce *coherent global structure* at every length. Two things make that work. First, the denoiser is a transformer — it is length-agnostic by design (attention works over any sequence length), unlike a fixed-input MLP. Second, the timing embedding *factorizes* the structure problem: instead of needing the network to infer "how much song is left" from the latent content alone (which is genuinely ambiguous mid-track — a held chord could be a pause or an ending), you *hand it* the global timing as conditioning. You have converted an inference problem (guess the structure) into a conditioning problem (you are told the structure). That is the conceptual move, and it is why timing conditioning, not a bigger model, is what unlocked long-form.

#### Worked example: same prompt, three lengths

Take the prompt "warm lo-fi hip-hop beat, vinyl crackle, mellow Rhodes." Run it three times with the timing knob set differently:

- `seconds_total = 11`: you get a tight loop-able phrase — one chord progression, a clean two-bar feel, no real development. Useful as a sample.
- `seconds_total = 47`: enough room for a small arc — an intro bar, a settled groove, a soft turnaround. This is the "social-clip" length.
- `seconds_total = 95` (near Stable Audio 1's max): the model paces an intro, a main section, and a wind-down. The *same* prompt, but the structure scales with the requested duration because the timing embedding told the denoiser how much canvas it had.

The audible difference between these is not "more of the same texture" — it is *different pacing of musical events*, which is exactly what timing conditioning buys you. (Lengths here track Stable Audio 1's training regime, which centered on clips up to roughly 95 seconds; Stable Audio 2 pushed the maximum to about three minutes.)

## 5. Running Stable Audio in diffusers

Enough theory. Here is the practical flow in 🤗 `diffusers`, which ships a `StableAudioPipeline`. This is real, copy-and-adapt code. I will time it and export a stereo wav.

```python
import torch
import soundfile as sf
import time
from diffusers import StableAudioPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableAudioPipeline.from_pretrained(
    "stabilityai/stable-audio-open-1.0",
    torch_dtype=dtype,
)
pipe = pipe.to(device)

prompt = "warm lo-fi hip-hop beat, vinyl crackle, mellow Rhodes chords, 80 BPM"
negative_prompt = "low quality, distorted, harsh, clipping"

# the timing knob: ask for a specific total duration
generator = torch.Generator(device=device).manual_seed(0)

t0 = time.time()
audio = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=100,        # denoising steps; quality/speed knob
    audio_end_in_s=30.0,            # <-- timing conditioning: total duration
    num_waveforms_per_prompt=1,
    guidance_scale=7.0,             # classifier-free guidance strength
    generator=generator,
).audios
elapsed = time.time() - t0

# audio shape: [batch, channels, samples] -> take first, transpose to [samples, channels]
wav = audio[0].T.float().cpu().numpy()    # stereo: [samples, 2]
sr = pipe.vae.sampling_rate               # 44100 for stable-audio-open
sf.write("lofi_30s.wav", wav, sr)

dur = wav.shape[0] / sr
print(f"generated {dur:.1f}s of stereo audio in {elapsed:.1f}s "
      f"(RTF {elapsed / dur:.2f}) at {sr} Hz")
```

The line that does the work is `audio_end_in_s=30.0`. That is the timing conditioning, exposed as a clean API knob. Set it to 47, or to 95, and you get a different-length, differently-paced track from the *same* prompt and the *same* weights. The pipeline internally (a) sizes the latent to the requested duration, (b) builds the timing embedding, (c) runs the DiT denoiser for `num_inference_steps`, and (d) decodes the latent through the VAE to a 44.1 kHz **stereo** waveform — note `wav` has shape `[samples, 2]`.

A few knobs worth understanding:

- **`num_inference_steps`** is the diffusion step count — the dominant speed/quality lever. Stable Audio Open's published defaults are in the low hundreds for the original sampler; the later DiT versions get good quality in roughly 50 steps. Fewer steps means faster but rougher; more means slower but cleaner. This is the same step-count economics as image diffusion (see the [diffusion-for-audio](/blog/machine-learning/audio-generation/diffusion-for-audio) sibling for the speech-vocoder version of this budget).
- **`guidance_scale`** is classifier-free guidance — how hard the model leans on the text prompt versus the unconditional model. The audio version works exactly like the image one; [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance) derives it. Too low and the text barely steers; too high and you get harsh, over-saturated audio (the audio analogue of the over-contrasted look you get from a too-high CFG image).
- **`negative_prompt`** steers *away* from undesired qualities — "distorted, clipping" is a useful default for music.

#### Worked example: the seconds-to-generate budget for a 30-second clip

Here is the honest RTF math, marked approximate because exact timings depend on the GPU, the sampler, and the diffusers version. On an A100 80GB, generating a 30-second stereo clip at 100 sampling steps with Stable Audio Open lands in the rough ballpark of 8 to 20 seconds of wall-clock time — call it an RTF (generation-time ÷ audio-duration) of roughly 0.3 to 0.7. Crucially, that RTF is *fixed in the step count, not the length*: a 95-second clip at the same 100 steps is *not* three times slower in *sequential* work — it is the same number of denoising passes over a longer latent, so the wall-clock grows sub-linearly with length on a GPU that can parallelize the longer sequence. Contrast an autoregressive model, whose wall-clock grows roughly linearly with length because every extra second is more sequential token decodes. To measure this honestly: do a warm-up generation first (the first call pays compilation and cache costs), fix the seed, and report the median of several runs on a named device. Always say which GPU — an RTF number without a device is meaningless.

## 6. Stable Audio 2.0: full three-minute songs, stereo, and audio-to-audio

Stable Audio 1 / Stable Audio Open generates high-quality stereo clips up to roughly 95 seconds. **Stable Audio 2.0** pushed this to a coherent **full-length track of about three minutes** — long enough for an actual song structure: intro, verse-like section, a change, an outro that resolves. Three things make that jump work, and all three are themes we have already met.

**A higher-compression, higher-fidelity autoencoder.** To fit three minutes in the denoiser's context, you need the latent even shorter per second of audio than before, *without* losing fidelity. Stable Audio 2's VAE (the oobleck-style continuous autoencoder) is tuned for exactly this — a higher temporal compression ratio so that three minutes maps to a latent the DiT can attend over, while the decoder still reconstructs clean 44.1 kHz stereo. This is the same lever from Section 2, turned harder: *fewer latents per second → minutes become tractable.* The fidelity-versus-compression tension is the rate-distortion trade you meet everywhere in audio (the codec version of it lives in [residual vector quantization (RVQ)](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq)); push compression too far and the decoder starts smearing high-frequency detail, which on music sounds like a loss of "air" and cymbal sheen.

**A DiT denoiser.** Stable Audio 2 leans on the diffusion-transformer denoiser (Section 3), which scales to the longer latent and the longer-range structure better than the original U-Net would. The transformer's non-causal attention over a three-minute latent is what keeps the key and tempo consistent from the first bar to the last.

**Audio-to-audio conditioning.** Beyond text, Stable Audio 2 added the ability to condition on an *input audio clip* — give it a reference and a text prompt and it transforms or extends the reference rather than starting from pure noise. Mechanically this is the audio analogue of image-to-image diffusion: instead of initializing the latent from pure Gaussian noise, you partially noise the *encoded reference latent* and denoise from there, so the output keeps the reference's broad structure while the text reshapes its character. The same `strength`-style knob from image-to-image applies: a low strength stays close to the reference, a high strength wanders further.

Here is the audio-to-audio idea expressed against the `StableAudioPipeline` interface, which accepts an `initial_audio_waveforms` argument. The mechanism is exactly the partial-noise-and-denoise of image-to-image, so what you already know from the image series transfers wholesale.

```python
import torch
import torchaudio
import soundfile as sf
from diffusers import StableAudioPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableAudioPipeline.from_pretrained(
    "stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16
).to(device)

# load a reference clip and match the model's sample rate
ref, sr = torchaudio.load("reference_loop.wav")          # [channels, samples]
target_sr = pipe.vae.sampling_rate                       # 44100
if sr != target_sr:
    ref = torchaudio.functional.resample(ref, sr, target_sr)
ref = ref.unsqueeze(0).to(device, torch.float16)         # [1, channels, samples]

generator = torch.Generator(device=device).manual_seed(0)
audio = pipe(
    prompt="the same groove, but with a warm analog synth lead on top",
    initial_audio_waveforms=ref,     # condition on the reference, not pure noise
    num_inference_steps=100,
    audio_end_in_s=30.0,
    guidance_scale=7.0,
    generator=generator,
).audios

wav = audio[0].T.float().cpu().numpy()
sf.write("transformed_loop.wav", wav, target_sr)
```

The `initial_audio_waveforms=ref` argument is the whole feature. Drop it and you are doing pure text-to-audio from noise; supply it and you are doing audio-to-audio. This is how you do style transfer, variation, and continuation — and it is one more place where the audio recipe is *the same code shape* as the image recipe, with the VAE and conditioning swapped for audio.

![A row-by-column matrix comparing AudioLDM 2, Stable Audio 1, and Stable Audio 2 across domain, maximum length, stereo support, conditioning, and sampling steps](/imgs/blogs/latent-diffusion-for-music-stable-audio-7.png)

The progression in the matrix is the story of the family: AudioLDM 2 is a general short-clip model (speech, sound effects, and music up to roughly ten seconds, mono, CLAP-conditioned); Stable Audio 1 specializes in music and sound effects, goes stereo, adds timing, and reaches ~95 seconds; Stable Audio 2 reaches full three-minute songs with audio-to-audio and a DiT backbone. The through-line is *length and fidelity climbing while step counts stay modest* — exactly the regime where diffusion's parallel sampling pays off.

#### Worked example: the long-form coherence stress test

Ask Stable Audio 2 for "an upbeat synthwave track with a clear build and a drop, 3 minutes." What does it get right and where does it strain? It gets *global* consistency right — the key and tempo hold across all three minutes, because the non-causal DiT enforces them everywhere at once; you will not hear the autoregressive-style drift from Section 1. Where it strains is *fine-grained long-range narrative* — a human producer's "the second chorus is bigger than the first, and the bridge introduces a new motif that pays off in the final chorus." Diffusion holds *texture and key* over minutes far better than it composes *deliberate, evolving song-writing*. So the honest stress-test verdict: timing conditioning plus a high-compression latent absolutely make a coherent three-minute *texture-and-groove* tractable; they do not (yet) make the model a songwriter. That gap — coherent-vs-composed — is where the commercial vocal-music systems are pushing next; the [Suno and Udio frontier](/blog/machine-learning/audio-generation/suno-udio-and-the-commercial-music-frontier) post takes that up.

## 7. AudioLDM and AudioLDM 2: a "language of audio"

Stable Audio is the music-specialized end of latent audio diffusion. **AudioLDM** and **AudioLDM 2** are the *general* end — latent diffusion models that aim to generate *any* audio (speech, music, and sound effects) from text, conditioned on **CLAP** embeddings.

CLAP — Contrastive Language-Audio Pretraining — is the audio twin of CLIP: it learns a *shared* embedding space where a text description and the audio it describes land near each other. AudioLDM's clever move is to condition the diffusion model on CLAP embeddings, and — because text and audio share the space — it can be *trained* using audio embeddings (no text caption needed for every clip) and *prompted* at inference with text embeddings. The text-audio alignment is what makes "type a caption, get the matching sound" work. (CLAP and the other conditioning encoders are covered in depth in [conditioning and control in audio generation](/blog/machine-learning/audio-generation/conditioning-and-control-in-audio-generation).)

AudioLDM 2 generalized this further into what its authors call a "**language of audio**" — a unified representation (built around a self-supervised audio model) that lets one architecture generate speech, music, *and* sound effects, where earlier systems needed a separate model per domain. The pitch is unification: one latent diffusion backbone, one conditioning interface, three domains.

The scientific bet behind "a language of audio" is worth pausing on, because it is the same bet that drives much of modern audio modeling. The claim is that speech, music, and sound effects, however different they sound, share a *common latent structure* — the same way that a single language model can write code, poetry, and email because they share the deep structure of token sequences. AudioLDM 2 instantiates this by predicting a shared intermediate representation (derived from a self-supervised model that learned general audio features) and then decoding that representation to a waveform through a latent diffusion stage. If the bet holds, you get *transfer*: training on abundant speech and effects data helps the music generation, and one model amortizes across three problems. The honest counterpoint — and the reason Stable Audio specializes rather than unifies — is that *depth* in any one domain (stereo, 44.1 kHz, three-minute structure for music) may demand domain-specific choices a generalist cannot afford. Unification and specialization are a real trade, not a settled question, and the two model families embody the two answers.

Here is AudioLDM 2 in `diffusers`. The API is the same shape as Stable Audio's — note that AudioLDM 2 is **mono** and short-form, which is exactly its trade against Stable Audio's stereo long-form.

```python
import torch
import soundfile as sf
import time
from diffusers import AudioLDM2Pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2", torch_dtype=dtype)
pipe = pipe.to(device)

prompt = "a dog barking, then a distant thunderclap, outdoor ambience"
negative_prompt = "low quality, average quality, noise"

generator = torch.Generator(device=device).manual_seed(0)
t0 = time.time()
audio = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=200,        # AudioLDM2 default is in the low hundreds
    audio_length_in_s=10.0,         # this model is short-form
    num_waveforms_per_prompt=1,
    guidance_scale=3.5,
    generator=generator,
).audios
elapsed = time.time() - t0

wav = audio[0]                       # mono: [samples]
sr = 16000                          # AudioLDM2 generates at 16 kHz
sf.write("sfx_10s.wav", wav, sr)
print(f"generated {len(wav)/sr:.1f}s mono in {elapsed:.1f}s "
      f"(RTF {elapsed/(len(wav)/sr):.2f})")
```

Two practical contrasts jump out against the Stable Audio code. First, **no `audio_end_in_s` timing knob with the same meaning** — AudioLDM 2 has `audio_length_in_s` but is fundamentally a short-form model; it does not do the structured-long-form trick that timing conditioning enables in Stable Audio. Second, **mono at 16 kHz** versus Stable Audio's **stereo at 44.1 kHz**. AudioLDM 2's strength is *breadth* (it will give you speech, a sound effect, or a short musical idea from one model); Stable Audio's strength is *music depth* (stereo, long, high sample rate). When your job is "a foley sound effect" or "a short ambient bed," AudioLDM 2's unification is the right tool; when it is "a stereo music track," Stable Audio is. The sound-effect and foley side of this is the subject of the [text-to-audio and sound effects](/blog/machine-learning/audio-generation/text-to-audio-and-sound-effects) post.

## 8. The diffusion-versus-MusicGen trade, decided honestly

Now the comparison the whole post has been building toward: latent diffusion (Stable Audio) versus the autoregressive codec language model (MusicGen) for music. These are the two dominant open paradigms, and they trade cleanly.

![A five-row matrix comparing diffusion Stable Audio against autoregressive MusicGen on maximum length, sampling style, stereo fidelity, melody control, and streaming](/imgs/blogs/latent-diffusion-for-music-stable-audio-5.png)

Read the matrix row by row:

- **Length.** Diffusion wins. Timing conditioning plus a high-compression latent give Stable Audio coherent multi-minute tracks; MusicGen is happiest in the ~30-second range before drift and context limits bite.
- **Sampling.** Diffusion is parallel (a fixed number of full-latent passes); MusicGen is serial (token by token). For long clips, diffusion's wall-clock scales better with length.
- **Stereo fidelity.** Diffusion wins. Continuous-latent diffusion decodes to 44.1 kHz stereo cleanly; MusicGen's base models are 32 kHz mono (stereo variants exist but it is not the native strength).
- **Melody control.** *MusicGen* wins. MusicGen-melody conditions on a chromagram of a reference melody — hum a tune and it generates a track that follows your *exact* melodic contour, time-aligned. Stable Audio conditions on text and (in 2.0) reference audio, but does not have the same tight, time-aligned melodic steering. If your job is "make a track that follows this specific melody," autoregressive melody conditioning is the better tool. (The mechanism is in [music generation with MusicLM and MusicGen](/blog/machine-learning/audio-generation/music-generation-musiclm-and-musicgen).)
- **Streaming.** *MusicGen* wins, decisively. Autoregression emits the first token before the last exists, so it can stream audio out as it generates. Diffusion needs the whole latent denoised jointly — you wait for the full render before you hear anything. For an interactive or real-time application, this is often the deciding factor.

The honest one-line summary: **diffusion for length, fidelity, and parallelism; autoregression for streaming and tight melody control.** Neither dominates. The right choice is a function of the job, not a ranking of the models.

Here is the same trade as a markdown table you can drop into a design doc. The "decides when" column is the part that actually matters — it tells you which property is load-bearing for a given job.

| Property | Diffusion (Stable Audio) | Autoregressive (MusicGen) | Decides when |
|---|---|---|---|
| Max coherent length | Minutes (timing + compression) | Tens of seconds | You need a full track, not a loop |
| Sampling | Parallel, fixed step count | Serial, token by token | Long clips on a GPU |
| Latency to first audio | Whole clip only | Streams immediately | Interactive / real-time use |
| Stereo + sample rate | 44.1 kHz stereo native | 32 kHz mono base | High-fidelity music delivery |
| Melody control | Text + reference audio | Time-aligned chromagram | "Follow this exact tune" |
| Global structure | Strong (non-causal) | Weaker (drift over minutes) | Key/tempo must hold for minutes |
| Failure mode | Holistic (re-roll seed) | Localizable (early token poisons) | How you debug a bad take |

Read the rightmost column top to bottom and you have a checklist: if any row's "decides when" is the dominant requirement for your job, that row picks the engine. Length, structure, stereo, and parallel batch rendering point at diffusion; streaming, melody-following, and short interactive clips point at autoregression. Most real disagreements between the two come down to *streaming* — it is the one property no diffusion knob can buy back.

#### Worked example: choosing an engine for two real jobs

*Job A — a background music bed for a three-minute product video.* You need: length (three minutes), stereo, coherent structure, no real-time constraint (it renders offline), and text control ("calm corporate piano, building gently"). This is Stable Audio's sweet spot on every axis. Diffusion wins; the lack of streaming does not matter because nothing is listening live.

*Job B — an interactive music toy where the user hums a melody and hears an arrangement form in near-real-time.* You need: melody-following (the user's exact tune), streaming (audio out as it generates, low latency), and short clips are fine. This is MusicGen-melody's sweet spot — chromagram melody conditioning plus token streaming. Autoregression wins; Stable Audio's superior length and stereo are irrelevant here, and its non-streaming nature is disqualifying.

Same field, opposite answers. That is the trade in practice.

## 9. Pushing past the trained length: overlapping segments

Timing conditioning gives you any length *up to the model's trained maximum* — roughly 95 seconds for Stable Audio 1, roughly three minutes for Stable Audio 2. What if you need *more*? You leave the single-pass regime and stitch, and it is worth seeing how, because it exposes exactly what the single-pass model was protecting you from.

The naive approach — generate two independent clips and concatenate — fails audibly. The two clips have no shared context, so at the seam the key, tempo, and arrangement jump. The fix is *overlapping generation with audio-to-audio conditioning*: generate the first segment, then generate the next segment *conditioned on the tail of the first* (via the `initial_audio_waveforms` path from Section 6), so the new segment continues the groove rather than starting fresh. You overlap the segments by a few seconds and crossfade the overlap to hide any residual seam.

```python
import numpy as np
import torch
import soundfile as sf
from diffusers import StableAudioPipeline

pipe = StableAudioPipeline.from_pretrained(
    "stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16
).to("cuda")
sr = pipe.vae.sampling_rate

def crossfade(a, b, overlap_samples):
    """Linear-crossfade the tail of a into the head of b."""
    fade = np.linspace(0.0, 1.0, overlap_samples)[:, None]   # [n, 1] for stereo
    a_tail = a[-overlap_samples:]
    b_head = b[:overlap_samples]
    blended = a_tail * (1 - fade) + b_head * fade
    return np.concatenate([a[:-overlap_samples], blended, b[overlap_samples:]], axis=0)

prompt = "driving techno, four-on-the-floor, dark analog bassline, 128 BPM"
overlap_s = 4
seg_len_s = 30

# first segment from pure noise
seg = pipe(prompt=prompt, num_inference_steps=100, audio_end_in_s=seg_len_s,
           guidance_scale=7.0).audios[0].T.float().cpu().numpy()   # [samples, 2]
song = seg

for _ in range(3):                       # extend three more times -> ~2 minutes
    # condition the next segment on the tail of what we have so far
    tail = torch.tensor(song[-overlap_s * sr:].T, dtype=torch.float16,
                        device="cuda").unsqueeze(0)               # [1, 2, n]
    nxt = pipe(prompt=prompt, initial_audio_waveforms=tail,
               num_inference_steps=100, audio_end_in_s=seg_len_s,
               guidance_scale=7.0).audios[0].T.float().cpu().numpy()
    song = crossfade(song, nxt, overlap_s * sr)

sf.write("extended_track.wav", song, sr)
print(f"stitched {song.shape[0]/sr:.0f}s from overlapping {seg_len_s}s segments")
```

This *works*, but it reintroduces exactly the problem the single-pass model avoided: a stitched track has no *global* plan. Each segment continues the last locally, but nothing enforces a song-wide arc — you can get a track that grooves consistently yet never builds to anything, because no single denoising pass ever saw the whole thing. That is the honest limit. Stitching buys you arbitrary length at the cost of the global coherence that made single-pass diffusion attractive in the first place. The lesson generalizes: **prefer to stay within the trained length and let one pass plan the whole track; only stitch when the length requirement is hard and you can accept a flatter global arc.** This is also why the field keeps pushing the *trained* maximum length up (95 s → 3 min → beyond) rather than relying on stitching — single-pass coherence is the prize.

#### Worked example: the seam-quality budget

Quantify the stitching trade. With a 4-second crossfade between 30-second segments, you keep roughly 26 seconds of "fresh" audio per segment and spend 4 seconds blending. Over a 2-minute stitched track that is four seams, so about 16 seconds (13%) of the track is crossfade region where two generations are blended — usually inaudible if the segments share key and tempo (which the audio-to-audio conditioning enforces), but audible as a slight "smearing" if the two segments disagreed on a transient like a kick. The tunable is the overlap: longer overlap hides seams better but wastes more compute and dilutes more of the track into blend regions. Around 2 to 4 seconds is a reasonable default for music at a steady tempo. Compare this to a *native* 2-minute generation from Stable Audio 2, which has *zero* seams and one global plan — which is why, whenever the length fits, the single-pass model wins.

## 10. Case studies and real numbers

Let me anchor this with named, sourced results. I will flag every figure that is approximate.

**Stable Audio Open 1.0 (Evans et al., Stability AI, 2024).** Open-weights, generates up to ~47 seconds (the Open release's training horizon) of 44.1 kHz stereo audio, conditioned on a T5 text encoder, denoising a continuous oobleck-VAE latent. Its FAD (the standard music-generation fidelity metric — Fréchet Audio Distance, the audio analogue of FID, measuring the distance between embedding distributions of generated and real audio; lower is better) on its benchmark is competitive with prior open music models, and it is the model the `StableAudioPipeline` code in Section 5 loads. The headline scientific contribution is the timing-conditioning recipe that lets one model emit variable-length structured clips.

**Stable Audio 2.0 (Stability AI, 2024).** The commercial successor, generating coherent full tracks up to ~3 minutes at 44.1 kHz stereo, with text *and* audio-to-audio conditioning, on a DiT backbone. The headline is *length*: the jump from sub-minute clips to multi-minute song-length structure, enabled by the higher-compression autoencoder plus the transformer denoiser. (Exact FAD/benchmark numbers vary by eval set; treat cross-model FAD comparisons cautiously — FAD is sensitive to the embedding model and sample count, a pitfall the [audio quality metrics](/blog/machine-learning/audio-generation/audio-quality-metrics) post details.)

**AudioLDM / AudioLDM 2 (Liu et al., 2023).** Latent diffusion over a mel/VAE latent, CLAP-conditioned, generating speech, music, and sound effects from text. AudioLDM 2's "language of audio" unifies the three domains under one backbone. It generates ~10-second mono clips at 16 kHz by default. Its contribution is *generality and the CLAP-conditioning trick* (train on audio embeddings, prompt with text embeddings), not music length or fidelity — which is exactly the trade against Stable Audio.

**MusicGen (Copet et al., Meta, 2023).** The autoregressive counterpoint. A single-stage codec language model over EnCodec tokens with a codebook-interleaving pattern, in sizes from ~300M to ~3.3B parameters, with a melody-conditioned variant. Generates 32 kHz mono (and stereo variants), tens of seconds at a time, *streamable*, with tight melody control. The headline scientific trick is the *delay pattern* / codebook interleaving that lets a single autoregressive transformer model EnCodec's multiple parallel codebooks without a separate model per codebook — a real engineering win that keeps it a one-pass model. The honest comparison: MusicGen and Stable Audio are the two open workhorses, and the choice between them is the trade in Section 8, not a quality ranking. Notably, the existence of a *melody-conditioned* MusicGen is the single clearest example of the control axis where autoregression still leads — there is no equally tight melody-following knob on the diffusion side.

#### Worked example: the cost of a music-bed service

Put rough dollars on it. Suppose you run a service that renders 30-second stereo music beds on a single rented A100 80GB at roughly \$2/hr. At an RTF around 0.5, a 30-second clip takes about 15 seconds of GPU time, so one A100-hour produces on the order of $3600 / 15 \approx 240$ clips, putting the marginal compute cost near \$0.008 per clip — under a cent. The economics are dominated not by the diffusion compute but by *utilization*: if your A100 sits idle between requests, your effective cost per clip balloons. This is the real production lesson — for an offline batch workload (render a catalog overnight), diffusion's parallel, GPU-friendly sampling is cheap and you keep the GPU saturated; for a spiky interactive workload, the idle time, plus diffusion's inability to stream, is what hurts. The full serving-and-cost treatment is the capstone's job; this is the music-engine slice of it. (Costs are illustrative at a representative spot price; your actual numbers depend on the GPU, region, and batch size.)

### How to actually measure FAD, honestly

A results table is only as trustworthy as the measurement behind it, and FAD is famously easy to misreport. Fréchet Audio Distance fits a Gaussian to the embeddings of a *reference* set of real audio and to the embeddings of your *generated* set, then computes the Fréchet distance between those two Gaussians:

$$
\text{FAD} = \lVert \mu_r - \mu_g \rVert^2 + \operatorname{Tr}\!\left( \Sigma_r + \Sigma_g - 2 (\Sigma_r \Sigma_g)^{1/2} \right),
$$

where $(\mu_r, \Sigma_r)$ and $(\mu_g, \Sigma_g)$ are the mean and covariance of the reference and generated embeddings. Lower is better. The catch is that *every* term depends on which embedding model you use and how many samples you fit the Gaussian on. A FAD computed with VGGish embeddings is not comparable to one computed with a CLAP or PANN embedding, and a FAD on 200 samples is noisier and biased relative to one on 2,000. This is why the [audio quality metrics](/blog/machine-learning/audio-generation/audio-quality-metrics) post insists you *never* compare FAD numbers across papers that used different embeddings or sample sizes.

Here is a runnable computation with the `frechet-audio-distance` package, which makes the embedding choice explicit:

```python
from frechet_audio_distance import FrechetAudioDistance

# choose ONE embedding model and report it alongside every number
fad = FrechetAudioDistance(
    model_name="clap",          # or "vggish", "pann"; the choice changes the number
    sample_rate=48000,
    use_pca=False,
    verbose=True,
)

# background_dir: real reference audio; eval_dir: your generated wavs
score = fad.score(
    background_dir="data/real_music_wavs",
    eval_dir="outputs/stable_audio_wavs",
)
print(f"FAD (clap embedding, {len(__import__('os').listdir('outputs/stable_audio_wavs'))} samples): {score:.3f}")
```

The honest reporting rule the print line enforces: *always* state the embedding model and the sample count next to the number. A bare "FAD 2.1" is unfalsifiable; "FAD 2.1, CLAP embedding, 1,000 samples, same reference set" is a measurement someone can reproduce and contest. The same discipline applies to RTF (name the device and warm up first) and to any listening-test MOS (state the rater count and whether it was crowd-sourced).

#### Worked example: an A/B you can run in an afternoon

Suppose you want to decide whether Stable Audio Open at 50 steps is "good enough" versus 100 steps for a music-bed product. Generate 200 clips from a fixed prompt set at each step count, with fixed seeds, export wavs, and compute FAD against a held-out real-music reference set using one CLAP embedding. Suppose you measure FAD 2.6 at 50 steps and FAD 2.3 at 100 steps (illustrative numbers). Now weigh it: the 100-step run takes roughly twice the wall-clock, for a 0.3 FAD improvement that may or may not be audible in a blind listen. Run a quick MOS listening test on 20 pairs with 5 raters; if they cannot reliably tell 50 from 100 steps, ship 50 and bank the 2× speedup. The discipline is to let the *measurement* — not a default config — pick your operating point.

Here is a compact results table. **Every number is approximate and source-dependent — verify against the current papers/model cards before quoting.**

| Model | Engine | Max length | Stereo | Sample rate | Conditioning | Steps (approx) | Notes |
|---|---|---|---|---|---|---|---|
| AudioLDM 2 | Latent diffusion (U-Net) | ~10 s | Mono | 16 kHz | CLAP text | ~100–200 | Unified speech/music/SFX |
| Stable Audio Open 1.0 | Latent diffusion | ~47 s | Stereo | 44.1 kHz | T5 + timing | ~100 | Open weights |
| Stable Audio 1 (commercial) | Latent diffusion | ~95 s | Stereo | 44.1 kHz | text + timing | ~100 | Timing conditioning |
| Stable Audio 2.0 | Latent diffusion (DiT) | ~3 min | Stereo | 44.1 kHz | text + audio-to-audio | ~50 | Full songs |
| MusicGen (medium) | Autoregressive codec LM | ~30 s | Mono (base) | 32 kHz | text (+ melody) | n/a (AR) | Streamable, melody control |

The pattern is the whole post in one table: the diffusion line climbs in length, stereo, and sample rate while step counts stay modest, and the autoregressive line trades that for streaming and melody control.

## 11. Stress tests: where latent music diffusion breaks

A technique you cannot break is a technique you do not understand. Here is where this one strains.

**You ask for four minutes and it loses the plot.** Push past the trained maximum length and two things go wrong. First, the latent is longer than any the DiT saw in training, so attention has to extrapolate over a horizon it never learned — the music can wander or repeat awkwardly. Second, even within range, diffusion holds *texture and key* but not *deliberate song narrative* (the Section 6 stress test). The mitigation is to stay within the trained length and, if you need longer, to generate in overlapping segments with the timing knob and crossfade — but that reintroduces a stitching problem the single-pass model was meant to avoid.

**The high-compression latent drops the high frequencies.** The whole long-form trick rests on aggressive VAE compression. Push the compression ratio too far and the decoder starts smearing fine high-frequency detail — on music this is audible as a loss of "air," dull cymbals, and a slightly underwater top end. This is the rate-distortion wall: every bit of extra compression that buys you length costs you some fidelity. The right operating point is the highest compression at which the decoder's reconstruction still passes a careful listen on cymbals, sibilance, and reverb tails.

**Guidance too high turns it harsh.** Crank `guidance_scale` and, exactly as in images, you over-saturate — the audio gets harsh, brittle, and clipped-sounding, the sonic equivalent of an over-contrasted image. The fix is the same: find the guidance scale where the prompt is clearly followed but the audio still sounds natural, usually a moderate value, and use a negative prompt ("distorted, clipping") rather than ever-higher positive guidance.

**Too few steps and it is mushy; too many and you waste an A100.** The step count is a real budget. Below some threshold the denoiser has not converged and the track sounds smeared and noisy; far above the quality knee you are paying linearly more compute for inaudible gains. The honest move is to *measure* the quality knee for your model and sampler (generate at 25 / 50 / 100 / 200 steps, listen and/or compute FAD, find where the curve flattens) and operate just past it.

**Streaming is simply impossible.** Worth restating as a hard limit, not a tuning issue: if your application needs the first audio out before the whole clip is rendered, latent diffusion is the wrong engine, full stop. No knob fixes this; it is structural. Reach for the autoregressive model.

**The decoder, not the diffusion, becomes the bottleneck.** Here is a failure mode that surprises people the first time. You optimize the denoiser — fewer steps, a faster sampler, bigger batches — and the wall-clock barely moves. The reason is that for a heavily compressed latent, the *VAE decode* (turning the final latent back into 44.1 kHz stereo samples) can be a meaningful fraction of total time, because it has to *expand* the few-thousand-frame latent back into millions of samples through a stack of transposed convolutions. If you have driven the diffusion steps down low enough, the decoder is now your critical path, and no amount of further step reduction helps. The fix is to profile honestly — time the denoising loop and the decode separately — and optimize whichever actually dominates. This mirrors a lesson the speech world learned with vocoders: past a point, the model is not the bottleneck, the *waveform synthesis* is. (The vocoder-as-bottleneck story is in [GAN vocoders, HiFi-GAN, and fast synthesis](/blog/machine-learning/audio-generation/gan-vocoders-hifi-gan-and-fast-synthesis).)

**The stereo image collapses to mono-in-disguise.** A subtle quality failure: the model emits two channels, but they are nearly identical, so the track sounds "narrow" — technically stereo, perceptually mono. This happens when the VAE or the training data under-represented true stereo width, and it is easy to miss if you only listen on a phone speaker (which sums to mono anyway). Check it by measuring the correlation between the left and right channels, or the mid/side energy ratio; a healthy stereo music track has meaningful side energy, a collapsed one does not. The honest move is to *listen on headphones* and *measure the side channel*, not to trust that "two channels out" means "real stereo."

## 12. When to reach for latent music diffusion (and when not to)

A decisive recommendation, because the whole point of understanding the trade is to act on it.

![A decision tree splitting on whether you need length and stereo, which points to Stable Audio, versus streaming and melody control, which points to MusicGen](/imgs/blogs/latent-diffusion-for-music-stable-audio-8.png)

**Reach for latent diffusion (Stable Audio) when:**

- You need **length** — anything beyond ~30 seconds, and especially full multi-minute tracks. This is the single strongest reason.
- You need **stereo at 44.1 kHz** and high musical fidelity.
- You need **coherent global structure** — a key and tempo that hold across the whole track.
- You are rendering **offline** (a music bed, a stock track, a backing loop), so the lack of streaming does not matter.
- Your control surface is **text** (and, for 2.0, a reference audio clip).

**Do NOT reach for it — use the autoregressive codec model (MusicGen) — when:**

- You need **streaming / low latency / real-time** interaction. Diffusion cannot stream; this is disqualifying.
- You need **tight, time-aligned melody control** — "follow this exact hummed tune." MusicGen-melody's chromagram conditioning is the right tool.
- You only need **short clips** and want the simplicity of a single-stage token model.

**And do not over-reach within diffusion:**

- **Do not push past the trained length** and expect coherence — generate within range and stitch only if you must.
- **Do not crank guidance or steps** past the quality knee; measure it and stop.
- **Do not use a general short-form model (AudioLDM 2) for a stereo music track** — it is mono, 16 kHz, and short by design; that breadth is the wrong trade for music depth.
- **Do not reach for diffusion when AudioLDM 2's breadth is the actual need** — a quick foley sound effect or a short ambient bed is squarely AudioLDM 2's job, and Stable Audio's music depth is wasted on it.

The capstone, [building an audio generation stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack), threads these choices into an end-to-end pipeline with serving, latency, and cost — this section is the music-engine decision node inside that larger flow.

## 13. Key takeaways

- **Diffusion beats autoregression for long music because it is parallel and non-causal.** It denoises the whole latent at once with full bidirectional context, so it does not suffer the serial drift that makes autoregressive tracks wander in key and tempo over minutes.
- **Latent diffusion for audio is the image-LDM recipe on a heavily compressed audio latent.** You diffuse in the small latent space and only decode to a waveform at the end — the same bargain as Stable Diffusion, made *more* essential by audio's enormous raw sample count.
- **The high-compression VAE is what makes minutes tractable.** Sixteen million stereo samples become a few thousand latent frames; without ~60×-plus temporal compression, a diffusion transformer could not attend over a song at all.
- **Timing conditioning is the unlock for variable-length, structured generation.** Feeding the model `seconds_start` and `seconds_total` converts an inference problem (guess the structure) into a conditioning problem (you are told the structure), so one model emits coherent clips of any length with real intros and outros.
- **Stable Audio 2.0 reaches coherent three-minute stereo songs** via a higher-compression autoencoder, a DiT denoiser, and audio-to-audio conditioning — but it holds texture and key far better than it composes deliberate song narrative.
- **AudioLDM 2 is the generalist** — CLAP-conditioned latent diffusion unifying speech, music, and sound effects ("the language of audio"), but mono, 16 kHz, and short-form: breadth, not music depth.
- **The diffusion-vs-MusicGen trade is clean and non-dominating:** diffusion for length, stereo, and parallelism; autoregression for streaming and tight melody control. Choose by the job.
- **Measure honestly.** Report RTF on a named device after a warm-up, treat cross-model FAD with suspicion (embedding- and sample-size-sensitive), and find your model's quality-vs-steps knee by listening, not by guessing.

## 14. Further reading

- **Evans et al., "Stable Audio Open" / "Long-form music generation with latent diffusion" (Stability AI, 2024)** — the timing-conditioning recipe and the long-form latent diffusion architecture; the canonical source for this post.
- **Liu et al., "AudioLDM 2: Learning Holistic Audio Generation with a Self-Supervised Pretraining" (2023/2024)** — the "language of audio," CLAP-conditioned latent diffusion unifying speech/music/SFX.
- **Copet et al., "Simple and Controllable Music Generation" (MusicGen, Meta, 2023)** — the autoregressive counterpoint; codebook interleaving and melody conditioning.
- **Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (2022)** — the latent diffusion recipe this whole approach ports to audio; see also the series' [latent diffusion and Stable Diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion).
- **Peebles & Xie, "Scalable Diffusion Models with Transformers" (DiT, 2023)** — the diffusion-transformer denoiser; the series' [diffusion transformers (DiT)](/blog/machine-learning/image-generation/diffusion-transformers-dit) walks it through.
- **Wu et al., "CLAP: Learning Audio Concepts from Natural Language Supervision" (2023)** — the text-audio contrastive embedding that AudioLDM conditions on.
- **Within this series:** [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard) (the foundation and the audio-stack frame), [diffusion for audio](/blog/machine-learning/audio-generation/diffusion-for-audio) (the broader audio-diffusion recipe), [music generation with MusicLM and MusicGen](/blog/machine-learning/audio-generation/music-generation-musiclm-and-musicgen) (the autoregressive sibling), [neural audio codecs, the tokenizer of sound](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound) and [conditioning and control in audio generation](/blog/machine-learning/audio-generation/conditioning-and-control-in-audio-generation), and the capstone [building an audio generation stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack).
- **🤗 `diffusers` audio docs** — `StableAudioPipeline` and `AudioLDM2Pipeline` API references for the runnable code above.
