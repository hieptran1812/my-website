---
title: "Diffusion for Audio: From DiffWave to Stable Audio"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "How the denoising engine behind modern image and video generation crossed over to sound, why it denoises latents instead of samples, and when it beats an autoregressive language model for audio."
tags:
  [
    "audio-generation",
    "audio-synthesis",
    "diffusion-models",
    "latent-diffusion",
    "music-generation",
    "generative-ai",
    "deep-learning",
    "stable-audio",
  ]
category: "machine-learning"
subcategory: "Audio Generation"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/diffusion-for-audio-1.png"
---

The first time I watched a diffusion model paint a photorealistic image out of pure static, my reaction was the same as everyone else's: *how is this not magic?* The second reaction, a few months later, was more useful for my day job: *the audio team should be doing this.* We had been autoregressing codec tokens for months — generating a clip of music one token at a time, left to right, the way a language model writes a sentence. It worked, but a forty-second clip meant thousands of sequential decoding steps, and the model could only ever see what it had already written. It could not plan the ending of a phrase before it reached it, because it generated the beginning first and committed to it. A diffusion model has the opposite shape: it sees the *whole* clip at every step, refining all of it at once, and it never decodes left to right. For music, where a chord progression is a global structure and not a causal stream of words, that felt like exactly the right inductive bias.

The catch, which I learned the hard way, is that you cannot simply point an image diffusion model at a waveform and press go. A 44.1 kHz stereo song is over five million numbers per minute, all of them tightly correlated across milliseconds and seconds at once. Run the denoising loop directly on those samples and a single step is a full-length convolution over millions of points — and you need dozens to hundreds of steps. The first generation of audio diffusion models, [DiffWave](https://arxiv.org/abs/2009.09761) and [WaveGrad](https://arxiv.org/abs/2009.00713), accepted that cost and used diffusion narrowly, as a **vocoder**: denoise a noisy waveform conditioned on a mel-spectrogram, turning a compact time-frequency picture back into samples. The second generation did to audio what Stable Diffusion did to images — it moved the denoising into a small learned **latent** so the loop became cheap, and that is what unlocked [AudioLDM](https://arxiv.org/abs/2301.12503), AudioLDM 2, and [Stable Audio](https://arxiv.org/abs/2407.14358), which can render minutes of music from a text prompt.

![Stacked diagram of the latent audio diffusion pipeline showing a text prompt and latent noise feeding an iterative denoiser, a clean latent, a decoder, and the final waveform](/imgs/blogs/diffusion-for-audio-1.png)

This post is about that crossover. By the end you will be able to: state the diffusion training objective in one line and know exactly which part of it is audio-specific; explain why DiffWave denoises samples but AudioLDM denoises a latent, and what that buys you; run a text-to-audio diffusion pipeline in 🤗 `diffusers` with the knobs that matter (`num_inference_steps`, `audio_length_in_s`, `guidance_scale`) and export a wav; sketch a mel-conditioned waveform-diffusion denoiser step in PyTorch; reason about the step-count-versus-quality trade and why few-step distillation is the active frontier; and — the decision that actually matters in production — choose between a diffusion model and an [autoregressive audio language model](/blog/machine-learning/audio-generation/autoregressive-audio-models-wavenet-to-audiolm) for a given job. Diffusion is the parallel, non-causal, full-context engine; AR is the causal, streaming, exact-likelihood engine. Knowing which one a problem wants is half the battle.

Throughout, keep the series spine in view: the audio stack runs waveform → codec or mel latent → generative model → vocoder or decoder → waveform, under the constant tension of **fidelity × controllability × speed × length**. Diffusion is one choice for the "generative model" box, and every design decision in this post — denoise samples or a latent, how many steps, whether to distill — is a move on one of those four axes. The waveform-diffusion vocoders sit in the last box of the stack; the latent-diffusion generators sit in the middle box. Holding the stack in mind keeps the many models below from blurring into an undifferentiated list.

This is the diffusion node in the generative-engines track of the series. It sits next to the [autoregressive audio models](/blog/machine-learning/audio-generation/autoregressive-audio-models-wavenet-to-audiolm) post (the AR alternative it competes with) and depends on the [neural audio codec](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound) that defines the latent it denoises. I will keep the core diffusion math brief and link OUT to the image series, which derives it properly in [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles) and [latent diffusion and Stable Diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion). If you have not read [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard), that post poses the central problem — a 1D high-rate signal our ears scrutinize mercilessly — and this is one of the two big answers to it.

## 1. The diffusion objective, in one paragraph, then what is audio-specific

I am going to state the diffusion machinery once, fast, and then spend the rest of the post on the part that is genuinely different for audio. If any of this is new, the [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles) post in the image series derives every step with pictures; I am compressing it on purpose.

A **denoising diffusion** model is defined by two processes. The **forward process** takes clean data $x_0$ and gradually adds Gaussian noise over $T$ steps, so that $x_T$ is indistinguishable from pure noise. Because the noise is Gaussian and the schedule is fixed, you can jump to any step in closed form:

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\,\epsilon, \qquad \epsilon \sim \mathcal{N}(0, I),
$$

where $\bar\alpha_t = \prod_{s=1}^{t}\alpha_s$ runs from near 1 (barely noised) to near 0 (pure noise) as $t$ grows. The **reverse process** learns to undo one step at a time. In the standard DDPM parameterization the network $\epsilon_\theta(x_t, t, c)$ predicts the noise that was added, given the noisy input $x_t$, the step index $t$, and any conditioning $c$ (a mel-spectrogram, a text embedding). Training is almost embarrassingly simple — sample a clean example, a random step, and a random noise vector, then minimize:

$$
\mathcal{L}_\text{simple} = \mathbb{E}_{x_0, t, \epsilon}\left[\,\big\|\,\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\; t,\; c)\,\big\|^2\,\right].
$$

That is it. There is an equivalent **score-matching** view — the network learns $\nabla_{x}\log p_t(x)$, the gradient of the log-density of the noised data, and sampling is following that gradient back toward the data manifold — and a continuous-time **stochastic differential equation** view that unifies the two. All three are derived in the image post; I will use the $\epsilon$-prediction form because it is what the audio papers use.

So what changes for audio? **The loss does not.** The forward process, the closed-form noising, the $\epsilon$-prediction objective — all of it transfers unchanged. What changes is three things, and they are the whole story of this post:

1. **What $x_0$ is.** In image diffusion $x_0$ is a pixel grid or a 2D VAE latent. In audio diffusion $x_0$ is one of three things: a raw **waveform** (DiffWave, WaveGrad), a **mel-spectrogram latent** (AudioLDM), or a **neural codec latent** (Stable Audio). Each choice sets the cost per step and the maximum length you can generate.

2. **The conditioning $c$.** Audio is almost never generated unconditionally. A vocoder is conditioned on a mel-spectrogram; a text-to-audio model is conditioned on a [CLAP](https://arxiv.org/abs/2211.06687) or T5 text embedding; Stable Audio is additionally conditioned on **timing** — the start time and total duration — which is the trick that lets one model produce variable-length clips up to minutes. We cover conditioning in depth in the [conditioning and control](/blog/machine-learning/audio-generation/conditioning-and-control-in-audio-generation) post; here I will show the mechanism and the classifier-free-guidance knob.

3. **The denoising network's shape.** Images use a 2D U-Net or a DiT over a square latent grid. Audio uses a 1D dilated-convolution stack (DiffWave), a 1D/2D U-Net over a spectrogram (AudioLDM), or a transformer/DiT over a sequence of latent frames (Stable Audio). The architecture follows the data's geometry: audio is a long 1D sequence, so the network is built for long 1D context.

Hold onto that list. Everything else — the speed, the length limits, the AR-versus-diffusion trade — falls out of those three choices.

There is a fourth, quieter audio-specific subtlety hiding in the **noise schedule** $\bar\alpha_t$, and it is worth a sentence because it bit the early waveform-diffusion papers. The schedule controls the **signal-to-noise ratio** at each step, $\text{SNR}(t) = \bar\alpha_t / (1 - \bar\alpha_t)$. For images, the standard cosine or linear schedules were tuned on 2D pixel data. Audio waveforms have a very different amplitude distribution — they are near-zero-mean, often quiet, with energy concentrated in bursts — so a schedule that adds "enough" noise to destroy an image's structure may leave audible low-frequency structure in a waveform at the same step, or wipe it out too aggressively at another. WaveGrad's choice to condition on the *continuous* noise level (rather than a discrete step index) is partly a response to this: it lets you re-tune the SNR schedule at inference without retraining. The practical takeaway is that you cannot blindly copy an image diffusion schedule onto a waveform and expect the same quality — the schedule is one of the things you tune per modality. For *latent* diffusion the issue is milder, because the VAE/codec normalizes the latent distribution, which is one more reason latents are friendlier to diffuse than raw samples.

## 2. Why audio resisted diffusion at first: the length problem

Before latent diffusion, the obstacle was brutal and quantitative. Let me make it concrete, because the number is the reason the whole field pivoted to latents.

Diffusion is **iterative**. Even with fast samplers you evaluate the network many times — call it $S$ steps, anywhere from 8 to 250. Each evaluation runs the full denoiser over the *entire* signal, because diffusion refines the whole thing at once (that is the point — it is non-causal). So the total compute is roughly

$$
\text{cost} \approx S \times (\text{cost of one network pass over the full signal}).
$$

For an image, "the full signal" is a 64×64 latent — about four thousand spatial positions. For one second of 24 kHz audio, "the full signal" is **24,000 samples**; for a minute of 44.1 kHz stereo it is over five million. A 1D convolution or attention pass over five million positions, repeated 200 times, is not a model you ship. This is the same wall the [neural audio codec](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound) post describes for autoregressive models — a raw waveform is simply too long — and diffusion hits it just as hard, only multiplied by the step count.

There are two escape routes, and the two generations of audio diffusion took one each.

**Route one: keep the waveform, narrow the job.** If you only ask diffusion to be a **vocoder** — turn a mel-spectrogram into a waveform for a *short* clip — the lengths stay bounded (a sentence of speech is a couple of seconds) and the mel condition does most of the work, so you can get away with very few steps. This is DiffWave and WaveGrad. They denoise samples, but they never try to generate a four-minute song; they are the last stage of a pipeline whose earlier stages produced the mel.

**Route two: move diffusion off the waveform entirely.** Train a [neural audio codec or VAE](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound) that compresses the waveform by a large factor — say 50× to 1000× fewer positions — into a learned latent, then run diffusion *in that latent*. Now "the full signal" the denoiser sees is fifty times shorter, each step is fifty times cheaper, and a minute of audio becomes tractable. Decode the final latent back to a waveform once, at the end. This is exactly what [latent image diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion) did for images, and it is what AudioLDM and Stable Audio do for audio.

![Before-and-after comparison contrasting waveform diffusion operating on millions of samples against latent diffusion operating on a compressed code](/imgs/blogs/diffusion-for-audio-4.png)

The two routes are not competitors so much as different layers of the stack. You can even combine them: generate a mel or codec latent with route-two latent diffusion, then convert that latent to a waveform with a route-one diffusion vocoder (or, more often in practice, a fast GAN vocoder like [HiFi-GAN](/blog/machine-learning/audio-generation/gan-vocoders-hifi-gan-and-fast-synthesis)). Keep that decomposition in mind; it is the spine of the rest of the post.

## 3. Waveform diffusion: DiffWave and WaveGrad as vocoders

Let us start where the field started — route one — because it is the cleanest illustration of the diffusion objective applied to sound, and because it teaches the conditioning mechanism you will reuse everywhere.

### What a vocoder is, and why diffusion fits

A **vocoder** is the module that turns a mel-spectrogram back into a waveform. Recall from the [representing sound](/blog/machine-learning/audio-generation/representing-sound-waveforms-spectrograms-and-perception) post that a mel-spectrogram is a compact, perceptually-weighted time-frequency picture that throws away **phase** — the fine timing information that says exactly where each waveform peak sits. Reconstructing a waveform from a mel is therefore an *inverse problem with a missing variable*: many waveforms share the same mel, and the vocoder has to pick a plausible, phase-coherent one. Classical methods like Griffin-Lim iterate to estimate the phase and sound metallic. Neural vocoders learn the mapping and sound clean.

Diffusion is a natural fit because picking "a plausible waveform consistent with this mel" is exactly a *conditional generation* problem. The mel is the condition $c$; the waveform is the $x_0$ you generate. DiffWave's network $\epsilon_\theta(x_t, t, \text{mel})$ predicts the noise in a noisy waveform, conditioned on the upsampled mel and the step embedding, using a stack of **dilated convolutions** borrowed from [WaveNet](/blog/machine-learning/audio-generation/autoregressive-audio-models-wavenet-to-audiolm) — but, crucially, **non-causal and parallel**. WaveNet generated samples one at a time; DiffWave's WaveNet-style network looks at the whole noisy waveform and predicts the noise for all samples *simultaneously*. The only sequential cost is the number of diffusion steps.

Before the model can be conditioned on a mel, you need the mel, and it is worth seeing how cheap that side is — it is plain DSP, not learned. Here is the `torchaudio` path that produces the exact condition a DiffWave-style vocoder consumes, plus a Griffin-Lim reconstruction so you can *hear* how much a vocoder has to add back. Griffin-Lim is the classical phase-estimation baseline: it iterates to guess a phase consistent with the mel's magnitudes, and it sounds metallic precisely because the phase it invents is not the true one. That metallic gap is the job the diffusion vocoder is learning to close.

```python
import torch
import torchaudio
import torchaudio.transforms as T

wav, sr = torchaudio.load("the_quick_brown_fox.wav")   # (channels, samples)
wav = wav.mean(0, keepdim=True)                         # mono
if sr != 22050:
    wav = T.Resample(sr, 22050)(wav); sr = 22050

# the mel-spectrogram the vocoder is conditioned on
n_fft, hop, n_mels = 1024, 256, 80
mel_tf = T.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop,
                          n_mels=n_mels, power=2.0)
mel = mel_tf(wav)                                        # (1, 80, frames)
log_mel = torch.log(mel.clamp(min=1e-5))                # log scale, as models use
print("mel:", log_mel.shape, "frames per second:", sr / hop)

# classical phase-free reconstruction (the baseline a neural vocoder beats)
inv_mel = T.InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=n_mels, sample_rate=sr)
griffin = T.GriffinLim(n_fft=n_fft, hop_length=hop, n_iter=64)
approx_lin = inv_mel(mel)                                # mel -> linear spectrogram
recon = griffin(approx_lin)                              # -> waveform (metallic!)
torchaudio.save("griffinlim_baseline.wav", recon.unsqueeze(0), sr)
```

Run that and listen: the Griffin-Lim output is intelligible but obviously synthetic — a buzzy, watery quality that comes entirely from the invented phase. A trained vocoder, diffusion or GAN, replaces that guesswork with a learned, phase-coherent mapping. The mel extraction itself (the `MelSpectrogram` call) is the same regardless of which vocoder you pair it with; only the inversion differs. This is also why the *frame rate* matters: at `hop=256` and 22.05 kHz the mel runs at about 86 frames per second, so a DiffWave denoiser must upsample the mel by 256× to align it with the per-sample waveform — that is the `mel_up` `ConvTranspose1d` plus the `interpolate` in the denoiser sketch below.

![Graph of mel-conditioned waveform diffusion where a mel-spectrogram, a noisy waveform, and a step embedding feed a dilated-convolution denoiser that predicts noise and recovers the clean waveform](/imgs/blogs/diffusion-for-audio-2.png)

### The math that makes it cheap: the mel does the heavy lifting

Here is the quantitative insight that lets a waveform diffusion vocoder run in as few as **six steps** while a text-to-image model needs dozens. The conditioning is *extremely* informative. The mel-spectrogram already specifies the rough magnitude content at every time-frequency cell; all the model has to add is consistent fine-scale phase and the details the mel dropped. The conditional distribution $p(x_0 \mid \text{mel})$ is far tighter than the unconditional $p(x_0)$. A tighter target needs fewer denoising steps to resolve, because each step has less uncertainty to remove. DiffWave reports near-ground-truth quality with around 6 reverse steps for speech; WaveGrad similarly trades steps for quality on a continuous noise schedule. That is the practical payoff of strong conditioning, and it generalizes: the more your condition pins down the output, the fewer steps you can afford to take.

### A mel-conditioned denoiser step in PyTorch

Let me sketch the inner loop concretely so the math has a body. This is a stripped-down DiffWave-style reverse step — not the full model, but the part that matters: predict the noise, then take one DDPM reverse step. The full implementation lives in the [official DiffWave repo](https://github.com/lmnt-com/diffwave); this is the shape of it.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffWaveDenoiser(nn.Module):
    """Predicts the noise epsilon in a noisy waveform x_t,
    conditioned on an upsampled mel-spectrogram and the diffusion step t.
    Real DiffWave uses ~30 dilated residual layers; this is the skeleton."""

    def __init__(self, n_mels=80, channels=64, n_layers=30):
        super().__init__()
        self.in_proj = nn.Conv1d(1, channels, 1)
        # condition: upsample mel to the waveform's time resolution
        self.mel_up = nn.ConvTranspose1d(n_mels, channels, 16, stride=8, padding=4)
        # step (diffusion timestep) embedding -> per-layer bias
        self.step_mlp = nn.Sequential(nn.Linear(128, channels), nn.SiLU(),
                                      nn.Linear(channels, channels))
        self.layers = nn.ModuleList([
            nn.Conv1d(channels, 2 * channels, 3, padding=2**(i % 10),
                      dilation=2**(i % 10))
            for i in range(n_layers)
        ])
        self.out = nn.Conv1d(channels, 1, 1)

    def forward(self, x_t, mel, t_emb):
        # x_t: (B, 1, L) noisy waveform; mel: (B, n_mels, L/hop); t_emb: (B, 128)
        h = self.in_proj(x_t)
        c = self.mel_up(mel)                      # (B, channels, ~L)
        c = F.interpolate(c, size=h.shape[-1])    # align lengths exactly
        s = self.step_mlp(t_emb).unsqueeze(-1)    # (B, channels, 1)
        for layer in self.layers:
            z = layer(h + c + s)                  # inject mel + step every layer
            a, b = z.chunk(2, dim=1)              # gated activation (WaveNet style)
            h = h + torch.tanh(a) * torch.sigmoid(b)
        return self.out(h)                        # predicted noise, shape (B, 1, L)


@torch.no_grad()
def ddpm_reverse_step(model, x_t, mel, t, t_emb, alphas, alpha_bars, betas):
    """One reverse step: x_t -> x_{t-1}. Parallel over all samples."""
    eps = model(x_t, mel, t_emb)                  # predict noise for the WHOLE waveform
    alpha_t, abar_t, beta_t = alphas[t], alpha_bars[t], betas[t]
    # posterior mean of x_{t-1} given x_t and the predicted noise
    mean = (x_t - beta_t / (1 - abar_t).sqrt() * eps) / alpha_t.sqrt()
    if t > 0:
        noise = torch.randn_like(x_t)
        return mean + beta_t.sqrt() * noise       # add stochasticity except at t=0
    return mean
```

Read the `forward` carefully — every dilated layer adds the mel condition `c` and the step bias `s` back in, so the model can never forget what waveform it is supposed to be producing. And read `ddpm_reverse_step`: the model is called once and predicts the noise for **all** samples in parallel. There is no left-to-right loop over time. The only loop you would wrap around this is over $t$, the diffusion steps — six of them for a fast speech vocoder. That is the entire difference in compute profile from an autoregressive vocoder, and it is why a diffusion vocoder can be fast despite being iterative: the iteration count is the step count, not the sample count.

#### Worked example: the step-count budget for a speech vocoder

Suppose you are vocoding the sentence "the quick brown fox" — about 2 seconds at 22.05 kHz, so $L \approx 44{,}100$ samples, mel hop 256, giving ~172 mel frames. A DiffWave-base model is roughly 2.6M parameters. With $S = 6$ reverse steps you call the network 6 times, each a parallel pass over 44,100 samples. On an RTX 4090 that is a handful of milliseconds per step, so the vocoder finishes in well under the 2 seconds of audio it produced — a **real-time factor** (generation time ÷ audio duration) comfortably below 1, often around 0.02–0.05 with a compiled model, i.e. 20–50× faster than real time. Push to $S = 50$ for the very best quality and you are still real-time, but you have spent 8× the compute for a quality gain most listeners will not hear in a clean speech setting. That is the whole trade in one example: **steps are a quality dial, and strong mel conditioning lets you turn it way down.** Contrast this with autoregressing 44,100 samples one at a time — 44,100 sequential steps — and the parallelism win is obvious.

The training loop is the other half, and it is even simpler than the sampler — it is the $\mathcal{L}_\text{simple}$ objective from Section 1 with the mel as the condition. Seeing it in code makes clear how little audio-specific machinery there is on the training side; almost everything is the generic diffusion recipe.

```python
import torch
import torch.nn.functional as F

def make_schedule(T_steps=50, device="cuda"):
    betas = torch.linspace(1e-4, 0.05, T_steps, device=device)  # tune per modality
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars

def step_embed(t, dim=128):
    # sinusoidal embedding of the diffusion step, like a transformer position
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, device=t.device) * (9.21 / half))
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)

def train_step(model, wav, mel, betas, alphas, alpha_bars, opt):
    # wav: (B, 1, L) clean waveform; mel: (B, n_mels, L/hop) condition
    B = wav.size(0)
    t = torch.randint(0, alpha_bars.numel(), (B,), device=wav.device)   # random step
    abar = alpha_bars[t][:, None, None]                                 # (B,1,1)
    eps = torch.randn_like(wav)                                         # target noise
    x_t = abar.sqrt() * wav + (1 - abar).sqrt() * eps                   # forward noising
    pred = model(x_t, mel, step_embed(t))                              # predict the noise
    loss = F.mse_loss(pred, eps)                                       # L_simple
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()
```

That `train_step` is the entire learning objective. Notice there is nothing in it specific to audio except that `wav` is a waveform and `mel` is the condition — swap a pixel tensor for `wav` and a text embedding for `mel` and you have image diffusion training. The audio-specific decisions live *outside* this loop: the noise schedule in `make_schedule` (tuned for a waveform's amplitude statistics, per Section 1), the mel extraction that produced `mel`, and the dilated-convolution architecture of `model`. The diffusion math is modality-agnostic; the representation and the conditioning are where audio shows up. This is exactly why the field could port image diffusion to sound so quickly — the hard, general part was already solved, and the work was in choosing what to denoise and how to condition it.

A WaveGrad note for completeness: WaveGrad uses a **continuous** noise level conditioning (it conditions on $\sqrt{\bar\alpha}$ directly rather than a discrete step index), which lets you choose the number of inference steps *after* training without retraining the schedule. DiffWave uses a small fixed set of discrete steps. Both reach near-vocoder-grade quality; the continuous schedule is more flexible at inference, the discrete one is simpler. Either way, you are spending single-digit-to-tens of steps, not hundreds, because the mel pins the target down.

## 4. Latent diffusion for audio: AudioLDM and AudioLDM 2

Now route two — the generation, not just the vocoding. This is where diffusion competes head-on with the autoregressive audio language model, because here diffusion is the thing that turns *text* into *new audio*, not just a mel into a waveform.

### The recipe is latent image diffusion, ported to sound

If you know how [Stable Diffusion works for images](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion), you already know the skeleton of AudioLDM. The recipe is:

1. Train a **VAE** that compresses the data into a small continuous latent and decodes it back. For images the data is pixels; for AudioLDM the data is a **mel-spectrogram**, and the VAE compresses the mel into a latent grid (it treats the mel like a 2D image).
2. Train a **diffusion U-Net** to denoise that latent, conditioned on a text embedding via cross-attention.
3. At inference: encode the text, denoise a random latent under that condition, decode the latent back to a mel with the VAE, then run a **vocoder** to turn the mel into a waveform.

So AudioLDM has an extra stage images do not: the VAE decoder gives you a *mel*, and you still need a vocoder (HiFi-GAN, in AudioLDM's case) to get from mel to waveform. The pipeline is text → CLAP → denoise latent → VAE-decode to mel → vocode to wav.

![Graph of the AudioLDM dataflow showing a text prompt encoded by CLAP, conditioning a diffusion U-Net that denoises a mel-VAE latent, decoded to a mel and vocoded to a waveform](/imgs/blogs/diffusion-for-audio-5.png)

The clever bit in the original AudioLDM is **how it conditions**, and it is worth dwelling on because it is a genuinely audio-flavored idea. [CLAP](https://arxiv.org/abs/2211.06687) is a contrastive model — the audio analogue of CLIP — that maps an audio clip and a text caption into a *shared* embedding space, so that a clip of a barking dog and the text "a dog barking" land near each other. AudioLDM exploits this in a way image models cannot: it trains the diffusion model conditioned on the **CLAP audio embedding** of the training clip (which you always have), and at inference swaps in the **CLAP text embedding** of the prompt (whose embedding is close, by construction). This sidesteps the need for a huge corpus of perfectly captioned audio during diffusion training — you condition on the audio's own embedding and lean on CLAP's shared space to make text prompting work at inference. It is a neat trick that made high-quality text-to-audio trainable on the messy, under-captioned audio data that actually exists.

**AudioLDM 2** generalizes the conditioning further with a "language of audio" idea — it predicts a sequence of self-supervised audio features (an [AudioMAE](https://arxiv.org/abs/2207.06405)-style representation) from text using a GPT-2-style model, then conditions the latent diffusion on those features. The headline is that AudioLDM 2 unifies speech, music, and sound effects under one model and one conditioning interface, with a continuous latent that decodes through a shared VAE and vocoder. For our purposes the key facts are: it is latent diffusion on a mel-VAE latent, conditioned on a learned audio-feature sequence derived from text, and it is the model you can call most easily today in 🤗 `diffusers`.

### Running AudioLDM 2 in `diffusers`

Here is the thing the brief most wants — a real, runnable text-to-audio diffusion call with the knobs that matter. This is `AudioLDM2Pipeline`, end to end, exporting a wav.

```python
import torch
import soundfile as sf
from diffusers import AudioLDM2Pipeline

# fp16 on GPU; the repo ships several sizes (audioldm2, audioldm2-large, -music)
pipe = AudioLDM2Pipeline.from_pretrained(
    "cvssp/audioldm2", torch_dtype=torch.float16
).to("cuda")

prompt = "lo-fi hip hop with a warm vinyl crackle and a mellow piano loop"
negative_prompt = "low quality, distorted, clipping"

# fix the seed so runs are comparable when you sweep the knobs
generator = torch.Generator("cuda").manual_seed(0)

audio = pipe(
    prompt,
    negative_prompt=negative_prompt,   # the "what to avoid" side of CFG
    num_inference_steps=200,           # the step-count / quality dial
    audio_length_in_s=10.0,            # how many seconds to generate
    guidance_scale=3.5,                # classifier-free guidance strength
    num_waveforms_per_prompt=3,        # generate a few, keep the best by CLAP-score
    generator=generator,
).audios

# audios is (num_waveforms, num_samples) at 16 kHz for audioldm2
sf.write("lofi.wav", audio[0], samplerate=16000)
print("samples:", audio.shape, "→ wrote lofi.wav")
```

Four knobs are doing the real work here, and they are the same four you will see in every diffusion-audio pipeline:

- **`num_inference_steps`** is the step count $S$. More steps, better fidelity, slower. AudioLDM 2 wants ~100–200 for its best quality (it is generating, not vocoding, so the target is far less constrained than DiffWave's). This is the single biggest lever on the speed/quality trade.
- **`audio_length_in_s`** sets the latent length. AudioLDM-family models are trained around 10 seconds; ask for much longer and quality degrades or the model loops, because it never learned long-range structure. This is the length limitation that Stable Audio's timing conditioning was invented to fix.
- **`guidance_scale`** is **classifier-free guidance** (CFG), which I unpack in the next subsection. Higher pushes the output to match the prompt more aggressively at some cost to diversity and naturalness.
- **`num_waveforms_per_prompt`** lets you sample several candidates cheaply and rank them — with [CLAP-score](/blog/machine-learning/audio-generation/audio-quality-metrics) (text-audio similarity), say — keeping the best. Generation is a distribution; sampling-and-ranking is a standard, effective trick.

### Classifier-free guidance, the audio version

CFG is borrowed wholesale from image diffusion — the [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance) post derives it — so I will only state the mechanism and the audio-specific gotcha. You train the model to predict noise *both* with the condition and without it (by randomly dropping the condition during training). At inference you push the prediction away from the unconditional and toward the conditional:

$$
\tilde\epsilon_\theta(x_t, c) = \epsilon_\theta(x_t, \varnothing) + w\,\big(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing)\big),
$$

where $w$ is the guidance scale (your `guidance_scale`). $w = 1$ recovers ordinary conditional sampling; $w > 1$ exaggerates the influence of the prompt. The audio-specific gotcha is the *cost* and the *sweet spot*. The cost: CFG doubles the per-step network evaluations, because you compute both the conditional and unconditional noise — so `guidance_scale > 1` makes every step ~2× more expensive. The sweet spot: audio is more sensitive to over-guidance than images in an audible way. Crank $w$ too high and music gets a brittle, over-saturated, "loudness-war" quality and speech gets unnaturally emphatic; the artifacts are immediately audible because, as the [why audio is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard) post argues, our ears are unforgiving. For most audio diffusion models a guidance scale around 3–7 is the practical band, lower than the 7.5–12 you often see quoted for images. The `negative_prompt` is the same machinery with a non-empty unconditional branch — instead of guiding away from "nothing," you guide away from "low quality, distorted."

#### Worked example: the seconds-to-generate budget for a 10-second clip

Let us put numbers on the latent-diffusion cost so you can size a deployment. AudioLDM 2 (the base `cvssp/audioldm2`, ~350M parameters in the U-Net) generating 10 seconds with `num_inference_steps=200` and CFG on calls the U-Net $200 \times 2 = 400$ times over a small mel latent. On an A100 80GB in fp16 that is roughly **15–25 seconds of wall-clock** for one 10-second clip — a real-time factor around 1.5–2.5, i.e. it takes a bit longer to generate than the audio lasts. Drop to `num_inference_steps=50` and you cut the U-Net calls 4× to ~4–7 seconds (RTF ~0.5), with a modest FAD increase. Turn CFG off (`guidance_scale=1.0`) and you halve it again — but the prompt adherence collapses, so nobody does that for text-to-audio. These are order-of-magnitude figures from the model size and step count, not a benchmarked table; the honest way to report them is to fix the seed, prompt, length, and device, warm up the GPU once (the first call includes compilation/allocation overhead you should not count), and time the median of several runs. Always name the device. "RTF 2 on an A100" means nothing without the A100.

## 5. Stable Audio: the codec latent and the timing trick

AudioLDM proved text-to-audio diffusion works at ~10 seconds. The frontier question was **length**: how do you generate a *minute* of coherent music, with an intro, a body, and an ending that lands on time? Stable Audio's two answers are the most important audio-specific contributions in this whole post, so let me give them room.

### Answer one: denoise a codec-style latent, not a mel

AudioLDM diffuses a mel-VAE latent and then needs a separate vocoder to get to a waveform. Stable Audio instead trains an **autoencoder that goes straight to and from the waveform** — an [EnCodec-style](/blog/machine-learning/audio-generation/encodec-dac-and-the-modern-codec) convolutional autoencoder (in Stable Audio 2 a heavily downsampling, high-compression variational autoencoder) whose **continuous** latent the diffusion model denoises directly. No mel, no separate vocoder stage: the autoencoder's decoder *is* the path back to samples.

This is the cleanest place to make the latent-image-diffusion analogy exact, because it is one-to-one. In [Stable Diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion), a [VAE](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) compresses a 512×512×3 image into a 64×64×4 latent — roughly a 48× reduction in elements — the diffusion U-Net or DiT denoises *that*, and the VAE decoder turns the final latent back into pixels once at the end. Stable Audio is the same three boxes with audio in the slots: a convolutional autoencoder compresses a 44.1 kHz stereo waveform into a low-frame-rate continuous latent (a large reduction in elements, since a second of stereo audio is ~88,000 samples but only a few dozen latent frames), a diffusion transformer denoises that latent under text-and-timing conditioning, and the autoencoder decoder reconstructs the waveform once at the end. The boxes line up perfectly; only the data's geometry differs — a 2D grid for images, a 1D sequence of frames for audio. If you have internalized latent image diffusion, you have internalized the skeleton of Stable Audio, and the only genuinely new pieces are the timing conditioning (below) and the fact that an audio decoder is reconstructing a signal our ears scrutinize harder than our eyes scrutinize pixels.

Why is a codec latent *friendlier* to diffuse than a raw waveform, beyond just being shorter? Three reasons, all of which the [codec post](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound) develops. First, **normalization**: the VAE is trained with a KL penalty that keeps the latent distribution close to a standard Gaussian, which is exactly the prior diffusion assumes — so the noise schedule and the data distribution are already well-matched, sidestepping the SNR-schedule headache from Section 1. Second, **perceptual weighting**: the autoencoder was trained with adversarial and spectral losses that make it spend its capacity on perceptually important detail, so diffusing in its latent automatically focuses the model's effort where ears care. Third, **decorrelation**: raw samples are massively redundant (adjacent samples are nearly identical), and that redundancy wastes diffusion capacity modeling structure the codec could have removed for free; the latent has already stripped it. The codec is not just a compressor — it reshapes the problem into one diffusion is good at. The compression is aggressive — the latent runs at a low frame rate relative to the 44.1 kHz stereo waveform — which is precisely what makes long durations affordable. Diffusing a minute of audio in this latent is diffusing a sequence of a few thousand latent frames, not five million samples. This is the cleanest expression of the [codec-as-the-tokenizer-of-sound](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound) idea meeting diffusion: the codec defines a compact space, and diffusion generates in it.

The denoiser itself is a **diffusion transformer** (DiT) operating on the sequence of latent frames, the audio analogue of the [DiT used in image and video latent diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion). Stable Audio's reported models use this transformer-over-latent-frames design rather than a convolutional U-Net, which scales better to the long latent sequences that long audio demands.

It is worth pausing to see the arc, because the three audio-specific choices from Section 1 moved in lockstep across four years. WaveGrad and DiffWave (2020) denoised raw waveforms and were vocoders, bounded to seconds. AudioLDM (2023) moved the denoising into a mel-VAE latent and conditioned on text via CLAP, turning diffusion from a vocoder into a *generator* — but still capped near ten seconds. AudioLDM 2 (2023) generalized the conditioning to a learned audio-feature sequence and unified speech, music, and sound effects. Stable Audio (2024) swapped the mel latent for a high-compression codec-style latent and added timing conditioning, finally breaking the length ceiling to minute-scale, full-fidelity stereo. Each step traded a little more representational distance from the raw waveform for a lot more length and controllability — the same bargain the [neural codec](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound) post frames as the central deal of modern audio modeling.

![Timeline of diffusion audio from WaveGrad and DiffWave through AudioLDM and AudioLDM 2 to Stable Audio, showing the move from waveform vocoders to text-conditioned latents to timing-aware codec-latent music](/imgs/blogs/diffusion-for-audio-7.png)

### Answer two: timing conditioning for variable-length generation

This is the genuinely clever part and it is unique to audio. A diffusion model generates a *fixed-size* tensor — the latent has a set number of frames. So how does one model produce clips of *different* lengths, from a 3-second sound effect to a 90-second track, and make them feel complete rather than truncated?

Stable Audio conditions the diffusion model on **timing signals**: two numbers, a *seconds-start* and a *seconds-total* (the offset of this chunk into the full piece, and the intended total duration), embedded and fed in alongside the text. During training, clips are taken from longer recordings, so the model *sees* examples where seconds-total is 30, 47, 90, and it learns what a piece of a given total length should sound like — including that it should resolve and end, not just fade or loop, when the requested duration is reached. At inference you generate the model's full fixed window but *tell it* you want, say, a 45-second piece starting at 0; it fills the first 45 seconds with a complete musical idea and the remainder with silence (which you trim). That is how a fixed-shape diffusion model produces variable-length, properly-resolved audio. It is the answer to the length problem that AudioLDM's fixed ~10-second window could not give, and it is why Stable Audio is the model people reach for when they need *long* audio. The dedicated [latent diffusion for music](/blog/machine-learning/audio-generation/latent-diffusion-for-music-stable-audio) post goes deeper on the music-specific design; here the takeaway is the timing-conditioning mechanism itself.

### Running Stable Audio in `diffusers`

The `diffusers` `StableAudioPipeline` exposes the timing conditioning through `audio_end_in_s`. Here is the call.

```python
import torch
import soundfile as sf
from diffusers import StableAudioPipeline

pipe = StableAudioPipeline.from_pretrained(
    "stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16
).to("cuda")

prompt = "a driving techno groove, 128 BPM, deep sub bass and a bright hi-hat"
negative_prompt = "low quality, muddy, off-beat"
generator = torch.Generator("cuda").manual_seed(7)

result = pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=200,      # DiT denoising steps
    audio_end_in_s=30.0,          # TIMING conditioning: total duration in seconds
    guidance_scale=7.0,
    num_waveforms_per_prompt=1,
    generator=generator,
).audios

# stable-audio-open is 44.1 kHz stereo: result is (1, 2, num_samples)
out = result[0].T.float().cpu().numpy()   # -> (num_samples, 2) for soundfile
sf.write("techno.wav", out, samplerate=pipe.vae.sampling_rate)
print("shape:", result.shape, "→ wrote 44.1 kHz stereo techno.wav")
```

Notice what is different from the AudioLDM call. The output is **44.1 kHz stereo** (`(1, 2, N)`), not 16 kHz mono — Stable Audio Open targets full-fidelity music. And `audio_end_in_s` is the timing knob: change it to 10, 30, or 47 and you get genuinely different-length, self-contained clips from the same model, because the model was *trained* to understand total duration. That single argument is the variable-length capability the rest of the field was missing, exposed as one number.

## 6. The autoregressive-versus-diffusion trade, decided honestly

This is the section the brief is really about, and it is the question I get asked most when someone is picking an audio architecture. You have two engines for generating audio from a model: an [autoregressive language model over codec tokens](/blog/machine-learning/audio-generation/autoregressive-audio-models-wavenet-to-audiolm) (VALL-E, MusicGen, AudioLM) and a diffusion model over a latent (AudioLDM, Stable Audio). They are genuinely different tools, and the right choice flips depending on the job. Let me lay out the axes, then give the rule.

![Matrix comparing autoregressive language models and diffusion across sampling, causality, length, and best use](/imgs/blogs/diffusion-for-audio-3.png)

**Sampling: sequential vs parallel.** An AR model generates tokens one at a time, left to right — each token conditions on all previous ones. To produce $N$ codec frames you run $N$ forward passes, inherently sequential. A diffusion model produces *all* frames at once and refines them over $S$ steps; $S$ is typically far smaller than $N$ for long audio. So for long clips diffusion can be dramatically faster in *step count*, though each step is heavier (it touches the whole sequence). For short clips, AR can win because $N$ is small and there is no fixed step overhead.

**Causality: streamable vs not.** This is the axis people underrate. An AR model is **causal** — it produces the first samples before it has decided the last ones — so it can **stream**: you can start playing audio while the model is still generating, which is the difference between a usable voice assistant and an unusable one. A diffusion model is **non-causal** — every step touches the whole clip, so you cannot emit the beginning until the whole thing is done. For real-time conversational speech (think [full-duplex dialogue](/blog/machine-learning/audio-generation/conditioning-and-control-in-audio-generation) and Moshi-style systems), causality is decisive and AR wins almost by default. For offline music generation, where you wait for the render anyway, non-causality is a *feature* — the model plans the whole piece globally.

**Likelihood: exact vs implicit.** AR models give you an **exact likelihood** — they factorize $p(x) = \prod_i p(x_i \mid x_{\lt i})$ and you can compute the probability of any sequence, which is useful for scoring, ranking, and some training tricks. Diffusion models give only a variational bound on the likelihood; you cannot cheaply score an arbitrary clip's probability. If your application needs exact likelihoods (some speech tasks, some detection setups), AR has an edge.

**Context: causal window vs full.** Because diffusion sees the whole clip at every step, it has **full bidirectional context** — the ending can influence the beginning during generation. This is why diffusion is so good at *global* structure like a coherent chord progression or a sound effect with a clear envelope. An AR model only ever conditions on the past, so long-range global coherence is harder (it has to *remember* what it committed to, rather than *see* the whole).

**Length and timing.** AR length grows linearly with sequential cost — a 4-minute song is a *very* long token sequence and a *lot* of sequential steps, and the model can drift off the beat or out of key as errors accumulate. Diffusion with timing conditioning (Stable Audio) generates a fixed window and can resolve to a target duration, which is a cleaner story for long, structured pieces — though it too has a maximum trained window.

So here is the rule I actually use:

- **Speech, dialogue, anything real-time or streaming → autoregressive.** Causality and streaming dominate. VALL-E-style and Moshi-style systems are AR for exactly this reason.
- **Music, sound effects, anything offline where global structure matters → diffusion.** Parallel sampling, full context, and timing conditioning win. Stable Audio and AudioLDM are diffusion for exactly this reason.
- **The lines blur.** MusicGen is AR and excellent at music; flow-matching TTS (the next post, [flow matching and consistency for audio](/blog/machine-learning/audio-generation/flow-matching-and-consistency-for-audio)) is diffusion-family and excellent at speech. These are tendencies, not laws. The axes above are how you reason about a *specific* case rather than cargo-culting a default.

#### Worked example: choosing an engine for two real jobs

*Job A — a real-time voice agent that must start speaking within 300 ms of the user finishing.* The hard constraint is **time-to-first-audio**. A diffusion model must finish all $S$ denoising steps over the whole utterance before it can emit a single sample — even at $S = 50$ and a fast model, that is hundreds of milliseconds of dead air *before* playback starts, and it grows with utterance length. An AR codec-token model emits the first frame after one forward pass and streams the rest while the user listens. **Choose AR.** The exact-likelihood and streaming properties are not nice-to-haves here; they are the product.

*Job B — generate 90-second instrumental tracks for a video-editing tool, offline, where users wait a few seconds for a render.* The hard constraints are **length, global coherence, and stereo fidelity**, not latency. Autoregressing 90 seconds of 44.1 kHz stereo codec tokens is thousands of sequential steps and risks drifting off the beat by the end. Stable Audio diffuses the whole 90 seconds in a couple hundred parallel steps with timing conditioning that makes it resolve on time, in full-fidelity stereo. **Choose diffusion.** Here non-causality is the feature, not the bug — the model plans the ending while it writes the intro.

## 7. The step-count cost and few-step distillation

Every diffusion model carries the same tax: it is **iterative**. You pay $S$ network evaluations (or $2S$ with CFG) per sample. That tax is the one thing diffusion has that AR does not have to the same degree, and the whole frontier of fast diffusion is about driving $S$ down without losing quality. For audio this matters even more than for images, because audio applications often want *real-time*, and a 200-step sampler is nowhere near it.

![Before-and-after comparison contrasting a full many-step diffusion sampler with a distilled few-step model trading a small fidelity loss for a large speedup](/imgs/blogs/diffusion-for-audio-8.png)

There are two families of fixes, both ported from the image world:

**Better samplers.** The reverse process does not have to take the same number of steps the forward process used. **DDIM** and higher-order ODE solvers (DPM-Solver and friends) integrate the same learned vector field with far fewer evaluations — dropping a 1000-step DDPM schedule to 20–50 steps with little quality loss. This is free: same trained model, different sampling loop, just pass a smaller `num_inference_steps`. It is always the first thing to try.

**Distillation.** To get to a *handful* of steps you distill: train a student model to take a big jump that the teacher takes in many small ones. **Consistency models** and **flow-matching/rectified-flow** distillation can collapse the sampler to 1–8 steps. The cost is a small fidelity drop (a modest FAD increase) and a separate distillation training run, but the payoff is near-real-time generation. This is the bridge to the next post, [flow matching and consistency for audio](/blog/machine-learning/audio-generation/flow-matching-and-consistency-for-audio), which covers the audio versions in depth — Stable Audio's later releases and several fast-TTS systems lean on exactly this to hit low latency. The image-series [consistency models post](/blog/machine-learning/image-generation/consistency-models-and-few-step-generation) derives the mechanism; the audio specifics (and why a codec latent makes distillation easier — fewer, more structured dimensions to match) are the next post's job.

The honest framing of the trade: **steps are a quality dial with a speed price, and distillation moves the whole curve, not just your position on it.** A 200-step model and an 8-step distilled student are not the same model run differently; the student was trained to be fast. For a vocoder with strong mel conditioning you may never need distillation — six steps is already real-time. For an unconstrained text-to-music generator at 200 steps, distillation is the difference between an offline tool and an interactive one.

#### Worked example: the RTF you can buy with distillation

Take a text-to-audio model generating a 10-second clip. At $S = 200$ steps with CFG, that is 400 network passes; on an A100 say it lands at RTF ~2 (20 seconds to make 10 seconds of audio). Switch to a good ODE sampler at $S = 30$: ~60 passes, RTF ~0.3, a ~6.7× speedup, and for many prompts the FAD barely moves. Now distill to an $S = 4$ consistency student (CFG often folded in, so ~4 passes): RTF drops toward ~0.05, roughly a 40× speedup over the original, at the cost of a measurable but often acceptable FAD increase — on the order of a few tenths to a point, depending on the model and how aggressively you distilled. The decision rule: if you are offline, run the full sampler and bank the quality; if you are interactive, distill and measure whether the FAD/MOS hit is one your users can hear. Always report the FAD with the *same* embedding network and the RTF on a *named* device with a warmed-up GPU — an unwarmed first call can lie by a factor of two.

#### Worked example: how much does the latent actually shrink the problem?

Let me make the "tens of times smaller" claim concrete with arithmetic, because the compression factor is *the* number that decides whether minute-long diffusion is affordable. Take 60 seconds of 44.1 kHz **stereo** audio. As raw samples that is $60 \times 44{,}100 \times 2 = 5{,}292{,}000$ numbers — the thing waveform diffusion would have to denoise, repeatedly. Now run it through a codec-style autoencoder that downsamples time by, say, 1024× and emits a latent with 64 channels per frame. The frame rate becomes $44{,}100 / 1024 \approx 43$ frames per second, so 60 seconds is about $60 \times 43 = 2{,}580$ frames, and the latent tensor is $2{,}580 \times 64 \approx 165{,}000$ numbers. That is a **32× reduction** in elements the diffusion model touches per step — and since stereo is folded into the channels, you have not paid extra for two channels the way raw stereo doubles the samples. At 200 steps, waveform diffusion would do 200 passes over ~5.3M points; latent diffusion does 200 passes over ~165k. That single ratio is why one is a research curiosity at minute scale and the other is a `diffusers` call. And note the second-order win from Section 5: those 165k latent numbers are already normalized and decorrelated, so the diffusion model spends them on musical structure rather than on the redundancy raw samples are riddled with.

The flip side — the stress test — is that the decoder is now the ceiling. If that 1024×-downsampling autoencoder cannot faithfully reconstruct a crisp hi-hat transient from 43 frames per second, then *no* diffusion model on top can produce a crisp hi-hat, because the only outputs available are the ones the decoder can render. This is the bitrate-versus-quality trade from the [RVQ](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq) and [codec](/blog/machine-learning/audio-generation/encodec-dac-and-the-modern-codec) posts showing up one level higher: a more aggressive latent compression makes diffusion cheaper and longer but lowers the fidelity floor. Stable Audio's whole engineering art is finding the compression that is aggressive enough for minute-scale diffusion yet rich enough that 44.1 kHz stereo still sounds like 44.1 kHz stereo. When you pick or train the autoencoder, you are choosing the ceiling; the diffusion model only decides how close to it you get.

## 8. Putting numbers on it: a comparison table you can act on

Let me consolidate the three diffusion audio systems into one table, because the differences are exactly the three audio-specific choices from Section 1 — what you denoise, how you condition, how long you can go. The FAD figures are approximate and embedding-dependent; treat them as order-of-magnitude, and never compare FADs computed with different embedders.

![Matrix comparing DiffWave, AudioLDM 2, and Stable Audio across what they denoise, conditioning, maximum length, and steps with approximate FAD](/imgs/blogs/diffusion-for-audio-6.png)

| Model | Domain (denoises) | Conditioning | Sample rate | Max length | Typical steps | Approx FAD | Role |
|---|---|---|---|---|---|---|---|
| DiffWave / WaveGrad | Raw waveform | Mel-spectrogram | 16–24 kHz | Seconds | 6–50 | — (vocoder; judge by MOS) | Vocoder |
| AudioLDM | Mel-VAE latent | CLAP embedding | 16 kHz | ~10 s | ~100–200 | ~1.6–2.0 (approx) | Text-to-audio |
| AudioLDM 2 | Mel-VAE latent | Learned audio feats from text | 16 kHz | ~10 s | ~100–200 | ~1.4 (approx) | Unified T2A |
| Stable Audio (Open) | Codec/VAE latent | Text + timing | 44.1 kHz stereo | Up to minutes | ~100–250 | low (music-tuned, approx) | Long-form music |

If you want to actually *produce* the FAD numbers in that table rather than quote them, the measurement is a short script, and showing it makes the caveats concrete. FAD fits a Gaussian to a frozen embedding of a set of real clips and another to your generated clips, then measures the Fréchet distance between the two Gaussians — so the **embedder** is the whole ballgame, and a FAD computed with VGGish is not comparable to one computed with PANNs or CLAP. Here is the `frechet-audio-distance` path, with the embedder named explicitly so the number is reproducible.

```python
from frechet_audio_distance import FrechetAudioDistance

# name the embedder — the FAD number is meaningless without it
fad = FrechetAudioDistance(
    model_name="vggish",   # or "pann", "clap" — DIFFERENT scales, never mix
    sample_rate=16000,
    use_pca=False,
    use_activation=False,
    verbose=True,
)

# background_dir: a few thousand real clips; eval_dir: your generated clips
score = fad.score(
    background_dir="data/real_audiocaps_16k",   # reference distribution
    eval_dir="out/audioldm2_200steps_16k",      # the model's generations
)
print(f"FAD (vggish) = {score:.3f}  (lower is better)")
# To compare two models fairly: SAME embedder, SAME sample count,
# SAME sample rate, and report the count alongside the number.
```

The honest reporting rules fall straight out of this script. Use a few thousand clips per side (FAD is biased on small samples — too few generations and the Gaussian fit is noisy and the score drifts). State the embedder. Match sample rates (resampling changes the embedding). And report FAD *next to* a CLAP-score for prompt alignment and a small MOS study for human preference, because — as the [audio quality metrics](/blog/machine-learning/audio-generation/audio-quality-metrics) post shows in detail — a model can win on FAD while losing to humans. A bare FAD number with no embedder named is not a result; it is a vibe.

A few honest caveats baked into that table. DiffWave's quality is measured by **MOS** against ground-truth speech, not FAD, because it is a vocoder reconstructing a known target — FAD is for distributional generation. The AudioLDM FAD numbers are the kind reported on [AudioCaps](https://arxiv.org/abs/1904.03340)-style benchmarks and depend heavily on the FAD embedding (VGGish vs PANNs vs CLAP give different absolute numbers); the [audio quality metrics](/blog/machine-learning/audio-generation/audio-quality-metrics) post explains why a bare FAD is so easy to misread. Stable Audio's strength is not a single FAD number but the *combination* of 44.1 kHz stereo and minute-scale length at competitive quality, which is a different point in the design space than the 16 kHz, 10-second AudioLDM family.

And here is the cross-engine table — the AR-versus-diffusion summary — as numbers rather than prose, so you can drop it into a design doc:

| Property | Autoregressive LM | Diffusion |
|---|---|---|
| Sampling | Sequential, $N$ token passes | Parallel, $S$ denoising steps (often $S \ll N$) |
| Streamable | Yes — causal, emits as it goes | No — whole clip finishes together |
| Time-to-first-audio | Low (one pass) | High (all $S$ steps first) |
| Likelihood | Exact factorized | Variational bound only |
| Context during gen | Past only (causal) | Full clip (bidirectional) |
| Long-range global structure | Harder (drift accumulates) | Easier (sees whole clip) |
| Variable length | Natural (stop when done) | Needs timing conditioning |
| Best fit | Speech, dialogue, real-time | Music, SFX, offline, global structure |

## 9. Case studies: real numbers from the literature

Let me ground the discussion in named results, accurately, with the uncertainty flagged where it exists.

**DiffWave as a fast vocoder (Kong et al., 2020).** DiffWave reported speech-vocoding quality competitive with the best autoregressive and GAN vocoders of its day while running **non-autoregressively** — and notably, near-top quality at as few as **6 reverse steps** for the conditional speech setting, with a base model around 2.6M parameters. The lesson that stuck with me: a strong conditioning signal (the mel) collapses the number of diffusion steps you need, because the conditional distribution is so much tighter than the unconditional one. This is the single most transferable idea from the waveform-diffusion era.

**AudioLDM and the CLAP-conditioning trick (Liu et al., 2023).** AudioLDM showed that latent diffusion on a mel-VAE latent, conditioned via CLAP, could generate text-to-audio at quality that was state-of-the-art at release on AudioCaps, while being far cheaper to train than pixel-space (here, mel-space) diffusion because the latent is small. The conditioning trick — train on the audio's own CLAP embedding, prompt with text's CLAP embedding at inference — is the part worth stealing: it lets you train high-quality text-to-audio without a perfectly captioned corpus. AudioLDM 2 then unified speech, music, and SFX under one conditioning interface, which is why it is the most generally useful of the family to call today.

**Stable Audio's length-and-fidelity leap (Evans et al., 2024).** Stable Audio's contribution was not a single benchmark win but a *capability*: high-fidelity **44.1 kHz stereo** audio up to the *minute* scale, from a single diffusion model, via the codec-latent-plus-timing-conditioning design. Stable Audio Open's release made a strong long-form music diffusion model runnable on consumer hardware, and it is the reason "generate a 90-second track" stopped being a research demo and became a `diffusers` one-liner. When someone asks me for *long* generated audio with a real ending, this is the architecture I point at.

**The AR counterpoint — MusicGen (Copet et al., 2023).** It is worth saying plainly that the best open *music* model for a long stretch was [MusicGen](/blog/machine-learning/audio-generation/music-generation-musiclm-and-musicgen), which is **autoregressive**, not diffusion — a single-stage codec language model with clever codebook interleaving and optional melody conditioning. This is the honest complication in the AR-vs-diffusion story: AR can be excellent at music too, and diffusion can be excellent at speech (flow-matching TTS). The architecture is a strong *prior* about what a task wants, not a guarantee. Measure both on *your* data, with the *same* FAD embedding and a small MOS study, before you commit.

## 10. Stress tests: where audio diffusion breaks

A technique is only as trustworthy as its failure modes. Here is where I have watched audio diffusion break, and what the break teaches.

**You ask AudioLDM for 30 seconds and it loops or drifts.** The AudioLDM family is trained around 10 seconds. Request much longer and it has no learned notion of long-range structure, so it repeats a motif or wanders. The fix is not "more steps" — it is the *right model*: Stable Audio's timing conditioning exists precisely because fixed-short-window models cannot do length. Lesson: **length is an architectural property, not an inference knob.**

**You crank `guidance_scale` to 12 and the music sounds harsh.** Over-guidance over-saturates audio audibly — a brittle, compressed, fatiguing sound. Because our ears are unforgiving (the thesis of [why audio is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard)), the over-guidance artifact is far more obvious than the equivalent in an image. Back off to 3–7. Lesson: **audio's guidance sweet spot is lower than images', and you can hear when you have left it.**

**The codec/VAE decoder is the quality ceiling, not the diffusion model.** A subtle one. In latent diffusion, the decoder defines the *best possible* output — diffusion can only produce latents the decoder knows how to turn into clean audio. If the VAE/codec was trained at a compression that drops high frequencies or smears transients, no amount of diffusion quality fixes it; you will hear a dull or smeared result that the spectrogram confirms. When latent-diffusion audio sounds slightly "off" in a way the prompt adherence does not explain, suspect the decoder. Lesson: **in a latent pipeline, the autoencoder is the fidelity floor.**

**Distillation eats the high end.** Push few-step distillation too hard and the first thing to go is often the high-frequency detail and the transient sharpness — cymbals lose their sizzle, sibilants soften. The few-step student matched the teacher's broad strokes and dropped the fine texture. Lesson: **measure the FAD *and* listen to the top octave after distilling; the metric can lag the audible loss.**

**The vocoder buzzes even though the diffusion latent is fine.** When AudioLDM's mel is correct but the output has a faint metallic buzz, the culprit is usually the **vocoder** (HiFi-GAN) hitting a mel it was not trained on, not the diffusion. The mel-to-waveform stage is a separate model with its own failure surface. Lesson: **debug the pipeline stage by stage — latent, mel, waveform — because a diffusion-audio system is a chain, and the weakest link sets the sound.**

**The prompt is too far from the training captions and CLAP shrugs.** AudioLDM leans on CLAP's shared text-audio space to make text prompting work despite training on audio embeddings. That bridge is only as good as CLAP's coverage. Prompt it with a concept CLAP never learned well — an obscure instrument, a very specific production style, a compound scene CLAP saw only in pieces — and the conditioning signal goes weak, so the diffusion model falls back toward its unconditional prior and produces something generic that ignores half your prompt. The tell is that `guidance_scale` stops helping: cranking it just amplifies a weak, wrong direction. Lesson: **text-to-audio adherence is bounded by the text encoder's coverage, not just the diffusion model's capacity** — when prompts fail, suspect the conditioning embedding before you suspect the denoiser.

**The render is fine but it is slower than the audio it makes.** This is not a bug so much as the defining property, and it catches people moving from AR mental models. A 200-step text-to-music diffusion at RTF ~2 spends *twice as long generating as the clip lasts*, and unlike an AR model it cannot stream a single sample early — you wait for the whole render. If that is unacceptable, the answer is not "more GPUs," it is a different point on the step curve (a faster sampler) or a different engine (distillation, or AR if you need streaming). Lesson: **diffusion's latency is structural; budget for it at design time, do not discover it in production.**

## 11. When to reach for diffusion (and when not to)

A decisive recommendation, because "it depends" is not an answer anyone can ship.

**Reach for diffusion when:**

- You are generating **music or sound effects offline**, where global structure matters and latency does not. Diffusion's full-context, parallel-step nature is exactly right, and timing conditioning (Stable Audio) handles length.
- You need **long, high-fidelity, stereo** output with a real ending. Stable Audio's codec-latent-plus-timing design is the current best open answer.
- You want a **vocoder** and quality is paramount over the last few percent — a diffusion vocoder (DiffWave) can edge out GAN vocoders in some settings, especially with enough steps. But read the next bullet first.

**Do not reach for diffusion when:**

- You need **streaming or real-time speech**. The non-causality is fatal to time-to-first-audio. Use an [autoregressive codec-token model](/blog/machine-learning/audio-generation/autoregressive-audio-models-wavenet-to-audiolm) and stream the frames. This is the single most important "don't."
- A **GAN vocoder hits your quality bar 100× faster**. For most TTS, [HiFi-GAN or Vocos](/blog/machine-learning/audio-generation/gan-vocoders-hifi-gan-and-fast-synthesis) vocode a mel in a single forward pass at an RTF far below a multi-step diffusion vocoder, at quality the listener cannot distinguish. Do not spend 50 diffusion steps to win an A/B test nobody passes. Reach for a diffusion vocoder only when you have measured a quality gap that matters.
- You need **exact likelihoods** for scoring or detection. Diffusion gives only a bound; an AR model gives the real thing.
- Your audio is **very short and latency-sensitive** (a UI sound, a short notification). The fixed step overhead of diffusion is poor value when $N$ is tiny; AR or even a non-generative method may be simpler.

The meta-rule, which ties back to the series spine of **fidelity × controllability × speed × length**: diffusion buys you length, global coherence, and (with strong conditioning) high fidelity, at the price of speed (the step tax) and streamability. If your application's hard constraint is on the *speed/streaming* axis, AR wins; if it is on the *length/global-structure* axis, diffusion wins. Name your hard constraint first, then the architecture chooses itself.

## 12. Key takeaways

- **The diffusion objective is unchanged for audio.** The $\epsilon$-prediction loss, the forward noising, the score view — all transfer from images. What changes is *what you denoise* (waveform, mel latent, or codec latent), *how you condition* (mel, CLAP/T5 text, timing), and the *1D network shape*.
- **Raw-waveform diffusion is too long to generate with; it is a vocoder.** DiffWave/WaveGrad denoise samples conditioned on a mel, in single-digit-to-tens of steps, because the mel pins the target down. Strong conditioning collapses the step count.
- **Latent diffusion is what makes text-to-audio generation tractable.** AudioLDM denoises a mel-VAE latent under CLAP conditioning; Stable Audio denoises a codec-style latent. Moving off the waveform shrinks each step by tens of times — the direct audio analogue of latent image diffusion.
- **Timing conditioning is the audio-specific trick for length.** Stable Audio conditions on seconds-start and seconds-total, so one fixed-shape diffusion model produces variable-length, properly-resolved clips up to minutes. Length is architectural, not an inference knob.
- **AR vs diffusion is a real decision with a clean rule.** Diffusion is parallel, non-causal, full-context — great for offline music and SFX. AR is sequential, causal, streamable, exact-likelihood — great for real-time speech and dialogue. Name your hard constraint (speed/streaming vs length/global-structure) and the engine chooses itself.
- **Classifier-free guidance is the same as images, with a lower sweet spot.** Audio over-guides audibly; 3–7 is the practical band, not 7.5–12. And CFG doubles the per-step cost.
- **The step count is a quality dial with a speed price.** Better ODE samplers cut 200 steps to ~30 for free; distillation (consistency/flow) reaches 1–8 steps at a small FAD cost — the bridge to the next post.
- **A latent pipeline is a chain, and the decoder is the fidelity floor.** Diffusion can only be as good as the VAE/codec decoder allows; debug stage by stage, and watch the high frequencies after distillation.
- **Report results honestly.** FAD only means something with a stated embedder and sample size; RTF only means something on a named, warmed-up device. Never compare FADs across embedders.

## 13. Further reading

- **DiffWave: A Versatile Diffusion Model for Audio Synthesis** — Kong, Ping, Huang, Zhao, Catanzaro (2020). The waveform diffusion vocoder, and the six-step result.
- **WaveGrad: Estimating Gradients for Waveform Generation** — Chen, Zhang, Zen, Weiss, Norouzi, Chan (2020). Continuous-noise-level waveform diffusion.
- **AudioLDM: Text-to-Audio Generation with Latent Diffusion Models** — Liu et al. (2023). Latent diffusion on a mel-VAE latent, the CLAP-conditioning trick.
- **AudioLDM 2: Learning Holistic Audio Generation with a Self-Supervised Pretraining** — Liu et al. (2023). Unified speech/music/SFX via a learned audio-feature conditioning.
- **Stable Audio Open / Long-form music generation with latent diffusion** — Evans et al., Stability AI (2024). Codec-latent diffusion with timing conditioning for 44.1 kHz stereo, minute-scale audio.
- **Denoising Diffusion Probabilistic Models** — Ho, Jain, Abbeel (2020), and **Classifier-Free Diffusion Guidance** — Ho, Salimans (2022). The core machinery, derived for images in this site's [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles) and [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance) posts.
- 🤗 `diffusers` audio pipelines docs — `AudioLDM2Pipeline`, `StableAudioPipeline` — the runnable APIs used above.
- Within this series: the [autoregressive audio models](/blog/machine-learning/audio-generation/autoregressive-audio-models-wavenet-to-audiolm) post (the engine diffusion competes with), the [neural audio codec](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound) post (the latent it denoises), the [audio quality metrics](/blog/machine-learning/audio-generation/audio-quality-metrics) post (how to read FAD/MOS/RTF honestly), and forward to [flow matching and consistency for audio](/blog/machine-learning/audio-generation/flow-matching-and-consistency-for-audio), [latent diffusion for music](/blog/machine-learning/audio-generation/latent-diffusion-for-music-stable-audio), [conditioning and control](/blog/machine-learning/audio-generation/conditioning-and-control-in-audio-generation), and the capstone, [building an audio generation stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack).
