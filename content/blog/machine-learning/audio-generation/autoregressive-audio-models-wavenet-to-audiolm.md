---
title: "Autoregressive Audio Models: From WaveNet to AudioLM"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "How the next-token idea conquered audio, from WaveNet predicting raw waveform samples one at a time to AudioLM's hierarchical token language model, and why moving from samples to codec tokens made it fast enough to use."
tags:
  [
    "audio-generation",
    "audio-synthesis",
    "autoregressive-models",
    "wavenet",
    "audiolm",
    "neural-audio-codec",
    "text-to-speech",
    "generative-ai",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Audio Generation"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/autoregressive-audio-models-wavenet-to-audiolm-1.png"
---

The first audio model that genuinely shocked me was WaveNet, and the thing that shocked me was not how good it sounded, though it did sound good. It was the rendering time. I had set it up to synthesize a single sentence, about four seconds of speech, and I went to get coffee expecting it to be done when I got back. It was not done when I got back. It was not done when I finished the coffee. The model was producing audio one sample at a time, and at 16,000 samples per second that meant 64,000 forward passes through a deep convolutional network for four seconds of sound, each pass waiting on the one before it because the network needed the sample it had just produced as input to produce the next one. The audio, when it finally arrived, was the best raw-waveform synthesis anyone had heard. It also took on the order of a minute of compute per second of audio on the hardware of the day. That single tension, that the next-token idea gives you extraordinary quality and a likelihood you can actually optimize but charges you one sequential step per output, is the whole story of autoregressive audio, and it is the story this post tells.

The arc runs from WaveNet modeling raw waveform samples directly, through the realization that you should not model raw samples at all but the discrete tokens of a neural codec, to AudioLM treating audio generation as language modeling over those tokens in a content-first hierarchy. By the end you will understand the chain-rule factorization that defines every autoregressive model, why WaveNet's dilated causal convolutions buy an exponentially large receptive field for linear cost, exactly why sample-level generation is two to three orders of magnitude slower than token-level generation, how AudioLM's semantic-then-acoustic hierarchy gives you both long-range coherence and high fidelity, and how the same recipe, an autoregressive transformer over codec tokens, became the backbone of modern text-to-speech in VALL-E and music in MusicGen. You will see runnable code for a dilated causal convolution block and for a token-level sampling loop over EnCodec tokens with temperature and top-k control.

![A vertical stack of the autoregressive audio paradigm from raw waveform samples through codec tokens to a token language model and back to a waveform](/imgs/blogs/autoregressive-audio-models-wavenet-to-audiolm-1.png)

The stack in the figure above is the shape of this whole post, and it is worth reading top to bottom before we start. At the bottom sits the autoregressive idea in its purest, slowest form: predict the next raw sample of the waveform, which is what WaveNet does. One layer up, you stop modeling raw samples and model the discrete tokens of a neural codec instead, which collapses tens of thousands of steps per second down to a few dozen. One layer above that, the model that does the predicting is just a transformer doing next-token prediction, an audio *language model*, and at the top a hierarchy of such models, the AudioLM design, plans content first and renders detail second. Every step up this stack trades a small amount of theoretical purity for an enormous amount of speed, and the series spine, the **audio stack** of waveform to latent to generative core to vocoder under the tension of **fidelity, controllability, speed, and length**, is exactly the axis we are moving along.

This post assumes you have met the neural codec already. If you have not, read [neural audio codecs, the tokenizer of sound](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound) first, because the entire jump from sample-level to token-level autoregression depends on having a codec that turns a waveform into a short sequence of discrete tokens you can predict one at a time. The autoregressive idea here is the audio twin of the one in the image series; if you want the parallel for pixels and patches, [autoregressive image models](/blog/machine-learning/image-generation/autoregressive-image-models) is the companion piece, and much of the sampling machinery, temperature and top-k and the exposure-bias problem, is shared. And the foundation post, [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard), set up the central pressure that makes all of this difficult: audio is short in wall-clock time but enormous in token count, so a per-step model pays dearly.

## The chain rule, and what "autoregressive" actually commits you to

Every autoregressive model rests on one identity, and it is exact, not an approximation. Take any sequence $x = (x_1, x_2, \dots, x_N)$, whether those are waveform samples, codec tokens, words, or pixels. The joint probability of the whole sequence factorizes by the chain rule of probability into a product of conditionals:

$$p(x) = p(x_1, x_2, \dots, x_N) = \prod_{t=1}^{N} p(x_t \mid x_1, \dots, x_{t-1}) = \prod_{t=1}^{N} p(x_t \mid x_{<t}).$$

This is not a modeling choice you can get wrong. It is true for *any* distribution over sequences. The modeling choice is to *parameterize* each conditional $p(x_t \mid x_{<t})$ with a neural network and to share parameters across positions $t$, so a single network with weights $\theta$ defines $p_\theta(x_t \mid x_{<t})$ for every $t$. Training is then pure maximum likelihood: you maximize the log-likelihood of your data, which because of the factorization is just a sum of per-position log-probabilities,

$$\log p_\theta(x) = \sum_{t=1}^{N} \log p_\theta(x_t \mid x_{<t}),$$

and if each conditional is a softmax over a discrete vocabulary, this is exactly the per-token cross-entropy loss you already know from language modeling. That equivalence is the reason the audio field could borrow the entire transformer language-modeling stack wholesale once it had tokens to predict.

Two consequences of this factorization define everything that follows, and they pull in opposite directions. The first is wonderful: an autoregressive model gives you an *exact* likelihood. Unlike a GAN, which has no tractable density, or a diffusion model, which optimizes a variational bound on the likelihood rather than the likelihood itself, an autoregressive model computes $\log p_\theta(x)$ exactly as that sum. You can compare two models by held-out likelihood, detect out-of-distribution audio by low likelihood, and trust that the training objective is the real thing and not a surrogate. This is a genuine and underrated advantage.

The second consequence is the curse. Because $p(x_t \mid x_{<t})$ conditions on *all* previous outputs, and because at generation time those previous outputs are themselves things the model produced, you cannot generate $x_t$ until you have generated $x_{t-1}$. Sampling is inherently sequential. To produce a sequence of length $N$ you run the network $N$ times, and each run waits on the previous one. Training parallelizes beautifully, because the true $x_{<t}$ is known for every position so you can score all positions in one batched forward pass with a causal mask, but *generation does not parallelize at all across time*. This is the single fact that makes the choice of *what* you autoregress over, raw samples or codec tokens, the most consequential decision in the entire design, because it sets $N$, and $N$ is your latency.

This train-fast, generate-slow asymmetry has a name worth knowing because it is the root of one of autoregression's deeper problems. During training you feed the model the *ground-truth* past at every position, a practice called **teacher forcing**: position $t$ is scored against the real $x_t$ given the real $x_{<t}$, and because all the real values are known up front, every position can be scored in parallel. But at generation there is no ground-truth past; the model conditions on its *own* samples, which inevitably contain small errors the model never encountered during training because the training past was always perfect. The distribution of contexts the model sees at generation drifts away from the distribution it was trained on. This mismatch between teacher-forced training and free-running generation is **exposure bias**, and we return to it in force later because it is the mechanism behind the drift that makes long autoregressive generations wander off-content or off-key. For now, hold the two facts together: the exact-likelihood advantage and the exposure-bias weakness are *two sides of the same teacher-forcing coin*, the thing that makes training tractable is the same thing that makes generation fragile.

It is also worth being concrete about what the likelihood number *means*, because audio people quote it less than language people do and it is genuinely useful. The per-token negative log-likelihood, averaged over a held-out set, is a *compression* measure: by Shannon's source-coding theorem it is, up to a constant, the number of bits per token an optimal coder using the model would need to encode the data. A model with lower held-out negative log-likelihood is literally a better compressor of audio tokens, and because it is an exact quantity, not a bound, you can trust comparisons between models in a way you cannot fully trust FAD or MOS, which have their own pitfalls covered in [audio quality metrics](/blog/machine-learning/audio-generation/audio-quality-metrics). The catch, and it is a real one, is that better likelihood does not always mean better *perceptual* quality: a model can assign high probability to audio that sounds wrong to a human, which is precisely why the field also reports FAD and MOS rather than likelihood alone. Likelihood is the honest training signal; perceptual metrics are the honest evaluation signal; you want both.

![A before and after comparison of sample-level autoregression at 24000 steps per second against token-level autoregression at about 600 steps per second](/imgs/blogs/autoregressive-audio-models-wavenet-to-audiolm-2.png)

The contrast in the figure above is the whole economic argument of the post in one image, and we will earn every number in it. On the left, the WaveNet world: model raw samples, pay one sequential step per sample, and at 24 kHz that is 24,000 steps for a single second of audio. On the right, the modern world: model codec tokens at roughly 75 frames per second, and even with several codebooks per frame you are paying a few hundred to a few thousand steps per second, a one-to-two-order-of-magnitude reduction in the number of sequential dependencies. Same autoregressive idea, same chain rule, radically different $N$. The rest of the speed story is just consequences of that left-to-right move.

## WaveNet: autoregression directly on the waveform

WaveNet, from van den Oord and colleagues at DeepMind in 2016, made the boldest possible choice. It modeled $p(x_t \mid x_{<t})$ where $x_t$ is a *raw audio sample*, the actual amplitude of the waveform at time step $t$. No spectrogram, no codec, no intermediate representation; the network looks at the literal sequence of sample values and predicts the next one. This is autoregression at the most granular level audio offers, and it is why WaveNet sounded so good and ran so slowly.

The first problem you hit is that a raw audio sample is a real number, typically stored as a 16-bit integer, which means 65,536 possible values. Modeling a softmax over 65,536 classes per step is wasteful and hard to train. WaveNet's fix is **µ-law companding quantization**, a classic telephony trick. The human ear's loudness perception is roughly logarithmic, so we have far finer discrimination at low amplitudes than at high ones. µ-law warps the amplitude axis logarithmically before quantizing, allocating more quantization levels to the quiet parts where the ear is sensitive and fewer to the loud parts where it is not. The transform, for a sample $x \in [-1, 1]$ and $\mu = 255$, is

$$f(x) = \operatorname{sign}(x) \, \frac{\ln(1 + \mu |x|)}{\ln(1 + \mu)},$$

and then the warped value is uniformly quantized into 256 bins. That single nonlinearity lets WaveNet model a 256-way softmax per sample, a far more tractable target than 65,536 classes, while keeping the perceptual quality of roughly 16-bit audio because the bins are placed where the ear cares. It is a beautiful example of using a fact about human perception, here that loudness is logarithmic, to shrink a modeling problem, and the same psychoacoustic instinct runs through everything in the [representing sound](/blog/machine-learning/audio-generation/representing-sound-waveforms-spectrograms-and-perception) post.

#### Worked example: how much resolution µ-law buys at low amplitude

Put numbers on why this works. Without companding, a uniform 8-bit quantizer would split the amplitude range $[-1, 1]$ into 256 equal bins, each of width about $2/256 \approx 0.0078$, everywhere. A quiet sound at amplitude $0.01$, just barely above one bin width, has essentially *one or two* bins of resolution to represent it, so quiet passages quantize to a coarse staircase and you hear quantization noise. Now apply µ-law with $\mu = 255$. Near zero the derivative of the companding curve $f(x)$ is large, $f'(0) = \mu / \ln(1 + \mu) \approx 255 / 5.55 \approx 46$, meaning the warped axis is stretched by about a factor of 46 near silence. So the small input interval $[0, 0.01]$ maps to a warped interval of roughly $0.46$, which spans about $0.46 \times 128 \approx 59$ of the 128 positive bins, versus the one or two bins a uniform quantizer gave it. µ-law has reallocated dozens of quantization levels into the quiet region where the ear is most sensitive, paid for by coarser bins up at high amplitude where the ear cannot tell the difference anyway. That is how 8 bits of µ-law audio sound roughly as clean as 16 bits of linear audio: the bits went where perception needed them, exactly the rate-allocation logic that residual vector quantization later generalized in [the RVQ post](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq).

### Dilated causal convolutions, and the receptive-field derivation

The second problem is context. A single audio sample is almost meaningless on its own; to predict the next sample of a vowel you need to know you are in a vowel, which means you need context spanning many milliseconds, which at 16 kHz is thousands of samples. A plain convolutional network would need an absurd number of layers to see that far back. WaveNet's central architectural idea, the one worth carrying away even if you never train a sample-level model, is the **dilated causal convolution**.

*Causal* means the convolution at position $t$ only looks at positions $\le t$, never into the future, which is exactly what the autoregressive factorization requires; you implement it by padding on the left and shifting so no output depends on a future input. *Dilated* means the convolution skips inputs by a fixed step called the dilation rate: a dilation of 1 is an ordinary convolution over adjacent samples, a dilation of 2 looks at every other sample, a dilation of 4 at every fourth, and so on. Stack layers with dilations that double, $1, 2, 4, 8, \dots, 2^{L-1}$, and the receptive field, the number of past samples that can influence the current prediction, grows *exponentially* with depth while the number of layers and the cost grow only *linearly*.

Here is the derivation, because the exponential is the entire point. With a kernel of size 2 and dilations doubling from layer to layer, layer $\ell$ (counting from 1) has dilation $2^{\ell-1}$. Each layer adds $2^{\ell-1}$ samples of reach to the receptive field. After $L$ layers the total receptive field is

$$R = 1 + \sum_{\ell=1}^{L} 2^{\ell-1} = 1 + (2^L - 1) = 2^L,$$

so the receptive field is $O(2^L)$ in the number of layers. Ten layers reach back $2^{10} = 1024$ samples; with a few stacked blocks of ten dilation layers each you reach thousands of samples, hundreds of milliseconds at 16 kHz, which is enough to capture the structure of phonemes and short prosodic units, for a network only a few dozen layers deep. That is the magic: long temporal context at logarithmic depth. Compare the alternative of a recurrent network, which can in principle carry unbounded context but must process samples strictly one at a time even during training and is far harder to parallelize; WaveNet's convolutional structure trains in parallel across all positions and still sees far into the past.

![A grid showing a stack of dilated causal convolution layers whose receptive field doubles from one to two to four to eight](/imgs/blogs/autoregressive-audio-models-wavenet-to-audiolm-3.png)

The grid in the figure above shows the structure that produces the exponential. Read it bottom to top as the input layer up through dilations 1, 2, 4, 8: each layer connects to a sparse, spread-out set of inputs from the layer below, and because the spacing doubles each level, the set of original input samples that can reach the top node, the receptive field, doubles too. Notice that the count of *connections per node* stays constant, two for a kernel of size two, so the compute per layer is constant; only the *reach* grows. This is the same trick that makes dilated convolutions useful in image segmentation, where you want a large spatial receptive field without downsampling, but in audio the axis being covered is time and the payoff is temporal context.

### A dilated causal convolution block in PyTorch

Here is a minimal WaveNet-style residual block, the actual computation: a causal pad, a dilated convolution, the gated activation that WaveNet uses in place of a plain nonlinearity, and a residual plus skip connection. The gated activation, $\tanh(W_f * x) \odot \sigma(W_g * x)$, splits the convolution output into a *filter* branch through $\tanh$ and a *gate* branch through a sigmoid and multiplies them elementwise, which empirically models audio better than a ReLU; it is the same multiplicative gating idea as in an LSTM cell, letting the network modulate how much of the filtered signal passes through.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedCausalConvBlock(nn.Module):
    """One WaveNet residual block: causal dilated conv + gated activation."""
    def __init__(self, channels: int, dilation: int, kernel_size: int = 2):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        # The left-pad that makes the conv causal: (kernel_size - 1) * dilation
        self.causal_pad = (kernel_size - 1) * dilation
        # 2*channels out: one half feeds tanh (filter), the other sigmoid (gate)
        self.conv = nn.Conv1d(channels, 2 * channels, kernel_size,
                              dilation=dilation)
        self.res_proj = nn.Conv1d(channels, channels, 1)   # 1x1 residual mix
        self.skip_proj = nn.Conv1d(channels, channels, 1)  # 1x1 skip output

    def forward(self, x):                       # x: (batch, channels, time)
        # Pad ONLY on the left so output[t] depends on input[<= t]: causality.
        y = F.pad(x, (self.causal_pad, 0))
        y = self.conv(y)                        # (batch, 2*channels, time)
        filt, gate = y.chunk(2, dim=1)          # split into filter and gate
        y = torch.tanh(filt) * torch.sigmoid(gate)   # the gated activation
        skip = self.skip_proj(y)                # collected across all blocks
        res = self.res_proj(y) + x              # residual keeps gradients alive
        return res, skip

# A small WaveNet body: a few stacks of doubling dilations.
class WaveNetBody(nn.Module):
    def __init__(self, channels=64, n_stacks=2, layers_per_stack=10):
        super().__init__()
        self.blocks = nn.ModuleList([
            DilatedCausalConvBlock(channels, dilation=2 ** layer)
            for _ in range(n_stacks)
            for layer in range(layers_per_stack)
        ])

    def forward(self, x):
        skips = 0
        for block in self.blocks:
            x, skip = block(x)
            skips = skips + skip       # sum of skip connections feeds the head
        return F.relu(skips)

# Receptive field of one stack of L doubling-dilation layers is 2**L samples.
body = WaveNetBody(channels=64, n_stacks=2, layers_per_stack=10)
# 2 stacks * (2**10) ≈ 2048 samples of context ≈ 128 ms at 16 kHz.
```

The thing to notice is how cheap the receptive field is. Two stacks of ten doubling-dilation layers give roughly 2,048 samples of context, about 128 milliseconds at 16 kHz, from twenty convolutional layers. The causal pad is the only subtle part: padding `(causal_pad, 0)` means pad on the left only, so the convolution at output position $t$ can only see inputs at positions $\le t$, never the future, which is the architectural enforcement of the autoregressive constraint. Get that pad wrong, pad on both sides, and you have leaked future information into the prediction and your training likelihood will look fantastic while generation produces garbage, a bug I have personally shipped and then spent an afternoon hunting.

There is a second subtlety worth internalizing, because it explains why people stack *multiple* blocks of dilations rather than one very deep block. If you keep doubling the dilation indefinitely, the convolution kernel becomes so spread out that adjacent layers sample wildly different, non-overlapping sets of inputs, and the network loses the ability to integrate fine local detail at the top of the stack. The standard remedy is to *reset* the dilation periodically: run a block of dilations $1, 2, 4, \dots, 512$, then start a new block again at $1, 2, 4, \dots$. Each block contributes its own $2^L$ of reach, so $K$ blocks of $L$ layers give a receptive field of roughly $K \cdot 2^L$ samples, but crucially every block re-grounds the representation in dense local context before spreading out again. This is why the original WaveNet used several stacks rather than one enormous one, and it is a pattern you see echoed in dilated-convolution architectures across domains: alternate spreading out for context with re-grounding for detail.

### Conditioning WaveNet: global and local

WaveNet was not only an unconditional sound generator; it was also a *vocoder* and a text-to-speech model, and the way it accepted conditioning is the direct ancestor of how modern token models accept it, so it is worth a paragraph. WaveNet distinguished two kinds of conditioning. **Global conditioning** is a single vector that holds for the entire utterance, a speaker-identity embedding, say, which biases the gated activation the same way at every time step so the whole output is rendered in one voice. **Local conditioning** is a time-varying signal aligned to the audio, such as a sequence of linguistic features or a mel-spectrogram, upsampled to the sample rate and injected into the gate at each step so the content of the audio follows the conditioning frame by frame. When WaveNet is used as a *neural vocoder*, the local conditioning is a mel-spectrogram produced by an acoustic model like Tacotron, and WaveNet's only job is to invert that mel-spectrogram into a high-fidelity waveform, sample by sample. That role, mel-to-waveform rendering, is exactly the job that the GAN vocoders in the [vocoder post](/blog/machine-learning/audio-generation/gan-vocoders-hifi-gan-and-fast-synthesis) later took over and did far faster, which is the clearest single example of sample-level autoregression being displaced not because it sounded worse but because it was too slow for the role.

The conditioning story matters here because it is the seed of the whole "recipe generalizes by conditioning" thesis we reach later. WaveNet already had the shape: an autoregressive core that predicts the next acoustic unit, plus a conditioning pathway that tells it *what* to render. Modern token models keep that exact shape and change two things, the acoustic unit from a raw sample to a codec token, and the conditioning pathway from upsampled mel features to a cross-attention over text or melody embeddings. The bones are WaveNet's.

## Why WaveNet was painfully slow, made precise

WaveNet's training is fully parallel. Because the true past samples are known for every position, you feed the whole waveform in at once, the causal convolutions compute the predicted distribution at every position simultaneously, and you take one gradient step on the average cross-entropy over all positions. This is fast and it scales. The problem is entirely at *generation* time, and it is worth being precise about why, because the same logic governs every autoregressive model.

At generation, you have no future samples; you are producing them. To sample $x_t$ you need the network's output at position $t$, which depends on $x_{<t}$, which includes $x_{t-1}$, which you only just produced. So you generate sample 1, feed it back, generate sample 2, feed it back, and so on, $N$ times, strictly in order. The naive implementation is even worse than it sounds, because a convolution recomputes overlapping windows; a single sample's generation in the naive scheme costs a full forward pass over the receptive field. Fast-WaveNet-style caching, which stores the intermediate activations at each dilation level in a queue and reuses them, brings the per-sample cost down to the minimum, but you still pay one sequential step per sample. At 16 kHz that is 16,000 steps per second of audio; at 24 kHz, 24,000.

Put a number on it. Suppose each forward pass for one sample takes 100 microseconds, an optimistic figure for a deep network even with caching on contemporary hardware. Then one second of 24 kHz audio costs $24{,}000 \times 100\,\mu s = 2.4$ seconds of compute, a real-time factor of 2.4, meaning generation takes more than twice as long as the audio lasts. Early WaveNet implementations without aggressive caching were far worse, with real-time factors in the tens or hundreds, which is the regime where my coffee got cold. The fidelity was state-of-the-art; the speed was unusable for anything interactive.

This sequential bottleneck spawned an entire subfield of fixes, and they split into two strategies that are worth naming because the second one wins the whole argument. The first strategy keeps modeling raw samples but tries to make sampling cheaper. WaveRNN replaced the deep convolutional stack with a compact single-layer recurrent network plus engineering tricks, sparse weights, subscaling, custom kernels, to get sample-level generation down toward real time. SampleRNN used a hierarchy of recurrent networks running at different time scales to model long sequences more efficiently. Parallel WaveNet went the most exotic route, distilling the autoregressive teacher into a parallel student via inverse-autoregressive flows so that generation became a single parallel pass, trading the exact-likelihood and simplicity of the teacher for speed. These are clever, and WaveRNN-class vocoders shipped in real products. But they are all fighting the same fundamental enemy, the sheer number of samples, with cleverness rather than by changing the target.

The second strategy is the one that reshaped the field, and it is almost embarrassingly simple in hindsight. **Stop modeling raw samples. Model codec tokens instead.**

## The shift to codec tokens: changing what you autoregress over

A neural audio codec, the subject of the [codecs post](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound), is an encoder-quantizer-decoder that turns a waveform into a short sequence of discrete tokens and back. Where WaveNet has 24,000 samples per second to predict, a typical EnCodec configuration on 24 kHz audio produces about 75 frames per second, and at each frame a residual vector quantizer emits one token per codebook. With, say, eight codebooks you have $75 \times 8 = 600$ tokens per second, but those 600 tokens reconstruct the same waveform that the 24,000 samples did. The codec has done the hard work of compressing the redundant, locally-predictable structure of the waveform into a compact code, and crucially that code is *discrete*, so it is exactly the kind of thing a next-token model eats.

This is the move. An autoregressive audio model in the modern sense does not look at the waveform at all. It is a transformer doing next-token prediction over the codec's discrete tokens, an audio *language model* in the most literal sense: same architecture as a text language model, same training objective, same sampling code, just a vocabulary of codec tokens instead of subword tokens. To generate audio you autoregress over tokens, then hand the generated token sequence to the codec's decoder, which renders it back to a waveform in a single parallel pass. The expensive sequential part now runs over hundreds of tokens per second instead of tens of thousands of samples, and the codec decoder, being a feed-forward convolutional network, costs essentially nothing.

The speedup is not subtle. Modeling 600 tokens per second instead of 24,000 samples per second is a 40-fold reduction in the number of sequential steps before you even account for the fact that a transformer can be optimized hard with key-value caching and batching. In practice, token-level autoregressive models reach real-time factors well below 1, generating audio faster than it plays, on a single modern GPU, which is the regime that makes interactive speech and on-demand music feasible. The entire reason the 2023-onward generation of audio models, VALL-E, MusicGen, AudioLM itself, are usable is that they autoregress over codec tokens, not samples.

Two engineering details compound the win, and both come for free from the language-modeling toolchain. The first is the **key-value cache**. A transformer step at position $t$ needs the keys and values of all previous positions to compute attention, and naively recomputing them every step would make generation cost $O(N^2)$ in total. The cache stores each position's keys and values once, when that position is first generated, and reuses them for all later steps, so each new step only computes attention against cached vectors and the total cost drops to $O(N)$. This is the same cache that makes text generation fast, and it transfers to audio tokens unchanged. The second is **batching across requests**: because the per-step transformer forward pass is bandwidth-bound rather than compute-bound at batch size one, you can generate audio for many requests at once with almost no per-request slowdown, which is what makes serving an audio model economical. Neither trick reduces the *number* of sequential steps, $N$ is still fixed by the token count, but together they make each step cheap and the throughput high, which is the practical difference between a model that demos and a model that serves.

It is worth pausing on a subtlety that confuses people coming from the sample-level world. You might think the codec *itself* must be slow because it has to process the waveform, but the codec is feed-forward and fully parallel: encoding and decoding a ten-second clip is a single pass over a convolutional network, milliseconds of compute, no sequential dependency at all. The codec moved all the per-sample work into a *non-autoregressive* network that runs in parallel, and left only the short token sequence for the *autoregressive* part. That division of labour, parallel codec on the outside, autoregressive transformer on the inside over a short code, is the architectural insight, and it is why the slogan "tokenize, then model the tokens" is the single most important sentence in modern audio generation.

![A graph of the modern token level autoregressive flow from a transformer through generated codec tokens to a codec decoder producing a waveform](/imgs/blogs/autoregressive-audio-models-wavenet-to-audiolm-4.png)

The flow in the figure above is the architecture every modern autoregressive audio model shares. A transformer, conditioned on whatever you are generating from, text for TTS, a melody for music, nothing for unconditional, predicts codec tokens one step at a time with key-value caching making each step cheap. The generated token sequence, a short discrete code, goes to the frozen codec decoder, the same decoder the codec was trained with, which renders it to a waveform in one parallel pass. The sequential cost lives entirely in the transformer and is paid over a few hundred tokens per second; the codec decoder is feed-forward and effectively free. Compare this to the WaveNet figure earlier, where the autoregressive model *was* the whole path from token to waveform and ran at sample rate. Changing the target from samples to tokens is the single highest-leverage decision in audio generation.

There is one wrinkle the codec introduces, and it is worth flagging because it is where designs differ. The codec emits multiple tokens per frame, one per RVQ codebook, so the per-frame data is not a single token but a small stack of them, ordered from the coarsest codebook, which carries the most important information, to the finest. A flat language model wants a single linear stream, so you have to decide how to flatten this $\text{frames} \times \text{codebooks}$ grid into a sequence. Flatten it fully and you are back to a long sequence; interleave or predict codebooks in parallel and you keep it short but complicate the model. MusicGen's *delay pattern*, which offsets each codebook by one step so the model predicts all codebooks of a frame in a staggered, parallel-ish way, is one influential answer; AudioLM's hierarchy is another. We will see both. The point for now is that "model codec tokens" is the winning idea, and the remaining design space is *how* you order and factor those tokens.

## AudioLM: autoregression as a content-first hierarchy

AudioLM, from Borsos and colleagues at Google in 2022, is the model that crystallized "audio generation is language modeling over tokens" and added the crucial insight that you should not use *one* kind of token but *two*, in a hierarchy. If you have read [semantic vs acoustic tokens](/blog/machine-learning/audio-generation/semantic-vs-acoustic-tokens) the motivation will be familiar, and I will not re-derive it here; the short version is that a flat language model over codec tokens produces locally gorgeous, globally incoherent audio, fluent speech in a language that does not exist, because the long-range content structure is too weak a signal in the acoustic token stream to compete with the overwhelmingly strong short-range acoustic correlations. AudioLM's fix is to factor the generation into stages, content first.

The hierarchy has three autoregressive stages, each a transformer, each predicting the next token in its own stream conditioned on what came before. The first stage models **semantic tokens**, derived from a self-supervised speech model like w2v-BERT, at a low rate; these capture phonetic and linguistic content, the *what is being said*, while being largely blind to speaker identity and acoustic detail. Because this stream is short and content-bearing, an autoregressive model over it learns long-range structure, grammar, the arc of a sentence, musical phrasing, that a flat acoustic model could not. The second stage models **coarse acoustic tokens**, the first few RVQ codebooks of the codec, conditioned on the semantic tokens; this fills in speaker identity and broad spectral structure. The third stage models **fine acoustic tokens**, the remaining codebooks, conditioned on the coarse ones, adding the fine recording detail. Decode the full acoustic token stack with the codec and you get the waveform.

![A graph of the AudioLM hierarchy from semantic tokens to coarse acoustic tokens to fine acoustic tokens feeding the codec decoder](/imgs/blogs/autoregressive-audio-models-wavenet-to-audiolm-5.png)

The hierarchy in the figure above is why AudioLM gives you both coherence and fidelity, two things that pull against each other in a flat model. The semantic stage *plans*: a short, content-rich sequence where an autoregressive model can actually learn that you are three words into a sentence about a brown fox, or four bars into a chord progression, and choose what comes next coherently. The coarse acoustic stage *renders the identity*: given the plan, it commits to a speaker or a timbre. The fine acoustic stage *renders the texture*: the breath, the room, the high-frequency detail. Each stage is an ordinary autoregressive transformer over a token stream; the power comes entirely from the factorization, from isolating content decisions onto a stream where they are the dominant signal rather than a rare minority drowned out by acoustic detail. This is the same inductive-bias-beats-brute-scale argument that the [semantic vs acoustic](/blog/machine-learning/audio-generation/semantic-vs-acoustic-tokens) post makes in full.

The reason this matters for *this* post specifically is that it shows autoregression's weakness, error accumulation over long sequences, being managed rather than ignored. A single flat autoregressive pass over thousands of acoustic tokens accumulates error: each sampled token feeds the next, so a small mistake early biases everything after it, and over seconds the model drifts off the content it should be following. AudioLM's hierarchy contains this by making the *content-determining* sequence, the semantic stream, short, so there are far fewer steps over which content error can accumulate, while the longer acoustic streams are tightly conditioned on the already-fixed plan and therefore have little room to drift in content even as they fill in detail. Factorization is not just for fidelity; it is also a brake on the error accumulation that is autoregression's native failure mode.

## A token-level sampling loop with temperature and top-k

Here is the practical heart of token-level autoregression: the sampling loop, with the controls you actually use to shape the output. This uses 🤗 `transformers` to encode a prompt with EnCodec and then samples codec tokens autoregressively with temperature and top-k, the same controls you know from text generation, applied to audio. The loop is deliberately explicit so you can see exactly where the sequential dependency lives.

```python
import torch
import torch.nn.functional as F
from transformers import EncodecModel, AutoProcessor

# The codec: turns a waveform into discrete tokens and back.
codec = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
codec.eval()

def sample_next_token(logits, temperature=0.9, top_k=250):
    """Apply temperature, then keep only the top_k logits, then sample."""
    logits = logits / temperature                  # flatter (>1) or sharper (<1)
    if top_k is not None:
        kth = torch.topk(logits, top_k).values[..., -1, None]
        logits = torch.where(logits < kth,         # mask out the long tail
                             torch.full_like(logits, float("-inf")), logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1) # ONE sequential draw

@torch.no_grad()
def generate_codec_tokens(model, prompt_tokens, n_steps, temperature=0.9, top_k=250):
    """Autoregressive token generation. Each step depends on the previous."""
    tokens = prompt_tokens                          # (batch, seq) seed tokens
    for _ in range(n_steps):                        # STRICTLY sequential loop
        logits = model(tokens)[:, -1, :]            # next-token distribution
        next_tok = sample_next_token(logits, temperature, top_k)
        tokens = torch.cat([tokens, next_tok], dim=1)  # feed back, then repeat
    return tokens

# Sampling controls, the levers you actually turn:
#   temperature < 1.0 -> more conservative, repetitive, "safe" audio
#   temperature > 1.0 -> more varied, riskier, more artifacts
#   top_k small (e.g. 50) -> tight, coherent; top_k large -> diverse
#   top_p (nucleus) -> keep the smallest set whose prob mass exceeds p
```

The `for` loop in `generate_codec_tokens` is the sequential bottleneck made literal: `torch.cat` appends the freshly sampled token, and the next iteration conditions on it. You cannot vectorize this loop across time, which is the autoregressive curse in code. In a real implementation `model(tokens)` would use a key-value cache so it only computes the new position rather than re-encoding the whole prefix each step, which is what makes each step cheap, but the *number* of steps is fixed by the sequence length and cannot be reduced without changing the model.

The sampling controls and their effect on audio are summarized in the table below; the through-line is that audio punishes the long tail of the distribution harder than text does, because a low-probability codec token tends to decode to an audible glitch rather than just an odd word.

| Control | Setting | Effect on audio | When to use |
| --- | --- | --- | --- |
| Temperature | < 1.0 | Conservative, stable, can repeat | Speech: keep voice intelligible |
| Temperature | > 1.0 | More varied, more artifacts | Music: more creative variety |
| Top-k | Small (e.g. 50) | Tight, coherent, fewer glitches | Speech, clean output |
| Top-k | Large (e.g. 250) | Diverse, riskier tail | Music, exploratory |
| Top-p | 0.9–0.95 | Adapts cutoff to distribution shape | General-purpose default |

The sampling controls are worth dwelling on because audio is less forgiving than text. **Temperature** scales the logits before the softmax: dividing by a temperature below 1 sharpens the distribution toward its most likely tokens, producing conservative, sometimes repetitive audio; a temperature above 1 flattens it, producing more variety but also more artifacts, the audio equivalent of a language model going off the rails. **Top-k** keeps only the $k$ most probable tokens and renormalizes, cutting off the long tail of low-probability tokens that, in audio, tend to be the ones that produce clicks, pops, and glitches. **Top-p** (nucleus) sampling keeps the smallest set of tokens whose cumulative probability exceeds $p$, adapting the cutoff to the shape of the distribution. For music, practitioners often run higher temperature and larger top-k for variety; for speech, lower temperature and tighter top-k to keep the voice intelligible and stable. These are the same knobs as in [autoregressive image models](/blog/machine-learning/image-generation/autoregressive-image-models), and the intuition transfers directly: you are trading diversity against the risk of sampling a token that derails coherence.

#### Worked example: pricing the sampling cost of a 10-second clip

Let me make the cost concrete with one running scenario, generating 10 seconds of audio with a token-level model and contrasting it with the sample-level alternative. Take EnCodec at 75 frames per second with 4 codebooks predicted via a delay pattern, so effectively the model produces on the order of 75 frames per second of *transformer steps* when codebooks are parallelized, call it roughly 750 sequential transformer steps for 10 seconds. On a single A100, a small audio transformer with KV caching might run each step in about 5 milliseconds, so 10 seconds of audio costs roughly $750 \times 5\,\text{ms} \approx 3.75$ seconds of compute, a real-time factor of about 0.38, comfortably faster than real time. Now the sample-level alternative: WaveNet at 24 kHz is 240,000 samples for 10 seconds, and even at an optimistic 100 microseconds per cached sample that is $240{,}000 \times 100\,\mu s = 24$ seconds, a real-time factor of 2.4. Same 10 seconds of audio, same autoregressive idea, and the token-level model is roughly *six times faster than real time while the sample-level model is more than twice slower than real time*, a swing of more than 6x in real-time factor that comes entirely from changing $N$ from 240,000 to 750. These are order-of-magnitude figures, not benchmarks; the exact numbers depend on model size, codebook count, and hardware, but the ratio is the durable lesson.

## The trade-offs of autoregression, stated honestly

Autoregressive models are not strictly better than their alternatives, and a principal engineer should be able to say exactly what they buy and what they cost. Here is the honest ledger, and it is the reason the field uses different generative engines for different jobs, a point the [diffusion for audio](/blog/machine-learning/audio-generation/diffusion-for-audio) post picks up directly.

On the credit side, autoregressive models give you three real advantages. First, *exact likelihood*: you optimize the actual log-probability of the data, not a bound, which makes training stable and well-understood and gives you a principled way to compare models and detect anomalies. Second, *quality and coherence on sequential structure*: because each token is generated conditioned on everything before it, autoregressive models are extraordinarily good at locally-coherent, causally-structured signals, which speech and music are, and they handle variable-length generation naturally because you just keep sampling until you decide to stop. Third, *simplicity and transfer*: an autoregressive audio model is a transformer language model, so it inherits the entire mature toolchain of language modeling, scaling laws, KV caching, sampling tricks, fine-tuning recipes, with almost no modification.

On the debit side, two costs are fundamental. First, *sequential inference*: generation cannot parallelize across time, so latency scales with sequence length, and even with codec tokens a long generation, a four-minute song, is a lot of sequential steps. This is the structural disadvantage versus diffusion and flow models, which generate the whole sequence in parallel across a fixed, often small, number of denoising steps regardless of length; for very long outputs, the parallel approaches pull ahead. Second, *error accumulation and exposure bias*: the model is trained on the *true* past, teacher-forced, but at generation it conditions on its *own* sampled past, which contains mistakes the model never saw during training. Errors compound. A token sampled slightly wrong shifts the context, making the next token a little more likely to be wrong, and over a long sequence the output can drift, off-key in music, off-content in speech, into repetition loops. AudioLM's hierarchy mitigates this by shortening the content-determining sequence, and sampling controls like top-k mitigate it by avoiding the worst tokens, but exposure bias is a real, named, structural weakness of the paradigm, and it is the same problem that plagues long-form text generation.

### Stress-testing the paradigm: where it breaks and what you do

It helps to walk the failure modes deliberately, because knowing how a model breaks tells you how to operate it. *What happens when you push the temperature too high?* The distribution flattens, the model starts sampling low-probability codec tokens, and in audio those tail tokens are disproportionately the ones that decode to clicks, pops, and metallic artifacts, because they live in rarely-visited corners of the codebook the codec never learned to render cleanly. The fix is a tighter top-k or a lower temperature, trading diversity for cleanliness. *What happens when you ask for a four-minute song in one autoregressive pass?* The sequence runs to tens of thousands of tokens, exposure bias has thousands of steps to compound, and the model tends to lose the beat, slide out of key, or fall into a repetition loop where it samples a token that makes the same token more likely next, locking into a groove that never resolves. The practical answers are to generate in overlapping windows and stitch, to add an explicit timing or position conditioning so the model knows where it is in the piece, or to switch to a parallel diffusion model whose error does not compound with length, which is exactly why long-form music skews toward the latent-diffusion approach.

*What happens when the codec drops the high frequencies at low bitrate?* If you run the model over a codec configured for very few codebooks to save tokens, the acoustic tokens simply cannot represent fine high-frequency detail, so even a perfect language model produces dull, muffled audio, the ceiling is set by the codec, not the model. This is a useful diagnostic: if your output sounds low-fi regardless of how well the language model is doing, suspect the codec's bitrate before you suspect the model, and revisit the rate-distortion trade in [the codec post](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound). *And what happens when the bottleneck is the vocoder or decoder, not the model?* For a token model the decoder is cheap, but in a mel-pipeline TTS system the vocoder can dominate latency, and there the right move is to swap a slow autoregressive or diffusion vocoder for a fast GAN vocoder. The general lesson from all four stress tests is the same: an autoregressive audio system is a chain of stages, and you must profile to find which stage is actually the bottleneck before you optimize, because intuition about where the time goes is frequently wrong.

![A matrix comparing WaveNet WaveRNN AudioLM and a token language model across what they model steps per second of audio real time factor and quality](/imgs/blogs/autoregressive-audio-models-wavenet-to-audiolm-6.png)

The matrix in the figure above lays the trade-offs out as numbers you can act on. Read across the rows from the slowest, purest model to the fastest, most practical one. WaveNet models raw samples and pays roughly 24,000 sequential steps per second of audio with a real-time factor well above 1 in its day, in exchange for the best raw-waveform fidelity of its era. WaveRNN models raw samples too but with a compact recurrent core and heavy engineering that pushes the real-time factor down toward 1, trading some architectural elegance for usable speed. AudioLM and the general token language model both model *codec tokens*, dropping to a few hundred sequential steps per second and real-time factors comfortably below 1, with AudioLM adding the hierarchy that buys long-range coherence. The single most important column is "steps per second of audio," because that number, set entirely by *what you autoregress over*, is the dominant term in latency, and watching it fall from 24,000 to a few hundred is the whole history of the field compressed into one column.

## How the recipe generalizes: VALL-E and MusicGen

The reason this post sits where it does in the series, at the head of the generative-engines track, is that the autoregressive-over-codec-tokens recipe is not one model, it is a *template*, and the two most influential audio models of the 2023 generation are both instances of it. Seeing them as the same recipe with different conditioning is the through-line of the whole series.

**VALL-E**, from Wang and colleagues at Microsoft in 2023, is text-to-speech reframed as exactly this recipe. It treats TTS as conditional codec-token language modeling: condition an autoregressive transformer on the phoneme sequence of the text you want spoken *and* on the codec tokens of a short reference clip of the target speaker, then autoregress the codec tokens of the output speech. Because the model is an in-context learner over codec tokens, it can clone a voice from a three-second reference prompt with no fine-tuning, the reference tokens simply become part of the conditioning prefix, and the model continues in that voice. It is the same content-first instinct as AudioLM, with a coarse autoregressive stage for the first codebook and a non-autoregressive stage for the rest, and it is the subject of the dedicated [VALL-E post](/blog/machine-learning/audio-generation/neural-codec-language-model-tts-vall-e). The headline is that zero-shot voice cloning fell out of *the autoregressive codec-token recipe plus speaker conditioning*, nothing more exotic.

**MusicGen**, from Copet and colleagues at Meta in 2023, is the same recipe pointed at music. A single autoregressive transformer over EnCodec tokens, conditioned on a text description, and optionally on a melody, generates music. Its key engineering contribution is the *codebook interleaving / delay pattern* I mentioned earlier: instead of flattening the four RVQ codebooks into a 4x-longer sequence or predicting them with four separate models, MusicGen offsets each codebook by a growing delay so a single model predicts all four codebooks per frame in a staggered pattern, keeping the sequence short while respecting the coarse-to-fine dependency between codebooks. That one trick is what lets a single-stage model match the quality of AudioLM's multi-stage hierarchy on music while being simpler to train and serve, and it is the subject of the [MusicGen post](/blog/machine-learning/audio-generation/music-generation-musiclm-and-musicgen). Again: autoregressive transformer, codec tokens, conditioning. The recipe is the same; only the conditioning signal and the codebook-handling differ.

This is the unifying claim of the series spine for the autoregressive family. Once you have a good neural codec, generating audio of any kind, speech, music, sound effects, is *next-token prediction over the codec's tokens with the right conditioning*, and the differences between famous models are mostly differences in conditioning and codebook ordering, not in the core idea. The same recipe also underlies the autoregressive half of the [autoregressive-vs-diffusion 2026 showdown](/blog/machine-learning/image-generation/autoregressive-vs-diffusion-the-2026-showdown) in the image series, which is worth reading for the cross-modal parallel: pixels-as-tokens and samples-as-tokens are the same bet.

## Case studies and real numbers

Let me ground all of this in figures from the literature, with the usual honesty about precision: where I give an exact number it comes from a paper or release, and where I am estimating I say so, because fabricating a precise FAD or MOS would betray the whole point of the [audio quality metrics](/blog/machine-learning/audio-generation/audio-quality-metrics) post.

The table below summarizes the four models on the axes that matter for an engineering decision. Treat the steps-per-second and real-time-factor columns as order-of-magnitude, hardware-dependent figures, not benchmarks; the *ratios* between rows are the durable signal, and they trace directly to the choice of modeling target.

| Model | Year | Models | Steps / sec audio | Real-time factor | What it bought |
| --- | --- | --- | --- | --- | --- |
| WaveNet | 2016 | Raw samples | ~24,000 | > 1 (slow) | Best raw-waveform fidelity of its era |
| WaveRNN | 2018 | Raw samples | ~24,000 | ~1 | Sample-level quality at real-time speed |
| AudioLM | 2022 | Codec tokens | ~hundreds | < 1 (fast) | Long-range coherence via hierarchy |
| MusicGen / token LM | 2023 | Codec tokens | ~hundreds | < 1 (fast) | Single-stage, conditioned, servable |

**WaveNet (van den Oord et al., 2016).** WaveNet's headline result was a mean opinion score for naturalness that substantially closed the gap to human speech on US-English and Mandarin text-to-speech, beating the best concatenative and parametric systems of the time; the original paper reported MOS gains of roughly 0.5 points over the previous best on a 5-point scale, a large jump perceptually. The cost was the sample-level autoregression we have dwelt on: generation at sample rate, real-time factors well above 1 in the original, which is why production systems used distilled or engineered variants. The lesson that endured was the dilated-causal-convolution receptive-field trick, which outlived sample-level modeling and shows up wherever you want long context cheaply.

**WaveRNN (Kalchbrenner et al., 2018).** WaveRNN's contribution was speed without abandoning sample-level quality: a single-layer recurrent core plus sparsification and custom kernels reached real-time or faster sample-level synthesis on mobile-class hardware, a real-time factor around or below 1 where WaveNet sat well above. It is the proof that you *can* make sample-level autoregression fast with enough engineering, but it is also the high-water mark of that strategy, because the token-level approach made the whole problem easier by shrinking $N$ rather than speeding up each step.

**AudioLM (Borsos et al., 2022).** AudioLM demonstrated that hierarchical token language modeling produces speech and piano continuations with both long-term coherence and high fidelity, generating coherent continuations of several seconds that maintained speaker identity, prosody, and, for music, key and rhythm, where flat models drifted. Its semantic-then-acoustic factorization is the design that the rest of the token-LM family inherited, and its ablations are the cleanest demonstration that the hierarchy, not scale, is what buys coherence.

**MusicGen (Copet et al., 2023).** MusicGen ships in sizes from roughly 300M to 3.3B parameters, generates at the codec frame rate so a 30-second clip is a few thousand transformer steps, and on a modern GPU runs comfortably faster than real time for the smaller sizes; the paper reports competitive FAD and human-preference scores against prior music models while being a single-stage model. It is the practical, open workhorse of autoregressive music generation and the clearest evidence that the delay-pattern trick lets one autoregressive model do what previously took a multi-stage hierarchy.

**VALL-E (Wang et al., 2023).** VALL-E is the case study that makes the "recipe plus conditioning" thesis tangible, and its headline result is the one that reset expectations for text-to-speech: zero-shot voice cloning from a *three-second* enrollment clip, with no per-speaker fine-tuning, simply by prepending the reference speaker's codec tokens to the conditioning prefix and letting the autoregressive model continue in that voice. The paper reported large improvements in speaker similarity and naturalness over the prior strong zero-shot baseline on the LibriSpeech test set, and crucially it showed the model preserving the *acoustic environment* and emotional tone of the prompt, because those are encoded in the acoustic tokens it conditions on. The lesson for this post is structural: nothing about VALL-E's architecture is exotic relative to AudioLM; it is the same autoregressive-over-codec-tokens core with the conditioning pointed at phonemes plus a speaker prompt. The capability, instant voice cloning, emerged from the *recipe*, and that is why understanding the recipe matters more than memorizing any single model.

What ties these four case studies together is the trajectory of the "steps per second of audio" number and what each model did with the headroom it bought. WaveNet spent everything on sample-level fidelity and had no headroom left for speed. WaveRNN spent its engineering on clawing back speed without changing the target. AudioLM, MusicGen, and VALL-E all changed the target to codec tokens, banked the order-of-magnitude reduction in steps, and *reinvested* the headroom into capability: AudioLM into a coherence-buying hierarchy, MusicGen into a single-stage model with melody conditioning, VALL-E into in-context voice cloning. The codec did not just make autoregression faster; it freed up the modeling budget that these models then spent on the things that actually made them useful.

#### Worked example: choosing an engine for a real product constraint

Suppose you are building a feature that generates a 15-second musical sting on demand inside an interactive app, and your product constraint is that the user should hear it within about two seconds of pressing the button. Walk the trade-offs. A sample-level autoregressive model is immediately disqualified: 15 seconds at 24 kHz is 360,000 samples, and even at 100 microseconds per cached sample that is 36 seconds of compute, an 18x violation of the budget. A token-level autoregressive model like MusicGen-small generates roughly 15 seconds of audio in a few thousand transformer steps; at a few milliseconds per step on a single A100 that is on the order of a few seconds of compute, right at the edge of the budget, and tightenable with a smaller model, fewer codebooks, or batching, this is the regime where token-level autoregression is the right tool. If the budget were tighter still, or the clip far longer, a *parallel* generator, a diffusion or flow model that produces the whole clip in a small fixed number of steps regardless of length, becomes attractive precisely because its latency does not scale with output length the way an autoregressive model's does; that is the [diffusion for audio](/blog/machine-learning/audio-generation/diffusion-for-audio) trade-off. The decision turns on the same quantity all post long: the number of sequential steps, which for autoregression is the output length and for the parallel families is a small constant.

## When to reach for autoregression, and when not to

Here is the decisive recommendation, because a comparison without a verdict is just a list.

**Reach for an autoregressive token model when** you need exact likelihood, when your output is naturally causal and variable-length and not too long, speech utterances, short-to-medium music clips, sound effects, when you want to clone a voice or follow a melody via in-context conditioning on a prompt, and when you value the maturity and transferability of the transformer language-modeling stack. For text-to-speech and for short music, autoregression over codec tokens is, as of 2026, the default that works, and VALL-E and MusicGen are why. The in-context conditioning property, where you prepend a reference prompt and the model continues in that style or voice, is a genuine superpower that the parallel families do not get for free.

**Do not reach for sample-level autoregression at all** in 2026 unless you have a very specific reason; raw-waveform autoregression was a landmark but its speed is a structural dead end now that codecs exist, and even as a vocoder the GAN vocoders covered in the [vocoder post](/blog/machine-learning/audio-generation/gan-vocoders-hifi-gan-and-fast-synthesis) hit the quality bar far faster. **Do not reach for token-level autoregression for very long outputs** where latency scales painfully with length, a four-minute song generated in a single autoregressive pass is a lot of sequential steps and a lot of room for the beat to drift; here a parallel diffusion or flow model that generates the whole length in a fixed number of steps, with timing conditioning for length, is usually the better engineering choice, which is exactly why Stable Audio and the latent-diffusion music models exist. **And do not reach for a flat autoregressive model over acoustic tokens** when you need long-range coherence; either add a semantic stage as AudioLM does, or accept the babble. The choice between autoregression, diffusion, and flow is the subject of the next several posts and ultimately of the [capstone](/blog/machine-learning/audio-generation/building-an-audio-generation-stack), and the honest answer is that they coexist because they sit at different points on the fidelity-controllability-speed-length surface.

## Putting the family in its place

It helps to see the autoregressive family next to its siblings so the boundaries are clear, because the next posts in this track, diffusion and flow matching, are defined partly by contrast with this one.

![A tree of the audio generative families splitting into autoregressive diffusion flow and gan vocoder branches](/imgs/blogs/autoregressive-audio-models-wavenet-to-audiolm-7.png)

The taxonomy in the figure above places autoregression among the generative families you will meet across this track. The autoregressive branch is the subject of this post: sequential next-token prediction, exact likelihood, excellent coherence, latency that scales with length. The diffusion branch, coming next, generates in parallel across a fixed number of denoising steps, decoupling latency from length at the cost of an approximate likelihood and many function evaluations. The flow branch, the [flow-matching post](/blog/machine-learning/audio-generation/flow-matching-and-consistency-for-audio), is a close cousin of diffusion that learns a straight-line transport and can generate in very few steps. The GAN-vocoder branch is a different beast, a fast feed-forward mel-to-waveform renderer rather than a full generative model, and it often sits *underneath* the others as the final synthesis stage. The thing the tree makes plain is that these are not competitors to be ranked once and for all; they are tools at different points on the trade-off surface, and a real system, the kind the [capstone](/blog/machine-learning/audio-generation/building-an-audio-generation-stack) builds, often uses several together, an autoregressive token model planning content and a fast vocoder rendering the final waveform.

It is also worth naming the cross-modal parallel explicitly, because it is more than an analogy. The move from sample-level to token-level autoregression in audio is the *exact same move* as the move from pixel-level to patch-token-level autoregression in images, the subject of [autoregressive image models](/blog/machine-learning/image-generation/autoregressive-image-models). In both cases the naive version, model the raw signal element by element, is theoretically clean and practically hopeless because the element count is enormous; in both cases the fix is a learned tokenizer, a codec for audio, a VQ-VAE or VQGAN for images, that compresses the signal into a short discrete code you can autoregress over cheaply. The codec is to audio what the patch tokenizer is to images, and the autoregressive transformer on top is, modulo the vocabulary, the same model. If you internalize one transferable idea from this post, make it this: *autoregression's cost is the number of elements, so the highest-leverage move is almost always to autoregress over a shorter, learned discrete representation rather than the raw signal.*

The parallel even extends to the failure modes and their fixes, which is a good sign that the analogy is structural rather than cosmetic. Image autoregressive models also suffer exposure bias, also drift on long generations, also use temperature and top-k to control the sample, and also benefit from a coarse-to-fine ordering of the tokens, the image analogue of AudioLM's semantic-then-acoustic hierarchy. Where the two diverge is in what "long" means and therefore in how much the sequential cost hurts: an image is a fixed, modest number of tokens, a few hundred to a couple thousand, so its autoregressive generation terminates quickly, whereas audio is *intrinsically* a long signal whose token count grows without bound as the clip lengthens. That single difference, audio is open-ended in time while an image is bounded in space, is why the audio field pushed so much harder on the codec to shrink the per-second token count, and why the parallel diffusion and flow families took a larger share of audio generation than they did of single-image generation. The lesson generalizes: the right generative engine depends not just on the modality but on how the modality's *length* scales, and audio's unbounded length is the pressure that shaped every choice in this post.

## A complete minimal pipeline, end to end

To close the loop between theory and something you can run, here is the full token-level autoregressive pipeline as a single flow, encode a seed waveform to codec tokens, autoregress new tokens, decode back to a waveform, using 🤗 `transformers` for the codec and the sampling loop from earlier. This is the skeleton that VALL-E and MusicGen flesh out with conditioning; stripped of the conditioning, it is the bare autoregressive-over-tokens recipe.

![A vertical stack of a minimal autoregressive pipeline from a seed waveform through codec encode to the token loop to codec decode and out to a generated waveform](/imgs/blogs/autoregressive-audio-models-wavenet-to-audiolm-8.png)

The stack in the figure above is the pipeline as five stages, and the colour of each stage tells you where the cost is. Loading and encoding the seed waveform is free, a single feed-forward pass through the codec's encoder. The autoregressive token loop in the middle is the only expensive stage, the sequential bottleneck we have circled all post, and it is the single place latency is spent. Decoding the generated tokens back to a waveform is free again, a single feed-forward pass through the codec's decoder. The shape of this pipeline, cheap-encode, expensive-loop, cheap-decode, is the same for every modern autoregressive audio model regardless of what conditioning you bolt onto the loop, and recognizing it is most of what you need to reason about the latency of any token-level system.

```python
import torch
import torchaudio
from transformers import EncodecModel, AutoProcessor

codec = EncodecModel.from_pretrained("facebook/encodec_24khz").eval()
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

# 1. Load a seed clip and resample to the codec's rate (24 kHz here).
wav, sr = torchaudio.load("seed.wav")
wav = torchaudio.functional.resample(wav, sr, 24000)

# 2. Encode the waveform to discrete codec tokens (the "vocabulary").
inputs = processor(raw_audio=wav.squeeze().numpy(),
                   sampling_rate=24000, return_tensors="pt")
with torch.no_grad():
    encoded = codec.encode(inputs["input_values"], inputs["padding_mask"])
codes = encoded.audio_codes        # (1, 1, n_codebooks, n_frames) discrete ids

# 3. Here a trained audio LM would autoregress NEW tokens from these as a prompt:
#       new_codes = generate_codec_tokens(audio_lm, codes, n_steps=750, ...)
#    using the sampling loop shown earlier (temperature + top_k). The codec's
#    tokens ARE the vocabulary; the LM never touches the waveform.

# 4. Decode tokens back to a waveform in ONE parallel pass (the cheap part).
with torch.no_grad():
    out = codec.decode(codes, [None])    # feed-forward, not autoregressive
out_wav = out.audio_values.squeeze(0)

# 5. Write the result. The whole generative cost was step 3's sequential loop.
torchaudio.save("generated.wav", out_wav, 24000)
print(f"codec frames: {codes.shape[-1]}, codebooks: {codes.shape[-2]}")
```

Step 3 is the only sequential, expensive part, and it is where the audio language model lives; steps 2 and 4, encode and decode, are feed-forward and effectively free. That asymmetry is the architectural payoff of modeling tokens: you pay the autoregressive tax only over the short token sequence, and the codec handles the waveform with parallel convolutions on either side. Swap in a conditioning prefix, text phonemes for TTS, a text description for music, and you have the skeleton of every modern autoregressive audio model. The whole jump from WaveNet, where the autoregressive model *was* the waveform path and ran at sample rate, is captured by the fact that here the autoregressive loop runs over `n_frames` codec frames, a number in the hundreds, not the tens of thousands.

## Key takeaways

- **The chain rule is exact; the parameterization is the model.** Every autoregressive model factorizes $p(x) = \prod_t p(x_t \mid x_{<t})$ and trains by maximizing the per-position log-likelihood, which for discrete tokens is exactly cross-entropy. This gives you an exact likelihood, autoregression's signature advantage over GANs and diffusion.
- **Sampling is sequential, so latency scales with sequence length.** Training parallelizes across positions; generation does not. The number of sequential steps, $N$, is the dominant term in latency, which makes *what you autoregress over* the highest-leverage decision in the design.
- **WaveNet modeled raw samples with dilated causal convolutions.** Doubling dilations give a receptive field of $O(2^L)$ for $L$ layers, long temporal context at logarithmic depth, plus µ-law quantization to a 256-way softmax and gated activations. It sounded great and ran at sample rate, which was the problem.
- **The decisive move was samples to codec tokens.** A neural codec turns 24,000 samples per second into a few hundred discrete tokens per second, cutting sequential steps by one to two orders of magnitude. A modern autoregressive audio model is a transformer doing next-token prediction over codec tokens, an audio language model.
- **AudioLM made it a content-first hierarchy.** Semantic tokens plan content on a short stream where coherence is learnable; coarse and fine acoustic tokens render identity and texture. The factorization buys both long-range coherence and high fidelity and brakes error accumulation.
- **The recipe generalizes by conditioning.** VALL-E is the recipe plus speaker conditioning, yielding zero-shot voice cloning from a 3-second prompt; MusicGen is the recipe plus text and melody conditioning plus the delay-pattern codebook trick. Same core idea, different conditioning.
- **Autoregression's weaknesses are sequential inference and exposure bias.** Latency scales with length, hurting very long outputs, and generation conditions on the model's own imperfect past, so errors accumulate and outputs can drift. Sampling controls (temperature, top-k, top-p) and hierarchical factorization mitigate but do not eliminate this.
- **Reach for autoregression for variable-length, coherent, not-too-long audio**, especially speech and short music, and when you want in-context conditioning. Reach for parallel diffusion or flow when outputs are very long and latency must not scale with length. Never use sample-level autoregression in 2026 when a codec is available.

## Further reading

- **van den Oord et al., "WaveNet: A Generative Model for Raw Audio" (2016)** — the dilated-causal-convolution architecture, µ-law quantization, and gated activations; the origin of sample-level autoregressive audio.
- **Kalchbrenner et al., "Efficient Neural Audio Synthesis" (WaveRNN, 2018)** — making sample-level autoregression fast with a compact recurrent core, sparsification, and custom kernels.
- **Mehri et al., "SampleRNN: An Unconditional End-to-End Neural Audio Generation Model" (2017)** — the multi-scale recurrent hierarchy for long-sequence raw-audio modeling.
- **Borsos et al., "AudioLM: a Language Modeling Approach to Audio Generation" (2022)** — the semantic-then-acoustic hierarchical token language model that defined modern autoregressive audio.
- **Wang et al., "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers" (VALL-E, 2023)** — TTS as codec-token language modeling with 3-second-prompt voice cloning.
- **Copet et al., "Simple and Controllable Music Generation" (MusicGen, 2023)** — single-stage autoregressive music generation with the codebook delay pattern.
- **Within-series:** [neural audio codecs](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound), [semantic vs acoustic tokens](/blog/machine-learning/audio-generation/semantic-vs-acoustic-tokens), [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard), and the [diffusion for audio](/blog/machine-learning/audio-generation/diffusion-for-audio) and [VALL-E](/blog/machine-learning/audio-generation/neural-codec-language-model-tts-vall-e) and [MusicGen](/blog/machine-learning/audio-generation/music-generation-musiclm-and-musicgen) follow-ups, building toward the [capstone](/blog/machine-learning/audio-generation/building-an-audio-generation-stack).
- **Cross-modal:** [autoregressive image models](/blog/machine-learning/image-generation/autoregressive-image-models) for the pixels-as-tokens parallel, the same bet on a shorter learned discrete representation.
