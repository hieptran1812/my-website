---
title: "Zipformer: How a U-Net Encoder, BiasNorm and ScaledAdam Beat the Conformer at a Third of the FLOPs"
publishDate: "2026-10-30"
date: "2026-10-30"
description: "A technique-by-technique reading of Zipformer: why squeezing the middle of an ASR encoder down to 6.25 Hz is nearly free, how one set of attention weights feeds three modules, and why a new normalizer, two new activations and a scale-aware optimizer matter as much as the architecture."
category: "paper-reading"
subcategory: "Speech Processing"
tags: ["paper-reading", "zipformer", "speech-recognition", "asr", "conformer", "scaledadam", "biasnorm", "transducer", "encoder-architecture", "icefall"]
author: "Hiep Tran"
featured: true
readTime: 13
image: "/imgs/blogs/zipformer-fig1.webp"
excerpt: "Conformer runs every layer at 25 Hz. Zipformer runs its middle at 6.25 Hz, reuses attention weights, and swaps LayerNorm, Swish and Adam for BiasNorm, Swoosh and ScaledAdam. The surprise: the optimizer swap is the biggest single win."
paper:
  title: "Zipformer: A Faster and Better Encoder for Automatic Speech Recognition"
  authors: "Zengwei Yao, Liyong Guo, Xiaoyu Yang, Wei Kang, Fangjun Kuang, Yifan Yang, Zengrui Jin, Long Lin, Daniel Povey"
  venue: "ICLR 2024"
  url: "https://arxiv.org/abs/2310.11230"
---

> [!tldr]
> **Claim.** Zipformer replaces the Conformer as an ASR encoder: faster, less memory, lower word error rate (WER) on LibriSpeech, Aishell-1 and WenetSpeech.
>
> **Mechanism.** Six stacks form a U-Net in time (50, 25, 12.5, 6.25, 12.5, 25 Hz); each block computes attention weights once and reuses them three times. Underneath: BiasNorm, SwooshR/SwooshL activations, and the ScaledAdam optimizer.
>
> **Headline.** Zipformer-L reaches 2.06 / 4.63 % WER (test-clean / test-other) at 107.7 GFLOPs. The authors' own Conformer-L reproduction gets 2.46 / 5.55 % at 294.2 GFLOPs, so Zipformer is better on both axes with 63% fewer FLOPs.
>
> **Surprise.** In the ablation, the largest single regression comes from swapping ScaledAdam back to Adam (+0.72 test-other), not from any architectural change.
>
> **Limits.** Sizes are matched on FLOPs, not parameters; ablations use one model size and single runs.

![Figure 1 from Yao et al. (2024): the overall Zipformer encoder, six stacks at 50, 25, 12.5, 6.25, 12.5 and 25 Hz joined by Downsample, Upsample and Bypass modules](/imgs/blogs/zipformer-fig1.webp)

The diagram above is the whole encoder; the rest of this post takes it apart one technique at a time.

## The problem

A 30-second utterance is 3,000 filterbank frames at 100 Hz. The [Conformer](/blog/machine-learning/speech-to-text/canary-parakeet-fastconformer-asr) subsamples to 25 Hz and runs every layer there, paying for multi-head self-attention (MHSA) twice over the length: once to compute a $T \times T$ weight matrix, once to apply it. At 750 frames that is 562,500 weights per head per layer, even though phonemes last 50 to 200 ms and most layers only need coarse context. Squeezeformer had tried running the middle of the encoder at half rate. Zipformer pushes that much harder and adds the optimization changes Daniel Povey's group had been accumulating in the icefall toolkit.

There is also a reproducibility gap the authors flag openly: the original Conformer paper reports 2.1 / 4.3 % on LibriSpeech, while open-source reimplementations land around 2.3 to 2.7 / 5.1 to 6.5 %.

## Contributions

1. A **U-Net-like encoder** whose middle stack runs at 6.25 Hz, eight times coarser than the first.
2. A **Zipformer block** that shares one set of attention weights across two self-attention modules and a new **Non-Linear Attention** (NLA) module.
3. **BiasNorm**, a LayerNorm replacement that keeps some vector-length information.
4. **SwooshR / SwooshL**, activations whose slope does not vanish for negative inputs.
5. **ScaledAdam**, an Adam variant that steps in proportion to each tensor's scale and learns that scale, plus the **Eden** schedule.

## Method

### U-Net downsampled stacks

**The problem.** Attention cost grows with $T^2$, and a constant frame rate spends that cost uniformly, even in layers that only need coarse context.

**Intuition.** Reading through a zoom lens: letters at full zoom, then sentences, then the paragraph, then back in to write each word down. The six stacks are those zoom levels. The coarse middle is cheap because there is little to look at, and Bypass connections carry fine detail around it, like a U-Net's skip connections.

**Mechanism.** A Conv-Embed front end (three 2-D convolutions plus a ConvNeXt layer) turns 100 Hz features into 50 Hz. Each stack after the first downsamples on entry, runs its blocks at the lower rate, upsamples on exit, and blends the result with its input. Between stacks the sequence is always 50 Hz; a final Downsample emits 25 Hz. Embedding widths differ per stack (Zipformer-M: 192, 256, 384, 512, 384, 256), so outputs are truncated or zero-padded to fit.

Downsampling by $k$ averages each group of $k$ frames with $k$ softmax-normalized learned weights; upsampling repeats each frame $k$ times. The stack output then passes through a Bypass:

$$
\text{Bypass}(x, y) = (1 - c) \odot x + c \odot y
$$

where $x \in \mathbb{R}^{T \times D}$ is the stack input, $y$ the upsampled output, and $c \in \mathbb{R}^{D}$ a learned per-channel weight. Early on $c$ is clamped to $[0.9, 1.0]$ so modules must learn something; after 20,000 steps the floor drops to 0.2.

**Worked example.** Downsample by 2 with learned logits $(0.3, -0.3)$: the softmax gives $(0.646, 0.354)$, so frames $a, b$ become $0.646a + 0.354b$. For the cost picture, take a 0.32-second clip:

![Frames seen by each Zipformer stack for a 0.32 s clip. Attention cost scales with T squared, so the 6.25 Hz middle stack pays 1/64 of the 50 Hz stack](/imgs/blogs/zipformer-1.webp)

Stack 1 sees 16 frames (a $16 \times 16$ attention matrix, 256 entries). Stack 4 sees 2 frames (4 entries), 64 times cheaper. Zipformer-M puts its most layers (4) and its widest embedding (512) exactly in that cheapest stack, so capacity lands where attention is nearly free.

**When it fails.** The 50 Hz first stack costs four times a 25 Hz layer, which is why it stays 2 layers deep. Coarse frames also need long chunks of audio, a problem for streaming. Still, removing downsampling (12 blocks at width 512, 94.2 M parameters) is worse: 5.09 vs 4.79 % test-other with 44% more parameters.

### The Zipformer block: compute attention once, use it three times

![Figure 2 from Yao et al. (2024): (left) the Zipformer block, (right) the Non-Linear Attention module](/imgs/blogs/zipformer-fig2.webp)

**The problem.** MHSA does two quadratic things: compute weights from queries and keys, then aggregate values with them. A Conformer block does both once per block.

**Intuition.** A committee draws its seating chart (who listens to whom) once, then reuses it for three rounds of discussion. Within one block the chart barely changes, so redrawing it is waste.

**Mechanism.** Zipformer splits MHSA into a **Multi-Head Attention Weight** module (MHAW), which only computes weights, and **Self-Attention** (SA) modules, which only aggregate. A block runs MHAW on its input, then feed-forward, NLA, and two groups of (SA, convolution, feed-forward), with Bypasses in the middle and at the end and a BiasNorm on the output. Both SA modules reuse MHAW's weights. There is no per-module LayerNorm; the ScaledAdam section explains why that works.

NLA uses a single head of those weights in a gated form. With block input $x \in \mathbb{R}^{T \times D}$, three linear projections produce $A, B, C \in \mathbb{R}^{T \times 3D/4}$, and

$$
\text{NLA}(x) = \text{Linear}\big(A \odot \text{attn}(\tanh(B) \odot C)\big)
$$

where $\odot$ is element-wise multiplication, $\text{attn}(\cdot)$ multiplies along time by one head's $T \times T$ weight matrix, and the outer linear maps $3D/4$ back to $D$. The $\tanh(B)$ gate decides what each frame broadcasts; $A$ decides what each frame takes from the aggregate. For Zipformer-M's widest stack ($D = 512$, 8 heads), one MHAW pass yields an $8 \times T \times T$ tensor; NLA reads one head, each SA reads all 8, and $A, B, C$ have 384 channels each.

**When it fails.** Reuse assumes the attention patterns within a block are similar. The ablation mildly supports it: dropping NLA costs 0.18 test-other points, and dropping NLA *and* sharing is slower and no better (5.10 %). Two Conformer-style blocks in place of each Zipformer block is 0.16 worse with more parameters (73.9 M vs 65.6 M). These are single runs, so gaps under 0.2 points deserve caution.

### BiasNorm

**The problem.** Trained Conformers showed two pathologies: one channel pinned at a huge constant such as 50, and "dead" modules with outputs around $10^{-6}$. The authors' reading: LayerNorm erases the vector's length, so the model fakes a length signal with a giant channel; and LayerNorm's scale $\gamma$ can drift to zero and oscillate around it, flipping the sign of the module's gradient on every crossing so it never learns.

$$
\text{LayerNorm}(x) = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} \odot \gamma + \beta
$$

Here $x \in \mathbb{R}^D$ is one frame's vector, the mean and variance run over its $D$ channels, and $\gamma, \beta \in \mathbb{R}^D$ are a learned scale and bias.

**Intuition.** LayerNorm is a scale that always reads "1 kg" regardless of what you put on it. BiasNorm puts a fixed counterweight on the pan. The reading is still normalized, but a heavier object now moves the needle.

**Mechanism and math.** BiasNorm drops mean subtraction, adds a learned bias inside the RMS, and makes the scale a single positive scalar:

$$
\text{BiasNorm}(x) = \frac{x}{\mathrm{RMS}[x - b]} \cdot \exp(\gamma)
$$

where $b \in \mathbb{R}^D$ is a learned per-channel bias, $\mathrm{RMS}$ is the root-mean-square over channels, and $\gamma$ is a scalar. The bias $b$ plays the role of the giant constant channel, but legitimately. Because $\exp(\gamma) > 0$, the scale can shrink but never cross zero, which removes the sign-flipping failure.

**Worked example.** Take $x = [1, 2, 3]$ and $3x = [3, 6, 9]$, the same direction at three times the length.

![Same direction, 3x the length. LayerNorm returns identical outputs; BiasNorm's bias b acts as a fixed reference so the longer input stays longer (output norm 0.70 vs 1.18)](/imgs/blogs/zipformer-2.webp)

LayerNorm maps both to $[-1.22, 0, 1.22]$. With $b = [0, 0, -6]$ and $\gamma = 0$, BiasNorm divides $x$ by $\mathrm{RMS}[1, 2, 9] = 5.35$ (output norm 0.70) and $3x$ by $\mathrm{RMS}[3, 6, 15] = 9.49$ (norm 1.18). With $b = 0$ it reduces to RMSNorm and the outputs match again: the bias carries the length.

**When it fails.** The gain is modest (LayerNorm costs 0.08 / 0.18 points), and how much length survives depends on the size of $b$ relative to $x$.

### SwooshR and SwooshL

Swish, $x \cdot \sigma(x)$, has a slope that goes to zero for large negative inputs. A unit stuck in that region gets almost no gradient, and Adam's denominator $\sqrt{v_t}$ for the weights feeding it can get dangerously small. The replacements are softplus curves with a linear term subtracted:

$$
\begin{aligned}
\text{SwooshR}(x) &= \log(1 + e^{x-1}) - 0.08x - 0.313261687 \\
\text{SwooshL}(x) &= \log(1 + e^{x-4}) - 0.08x - 0.035
\end{aligned}
$$

As $x \to -\infty$ the softplus term flattens, so the slope tends to $-0.08$ instead of 0. At $x = -10$, Swish's slope is about $-0.0004$; SwooshR's is $-0.08$, two hundred times larger. The constant in SwooshR makes it pass through the origin, since $\log(1 + e^{-1}) = 0.3133$.

![Figure A.3 from Yao et al. (2024): Swish, SwooshR and SwooshL. Both Swoosh curves keep a small negative slope on the left](/imgs/blogs/zipformer-fig4.webp)

SwooshL is SwooshR shifted right. Feed-forward and ConvNeXt modules learned a large negative bias to be "normally off", so SwooshL bakes that offset in for them; convolution modules keep SwooshR. Using only SwooshR costs 0.42 test-other points; Swish costs 0.58.

### ScaledAdam and Eden

**The problem.** Adam normalizes each element's gradient, so its step is roughly the learning rate $\alpha_t$ in absolute terms, whatever the parameter's scale: large for a tensor with RMS 0.01, tiny for one with RMS 1. Adam also struggles to change a tensor's overall scale, because "grow or shrink everything together" is one specific direction in a huge space, and noisy steps inflate the norm.

**Intuition.** Telling every holding in a portfolio "move one dollar" doubles the penny stock and leaves the blue chip untouched. Say "move 1% of your position" instead, and add a separate knob that resizes each position.

**Mechanism and math.** Start from Adam's update with gradient $g_t$, first and second moments $m_t, v_t$, decay rates $\beta_1, \beta_2$, and $\epsilon = 10^{-8}$:

$$
\Delta_t = -\alpha_t \cdot \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t} \cdot \frac{m_t}{\sqrt{v_t} + \epsilon}
$$

ScaledAdam multiplies it by the tensor's scale $r_{t-1} = \mathrm{RMS}[\theta_{t-1}]$:

$$
\Delta'_t = -\alpha_t \cdot r_{t-1} \cdot \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t} \cdot \frac{m_t}{\sqrt{v_t} + \epsilon}
$$

Now the relative change $\Delta'_t / r_{t-1}$ is about $\alpha_t$ for every tensor.

![With step size 0.001, Adam moves every parameter by about 0.001 in absolute terms, so a tensor with RMS 0.01 changes 10% per step and one with RMS 10 changes 0.01%. ScaledAdam multiplies the step by r, holding the relative change constant](/imgs/blogs/zipformer-3.webp)

To learn the scale itself, treat the parameter as $\theta = r \cdot \theta'$ and take a gradient step on $r$. Its gradient is the scalar $h_t = g_t \cdot \theta_{t-1}$, a dot product over the whole tensor that says whether growing the tensor would lower the loss. ScaledAdam keeps Adam-style moments $n_t, w_t$ of $h_t$ and adds a second update that rescales the whole tensor along its own direction:

$$
\Delta'_{t,r} = -\eta \cdot \alpha_t \cdot \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t} \cdot \frac{n_t}{\sqrt{w_t} + \epsilon} \odot \theta_{t-1}
$$

with $\eta = 0.1$. The parameter update is $\theta_t = \theta_{t-1} + \Delta'_t + \Delta'_{t,r}$. Here is a minimal single-tensor sketch of Algorithm 1 (the real icefall implementation batches tensors by shape and adds clipping):

```python
import torch

def scaled_adam_step(p, grad, state, lr, beta1=0.9, beta2=0.98, eta=0.1, eps=1e-8):
    t = state["step"] = state.get("step", 0) + 1
    if t == 1:
        for k in ("m", "v"):
            state[k] = torch.zeros_like(p)
        state["n"] = torch.zeros(())  # moments of the scalar scale-gradient
        state["w"] = torch.zeros(())
    bias_corr = (1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t)

    rms = p.pow(2).mean().sqrt()           # r_{t-1}: this tensor's scale
    h = (grad * p).sum()                   # gradient w.r.t. the scale

    state["m"].mul_(beta1).add_(grad, alpha=1 - beta1)
    state["v"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    delta = -lr * rms * bias_corr * state["m"] / (state["v"].sqrt() + eps)

    state["n"].mul_(beta1).add_(h, alpha=1 - beta1)
    state["w"].mul_(beta2).add_(h * h, alpha=1 - beta2)
    delta_r = -eta * lr * bias_corr * state["n"] / (state["w"].sqrt() + eps) * p

    p.add_(delta + delta_r)
```

**Worked example.** With peak $\alpha_{\text{base}} = 0.045$ and $\eta = 0.1$, the scale term resizes a tensor by up to $0.1 \times 0.045 = 0.45\%$ per step, so a consistent "shrink" signal halves it in about $\ln 2 / 0.0045 \approx 154$ steps. That is why the block needs no per-module LayerNorm: each module learns its own output scale. In the ablation, Adam only trains Zipformer if BiasNorm is put back into every module.

**Eden.** The schedule decays with both step $t$ and epoch $e$:

$$
\alpha_t = \alpha_{\text{base}} \left(\frac{t^2 + \alpha_{\text{step}}^2}{\alpha_{\text{step}}^2}\right)^{-0.25} \left(\frac{e^2 + \alpha_{\text{epoch}}^2}{\alpha_{\text{epoch}}^2}\right)^{-0.25} \cdot \text{linear}(\alpha_{\text{start}}, t_{\text{warmup}}, t)
$$

Each factor is about 1 early and decays like $t^{-0.5}$ past its knee (at $t = \alpha_{\text{step}}$ it is $2^{-0.25} = 0.84$). Warmup is only 500 steps because scale-relative updates diverge less, and the epoch term keeps change per hour of audio stable across batch sizes. The paper does not give the two knee values.

**When it fails.** For a tensor near zero, $r \approx 0$ and the main update stalls, leaving only the scale term to grow it. Biases and scalars are the obvious edge cases, and the paper does not say how it treats them.

## Experiments & results

Zipformer uses a pruned transducer loss, SpecAugment, beam size 4 and no external LM; LibriSpeech models train 50 epochs on 2 to 4 V100s. FLOPs are encoder-only on 30 s of audio (Table 2).

| Model | Type | Params (M) | GFLOPs | test-clean | test-other |
| --- | --- | --- | --- | --- | --- |
| Squeezeformer-L | CTC | 236.3 | 333.7 | 2.47 | 5.97 |
| E-Branchformer-L | CTC/AED | 148.9 | 284.4 | 2.14 | 4.55 |
| Conformer-L (original paper) | transducer | 118.8 | n/a | 2.1 | 4.3 |
| Conformer-M (their repro) | pruned transducer | 28.4 | 77.0 | 2.96 | 7.11 |
| Conformer-L (their repro) | pruned transducer | 122.5 | 294.2 | 2.46 | 5.55 |
| Zipformer-S | pruned transducer | 23.3 | 40.8 | 2.42 | 5.73 |
| Zipformer-M | pruned transducer | 65.6 | 62.9 | 2.21 | 4.79 |
| Zipformer-L | pruned transducer | 148.4 | 107.7 | 2.06 | 4.63 |
| Zipformer-L (170 ep, 8 A100) | pruned transducer | 148.4 | 107.7 | 2.00 | 4.38 |

On Aishell-1, Zipformer-L reaches 4.28 % test CER against 4.43 % for Branchformer. On WenetSpeech it reaches 7.24 / 12.06 % on Test Net / Test Meeting, beating a 425 M-parameter Conformer-MoE (8.28 / 13.96 %).

![Figure 3 from Yao et al. (2024): average LibriSpeech WER against encoder inference time (left) and peak inference memory (right), batches of 30 s audio on one V100](/imgs/blogs/zipformer-fig3.webp)

Zipformer sits on the lower-left frontier for both time and memory, widest at the large size, where the paper reports over 50% faster inference. The ablation (Table 5, Zipformer-M) is the most interesting part:

![Test-other WER increase when one component is removed from Zipformer-M (baseline 4.79%). The single largest regression is swapping the optimizer, not the architecture](/imgs/blogs/zipformer-4.webp)

Three of the four largest regressions are training changes: Adam (+0.72), Swish (+0.58) and only-SwooshR (+0.42). The best architectural item, attention sharing plus NLA, is worth +0.31.

**What might not transfer.** The recipe leans on pruned-transducer training, icefall's Balancer and Whitener activation constraints (removing the Whitener alone costs 0.24 test-other, Appendix A.3), and 50-epoch schedules; another toolkit would need to rebuild those. FLOPs come from DeepSpeed's profiler, so the measured latencies in Figure 3 are the safer evidence.

## Critique

**Strong.** The ablation tests modeling and optimization on equal footing, with a learning-rate sweep for both optimizers. Efficiency is measured, and the code is public.

**Weak.** Sizes are matched on FLOPs, not parameters: Zipformer-M has 2.3 times Conformer-M's parameters (65.6 M vs 28.4 M). The Adam baseline needed BiasNorm in every module, so the optimizer comparison is not perfectly clean. Single runs and 0.07 to 0.18 point gaps sit near seed noise. Balancer and Whitener are, by the authors' account, placed where specific failures appeared, which is honest but hard to reproduce. Streaming is not evaluated.

**Missing ablation.** The reverse experiment: a Conformer trained with ScaledAdam, BiasNorm and Swoosh. That is the clean way to separate "better encoder" from "better training recipe".

**What would change my mind:** if a Conformer trained with ScaledAdam, BiasNorm and SwooshR/L closed most of the gap to Zipformer at equal FLOPs, I would read this paper mainly as an optimizer paper with an efficient architecture attached. The ablation numbers make that plausible.

## What I'd build with this

These are my extrapolations, not claims from the paper.

1. **ScaledAdam outside speech**: a small Transformer LM with its norm layers removed.
2. **The reverse ablation** above, at matched FLOPs and three seeds.
3. **WER against chunk size for streaming.** A 6.25 Hz frame spans 160 ms of audio, a hard latency floor.

## References

- Yao et al., *Zipformer: A Faster and Better Encoder for Automatic Speech Recognition*, ICLR 2024. [arXiv:2310.11230](https://arxiv.org/abs/2310.11230)
- Code: [k2-fsa/icefall](https://github.com/k2-fsa/icefall)
- Gulati et al., *Conformer: Convolution-augmented Transformer for Speech Recognition*, Interspeech 2020.
- On this blog: [CTC vs RNN-T vs attention encoder-decoder](/blog/machine-learning/speech-to-text/ctc-vs-rnnt-vs-attention-asr-alignment), [Canary, Parakeet and FastConformer](/blog/machine-learning/speech-to-text/canary-parakeet-fastconformer-asr), [Building a modern ASR system end to end](/blog/machine-learning/speech-to-text/modern-asr-system-end-to-end), [Training ASR models](/blog/machine-learning/speech-to-text/training-asr-models)
