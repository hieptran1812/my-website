---
title: "Residual Vector Quantization: How Codecs Squeeze Audio Into Tokens"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Build the quantizer at the heart of every neural audio codec from scratch — vector quantization, the straight-through gradient, codebook collapse and its fixes, and the residual VQ stack that turns a waveform into a coarse-to-fine token hierarchy — with runnable PyTorch and a worked bitrate budget."
tags:
  [
    "audio-generation",
    "audio-synthesis",
    "neural-audio-codec",
    "residual-vector-quantization",
    "vector-quantization",
    "soundstream",
    "rate-distortion",
    "generative-ai",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Audio Generation"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/residual-vector-quantization-rvq-1.png"
---

The first RVQ I trained looked like it was working. The reconstruction loss went down, the spectrograms looked plausible, and the decoded audio was clearly recognizable. Then I printed the codebook usage. I had asked for a codebook with 1024 entries; 47 of them were doing all the work. The other 977 had been initialized once, never selected, and left to rot — pure dead weight in a tensor I was paying gradient and memory for, and a brutal cap on how much information each token could carry. The model had quietly collapsed onto a tiny corner of its own codebook, and nothing in the loss curve told me. That bug — and the fixes for it — is half of what this post is about. The other half is the idea that makes neural audio codecs actually work at high fidelity: instead of fighting to make one enormous codebook behave, you stack many small ones, each cleaning up the leftovers of the last.

This is the quantization machinery at the heart of every neural audio codec — SoundStream, EnCodec, DAC, Mimi — and therefore at the heart of nearly every modern audio generation system. A [neural audio codec](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound) is the audio analogue of the image VAE: an encoder squeezes a waveform into a compact latent, a quantizer turns that latent into a short sequence of discrete tokens, and a decoder turns the tokens back into a waveform. The tokens are what an autoregressive language model or a diffusion model actually generates. Everything in the [audio stack](/blog/machine-learning/audio-generation/why-audio-generation-is-hard) — waveform to codec tokens to generative model to decoder back to waveform — flows through this quantizer, and the quantizer's design sets the fundamental trade between fidelity, bitrate, and how hard the downstream model has to work.

![A vertical stack figure showing a 128-dimensional encoder latent feeding into codebook one, the residual feeding codebook two, and further residuals feeding later codebooks, with total bits growing as N times ten per frame.](/imgs/blogs/residual-vector-quantization-rvq-1.png)

By the end of this post you will be able to: explain exactly why the nearest-codebook lookup blocks gradients and how the straight-through estimator routes around it; write a minimal VQ layer with an exponential-moving-average codebook update in PyTorch and stack it into a residual VQ; diagnose codebook collapse from usage and perplexity numbers and fix it with k-means init, EMA, random restarts, and expiry; compute the bitrate of any RVQ configuration in your head (frames per second times number of codebooks times log-two of the codebook size); read a rate-distortion curve and know where the diminishing returns kick in; and understand why generative models predict the coarse codebook first. This is SoundStream's core contribution (Zeghidour et al., 2021), and it is the load-bearing idea under EnCodec, DAC, and the codec language models that consume their tokens. The discrete-token-from-a-codebook idea will be familiar if you have read the image series on [autoregressive image models](/blog/machine-learning/image-generation/autoregressive-image-models) and the [VQ-VAE / VQ-GAN](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) tokenizers — this is the same family of tricks, specialized to a one-dimensional signal at very high sample rate where one codebook is nowhere near enough.

Throughout I will keep returning to one concrete target: a 24 kHz mono recording that we want to compress to **6 kbps** — about 250 times smaller than the raw PCM — and reconstruct so well that a listener struggles to tell it from the original. That single number, 6 kbps, is roughly where SoundStream and EnCodec aimed, and chasing it forces every design decision we are about to make.

## 1. Why one continuous latent is not yet a token

Start with what the encoder hands us. A codec encoder is a stack of strided 1D convolutions that takes a waveform — say 24,000 samples per second — and downsamples it heavily. SoundStream uses a total stride of 320, so 24 kHz audio comes out at 24000 / 320 = **75 frames per second**. At each of those 75 frames per second the encoder emits a continuous vector — call it $z \in \mathbb{R}^D$, with $D$ maybe 128 or 256. So one second of audio is now 75 vectors of dimension 128. That is already a big compression of the million-ish numbers in a second of raw audio, but it is still *continuous*: each of those 128 numbers is a 32-bit float. We have not made tokens yet.

A token is a discrete symbol from a finite vocabulary — an integer index. To get tokens we need to replace each continuous vector $z$ with one of a finite set of allowed vectors, and emit the *index* of the chosen one. That finite set is the **codebook**: a learned matrix $C \in \mathbb{R}^{K \times D}$ of $K$ entries, each a $D$-dimensional vector. Quantization is the map

$$
q(z) = \arg\min_{k \in \{1, \dots, K\}} \lVert z - C_k \rVert_2^2,
$$

the index of the nearest codebook entry to $z$ under Euclidean distance. The token we emit is that integer $k$; the vector the decoder sees is $C_k$, the quantized stand-in for $z$, which I will write $z_q = C_{q(z)}$. This is **vector quantization** (VQ): not quantizing each scalar independently, but snapping the whole vector to its nearest neighbor in a learned dictionary. It is the same operation as a single step of k-means assignment, except the centroids are learned by gradient descent (mostly) and shared across the whole dataset.

Why is this the right move for audio? Two reasons. First, discreteness is what makes the downstream generative model tractable. An autoregressive language model predicts a categorical distribution over a vocabulary; it cannot directly emit a 128-dimensional float vector, but it can happily emit one of $K$ token indices. Discretizing the latent turns "generate audio" into "predict the next token," which is exactly the problem transformers are good at — and it lets the entire toolbox built for text language models (cross-entropy training, sampling temperatures, beam search, KV-caching, in-context prompting) transfer directly to audio. The discreteness is not a compromise; it is what unlocks the whole language-model paradigm for sound. Second, the codebook is a strong prior: by forcing every frame onto one of $K$ learned vectors, you bake the statistics of real audio into the representation. The encoder cannot emit nonsense; it can only emit combinations of vectors the codebook has learned are useful. A diffusion or flow model *can* work on the continuous latent directly and many do (we will draw that line sharply at the end) — but if you want a language model over audio, you need discrete tokens, and that means a quantizer.

There is a subtlety in "the index is the token" worth stating outright, because it trips people coming from text. In a text tokenizer the token *is* the unit of meaning — "cat" is a token, and its embedding is looked up from a table. In an audio codec the token is an *index into a learned codebook*, and the thing that carries meaning is the codebook *vector* it points to. So when a language model predicts an audio token, it is predicting "use codebook entry 412," and the decoder reconstructs the audio from the *vector* at row 412, not from the integer 412 itself. The integer is just a name; the vector is the content. This matters because it means the codebook is shared, learned, and finite — two different audio frames that happen to be acoustically similar get the same token, which is precisely the compression we want.

The cost of all this is information. A single codebook of size $K$ can encode exactly $\log_2 K$ bits per frame, because choosing one of $K$ options is $\log_2 K$ bits of choice. With $K = 1024$ that is 10 bits per frame. At 75 frames per second that is 750 bits per second — **0.75 kbps**. That is wildly too little for high-fidelity audio. To hit our 6 kbps target with a single codebook we would need $K = 2^{80}$ entries, which is absurd: you cannot store, train, or search a codebook with $10^{24}$ rows, and even if you could, almost all of it would be dead. This is the **bitrate wall** of plain VQ, and the entire reason residual VQ exists. We will get there in section 5; first we have to make a single VQ layer actually train, because the lookup we just wrote has a fatal property: it has no gradient.

It helps to put the compression in human terms. Raw 24 kHz audio at 16 bits per sample is $24000 \times 16 = 384{,}000$ bits per second — 384 kbps, or about 2.8 megabytes per minute. MP3 at "transparent" quality is ~128–256 kbps. The Opus codec, the current champion of classical (hand-designed) speech-and-music coding, gets usable speech down to ~6–12 kbps and good music around 32–64 kbps. Our target of 6 kbps for *general* high-fidelity audio is below what any classical codec achieves at that quality — that gap is exactly what neural codecs opened, and RVQ is the mechanism that let them do it. So when I say "6 kbps reconstruction a listener struggles to distinguish from the original," understand that this was, before SoundStream, considered roughly impossible for music. The whole point of learning the encoder, decoder, *and* the quantizer jointly with a neural network and adversarial losses is that the network discovers a representation far more compact than any signal-processing engineer could hand-design — and RVQ is the part of that system that turns the compact representation into the discrete tokens a generative model can predict.

One more orienting fact before we make this trainable: the *number of tokens* matters as much as the *bits per token* for everything downstream. At 75 fps with 8 codebooks, one second of audio is $75 \times 8 = 600$ tokens. A 30-second clip is 18,000 tokens — already a long sequence for a transformer. This is why frame rate is a first-class design choice and why a codec like Mimi pushes it all the way down to 12.5 fps: not to save bits (it uses a bigger codebook to compensate) but to make the *token sequence short enough* that a real-time language model can keep up. Hold that tension — bits per frame versus frames per second versus codebooks per frame — in mind; it is the budget every codec spends differently.

## 2. The non-differentiable argmin and the straight-through estimator

Look at the quantization map again: $z_q = C_{\arg\min_k \lVert z - C_k \rVert^2}$. The $\arg\min$ is a discrete selection. Nudge $z$ by an infinitesimal amount and, almost everywhere, the nearest codebook entry does not change at all — so $\partial z_q / \partial z = 0$ — and at the boundaries where it does change, it jumps discontinuously. Either way the gradient is useless: zero almost everywhere, undefined on a measure-zero set. If we backpropagate through the quantizer honestly, no gradient reaches the encoder, and the encoder never learns. The decoder learns to use whatever codebook vectors it is fed, but the encoder that produces $z$ gets a flat zero signal and stays at its initialization forever.

The fix, introduced for exactly this problem in the VQ-VAE paper (van den Oord et al., 2017), is the **straight-through estimator** (STE). The idea is brutally simple: on the forward pass, quantize honestly ($z_q = C_{q(z)}$); on the backward pass, *pretend the quantizer was the identity function* and copy the gradient from $z_q$ straight onto $z$. Formally, we define

$$
z_q = z + \text{sg}(C_{q(z)} - z),
$$

where $\text{sg}(\cdot)$ is the stop-gradient operator (identity on the forward pass, zero gradient on the backward pass). On the forward pass the two $z$ terms cancel inside the stop-gradient and you get exactly $z_q = C_{q(z)}$. On the backward pass, the stop-gradient term contributes nothing, so $\partial z_q / \partial z = 1$ — the decoder's gradient flows through the quantizer to the encoder unchanged, as if the codebook had not been there. That is the straight-through gradient identity: $\nabla_z \mathcal{L} \approx \nabla_{z_q} \mathcal{L}$.

![A dataflow graph showing the encoder producing a continuous latent, an argmin lookup selecting the nearest entry, a stop-gradient combine node where the gradient is copied straight past the lookup, the decoder, and the reconstruction-plus-commitment loss.](/imgs/blogs/residual-vector-quantization-rvq-2.png)

This is an *estimator*, not the true gradient — it is biased, because we are lying about what the quantizer did. But it works astonishingly well in practice, for the same reason it works for image VQ-VAEs: as long as $z_q$ is close to $z$ (which the codebook and commitment losses below enforce), the identity is a good local approximation, and the bias is small enough that training converges to a useful codebook. There are unbiased alternatives — Gumbel-softmax relaxation, which makes the selection a soft, temperature-annealed mixture you can differentiate honestly, and rotation-trick variants that preserve more gradient structure — but for audio codecs the plain straight-through estimator is what SoundStream, EnCodec, and DAC all use, and it is what you should reach for first.

Why does the bias not accumulate into something catastrophic? Two reasons. First, the error you make by pretending the quantizer is the identity is exactly $z_q - z$, the quantization residual, and the commitment loss actively drives that residual toward zero — so the lie gets smaller as training proceeds. Second, the gradient the encoder actually needs from the decoder is a *direction to move $z$ in*, and that direction is almost always still correct even though $z_q \ne z$: if the decoder wants more energy at 4 kHz, it wants the encoder to produce more 4-kHz energy regardless of which codebook cell $z$ currently lands in. The straight-through estimator gets the direction right and only the magnitude slightly wrong, and slightly-wrong magnitudes are exactly what an optimizer with a learning rate is built to tolerate. This is the same reason the trick works for the [VQ-VAE/VQ-GAN image tokenizers](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) — the geometry of "which way should the encoder move" survives the quantization boundary even when the exact value does not.

The STE only handles the encoder. The codebook itself still needs a learning signal — the argmin tells you *which* entry was chosen but gives no gradient to *move* that entry toward the data. That is what the two auxiliary losses do.

### The codebook and commitment losses

The full VQ-VAE training objective has three terms:

$$
\mathcal{L} = \underbrace{\lVert x - \hat{x} \rVert^2}_{\text{reconstruction}} \;+\; \underbrace{\lVert \text{sg}(z) - C_{q(z)} \rVert^2}_{\text{codebook}} \;+\; \beta \underbrace{\lVert z - \text{sg}(C_{q(z)}) \rVert^2}_{\text{commitment}}.
$$

The reconstruction term trains the encoder and decoder to make $\hat{x}$ match the input $x$ (for an audio codec this is not a bare L2 on the waveform but a multi-resolution STFT loss plus adversarial losses — covered in the [EnCodec and DAC](/blog/machine-learning/audio-generation/encodec-dac-and-the-modern-codec) post — but the structure is the same).

The **codebook loss** $\lVert \text{sg}(z) - C_{q(z)} \rVert^2$ pulls the *chosen codebook entry* toward the encoder output, with the encoder held fixed (stop-gradient on $z$). This is what trains the codebook: each selected entry slides toward the average of the latents assigned to it — gradient-descent k-means, essentially.

The **commitment loss** $\beta \lVert z - \text{sg}(C_{q(z)}) \rVert^2$ pulls the *encoder output* toward its chosen entry, with the codebook held fixed (stop-gradient on $C$). Without it the encoder is free to let $z$ drift arbitrarily far from any codebook vector — the encoder's output volume grows unboundedly, the codebook can never catch up, and training diverges. The commitment term "commits" the encoder to actually living near the codebook it is being quantized against. The weight $\beta$ trades these off; $\beta = 0.25$ is the canonical default from the VQ-VAE paper and a perfectly good starting point.

There is a subtlety worth internalizing: the codebook loss and the straight-through estimator are doing different jobs. The STE moves gradient *through* the quantizer so the encoder can learn from reconstruction. The codebook loss moves the *codebook entries* toward the encoder. The commitment loss keeps the encoder from running away. Three mechanisms, three jobs, and you need all three or VQ training falls apart in one of three characteristic ways: dead encoder (no STE), drifting encoder (no commitment), or static codebook (no codebook loss).

## 3. A minimal VQ layer in PyTorch

Let us make this concrete. Here is a minimal but real vector-quantization layer with the straight-through estimator and both auxiliary losses, written the way you would actually drop it into a codec. The only non-obvious line is the straight-through trick itself.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """A single VQ codebook with straight-through gradient and the
    codebook + commitment losses (the loss-based variant)."""

    def __init__(self, num_entries: int, dim: int, beta: float = 0.25):
        super().__init__()
        self.num_entries = num_entries
        self.dim = dim
        self.beta = beta
        self.codebook = nn.Embedding(num_entries, dim)
        # Uniform init in [-1/K, 1/K] is the VQ-VAE default.
        self.codebook.weight.data.uniform_(-1.0 / num_entries, 1.0 / num_entries)

    def forward(self, z):
        # z: (batch, dim, frames)  ->  (batch, frames, dim) for lookup
        z_e = z.transpose(1, 2).contiguous()
        flat = z_e.reshape(-1, self.dim)                     # (B*T, dim)

        # Squared Euclidean distance to every codebook entry.
        # ||a-b||^2 = ||a||^2 - 2 a.b + ||b||^2
        cb = self.codebook.weight                            # (K, dim)
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ cb.t()
            + cb.pow(2).sum(1)
        )                                                    # (B*T, K)
        indices = dist.argmin(1)                             # (B*T,)  the TOKENS
        z_q = self.codebook(indices).view_as(z_e)           # nearest entries

        # Codebook loss (move entries to z) + commitment loss (move z to entries).
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commit_loss = F.mse_loss(z_e, z_q.detach())
        loss = codebook_loss + self.beta * commit_loss

        # Straight-through: forward = z_q, backward gradient = identity to z_e.
        z_q = z_e + (z_q - z_e).detach()

        z_q = z_q.transpose(1, 2).contiguous()              # back to (B, dim, T)
        return z_q, indices.view(z.size(0), -1), loss
```

The line `z_q = z_e + (z_q - z_e).detach()` is the entire straight-through estimator. The forward value is `z_q` (the `z_e` terms cancel); the backward gradient is the gradient of `z_e` (because the `.detach()` zeroes the gradient of the bracketed difference). Everything else is bookkeeping: reshape to put the channel dimension last for the distance computation, expand the squared-distance with the $\lVert a - b \rVert^2 = \lVert a \rVert^2 - 2 a \cdot b + \lVert b \rVert^2$ identity (cheaper than materializing all the differences), take the argmin to get the tokens, and gather the chosen entries.

That `argmin(1)` returns one integer per frame per batch element — those integers are your codec tokens, the things a language model will eventually predict. The returned `loss` gets added to your reconstruction loss and the whole thing trains end to end.

### Worked example: the bitrate of one codebook

Take the configuration in that code with `num_entries = 1024` running on the 75-fps encoder. Each frame emits one index in $\{0, \dots, 1023\}$, which is $\log_2 1024 = 10$ bits. The bitrate is

$$
75 \ \text{frames/s} \times 1 \ \text{codebook} \times 10 \ \text{bits} = 750 \ \text{bits/s} = 0.75 \ \text{kbps}.
$$

For comparison, the raw 24 kHz 16-bit PCM input is $24000 \times 16 = 384$ kbps. So one codebook already gives a 512x compression — but the reconstruction at 0.75 kbps is muffled and metallic, nowhere near our 6 kbps fidelity target. We are off by a factor of 8 in bitrate, and the only knob a single VQ layer gives us is $K$, which we have already seen scales hopelessly. We need a different axis. But before we add codebooks, we have to deal with the bug from the opening paragraph, because it will sabotage every codebook we add.

## 4. Codebook collapse and the four fixes

**Codebook collapse** is the failure where most codebook entries are never selected. It is the single most common VQ training pathology, and on audio it is vicious because audio latents are highly non-uniform — a few frames (silence, sustained tones) dominate, and the codebook happily collapses onto them. When 977 of 1024 entries are dead, you are not getting 10 bits per frame; you are getting $\log_2 47 \approx 5.5$ bits of *effective* capacity, and you are paying full bitrate for it. The token vocabulary the downstream model sees is mostly junk it never observes.

The right way to *measure* collapse is **codebook perplexity**, the exponential of the entropy of the entry-usage distribution over a batch:

$$
\text{perplexity} = \exp\!\Big(-\sum_{k=1}^{K} p_k \log p_k\Big), \qquad p_k = \frac{\text{count of frames assigned to } k}{\text{total frames}}.
$$

If all $K$ entries are used equally, perplexity equals $K$ (maximum) — every token is informative. If only one entry is ever used, perplexity is 1. So perplexity is the *effective vocabulary size*. A healthy 1024-entry codebook on audio should sit in the high hundreds; if you print perplexity and see 47, you have collapse. Always log it — it is the diagnostic the loss curve hides.

![A matrix figure with rows for naive VQ, EMA update, k-means init, random restart, and codebook expiry, and columns describing what each does and its effect on codebook usage from many-dead to near-full utilization.](/imgs/blogs/residual-vector-quantization-rvq-6.png)

There are four standard fixes, and production codecs use all of them together.

**1. EMA codebook updates.** Instead of learning the codebook by gradient descent on the codebook loss, update each entry as an exponential moving average of the encoder vectors assigned to it. This is the variant SoundStream and EnCodec actually use, and it is far more stable than the loss-based update because it is exactly online k-means rather than a noisy gradient approximation of it. For each entry $k$, maintain a running count $N_k$ and a running sum $m_k$ of assigned vectors, both decayed by $\gamma$ (typically 0.99) each step:

$$
N_k \leftarrow \gamma N_k + (1-\gamma)\, n_k, \qquad m_k \leftarrow \gamma m_k + (1-\gamma) \sum_{i : q(z_i)=k} z_i, \qquad C_k \leftarrow \frac{m_k}{N_k},
$$

where $n_k$ is how many vectors in the current batch chose entry $k$. With the EMA variant you drop the codebook loss term entirely (the EMA *is* the codebook update); you keep only the commitment loss to anchor the encoder.

Why is the EMA more stable than just minimizing the codebook loss with the optimizer? Because the codebook-loss gradient for an entry is proportional to how *often* that entry is chosen — a popular entry gets a strong gradient and moves fast, a rarely-chosen entry gets a weak gradient and barely moves, which *accelerates* collapse (the rich get richer). The EMA, by dividing the running sum by the running count $C_k = m_k / N_k$, always places each entry at the *centroid* of its assigned vectors regardless of how popular it is, so a rarely-used entry still sits sensibly in the middle of the few points that chose it rather than drifting away. The EMA also sidesteps the interaction between the codebook's learning rate and the encoder's — with one optimizer over both, a learning rate good for the encoder is often wrong for the codebook. Decoupling the codebook update from the optimizer is most of why EMA codecs train more reliably.

**A note on $\gamma$.** The decay $\gamma$ sets how fast the codebook tracks the encoder. Too high (0.999) and the codebook lags the encoder badly early in training, when the encoder is changing fast; too low (0.9) and the codebook is noisy and chases minibatch fluctuations. $\gamma = 0.99$ is the robust default and is what EnCodec ships; only touch it if you see the codebook visibly trailing or jittering relative to the encoder outputs.

**2. K-means initialization.** Initialize the codebook from a k-means clustering of the first batch's encoder outputs instead of from random uniform noise. A randomly initialized codebook in $[-1/K, 1/K]$ has tiny entries clustered near the origin, and the encoder's outputs may land far from all of them, so a few nearest entries win everything from step one. Seeding the codebook *with actual data* gives every entry a fighting chance and dramatically reduces early collapse. DAC and most modern codecs do this.

**3. Random restarts (codebook reset).** Periodically, find entries whose usage has dropped to (near) zero and *reseed* them — copy a randomly chosen high-usage encoder vector from the current batch into the dead slot, plus a little noise. This is "expire-and-resurrect": a dead entry gets a second life near where the data actually is, instead of staying stranded.

**4. Codebook expiry.** A cleaner formalization of the restart: track how many batches each entry has gone unused, and once an entry has been idle for more than a threshold (say 2 epochs), evict it and replace it with a live latent. EnCodec uses exactly this — entries unused for a configured number of batches are replaced by random candidate vectors from the current batch.

Here is the EMA variant with expiry, which is what you actually want in a codec:

```python
class EMAVectorQuantizer(nn.Module):
    """EMA codebook update + dead-entry expiry (the codec-grade variant)."""

    def __init__(self, num_entries, dim, beta=0.25, gamma=0.99,
                 expire_after=2.0, eps=1e-5):
        super().__init__()
        self.num_entries, self.dim = num_entries, dim
        self.beta, self.gamma, self.eps = beta, gamma, eps
        self.expire_after = expire_after  # threshold on the EMA usage count
        self.register_buffer("codebook", torch.randn(num_entries, dim))
        self.register_buffer("ema_count", torch.zeros(num_entries))
        self.register_buffer("ema_sum", self.codebook.clone())

    def forward(self, z):
        z_e = z.transpose(1, 2).contiguous()
        flat = z_e.reshape(-1, self.dim)
        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.codebook.t()
                + self.codebook.pow(2).sum(1))
        idx = dist.argmin(1)
        onehot = F.one_hot(idx, self.num_entries).type(flat.dtype)  # (N, K)
        z_q = F.embedding(idx, self.codebook).view_as(z_e)

        if self.training:
            n = onehot.sum(0)                      # usage count this batch
            cluster_sum = onehot.t() @ flat        # (K, dim) sum of assigned z
            self.ema_count.mul_(self.gamma).add_(n, alpha=1 - self.gamma)
            self.ema_sum.mul_(self.gamma).add_(cluster_sum, alpha=1 - self.gamma)
            # Laplace-smooth the counts so empty clusters do not divide by zero.
            total = self.ema_count.sum()
            counts = ((self.ema_count + self.eps)
                      / (total + self.num_entries * self.eps) * total)
            self.codebook.copy_(self.ema_sum / counts.unsqueeze(1))
            # Expire entries whose EMA count fell below the threshold.
            dead = self.ema_count < self.expire_after
            if dead.any():
                live = flat[torch.randint(0, flat.size(0), (int(dead.sum()),))]
                self.codebook[dead] = live
                self.ema_count[dead] = 1.0

        commit_loss = self.beta * F.mse_loss(z_e, z_q.detach())
        z_q = z_e + (z_q - z_e).detach()           # straight-through
        # Perplexity = effective codebook size (log this!).
        p = onehot.mean(0)
        perplexity = torch.exp(-(p * (p + 1e-10).log()).sum())
        z_q = z_q.transpose(1, 2).contiguous()
        return z_q, idx.view(z.size(0), -1), commit_loss, perplexity
```

Note what changed: no codebook loss in the returned loss (the EMA update replaces it), the codebook lives in a buffer and is updated in-place rather than by the optimizer, dead entries are reseeded from live latents every step, and we return perplexity so you can watch collapse. The Laplace smoothing on the counts is the standard trick to keep empty clusters from producing NaNs. This single layer, used well, is the building block. Now we stack it.

## 5. Residual VQ: a stack of codebooks

Here is the idea that breaks the bitrate wall. One codebook of size $K$ gives $\log_2 K$ bits and snaps $z$ to its nearest entry $C^{(1)}_{q_1(z)}$. That entry is close to $z$ but not equal — there is an error, the **residual** $r_1 = z - C^{(1)}_{q_1(z)}$. Plain VQ throws this residual away. Residual VQ *quantizes it too*, with a second, independent codebook. Then it takes the new residual and quantizes that with a third codebook, and so on through $N$ codebooks. The recursion is:

$$
\begin{aligned}
r_0 &= z, \\
r_i &= r_{i-1} - C^{(i)}_{q_i(r_{i-1})}, \qquad i = 1, \dots, N, \\
\hat{z} &= \sum_{i=1}^{N} C^{(i)}_{q_i(r_{i-1})}.
\end{aligned}
$$

Each codebook quantizes what the previous ones could not represent. The reconstruction $\hat{z}$ is the *sum* of one chosen entry from each codebook, and the token for frame $t$ is now a tuple of $N$ indices $(q_1, q_2, \dots, q_N)$ — one stream per codebook. This is **residual vector quantization**, the core contribution of SoundStream (Zeghidour et al., 2021), borrowed from classical multi-stage vector quantization in speech coding and made to train end to end inside a neural codec.

It is worth pausing on *why* this is so much better than one big codebook, because the answer is geometric and it is the crux of the whole technique. A single codebook of size $K$ tiles the latent space $\mathbb{R}^D$ into $K$ Voronoi cells — regions, each owned by one entry, where every point in the region snaps to that entry. To make the quantization error small you need the cells small, which means you need *many* of them, and the number you need grows exponentially with $D$ (this is the curse of dimensionality biting the quantizer: to halve the cell diameter in $D$ dimensions you need $2^D$ times as many cells). For $D = 128$ that is hopeless — no codebook you can build comes close to tiling the space finely. Residual VQ escapes this by tiling *hierarchically*: codebook 1 lays down a coarse tiling, and then codebook 2 lays a second coarse tiling *over each residual cell*, and codebook 3 a third over those, so the effective tiling is the product of $N$ coarse tilings — $K^N$ cells from $N \times K$ entries. You get exponentially many effective cells from a linear number of stored vectors, because the cells are defined by *combinations* of codebook entries, not by individual entries. That product structure is the entire reason RVQ works where flat VQ cannot.

There is one more thing the recursion buys you that a flat codebook never could: an **ordering**. In the recursion above, codebook 1 always quantizes the full latent $r_0 = z$, codebook 2 always quantizes $r_1$, and so on, so codebook 1 is *by construction* operating on the highest-energy signal and codebook $N$ on the smallest residual. The codebooks are not interchangeable — they form a strict coarse-to-fine sequence. We will lean on this ordering twice: in section 7 to drop the tail of the stack for a lower bitrate, and in section 8 to let a generative model predict the coarse codebook first. A flat codebook has no such ordering — every entry is peer to every other — which is precisely why a flat tokenizer cannot give you graceful bitrate scaling or a natural prediction order.

The bitrate is now beautifully linear in the number of codebooks. With $N$ codebooks each of size $K$, every frame carries $N \log_2 K$ bits, so

$$
\text{bitrate} = (\text{frames/s}) \times N \times \log_2 K.
$$

This is the single most important formula in the post. Each codebook stays small and stable (k-means-sized, $K = 1024$), but stacking them multiplies your effective vocabulary to $K^N$ *combinations* — without ever building a codebook with $K^N$ rows. With $K = 1024$ and $N = 8$ you get $1024^8 = 2^{80}$ effective combinations from eight tiny, trainable codebooks. That is the trick: residual VQ gives you the capacity of an astronomically large codebook with the stability of a small one.

![A before-and-after figure contrasting a single giant unstable codebook that needs an impossible number of entries and suffers collapse against a stack of small stable well-used codebooks that reach the same bits and support coarse-to-fine scaling.](/imgs/blogs/residual-vector-quantization-rvq-3.png)

### Worked example: the 6 kbps budget

Now we can hit our target exactly. Take the 75-fps encoder, $K = 1024$ (so 10 bits per codebook), and ask how many codebooks we need for 6 kbps:

$$
6000 \ \text{bits/s} = 75 \ \text{frames/s} \times N \times 10 \ \text{bits} \implies N = \frac{6000}{750} = 8 \ \text{codebooks}.
$$

So **75 fps x 8 codebooks x 10 bits = 6 kbps** — eight stacked 1024-entry codebooks, each adding 0.75 kbps, get us precisely to the SoundStream/EnCodec target. Want 3 kbps for a bandwidth-constrained deployment? Use $N = 4$. Want 12 kbps for near-transparent quality? Use $N = 16$. The bitrate is a dial you turn by choosing $N$, and — crucially — you can turn it *at inference time without retraining*, which we will get to in section 7.

Here is the RVQ stack, built directly on the EMA layer from section 4. The whole thing is a loop over codebooks operating on the running residual.

```python
class ResidualVQ(nn.Module):
    """N stacked VQ codebooks over a running residual (SoundStream-style)."""

    def __init__(self, num_quantizers, num_entries, dim, **vq_kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [EMAVectorQuantizer(num_entries, dim, **vq_kwargs)
             for _ in range(num_quantizers)]
        )

    def forward(self, z, n_active=None):
        # n_active lets us DROP later codebooks at inference (bitrate scaling).
        layers = self.layers[: n_active] if n_active else self.layers
        residual = z
        z_q = torch.zeros_like(z)
        all_indices, total_commit, perplexities = [], 0.0, []
        for vq in layers:
            q, idx, commit, ppl = vq(residual)
            residual = residual - q          # the RVQ recursion: peel off q
            z_q = z_q + q                     # accumulate the reconstruction
            all_indices.append(idx)
            total_commit = total_commit + commit
            perplexities.append(ppl)
        # tokens: (batch, num_quantizers, frames)
        codes = torch.stack(all_indices, dim=1)
        return z_q, codes, total_commit, torch.stack(perplexities)
```

Two lines carry the whole idea: `residual = residual - q` peels off what the current codebook captured, and `z_q = z_q + q` accumulates the running reconstruction. The straight-through estimator inside each `EMAVectorQuantizer` means gradient still flows from the final reconstruction loss back through the whole stack to the encoder — each codebook learns to clean up the residual the earlier ones leave, and the encoder learns to produce a latent the whole stack can represent. One subtlety production codecs add: **quantizer dropout** — during training, randomly truncate the stack to a random $N' \le N$ so the codec learns to reconstruct decently from *any prefix* of the codebooks, which is what makes inference-time bitrate scaling graceful.

![A dataflow graph showing a waveform entering a strided convolutional encoder at 75 frames per second, the residual VQ stack producing N token streams that branch to both an LM-bound token output and a summed-entry latent, and a mirror convolutional decoder reconstructing the waveform.](/imgs/blogs/residual-vector-quantization-rvq-5.png)

## 6. Why the rate-distortion curve bends

Each codebook you add costs a fixed $\log_2 K$ bits but buys a *shrinking* amount of quality. To see why, think about what each codebook is quantizing. Codebook 1 quantizes the full latent $z$ — it has the largest energy to work with, so getting it roughly right removes most of the reconstruction error. Codebook 2 quantizes $r_1$, the residual *after* codebook 1, which has much smaller energy. Codebook 3 quantizes $r_2$, smaller still. Each codebook operates on a residual with less energy than the last, so each one removes less absolute error than the last. The first codebook does the heavy lifting; the eighth is polishing.

Quantitatively, this is the **rate-distortion** trade. Distortion (mean squared residual energy $\lVert r_N \rVert^2$, or equivalently a perceptual metric like ViSQOL or a multi-resolution STFT distance) falls roughly geometrically as you add codebooks, while rate (bits, hence kbps) grows *linearly*. Plot quality against bitrate and you get a concave curve that rises steeply at first and then flattens — the classic diminishing-returns shape. The reason is information-theoretic: under a high-resolution quantization model, each additional bit of rate reduces squared distortion by a roughly constant *factor*, so distortion decays exponentially in rate, which means *log* distortion is linear in rate and the perceptual-quality-versus-rate curve is concave. You are paying linearly and being repaid logarithmically.

Let us make that concrete with a simple model. Suppose each codebook is well-trained enough to reduce the residual energy by a constant factor $\rho < 1$ — that is, $\lVert r_i \rVert^2 \approx \rho \, \lVert r_{i-1} \rVert^2$. Then after $N$ codebooks the residual energy is

$$
\lVert r_N \rVert^2 \approx \rho^N \, \lVert z \rVert^2,
$$

so the distortion decays *geometrically* in $N$. Meanwhile the rate is $R = N \log_2 K$ bits per frame, which grows *linearly* in $N$. Substitute $N = R / \log_2 K$ into the distortion:

$$
D(R) \approx \lVert z \rVert^2 \cdot \rho^{\,R / \log_2 K} = \lVert z \rVert^2 \cdot 2^{-cR}, \qquad c = \frac{\log_2(1/\rho)}{\log_2 K} > 0.
$$

Distortion falls off as $2^{-cR}$ — exponential in rate. Take the log and you get $\log D \approx \text{const} - cR$: log-distortion is *linear* in rate, exactly the canonical high-resolution rate-distortion law $D(R) \propto 2^{-cR}$ that classical quantization theory predicts for any reasonable source. This is not an accident of RVQ; it is RVQ realizing the theoretical shape of the rate-distortion function with stacked codebooks. The practical consequence is the concave curve: in *quality* units (which are roughly logarithmic in distortion, because perception is logarithmic — a decibel is a log unit), quality is roughly linear in rate near the bottom and saturates as $D \to$ the irreducible floor set by the encoder's bottleneck dimension and the metric. The knee of the curve is where $D$ has dropped enough that further halving is below the perceptual threshold, and that is where you want to live.

A caveat worth stating plainly: the constant-factor assumption ($\rho$ fixed across codebooks) is an idealization. In practice the *first* codebook often reduces the residual by more than $\rho$ (the latent has strong low-rank structure the first codebook grabs cheaply) and the later codebooks by less (the residual becomes whiter, harder to compress) — which makes the real curve bend *even more sharply* than the geometric model predicts. The diminishing returns are, if anything, worse than the clean exponential suggests. That only strengthens the conclusion: stop at the knee.

![A matrix figure with rows for one, two, four, eight, and twelve codebooks and columns for bitrate in kbps, ViSQOL score, and the per-step quality gain, showing bitrate rising linearly while the quality gain shrinks from large early jumps to tiny late ones.](/imgs/blogs/residual-vector-quantization-rvq-4.png)

The numbers in that figure follow the shape SoundStream and EnCodec report (the absolute values are approximate and depend on the encoder, the discriminators, and the eval set — treat them as the *trend*, not exact published figures): going from 1 to 2 codebooks is a big quality jump, 2 to 4 is solid, 4 to 8 is a clear but smaller gain, and 8 to 12 is barely perceptible. This is exactly why SoundStream and EnCodec target around 6 kbps (eight codebooks at 75 fps): it sits at the knee of the curve, where you have captured nearly all the perceptual quality you are going to get and adding more codebooks mostly burns bitrate and makes the downstream model's job harder for diminishing return.

The crucial second-order point is that the codec is almost never the thing whose cost you are optimizing — the *generator* is. Each codebook you add is another token stream the downstream language model must predict, so the generator's compute, latency, and (for autoregressive models) its tendency to drift over long sequences all scale with $N$. So the right way to read the rate-distortion curve is not "where does codec quality stop improving" but "where does the *marginal codec quality per unit of generator cost* stop being worth it." Those are different knees, and the second is at a *lower* $N$ than the first, because the generator cost grows while the codec quality saturates. In practice this pushes you toward the smallest $N$ that clears your quality bar — which is the opposite of the instinct to "use all the codebooks for safety." More codebooks is not free insurance; it is a tax the generator pays on every frame it produces.

### Worked example: where the knee is

Suppose you measure ViSQOL (a perceptual similarity score from 1 to ~4.5; higher is better) at a few codebook counts and get, for a given codec, roughly: $N=1 \to 2.6$, $N=2 \to 3.3$, $N=4 \to 3.9$, $N=8 \to 4.2$, $N=12 \to 4.3$. The marginal gain per *added codebook* is $+0.7$ (going to 2), then averaging $+0.3$ per codebook (going to 4), then $+0.075$ per codebook (going to 8), then $+0.025$ per codebook (going to 12). The marginal quality per bit collapses by an order of magnitude across the range. If your application is a music generation model where each extra codebook is another token stream the LM must predict (8 codebooks means the model emits 8x as many tokens per frame), the right call is almost always to live at $N=4$ to $N=8$, not $N=12$. The curve says the last codebooks are nearly free quality you are paying full bitrate *and* full generation cost for.

## 7. Bitrate scalability: drop codebooks at inference

The most operationally useful property of RVQ falls straight out of its structure. Because the reconstruction is a *sum* of independent codebook contributions, and because codebook 1 carries the coarsest, most important information, you can **drop the later codebooks at inference time** and still decode — you just decode from a prefix of the stack. Use all 8 codebooks for 6 kbps; use the first 4 for 3 kbps; use the first 2 for 1.5 kbps. Same trained codec, no retraining, a bitrate dial you turn per request. That `n_active` argument in the `ResidualVQ.forward` above is exactly this: feed `n_active=4` and the stack stops after four codebooks, the residuals from codebooks 5 through 8 are simply left in (unquantized error), and the decoder — if it was trained with quantizer dropout — handles it gracefully.

![A before-and-after figure contrasting decoding from two of eight codebooks at 1.5 kbps with smeared high frequencies and a lower ViSQOL against decoding from all eight codebooks at 6 kbps with restored highs, clean timbre, and a near-transparent ViSQOL.](/imgs/blogs/residual-vector-quantization-rvq-7.png)

What does dropping codebooks actually cost perceptually? The later codebooks encode the *fine* residual, which is disproportionately high-frequency detail and timbral nuance — the air on a cymbal, the breathiness of a voice, the texture of a snare. So a low-bitrate prefix keeps the coarse structure (you can tell what is being said or played, the pitch and rhythm are intact) but smears the highs: speech sounds muffled, music sounds like it is behind a curtain. This is graceful degradation, not a cliff — exactly the behavior you want for adaptive-bitrate streaming, where you cut to a 2-codebook prefix under network pressure and restore the full stack when bandwidth returns, all from one codec.

This is also why the *ordering* of codebooks is not arbitrary: codebook 1 must carry the most important information, because it is the one that survives every truncation. RVQ enforces this automatically — codebook 1 quantizes the full-energy latent, so it *is* the most important by construction. The hierarchy is coarse-to-fine, and that ordering is what makes both bitrate scaling and the generative-model strategy in the next section work.

The trick that makes truncation actually decode well is **quantizer dropout** during training, and it is worth being precise about why it is necessary. If you train a codec with all $N$ codebooks always active, the decoder learns to expect the *sum* of all $N$ contributions. Feed it only the first 2 at inference and it sees a latent that is systematically wrong — missing the energy the other 6 codebooks would have added — and it reconstructs garbage, because it never saw a 2-codebook latent during training. Quantizer dropout fixes this by randomly truncating the stack to a random $N' \in \{1, \dots, N\}$ on each training step, so the decoder is forced to produce a reasonable waveform from *any* prefix length. The decoder learns a graceful "given fewer codebooks, fall back to a band-limited reconstruction" behavior. Without quantizer dropout you do not get bitrate scaling — you get a codec that only works at exactly the bitrate it was trained at. SoundStream introduced this; EnCodec and DAC inherit it.

#### Worked example: an adaptive-bitrate streaming budget

Suppose you stream voice over a network whose available bandwidth swings between 1 and 8 kbps. With a classical fixed-bitrate codec you would pick a single conservative bitrate and live with it. With an RVQ codec and quantizer dropout you adapt per packet: when the network gives you 6 kbps of headroom you send all 8 codebook streams (full fidelity); when it drops to 1.5 kbps you send only the first 2 streams ($75 \times 2 \times 10 = 1.5$ kbps) and the receiver decodes a muffled-but-intelligible signal; when it recovers you go back to 8. The *same encoded coarse tokens* (codebooks 1–2) are a prefix of the full encoding, so you are not re-encoding — you are choosing how many of the already-computed streams to transmit. One codec, one encode pass, a bitrate you dial per packet. This is exactly the property that makes RVQ codecs attractive for real-time communication, not just generation.

## 8. How the coarse-to-fine hierarchy shapes the generative model

The token stream RVQ produces is not a flat sequence — it is $N$ parallel streams with a clear importance ordering, and the generative models that consume codec tokens are built around that structure. This is where the codec design reaches forward into the model design, and it is worth being precise about, because "predict the tokens" hides real architectural choices.

![A vertical stack figure showing codebook-one tokens predicted first as the coarse layer carrying the most information, codebook-two tokens conditioned on them, finer codebooks three through N, a delay-and-interleave pattern across the streams, and a final codec decode summing the entries into a waveform.](/imgs/blogs/residual-vector-quantization-rvq-8.png)

The foundational pattern, from AudioLM (Borsos et al., 2022), is **coarse-to-fine prediction**: predict codebook 1 for the whole sequence first (or the coarsest few codebooks), then predict the finer codebooks *conditioned on* the coarse ones. This matches the information hierarchy — the model spends its hardest modeling on the tokens that matter most, and the fine tokens become a comparatively easy "fill in the detail given the structure" problem. AudioLM goes further and stacks a *semantic* token layer on top, but the coarse-to-fine codec-token idea is the part that comes straight from RVQ. (The full semantic-versus-acoustic story is its own post: [semantic vs acoustic tokens](/blog/machine-learning/audio-generation/semantic-vs-acoustic-tokens).)

The problem this raises: with $N$ codebooks per frame, you have a *two-dimensional* array of tokens — frames along one axis, codebooks along the other — and an autoregressive model needs a *one-dimensional* order to predict them in. How you flatten that 2D grid into a 1D sequence is a real design choice with real consequences. The naive flattening — emit all $N$ tokens for frame 1, then all $N$ for frame 2 — makes the sequence $N$ times longer and forces the model to predict fine tokens before it has committed to coarse structure several frames ahead. That ordering also wastes the coarse-to-fine prior: the model has to guess the fine residual of frame 1 with no idea what the coarse structure of frame 2 will be.

MusicGen (Copet et al., 2023) solved this with a **delay/interleaving pattern**: it offsets each codebook stream by a growing delay so that within a single transformer step the model predicts codebook 1 of frame $t$, codebook 2 of frame $t-1$, codebook 3 of frame $t-2$, and so on. This keeps the sequence length manageable (one transformer step per frame, not $N$), preserves the coarse-to-fine conditioning (by the time the model commits codebook 2 of a frame it has already committed codebook 1), and was a big part of what made MusicGen a single-stage model rather than the multi-stage cascade AudioLM used. The delay means the model only ever predicts each frame's coarse token *before* its fine tokens, which is exactly the order the RVQ hierarchy wants.

VALL-E (Wang et al., 2023) takes yet another split for TTS: an autoregressive model predicts codebook 1 (the coarse acoustic token) frame by frame, and a separate non-autoregressive model predicts codebooks 2 through $N$ in parallel given codebook 1 — coarse-to-fine again, but with the fine codebooks done in one non-autoregressive shot for speed. The AR model carries the hard, sequential, prosody-and-content part (codebook 1); the NAR model fills in acoustic detail in a single forward pass per codebook because, conditioned on the coarse tokens, the fine ones are nearly independent across frames. This is why VALL-E is fast enough to be practical: only one of its $N$ codebooks is predicted autoregressively.

Here is the same idea seen three ways:

| Model | Coarse-token prediction | Fine-token prediction | Sequence length | Why it is shaped this way |
|---|---|---|---|---|
| AudioLM | AR over coarse acoustic tokens (after semantic stage) | AR over fine tokens, separate stage | long (cascaded stages) | clean coarse-to-fine separation; multi-stage |
| MusicGen | AR, codebook 1 of frame $t$ first | AR, delayed by 1 frame per codebook | ~1 step/frame | single-stage; delay preserves coarse-first order |
| VALL-E | AR over codebook 1, frame by frame | NAR, all of codebooks 2..N in parallel | short (1 AR stream) | fine tokens ~independent given coarse; fast |

Read the table as three answers to one question — *how do you serialize a coarse-to-fine RVQ grid for a generative model?* — and notice that all three put codebook 1 first and treat the fine codebooks as cheaper. That is the RVQ hierarchy dictating the architecture, not a coincidence of three independent design teams.

The throughline: **the codec's RVQ structure dictates the generative model's factorization.** Whether you cascade (AudioLM), interleave with delays (MusicGen), or split AR-plus-NAR (VALL-E), every one of these is a strategy for predicting a coarse-to-fine RVQ token hierarchy, and every one of them exploits the fact that codebook 1 carries the most information. You cannot understand why these models are shaped the way they are without understanding RVQ first. That is why this post sits where it does in the [audio stack](/blog/machine-learning/audio-generation/why-audio-generation-is-hard): the quantizer is upstream of, and constrains, everything that generates on top of it.

## 9. Encoding and decoding with a real codec

Enough theory — let us run a real RVQ-based codec end to end. The 🤗 `transformers` library ships EnCodec with a clean API, so you can encode a waveform to integer RVQ codes, inspect the codebook streams, drop later codebooks to lower the bitrate, and decode back to audio, all in a few lines. This is the exact flow your generative model sits inside: it consumes the integer codes `encode` produces and feeds new codes to `decode`.

```python
import torch
import torchaudio
from transformers import EncodecModel, AutoProcessor

# EnCodec 24 kHz, the SoundStream-lineage RVQ codec.
model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

# Load and resample our running example to the codec's 24 kHz.
wav, sr = torchaudio.load("example.wav")            # (channels, samples)
wav = torchaudio.functional.resample(wav, sr, 24000).mean(0, keepdim=True)
inputs = processor(raw_audio=wav.squeeze().numpy(),
                   sampling_rate=24000, return_tensors="pt")

# Encode at a chosen bitrate -> integer RVQ codes.
with torch.no_grad():
    enc = model.encode(inputs["input_values"], inputs["padding_mask"],
                       bandwidth=6.0)               # 6.0 kbps -> 8 codebooks
codes = enc.audio_codes        # (1, 1, num_codebooks, frames) of int indices
print("codes shape:", codes.shape)                 # e.g. (1, 1, 8, T)
print("codebooks used:", codes.shape[-2])          # 8 at 6 kbps
print("frames:", codes.shape[-1], "fps:", codes.shape[-1] / (wav.shape[-1] / 24000))
```

The `bandwidth` argument is the bitrate dial from section 7 made literal: EnCodec maps `1.5, 3.0, 6.0, 12.0, 24.0` kbps to `2, 4, 8, 16, 32` codebooks respectively (at 75 fps, `bandwidth / 0.75` codebooks). Print `codes.shape[-2]` and you will see exactly the number of codebooks the formula predicts. Those integers are the tokens — the *only* thing a downstream language model ever sees of the audio. To go back to a waveform:

```python
with torch.no_grad():
    recon = model.decode(enc.audio_codes, enc.audio_scales,
                         inputs["padding_mask"])[0]
torchaudio.save("recon_6kbps.wav", recon.squeeze(0), 24000)

# Now drop to 1.5 kbps by re-encoding with fewer codebooks (bitrate scaling).
with torch.no_grad():
    enc_lo = model.encode(inputs["input_values"], inputs["padding_mask"],
                          bandwidth=1.5)            # 2 codebooks
    recon_lo = model.decode(enc_lo.audio_codes, enc_lo.audio_scales,
                            inputs["padding_mask"])[0]
torchaudio.save("recon_1p5kbps.wav", recon_lo.squeeze(0), 24000)
```

Listen to the two files and you will hear exactly the degradation section 7 described: the 6 kbps reconstruction is close to transparent, the 1.5 kbps version is intelligible but muffled, with the highs smeared. Same codec, same weights, a bitrate you chose at call time. And here is the diagnostic from section 4 — codebook usage — computed directly on the integer codes the codec emits, so you can check on *real* audio that the codebooks are not collapsing:

```python
def codebook_usage(codes, num_entries=1024):
    """Per-codebook utilization and perplexity from integer RVQ codes."""
    codes = codes.reshape(codes.shape[-2], -1)      # (num_codebooks, frames)
    for cb in range(codes.shape[0]):
        idx = codes[cb]
        used = idx.unique().numel()
        counts = torch.bincount(idx, minlength=num_entries).float()
        p = counts / counts.sum()
        nz = p[p > 0]
        perplexity = torch.exp(-(nz * nz.log()).sum())
        print(f"codebook {cb}: {used}/{num_entries} used "
              f"({100 * used / num_entries:.1f}%), perplexity {perplexity:.0f}")

codebook_usage(codes[0])
```

Run this on a few minutes of varied audio and a healthy codec shows high utilization on codebook 0 (it sees the most diverse signal) and progressively lower-but-still-substantial usage on the fine codebooks. If codebook 0 shows 40/1024 used, you have caught the collapse from the opening paragraph on a real model — and you caught it from the *codes*, not the loss curve. (For training your own codec rather than using a pretrained one, swap in `descript-audio-codec` or `audiocraft`'s EnCodec trainer; the diagnostic above works on any RVQ codec's integer codes.)

## 10. Case studies and real numbers

Here are concrete, citable configurations so the formulas above attach to real systems. Where I give a perceptual number I flag it as approximate — codec quality numbers depend heavily on the eval set, the metric's embedding, and the encoder/discriminator details, and you should reproduce them on your own data before trusting them.

| Codec | Sample rate | Frame rate | Codebooks (N) | Codebook size (K) | Bitrate | Notes |
|---|---|---|---|---|---|---|
| SoundStream | 24 kHz | 75 fps | up to 12 | 1024 | 0.75–9 kbps | Introduced RVQ for neural codecs; quantizer dropout for scalable bitrate |
| EnCodec | 24 / 48 kHz | 75 fps | up to 32 (variable) | 1024 | 1.5–24 kbps | EMA codebooks + entry expiry; MS-STFT discriminators |
| DAC (Descript) | 44.1 kHz | ~86 fps | 9 (default) | 1024 | ~8 kbps | k-means init, factorized + L2-normalized codes for near-100% usage |
| Mimi (Moshi) | 24 kHz | 12.5 fps | 8 | 2048 | ~1.1 kbps | Split semantic + acoustic RVQ; very low frame rate for streaming dialogue |

A few things to read off this table. EnCodec and SoundStream share the 75-fps / $K=1024$ design and differ mainly in the discriminator and the codebook-maintenance details. DAC's headline contribution was *fixing codebook usage*: by projecting codes to a low-dimensional space and L2-normalizing before the lookup (factorized codes), DAC pushes codebook utilization to near 100%, which is why it reaches high fidelity at a given bitrate with fewer dead entries — a direct, measured win against the collapse problem in section 4. Mimi is the interesting outlier: it runs at an aggressively low 12.5 fps (so the LM in Moshi has very few tokens per second to predict, essential for real-time full-duplex dialogue) and compensates with a larger codebook and a semantic/acoustic split.

#### Worked example: DAC's usage win in kbps terms

Suppose a vanilla RVQ at 8 codebooks of size 1024 effectively uses only ~512 entries per codebook on your data (perplexity ~512). Your *nominal* bitrate is $75 \times 8 \times 10 = 6$ kbps, but your *effective* information is $75 \times 8 \times \log_2 512 = 75 \times 8 \times 9 = 5.4$ kbps — you are wasting roughly 10% of your bitrate on dead entries. DAC's factorized, L2-normalized codes drive perplexity to nearly the full 1024 (~10 effective bits per codebook), recovering that 0.6 kbps as real reconstruction quality at the same nominal bitrate. That is the practical payoff of solving collapse: not a different bitrate, but *every bit you pay for actually carrying information*.

#### Worked example: choosing N for a TTS deployment

You are shipping a voice assistant. Latency budget is tight, you run a codec-token LM on the GPU, and each codebook is another set of tokens the LM predicts. You measure: at $N=8$ the LM emits 600 tokens/s and MOS (mean opinion score, 1–5, from human raters) is ~4.1; at $N=4$ it emits 300 tokens/s and MOS is ~3.9; at $N=2$ it emits 150 tokens/s and MOS drops to ~3.4 (audibly muffled). The $N=8 \to N=4$ move halves your generation cost (and roughly halves real-time factor) for a 0.2 MOS hit most users will not notice on speech; the $N=4 \to N=2$ move is a false economy — you save another 150 tokens/s but cross into clearly-degraded territory. Conclusion: ship $N=4$ for this use case. The rate-distortion knee told you where to stop, and the per-token generation cost told you which side of the knee to err toward. (How to measure MOS honestly — rater counts, CMOS against a reference, fixed text and seed — is covered in [audio quality metrics](/blog/machine-learning/audio-generation/audio-quality-metrics).)

## 11. Stress tests: where RVQ bites back

Pose the failures honestly, because they are the ones you will actually hit.

**What happens at very low bitrate (one or two codebooks)?** The decoder gets only the coarsest residual. Speech stays intelligible (the structure survives) but sounds muffled and "underwater"; music keeps pitch and rhythm but loses all air and texture. This is graceful, not catastrophic — but it is a hard floor. Below about 1.5 kbps for general audio you are into "recognizable but clearly degraded," and no amount of decoder cleverness fixes missing bits.

**What happens when the codec drops the high frequencies?** Because the fine codebooks encode high-frequency residual, an under-trained or under-provisioned fine stack smears the top octave. The tell is a spectrogram with a soft, blurry ceiling above ~8 kHz. The fix is not more codebooks but better fine-codebook training — quantizer dropout so the fine codebooks actually get gradient, and the multi-resolution STFT loss weighting high frequencies enough that the model cares about them.

**What happens when codebook collapse hits a *middle* codebook?** Subtle and nasty. If codebook 5 collapses to a handful of entries, you are paying for 10 bits there and getting 3, so your effective bitrate is below nominal and your quality plateaus early no matter how many codebooks you stack. This is why you log per-codebook perplexity, not just an aggregate — collapse can hide in one layer of the stack while the others look healthy.

**What happens when you crank N way up for "quality"?** Past the knee, extra codebooks barely move perceptual quality but multiply the downstream model's token count and generation time. A music LM at $N=16$ predicts twice the tokens of $N=8$ for a fraction of a ViSQOL point — you have made the codec marginally better and the *generator* twice as slow and twice as likely to drift over a long sequence. The codec and the generator share a bitrate budget; spending it all on the codec starves the model.

**What happens when the encoder fights the codebook?** If the commitment weight $\beta$ is too low, the encoder's outputs drift far from the codebook, the quantization error explodes, and reconstruction degrades while the loss looks deceptively fine. If $\beta$ is too high, the encoder is over-constrained and cannot use its full representational capacity. The symptom of "too low" is a large gap between $z$ and $z_q$ (log it); the symptom of "too high" is a reconstruction that plateaus below what the bitrate should allow. $\beta = 0.25$ with EMA codebooks is the robust default; reach for the knobs only when the diagnostics tell you to.

**What happens when the input is out-of-distribution — a 3-second noisy speaker prompt for voice cloning?** This is the case that connects RVQ to TTS, and it is instructive. A codec trained mostly on clean speech and music will encode a noisy, reverberant 3-second clip *faithfully* — including the noise and reverb, because the codec's job is reconstruction, not denoising. The tokens it produces carry the speaker's timbre but also the room and the hiss, and a downstream voice-cloning model (VALL-E-style) conditioned on those tokens will happily clone the noise along with the voice. The codec is not the place to fix this; it is doing exactly what it was trained to do. The lesson is a clean division of labor: the codec is a faithful, content-agnostic tokenizer, and quality control (denoising, dereverberation, loudness normalization) belongs *before* the encoder, not inside it. If your clones sound noisy, clean the prompt audio first — do not blame the RVQ. This is a recurring theme in the [TTS and voice-cloning](/blog/machine-learning/audio-generation/semantic-vs-acoustic-tokens) parts of the stack: the codec faithfully tokenizes whatever you give it, so garbage in is garbage tokenized.

**What happens when the codec, not the model, is your fidelity ceiling?** Every generative model that produces codec tokens is upper-bounded by the codec's own reconstruction quality — the model can at best produce tokens the codec decodes as well as it decodes *real* audio's tokens. If your codec's reconstruction MOS is 4.1, no language model on top of it will ever exceed MOS 4.1, because the codec decode is the last step and it cannot un-lose the information the codec threw away. This is the "topline" check every audio-gen project should run first: encode and immediately decode real audio (no generation), measure the reconstruction quality, and treat that as the hard ceiling for the whole system. If the ceiling is too low, you fix the codec (more codebooks, better discriminators, a higher-fidelity model) before you touch the generator — improving a generator under a leaky codec is wasted effort.

## When to reach for RVQ (and when not to)

RVQ is the right tool when you need **discrete tokens at a target bitrate for a downstream generative model** — which is essentially every codec-LM audio system (VALL-E, MusicGen, AudioLM, Moshi). If your generator is an autoregressive or masked language model over audio tokens, you need a codec, and a codec needs RVQ to hit high fidelity without an unmanageable single codebook. Use it.

Reach for it specifically when: you want **inference-time bitrate scaling** (RVQ gives it for free via codebook dropping); you want a **coarse-to-fine hierarchy** the generator can exploit (predict structure first, detail later); or you want **small, stable, trainable codebooks** instead of one giant one that collapses.

Do *not* reach for RVQ — or any discrete quantizer — when your generative model is a **continuous diffusion or flow-matching model on a continuous latent**. Stable Audio and many waveform-diffusion systems generate a *continuous* VAE latent and never quantize it; forcing discreteness there only throws away information for no benefit, because the diffusion model is perfectly happy predicting continuous values. The discrete codec exists to serve *language-model-style* generators; if yours is a diffusion model, use a continuous autoencoder (link out to the [VAE post](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) for the continuous analogue) and skip quantization entirely. Likewise, do not stack more codebooks than your generator can afford to predict — past the rate-distortion knee you are buying codec quality your model cannot exploit and paying for it in generation latency and long-sequence drift. And do not ship a codec without logging per-codebook perplexity; collapse is invisible in the loss and lethal to effective bitrate.

A useful decision rule, then. If you are building a music or speech *language model* (AR or masked) — codec with RVQ, pick $N$ at the knee for your quality bar, train with EMA codebooks and quantizer dropout, log per-codebook perplexity, and verify the codec's reconstruction ceiling before you train the generator. If you are building a *diffusion or flow* audio model — continuous VAE latent, no quantizer, and spend your engineering on the latent's bottleneck dimension and the diffusion schedule instead. If you are building a *real-time communication* codec rather than a generator — RVQ with quantizer dropout for adaptive bitrate, and push the frame rate down as far as quality allows to keep latency low. Three different builds, one quantizer family, and the choice between them is set entirely by what consumes the tokens. Get that match right and the rest of the codec is tuning; get it wrong and you will spend weeks fighting a quantizer your generator never needed.

## Key takeaways

- **Vector quantization** replaces a continuous encoder latent with its nearest entry in a learned codebook of size $K$; the emitted integer index is the token, worth $\log_2 K$ bits.
- The nearest-entry **argmin is non-differentiable**, so the **straight-through estimator** copies the decoder's gradient straight onto the encoder ($\nabla_z \mathcal{L} \approx \nabla_{z_q} \mathcal{L}$), while the **codebook loss** trains the entries and the **commitment loss** anchors the encoder.
- **Codebook collapse** (most entries dead) is the dominant VQ failure; measure it with **perplexity** (effective vocabulary size) and fix it with **EMA updates, k-means init, random restarts, and codebook expiry** — production codecs use all four.
- A single codebook caps you at $\log_2 K$ bits per frame; a huge codebook is unstable and underused. **Residual VQ** sidesteps this by stacking $N$ small codebooks, each quantizing the previous one's **residual** ($r_i = r_{i-1} - C^{(i)}$).
- **Bitrate $= \text{fps} \times N \times \log_2 K$** — linear in the number of codebooks. The canonical 6 kbps target is $75 \times 8 \times 10$.
- The **rate-distortion curve is concave**: each added codebook quantizes a smaller residual, so quality climbs with diminishing returns — live at the knee ($N \approx 4$–8), not past it.
- RVQ gives **inference-time bitrate scaling for free**: drop later codebooks to lower the bitrate without retraining, with graceful (highs-first) degradation — train with **quantizer dropout** so the prefix decodes cleanly.
- The **coarse-to-fine** token hierarchy (codebook 1 carries the most information) dictates how generators are built — AudioLM cascades, MusicGen interleaves with delays, VALL-E splits AR + NAR — all predicting coarse tokens first.

## Further reading

- **Zeghidour, Luebs, Omran, Skoglund, Tagliasacchi — "SoundStream: An End-to-End Neural Audio Codec" (2021).** The paper that introduced RVQ to neural codecs, with quantizer dropout for scalable bitrate. The primary source for this post.
- **van den Oord, Vinyals, Kavukcuoglu — "Neural Discrete Representation Learning" (VQ-VAE, 2017).** Where the straight-through estimator, codebook loss, and commitment loss were introduced.
- **Défossez, Copet, Synnaeve, Adi — "High Fidelity Neural Audio Compression" (EnCodec, 2022).** EMA codebooks with entry expiry, multi-scale STFT discriminators, and variable-bitrate RVQ in production.
- **Kumar, Seetharaman, Luebs, Kumar, Kumar — "High-Fidelity Audio Compression with Improved RVQGAN" (DAC, 2023).** Factorized, L2-normalized codes that drive codebook utilization to near 100% — the definitive fix for collapse.
- **Borsos et al. — "AudioLM: a Language Modeling Approach to Audio Generation" (2022)** and **Copet et al. — "Simple and Controllable Music Generation" (MusicGen, 2023).** How the coarse-to-fine RVQ hierarchy and codebook interleaving shape the generative model.
- Within this series: the [neural audio codec](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound) overview this post zooms into, [EnCodec, DAC, and the modern codec](/blog/machine-learning/audio-generation/encodec-dac-and-the-modern-codec), [semantic vs acoustic tokens](/blog/machine-learning/audio-generation/semantic-vs-acoustic-tokens), and the [capstone audio-generation stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack); plus the image series' [autoregressive image models](/blog/machine-learning/image-generation/autoregressive-image-models) for the VQ-VAE/VQ-GAN token parallel.
