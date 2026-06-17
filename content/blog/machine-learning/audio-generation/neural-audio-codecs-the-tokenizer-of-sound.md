---
title: "Neural Audio Codecs: The Tokenizer of Sound"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "The single enabler behind modern audio generation: a learned encoder, quantizer, and decoder that turn a waveform into a short sequence of discrete tokens a transformer can model like text."
tags:
  [
    "audio-generation",
    "audio-synthesis",
    "neural-audio-codec",
    "encodec",
    "vector-quantization",
    "generative-ai",
    "deep-learning",
    "tokenization",
  ]
category: "machine-learning"
subcategory: "Audio Generation"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/neural-audio-codecs-the-tokenizer-of-sound-1.png"
---

The first time I tried to generate audio with a sequence model, I did the naive thing. I took a 10-second clip at 24 kHz, flattened it to its 240,000 raw samples, and fed those into an autoregressive model the way you would feed it tokens of text. The model never trained. The sequence was so long that a single forward pass through the attention layers exhausted the GPU before the loss even printed. Even if it had fit, generating 10 seconds would mean 240,000 sequential decoding steps — one per sample — which on any real hardware is not a model, it is a waiting room. I had walked straight into the wall that every audio engineer hits on day one: **a waveform is far, far too long to model directly.**

The fix is not a bigger GPU or a cleverer attention kernel. It is a different representation. Modern audio generation does not model raw samples and it does not model the spectrogram pixels either. It models the output of a **neural audio codec** — a small learned network that compresses a waveform into a short sequence of **discrete tokens**, and reconstructs the waveform back from those tokens with quality good enough to fool your ears. The codec is the piece that makes everything downstream possible. VALL-E clones a voice from a 3-second prompt by autoregressing codec tokens. MusicGen writes music by autoregressing codec tokens. AudioLM models speech continuations as codec tokens. Stable Audio runs diffusion on a codec's continuous latent. The codec is the bridge between the messy continuous world of pressure waves and the clean discrete world a transformer was built for.

If you have read the image-generation series, you already know the shape of this idea. There, the [variational autoencoder](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) compresses a 512×512×3 image into a tiny latent grid, and the diffusion model works in that latent space instead of pixel space. The neural audio codec is the audio analogue of exactly that VAE — with one crucial twist that defines the whole field: the audio codec usually **quantizes** its latent into discrete tokens, so that a *language model*, not just a diffusion model, can generate audio. That single design choice is why so much of audio generation in 2024–2026 looks like text modeling with a different vocabulary.

![Stacked diagram showing a waveform passing through an encoder, a latent, a quantizer producing tokens, and a decoder reconstructing the waveform, with the frame and token rates labeled at each stage](/imgs/blogs/neural-audio-codecs-the-tokenizer-of-sound-1.png)

By the end of this post you will be able to: explain why a raw waveform is too long to model and compute exactly how much the codec shrinks it; describe the three blocks of a codec (encoder, quantizer, decoder) and the GAN-style loss that trains them; calculate a codec's bitrate from its frame rate, number of codebooks, and codebook size; tell the difference between a *discrete* codec for language models and a *continuous* latent for diffusion, and know which to reach for; and actually run a codec in code with 🤗 `transformers`, encode audio to integer tokens, decode it back, and measure the reconstruction error. This is the key-enabler post of the codec track. It sets up [residual vector quantization](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq), the [EnCodec and DAC](/blog/machine-learning/audio-generation/encodec-dac-and-the-modern-codec) designs, and the [semantic-vs-acoustic token](/blog/machine-learning/audio-generation/semantic-vs-acoustic-tokens) split that the rest of the series builds on. If you have not read [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard), that is the foundation; this post is the answer to the central problem it poses.

## 1. The problem, in numbers you can feel

Let us make the wall concrete, because the numbers are the whole motivation and they are worth internalizing.

Sound that reaches your ear is a continuous pressure wave. To put it in a computer we **sample** it — measure the pressure at regular instants — and **quantize** each measurement to a number with finite precision. Speech is usually sampled at 16 kHz (16,000 samples per second), general audio and many TTS systems at 22.05 or 24 kHz, and music at 44.1 kHz (CD quality) or 48 kHz. Each sample is typically a 16-bit integer. We covered the *why* of those choices — the Nyquist sampling theorem, bit depth, the dynamic range of hearing — in [representing sound: waveforms, spectrograms, and perception](/blog/machine-learning/audio-generation/representing-sound-waveforms-spectrograms-and-perception). What matters here is the *rate*.

At 24 kHz, **one second of audio is 24,000 numbers.** Ten seconds is 240,000. A three-minute song at 44.1 kHz is

$$
44{,}100 \ \frac{\text{samples}}{\text{s}} \times 180 \ \text{s} \approx 7.9 \times 10^{6} \ \text{samples}.
$$

Nearly eight million numbers for one song. Now recall what a sequence model costs. A transformer's self-attention is $O(L^2)$ in sequence length $L$. Even a model with linear-attention tricks pays $O(L)$ memory and, for *autoregressive* generation, makes one sequential forward pass **per output position**. At $L = 240{,}000$, the quadratic attention term alone is $5.76 \times 10^{10}$ — and that is before any feature dimension. An autoregressive model would need 240,000 sequential steps to emit ten seconds, which even at a generous 1,000 steps per second of wall-clock generation is four minutes of compute for ten seconds of sound. The arithmetic is hopeless.

There is a second, subtler problem hiding behind the length: **redundancy.** Adjacent audio samples are enormously correlated. The waveform of a vowel is a smooth, near-periodic curve; consecutive samples differ by tiny amounts. A model that predicts raw samples spends almost all of its capacity learning to predict "roughly the same as the last sample," which is the audio equivalent of an LLM spending its parameters predicting that the next character after a space is usually a letter. There is real information in audio — pitch, timbre, phonetic identity, the attack of a drum — but it lives at a far lower rate than 24,000 events per second. Speech carries on the order of a few hundred phonetic and prosodic decisions per second, not 24,000. The waveform is a high-rate *encoding* of a low-rate *message*.

That gap is exactly what a codec exploits. If the true information rate of speech is, say, a few thousand bits per second, then a good codec should be able to throw away most of the 384,000 bits per second (24,000 samples × 16 bits) that a raw 24 kHz/16-bit stream carries and lose nothing a listener cares about. Classical codecs — MP3, AAC, Opus — already do this with hand-designed psychoacoustic models. A **neural** codec does it with a learned encoder and decoder, and crucially, it produces a representation that a generative model can work in.

It is worth pausing on *why* compression this aggressive is even possible, because the information-theory framing is the foundation the whole field stands on. A raw audio stream's bit budget — its 384 kbps at 24 kHz/16-bit — is an *upper bound* on how much information it could carry, assuming every sample were independent and uniformly distributed. Real audio is nowhere near that. Two things shrink the true entropy far below the bound. First, **statistical redundancy**: because adjacent samples are highly correlated, knowing the recent past makes the next sample largely predictable, so its *conditional* entropy — the genuinely new information it adds — is small. A predictive coder that models $p(x_t \mid x_{<t})$ pays only for the surprise, and audio holds few surprises sample to sample. Second, **perceptual irrelevance**: a large fraction of what *is* present in the signal cannot be heard — quiet frequency components masked by louder nearby ones, detail above the frequencies your cochlea resolves, phase relationships your auditory system is insensitive to. The [psychoacoustics post](/blog/machine-learning/audio-generation/representing-sound-waveforms-spectrograms-and-perception) covers masking and the loudness curve; the consequence here is that a codec is allowed to discard not just the *redundant* bits but the *imperceptible* ones, and the listener never knows.

A neural codec attacks both at once, and that is its edge over a hand-tuned classical codec. The learned encoder discovers the statistical structure automatically — it does not need an engineer to specify which frequency bands to group — and the *adversarial* training objective, which we will meet shortly, teaches the decoder to produce audio that is perceptually convincing rather than numerically exact, which is exactly the license to drop imperceptible detail. Where MP3 hard-codes a psychoacoustic model, a neural codec *learns* one, jointly with a synthesis network that can hallucinate plausible detail at decode time instead of transmitting it. That is why neural codecs reach usable quality at one-tenth the bitrate of the classical codecs that dominated the 2000s — and why their output is a clean, low-rate representation that a downstream transformer or diffusion model can actually model.

#### Worked example: the length budget

Take a transformer with a comfortable context of 4,096 positions — the kind of length budget that trains without exotic engineering. How much audio fits?

- **Raw samples at 24 kHz:** $4096 / 24000 \approx 0.17$ seconds. You cannot model a single word.
- **Codec frames at 75 fps:** $4096 / 75 \approx 54.6$ seconds. Now you have nearly a minute of audio in context.
- **Codec tokens at 8 codebooks per frame (600 tokens/s):** $4096 / 600 \approx 6.8$ seconds. Still a usable utterance, and with codebook *interleaving* or *delay* patterns (which we will meet in the MusicGen post) you can stretch this further.

The codec buys you a **320-fold** reduction in the number of frames (24,000 → 75), and even after spending some of that back on multiple codebooks per frame, you land in a range where a sequence model is not just feasible but pleasant to train. That is the entire reason the field exists.

## 2. The codec idea: encoder, quantizer, decoder

A neural audio codec has exactly three parts, and once you see them you see them everywhere.

**The encoder** is a stack of 1D convolutions with *stride*. Stride is the trick: a convolution with stride 2 outputs half as many frames as it ingests, because it hops two input samples per output. Chain several strided conv blocks and the downsampling multiplies. EnCodec's encoder, for instance, uses strides of $(2, 4, 5, 8)$, whose product is $2 \times 4 \times 5 \times 8 = 320$. Feed it a 24,000-sample second and it emits $24{,}000 / 320 = 75$ frames. Each frame is not a scalar but a **vector** — EnCodec uses a 128-dimensional latent — that summarizes everything happening in the ~13 ms window of audio it came from. So the encoder's job is: take the high-rate, low-dimensional waveform and produce a low-rate, higher-dimensional **latent sequence** at, say, 75 frames per second.

**The quantizer** is what makes it a *codec* and not just an autoencoder. The 128-dim latent vector for each frame is still a continuous float vector — there are uncountably many of them. The quantizer **snaps each latent to the nearest entry in a learned codebook** of, say, 1,024 vectors. After quantization, each frame is no longer a float vector; it is an *integer index* into the codebook, a number from 0 to 1,023. That integer is the **token**. A single codebook of 1,024 entries carries $\log_2 1024 = 10$ bits per frame, which is usually not enough fidelity, so real codecs stack several codebooks per frame using **residual vector quantization** — the first codebook quantizes the latent, the second quantizes what the first one got wrong (the residual), the third quantizes that residual, and so on. We give RVQ its own full treatment in [the next post](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq); for now hold the picture that one frame becomes a small *stack* of integer tokens, one per codebook.

**The decoder** mirrors the encoder. It is a stack of *transposed* (strided) convolutions that *upsample* — each block roughly doubles or quadruples the rate — taking the low-rate quantized latent back up to the full 24,000-sample-per-second waveform. The decoder is where the real synthesis happens, and it is trained adversarially, like a GAN vocoder, which is why a good codec's output sounds like real audio rather than a muffled average. We will get to that loss shortly.

The whole thing is trained **end to end as an autoencoder**: feed in a waveform, encode it, quantize it, decode it, and push the reconstruction to match the input. The quantizer sits in the middle as a bottleneck. Because quantization is not differentiable (rounding to the nearest codebook entry has zero gradient almost everywhere), the codec uses a **straight-through estimator** — in the backward pass it pretends the quantizer was the identity and copies the gradient straight through to the encoder. That trick, plus a small commitment loss that pulls the encoder's outputs toward their assigned codebook entries, is what lets gradients flow past the discrete bottleneck. Again, the details live in the RVQ post; here the key point is that quantization is *learnable* and the codebook is trained jointly with the encoder and decoder.

Here is the mental frame that makes all of this click, and it is the title of this post: **the codec is a tokenizer for sound.** A text tokenizer maps a string to a sequence of integers from a fixed vocabulary, and a detokenizer maps those integers back to text. A neural audio codec maps a waveform to a sequence of integers from a learned codebook (the encoder + quantizer), and maps those integers back to a waveform (the decoder). The vocabulary is the codebook. Once you have that, **you can model audio with the exact same machinery you use for text** — an autoregressive transformer that predicts the next token, trained with cross-entropy, sampled with temperature and top-p. That equivalence is the engine of AudioLM, VALL-E, MusicGen, and the whole audio-language-model paradigm.

## 3. Why *discrete*? The fork that defines the field

You might reasonably ask: if the encoder already produces a short, compact latent at 75 frames per second, why bother quantizing it at all? The continuous latent is already 320× shorter than the waveform. Why throw away precision by snapping it to a codebook?

The answer is that **discreteness buys you a language model**, and that turns out to be worth a great deal.

A transformer that predicts a continuous vector has to either regress it (predict the mean, which collapses to blurry averages for multimodal targets) or model its distribution with something like a mixture density network or a diffusion head (which works but is more delicate). A transformer that predicts a **discrete token** does the thing transformers are best at on Earth: a softmax over a fixed vocabulary, trained with cross-entropy, with all the well-understood sampling controls — temperature, top-k, top-p, beam search, classifier-free guidance applied to logits. When each audio frame is an integer from a codebook of 1,024 entries, "generate audio" becomes literally "predict the next code," indistinguishable in its mechanics from "predict the next word." Every advance in language modeling — better architectures, better sampling, better scaling laws, in-context learning — transfers for free.

This is the **discrete path**, and it powers the audio-language-model lineage:

- **AudioLM** (Borsos et al., 2022) models speech and piano continuations as sequences of codec tokens, hierarchically — semantic tokens first, then acoustic codec tokens — which we unpack in [semantic vs acoustic tokens](/blog/machine-learning/audio-generation/semantic-vs-acoustic-tokens).
- **VALL-E** (Wang et al., 2023) reframes text-to-speech as "predict the EnCodec tokens of the target speech, conditioned on the text and a 3-second acoustic prompt of the target speaker." Zero-shot voice cloning falls out of in-context learning over codec tokens.
- **MusicGen** (Copet et al., 2023) is a single-stage transformer that autoregresses EnCodec tokens to generate music, with a clever codebook-interleaving pattern to handle the multiple codebooks per frame.

The alternative is the **continuous path**. Stop the codec *before* quantization — keep the encoder's continuous latent and skip the codebook — and you have a compact continuous representation that is the natural home for a **diffusion or flow-matching** model. This is exactly the [latent diffusion](/blog/machine-learning/image-generation/diffusion-from-first-principles) recipe from the image world, transplanted to audio: a VAE-style encoder compresses to a latent, a diffusion model learns to denoise *in that latent space*, and the decoder turns the denoised latent back into a waveform. **Stable Audio** and **AudioLDM** take this path. They do not need discreteness because diffusion models are perfectly happy generating continuous vectors; in fact discreteness would only get in their way.

![Matrix comparing raw waveform, continuous latent, and discrete codec tokens across rate, whether they are discrete, which generative model consumes them, and resulting sequence length](/imgs/blogs/neural-audio-codecs-the-tokenizer-of-sound-3.png)

So there is a fork in the road, and the quantizer is where it splits. Keep the latent continuous → feed a diffusion model. Quantize to discrete tokens → feed an autoregressive language model. The *same codec architecture* can serve either path; EnCodec's continuous encoder output is used by some diffusion systems while its quantized tokens feed the language-model systems. This is one of the most important organizing facts in audio generation, and we will return to it in [diffusion for audio](/blog/machine-learning/audio-generation/diffusion-for-audio) and [autoregressive audio models](/blog/machine-learning/audio-generation/autoregressive-audio-models-wavenet-to-audiolm). Hold onto it: **discrete for LMs, continuous for diffusion.**

A useful way to think of it: the codec is a compression engine first, and a tokenizer second. Whether you tap its *continuous* compressed latent or its *discrete* quantized tokens depends entirely on which kind of generative model you are going to put downstream. The codec does not care; the generative engine does.

## 4. The codec is the bridge — and the rest of the stack hangs off it

Step back and look at the whole audio stack, the spine of this entire series:

$$
\text{waveform} \rightarrow \underbrace{\text{codec tokens / latent}}_{\text{the bridge}} \rightarrow \text{generative model} \rightarrow \underbrace{\text{decoder / vocoder}}_{\text{back to waveform}} \rightarrow \text{waveform}.
$$

The codec is the bridge on **both ends**. Its encoder is the on-ramp from the waveform world into the sequence-model world; its decoder is the off-ramp back to a waveform. Everything in the middle — the autoregressive LM, the diffusion model, the flow-matching model — operates in the codec's representation, never touching raw samples. That is what lets the generative model be reasonable in size: it never has to model 24,000 events per second, only ~75 frames or ~600 tokens.

This bridging role is why I call the codec the **single most important enabler** of modern audio generation. You can swap the generative engine — AR, diffusion, flow — and keep the codec. You can swap the modality — speech, music, sound effects — and keep the codec idea. But take the codec away and you are back to modeling raw waveforms, which is the wall we started with. The image series had the [VAE](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) playing this exact role; the video series had a 3D-VAE. Audio's version is the neural codec, with the discrete-token twist that the image and video worlds mostly did not take.

There is one more thing the codec quietly enables: **streaming and low latency.** Because the encoder is causal convolutions (or can be made causal), a codec like Mimi (the codec inside Moshi) can encode audio as it arrives and decode it frame by frame, which is what full-duplex spoken dialogue needs. We will see that in [real-time streaming and full-duplex speech](/blog/machine-learning/audio-generation/real-time-streaming-and-full-duplex-speech). The point for now: the codec is not a passive preprocessing step. Its design — frame rate, codebook count, causality, fidelity — shapes what the whole stack can do.

## 5. The codec is the image VAE, with a discrete twist

If you came to audio from the image-generation world, the cleanest way to internalize the codec is by direct analogy to the [variational autoencoder](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) that powers latent diffusion. The parallel is almost exact, and the one place it diverges is precisely the place that defines audio generation.

In latent image diffusion, a VAE encoder compresses a $512 \times 512 \times 3$ image — about 786,000 numbers — into a latent grid of roughly $64 \times 64 \times 4 \approx 16{,}000$ numbers, a 48× spatial-plus-channel compression. The diffusion model never touches pixels; it denoises in that latent space, and the VAE decoder turns the final latent back into an image. The VAE does the dull-but-essential job of *removing the perceptually irrelevant redundancy* so the generative model can spend its capacity on structure rather than on pixel-level texture. The neural audio codec plays this role note for note: its encoder removes the sample-level redundancy of the waveform, the generative model works in the compact representation, and the decoder reconstructs. Both are convolutional autoencoders with a bottleneck. Both are trained with reconstruction plus adversarial losses (the popular image VAEs use a patch discriminator and an LPIPS perceptual loss; the codec uses waveform discriminators and a multi-scale STFT loss — different losses, same recipe). Both are *frozen* after training and reused as fixed infrastructure under many generative models.

There are two real differences, and they are instructive. The first is **dimensionality**: an image VAE compresses a 2D grid, so its convolutions are 2D and its downsampling is spatial; an audio codec compresses a 1D stream, so its convolutions are 1D and its downsampling is purely temporal. That sounds trivial but it changes the *shape* of the redundancy. Images are redundant in two spatial directions (nearby pixels look alike both horizontally and vertically); audio is redundant in one time direction but spans an enormous *dynamic range of timescales* — a single waveform contains both a 50-millisecond plosive and a 4-minute song structure, and the codec's strided convolutions have to summarize the fast detail while a downstream model handles the slow structure.

The second difference is the one that matters: **the audio codec usually quantizes; the image VAE usually does not.** Latent diffusion keeps the VAE latent *continuous* because diffusion models generate continuous vectors. Audio took a different road. Because so much of audio generation is *autoregressive language modeling* — sequence in, sequence out, one symbol at a time — the audio field leaned hard into **discrete** codec tokens, so it could use the full text-modeling toolbox. The image world has discrete-token models too ([autoregressive image models](/blog/machine-learning/image-generation/autoregressive-image-models) over VQ-GAN tokens), so the discrete path is not unique to audio — but audio is where it became the *dominant* path, because the audio-LM systems (AudioLM, VALL-E, MusicGen) are so central. So: the codec is the audio VAE, and its quantizer is the extra step that turns a continuous latent into a discrete vocabulary. Hold both halves of that and you understand where the codec sits in the family tree of generative media.

The practical consequence of the analogy is reuse of intuition. Everything you learned about VAE latents in the image series transfers: the codec is lossy, so its reconstruction caps your output quality; a too-aggressive bottleneck loses detail your generative model can never recover; the codec is trained on a distribution and degrades off it; and you freeze it and treat it as a fixed substrate. The audio-specific part — quantization, RVQ, the bitrate math — is what the rest of this codec track is about.

## 6. Inside the encoder and decoder: convolutions and receptive fields

Let us open the encoder and decoder, because the convolution mechanics are where the frame rate, the receptive field, and the streaming behavior all come from, and a little detail here pays off across the whole series.

The encoder is a stack of **residual convolution blocks** separated by **strided downsampling convolutions**. A typical EnCodec encoder looks like: an initial $1\text{D}$ convolution lifts the single waveform channel to a wide feature dimension (say 32 or 64 channels); then four downsampling stages, each a strided convolution that halves or quarters the time axis while doubling the channel count, interleaved with residual blocks that refine the features without changing the rate; then a final convolution projects to the latent dimension (128). The strides multiply: with strides $(2,4,5,8)$ the time axis shrinks by $2 \times 4 \times 5 \times 8 = 320$, and the channels grow so that *information density per frame* rises as the rate falls — exactly the trade an autoencoder bottleneck is supposed to make. The decoder is the mirror image: it starts from the quantized 128-dim latent and runs **transposed** (fractionally-strided) convolutions with the strides in reverse, $(8,5,4,2)$, each one upsampling the time axis, until it lands back at the original sample rate with a single output channel — the reconstructed waveform.

Two properties of this convolutional design matter downstream.

**Receptive field.** Each output frame of the encoder "sees" a window of input samples determined by the kernel sizes and strides of all the layers below it. EnCodec's encoder has a receptive field on the order of a few hundred milliseconds — wide enough that a single latent frame is informed by its neighborhood, which is why the codec captures local timbre and short-term structure well, and why it does *not* need to model long-range structure (that is the generative model's job). The decoder's transposed convolutions similarly give each output sample a receptive field over several latent frames, which is what lets it synthesize smooth, artifact-free transitions between frames rather than blocky seams.

**Causality and streaming.** If you constrain every convolution to look only *backward* in time (causal padding), the codec becomes **streaming-capable**: it can encode and decode audio with bounded latency as samples arrive, never waiting for future context. This is exactly what a real-time voice agent or a full-duplex dialogue model needs. The cost is a small quality hit (the encoder cannot use future context to disambiguate a frame), which is usually worth it for live applications. EnCodec offers both causal and non-causal variants; Mimi is built causal from the ground up for Moshi. The convolutional architecture is what makes this dial available — a transformer-based codec front end would be far harder to stream frame by frame.

There is also a subtle but important detail about **what the latent dimension buys you**. The encoder outputs a 128-dimensional vector per frame *before* quantization. That is a rich, continuous representation. Quantization with a single 1,024-entry codebook can only express $\log_2 1024 = 10$ bits per frame — nowhere near enough to capture a 128-dim float vector faithfully. That mismatch is the entire motivation for residual VQ: one codebook is a coarse approximation, so you stack several, each cleaning up the previous one's error, until the *sum* of their codebook vectors approximates the continuous latent well. The encoder's job is to produce a latent that quantizes *well* (entries cluster in ways a codebook can capture); the RVQ's job is to capture it at a controllable bitrate. We will derive exactly how the distortion falls as you add codebooks in [the RVQ post](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq) — the short version is that each additional codebook removes a roughly geometric fraction of the remaining error, which is why the rate-distortion curve bends the way it does and why the early codebooks matter most.

## 7. The science: deriving the compression and the bitrate

Let us make the codec's economics precise, because once you can compute a codec's bitrate you can reason about every trade-off the rest of the field argues over.

**Sample rate to frame rate.** The encoder downsamples by a fixed factor, the product of its strides. Call that the *hop* $H$ (the number of input samples per output frame). The frame rate is

$$
f_\text{frame} = \frac{f_\text{sample}}{H}.
$$

For EnCodec at 24 kHz with $H = 320$: $f_\text{frame} = 24{,}000 / 320 = 75$ frames per second. For DAC at 44.1 kHz with a hop of 512: $f_\text{frame} = 44{,}100 / 512 \approx 86$ frames per second. The frame rate is the codec's true *temporal resolution* — it is how finely the codec can place events in time. Lower frame rate means a shorter sequence (cheaper to model) but coarser timing; this is a real trade-off, not a free lunch.

**Frame rate to token rate.** If the codec uses $N_q$ codebooks per frame (residual VQ stacks several), then each frame produces $N_q$ tokens, so

$$
f_\text{token} = N_q \cdot f_\text{frame}.
$$

At 75 fps with $N_q = 8$ codebooks: $f_\text{token} = 8 \times 75 = 600$ tokens per second. That is the number that matters to the downstream language model, because that is the length of the sequence it must generate.

**Bitrate.** Each codebook holds $K$ entries, so each token carries $\log_2 K$ bits. The total bitrate is

$$
R = f_\text{frame} \cdot N_q \cdot \log_2 K \quad \text{bits per second}.
$$

With $f_\text{frame} = 75$, $N_q = 8$, and $K = 1024$ (so $\log_2 K = 10$ bits):

$$
R = 75 \times 8 \times 10 = 6{,}000 \ \text{bits/s} = 6.0 \ \text{kbps}.
$$

Six kilobits per second to represent 24 kHz audio. Compare that to the raw stream: $24{,}000 \times 16 = 384{,}000$ bits/s $= 384$ kbps. The codec is a **64-fold** compression over raw PCM, and it does it while staying decodable to audio a listener accepts. (For reference, MP3 at "good" quality is ~128–256 kbps and Opus speech can go down to ~16–24 kbps; neural codecs reach far lower bitrates at comparable or better perceptual quality, which is one of the headline results of the SoundStream and EnCodec papers.)

![Stacked diagram of the compression cascade showing raw waveform at 24000 floats per second dropping to 75 frames per second after the encoder, then to 75 or 600 tokens per second depending on codebook count, landing in transformer range](/imgs/blogs/neural-audio-codecs-the-tokenizer-of-sound-2.png)

The bitrate formula is the lever every codec paper pulls. You can trade quality for bitrate three ways:

- **Fewer codebooks** ($N_q$): drop from 8 to 4 and you halve the bitrate to 3 kbps and halve the token sequence length, at the cost of coarser reconstruction (the later residual codebooks were correcting fine errors, so you lose high-frequency detail). RVQ is specifically designed so you can *drop trailing codebooks at inference time* — train at 8, decode with 4 — which is why one trained EnCodec serves a whole range of bitrates. We will derive that rate-distortion curve in the RVQ post.
- **Smaller codebooks** ($K$): fewer entries, fewer bits per token, but the quantization is coarser.
- **Lower frame rate** ($f_\text{frame}$): a bigger hop means fewer frames per second — shorter sequences, lower bitrate — but worse temporal precision (transients smear).

#### Worked example: a bitrate ladder

Suppose you are choosing an EnCodec configuration for a music model and you want to know your options. Fix 24 kHz, 75 fps, $K = 1024$ ($\log_2 K = 10$ bits). The bitrate ladder is $R = 75 \times N_q \times 10$:

| Codebooks $N_q$ | Tokens/s | Bitrate | Sequence for 30 s | Typical use |
|---|---|---|---|---|
| 1 | 75 | 0.75 kbps | 2,250 tokens | Lowest fidelity, longest context |
| 2 | 150 | 1.5 kbps | 4,500 tokens | Coarse but recognizable |
| 4 | 300 | 3.0 kbps | 9,000 tokens | Common LM-gen sweet spot |
| 8 | 600 | 6.0 kbps | 18,000 tokens | High fidelity, heavier sequence |
| 16 | 1,200 | 12.0 kbps | 36,000 tokens | Near-transparent, expensive |

A 30-second generation at $N_q = 4$ is 9,000 tokens — squarely in transformer range — at 3 kbps, which EnCodec reconstructs at quite acceptable music quality. Bump to $N_q = 8$ and you double both the fidelity headroom and the sequence length. This table *is* the design decision: more codebooks means better audio and a longer, costlier sequence for the LM to generate. MusicGen, for what it is worth, uses 4 codebooks at 50 fps for its 32 kHz EnCodec, landing around 2 kbps, and handles the 4 codebooks per frame with an interleaving pattern rather than emitting them as a flat 4× longer sequence.

## 8. The training losses: a codec is a GAN vocoder

A naive autoencoder trained with only mean-squared error on the waveform produces *muffled, lifeless* audio. The reason is the same as why MSE produces blurry images: when the target is multimodal (many plausible waveforms could explain the same coarse latent), MSE predicts their average, and the average of many sharp waveforms is a smear. Audio is brutal here because our ears are exquisitely sensitive to the high-frequency detail that averaging destroys. So a good codec is trained with a *combination* of losses, and the decisive ingredient is **adversarial training** — the codec's decoder is, functionally, a GAN vocoder.

There are four loss families, summed with weights:

**1. Time-domain reconstruction.** An $L_1$ or $L_2$ loss directly between the reconstructed waveform $\hat{x}$ and the original $x$:

$$
\mathcal{L}_\text{time} = \lVert x - \hat{x} \rVert_1.
$$

This anchors the overall shape but, alone, gives the blurry average.

**2. Multi-scale spectral (STFT) loss.** Compute the [short-time Fourier transform](/blog/machine-learning/audio-generation/the-mathematics-of-audio-signals) of both $x$ and $\hat{x}$ at *several* window sizes, and penalize the difference in their magnitude spectrograms. A small window captures fine temporal detail (transients), a large window captures fine frequency detail (pitch); summing across scales forces the codec to get both right:

$$
\mathcal{L}_\text{spec} = \sum_{s} \Big( \lVert |S_s(x)| - |S_s(\hat{x})| \rVert_1 + \lVert \log|S_s(x)| - \log|S_s(\hat{x})| \rVert_1 \Big),
$$

where $S_s$ is the STFT at scale $s$ (different `n_fft`/`hop_length`). The log term matters because hearing is roughly logarithmic in amplitude — a 3 dB error is perceptually similar whether the signal is loud or quiet. This is the perceptual workhorse of the loss.

**3. Adversarial loss.** One or more *discriminator* networks are trained to distinguish real waveforms from the codec's reconstructions, and the codec is trained to fool them. The discriminators are the same kind used in GAN vocoders like HiFi-GAN: a **multi-period discriminator** (MPD) that reshapes the 1D waveform into 2D at several periods to catch periodic structure (crucial for the harmonic structure of voiced speech and musical tones), and a **multi-scale discriminator** (MSD) that looks at the waveform at several downsampled resolutions. The adversarial loss is what pushes the decoder to synthesize *crisp, realistic* high-frequency detail instead of an average — it is the single biggest reason neural codecs sound good. We cover these discriminators in depth in [GAN vocoders: HiFi-GAN and fast synthesis](/blog/machine-learning/audio-generation/gan-vocoders-hifi-gan-and-fast-synthesis).

**4. Feature-matching loss.** A stabilizer for the adversarial training: match the *intermediate activations* of the discriminator between real and reconstructed audio, not just its final real/fake verdict. This gives the generator a smoother, denser gradient than the adversarial term alone and is standard in HiFi-GAN-style training.

![Graph showing the reconstructed and original waveforms feeding a time-domain loss, a multi-scale STFT loss, and discriminator losses, which merge into a single weighted total loss](/imgs/blogs/neural-audio-codecs-the-tokenizer-of-sound-4.png)

Plus the **quantizer's own losses** — the commitment loss and codebook loss that train the codebook and keep the encoder outputs near their assigned entries (RVQ post). The grand total is roughly

$$
\mathcal{L} = \lambda_t \mathcal{L}_\text{time} + \lambda_s \mathcal{L}_\text{spec} + \lambda_g \mathcal{L}_\text{adv} + \lambda_f \mathcal{L}_\text{feat} + \lambda_q \mathcal{L}_\text{quant}.
$$

Two things are worth absorbing here. First, **the codec is trained exactly like a GAN vocoder** — the decoder *is* a neural vocoder, and the only thing distinguishing a codec from a standalone vocoder is the encoder+quantizer front end that produces the tokens. That is why the codec lineage (SoundStream, EnCodec, DAC) and the vocoder lineage (HiFi-GAN, BigVGAN, Vocos) share so much DNA. Second, **none of these losses are perceptual rocket science individually** — they are the same multi-resolution STFT loss and HiFi-GAN discriminators the vocoder world already had. The codec's contribution is the *quantized bottleneck* in the middle, which turns a vocoder into a tokenizer.

## 9. The practical flow: encode and decode with EnCodec

Enough theory. Let us actually run a codec, encode audio to integer tokens, decode it back, and measure how much we lost. The cleanest path is 🤗 `transformers`, which ships `EncodecModel` and an `AutoProcessor`. We will use the 24 kHz EnCodec checkpoint.

```bash
pip install "transformers>=4.40" torch torchaudio soundfile
```

First, load a model and a piece of audio. We will use a sentence-length clip — keep returning to the running example of synthesizing or reconstructing a short utterance.

```python
import torch
import torchaudio
from transformers import EncodecModel, AutoProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

# 24 kHz EnCodec. There is also facebook/encodec_48khz for stereo music.
model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device).eval()
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

# Load audio and resample to the codec's native rate.
wav, sr = torchaudio.load("speech.wav")          # wav: [channels, samples]
wav = wav.mean(dim=0, keepdim=True)              # mono
target_sr = processor.sampling_rate              # 24000
if sr != target_sr:
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)

duration_s = wav.shape[-1] / target_sr
print(f"loaded {duration_s:.2f} s of audio at {target_sr} Hz "
      f"({wav.shape[-1]} samples)")
```

Now encode to discrete codes. The processor handles normalization and batching; `model.encode` returns the integer codes. We pass a target `bandwidth` (in kbps), which EnCodec maps to a number of codebooks — this is the bitrate lever from the science section made concrete.

```python
inputs = processor(raw_audio=wav.squeeze(0).numpy(),
                   sampling_rate=target_sr,
                   return_tensors="pt").to(device)

with torch.no_grad():
    # bandwidth picks how many RVQ codebooks to use.
    # 24 kHz EnCodec supports 1.5, 3.0, 6.0, 12.0, 24.0 kbps.
    encoded = model.encode(inputs["input_values"],
                           inputs["padding_mask"],
                           bandwidth=6.0)

codes = encoded.audio_codes        # [batch, n_q, n_frames]
codes = codes[0, 0]                # drop batch and chunk dims -> [n_q, n_frames]
n_q, n_frames = codes.shape
print(f"codes shape: [n_q={n_q}, n_frames={n_frames}]")
print(f"frame rate: {n_frames / duration_s:.1f} frames/s")
print(f"token rate: {n_q * n_frames / duration_s:.1f} tokens/s")
print(f"value range: {codes.min().item()}..{codes.max().item()} "
      f"(codebook size 1024 -> 0..1023)")
```

For a 24 kHz, 6 kbps configuration you will see `n_q = 8` codebooks, a frame rate near **75 frames/s**, a token rate near **600 tokens/s**, and code values in `0..1023`. Those integers are the tokens — *this is the sequence a language model would generate.* Compute the bitrate to confirm the formula from the science section:

```python
import math

codebook_size = model.config.codebook_size      # 1024
bits_per_token = math.log2(codebook_size)        # 10.0
frames_per_s = n_frames / duration_s
bitrate_kbps = frames_per_s * n_q * bits_per_token / 1000
print(f"bitrate = {frames_per_s:.1f} fps x {n_q} codebooks "
      f"x {bits_per_token:.0f} bits = {bitrate_kbps:.2f} kbps")
# bitrate = 75.0 fps x 8 codebooks x 10 bits = 6.00 kbps
```

The number you print should land right on 6.0 kbps — the codec's bitrate is fully determined by frame rate, codebook count, and codebook size, exactly as the formula says. Now decode the tokens back to a waveform and measure how much we lost.

```python
with torch.no_grad():
    decoded = model.decode(encoded.audio_codes,
                           encoded.audio_scales,
                           inputs["padding_mask"])
recon = decoded.audio_values[0].cpu()            # [1, samples]

# Align lengths (decoder may emit a few extra samples).
n = min(wav.shape[-1], recon.shape[-1])
x, x_hat = wav[..., :n], recon[..., :n]

mse = torch.mean((x - x_hat) ** 2).item()
# Signal-to-noise ratio of the reconstruction, in dB.
sig = torch.sum(x ** 2).item()
err = torch.sum((x - x_hat) ** 2).item()
snr_db = 10 * math.log10(sig / (err + 1e-12))
print(f"reconstruction MSE: {mse:.2e}")
print(f"reconstruction SNR: {snr_db:.1f} dB  (higher is better)")

torchaudio.save("recon_6kbps.wav", x_hat, target_sr)
```

A few practical notes that save you debugging time:

- **Codes are plain integers.** `encoded.audio_codes` is a `LongTensor` of shape `[batch, n_q, n_frames]`. You can save them, ship them to a language model as token ids, sample new ones, and feed them back to `model.decode`. That round trip — encode → (model does something to the tokens) → decode — is the entire VALL-E/MusicGen loop.
- **The `padding_mask` matters.** EnCodec chunks long audio; passing the mask keeps the reconstruction aligned. Forget it and you get subtle length and boundary glitches.
- **Numerical SNR is not perceptual quality.** A codec can have a mediocre SNR yet sound excellent, because the adversarial loss optimized for *perceptual* realism, not sample-exact reconstruction. Trust your ears, or a perceptual metric like [FAD or a CLAP score](/blog/machine-learning/audio-generation/audio-quality-metrics), over raw MSE. We will return to this.

#### Worked example: hearing the bitrate ladder

Run the encode/decode loop above at `bandwidth=1.5`, `3.0`, `6.0`, and `12.0` kbps on the same speech clip and listen, while printing the codes' shape each time:

| `bandwidth` | `n_q` | Tokens/s | What you hear |
|---|---|---|---|
| 1.5 kbps | 2 | 150 | Intelligible but metallic, thin highs |
| 3.0 kbps | 4 | 300 | Clearly intelligible, slight roughness |
| 6.0 kbps | 8 | 600 | Natural, hard to fault on speech |
| 12.0 kbps | 16 | 1,200 | Near-transparent |

The same trained EnCodec gives you the whole ladder by selecting how many RVQ codebooks to use — you do not retrain. That is the practical payoff of residual quantization, and it is why "what bitrate?" is a *decode-time* decision, not a training-time one. The lesson for downstream modeling: pick the lowest $N_q$ whose reconstruction clears your quality bar, because every extra codebook is a proportionally longer sequence for your generative model to produce.

### The generation loop: tokens in, tokens out, sound out

The encode/decode round trip above is the *infrastructure*. The actual point of a codec is to put a generative model in the middle of it. The loop, in full, is: encode a prompt to tokens → a language model continues or generates new tokens → decode those tokens to a waveform. To make this concrete without pulling in a full audio-LM, here is the skeleton — encode a prompt, then (where a real system would call a trained transformer) simply pass the tokens through and decode, which proves the round trip a generative model plugs into.

```python
# 1. Encode a short prompt to codec tokens (the "context" for the LM).
with torch.no_grad():
    prompt = model.encode(inputs["input_values"],
                          inputs["padding_mask"],
                          bandwidth=6.0)
prompt_codes = prompt.audio_codes        # [1, 1, n_q, n_frames]

# 2. A real audio LM (VALL-E / MusicGen) would now autoregress new codes:
#    logits = transformer(prompt_codes); next_code = sample(logits) ...
#    Here we just reuse the prompt codes to demonstrate the decode path.
generated_codes = prompt_codes           # placeholder for LM output

# 3. Decode whatever codes the model produced back to a waveform.
with torch.no_grad():
    out = model.decode(generated_codes,
                       prompt.audio_scales,
                       inputs["padding_mask"])
torchaudio.save("generated.wav", out.audio_values[0].cpu(), target_sr)
```

The only thing a real system adds is step 2: a transformer trained to predict the next codec token, conditioned on text (MusicGen), a phoneme sequence plus a voice prompt (VALL-E), or earlier tokens (AudioLM). The codec's encoder gives the model its training targets and its prompt; the codec's decoder turns the model's output back into sound. *That is the whole architecture of codec-based audio generation*, and the codec is the half of it that never changes when you swap modalities. Everything in [autoregressive audio models](/blog/machine-learning/audio-generation/autoregressive-audio-models-wavenet-to-audiolm) is a fleshing-out of step 2.

### Measuring codebook utilization (the codec-training metric that matters)

If you ever train your own codec — or just want to diagnose a checkpoint — the single most informative number is **codebook utilization**: what fraction of each codebook's entries actually get used. A healthy codec uses nearly all entries; a collapsed one uses a handful, which means your *effective* bitrate is far below your *nominal* bitrate and reconstruction quality is paying for codebook entries you never spend. You can estimate it directly from the codes:

```python
import torch

def codebook_utilization(codes, codebook_size=1024):
    # codes: LongTensor [n_q, n_frames]
    n_q = codes.shape[0]
    for q in range(n_q):
        used = torch.unique(codes[q]).numel()
        frac = used / codebook_size
        print(f"codebook {q}: {used:4d}/{codebook_size} entries used "
              f"({frac:5.1%})")

codebook_utilization(codes)          # codes from the encode step above
```

On a single short clip you will see low utilization simply because one clip cannot exercise a 1,024-entry codebook — you need to aggregate over a corpus. But the *pattern* is telling: the first codebook (coarsest) is usually used most heavily, and later residual codebooks should still show broad usage. If a late codebook collapses to a few entries during training, that codebook is wasted and the codec's rate-distortion curve flattens early. DAC's headline improvement over EnCodec was precisely raising utilization across all codebooks, which is why it reconstructs better at the same nominal bitrate. We measure and fight collapse properly in [the RVQ post](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq); the habit to build now is *always watch utilization*, because nominal bitrate lies when codebooks collapse.

## 10. Results: representations side by side

Let us put the three ways of representing audio next to each other with real rate numbers, so the trade-off is concrete. The point of this table is the *sequence length* and *what consumes it* columns — those are what decide your architecture.

| Representation | Rate (24 kHz) | Discrete? | 10 s sequence | Consumed by | Why |
|---|---|---|---|---|---|
| Raw waveform | 24,000 /s | No (float) | 240,000 | Vocoder / direct synthesis | Too long for any sequence model |
| Mel-spectrogram | ~86 frames/s (varies) | No (float grid) | ~860 frames | Vocoder, some diffusion | Compact, but throws away phase |
| Continuous codec latent | ~75 frames/s | No (float vec) | 750 frames | Diffusion / flow | Short, full info, continuous |
| Discrete codec tokens | ~75 frames/s × $N_q$ | Yes (codebook ids) | 750 × $N_q$ | AR language model | Short, integer, LM-friendly |

![Before-and-after comparison showing that generating one second of audio means 24000 autoregressive steps as raw samples versus about 600 steps as codec tokens](/imgs/blogs/neural-audio-codecs-the-tokenizer-of-sound-5.png)

The jump from 240,000 to 750 (or 750 × a small $N_q$) is the headline. It is why audio generation went from "research curiosity that takes minutes to make seconds of sound" to "ship a real-time voice assistant." And notice that the **mel-spectrogram** — the workhorse of the classic [Tacotron-to-VITS](/blog/machine-learning/audio-generation/text-to-speech-from-tacotron-to-vits) TTS pipeline — is *also* a compact representation, but it is not discrete and, more importantly, it throws away **phase** (it keeps only the magnitude of each STFT bin), which is why a mel pipeline always needs a vocoder to *invent* a plausible phase when reconstructing the waveform. The neural codec keeps everything needed to reconstruct the waveform — phase included, implicitly, in the decoder — which is part of why codec-based systems can hit such high fidelity. The phase problem and why mel discards it is covered in [representing sound](/blog/machine-learning/audio-generation/representing-sound-waveforms-spectrograms-and-perception); the takeaway here is that the codec sidesteps it by learning a decoder that synthesizes phase directly.

#### Worked example: a compression accounting, end to end

Take ten seconds of 24 kHz mono speech and trace the bytes:

- **Raw PCM (16-bit):** $24{,}000 \times 10 \times 2 = 480{,}000$ bytes ≈ 469 KB.
- **EnCodec at 6 kbps:** $6{,}000 \times 10 / 8 = 7{,}500$ bytes ≈ 7.3 KB. A **64×** reduction.
- **As a token sequence for an LM:** $75 \times 10 \times 8 = 6{,}000$ tokens. With a codebook of 1,024 entries, that is a vocabulary the size of a small BPE tokenizer and a sequence shorter than a single page of text. A 6.7-billion-parameter language model would not blink at it.

That last line is the whole point. **Ten seconds of speech becomes a 6,000-token sequence over a 1,024-symbol vocabulary** — a problem a language model treats exactly like a paragraph. The codec did the hard part of turning continuous sound into discrete language; the language model just does what it always does.

## 11. The lineage: SoundStream → EnCodec → DAC

The neural codec did not appear fully formed. It is worth knowing the three landmarks because the whole rest of the codec track refers back to them, and because the *trend* across them tells you what "better" means for a codec.

![Timeline from WaveNet through SoundStream, EnCodec, and DAC to Mimi, marking the shift from raw-sample autoregression to learned residual-VQ codecs and then to streaming codecs](/imgs/blogs/neural-audio-codecs-the-tokenizer-of-sound-7.png)

**SoundStream** (Zeghidour et al., Google, 2021) is the origin. It introduced the now-standard recipe: a fully convolutional encoder–decoder, a **residual vector quantizer** in the bottleneck, and adversarial + reconstruction training. SoundStream showed you could hit *transparent-ish* quality at a few kbps and — critically — that you could train a single model that operates across a *range* of bitrates by varying how many RVQ codebooks you use at inference. It targeted 24 kHz and demonstrated the codec could carry both speech and general audio. SoundStream is where "learned codec with discrete tokens" became a thing.

**EnCodec** (Défossez et al., Meta, 2022) scaled and hardened the recipe, and — just as importantly — it became the **default tokenizer for the audio-LM era.** EnCodec brought a multi-scale STFT loss, a careful balance of the loss terms (a loss balancer that keeps any single term from dominating), an optional small transformer entropy model to squeeze the bitrate further, and clean 24 kHz and 48 kHz (stereo) checkpoints with selectable bitrate. When VALL-E and MusicGen needed discrete audio tokens, they reached for EnCodec. If you use one codec in 🤗 `transformers`, it is EnCodec, which is why this post's code uses it.

**DAC** — the **Descript Audio Codec** (Kumar et al., 2023) — pushed *reconstruction fidelity* up at the same low bitrate. Its headline contributions were practical fixes to RVQ training: improved codebook learning (better codebook usage, less of the **codebook collapse** that plagues naive VQ — where most codebook entries go unused), a snake activation for better periodic modeling, and careful discriminator design. DAC operates at 44.1 kHz and reconstructs music and general audio at a quality that, at ~8 kbps, many listeners cannot distinguish from the original. DAC is the answer to "I need the best reconstruction quality I can get."

| Codec | Year · Lab | Sample rate | Frame rate | Bitrate range | Headline |
|---|---|---|---|---|---|
| SoundStream | 2021 · Google | 24 kHz | 75 fps | 3–18 kbps | First learned RVQ codec |
| EnCodec | 2022 · Meta | 24 / 48 kHz | 75 fps | 1.5–24 kbps | Default audio-LM tokenizer |
| DAC | 2023 · Descript | 44.1 kHz | ~86 fps | ~8 kbps | Highest reconstruction fidelity |
| Mimi | 2024 · Kyutai | 24 kHz | 12.5 fps | ~1.1 kbps | Streaming, semantic-distilled |

![Matrix of the codec lineage comparing SoundStream, EnCodec, and DAC across sample rate, frame rate, bitrate, and the contribution each one added](/imgs/blogs/neural-audio-codecs-the-tokenizer-of-sound-6.png)

The trend across the lineage is the thing to remember: **bitrate held roughly constant (a few kbps) while reconstruction quality climbed and special-purpose variants appeared.** That progression is what made codec tokens trustworthy enough to *generate* — if your codec's reconstruction is noticeably worse than the original, then even a perfect generative model produces noticeably-worse audio, because the decoder is the ceiling. A better codec raises the ceiling for *everything* downstream. And note Mimi at the bottom: a 2024 streaming codec running at a startlingly low **12.5 frames per second** with semantic distillation baked in, built so Moshi can do full-duplex conversation. We give EnCodec, DAC, and Mimi their full treatment in [the modern codec post](/blog/machine-learning/audio-generation/encodec-dac-and-the-modern-codec).

## 12. The engineering decision: which codec, which path

Now the part you actually act on. You are building an audio system. The codec choice and the discrete-vs-continuous choice are among the first and most consequential decisions you make. Here is how I reason about it.

**Start from the generative engine you intend to use.**

- **Autoregressive language model** (you want VALL-E-style voice cloning, MusicGen-style music, in-context conditioning, or you just want to reuse your LLM infrastructure): you need **discrete tokens**. Use EnCodec (well-supported, the de facto standard) or DAC (if you need higher fidelity and can afford the slightly higher token rate). The number of codebooks $N_q$ is your main dial: more codebooks = better audio but a longer sequence, and the multiple codebooks per frame need a handling strategy (flattening, interleaving, or a delay pattern — MusicGen's territory).
- **Diffusion or flow-matching model** (you want Stable Audio-style high-fidelity long-form generation, smooth continuous control, or few-step sampling): use the codec's **continuous latent** and skip quantization, or use a VAE trained for this purpose. Diffusion does not want discreteness.

**Then pick the bitrate / codebook count by your quality bar.** Encode a representative clip at each bandwidth, decode, and listen (or measure FAD against your reference set). Choose the **lowest** $N_q$ that clears the bar, because every extra codebook is a proportionally longer and more expensive sequence for your generative model. For speech, 6 kbps EnCodec is usually overkill quality and 3 kbps is often fine; for music you tend to want more.

![Before-and-after diagram showing the quantizer as a fork where keeping the continuous latent feeds a diffusion model and quantizing to discrete tokens feeds an autoregressive language model](/imgs/blogs/neural-audio-codecs-the-tokenizer-of-sound-8.png)

**When NOT to reach for a codec at all:**

- **Pure vocoding** (you already have a mel-spectrogram and just need a waveform): you do not need a codec's encoder/quantizer. Use a dedicated vocoder — HiFi-GAN, BigVGAN, or Vocos — which is faster and simpler. The codec is for when you need to *tokenize* audio for a sequence model, not merely invert a spectrogram.
- **Classic mel-based TTS** (Tacotron/FastSpeech/VITS-style): these model a mel-spectrogram, not codec tokens, and pair it with a vocoder. A codec is the *newer* (VALL-E-style) way to do TTS, not the only way, and the mel pipeline is still excellent and often cheaper. Don't bolt a codec onto a system that doesn't need discrete tokens.
- **Lossless or archival** needs: neural codecs are *lossy* by design. If you need bit-exact audio, use FLAC, not a neural codec.

**A stress test, because the edges are where you learn the codec's limits.**

- *What happens at a very low bitrate?* Drop to 1.5 kbps (2 codebooks) and the codec keeps speech intelligible but strips the highs — sibilants get metallic, music loses air and sparkle. The later residual codebooks were carrying exactly that high-frequency detail; remove them and it is gone. This is a graceful, predictable degradation, which is one of RVQ's virtues.
- *What happens to out-of-domain audio?* A codec trained mostly on speech and music will mangle, say, a dense orchestral fortissimo or a noisy field recording it never saw — quantization error spikes where the codebook has no nearby entry. Codecs are only as good as their training distribution, same as any learned model.
- *When the codec is the bottleneck, not the model.* Sometimes your generated tokens are great but the audio sounds off — that is the **decoder** showing its ceiling. No generative model can exceed the reconstruction quality of the codec it generates tokens for. If your output quality is capped, upgrade the codec (EnCodec → DAC) before you blame the language model. I have spent more than one afternoon tuning a generation model when the real fix was a better codec underneath it.
- *When codebook collapse bites.* If you train your own codec and most of the codebook entries go unused (collapse), your effective bitrate is far below your nominal bitrate and reconstruction suffers. DAC's main contributions were fixes for exactly this. Watch codebook **utilization** during training; it is the single most informative codec-training metric. We dig into collapse in the [RVQ post](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq).
- *When the codec's own latency dominates.* For a real-time system, the codec's encode and decode passes are part of your latency budget, and they are not free. A convolutional codec decoder is fast — on a modern GPU, decoding runs at a real-time factor well below 1 (it synthesizes faster than the audio plays), often by an order of magnitude, and even on CPU a small codec can hit real-time for speech. But it is not zero, and for full-duplex dialogue every millisecond counts: the codec, the generative model, and any network hop all share a budget of roughly 200 ms before a conversation feels laggy. This is part of why streaming codecs like Mimi push the frame rate so low (12.5 fps) and keep the decoder light — fewer frames per second means fewer decode steps per second of audio. If you are serving offline batch generation, codec latency is a rounding error; if you are serving a live voice agent, profile the codec alongside the model, because either one can be the bottleneck. We quantify real-time factors for the whole stack in the [streaming post](/blog/machine-learning/audio-generation/real-time-streaming-and-full-duplex-speech).

The throughline across all of these stress tests is the same lesson: the codec is a *learned, lossy, distribution-bound* component, and treating it as a magic lossless pipe will eventually surprise you. It degrades gracefully at low bitrate, it has a quality ceiling that caps everything downstream, it can collapse during training, it is only as good as its data, and it costs real time at inference. Know those five edges and you will reach for the right codec — and the right bitrate — without learning each one the hard way on a shipping deadline.

## 13. Why "tokenizer of sound" is the right frame

I want to close the conceptual arc by defending the title, because the metaphor is not decoration — it is the most useful way to hold the whole field in your head.

A text tokenizer has three properties that make it useful: it is **lossy-but-faithful** (it commits to a fixed vocabulary, but reconstructs the original string exactly via detokenization), it is **discrete** (integers from a fixed set), and it **shortens** the sequence (BPE merges characters into subwords, cutting length). A neural audio codec has the same three properties with one relaxation: it is **lossy-but-perceptually-faithful** (it cannot reconstruct the waveform bit-exactly, but it reconstructs audio your ears accept), it is **discrete** (integers from a learned codebook), and it **dramatically shortens** the sequence (24,000/s → ~600/s). The relaxation — perceptual rather than exact reconstruction — is forced by the fact that audio is continuous and a finite codebook cannot represent every waveform. But everything else lines up.

And the payoff is the same as text tokenization's payoff: **it lets a sequence model operate on a sane representation.** A language model does not model raw bytes of UTF-8 (well, mostly not); it models tokens. An audio language model does not model raw samples; it models codec tokens. The codec is to a waveform what a BPE tokenizer is to a string. Once you see it that way, the audio-LM papers stop being mysterious — VALL-E is "GPT for EnCodec tokens conditioned on text and a voice prompt," MusicGen is "GPT for EnCodec tokens conditioned on a text description," AudioLM is "GPT for hierarchical audio tokens." They are all the same move: tokenize the sound, then model the tokens.

That is also why the codec is the *enabler*. Improve the tokenizer and you improve everything that models its tokens — every TTS system, every music model, every sound-effect generator that uses it. It is the leveraged component of the whole audio stack. When you build an end-to-end system in the [capstone](/blog/machine-learning/audio-generation/building-an-audio-generation-stack), the codec is the first brick you lay, and the one whose quality silently caps the whole edifice.

The next posts make good on the promissory notes scattered through this one. [Residual vector quantization](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq) derives the RVQ machinery, the straight-through gradient, the rate-distortion curve, and codebook collapse. [EnCodec, DAC, and the modern codec](/blog/machine-learning/audio-generation/encodec-dac-and-the-modern-codec) goes deep on the three landmark designs and what each one actually changed. [Semantic vs acoustic tokens](/blog/machine-learning/audio-generation/semantic-vs-acoustic-tokens) explains the two-token paradigm — why AudioLM and VALL-E pair *semantic* tokens (from a self-supervised model like HuBERT, capturing content) with *acoustic* tokens (from a codec, capturing fine detail) — and how they compose. With those three, you will understand the codec layer of the audio stack completely.

## Case studies: real numbers from the literature

A few concrete results from the papers, so the claims above are anchored. Where a number is approximate or I am recalling an order of magnitude, I say so — never trust a precise figure you cannot cite.

**EnCodec at low bitrate beats Opus and MP3 perceptually.** The EnCodec paper (Défossez et al., 2022) reports that at 24 kHz, EnCodec at ~6 kbps matches or exceeds the perceptual quality (in MUSHRA listening tests) of much higher-bitrate classical codecs, and that even at 1.5–3 kbps it stays usable. The headline is the *bitrate-quality* trade: neural codecs reach acceptable quality at bitrates where classical codecs fall apart, because the decoder *learns* to synthesize plausible detail rather than transmitting it. The exact MUSHRA scores depend on the test set and the comparison points; the robust takeaway is "neural codec ≈ classical codec at a fraction of the bitrate."

**DAC's reconstruction fidelity at 44.1 kHz.** The Descript Audio Codec paper (Kumar et al., 2023) reports state-of-the-art reconstruction across speech, music, and general audio at ~8 kbps for 44.1 kHz audio, with substantially improved codebook utilization over prior RVQ codecs (close to full usage, versus the partial collapse common before). The practical upshot reported by users is that DAC reconstructions are frequently indistinguishable from the source on casual listening — which is exactly the property you want from a tokenizer you intend to generate over.

**SoundStream's single-model bitrate scalability.** The SoundStream paper (Zeghidour et al., 2021) demonstrated a single model spanning roughly 3–18 kbps by selecting the number of RVQ codebooks at inference, with quality degrading gracefully as you drop codebooks. This *scalability* property — train once, decode at many bitrates — is inherited by EnCodec and DAC and is the reason "bandwidth" is a runtime argument in the code above.

**Codec quality caps generative quality (the system-level result).** Across VALL-E, MusicGen, and AudioLM, the codec is the reconstruction ceiling: the generated audio can never sound *better* than the codec's own reconstruction of real audio at the same bitrate. This is not a single-paper number but a structural fact that every codec-based system inherits, and it is why the codec lineage's fidelity gains (SoundStream → EnCodec → DAC) directly lift the quality of everything built on top. When MusicGen sounds better at higher bitrate, a large part of that is simply EnCodec reconstructing better with more codebooks.

**Mimi's extreme frame rate for streaming.** The Moshi/Mimi work (Kyutai, 2024) reports a streaming codec at a remarkably low 12.5 frames per second around 1.1 kbps, with semantic information distilled into the first codebook so the downstream model gets content and acoustics in one stream — a design tuned for the low-latency, full-duplex conversation Moshi targets. The lesson: codec design is not one-size-fits-all; a streaming-dialogue codec makes different trade-offs (very low frame rate, semantic distillation) than a high-fidelity music codec (higher rate, more codebooks).

## When to reach for this (and when not to)

A decisive summary, because the whole point of understanding the codec is to make the right call.

- **Reach for a discrete codec** (EnCodec/DAC tokens) when your generative engine is an **autoregressive language model**: TTS with voice cloning (VALL-E-style), music generation (MusicGen-style), any audio-LM. You want the LM mechanics — softmax, cross-entropy, in-context conditioning — and discreteness is the price of admission.
- **Reach for the continuous latent** (skip quantization) when your engine is **diffusion or flow matching**: Stable Audio, AudioLDM-style high-fidelity long-form generation. Diffusion is happiest on continuous vectors; quantization would only hurt.
- **Use DAC over EnCodec** when reconstruction fidelity is the binding constraint and you can afford the token rate — music, high-quality content, anything where listeners will scrutinize.
- **Use EnCodec** when you want the best-supported, most-documented option that every audio-LM example already uses, especially in 🤗 `transformers`.
- **Use a streaming codec** (Mimi-class) when latency and full-duplex conversation dominate; accept the lower fidelity that a 12.5 fps codec implies.
- **Do NOT use a codec** when you only need to invert a mel-spectrogram — use a vocoder (HiFi-GAN/BigVGAN/Vocos), which is faster and simpler. Do NOT use a codec for lossless/archival audio — it is lossy by design; use FLAC. Do NOT bolt a codec onto a classic mel TTS pipeline that does not need discrete tokens.
- **Pick the lowest codebook count that clears your quality bar.** Every extra RVQ codebook is a proportionally longer, costlier sequence for your generative model. Measure, listen, then choose.

## Key takeaways

- A 24 kHz waveform is 24,000 floats per second — far too long and too redundant for a sequence model to generate directly. The codec is the fix.
- A neural codec has three blocks: an **encoder** (strided 1D convs, downsamples ~320× to ~75 frames/s), a **quantizer** (snaps each frame to learned codebook tokens), and a **decoder** (transposed convs, GAN-trained, reconstructs the waveform).
- **Discrete tokens let a language model generate audio like text** — softmax over a codebook, cross-entropy, in-context conditioning. This is the VALL-E / MusicGen / AudioLM paradigm. A **continuous latent** (skip quantization) feeds diffusion instead. *Discrete for LMs, continuous for diffusion.*
- Bitrate is fully determined: $R = f_\text{frame} \times N_q \times \log_2 K$. At 75 fps, 8 codebooks, 1,024-entry codebooks, that is 6 kbps — a 64× compression over raw 16-bit PCM.
- The codec is trained like a **GAN vocoder**: time-domain loss + multi-scale STFT loss + adversarial and feature-matching losses from waveform discriminators (plus the quantizer's commitment loss). The adversarial term is why it sounds crisp instead of muffled.
- The lineage **SoundStream → EnCodec → DAC** held bitrate near a few kbps while reconstruction quality climbed; the codec's reconstruction is the ceiling for everything generated over its tokens.
- Pick the lowest codebook count that clears your quality bar — fidelity costs sequence length. And remember the codec, not the generative model, is often the real bottleneck.
- The right mental frame is **tokenizer of sound**: the codec is to a waveform what a BPE tokenizer is to a string, with perceptual rather than exact reconstruction.

## Further reading

- **Zeghidour, Luebs, Omran, Skoglund, Tagliasacchi — "SoundStream: An End-to-End Neural Audio Codec" (2021)** — the origin of the learned RVQ codec and bitrate-scalable single-model decoding.
- **Défossez, Copet, Synnaeve, Adi — "High Fidelity Neural Audio Compression" (EnCodec, 2022)** — the codec that became the default audio-LM tokenizer; multi-scale STFT loss, loss balancing, optional entropy coding.
- **Kumar, Seetharaman, Luebs, Kumar, Kumar — "High-Fidelity Audio Compression with Improved RVQGAN" (Descript Audio Codec, 2023)** — fixes for codebook collapse and the highest-fidelity open codec.
- **Borsos, Marinier, Vincent, et al. — "AudioLM: a Language Modeling Approach to Audio Generation" (2022)** — the audio-language-model paradigm over codec and semantic tokens.
- **Wang, Chen, Wu, et al. — "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers" (VALL-E, 2023)** — TTS as codec-token language modeling with 3-second voice cloning.
- **Copet, Kreuk, Gat, et al. — "Simple and Controllable Music Generation" (MusicGen, 2023)** — single-stage codec-LM music generation with codebook interleaving.
- **Défossez, Mazaré, Orsini, et al. — "Moshi: a speech-text foundation model for real-time dialogue" (2024)** — the Mimi streaming codec and full-duplex conversation.
- 🤗 `transformers` audio docs — `EncodecModel` and `AutoProcessor` usage, the code in this post.
- Within this series: [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard) (the foundation), [representing sound: waveforms, spectrograms, and perception](/blog/machine-learning/audio-generation/representing-sound-waveforms-spectrograms-and-perception), [residual vector quantization (RVQ)](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq), [EnCodec, DAC, and the modern codec](/blog/machine-learning/audio-generation/encodec-dac-and-the-modern-codec), and the [capstone: building an audio generation stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack). The codec is the audio analogue of the image [variational autoencoder](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch).
