---
title: "Music Generation: MusicLM and MusicGen"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "How the codec-language-model approach turns a text prompt or a hummed tune into music: MusicLM's hierarchical tokens, MusicGen's single-stage delay pattern, and the runnable code to generate a thirty-second clip yourself."
tags:
  [
    "audio-generation",
    "audio-synthesis",
    "music-generation",
    "musicgen",
    "musiclm",
    "neural-audio-codec",
    "generative-ai",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Audio Generation"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/music-generation-musiclm-and-musicgen-1.png"
---

The first time I asked a music model for "a calm lo-fi hip-hop beat with a mellow piano, 80 BPM," it gave me back thirty seconds that started exactly right and then, around the twenty-second mark, quietly changed key, lost the kick drum, and wandered off into something that was technically still lo-fi but no longer the same song. The piano was lovely. The vibe was right. But the *structure*, the thing that makes a piece of music feel like one intentional object rather than a stream of plausible sounds, fell apart the moment the model ran past the span it could hold in its head. That failure is the whole story of music generation in one clip. Speech has a transcript to anchor on and rarely runs past a paragraph. Music has no transcript, has to keep harmony, rhythm, and timbre coherent simultaneously, and is judged over *minutes* by ears that catch a wrong note in milliseconds.

This post is about the approach that made open, controllable music generation real for the rest of us: the **codec language model**. The idea is the one we have built up across this series. Take a [neural audio codec](/blog/machine-learning/audio-generation/encodec-dac-and-the-modern-codec) that turns a waveform into a short sequence of discrete tokens, then train a [transformer language model](/blog/machine-learning/audio-generation/autoregressive-audio-models-wavenet-to-audiolm) to predict those tokens the way GPT predicts words. Two systems define the era. **MusicLM** from Google took the hierarchical [semantic-then-acoustic token](/blog/machine-learning/audio-generation/semantic-vs-acoustic-tokens) recipe of AudioLM and pointed it at music, using a contrastive text-music model called MuLan to inject the text prompt. **MusicGen** from Meta then did something quietly radical: it threw out the hierarchy, modeled everything with a *single* transformer over EnCodec tokens, and made the multi-codebook problem tractable with a clever **delay pattern**. MusicGen shipped open weights, ran on a single consumer GPU, took a hummed melody as a second condition, and became the community default almost overnight.

![A vertical stack comparing the MusicLM three-stage token route and the MusicGen single-model route from a text prompt down to a waveform](/imgs/blogs/music-generation-musiclm-and-musicgen-1.png)

The figure above is the shape of the whole post. Both systems sit on the series spine, the **audio stack**: a waveform becomes [codec tokens](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound), a generative model predicts those tokens, and a decoder turns them back into a waveform, all under the tension of **fidelity, controllability, speed, and length**. MusicLM spends its complexity on a *hierarchy* of token models. MusicGen spends its cleverness on a *scheduling trick* that lets one model do the hierarchy's job. By the end of this post you will understand the delay pattern well enough to derive its step count from scratch, you will have runnable `transformers` and `audiocraft` code to generate a clip and to hum a melody into the model, and you will know exactly why both systems run out of road at around thirty seconds and have nothing to say about vocals or lyrics. We will also preview what comes next: [latent diffusion for music](/blog/machine-learning/audio-generation/latent-diffusion-for-music-stable-audio) with Stable Audio, and the [commercial frontier](/blog/machine-learning/audio-generation/suno-udio-and-the-commercial-music-frontier) of Suno and Udio that finally cracked full songs with vocals.

If you have not read [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard), the foundation post, the short version you need here is this: a song is a one-dimensional signal sampled tens of thousands of times per second, so even a codec that compresses aggressively leaves you with a *long* token sequence, and the model's context window is the hard ceiling on how much musical structure it can keep coherent. Hold that thought. It is the single fact that explains most of what follows.

## Why music is harder than speech

It is tempting to think music generation is "just" speech generation with different training data. It is not, and the differences are worth stating plainly because they shape every design decision downstream.

**There is no transcript to anchor on.** Text-to-speech has a beautiful crutch: the text. The model is handed the exact sequence of words it must produce, and the hard problem is "how should this be said," not "what should be said." A [TTS system](/blog/machine-learning/audio-generation/text-to-speech-from-tacotron-to-vits) can lean on a grapheme-to-phoneme front-end and an explicit alignment between phonemes and audio frames. Music has no such anchor. A prompt like "uplifting orchestral piece" constrains the *style* but specifies essentially nothing about the actual notes, chords, dynamics, or arrangement. The model must invent the entire content, not just its delivery. This is why text conditioning for music is a fundamentally fuzzier, looser steering signal than text conditioning for speech, and why so much engineering goes into making that signal usable at all.

**Everything happens at once.** Speech is, to a first approximation, monophonic: one speaker, one stream of phonemes, one fundamental pitch contour at a time. Music is **polyphonic** by nature. A single moment of a song contains a bass line, a chord progression, a melody, a drum pattern, and a wash of reverb, all sounding simultaneously and all required to be consistent with each other. The bass note has to belong to the chord. The kick has to land on the beat the hi-hat is subdividing. The melody has to sit in the key. A model that gets the timbre right but the harmony wrong produces something that sounds like a real recording of musicians who cannot play together, which is far more jarring than a single wrong word in synthesized speech.

**The structure spans minutes, the detail spans milliseconds.** This is the brutal one. A song has structure at every timescale at once. At the millisecond scale there is the waveform itself and the transient attack of a snare. At the scale of tens of milliseconds there is timbre, the spectral fingerprint that makes a piano a piano. At the scale of a second there is the beat and the chord. At the scale of ten seconds there is the phrase. At the scale of a minute there is the section: verse, chorus, bridge. And at the scale of the whole piece there is *form*, the large-scale arrangement that makes a chorus feel like a return. A generative model has to be coherent across **six or seven orders of magnitude in time** simultaneously. The codec compresses the millisecond detail into tokens, but the model still has to plan structure over a token sequence that, at a typical 50 Hz frame rate times several codebooks, runs into the thousands of tokens for just thirty seconds. Plan over more than the context window and the structure drifts, exactly as my lo-fi clip did.

**Our ears are merciless and trained.** Most people cannot tell a slightly-off color from a correct one, but almost everyone can hear a note that is out of tune, a beat that is off, or a chord that is "wrong" even if they cannot name why. Decades of listening to professionally produced music have given every listener an exacting, if unconscious, model of what real music sounds like. A music generator is graded against that model. There is no slack.

There is a fifth difference that is easy to miss but shapes the data side of the problem: **music has far less paired supervision than speech**. Speech comes with transcripts at enormous scale, every podcast, audiobook, and captioned video is a (text, audio) pair, so TTS and ASR sit on oceans of aligned data. Music captions are rare, inconsistent, and shallow ("upbeat pop" tells you almost nothing about the notes). This scarcity is exactly why MusicLM had to invent MuLan, a way to learn text-music alignment from limited pairs, and why MusicGen leans on a generic T5 encoder plus end-to-end training rather than expecting rich captions. The data poverty of music conditioning is a first-class constraint, not an afterthought, and it explains why text control of music is so much looser than text control of speech: there simply was not enough labeled data to learn a tight mapping.

Put these together and the design problem becomes clear. You need a representation that handles the millisecond detail cheaply (the codec), a model that can plan structure over a long sequence (the language model and its context window), and a conditioning signal that can steer style despite having no transcript and little paired data (MuLan, T5, CLAP). MusicLM and MusicGen are two different bets on how to assemble exactly those three pieces, and the rest of this post is a careful walk through both bets and the trade-offs each one makes.

## The codec-LM idea in one paragraph

Before the two systems, fix the shared idea, because both are variations on it. A neural audio codec is an encoder-quantizer-decoder trained to reconstruct a waveform through a discrete bottleneck. The encoder downsamples the waveform to a low frame rate (EnCodec runs at 50 frames per second for 32 kHz music in MusicGen's configuration); at each frame, a **residual vector quantizer** (RVQ) assigns the frame to a stack of codebook entries, one per codebook, each refining the residual the previous ones left behind. We covered the rate-distortion math of RVQ in the [residual vector quantization post](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq); the one fact you need here is that each frame of audio becomes not *one* token but a small *stack* of tokens, typically four, one per codebook, and stacking more codebooks buys more fidelity at a higher bitrate.

So a thirty-second clip at 50 Hz with 4 codebooks is $30 \times 50 \times 4 = 6000$ tokens. A music language model is just a transformer trained to predict those tokens autoregressively, conditioned on a text prompt. Decode the predicted tokens back through the codec and you have a waveform. That is the entire trick. The reason it works at all is the codec: predicting 6000 discrete tokens is a problem a transformer can do; predicting $30 \times 32000 = 960{,}000$ raw waveform samples directly is not. The codec is what makes the language-model framing tractable, the same way a [VAE latent](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) makes latent diffusion tractable for images.

The only genuinely awkward part, and the part where MusicLM and MusicGen diverge, is the **multi-codebook problem**: each frame is a *stack* of tokens, not a single token, so what exactly does "predict the next token" mean when there are four parallel streams? Hold that question. It is the crux, and the delay pattern is MusicGen's answer to it.

### The token budget is the whole constraint

It is worth dwelling on the arithmetic because almost every limitation of these models falls out of it. The codec's job is to push the bitrate of the audio down to something a transformer can model, and the bitrate is set by three numbers: the frame rate $f$ (frames per second), the number of codebooks $K$, and the codebook size $V$ (entries per codebook). Each codebook entry carries $\log_2 V$ bits, and there are $K$ of them per frame at $f$ frames per second, so the bitrate is

$$
\text{bitrate} = f \cdot K \cdot \log_2 V \quad \text{bits per second}.
$$

For MusicGen's EnCodec configuration, $f = 50$, $K = 4$, and $V = 2048$ (so $\log_2 V = 11$ bits), giving $50 \times 4 \times 11 = 2200$ bits per second, about 2.2 kbps. Compare that to the raw waveform: 32 kHz at 16-bit samples is $32000 \times 16 = 512{,}000$ bits per second, 512 kbps. The codec achieves roughly a **230:1** compression of the *bitrate*, and a comparable reduction in the *sequence length* the model must process. That compression is the entire reason a transformer can model music at all, and it is also why fidelity is capped: at 2.2 kbps the codec has thrown away a great deal, and no language model can put back detail the codec did not preserve. The codec sets the ceiling on quality; the language model can only reach it, never exceed it. This is the same ceiling-setting role the [VAE plays in latent diffusion](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch), and it is why codec quality (covered in the [EnCodec and DAC post](/blog/machine-learning/audio-generation/encodec-dac-and-the-modern-codec)) matters so much for downstream generation.

The other quantity the budget fixes is **how much music fits in context**. A transformer with a context window of, say, $L$ tokens can hold $L / (f \cdot K)$ seconds of music in the delay-pattern layout (roughly, ignoring the $K-1$ warmup). With $f = 50$ and $K = 4$, a 6000-token context holds $6000 / 200 = 30$ seconds. Double the context to 12000 tokens and you hold a minute, but attention cost grows with the square of context length, so doubling the window roughly quadruples the attention compute. This quadratic wall is the real reason the models stop at tens of seconds rather than minutes: it is not that nobody wants longer music, it is that the context length you would need to plan a four-minute song coherently is expensive enough that, at training time on the hardware budgets these were trained with, thirty seconds was the pragmatic choice. Every "why only thirty seconds" question reduces to this token budget and the quadratic cost of extending it.

#### Worked example: bitrate versus codebooks

Suppose you want to push MusicGen's quality up by using more codebooks. EnCodec can be configured with $K = 8$ instead of $K = 4$. What does that cost you in the language model?

- **Bitrate** doubles: $50 \times 8 \times 11 = 4400$ bits per second, 4.4 kbps. The reconstruction is crisper; the codec preserves more high-frequency detail.
- **Delay-pattern steps** for 30 seconds: $S + K - 1 = 1500 + 7 = 1507$, barely more than the 1503 for $K=4$. The *step count* is almost unchanged, because the delay pattern adds only $K-1$ steps total, not a factor of $K$.
- **But** each step now emits 8 tokens, not 4, so the model needs 8 output heads, the per-step compute is higher, and crucially the *causal warmup* is longer and the residual dependency chain is deeper.

So more codebooks buys fidelity at modest *step-count* cost but higher per-step cost and a harder modeling problem (the model must get all 8 residual codebooks right per frame). The released MusicGen uses $K=4$ as the balance point. This is the kind of trade the delay pattern makes cheap: because it costs $S + K - 1$ rather than $K \times S$, adding codebooks is far less painful than it would be under flattening, where $K=8$ would *double* the step count to 12000. The delay pattern is precisely what makes higher-codebook configurations even contemplatable.

## MusicLM: the hierarchical bet

MusicLM, published by Agostinelli et al. at Google in January 2023, took the cleanest possible route: apply the AudioLM recipe to music and bolt on a text-conditioning model. The AudioLM recipe, which we covered in the [autoregressive audio post](/blog/machine-learning/audio-generation/autoregressive-audio-models-wavenet-to-audiolm), splits sound into two kinds of tokens. **Semantic tokens**, extracted from a self-supervised model like w2v-BERT, capture the long-range *content and structure*, the melody and the phrasing, at a low rate. **Acoustic tokens**, from a SoundStream codec, capture the fine *recording detail*, the timbre and the room, at a higher rate. The [semantic-vs-acoustic split](/blog/machine-learning/audio-generation/semantic-vs-acoustic-tokens) exists precisely because one model is bad at doing both jobs: ask one transformer to plan structure *and* render fidelity and it does neither well.

![A vertical stack showing MusicLM generating semantic tokens for structure then coarse and fine acoustic tokens for fidelity](/imgs/blogs/music-generation-musiclm-and-musicgen-2.png)

So MusicLM is a three-stage cascade, shown above. **Stage one** generates semantic tokens, conditioned on the text. These set the high-level musical structure: roughly, the melodic and rhythmic plan. **Stage two** generates *coarse* acoustic tokens (the first few SoundStream codebooks) conditioned on the semantic tokens. **Stage three** generates *fine* acoustic tokens (the remaining codebooks) conditioned on the coarse ones, filling in the high-frequency detail. Each stage is its own autoregressive transformer. Decode the full acoustic token stack through SoundStream and you have 24 kHz audio. The hierarchy is the whole point: the cheap, low-rate semantic stage gets to plan structure without drowning in acoustic detail, and the later stages get a structural skeleton to flesh out rather than having to invent structure and fidelity at once.

The genuinely new piece MusicLM contributed is **MuLan**, the text conditioning. Recall the problem: there is no transcript, and worse, there is very little paired text-music data. You cannot caption music at the scale you can caption images. MuLan, from Huang et al., solves this with **contrastive learning**, the same machinery as [CLIP](/blog/machine-learning/image-generation/text-encoders-and-prompt-conditioning) for images and CLAP for general audio.

![A graph of MuLan's text tower and audio tower projecting captions and music clips into one shared contrastive embedding space](/imgs/blogs/music-generation-musiclm-and-musicgen-3.png)

MuLan trains two encoders, a text tower and an audio tower, shown above. The text tower reads a caption; the audio tower reads a music clip. The contrastive objective pulls the embedding of a clip and the embedding of *its* caption together, and pushes mismatched pairs apart, until matching text and music land near each other in a shared space. The loss is the standard InfoNCE: for a batch of $N$ paired clip-caption embeddings $\{(a_i, t_i)\}$ with cosine similarity $s(a, t)$ and temperature $\tau$,

$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(s(a_i, t_i)/\tau)}{\sum_{j=1}^{N} \exp(s(a_i, t_j)/\tau)}.
$$

The trick that makes MuLan useful for *generation* is what you do at training versus inference time. During training, MusicLM conditions on the *audio* MuLan embedding, computed from the training clip itself, which is always available. At inference time, you have no audio, only text, so you swap in the *text* MuLan embedding. Because the contrastive objective forced the two embeddings into the same space, the model that learned to condition on audio embeddings can be steered by text embeddings it never saw during conditioning training. This is the clever move: it sidesteps the scarcity of paired text-music data by training the *generator* on plentiful audio-to-audio conditioning and only using text at the very end through the shared embedding. It is the same idea that makes [CLIP-guided image generation](/blog/machine-learning/image-generation/text-encoders-and-prompt-conditioning) work.

MusicLM was, by every account, excellent. It produced coherent, on-prompt music at 24 kHz and demonstrated capabilities like "story mode" (a sequence of prompts) and melody conditioning from a hummed input. But Google never released the weights. There was no public checkpoint to download, finetune, or build on. For the open community, MusicLM was a paper to admire and a target to reimplement, not a tool to use. That gap is exactly what MusicGen filled.

## MusicGen: the simplification

MusicGen, from Copet et al. at Meta in June 2023, asked an uncomfortable question: do you actually *need* the hierarchy? The semantic-then-acoustic cascade is principled, but it is also three models to train, three models to serve, and three places for things to go wrong. What if a single transformer, conditioned on text, could model the EnCodec acoustic tokens directly, structure and detail at once?

![A graph of MusicGen as one transformer reading a T5 text prefix and optional melody chromagram and emitting four EnCodec codebooks under a delay pattern](/imgs/blogs/music-generation-musiclm-and-musicgen-4.png)

The answer, shown above, is yes, with two enabling tricks. The first is conditioning: MusicGen drops MuLan in favor of a **T5 text encoder**, the same off-the-shelf encoder used across many text-to-X models, whose output is injected into the transformer via cross-attention (and optionally prepended as a prefix). T5 is trained on text alone, not on text-music pairs, but it turns out a strong general text encoder plus end-to-end training on the (text, music) pairs MusicGen *does* have is enough; the model learns to map T5 features to musical attributes during its own training. This is simpler than MuLan and avoids needing a separate contrastive pretraining stage, at the cost of less specialized text-music alignment.

The second trick, the one that makes the single-stage idea actually work, is the **delay pattern** (also called codebook interleaving). This is the heart of the post, and it deserves its own section, because it is the elegant answer to the multi-codebook problem that the hierarchy was invented to avoid.

## The delay pattern, derived from scratch

Here is the problem stated precisely. EnCodec gives you, at each of $S$ frames, a stack of $K$ codebook tokens (MusicGen's default is $K=4$). Call the token from codebook $k$ at frame $s$ the value $c_{k,s}$. A language model produces one token per step. How do you arrange the $K \times S$ tokens into a sequence the model emits one step at a time, and what are the causal dependencies?

**Option one: flatten.** Lay the tokens out frame by frame: all $K$ codebooks of frame 1, then all $K$ of frame 2, and so on. The sequence is $c_{1,1}, c_{2,1}, c_{3,1}, c_{4,1}, c_{1,2}, c_{2,2}, \ldots$, of length $K \times S$. One model, fully causal, clean dependencies, the language-model framing in its purest form. The cost: the sequence is $K$ times longer than the number of frames. For 30 seconds at 50 Hz and $K=4$ that is $4 \times 1500 = 6000$ autoregressive steps. Every step is a full forward pass of the transformer. Generation time scales with the number of steps, so flattening makes you pay $4\times$ the steps, which is $4\times$ the latency. For a model meant to run on a consumer GPU, that is the difference between usable and not.

**Option two: parallel.** Predict all $K$ codebooks of a frame at *once*, in a single step, with $K$ separate output heads. The sequence is now only $S$ steps long, a $4\times$ speedup. The cost: within a frame, the $K$ codebook predictions are made *independently and simultaneously*, so the model cannot let $c_{2,s}$ depend on $c_{1,s}$. But RVQ codebooks are *residual*, the whole point of codebook 2 is that it refines the residual codebook 1 left behind, so $c_{2,s}$ is highly dependent on $c_{1,s}$. Predicting them independently throws away exactly the dependency structure that makes RVQ work, and quality suffers badly. The parallel pattern is fast and wrong.

**Option three: the delay pattern.** This is MusicGen's answer, and it threads the needle. The idea: keep the parallel pattern's one-step-per-frame speed, but restore the causal dependency between codebooks by *offsetting each codebook in time*. Specifically, delay codebook $k$ by $k-1$ steps. Codebook 1 is not delayed. Codebook 2 is shifted one step later. Codebook 3 is shifted two steps later. Codebook 4 is shifted three steps later. (Padding tokens fill the gaps at the start.)

![A grid showing the codebook delay pattern where each RVQ codebook is offset by one extra step so one autoregressive step emits several codebooks](/imgs/blogs/music-generation-musiclm-and-musicgen-5.png)

The grid above shows the resulting layout. Look at what one column, one autoregressive step, now contains. At step $s$ (for $s$ large enough that all codebooks have started), the model emits codebook 1 of frame $s$, codebook 2 of frame $s-1$, codebook 3 of frame $s-2$, and codebook 4 of frame $s-3$. The model still emits $K$ tokens per step, one per codebook head, so you keep the parallel pattern's speed. But now the crucial dependency is **causal across steps**, not within a step. When the model predicts codebook 2 of frame $s-1$ at step $s$, it has *already* emitted codebook 1 of frame $s-1$ back at step $s-1$, so codebook 1 is in the context and codebook 2 can attend to it. The residual dependency $c_{2,s} \mid c_{1,s}$ is preserved because the two tokens live in *different steps*, with codebook 1 strictly earlier.

Let me make the dependency concrete. With the delay, codebook $k$ of frame $f$ is emitted at step $s = f + (k-1)$. So codebook 1 of frame $f$ is at step $f$; codebook 2 of frame $f$ is at step $f+1$; codebook 3 at $f+2$; codebook 4 at $f+3$. Because steps are strictly ordered and the transformer is causal, every codebook of a frame is conditioned on all *lower-numbered* codebooks of the *same* frame (they came earlier) and on all codebooks of *earlier* frames. That is exactly the dependency structure RVQ wants, recovered at parallel-pattern speed.

Now the step count. The last codebook ($k=K$) of the last frame ($f=S$) is emitted at step $S + (K-1)$. The first token is at step 1. So the delay pattern takes

$$
S + K - 1 \quad \text{steps},
$$

versus $K \times S$ for flattening. For $S = 1500$ and $K = 4$: flattening is $6000$ steps; the delay pattern is $1503$ steps. That is a **3.99×** reduction, essentially the full $K\times$ speedup, while keeping almost all of the causal structure flattening gave you. The only thing you give up is the dependency of $c_{1,s}$ on $c_{2,s-1}$ and similar cross-codebook-cross-frame links that the strict flatten ordering would have captured, and empirically those matter little. This is why the delay pattern is the default: it is within a hair of flattening's quality at a quarter of flattening's cost.

#### Worked example: counting the savings

Take a concrete generation: 30 seconds of music, EnCodec at 50 Hz, $K=4$ codebooks. The frame count is $S = 30 \times 50 = 1500$.

- **Flatten:** $K \times S = 4 \times 1500 = 6000$ forward passes.
- **Delay:** $S + K - 1 = 1500 + 3 = 1503$ forward passes.
- **Ratio:** $6000 / 1503 = 3.99$.

If a single forward pass of MusicGen-medium takes, say, roughly 15 ms on an RTX 4090 (order of magnitude; depends on batch, context length, and `torch.compile`), then flatten is about $6000 \times 0.015 = 90$ seconds and delay is about $1503 \times 0.015 = 22.5$ seconds for the same 30 seconds of audio. The real-time factor (RTF, generation-time over audio-duration) drops from about 3.0 to about 0.75. Crossing RTF below 1.0, generating faster than real time, is the difference between a model you wait on and a model you could stream. Treat the per-step millisecond figure as approximate; the *ratio* of 3.99 is exact and is the part you can bank on.

There are other interleaving patterns in the MusicGen paper, "partial delay" and "VALL-E pattern" (a coarse-first split like VALL-E's), but the simple "delay by $k-1$" pattern is the default and the one the released checkpoints use. The general lesson, worth carrying beyond music: when you have $K$ parallel streams with a residual dependency, *offsetting them in time* converts a within-step dependency you cannot model into a cross-step dependency you can, at the cost of $K-1$ extra steps total rather than a multiplicative blowup.

### The training objective

With the layout fixed, the training objective is exactly the one you know from language modeling, applied per codebook. Let the delayed token sequence have columns $1 \ldots T$ where $T = S + K - 1$, and let $x_{k,t}$ be the token of codebook $k$ in column $t$. The model, conditioned on the T5 text encoding $c$ and on all earlier columns, predicts a distribution over each codebook's vocabulary. The loss is the sum of cross-entropies across codebooks and columns,

$$
\mathcal{L} = -\sum_{t=1}^{T} \sum_{k=1}^{K} \log p_\theta\!\left(x_{k,t} \mid x_{<t}, c\right),
$$

where $x_{<t}$ is every token in columns strictly before $t$ (the causal mask enforces this). Each codebook gets its own output head and its own softmax over the $V = 2048$ codebook entries, but they share the transformer trunk. Three details make this work in practice and are worth knowing because they are where the subtle bugs live.

First, the **padding tokens** that fill the delay gaps at the start (and the corresponding gaps at the end) are *masked out of the loss*. You do not want the model spending capacity predicting "pad," and you do not want pad predictions polluting the gradient. The loss above runs only over real token positions.

Second, the **conditioning is dropped at random during training** to enable classifier-free guidance at inference. With some probability (commonly 10–20 percent), the text condition $c$ is replaced by a null embedding, so the model learns both the conditional distribution $p_\theta(x \mid c)$ and the unconditional $p_\theta(x)$. At inference, guidance interpolates between them, the mechanism we [derived for images](/blog/machine-learning/image-generation/classifier-free-guidance) and reuse unchanged for audio. Without this conditioning dropout at training time, `guidance_scale` at inference would have no unconditional branch to push away from.

Third, the **codebooks are not weighted equally in importance** even though the loss weights them equally. Codebook 1 carries the coarse, structurally important information (it is the first RVQ level, the biggest chunk of the signal); codebook 4 carries fine residual detail. An error in codebook 1 is musically catastrophic; an error in codebook 4 is a slight texture blemish. The uniform cross-entropy does not know this, which is one reason sampling temperature sometimes wants to be lower for the coarse codebooks than the fine ones, though the released models keep it simple. This asymmetry, coarse codebooks matter more, is the same insight that motivated MusicLM's explicit coarse-then-fine split; MusicGen folds it into one model and lets the delay ordering (coarse codebook 1 always emitted first within a frame) carry it implicitly.

## Generating music with MusicGen, for real

Enough theory. Here is the code to generate a clip with 🤗 `transformers`. The model is `MusicgenForConditionalGeneration` and it bundles the T5 text encoder, the transformer LM, and the EnCodec decoder behind one `generate` call.

```python
import torch
import soundfile as sf
from transformers import AutoProcessor, MusicgenForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

# medium = 1.5B params, the usual default; "facebook/musicgen-small" (300M)
# is faster for prototyping, "facebook/musicgen-large" (3.3B) is the quality bar.
processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
model = MusicgenForConditionalGeneration.from_pretrained(
    "facebook/musicgen-medium", torch_dtype=torch.float16
).to(device)

prompt = "calm lo-fi hip-hop beat with a mellow piano, 80 BPM, soft vinyl crackle"
inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)

# guidance_scale is classifier-free guidance: higher = obey the prompt harder.
# max_new_tokens sets the length: at 50 Hz, ~1500 tokens is ~30 s.
import time
t0 = time.time()
with torch.no_grad():
    audio = model.generate(
        **inputs,
        do_sample=True,
        guidance_scale=3.0,
        max_new_tokens=1503,   # ~30 s = S + K - 1 for S=1500, K=4
    )
elapsed = time.time() - t0

sr = model.config.audio_encoder.sampling_rate   # 32000 for MusicGen
wav = audio[0, 0].cpu().float().numpy()
sf.write("lofi.wav", wav, sr)
duration = len(wav) / sr
print(f"generated {duration:.1f} s in {elapsed:.1f} s  ->  RTF {elapsed/duration:.2f}")
```

A few things to notice, because they are exactly the levers you will reach for. `guidance_scale` is [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance) applied to audio: the model runs once with the prompt and once without, and pushes the prediction away from the unconditional and toward the conditional by the guidance factor. MusicGen's default is 3.0; raise it and the output hews harder to the prompt but can get harsh and lose diversity, lower it and the prompt becomes a loose suggestion. `max_new_tokens` is your length knob, and the comment makes the delay-pattern arithmetic concrete: you ask for $S + K - 1$ tokens to get $S$ frames of audio. The `do_sample=True` matters: greedy decoding of music sounds dead and repetitive, so you sample, and `temperature`, `top_k`, and `top_p` (all `generate` kwargs) control how adventurous the sampling is.

The `audiocraft` library, Meta's own, gives a slightly higher-level interface and is the reference implementation:

```python
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained("facebook/musicgen-medium")
model.set_generation_params(
    duration=30,           # seconds; audiocraft handles the token math for you
    use_sampling=True,
    top_k=250,
    cfg_coef=3.0,          # classifier-free guidance coefficient
)

descriptions = [
    "calm lo-fi hip-hop beat with a mellow piano, 80 BPM",
    "energetic synthwave with a driving bassline, 120 BPM",
]
wav = model.generate(descriptions)   # [batch, channels, samples]

for i, one in enumerate(wav):
    audio_write(f"clip_{i}", one.cpu(), model.sample_rate,
                strategy="loudness")   # writes clip_0.wav, clip_1.wav, loudness-normalized
```

Note the `strategy="loudness"` in `audio_write`: it loudness-normalizes the output, which matters because raw model output can clip or be inconsistently loud across generations. This is the kind of production detail that separates "it generates audio" from "it generates audio you can ship."

## Melody conditioning: hum a tune

The feature that made MusicGen genuinely fun, and genuinely useful, is **melody conditioning**. The `facebook/musicgen-melody` checkpoint takes a second input alongside the text: a reference audio clip whose *melody* the generation should follow, while the *style* still comes from the text. You hum a tune, or feed in a few bars of any song, and ask for "the same melody, but as a string quartet."

The mechanism is a **chromagram**. A chromagram is a representation that collapses the spectrogram onto the twelve pitch classes (C, C#, D, ... B), discarding octave and timbre and keeping only *which notes are sounding*. Concretely, you take the [STFT](/blog/machine-learning/audio-generation/the-mathematics-of-audio-signals), map each frequency bin to its pitch class, and sum the energy within each of the twelve classes per frame. The result is a 12-dimensional vector per frame that says, roughly, "these pitch classes are active now." It deliberately throws away timbre and octave precisely so that the *melody*, the pitch contour, survives but the *instrument* does not, leaving the text free to set the instrument.

![A before-and-after comparison of text-only generation where the tune is arbitrary versus melody-conditioned generation where the model follows a hummed pitch contour](/imgs/blogs/music-generation-musiclm-and-musicgen-7.png)

The figure above is the contrast. Without melody, the text sets the genre and the model picks whatever tune fits, which is fine if you have no opinion about the melody and useless if you do. With a chromagram, the melody is pinned frame-by-frame to your reference while the text still controls the arrangement. Crucially the chromagram is **time-aligned**: it is one 12-vector per frame, lined up with the output frames, so the melody at second three conditions the audio at second three. This makes it a time-aligned conditioning signal in the sense of the [conditioning post](/blog/machine-learning/audio-generation/conditioning-and-control-in-audio-generation), added at every position rather than once at the start, exactly like a per-frame control track.

Here is melody conditioning in `audiocraft`:

```python
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained("facebook/musicgen-melody")
model.set_generation_params(duration=15, cfg_coef=3.0, use_sampling=True, top_k=250)

# load any reference whose tune you want to keep (your humming, a recording, ...)
melody, sr = torchaudio.load("hummed_tune.wav")

descriptions = ["a warm string quartet, expressive, legato"]
wav = model.generate_with_chroma(
    descriptions,
    melody_wavs=melody[None],     # [batch, channels, samples]
    melody_sample_rate=sr,
)
audio_write("quartet_from_hum", wav[0].cpu(), model.sample_rate, strategy="loudness")
```

`generate_with_chroma` is where the chromagram is computed internally from your reference and injected as the time-aligned condition. The output is a string quartet that follows your hummed melody. This is a remarkably direct form of control for a generative music model, and it is the single biggest reason MusicGen found real users: it turns "generate something in this style" into "generate *this tune* in this style," which is a far more useful instruction.

#### Worked example: when the melody fights the text

A real failure to keep in mind. Suppose you hum a melody that lives in a major key and ask for "dark, dissonant horror score." The chromagram pins the pitch classes; the text pulls toward dissonance. The model has to satisfy both, and the result is often a compromise that satisfies neither cleanly: a major-ish melody with uneasy harmony bolted on, which can sound either interestingly tense or simply confused. The lesson is that melody conditioning is a *hard* constraint on pitch class and the text is a *soft* constraint on everything else, so when they conflict, pitch usually wins and the style bends. If you want the dark version, hum a darker melody. The constraint that is time-aligned and dense (the chromagram, 12 numbers every frame) dominates the constraint that is global and sparse (a sentence of text). That ranking, dense time-aligned signal beats sparse global signal, is a general rule of conditioning worth internalizing.

## A sketch of the delay-pattern code

To make the delay pattern fully concrete, here is a minimal sketch of the interleaving itself, the function that takes a $[K, S]$ array of codebook tokens and produces the delayed $[K, S']$ layout the model trains and generates on. This is the operation hidden inside MusicGen's data pipeline.

```python
import torch

def apply_delay_pattern(codes: torch.Tensor, pad_token: int) -> torch.Tensor:
    """
    codes: [K, S] long tensor of RVQ codebook tokens (K codebooks, S frames).
    Returns a [K, S + K - 1] tensor where codebook k is shifted right by k steps,
    with pad_token filling the gaps. This is the layout the LM emits one column
    at a time; column t holds (cb0 of frame t, cb1 of frame t-1, ...).
    """
    K, S = codes.shape
    out = codes.new_full((K, S + K - 1), pad_token)
    for k in range(K):
        # codebook k starts at column k (delay = k) and runs for S frames
        out[k, k:k + S] = codes[k]
    return out


def undo_delay_pattern(delayed: torch.Tensor, K: int, S: int, pad_token: int) -> torch.Tensor:
    """Inverse: pull each codebook back to its frame-aligned position, drop pads."""
    out = delayed.new_full((K, S), pad_token)
    for k in range(K):
        out[k] = delayed[k, k:k + S]
    return out


# tiny demonstration
K, S = 4, 6
codes = torch.arange(K * S).reshape(K, S)   # fake tokens 0..23
delayed = apply_delay_pattern(codes, pad_token=-1)
print(delayed.shape)        # torch.Size([4, 9])  ==  [K, S + K - 1]
recovered = undo_delay_pattern(delayed, K, S, pad_token=-1)
assert torch.equal(recovered, codes)        # round-trips exactly
```

Read the shapes. The input is $[4, 6]$, the output is $[4, 9] = [K, S + K - 1]$. Each codebook row is shifted right by its index, and the model is trained to predict each column given the columns before it. At inference, the model emits columns left to right; once all $K$ codebooks have started (after $K-1$ warmup columns) every column produces a complete set of $K$ tokens, one per codebook, that you accumulate, de-delay with `undo_delay_pattern`, and hand to the EnCodec decoder. This is the whole mechanism in twenty lines. The real implementation handles batching, per-codebook output heads, and the cross-attention to the T5 prefix, but the scheduling logic is exactly this shift.

## Sampling and decoding: the knobs that make or break a clip

The single most underappreciated part of running a music codec-LM is the **decoding strategy**, the procedure that turns the model's per-step probability distribution into an actual chosen token. Get this wrong and a perfectly good model produces garbage; get it right and a modest model sounds great. Because this is a language model, all the decoding machinery you know from text generation applies, but the right *settings* are different because music tolerates, and indeed requires, more randomness than text.

**Why you must sample.** Greedy decoding, always taking the most probable token, is a disaster for music. It collapses into repetition, the model finds a locally safe loop and stays there, producing a clip that drones on one bar forever. Beam search, which helps for translation, is similarly bad here because there is no single "correct" continuation to search toward; music is genuinely one-to-many. So you *sample* from the distribution, and the question becomes how to shape that distribution.

**Temperature** rescales the logits before the softmax: a logit vector $z$ becomes $z / \tau$. At $\tau = 1$ you sample from the model's raw distribution. Below 1 the distribution sharpens (more conservative, more repetitive, more "safe"); above 1 it flattens (more adventurous, more surprising, more prone to going off the rails). For music, $\tau$ around 1.0 is typical; pushing it down makes the output dull and loopy, pushing it up makes it incoherent. The drift problem discussed later is partly a temperature problem: every sampled token is a small random step away from the plan, and temperature is the step size.

**Top-k and top-p (nucleus) sampling** truncate the distribution before sampling to keep you out of the long tail of implausible tokens. Top-k keeps only the $k$ most probable tokens (MusicGen's default is $k = 250$); top-p keeps the smallest set of tokens whose probability mass exceeds $p$. Both prevent the rare catastrophic token, an out-of-key note, a glitch, that even a small probability would otherwise let through over thousands of steps. Over a 6000-token generation, a per-token glitch probability of even 0.1 percent compounds to roughly a 45 percent chance of at least one audible glitch ($1 - 0.999^{6000}$), so truncation is not optional; it is what keeps long clips clean.

```python
from audiocraft.models import MusicGen

model = MusicGen.get_pretrained("facebook/musicgen-medium")

# the four knobs that shape every generation
model.set_generation_params(
    duration=20,
    use_sampling=True,   # never greedy for music
    temperature=1.0,     # step size of the sampling random walk
    top_k=250,           # truncate the long tail (MusicGen default)
    top_p=0.0,           # 0 disables nucleus; use one of top_k OR top_p
    cfg_coef=3.0,        # classifier-free guidance strength
)
wav = model.generate(["a bright acoustic guitar fingerpicking pattern, folk"])
```

The interaction between `cfg_coef` (guidance) and the sampling knobs is the subtle part. Guidance sharpens the distribution toward the prompt *before* you sample; high guidance plus low temperature is double-conservative and sounds rigid; high guidance plus high temperature can fight itself. The practical recipe most people converge on is moderate guidance (3.0) with temperature near 1.0 and top-k truncation, then adjust *one* knob at a time. There is no universally best setting because the right amount of randomness depends on the genre: ambient textures tolerate more entropy than a tight techno groove where one off-beat token is glaring.

#### Worked example: diagnosing a loopy generation

You generate a 30-second clip and it repeats the same two-bar phrase the whole time. Which knob? The repetition signature points to *too little* entropy in the sampling. Check, in order: is `use_sampling=True` (greedy will always loop)? Is `temperature` too low (try raising from 0.8 to 1.0)? Is `top_k` too small (a top_k of 5 is far too tight for music, leaving the model almost no room to vary)? Conversely, if the clip is *incoherent*, random notes, no groove, lower the temperature and tighten top_k. The diagnostic rule: **looping means raise entropy, chaos means lower entropy**, and you turn exactly one knob per generation so you can attribute the change. This single-variable discipline, the same you would use debugging any stochastic system, saves enormous time versus turning three knobs at once and not knowing which one helped.

## Finetuning MusicGen on your own style

The reason MusicGen's open release mattered so much is that you can *finetune* it, adapt the model to a specific instrument, genre, or artist's catalogue, which is impossible with a closed API. The workflow is the standard transformer finetune, with audio-specific data prep.

The data pipeline is: collect audio in your target style, encode each clip to EnCodec tokens with the frozen codec, pair each with a text caption (either hand-written or auto-generated by a captioning model), and train the transformer to predict the tokens conditioned on the caption. The codec stays frozen, you are not retraining the tokenizer, only the language model over its tokens. This is exactly analogous to finetuning a text LLM: the tokenizer is fixed, you adapt the model that predicts tokens.

```python
# conceptual finetuning loop (audiocraft / a custom trainer); shapes shown
import torch

# 1) freeze the codec, use it only to tokenize
encodec = load_encodec().eval()
for p in encodec.parameters():
    p.requires_grad_(False)

# 2) MusicGen's transformer is what we train
lm = load_musicgen_lm("facebook/musicgen-medium")   # trainable
opt = torch.optim.AdamW(lm.parameters(), lr=1e-5, weight_decay=0.01)

for wav, caption in style_dataloader:           # wav: [B, 1, samples]
    with torch.no_grad():
        codes = encodec.encode(wav)             # [B, K, S] discrete tokens
        codes = apply_delay_pattern(codes, pad_token=lm.pad)  # [B, K, S+K-1]
    cond = lm.encode_text(caption)              # T5 features [B, L, d]
    logits = lm(codes, condition=cond)          # per-codebook logits
    loss = masked_cross_entropy(logits, codes, ignore=lm.pad)  # mask pads
    loss.backward(); opt.step(); opt.zero_grad()
```

Two practical notes that save real pain. First, use a **small learning rate** (around $10^{-5}$) and few epochs; MusicGen is already a strong base, and a high learning rate on a small style dataset will catastrophically forget the general musicality that makes the base useful, leaving you with a model that only does your twelve training clips. Second, for small datasets, prefer **parameter-efficient finetuning** (LoRA-style adapters) over full finetuning, the [PEFT machinery](/blog/machine-learning/training-techniques) that keeps the base frozen and trains a small number of added parameters, which both reduces forgetting and lets you keep multiple style adapters for one base model. The captioning quality matters too: garbage captions teach the model garbage text-to-music associations, so spend effort on the text side even though the audio is the "interesting" part.

#### Worked example: a 50-clip style finetune

Say you have 50 clips of a specific lo-fi producer's beats, each about 20 seconds, and you want MusicGen to generate "in their style." Full finetuning of the 1.5B medium model on 50 clips will overfit hard, with so little data the model memorizes rather than generalizes. The right move is a LoRA adapter (a few million trainable parameters) at a learning rate around $10^{-4}$ for the adapter, a handful of epochs, with the base frozen. Caption each clip consistently (for example "lo-fi hip-hop, mellow, vinyl crackle, [tempo]"). Expect the adapter to capture the *texture and instrumentation* well (the vinyl crackle, the specific drum sound) and the *structure* less so, because 50 clips is too few to teach long-range form. This matches the general rule: small finetunes move timbre and surface style easily and move deep structure barely, because structure is the part that needs scale. Budget accordingly, you are buying a style filter, not a new composer.

## MusicLM versus MusicGen, side by side

We now have both systems on the table. The comparison is the crux of the post, so let me lay it out explicitly.

![A matrix comparing MusicLM and MusicGen across number of stages, conditioning, context, open weights, and quality](/imgs/blogs/music-generation-musiclm-and-musicgen-6.png)

The matrix above summarizes it, and the table below puts numbers and citations to it. Treat any precise FAD figure as approximate and embedding-dependent; the *relative* ordering is the robust part.

| Property | MusicLM (Google, 2023) | MusicGen (Meta, 2023) |
| --- | --- | --- |
| Stages | 3 (semantic → coarse → fine acoustic) | 1 (single transformer over EnCodec tokens) |
| Multi-codebook handling | hierarchy of separate models | delay pattern in one model |
| Text conditioning | MuLan (contrastive text-music) | T5 text encoder + cross-attention |
| Melody conditioning | yes (from a hummed input) | yes (chromagram, `musicgen-melody`) |
| Context / length | ~30 s coherent | ~30 s coherent (50 Hz tokens) |
| Sample rate | 24 kHz | 32 kHz |
| Open weights | no public checkpoint | yes, small/medium/large/melody |
| Quality (FAD, VGGish) | strong, no public reproduction | FAD ≈ 4 (large), competitive with MusicLM |

The headline is that MusicGen reached roughly MusicLM-class quality with **one** model instead of three, conditioning on a generic text encoder instead of a bespoke contrastive one, and it *shipped the weights*. The single-stage design is not just simpler to reason about; it is simpler to finetune (one model, one loss), simpler to serve (one forward pass per step), and simpler to extend (the melody variant is the same model with one extra conditioning input). The hierarchy MusicLM used is genuinely elegant and the semantic-token idea remains important, but the delay pattern showed you could get the multi-codebook benefit without paying for a multi-model cascade.

This is a recurring pattern in generative modeling: an early system establishes that something is *possible* with a principled, complex design, and a follow-up establishes that it is *practical* with a simpler one. MusicLM was the proof of concept; MusicGen was the workhorse. The same arc runs through the image and video series.

## MusicGen's model sizes

MusicGen shipped as a family, and choosing among them is a real decision, so let us be concrete about the trade.

![A matrix of MusicGen model sizes from small to large to melody with parameters, quality, melody input, and recommended use](/imgs/blogs/music-generation-musiclm-and-musicgen-8.png)

The figure above lays out the four checkpoints. The numbers below are the ones to act on.

| Checkpoint | Params | Relative quality | Melody input | Reach for it when |
| --- | --- | --- | --- | --- |
| `musicgen-small` | 300M | decent, fastest | no | prototyping, tight latency, CPU-ish |
| `musicgen-medium` | 1.5B | good, balanced | no | the default text-to-music workhorse |
| `musicgen-large` | 3.3B | best FAD, slowest | no | you need the quality bar and have a GPU |
| `musicgen-melody` | 1.5B | good + tune control | yes (chromagram) | you want to hum or supply a melody |

The quality-versus-latency trade is monotone and predictable: bigger model, lower FAD and better prompt adherence, higher latency and VRAM. `small` at 300M fits comfortably on modest hardware and is genuinely usable for drafts; `large` at 3.3B is the one you reach for when output quality is the constraint and you can spend the seconds and the 16+ GB of VRAM. `medium` is the sweet spot most people live on. The `melody` checkpoint is `medium`-sized but trained to accept the chromagram, so you pick it specifically for the hum-a-tune capability, not for raw quality.

#### Worked example: budgeting a generation run

Say you want to generate 200 candidate 10-second clips to pick the best one for a video. On `musicgen-large` (3.3B) at an RTF of roughly 1.5 on an RTX 4090 (order of magnitude; the large model is comfortably slower than real time on consumer hardware), each 10-second clip takes about 15 seconds, so 200 clips is about 50 minutes. On `musicgen-small` (300M) at an RTF of maybe 0.4, each clip takes about 4 seconds, so 200 clips is about 13 minutes, but the per-clip quality is lower so you may need *more* candidates to find a keeper. The practical move is a two-stage funnel: generate many candidates on `small` to explore prompts and seeds cheaply, lock the prompt, then regenerate the finalists on `large` for quality. This explore-cheap-then-refine-expensive funnel is the standard way to use a model family with a steep quality-cost curve, and it applies just as much to image and video generation.

## The hard limits: length, vocals, drift

MusicGen and MusicLM are wonderful within their box. The box has three walls, and you need to know exactly where they are.

**The ~30-second wall.** Both models are coherent for roughly thirty seconds and then degrade. The reason is the context window, and it follows directly from the token math. At 50 Hz with the delay pattern, thirty seconds is about 1500 columns; the transformer's context is finite, and once the structure you are continuing falls out of the window, the model has nothing to anchor the next section to. You *can* generate longer by **sliding-window continuation**: generate thirty seconds, take the last few seconds of tokens as a prompt, generate the next thirty conditioned on them, and stitch. `audiocraft` supports this. But it is a patch, not a fix. The model only sees the recent past, so it maintains *local* coherence (the texture and key carry over) while *global* structure drifts, the chorus does not come back, the energy meanders, because nothing in the model's view remembers that there was supposed to be a chorus. My lo-fi clip's twenty-second key change was a baby version of this; at four minutes it is severe. Long-form structure is genuinely unsolved by the pure codec-LM approach and is one of the main reasons the field looked to [diffusion with timing conditioning](/blog/machine-learning/audio-generation/latent-diffusion-for-music-stable-audio) and to the commercial systems.

**No vocals, no lyrics.** MusicGen generates instrumental music. It has no notion of words, no lyric conditioning, and the EnCodec tokens it models were not trained to render intelligible singing. Ask for "a pop song with vocals" and you get instrumental music that may have a wordless vocal-like texture, never actual sung lyrics. This is not a tuning problem; it is architectural. Singing requires modeling phonemes *and* pitch *and* timbre jointly, aligned to a melody and a lyric, which is a substantially harder conditioning problem. Cracking it is precisely what [Suno and Udio](/blog/machine-learning/audio-generation/suno-udio-and-the-commercial-music-frontier) did, and it is the single biggest capability gap between the open codec-LM models and the commercial frontier.

**Drift on long generations.** Even within a single generation, and especially in sliding-window continuation, the output **drifts**: the tempo wobbles, the key slides, the energy decays. Sampling is partly to blame, every sampled token is a small random walk away from the plan, and over thousands of tokens those steps accumulate. Lowering temperature reduces drift but also reduces musicality, the same diversity-stability tension as everywhere in generation. There is no free lunch here; the codec-LM approach trades long-range stability for the simplicity and controllability that make it so usable in the first place.

It is worth being precise about *why* sliding-window continuation drifts rather than diverging into noise, because the distinction tells you what the failure is and is not. When you continue from the last few seconds of tokens, the model sees a perfectly coherent local context, so the *next* few seconds are locally fine: the texture, the key, and the tempo carry over smoothly because they are all visible in the window. What does not carry over is anything *outside* the window, the fact that the piece began in a particular section, that a chorus is due, that the energy arc was supposed to build then resolve. The model has no memory of those, so it makes a locally optimal choice that is globally aimless. The result is music that is *always* plausible moment to moment and *never* purposeful across the whole, like a conversation where each sentence follows from the last but the speaker has forgotten what they were talking about. That is the precise signature of a fixed-context model asked to produce structure longer than its context, and it is why the genuine fix is longer context or an explicit plan, not better sampling. Both [Stable Audio's timing conditioning](/blog/machine-learning/audio-generation/latent-diffusion-for-music-stable-audio) and the commercial systems' long-context training attack exactly this, giving the model either a global position signal or a window large enough to hold the whole piece, so that the chorus can come back because the model can still see that there was a verse.

## Measuring music generation honestly

You cannot improve what you cannot measure, and music generation is notoriously hard to measure, so a word on how to evaluate these models without fooling yourself. The two metrics that matter are **FAD** (Fréchet Audio Distance) for quality and a **CLAP score** for text adherence, plus human listening for anything you actually ship. We covered the metric zoo in the [audio quality metrics post](/blog/machine-learning/audio-generation/audio-quality-metrics); here is how to wire them up for a music model specifically.

FAD measures how close the *distribution* of your generated clips is to the distribution of real music, in the embedding space of a pretrained audio classifier (VGGish, classically, or a CLAP/PANN embedding). You embed a few thousand real clips and a few thousand generated clips, fit a Gaussian to each set, and compute the Fréchet distance between the two Gaussians. Lower is better. The two traps, both stressed in the [evaluation post](/blog/machine-learning/audio-generation/audio-quality-metrics), are the *embedding choice* (FAD with VGGish and FAD with CLAP are different numbers and rank models differently, so you cannot compare across papers unless the embedding matches) and the *sample size* (FAD is biased at small sample sizes and needs at least a couple thousand clips per set to stabilize). Report both the embedding and the sample count or the number is meaningless.

```python
# FAD between a set of real clips and a set of generated clips
from frechet_audio_distance import FrechetAudioDistance

fad = FrechetAudioDistance(
    model_name="vggish",     # ALWAYS state the embedding; vggish vs clap differ
    sample_rate=16000,
    use_pca=False,
)
score = fad.score(
    background_dir="real_music_2k/",    # >= ~2000 real reference clips
    eval_dir="musicgen_outputs_2k/",    # >= ~2000 generated clips, same count
)
print(f"FAD(vggish) = {score:.2f}  over 2000 clips each")
```

CLAP score measures *text adherence*: does the generated clip match its prompt? You embed the prompt with CLAP's text tower and the generated audio with CLAP's audio tower, and take the cosine similarity, exactly the shared-space idea MuLan uses for conditioning, repurposed here for evaluation. A higher CLAP score means the audio better matches what the text asked for. It is the right metric for asking "did `guidance_scale` actually make the output follow the prompt more," because you can sweep guidance and watch the CLAP score climb (and then watch FAD worsen as guidance gets too high, the same control-versus-fidelity trade quantified).

```python
import laion_clap
import numpy as np

clap = laion_clap.CLAP_Module(enable_fusion=False)
clap.load_ckpt()  # downloads a pretrained CLAP checkpoint

prompts = ["calm lo-fi hip-hop with mellow piano"]
text_emb = clap.get_text_embedding(prompts)                      # [1, d]
audio_emb = clap.get_audio_embedding_from_filelist(["lofi.wav"]) # [1, d]

cos = (text_emb @ audio_emb.T) / (
    np.linalg.norm(text_emb) * np.linalg.norm(audio_emb)
)
print(f"CLAP text-audio similarity = {float(cos):.3f}")  # higher = better adherence
```

Neither metric replaces **human listening**. FAD can be fooled (a model that produces tasteful-but-generic music can score well while sounding boring), and CLAP rewards surface-level prompt matching over musical quality. For anything you ship, run a small MOS-style listening test (a handful of raters, fixed prompts, blind A/B against a baseline) and treat the automatic metrics as a fast proxy you validate against ears periodically. The honest harness is: FAD with a *stated* embedding and sample size for distributional quality, CLAP for adherence, and human MOS for the final call. Report all three and you are being honest; report one cherry-picked number and you are doing marketing.

#### Worked example: a guidance sweep evaluated

You want to set `guidance_scale` well. Generate 500 clips at each of guidance ∈ {1, 3, 5, 7}, with fixed prompts and seeds so only guidance varies. Compute FAD (CLAP embedding, 500 clips) and mean CLAP score at each setting. The pattern you will typically see: CLAP score rises with guidance (the output follows the prompt harder) while FAD is flat or improving up to a point and then *worsens* as high guidance introduces artifacts and reduces diversity. The best operating point is the guidance where CLAP is high but FAD has not yet turned up, usually around the model's default of 3.0, which is the default precisely because someone ran this sweep. The lesson: the default is not arbitrary; it is the elbow of exactly this curve, and you only need to re-tune it if your prompts or domain differ a lot from the training distribution.

## Case studies: real numbers from the literature

Let me ground the claims in named results, with honest caveats about precision.

**MusicGen's own ablation of the codebook pattern.** The MusicGen paper (Copet et al., 2023) directly compared the interleaving patterns. The headline result is the one this whole post was built around: the **delay pattern roughly matches the flatten pattern's quality at roughly a quarter of the autoregressive steps** for $K=4$, while the fully parallel pattern is meaningfully worse on FAD because it severs the residual codebook dependency. This is the empirical justification for the entire delay-pattern design, and it is the number to remember: same quality, $\approx 4\times$ fewer steps. The exact FAD deltas depend on the embedding and the eval set; the *ordering* (delay ≈ flatten > parallel on quality, delay ≈ parallel ≪ flatten on cost) is robust.

**MusicGen FAD and the open-model bar.** Across the reported evaluations, MusicGen-large lands at a Fréchet Audio Distance (FAD with the VGGish embedding) in the low single digits on MusicCaps-style prompts, competitive with MusicLM as reported by Google, with the enormous practical advantage that you can actually download and run it. The exact figure varies by eval protocol; treat "FAD in the low single digits, competitive with MusicLM" as the defensible claim and the embedding-and-set-specific number as something to recompute yourself if it matters. As the [evaluation post](/blog/machine-learning/audio-generation/audio-quality-metrics) stresses, FAD is sensitive to the embedding network and the sample size, so cross-paper FAD comparisons are only loosely meaningful unless the protocol matches.

**Melody conditioning's measured effect.** The paper reports that melody conditioning improves *melodic* adherence (measured by chroma similarity to the reference) substantially, at a small cost to the unconditioned FAD, because constraining the pitch contour slightly reduces the model's freedom to produce the most natural-sounding output. This is the control-versus-fidelity trade from the conditioning post, made quantitative: more control (the chromagram pins the melody), marginally less raw fidelity (a touch higher FAD). It is a trade you take gladly when you care about the tune.

**The open release as the real result.** The most consequential "number" is not a metric at all: MusicGen's permissive open release turned it into the community default essentially overnight. The flood of finetunes, the integration into `transformers` and `diffusers`-adjacent tooling, the Hugging Face Spaces demos, the downstream products, all of it followed from "you can download the weights." MusicLM's quality was arguably comparable, but its impact on the open ecosystem was near zero because there was nothing to download. For a practitioner, *availability is a feature*, and it is the one that most determines what you actually build with.

## What comes after: diffusion and the commercial frontier

The codec-LM approach is one of three lines in music generation, and it is worth previewing the other two so you know where MusicGen sits.

**Latent diffusion for music** is the second line. Instead of autoregressing over discrete codec tokens, you run a [diffusion model](/blog/machine-learning/audio-generation/diffusion-for-audio) over a *continuous* latent (a codec or autoencoder latent), the same recipe as latent diffusion for images. The standout is **Stable Audio** (Evans et al., 2024), which added **timing conditioning**: the model is told the desired duration and where in the clip it is, which lets it generate *variable-length* audio with a real beginning and end rather than the fixed-window output of the AR models, and it generates the whole clip in parallel denoising steps rather than token by token. We dig into this in the [Stable Audio post](/blog/machine-learning/audio-generation/latent-diffusion-for-music-stable-audio); the one-line contrast is that diffusion-music trades the AR model's clean causal conditioning for parallel generation and better length control.

**The commercial frontier** is the third line. **Suno** and **Udio** generate full songs, with vocals, lyrics, verse-chorus structure, and minutes of coherent form, the exact three walls MusicGen runs into. Their recipes are not public, but they almost certainly combine a codec-LM-style backbone with heavy investment in long-context modeling, vocal and lyric conditioning, and large proprietary datasets. The [commercial-frontier post](/blog/machine-learning/audio-generation/suno-udio-and-the-commercial-music-frontier) treats them as a report with honest uncertainty. The takeaway for now: the open codec-LM models gave the community a real, controllable, instrumental music generator; the commercial systems pushed past the length and vocal walls that the open models still hit.

MusicGen remains the right tool for a large class of jobs precisely because it is open, controllable, melody-conditionable, and runs on hardware you own. Knowing exactly where its walls are is what lets you reach for it confidently when the job fits inside them.

## When to reach for this (and when not to)

A decisive guide, because "use a music model" is not specific enough.

**Reach for MusicGen when** you need open weights you can run and finetune locally; when you want instrumental music up to about thirty seconds; when you want *melody control*, the hum-a-tune capability is genuinely unique among open models; when you are building a product where data must stay on your own infrastructure; or when you want a base to finetune on a specific style or instrument. It is the open workhorse for a reason.

**Do not reach for MusicGen when** you need vocals or lyrics, it simply cannot do them; when you need a coherent multi-minute piece with real verse-chorus form, the ~30-second wall and the drift will defeat you, and sliding-window continuation is a patch, not a solution; when you need the absolute top-of-market quality and structure that the commercial systems deliver and you are willing to use a closed API; or when variable-length output with clean endings matters, where [Stable Audio's timing conditioning](/blog/machine-learning/audio-generation/latent-diffusion-for-music-stable-audio) is the better fit.

**Do not reach for MusicLM at all as a tool**, because there is no public checkpoint. Reach for it as *ideas*: the hierarchical semantic-then-acoustic decomposition and the MuLan contrastive-conditioning trick are both important and both reappear elsewhere. But to actually generate music today, MusicGen or a diffusion model is what you run.

**Do not pick the largest model reflexively.** The two-stage funnel, explore cheaply on `small`, refine finalists on `large`, beats running everything on `large` for any workflow where you generate many candidates. Match the model size to the stage of your workflow, not to a vague desire for "best."

## Key takeaways

- **Music is harder than speech** because there is no transcript to anchor on, everything (harmony, rhythm, timbre) must be coherent at once, structure spans minutes while detail spans milliseconds, and trained ears catch every error.
- **The codec-LM idea** is the unifier: a [neural codec](/blog/machine-learning/audio-generation/encodec-dac-and-the-modern-codec) turns the waveform into a short token sequence and a transformer predicts those tokens. The codec is what makes the language-model framing tractable.
- **MusicLM is the hierarchical bet**: three stages (semantic → coarse → fine acoustic) plus MuLan, a contrastive text-music encoder that sidesteps scarce paired data by training on audio conditioning and steering with text at inference. Excellent, but closed.
- **MusicGen is the simplification**: one transformer over EnCodec tokens, T5 text conditioning, and a **delay pattern** that offsets codebook $k$ by $k-1$ steps so one model emits all $K$ codebooks with the right causal dependencies.
- **The delay pattern costs $S + K - 1$ steps** versus $K \times S$ for flattening, a ~$4\times$ speedup at $K=4$, while preserving the residual codebook dependency that the fully parallel pattern destroys. Same quality, a quarter of the steps.
- **Melody conditioning via a chromagram** is MusicGen's standout: a time-aligned 12-dimensional pitch-class track pins the tune while text sets the style, and the dense time-aligned signal dominates the sparse text signal when they conflict.
- **The model family trades quality for latency monotonically** (300M small → 1.5B medium → 3.3B large), and the `melody` checkpoint adds chromagram input; funnel cheaply on small, refine on large.
- **The three walls are real**: ~30-second coherence, no vocals or lyrics, and drift on long generations. Knowing exactly where they are is what makes MusicGen safe to reach for.
- **Availability is a feature.** MusicGen's open release, not just its quality, is why it became the default. The next leaps, [Stable Audio's length control](/blog/machine-learning/audio-generation/latent-diffusion-for-music-stable-audio) and [Suno/Udio's vocals](/blog/machine-learning/audio-generation/suno-udio-and-the-commercial-music-frontier), addressed exactly the walls MusicGen hit.

This post is one stop on the [audio stack](/blog/machine-learning/audio-generation/why-audio-generation-is-hard); when you are ready to wire a music model into a real serving pipeline with a codec, a vocoder, eval, and cost, the [capstone on building an audio generation stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack) puts all the pieces together. For the discrete-token-versus-continuous-latent framing applied to a different modality, the image series' [autoregressive image models](/blog/machine-learning/image-generation/autoregressive-image-models) post covers the same codec-LM tension for pixels.

## Further reading

- **Agostinelli et al., "MusicLM: Generating Music From Text" (2023)** — the hierarchical semantic-then-acoustic music model and the MuLan-conditioning recipe.
- **Huang et al., "MuLan: A Joint Embedding of Music Audio and Natural Language" (2022)** — the contrastive text-music model that conditions MusicLM.
- **Copet et al., "Simple and Controllable Music Generation" (MusicGen, 2023)** — the single-stage codec LM, the codebook-interleaving ablation, and melody conditioning.
- **Borsos et al., "AudioLM: a Language Modeling Approach to Audio Generation" (2022)** — the semantic-then-acoustic recipe MusicLM applies to music.
- **Défossez et al., "High Fidelity Neural Audio Compression" (EnCodec, 2022)** — the codec whose RVQ tokens MusicGen models.
- **Evans et al., "Stable Audio: Fast Timing-Conditioned Latent Audio Diffusion" (2024)** — the diffusion alternative with variable-length, timing-conditioned generation.
- **🤗 `transformers` MusicGen docs** and the **`audiocraft`** repository — the runnable APIs (`MusicgenForConditionalGeneration`, `MusicGen.get_pretrained`, `generate_with_chroma`) used in this post.
- Within this series: [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard), [semantic vs acoustic tokens](/blog/machine-learning/audio-generation/semantic-vs-acoustic-tokens), [EnCodec, DAC, and the modern codec](/blog/machine-learning/audio-generation/encodec-dac-and-the-modern-codec), [conditioning and control](/blog/machine-learning/audio-generation/conditioning-and-control-in-audio-generation), and the [capstone](/blog/machine-learning/audio-generation/building-an-audio-generation-stack).
