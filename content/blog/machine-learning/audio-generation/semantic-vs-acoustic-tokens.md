---
title: "Semantic vs Acoustic Tokens: The Two-Stream Trick Behind Audio LMs"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Why audio language models split sound into two token streams, and how the semantic-then-acoustic hierarchy made AudioLM, MusicLM, and VALL-E actually work."
tags:
  [
    "audio-generation",
    "audio-synthesis",
    "semantic-tokens",
    "acoustic-tokens",
    "neural-audio-codec",
    "text-to-speech",
    "music-generation",
    "generative-ai",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Audio Generation"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/semantic-vs-acoustic-tokens-1.png"
---

The first time I tried to build an audio language model, I did the obvious thing. I took a neural codec, the kind we spent the whole of [the codec post](/blog/machine-learning/audio-generation/encodec-dac-and-the-modern-codec) on, encoded a few thousand clips of speech into discrete tokens, and trained a plain decoder-only transformer to predict the next token, exactly the way you would train a tiny language model on text. The codec was good. Decoded ground-truth tokens sounded clean. The transformer's loss went down nicely. And when I sampled from it, the output was a catastrophe that was somehow also impressive: every individual 50-millisecond fragment sounded like a real human voice, with realistic timbre and breath and room tone, but stitched together they were babble. The model produced the *texture* of speech with no *content*. It was a person fluently speaking a language that does not exist.

That failure is the subject of this entire post, and the fix for it is one of the most important ideas in modern audio generation. The problem was not the codec and it was not the transformer. The problem was that the tokens I was modeling, the codec's residual-vector-quantization tokens, encode *acoustic detail* beautifully but carry their long-range structure in a form that a flat language model struggles to learn. Words, melodies, the arc of a sentence over several seconds: that structure exists in the audio, but it is smeared across a token stream that is mostly preoccupied with how things sound rather than what is being said. AudioLM's authors named the cure precisely. You need **two kinds of tokens**: *semantic* tokens that capture the linguistic and melodic content and *acoustic* tokens that capture the recording detail, and you model them in a hierarchy, content first.

![A vertical stack showing a waveform splitting into a semantic token stream that captures content and an acoustic token stream that captures recording detail](/imgs/blogs/semantic-vs-acoustic-tokens-1.png)

The figure above is the whole idea on one slide, and it is the thing to hold in your head for the rest of this post. A single waveform gets encoded *twice*, in parallel, by two different systems. One system, a self-supervised speech or audio model like HuBERT or w2v-BERT, produces **semantic tokens** at a low rate that encode phonetic, linguistic, and melodic content while being remarkably blind to who is speaking. The other system, an RVQ neural codec like EnCodec, produces **acoustic tokens** at a higher rate that encode speaker identity, timbre, and the fine texture of the recording. The generative model then works content-first: a semantic model lays down the plan, and acoustic models render it into sound. This is the **information factorization** that made audio language models work, and by the end of this post you will understand exactly what each token type captures, why a self-supervised model yields good semantic tokens, why the hierarchy is more sample-efficient than a flat acoustic language model, how it maps onto text-to-speech in VALL-E and music in MusicLM, and how later single-stage models like MusicGen simplified the whole thing back down again. You will also see the actual code to extract both token streams and wire a coarse acoustic model onto semantics.

This sits squarely on the series spine: the **audio stack** of waveform to latent to generative core to vocoder back to waveform, under the tension of **fidelity, controllability, speed, and length**. The two-token trick is fundamentally a *length-and-coherence* intervention. It buys long-range structure that a flat acoustic model cannot hold, at the cost of two tokenizers and multiple inference stages. Whether that trade is worth it, and when it is not, is the running question. If you have not read [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard), the foundation post, the framing there about audio being short in wall-clock time but enormous in token count is exactly the pressure that makes this idea necessary.

## The flat acoustic model and why it babbles

Let me make the opening anecdote precise, because the failure mode is not a bug, it is structural, and understanding it tells you exactly what semantic tokens are for.

A neural audio codec encodes a waveform into a grid of discrete tokens. For a typical EnCodec configuration on 24 kHz audio, you get about 75 frames per second, and at each frame the residual vector quantizer emits one token per codebook, with eight codebooks stacked. So one second of audio becomes $75 \times 8 = 600$ tokens arranged as a $75 \times 8$ grid. The codec was trained, as we covered in [residual vector quantization](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq), purely to *reconstruct* the waveform with minimal perceptual distortion. Its objective contains no notion of words, phonemes, or musical phrases. It is a compression artifact, optimized for rate and distortion, nothing more.

Now think about what a flat autoregressive language model over those tokens has to learn. To produce a coherent sentence, the model must keep track of which word it is in the middle of, which word should come next given the grammar and meaning so far, the speaker's identity so the voice stays consistent, the prosodic contour, and the exact spectral detail of the current 13-millisecond frame, all at once, all encoded in the same flat token stream. The catch is that the *acoustic* dependencies are short-range and overwhelmingly strong, while the *linguistic* dependencies are long-range and comparatively weak in the token statistics. Frame $t$ is enormously predictable from frame $t-1$ because adjacent audio frames are nearly identical. The fact that you are three words into a sentence about a brown fox is a far subtler signal, spread across hundreds of tokens.

Gradient descent goes for the easy wins. The model rapidly learns the short-range acoustic correlations because they crush the loss, and it largely ignores the long-range linguistic structure because modeling it correctly barely moves the cross-entropy. The result is exactly what I heard: locally gorgeous audio, globally incoherent. The voice is real; the speech is nonsense. In music the analogous failure is a model that produces a convincing instrument timbre but wanders out of key and loses the beat over a few bars. The codec did its job. The language model did *a* job, just not the one you wanted, because the signal that would have taught it the right job was buried.

This is the core motivation, and it generalizes beyond audio: when a representation packs many kinds of structure into one stream with wildly different strengths, a single flat model tends to capture the loud, local structure and miss the quiet, global structure. The fix in audio was to *separate* the streams.

It is worth being precise about *why* the loss barely moves when the model gets the content wrong, because that is the crux. Cross-entropy loss on the next acoustic token is overwhelmingly dominated by the *easy* tokens. The vast majority of frames in a speech signal are smoothly continuing a vowel or a steady-state sound, and for those the next token is nearly deterministic given the last few. The model nails them, the per-token loss on them is tiny, and they make up most of the average. The *hard* tokens, the ones at word and phrase boundaries where the content genuinely branches, are a small minority of positions and contribute little to the mean loss even when the model gets them completely wrong. So a model that produces locally-perfect, globally-incoherent audio sits at a *low average loss* that looks like success on a training curve. This is the trap: your metric says the model is good, your ears say it is babbling, and both are correct, because the metric is averaging over a token population where content decisions are rare. The semantic stream fixes this by *isolating* those content decisions onto a separate, short sequence where they are no longer a rare minority but the dominant signal, so the loss on the semantic model actually rewards getting them right.

There is a tempting wrong fix worth dismissing: just make the flat acoustic model bigger and train it longer. Scale does help, and a large enough acoustic LM with enough data does learn *some* long-range structure, which is part of why heavily-conditioned single-stage models work at all. But scale is an expensive and indirect way to buy what the factorization gives you cheaply. You are paying for the model to *rediscover* the content-versus-detail separation from raw token statistics, when you could have handed it that separation for free by tokenizing twice. The two-stream trick is, in this light, an *inductive bias*: it bakes the right factorization into the representation rather than hoping the model learns it. Good inductive biases beat brute scale on sample efficiency, and that is exactly the comparison AudioLM's ablation made.

![A before and after comparison contrasting a flat acoustic language model that drifts on words and melody against a two-stream semantic plus acoustic model that stays coherent over seconds](/imgs/blogs/semantic-vs-acoustic-tokens-3.png)

The contrast above is the thesis of the post stated as an engineering result. On the left, one token stream and one model: great local fidelity, but content drifts because the long-range signal is too weak to compete. On the right, a semantic stream that *plans* the content and an acoustic stream that *renders* it: the planning model only ever sees the content-bearing tokens, so it cannot be distracted by acoustic detail, and coherence over seconds becomes learnable. We will spend the rest of the post making each half of that picture concrete.

## What a semantic token captures, and what an acoustic token captures

The word "semantic" here is a term of art and slightly overloaded, so let me pin it down. A **semantic token** in the AudioLM sense is not a word embedding or a meaning vector in the NLP sense. It is a discrete index derived from the hidden representations of a self-supervised speech or audio model, and the empirical fact that earns it the name is this: those representations correlate strongly with *phonetic and linguistic content* and weakly with *speaker identity and acoustic detail*. They tell you roughly *what sound* is being made, not *who* is making it or *in what room*.

An **acoustic token** is the opposite. It is the codec's RVQ token, and it encodes everything you need to *reconstruct the exact waveform*: the speaker's vocal-tract timbre, the pitch, the recording's noise floor and reverb, the microphone coloration, the breath. It is a near-complete description of the sound as a physical signal, with very little explicit notion of linguistic content.

So the two token types **factorize the information** in a waveform along a content-versus-detail axis. This is the single most important conceptual point in the post, so let me state it as cleanly as I can. A speech waveform $x$ carries, roughly, two largely separable kinds of information: linguistic content $c$ (the words, the phonemes, the melody) and acoustic identity $a$ (the speaker, the timbre, the recording conditions). Semantic tokens are a lossy code for $c$ that throws away most of $a$. Acoustic tokens are a near-lossless code for $x$ that contains both $c$ and $a$ but in a form where $c$ is hard to read off. The generative recipe exploits this by modeling $c$ first, then modeling $a$ conditioned on $c$.

![A matrix comparing semantic, coarse acoustic, and fine acoustic tokens across their source model, what they capture, their token rate, and what they discard](/imgs/blogs/semantic-vs-acoustic-tokens-4.png)

The matrix above lays out the three token roles you will meet in a full AudioLM-style stack. Note there are *three*, not two, because the acoustic tokens themselves split into coarse and fine. Semantic tokens come from a self-supervised model at a low rate, around 25 to 50 tokens per second, and discard speaker and timbre. **Coarse acoustic tokens** are the first one or two RVQ codebooks; they carry the bulk of the speaker identity and pitch and run at the codec frame rate. **Fine acoustic tokens** are the remaining codebooks; they carry the last increments of texture and room detail. The hierarchy walks down this matrix top to bottom: content, then coarse sound, then fine sound.

Why does this factorization exist at all? Why should a self-supervised model's features happen to encode content and discard speaker? That is not a coincidence, and the answer is the most beautiful part of the story.

### Worked example: how speaker-invariant are semantic tokens, really?

#### Worked example: measuring the content-vs-speaker split

Here is a concrete way to see the factorization, the kind of probe I run whenever I am told a representation is "speaker-invariant" and want to check. Take a self-supervised model such as HuBERT, extract frame features for a labeled speech dataset, and quantize them to discrete tokens with k-means (the standard recipe, more on this below). Now train two tiny linear classifiers directly on those discrete tokens: one to predict the *phoneme* at each frame, and one to predict the *speaker identity* of the clip. In the published HuBERT and AudioLM analyses, the phoneme probe is strong, frame-level phone classification well above chance and competitive with supervised features, while the speaker probe is weak relative to what you get from raw spectral features or from codec tokens. Concretely, reported phoneme purity for HuBERT k-means clusters lands in the rough vicinity of 60 to 70 percent depending on the layer and cluster count, while the same clusters carry much less speaker information than an equal-bitrate acoustic code.

The number to internalize is not the exact percentage, which depends heavily on layer choice, cluster count, and dataset, so treat these as order-of-magnitude. The point is the *gap*: semantic tokens are far more phonetically informative than speaker-informative, and acoustic tokens are the reverse. That gap is the entire engineering lever. If you wanted a single takeaway test for "is this a good semantic tokenizer," it is "does phoneme purity stay high while speaker information drops."

Here is the actual measurement, the probe I run to pick a layer and a cluster count. **Phoneme purity** is computed by aligning each frame's semantic token to its ground-truth phoneme label (you need a forced alignment, which a tool like the Montreal Forced Aligner provides), then asking, for each *cluster*, what fraction of its frames share the most common phoneme in that cluster. A purity near 1.0 means each semantic token cleanly corresponds to a single phone; a purity near chance means the tokens are phonetically meaningless. The mirror-image metric, **cluster purity** of speaker labels, tells you how much speaker identity leaked.

```python
import numpy as np
from collections import Counter

def cluster_purity(token_ids, labels):
    """token_ids and labels are aligned 1-D arrays over frames.
    Returns the weighted fraction of frames whose label matches the
    majority label of their assigned cluster (1.0 = perfectly pure)."""
    correct, total = 0, 0
    for tok in np.unique(token_ids):
        members = labels[token_ids == tok]
        if len(members) == 0:
            continue
        majority = Counter(members).most_common(1)[0][1]
        correct += majority
        total += len(members)
    return correct / total

phoneme_purity = cluster_purity(semantic_tokens.numpy(), phoneme_labels)  # want HIGH
speaker_purity = cluster_purity(semantic_tokens.numpy(), speaker_labels)  # want LOW-ish
print(f"phoneme purity {phoneme_purity:.3f}   speaker purity {speaker_purity:.3f}")
```

Sweep the HuBERT layer index from, say, 6 through 11 and the cluster count $K$ from 100 through 2000, and plot phoneme purity against speaker purity. The layer-and-$K$ you want is the knee where phoneme purity is near its peak while speaker purity is as low as you can keep it. That single sweep is the most useful hour you can spend before committing to a semantic tokenizer, and it is the thing most people skip and then wonder why their cloning is fighting their content.

## Why a self-supervised model gives you good semantic tokens

To understand why HuBERT or w2v-BERT features are content-rich and speaker-poor, you have to look at what task produced them, because a representation is shaped by the prediction problem it was trained to solve. Both models are trained with **masked prediction**, the audio analogue of BERT's masked language modeling, and that objective has a specific consequence.

HuBERT (Hidden-unit BERT, from Hsu and colleagues at Meta, 2021) works in two phases. First, it runs a clustering step: it takes some acoustic features, in the first iteration just MFCCs, the classic mel-frequency cepstral coefficients, and runs k-means to assign every frame a discrete cluster label. These labels are crude and noisy, but they carry a weak signal about which broad sound class a frame belongs to. Then it trains a transformer: it masks spans of the input audio frames and asks the model to predict the *cluster labels* of the masked frames from the surrounding unmasked context. After training, it re-clusters using the model's own learned features, which are far better than MFCCs, and trains again. Each iteration sharpens the labels and the representation.

Now reason about what masked prediction forces the model to learn. To predict the cluster label of a masked region from its neighbors, the model must use *context*, which means it must learn the structure of how sounds follow each other, which is phonotactics and, ultimately, linguistic structure. Predicting a masked phoneme-like unit from surrounding ones is fundamentally a content-modeling task. What the objective does *not* reward is preserving speaker identity, because speaker identity is roughly constant across the whole clip and therefore provides almost no help in the contextual prediction of one masked frame from its neighbors. The masking objective pushes the model toward features that are *predictive of local content given context* and lets speaker information atrophy because it is not useful for the task. That is the mechanism. Speaker invariance is not designed in; it falls out of the prediction problem.

w2v-BERT (Chung and colleagues, 2021) combines a contrastive objective in the style of wav2vec 2.0 with a masked-prediction objective in the style of BERT, and it is the model AudioLM and MusicLM actually used. The contrastive part learns to discriminate the true latent for a masked frame from distractors, which again rewards contextual content prediction; the masked-prediction part is HuBERT-like. The upshot is the same: features that are excellent at content and lossy about speaker.

This is exactly analogous to how masked image modeling and contrastive learning produce features useful for recognition, and if you want the parallel from the image world, the way self-supervised pretraining shapes which information a representation keeps is the same phenomenon we leaned on in [autoregressive image models](/blog/machine-learning/image-generation/autoregressive-image-models) and in the VAE-as-codec analogy in [variational autoencoders from scratch](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch). The lesson travels across modalities: the pretraining task selects what survives in the features.

### From continuous features to discrete semantic tokens

A self-supervised model gives you *continuous* feature vectors, one per frame, typically 768 or 1024 dimensions. To turn those into a token *language* a transformer can model autoregressively, you need to discretize them, and the standard, almost boringly simple, method is **k-means**. You take the feature vectors from a chunk of training audio, run k-means with some number of clusters $K$ (commonly several hundred to a thousand, AudioLM used 1024 for w2v-BERT), and the cluster index of each frame's feature vector becomes its semantic token. The codebook is the set of $K$ centroids; encoding a new frame is a nearest-centroid lookup.

Two design knobs matter. The **layer** you extract from: middle-to-late layers of HuBERT and w2v-BERT tend to be the most content-rich, and the exact best layer is found empirically by the phoneme-purity probe above. The **number of clusters** $K$: more clusters give a finer, higher-bitrate semantic code that captures more nuance but is harder to model and starts to leak acoustic detail; fewer clusters give a coarser, more purely-content code. The token *rate* also matters: HuBERT runs at 50 frames per second, but AudioLM further deduplicates *runs* of identical semantic tokens, collapsing them to a lower effective rate, which both shortens the sequence and emphasizes content over duration. A common simplification, used in textless-NLP and many TTS systems, is to collapse consecutive duplicate tokens entirely, so "a a a b b c" becomes "a b c", trading away explicit duration for a much shorter, more linguistic sequence.

## The hierarchical recipe: semantic, then coarse acoustic, then fine acoustic

Now we can assemble the full AudioLM generation pipeline, the three-stage hierarchy that is the heart of this post.

![A graph of AudioLM's three-stage hierarchy where a semantic language model feeds coarse and fine acoustic models conditioned by a speaker prompt](/imgs/blogs/semantic-vs-acoustic-tokens-2.png)

The figure above shows the data flow. Read it as three transformers chained, each consuming what the previous one produced.

**Stage one, the semantic language model.** A decoder-only transformer is trained on the *semantic token* sequences alone. Given a prompt (which can be empty for unconditional generation, a short audio continuation, or, in MusicLM, a text-derived conditioning), it autoregressively generates a semantic token sequence. Because this model only ever sees content-bearing tokens, it can devote all its capacity to long-range structure: words following words, phrases following phrases, a melody that resolves. It is, functionally, a language model over a phonetic-melodic alphabet. This is where the coherence comes from.

**Stage two, the coarse acoustic language model.** A second transformer is trained to generate the *first one or two RVQ codebooks* of the acoustic tokens, conditioned on the full semantic token sequence. Its input is the semantic tokens (the content plan) plus, optionally, a short acoustic prompt that fixes the speaker. Its output is the coarse acoustic stream. This stage decides *how the content sounds*: it picks a voice, a pitch contour, the gross spectral shape, all consistent with the semantic plan it was handed.

**Stage three, the fine acoustic language model.** A third transformer generates the *remaining RVQ codebooks*, conditioned on the coarse acoustic tokens (and within AudioLM, a local window of them, since fine detail is short-range). This stage adds the last increments of fidelity, the texture and room and high-frequency detail that the first codebooks omit. Finally the full stack of acoustic tokens is handed to the codec decoder, which reconstructs the waveform.

The flow is **semantic to acoustic, and coarse to fine**. Content is decided before sound; gross sound is decided before fine sound. Each stage conditions on a more abstract, longer-range plan and adds shorter-range detail. The speaker prompt enters at the acoustic stage, not the semantic stage, which is exactly why AudioLM can take three seconds of an unseen speaker and continue in their voice: the semantic tokens carry the content, and the acoustic model paints that content in the prompted speaker's timbre.

### Why the hierarchy is more sample-efficient than a flat model

This is the science block, the part where the *why* becomes provable rather than asserted. The claim is that factoring the joint distribution into a hierarchy is easier to learn than modeling the flat acoustic stream directly. Here is the argument in the language of probability.

Write the full acoustic token sequence as $A$ and the semantic token sequence as $S$. A flat acoustic language model directly factorizes and learns

$$
p(A) = \prod_{t} p(a_t \mid a_{\lt t}).
$$

The hierarchical model instead learns a *factorized* joint over both streams,

$$
p(S, A) = p(S)\, p(A \mid S) = \underbrace{\prod_i p(s_i \mid s_{\lt i})}_{\text{semantic LM}} \cdot \underbrace{\prod_t p(a_t \mid a_{\lt t}, S)}_{\text{acoustic LM}},
$$

and at generation time it samples $S \sim p(S)$ and then $A \sim p(A \mid S)$. To see why this helps, look at the second factor, the conditional $p(A \mid S)$. *Given the semantic tokens*, the acoustic model no longer has to invent the content; the content is fixed by $S$. Its only job is to render that content acoustically, which is a far more local, lower-entropy problem. The long-range dependencies that were so hard to learn in the flat $p(A)$ have been *moved into* $S$, and $S$ is modeled by a dedicated network on a short, content-pure sequence where those dependencies are the dominant signal rather than a faint one.

You can sharpen this with an information-theoretic lens. The entropy of the acoustic stream decomposes as

$$
H(A) = H(S) + H(A \mid S) - I(A; S) + H(S \mid A),
$$

but the practical point is simpler than the full identity. Because $S$ is a near-deterministic function of $A$ (you can extract semantic tokens from the audio that the acoustic tokens encode), $H(S \mid A) \approx 0$, and the useful reading is that the hard, high-mutual-information long-range part of $A$ is captured by $S$. The conditional $p(A \mid S)$ has had its long-range uncertainty largely removed; what remains is dominated by short-range acoustic detail that a model learns easily. Modeling $p(S)$ on a short sequence and $p(A \mid S)$ as a local problem is strictly easier than modeling $p(A)$, which had to do both jobs in one network on a long sequence.

There is a second, very concrete win: **sequence length**. Semantic tokens run at perhaps 25 to 50 per second after deduplication, while acoustic tokens run at 600 per second (75 frames times 8 codebooks). The expensive long-range modeling happens on the *short* semantic sequence, where a transformer's quadratic attention cost is cheap and its effective context in seconds is long. The acoustic model's context only needs to span the short range over which acoustic detail correlates, so it too can be efficient. A flat acoustic model, by contrast, has to attend across the full 600-tokens-per-second stream to capture the same long-range linguistic structure, which is both expensive and, as we saw, statistically swamped. The hierarchy puts each kind of dependency on a sequence whose length and statistics suit it. That is the sample-efficiency and the compute-efficiency argument in one.

### Why acoustic tokens split into coarse and fine

A natural question is why the acoustic side needs *two* stages rather than one. The answer is in the structure of residual vector quantization, which we derived in full in [the RVQ post](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq), but the relevant fact here is this. RVQ quantizes a vector by a *stack* of codebooks: the first codebook picks the nearest centroid to the input, the second codebook quantizes the *residual* (input minus first centroid), the third quantizes the residual of that, and so on. Each successive codebook refines the approximation. This means the codebooks are *ordered by importance*: the first one or two carry the bulk of the signal's energy and the gross structure (which for speech is dominated by the speaker's vocal-tract shape and pitch), while later codebooks carry progressively finer corrections (the high-frequency texture, the subtle room reverberation).

That ordering is exactly why the split into coarse and fine is natural and not arbitrary. The first codebooks are *high-information and long-range-correlated*: the speaker does not change mid-utterance, so codebook 1 is slowly varying and strongly tied to the content plan. The later codebooks are *low-information and short-range*: the exact fine texture at frame $t$ depends mostly on frames near $t$, not on what was said three seconds ago. So you model the coarse codebooks with a model that attends to the semantic plan and a wide context, and you model the fine codebooks with a model that only needs a local window. AudioLM's fine stage literally restricts its attention to a short window for this reason, and it makes the fine stage cheap. The coarse-to-fine split is the RVQ codebook ordering reflected directly into the model hierarchy: model the codebooks that carry structure first and with full context, model the codebooks that carry only detail later and locally.

This also explains a practical observation: you can often *drop* the last few fine codebooks at inference for a speed-or-bandwidth win and lose only a little perceptual quality, because they were the low-information corrections. You cannot drop the first codebook without the output collapsing, because it carries the speaker and the gross spectrum. The hierarchy's stages inherit this asymmetry, which is why the coarse stage is the one that must be right and the fine stage is the one you can economize.

#### Worked example: the token budget of ten seconds of speech

Take the running example of generating ten seconds of 24 kHz speech. Flat acoustic: $10 \times 75 \times 8 = 6{,}000$ acoustic tokens, and the model must learn word-level coherence across all 6,000 of them. Hierarchical: stage one generates roughly $10 \times 35 \approx 350$ semantic tokens (taking 35 per second after dedup), where word-level structure spans only a few hundred tokens and is the dominant signal; stage two then generates $10 \times 75 \times 2 = 1{,}500$ coarse acoustic tokens conditioned on those 350; stage three generates the remaining $10 \times 75 \times 6 = 4{,}500$ fine tokens conditioned on the coarse ones, but each fine token depends on only a *local* window, not the whole sequence. The hard long-range learning happened on a 350-token sequence instead of a 6,000-token one, a roughly 17× shorter sequence for the part of the problem that was actually hard. That is why AudioLM produced coherent speech where my flat model babbled.

## The code: extracting both token streams in PyTorch

Enough theory. Here is the practical flow, the actual code to extract semantic and acoustic tokens from a waveform. We will use HuBERT from 🤗 `transformers` for semantic features plus k-means for quantization, and EnCodec for acoustic tokens. This is the copy-and-adapt skeleton you would build a real pipeline on.

First, load audio and resample to the rate each model expects. HuBERT wants 16 kHz; EnCodec's 24 kHz model wants 24 kHz.

```python
import torch
import torchaudio

# Load a wav and keep two resampled copies: 16 kHz for the SSL model, 24 kHz for the codec.
wav, sr = torchaudio.load("the_quick_brown_fox.wav")  # shape: (channels, samples)
wav = wav.mean(dim=0, keepdim=True)                   # force mono: (1, samples)

wav16 = torchaudio.functional.resample(wav, sr, 16_000)
wav24 = torchaudio.functional.resample(wav, sr, 24_000)
```

Now extract continuous semantic features from HuBERT. We take the hidden states from a middle-to-late layer, which the literature finds most content-rich.

```python
from transformers import HubertModel, Wav2Vec2FeatureExtractor

extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960").eval()

with torch.no_grad():
    inputs = extractor(wav16.squeeze(0), sampling_rate=16_000, return_tensors="pt")
    # output_hidden_states gives every layer; layer 9 is a common content-rich choice for HuBERT-base.
    out = hubert(**inputs, output_hidden_states=True)
    feats = out.hidden_states[9].squeeze(0)   # shape: (frames, 768), ~50 frames/sec
```

These continuous features are not yet tokens. We quantize them with a k-means codebook that you fit once on a corpus of features and then reuse. Fitting the codebook is an offline step; here is both the fit and the encode.

```python
import numpy as np
from sklearn.cluster import MiniBatchKMeans

# ---- offline, once: fit a k-means codebook on a big pile of HuBERT features ----
# corpus_feats is a (N, 768) numpy array of features pooled across many clips.
kmeans = MiniBatchKMeans(n_clusters=1024, batch_size=10_000, n_init=10)
kmeans.fit(corpus_feats)                       # learns 1024 centroids = the semantic codebook
np.save("hubert_km1024.npy", kmeans.cluster_centers_)

# ---- at encode time: nearest-centroid lookup turns features into semantic tokens ----
centroids = torch.from_numpy(np.load("hubert_km1024.npy"))         # (1024, 768)
dists = torch.cdist(feats, centroids)                              # (frames, 1024)
semantic_tokens = dists.argmin(dim=1)                             # (frames,) ints in [0, 1024)

# Deduplicate consecutive identical tokens to emphasize content over duration.
keep = torch.ones_like(semantic_tokens, dtype=torch.bool)
keep[1:] = semantic_tokens[1:] != semantic_tokens[:-1]
semantic_tokens_dedup = semantic_tokens[keep]                     # shorter, content-pure sequence
```

That `semantic_tokens_dedup` tensor is the content stream, a sequence of integers in `[0, 1024)` at roughly 25 to 50 per second. Now the acoustic stream from EnCodec:

```python
from transformers import EncodecModel

codec = EncodecModel.from_pretrained("facebook/encodec_24khz").eval()

with torch.no_grad():
    # bandwidth selects how many RVQ codebooks are active; 6.0 kbps uses 8 codebooks at 24 kHz.
    enc = codec.encode(wav24.unsqueeze(0), bandwidth=6.0)
    acoustic_tokens = enc.audio_codes  # shape: (1, 1, num_codebooks=8, frames=~75/sec)

acoustic_tokens = acoustic_tokens.squeeze(0).squeeze(0)  # (8, frames)
coarse = acoustic_tokens[:2]                             # first 2 codebooks = coarse stream
fine = acoustic_tokens[2:]                               # remaining 6 codebooks = fine stream
```

![A graph showing one waveform feeding two parallel encoders, a self-supervised model producing semantic tokens and a codec producing acoustic tokens](/imgs/blogs/semantic-vs-acoustic-tokens-5.png)

The figure above is exactly what this code does: one waveform, two parallel encoders, two token streams. Note the asymmetry that the figure encodes. The semantic path has an *extra* step, the k-means quantization, because the self-supervised model emits continuous features and you must discretize them yourself. The acoustic path's quantization is *built in*, because the codec's RVQ already produces discrete tokens. This is the single most common practical snag for people building these pipelines: an off-the-shelf codec hands you tokens, but an off-the-shelf SSL model hands you floats, and you own the k-means.

You now have everything an AudioLM-style stack needs: `semantic_tokens_dedup` for stage one, `coarse` for stage two, `fine` for stage three. The remaining work is the three transformers and their conditioning wiring, which we sketch next.

## Wiring a coarse acoustic model onto semantics

The conditioning of the coarse acoustic model on the semantic tokens is the join that makes the hierarchy work, so let me show it concretely. The cleanest way to condition one token stream on another in a decoder-only transformer is to *prefix* the conditioning sequence: you concatenate the semantic tokens (the condition) and the acoustic tokens (the target) into one sequence with distinct embedding tables and a separator, and you compute the autoregressive loss *only on the acoustic positions*. The model attends back over the whole semantic prefix while predicting each acoustic token.

```python
import torch
import torch.nn as nn

class CoarseAcousticLM(nn.Module):
    """A decoder-only transformer that predicts coarse acoustic tokens conditioned
    on a semantic-token prefix. Semantic and acoustic use separate embedding tables."""
    def __init__(self, n_sem=1024, n_aco=1024, n_codebooks=2, d_model=1024, n_layer=12):
        super().__init__()
        self.sem_emb = nn.Embedding(n_sem + 1, d_model)          # +1 for a separator id
        # One embedding per coarse codebook, summed at each acoustic frame.
        self.aco_emb = nn.ModuleList(nn.Embedding(n_aco, d_model) for _ in range(n_codebooks))
        layer = nn.TransformerDecoderLayer(d_model, nhead=16, batch_first=True)
        self.backbone = nn.TransformerDecoder(layer, num_layers=n_layer)
        self.heads = nn.ModuleList(nn.Linear(d_model, n_aco) for _ in range(n_codebooks))
        self.sep_id = n_sem

    def forward(self, sem_tokens, aco_tokens):
        # sem_tokens: (B, S) semantic ids.  aco_tokens: (B, T, n_codebooks) coarse ids.
        B, S = sem_tokens.shape
        sep = torch.full((B, 1), self.sep_id, device=sem_tokens.device)
        sem_x = self.sem_emb(torch.cat([sem_tokens, sep], dim=1))       # (B, S+1, d)
        # Sum the per-codebook embeddings to get one vector per acoustic frame.
        aco_x = sum(emb(aco_tokens[..., k]) for k, emb in enumerate(self.aco_emb))  # (B, T, d)
        x = torch.cat([sem_x, aco_x], dim=1)                           # prefix + target
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        h = self.backbone(x, x, tgt_mask=mask, memory_mask=mask)
        h_aco = h[:, sem_x.size(1):]                                   # only acoustic positions
        return [head(h_aco) for head in self.heads]                   # per-codebook logits
```

The loss is a sum of cross-entropies over the coarse codebooks, computed only on the acoustic positions:

```python
import torch.nn.functional as F

def coarse_loss(model, sem_tokens, aco_tokens):
    logits_per_cb = model(sem_tokens, aco_tokens[:, :-1])   # teacher forcing, predict next frame
    targets = aco_tokens[:, 1:]                             # (B, T-1, n_codebooks)
    loss = 0.0
    for k, logits in enumerate(logits_per_cb):
        loss = loss + F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets[..., k].reshape(-1),
        )
    return loss / len(logits_per_cb)
```

The key lines are the prefix concatenation and the slice `h[:, sem_x.size(1):]` that restricts the prediction to acoustic positions. The semantic tokens are *context the model reads*, never something it predicts, which is exactly the conditional $p(A \mid S)$ from the math: the model learns the distribution of acoustic tokens *given* the semantic plan. The fine model is the same pattern, conditioned on the coarse tokens instead of semantics, often with a restricted local attention window because fine detail is short-range.

This prefix-conditioning trick is the same mechanism you would use to condition any AR model on a control signal, and it is worth recognizing it as a general tool. Text-to-speech, melody-conditioned music, and speaker-prompted cloning are all just *different things put in the prefix*.

## How this maps onto text-to-speech: VALL-E

The hierarchy I have described is the *audio-to-audio* AudioLM form, where the semantic tokens come from a self-supervised model. The moment you want **text-to-speech**, you have a better source for the content stream than self-supervised tokens: the *text itself*. This is the insight behind VALL-E (Wang and colleagues, Microsoft, 2023), and it is a clean specialization of the two-stream idea.

![A vertical stack showing VALL-E mapping phonemes and a three-second acoustic prompt through an autoregressive then non-autoregressive model to a cloned voice](/imgs/blogs/semantic-vs-acoustic-tokens-8.png)

The stack above shows VALL-E's structure. VALL-E reframes TTS as *codec-token language modeling*. The content stream is **phonemes** derived from the input text (a grapheme-to-phoneme step), playing exactly the role AudioLM's semantic tokens play: they specify *what* is said while carrying no speaker identity. The acoustic stream is **EnCodec tokens**. The model has two parts that mirror AudioLM's coarse and fine stages:

- An **autoregressive (AR) model** predicts the *first* codec codebook conditioned on the phonemes and on a short *acoustic prompt*, three seconds of the target speaker's EnCodec tokens. This is the stage that determines duration and the coarse acoustic content, and crucially, by conditioning on the three-second prompt, it performs **zero-shot voice cloning**: the model has never seen this speaker in training, but it continues the prompt's voice the way a text language model continues a writing style from a few sentences. This is in-context learning, applied to timbre.
- A **non-autoregressive (NAR) model** predicts the *remaining seven codebooks* in parallel, conditioned on the phonemes, the prompt, and the already-generated coarser codebooks. Parallel generation here is fine because, given the first codebook, the residual codebooks are largely conditionally independent across time, so you do not pay the autoregressive latency for the bulk of the tokens.

The mapping to our framework is exact. Phonemes are the content stream (the "semantic" tokens, but obtained for free from text rather than learned). The AR model is the coarse acoustic stage. The NAR model is the fine acoustic stage. The three-second prompt is the speaker conditioning entering at the acoustic stage, precisely as in AudioLM. The reason VALL-E can clone a voice from three seconds while a conventional TTS system needs minutes of target-speaker data to fine-tune is *the factorization*: content comes from text, so the only thing the prompt must supply is identity, and identity is exactly what acoustic tokens encode densely. We will go much deeper on VALL-E and the cloning frontier in [the neural codec language model TTS post](/blog/machine-learning/audio-generation/neural-codec-language-model-tts-vall-e); here the point is that it is the two-stream trick with phonemes standing in for learned semantic tokens.

Here is what running a modern descendant of this idea looks like in practice with a high-level toolkit, so the mapping stays grounded in real code:

```python
# A practical zero-shot clone with Coqui XTTS, a VALL-E-lineage system.
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

# speaker_wav is the ~3-6 second reference clip; the content comes from `text`.
tts.tts_to_file(
    text="The quick brown fox jumps over the lazy dog.",
    speaker_wav="reference_3s.wav",   # supplies identity (the acoustic prompt)
    language="en",
    file_path="cloned_fox.wav",
)
```

The `text` argument is the content stream and the `speaker_wav` is the acoustic prompt; the same content-versus-identity split runs all the way from AudioLM's research code to a one-line production API.

## How this maps onto music: MusicLM

Music has no phonemes, so where does the content stream come from? MusicLM (Agostinelli and colleagues, Google, 2023) answers with a clever twist on the AudioLM recipe. It keeps the three-stage **semantic to coarse to fine** hierarchy and the w2v-BERT semantic tokens, but it adds a *text-to-semantic* conditioning bridge so you can drive generation from a text prompt like "a calming violin melody backed by a distorted guitar riff."

The bridge is **MuLan**, a joint text-audio embedding model trained contrastively (the audio analogue of CLIP, which we used throughout the image series). At training time, MusicLM conditions the semantic stage on the MuLan *audio* embedding of the training clip; at inference time, it swaps in the MuLan *text* embedding of the prompt. Because MuLan aligns text and audio in a shared space, the model trained on audio embeddings generalizes to text embeddings, which is how a model trained without paired text-audio captions can still follow text prompts. The semantic tokens then carry the *melodic and structural content* of the music, just as they carry phonetic content for speech, and the acoustic stages render that content into a full-fidelity waveform with the timbre and texture of real instruments.

So across speech and music the *content stream* has three possible sources, and recognizing this unifies the whole landscape: learned self-supervised semantic tokens (AudioLM, MusicLM), phonemes from text (VALL-E and most TTS), or a text embedding bridged into semantic space (MusicLM's MuLan conditioning). The acoustic stream is almost always RVQ codec tokens. The hierarchy is the same.

![A matrix comparing AudioLM, MusicLM, VALL-E, and MusicGen across their semantic source, acoustic codec, number of stages, and task](/imgs/blogs/semantic-vs-acoustic-tokens-7.png)

The matrix above is the field on one slide. AudioLM and MusicLM use w2v-BERT semantic tokens and a three-stage hierarchy; MusicLM adds MuLan text conditioning. VALL-E uses phonemes and an AR-plus-NAR pair over EnCodec. MusicGen, the row that breaks the pattern, uses *no* semantic tokens at all, which brings us to the simplification.

## The single-stage simplification: MusicGen and codebook interleaving

The two-stream hierarchy is powerful, but look at what it costs: two tokenizers (a self-supervised model *and* a codec) and three sequential transformers at inference. That is a lot of moving parts, a lot of latency, and a lot of places for the stages to disagree. MusicGen (Copet and colleagues, Meta, 2023) asked whether you actually need the semantic tokens and the multi-stage hierarchy, and its answer was a qualified *no*, at least for text-conditioned music.

![A before and after comparison of the hierarchical AudioLM approach with two tokenizers and three models against the single-stage MusicGen approach with one codec and one model](/imgs/blogs/semantic-vs-acoustic-tokens-6.png)

The comparison above is the architectural simplification. MusicGen drops the semantic tokenizer and the semantic language model entirely. It models *only* the EnCodec acoustic tokens, conditioning directly on a text embedding from a frozen T5 encoder (and optionally a melody chroma feature for melody conditioning). It is a *single-stage* codec language model. But that immediately raises the problem we opened the post with: a flat model over codec tokens should babble. How does MusicGen avoid that?

The answer is two-fold. First, the conditioning is strong: a T5 text embedding is a rich content signal injected via cross-attention at every layer, which supplies much of the long-range structure that semantic tokens supplied in AudioLM. The content plan comes from text rather than from a separate generated stream. Second, MusicGen introduces **codebook interleaving patterns**, the clever bit, to handle the fact that EnCodec produces *eight parallel codebooks per frame* rather than one token per step.

The naive way to flatten a $K$-codebook, $T$-frame grid into a 1D sequence is to emit all $K$ codebooks of frame 1, then all $K$ of frame 2, and so on, giving a sequence of length $K \times T$ that is 8× longer than the frame count. MusicGen's **delay pattern** instead staggers the codebooks: codebook $k$ at frame $t$ is emitted at sequence position $t + k$, so the eight codebooks of a given frame are spread diagonally across eight consecutive steps. This lets the model predict all eight codebooks of a frame in parallel at inference (each conditioned on the appropriate delayed history) while keeping the sequence length at roughly $T + K$ rather than $K \times T$. The delay pattern recovers most of the efficiency of parallel codebook prediction without the full quality loss of predicting them fully independently, because each codebook still sees the lower-index codebooks of the same frame.

```python
# MusicGen single-stage text-to-music: one model, one codec, no semantic tokens.
import torch
from transformers import MusicgenForConditionalGeneration, AutoProcessor

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to("cuda")

inputs = processor(
    text=["lo-fi hip hop with a mellow piano and vinyl crackle"],
    padding=True, return_tensors="pt",
).to("cuda")

# guidance_scale is classifier-free guidance; max_new_tokens sets the length (50 tok/s here).
with torch.no_grad():
    audio = model.generate(**inputs, guidance_scale=3.0, max_new_tokens=512)
# audio: (1, 1, samples) at 32 kHz; ~10 seconds from 512 tokens.
```

The trade is real and worth stating plainly. The single-stage model is simpler, faster to run, and easier to train and serve, and for *text-conditioned* music it works extremely well, because text is a strong enough content signal to replace the semantic stream. What it gives up is the explicit, separately-modeled content plan, which matters more for *unconditional* generation and for tasks where the long-range structure is linguistic rather than describable in a short text prompt. AudioLM's unconditional speech continuation, where there is no text to lean on, genuinely benefits from the semantic stream in a way MusicGen's text-conditioned music does not. The field's practical resolution: if you have a strong conditioning signal (text, phonemes), a single-stage codec LM often suffices; if you need long-range coherence *without* a strong external content signal, the semantic stream earns its keep.

#### Worked example: comparing inference cost of hierarchical vs single-stage

Consider generating ten seconds of audio. The hierarchical AudioLM path runs three transformer decodes in sequence: a semantic decode over ~350 tokens, a coarse decode over ~1,500 tokens, and a fine decode over ~4,500 tokens, plus two encode passes at training/conditioning time. The single-stage MusicGen path runs *one* transformer decode; with the delay pattern over eight codebooks at ~50 frames per second, that is on the order of $10 \times 50 \approx 500$ autoregressive steps (each step emitting the delayed codebook set) rather than three separate decodes totaling thousands of steps. On a single A100, MusicGen-small generates roughly real-time or faster for short clips, with reported figures in the rough vicinity of a real-time factor near or below 1.0 for the small model and slower for the 3.3B model; treat these as order-of-magnitude, since RTF depends heavily on batch size, sequence length, and whether you compiled the model. The headline is that collapsing three stages to one, plus the delay pattern's parallelism, is a meaningful latency and simplicity win, which is why most *open* music generation today is single-stage.

## Results: what the two-stream trick actually buys

Now the proof, the measured side. The comparison that matters is coherence and controllability, and the honest summary from the literature is this.

AudioLM (Borsos and colleagues, 2022) demonstrated that the semantic-then-acoustic hierarchy produces speech and piano continuations with *long-term coherence* that flat acoustic models lacked: syntactically and semantically plausible speech continuations without any text supervision, and piano that stays in a consistent musical structure over several seconds. The key ablation in the paper is the one to internalize: *removing the semantic tokens and modeling acoustic tokens directly collapses long-term coherence*, exactly the babble failure. The semantic stream is load-bearing for coherence.

For evaluation, the relevant metrics differ by task. For the speech case you measure intelligibility with **Word Error Rate (WER)** by running an ASR system (Whisper or similar) over the generated audio and comparing to the intended transcript, and you measure speaker similarity with a speaker-verification embedding cosine. For voice cloning, VALL-E reported strong zero-shot speaker similarity and competitive WER from a *three-second* prompt, the headline result being that in-context cloning from three seconds rivaled systems that needed far more target-speaker data. For music, you measure **Fréchet Audio Distance (FAD)**, which compares the distribution of generated audio embeddings to real audio embeddings in a pretrained audio classifier's feature space (lower is better), and **CLAP score** for text-audio alignment. We cover all of these honestly, including FAD's well-known sensitivity to the embedding and sample size, in [the audio quality metrics post](/blog/machine-learning/audio-generation/audio-quality-metrics).

Here is a consolidated comparison of the landmark systems and how their token choices map to their behavior. Numbers are drawn from the respective papers and should be read as the *reported* figures under each paper's conditions; reproductions vary.

| System | Semantic source | Acoustic codec | Stages | Strength | Headline metric |
|---|---|---|---|---|---|
| AudioLM (2022) | w2v-BERT k-means | SoundStream | 3 (sem→coarse→fine) | Long-term coherence, no text | Coherent speech/piano continuation |
| MusicLM (2023) | w2v-BERT + MuLan | SoundStream | 3 + text bridge | Text-to-music coherence | FAD and CLAP on MusicCaps |
| VALL-E (2023) | phonemes (from text) | EnCodec | AR + NAR | 3-second zero-shot clone | Strong speaker-sim, competitive WER |
| MusicGen (2023) | none (T5 text) | EnCodec | 1 (delay pattern) | Simplicity, melody control | FAD ~ competitive with MusicLM, simpler |

The pattern across the table is the unifying message: every system is the same two-stream idea with the content stream sourced differently (learned semantic tokens, phonemes, or strong text conditioning) and the acoustic stream always being RVQ codec tokens. The number of *modeled* streams ranges from one (MusicGen, text does the planning) to three (AudioLM, explicit semantic stream), and that count is the main axis of the design space.

A second table, the trade-off the builder actually acts on:

| Choice | Long-range coherence | Inference simplicity | Best when |
|---|---|---|---|
| Flat acoustic LM | Poor (babbles) | Simplest | Almost never for >1-2 sec |
| Single-stage + strong text cond. (MusicGen) | Good (text plans) | Simple, 1 model | You have a strong text/phoneme prompt |
| Hierarchical semantic→acoustic (AudioLM) | Best (explicit plan) | Complex, 2 tokenizers, 3 models | Weak/no external content signal; unconditional |
| AR + NAR over codec (VALL-E) | Good (phonemes plan) | Moderate, 2 models | Zero-shot TTS / voice cloning |

#### Worked example: when the semantic stream is worth its cost

Suppose you are building two products. Product one is a *text-to-music* service: users type a prompt and get 30 seconds of music. Product two is an *unconditional ambient-sound continuation* tool: it takes a few seconds of an environment and extends it for a minute with no text. For product one, the text prompt is a strong content signal, so a single-stage MusicGen-style model is the right call: one codec, one model, simpler serving, and the text does the long-range planning. Adding a semantic stream here would roughly double your tokenizer and modeling complexity for marginal coherence gain, because the bottleneck is not content planning, it is rendering. For product two, there is *no* text to plan with, so a flat acoustic model would drift within seconds; the semantic stream is exactly what supplies the missing long-range structure, and the three-stage hierarchy earns its complexity. The decision rule generalizes: *the semantic stream is worth its cost in inverse proportion to the strength of your external conditioning signal.*

## Case studies: real systems and real numbers

**AudioLM, speech continuation without text (Borsos et al., 2022).** AudioLM trained a w2v-BERT model, k-means clustered its features to 1024 semantic tokens, and used SoundStream acoustic tokens. Its standout result was generating syntactically and semantically coherent *speech continuations* from a short prompt with no transcript at any point in the pipeline, something flat acoustic models could not do. Human raters could not reliably distinguish AudioLM's continuations from real speech in the ablation conditions the paper reported, and removing the semantic stage visibly degraded long-term structure. The lesson the field took: semantic tokens are the mechanism for textless long-range coherence.

**VALL-E, three-second zero-shot cloning (Wang et al., 2023).** VALL-E was trained on roughly 60,000 hours of speech (the LibriLight corpus), far more than typical TTS training sets, which is itself a lesson: the in-context cloning ability emerged at scale, much as in-context learning emerged in large text LMs. From a three-second enrolled clip it matched the target speaker's timbre and even the acoustic environment of the prompt, and it preserved emotion and acoustic conditions because those live in the acoustic tokens the prompt provided. Its reported zero-shot speaker similarity and WER beat the strong prior baseline (YourTTS) in the paper's evaluation. The caveat, and an honest one: quality degrades when the three-second prompt is noisy or atypical, because the only speaker information the model has is in that short, possibly-corrupted acoustic prompt.

**MusicGen, single-stage with melody control (Copet et al., 2023).** MusicGen showed a single-stage codec LM with the delay-pattern interleaving could match or approach the quality of the multi-stage MusicLM while being dramatically simpler, and it added *melody conditioning* by feeding a chromagram of a reference, letting users hum a tune and get it orchestrated. Released in 300M, 1.5B, and 3.3B sizes, it became the open music-generation workhorse. Its FAD and human-preference numbers on the standard benchmarks were competitive with the closed MusicLM despite dropping the semantic stream, which is the strongest single piece of evidence that *strong text conditioning can substitute for an explicit semantic stream*. We go deep on MusicGen in [the music generation post](/blog/machine-learning/audio-generation/music-generation-musiclm-and-musicgen).

**The codec choice underneath all of them.** None of this works without good acoustic tokens, which is why the codec posts come first in this track. The acoustic stream's quality ceiling *is* the codec's reconstruction quality: the generative model can only sound as good as the codec can decode, since it is generating codec tokens. If your EnCodec is dropping high frequencies at a low bitrate, no amount of clever hierarchy fixes the dull result, because the information was never in the token vocabulary to begin with. The semantic-versus-acoustic split, residual vector quantization, and the modern codecs are one connected story: [neural audio codecs](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound) build the tokenizer, [RVQ](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq) makes it efficient, and this post puts a second, content-bearing token type alongside it.

### Stress-testing the idea

Let me push on the design the way you should before betting a system on it.

*What if the semantic tokens leak speaker information?* They do, a little, and the amount depends on the cluster count $K$ and the extraction layer. If $K$ is too large or you extract from too late a layer, semantic tokens start to carry timbre, which contaminates the clean content-versus-identity split and can cause the speaker prompt to be ignored or fought. The mitigation is to keep $K$ modest and pick the layer by the phoneme-purity-minus-speaker-info probe. This is a real tuning burden that single-stage models sidestep entirely.

*What happens at a very low acoustic bitrate?* If you cut the codec to two codebooks to save sequence length, the fine acoustic detail vanishes and the output sounds muffled and metallic regardless of how good the semantic plan is, because the acoustic *vocabulary* can no longer represent the detail. The hierarchy does not rescue a starved codec; content coherence and acoustic fidelity are separately bottlenecked.

*What if the three-second speaker prompt is three seconds of noise?* Then the only identity signal the acoustic model has is corrupted, and VALL-E-style cloning will faithfully reproduce the noise *and* a distorted timbre, because it cannot tell the speaker apart from the environment in such a short, noisy clip. The factorization that makes three-second cloning possible is also its fragility: all of identity rides on the prompt, so a bad prompt poisons the output.

*What about the two-tokenizer training cost?* Maintaining a self-supervised model, a k-means codebook, *and* a codec, keeping them in sync across data versions, is genuine operational overhead. Every time you change the audio preprocessing you risk invalidating the k-means centroids. This unglamorous friction is a large part of why single-stage models won for text-conditioned use cases: one fewer tokenizer to babysit, and one fewer thing to silently drift out of alignment when someone bumps the resampler.

*What if the stages disagree?* This is the subtle failure I have actually shipped and had to debug. Because the stages are trained separately, the coarse acoustic model assumes the semantic tokens it receives at inference come from the *same distribution* it saw during training, namely real extracted semantic tokens. But at generation time those semantic tokens are *sampled* from the stage-one model, and if stage one occasionally produces an unusual or low-probability semantic token sequence, the coarse model is now conditioned on something slightly off-distribution, and it can respond with an artifact, a glitch, a momentary wrong speaker. This **exposure bias between stages** is a real and under-discussed cost of the hierarchy: each stage is only as robust as the *worst* sequence the previous stage hands it, and errors compound down the chain. The standard mitigations are to train later stages on *sampled* (not just ground-truth) outputs of earlier stages, or to use a single unified codec so there is no inter-stage distribution gap at all. It is one more reason the field drifted toward unified codecs, and a concrete instance of the general lesson that multi-stage generative pipelines pay an integration tax that single-stage pipelines avoid.

*What if you need to edit the output?* The factorization is a gift here, not a cost. Because content lives in the semantic stream and identity lives in the acoustic stream, you can *swap* one while holding the other: keep the semantic tokens and re-render with a different speaker prompt to change the voice without changing the words, or keep the acoustic prompt and feed new semantic tokens to make the same voice say something new. This editability is a real advantage of the explicit two-stream form over a single entangled stream, and it is why voice-conversion and content-preserving editing systems lean on exactly this split. A single-stage model that entangles everything into one token vocabulary makes this kind of surgical edit much harder, which is the flip side of its simplicity.

## The frontier: folding semantics into the codec itself

The two-tokenizer cost I keep flagging, a self-supervised model plus a codec, plus a k-means codebook to babysit, was annoying enough that the 2024 frontier asked an obvious question: can the codec produce *both* kinds of tokens itself, so that the first codebook carries semantics and the rest carry acoustic detail? The answer is yes, and it is one of the most important recent moves in audio tokenization.

The cleanest example is **Mimi**, the codec inside Moshi (Défossez and colleagues, Kyutai, 2024). Mimi is an RVQ codec like EnCodec, but it is trained with an extra objective: its *first* codebook is distilled to match the semantic tokens of a self-supervised model (a WavLM-style teacher), while the *remaining* codebooks are trained for reconstruction as usual. The result is a single tokenizer whose codebook 1 is a *semantic* token and whose codebooks 2 through $N$ are *acoustic* tokens. You get the content-versus-detail factorization for free, in one model, at one frame rate, with no separate k-means step. This is sometimes called **semantic distillation** into the codec, and it is the dominant design in the newest audio language models because it removes the operational friction that made the two-tokenizer hierarchy painful.

The conceptual payoff is large. With a Mimi-style codec, the two-stream idea is no longer an architectural decision about wiring three transformers together; it is *baked into the token vocabulary*. A single autoregressive model over the codec tokens, generating codebook 1 (semantic) first and then the acoustic codebooks, *is* the semantic-then-acoustic hierarchy, but as one model over one token stream with the delay pattern handling the codebooks. This is exactly how Moshi's full-duplex spoken-dialogue model works, and it is why the two-stream factorization, far from being obsoleted by single-stage models, is *more* central than ever: it just moved from the model architecture into the tokenizer.

So the arc of the field is worth stating cleanly. AudioLM (2022) separated semantics and acoustics into *two tokenizers and three models*. MusicGen (2023) showed that strong text conditioning lets you *drop* the semantic stream for one model and one codec. And Mimi (2024) put the semantic token *back*, but *inside* the codec, so you get one tokenizer, one model, and the factorization for free. Each step kept the core insight, content tokens plus detail tokens, and changed only *where* the split lives. That is the sign of a durable idea: it survives every architectural fashion by relocating rather than disappearing.

```python
# A unified semantic+acoustic codec exposes both roles from ONE encode call.
# (Mimi ships with the `moshi` package; the shape is the teaching point here.)
from moshi.models import loaders

mimi = loaders.get_mimi("kyutai/mimi", device="cuda")
mimi.set_num_codebooks(8)

with torch.no_grad():
    codes = mimi.encode(wav24.unsqueeze(0))   # (1, num_codebooks, frames)

semantic_stream = codes[:, 0]    # codebook 0 was distilled toward SSL semantics
acoustic_stream = codes[:, 1:]   # codebooks 1..N carry the acoustic residual
# One tokenizer, both token roles -- no separate HuBERT + k-means pipeline.
```

The `set_num_codebooks` knob is the same coarse-versus-fine bitrate lever from RVQ, now spanning a semantic codebook and acoustic ones in a single stack. This is the form most new systems are converging on, and it is where the capstone's pipeline recommendations will land.

## Connecting to understanding-and-generation unification

There is a forward-looking reason semantic tokens matter beyond generation, and it is worth flagging because it points at where the field is going. A semantic token is, by construction, a discrete code for *content* that is shared between *understanding* and *generation*. The same w2v-BERT/HuBERT tokens that an audio language model generates can be the input to an audio *understanding* model: speech recognition, audio captioning, spoken-language understanding. Because semantic tokens align with linguistic content, they form a natural interface between a generative audio model and a text-based language model.

This is exactly the bridge that recent speech-text models exploit: represent speech as semantic tokens, treat them as just more tokens in a shared vocabulary with text, and a single transformer can both understand spoken input and generate spoken output. Acoustic tokens then handle the final rendering to a waveform with the right voice. The two-stream factorization, content tokens plus detail tokens, turns out to be the natural seam along which to join audio with language models, which is why it underpins the full-duplex spoken-dialogue systems we will reach later in the series. The autoregressive-over-tokens framing, and how it descends from WaveNet through AudioLM, is the subject of [autoregressive audio models](/blog/machine-learning/audio-generation/autoregressive-audio-models-wavenet-to-audiolm), the next post you should read after this one.

## When to reach for the two-stream hierarchy (and when not to)

Decisive guidance, because the whole point of understanding this is to make a good architectural call.

**Reach for an explicit semantic stream (the full hierarchy) when** your task has weak or no external content conditioning and demands long-range coherence: unconditional or audio-prompted continuation, textless generation, or any setting where the structure you need is linguistic or melodic and is not handed to you as text. The semantic stream is the only thing that will keep you coherent over seconds, and the two-tokenizer cost is justified.

**Reach for a single-stage codec LM with strong conditioning (MusicGen-style) when** you have a powerful content signal already: a text prompt, phonemes, or a melody. The text or phonemes do the long-range planning, the delay pattern keeps inference efficient, and you save an entire tokenizer and two inference stages. This is the right default for *most* text-to-music and is increasingly competitive for TTS.

**Reach for the AR-plus-NAR split (VALL-E-style) when** you are doing zero-shot TTS or voice cloning specifically: phonemes give you content for free, the AR stage handles the coarse acoustic and duration, and the NAR stage parallelizes the residual codebooks for speed. The three-second-prompt cloning ability is the payoff.

**Do not** build a flat acoustic language model for anything longer than a second or two and expect coherence; it will babble, and you will waste weeks rediscovering AudioLM's motivation, as I did. **Do not** add a semantic stream to a system that already has a strong text prompt unless you have measured a coherence problem that the text conditioning genuinely fails to fix; you are buying complexity you may not need. **Do not** trust a three-second voice clone from a noisy prompt; the factorization makes the output only as clean as the prompt's identity signal. And **do not** expect the hierarchy to compensate for a weak codec: acoustic fidelity is capped by the codec's reconstruction, full stop.

The capstone, [building an audio generation stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack), turns these rules into a concrete decision tree for assembling a real pipeline, and the parallel between this token hierarchy and the coarse-to-fine, next-scale token hierarchies in image generation is drawn out in [autoregressive image models](/blog/machine-learning/image-generation/autoregressive-image-models), which is worth reading side by side with this post: the same idea, plan-the-structure-then-fill-the-detail, recurs across modalities.

## Key takeaways

- **A flat language model over codec (acoustic) tokens babbles**: it nails local timbre and drifts on long-range content, because the acoustic dependencies are loud and local while the linguistic ones are quiet and global, and gradient descent chases the loud signal.
- **The fix is two token types.** *Semantic* tokens (from a self-supervised model like HuBERT/w2v-BERT, discretized with k-means) capture phonetic, linguistic, and melodic *content* while discarding speaker and timbre. *Acoustic* tokens (RVQ codec tokens) capture speaker identity, timbre, and recording detail.
- **Self-supervised masked prediction is why semantic tokens are content-rich and speaker-poor**: predicting a masked unit from context rewards modeling how sounds follow each other (content) and lets speaker identity, useless for that prediction, atrophy.
- **The hierarchy is semantic → coarse acoustic → fine acoustic.** A semantic LM plans content on a short, content-pure sequence; coarse and fine acoustic LMs render it conditioned on the plan. This factorizes $p(S,A) = p(S)\,p(A \mid S)$, moving the hard long-range learning onto a ~17× shorter sequence.
- **The speaker prompt enters at the acoustic stage**, which is why three seconds of an unseen voice is enough to clone it: content comes from semantics/text, so the prompt only has to supply identity, and acoustic tokens encode identity densely.
- **VALL-E is the TTS specialization**: phonemes replace learned semantic tokens, an AR model does the coarse stage and the three-second clone, a NAR model parallelizes the fine codebooks. **MusicLM is the music specialization**: w2v-BERT semantics plus a MuLan text-to-semantic bridge.
- **MusicGen simplified it back to one stage** by dropping the semantic stream, leaning on strong T5 text conditioning, and using a codebook delay pattern to predict eight codebooks efficiently. Strong external conditioning can substitute for an explicit semantic stream.
- **The decision rule**: the semantic stream is worth its two-tokenizer, multi-stage cost in inverse proportion to the strength of your external conditioning signal. Strong text/phonemes → single-stage; weak/no signal → hierarchy.
- **Acoustic fidelity is capped by the codec.** The generative model can only sound as good as the codec decodes, because it is generating codec tokens; the hierarchy fixes coherence, not fidelity.
- **Semantic tokens are the seam between understanding and generation**: a shared content vocabulary that lets one transformer both recognize and synthesize speech, which is why the two-stream idea underpins modern spoken-dialogue models.

## Further reading

- Borsos et al., *AudioLM: a Language Modeling Approach to Audio Generation* (2022) — the paper that introduced the semantic-then-acoustic hierarchy and the two-token paradigm.
- Hsu et al., *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units* (2021) — the masked-prediction self-supervised model that yields good semantic tokens.
- Chung et al., *w2v-BERT: Combining Contrastive Learning and Masked Language Modeling for Self-Supervised Speech Pre-Training* (2021) — the semantic model AudioLM and MusicLM actually used.
- Wang et al., *Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers* (VALL-E, 2023) — TTS as codec-token language modeling with three-second cloning.
- Agostinelli et al., *MusicLM: Generating Music From Text* (2023) — the AudioLM recipe for music with the MuLan text-to-semantic bridge.
- Copet et al., *Simple and Controllable Music Generation* (MusicGen, 2023) — the single-stage codec LM with codebook delay-pattern interleaving.
- Défossez et al., *High Fidelity Neural Audio Compression* (EnCodec, 2022) — the RVQ codec that produces the acoustic tokens used throughout.
- Within this series: [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard), [residual vector quantization](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq), [EnCodec, DAC, and the modern codec](/blog/machine-learning/audio-generation/encodec-dac-and-the-modern-codec), and the capstone [building an audio generation stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack).
