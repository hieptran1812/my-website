---
title: "Evaluating Audio Generation Honestly: Beyond a Single Number"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "A practitioner's harness for judging audio models you actually ship — why FAD's embedding decides the ranking, how to run a MOS study that survives review, and how to report it all honestly."
tags:
  [
    "audio-generation",
    "audio-synthesis",
    "evaluation",
    "fad",
    "mos",
    "clap-score",
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
image: "/imgs/blogs/evaluating-audio-generation-honestly-1.png"
---

Here is a story I have lived through more than once. You train a new text-to-speech model on Friday, it posts a Fréchet Audio Distance of 1.4 against your old checkpoint's 1.9, the leaderboard updates, the channel gets a green checkmark, and you ship it Monday. By Wednesday the support queue has a theme: "the new voice sounds flatter," "it swallows the ends of sentences," "I switched back to the old one." You run a quick listening test in the hallway and everyone agrees the old model sounds better. You did not make a math error. The metric did exactly what it promised — it measured the distance between two clouds of embeddings — and the thing it measured was not the thing your users feel. You optimized a proxy until the proxy and the product came apart, which is the single most expensive failure mode in this entire field.

This post is about not doing that. It is the deeper, opinionated companion to the metrics primer in [audio quality metrics](/blog/machine-learning/audio-generation/audio-quality-metrics): where that post catalogs the instruments, this one is about how to actually judge a model you are about to put in front of people. I am going to make four arguments and defend each with math, code, and numbers. First, that FAD is not one metric but a family of metrics indexed by the embedding you choose, and that the embedding — not the audio — frequently decides the ranking. Second, that almost every reported FAD is computed on too few clips to be stable, and small-sample FAD is biased upward in a way that quietly flatters small models. Third, that human evaluation is the gold standard precisely because it is the only thing that measures what you care about, but only if you run it like an experiment instead of a vibe check. And fourth, that the way out of the eval crisis is not a better single number — there is no better single number — but a small, honestly-reported *basket* of complementary measurements plus a willingness to look at the gap between offline metrics and live preference.

By the end you will be able to: compute FAD with two different embeddings and watch the ranking flip; reason about how many clips and how many raters you actually need for a result you can defend; build a reproducible eval harness in PyTorch that scores realism, relevance, and intelligibility together; design a MOS/CMOS study with anchors and rater screening so the confidence interval means something; and report the whole thing in a way that does not lie to your future self. We are walking a path the image-generation people walked first — the FID-and-eval-crisis story in [evaluating image generation honestly](/blog/machine-learning/image-generation/evaluating-image-generation-honestly) is the direct ancestor of everything here, and where the parallels are exact I will point at them rather than re-derive them.

![A matrix comparing FAD, CLAP-score, WER, and human MOS across what each captures, how gameable each is, and whether it correlates with human judgement](/imgs/blogs/evaluating-audio-generation-honestly-1.png)

This connects to the spine of the whole series — the audio stack of waveform to codec or mel latent to a generative model to a vocoder back to waveform, pulled in four directions by fidelity, controllability, speed, and length. Evaluation is how you find out which of those four you actually bought with your last architecture change, and a dishonest metric will tell you that you bought fidelity when you spent it on the leaderboard. If you have not read [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard), skim it for why our ears are so unforgiving; this post assumes you know what a waveform, an STFT, a mel-spectrogram, and an embedding are.

## 1. FAD is a family of metrics, not one metric

Let me state the central, under-appreciated fact plainly: **Fréchet Audio Distance is parameterized by an embedding network, and changing that network changes what the metric is.** People write "FAD = 2.3" as if FAD were a property of the audio. It is not. It is a property of the audio *as seen through a particular frozen feature extractor*, and different extractors see different things.

Start with the definition so the dependence is impossible to miss. FAD takes a set of real reference clips and a set of generated clips. It pushes every clip through an embedding network $f$, producing a cloud of vectors for each set. It fits a single multivariate Gaussian to each cloud — mean $\mu_r$ and covariance $\Sigma_r$ for the real set, $\mu_g$ and $\Sigma_g$ for the generated set — and reports the Fréchet (2-Wasserstein) distance between those two Gaussians:

$$
\text{FAD} = \lVert \mu_r - \mu_g \rVert_2^2 + \operatorname{Tr}\!\left( \Sigma_r + \Sigma_g - 2\left(\Sigma_r \Sigma_g\right)^{1/2} \right).
$$

Lower is closer; zero means the two Gaussians coincide. This is the audio analogue of FID, with VGGish historically playing the role that Inception-v3 plays for images. Everything downstream of "push every clip through $f$" inherits whatever geometry $f$ imposes. And here is the thing that breaks the naive reading: $\mu$, $\Sigma$, and therefore the entire distance live in the embedding space of $f$. Two embedding networks define two completely different spaces. There is no canonical "audio feature space" the way there is no canonical "image feature space" — there is only whatever a particular network learned to care about.

It is worth knowing *why* that particular closed form is the distance, because the derivation tells you what the metric assumes. The Fréchet distance — the 2-Wasserstein distance under squared-Euclidean cost — between two distributions $P$ and $Q$ is the optimal-transport cost of morphing one cloud of points into the other:

$$
W_2^2(P, Q) = \inf_{\gamma \in \Gamma(P,Q)} \mathbb{E}_{(x,y) \sim \gamma}\,\lVert x - y \rVert_2^2,
$$

where $\Gamma(P,Q)$ is the set of all couplings with marginals $P$ and $Q$. In general this infimum is intractable. But for two *Gaussians* it has the closed form above — the mean term plus the covariance trace term — and that is the whole reason FAD fits Gaussians: it is a deliberate simplification that turns an intractable transport problem into a formula you can evaluate from two means and two covariance matrices. The cost of that convenience is the **Gaussian assumption itself**: FAD pretends both embedding clouds are unimodal ellipsoids. When a generated distribution is multimodal — say a music model that produces clips in two distinct sub-genres — the single fitted Gaussian smears across both modes, and the metric can be blind to a mode it should have penalized. This is a real and under-discussed limitation: FAD measures the distance between the *Gaussian approximations* of your distributions, not between the distributions themselves, and the approximation error is largest exactly when the output is diverse and multimodal — which is often what you want.

So FAD has two layers of assumption stacked on top of each other, and either can dominate. The outer layer is the embedding $f$ — the geometry the distance is measured in. The inner layer is the Gaussian fit — the shape the distance assumes each cloud has. A clean ranking requires *both* to be benign for your data, and neither is guaranteed. This is why I keep insisting the number is fragile: it is not one approximation but two, composed.

![A branching graph of the FAD pipeline showing audio fed through either a VGGish or a CLAP embedding before fitting Gaussians and taking a Fréchet distance, with the ranking able to flip](/imgs/blogs/evaluating-audio-generation-honestly-2.png)

### Why the embedding dominates the result

Here is the proof, and it is short. Suppose you have two candidate models, A and B, and a reference set. Under embedding $f$, the generated clouds get summarized as Gaussians $(\mu_A^f, \Sigma_A^f)$ and $(\mu_B^f, \Sigma_B^f)$, and you rank A above B if and only if

$$
d_f(\text{A}, \text{ref}) < d_f(\text{B}, \text{ref}),
$$

where $d_f$ is the Fréchet distance computed in $f$'s space. Now change to a second embedding $g$. There is no reason whatsoever that the sign of $d_f(\text{A}) - d_f(\text{B})$ has to match the sign of $d_g(\text{A}) - d_g(\text{B})$, because $f$ and $g$ apply different nonlinear maps to the raw audio. If $f$ is a classifier trained on AudioSet event labels (VGGish), it organizes its space around *what kind of sound* this is — dog bark versus piano versus speech — and it is largely deaf to fine perceptual quality within a class, because that distinction was never useful for its training objective. If $g$ is a contrastive text-audio model (CLAP) or a codec encoder (EnCodec), it organizes its space around features that had to survive reconstruction or had to align with language, which turn out to track perceived quality far more closely.

So the two metrics are measuring genuinely different quantities. The reason this is not just a theoretical worry is empirical: a body of recent work has shown that **VGGish-based FAD correlates poorly with human judgement**, and that swapping in a CLAP embedding, an EnCodec embedding, or a self-supervised speech model produces a FAD that tracks listener preference much better. Gui, Gamper, Braun, and Emmanouilidou's study of FAD as a music-quality metric is the standard citation here: they found the embedding choice changes the ranking, and that more perceptually-aligned embeddings give more trustworthy distances. The practical upshot is brutal and simple — **a paper that reports "VGGish-FAD 2.3 vs 2.5" has not actually shown you which model is better.** It has shown you which model VGGish prefers, and VGGish has opinions you did not ask for.

#### Worked example: the ranking that flips

Let me make this concrete with the kind of numbers you will actually see. You evaluate three text-to-speech checkpoints — call them A (crisp and intelligible), B (smooth but slurs consonants), and C (clearly worse). You compute FAD twice, once with VGGish and once with a CLAP embedding, on the same 2,000 clips with the same reference set.

![A before-and-after figure showing three TTS models ranked one way under VGGish-FAD and re-ordered under CLAP-FAD so the crisp model rises to first](/imgs/blogs/evaluating-audio-generation-honestly-3.png)

Under VGGish-FAD you read off B at 1.92, A at 2.41, C at 3.10, and you ship B — the smooth one. Your users immediately complain that B swallows words. Under CLAP-FAD the order is A at 0.28, B at 0.41, C at 0.55, and A — the crisp one — wins. (The absolute scales differ between embeddings, which is itself a reason never to compare FAD across embeddings; only the *ranking* is comparable, and even that flipped.) The audio did not change between the two computations. Only the judge did. If you had reported a single VGGish-FAD number and called it a day, you would have shipped the worse model and had a leaderboard that endorsed the decision. The fix is not "trust CLAP-FAD blindly" — it is to pick a perceptually-validated embedding *on purpose*, state which one you used, and never report a bare "FAD" without it.

The bias here is structural, not adversarial. Nobody set out to game VGGish. The model that wins under VGGish wins because VGGish's feature geometry happens to reward smoothness-within-class over crispness, and a model that produces smooth, slightly-mushy speech sits closer to the AudioSet "speech" centroid than a model that produces sharp, fully-articulated speech with more spectral energy in the consonants. The metric is not broken; it is measuring the wrong thing, confidently.

## 2. Sample size: most FAD numbers are computed on too few clips

The second way FAD lies is statistical, and it is independent of the embedding. **FAD is biased upward at small sample sizes, and the bias is large enough to change conclusions.** The reason is the covariance term. Estimating a mean from $n$ samples is easy and converges fast. Estimating a $D \times D$ covariance matrix from $n$ samples is hard — you are estimating $O(D^2)$ parameters, and for VGGish $D = 128$, for CLAP $D = 512$. With only a few hundred clips your $\hat{\Sigma}$ is a noisy, structurally distorted estimate of the true covariance, and that noise does not average out of the Fréchet distance — it inflates it.

You can see why from the trace term. The matrix square root $(\Sigma_r \Sigma_g)^{1/2}$ is sensitive to the small eigenvalues of $\hat{\Sigma}$, which are exactly the directions estimated worst from few samples. Sampling noise pushes the two covariances apart even when the underlying distributions are identical, so the expected FAD between two independent samples of the *same* distribution is strictly positive and decreases toward zero only as $n$ grows. In practice this means a model evaluated on 500 clips can post a meaningfully better FAD than the same model evaluated on 5,000 clips, purely because the smaller sample's covariance was noisier in a direction that happened to help.

![A matrix showing FAD bias and run-to-run spread shrinking as the number of clips grows from 200 to 5000, with small samples flagged as untrustworthy](/imgs/blogs/evaluating-audio-generation-honestly-5.png)

The numbers in that figure are the shape you should expect, not exact constants — the magnitude depends on the embedding dimension and the diversity of your data — but the trend is robust and worth internalizing: at $n = 200$ the bias relative to the large-sample limit is large and the run-to-run standard deviation across resampled subsets is enormous; by $n = 2{,}000$ both have collapsed to something usable; past $n = 5{,}000$ you are mostly buying decimal places. This is why the FAD methodology papers recommend on the order of a few thousand clips and a *fixed* evaluation set, and why I treat any FAD computed on a few hundred clips as a directional hint, not a result.

There is a second, subtler trap inside the first: **the bias is not the same for every model.** A model with low output diversity — one that mode-collapses onto a few safe outputs — produces a *tighter* generated cloud, which interacts with small-sample covariance estimation differently from a high-diversity model. So small-$n$ FAD does not just add noise uniformly; it can systematically flatter the lower-diversity model, which is often the worse model. This is the audio version of a well-known FID pathology, and it is one more reason the single number is dangerous: the noise has a direction.

#### Worked example: the same model, two sample sizes

You have one model. You compute its FAD against the reference set three times: once on a random 300-clip subset, once on 1,500 clips, once on 4,000. You get 2.7, 2.1, and 1.95. A colleague who only ran the 300-clip version on a *different* model got 2.3 and concluded their model beats yours. Both of you were measuring noise. The honest move is to fix the evaluation set — same clips, same count, same seed for any subsampling — across every model you compare, report the $n$ you used, and ideally report a bootstrap confidence interval by resampling the clip set. When I see a FAD in a paper with no $n$ and no interval, I mentally widen the number by a few tenths and refuse to break ties below that margin.

## 3. CLAP-score and its blind spots

FAD asks "does this sound real?" It does not ask "does this match the prompt?" For that, the field reaches for **CLAP-score**: you embed the generated audio and the text prompt into the shared CLAP space and take the cosine similarity. It is cheap, reference-free, and it correlates reasonably with whether the audio is *about* the right thing. If you prompt "a dog barking over rain" and the model produces a piano sonata, CLAP-score will be low; if it produces barking and rain, CLAP-score will be high. For text-to-audio and text-to-music relevance, it is the standard relevance number, and you should compute it.

But CLAP-score has blind spots you have to know about or it will mislead you in the other direction. It measures *semantic alignment*, not *quality*. A model can produce a low-fidelity, artifact-ridden clip of a barking dog and score a perfectly good CLAP-score, because the embedding recognizes the *category* "dog" even through the artifacts. CLAP is also coarse about *compositional* and *count* details — "three dogs, then silence, then one dog" is largely flattened to "dogs" in the embedding, the same way CLIP-score is coarse about object counts and spatial relations in images. And CLAP-score is gameable: because the metric rewards similarity to the prompt's text embedding, a model that learns to produce a slightly generic but unambiguously on-category sound can score higher than a model that produces a faithful but specific rendering, the same keyword-stuffing failure that haunts text-image retrieval. So CLAP-score belongs in the basket as the *relevance* axis, paired with a *quality* axis (FAD with a good embedding) and a *human* axis — never on its own. Reporting CLAP-score without a fidelity metric is how you ship audio that is recognizably-correct and audibly-bad.

There is also a quieter trap: CLAP-score has no absolute scale you can reason about across prompts. A cosine similarity of 0.4 on one prompt and 0.4 on another do not mean the same thing, because different prompts sit at different baseline similarities to *any* on-topic audio — abstract prompts ("a sense of unease") have lower achievable CLAP-scores than concrete ones ("a piano"), simply because the text embedding is more diffuse. So you cannot threshold CLAP-score at a fixed value and call anything below it a failure, and you cannot average it naively across a prompt set with mixed concreteness. The defensible use is *relative*: compare two models on the *same* prompt set, where the per-prompt baselines cancel, and look at the delta. As an absolute bar it is close to meaningless; as a paired comparison on a fixed prompt set it is informative.

#### Worked example: the high-CLAP, low-fidelity trap

You evaluate a text-to-audio model on 500 prompts. It posts a CLAP-score of 0.43, comfortably above the previous model's 0.39, and the relevance leaderboard says ship it. Then you listen: the new model produces audio that is unmistakably on-topic — every "rainstorm" prompt makes rain, every "engine" prompt makes an engine — but the clips are gritty, with a persistent high-frequency buzz the old model did not have. The CLAP-score went *up* because the new model is more reliably on-category; the FAD, which you also computed, went up too (worse): CLAP-FAD 0.55 versus the old model's 0.38. The two numbers disagree because they measure different things — CLAP-score says "more on-topic," FAD says "less realistic" — and the disagreement is the signal. Reading CLAP-score alone, you ship buzzy-but-relevant audio. Reading the two together, you see a relevance gain bought with a fidelity loss, and you go to a human study to decide whether the trade is worth it. The lesson is the same one this whole post keeps making: the value is in the *disagreements between metrics*, which a single number erases.

## 4. The speech and TTS battery

Speech has its own evaluation traditions, and they are more mature than music's because the field is older and the use case is sharper: you usually know the words that were supposed to come out. That gives you measurements music does not have.

**Intelligibility via WER.** Run a strong automatic speech recognition model — Whisper is the default — over your synthesized audio, take its transcript, and compute the word error rate against the text you asked the model to say. WER is an edit-distance metric:

$$
\text{WER} = \frac{S + D + I}{N},
$$

where $S$, $D$, $I$ are the substitutions, deletions, and insertions in the minimum-edit alignment between the ASR transcript and the reference text, and $N$ is the number of words in the reference. It is the Levenshtein distance at the word level, normalized by reference length. A WER of 0 means the ASR heard exactly the words you intended; a WER of 0.10 means roughly one word in ten was wrong. This is the single most useful TTS number, because it is hard to game in the worst direction: a model that slurs consonants — the exact failure VGGish-FAD missed in section 1 — gets caught, because the ASR mishears the slurred words and WER climbs. The caveat is that **WER is an ASR-mediated proxy, not ground truth.** A very strong ASR is robust enough that WER tracks human intelligibility well, but a weak or domain-mismatched ASR will both miss real errors and hallucinate fake ones. Use a large, recent ASR, use the same one across all systems, and normalize text the same way (lowercase, strip punctuation, expand numbers) before scoring — inconsistent normalization is the most common way to get a WER that is silently wrong by a few points.

**Speaker similarity.** When you are cloning a voice, intelligibility is not enough — the output has to *sound like the target speaker*. The objective version is **SIM**: push the reference speaker's audio and the generated audio through a speaker-verification embedding (a model like WavLM-based or ECAPA-TDNN x-vectors) and take the cosine similarity between the speaker vectors. A high SIM means the verification model thinks it is the same person. The subjective version is **SMOS** (similarity mean opinion score), where human raters score how similar the cloned voice is to a reference. These are the metrics that matter for the zero-shot cloning frontier in [zero-shot voice cloning and the TTS frontier](/blog/machine-learning/audio-generation/zero-shot-voice-cloning-and-the-tts-frontier), and SIM has the same caveat as WER — it is mediated by a frozen verification model, so it measures "does this fool the verifier," which is close to but not identical to "does this sound like the person to a human."

**Prosody and naturalness.** This is the part with no good objective metric. Whether speech has natural rhythm, appropriate emphasis, and believable emotion is exactly the kind of perceptual judgement that no frozen network captures well, which is why naturalness lives almost entirely in human evaluation (the MOS of section 6). There are proxies — you can compare pitch (F0) contours and duration distributions against real speech, and large gaps there flag obviously-wrong prosody — but a model can match the statistics of pitch and duration and still sound robotic, so these proxies catch gross failures and miss subtle ones. For prosody, the honest answer is: measure what you can objectively, then put it in front of humans.

**Reference-based metrics, when you have a clean target.** Everything above is reference-free or reference-text-based, because in generation you usually do not have a "correct" waveform to compare against — there is no single right way to say a sentence. But there is one important case where you *do* have a paired reference: codec and vocoder evaluation, voice conversion against a held-out recording, and speech enhancement, where the ground-truth waveform exists and you are measuring how faithfully you reconstructed it. There the classic signal-quality metrics apply, and they are worth keeping in the toolkit even though they are useless for free-form generation:

- **PESQ** (perceptual evaluation of speech quality) is a psychoacoustic model that compares a degraded signal to a clean reference and outputs a score on roughly a 1–4.5 scale that was tuned to predict listening-test MOS for telephony-band speech. It is a real perceptual model — it does loudness warping and masking — which makes it far better than a raw signal-to-noise number for "how bad does this degradation sound," but it is band-limited and built for speech, so do not point it at music.
- **STOI** (short-time objective intelligibility) predicts *intelligibility* rather than quality, correlating short-time spectral envelopes of the reference and the degraded signal; it returns a number in $[0,1]$ that tracks how many words a listener would get right. It is the reference-based cousin of WER and is the right metric for enhancement and low-bitrate codec intelligibility.
- **SI-SDR** (scale-invariant signal-to-distortion ratio) is the cleanest of the three to reason about: it decomposes the estimate into the part aligned with the target and the residual, and reports their ratio in dB, invariant to overall scale. $\text{SI-SDR} = 10 \log_{10} \frac{\lVert s_\text{target} \rVert^2}{\lVert e_\text{residual} \rVert^2}$. Higher is better; it is the standard for source separation and reconstruction.

The reason these are *not* in the generation harness of section 8 is the same reason they exist: they require a paired reference, and generation does not have one. But the moment your evaluation has a clean target — you are scoring a vocoder against the original waveform, or a codec at a given bitrate against the uncompressed signal, the bitrate–quality trade explored in [EnCodec, DAC, and the modern codec](/blog/machine-learning/audio-generation/encodec-dac-and-the-modern-codec) — these are exactly the right numbers, and PESQ/STOI with MUSHRA is the standard codec-eval triplet.

#### Worked example: the speech eval that would have saved Monday

Go back to the opening story. The model with FAD 1.4 that you shipped — suppose you had also run the speech battery on a fixed 200-utterance test set, same text for every checkpoint. The old model scored WER 3.1% and SMOS 4.1; the new one scored WER 6.8% and SMOS 3.6. The WER nearly doubling is the slurred-consonant failure showing up as a hard number, and the SMOS drop says raters heard the voice as less faithful. Those two numbers, computed in twenty minutes on a GPU plus a one-afternoon listening test, would have overruled the FAD before you shipped. The FAD was not wrong about distributional realism; it was answering a question that did not include "are the words intact."

## 5. Music-specific evaluation: the hard-to-measure stuff

Music is the hardest modality to evaluate, because the thing you care about — does this sound *good*, is it *musical*, does it have *structure* — is the least reducible to a frozen network. You can and should compute FAD with a good embedding for distributional realism and CLAP-score for prompt relevance, and those catch a lot: a model that produces noise, or music in the wrong genre, or clips that drift out of key into atonal mush, will move those numbers. But the things that separate a good music model from a great one are mostly invisible to them.

**Musicality and structure.** Whether a 90-second clip has a coherent arrangement — an intro, a development, a sense that the second half relates to the first — is a long-range structural property, and embedding-based metrics computed on short windows or on a single pooled vector are largely blind to it. A model can produce locally-plausible bars that, stitched together, go nowhere, and FAD on short clips will not notice because every short clip looks fine. This is why the music section of [Suno, Udio, and the commercial music frontier](/blog/machine-learning/audio-generation/suno-udio-and-the-commercial-music-frontier) leans so heavily on human listening: the commercial gap between open music models and the frontier is mostly a *structure and production-polish* gap, and it does not show up cleanly in any single number.

**Genre and instrument fidelity** are partly objectively checkable — you can run a genre or instrument classifier and see whether "lo-fi hip hop" actually produces lo-fi hip hop — but these are coarse. The practical stance for music: use FAD-with-a-good-embedding and CLAP as a *screen* (they cheaply reject the disasters), then spend your human budget on the things they cannot see — structure, musicality, production quality, and whether the vocals (if any) are intelligible and in tune. For music more than anywhere else, the offline metrics tell you which checkpoints are *not* broken, and humans tell you which one is *good*.

A few music-specific objective proxies are worth running even though none is decisive. **Tonal stability** — does the clip stay in a consistent key, or drift — can be estimated from a chroma feature over the duration, and a large key-variance flags the "drifts out of tune" failure that a short-window FAD misses entirely. **Rhythmic consistency** can be probed by tracking a beat and checking whether the tempo stays stable, catching the model that loses the beat halfway through a long generation. And for sung material, you can run the same WER-via-ASR trick on the vocals to check lyric intelligibility, and a pitch tracker against the intended melody to check whether the singing is in tune. None of these replaces a human ear — a clip can pass all of them and still sound lifeless — but they are cheap automated guards against the *specific* gross failures (drift, lost beat, garbled lyrics) that plague long-form music generation, and they belong in the probe layer of the harness for any music model.

## 6. Human evaluation, done like an experiment

Human evaluation is the gold standard, and it deserves the phrase, because it is the only thing on this list that measures the actual target — what a person thinks of the audio — instead of a proxy for it. But "we asked some people and they liked it" is not human evaluation; it is an anecdote with a sample size of three. To get a number you can defend in review, you have to run it like an experiment, with a protocol, anchors, enough raters, screening, and a confidence interval.

![A branching graph of a MOS study where stimuli and hidden anchors feed raters, who are screened on the anchors before the scores are aggregated with a confidence interval](/imgs/blogs/evaluating-audio-generation-honestly-6.png)

### The protocols: ACR, DCR, MUSHRA, CMOS

There are a few standard listening-test designs, and picking the right one matters more than people think.

**ACR — absolute category rating — produces MOS.** Each rater hears one clip at a time and rates it on a five-point scale (1 = bad, 5 = excellent). Average over raters and clips and you get the **mean opinion score (MOS)**. ACR is simple and gives you an absolute number, but it is noisy and *contextual*: a rater's sense of "5" drifts depending on what they have heard recently, and absolute MOS is hard to compare across studies run on different days with different rater pools. Use it when you need a standalone quality number.

**DCR — degradation category rating** — plays the reference first, then the test clip, and asks how degraded the test sounds relative to the reference. It is more sensitive than ACR for *codec and vocoder* evaluation where you have a clean reference and are measuring how much you lost.

**MUSHRA** plays multiple systems side by side against a hidden reference and a hidden low-anchor (often a low-pass-filtered version), and asks raters to score all of them on a 0–100 scale in one screen. Because everything is heard together, MUSHRA is far more discriminating than ACR — small quality differences that vanish in absolute rating become visible in direct comparison — and the hidden anchors let you screen out raters who score the reference low or the low-anchor high. MUSHRA is the right tool when you are comparing several systems at similar quality.

**CMOS — comparative MOS** — is the one I reach for most for shipping decisions. Raters hear two clips (your candidate and the baseline) in random order and score the *preference* on a scale like −3 (B much better) to +3 (A much better). CMOS directly measures the thing you actually want to know — "is the new model better than the one we have" — and because it is a *paired* comparison it cancels a huge amount of per-clip and per-rater variance. A CMOS of +0.3 with a confidence interval that excludes zero is a real, shippable win; a CMOS of +0.05 with an interval straddling zero means you changed nothing a human can hear, no matter what FAD says.

### The statistics: how many raters do you actually need?

Here is the part teams skip, and it is the part that decides whether your study means anything. MOS is a mean of noisy per-rating samples, so it has a standard error, and the standard error sets how small a difference you can resolve.

Model a single rating as the true quality $\mu$ plus noise: $x_{ij} = \mu + \varepsilon_{ij}$, with $\varepsilon$ capturing both how the clip varies and how raters disagree, with variance $\sigma^2$. If you collect $R$ ratings (raters times clips, roughly), the standard error of the mean is

$$
\text{SE} = \frac{\sigma}{\sqrt{R}},
$$

and the 95% confidence interval is about $\pm 1.96 \cdot \text{SE}$. Invert it: to resolve a true difference of size $\Delta$ between two systems with a half-width $\pm 1.96\,\sigma/\sqrt{R}$ smaller than $\Delta$, you need on the order of

$$
R \gtrsim \left( \frac{1.96\,\sigma}{\Delta} \right)^2
$$

ratings. Put real numbers in. ACR ratings on a 1–5 scale typically have a per-rating standard deviation around $\sigma \approx 0.9$. If you want to reliably detect a MOS gap of $\Delta = 0.2$ — a difference small enough that you cannot trust your own ears about it — you need roughly $R \gtrsim (1.96 \times 0.9 / 0.2)^2 \approx 78$ ratings *per system*, and that is the optimistic case where the variance is well-behaved. In practice, with crowd raters and harder material, you want more: a common rule of thumb is at least 15–30 distinct raters each scoring a couple dozen clips, giving several hundred ratings per system, which buys you confidence intervals tight enough to resolve the ~0.2–0.3 MOS differences that actually distinguish modern models. **Paired designs (CMOS, MUSHRA) need far fewer raters for the same resolving power** because pairing cancels the between-clip and between-rater variance, which is the main reason I prefer them — you get a defensible answer from 15–20 screened raters instead of needing a hundred.

### Why crowd MOS is noisy, and how to de-noise it

Crowdsourced listening tests (think a crowd-work platform) are cheap and fast and *noisy*, for reasons that are entirely fixable if you know them. Raters use cheap earbuds or laptop speakers. They drift — their internal "5" wanders over a long session. Some are inattentive or adversarial, clicking through for the payout. The session has no calibration, so one rater's 3 is another's 4. Left alone, this noise can swamp a 0.2 MOS difference entirely. The fixes are standard and you should apply all of them:

- **Hidden anchors.** Slip in a known-excellent clip (a real recording) and a known-bad clip (heavily degraded) in every batch. A rater who scores the excellent anchor low or the bad anchor high is not listening carefully — you drop them. Anchors are also how MUSHRA defines the top and bottom of its scale.
- **Enough raters, with overlap.** More raters and multiple ratings per clip average out individual drift and disagreement. Per-clip averaging is where most of your noise reduction comes from.
- **Screening and outlier removal.** Beyond anchors, remove raters whose scores are uncorrelated with the consensus, or who show no within-session variance (everyone gets a 4). Report how many you removed and why — silently dropping raters until the result looks good is p-hacking with extra steps.
- **Report the confidence interval, always.** A MOS without an interval is a number pretending to be a result. Compute it (analytically from the SE above, or by bootstrapping over raters/clips) and report it. If the intervals of two systems overlap heavily, say so and do not declare a winner.
- **Calibrate and fix the setup.** Tell raters to use headphones, normalize loudness across clips (a louder clip is rated higher, all else equal — a classic confound), randomize presentation order, and use the same instructions every time.

Do all of that and crowd MOS becomes a real instrument. Skip it and you have an expensive random number generator.

#### Worked example: sizing a MOS study before you pay for it

You are about to commission a crowd listening test to decide whether your new TTS checkpoint beats the baseline, and you want to know what to buy before you spend the budget. You expect the true quality gap, if any, to be small — call it $\Delta = 0.2$ MOS, a difference you genuinely cannot trust your own ears about. Per-rating standard deviation on a 1–5 ACR scale for this kind of material runs around $\sigma \approx 0.9$. Plug into the resolving-power formula: you need on the order of $R \gtrsim (1.96 \times 0.9 / 0.2)^2 \approx 78$ ratings *per system* just for the confidence half-width to drop below $\Delta$ — and that is the floor, not a comfortable margin. If you ran an absolute ACR study, that means roughly 30 raters each scoring ~6 clips per system, and you would still be borderline. Now switch the design to **CMOS**, a paired comparison. Pairing cancels the between-clip variance (both systems are heard on the same content) and a chunk of the between-rater variance (each rater is their own control), which in practice shrinks the effective $\sigma$ by something like a factor of two — so the same resolving power comes from roughly a quarter of the ratings, on the order of 15–20 screened raters. Same decision, a quarter of the cost, and a tighter interval. That is why I default to CMOS for ship calls: the statistics are simply more efficient, and the question it answers — "is the new one better than what we have" — is the question I actually have. Before you launch *any* listening test, do this arithmetic with your own $\Delta$ and $\sigma$; commissioning a study too small to resolve the difference you care about is the most common way to waste a human-eval budget and walk away with an inconclusive result you then over-interpret.

## 7. The eval crisis: Goodhart, contamination, and the offline–online gap

Step back from the individual metrics and look at the system, because the failures compound into something the field half-jokingly calls the eval crisis, and it is real.

**Goodhart's law.** "When a measure becomes a target, it ceases to be a good measure." The moment FAD becomes the leaderboard, people — sometimes deliberately, usually by gradient descent and architecture search — start optimizing FAD, and a model tuned to minimize FAD is not the same as a model tuned to sound good. Concretely: because small-sample FAD flatters low-diversity models (section 2), a model that quietly reduces its output diversity can lower its FAD while getting *worse* for users; because VGGish-FAD rewards smoothness-within-class (section 1), a model that smooths its outputs can lower its FAD while losing crispness. Neither of these is cheating in the legal sense. They are the metric doing exactly what an optimizer asks of it, which is the whole danger. The defense is to never let a single number be the target — rotate which metrics you watch, hold some out, and keep a human study in the loop as the un-gameable backstop.

![A before-and-after figure where a checkpoint with the best offline FAD loses a live A/B preference test against the previous version, exposing the gap between proxy and product](/imgs/blogs/evaluating-audio-generation-honestly-8.png)

**Benchmark contamination.** As public eval sets age, they leak into training data — scraped, mirrored, redistributed — until a model has effectively seen the test set during pretraining and its scores no longer measure generalization. This is the same contamination story that haunts LLM benchmarks, and it haunts audio too: a music model trained on a web-scale crawl may well have ingested the very clips your benchmark draws its reference distribution from, and a TTS model may have seen the exact sentences in your WER test set. The defenses are unglamorous: maintain a *private, held-out* eval set you never publish; refresh it periodically; and treat any benchmark older than a couple of years with suspicion that it has leaked.

The audio version of contamination has a nasty twist the text version mostly lacks: the *reference set* of FAD can itself be contaminated. FAD measures distance to a reference distribution of "real" clips. If your model was trained on those exact reference clips — entirely plausible when both your training crawl and the benchmark draw from the same public music or speech corpora — then your model has been optimized, however indirectly, to produce outputs near the reference centroid, and its FAD is flattered for a reason that has nothing to do with quality. You can get a suspiciously good FAD by *memorizing* slices of the reference distribution, which is the opposite of what you want a generative model to do. So for FAD specifically, the held-out discipline has to cover both ends: a reference set the model never trained on, *and* a generation-prompt set the model never tuned for. A FAD computed against a reference set that overlaps training data is measuring memorization, not generation, and it will look great right up until a user asks for something the model did not memorize.

**The offline–online gap.** This is the deepest one, and section 8's worked example makes it concrete: the checkpoint with the best offline FAD can lose a live A/B preference test against the previous version. Offline metrics are computed on a fixed distribution with a frozen judge; real users bring prompts you did not anticipate, listen on devices you did not test, and care about things (latency, consistency, the absence of rare catastrophic failures) that your averaged offline number washes out. A model can win on average FAD and still produce one jarring artifact every fifty clips that drives users away — and average metrics are exactly the wrong tool for catching rare catastrophic failures. The only cure is to close the loop: ship to a fraction of traffic, measure real preference (A/B tests, win rates, retention, thumbs), and trust *that* over any offline leaderboard when they disagree.

## 8. A recommended harness, and how to report it

Enough diagnosis. Here is the prescription: a small, honestly-reported basket that covers the axes no single metric can cover alone.

![A stacked layered figure of the recommended evaluation basket, from a frozen corpus up through FAD, CLAP, WER, targeted probes, and a human CMOS study](/imgs/blogs/evaluating-audio-generation-honestly-4.png)

The basket, bottom to top, is: a **frozen evaluation corpus** (fixed clips, fixed count of ~2,000, fixed seed for any subsampling, ideally private and refreshed); **FAD with a perceptually-validated embedding** (CLAP or EnCodec, *stated explicitly*, not VGGish) for distributional realism; **CLAP-score** for text-audio relevance; **WER via a strong ASR** for speech intelligibility (and SIM/SMOS when cloning); **targeted probes** for the things averages hide — long-form generation, out-of-distribution prompts, the rare-failure tail; and on top, a **small, well-designed CMOS study** with anchors, screened raters, and a reported confidence interval. None of these is sufficient. Together they are hard to fool all at once, because gaming one tends to cost you on another — smoothing to win VGGish-style FAD costs you WER; reducing diversity to win FAD costs you on the probes; keyword-stuffing for CLAP-score costs you on the human study.

### The probes: what averaged metrics structurally cannot see

The "targeted probes" layer deserves its own treatment, because it catches a class of failure that *every averaged metric is mathematically blind to*. An average is a sum over a fixed test distribution divided by its size. By construction it tells you nothing about the *tail* — the one-clip-in-fifty catastrophic artifact — because one bad clip in fifty barely moves a mean. And it tells you nothing about anything *outside the test distribution* — the prompts your fixed corpus never thought to include. These are exactly the failures that drive the offline–online gap, so probes are how you close it before users do.

A probe is a small, *adversarially-chosen* test set aimed at a specific weakness, scored separately and never averaged into the headline number. The ones I always run:

- **Long-form coherence.** Generate at the longest length you support and check whether the model holds together — does the song keep its key and tempo for the full duration, or drift; does the speech keep the speaker's voice consistent across a long passage, or wander. A short-clip FAD will not see this because every short window looks fine while the whole goes nowhere, the failure described in [latent diffusion for music, Stable Audio](/blog/machine-learning/audio-generation/latent-diffusion-for-music-stable-audio).
- **Out-of-distribution prompts.** Feed prompts your training data under-covers — rare instruments, accented or code-switched text, unusual emotional registers — and listen for collapse. Models look great on the prompts they were tuned for and fall apart just off-distribution, and your fixed corpus, drawn from the same distribution as training, will never reveal it.
- **The rare-failure tail.** Generate a few hundred clips and *listen to the worst ten*, sorted by some cheap proxy (lowest CLAP-score, highest WER) or just by ear. The question is not "what is the average quality" but "how bad is the worst thing this model will hand a user, and how often." A model with a great average and a 2%-rate of jarring glitches is worse to ship than a slightly-lower-average model that never glitches.
- **Stress conditions for cloning.** When the use case is voice cloning, probe with a deliberately *bad* speaker prompt: three seconds of noisy, reverberant audio instead of a clean studio reference. SIM and SMOS measured only on clean prompts tell you nothing about the case your users will actually hit.

Probes are reported as a small table of pass/fail or worst-case numbers next to the headline metrics, never folded in. The discipline is: averages for "is it good on the common case," probes for "what is the worst it will do." You need both, and only the second one catches the failures that actually generate support tickets.

### Build it: FAD with two embeddings, and watch it flip

First, the centerpiece — compute FAD twice and show yourself the embedding dependence on your own data. The `frechet-audio-distance` package supports multiple embeddings; here is the core of an honest FAD harness.

```python
import torch
from frechet_audio_distance import FrechetAudioDistance

# Two judges for the SAME audio. We will compare the rankings, not the scales.
fad_vggish = FrechetAudioDistance(
    model_name="vggish",      # the historical default — known to track humans poorly
    sample_rate=16000,
    use_pca=False,
    use_activation=False,
    verbose=False,
)
fad_clap = FrechetAudioDistance(
    model_name="clap",        # perceptually better aligned (also try "encodec")
    sample_rate=48000,
    submodel_name="630k-audioset",
    verbose=False,
)

def rank_models(reference_dir, candidate_dirs):
    """Return each candidate's FAD under both embeddings."""
    results = {}
    for name, gen_dir in candidate_dirs.items():
        v = fad_vggish.score(reference_dir, gen_dir)   # lower = closer
        c = fad_clap.score(reference_dir, gen_dir)
        results[name] = {"vggish_fad": round(v, 3), "clap_fad": round(c, 3)}
    return results

candidates = {"A_crisp": "gen/A", "B_slurred": "gen/B", "C_worse": "gen/C"}
scores = rank_models("ref/", candidates)

# Rank under each embedding and check whether the order agrees.
order_vggish = sorted(scores, key=lambda k: scores[k]["vggish_fad"])
order_clap   = sorted(scores, key=lambda k: scores[k]["clap_fad"])
print("VGGish ranking:", order_vggish)   # e.g. ['B_slurred', 'A_crisp', 'C_worse']
print("CLAP   ranking:", order_clap)     # e.g. ['A_crisp', 'B_slurred', 'C_worse']
if order_vggish != order_clap:
    print("WARNING: the embedding changed the ranking. Do not report a bare FAD.")
```

The point of this script is not to pick a winner automatically; it is to *force the disagreement into the open*. When the two rankings differ, you know the FAD number alone cannot decide, and you escalate to WER and the human study. When they agree, you have a stronger signal. Either way you report *which embedding* you used and never compare FAD across embeddings — the scales are not commensurable, only the within-embedding ranking is.

### Add a confidence interval to FAD by bootstrapping

A single FAD is a point estimate on a noisy covariance. Bootstrap over the clip set to get an interval, so you stop breaking ties inside the noise.

```python
import numpy as np

def fad_bootstrap(embed_real, embed_gen, n_boot=200, rng=None):
    """Bootstrap CI for FAD from precomputed embeddings (N x D arrays)."""
    rng = rng or np.random.default_rng(0)
    n_r, n_g = len(embed_real), len(embed_gen)
    samples = []
    for _ in range(n_boot):
        r = embed_real[rng.integers(0, n_r, n_r)]   # resample with replacement
        g = embed_gen[rng.integers(0, n_g, n_g)]
        samples.append(frechet_from_embeddings(r, g))  # Gaussian-fit + Frechet
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(np.mean(samples)), float(lo), float(hi)

mean, lo, hi = fad_bootstrap(real_emb, gen_emb)
print(f"FAD = {mean:.3f}  (95% CI [{lo:.3f}, {hi:.3f}])")
# If two models' CIs overlap, you have NOT shown one is better.
```

If you cannot afford a full bootstrap, at minimum compute FAD on two disjoint halves of your generated set and look at the spread — a large gap between the halves is a loud warning that your $n$ is too small and the number is unstable.

### CLAP-score for relevance

```python
import laion_clap
import torch.nn.functional as F

clap = laion_clap.CLAP_Module(enable_fusion=False)
clap.load_ckpt()  # downloads a pretrained CLAP checkpoint

def clap_score(audio_paths, prompts):
    """Mean cosine similarity between generated audio and its text prompt."""
    a = clap.get_audio_embedding_from_filelist(x=audio_paths, use_tensor=True)
    t = clap.get_text_embedding(prompts, use_tensor=True)
    a, t = F.normalize(a, dim=-1), F.normalize(t, dim=-1)
    return (a * t).sum(dim=-1).mean().item()  # higher = better aligned

print("CLAP-score:", clap_score(gen_paths, gen_prompts))
```

Remember the blind spot from section 3: a high CLAP-score with a bad FAD means "on-topic but low-fidelity." Read the two numbers together, never CLAP-score alone.

### WER via Whisper for speech intelligibility

```python
import whisper
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces

asr = whisper.load_model("large-v3")   # use a STRONG, consistent ASR

# Identical normalization for hypothesis and reference, or WER is meaningless.
normalize = Compose([ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces()])

def speech_wer(audio_paths, reference_texts):
    hyps, refs = [], []
    for path, ref in zip(audio_paths, reference_texts):
        text = asr.transcribe(path, language="en")["text"]
        hyps.append(normalize(text))
        refs.append(normalize(ref))
    return wer(refs, hyps)   # the (S + D + I) / N edit-distance metric

print(f"WER = {speech_wer(tts_paths, tts_texts) * 100:.2f}%")
```

This is the number that would have caught the slurred-consonant model in section 1. Run it on a fixed sentence set, the same ASR, the same normalization, for every checkpoint you compare.

### A MOS/CMOS study scaffold

Finally, the human study — the part no code can replace, but that code can *structure* so it is reproducible. Here is the scaffold: build the stimulus list with hidden anchors, collect ratings, screen raters on the anchors, and aggregate with a confidence interval.

```python
import numpy as np
import pandas as pd

def build_cmos_session(candidate_clips, baseline_clips, hi_anchor, lo_anchor,
                       rng=np.random.default_rng(0)):
    """One rater's session: paired A/B trials in random order, anchors hidden."""
    trials = []
    for cand, base in zip(candidate_clips, baseline_clips):
        flip = rng.random() < 0.5             # randomize which side is the candidate
        left, right = (cand, base) if flip else (base, cand)
        trials.append({"left": left, "right": right, "candidate_is": "left" if flip else "right"})
    # hidden anchors: a real-vs-degraded pair the rater MUST get right
    trials.append({"left": hi_anchor, "right": lo_anchor, "candidate_is": "anchor"})
    rng.shuffle(trials)
    return trials

def aggregate_cmos(ratings_df):
    """ratings_df: rater_id, trial_is_anchor, anchor_correct, cmos_score in [-3, 3]."""
    # 1. screen: drop raters who fail the hidden anchor
    bad = ratings_df[ratings_df.trial_is_anchor & ~ratings_df.anchor_correct].rater_id.unique()
    clean = ratings_df[~ratings_df.rater_id.isin(bad) & ~ratings_df.trial_is_anchor]
    print(f"Dropped {len(bad)} raters who failed the anchor")
    # 2. per-rater mean, then mean of means (cancels per-rater drift)
    per_rater = clean.groupby("rater_id").cmos_score.mean()
    mean = per_rater.mean()
    # 3. bootstrap CI over raters
    boot = [per_rater.sample(len(per_rater), replace=True).mean() for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return mean, lo, hi

cmos, lo, hi = aggregate_cmos(collected_ratings)
print(f"CMOS = {cmos:+.2f}  (95% CI [{lo:+.2f}, {hi:+.2f}])")
verdict = "SHIP" if lo > 0 else ("DO NOT SHIP" if hi < 0 else "INCONCLUSIVE — tied")
print(verdict)
```

Notice the structure: anchors are *built into* the stimulus list, screening is automatic and *reported* (we print how many raters were dropped), and the verdict is gated on the confidence interval excluding zero. A CMOS of +0.4 with a CI of [+0.1, +0.7] ships; a CMOS of +0.1 with a CI of [−0.2, +0.4] is a tie and you say so. This is the difference between a listening test that survives review and a vibe check that does not.

### Reporting it honestly

The harness is only half the job; the other half is reporting it in a way that does not let your future self — or a reader — draw a conclusion the data does not support. The rules I hold myself to: **state the embedding** on every FAD and never compare FAD across embeddings. **State the $n$** (clip count, rater count) on every number, and attach a confidence interval or at least a run-to-run spread; a metric without an interval is a point estimate masquerading as a fact. **Report what you dropped** — how many raters failed the anchors, how many clips were excluded and why — because silent filtering is how a tie becomes a "win." **Do not declare a winner when the intervals overlap;** write "indistinguishable on this metric" and move to a more discriminating one. **Keep the probe results next to the headline numbers,** not folded into them, so the tail stays visible. And when you write the summary, lead with the *basket*, not the best single number: "CLAP-FAD 0.31, CLAP-score 0.42, WER 3.1%, CMOS +0.3 [+0.1, +0.5] over baseline, no regressions on the long-form and OOD probes" is a sentence a reviewer can trust. "FAD 1.4, ship it" is the sentence that started this post. The whole discipline reduces to one habit: report enough that someone who distrusts you could re-derive your conclusion — and refuse to state a conclusion the numbers, with their intervals, do not actually support.

## 9. A results table you can act on

Here is the basket as a decision table — each metric, what it captures, how it is gamed, and whether it tracks humans — the thing to pin above your desk.

| Metric | Captures | Cost | How it is gamed | Tracks human judgement? |
| --- | --- | --- | --- | --- |
| FAD (VGGish) | Distributional realism, by AudioSet classes | Cheap | Smoothness-within-class; low diversity at small $n$ | Poorly — known weak correlation |
| FAD (CLAP / EnCodec) | Distributional realism, perceptual | Cheap | Still low-diversity bias at small $n$ | Better — use this embedding |
| CLAP-score | Text-audio relevance | Cheap | Generic on-category outputs; keyword match | Coarse — relevance only, not quality |
| WER (strong ASR) | Speech intelligibility | Cheap | Hard to game downward; ASR can mishear | Strong for clarity |
| SIM / SMOS | Speaker similarity (cloning) | Cheap / human | Fool the verifier vs fool a human | Good, verifier-mediated |
| MUSHRA | Fine quality differences, side-by-side | Human | Loudness confound; bad raters | Strong with anchors + screening |
| CMOS | Preference vs a baseline | Human | Loudness/order confounds | Gold standard for ship decisions |
| Live A/B | Real-world preference | Slow | None — it is the target | The ground truth |

And here is the harness as a comparison of *what each axis buys you*, with representative numbers from the kind of evaluation runs you will do (treat the FAD scales as embedding-specific and not comparable across rows):

| Axis | Tool | Sample / raters | Typical resolving power | Failure it catches |
| --- | --- | --- | --- | --- |
| Realism | FAD (CLAP embed) | ~2,000 clips | ~0.05 with bootstrap CI | Off-distribution, artifacts, noise |
| Relevance | CLAP-score | ~2,000 clips | coarse | Wrong content / off-prompt |
| Intelligibility | WER (Whisper large) | ~200 utterances | ~0.5% | Slurring, dropped words |
| Similarity | SIM (verification) | ~200 pairs | ~0.02 cosine | Wrong-speaker clone |
| Quality | CMOS | 15–25 screened raters | ~0.2 MOS, CI-gated | The thing all proxies miss |

The discipline is to compute the cheap axes (FAD, CLAP, WER, SIM) on every checkpoint as a fast screen, then spend the expensive human axis (CMOS/MUSHRA) only on the two or three checkpoints that survive the screen and on your final ship decision. That keeps the per-checkpoint cost low while keeping a human as the un-gameable backstop.

#### Worked example: the basket overruling the leaderboard

Put it all together on one real-shaped decision. You have two TTS checkpoints, the incumbent v1 and the candidate v2, and you run the full basket on a fixed 200-utterance test set and a frozen 2,000-clip corpus. The headline VGGish-FAD says v2 wins: 1.40 versus 1.92. If that were the whole story you would ship v2 — exactly the Monday-morning mistake. But the basket keeps going. CLAP-FAD (the better-correlated embedding) reverses it: v1 at 0.31, v2 at 0.46. WER via Whisper large-v3 also reverses it: v1 at 3.1%, v2 at 6.8% — v2 nearly doubled the error rate, the slurred-consonant signature. SIM is a wash, 0.71 versus 0.70. So you escalate the two to a CMOS study, 18 screened raters, anchors built in: CMOS comes back −0.34 in favor of v1, with a 95% CI of [−0.52, −0.16] that cleanly excludes zero. The probes confirm it: on the long-form passage, v2 drifts in speaker timbre after thirty seconds; on the OOD accented prompts, v2's WER blows out to 14%. The verdict is unambiguous — **do not ship v2** — and the only metric that pointed the wrong way was the one everyone quotes. The basket cost a GPU afternoon and an 18-rater study, and it saved you the support queue. Report it as: "v1 retained; v2 regresses on CLAP-FAD, WER (+3.7 pts), and CMOS (−0.34 [−0.52, −0.16]), with long-form and OOD probe failures; VGGish-FAD favored v2 and was overruled."

This is what "honest" means operationally. Not one number, not a vibe, but a small set of complementary measurements where the *disagreements* carry the information, each reported with its sample size and interval, with a human study as the tiebreaker and probes guarding the tail. Any one of those numbers alone would have been wrong or insufficient. Together they made a decision you can defend in a review and to a user.

## 10. Case studies and real numbers from the literature

A few grounded examples of these ideas showing up in published work — accurate in shape; where I am not certain of an exact figure I say so.

**The FAD-embedding finding (Gui et al., 2024).** The study that crystallized section 1 evaluated FAD as a music-quality metric across several embedding networks and found that VGGish-FAD correlates relatively poorly with human perceptual judgement, while embeddings like CLAP and a trained music encoder correlate substantially better — and crucially that the *ranking* of systems can change with the embedding. This is the empirical backbone of "report which embedding."

**WER as the TTS intelligibility standard.** Modern zero-shot TTS papers — VALL-E (Wang et al., 2023) and its successors, and the F5-TTS and NaturalSpeech 3 line — report WER computed with a strong ASR as a primary intelligibility metric alongside speaker-similarity (SIM via a verification model) and human MOS/SMOS. The pattern is exactly the basket of section 8: an objective intelligibility number, an objective similarity number, and a human quality number, because no one of them is enough. The reported WERs for strong open systems sit in the low single digits of percent on clean read-speech test sets, which is the bar to beat; exact values vary by ASR and test set, so compare only within a fixed evaluation protocol.

**MUSHRA for codecs and vocoders.** Codec and vocoder papers — EnCoder/DAC-class neural codecs, HiFi-GAN-class vocoders covered in [GAN vocoders, HiFi-GAN, and fast synthesis](/blog/machine-learning/audio-generation/gan-vocoders-hifi-gan-and-fast-synthesis) — lean on MUSHRA and DCR rather than ACR, because they have a clean reference and are measuring *degradation*, and MUSHRA's side-by-side design with hidden anchors resolves the small quality gaps between, say, two bitrates that ACR would blur together. This is the right protocol choice for that problem.

**MusicGen and the FAD-plus-human pattern (Copet et al., 2023).** The MusicGen paper reports both FAD (computed with a specified embedding) and human ratings, and the design choice is instructive: the authors do not lean on FAD alone, they pair it with human evaluation of overall quality and of *adherence to the melody conditioning*, because the conditioning-faithfulness question — does the generated music actually follow the melody you gave it — is one that no distributional metric answers. This is the music basket in miniature: a cheap distributional number for realism, a relevance number for conditioning, and humans for the musicality that neither captures. The covered model-size-versus-quality trade in [music generation, MusicLM and MusicGen](/blog/machine-learning/audio-generation/music-generation-musiclm-and-musicgen) is exactly the kind of result you can only read correctly with that full basket — a bigger model with a marginally better FAD is not automatically the one humans prefer.

**The offline–online gap in production.** Teams shipping TTS and music at scale routinely report that offline FAD/MOS improvements do not always translate to live preference, and that the disagreements are informative — usually the offline metric missed a rare failure mode or a latency/consistency property that users feel. The general lesson, echoed across image and audio generation alike, is that the live A/B is the ground truth and the offline leaderboard is a fast, fallible proxy. A concrete pattern worth naming: a checkpoint that improves *average* quality while introducing a small rate of catastrophic glitches will win offline and lose online, because the average rewards the typical case and users punish the worst case. Averages and tails pull in opposite directions, and only a live test — or a deliberate tail probe — measures the tail. When the offline win is small and the model is going in front of real users, I trust the A/B and treat the offline number as the hypothesis the A/B is there to test.

## 11. When to reach for which metric (and when not to)

Decisive guidance, because the worst eval is the one that measures everything badly instead of the right things well.

![A decision tree routing speech, music, and relevance evaluations to WER plus SMOS, FAD with a good embedding, and CLAP-score plus MUSHRA respectively](/imgs/blogs/evaluating-audio-generation-honestly-7.png)

- **Do not report a bare "FAD."** Always state the embedding. If you must pick one, use CLAP or EnCodec, not VGGish. And never compare FAD across embeddings — the scales are not commensurable.
- **Do not trust a FAD computed on a few hundred clips.** Use ~2,000+, fix the set and seed, and bootstrap a CI. Below that, FAD is a directional hint, not a result, and its bias flatters low-diversity models.
- **Do not use CLAP-score as a quality metric.** It is a *relevance* metric. Pair it with a fidelity metric or it will endorse on-topic garbage.
- **For speech, lead with WER and SIM, not FAD.** Intelligibility and speaker identity are what users notice; a distributional metric that misses slurring is the wrong primary number for TTS.
- **For music, use FAD-with-a-good-embedding and CLAP as a screen, then spend your human budget on structure and musicality.** The things that separate good music models from great ones are mostly invisible to any frozen network.
- **Use ACR/MOS for a standalone number, MUSHRA for fine side-by-side quality, CMOS for ship decisions.** CMOS resolves the most with the fewest raters because pairing cancels variance.
- **Do not run a listening test without anchors, screening, and a confidence interval.** Without them it is an anecdote. With them it is the gold standard.
- **Do not let any single number be the optimization target.** That is Goodhart; rotate metrics, hold some out, keep a human in the loop, and refresh a private eval set to fight contamination.
- **When offline and online disagree, trust online.** The live A/B is the target; the offline metric is a proxy that can be — and eventually will be — gamed.

## 12. Key takeaways

- **FAD is a family indexed by its embedding.** The embedding, not the audio, often decides the ranking — VGGish-FAD correlates poorly with humans; CLAP/EnCodec-FAD is better. Always state the embedding; never compare across embeddings.
- **Most reported FAD is computed on too few clips.** The covariance term is biased upward and high-variance at small $n$, and the bias flatters low-diversity models. Use ~2,000+ clips, a fixed set and seed, and a bootstrap CI.
- **CLAP-score measures relevance, not quality.** Pair it with a fidelity metric; it will happily reward on-topic, low-fidelity audio and is gameable by generic on-category outputs.
- **For speech, WER (via a strong, consistent ASR) and SIM/SMOS are the load-bearing numbers.** WER is an edit-distance metric that catches exactly the slurring that distributional metrics miss; SIM catches wrong-speaker clones.
- **Music quality lives mostly in human listening.** Use FAD and CLAP as a cheap screen, then spend humans on structure, musicality, and production polish that no frozen network sees.
- **Human evaluation is the gold standard only if you run it like an experiment** — anchors, 15–30 screened raters (fewer for paired CMOS/MUSHRA because pairing cancels variance), and a reported confidence interval. CMOS gated on a CI excluding zero is a shippable win.
- **The way out of the eval crisis is a basket, not a better single number** — FAD-with-a-good-embedding + CLAP + WER + targeted probes + a small CMOS study, reported honestly with its $n$ and intervals.
- **Beware Goodhart, contamination, and the offline–online gap.** A model tuned to FAD is not a model tuned to sound good; old benchmarks leak into training; and the live A/B is the ground truth that overrules any offline leaderboard.

For where these numbers feed into a real shipping decision, see the capstone, [building an audio generation stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack), which folds this harness into a serving and safety pipeline, and [the 2026 audio model landscape](/blog/machine-learning/audio-generation/the-2026-audio-model-landscape), which compares the frontier models you will be running this harness against. The image-side parallel in [evaluating image generation honestly](/blog/machine-learning/image-generation/evaluating-image-generation-honestly) tells the same story for FID — read it for how exactly the two crises rhyme.

## Further reading

- Gui, Gamper, Braun, Emmanouilidou, *Adapting Frechet Audio Distance for Generative Music Evaluation* (2024) — the embedding-choice finding that anchors section 1.
- Kilgour, Zuluaga, Roblek, Sharifi, *Fréchet Audio Distance: A Reference-Free Metric for Evaluating Music Enhancement Algorithms* (2019) — the original FAD definition.
- Wang et al., *VALL-E: Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers* (2023) — WER + SIM + human MOS as the TTS battery.
- Wu et al., *Large-Scale Contrastive Language-Audio Pretraining (CLAP)* (2023) — the embedding behind CLAP-score and CLAP-FAD.
- Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)* (2022) — the ASR you compute WER with.
- ITU-T Recommendation P.800 (ACR/DCR MOS) and ITU-R BS.1534 (MUSHRA) — the standard listening-test protocols.
- 🤗 `transformers`, `frechet-audio-distance`, and `laion-clap` documentation — the practical toolchain for the harness in section 8.
- Within the series: [audio quality metrics](/blog/machine-learning/audio-generation/audio-quality-metrics) (the primer), [the mathematics of audio signals](/blog/machine-learning/audio-generation/the-mathematics-of-audio-signals), [zero-shot voice cloning and the TTS frontier](/blog/machine-learning/audio-generation/zero-shot-voice-cloning-and-the-tts-frontier), and [Suno, Udio, and the commercial music frontier](/blog/machine-learning/audio-generation/suno-udio-and-the-commercial-music-frontier).
