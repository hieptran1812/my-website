---
title: "Audio Quality Metrics: FAD, MOS, and What the Numbers Hide"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "A practitioner's guide to measuring generated audio — FAD, CLAP-score, WER, PESQ/STOI/SI-SDR, and human MOS — with runnable code and the failure mode of each number."
tags:
  [
    "audio-generation",
    "audio-synthesis",
    "evaluation",
    "fad",
    "mos",
    "text-to-speech",
    "music-generation",
    "generative-ai",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Audio Generation"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/audio-quality-metrics-1.png"
---

You trained two text-to-speech models over the weekend. Model A posts a Fréchet Audio Distance of 1.9; Model B posts 2.6. Lower is better, so you ship A, write it up, and move on. Then the first user emails: "the new voice swallows half its words." You play a few samples and they are right — A sounds *plausibly like speech* in a way that fools the metric, but it slurs consonants into mush. B, the one you discarded, was crisp and intelligible. You optimized a number that did not measure the thing you cared about, and the number rewarded you for it.

This is the single most expensive mistake in audio generation, and almost everyone makes it once. Audio is unusually good at hiding the gap between "scores well" and "sounds good," because the metrics we lean on are proxies built on frozen neural networks, reference signals you may not have, or human panels you cannot afford to run on every checkpoint. The same FAD that ranks two music models can flip its verdict if you swap the embedding network underneath it. The same listening test that is the gold standard is also noisy, slow, and biased by how you phrase the question.

This post is the metrics primer the rest of this series leans on. By the end you will be able to: compute FAD with a deliberate choice of embedding and know why that choice dominates the result; score text-audio alignment with CLAP; measure real speech intelligibility with an ASR-based word error rate; reach for PESQ, STOI, or SI-SDR when you have a reference signal; run a small but statistically honest MOS or CMOS study; and — most importantly — assemble these into a *basket* so no single number can quietly lie to you. We will be opinionated, because the field's habit of reporting one decimal place of FAD as if it settled the question is exactly how you end up shipping Model A.

![A matrix table comparing reference-free, reference-based, and human evaluation families across whether they need ground truth, what they capture, their cost, and how gameable they are](/imgs/blogs/audio-quality-metrics-1.png)

If you have not read [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard) and [the mathematics of audio signals](/blog/machine-learning/audio-generation/the-mathematics-of-audio-signals), skim them first — this post assumes you know what a waveform, an STFT, and a mel-spectrogram are. We are also walking a path the image people walked before us: the FID-and-eval-crisis story in [evaluating image generation honestly](/blog/machine-learning/image-generation/evaluating-image-generation-honestly) is the direct ancestor of everything here, and the parallels are exact enough that I will point at them rather than re-derive them.

## 1. The three families of audio metrics

Every audio metric you will meet falls into one of three families, and the families answer genuinely different questions. Confusing them is how teams end up arguing past each other in review meetings.

**Reference-free distributional metrics** ask: *does the set of clips my model produced look like a set of real clips, statistically?* They never compare a generated clip to a specific "correct" clip — there is no correct clip when you generate ten seconds of lo-fi from a prompt. Instead they compare two *distributions*: the cloud of real audio and the cloud of generated audio, summarized in some embedding space. FAD is the flagship. CLAP-score is a cousin that measures alignment to a text prompt rather than realism. These are cheap (you run them on a GPU over a couple thousand clips), they need no paired ground truth, and they are the easiest to game — which is the whole problem.

**Reference-based signal metrics** ask: *how close is this output to a specific target waveform?* This only makes sense when a target exists: speech enhancement (clean up this noisy recording, and I have the clean original to compare against), source separation (split this mix, and I have the true stems), or TTS where you are reading a sentence whose ground-truth recording you hold. PESQ and STOI model perceptual quality and intelligibility against a reference; SI-SDR measures how much of the target signal you recovered versus how much you smeared. The killer member of this family for generative speech is WER — transcribe the generated speech with an ASR model and edit-distance the transcript against the text you asked it to say. That is the realest intelligibility test there is.

**Human evaluation** asks the only question that ultimately matters: *do listeners actually prefer it?* MOS (mean opinion score), CMOS (comparative MOS), and MUSHRA are the protocols. Humans are the gold standard precisely because they are not a proxy — they *are* the target. But they are noisy (raters disagree, get tired, and anchor on the first sample they hear), expensive (you are paying people or running a panel), and slow (you cannot run a listening study on every training checkpoint). And the question framing leaks: "rate the quality" and "which do you prefer" pull different answers out of the same clips.

The figure above lays out the trade. The honest takeaway, which the rest of this post elaborates, is that *no single family is sufficient.* Reference-free metrics are gameable. Reference-based metrics need a reference you often do not have. Human eval is the truth but you cannot afford it at scale. The competent move is to spend cheap metrics liberally as a fast signal and reserve human eval for the decisions that matter — and to always read more than one number.

It helps to know *why audio is harder to evaluate than images*, because the families exist for audio-specific reasons. Three properties of sound make objective metrics shakier than their image counterparts. First, **the ear is a phase-aware time-frequency instrument with brutal temporal acuity** — we localize clicks to within tens of microseconds and we hear a 50 ms gap as a stutter, so artifacts that a pooled embedding averages away are perfectly audible to a listener. Second, **perception is strongly nonlinear and content-dependent**: a given amount of distortion is inaudible under a loud broadband sound (masking) and glaring in a quiet passage, so a single distortion number does not map cleanly to a single perceived-quality number. Third, **the goal is often not reconstruction at all** — when you generate ten seconds of music from a prompt, there is no "right answer" to compare against, which is precisely why the reference-free family had to be invented. Images share some of this, but the eye is far more forgiving of high-frequency detail than the ear is of a buzz, which is why audio leans harder on human eval and why the metric situation is, frankly, worse. We dug into the perceptual side in [the mathematics of audio signals](/blog/machine-learning/audio-generation/the-mathematics-of-audio-signals); the consequence for evaluation is that you should distrust any single objective number a little more than you would in vision.

There is one habit to break before going further: **false precision.** It is normal in audio papers to see "FAD 2.13" reported to two decimals as if that resolution were meaningful. It almost never is. The run-to-run variance of FAD from resampling alone is often a few tenths; the embedder and reference choice swing it by whole points. So a difference in the second decimal place is noise dressed as signal. When you read or write a FAD, mentally round it — "about 2," "about 4" — and reserve fine distinctions for metrics and protocols that can actually support them (a CMOS study with a tight CI can; a single-seed FAD cannot). The decimals are not a lie because someone is dishonest; they are a lie because the metric does not have that much resolution, and reporting them anyway trains everyone to over-read the number.

A note on direction before we go further, because it trips people up: for FAD, WER, PESQ-distance-style and SI-SDR-error framings, *lower is the goal* — they are distances or errors. For CLAP-score, MOS, CMOS, STOI, and SI-SDR-as-ratio, *higher is the goal*. When you read a table, the first thing to check is which way each arrow points, because half of all "our model beats theirs" claims are arrow-direction confusion.

## 2. Fréchet Audio Distance: FID, lifted to audio

FAD is the most reported number in audio generation, so it deserves the most scrutiny. It is FID — the Fréchet Inception Distance from the image world — with the Inception network swapped for an audio embedding network. The math is identical; the consequences are not, and the difference is entirely about that swap.

### The science: a distance between two Gaussians

Here is the construction, stripped to its bones. You have a set of real reference clips and a set of generated clips. You push every clip through a *frozen* embedding network and collect the embedding vectors. Each set is now a cloud of points in embedding space. You fit a multivariate Gaussian to each cloud — meaning you compute a mean vector and a covariance matrix and pretend the cloud is Gaussian. Then you measure how far apart the two Gaussians are, using the Fréchet distance (also called the 2-Wasserstein distance) between Gaussians, which has a closed form:

$$
\text{FAD} = \lVert \mu_r - \mu_g \rVert_2^2 + \operatorname{tr}\!\left(\Sigma_r + \Sigma_g - 2\left(\Sigma_r \Sigma_g\right)^{1/2}\right)
$$

where $(\mu_r, \Sigma_r)$ are the mean and covariance of the *real* embeddings and $(\mu_g, \Sigma_g)$ are those of the *generated* embeddings. The first term penalizes the two clouds for having different centers (different average timbre, loudness, content). The second term penalizes them for having different shapes and spreads (one model produces a narrower or differently-correlated range of sounds than the reference). The matrix square root $(\Sigma_r \Sigma_g)^{1/2}$ is the only awkward piece numerically; everything else is means and traces. This is the exact same formula as FID — derive it once in the image series and you never have to derive it again, which is why I will point you at [evaluating image generation honestly](/blog/machine-learning/image-generation/evaluating-image-generation-honestly) for the geometry and spend my words here on what audio breaks.

It is worth pausing on *why this particular distance*, because it is not arbitrary. The 2-Wasserstein distance is the minimum "cost" to transport one probability mass onto another, where cost is squared Euclidean distance. For two Gaussians that transport problem has the closed form above — no optimization needed at evaluation time, just linear algebra on the two covariance matrices. The reason we fit Gaussians at all (rather than comparing the raw point clouds with some nonparametric two-sample test) is tractability and stability: a Gaussian is summarized by a mean and a covariance, both of which are smooth functions of the data, so the metric does not jitter wildly when you resample. The cost of that convenience is a strong assumption — that the embedding cloud *is* roughly Gaussian, which it is not, exactly. Embedding distributions of audio are typically heavy-tailed and multimodal (a music set has clusters for genres, a speech set has clusters for speakers), and squashing them into a single Gaussian throws away that structure. FAD cannot tell the difference between a model that covers all the modes evenly and a model that piles onto one mode but happens to land the same mean and covariance. This *mode-coverage blindness* is the same critique leveled at FID, and it is why a model can post a great FAD while quietly suffering mode collapse — generating a narrow band of very-realistic-but-repetitive audio. The metric sees a well-matched Gaussian and is satisfied.

The matrix square root deserves a practical warning too, because it is where FAD implementations silently go wrong. $(\Sigma_r \Sigma_g)^{1/2}$ is a *matrix* square root, not an element-wise one, and the product $\Sigma_r \Sigma_g$ of two symmetric positive-definite matrices is not itself symmetric, so naive eigendecomposition can return small *complex* eigenvalues from floating-point error. Good implementations (the one in `scipy.linalg.sqrtm`, which the FAD packages call) handle this by discarding tiny imaginary parts and clipping negative eigenvalues to zero — but if your covariance is rank-deficient because you fed it fewer samples than the embedding dimension, the square root becomes numerically unstable and your FAD can come out as a small negative number or NaN. That is not a deep mystery; it is a sign you used too few samples to estimate a full-rank covariance, which brings us straight to the sample-size problem below.

![A dataflow graph showing generated and reference audio sets both entering a frozen embedding network, each producing a fitted Gaussian, which together feed the Frechet distance](/imgs/blogs/audio-quality-metrics-2.png)

The pipeline in the figure makes the dependency structure obvious. Two sets of audio go in. *One* frozen embedder turns both into point clouds. A Gaussian is fit to each. The distance between those Gaussians is your FAD. Notice what carries the entire result: the embedder. It is the only learned component, and it was trained on some specific data for some specific task that has nothing to do with judging your model. Whatever that network was built to hear is what FAD will reward you for matching; whatever it is deaf to, FAD is deaf to as well.

### Why the embedding choice dominates

The classic FAD, from Kilgour et al. (2019), used **VGGish** — a small CNN trained to predict AudioSet tags from 16 kHz log-mel patches. Sixteen kilohertz. That means VGGish-FAD is structurally incapable of caring about anything above 8 kHz, because the audio was low-pass filtered to fit the model's input before it ever saw it. A 44.1 kHz music model that produces gorgeous, airy cymbals and a model that produces dull, band-limited cymbals can score *identically* under VGGish-FAD, because the metric threw the cymbals' brilliance away at the front door. This is not a subtle bias; it is a deaf spot the size of an octave.

So the field moved. **PANNs** (the CNN14 model, Kong et al. 2020) runs at 32 kHz and was trained as a much stronger audio tagger; it hears more event structure. **CLAP** (Wu et al. 2023, the LAION variant) was trained contrastively on text-audio pairs at 48 kHz, so its embedding space is organized around semantic meaning — "is this a dog bark, sung in a minor key, with reverb" — rather than low-level acoustics. Each of these embedders carves the space of sounds differently. Match VGGish's notion of similarity and you can still fail CLAP's; ace CLAP's semantic alignment and you can still have audible artifacts VGGish would have caught.

![A matrix comparing VGGish, PANNs, and CLAP embeddings across their training data, sample rate, what they are sensitive to, and what they are blind to](/imgs/blogs/audio-quality-metrics-3.png)

The figure above is the practical consequence, and it is the most important thing to internalize about FAD: **the embedding is not an implementation detail, it is the metric.** Reported FAD numbers across two papers are not comparable unless both used the same embedder *and* the same reference set *and* a comparable sample size. "FAD 2.1" means nothing on its own. "VGGish-FAD 2.1 against the MusicCaps reference with 2,000 samples" is a claim you can reproduce and argue with. Always report the embedder. Always.

### The other two biases: sample size and the top-end blindness

Two more FAD pathologies will bite you.

**Sample-size sensitivity.** The covariance matrix $\Sigma$ has to be estimated from your samples, and covariance estimation is data-hungry — for a $d$-dimensional embedding you are estimating on the order of $d^2/2$ parameters. VGGish embeddings are 128-dimensional, so $\hat{\Sigma}$ has about 8,000 free entries; CLAP's are larger still. To estimate that many parameters even roughly well you want at least several times $d$ samples, and ideally an order of magnitude more. With too few, $\hat{\Sigma}$ is a noisy, biased estimate, and — this is the part that bites — the bias is *not* mean-zero. The estimated distance between two clouds is *inflated* by finite-sample noise, and the inflation shrinks as you add samples. So FAD has a built-in pull toward larger values at small $n$ that decays roughly like $1/n$. The upshot: FAD computed on 500 clips and FAD computed on 5,000 clips are *different metrics*, and the small-sample one reads systematically worse — not because the audio is worse, but because the estimator is noisier. If you compare your model on 800 generated clips to a baseline's reported number computed on 2,000, you may be losing entirely on sample count, with the audio quality identical.

There is a clean way to diagnose this: compute FAD at several sample sizes (say 500, 1,000, 2,000, 4,000) and plot the curve. A trustworthy FAD has flattened out by your chosen $n$ — adding more clips barely moves it — which tells you the estimate has converged. If the curve is still dropping steeply at your reported $n$, your number is dominated by sample noise and is not yet measuring the model. The discipline that follows is simple and non-negotiable: **fix a sample size, pick it large enough to be on the flat part of that curve, report it, and use the exact same size for everything you compare.** A FAD number without an $n$ next to it is as meaningless as a FAD number without an embedder next to it, and for the same reason — you have not specified the metric, only gestured at it.

**Weak correlation with humans at the top end.** This is the deep one. FAD correlates reasonably with human judgement when models are *far apart* in quality — a clearly broken model has high FAD and a clearly good one has low FAD, and humans agree. But at the frontier, where two strong models differ by a tenth of a FAD point, that correlation collapses. The metric is measuring distributional overlap in an embedding space, and once both models are close to the reference distribution, the residual FAD difference is mostly noise from the embedder's idiosyncrasies and your sample estimate, not a real perceptual difference. **FAD is a coarse filter, not a fine ranker.** Use it to catch a model that is badly off; do not use it to declare a winner between two good models by 0.1.

#### Worked example: VGGish says A wins, CLAP says B wins

You have two music models. On VGGish-FAD over 2,000 MusicCaps-style clips, Model A scores 3.1 and Model B scores 3.8 — A wins by 0.7, a gap that looks decisive. You are about to ship A. Out of diligence you recompute with CLAP-FAD on the same clips: now A scores 0.42 and B scores 0.33 — *B* wins. What happened? A produces clean mid-band audio (VGGish, deaf above 8 kHz, loves it) but its high frequencies are slightly metallic and its instruments are generic (CLAP, semantic and full-band, dings it). B has a touch more low-band roughness (VGGish penalizes) but richer, more prompt-faithful instrumentation and crisp highs (CLAP rewards). Neither metric is wrong; they are measuring different things. The resolution is not to pick the embedder that flatters your model — it is to run a small listening test and let humans break the tie, then report *both* FADs so a reader knows the ranking was embedder-dependent. This exact failure is why "we improved FAD by 0.7" is a sentence I no longer trust on its own.

### Computing FAD in practice

The `frechet-audio-distance` package wraps all of this, including the embedder zoo, so you do not implement the matrix square root yourself. The thing to be deliberate about is the `model_name`.

```python
# pip install frechet-audio-distance
from frechet_audio_distance import FrechetAudioDistance

# Pick the embedder ON PURPOSE. This choice IS the metric.
# "vggish" = 16 kHz classic (band-limited, blind above 8 kHz)
# "pann"   = 32 kHz AudioSet tagger (more event structure)
# "clap"   = 48 kHz text-audio contrastive (semantic, full band)
fad = FrechetAudioDistance(
    model_name="clap",          # report this in every result
    sample_rate=48000,
    use_pca=False,
    use_activation=False,
    verbose=True,
)

# Two folders of .wav files. Fix the count and keep it constant
# across everything you compare (sample size changes the metric).
score = fad.score(
    background_dir="data/reference_2000",   # real clips
    eval_dir="data/generated_2000",         # your model's clips
)
print(f"CLAP-FAD (n=2000): {score:.3f}")
```

A few non-obvious operational notes. Resample everything to the embedder's native rate *before* scoring (feeding 16 kHz audio to the CLAP embedder wastes its high-band capacity and feeding 48 kHz to VGGish just gets downsampled anyway). Use the *same* reference set every time — if you regenerate the reference embeddings from a different subset, your FAD drifts for reasons that have nothing to do with your model. And cache the reference statistics $(\mu_r, \Sigma_r)$; they do not change between runs, so there is no reason to recompute them on every evaluation.

## 3. CLAP-score: measuring text-audio alignment

FAD tells you whether your generated audio *looks like real audio*. It says nothing about whether it matches the *prompt*. A text-to-audio model that ignores your prompt entirely and produces a beautiful but unrelated clip can score a great FAD — the clip is realistic, just wrong. To catch that, you need a metric that compares the audio to the text you asked for. That is CLAP-score.

CLAP (Contrastive Language-Audio Pretraining) is the audio analogue of CLIP. It has a text encoder and an audio encoder trained so that a clip and its caption land near each other in a shared embedding space, and a clip and an unrelated caption land far apart. CLAP-score uses this directly: embed the generated audio with the audio tower, embed the prompt with the text tower, and take the cosine similarity between them.

$$
\text{CLAP-score} = \cos\!\big(E_\text{audio}(x_\text{gen}),\; E_\text{text}(c)\big) = \frac{E_\text{audio}(x_\text{gen}) \cdot E_\text{text}(c)}{\lVert E_\text{audio}(x_\text{gen})\rVert \, \lVert E_\text{text}(c)\rVert}
$$

Higher is better: a high cosine means the audio sits where the prompt's text says it should. It typically lands in roughly the 0.2–0.5 range for good text-to-audio systems (cosine similarities in a contrastive space are rarely near 1.0), and you read *relative* differences between models, not absolute values.

```python
# pip install laion-clap
import laion_clap
import numpy as np

model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt()  # downloads a pretrained LAION-CLAP checkpoint

prompts = ["a dog barking over upbeat lo-fi hip hop"]
wav_paths = ["generated/clip_001.wav"]  # 48 kHz mono

audio_embed = model.get_audio_embedding_from_filelist(
    x=wav_paths, use_tensor=False
)                                        # shape (N, D), L2-normalized
text_embed = model.get_text_embedding(prompts, use_tensor=False)

# cosine similarity = dot product of L2-normalized vectors
clap_score = float(np.dot(audio_embed[0], text_embed[0]))
print(f"CLAP-score: {clap_score:.3f}")   # higher = better prompt match
```

CLAP-score's blind spot is the mirror image of FAD's: it measures *semantic alignment* and is largely indifferent to *fidelity*. A clip that is unmistakably "a dog barking over lo-fi" but is riddled with codec artifacts and clipping can score a strong CLAP-score, because the semantic content is there and CLAP was not trained to be an audiophile. So CLAP-score and FAD are *complements*: FAD catches the unrealistic-but-on-topic failure, CLAP-score catches the realistic-but-off-prompt failure, and you need both. A subtle gotcha: use the *same* CLAP checkpoint for scoring that you would never use for training or guidance in the same system, or you risk a model that has learned to satisfy that specific CLAP rather than the underlying concept — the audio version of Goodhart's law, which we will hit again later.

A second subtlety worth flagging: CLAP-score has two flavors and they answer slightly different questions. The version above is *text-to-audio* similarity — embed the audio, embed the prompt, take the cosine — which is what you want for evaluating a text-to-audio generator. There is also an *audio-to-audio* variant that compares a generated clip to a *reference* clip in CLAP space, which behaves more like a semantic FAD on single pairs and is useful when you have a target style clip rather than a text prompt. Be explicit about which one you report, because a reader will assume the text-to-audio version unless you say otherwise. And do not over-interpret the absolute value: a CLAP-score of 0.35 is not "35% aligned." Cosine similarities in a contrastive space have no natural zero-to-one interpretation; they are only meaningful *relative* to other systems scored with the same checkpoint on the same prompts. The number is a ranking signal, not a percentage.

## 4. The speech metrics: PESQ, STOI, SI-SDR, and WER

Speech has a luxury music does not: very often there *is* a reference. When you synthesize a sentence whose recorded ground truth you hold, when you enhance a noisy clip you also have clean, when you separate a mix you also have the true stems — you can measure against the target directly. This unlocks the reference-based family, and for speech it is where the real signal lives.

### PESQ and STOI: what they actually model

**PESQ** (Perceptual Evaluation of Speech Quality, ITU-T P.862) was built to predict telephone-network MOS automatically. It aligns the degraded signal to the reference in time, maps both onto a perceptual (Bark-band, loudness-warped) representation that mimics the ear's frequency and loudness sensitivity, computes a disturbance between them, and regresses that onto a MOS-like score, roughly 1 to 4.5, higher is better. PESQ is good at what it was made for — narrowband and wideband *telephony* degradations like packet loss, codec noise, and level changes. It is shakier on the kinds of artifacts neural vocoders introduce (a faint metallic buzz, a phase smear) because those were not in its design distribution. Treat PESQ as a quality proxy with a known comfort zone, not a universal oracle.

**STOI** (Short-Time Objective Intelligibility) predicts *intelligibility* specifically — how many words a listener would get right — rather than overall pleasantness. It correlates short-time envelopes of the reference and the degraded signal across frequency bands and returns a number from 0 to 1, higher is better, where the correlation of the temporal envelopes stands in for "are the speech cues that carry word identity still present." STOI is the right tool when your question is "can people understand it," PESQ when your question is "does it sound clean." They answer different questions and a system can win one and lose the other.

```python
# pip install pesq pystoi soundfile
import soundfile as sf
from pesq import pesq
from pystoi import stoi

ref, sr = sf.read("clean_reference.wav")   # ground-truth clean speech
deg, _  = sf.read("model_output.wav")      # degraded / generated
# both must be the SAME length and aligned; trim or pad as needed.

# PESQ: 'wb' = wideband (16 kHz), 'nb' = narrowband (8 kHz)
pesq_wb = pesq(sr, ref, deg, "wb")          # ~1.0 .. 4.5, higher better
stoi_sc = stoi(ref, deg, sr, extended=False)  # 0 .. 1, higher better
print(f"PESQ {pesq_wb:.2f}   STOI {stoi_sc:.3f}")
```

The operational trap with both is *alignment*. PESQ and STOI compare frame-by-frame, so if your output is shifted by even a few tens of milliseconds relative to the reference, the score craters for a reason that has nothing to do with quality. For generative TTS where timing is not sample-aligned to a reference, this makes PESQ/STOI awkward — they shine for enhancement and separation where input and output share a timeline, and they are clumsy for free-running synthesis.

### SI-SDR: the separation and enhancement workhorse

**SI-SDR** (Scale-Invariant Signal-to-Distortion Ratio) is the standard for source separation and enhancement. It asks: of the energy in your output, how much is the target signal and how much is everything else (interference, artifacts, noise)? "Scale-invariant" means it first projects your estimate onto the target to remove any overall gain difference — a quieter-but-correct estimate is not punished for being quiet. The decomposition: project the estimate $\hat{s}$ onto the target $s$ to get the part that is genuinely target, $s_\text{target}$, call the rest error $e_\text{noise}$, and take the ratio in decibels.

$$
s_\text{target} = \frac{\langle \hat{s}, s\rangle}{\lVert s\rVert^2}\, s,
\qquad
\text{SI-SDR} = 10\log_{10}\frac{\lVert s_\text{target}\rVert^2}{\lVert \hat{s} - s_\text{target}\rVert^2}
$$

Higher is better, in dB; strong separation systems hit the high teens to twenties. The reason SI-SDR is loved is that scale-invariance removes a whole class of trivial cheating and false penalties. It is the wrong tool for free-running TTS or music generation — there is no aligned reference signal to project onto — but for "I have the true stem and want to know how cleanly I recovered it," it is exactly right.

```python
# pip install torchmetrics
import torch
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr

target = torch.tensor(ref, dtype=torch.float32)   # true source
est    = torch.tensor(deg, dtype=torch.float32)   # estimated source
print(f"SI-SDR: {si_sdr(est, target).item():.2f} dB")  # higher better
```

### WER: the real intelligibility test for TTS

Here is the metric I trust most for generative speech, and it is almost embarrassingly direct: have a machine *listen to your output and write down what it heard*, then compare that transcript to the text you asked the model to say. If your TTS swallows words, the ASR will mis-transcribe them, and the error rate climbs. **Word Error Rate** is the edit distance between the ASR transcript and the target text, normalized by the target length:

$$
\text{WER} = \frac{S + D + I}{N}
$$

where $S$ is the number of substituted words, $D$ deleted, $I$ inserted, and $N$ the number of words in the reference text. It is the Levenshtein (edit) distance at the word level over the reference length. The edit distance itself is computed by dynamic programming: build a table whose cell $(i, j)$ holds the minimum number of word-level edits to turn the first $i$ reference words into the first $j$ hypothesis words, filling it with the recurrence that each cell is the cheapest of (a match or substitution from the diagonal), (a deletion from above), or (an insertion from the left). The bottom-right cell is $S + D + I$, and dividing by $N$ gives WER. You never code this yourself — `jiwer` does it — but knowing the mechanism explains two of its quirks. First, WER can exceed 100%: if the model hallucinates a long run of extra words, the insertion count $I$ alone can be larger than $N$, so a badly-degenerate TTS output can post a WER of 150% or more. Second, a single dropped word in a short sentence is a brutal hit — drop one word from a five-word target and you are already at 20% WER — so always report WER over a *corpus* of sentences with enough total words that one unlucky deletion does not dominate.

Lower is better; a clean TTS system on clean text should land in the low single-digit percentages, and anything in the double digits means listeners are losing words. The beauty of WER is that it measures the thing that actually matters for speech — *can you understand it* — and it needs no recorded reference audio, only the *text* you already have, because you wrote the prompt. A close relative worth knowing is **CER** (Character Error Rate), the same edit distance at the character level; CER is more forgiving of single-phoneme slips and is the better choice for languages without clear word boundaries or for measuring fine pronunciation, while WER is the standard headline for English intelligibility. A second relative, important for *zero-shot voice cloning*, is **speaker-similarity** (SIM): embed the generated speech and the reference speaker's enrollment audio with a speaker-verification network (a WavLM or ECAPA-TDNN model) and take the cosine similarity. WER tells you the clone is intelligible; SIM tells you it actually sounds like the target speaker. A clone can ace WER while sounding like a generic voice — high intelligibility, low similarity — so cloning systems report both, and we return to this pairing in the voice-cloning posts of this series.

```python
# pip install jiwer openai-whisper
import whisper
from jiwer import wer

asr = whisper.load_model("base.en")     # the ASR acts as the "listener"
target_text = "the quick brown fox jumps over the lazy dog"

result = asr.transcribe("tts_output.wav")
hypothesis = result["text"].strip().lower()

score = wer(target_text.lower(), hypothesis)
print(f"ASR hypothesis: {hypothesis!r}")
print(f"WER: {score:.1%}")              # lower = more intelligible
```

Two honest caveats so you do not over-trust WER. First, *the ASR is the judge*, and ASR models have their own biases — a WER computed with `whisper-base` and one computed with `whisper-large-v3` differ, and a weak ASR can blame your TTS for the ASR's own errors. Pick a strong, fixed ASR and report which one, exactly as you report the FAD embedder. Second, normalize text consistently before scoring — lowercase, strip punctuation, expand or fix numbers ("\$5" versus "five dollars" versus "5 dollars" are three different strings to a naive WER) — or you will measure formatting differences and call them intelligibility. `jiwer` ships transforms for exactly this; use them, and use the *same* normalization for every system you compare.

To keep the four speech metrics straight, here is when each is the right tool and what it needs:

| Metric | Use case | Needs | Direction | Watch out for |
| --- | --- | --- | --- | --- |
| WER | TTS intelligibility | Target text + ASR | Lower | ASR choice, text normalization |
| PESQ | Speech quality (telephony) | Aligned clean reference | Higher | Misses neural-vocoder artifacts |
| STOI | Intelligibility prediction | Aligned clean reference | Higher | Time-alignment sensitivity |
| SI-SDR | Enhancement / separation | Aligned target signal | Higher | Useless without an aligned target |

The decisive split in this table is the *Needs* column. WER needs only text, which you always have for TTS, so it works for free-running synthesis. The other three need an aligned reference *signal*, which you have for enhancement and separation but not for open-ended generation — which is exactly why WER is the speech metric you reach for first in a generative pipeline and the others are specialists for the reference-rich tasks.

## 5. Human evaluation: MOS, CMOS, and MUSHRA

Every metric so far is a proxy. Human evaluation is not — it is the target itself, which is why, when a cheap metric and a listening test disagree, the listening test wins. The catch is that humans are a *noisy instrument*, and running them well is a small science.

**MOS** (Mean Opinion Score) is the classic protocol: play a clip, ask a listener to rate its quality on a 1-to-5 scale (1 bad, 5 excellent), average across many clips and many raters. The headline number is the mean, but the number you must report alongside it is the *confidence interval*, because MOS without a CI is just a vibe. A "MOS of 4.1" from eight raters on twenty clips might have a 95% CI of ±0.3 — wide enough that a "4.1 model" and a "3.9 model" are statistically indistinguishable, and announcing the 4.1 as a win is a lie of precision.

**CMOS** (Comparative MOS) sidesteps a major weakness of absolute MOS. Listeners are bad at absolute ratings (one person's 4 is another's 3) but good at *A-vs-B preferences*. CMOS plays two clips — yours and a baseline — and asks which is better and by how much, on a scale like −3 (B much better) to +3 (A much better), with 0 meaning no difference. The result is a *difference* with a CI, and because each listener judges both clips back-to-back, a lot of inter-rater variance cancels. **When you are comparing two strong models, CMOS is almost always the right protocol** — it is far more sensitive to small real differences than two separate absolute-MOS runs, exactly where FAD goes blind.

**MUSHRA** (Multi-Stimulus test with Hidden Reference and Anchor, ITU-R BS.1534) is the heavyweight, used when you need fine discrimination among several systems at once. The listener sees all systems for one item on sliders from 0 to 100, plus a *hidden reference* (the clean original, which a careful listener should rate ~100) and one or more *anchors* (deliberately degraded versions, e.g. a 3.5 kHz low-pass, which should rate low). Presenting all systems side by side on one screen is what makes MUSHRA so sensitive: the listener can A/B/C/D-compare directly rather than rating each clip in isolation against a remembered standard, which removes the absolute-scale drift that plagues plain MOS. The hidden reference and anchors are quality controls: a rater who scores the hidden reference at 60 or the anchor at 90 was not listening, and you screen them out — the standard prescribes excluding a listener who rates the hidden reference below 90 on too many trials. MUSHRA gives you the tightest, most defensible ranking, at the highest cost in listener time and setup, and it is overkill for a simple two-system comparison where CMOS is faster and just as conclusive. Reach for MUSHRA when you have three-plus systems and need to rank them all at once with audiophile-grade discrimination; reach for CMOS when it is just you versus a baseline.

```python
# A MOS / CMOS study scaffold: structure the trials, then analyze.
import numpy as np
import pandas as pd

# Each row = one (rater, clip, system) judgement collected from your UI.
# For CMOS, `score` is in [-3, +3]: + means systemA preferred over baseline.
ratings = pd.read_csv("listening_study.csv")   # cols: rater, clip, system, score

def mos_with_ci(scores):
    scores = np.asarray(scores, dtype=float)
    mean = scores.mean()
    # 95% CI via standard error of the mean (report it ALWAYS)
    ci95 = 1.96 * scores.std(ddof=1) / np.sqrt(len(scores))
    return mean, ci95

for system, grp in ratings.groupby("system"):
    mean, ci = mos_with_ci(grp["score"])
    print(f"{system:>12}: {mean:.2f} +/- {ci:.2f}  (n={len(grp)})")

# A simple gate: a CMOS difference is only meaningful if its CI excludes 0.
cmos = ratings[ratings.system == "ours_vs_baseline"]["score"]
m, ci = mos_with_ci(cmos)
verdict = "significant" if abs(m) > ci else "NOT significant"
print(f"\nCMOS ours vs baseline: {m:+.2f} +/- {ci:.2f}  -> {verdict}")
```

Running human eval honestly comes down to a few disciplines. *Rater count and trials:* aim for at least 15–20 raters and enough clips that your CI is tight enough to resolve the difference you care about — if your CI is wider than the gap, you have not run a big enough study to make the claim. *Calibration and screening:* use MUSHRA's hidden-reference/anchor trick (or attention-check clips in MOS) to catch raters who are clicking randomly, and discard them before averaging. *Counterbalancing:* randomize clip order per rater, because people anchor hard on the first thing they hear. *Fixed content:* score every system on the *same* set of prompts/sentences, or you are comparing prompts, not systems. None of this is exotic, but skipping it is how a "listening study" produces a number no more reliable than the FAD it was supposed to validate.

#### Worked example: a CMOS study that flips a FAD ranking

Two TTS systems, X and Y, tie on FAD at 2.3 (CLAP-FAD, 2,000 clips). You cannot separate them on the distributional metric, so you run a CMOS study: 20 raters, 30 sentences, each rater hears X and Y back-to-back for every sentence and scores −3 to +3 (positive favors X). The mean CMOS comes back +0.4 with a 95% CI of ±0.15. Because the CI excludes zero, the preference for X is real, not noise — listeners reliably prefer X by a small but genuine margin. You also run WER with a fixed `whisper-large-v3`: X is 4%, Y is 12%. Now the picture is coherent: the systems looked tied on FAD, but Y muddies diction, the ASR catches it as a higher WER, and humans feel it as a CMOS preference for X. The FAD tie was not wrong — FAD genuinely could not see this difference — it was just *insufficient*, and the basket of metrics resolved what the headline number could not.

![A before-and-after figure contrasting two systems that tie on FAD but split on WER and CMOS, with system X winning on intelligibility and preference](/imgs/blogs/audio-quality-metrics-8.png)

## 6. The eval crisis: when models are tuned to the metric

Now the uncomfortable part, and it is the same crisis the image field lived through with FID. Once a number becomes the number everyone reports, the field starts optimizing *it* rather than the thing it was meant to proxy. This is Goodhart's law — when a measure becomes a target, it stops being a good measure — and audio is deep in it.

The mechanism is mundane. FAD is cheap, so you compute it every checkpoint. You make architecture and hyperparameter choices that move FAD down, because that is the number on your dashboard. Some of those choices genuinely improve the audio; some of them just exploit the embedder's quirks — they produce audio that sits closer to the reference *in VGGish space* without sounding better to a person. Over enough iterations, you have a model that is *tuned to FAD*, and its FAD-to-MOS relationship has quietly decoupled. The leaderboard says you are winning. The listeners say "it sounds a bit off, I can't say why." Both are telling the truth about different things.

This decoupling is worst exactly where it hurts most: at the frontier, between strong models. We saw why in §2 — FAD's correlation with human judgement is decent across a wide quality range and collapses to noise when models are close. So the regime where you most need to tell two good models apart is precisely the regime where FAD is least able to, and yet it is the regime where people report FAD to three decimals and crown winners. The number is being asked to do the one thing it cannot.

![A before-and-after figure showing a low-FAD but unintelligible model losing to a higher-FAD model that humans prefer because it is clear](/imgs/blogs/audio-quality-metrics-5.png)

The figure makes the failure concrete. Model A wins on FAD and loses to humans because it is unintelligible; Model B looks worse on the headline metric and wins where it counts. If you had only looked at FAD, you would have shipped A — which is exactly the mistake in this post's opening. The lesson is not "FAD is useless." FAD is a fine coarse filter and a useful smoke test. The lesson is that *no single number suffices*, and that treating any one metric as the objective will, given enough optimization pressure, produce a model that games it.

The image people wrote this story down first, and it is worth reading their version because the structure is identical — FID's embedder bias, sample-size sensitivity, and top-end blindness are FAD's, one field over. See [evaluating image generation honestly](/blog/machine-learning/image-generation/evaluating-image-generation-honestly); everything they say about FID transfers, and the few audio-specific twists (intelligibility via WER, the phase artifacts PESQ misses) are what this post adds.

#### Worked example: chasing FAD into a worse model

A concrete version of the crisis, because it is easy to nod along and still do it. You are training a music model and your dashboard plots VGGish-FAD every 5,000 steps. At step 50k you are at FAD 4.2. You try a change — a heavier reconstruction weight in the codec, which sharpens the mid-band the embedder is most sensitive to — and FAD drops to 3.6. Great, ship the recipe. You try another change that biases the model toward the most common textures in the training set, and FAD drops again to 3.1, because matching the *bulk* of the reference distribution is exactly what the metric rewards. You now have FAD 3.1, a full point better than where you started, and a model that has quietly learned to produce safe, average, slightly-repetitive music — because "closer to the mean of the reference Gaussian" and "more interesting and varied" are not the same goal, and FAD only measures the first. You run a CMOS study against the step-50k checkpoint and listeners *prefer the old one* by +0.3 (CI ±0.12): the old model was more varied and engaging even though its FAD was worse. The 1.1-point FAD "improvement" was partly real (the codec change) and partly a march into mode collapse that the metric applauded. The fix is not to abandon FAD — it caught nothing wrong here, it just could not see variety — but to keep a CLAP-score (which would have flagged the drop in prompt-faithful variety) and a periodic small listening test in the loop so the metric you optimize cannot quietly diverge from the thing you want.

## 7. The honest harness: a basket, not a number

If no single number works, what do you actually do? You build a *basket* — a small panel of metrics chosen so that each one covers another's blind spot — and you read them together. For a generic audio-generation system, the basket I reach for is:

- **FAD** (named embedder, fixed reference, fixed sample size) for distributional realism — the coarse "is this even in the right ballpark" filter.
- **CLAP-score** for prompt alignment — catches the realistic-but-off-prompt failure FAD is blind to.
- **WER** (named ASR, consistent text normalization) whenever there is speech — the real intelligibility test, the one that would have caught Model A.
- **A small MOS or CMOS study** for the decisions that matter — the only metric that is not a proxy, reserved for final candidates because it is expensive.
- Plus, *when a reference exists* (enhancement, separation, reference TTS): **PESQ/STOI** for perceptual quality/intelligibility and **SI-SDR** for separation fidelity.

![A layered stack showing the evaluation basket of FAD, CLAP-score, WER, reference-based signal metrics, and a MOS or CMOS study feeding a single decision](/imgs/blogs/audio-quality-metrics-4.png)

The stack in the figure is the whole philosophy in one picture: cheap metrics run constantly as a fast signal, the human study runs rarely on the finalists, and the *decision* comes from reading them together rather than from any single row. The cheap metrics tell you when something is badly wrong and let you iterate fast; the human study tells you which of two good models to ship. A model that wins the cheap metrics but loses the human study does not get shipped — the human study is the tiebreaker by construction, because it is the only one that is not a proxy.

Here is the basket as a single evaluation function — the scaffold I actually use, with each metric named and its embedder/ASR pinned so the result is reproducible and comparable.

```python
import numpy as np
from frechet_audio_distance import FrechetAudioDistance
import laion_clap, whisper
from jiwer import wer

def evaluate_audio_model(gen_dir, ref_dir, prompts, target_texts, gen_wavs):
    report = {}

    # 1) FAD -- distributional realism. Pin the embedder + sample size.
    fad = FrechetAudioDistance(model_name="clap", sample_rate=48000)
    report["fad_clap_n2000"] = fad.score(ref_dir, gen_dir)

    # 2) CLAP-score -- prompt alignment (mean cosine over the set).
    clap = laion_clap.CLAP_Module(enable_fusion=False); clap.load_ckpt()
    a = clap.get_audio_embedding_from_filelist(gen_wavs, use_tensor=False)
    t = clap.get_text_embedding(prompts, use_tensor=False)
    report["clap_score"] = float(np.mean([np.dot(a[i], t[i])
                                          for i in range(len(prompts))]))

    # 3) WER -- intelligibility, only where there is speech. Pin the ASR.
    asr = whisper.load_model("large-v3")
    wers = []
    for wav, txt in zip(gen_wavs, target_texts):
        hyp = asr.transcribe(wav)["text"].strip().lower()
        wers.append(wer(txt.lower(), hyp))
    report["wer"] = float(np.mean(wers))

    # 4) MOS/CMOS is run SEPARATELY with human raters on the finalists.
    report["mos_note"] = "run a 20-rater CMOS study on the top 2 candidates"
    return report
```

Notice what the function does *not* do: it does not collapse the basket into a single weighted score. Resist that urge. A weighted sum just recreates the single-number problem with extra steps, and the weights are arbitrary. Report the vector — FAD, CLAP, WER, and the human result — and make the decision by reading it. The metrics that disagree are the *interesting* ones; they are telling you something a scalar would have hidden.

![A metric cheat-sheet matrix listing FAD, CLAP-score, WER, PESQ, STOI, SI-SDR, and MOS with whether each is reference-free, what it captures, and its failure mode](/imgs/blogs/audio-quality-metrics-6.png)

The cheat-sheet matrix above is the one to pin above your desk: every metric, whether it needs a reference, what it captures, and how it fails. The failure-mode column is the load-bearing one — knowing that FAD's failure is embedder bias, CLAP's is fidelity-blindness, and WER's is ASR-dependence is what lets you read a disagreement correctly instead of trusting whichever number is highest.

For reference, here is the same information as a table you can copy into a model card, with the direction each metric points and a typical good-range so you can sanity-check a number at a glance. Treat the ranges as rough — they depend on the embedder, ASR, reference, and content — and never as universal thresholds.

| Metric | Reference-free? | Direction | What it captures | Typical good range | Failure mode |
| --- | --- | --- | --- | --- | --- |
| FAD | Yes (distribution) | Lower | Realism / style match | 1–5 (embedder-dependent) | Embedder bias, sample-size, top-end blindness |
| CLAP-score | Yes (prompt only) | Higher | Text-audio alignment | ~0.2–0.5 cosine | Ignores fidelity; Goodhart if used for guidance |
| WER | Needs target text | Lower | Speech intelligibility | <5% clean TTS | ASR is the judge; text normalization |
| CER | Needs target text | Lower | Fine pronunciation | <2% clean TTS | Same as WER; ASR-dependent |
| PESQ | Needs clean ref | Higher | Perceptual quality | 3.5–4.5 good | Telephony-tuned; misses neural artifacts |
| STOI | Needs clean ref | Higher | Intelligibility (envelope) | >0.9 good | Needs time alignment |
| SI-SDR | Needs ref signal | Higher | Separation fidelity | 10–20 dB good | Meaningless for free-running gen |
| Speaker-SIM | Needs enroll audio | Higher | Voice match (cloning) | >0.6 cosine | Network-dependent; orthogonal to WER |
| MOS | No (listeners) | Higher | Absolute quality | 4.0–4.5 good | Noisy; needs CI and rater count |
| CMOS | No (listeners) | Higher (vs base) | A-vs-B preference | CI must exclude 0 | Only relative; needs counterbalancing |

The two columns that save you from embarrassment are *direction* and *reference-free?*. Half of all "we beat the baseline" mistakes are reading a lower-is-better metric as higher-is-better; the other half are comparing a number that needs a reference against one computed without the same reference. Check both before you believe any row.

## 8. Measuring honestly: the discipline behind every number

A metric is only as trustworthy as the protocol that produced it, and most reported audio numbers are wrong not because the metric is bad but because the protocol was sloppy. Here is the checklist I run before I believe any number — mine or anyone else's.

**Fix the reference set and never quietly change it.** FAD and CLAP-FAD are distances *to a specific reference distribution*. If you evaluate against MusicCaps one week and against your own curated set the next, your FAD has changed for reasons that have nothing to do with your model, and any trend line across that change is fiction. Pick the reference set deliberately (it should represent the distribution you actually want to match), freeze it, cache its embeddings and Gaussian statistics, and write the set's identity into the result. A FAD trend is only a model trend if the reference held still.

**Pin every frozen model.** FAD has an embedder; WER has an ASR; CLAP-score has a CLAP checkpoint; speaker-similarity has a verification network. Each of these is a learned judge with its own version, and upgrading it silently shifts every score. Record the exact checkpoint — "VGGish from the TensorFlow Hub release," "whisper-large-v3," "LAION-CLAP `630k-audioset-best`" — in the result, the way you would record a random seed. Two numbers from different judges are not comparable, full stop.

**Control the generation, not just the evaluation.** Fixed seed, fixed text or prompt set, fixed decoding parameters (temperature, guidance scale, number of diffusion steps). If model A used 50 diffusion steps and model B used 25, you are comparing step counts, not models. If A's WER was measured on easy sentences and B's on hard ones, you are comparing sentence difficulty. The single most common way a "fair comparison" is unfair is that the generation settings drifted while everyone watched the evaluation settings.

**Warm up before timing.** This one is for the speed metrics that the rest of this series cares about — real-time factor and seconds-to-generate. The first call to a model includes one-time costs: CUDA kernel compilation, `torch.compile` tracing, lazy weight loading, cache allocation. Time *that* call and you will report a real-time factor two or three times worse than the steady state. Always discard the first one or two runs as warm-up, then time the median of several runs, and name the device — "RTF 0.04 on an A100 80GB, fp16, batch 1, after warm-up" is a number a reader can reproduce; "RTF 0.1" is not.

**Report uncertainty, not just point estimates.** For human eval this means a confidence interval, as hammered in §5. For the cheap metrics it means at least an awareness of run-to-run variance: FAD wobbles with the particular sample you drew, so if two models differ by less than that wobble, they are tied. A cheap and honest move is to compute FAD on two or three disjoint subsets of your generated clips and look at the spread; if the spread is 0.3 and your "improvement" is 0.2, you have not improved anything.

**Separate the dev metric from the report metric.** The cheap metrics you watch every checkpoint are a *development signal* — you want them fast and frequent, and it is fine that they are imperfect. The numbers you put in a report or a model card are a *claim*, and they deserve the full protocol: large fixed sample size, pinned judges, multiple seeds, a confidence interval, and a human study on the finalists. Conflating the two — reporting your noisy every-checkpoint FAD as if it were a settled result — is how the precision theater starts.

#### Worked example: an apples-to-oranges FAD comparison, corrected

A paper claims their music model beats MusicGen with "FAD 2.1 vs 2.4." You want to verify before citing it. You check the fine print: their 2.1 was VGGish-FAD on 1,000 of their own generated clips against a reference set they curated; MusicGen's published 2.4 was VGGish-FAD on 2,000 MusicCaps-conditioned clips against the MusicCaps reference. Three things differ — sample size (1,000 vs 2,000, and smaller $n$ reads *worse*, so their real number is if anything understated, but it is still a different metric), the reference set (their curation vs MusicCaps), and the prompt conditioning. The "2.1 vs 2.4" comparison is meaningless as stated. To make it real you would regenerate both models' outputs on the *same* prompts, score against the *same* reference, at the *same* $n$, with the *same* embedder — and quite possibly the gap vanishes or flips. This is not a hypothetical pedantry; it is the single most common reason a published audio comparison does not replicate, and it is why the measurement discipline above is the actual content of "evaluating honestly."

## 9. Case studies and real numbers

Concrete numbers from the literature, with the caveats that make them honest. I am giving order-of-magnitude figures and the conditions under which they were measured; treat any single decimal as approximate and tied to a specific setup.

**EnCodec/DAC bitrate-vs-fidelity.** Neural codecs are evaluated heavily by reference-based metrics because the reference is the input audio. The Descript Audio Codec paper (Kumar et al. 2023) reports that DAC reconstructs 44.1 kHz audio at ~8 kbps with quality competitive with EnCodec at much higher bitrates, measured with a mix of objective scores and MUSHRA-style listening tests — the listening test is what carries the claim, because at high fidelity the objective metrics compress together. The takeaway for evaluation: for codecs, report reconstruction quality with a *reference-based* metric plus a listening test, not FAD; FAD measures distribution, codecs are about reconstruction.

**MusicGen and FAD.** The MusicGen paper (Copet et al. 2023) reports VGGish-FAD on MusicCaps along with a KL-divergence content metric, a CLAP-score for text alignment, and human studies for overall quality and prompt relevance — *a basket, exactly as argued here.* Their human study, not the FAD, is what supports the headline quality claims, and the CLAP-score is what supports the "follows the prompt" claims. It is a model paper that evaluates the way this post recommends, which is not an accident.

**TTS WER/MOS at the frontier.** Modern zero-shot TTS systems (the VALL-E lineage, Wang et al. 2023, and successors like F5-TTS and XTTS) report WER via a fixed ASR and speaker-similarity alongside MOS and CMOS. The pattern across these papers: WER differences in the low single digits are where the systems actually separate on intelligibility, and CMOS against a strong baseline is the protocol used to claim a quality win, precisely because absolute MOS is too noisy to resolve frontier-level differences. When a TTS paper reports only FAD-style numbers and no WER, be suspicious — they have not measured intelligibility, the thing TTS exists to deliver.

**The FAD-embedder reshuffle, in the wild.** Multiple text-to-audio evaluations have shown that swapping VGGish for a stronger embedder (PANNs or CLAP) changes model rankings, which is exactly the §2 worked example playing out across real papers. This is why cross-paper FAD comparison is so treacherous: unless two papers pin the *same* embedder, reference set, and sample size, their FAD numbers are not on the same axis, and "ours is 0.3 lower" may be measuring the embedder, not the model.

**Stable Audio and long-form evaluation.** The Stable Audio work (Evans et al. 2024) generates variable-length, full-length music up to minutes long, which breaks a hidden assumption baked into a lot of metric tooling: that clips are short and roughly uniform in length. FAD embedders chunk audio into fixed windows and pool, so a 90-second piece and a 10-second piece are not embedded comparably, and a metric computed over mixed lengths is measuring length as much as quality. The honest move for long-form is to fix the clip duration you score (or to score consistent sub-windows) and to report it. It is another instance of the same lesson: the metric has a precondition — here, comparable clip length — and violating it silently corrupts the number.

**Vocoder evaluation and the artifacts objective metrics miss.** GAN vocoders like HiFi-GAN (Kong et al. 2020) are judged largely by listening tests and reference-based scores, because their characteristic failures — a faint metallic ring, a periodic buzz, a subtle phase smear in transients — are exactly the artifacts PESQ was not built to catch and that FAD's pooled embedding can wash out. The HiFi-GAN paper leans on MOS for its quality claims and on real-time factor for its speed claims (it synthesizes far faster than real time on a modern GPU), and that pairing — MOS for quality, RTF for speed — is the right basket for a vocoder, where the whole point is high quality *fast*. We dig into vocoder trade-offs in the GAN-vocoder post of this series; the evaluation lesson here is that when the failure mode is a subtle artifact, the objective metrics get quiet and the human ear gets loud, so you must run the listening test.

The thread through all four cases is the same: every metric carries an implicit precondition (matched embedder, comparable length, an aligned reference, audibility of the failure mode), and a number is only honest when its precondition held. The papers that evaluate well do not report a single hero number — they report a basket, name their judges, and let the human study carry the claims the cheap metrics cannot.

## 10. When to reach for each metric (and when not to)

Decisive recommendations, because that is what a metrics primer is for. The decision starts not from the metric but from the *task* — what you are evaluating selects which branch of metrics is even relevant, and most real projects need a leaf from more than one branch.

![A decision tree branching from the evaluation task into realism, speech, and alignment, each selecting the appropriate leaf metric such as FAD, WER, PESQ, or CLAP-score](/imgs/blogs/audio-quality-metrics-7.png)

The tree above is the routing logic. If you are evaluating *realism* of music or general audio with no per-clip target, you are in distributional territory: FAD (named embedder) is your branch. If you are evaluating *speech*, split again — intelligibility of free-running TTS goes to WER, while quality of enhancement or separation against a reference goes to PESQ/STOI and SI-SDR. If you are evaluating *alignment to a prompt*, CLAP-score is the leaf. The point of drawing it as a tree is that the wrong branch produces a confidently-wrong number: reach for SI-SDR on generative TTS and you will get a meaningless value because there is no aligned target to project onto; reach for FAD to judge intelligibility and you will miss slurred words entirely. Pick the branch from the task first, then the leaf.

**Reach for FAD** when you want a cheap, fast smoke test of distributional realism for music or general audio, run every checkpoint. **Do not** reach for FAD to declare a winner between two strong models by a tenth of a point — that gap is noise — and never compare FAD across papers without matched embedder, reference, and sample size.

**Reach for CLAP-score** whenever there is a text prompt and you want to know if the model followed it; it is the cheapest catch for the off-prompt failure. **Do not** treat it as a fidelity metric — a high CLAP-score with audible artifacts is entirely possible — and do not train against the same CLAP you evaluate with, or you will Goodhart it.

**Reach for WER** for *any* speech system; it is the single most informative cheap metric for TTS because it measures intelligibility directly, needs only the text you already have, and would have caught the Model A disaster. **Do not** trust a WER without naming the ASR and fixing the text normalization, and do not blame your TTS for a weak ASR's errors — use a strong, fixed ASR.

**Reach for PESQ/STOI/SI-SDR** when you have an aligned reference: enhancement, separation, reference-anchored TTS. **Do not** use them for free-running generation where there is no aligned target — the frame-by-frame comparison will punish timing differences as if they were quality defects, which they are not. SI-SDR in particular is meaningless for generative TTS or music; it is a separation/enhancement metric.

**Reach for human eval — CMOS for two strong models, MUSHRA for several, MOS for absolute quality** — for the decisions that actually ship. **Do not** run it on every checkpoint (too slow and costly), report it without a confidence interval (precision theater), or run it with too few raters to resolve the gap you are claiming (an underpowered study is worse than none, because it manufactures false confidence).

And the meta-recommendation that overrides all of these: **do not ship on a single number.** Read the basket. When the cheap metrics and the human study agree, you have a clear result. When they disagree, the disagreement is the finding — trace it (is it a fidelity issue FAD missed, an intelligibility issue WER caught, an alignment issue CLAP flagged?), and let the human study break the tie. This is the eval discipline the whole rest of the series assumes, and we go much deeper on the honest-harness mechanics in [evaluating audio generation honestly](/blog/machine-learning/audio-generation/evaluating-audio-generation-honestly), then put it to work end-to-end in the capstone, [building an audio generation stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack).

One last practical note on *how often* to run each. The cheap metrics — FAD, CLAP-score, WER — are fast enough to run on every evaluation checkpoint and cheap enough that there is no reason not to; they are your gauges in the cockpit during training. The reference-based signal metrics run on whatever validation set has a reference, also cheap, also frequent. The human study is the expensive instrument you bring out for the decisions that ship: typically once, on the two or three finalists, after the cheap metrics have already narrowed the field. Spending the cheap metrics liberally and the expensive one surgically is the entire economics of honest audio evaluation, and it is what lets you iterate fast without ever quite trusting the number you iterated on.

## Key takeaways

- **Three families, three questions.** Reference-free (FAD, CLAP-score) ask "is it realistic / on-prompt," reference-based (PESQ, STOI, SI-SDR, WER) ask "how close to a target," human eval (MOS, CMOS, MUSHRA) asks "do listeners prefer it." Know which question you are asking before you pick a metric.
- **For FAD, the embedding *is* the metric.** VGGish (16 kHz, band-limited), PANNs (32 kHz), and CLAP (48 kHz, semantic) rank models differently. Always report the embedder, reference set, and sample size; never compare FAD across papers without all three matched.
- **FAD is a coarse filter, not a fine ranker.** It correlates with humans when models are far apart and decouples at the frontier — exactly where people misuse it to crown winners by 0.1.
- **WER is the most informative cheap metric for speech.** Transcribe with a fixed strong ASR, normalize text consistently, and you measure intelligibility directly using only the text you already wrote.
- **CLAP-score and FAD are complements.** FAD catches realistic-but-off-prompt; CLAP catches on-prompt-but-unrealistic. Use both for text-to-audio.
- **Reference-based metrics need an aligned reference.** PESQ/STOI/SI-SDR shine for enhancement and separation and are clumsy or meaningless for free-running generation.
- **Human eval is the gold standard but a noisy instrument.** Report a confidence interval always; use CMOS for two strong models; screen raters with hidden references and anchors; counterbalance order.
- **The eval crisis is Goodhart's law.** Tune to any single metric long enough and you get a model that games it. No single number suffices.
- **Ship on a basket, not a number.** FAD + CLAP + WER + a small MOS/CMOS study, read together. Disagreements among them are the finding, not an annoyance.

## Further reading

- Kilgour, Zuluaga, Roblek, Sharifi — *Fréchet Audio Distance: A Reference-Free Metric for Evaluating Music Enhancement Algorithms* (2019). The paper that introduced FAD with VGGish.
- Heusel, Ramsauer, Unterthiner, Nessler, Hochreiter — *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium* (2017). The original FID, whose formula FAD reuses verbatim.
- Wu, Chen, Zhang, Berg-Kirkpatrick, Dubnov — *Large-Scale Contrastive Language-Audio Pretraining (CLAP)* (2023). The embedding behind CLAP-score and CLAP-FAD.
- ITU-T Recommendation P.862 — *Perceptual Evaluation of Speech Quality (PESQ)*; and Taal, Hendriks, Heusdens, Jensen — *An Algorithm for Intelligibility Prediction of Time-Frequency Weighted Noisy Speech (STOI)* (2011).
- Le Roux, Wisdom, Erdogan, Hershey — *SDR — Half-baked or Well Done?* (2019). The paper that introduced SI-SDR and explains why scale invariance matters.
- Copet, Kreuk, Gat, Remez, Kant, Synnaeve, Adi, Défossez — *Simple and Controllable Music Generation (MusicGen)* (2023). A model paper that evaluates with a metric basket the way this post recommends.
- Within this series: [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard), [the mathematics of audio signals](/blog/machine-learning/audio-generation/the-mathematics-of-audio-signals), the deeper [evaluating audio generation honestly](/blog/machine-learning/audio-generation/evaluating-audio-generation-honestly), and the capstone [building an audio generation stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack).
- The image-world ancestor of this whole story: [evaluating image generation honestly](/blog/machine-learning/image-generation/evaluating-image-generation-honestly) — FID's embedder bias, sample-size sensitivity, and top-end blindness are FAD's, one field over.
