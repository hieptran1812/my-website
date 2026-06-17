---
title: "The 2026 Audio Model Landscape: Who Leads What, and How to Choose"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "A buyer's-and-builder's map of the 2025-2026 audio generation field — the leaders in speech, music and sound, the axes that actually differentiate them, an honest comparison harness you can run, and a decision guide for picking the right model for each job."
tags:
  [
    "audio-generation",
    "audio-synthesis",
    "text-to-speech",
    "music-generation",
    "neural-audio-codec",
    "model-comparison",
    "generative-ai",
    "deep-learning",
    "mlops",
  ]
category: "machine-learning"
subcategory: "Audio Generation"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/the-2026-audio-model-landscape-1.png"
---

A colleague pinged me last month with a question that, on its face, sounded simple: *"We need to add a voice to our product, generate background music for some clips, and add sound effects to a few auto-edited videos. Which models do we use?"* I started typing an answer and stopped after the third paragraph, because the honest reply is that there is no single answer — there are at least three answers, the right one in each case depends on a constraint they had not told me yet (latency? budget? can it leave our VPC?), and half of what they'd read online was either a leaderboard screenshot with no methodology or a vendor's own benchmark. So instead of a Slack message I wrote them a map. This post is that map, cleaned up and made public.

If you have read the rest of this series, you have the machinery: you know what a [neural audio codec](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound) is, why [residual vector quantization](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq) bends its rate-distortion curve the way it does, how an [autoregressive codec language model](/blog/machine-learning/audio-generation/autoregressive-audio-models-wavenet-to-audiolm) differs from a [latent diffusion model for music](/blog/machine-learning/audio-generation/latent-diffusion-for-music-stable-audio), and what it takes to clone a voice from three seconds of audio. What you may *not* have is a synthesis: a single view of who leads what in 2026, why they lead, on which axis, and how to choose between them when you have a real job to ship. That is the gap this post fills. It is opinionated on purpose — a survey that refuses to recommend anything is useless — but every standing here is dated, every number is marked approximate, and I will tell you exactly where the benchmarks lie so you can build your own ranking instead of trusting mine.

![A five-row matrix comparing ElevenLabs, Suno, open TTS, MusicGen and MMAudio across domain, access, fidelity, latency and license, showing no single overall winner](/imgs/blogs/the-2026-audio-model-landscape-1.png)

The frame for the whole post is the one this series has carried throughout: the **audio stack** — waveform to neural-codec tokens or a mel latent, into a generative core (an autoregressive language model, a diffusion model, or a flow-matching model), back out through a vocoder or codec decoder to a waveform — under the four-way tension of **fidelity, controllability, speed, and length**. Every model below is a different point in that space, optimized for a different corner of it. By the end you will be able to: name the leader and the open challenger in each of the three sub-domains; define each comparison axis *operationally* so your ranking is grounded in numbers rather than vibes; run a small reproducible harness that scores several open models on the same input; read a comparison table without being misled; and walk away from a job description with a defensible model choice and a rough cost. Let's build the map.

## The three sub-domains (and why you must pick one first)

The single most common mistake I see — in Slack questions, in RFP documents, in conference-talk Q&A — is treating "audio generation" as one thing. It is not. It is three loosely related fields that happen to share a substrate (the codec) and a lot of machinery (the generative core), but that diverge sharply in what "good" means, who leads, and what you can self-host. Before you compare any two models, you have to know which sub-domain you are in, because a music model and a speech model are not competing — they are answering different questions.

The three are **TTS / voice** (turn text into speech, optionally in a specific person's voice), **music** (turn a text prompt, and increasingly lyrics, into instrumental loops or full songs), and **sound / SFX** (turn a text prompt or a *video* into sound effects, foley, and ambience). They differ in their hardest constraint. For speech the hard constraint is *intelligibility and speaker identity* — a listener will forgive a slightly artificial timbre but not a mispronounced word or a voice that drifts away from the target speaker. For music the hard constraint is *long-range musical coherence* — staying in key, holding a groove, and keeping a verse-chorus structure over minutes, which is a far longer horizon than a sentence of speech. For sound effects, when they are conditioned on video, the hard constraint is *temporal synchronization* — the clink has to land on the frame where the glasses touch, or the whole illusion collapses. Three different "hard parts," three different leaderboards.

![A tree mapping audio generation into TTS, music and sound sub-domains, each with a closed leader and an open workhorse](/imgs/blogs/the-2026-audio-model-landscape-2.png)

Here is the leader map as it stands in mid-2026, and I'll defend each entry in its own section below. In **TTS / voice**, the closed quality leader is **ElevenLabs**, with the open field led by **XTTS v2**, **F5-TTS**, **Fish-Speech**, and the tiny-but-mighty **Kokoro**; the research frontier on naturalness is **NaturalSpeech 3**; and a fourth, newer category — **speech-native full-duplex** models like **Moshi** and the GPT-4o voice mode — is its own thing entirely, because it dissolves the boundary between TTS and ASR into a single real-time conversational model. In **music**, the closed leaders that generate *full songs with vocals* are **Suno** and **Udio**; the open field for instrumental and short-form is **MusicGen** and **Stable Audio Open**, with **YuE** as the first credible open attempt at full songs with vocals. In **sound / SFX**, the open workhorses are **AudioGen** and **AudioLDM 2** for text-to-sound, and **MMAudio** and Meta's **Movie Gen Audio** for video-to-audio. Underneath all of them sit the **codecs** — **EnCodec**, **DAC**, **Mimi** — which set the quality ceiling and the token budget for everything above.

Notice what that map implies: if your colleague's three jobs (voice, music, SFX) really are three jobs, they need at least three models, and quite possibly three *vendors*, because the leader in each sub-domain is a different organization with a different access model. There is no "GPT-4 of audio" that does all three well. The closest thing to a unified provider is a large lab's omni-model (the GPT-4o family handles speech well and can describe audio), but for *generation* quality in music and SFX you still reach for specialists. Accept that up front and the rest of the decision gets much easier.

It is worth understanding *how* the field arrived at this three-specialist shape, because the history explains the present and predicts the trajectory. The lineage runs through one decisive turn. The first wave of neural audio generation, around 2016, modeled the **raw waveform** directly — WaveNet predicted one 16-bit sample at a time with dilated causal convolutions, which sounded remarkable and ran at a glacial fraction of real time because a second of 24 kHz audio is 24,000 sequential predictions. The second wave, Tacotron 2 and its descendants around 2017-2019, split the problem into text-to-mel-spectrogram plus a separate vocoder, which was the right factorization and made TTS practical, but still treated the signal as a continuous spectrogram. The decisive turn came in 2021 with **SoundStream** and then EnCodec: a learned encoder-quantizer-decoder that compressed the waveform into a short sequence of *discrete tokens*. Once audio was tokens, the entire transformer toolkit from language modeling transferred wholesale — and the field accelerated. VALL-E reframed TTS as token language modeling in 2023; MusicGen did the same for music the same year; Suno shipped full songs with vocals in 2024; Moshi made real-time full-duplex speech an open capability that same year. The pace from 2021 onward is not a coincidence — it is what happens when a hard modality suddenly inherits a decade of mature machinery.

![A timeline of audio generation milestones from 2016 WaveNet through 2021 neural codecs to the 2026 real-time and long-form frontier](/imgs/blogs/the-2026-audio-model-landscape-3.png)

The shape of that timeline carries the single most useful prediction in this post: the gains that matter now arrive at the *codec* and *data-and-tuning* layers, not at the backbone, because the backbone question (AR transformer vs diffusion) was effectively settled once tokens existed and both families proved out. Keep that in mind as we move from the map to the axes — it is the throughline that tells you where to look when the next model drops.

#### Worked example: routing one product's three jobs

Take the colleague's product concretely. Job one is a product voice that reads notifications aloud — short utterances, must be fast and reliable, English plus five other languages. Job two is 20-to-30-second background music beds for marketing clips — instrumental, no vocals needed, but they must not sound stock-library generic. Job three is auto-generated SFX for short product demo videos — footsteps, UI clicks, a whoosh on a transition, synced to the cut. Three jobs, three picks: job one goes to a streaming TTS API (ElevenLabs for polish, or self-hosted **F5-TTS** if the audio must stay in the VPC); job two goes to **MusicGen-melody** or **Stable Audio Open** self-hosted (a one-time GPU cost, no per-clip fee, and you can fine-tune on a house style); job three goes to **MMAudio** conditioned on the rendered video frames so the SFX land on the right moment. One product, three sub-domains, three models — and notice that two of the three can be open and self-hosted, which materially changes the cost and privacy story. We will quantify that cost in the decision section.

## The axes that actually differentiate (define them operationally)

Here is where most "comparisons" go wrong: they rank models on *fidelity* as if it were a single number, when in practice you care about a vector of axes that trade against each other, and most of those axes have a precise operational definition that turns a vibe into a measurement. If you want a ranking you can defend in a design review, you measure each axis the same way for every model, on the same inputs, and you report the method alongside the number. Let me define the axes I use, each with the metric that grounds it. (For the deep version of this, read the dedicated post on [evaluating audio generation honestly](/blog/machine-learning/audio-generation/evaluating-audio-generation-honestly) — this is the working summary.)

**Fidelity** — how good does it *sound*, independent of whether it's the right content. The reference-free metric is **FAD (Fréchet Audio Distance)**: embed a set of generated clips and a set of reference clips with an audio embedding model, fit a Gaussian to each set, and compute the Fréchet distance between the two Gaussians. Lower is better. The single most important and most-ignored caveat: FAD is only as meaningful as its embedding. The original FAD used a VGGish embedding trained on AudioSet, which is fine for environmental sound but a poor proxy for music or speech quality; a FAD computed with a **CLAP** or a music-specific embedding correlates far better with human judgment for music. So "FAD 1.8" is meaningless on its own — you must say *FAD-with-which-embedding, on-which-reference-set, with-how-many-samples* (small sample counts bias FAD upward). I treat any FAD number without those three facts as decorative.

**Intelligibility** — for speech, can a listener (or a machine) recover the words. The operational metric is **WER (Word Error Rate)**: run the generated speech through a strong ASR model (Whisper-large, say), compare the transcript to the input text, and compute `(substitutions + insertions + deletions) / reference_words`. Lower is better; a good zero-shot TTS lands around 2-5% WER on clean English, which is roughly the WER of *real human speech* through the same ASR (so you are partly measuring the ASR's own errors — always report the human-speech baseline on the same pipeline). WER is the most honest single number in all of TTS because it is reproducible and hard to game.

![A five-layer stack diagram of the converged 2026 recipe: codec, generative core, conditioning, scale, preference tuning, decoder](/imgs/blogs/the-2026-audio-model-landscape-4.png)

**Speaker similarity** — for voice cloning, does the output sound like the *target* speaker. The operational metric is **SECS (Speaker Encoder Cosine Similarity)**, sometimes reported as SIM: pass the reference clip and the generated clip through a speaker-verification embedding model (a WavLM-based or ECAPA-TDNN x-vector network) and take the cosine similarity of the two embeddings. Higher is better; strong zero-shot cloners reach 0.6-0.7 on this scale, where same-speaker real recordings sit around 0.7-0.8 and different speakers near 0.2-0.3. The key honesty move is to *use the same verification model for every system you compare*, because the absolute scale shifts between embedding networks.

**Controllability** — how precisely can you steer it. This one resists a single number, so I score it on a rubric: can you condition on text (everyone), on a speaker prompt (zero-shot cloners), on a melody (MusicGen-melody, Suno's "cover"), on explicit prosody or emotion ([expressive TTS](/blog/machine-learning/audio-generation/prosody-emotion-and-expressive-speech)), on timing or structure (Stable Audio's duration conditioning, Suno's section tags), and can you *edit* an existing clip (inpainting, continuation). More handles, higher controllability — but more handles also means a harder prompt interface, so I weight it by how often the job actually needs each handle.

**Latency and streaming** — how fast, and can it start emitting before it finishes. The right metric depends on the use case. For batch generation, use **RTF (Real-Time Factor)** = `generation_time / audio_duration` — an RTF of 0.3 means you produced 10 seconds of audio in 3 seconds (faster than real time, good for streaming); an RTF of 3 means 10 seconds took 30 seconds (fine for offline). For interactive use, RTF is the wrong number — what matters is **TTFA (Time To First Audio)**, the latency from request to the first chunk of sound, because a streaming model with RTF 0.5 but 2 seconds of TTFA feels worse in a conversation than one with RTF 0.8 and 200 ms TTFA. Always state the device — "RTF 0.4 on an A100 80GB" and "RTF 0.4 on an M2 MacBook CPU" are wildly different claims — and always discard the first run (warm-up compiles kernels and loads weights; the steady-state RTF is what you ship).

**Max length, multilingual coverage, price, license, self-hostability** round out the vector. Max length is where music models diverge hardest — a 30-second cap versus a 4-minute song is a *capability* difference, not a quality one. Multilingual coverage ranges from English-only research models to 30-plus-language production APIs. Price is per-character or per-second for APIs, or amortized GPU cost for self-hosted. License determines whether you can ship the output commercially — and this is a minefield, because several "open" models carry non-commercial or research-only licenses on the *weights* even when the code is permissive. Self-hostability is binary and often decisive: can this run inside your VPC with no data leaving, yes or no. Score every candidate on all of these, the same way, and the "which is best" question dissolves into "best *for what*."

### The science: what FAD actually computes (and why the embedding is the whole game)

Because "fidelity" is the axis most people quote and least understand, it is worth deriving FAD precisely — this is the rigorous block this comparison stands on, and once you see the math you will never again read a bare "FAD 1.8" as meaningful. FAD borrows the structure of the Fréchet Inception Distance from image generation, adapted to audio. You take a set of generated clips and a set of real reference clips, push each clip through a fixed audio embedding network $\phi(\cdot)$ to get a feature vector, and then model each set of feature vectors as a multivariate Gaussian: the real set as $\mathcal{N}(\mu_r, \Sigma_r)$ and the generated set as $\mathcal{N}(\mu_g, \Sigma_g)$, where $\mu$ is the mean vector and $\Sigma$ the covariance matrix of the embeddings. FAD is then the squared 2-Wasserstein (Fréchet) distance between those two Gaussians, which has a closed form:

$$\text{FAD} = \lVert \mu_r - \mu_g \rVert_2^2 + \operatorname{Tr}\!\left(\Sigma_r + \Sigma_g - 2\left(\Sigma_r \Sigma_g\right)^{1/2}\right)$$

Read the two terms physically. The first term, $\lVert \mu_r - \mu_g \rVert_2^2$, measures how far the *average* generated clip sits from the average real clip in feature space — a systematic timbre or spectral bias shows up here. The second term, the trace expression with the matrix square root $(\Sigma_r \Sigma_g)^{1/2}$, measures whether the generated set has the same *spread and correlation structure* as the real set — a model that produces clips that are individually plausible but collectively too similar (mode collapse) or too varied (incoherent) is penalized here even if its mean is perfect. That second term is why FAD catches diversity failures that a per-clip metric would miss.

Now the punchline, and the reason the embedding is the whole game: *every quantity in that formula is computed on $\phi(x)$, never on the raw audio.* The distance is entirely a statement about the geometry of the embedding space. If $\phi$ is VGGish trained on AudioSet, the space is organized around *environmental sound categories*, so the FAD is sensitive to whether your clips sound like the right *class of sound* and relatively blind to musical or speech subtleties. If $\phi$ is a CLAP audio encoder or a music-specific network, the space is organized around features that track musical and timbral quality, so the FAD correlates with how good the music *sounds*. Same formula, same audio, wildly different number and meaning — purely because of $\phi$. This is not a footnote; it is the single most important fact about FAD. When a paper reports FAD without naming the embedding, the number is uninterpretable, and you should treat the comparison as incomplete. There is one more wrinkle the formula hides: estimating $\Sigma$ reliably needs enough samples relative to the embedding dimension, so a FAD on 20 clips is badly biased upward versus a FAD on 2,000 — always report the sample count alongside the embedding. With those two facts in hand (embedding and sample count), FAD becomes a genuinely useful fidelity proxy; without them it is decoration.

The same operational rigor applies to the human metric. **MOS** is an average of 1-5 ratings, and its sampling error scales like $\sigma/\sqrt{n}$ in the number of raters $n$ — a MOS from 8 raters has a standard error wide enough to swallow most of the differences people report as meaningful, which is why a paired **CMOS** (raters hear A and B back-to-back and rate the *difference*) is far more reliable than two independent MOS numbers compared across studies. The discipline is identical across every axis: name the estimator, name the sample size, name the model that produced the embedding or transcript, and only then compare. Do that and your ranking survives scrutiny; skip it and you are ranking on noise.

## A reproducible comparison harness you can actually run

The antidote to leaderboard-trust is to build your own leaderboard on *your* inputs. The harness below runs the same text (for TTS) or the same prompt (for music) through several open models, then scores each output on the axes we just defined. I am deliberately using only open, self-hostable models here so the whole thing is reproducible end to end — you cannot script ElevenLabs into a fair local benchmark without an API key and a budget, but you *can* anchor where the open field sits and then spot-check the closed models by ear.

Start with the TTS side. We synthesize one fixed sentence with two open cloners and score WER (intelligibility) and SECS (speaker similarity) against a reference clip. The running example throughout this series has been the sentence *"the quick brown fox jumps over the lazy dog"* — we keep it.

```python
# pip install TTS f5-tts transformers torchaudio jiwer speechbrain
import torch, torchaudio, jiwer
from transformers import pipeline

TEXT = "the quick brown fox jumps over the lazy dog"
REF_WAV = "reference_speaker.wav"   # 6-10 s of the target voice, 24 kHz mono

# --- ASR for WER (Whisper-large-v3) ---
asr = pipeline("automatic-speech-recognition",
               model="openai/whisper-large-v3",
               device=0 if torch.cuda.is_available() else -1)

def wer_of(wav_path: str, reference_text: str) -> float:
    hyp = asr(wav_path)["text"].strip().lower()
    # jiwer normalizes punctuation/casing for a fair word-error-rate
    return jiwer.wer(reference_text, hyp)

# --- Speaker similarity (SECS) with a verification embedding ---
from speechbrain.inference.speaker import EncoderClassifier
spk = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

def secs(wav_a: str, wav_b: str) -> float:
    ea = spk.encode_batch(torchaudio.load(wav_a)[0]).squeeze()
    eb = spk.encode_batch(torchaudio.load(wav_b)[0]).squeeze()
    return torch.nn.functional.cosine_similarity(ea, eb, dim=0).item()
```

Now generate with each model and time it for RTF. XTTS v2 (Coqui) and F5-TTS are the two open cloners worth comparing first.

```python
import time, soundfile as sf

def rtf_of(generate_fn, out_path):
    t0 = time.perf_counter()
    generate_fn(out_path)                       # writes the wav
    gen_time = time.perf_counter() - t0
    audio_dur = sf.info(out_path).duration
    return gen_time / audio_dur                 # RTF = gen / audio

# --- XTTS v2 ---
from TTS.api import TTS
xtts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
def gen_xtts(out):
    xtts.tts_to_file(text=TEXT, speaker_wav=REF_WAV, language="en", file_path=out)

# warm up once (compiles kernels, loads weights) then measure steady state
gen_xtts("warmup.wav")
rtf_xtts = rtf_of(gen_xtts, "xtts.wav")

print(f"XTTS v2  | WER {wer_of('xtts.wav', TEXT):.2%}"
      f" | SECS {secs(REF_WAV, 'xtts.wav'):.2f}"
      f" | RTF {rtf_xtts:.2f}")
# F5-TTS would slot in identically with its own generate fn.
```

That gives you a real, three-number row per TTS model — WER, SECS, RTF — on *your* reference voice and *your* hardware. The discipline that makes it honest: one warm-up run discarded, the same Whisper model and the same ECAPA verifier for every system, a fixed reference clip, and the device named in the report. Run it on five reference speakers and average; a single speaker is noisy.

The music side is structurally the same but the metric changes from WER/SECS to FAD, because there is no "correct transcript" for a 10-second lo-fi loop. We generate the same prompt with two open music models and score FAD against a small reference set of real music in the target style.

```python
# pip install audiocraft frechet-audio-distance stable-audio-tools
import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

PROMPT = "warm lo-fi hip hop beat, vinyl crackle, mellow piano, 80 bpm"

# --- MusicGen (open, codec language model) ---
mg = MusicGen.get_pretrained("facebook/musicgen-medium")  # 1.5B params
mg.set_generation_params(duration=10)                     # seconds
wav = mg.generate([PROMPT])                               # [B, C, T] at 32 kHz
audio_write("musicgen_lofi", wav[0].cpu(), mg.sample_rate,
            strategy="loudness")                          # writes musicgen_lofi.wav

# --- FAD with a CLAP embedding (not VGGish) for music ---
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance(model_name="clap",             # music-aware embedding
                           sample_rate=48000, use_pca=False)
score = fad.score("ref_real_lofi_dir/", "gen_lofi_dir/")  # dirs of wavs
print(f"MusicGen-medium | FAD-CLAP {score:.2f}"
      f" (on {len(...)} samples vs real lo-fi reference)")
```

Two things to internalize from the music snippet. First, I passed `model_name="clap"` to the FAD scorer *on purpose* — the default VGGish embedding would give you a number that barely correlates with how the loops actually sound. Second, FAD needs a *set* on both sides; scoring one generated clip against one reference clip is statistically meaningless. Generate 50-plus clips, collect 50-plus real references in the same style, and only then is the distance interpretable — and even then, report the sample count, because small sets inflate FAD. This is exactly the kind of methodological caveat the [honest-evaluation post](/blog/machine-learning/audio-generation/evaluating-audio-generation-honestly) was written to drill in.

#### Worked example: the harness output, and reading it correctly

Suppose you run the TTS harness on five English reference speakers, A100 80GB, and get roughly: XTTS v2 averages WER 4.1%, SECS 0.58, RTF 0.55; F5-TTS averages WER 2.6%, SECS 0.65, RTF 0.30; and your real-human baseline through the same Whisper model is WER 2.2%. How do you read this? F5-TTS wins on all three axes on this set — lower WER, higher similarity, faster — so for a self-hosted English cloner it is the pick. But notice the WER numbers are *close to the human baseline of 2.2%*, which means you are partly measuring Whisper's own errors, not the TTS — the intelligibility race is essentially won by both, and the real differentiator is the 0.07 SECS gap (audible as "sounds more like the target") and the 0.25 RTF gap (audible as "starts faster"). The lesson: when a metric saturates against its human baseline, stop ranking on it and move your decision weight to the axis that still has headroom. (These figures are illustrative of the *shape* of a real result, not exact published numbers; run the harness to get yours.)

#### Worked example: scoring two open music models on the same prompt

Now the music side, where the metric is FAD and the reading is subtler. You generate 64 lo-fi clips each from MusicGen-medium (1.5B params) and Stable Audio Open on the prompt above, collect 64 real lo-fi reference clips, and score FAD with a CLAP embedding on an A100. Say you get roughly: MusicGen-medium FAD-CLAP 1.9 at RTF ~2.5, and Stable Audio Open FAD-CLAP 2.3 at RTF ~1.2 — and as a control you also compute the *same prompt's* FAD with the default VGGish embedding and get 4.1 and 4.4, numbers that are both higher and *closer together*, ranking the two models almost the same. How do you read this? First, the VGGish control demonstrates the embedding caveat live: on VGGish the two models look nearly tied, while on CLAP MusicGen pulls clearly ahead — so if you had quoted VGGish FAD you would have made a worse decision. Second, MusicGen wins on fidelity (lower CLAP FAD) but Stable Audio is twice as fast (lower RTF) and generates longer coherent clips — so the pick depends on the job: MusicGen for the best 10-second loop where you'll also use melody conditioning, Stable Audio for longer-form texture where speed and length matter more than the last fraction of a FAD point. Third, with only 64 samples both FAD numbers carry meaningful estimation error — push to 500+ clips before you treat a 0.4 FAD gap as decisive. The shape of this result is real even though the exact numbers are illustrative; the harness is what turns "MusicGen feels better" into "MusicGen is 0.4 CLAP-FAD better at 2x the cost on my prompt set."

## The converged recipe (why the backbone is not the moat)

Here is the most important structural fact about the 2026 landscape, and the one that should most change how you evaluate a new model announcement: **almost every serious system is the same recipe.** Peel back the marketing and you find, with remarkable consistency, five layers. A **neural codec** turns the waveform into a short sequence of discrete tokens or a continuous latent — EnCodec, DAC, or a model-specific variant. A **generative core** models those tokens or that latent — either an autoregressive transformer (a token language model, as in VALL-E, MusicGen, Suno's likely architecture) or a diffusion/flow model (as in Stable Audio, AudioLDM, F5-TTS). A **conditioning path** injects control — a T5 or CLAP text encoder, a speaker embedding, a melody chroma, a duration token. Then **scale** — more data, more parameters, more compute. Then **preference tuning** — DPO or RLHF on human ratings, which is increasingly where the last mile of "it sounds good" actually comes from.

That convergence is not an accident; it is the same story that played out in language modeling and image generation. Once neural codecs (SoundStream in 2021, then EnCodec) turned a continuous waveform into a discrete token stream, *all of the transformer machinery from NLP transferred directly* — next-token prediction, in-context learning, scaling laws, instruction tuning, preference optimization. The codec was the unlock; everything after it was the field running the well-worn language-model and diffusion playbooks on a new modality. Which means the *backbone* — "is it an AR transformer or a diffusion model?" — is largely a solved, commoditized choice. Both work. The interesting differences are elsewhere.

So where does a real quality lead come from, if not the architecture? Three places, and they compound.

![A branching graph showing a shared backbone splitting into data, codec quality and preference tuning, which converge into perceived quality and a market lead](/imgs/blogs/the-2026-audio-model-landscape-5.png)

**Data.** The single biggest differentiator, and the least discussed because it is the least defensible legally. ElevenLabs and Suno's leads are, to a large degree, *data* leads — more hours, cleaner labels, broader coverage of speakers/styles/languages, and (for the closed players) a willingness to train on data that open projects cannot touch for licensing reasons. An open model trained on 50,000 hours of permissively licensed speech will lose to a closed model trained on 1,000,000 hours of mixed-provenance speech, holding architecture constant, simply because it has seen more of the world. This is why the open-vs-closed gap is *narrowing but persistent*: open data is growing, but the closed players' data advantage is real and largely about scale and rights, not cleverness.

**Codec quality.** The codec sets the ceiling. A generative model can never sound better than what its codec can reconstruct — if the codec smears sibilants or mangles cymbals at its operating bitrate, every sample inherits that flaw. So a team that ships a better codec (DAC's improvements over EnCodec, Mimi's speech-tuned design for Moshi) raises the ceiling for everything above it. This is a genuine, defensible technical moat, and it is why the [codec post](/blog/machine-learning/audio-generation/encodec-dac-and-the-modern-codec) sits so central in this series.

**Preference tuning.** The last mile. Two models with the same backbone, similar data, and similar codecs can still differ audibly because one has been preference-tuned on millions of human "this one sounds better" judgments and the other has not. Suno's musicality, ElevenLabs' naturalness — a meaningful slice of that is DPO/RLHF on a scale of human feedback that open projects struggle to gather. This is the layer that is hardest to see in a paper and easiest to hear in a demo.

The practical upshot for evaluating any new release: when a vendor announces a model, ask *what is the data story, what codec is underneath, and was it preference-tuned* — not *what backbone did they use*. The backbone is table stakes. If a new model claims a leap purely from a clever architecture and says nothing about data, codec, or tuning, be skeptical; the field's history says the gains live in those three.

This reframes the open-vs-closed debate in a way that is more useful than the usual "closed is better." The closed leaders are not better because they have a secret architecture; they are ahead on the three compounding levers — more data (often more than open projects can legally touch), heavier preference tuning on more human ratings, and the polish of a closed-loop product (low latency, high uptime, a tuned default voice or musical style). The open challengers give up some of each in exchange for the one thing closed cannot offer: you can self-host them, inspect them, fine-tune them on your own data, and keep every byte inside your infrastructure. Seen this way, the gap is *narrowing on the levers that scale with the open ecosystem* (data is slowly opening, community tuning is improving) and *persistent on the levers that need a product organization* (millions of human preference judgments, an SRE team keeping a low-latency API up). That is why open keeps closing the quality gap while closed keeps a polish-and-breadth lead — and why the right choice is set by whether self-hosting is a hard requirement, not by a global "which is best."

![A before-and-after comparison contrasting a closed leader's data scale and preference tuning against an open challenger's lighter tuning but self-hostability](/imgs/blogs/the-2026-audio-model-landscape-6.png)

## TTS / voice: the leaders, ranked and qualified

Speech is the most mature of the three sub-domains and the one with the clearest leader. **ElevenLabs** is the closed quality leader — top-tier MOS, the lowest perceptible artifact rate, 30-plus languages, instant cloning, and a streaming API with low TTFA that makes it the default for production voice agents. It is API-only and paid; you cannot self-host it, and your audio passes through their servers. That is the trade: best-in-class polish and breadth in exchange for a per-character fee and no on-prem option. For a customer-facing product where voice quality is the product, it is hard to beat and usually the right call.

The **open field** has closed most of the *quality* gap and none of the *operational* one. **XTTS v2** (Coqui lineage) is the workhorse — 17 languages, solid zero-shot cloning from a few seconds, an RTF around 0.5 on a good GPU, and a permissive-enough license for most uses (read it carefully; the Coqui public model license is non-commercial in its strictest reading, which has pushed many teams toward alternatives). **F5-TTS** is the current open favorite for English and Chinese: a flow-matching, alignment-free design that hits low WER and high speaker similarity, runs fast (RTF often below 0.4), and ships under a permissive license — for a self-hosted English cloner it is my default recommendation right now. **Fish-Speech** is another strong open multilingual option with good latency. And **Kokoro** deserves special mention: it is *tiny* (on the order of 82M parameters), runs comfortably on CPU and even in a browser via ONNX, sounds genuinely good for its size, and is the model I reach for when the constraint is "must run on-device with no GPU." The research frontier on raw naturalness is **NaturalSpeech 3**, which factorizes speech into disentangled content/prosody/timbre/detail codecs and hits state-of-the-art similarity — but it is a paper, not a downloadable product, so treat it as a signpost for where the field is heading rather than something you ship this quarter.

For the full architectural breakdown of these open systems — why F5-TTS's alignment-free flow design works, how XTTS does in-context cloning — see the dedicated post on [zero-shot voice cloning and the TTS frontier](/blog/machine-learning/audio-generation/zero-shot-voice-cloning-and-the-tts-frontier). Here I want to land the *decision*: closed (ElevenLabs) for max polish and breadth with no self-hosting need; F5-TTS for a fast permissive English/Chinese self-hosted cloner; XTTS v2 for broad multilingual self-hosting; Kokoro for on-device and CPU.

There is a fourth category that does not fit the TTS frame at all, and it is the most exciting frontier in speech: **speech-native, full-duplex** models. **Moshi** (from Kyutai) and the **GPT-4o voice mode** are not "text-to-speech" — they are end-to-end spoken-dialogue models that listen and speak *simultaneously*, with no separate ASR and TTS, achieving conversational latency around 200 ms and the ability to be interrupted mid-sentence. Moshi is open and built on the Mimi codec (1.1 kbps, 12.5 Hz frame rate — extremely aggressive compression tuned for speech, which is what makes the real-time loop tractable). If your job is a real-time voice *agent* rather than a TTS endpoint, this is a different and better tool, and the dedicated post on [real-time streaming and full-duplex speech](/blog/machine-learning/audio-generation/real-time-streaming-and-full-duplex-speech) covers why the architecture matters.

The reason full-duplex deserves its own category, rather than being filed as "fast TTS," is architectural and worth making explicit because it changes the buying decision. A conventional voice agent is a *pipeline*: speech-to-text (ASR) feeds a language model, whose text output feeds text-to-speech, and each stage adds latency and, worse, each stage is *turn-based* — the system must wait for you to stop talking before it starts processing, and it cannot react while you speak. A full-duplex model collapses that pipeline into one model that processes incoming audio and emits outgoing audio on the same clock, so it can start formulating a response before you finish, back-channel ("mm-hm") while you talk, and stop instantly when you interrupt. That is not a latency improvement; it is a different *interaction model*, and it is why a stitched ASR-LLM-TTS stack always feels a beat behind no matter how fast each component is. The trade is that full-duplex models are harder to control precisely (you cannot easily insert a specific SSML pause the way you can in a TTS API) and the open ones are newer and rougher. So the rule is: if the job is genuinely conversational — interruption, overlap, real-time reaction — reach for full-duplex (GPT-4o voice managed, Moshi self-hosted); if the job is *reading text aloud* with no live interlocutor (audiobooks, IVR prompts, video narration), a streaming TTS is simpler, more controllable, and entirely sufficient. Misclassifying these two is the most common architecture mistake in voice products.

#### Worked example: a multilingual dubbing pipeline, open vs closed

You need to dub 200 hours of training videos into eight languages, preserving each original presenter's voice. Closed path: ElevenLabs' dubbing — clone each presenter once, translate, synthesize; quality is excellent, but at a per-character rate over 200 hours of speech across eight languages the bill runs into the thousands of dollars and every second of your training content transits a third-party API. Open path: XTTS v2 self-hosted (17 languages covers your eight), clone each presenter from a 10-second sample, run the translated scripts through it on a single A100; the GPU cost is roughly \$2/hr and the whole batch might take a day or two of compute — call it \$50-100 of GPU time, with nothing leaving your infrastructure. The quality gap is real but narrowing — XTTS will be a notch below ElevenLabs on naturalness and a notch above on "it's in our VPC." For internal training content, the open path's cost and privacy story usually wins; for a flagship customer-facing product, the closed polish often justifies the bill. That is the trade, quantified.

## Music: open loops, closed songs

Music is where the open-closed gap is widest, and it is a *capability* gap, not just a quality one. The closed leaders, **Suno** and **Udio**, generate full songs — verse, chorus, bridge, vocals that sing your supplied lyrics, multiple minutes of coherent structure — from a text box. No open model does this at the same level yet. The open field is excellent at *instrumental loops and short-form* and only beginning to attempt full songs with vocals.

On the open side, **MusicGen** (Meta) is the workhorse: a single-stage codec language model over EnCodec tokens, available at 300M / 1.5B / 3.3B parameters, with a **melody-conditioned** variant that lets you hum or feed a chroma to steer the tune. It generates up to ~30 seconds in one pass (longer via windowed continuation, with some drift), runs at an RTF around 1-3 depending on size and device, and ships under a permissive license with CC-licensed training data — which matters enormously if you need to *ship* the output commercially. **Stable Audio Open** is the open latent-diffusion alternative: a diffusion model on a codec latent with timing/duration conditioning, strong on textured instrumental and sound-design material, also openly licensed. **YuE** is the most interesting recent open development — the first credible open attempt at *full songs with vocals and lyrics*, closing toward Suno's capability frontier, though still behind on polish and length. The architectural deep-dives live in [music generation: MusicLM and MusicGen](/blog/machine-learning/audio-generation/music-generation-musiclm-and-musicgen) and [latent diffusion for music: Stable Audio](/blog/machine-learning/audio-generation/latent-diffusion-for-music-stable-audio).

On the closed side, [Suno and Udio](/blog/machine-learning/audio-generation/suno-udio-and-the-commercial-music-frontier) define the frontier and I will not re-derive their inferred recipe here — the short version is a high-compression codec for length, a lyrics-and-phoneme-aligned conditioning path for the singing, section/structure conditioning, and heavy preference tuning. The honest caveat on standings: both are closed, neither has published a full technical report, and the legal picture is unsettled (the RIAA lawsuits over training data are ongoing as of mid-2026), so treat their *capability* as well-established and their *recipe* and *legal footing* as inferred and in flux.

The decision for music is unusually clean. Need a full song with vocals from lyrics? You go closed — Suno or Udio — because nothing open matches it yet, and you accept the per-song cost, the SaaS dependency, and the open legal questions around the output's provenance. Need instrumental beds, loops, sound-design material, or anything you must self-host and ship commercially with a clear license? You go open — MusicGen for tune-controllable loops, Stable Audio Open for texture and longer-form instrumental — and you accept that it will not write you a chorus with sung lyrics. Want to experiment with open full-song generation? Try YuE, with the expectation that it is the frontier of *open*, not of the field.

It is worth being precise about *why* the open-closed gap is a capability gap in music specifically, because it illuminates the whole landscape. Two things make full-song-with-vocals hard, and both are where the closed players have invested. The first is **length and structure**: a token-LM like MusicGen runs out of coherent context after about 30 seconds because the token rate (EnCodec frames times codebooks) makes a 4-minute song an enormous sequence, and the model loses the thread — the groove drifts, the key wanders. Suno's length advantage almost certainly comes from a much higher-compression codec (fewer tokens per second of audio) plus structure conditioning (explicit verse/chorus/bridge section tags), which is why it holds a song together where an open token-LM falls apart. The second is **sung vocals**, which is genuinely harder than text-to-speech: in speech the timing of phonemes is relatively free, but in singing the phonemes must align to a *melody* — each syllable lands on a specific note for a specific duration — and the vocal must mix coherently with the backing track rather than sit on top of silence. That lyrics-to-melody-aligned conditioning path is the hardest engineering in the whole sub-domain, and it is exactly what open models have not yet matched. So when you read "Suno is better than MusicGen," translate it: Suno solved length-via-codec-and-structure and sung-vocals-via-melody-alignment, two specific capabilities, not a vague "higher quality." That translation tells you precisely when MusicGen is *fine* — short instrumental loops, where neither hard part applies — and when only the closed players will do.

## Sound, SFX, and the video-to-audio link

The third sub-domain is the least mature commercially and the most interesting technically right now, because it includes the **video-to-audio** problem, which forces the model to solve temporal synchronization that text-conditioned models never face.

For **text-to-sound** — "a dog barking in a large room," "rain on a tin roof," "a sci-fi door whoosh" — the open workhorses are **AudioGen** (Meta, a codec language model like MusicGen but trained on sound events) and **AudioLDM 2** (a latent diffusion model spanning speech, sound, and music). Both are openly available, both generate a few seconds of conditioned sound, and both are strong enough for foley and ambience in production pipelines. The [text-to-audio and sound effects post](/blog/machine-learning/audio-generation/text-to-audio-and-sound-effects) goes deep on these.

The frontier is **video-to-audio**: given silent video, generate synchronized sound. **MMAudio** is the leading open option — it conditions on video frames (and optionally text) and produces sound that lands on the visual events, which is the hard part; a clink that arrives two frames late breaks the illusion entirely. Meta's **Movie Gen Audio** is the closed counterpart, generating synced sound and music for generated or real video at film quality, integrated into their Movie Gen video stack. This is also the bridge to the video series — the joint audio-video generation story is covered in [audio and joint AV generation](/blog/machine-learning/video-generation/audio-and-joint-av-generation), and the whole reason video-to-audio is hard (and why sync, not fidelity, is the constraint) connects directly to the temporal-modeling machinery there.

The operational note for SFX: do not score these on WER (no transcript) and be careful scoring them on FAD with the wrong embedding. For environmental sound, the original VGGish/AudioSet FAD is actually *reasonable* (it was trained on exactly this kind of data), which is a nice inversion of the music case where VGGish FAD misleads. The metric that matters most for video-to-audio has no clean automatic proxy — it is *temporal alignment*, "does the sound land on the event," which today still needs human eyes-and-ears evaluation or a custom onset-alignment score. When someone shows you a video-to-audio FAD number with no sync metric, they have measured the easy axis and skipped the hard one.

Since the gate for this sub-domain is sync rather than fidelity, here is the metric that actually matters, sketched: detect onsets in the generated audio, detect visual events in the video, and measure how well the two line up in time. A crude but useful onset-alignment score is the median absolute time offset between each audio onset and its nearest visual event.

```python
# pip install librosa numpy
import librosa, numpy as np

def onset_alignment(audio_wav, visual_event_times, sr=16000):
    y, _ = librosa.load(audio_wav, sr=sr)
    # audio onsets in seconds (energy/spectral-flux peaks)
    audio_onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    if len(audio_onsets) == 0 or len(visual_event_times) == 0:
        return float("inf")
    # for each visual event, the nearest audio onset; report the median gap
    gaps = [min(abs(a - v) for a in audio_onsets) for v in visual_event_times]
    return float(np.median(gaps))   # seconds; < 0.05 s is good, > 0.1 s is audible lag

# visual_event_times come from a detector on the video (e.g. motion-energy peaks)
score = onset_alignment("mmaudio_out.wav", visual_event_times=[0.42, 1.10, 1.95])
print(f"median onset offset: {score*1000:.0f} ms")   # ship target: under ~50 ms
```

The deeper reason video-to-audio is the most technically interesting corner of the whole field is that it forces a *cross-modal* alignment that none of the other sub-domains face. A text-to-sound model only has to make a plausible bark; a video-to-sound model has to make the bark *start on the frame where the dog's mouth opens* and *stop when it closes*, which means the model must read fine-grained motion cues from the video and bind them to acoustic onsets with sub-100-millisecond precision — humans are exquisitely sensitive to audio-visual lag, and a 50-millisecond offset is already perceptible as "off." MMAudio tackles this by conditioning the audio generator on per-frame video features so the generation is locked to the visual timeline; Movie Gen Audio does it inside a larger video-and-audio joint stack. This is why the sub-domain's real benchmark is a sync score, not a fidelity score: a video-to-audio model that produces gorgeous foley two frames late has failed at the only thing that distinguishes it from a text-to-sound model. If you are building here, instrument *onset alignment* first and fidelity second — and budget for human review, because no automatic proxy yet captures "it feels in sync" the way a person watching the clip does. This is also precisely the bridge to the video series, where the joint audio-video modeling machinery and the temporal-conditioning tricks are derived in full.

## The codecs underneath: the quietest leverage point

Every model above sits on a codec, and the codec is the quietest, highest-leverage choice in the whole stack — because it sets both the **quality ceiling** (the model can't beat what the codec reconstructs) and the **token budget** (the codec's frame rate times the number of RVQ codebooks determines how many tokens per second the generative model must produce, which sets the speed and the max length).

![A three-row matrix comparing EnCodec, DAC and Mimi codecs across bitrate, fidelity, token rate and which models use them](/imgs/blogs/the-2026-audio-model-landscape-7.png)

**EnCodec** (Meta) is the workhorse general-purpose codec — operates from roughly 1.5 to 24 kbps, a 75 Hz frame rate, multiple RVQ codebooks, and it is what MusicGen and AudioGen tokenize with. **DAC (Descript Audio Codec)** improved on EnCodec's fidelity at a given bitrate — better high-frequency reconstruction, fewer artifacts at low rates — and is widely used under modern TTS systems. **Mimi** (Kyutai, for Moshi) is the specialist: a *speech-tuned* codec running at ~1.1 kbps with an extraordinarily low 12.5 Hz frame rate, which is the key to Moshi's real-time loop — fewer frames per second means fewer tokens to generate per second of audio, which is what makes full-duplex conversation tractable. That low frame rate is a *design choice for speed*, traded against the fidelity you'd want for music (which is why you would not use Mimi for a music model).

There is a quantitative relationship here worth stating because it explains so much downstream behavior. The number of tokens a generative model must produce per second of audio is the codec's *frame rate* times the *number of residual codebooks* it uses. EnCodec at 75 Hz with, say, 4 codebooks is 300 tokens per second; Mimi at 12.5 Hz with 8 codebooks is 100 tokens per second despite carrying speech well. Halve the token rate and you roughly halve the sequence length a language model must model for a given duration of audio — which directly improves RTF, extends the coherent max length, and lowers serving cost, all without touching the generative architecture. This is why codec progress is such high leverage: a better codec at a lower frame rate is a free win for *speed and length* across every model that adopts it, and it is the lever that quietly enabled real-time speech (Mimi) and will likely enable open long-form music next. When you read that a new music model generates four-minute songs where the last one capped at thirty seconds, look first at the codec — the odds are the breakthrough lives there, not in the transformer on top.

The lesson for builders: if you are assembling your own stack (the subject of the [capstone post](/blog/machine-learning/audio-generation/building-an-audio-generation-stack)), pick the codec to match the job — DAC for high-fidelity general audio and TTS, EnCodec for the MusicGen/AudioGen ecosystem, Mimi or a similarly aggressive low-frame-rate codec when real-time speech latency is the constraint. The codec choice ripples through everything: it sets your token rate (hence your RTF and your max length), your fidelity ceiling, and your compatibility with off-the-shelf generative models. It is the foundation, and foundations are worth getting right first.

## The benchmarks and arenas — and how they lie

You will be tempted to settle arguments with a leaderboard. Resist the urge to trust them blindly, because every audio benchmark lies in a specific, knowable way, and a buyer who knows the lie can still use the benchmark correctly.

**TTS Arena** and similar human-preference arenas (the audio analogue of Chatbot Arena) collect blind A/B votes between TTS systems and produce an Elo-style ranking. These are genuinely useful — human preference is the gold standard for "does it sound good" — but they have two failure modes. First, **they are not reproducible**: the ranking shifts with the voter pool, the prompt set, and the time of day, and you cannot re-run last month's standings. Second, **they conflate axes**: a model can win on Arena because its default voice is pleasant, not because its *cloning* is accurate, and the Arena doesn't separate those. Read Arena rankings as a directional signal of overall pleasantness, not as a per-axis measurement.

**FAD** lies through its embedding, as covered above — a FAD number is only interpretable with its embedding, reference set, and sample count attached. **CLAP-score** (text-audio cosine similarity via a CLAP model) measures whether the audio *matches the prompt*, which is a different thing from whether it sounds good — a model can score high CLAP by being on-topic while sounding mediocre. **MOS** (Mean Opinion Score) is the classic human 1-5 rating, and it lies through rater count and calibration: a MOS from 10 raters has huge error bars, and MOS scales are not comparable across studies (one paper's "4.1" is another's "3.8" for the same audio) — only *within-study* MOS comparisons (CMOS, a direct A/B) are trustworthy. **WER** is the most honest because it is reproducible and machine-computed, but remember it is bounded below by the ASR's own error and by the human baseline.

There is a meta-lie worth naming too: **vendor benchmarks**. Almost every model announcement ships with a table showing the new model winning, and almost every such table is constructed — consciously or not — to favor the home team, through the choice of eval set, the choice of baselines, the choice of metric, and the choice of which axis to report. This is not usually fraud; it is the natural result of a team measuring what they optimized for on the data they optimized on. The defense is simple and non-negotiable: never let a vendor's own number be the last word. If model X claims to beat model Y, reproduce the comparison yourself on a *neutral* set with the *same* ASR/embedding/verifier for both, and you will frequently find the gap is smaller, or reversed, or specific to one language or one style. The harness in this post exists precisely to give you that independent check; running it on even ten of your own examples is worth more than reading ten vendor tables.

The synthesis: no single benchmark settles a model choice. Triangulate — a low WER (reproducible intelligibility) *plus* a strong within-study SECS (similarity) *plus* a decent Arena standing (human pleasantness) *plus* your own ears on *your* content is a defensible basis for a decision. Any one of them alone is a way to be confidently wrong. The [honest-evaluation post](/blog/machine-learning/audio-generation/evaluating-audio-generation-honestly) exists precisely because the audio-eval situation is this messy, and a buyer who internalizes the messiness makes better calls than one chasing a single number.

## The decision guide: which model for which job

Now the payoff — the part your colleague actually needed. Before the job-to-model map, here is the master comparison table — every model discussed, across domain, access, fidelity, the intelligibility/similarity number that applies, latency, max length, and license. Treat every cell as approximate and dated to mid-2026; the point is the *relative* shape, not a spec sheet. Where a number does not apply (WER is meaningless for instrumental music; SECS is meaningless for SFX), the cell says so.

| Model | Domain | Access | Fidelity | WER / Sim. | Latency | Max length | License |
|---|---|---|---|---|---|---|---|
| ElevenLabs | TTS / voice | API only | top MOS | very low WER · top SIM | ~300 ms TTFA | long (chunked) | paid SaaS |
| F5-TTS | TTS / voice | open weights | near-top | WER ~2-3% · SIM ~0.65 | RTF ~0.3 | sentence/para | permissive |
| XTTS v2 | TTS / voice | open weights | strong | WER ~3-5% · SIM ~0.58 | RTF ~0.5 | sentence/para | non-comm.-ish |
| Kokoro | TTS / voice | open · 82M | good | WER ~4-6% | CPU real-time | sentence | permissive |
| NaturalSpeech 3 | TTS / voice | paper only | SOTA SIM | WER ~1.8% · top SIM | n/a | sentence | closed |
| Moshi | duplex speech | open weights | good (speech) | conversational | ~200 ms duplex | continuous | permissive |
| GPT-4o voice | duplex speech | API only | top (speech) | conversational | ~300 ms duplex | continuous | paid SaaS |
| Suno / Udio | music · songs | API only | full songs | n/a (vocals) | ~30 s / song | minutes | paid SaaS |
| MusicGen | music · loops | open weights | good 30 s | n/a (instrum.) | RTF ~1-3 | ~30 s / pass | MIT + CC data |
| Stable Audio Open | music · texture | open weights | good | n/a (instrum.) | RTF ~1-2 | longer-form | open |
| YuE | music · songs | open weights | open frontier | n/a (vocals) | slow | song-length | open |
| AudioGen | sound / SFX | open weights | good foley | n/a (SFX) | few sec | ~10 s | research |
| AudioLDM 2 | sound / SFX | open weights | good | n/a (SFX) | few sec | ~10 s | open |
| MMAudio | video-to-audio | open weights | synced SFX | n/a · sync metric | few sec | clip-length | research |
| Movie Gen Audio | video-to-audio | closed | film-quality | n/a · sync metric | n/a | clip-length | closed |

Read this table as a *starting hypothesis*, not a verdict — every "approximate" number here is exactly the kind you should re-measure with the harness on your own inputs before you commit budget. With that hypothesis in hand, here is the job-to-model map, with the open backup and the constraint that drives each pick. This is opinionated and dated to mid-2026; re-run the harness before you commit, because standings move monthly.

![A six-row matrix mapping job scenarios to a top model pick, an open backup, and the key constraint for each](/imgs/blogs/the-2026-audio-model-landscape-8.png)

**Production TTS (a product voice, English-heavy):** top pick ElevenLabs for polish and a reliable low-latency streaming API; open backup F5-TTS self-hosted when the audio must stay in your VPC or you want zero per-character cost. The driving constraint is MOS-plus-uptime — a product voice that occasionally rate-limits or sounds robotic costs you more than the API fee saves.

**Multilingual dubbing (preserve voices across many languages):** top pick ElevenLabs (30-plus languages, excellent cloning); open backup XTTS v2 (17 languages, self-hostable, cheap at scale). The constraint is language coverage *and* clone fidelity together — and as the worked example showed, the open path wins decisively on cost and privacy for internal content.

**A song with vocals from lyrics:** top pick Suno or Udio — nothing open matches full-song-with-vocals yet; open frontier YuE if you must self-host and will accept a quality and length gap. The constraint is the *capability* of sung vocals over song-length structure, which is a hard wall, not a quality dial.

**SFX for video (synced sound for clips):** top pick MMAudio (open, video-conditioned, lands on the cut); backup AudioGen for text-only foley when you don't need sync. The constraint is temporal synchronization — fidelity is secondary to landing on the frame.

**Real-time voice agent (conversational, interruptible):** top pick the GPT-4o voice mode for a managed low-latency duplex experience; open backup Moshi when you need to self-host the full-duplex loop. The constraint is end-to-end *duplex latency* (TTFA plus the ability to be interrupted), which a TTS-plus-ASR pipeline cannot match — you need a speech-native model.

**On-device / no GPU (edge, browser, privacy-max):** top pick Kokoro (tiny, runs on CPU, sounds good for its size); backup Piper for the lowest-resource embedded targets. The constraint is model size and CPU RTF — quality is good-enough rather than best, by design. For the broader on-device story, the [edge-AI series on speech at the edge](/blog/machine-learning/edge-ai/multimodal-and-speech-at-the-edge) covers quantization and runtime trade-offs that compound here.

Cross-cutting the table: **open vs closed** is set by whether self-hosting is a hard requirement (data residency, cost at scale, no-API-dependency) — if it is, you accept a quality notch and pick the best open option; if it isn't, the closed leaders usually win on polish. And **price** scales differently — APIs are pay-per-use (great at low volume, painful at high volume), while self-hosted is a fixed GPU cost (painful at low volume, great at high volume). The crossover volume is the number to compute for your specific job, and it is a two-line calculation worth doing before you commit:

```python
# Break-even between a pay-per-use TTS API and a self-hosted GPU model.
api_price_per_min = 0.30        # USD per audio-minute (illustrative API rate)
gpu_cost_per_hour = 2.00        # USD/hr for the GPU you'd rent (e.g. an A100)
model_rtf = 0.4                 # self-hosted real-time factor (gen/audio time)

# self-hosted cost per audio-minute = GPU cost/min * compute-minutes-per-audio-min
gpu_cost_per_min = gpu_cost_per_hour / 60.0
self_host_per_audio_min = gpu_cost_per_min * model_rtf
print(f"API: {api_price_per_min:.3f}/min  vs  "
      f"self-host: {self_host_per_audio_min:.4f}/min  (USD)")
# Below the crossover monthly volume, the API is cheaper (no idle GPU);
# above it, self-hosting wins. Add engineering + ops time to the self-host side.
```

Run that with your real rates and you will usually find the API is cheaper until you cross into tens of thousands of audio-minutes a month, after which the fixed GPU cost dominates — but the self-hosted side must also carry the engineering and on-call cost of running the stack, which the napkin math omits and which often pushes the real crossover higher than the pure-compute number suggests.

## When to reach for each (and when not to)

A decision guide is incomplete without the *negative* recommendations — the places where the obvious choice is wrong. These are the calls I have seen teams get backwards.

**Don't reach for a closed music API to make instrumental loops you'll ship commercially.** Suno is built for songs; for a 15-second instrumental bed that must have clean commercial rights, self-hosted MusicGen (CC-licensed training data, permissive weights) is cheaper, controllable, and legally cleaner. Using Suno here buys you nothing and adds a SaaS dependency and murkier output provenance.

**Don't build a real-time voice agent out of a streaming TTS plus a separate ASR.** It feels like the modular, sensible choice, but the round-trip latency and the inability to handle interruption (barge-in) make it feel laggy and rigid next to a speech-native duplex model. If the job is *conversation*, reach for Moshi or a duplex API, not a TTS endpoint stitched to a recognizer.

**Don't self-host a giant model when a tiny one clears the bar.** If your TTS job is reading short notifications, Kokoro at 82M parameters on a CPU may be entirely sufficient, and you have just saved a GPU and a serving stack. The reflex to reach for the biggest, best-MOS model is often over-engineering — match the model to the quality the job actually needs, not the quality the leaderboard celebrates.

**Don't trust a single benchmark number to settle a model choice.** As the benchmarks section labored, every number lies in its own way. A model that wins one Arena snapshot may lose on *your* content, your languages, your voices. Run the harness, listen to the output on your actual material, triangulate across reproducible (WER, within-study SECS) and human (Arena, CMOS) signals before you commit budget.

**Don't ignore the license on "open" weights.** Several models that are "open" in the sense of downloadable carry non-commercial or research-only weight licenses even when the code is permissive. Read the *weight* license, not just the GitHub LICENSE file, before you ship a product on it — this has bitten teams who assumed "open" meant "commercially usable."

## Case studies: real numbers from the literature

A few concrete, citable anchors so the standings above are grounded in published results rather than assertion. I mark each with its source and flag where I am giving an approximate figure.

**EnCodec's bitrate-quality trade.** Défossez et al. (2022) showed EnCodec achieving high-fidelity 24 kHz reconstruction across a 1.5-to-24 kbps range, with the rate-distortion curve bending exactly as RVQ theory predicts — each additional residual codebook adds bits and reduces distortion with diminishing returns. The practical takeaway that propagates up the stack: at the low end of that range you get a token stream short enough for a language model to handle, at the cost of fidelity the generative model can never recover.

**DAC's improvement.** The Descript Audio Codec (Kumar et al., 2023) reported better reconstruction quality than EnCodec at matched bitrates, particularly in the high frequencies where EnCodec tended to smear — which is why DAC became a popular foundation under modern TTS. The lesson: a codec improvement is a free quality gain for every model above it, no change to the generative architecture required.

**MusicGen's sizes and melody conditioning.** Copet et al. (2023) released MusicGen at 300M, 1.5B, and 3.3B parameters, with a melody-conditioned variant; the larger models score better on FAD and human preference, and melody conditioning lets you steer the tune with a reference chroma. Generation is capped near 30 seconds per pass at 32 kHz. The takeaway: open music generation is genuinely usable for short-form, and controllable via melody, with a clean license.

**Moshi's full-duplex latency.** Kyutai's Moshi (Défossez et al., 2024) reported conversational latency on the order of 200 ms with full-duplex operation, enabled by the Mimi codec's aggressive ~1.1 kbps, 12.5 Hz design. The takeaway: real-time conversational speech is now an *open*, self-hostable capability, not just a closed-API one — a meaningful shift since 2023.

**Stable Audio's length.** Evans et al. (2024) demonstrated variable-length generation up to and beyond the typical 30-second cap of token-LM music models, via latent diffusion with timing conditioning, producing longer coherent instrumental and sound-design audio. The takeaway: latent diffusion's advantage over token-LMs for music is most visible on the *length* axis.

Where I have given a round number above (latency "~200 ms," WER "~2-5%," SECS "0.6-0.7"), treat it as an order-of-magnitude anchor from the literature, not a precise spec — the exact figure depends on the eval set, the device, and the embedding/ASR used, which is exactly why the harness exists: to get *your* numbers on *your* inputs.

## The trajectory, and the open questions

Three trends define where this is going, and a buyer who sees them coming makes more durable choices.

**Real-time, speech-native dialogue is eating TTS.** The boundary between "synthesize this text" and "have a conversation" is dissolving into single full-duplex models (Moshi, GPT-4o voice, and the wave of followers). Within a year or two, the default for any *interactive* voice product will be a speech-native model, not a TTS endpoint — TTS will remain the right tool for *batch* narration (audiobooks, dubbing, notifications) where there is no live interlocutor. Build interactive products on the duplex stack now.

**Open full-song-with-vocals is coming.** YuE is the first credible open attempt; the trajectory of every other sub-domain (TTS, instrumental music, SFX) says open will close most of the gap to Suno/Udio within a couple of years, gated mostly by *data and tuning*, not architecture — exactly the differentiation pattern this post has hammered. The legal fight over training data is the wildcard that could slow or reshape this; the technical path is fairly clear.

**The codec arms race continues.** Lower bitrate, higher fidelity, lower frame rate — every improvement raises the ceiling and lowers the cost for everything above it, and it is the most underrated place to watch. Mimi's 12.5 Hz frame rate for real-time speech is a preview; expect the same aggressive compression to spread, unlocking longer-form and lower-latency generation across all three sub-domains.

The open questions are real and worth holding honestly. *Data provenance and copyright* — the Suno/Udio lawsuits could materially reshape what closed models are allowed to train on, and by extension the size of the open-closed gap. *Watermarking and detection* — as voice cloning and song generation get good enough to fool people, the [safety stack](/blog/machine-learning/audio-generation/audio-deepfakes-watermarking-and-voice-safety) (AudioSeal, SynthID-audio, C2PA-for-audio) goes from nice-to-have to load-bearing, and provenance may become a *product requirement*, not just an ethics footnote. *Evaluation* — the field still lacks a reproducible, axis-separated, human-correlated benchmark that everyone trusts, which is why "run your own harness" remains the only honest advice. And *the unification question* — whether a single omni-model eventually does speech, music, and SFX well, or whether the three sub-domains stay specialist — is genuinely open; today the specialists lead, but the language and image fields both eventually consolidated toward generalists, and audio may follow.

## Key takeaways

- **There is no "GPT-4 of audio."** Pick the sub-domain first — TTS, music, or SFX — then the model. The leaders are different organizations with different access models, and a single product often needs three of them.
- **The backbone is commoditized; the moat is data, codec, and tuning.** When you evaluate a new model, ask about its data, its codec, and whether it was preference-tuned — not whether it's AR or diffusion. The architecture is table stakes.
- **Define every axis operationally.** Intelligibility is WER via an ASR; speaker similarity is cosine on a verification embedding; latency is RTF (batch) or TTFA (interactive); fidelity is FAD *with a stated embedding, reference set, and sample count*. Vibes don't survive a design review.
- **Run your own harness.** Score open models on *your* text/prompt with WER + SECS + FAD + RTF; spot-check the closed leaders by ear. A leaderboard screenshot is not a decision.
- **Every benchmark lies in a knowable way** — Arena isn't reproducible and conflates axes; FAD depends on its embedding; MOS isn't comparable across studies; WER is bounded by the ASR. Triangulate; never trust one number.
- **Open vs closed is an operational choice, not a quality verdict.** If self-hosting is a hard requirement, accept a quality notch and pick the best open option (F5-TTS, MusicGen, MMAudio, Kokoro). If it isn't, the closed leaders (ElevenLabs, Suno) usually win on polish.
- **Match the model to the job's hard constraint** — latency for agents, length and vocals for songs, sync for video SFX, size and CPU-RTF for on-device. The constraint, not the global ranking, picks the model.
- **Read the weight license, not just the code license.** "Open" sometimes means downloadable-but-non-commercial. Check before you ship.
- **Real-time speech-native is eating interactive TTS, open full-songs are coming, and the codec arms race is the quiet driver.** Build interactive products on the duplex stack, watch YuE for open songs, and watch the codecs for the next ceiling-raise.
- **The capstone ties it together:** when you actually assemble a stack, the [building-an-audio-generation-stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack) post walks the end-to-end pipeline (codec → model → vocoder → serving → eval → safety) for whichever sub-domain you landed in.

## Further reading

- van den Oord et al., **"WaveNet: A Generative Model for Raw Audio"** (2016) — where waveform-level generation began, and why it was too slow to ship.
- Zeghidour et al., **"SoundStream: An End-to-End Neural Audio Codec"** (2021) — the codec that turned audio into tokens and unlocked the language-model playbook.
- Défossez et al., **"High Fidelity Neural Audio Compression"** (EnCodec, 2022) — the codec under MusicGen/AudioGen and the rate-distortion story that propagates up the stack.
- Borsos et al., **"AudioLM: a Language Modeling Approach to Audio Generation"** (2022) and Wang et al., **"VALL-E"** (2023) — TTS and audio reframed as token language modeling.
- Copet et al., **"Simple and Controllable Music Generation"** (MusicGen, 2023) and Evans et al., **"Stable Audio"** (2024) — the open music workhorses, token-LM vs latent diffusion.
- Défossez et al., **"Moshi: a speech-text foundation model for real-time dialogue"** (2024) — open full-duplex speech and the Mimi codec.
- Series cross-links: [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard) (the foundation), [building an audio generation stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack) (the capstone), [evaluating audio generation honestly](/blog/machine-learning/audio-generation/evaluating-audio-generation-honestly), [encodec, dac and the modern codec](/blog/machine-learning/audio-generation/encodec-dac-and-the-modern-codec), and [audio and joint AV generation](/blog/machine-learning/video-generation/audio-and-joint-av-generation) in the video series.
- 🤗 docs: `transformers` audio models (`MusicgenForConditionalGeneration`, `EncodecModel`, `VitsModel`), `diffusers` audio (`AudioLDM2Pipeline`, `StableAudioPipeline`), and the `audiocraft` and `descript-audio-codec` repositories for the open toolchain.

The map is not the territory, and standings move monthly — but the *method* here outlasts any snapshot. Pick the sub-domain, define your axes operationally, run the harness on your own inputs, read the license, and choose by the job's hard constraint rather than the leaderboard's headline. Do that and you will pick well in 2026, and you will still pick well when half the model names in this post have been replaced.
