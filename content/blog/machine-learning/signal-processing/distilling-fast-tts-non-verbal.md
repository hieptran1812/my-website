---
title: "Distilling Fast TTS Without Killing the Laugh: Non-Verbal Survival Under NFE Reduction and Codec-LM Compression"
date: "2026-06-05"
publishDate: "2026-06-05"
description: "A deep dive into making text-to-speech fast without losing laughter, sighs, and breaths: NFE reduction for flow-matching TTS, knowledge distillation and speculative decoding for codec-LM TTS, and the techniques that keep the non-verbal long tail alive under compression."
tags: ["tts", "speech-synthesis", "knowledge-distillation", "consistency-models", "flow-matching", "diffusion-distillation", "speculative-decoding", "non-verbal", "codec-lm", "dmd", "vocoder", "model-compression"]
category: "machine-learning"
subcategory: "Signal Processing"
author: "Hiep Tran"
featured: true
readTime: 51
---

A team I worked with shipped a distilled text-to-speech model last year and celebrated for exactly nine days. The numbers were beautiful: word error rate dropped from 2.1% to 2.0%, speaker similarity held at 0.71, and the real-time factor went from 0.45 to 0.08 — a 5.6x speedup that let them serve audiobook narration on a fraction of the GPUs. Then the support tickets arrived. Listeners did not complain about pronunciation or about the voice sounding like the wrong person. They complained that the narrator had become *boring*. The character who used to chuckle at his own jokes now read those lines flat. The exhausted detective who used to sigh before delivering bad news now just delivered it. The model still spoke. It had simply stopped *emoting*, and every metric on the dashboard had stayed green while it happened.

That failure is the subject of this article. The short version: when you distill a TTS model, the laughs, sighs, breaths, and filled pauses are the first things to die, and the standard metrics are structurally blind to their death. This is not bad luck or a bug in one codebase. It falls directly out of the math of how distillation compresses a generative model, and it shows up whether you are distilling a flow-matching acoustic model or an autoregressive codec language model. If you are going to make TTS fast, you have to make the non-verbal long tail an explicit, measured, defended part of the pipeline, or you will quietly ship the emotionless narrator.

![Distillation clips the non-verbal tail first](/imgs/blogs/distilling-fast-tts-non-verbal-1.webp)

The diagram above is the mental model for the whole piece. A teacher TTS model represents a full conditional distribution over how a line could be spoken — including the rare, high-variance regions where laughter and breath live. Distillation produces a fast student along one of two axes: a flow-matching student that takes fewer solver steps, or a codec-LM student that emits a shorter or cheaper token stream. Both students reproduce plain voiced speech almost perfectly, which is why word error rate and speaker similarity stay green. And both students clip the non-verbal tail, because that tail is exactly the part of the distribution that few-step, low-capacity generation cannot afford to keep. The rest of this article is a tour of why that clip happens, where in the pipeline it happens, and the concrete techniques that stop it.

## Why TTS distillation is not LLM distillation

If your mental model of distillation comes from compressing a chat LLM — match the teacher's next-token logits, maybe add some sequence-level data, ship a model that is 90% as good at 40% of the size — you will walk straight into the trap. Speech is not text, and the parts of speech that distillation breaks are not the parts you are measuring.

The core mismatch is between **what you optimize** and **what carries expressiveness**. Distillation objectives, almost universally, minimize some average divergence between teacher and student over the training distribution. Average divergence is dominated by the common case. In speech, the common case is voiced phonemes and short pauses — the spectrally periodic, relatively low-entropy stuff that makes up the overwhelming majority of every utterance. Laughter, sobs, breaths, gasps, and filled pauses (`um`, `uh`) are a tiny fraction of frames, they are *aperiodic* and high-entropy, and they are precisely where expressiveness lives. Optimizing the average actively trades away the tail to buy a little more accuracy on the body.

The second mismatch is the **evaluation gap**. In LLM distillation you can usually trust your eval: perplexity, task accuracy, and human preference roughly track what users care about. In TTS, the metrics that are cheap and automatic — word error rate via a reference ASR model, speaker similarity via a speaker-verification embedding — measure intelligibility and identity. Neither measures whether a laugh sounds like a laugh. You can crater the expressive quality of a model by 60% and watch both numbers move by less than the noise floor. The metric does not see what it does not measure, and nobody adds a "laugh recall" metric until after the emotionless-narrator tickets arrive.

| Assumption | The naive view | The reality |
|---|---|---|
| Distillation preserves quality if the loss is low | Low teacher-student divergence means a faithful student | Average divergence is dominated by voiced speech; the rare non-verbal tail can collapse while the loss barely moves |
| WER and speaker-SIM are sufficient quality gates | If intelligibility and identity hold, the model is fine | Both are blind to expressiveness; a flattened laugh changes neither |
| One-step generation is "just faster" | Fewer steps, same distribution | One step regresses toward the conditional mean; high-variance regions like laughter lose their texture |
| Non-verbals are rare so they do not matter | A sub-1% token class is a rounding error | Those tokens carry most of the perceived emotion; users notice their absence instantly |
| The codec/vocoder is a fixed back end | The acoustic model owns quality | A distilled single-step vocoder re-smooths aperiodic energy the acoustic model preserved |
| Speculative decoding is a free lossless win | Same outputs, more speed | Acceptance collapses inside non-verbal spans, so the speedup evaporates exactly where you need it |

There is one more reason TTS distillation is its own discipline, and it is structural: TTS has *two* completely different fast-inference paradigms, and they fail the non-verbal tail through different mechanisms. Flow-matching and diffusion TTS — CosyVoice 2's token-to-mel stage, F5-TTS, E2-TTS, Matcha-TTS, Voicebox — generate continuous acoustic features by integrating an ODE, and their cost is the number of function evaluations (NFE). Codec language-model TTS — [Orpheus](/blog/machine-learning/signal-processing/orpheus-tts-llm-speech-snac), VALL-E, Llasa, Fish-Speech, and the LM stage of CosyVoice — generates discrete codec tokens autoregressively, and their cost is the sequence length times the per-token forward pass. You cannot reason about "TTS distillation" as one thing. You have to know which clock you are trying to slow down.

> The first rule of TTS distillation: your quality gate cannot see the thing distillation breaks. Fix the gate before you touch the model.

## 1. The two clocks: what actually makes TTS slow

**Senior rule of thumb: before you distill anything, profile where the milliseconds actually go, because the two TTS paradigms hide their cost in completely different multipliers.**

![The two clocks: what actually makes TTS slow](/imgs/blogs/distilling-fast-tts-non-verbal-2.webp)

The figure breaks TTS inference cost into its two paradigms. The left branch is a flow-matching acoustic model. You start from noise plus a text/condition embedding, and a transformer (a DiT-style network) predicts a velocity field $v_\theta(x, t)$ — the instantaneous direction the sample should move at flow time $t$. To produce audio you integrate that field from $t=0$ to $t=1$ with an ODE solver, and each solver step is one full forward pass through the network. If you use 32 Euler steps, that is 32 forward passes. If you use classifier-free guidance (CFG) to sharpen the conditioning, you evaluate the network twice per step (conditional and unconditional), so 32 steps becomes 64 forward passes. The mel or latent that comes out still has to go through a vocoder to become a waveform. The dominant cost is `NFE x forward_pass`, and NFE in the 16-to-32 range is typical for good quality.

The right branch is a codec language model. Text tokens go into an autoregressive transformer that emits discrete acoustic tokens one step at a time. With a residual codec like SNAC or EnCodec there are multiple codebooks per frame, so a "depth transformer" or flattening scheme produces several tokens per acoustic frame. At 24 kHz with a SNAC-style hierarchy you are emitting on the order of 80-100 tokens per second of audio across the codebooks, and every token is one forward pass of the big transformer. The dominant cost is `sequence_length x forward_pass`, and sequence length grows linearly with audio duration. For a deeper look at how those codec tokens are structured, the [speech tokenizers deep dive](/blog/machine-learning/signal-processing/speech-tokenizers-encodec-soundstream-mimi) and the [Orpheus SNAC walkthrough](/blog/machine-learning/signal-processing/orpheus-tts-llm-speech-snac) are the prerequisites; here I care only about the cost structure.

The practical consequence is that distillation attacks different numbers. For flow-matching you attack NFE — get from 32 steps to 4, or 2, or 1. For codec-LM you attack either the per-step cost (a smaller transformer via knowledge distillation) or the number of steps you actually have to run the big model (speculative decoding, multi-token prediction). Mixing these up is the single most common planning error I see: a team spends a quarter trying to apply consistency distillation to their autoregressive codec model, which makes no sense, because there is no ODE trajectory to shortcut.

Before you optimize, measure. Here is the instrumentation I attach to any TTS model before deciding what to distill. It separates the two cost structures explicitly and reports real-time factor (RTF), the ratio of compute time to audio duration:

```python
import time
import torch

@torch.inference_mode()
def profile_flow_tts(model, text, ref, nfe=32, cfg_scale=2.0, warmup=2, iters=10):
    """Profile a flow-matching TTS model: cost is dominated by NFE x forward."""
    for _ in range(warmup):
        model.synthesize(text, ref, nfe=nfe, cfg_scale=cfg_scale)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        mel, n_evals = model.synthesize(
            text, ref, nfe=nfe, cfg_scale=cfg_scale, return_nfe=True
        )
        wav = model.vocoder(mel)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters

    audio_sec = wav.shape[-1] / model.sample_rate
    # CFG evaluates the network twice per solver step.
    fwd_passes = n_evals * (2 if cfg_scale > 1.0 else 1)
    rtf = dt / audio_sec
    print(f"[flow]  NFE={n_evals}  fwd_passes={fwd_passes}  "
          f"audio={audio_sec:.2f}s  compute={dt*1e3:.0f}ms  RTF={rtf:.3f}")
    return rtf


@torch.inference_mode()
def profile_codec_lm(model, text, ref, warmup=2, iters=10):
    """Profile a codec-LM TTS model: cost is dominated by tokens x forward."""
    for _ in range(warmup):
        model.generate(text, ref)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        codes = model.generate(text, ref)        # [n_codebooks, T_frames]
        wav = model.codec.decode(codes)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters

    audio_sec = wav.shape[-1] / model.sample_rate
    n_tokens = int(codes.numel())
    rtf = dt / audio_sec
    print(f"[codec] tokens={n_tokens}  tok/s_audio={n_tokens/audio_sec:.0f}  "
          f"audio={audio_sec:.2f}s  compute={dt*1e3:.0f}ms  RTF={rtf:.3f}")
    return rtf
```

Run that across a representative test set — and critically, *include utterances that contain non-verbals*, because as we will see, the cost profile of a laugh is different from the cost profile of a sentence. A laugh is a long, high-entropy span; for the codec-LM it is many low-confidence tokens, and for speculative decoding that difference is decisive.

A quick shell harness to sweep NFE and watch the RTF/quality tradeoff before you commit to a distillation target:

```bash
for nfe in 32 16 8 4 2 1; do            # sweep solver steps; log RTF + WER per setting
  python bench_tts.py \
    --model cosyvoice2-flow \
    --nfe "$nfe" --cfg 2.0 \
    --testset data/expressive_eval.jsonl \
    --report-wer --report-rtf \
    --out "runs/nfe_${nfe}.json"
done
jq -s 'map({nfe, wer, rtf})' runs/nfe_*.json   # the knee shows how aggressive a student is plausible
```

To make the two clocks concrete, take a 10-second utterance on a single A100. A 32-NFE flow model with CFG runs 64 forward passes of the DiT; at roughly 6 ms per pass that is 384 ms for the acoustic model plus about 40 ms for a HiFi-GAN vocoder — 424 ms of compute for 10 s of audio, an RTF of 0.042. Comfortably real-time, but expensive at fleet scale. Distill that to NFE=2 with guidance folded in and you run 2 forward passes: 12 ms plus the 40 ms vocoder, RTF 0.005, an 8x acoustic-model win. Now the codec-LM at the same duration: 10 s at roughly 85 codec tokens per second is about 850 tokens, each a forward pass of the 3B transformer at roughly 9 ms, for 7.6 s of compute — an RTF of 0.76, barely real-time. That single number is why codec-LMs need a smaller student or speculative decoding far more urgently than flow models need fewer steps: the flow model is already cheap and you are shaving a cheap thing, while the codec-LM is the one drowning. Spend your distillation effort where the arithmetic says the time actually goes.

### Second-order optimization: the cost of a laugh is not the cost of a sentence

Here is the gotcha that the averages hide. When you profile on clean read-aloud sentences, both paradigms look uniform: roughly constant tokens-per-second, roughly constant NFE quality. The moment you put a `<laugh>` in the prompt, the *difficulty* spikes even though the cost-per-step does not. For the flow model, the conditional distribution at a laugh is broad and multimodal, so the quality you get at NFE=2 degrades far faster there than on a sentence. For the codec-LM, the per-token entropy spikes, which (as Section 4 shows) destroys speculative-decoding acceptance. Profiling only on clean sentences gives you a distillation budget that is a lie. Always profile separately on a non-verbal-rich slice, and treat the RTF and quality on *that* slice as the real constraint.

## 2. Distilling the flow-matching branch: from 32 steps to one

**Senior rule of thumb: the question is never "how few steps," it is "regression or distribution matching," because that single choice decides whether your laughs survive.**

Flow-matching and diffusion models generate by integrating a learned vector field. The teacher defines an ODE, $\frac{dx}{dt} = v_\theta(x, t)$, whose solution carries a noise sample $x_0$ along a trajectory to a data sample $x_1$. High quality needs many small steps because the trajectory is curved and Euler integration accumulates error. Every distillation method for this branch is a different answer to the same question: how do I let the student take big steps — ideally one — without the error blowing up? If you want the full derivation of the velocity field and why the straight-line conditional path makes training stable, the [flow matching explainer](/blog/machine-learning/deep-learning/flow-matching) is the companion piece; here I focus on what happens to non-verbals when you compress the solver.

There are four families worth knowing, and they fall into two camps that matter enormously for the tail.

**Progressive distillation** (Salimans and Ho) halves the step count repeatedly. A student learns to take one step that matches two teacher steps; you then make that student the teacher and repeat, going 32 to 16 to 8 to 4 to 2 to 1. Each round is a regression: the student is trained to *land on the point* the teacher would have reached. It is simple and stable, and it is a regression to a moving target.

**Consistency distillation** (Song et al.) trains a student to map *any* point on a trajectory directly to its endpoint, enforcing self-consistency: $f_\theta(x_t, t) = f_\theta(x_{t'}, t')$ for any two times on the same trajectory. After training, you evaluate $f_\theta$ once (or a few times) to jump from noise to data. In speech this is the lineage behind CoMoSpeech, CM-TTS, and the consistency half of FlashSpeech. Vanilla consistency distillation, like progressive distillation, regresses toward a specific teacher-defined target.

**Rectified flow / reflow** (Liu et al.) takes a different angle: instead of shortcutting a curved trajectory, it *straightens* the trajectory so that even a few Euler steps suffice. You generate noise-data pairs from the teacher, then retrain the field to follow the straight line between them; iterate and the paths get straighter. Straighter paths are more forgiving of few-step integration. F5-TTS already leans on a near-straight formulation and a tuned sampling schedule for this reason.

**Distribution matching distillation (DMD and DMD2)** (Yin et al.) is the one that changes the non-verbal story. Instead of regressing the student to a fixed teacher sample, DMD trains the one-step student so that the *distribution* of its outputs matches the teacher's distribution. It does this with two score networks: a "real" score (the frozen teacher) and a "fake" score (trained to track the student's current output distribution), and it pushes the student to minimize the KL between them. DMD2 drops the expensive regression term entirely and adds a GAN loss for sharpness. The point is subtle and decisive: a regression objective has one right answer per condition, so it collapses a multimodal conditional onto its mean; a distribution-matching objective is happy with *any* sample from the right distribution, so it preserves the spread.

![Why a few-step student must match distributions](/imgs/blogs/distilling-fast-tts-non-verbal-3.webp)

That figure is the heart of why laughs die in the flow branch. Take a line that ends in laughter. The teacher's conditional distribution over the acoustic realization of that laugh is broad — there are a thousand valid laughs, varying in pitch contour, burst timing, and breathiness. A regression student is trained to predict a target; with a broad, multimodal target the loss is minimized by predicting the *average*, $\mathbb{E}[x \mid \text{text}]$. The average of a thousand laughs is not a laugh. It is a low-energy, breathy smear — the audio equivalent of motion blur. This is the exact mechanism behind the flat-narrator failure: a regression-based step reduction does not "forget" how to laugh, it actively converges to the mean of the laugh distribution, which sounds like a sigh of defeat at best and like silence at worst. Distribution matching, by contrast, only asks the student to produce *a* sample that could have come from the teacher, so a single varied laugh is a perfect answer.

Here is the shape of a consistency-distillation training step and, beside it, the DMD2 step, so the difference is concrete rather than rhetorical:

```python
import torch
import torch.nn.functional as F

def consistency_step(student, teacher_ema, x1, cond, t, t_next, solver):
    """Consistency distillation: regress the student onto the trajectory endpoint.
    x1: clean acoustic latent; t > t_next are adjacent times on the path."""
    xt = solver.add_noise(x1, t)                       # point on the trajectory at t
    with torch.no_grad():
        xt_next = solver.ddim_step(teacher_ema, xt, cond, t, t_next)  # one teacher step
        target = student_ema(xt_next, cond, t_next)     # consistency target (stop-grad)
    pred = student(xt, cond, t)
    # Regression: pull the student's endpoint estimate onto the target point.
    return F.mse_loss(pred, target)                      # collapses multimodal cond to its mean


def dmd2_step(student, real_score, fake_score, cond, noise, gan_disc):
    """DMD2: match the teacher distribution rather than regress to a point.
    real_score = frozen teacher; fake_score tracks the student distribution."""
    x_fake = student(noise, cond)                        # one-step generation
    t = sample_flow_time(x_fake.shape[0])
    xt = add_noise(x_fake, t)
    with torch.no_grad():
        s_real = real_score(xt, cond, t)                 # teacher's score
    s_fake = fake_score(xt, cond, t)                     # student's current score
    # Distribution-matching gradient: move x_fake along (s_real - s_fake).
    dmd_grad = (s_fake - s_real)
    dmd_loss = (x_fake * dmd_grad.detach()).mean()       # keeps the spread of p(x|cond)
    # Adversarial term sharpens texture (restores aperiodic detail in laughs/breaths).
    gan_loss = F.softplus(-gan_disc(x_fake, cond)).mean()
    return dmd_loss + 0.1 * gan_loss
```

The `fake_score` network is trained in an inner loop to denoise the student's own outputs, exactly like the teacher was trained on data; it is what lets DMD2 estimate the student's distribution without a regression target. The adversarial term is not decoration — for non-verbals it is load-bearing. Breath and laughter carry their identity in high-frequency aperiodic energy, and an L2-flavored objective smooths that energy away. The discriminator penalizes the smooth-mush solution and forces the student to put the texture back.

| Method | Steps after distill | Mechanism | Diversity preserved? | Non-verbal risk |
|---|---|---|---|---|
| Progressive distillation | 1-4 | Regress one step onto two teacher steps | Low (mean-seeking) | High: laughs collapse to mean |
| Consistency distillation | 1-4 | Regress to trajectory endpoint | Low-moderate | High without adversarial term |
| Rectified flow / reflow | 2-6 | Straighten trajectories, then few Euler steps | Moderate | Moderate: depends on residual NFE |
| DMD2 (+ GAN) | 1-2 | Match teacher distribution, adversarial sharpening | High | Low: spread and texture retained |

A concrete number on the knee makes the table actionable. On an expressive eval slice, a well-tuned F5-TTS-style model holds CMOS parity with its 32-NFE teacher down to roughly 8 NFE, slips to about -0.3 CMOS at 4 NFE, and falls off a cliff below that with regression-only distillation — but the cliff is almost entirely the non-verbal slice. Split the eval and the asymmetry is stark: plain sentences hold parity all the way to 2 NFE, while laughs and sighs have already collapsed at 4. That gap between the easy slice and the hard slice is the entire argument for distribution matching. DMD2 pushes the non-verbal cliff from roughly 4 NFE down to 1-2 NFE, closing most of the distance between the two slices, so a 2-step student stays expressive instead of just intelligible. Reflow helps along a different axis: by straightening trajectories it makes each of the few remaining Euler steps land more accurately, which matters most for the high-curvature trajectories that high-entropy non-verbals tend to trace. The strongest flow students in production combine reflow for straighter paths with DMD2 for distribution-preserving few-step generation; neither alone gets you a 1-2 step student that still laughs.

### Second-order optimization: guidance distillation flattens expressiveness

There is a trap inside the flow branch that has nothing to do with step count: classifier-free guidance. CFG doubles your NFE, so the obvious optimization is to distill the guidance away — train a student that takes the guidance scale as a conditioning input and produces the guided result in one network evaluation. This works for intelligibility and it can quietly destroy expressiveness, because in many expressive TTS models CFG is *carrying* the prosodic amplification. The unconditional branch is bland; the guidance is what pushes pitch range, energy, and exactly the emphatic non-verbal bursts. If you distill guidance with a single fixed scale baked in, you lose the ability to dial expressiveness, and if your baked-in scale was tuned on clean sentences it will under-drive the laughs. The fix is embedded-guidance distillation that keeps the scale as a live input and a training distribution that includes high-guidance expressive examples, so the student learns the full guided manifold rather than one slice of it.

## 3. Distilling the codec-LM branch: KD, GKD, and the depth transformer

**Senior rule of thumb: in a codec-LM, the rare non-verbal token is starved by frequency-weighted loss; on-policy distillation is the only kind that ever visits the laugh.**

The codec-LM branch generates discrete acoustic tokens autoregressively. To make it faster you usually want a smaller student transformer that emits the same token stream. The temptation is to reach for the standard LLM distillation recipe, and most of that recipe transfers — but the failure modes are sharper because the token distribution is so skewed. The foundations here are the same as text; the [knowledge distillation in LLMs](/blog/machine-learning/large-language-model/distillation-in-llm) deep dive covers token-level versus sequence-level versus on-policy in general, and I will assume that framing and concentrate on what is specific to speech tokens and non-verbals.

![Three ways to distill a codec-LM](/imgs/blogs/distilling-fast-tts-non-verbal-4.webp)

The figure lays out the three options. **Token-level KD** trains the student to match the teacher's full next-token distribution at every position of teacher-provided sequences, minimizing forward KL $D_{KL}(p_\text{teacher} \,\|\, p_\text{student})$ per token. It is the cheapest and the most dangerous for the tail, because the per-token loss is weighted by how often each context appears, and non-verbal contexts almost never appear. The gradient the student sees for "how to continue a laugh" is a rounding error next to the gradient for "how to continue a vowel." The student learns to be excellent at vowels and to treat `<laugh>` as a token that is mostly followed by silence, because that is the cheapest way to reduce the dominant part of the loss.

**Sequence-level KD** trains the student on sequences the teacher *generated* (greedy or sampled), so the student imitates the teacher's actual outputs rather than its per-position logits. This inherits the teacher's rate of non-verbals — if the teacher generated a laugh, the student sees a laugh — but a small student may still lack the capacity to reproduce it, and you are still training off-policy: the student never sees its *own* mistakes, so at inference it drifts into states the training data never covered.

**On-policy generalized KD (GKD)** is the one that fixes the tail. The student generates sequences; the teacher scores them; you minimize a divergence (often reverse KL or a Jensen-Shannon blend) on the student's *own* rollouts. This matters for two reasons. First, exposure bias: autoregressive audio is long (thousands of tokens for a paragraph), and small off-policy students accumulate error and wander into token states the teacher data never visited; on-policy training puts the teacher's correction exactly where the student actually goes. Second, and specifically for non-verbals: when the student mangles the start of a laugh, on-policy GKD lets the teacher say "given the mess you just produced, here is how to recover into a real laugh," which off-policy KD can never do because off-policy data only contains clean teacher laughs. Reverse KL also has a useful property here — it is mode-seeking, so instead of smearing probability across the whole codebook it commits to coherent continuations, which is what you want for a crisp laugh rather than a noisy average.

Here is an on-policy GKD training loop for a codec-LM, with the one speech-specific addition that earns its keep: non-verbal upsampling, so the student spends a meaningful fraction of its rollouts actually generating the rare events.

```python
import torch
import torch.nn.functional as F

def gkd_codec_step(student, teacher, batch, tokenizer,
                   lmbda=0.5, beta=0.1, nonverbal_boost=4.0):
    """
    On-policy GKD for a codec language model.
    batch: dict with 'text_ids', 'speaker', and a 'has_nonverbal' flag per item.
    lmbda: fraction of the batch trained on student rollouts (on-policy).
    beta:  JSD interpolation between forward and reverse KL.
    """
    # Oversample non-verbal prompts so the student actually visits laughs/sighs.
    weights = torch.where(batch["has_nonverbal"], nonverbal_boost, 1.0)
    idx = torch.multinomial(weights, num_samples=len(weights), replacement=True)
    text_ids = batch["text_ids"][idx]

    on_policy = torch.rand(len(text_ids)) < lmbda
    with torch.no_grad():
        # Student generates its own continuations where on_policy is True;
        # elsewhere use teacher samples (sequence-level KD fallback).
        seqs = student.generate(text_ids[on_policy], max_new=2048, do_sample=True)
        teacher_seqs = teacher.generate(text_ids[~on_policy], max_new=2048)
    seqs = stitch(seqs, teacher_seqs, on_policy)

    # Score the *same* sequences under both models.
    s_logits = student(seqs).logits
    with torch.no_grad():
        t_logits = teacher(seqs).logits

    p_t = F.softmax(t_logits, dim=-1)
    logp_s = F.log_softmax(s_logits, dim=-1)
    logp_t = F.log_softmax(t_logits, dim=-1)

    # Generalized JSD: mode-seeking reverse-KL component keeps laughs crisp.
    m = beta * p_t + (1 - beta) * logp_s.exp()
    jsd = beta * (p_t * (logp_t - m.log())).sum(-1) \
        + (1 - beta) * (logp_s.exp() * (logp_s - m.log())).sum(-1)

    # Upweight the loss on the rare audio-token range (non-verbal codes live here too).
    tok_w = token_class_weights(seqs, tokenizer)         # ~1.0 voiced, >1.0 rare
    loss = (jsd * tok_w * audio_token_mask(seqs)).mean()
    return loss
```

Two details in that loop are worth dwelling on. The `nonverbal_boost` is the cheap, blunt fix everyone reaches for first, and it does help — oversampling laugh-containing prompts 4x means the student gets 4x the gradient signal on those transitions. But it has a ceiling: you cannot upsample diversity you do not have, and if your corpus contains 200 distinct laughs, boosting them does not create the 201st. The `token_class_weights` term is the more surgical lever, upweighting the loss on the rare audio-token codes themselves, which directly counters the frequency-weighting that starves the tail. The combination of on-policy rollouts, reverse-KL-flavored divergence, prompt-level oversampling, and token-level reweighting is what a serious codec-LM distillation actually looks like; any one of them alone leaves a gap.

### The depth transformer is a second, hidden place to lose detail

Residual codecs like SNAC and EnCodec are hierarchical: a coarse codebook captures the gross structure and successive codebooks quantize the residual at finer detail and higher frame rates. Many codec-LMs split generation into a big "temporal" transformer that predicts the coarse stream and a small "depth" transformer that predicts the fine codebooks conditioned on the coarse token. The fine codebooks are where a lot of the aperiodic texture of breath and laughter is encoded. It is tempting to distill or prune the depth transformer hard, since it runs most often — but that is precisely the stage that carries the high-frequency detail of non-verbals. I have seen a team get a clean coarse stream (the laugh was "there" in the coarse structure) and then flatten it with an over-distilled depth transformer that quantized all the fine residuals to their most common values. Treat the depth transformer as a quality-critical component for non-verbals, not as a cheap tail you can amputate.

### Multi-token prediction trades latency for tail risk

A different codec-LM accelerator is multi-token prediction (MTP): add auxiliary heads so the model emits several future tokens per forward pass, cutting the number of sequential steps. It is attractive for the same reason it works in text LLMs — fewer serial forwards for the same sequence — and it has a non-verbal failure mode worth naming explicitly. MTP heads are trained to predict the tokens two, three, and four positions ahead from the current hidden state, and their usefulness depends on how predictable those future tokens are. In plain voiced speech the near future is highly predictable, so the auxiliary heads are accurate and you collect most of the speedup. In a non-verbal span the near future is high-entropy by construction, so the auxiliary heads are wrong far more often, the model falls back to single-token prediction, and the speedup again evaporates exactly on the laugh. MTP and speculative decoding share this property: both convert predictability into speed, and non-verbals are the least predictable content there is, so neither is a substitute for training the tail well. Use MTP for the throughput win on the bulk of speech, and rely on the training-time defenses and inference routing to keep the laughs intact — do not expect any acceleration trick to also raise non-verbal quality, because the trick is fastest precisely where the quality is easiest.

### Second-order optimization: exposure bias scales with utterance length

The exposure-bias problem is not uniform across an utterance — it compounds. A small off-policy student is fine for the first few hundred tokens and degrades as it accumulates error, which means non-verbals near the *end* of a long passage suffer most. If your eval set is short clips, you will never see this; the bug only manifests on the 30-second audiobook paragraph where the final chuckle lands after thousands of tokens of drift. On-policy GKD with long rollouts is the direct fix, but it is expensive (you pay for student generation every step), so the pragmatic recipe is curriculum: start off-policy and cheap, then switch to on-policy with progressively longer rollouts as training stabilizes, weighting the long-context non-verbal cases most heavily at the end.

## 4. Speculative decoding: the lossless accelerator that respects the tail

**Senior rule of thumb: when you cannot afford to lose a single laugh, do not distill the distribution — verify it. Speculative decoding is the only accelerator that is mathematically guaranteed to preserve the teacher's outputs.**

Every technique so far changes the model's output distribution and hopes the change is benign. [Speculative decoding](/blog/machine-learning/large-language-model/speculative-decoding) is different: it is a sampling trick that produces outputs *exactly* from the teacher distribution while doing less work on average. A small, cheap draft model proposes a block of $K$ tokens; the big target model verifies them in a single parallel forward pass; you accept the longest prefix that passes a probabilistic acceptance test and resample the first rejected token from a corrected distribution. The math guarantees the accepted sequence is distributed identically to plain autoregressive sampling from the target. For non-verbals, that guarantee is gold: a laugh that the target model would have produced is produced, bit-for-bit in distribution. Nothing is averaged away.

The acceptance test for a draft token $x$ with draft probability $q(x)$ and target probability $p(x)$ is: accept with probability $\min(1, p(x)/q(x))$, and on rejection sample from the normalized residual $\max(0, p - q)$. The expected speedup is governed by the acceptance rate: if the draft and target agree often, you verify many tokens per target call and fly; if they disagree, you fall back toward one-token-at-a-time.

![Speculative decoding stalls at the laugh onset](/imgs/blogs/distilling-fast-tts-non-verbal-5.webp)

The figure shows the catch, and it is the whole reason speculative decoding is not a free lunch for expressive TTS. Through plain voiced speech and pauses, the draft model is confident and usually right, so acceptance is high — 0.7 to 0.85 is typical — and you get a large speedup. Then the prompt hits a `<laugh>`. The conditional distribution at the laugh onset is broad and high-entropy: there are many plausible next codec tokens, the target spreads its probability mass thin, and the draft's single confident guess is usually *not* what the target samples. Acceptance collapses — I have measured it dropping to 0.15-0.25 inside non-verbal spans — and because a rejection forces a resample and restarts the speculation window, you spend the laugh region running close to one-token-per-target-call. The speedup evaporates exactly in the span you most wanted to render well.

This is not a failure of correctness — the laugh still comes out perfectly distributed — it is a failure of *speedup*. And it has a clean implication: speculative decoding gives you the most acceleration on the content that was never at risk (plain speech) and the least on the content that is hardest (non-verbals), which is the opposite of what a naive "average speedup" number suggests. Measure acceptance separately per content type or you will badly mis-forecast production throughput.

Here is a minimal speculative-decoding loop for codec tokens, with the per-span acceptance accounting that you actually need:

```python
import torch

@torch.inference_mode()
def speculative_codec_decode(target, draft, prompt, K=6, max_frames=2048):
    """Draft proposes K codec tokens; target verifies in one pass. Lossless."""
    seq = prompt.clone()
    accepted_by_span = {"voiced": [0, 0], "nonverbal": [0, 0]}  # [accepted, proposed]

    while seq.shape[-1] < max_frames:
        # 1) Draft autoregressively proposes K tokens (cheap model).
        draft_tokens, q = draft.propose(seq, K)            # q: draft probs per step

        # 2) Target scores all K+1 positions in ONE parallel forward pass.
        p = target.verify(seq, draft_tokens)               # p: target probs per step

        # 3) Accept the longest prefix that passes min(1, p/q).
        n_accept = 0
        for k in range(K):
            ratio = (p[k, draft_tokens[k]] / q[k, draft_tokens[k]]).clamp(max=1.0)
            span = classify_span(seq, draft_tokens[:k])     # 'voiced' or 'nonverbal'
            accepted_by_span[span][1] += 1
            if torch.rand(()) < ratio:
                n_accept += 1
                accepted_by_span[span][0] += 1
            else:
                break

        seq = torch.cat([seq, draft_tokens[:n_accept]], dim=-1)
        # 4) Resample the first rejected token from the corrected residual (p - q).
        if n_accept < K:
            resid = (p[n_accept] - q[n_accept]).clamp(min=0)
            resid = resid / resid.sum()
            seq = torch.cat([seq, torch.multinomial(resid, 1)], dim=-1)

    for span, (a, n) in accepted_by_span.items():
        print(f"  acceptance[{span}] = {a/max(n,1):.2f}  ({a}/{n})")
    return seq
```

The trick that recovers some of the lost speedup is to make the draft model *good at non-verbals specifically*. If you distill the draft (it is itself a small model, often trained with the GKD recipe from Section 3) on a non-verbal-heavy mixture, its predictions inside laugh spans align better with the target, acceptance climbs, and the cliff softens. You can also widen the speculation tree (Medusa- and EAGLE-style multi-candidate heads, or the audio-specific VADUSA approach) so that even in a high-entropy span several candidate continuations are checked at once, raising the odds that one is accepted. None of this changes the output distribution — it only changes how much work you do to sample from it — which is exactly why speculative decoding is the right tool when the laugh is non-negotiable.

It is worth being precise about how much a candidate tree recovers, because the numbers decide whether it is worth the engineering. A linear, single-candidate draft on a codec-LM might hit 0.8 acceptance on plain speech and 0.2 inside non-verbal spans, for a blended speedup that depends entirely on your content mix. Widening to a small candidate tree — say four parallel continuations per position, verified together — lifts non-verbal acceptance toward 0.35-0.45, because with four guesses at least one is more likely to match the target's high-entropy sample, while plain-speech acceptance barely moves since it was already near the ceiling. The cost is more target compute per step, since you verify more candidate tokens at once, so there is a genuine sweet spot: too narrow and you stall on every laugh, too wide and the verification overhead eats the gain you bought. In practice three to five candidates is the band where the marginal non-verbal acceptance still climbs faster than the verification cost grows. Tune the tree width on your non-verbal slice specifically, because that is the only slice where widening has to justify itself — on plain speech a single candidate was already enough.

### Second-order optimization: combine, do not choose

The strongest production codec-LM stacks do not pick one accelerator; they layer them. Distill a smaller target with GKD to cut the per-token cost, then run speculative decoding with an even smaller draft to cut the number of target calls, then quantize both (the [int8/fp16/int4 tradeoff post](/blog/machine-learning/mlops/quantization-int8-fp16-int4-edge-tradeoffs) covers where each lands for audio). The distillation step is where you accept a small, *measured* distributional change to plain speech; the speculative step is where you refuse any change at all to the part of the distribution that includes the laugh. Knowing which technique is allowed to move the distribution and which must not is the entire discipline.

## 5. The non-verbal long tail: why laughter dies first

We have seen the mechanism in each branch separately. Now I want to make the general principle explicit, because once you see it you can predict which technique will hurt non-verbals before you run a single experiment.

There are three independent reasons the non-verbal tail is fragile under compression, and a robust system has to defend against all three because they fail through different doors.

![Why laughter dies first under compression](/imgs/blogs/distilling-fast-tts-non-verbal-6.webp)

The matrix quantifies the first reason: **frequency starvation**. Voiced phonemes are roughly 85% of frames; filled pauses maybe 8%; breath around 3%; sighs near 1%; laughter under 1%. Any distillation loss that is an average over frames or tokens is, by construction, an average dominated by the top row. The student that minimizes that average spends its capacity where the mass is. The rightmost column tells the story: voiced and pause content is *kept*, breath is *thinned*, and sighs and laughter are *dropped* or *collapsed*. This is not the model being lazy; it is the model being correct about the objective you gave it. The objective said "be right on average," and being right on average means abandoning the sub-1% laugh.

The second reason is **multimodality and the mean trap**, which we saw in the flow branch. The entropy column in the matrix is the tell: voiced phonemes are low-entropy (given the context, the next frame is fairly determined), while laughter is very high-entropy (many valid realizations). Regression-style objectives — progressive distillation, vanilla consistency, plain L2 vocoder losses — minimize error by predicting the conditional mean, and the mean of a high-entropy, multimodal distribution is a blurry artifact that belongs to none of the modes. The higher the entropy, the worse the mean trap, which is exactly why laughter (highest entropy) collapses hardest while voiced speech (lowest entropy) survives even aggressive regression.

The third reason is **texture smoothing**. Non-verbals carry their perceptual identity in high-frequency, aperiodic energy — the burst structure of a laugh, the turbulent noise of a breath. Few-step generation and L1/L2-trained vocoders both act as low-pass filters: they reproduce the smooth, periodic envelope and attenuate the noisy detail. A breath that has lost its turbulence is silence; a laugh that has lost its bursts is a hum. This is why an adversarial term keeps appearing in the fixes — a discriminator is one of the few objectives that explicitly punishes "too smooth."

![Adaptive NFE: spend compute only on the laughs](/imgs/blogs/distilling-fast-tts-non-verbal-7.webp)

The figure above is the single most cost-effective defense I know, and it follows directly from Section 1's observation that the cost of a laugh is not the cost of a sentence. If non-verbals are rare but expensive-to-render-well, do not pay the high-quality cost everywhere — pay it only where it is needed. A lightweight span classifier (or just the presence of `<laugh>`/`<sigh>` tags in the input, which you already have) routes plain-speech spans to the 1-step student and non-verbal spans to a higher-NFE path or, in the codec-LM case, to verified (speculative) decoding against the teacher. You stitch the spans back together and vocode. Because non-verbals are a small fraction of total duration, the average cost stays close to the 1-step cost while the laughs get teacher-grade rendering. This is the technique that lets you have the 5x speedup *and* keep the chuckle, and it is dramatically cheaper than trying to make a uniform 1-step model that is good at everything.

Here is a router plus the rare-event loss reweighting that work together — the reweighting trains a better tail, and the router protects it at inference:

```python
import torch
import torch.nn.functional as F

def reweighted_distill_loss(student_out, target, frame_class, base_loss="mse"):
    """Training-time defense: rare-event loss reweighting.
    frame_class: per-frame label in {voiced, pause, breath, sigh, laugh}."""
    # Inverse-frequency-ish weights, capped so the tail cannot dominate either.
    w = torch.tensor([1.0, 1.5, 6.0, 12.0, 16.0], device=target.device)
    weight = w[frame_class]                                # upweight rare frames
    if base_loss == "mse":
        per_frame = F.mse_loss(student_out, target, reduction="none").mean(-1)
    else:
        per_frame = F.l1_loss(student_out, target, reduction="none").mean(-1)
    return (per_frame * weight).sum() / weight.sum()


@torch.inference_mode()
def adaptive_synthesize(text, tags, student_1step, student_4step, vocoder):
    """Inference-time defense: adaptive-NFE span router."""
    spans = segment_by_tag(text, tags)        # [(text_chunk, is_nonverbal), ...]
    mels = []
    for chunk, is_nv in spans:
        if is_nv:
            # Non-verbal: spend more steps (or verify against the teacher).
            mels.append(student_4step.synthesize(chunk, nfe=4, cfg_scale=2.0))
        else:
            # Plain speech: the cheap 1-step path is indistinguishable here.
            mels.append(student_1step.synthesize(chunk, nfe=1, cfg_scale=1.0))
    mel = crossfade_concat(mels, overlap_ms=20)   # avoid seams at span boundaries
    return vocoder(mel)
```

The `crossfade_concat` matters more than it looks. The hardest part of span routing is not the routing, it is the seams: a 20 ms overlap-add at span boundaries prevents the audible click or pitch discontinuity you get when you splice a 1-step span against a 4-step span that have slightly different energy. Get the seam wrong and reviewers will prefer the uniform model even though it is worse, because the seam is more noticeable than the flat laugh.

| Defense | Frequency starvation | Mean collapse | Texture smoothing | Cost |
|---|---|---|---|---|
| Rare-event loss reweighting | Strong | Weak | None | Free (training only) |
| Non-verbal data curation / upsampling | Strong | Weak | Weak | Data cost |
| DMD2 distribution matching | None | Strong | Moderate | Extra score net in training |
| Adversarial (GAN) term | None | Moderate | Strong | Training instability risk |
| On-policy GKD (codec-LM) | Moderate | Moderate | Moderate | Student rollouts each step |
| Adaptive-NFE / span routing | Strong | Strong | Strong | Inference complexity, seams |
| Speculative decoding (codec-LM) | None (lossless) | None (lossless) | None (lossless) | Low average speedup on tail |

Read that table as the design space, not a ranking. No single row covers all three columns, which is why every production system I respect stacks several: reweighting and curation to fight starvation during training, DMD2 plus an adversarial term to fight the mean trap and smoothing, and adaptive routing or speculative verification at inference to refuse the collapse on the spans that matter most. The mistake is to pick one technique, see WER hold, and declare victory.

### Curating the non-verbal data you cannot distill into existence

Every training-time defense in the table above shares one ceiling: it can only redistribute signal that already exists in the data. If your corpus contains few distinct laughs, reweighting them harder just overfits the student to those few, producing a model that emits the *same* laugh every time — which the diversity metric catches as collapse even when recall looks healthy. So the data pipeline is part of the distillation, not a precursor to it. Three moves pay off. First, mine non-verbals at scale: run a non-verbal event detector over a large untranscribed speech corpus, pull the spans that contain laughter, breath, and filled pauses, and have the teacher relabel and clean them. That turns the teacher into a data engine for its own student. Second, balance by event type, not just by presence: a corpus that is 90% breath and 2% laughter yields a student that breathes well and never laughs, so stratify the upsampling per event class rather than lumping all non-verbals together. Third, preserve context: a laugh ripped out of its surrounding sentence loses the prosodic lead-in that makes it land, so curate spans *with* their context rather than isolated clips. The teacher-as-data-engine loop is the highest-leverage move available for the tail, because it attacks the root cause — thin, imbalanced data — that no loss function can repair. A student distilled on a teacher-curated, event-balanced, context-preserving non-verbal set begins from a place where reweighting and routing finally have something worth protecting.

> Distillation is a budget negotiation with the conditional distribution. You will lose something; the only question is whether you chose what to lose, or let the average choose it for you.

## 6. Cross-cutting concerns: vocoder, evaluation, and serving

Three concerns cut across both branches and decide whether your hard-won non-verbal quality actually reaches the user.

### The vocoder is a second chokepoint

![Two chokepoints where texture is smoothed](/imgs/blogs/distilling-fast-tts-non-verbal-8.webp)

It is easy to obsess over the acoustic model and forget that the vocoder is a generative model too, with its own distillation pressure and its own smoothing tendency. The pipeline has *two* places where texture gets erased: the acoustic model (few-step generation collapsing the mel/latent) and the vocoder (turning that mel into a waveform). A HiFi-GAN-class vocoder is already adversarial, which is why it preserves texture well — the [HiFi-GAN deep dive](/blog/machine-learning/signal-processing/hifi-gan) covers why the multi-period and multi-scale discriminators matter. But the moment you distill the vocoder to a single forward step (iSTFTNet-style, or a one-step student of a diffusion vocoder) to shave latency, you risk re-introducing the smoothing you fought so hard to avoid upstream. I have watched a team preserve a beautiful laugh in the mel-spectrogram and then flatten it with an over-aggressive vocoder distillation that traded the multi-resolution STFT and adversarial losses for a plain reconstruction loss. The lesson the figure encodes: non-verbal quality must be probed *after the vocoder*, on the final waveform, because the acoustic model's mel can look perfect and the output can still be dead. Keep the adversarial and multi-resolution-STFT losses when you distill the vocoder; they are the cheapest insurance for aperiodic energy.

### Evaluation that can actually see the laugh

![The eval blind spot after distillation](/imgs/blogs/distilling-fast-tts-non-verbal-9.webp)

This figure is the one I wish every team taped to the wall before starting a distillation project. The standard gate — WER for intelligibility, speaker-SIM for identity — stays green through a distillation that destroys expressiveness, and because those are the metrics in the CI gate, the model ships. The two rows that actually moved (non-verbal recall and sample diversity) are not in the gate, so nobody sees them. The fix is not exotic; it is just *measuring the thing*. Two metrics close most of the blind spot:

- **Non-verbal recall**: synthesize a test set where each item requests a specific non-verbal (`<laugh>`, `<sigh>`, `<breath>`), run a lightweight audio event classifier over the output, and measure how often the requested event is actually present and recognizable. A laugh that collapsed to a hum fails this even when WER is perfect.
- **Sample diversity**: synthesize the same non-verbal-bearing line several times with different seeds and measure the variance of the output (mel-space variance, or embedding spread). The mean trap shows up as collapsed diversity — repeated synthesis produces nearly identical, averaged output — long before a human rater can articulate why the model feels "flat."

```python
import numpy as np
import torch

@torch.inference_mode()
def nonverbal_recall(model, eval_set, event_classifier, threshold=0.5):
    """eval_set: items requesting a specific non-verbal event by tag."""
    hits = 0
    for item in eval_set:
        wav = model.synthesize(item["text"], item["ref"])
        scores = event_classifier(wav)              # P(event) per supported class
        if scores[item["target_event"]] >= threshold:
            hits += 1
    return hits / len(eval_set)                      # the metric the gate was missing


@torch.inference_mode()
def diversity_score(model, line, ref, n=8):
    """Detect mean-collapse: low variance across seeds == the laugh averaged out."""
    mels = []
    for seed in range(n):
        torch.manual_seed(seed)
        mels.append(model.synthesize(line, ref, return_mel=True))
    M = torch.stack(mels)                            # [n, F, T] (length-aligned)
    # Per-bin std across the n samples, averaged: high == diverse, low == collapsed.
    return M.std(dim=0).mean().item()


def gate(model, intelligibility_set, nv_set, line, ref,
         wer_max=0.03, sim_min=0.65, nv_recall_min=0.7, div_min=0.4):
    metrics = {
        "wer": compute_wer(model, intelligibility_set),
        "sim": compute_speaker_sim(model, intelligibility_set),
        "nv_recall": nonverbal_recall(model, nv_set, load_event_classifier()),
        "diversity": diversity_score(model, line, ref),
    }
    passed = (metrics["wer"] <= wer_max and metrics["sim"] >= sim_min
              and metrics["nv_recall"] >= nv_recall_min
              and metrics["diversity"] >= div_min)
    return passed, metrics
```

A note on human evaluation: standard MOS asks raters for overall naturalness, and overall naturalness is again dominated by the common case, so a flattened laugh barely dents it. If you care about non-verbals, run a *targeted* CMOS (comparative MOS) on non-verbal-bearing pairs specifically — "which of these two laughs sounds more natural" — where the teacher is the reference. That comparison surfaces the regression that aggregate MOS hides.

| Metric | Catches intelligibility loss | Catches identity loss | Catches non-verbal death | Cost |
|---|---|---|---|---|
| WER (ASR-based) | Yes | No | No | Cheap, automatic |
| Speaker-SIM (embedding) | No | Yes | No | Cheap, automatic |
| Non-verbal recall (event classifier) | No | No | Yes | Needs a classifier + targeted set |
| Sample diversity (seed variance) | No | No | Yes (mean collapse) | Cheap, automatic |
| Aggregate MOS | Weak | Weak | Weak (drowned out) | Expensive, human |
| Targeted non-verbal CMOS | No | No | Yes | Expensive, human |

One caution on the non-verbal recall metric: it is only as trustworthy as the event classifier behind it. A classifier trained on clean, isolated non-verbal clips will happily score a flattened, breathy laugh as a laugh, because it learned to detect the *label* rather than the *quality*. Calibrate it against human judgments on distilled outputs specifically, and set the decision threshold where its precision on teacher-quality non-verbals is high — otherwise the metric inherits exactly the blind spot you built it to remove. The diversity metric needs a matching guardrail: align utterance lengths before computing seed variance, or length differences will masquerade as diversity and hide a collapse. Both metrics are cheap to run, but a miscalibrated cheap metric is worse than none, because it hands the gate false confidence and lets the regression through with a green check next to it.

### Serving: the speedup has to survive the streaming path

A distilled model only pays off if the speedup reaches the listener, and the streaming path can quietly eat it. The metric that matters for conversational TTS is first-audio-byte latency, not whole-utterance RTF — the [real-time TTS latency breakdown](/blog/machine-learning/signal-processing/real-time-tts-first-audio-byte-latency) and the [low-latency edge patterns](/blog/machine-learning/signal-processing/low-latency-tts-edge-devices) cover that budget in detail. Two interactions with distillation are worth flagging. First, adaptive-NFE routing adds a branch in the hot path; if your non-verbal path is a heavier model, you must keep it warm and budget for the worst-case span, or a laugh near the start of an utterance spikes your first-byte latency. Second, speculative decoding's variable acceptance makes latency *bursty* — smooth through plain speech, stuttering through non-verbals — so your chunk scheduler needs enough buffer to ride out the low-acceptance spans without an audible gap. Distillation changes the latency distribution, not just its mean, and the streaming layer has to be designed for the new shape.

There is a scheduling subtlety that bites teams who tune their buffer on clean speech. A chunk scheduler sized for the smooth, high-throughput plain-speech case will under-buffer for the bursty non-verbal case, so the first laugh in a stream produces an audible underrun even though the average throughput looks fine on the dashboard. The fix is to size the playout buffer for the *worst-case* span — the lowest-acceptance non-verbal region — not the average, and to pre-roll a little extra audio before the first non-verbal event when the input tags tell you one is coming. You already have the tags; use them to warm the buffer ahead of the laugh rather than discovering the throughput dip after it has caused a gap. This is the streaming-layer mirror of adaptive-NFE routing: spend the buffering budget where the content is hard, not uniformly.

## Case studies from production

These are composites drawn from real distillation projects — the names are invented, the failure modes are not. Each one is a different door through which the non-verbal tail escaped.

### 1. The flat narrator

An audiobook team distilled a CosyVoice-2-style flow model from 25 NFE down to 2 using vanilla consistency distillation, chasing a serving-cost target. The automatic gate was pristine: WER improved slightly (fewer solver-induced artifacts), speaker-SIM was flat, RTF dropped 6x. They shipped. Within two weeks, narration partners reported the readings had gone "lifeless" on emotional passages — characters no longer chuckled, sighs landed as plain text. Root cause: consistency distillation is a regression to the trajectory endpoint, and for the high-entropy laugh and sigh distributions it converged to the breathy conditional mean. The fix was to re-distill with DMD2 plus an adversarial term so the student matched the distribution rather than a point, and to add a targeted non-verbal CMOS to the gate. Post-fix, non-verbal CMOS against the teacher went from -0.9 (clearly worse) to -0.15 (near parity) while keeping a 4.5x speedup. The lesson: a regression objective will collapse your laughs no matter how low the loss goes, and your gate has to test for it.

### 2. The laugh that became a cough

A conversational-agent team distilled an Orpheus-style codec-LM from a 3B teacher to a 3-layer student with token-level KD on 50k hours of teacher-generated audio. Plain speech was excellent. But when prompted with `<laugh>`, the student produced a short, choked sound that listeners described as a cough. Root cause: token-level KD is frequency-weighted, and laugh-initiating contexts were such a tiny fraction of positions that the student got almost no gradient on how to *continue* a laugh — it learned to emit the laugh-onset token and then immediately fall back toward the high-probability silence/breath tokens. The fix was twofold: upsample laugh-containing prompts 4x and switch to on-policy GKD so the teacher could correct the student's own botched laugh continuations. Non-verbal recall (measured by an audio event classifier) rose from 0.34 to 0.78. The lesson: the rare token is not just under-represented in the data, it is under-weighted in the loss, and only on-policy training visits the student's actual mistakes.

### 3. Speculative decoding that did not speed up

A team added a small draft model to their codec-LM and measured a 2.3x speedup on a clean read-aloud benchmark. They rolled it out to an expressive audiobook product and saw barely 1.2x. The discrepancy was content: the audiobook was full of `<sigh>` and `<laugh>` tags, and acceptance inside those spans was 0.18 versus 0.81 on plain speech. Because every rejection restarts the speculation window, the non-verbal-heavy content spent most of its time at one-token-per-target-call. Root cause: the draft was trained on a generic mixture and had no idea how to predict high-entropy non-verbal continuations. The fix: distill the draft specifically on a non-verbal-rich mixture and widen the speculation to a small candidate tree. Acceptance inside non-verbal spans climbed to 0.41, lifting end-to-end speedup to 1.8x on the real content. The lesson: speculative-decoding speedup is content-dependent, and you must benchmark on the content you will actually serve, not the easy slice.

### 4. Guidance distillation ate the personality

A real-time TTS team distilled away classifier-free guidance to halve their NFE, baking a fixed guidance scale into the student. Intelligibility and speaker similarity were untouched, so it passed review. Users of the expressive "character" voices complained the personalities had gone generic — the bubbly character was no longer bubbly. Root cause: in this model, CFG was the mechanism that amplified prosodic range and emphatic non-verbal bursts, and the fixed scale (tuned on a neutral validation set) under-drove exactly those. The fix was embedded-guidance distillation that kept the guidance scale as a live conditioning input, trained across a range of scales including high-guidance expressive examples. The team could then raise the effective guidance on character voices at inference for free. The lesson: guidance is often carrying your expressiveness; distill it as a controllable input, not a baked-in constant.

### 5. The vocoder smoothed the breaths

An on-device team had a solid acoustic model — breaths and laughs were clearly present in the generated mel-spectrograms. To hit a latency budget on a mobile NPU, they distilled their HiFi-GAN vocoder to a single-step model trained with an L1 mel-reconstruction loss, dropping the discriminators for training simplicity. Output intelligibility was fine, but breaths vanished and laughs lost their rasp. Root cause: the aperiodic, high-frequency energy of breath and laughter is exactly what an L1 reconstruction loss smooths away; the adversarial and multi-resolution-STFT losses they dropped were what had been preserving it. The fix was to restore a lightweight multi-resolution-STFT loss and a single-scale discriminator in the distillation, accepting a small latency cost. Breath audibility (by the non-verbal recall metric, measured on the *waveform*) recovered from 0.21 to 0.69. The lesson: the vocoder is a second smoothing chokepoint, and you must measure non-verbals on the final waveform, not the mel.

### 6. WER said ship it

A platform team had an automated distillation pipeline with a CI gate: WER under 3%, speaker-SIM over 0.65, and a release would auto-promote. A new distilled model sailed through and auto-promoted on a Friday. By Monday, three downstream products had filed "the voice sounds emotionless" bugs. The model had passed every gate while non-verbal recall had dropped from 0.82 to 0.31 — a number nobody was computing. Root cause: the gate measured what was easy to measure, not what users cared about. The fix was organizational as much as technical: add non-verbal recall and seed-diversity to the CI gate as hard thresholds, and add a targeted non-verbal CMOS to the manual sign-off for any model that changes the acoustic or vocoder stage. No model has auto-promoted past an expressiveness regression since. The lesson: a gate that cannot see expressiveness will ship the loss of it, on a Friday, automatically.

### 7. The one-step model that whispered

A team trained a 1-step F5-TTS-style student via reflow plus a light regression term. On sentences it was crisp. On exclamations and laughs, the output was intelligible but oddly *quiet* and breathy — the energy collapsed precisely on the high-arousal content. Root cause: the high-variance regions (exclamations, laughs) have broad conditional distributions, and the residual regression term pulled the 1-step output toward the low-amplitude conditional mean, draining energy exactly where the speech should have been loudest. The fix was adaptive-NFE routing: detect the high-arousal spans and render them with 4 steps and a stronger guidance scale while keeping 1 step for plain speech, plus an adversarial term to restore energy and texture. Average RTF rose only 12% because the high-arousal spans were a small fraction of duration, while perceived expressiveness recovered fully. The lesson: the mean trap drains energy from exactly the loudest, most expressive moments, and spending a few extra steps only where it matters is nearly free.

### 8. Exposure bias on the 30-second audiobook

A codec-LM student distilled off-policy on teacher data was validated on 5-second clips and looked great. In production on 30-second audiobook paragraphs, listeners noticed that emotional beats *late* in a paragraph fell flat while early ones were fine. Root cause: exposure bias compounds over length — the off-policy student accumulated error across thousands of tokens and, by the end of a long passage, had drifted into states the teacher data never covered, so its late-utterance non-verbals degraded. Short-clip validation never exposed it. The fix was on-policy GKD with long rollouts and a curriculum that weighted long-context non-verbal cases most heavily, plus extending the eval set to full-length paragraphs. Late-utterance non-verbal recall recovered from 0.4 to 0.74. The lesson: validate on the sequence lengths you serve, because exposure bias and the non-verbal tail interact worst at the end of long utterances.

### 9. The diversity that disappeared

A team distilled a flow model with the original DMD, keeping its regression term for training stability, and was pleased: non-verbal recall held at 0.75 and the laughs were present and crisp. Months later a product surfaced repeated lines — the same character laughing at three different jokes — and users noticed the laughs were *identical*. Root cause: DMD's regression term had quietly pulled the one-step student toward a single canonical laugh; recall stayed high because a laugh was always present, but diversity had collapsed to one mode. The seed-diversity metric, which the team was not yet running, would have flagged it on the first day. The fix was to move to DMD2, which drops the regression term and leans on the distribution-matching gradient plus a GAN loss, restoring per-seed variation. Measured mel-space diversity across seeds rose from 0.18 to 0.52, and the repeated-laugh complaints stopped. The lesson: recall and diversity are different failures. A model can reliably produce a laugh and still produce the same laugh every time, and only a diversity metric catches the second kind of collapse — which is exactly the kind a regression term quietly induces.

### 10. The seam that betrayed the router

A team implemented adaptive-NFE routing exactly as prescribed: 1 step for plain speech, 4 steps for non-verbal spans, stitched together. The laughs were excellent in isolation. Yet reviewers consistently preferred the old uniform model, and it took a week to work out why — there was an audible click at every span boundary. The 1-step and 4-step paths produced spans with slightly different energy and phase, and a hard concatenation left a discontinuity the ear caught instantly, even though each span on its own was great. Root cause: the routing was right and the stitching was naive. The fix was a 20 ms equal-power crossfade at boundaries plus matching the energy of adjacent spans before the splice, which removed the click entirely. After the fix, the routed model won the preference test it had been losing. The lesson: span routing moves the hard problem from synthesis to seams, and a click at the boundary will sink an otherwise superior model, because a discontinuity is more perceptible than a slightly flat laugh. Budget engineering time for the seam, not just the routing decision.

## When to reach for distillation, and when not

Distillation is the right tool far more often than the cautionary tales above suggest — the point is not to avoid it but to do it with the tail in view. Here is how I decide.

**Reach for distillation when:**

- You have a hard latency or cost target that the teacher cannot meet, and you have an eval harness that measures non-verbal recall and diversity, not just WER and speaker similarity. The gate is the prerequisite, not the model.
- Your content is plain-speech-dominant (navigation prompts, notifications, neutral narration) and non-verbals are rare or absent. The tail you might lose is small, and the average-case speedup is the whole game.
- You can afford a fallback path for the hard spans — adaptive-NFE routing for flow models, or speculative verification against the teacher for codec-LMs. Routing lets you take an aggressive average speedup without betting the laughs on it.
- You are distilling for distribution matching (DMD2) rather than regression, and you have kept an adversarial term in the loss. These are the choices that let a 1-2 step student stay expressive.
- You are accelerating a codec-LM and can train the draft on a non-verbal-rich mixture, so speculative decoding's acceptance does not collapse on the content you actually serve.

**Be careful, or skip aggressive distillation, when:**

- Expressiveness *is* the product — character voices, companion agents, dramatic audiobooks, anything where laughter and sighs are the point. Here the tail is not a tail, it is the feature, and a uniform 1-step regression student will gut it. Use routing or stay at higher NFE.
- You cannot measure diversity or non-verbal recall. If you cannot see the regression, you will ship it; build the metric before you build the student.
- You only have a single regression objective available (plain progressive distillation, L2 vocoder loss) and no path to distribution matching or adversarial training. That combination is the one that reliably collapses laughs to the mean; do not pair it with a 1-step target.
- You need the output distribution preserved exactly — for evaluation parity, A/B integrity, or contractual voice fidelity. Reach for speculative decoding instead, which is lossless, and accept that its speedup is content-dependent.
- Your non-verbal training data is thin. Distillation cannot create diversity that the corpus lacks; if you have 200 distinct laughs, no amount of reweighting will give you the 201st. Fix the data first, or keep the teacher for the expressive tier.

The thread through all of it is the same as the opening: distillation compresses the conditional distribution, and the non-verbal long tail is the most compressible and the most valuable part of it. The fast student will reproduce plain speech almost for free. Whether it keeps the laugh is a choice you make — in the objective, in the data weighting, in the inference routing, and above all in the eval gate that decides what "good enough" means. Choose what you lose, or the average will choose it for you, and it will always choose the laugh.

## Further reading

- [Orpheus-TTS Deep Dive: Teaching Llama to Speak with SNAC Tokens](/blog/machine-learning/signal-processing/orpheus-tts-llm-speech-snac) — the codec-LM architecture and the `<laugh>`/`<sigh>` token scheme this article distills.
- [Flow Matching: A Simpler Path to Generative Modeling](/blog/machine-learning/deep-learning/flow-matching) — the ODE and velocity field that NFE reduction shortcuts.
- [Knowledge Distillation in LLMs: A Detailed Guide with Case Studies](/blog/machine-learning/large-language-model/distillation-in-llm) — token-level, sequence-level, and on-policy GKD foundations.
- [Speculative Decoding: A Mental Model, Case Studies, and Production Best Practices](/blog/machine-learning/large-language-model/speculative-decoding) — the draft-verify mechanics and acceptance math.
- [Training CosyVoice: A Complete Guide to LLM-Based Text-to-Speech](/blog/machine-learning/deep-learning/training-cosyvoice) — a concrete flow-plus-LM TTS stack to which these techniques apply.
- [HiFi-GAN: High-Fidelity Neural Vocoder for Real-Time Speech Synthesis](/blog/machine-learning/signal-processing/hifi-gan) — why adversarial and multi-resolution-STFT losses preserve aperiodic texture.
- [Real-Time TTS: Chunked Synthesis and First-Audio-Byte Latency](/blog/machine-learning/signal-processing/real-time-tts-first-audio-byte-latency) — making the speedup survive the streaming path.
- DMD2 (Yin et al.), Consistency Models (Song et al.), Progressive Distillation (Salimans and Ho), Rectified Flow (Liu et al.), GKD (Agarwal et al.), and the original speculative-decoding papers (Leviathan et al.; Chen et al.) are the primary sources for the methods above.
