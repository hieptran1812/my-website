---
title: "Streaming vs Offline Mismatch: When Train-Offline, Deploy-Streaming Breaks"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Your ASR model scores 8.7% WER in the offline eval and 16% in production, and the model never changed — here is how to localize the streaming gap and close it."
tags:
  [
    "debugging",
    "model-training",
    "speech",
    "asr",
    "streaming",
    "finetuning",
    "deep-learning",
    "pytorch",
    "torchaudio",
    "train-serve-skew",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/streaming-vs-offline-mismatch-1.png"
---

You shipped a speech model. In the offline evaluation it scored 8.7% word error rate (WER) on `test-clean`, the best number your team had ever seen. Three days after it went live behind a real-time microphone, the support channel filled up: the model drops the first word of every sentence, garbles the ends of phrases, and produces nonsense for the first second of any recording. You pull the production logs, re-run the *exact same checkpoint* on the *exact same audio* in your offline harness, and get 8.7% again. The weights did not change. The audio did not change. And yet production is running at roughly 16% WER — almost double.

This is one of the most disorienting bugs in applied machine learning, because every instinct you have says "the model is broken" and every measurement you take says "the model is fine." The model is fine. What broke is the *contract* between how you trained and evaluated it (offline, whole utterances, full context, statistics computed over the entire clip) and how you deploy it (streaming, small chunks, only past context plus a sliver of the future, statistics you have to estimate on the fly). The training distribution and the serving distribution diverged, silently, and your offline eval was blind to it because the offline eval *is* the training distribution.

This post is about that divergence — what the streaming literature calls the offline-streaming or train-serve mismatch — and how to find it, prove it, and close it. The running example is automatic speech recognition (ASR), because that is where the bug bites hardest and where the mechanism is cleanest to explain. But the principle is general: any time you train a model that consumes a whole sequence at once and then deploy it to consume that sequence in pieces — streaming video, real-time translation, incremental text processing, online anomaly detection — you can hit the exact same gap. The fix is always the same shape: **make training and evaluation see what serving sees.** Figure 1 is the map of where the gap leaks in; we will walk every layer.

![Six stacked layers between an offline acoustic checkpoint and a live streaming transcript, with the chunking, normalization, context window, and encoder state layers flagged as the places the gap enters](/imgs/blogs/streaming-vs-offline-mismatch-1.png)

By the end you will be able to: simulate streaming inference inside your eval loop so the offline number stops lying; bisect a good-offline / bad-streaming run to one of five causes — context, lookahead, normalization, chunk boundaries, or encoder state — using a runnable chunked-eval harness; swap a global normalization for a causal one and measure the gap close; verify that your encoder cache actually carries across chunk boundaries; and reason about the latency-accuracy trade-off that the lookahead window controls. This is a member of the six-places-a-bug-hides frame from the series: the streaming gap is fundamentally a **data / distribution-shift** bug wearing an **evaluation** bug's clothing. The model code is innocent. The instruments — WER measured *under streaming conditions*, the per-chunk activation statistics, the normalization stats over time — are what catch it. We will read those instruments rather than guess.

## The symptom: a number that is correct and useless

Let us be precise about the symptom, because the precision is what makes this bug bisectable rather than mysterious.

You have two numbers. The **offline WER** is what your eval harness reports: you load each test utterance fully into memory, run one forward pass over the whole thing, decode, and compute the error rate. The **streaming WER** is what production experiences: audio arrives in chunks (say 640 ms at a time), each chunk is fed to the model as it arrives, the model can only attend to what it has seen so far plus a small lookahead, and partial transcripts are emitted as they stabilize. Same model, same audio, two numbers. When they diverge, you have a streaming gap.

The gap has a *signature*, and the signature is diagnostic. Here is what it actually looks like in practice, and what each pattern points to:

| Symptom in production | What the offline eval shows | Most likely cause |
| --- | --- | --- |
| First 0.5–1.0 s of every utterance is garbage, then it recovers | Perfect — full clip normalized globally | Normalization (global vs running CMVN) |
| Words near silence boundaries or pauses get dropped or doubled | Perfect — continuous feature stream | Chunk-boundary effects, state reset |
| Degrades uniformly across the whole utterance, not just the start | Perfect — full bidirectional context | Context: trained with future context, removed at serve |
| Accuracy is fine but latency is 2 s and users complain | Latency not measured offline | Lookahead too large (the other side of the trade-off) |
| Random degradation that worsens the longer the utterance runs | Perfect — single forward pass | Encoder state not carried, or carried wrong |

Notice the column that is constant: **the offline eval shows "perfect" for every one of these.** That is the entire problem. The offline eval cannot distinguish a model that will stream well from a model that will stream catastrophically, because the offline eval never streams. It is measuring a quantity — accuracy under full-context, globally-normalized, single-forward-pass conditions — that production will never reproduce. The first and most important fix in this whole post is not a model change. It is an *eval* change: make your eval stream. Everything else follows from being able to see the gap.

#### Worked example: the 8.7 vs 16 gap, decomposed

Concretely, suppose your offline WER is 8.7% and your streaming WER is 16.2%, a gap of 7.5 points. When you instrument the streaming run (we will build the harness below) and decompose the gap by ablating one cause at a time, you might find:

- Switching the eval to **chunked causal context** (full → past + 320 ms lookahead) alone: 8.7% → 12.1% (+3.4).
- Adding **running normalization** instead of global: 12.1% → 14.0% (+1.9).
- **Dropping the encoder cache** at chunk boundaries (a real production bug): 14.0% → 16.2% (+2.2).

That decomposition is gold. It tells you that context removal is the largest contributor (3.4 points, intrinsic to streaming and only fixable by training with limited context), normalization is the second (1.9 points, fixable by matching train and serve normalization), and the cache bug is the third (2.2 points, a pure implementation bug you can fix today for free). Without the decomposition you would be staring at "16% in prod" with no idea which lever to pull. With it, you have a ranked to-do list. The rest of this post is how to produce that decomposition and act on each line.

## Why removing future context shifts the distribution

This is the science block — the part that explains *why* the gap is not a bug in the usual sense but an inevitable consequence of changing what the model conditions on. If you understand this section, the rest of the post is mechanical.

### Conditioning, formally

An acoustic model that emits a sequence of labels $y_{1:T}$ from a feature sequence $x_{1:T}$ is, at its core, estimating a conditional distribution. The question is: *conditioned on what?* In an offline, bidirectional model, the representation at frame $t$ — call it $h_t$ — is a function of the **entire** input:

$$h_t = f(x_1, x_2, \dots, x_T)$$

Every frame's encoding depends on every other frame, both past and future. A bidirectional LSTM literally runs a backward pass from $x_T$ to $x_1$; a full-context Transformer encoder lets frame $t$ attend to all $T$ frames through self-attention. The model learned, during training, to use that future context. It learned that the right interpretation of an ambiguous phoneme at frame 40 often depends on what happens at frame 80.

In streaming, the representation at frame $t$ can only depend on the past and a bounded lookahead $R$ (the right context, in frames):

$$h_t^{\text{stream}} = f(x_1, \dots, x_{t+R}), \qquad R \ll T$$

If $R = 0$ the model is fully causal; if $R$ is, say, 16 frames (320 ms at a 20 ms frame shift), the model gets a small peek ahead. Either way, $h_t^{\text{stream}} \neq h_t$ for the vast majority of frames, because the function $f$ was *trained* expecting arguments $x_{t+R+1}, \dots, x_T$ that are now missing.

Here is the precise statement of why this is a distribution shift and not just "less information." The model is a fixed function. At training time, it saw inputs drawn from the distribution of *full-context representations*. At serving time, you feed it inputs from the distribution of *limited-context representations*. These are different distributions over the encoder's internal activations. The decoder, the output projection, the loss-implied calibration — all of them were tuned for the first distribution and are now being asked to operate on the second. This is exactly the train-serve skew that shows up across this series under different masks; in [distribution shift, train vs the real world](/blog/machine-learning/debugging-training/distribution-shift-train-vs-the-real-world) it is the data changing under a fixed model, and here it is the *conditioning context* changing under a fixed model. Same mathematics, different trigger.

### How much does it cost? An information argument

You can put a rough lower bound on the damage. Speech is highly coarticulated: the acoustic realization of a phoneme depends on its neighbors on both sides. A model with access to future context can resolve ambiguities that a causal model cannot resolve until the ambiguity has already passed. Empirically, in the streaming-ASR literature, moving from full bidirectional context to a causal or limited-right-context encoder costs on the order of **10–30% relative WER** depending on the language, the architecture, and the lookahead budget — for example a model at 8.7% absolute might land between roughly 9.6% and 11.3% absolute purely from the context change, before any normalization or implementation issues. (These are order-of-magnitude figures synthesized from streaming Transformer/Conformer results; treat the exact number for your model as something you must measure, not assume.) The point is the *sign and rough magnitude*: limited context costs you something real and unavoidable, and the cost grows as the lookahead shrinks.

Figure 2 contrasts the two conditioning regimes directly — the full-context offline path versus the chunked causal streaming path — and shows the WER moving the wrong way for the same weights.

![A before-after figure contrasting offline full-bidirectional context against streaming past-plus-lookahead context, with WER rising from 8.7 percent to 16.2 percent on the identical model](/imgs/blogs/streaming-vs-offline-mismatch-2.png)

### The effective right context grows with depth — and so does the latency

There is a subtle, important multiplier hiding in deep encoders that surprises people the first time they hit it: **the lookahead you grant compounds with depth.** If a single self-attention or convolution layer lets a frame peek $r$ frames into the future, then a stack of $L$ such layers lets the *output* frame depend on inputs up to roughly $L \cdot r$ frames into the future, because each layer's lookahead chains onto the previous one's. The effective right context of the whole encoder is not the per-layer lookahead; it is the per-layer lookahead times the number of layers (for convolutions, it accumulates as $\sum_\ell (k_\ell - 1)/2$ over layers, scaled by any downsampling).

This matters in two directions, and both bite. First, for **latency**: an architecture with a modest per-layer lookahead can still have an enormous total algorithmic latency once you stack 12 or 18 layers — a 2-frame-per-layer peek across 18 layers is 36 frames, $720$ ms, of built-in delay you did not consciously choose. If your product budget is 500 ms and your encoder has a 720 ms effective right context, no amount of chunk tuning will save you; you must architecturally limit the per-layer lookahead (causal convolutions, capped attention) so the *accumulated* right context fits the budget. Second, for **correctness in the harness**: when your window-cap eval grants `right_context` frames, it is granting them to the *input*, but the *deepest* layer's effective lookahead is what actually determines how much future each output frame consumed. If you cap the input window at 16 frames of lookahead but your 12-layer encoder wants 12 × 2 = 24 frames to compute the last emitted frame correctly, the emitted frame is computed with truncated context and your harness silently under-reports the achievable streaming quality. The clean way to avoid this confusion is the chunked attention mask (below): it limits lookahead *per layer in a way that composes predictably*, so the effective right context is exactly the chunk's future plus the configured lookahead, independent of depth, because every layer respects the same chunk boundary.

The practical rule that falls out: **decide your total algorithmic-latency budget in milliseconds, divide by your frame shift to get a frame budget, and make sure the encoder's *accumulated* right context fits inside it.** Audit it explicitly — sum the per-layer lookaheads, multiply convolution kernels, account for downsampling factors — rather than assuming the per-layer number is the whole story. A 30 ms-per-layer lookahead feels harmless until you multiply by depth and discover you have promised the user half a second of latency in the architecture before a single chunk is processed.

### The crucial corollary: you cannot fix this at serve time alone

If the model was *trained* with full context and you *serve* it with limited context, no clever serving trick fully recovers the loss, because the weights encode an expectation of future context that is simply not there. The only complete fix is to **train the model with the same limited context it will see at serve time** — train it causally, or with a chunked attention mask, or with the exact right-context budget the product allows. This is why the streaming-ASR field builds streaming-aware architectures (chunked attention, Emformer-style block processing, streaming Conformers) rather than trying to bolt streaming onto an offline model. The training objective has to match the serving constraint. We will see the code for a chunked attention mask shortly; the mental model is: the mask you train with *is* the streaming constraint, made differentiable.

## The diagnostic: a chunked streaming eval harness

Everything above is theory until you can *see* it. The single most valuable artifact in this entire post is a harness that runs your model the way production does — in chunks, causally, with the normalization production uses — and reports a WER you can trust. If you build only one thing from this post, build this.

The idea is simple. Instead of feeding the whole feature sequence to the model, feed it chunk by chunk, masking out any future beyond the allowed lookahead, and concatenate the per-chunk outputs. If your offline WER and your chunked WER disagree, you have just reproduced production on your laptop, deterministically, in a loop you control.

```python
import torch
import torchaudio

@torch.no_grad()
def streaming_eval_wer(model, feats, frame_chunk, right_context, decode_fn):
    """
    Simulate streaming inference over a single utterance.

    feats:          (T, F) feature tensor for one utterance, already extracted.
    frame_chunk:    how many frames to emit per step (e.g. 32 frames = 640 ms).
    right_context:  allowed lookahead in frames (e.g. 16 frames = 320 ms).
    decode_fn:      maps encoder logits -> token ids (greedy/CTC/beam).

    Returns the decoded hypothesis for the utterance under streaming conditions.
    """
    T = feats.size(0)
    logits_chunks = []
    pos = 0
    while pos < T:
        # The model is allowed to see everything up to `pos + frame_chunk`,
        # plus `right_context` frames of lookahead, but NOTHING beyond that.
        end = min(pos + frame_chunk + right_context, T)
        window = feats[:end]  # past is always visible; future is capped at `end`

        # Forward the visible window. A streaming encoder would reuse cached
        # state here; for a first-pass diagnostic we recompute and slice.
        enc = model.encoder(window.unsqueeze(0))          # (1, end, H)
        # We only "emit" the new chunk's frames, not the lookahead tail.
        emit_end = min(pos + frame_chunk, T)
        logits_chunks.append(enc[:, pos:emit_end, :])
        pos += frame_chunk

    logits = torch.cat(logits_chunks, dim=1)              # (1, T, H)
    return decode_fn(logits)
```

Run this for every utterance in your test set, compute WER against the references with `torchaudio` or `jiwer`, and compare to your offline number. The first time you do this, the gap usually appears immediately, and you have converted "production is mysteriously bad" into "my chunked harness reproduces production, now I can ablate." That is the entire game: turn an unmeasurable field problem into a measurable, reproducible loop. For why reproducibility is a precondition for debugging anything, see [reproducibility and determinism in training](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training) — the same discipline applies here, and a non-deterministic chunked harness will waste your week.

A few things about this harness are deliberately naive, and you should know which:

- **It recomputes the encoder over the growing window every chunk** instead of caching. That is `O(T^2)` work and not how you would serve, but it is *correct* for measuring the context effect in isolation: each emitted frame sees exactly the context a streaming model with carried state would see, so the WER it reports is the WER a correctly-cached streaming model achieves. We separate "what should the WER be" from "is my cache implemented right" — and that separation is itself a bisection step.
- **It uses whatever normalization `model.encoder` applies.** If that normalization is global (computed over the whole window), you have *not* yet simulated the streaming normalization gap — you have only simulated the context gap. We fix that next.
- **The lookahead is enforced by the window cap, not by an attention mask.** For a Transformer/Conformer, you would also want to apply a chunked attention mask so frames cannot attend past `end` even within the window. We cover that below.

### Adding the attention mask (the right way to enforce lookahead)

The window cap above is a coarse way to limit context: it physically excludes future frames from the forward pass. For attention-based encoders there is a cleaner, training-compatible way — a **chunked attention mask** that lets a frame attend to all past frames and to future frames only within its chunk plus the lookahead. This is exactly the mask you would *train* with to make the model streaming-native, so building it for eval and for training is the same code.

```python
def chunked_attention_mask(T, chunk_size, right_context, device):
    """
    Build a (T, T) boolean mask where mask[i, j] = True means query i
    is FORBIDDEN to attend to key j. Each query can see:
      - all keys in earlier chunks (full left context), and
      - keys up to (its chunk end + right_context).
    This is the streaming constraint expressed as a mask.
    """
    idx = torch.arange(T, device=device)
    # The last frame each query is allowed to attend to.
    chunk_end = ((idx // chunk_size) + 1) * chunk_size - 1
    allowed_to = torch.clamp(chunk_end + right_context, max=T - 1)
    # query i may attend to key j iff j <= allowed_to[i]
    j = idx.unsqueeze(0)                  # (1, T)
    limit = allowed_to.unsqueeze(1)       # (T, 1)
    forbidden = j > limit                 # True where attention is blocked
    return forbidden
```

Pass `forbidden` as the additive `-inf` mask (or boolean mask) to your attention layers. The beautiful property: **the same mask you eval with is the mask you train with.** If you train with `chunk_size=32, right_context=16` and serve with the same, there is no context mismatch — the model learned exactly the conditioning it will be given. That is the structural fix for the largest component of the gap, and it is why streaming and offline are best thought of as two different *training* recipes, not two ways of running one model.

### Confirming the cause with a controlled ablation

Now you can decompose the gap mechanically. Run four configurations and tabulate:

```python
configs = {
    "offline_full":   dict(chunk=10_000, rc=10_000, norm="global"),  # whole clip
    "chunk_causal":   dict(chunk=32,     rc=16,     norm="global"),  # context only
    "chunk_running":  dict(chunk=32,     rc=16,     norm="running"), # + norm
    "chunk_nocache":  dict(chunk=32,     rc=16,     norm="running", drop_cache=True),
}

results = {}
for name, cfg in configs.items():
    wers = []
    for feats, ref in test_set:
        hyp = streaming_eval_wer(model, feats, cfg["chunk"], cfg["rc"], decode_fn,
                                 norm=cfg["norm"], drop_cache=cfg.get("drop_cache", False))
        wers.append(wer(ref, hyp))
    results[name] = sum(wers) / len(wers)

for name, w in results.items():
    print(f"{name:16s} WER = {w:.1%}")
```

The printed ladder *is* the decomposition from the worked example: each line isolates one cause, so the deltas between consecutive lines attribute the gap. This is the make-it-fail-small philosophy applied to a distribution problem: you do not theorize about which cause dominates, you turn each one on in isolation and read the WER. Figure 3 shows the structural reason the boundary handling matters — a frame near a chunk edge has a receptive field that crosses the boundary, and how you handle that crossing (hard cut, overlap, or carried cache) decides whether that frame is starved or correct.

![A branching dataflow figure showing a frame at a chunk edge whose receptive field needs the next chunk, with a hard split starving it while an overlap window or a carried cache feeds it the missing context](/imgs/blogs/streaming-vs-offline-mismatch-3.png)

### Reading the per-chunk activation statistics as an instrument

WER is the bottom-line metric, but it is a lagging indicator — by the time WER moves, the damage is in the transcript. A leading indicator, and one of the most useful instruments for this bug, is the **distribution of the encoder's internal activations, measured per chunk and compared against the offline pass.** The logic is direct: if streaming feeds the model a different input distribution (because context or normalization changed), the encoder's hidden activations will *also* shift, and they shift *before* the decoding error shows up. You can catch the gap by watching the activations, not just the WER.

The cleanest version registers a forward hook on a mid-encoder layer and records, for each emitted chunk, the mean and standard deviation of the activations. Then you run the same audio offline and compare. Where the streaming and offline activation statistics diverge — and by how much, and *which chunks* — localizes the cause exactly as the cache-correctness test does, but as a continuous signal you can plot.

```python
import torch

def attach_activation_probe(model, layer_name="encoder.layers.6"):
    """
    Register a forward hook that records per-call activation mean/std.
    Run streaming and offline through the same audio, then diff the logs.
    """
    log = []

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        log.append((h.mean().item(), h.std().item()))

    layer = dict(model.named_modules())[layer_name]
    handle = layer.register_forward_hook(hook)
    return log, handle


# Usage: probe the same utterance offline and streaming, then compare.
off_log, h1 = attach_activation_probe(model)
_ = model.offline_forward(feats)                 # one call -> one (mean, std)
h1.remove()

st_log, h2 = attach_activation_probe(model)
_ = run_streaming(model, feats, chunk=32, rc=16) # one call per chunk
h2.remove()

# The offline pass logs once; streaming logs per chunk. Compare the streaming
# per-chunk mean/std against the offline whole-utterance mean/std.
off_mean, off_std = off_log[0]
for i, (m, s) in enumerate(st_log):
    dm = (m - off_mean)
    print(f"chunk {i:2d}: mean drift {dm:+.3f}  std {s:.3f} (offline {off_std:.3f})")
```

The print-out has a diagnostic shape. **If the first few chunks show a large mean/std drift that shrinks toward zero as you go**, that is the normalization warm-up signature — running CMVN is noisy early and stabilizes, so the activations start off-distribution and settle. **If the drift is roughly constant across all chunks**, that is the context-removal signature — every frame lost the same kind of future context, so every chunk is uniformly shifted. **If the drift grows monotonically over the utterance**, that is the dropped-cache signature — left context is being lost cumulatively, so the activations wander further off-distribution the longer the utterance runs. You have just read three different bugs off one instrument, before computing a single WER. This is exactly the spirit of [instrumenting a training run — what to log](/blog/machine-learning/debugging-training/instrumenting-a-training-run-what-to-log): the activation statistics are an instrument, and the *shape* of the signal over chunks names the suspect.

A practical refinement: log the per-dimension activation drift, not just the scalar mean, and you can sometimes see *which* feature dimensions are most affected — frequently the low-order cepstral/energy dimensions, which is consistent with a normalization (CMVN) cause rather than a context cause. The more your instrument distinguishes causes, the less you guess.

### Decoding under streaming: CTC, transducer, and partial-hypothesis stability

The diagnostic harness above stops at the encoder logits and hands them to a `decode_fn`, but the decoder is itself a place the offline-streaming gap can hide, and it deserves a paragraph because the failure mode is sneaky. The two dominant streaming-friendly losses are **CTC** (connectionist temporal classification) and the **transducer** (RNN-T and its Transformer variants). Both are designed to emit incrementally, but they emit differently, and an offline decode does not exercise the incremental path.

For CTC, offline decoding typically does a greedy or beam search over the whole logit sequence at once, with the blank-collapsing and repeat-merging applied globally. Streaming CTC must collapse and merge *incrementally*, which means a token that looked like a repeat within one chunk might actually be a genuine repeat spanning two chunks — and if your chunk boundary falls in the middle of a repeated token, naive per-chunk collapsing can drop or duplicate it. This is a chunk-boundary bug specific to the decoder rather than the encoder, and the cache-correctness test on encoder logits will *not* catch it because the encoder logits are fine; the bug is in how the decoder stitches per-chunk emissions. The detector is to compare the streaming-decoded hypothesis against the offline-decoded hypothesis *on identical encoder logits* — if the encoder outputs match (your assertion passes) but the decoded strings differ, the bug is in the streaming decoder's collapsing logic.

For the transducer, the analogous issue is the **partial-hypothesis stability** problem. A streaming transducer emits a partial transcript as it goes and may *revise* earlier tokens as more audio arrives — the user sees "I want to..." then "I want two..." then "I want to buy." Excessive revision (a "flickering" transcript) is a streaming-specific quality defect that WER does not measure at all: WER only scores the final hypothesis, so a transcript that flickers wildly and lands on the right answer scores 0% WER while feeling broken to the user. The diagnostic is a separate metric — token-level stability or "expected number of revisions per word" — that you must add to your streaming eval explicitly. The fix usually involves a stabilization delay (only display tokens that have not changed for $N$ chunks) which is, predictably, *another* latency-quality trade-off: more delay means a more stable display but a laggier one. The lesson, again: the offline eval measures one quantity (final-hypothesis WER); production has quality dimensions (stability, latency, end-of-utterance behavior) that the offline eval is structurally blind to, and you have to instrument each one to ship confidently.

## The lookahead and chunk-size trade-off

The single knob that most directly controls the context part of the gap is the **lookahead** (right context, $R$) — how far into the future a frame is allowed to peek. It is also the knob that controls latency. They pull against each other, and understanding the trade-off is what separates "I set chunk size to whatever the tutorial used" from "I chose the operating point my product can afford."

### The latency-accuracy curve

Latency in a chunked streaming system has two main components. **Algorithmic latency** is the unavoidable delay from the lookahead: to emit frame $t$, you must wait for frame $t + R$ to arrive. If $R$ is 16 frames at a 20 ms frame shift, you have built in $16 \times 20 = 320$ ms of latency before any computation. **Chunk latency** is the delay from batching frames into chunks: if you process 32-frame chunks, a frame at the start of a chunk waits up to $32 \times 20 = 640$ ms for its chunk to fill. The user-perceived latency is roughly the sum, plus compute time.

Accuracy moves the opposite way. More lookahead means more future context, which means representations closer to the full-context ones the model was trained on (if trained offline) or simply more information (if trained streaming). The relationship is monotonic but with diminishing returns: the first few hundred milliseconds of lookahead buy a lot of accuracy because most coarticulation effects are local; beyond ~500–600 ms the marginal WER improvement flattens while the latency cost keeps growing linearly.

| Right context (lookahead) | Algorithmic latency | Typical relative WER vs offline | Use case |
| --- | --- | --- | --- |
| 0 ms (fully causal) | ~0 ms | +20–30% | Lowest-latency captions, dictation |
| 160 ms (8 frames) | 160 ms | +12–18% | Voice assistants, commands |
| 320 ms (16 frames) | 320 ms | +8–12% | Real-time transcription |
| 640 ms (32 frames) | 640 ms | +4–8% | Latency-tolerant streaming |
| Full (offline) | unbounded | 0% (baseline) | Batch transcription, captions post-hoc |

The exact numbers depend on your model and language and should be measured with the harness above — but the *shape* is robust: a knee where small lookahead buys most of the accuracy, then diminishing returns. Pick the largest lookahead your latency budget allows up to the knee; past the knee you are paying latency for nothing.

There is a further latency component people forget until it bites them in production: **compute latency that scales with the carried left context.** If your streaming attention attends to all of history (unbounded left context), then each chunk's attention cost grows linearly with how far into the utterance you are, so chunk 50 is far more expensive to process than chunk 1. On a tight per-chunk deadline this means the *tail* of a long utterance can blow the real-time factor even though the *head* was comfortable — the recognizer falls behind the audio and the perceived latency balloons exactly when the user has been talking a while. This is why block-processing architectures cap the left context (a fixed memory bank rather than full history): bounded left context gives a constant per-chunk cost and a stable real-time factor. When you measure latency, measure it at the *worst case* (a long utterance), not the average, because the worst case is what trips the deadline and the worst case is at the tail, not the head. Algorithmic latency from lookahead is a constant; compute latency from unbounded left context is a creeping cost that the offline eval — which has no notion of per-chunk deadlines — cannot show you at all.

#### Worked example: choosing the operating point under a latency SLA

Suppose your product requires end-to-end latency under 500 ms (a voice assistant) and your measured curve is: fully causal 11.0% WER, 160 ms lookahead 10.1%, 320 ms lookahead 9.6%, 640 ms lookahead 9.3%, offline 8.7%. Your latency budget is roughly: 500 ms total minus ~80 ms network and ~60 ms compute leaves ~360 ms for algorithmic + chunk latency. With a 16-frame chunk (320 ms chunk latency) you cannot also afford 320 ms of lookahead — that would be 640 ms algorithmic + chunk, over budget. So you either shrink the chunk to 8 frames (160 ms) and use 160 ms lookahead (total 320 ms, WER ~10.1%), or shrink lookahead to 160 ms with a 16-frame chunk and accept the chunk latency. The WER difference between these operating points is about 0.5 points; the latency difference is what your users feel. The lesson: **the chunk size and lookahead are a product decision constrained by an SLA, not a hyperparameter you tune for WER alone.** And critically — whatever operating point you pick, you must *train* at that operating point (or at least eval at it), or the offline number will keep lying.

## Normalization: the statistics you cannot compute online

The second-largest contributor to the gap, and the most counterintuitive, is normalization. It is counterintuitive because normalization feels like a preprocessing detail, not a modeling choice — and yet a normalization mismatch alone can add a couple of points of WER and, characteristically, *only at the start of every utterance.* If your production symptom is "the first second is garbage and then it recovers," normalization is your prime suspect.

### Why global CMVN is a serving impossibility

The standard normalization in speech is **cepstral mean and variance normalization** (CMVN): for each feature dimension, subtract the mean and divide by the standard deviation so the features are zero-mean, unit-variance. The question is: mean and variance computed over *what*? In offline training, the natural and most effective choice is **global** (per-utterance) CMVN: compute the mean and variance over the *entire utterance* and apply them to every frame.

$$\hat{x}_t = \frac{x_t - \mu_{\text{utt}}}{\sigma_{\text{utt}}}, \qquad \mu_{\text{utt}} = \frac{1}{T}\sum_{i=1}^{T} x_i$$

Stare at that $\mu_{\text{utt}}$ for a second. It is a sum over all $T$ frames — including frames in the *future*. To normalize frame 1, global CMVN needs frame $T$. In a streaming system, frame $T$ has not arrived yet when you process frame 1. **Global CMVN is not approximately hard to compute online; it is impossible** — it requires information from the future by construction. A model trained on globally-normalized features is being fed, at the start of every streaming utterance, features normalized with statistics it has never seen, because production *cannot* produce those statistics until the utterance ends.

### Running (causal) CMVN and the warm-up skew

The serving-compatible alternative is **running** or **causal** CMVN: at frame $t$, normalize using the mean and variance over frames $1 \dots t$ only.

$$\hat{x}_t^{\text{run}} = \frac{x_t - \mu_{1:t}}{\sigma_{1:t}}, \qquad \mu_{1:t} = \frac{1}{t}\sum_{i=1}^{t} x_i$$

This *is* computable online — it only uses the past. But it has a warm-up problem: when $t$ is small, the estimate $\mu_{1:t}$ is noisy (you are averaging over very few frames), so the early frames of a streaming utterance are normalized with a different, higher-variance transform than the late frames. If the model was *trained* on clean global CMVN, those noisy early frames are off-distribution, and that is precisely why the first ~0.5–1.0 s degrades while the rest recovers as the running estimate stabilizes.

#### Worked example: how long the warm-up actually lasts

Put numbers on the warm-up. The variance of a sample mean over $t$ frames scales as $\sigma^2 / t$, so the *standard error* of your running mean estimate shrinks as $\sigma/\sqrt{t}$ — it falls fast at first and then crawls. Suppose the per-dimension feature standard deviation is $\sigma = 1.0$ in raw (pre-normalization) units and you decide the running estimate is "good enough" once its standard error is below $0.1$ (10% of the feature scale). You need $\sigma/\sqrt{t} < 0.1$, i.e. $\sqrt{t} > 10$, i.e. $t > 100$ frames. At a 10 ms frame shift, 100 frames is **1.0 second of audio** before your running CMVN is trustworthy. That is a quantitative prediction of exactly the symptom teams report: roughly the first second of every streaming utterance is degraded. It also explains why short commands suffer most — a 0.8 s "stop the timer" never reaches 100 frames, so the *entire* utterance is in the warm-up zone and the running statistics never stabilize. And it tells you the fix's magnitude: if you prime the running statistics with, say, 30 frames' worth of a global prior estimated from the first chunk, you start the standard error at the $t=30$ level ($\approx 0.18$) instead of $t=1$, cutting the warm-up time roughly in half. The point of the worked example is that the warm-up is not a vague "the start is bad" — it is a $\sigma/\sqrt{t}$ curve you can compute, predict, and engineer against.

Figure 4 lays out all five axes of the gap side by side — context, lookahead, normalization, boundaries, and state — with the offline behavior, the streaming behavior, and the fix for each, so you can see normalization as one row of a systematic problem rather than a one-off surprise.

![A matrix mapping the five axes of the streaming gap, context, lookahead, normalization, chunk boundary, and encoder state, each with its offline behavior, its streaming behavior, and the concrete fix that closes it](/imgs/blogs/streaming-vs-offline-mismatch-4.png)

### The fix: train with the normalization you will serve with

The fix is the same shape as the context fix: **train with running CMVN if you will serve with running CMVN.** If the model sees noisy, warming-up running statistics during training, it learns to be robust to them, and the start-of-utterance degradation largely disappears. You can also blend strategies — for example, prime the running statistics with a global estimate from a small calibration buffer, or use an exponential moving average with a sensible decay so the estimate stabilizes faster — but the non-negotiable principle is **train-serve normalization parity.** Here is the running CMVN you would use in both places:

```python
import torch

def running_cmvn(feats, eps=1e-5):
    """
    Causal CMVN: normalize each frame using only the statistics of the
    frames up to and including it. Computable online; safe for streaming.
    feats: (T, F)
    """
    T, F = feats.shape
    cumsum = torch.cumsum(feats, dim=0)                       # (T, F)
    cumsq  = torch.cumsum(feats ** 2, dim=0)                  # (T, F)
    counts = torch.arange(1, T + 1, device=feats.device).unsqueeze(1)  # (T, 1)
    mean = cumsum / counts
    var  = (cumsq / counts) - mean ** 2
    return (feats - mean) / torch.sqrt(var.clamp_min(eps))


def streaming_cmvn_state(prev_count, prev_sum, prev_sumsq, chunk, eps=1e-5):
    """
    The truly online version: carry (count, sum, sumsq) across chunks so the
    running statistics are continuous over the whole utterance, not reset
    per chunk. This is what serving must do; eval must match it.
    """
    n = chunk.size(0)
    new_count = prev_count + n
    new_sum   = prev_sum + chunk.sum(dim=0)
    new_sumsq = prev_sumsq + (chunk ** 2).sum(dim=0)
    mean = new_sum / new_count
    var  = (new_sumsq / new_count) - mean ** 2
    normed = (chunk - mean) / torch.sqrt(var.clamp_min(eps))
    return normed, new_count, new_sum, new_sumsq
```

Note the second function carries the statistics state across chunks — that is the streaming-correct version, and it connects directly to the next section, because normalization state is *one kind* of state that must survive chunk boundaries. Figure 5 contrasts the two normalization regimes and shows that the skew vanishes precisely when training matches serving.

![A before-after figure showing global CMVN normalizing with whole-utterance statistics that require future frames, versus running CMVN whose early-frame noise disappears once training uses the same causal statistics](/imgs/blogs/streaming-vs-offline-mismatch-5.png)

## Training with the chunked mask: the structural fix

We have established that the context part of the gap cannot be fully repaired at serve time — the weights have to be trained for the conditioning they will be given. This section is the recipe for doing that, because it is the single highest-impact model change and it is genuinely simple once you have the mask function from earlier. The idea: during training, apply the same chunked attention mask the model will be served with, so that the loss is computed over exactly the limited-context representations production will produce. The model learns to do its best with past plus a small lookahead, rather than learning to lean on a future it will not have.

### Dynamic chunk training: one model, many latencies

There is a refinement worth adopting from the start, because it is nearly free and it pays off operationally. Instead of training with a *single* fixed chunk size and lookahead, randomize them per batch — sample the chunk size from a small set (say 8, 16, 32 frames) and the lookahead similarly — so the model learns to operate across a *range* of streaming configurations. This is sometimes called dynamic chunk training or dynamic right-context training, and the payoff is one checkpoint that you can serve at multiple latency operating points without retraining: a low-latency mode for commands, a higher-latency higher-accuracy mode for transcription, both from the same weights. The cost is a marginal increase in offline WER (the model spreads its capacity across configurations) in exchange for a large operational simplification and surprisingly little streaming-WER loss at any single operating point.

```python
import random
import torch

def training_step(model, batch, chunk_choices=(8, 16, 32), rc_choices=(0, 8, 16)):
    """
    One SFT-style training step for a streaming encoder. We sample a chunk
    size and lookahead PER BATCH and build the matching attention mask, so
    the model learns to operate under the streaming constraint it will serve.
    """
    feats, targets, target_lens, feat_lens = batch
    T = feats.size(1)

    # Sample this batch's streaming configuration (dynamic chunk training).
    chunk = random.choice(chunk_choices)
    rc = random.choice(rc_choices)
    mask = chunked_attention_mask(T, chunk, rc, feats.device)  # (T, T), True = block

    # Forward with the streaming mask AND running CMVN (train-serve parity on
    # BOTH context and normalization, not just one).
    normed = running_cmvn(feats)                  # causal normalization at train
    logits = model.encoder(normed, attn_mask=mask)

    # CTC (or transducer) loss over the limited-context logits.
    log_probs = logits.log_softmax(-1).transpose(0, 1)   # (T, B, V) for CTC
    loss = torch.nn.functional.ctc_loss(
        log_probs, targets, feat_lens, target_lens, blank=0, zero_infinity=True
    )
    return loss
```

Two details in that snippet are the whole point and easy to skip. First, the mask uses the *same* `chunked_attention_mask` you eval and serve with — there is one source of truth for "what context is allowed," used in training, eval, and serving. Second, the features are normalized with `running_cmvn` *at training time*, not global CMVN — so the model trains on the same warming-up running statistics it will see live. Training with the chunked mask but *global* CMVN fixes context and leaves the normalization warm-up bug intact; you have to do both, which is why train-serve parity is a property of the whole pipeline, not one layer. (Note `zero_infinity=True` on the CTC loss — under aggressive chunking some alignments can hit the input-shorter-than-target trap; that is a separate bug class covered in CTC-specific debugging, but the flag keeps a streaming finetune from blowing up on the occasional degenerate chunk.)

### Why a finetune, not a from-scratch train

You almost never train a streaming model from random initialization. You take a strong offline checkpoint and *finetune* it with the chunked mask and running CMVN for a small number of epochs. This works because the acoustic features the offline model learned are largely reusable; what changes is how the model *combines* them under limited context. Practically this is a 1–3 epoch finetune at a *lower* learning rate than the original training (the model is being adapted, not rebuilt — the right LR is often 10× smaller than the from-scratch LR, the same finetuning-LR caution that recurs across modalities). Watch the streaming WER on your held-out streaming eval each epoch; it typically drops fast in the first epoch as the model stops expecting future context, then flattens. If your streaming WER does *not* improve with chunked-mask finetuning, suspect that your mask is wrong (it is not actually limiting context — assert it against the cache-correctness test) or that normalization is still mismatched (you fixed context but not CMVN). The chunked-mask finetune is also where the small offline-WER regression in the closing before-after table comes from, and that regression is the *correct* trade: you spent a fraction of a point of a number you do not deploy to halve a number you do.

## Chunk boundaries and state carryover

The last two axes — chunk-boundary effects and encoder-state carryover — are where the pure *implementation* bugs live, as opposed to the intrinsic distribution shifts of context and normalization. These are the bugs you can fix for free, the ones that show up as "we lost two points of WER and there was no reason for it," and the ones the cache-correctness test below catches in seconds.

### What "carry the state" actually means

When you process an utterance offline in one forward pass, every layer's computation at frame $t$ naturally has access to its proper left context — earlier frames flowed through the same forward pass. When you process the utterance in chunks, each chunk is a *separate* forward pass, and unless you explicitly carry the relevant state from one chunk to the next, chunk $k+1$ starts as if chunk $k$ never happened. The transcript then degrades cumulatively the further into the utterance you go, because every chunk after the first is missing its left context.

What needs to be carried depends on the architecture:

- **Self-attention / Transformer / Conformer:** the keys and values (the KV cache) of past frames, so that queries in the new chunk can attend to the real left context instead of only to the current chunk. Reset this, and every chunk attends only to itself — a 32-frame attention window, which is far too short.
- **Convolution layers:** the last `kernel_size - 1` input frames at the chunk boundary, so the convolution's receptive field spans the boundary instead of zero-padding at the edge. This is the convolution cache. Conformers have both attention and convolution, so they need both caches.
- **RNN / LSTM / GRU:** the hidden and cell state $(h, c)$, carried verbatim from the last frame of chunk $k$ to the first frame of chunk $k+1$. An RNN with reset state at every chunk is effectively re-initialized every 640 ms.
- **Normalization:** the running CMVN statistics (count, sum, sum-of-squares) from the previous section, so the normalization is continuous over the utterance rather than restarting noisy at every chunk.

Figure 6 shows the chunk-by-chunk timeline with the cache carried correctly for the first three chunks, then dropped — and the characteristic recovery when you carry it back.

![A timeline of streaming inference where chunks one through three carry the encoder cache correctly, chunk four starts cold because the cache was dropped, and carrying the cache back recovers WER from 16 percent to 9.6 percent](/imgs/blogs/streaming-vs-offline-mismatch-6.png)

### The double-count / gap trap at boundaries

Even when you carry state, the *boundary arithmetic* is easy to get wrong, and the two failure modes are mirror images:

- **Gaps:** if you emit frames `pos:pos+chunk` but the receptive field of those frames needed `kernel-1` extra frames you did not provide, the boundary frames are computed with zero-padding instead of real context — they are *starved*. This is the hard-split case in Figure 3.
- **Double-counting:** if you use an overlap-and-recompute strategy (process `pos-overlap : pos+chunk+rc` and emit only the middle) but then *also* carry a cache that already includes the overlap region, frames in the overlap get their contribution counted twice in the running statistics, or attended to twice — a subtler corruption that shows up as a low-level accuracy sag rather than an obvious break.

The way to never have this bug is a single source of truth for "which frames have been consumed," and an assertion that the streaming forward and the offline forward produce *bit-identical* (or float-close) encoder outputs for the same frames. That assertion is the most valuable test in the whole streaming stack.

### The cache-correctness test

Here is the test. It is short, it is decisive, and it should be in your CI. It says: the encoder output for frame $t$, computed via streaming with carried cache, must equal the encoder output for frame $t$ computed via one offline forward pass over the same prefix. If they disagree, your cache is wrong, full stop.

```python
import torch

@torch.no_grad()
def assert_streaming_matches_offline(model, feats, chunk, rc, atol=1e-4):
    """
    Forward the utterance in streaming chunks WITH carried cache, and compare
    against the offline forward over the matching prefix. Encoder outputs for
    emitted frames must agree, or the cache/state carryover is buggy.
    """
    T = feats.size(0)
    cache = model.init_streaming_cache()      # KV cache, conv cache, CMVN state
    stream_out = []
    pos = 0
    while pos < T:
        end = min(pos + chunk + rc, T)
        new = feats[pos:end]
        out, cache = model.streaming_step(new, cache)   # carries state
        emit = min(pos + chunk, T) - pos
        stream_out.append(out[:emit])
        pos += chunk
    stream_out = torch.cat(stream_out, dim=0)           # (T, H)

    # Offline reference: one forward over the whole utterance, same masking.
    offline_out = model.offline_forward(feats, chunk=chunk, rc=rc)  # (T, H)

    max_err = (stream_out - offline_out).abs().max().item()
    ok = max_err < atol
    print(f"max |stream - offline| = {max_err:.2e}  -> {'PASS' if ok else 'FAIL: cache bug'}")
    assert ok, "Streaming encoder output diverges from offline; check state carryover."
    return max_err
```

When this passes, you have *proven* that your streaming implementation reproduces the intended computation, and any remaining gap is an intrinsic context/normalization effect, not a bug. When it fails, the magnitude and *which frames* diverge tell you where: if only the first frame of each chunk diverges, your convolution or attention cache is missing the boundary context; if the divergence grows over the utterance, your running normalization state is not carried; if everything after the first chunk is wrong, you forgot to carry the KV cache at all. This is bisection-by-symptom applied to a cache bug, and it is the same discipline as hunting any silent training bug — see [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) for the general decision tree this test slots into.

#### Worked example: the dropped KV cache that cost 2.2 points

A team finetunes a streaming Conformer for medical dictation. Offline eval (their CI) reports 8.9% WER and they ship. Field WER is 14%, and curiously, *short* utterances are fine while *long* utterances are terrible — the longer someone talks, the worse it gets. They run `assert_streaming_matches_offline` and it fails: `max |stream - offline| = 3.1e+00`, and the divergence is near zero for the first chunk and grows monotonically afterward. That signature — fine first chunk, degrading after — is the fingerprint of a KV cache that is initialized but never *updated* between chunks: every chunk attends only to its own 32 frames. They find the bug: the serving wrapper called `model.streaming_step(new)` but discarded the returned `cache` instead of feeding it back in. One line. After fixing it, the assertion passes at `4e-5`, and the field WER drops from 14% to 9.8% — 2.2 points recovered for free, exactly matching the `chunk_nocache` line in the decomposition ladder. The offline eval never caught it because the offline eval never chunked, so it never exercised the cache at all. This is the entire moral of the post in one bug: **if your eval does not stream, your eval cannot see the streaming bugs.**

## The eval that does not reproduce streaming

We have circled this point several times because it is the root cause beneath all the others, so let us make it a first-class section. The reason the streaming gap is so painful in practice is almost never that the fixes are hard — running CMVN is ten lines, carrying a cache is one line, training with a chunked mask is a mask function. The reason it is painful is that the *standard evaluation pipeline is structurally incapable of detecting it.* The offline eval feeds whole utterances, full context, global normalization, single forward pass — and reports a number that production will never reproduce.

### Why the default eval is misleading by construction

Think about what your eval loop actually does. It iterates the test set, and for each utterance it does something like `logits = model(features); hyp = decode(logits); wer = score(hyp, ref)`. Every word of that is offline. `model(features)` is one forward pass over the whole clip. The features were extracted with whatever normalization your data pipeline uses — almost always global, because global is best offline. There is no chunking, no causal mask, no carried state, no warm-up. The eval is a faithful measurement of *offline* accuracy and a faithful *non*-measurement of streaming accuracy. It is not buggy; it is answering a different question than the one production asks.

This is the same trap as the eval-doesn't-match-serving problem that recurs throughout this series in non-streaming forms — the LLM that scores well teacher-forced and falls apart in free-running generation, the classifier whose val set does not match the deployment distribution. The streaming case is just the most mechanical instance: the mismatch is not in the data but in *how the model consumes the data over time.* For the autoregressive-generation cousin of this exact problem, see [train-infer mismatch for LLMs](/blog/machine-learning/debugging-training/train-infer-mismatch-for-llms) — the symptom (great offline metric, bad live behavior) and the cure (eval the way you serve) are identical; only the mechanism differs.

### Making the eval honest

The fix is to make a second eval — your *streaming* eval — your release gate, not the offline one. It must:

1. **Chunk the input** at the production chunk size, feeding frames in the order and granularity they arrive.
2. **Enforce the lookahead** with a chunked attention mask (or window cap), so no frame sees beyond `chunk_end + right_context`.
3. **Use causal/running normalization**, the same one serving uses, carried across chunks.
4. **Carry the encoder state** (KV cache, conv cache, RNN state, CMVN stats) across chunk boundaries, and assert it matches offline (the cache-correctness test).
5. **Report WER under those conditions** as the number you ship against.

Figure 7 is the decision tree you walk when offline is good but streaming is bad — replay chunked first, then check normalization, then check cache — each branch landing on one of the causes we have dissected.

![A decision tree for bisecting a good-offline bad-streaming run, starting from whether a chunked replay reproduces the gap, then whether normalization is causal, then whether the encoder cache carries across chunks](/imgs/blogs/streaming-vs-offline-mismatch-7.png)

When you do this, the offline number and the streaming number tell you two genuinely different things, and you stop being surprised in production. The offline number is the ceiling — the best you could do with unlimited context. The streaming number is the floor under your latency budget — what you will actually deliver. The gap between them is your streaming tax, and now you can see it, decompose it, and pay down the parts that are bugs rather than physics. Figure 8 shows the two evals side by side: the offline one reporting 8.7% that production never sees, and the streaming-matched one reporting 9.6% that production *does* see.

![A before-after figure contrasting a misleading offline eval that reports 8.7 percent WER against a streaming-matched eval with chunked causal input and carried cache that reports 9.6 percent and matches production](/imgs/blogs/streaming-vs-offline-mismatch-8.png)

## The full before-after: closing the gap

Let us put the whole repair together as one before-after, because the series mandate is concrete evidence, not just mechanism. Here is a realistic progression for a streaming Conformer finetune, with the instrument readings at each step:

| Stage | What changed | Offline WER | Streaming WER | Streaming-offline gap |
| --- | --- | --- | --- | --- |
| 0. Baseline | Trained offline, served streaming | 8.7% | 16.2% | 7.5 |
| 1. Streaming eval added | No model change, eval now chunks + caches | 8.7% | 16.2% | 7.5 (now *visible*) |
| 2. Fix dropped KV cache | One line: feed cache back in | 8.7% | 14.0% | 5.3 |
| 3. Running CMVN parity | Train + serve with causal CMVN | 8.9% | 12.1% | 3.2 |
| 4. Chunked-mask training | Train with chunk=32, rc=16 mask | 9.1% | 9.6% | 0.5 |

Read the table top to bottom and the lesson is stark. Stage 1 changes *nothing about the model* and still matters most, because it makes the 7.5-point gap visible — you cannot fix what you cannot measure. Stage 2 is a free implementation fix worth 2.2 points. Stage 3 trades a hair of offline accuracy (8.7 → 8.9, because running CMVN is slightly worse offline than global) for 1.9 points of streaming accuracy — a trade you make gladly because streaming is what ships. Stage 4 is the big structural fix: training with the chunked mask costs another 0.2 points offline (9.1 vs 8.9) but collapses the gap to 0.5 points, because now the model was *trained* on exactly the conditioning it is *served* with. The offline number got slightly worse at every model-changing step, and that is correct and healthy: you are no longer optimizing for a regime you do not deploy. The streaming number — the one that matters — went from 16.2% to 9.6%, a 6.6-point absolute improvement and a 41% relative reduction, with no change to the data and no exotic architecture, just train-serve parity on five axes.

#### A note on cost

These fixes are also cheap in compute terms relative to the alternative of "ship a worse model and lose users." Re-finetuning a streaming Conformer with the chunked mask is typically a 1–3 epoch finetune from the offline checkpoint, not a from-scratch train — for a mid-sized model that might be a handful of GPU-hours, on the order of a few dollars to low tens of dollars of compute at roughly \$2–\$3 per GPU-hour, versus the open-ended cost of degraded production accuracy. The cache fix and the running-CMVN swap cost essentially nothing — they are code changes you validate with the assertion test on a CPU. The expensive thing was never the fix; it was the weeks of confusion before someone made the eval stream.

There is a deeper organizational lesson in that table, beyond the per-line numbers. The reason this bug class consumes weeks rather than hours is that it sits in the seam *between* teams: the research team that trained the offline model measures offline WER and declares victory; the serving team that runs the streaming wrapper sees bad production numbers and assumes the model is weak; and because the two never run the *same* eval, neither owns the gap. The fix that prevents recurrence is not technical at all — it is to make the streaming eval the *single shared release gate* that both teams report against, so the offline number is treated as a research diagnostic and the streaming number is treated as the product number. Once the streaming eval is the contract, the gap becomes visible the first day a model is a candidate for release, and the five fixes above become a routine pre-launch checklist rather than a post-incident scramble. The most valuable artifact in this post — the chunked-causal harness — is therefore as much a process change as a code change: it relocates the moment of discovery from "three days after launch, in the support channel" to "before merge, in CI."

## Case studies and real signatures

These are real, named patterns from the streaming-ASR literature and from common production post-mortems. Where I give a number, it is either from the cited work or flagged as an order-of-magnitude estimate; do not quote my approximations as exact.

**Streaming Transformers / Transformer-Transducer (Google, 2019–2020).** The streaming-ASR line of work that productionized on-device recognition explicitly trades context for latency. Models like the Transformer-Transducer and the RNN-Transducer family are *designed* causal or limited-right-context from the start, precisely because, as we derived, you cannot bolt streaming onto a full-context model without paying the context tax. The consistent finding across this literature is that limited-context streaming costs single-digit-to-low-double-digit relative WER versus offline, and that the cost shrinks as you spend lookahead — exactly the latency-accuracy knee we tabulated. The engineering lesson the field internalized: pick your streaming constraint first, then train under it.

**Emformer / block processing (Facebook/Meta, 2020).** The Emformer (efficient memory Transformer) is a direct response to the chunk-boundary and state-carryover problems in this post. It processes audio in blocks, carries a compressed memory bank across blocks (so the model has long left context without re-attending to all of history), and uses a limited lookahead. Its whole design is an answer to "how do I carry the right state across chunk boundaries cheaply" — the same question our cache-correctness test interrogates. If you find yourself hand-rolling KV-cache carryover and fighting boundary arithmetic, you are reinventing a slice of what block-processing architectures formalize.

**Chunked attention in streaming Conformers (2021 onward).** The Conformer combines convolution and self-attention, which means streaming it requires carrying *both* a convolution cache and an attention KV cache across boundaries — exactly the double-cache requirement we listed. A common production bug, and one worth naming, is carrying the attention cache correctly but forgetting the convolution cache, which produces a small but persistent WER sag concentrated at chunk boundaries (every `chunk_size` frames). The cache-correctness assertion catches it because the divergence localizes to the boundary frames. The architectural fix the field uses is causal convolutions with explicit cache state, trained with the chunked attention mask.

**The CMVN warm-up post-mortem (common production pattern).** A frequently-seen post-mortem: a team trains with utterance-global CMVN (because it is the default and it is best offline), serves with running CMVN (because global is impossible online), and ships a model whose first ~0.5–1.0 s is unreliable on every utterance. In a command-and-control product where utterances are short ("turn off the lights"), this is catastrophic because the *entire* utterance is in the warm-up zone — the running statistics never stabilize. The fix is train-serve CMVN parity, and for very short utterances, priming the running stats from a short calibration buffer or a global prior. This pattern is the cleanest illustration that normalization is a modeling decision with a serving constraint, not a preprocessing afterthought.

**Whisper finetuned and served chunked (a finetuning-specific trap).** Whisper is trained on 30-second windows with full bidirectional attention in the encoder. When teams finetune Whisper and then serve it on shorter chunks for lower latency, they hit a textbook offline-streaming gap: the encoder was trained to see a full 30 s context, and chunking it strips most of that away. The naive chunked-Whisper serving (slice the audio, run the encoder per slice, stitch) shows exactly the boundary garbling and uniform context degradation we have dissected, and the offline finetuning eval — which runs full 30 s windows — never reveals it. The honest path is either to serve Whisper offline (it is an offline architecture) and accept its latency, or to finetune with a chunked/causal encoder mask so the model learns the limited context. This is also a reminder that the metric itself can hide the gap: WER computed after text normalization can mask the start-of-utterance errors if those errors land on tokens that normalization collapses, so always inspect raw hypotheses for the first few words, not just the normalized WER. The WER-that-lies dimension of this is its own bug class; the streaming gap and the metric bug can stack, and you want to separate them.

**The endpointer interaction (a second-order streaming bug).** In a real streaming product the recognizer does not run forever — an endpointer (voice-activity / end-of-speech detector) decides when an utterance is over and the recognizer is reset. A subtle compound bug appears when the endpointer fires *early*, truncating the audio before the recognizer's lookahead is satisfied: the last few frames are emitted with less right context than the model was trained or even eval-ed with, so the *ends* of utterances degrade specifically in production, not in the offline-chunked eval (which always has the full audio available). The signature is "the last word is often wrong, but only live." The fix is to make the streaming eval honor the *same endpointing policy* as production — feed the harness the truncated audio the endpointer would have produced — so the eval reproduces the end-of-utterance context starvation too. This is the deepest version of the post's thesis: matching serving means matching *everything* serving does, including the components around the model, not just the chunking of the model itself.

## When this is — and isn't — your bug

A decisive section, because misattributing this bug wastes days. The streaming gap has a fingerprint, and so do the things it is often confused with.

**It is the streaming gap when:** your offline WER is good and stable, your streaming WER is meaningfully worse on the *same checkpoint and same audio*, and the degradation has a streaming-shaped signature — concentrated at the start of utterances (normalization), at chunk boundaries (state/boundary), growing over the utterance (cache not carried), or uniform but milder (context removal). The clinching test is that your chunked-causal eval *reproduces* production while your offline eval does not. If you can reproduce the gap in a chunked loop on your laptop, it is the streaming gap.

**It is *not* the streaming gap when:**

- **The model is bad offline too.** If offline WER is also poor, the streaming gap is not your first problem — fix the model first. The streaming gap is, by definition, a *gap*; it requires a good offline baseline to exist.
- **The audio is genuinely different in production.** Different microphone, sample rate, codec, or background noise is a *data* distribution shift, not a streaming-mechanism shift. Check this first: re-run production audio through the offline eval. If offline WER on production audio is *also* bad, you have an acoustic-domain problem (different microphones, far-field, codec artifacts), not a streaming problem. The streaming gap only manifests as offline-good / streaming-bad on the *same* audio. For the general framing of "the data changed under a fixed model," see [distribution shift, train vs the real world](/blog/machine-learning/debugging-training/distribution-shift-train-vs-the-real-world).
- **Decoding parameters differ.** If production uses a different beam size, language-model weight, or endpointing threshold than your eval, the gap might be a decoder-config skew, not a streaming-encoder issue. Pin the decoder config across eval and serve before blaming the encoder.
- **It is a latency complaint, not an accuracy complaint.** "Streaming is slow" is the *other* side of the trade-off — you over-budgeted lookahead or chunk size. That is a tuning decision (shrink the lookahead toward the knee), not a correctness bug, and the accuracy is probably fine.

The general rule: a smooth offline-good / streaming-bad split on identical audio is a streaming-mechanism bug; a uniformly-bad result, or a result that is bad offline too, points to data or model, not streaming. Bisect with the offline eval on production audio first — it is one line and it instantly separates "acoustic domain shift" from "streaming mechanism."

## Generalizing beyond ASR

Although every concrete example here is speech, the mechanism is modality-agnostic, and naming the generalization helps you recognize the bug in other settings.

**Streaming video / online action recognition.** A model trained on full clips with 3D convolutions or full temporal attention, deployed to classify frames as they arrive, hits the same context-removal gap: the full-clip model used future frames to disambiguate the present. Causal temporal convolutions and limited-lookahead attention are the video analog of streaming ASR, and per-clip normalization (the video analog of global CMVN) has the identical "needs the future" impossibility online.

**Incremental / simultaneous translation.** A translation model that reads the whole source sentence before emitting target tokens cannot operate simultaneously (translating as the speaker talks) — it must commit to output before seeing the rest of the source. The "wait-k" family of policies is literally a lookahead budget: read $k$ source tokens of right context before emitting, the exact same latency-quality knob as ASR lookahead.

**Online time-series / anomaly detection.** A detector trained on windows normalized over the whole window (using the window's future) and deployed to score points as they stream in hits the running-normalization gap precisely as CMVN does. Any z-score computed over a window that includes future points is a global-CMVN-style serving impossibility.

The unifying statement: **whenever training consumes a sequence whole and serving consumes it incrementally, the conditioning context and the available statistics change, and a model is a function that was tuned for specific arguments.** Change the arguments — remove the future, swap the normalization, reset the state at boundaries — and you change the function's behavior. The cure is invariant across modalities: *evaluate the way you serve, and train the way you serve.*

## A streaming-readiness checklist

If you are about to ship a model that was trained offline into a streaming product, run this checklist before you trust the offline number. It is the operational distillation of everything above, ordered so each step rules out one cause and the cheap checks come first.

1. **Re-run production audio through the offline eval.** If offline WER on the production audio is *also* bad, stop — you have an acoustic-domain shift (microphone, codec, noise, far-field), not a streaming bug. Fix that first. This is one line and it saves days of chasing the wrong cause.
2. **Build the chunked-causal eval harness** at your production chunk size and lookahead, and compare its WER to the offline WER on the *same* audio. If they match, you do not have a streaming gap and you can ship. If they diverge, you have reproduced production on your laptop — proceed.
3. **Run the four-config ablation ladder** (offline-full → chunk-causal → chunk-running-CMVN → chunk-no-cache) and read the deltas. You now have the gap decomposed into context, normalization, and implementation, ranked by contribution.
4. **Run the cache-correctness assertion** (`max |stream - offline| < 1e-4`). If it fails, fix the state carryover before anything else — it is a free win and the divergence shape (first chunk fine vs growing vs boundary-localized) names the missing cache. Add the assertion to CI so it never regresses.
5. **Check normalization parity.** Confirm training and serving use the same CMVN (running, not global). If training used global, you will need to finetune with running CMVN, and you can predict the warm-up duration from the $\sigma/\sqrt{t}$ math.
6. **Audit the encoder's accumulated right context** against your latency budget in milliseconds. Sum per-layer lookaheads and convolution kernels, account for downsampling, and confirm the total fits — depth multiplies lookahead, and the per-layer number lies.
7. **Add streaming-specific quality metrics** the offline eval cannot see: partial-hypothesis stability (revisions per word), end-of-utterance accuracy under the production endpointing policy, and measured latency at the chosen operating point. WER alone is necessary but not sufficient for a streaming product.
8. **If context is the dominant gap component, finetune with the chunked attention mask** (and running CMVN) for 1–3 epochs at a reduced learning rate, watching the streaming-eval WER per epoch. Accept the small offline-WER regression; judge the model on the streaming number.

The checklist is deliberately bisective: each step either clears a suspect or localizes the bug, and the order front-loads the cheapest, highest-information checks. Most production streaming incidents I have seen are resolved by steps 1, 2, and 4 alone — a domain shift mistaken for a streaming bug, an eval that never streamed, or a cache that was never carried. The expensive structural fix (step 8) is only necessary when the cheap checks have proven the gap is intrinsic context loss, not an implementation mistake.

## Key takeaways

- **The offline eval cannot see the streaming gap.** A single-forward, full-context, globally-normalized eval reports a number production never reproduces. Build a chunked, causal, cache-carrying eval and make *it* your release gate. This one change, with no model edit, is the highest-leverage move in the whole post.
- **Removing future context is a distribution shift, not just less information.** A model trained with bidirectional/full context conditions every frame on the future; serving causally feeds it a distribution it never trained on. The only complete fix is to *train* with the limited context you will serve — a chunked attention mask, trained and evaled identically.
- **Global CMVN is a serving impossibility.** It needs whole-utterance statistics that do not exist until the utterance ends. Use running/causal CMVN, carry its state across chunks, and train with the same — or watch the first second of every utterance degrade.
- **Carry the encoder state across chunk boundaries — all of it.** KV cache for attention, conv cache for convolutions, hidden state for RNNs, running stats for normalization. The signature of a dropped cache is degradation that *grows over the utterance*; a dropped conv cache sags *at every boundary*.
- **Assert streaming equals offline frame-for-frame.** The cache-correctness test (`max |stream - offline| < 1e-4`) proves your implementation reproduces the intended computation. Put it in CI. It separates "intrinsic context tax" from "I have a bug."
- **Lookahead trades latency for accuracy with a knee.** Spend lookahead up to the knee (most accuracy is bought in the first few hundred milliseconds), then stop — past the knee you pay latency for nothing. The operating point is a product SLA decision, not a WER hyperparameter.
- **Decompose the gap by ablation.** Turn context, normalization, and cache on one at a time and read the WER ladder. The deltas attribute the gap and give you a ranked to-do list instead of a vague "prod is bad."
- **Bisect before blaming streaming.** Re-run production audio through the offline eval first: if it is *also* bad, you have an acoustic-domain shift, not a streaming bug. Streaming bugs are offline-good / streaming-bad on *identical* audio.
- **A slightly worse offline number after the fix is correct.** Running CMVN and chunked-mask training cost a fraction of a point offline; you stopped optimizing for a regime you do not deploy. Judge the model on the streaming number, because that is what ships.

## Further reading

- **PyTorch and torchaudio streaming docs** — `torchaudio.io.StreamReader`, the Emformer RNN-T pipeline (`torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH`), and streaming inference tutorials show a production-grade carried-cache implementation you can read instead of re-deriving.
- **"Transformer Transducer: A Streamable Speech Recognition Model with Transformer Encoders and RNN-T Loss"** (Zhang et al., 2020) — the streaming Transformer-Transducer; how limited context and the transducer loss combine for low-latency streaming, with the context-vs-WER trade-offs quantified.
- **"Emformer: Efficient Memory Transformer Based Acoustic Model for Low Latency Streaming Speech Recognition"** (Shi et al., 2021) — block processing with a carried memory bank; the canonical answer to chunk-boundary state carryover.
- **"Conformer: Convolution-augmented Transformer for Speech Recognition"** (Gulati et al., 2020) — the architecture whose convolution-plus-attention structure forces the double-cache requirement when streamed.
- **"STACL: Simultaneous Translation with Implicit Anticipation and Controllable Latency using Prefix-to-Prefix Framework"** (Ma et al., 2019) — the wait-k policy; the same lookahead-vs-quality knob in machine translation, useful for seeing the generality.
- **Within this series:** [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) for where the streaming gap sits in the six-places decision tree; [the training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) for the symptom-to-fix master checklist; [train-infer mismatch for LLMs](/blog/machine-learning/debugging-training/train-infer-mismatch-for-llms) for the autoregressive cousin of this exact eval-the-way-you-serve problem; and [distribution shift, train vs the real world](/blog/machine-learning/debugging-training/distribution-shift-train-vs-the-real-world) for the general framing of train-serve skew.
