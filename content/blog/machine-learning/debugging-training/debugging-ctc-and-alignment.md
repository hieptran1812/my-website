---
title: "Debugging CTC and Alignment: The Blank Token, Inf Loss, and Collapsed Output"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "A first-principles guide to the Connectionist Temporal Classification loss and the bugs unique to it: the inf-loss trap when input frames run short, blank-index mismatches that train the wrong column, and the all-blank collapse, with runnable asserts and alignment visualizations that catch each before it wastes a run."
tags:
  [
    "debugging",
    "model-training",
    "ctc",
    "speech",
    "asr",
    "alignment",
    "pytorch",
    "torchaudio",
    "finetuning",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/debugging-ctc-and-alignment-1.png"
---

You finetuned a wav2vec2 model on a few hundred hours of audio overnight, and the morning report is confusing. For the first thousand steps the training log printed a perfectly normal-looking loss around 3.4, descending slowly. Then, somewhere around step 1,200, a handful of lines read `loss = inf`, and a few steps later the whole thing tipped into `nan` and stayed there. When you load the checkpoint and run greedy decoding on a held-out clip, the model transcribes a clear spoken sentence as the empty string. Every single frame's argmax is the same integer: the blank token. The model that was supposed to learn English has, as far as you can tell, learned to say nothing at all, very confidently, on every frame.

This is not a generic NaN bug and it is not a generic "my model won't learn" bug, even though it shares symptoms with both. It is a *CTC* bug, and CTC — Connectionist Temporal Classification, the loss that lets you train a sequence model without frame-level labels — fails in ways that have no analogue in ordinary classification. The `inf` loss is not floating-point overflow; it is the loss function correctly telling you that the target you handed it is *impossible to produce* given the number of output frames you gave it. The all-blank output is not a dead network; it is a known, predictable phase that healthy CTC training passes through and that broken CTC training gets stuck in. Once you understand the small amount of math behind CTC — a sum over alignments, a blank token, and a collapse rule — every one of these symptoms becomes diagnosable in minutes instead of hours.

![A branching dataflow figure showing one target expanding into many blank-padded frame alignments that all collapse to the same string and whose probabilities are summed into the CTC loss](/imgs/blogs/debugging-ctc-and-alignment-1.png)

CTC bugs live in two of the six places a training bug can hide — *numerics* (the inf and the NaN) and *model code / data bookkeeping* (the blank index, the lengths, the log-softmax, the decode rule) — and the discipline is the same as everywhere else in this series: read the instruments, make it fail small, and bisect to the right place before you touch the model. This post is built around a single running example, a small ASR model that emits all-blank and throws `inf` loss, and we will take it from "garbage on every frame" to "WER 9.1%" by fixing, in order, the length constraint, the blank index, the length bookkeeping, and the decode rule. Along the way I will give you the math that makes each symptom *predictable* — why `inf` is the correct answer when input length drops below target length, why the network rationally starts by predicting only blank, why a blank-index mismatch trains the exact wrong column — and the runnable asserts and visualizations that turn each silent failure into a loud one. By the end you will be able to look at a CTC run that is throwing `inf` or collapsing to blank and name the cause before you have read a single line of the model definition.

## 1. What CTC actually computes

To debug CTC you have to know precisely what it computes, because almost every CTC bug is a violation of one of its assumptions. CTC was introduced by Graves, Fernández, Gomez, and Schmidhuber in 2006 to solve a specific problem: you have an input sequence of $T$ frames (audio frames, image columns in handwriting recognition, video frames) and an output sequence of $U$ labels (characters, phonemes, words), and you do *not* know which frame corresponds to which label. In speech, the word "cat" might span 40 frames or 4 frames depending on how slowly it is spoken, and nobody has annotated, frame by frame, where the `c` ends and the `a` begins. CTC lets you train a frame-level classifier anyway, by being *alignment-free*: it considers every possible way the frames could line up with the labels and optimizes the sum of their probabilities.

The mechanism rests on three pieces. The first is an extra symbol added to your vocabulary: the **blank** token, written here as `-`. Blank does not mean silence and it does not mean "space between words." It means "no label emitted at this frame" — a way for the network to say "I am still on the same sound" or "there is nothing to output right now." If your real alphabet has 28 characters (26 letters plus space and apostrophe, say), your network outputs a distribution over 29 classes at every frame: the 28 real characters plus blank.

The second piece is the **collapse rule** (sometimes called the squash or many-to-one map), which turns a frame-level sequence of length $T$ into a label sequence. The rule is exactly two steps, applied in this order: first, **merge consecutive repeated symbols**; second, **remove all blanks**. So the frame sequence `c c - a a t` collapses by first merging the repeated `c`s and `a`s to get `c - a t`, then removing the blank to get `c a t`. Crucially, the merge happens *before* the blank removal, which is the whole reason blank exists. Consider the word "hello" with its double `l`. If there were no blank, the frame sequence `h e l l o` would collapse (by merge-repeats) to `h e l o` — you would lose a letter. With blank, the network can emit `h e l - l o`, which merges nothing across the blank and then drops the blank, giving the correct `h e l l o`. **Blank is the device that lets the model emit a genuine repeated character.** This single fact explains a class of subtle bugs we will return to.

The third piece is the **summation over alignments**. For a given target label sequence $\mathbf{y}$ (length $U$), there are many frame-level sequences (each length $T$) that collapse to it. Call each such frame sequence an *alignment* or a *path*, $\pi$. The set of all paths that collapse to $\mathbf{y}$ is written $\mathcal{B}^{-1}(\mathbf{y})$, where $\mathcal{B}$ is the collapse map. CTC defines the probability of the target as the sum of the probabilities of all those paths:

$$
P(\mathbf{y} \mid X) = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{y})} P(\pi \mid X) = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{y})} \prod_{t=1}^{T} p_t(\pi_t \mid X)
$$

where $p_t(k \mid X)$ is the network's softmax probability of class $k$ at frame $t$, and the per-path probability is just the product of the per-frame probabilities (CTC assumes the per-frame outputs are conditionally independent given the input — a strong assumption that, notably, is what creates the train–infer mismatch we discuss at the end). The CTC **loss** is the negative log of this probability:

$$
\mathcal{L}_{\text{CTC}} = -\log P(\mathbf{y} \mid X)
$$

Figure 1 above shows this as a branch-and-merge: one target fans out into many candidate alignments (`c c a t -`, `- c a a t`, `c a a t t`, and so on), each collapses back to "cat," and their probabilities are summed into one loss. The number of such paths is combinatorially large — for a target of length $U$ and $T$ frames it grows roughly like the number of monotone paths through a lattice — so you cannot enumerate them. The genius of CTC, exactly like the forward algorithm for hidden Markov models, is that you can compute the sum efficiently with dynamic programming, in $O(T \cdot U)$ time, using a forward-backward recursion. We will look at that recursion in the next section, because the place where the recursion has *zero valid paths* is exactly where you get `inf` loss.

Here is the punchline you need to keep in your head for the rest of the post. **CTC does not learn an alignment; it marginalizes over all of them.** It never commits to "the `c` is at frame 7." It computes the total probability mass of every legal alignment and pushes that up. That is why CTC is wonderful (no frame labels needed) and why it is buggy in specific ways: the set of legal alignments can be *empty* (inf loss), the network can rationally put all its mass on the all-blank alignment early in training (collapse), and the blank token has to be at a consistent index in three different places (the loss, the label space, and the decoder) or the whole accounting silently breaks.

It is worth placing CTC against its alternatives for a moment, because the choice of loss determines the bug class you will fight. An attention-based encoder-decoder (the Listen-Attend-Spell family, and the decoder half of Whisper) learns the alignment *implicitly* through cross-attention and is trained with ordinary token-level cross-entropy — it has no blank, no length constraint, and no inf-loss trap, but it pays with exposure bias and the freedom to hallucinate fluent wrong text. RNN-Transducer (RNN-T) keeps a blank-like symbol and a monotonic alignment like CTC but adds a prediction network so the outputs are no longer conditionally independent; it removes CTC's independence assumption at the cost of a more expensive loss and its own diagonal-of-the-lattice bugs. CTC sits at the simple end: one linear head, a closed-form forward-backward, a hard length constraint, and a conditional-independence assumption that makes decoding fast and training stable but creates the train–infer gap we cover in section 8. The reason CTC remains everywhere despite its age — most production streaming ASR, a great deal of OCR and handwriting recognition, keyword spotting — is precisely that simplicity: it is the cheapest way to train a sequence labeler with no frame labels. The price of that simplicity is the exact set of bugs this post is about, and none of them exist in the attention world. Knowing *which* loss you are using tells you which family of failures to expect before you have seen a single symptom.

## 2. The forward-backward recursion and where inf comes from

Let me make the length constraint rigorous, because the most expensive CTC bug — the one that wastes whole runs — is the `inf` loss, and `inf` is not a numerical accident. It is the dynamic program reporting that the set $\mathcal{B}^{-1}(\mathbf{y})$ is empty: there is *no* alignment of $T$ frames that collapses to your target. To see exactly when that happens, we need the extended label sequence and the lattice.

CTC builds an **extended label sequence** $\mathbf{y}'$ from your target $\mathbf{y}$ by inserting a blank between every pair of labels and at both ends. For the target `cat` (length $U=3$), the extended sequence is `- c - a - t -`, which has length $2U + 1 = 7$. The forward variable $\alpha_t(s)$ is the total probability of all paths that, by frame $t$, have produced the first $s$ symbols of the extended sequence $\mathbf{y}'$. The recursion lets a path at frame $t$, position $s$, come only from a small set of previous positions at frame $t-1$:

$$
\alpha_t(s) = \big(\alpha_{t-1}(s) + \alpha_{t-1}(s-1) + \mathbb{1}[\,\mathbf{y}'_s \neq \text{blank} \text{ and } \mathbf{y}'_s \neq \mathbf{y}'_{s-2}\,]\,\alpha_{t-1}(s-2)\big)\, p_t(\mathbf{y}'_s)
$$

The first two terms (stay at $s$, or advance by one) are always allowed. The third term (skip a blank, advancing by two) is allowed only when the current symbol is a real label *and* it differs from the label two positions back — this is precisely the rule that forces a blank between two identical labels (the "hello" case). The total probability is read off the last frame: $P(\mathbf{y} \mid X) = \alpha_T(2U+1) + \alpha_T(2U)$, the two ways a valid path can finish (on the final label or on the trailing blank).

Now the constraint falls out of the geometry. Look at figure 2: the lattice has one row per extended-label symbol and one column per frame, and a valid path is a monotone walk from the top-left region to the bottom-right that moves only rightward (advance a frame) and downward (advance a label position), respecting the skip rule. To produce all $U$ labels, the path must descend far enough down the rows. **Each real label needs at least one frame, and any two identical adjacent labels need at least one extra frame for the mandatory blank between them.** If $r$ is the number of adjacent-duplicate label pairs in your target, then the minimum number of frames needed is:

$$
T_{\min} = U + r
$$

![A row-by-column lattice grid showing one extended-label symbol per row and one frame per column with a monotone path from the first blank down through c, a, and t to the collapsed output cat](/imgs/blogs/debugging-ctc-and-alignment-2.png)

For `cat` there are no adjacent duplicates, so $r = 0$ and you need at least $T = 3$ frames. For `hello`, the `ll` is one duplicate pair, so $r = 1$ and you need at least $T = 6$ frames to spell five letters. For `bookkeeper` — `oo`, `kk`, `ee` are three duplicate pairs — you need $U + r = 10 + 3 = 13$ frames. If your input has fewer than $T_{\min}$ frames, then *no monotone path can reach the bottom-right corner*, the set of valid alignments is empty, $P(\mathbf{y} \mid X) = 0$, and the loss is $-\log 0 = +\infty$. That is the entire mechanism of the inf-loss trap. PyTorch's `nn.CTCLoss` returns `inf` for that sample (and, in older default behavior, a NaN gradient that then poisons your weights), exactly as the math requires.

Figure 3 below frames the trap and its fix as a before–after: a target that needs 10 frames, a 4× downsampling stack that only leaves 7, no valid path, `inf` at step 1 — and the fixed version with less downsampling, a length assert, and a finite loss of 3.1.

![A two-column before and after figure showing a target needing ten frames downsampled to seven producing inf loss versus less downsampling giving fourteen frames and a finite loss](/imgs/blogs/debugging-ctc-and-alignment-3.png)

The deeply practical consequence: **CTC has a hard, checkable precondition, and almost nobody checks it until a run dies.** The most common way to violate it is *downsampling*. Speech and OCR encoders downsample aggressively in time — a wav2vec2-style CNN front end might reduce a 16 kHz waveform to one frame every 20 ms (a 320× reduction in samples), and a Conformer or a stacked-stride CNN can downsample the resulting feature sequence by another 4× or 8×. If your raw input is short and your transcript is long — a fast speaker, a noisy clip trimmed to 0.8 seconds, a long word in a short utterance — the post-downsampling frame count $T$ can fall below $U + r$, and that single sample throws `inf`. It does not matter that 99.9% of your batches are fine; one bad sample makes the batch loss `inf`, the gradient `nan`, and (without guardrails) your whole model `nan` one step later. We will write the assert that catches it in section 4.

#### Worked example: counting the minimum frames

Take the target word `mississippi`. Spell it out and count adjacent duplicate pairs: `m i s s i s s i p p i` — the pairs are `ss` (positions 3-4), `ss` (positions 6-7), and `pp` (positions 9-10), so $r = 3$. The length is $U = 11$. The CTC minimum is $T_{\min} = U + r = 11 + 3 = 14$ frames. Now suppose your audio for this word is 0.9 seconds, your feature extractor produces one frame every 20 ms (so 45 frames), and your encoder downsamples by 4× — leaving $T = \lfloor 45 / 4 \rfloor = 11$ frames. You have 11 frames and you need 14. The loss for this sample is `inf`, deterministically, every epoch, until you either lower the downsampling factor (4× → 2× gives $T = 22$, comfortably above 14) or ensure such short-and-dense samples are filtered or padded. If you had only checked the word "cat" by hand you would never have seen it; the bug lives in the long words spoken quickly. This is why you assert over the *whole dataset*, not a sample you eyeball.

There is a subtlety in the constraint that bites people who use space or word-piece tokens. The repeat count $r$ is computed over the *symbol sequence CTC actually sees*, which is your tokenized target, not the raw text. If your tokenizer splits "mississippi" into sub-word pieces, the adjacent-duplicate pairs are counted over the piece ids, not the letters — two different pieces that happen to render the same character are *not* a duplicate and need no separating blank, while two occurrences of the same piece id *are* and do. The practical failure mode is a character-level model where someone forgot that the space between words is itself a token: the transcript "a a a" (three single-letter words) tokenizes as `a <space> a <space> a`, which has no adjacent duplicates at the token level (the spaces separate them), so $r = 0$ and $T_{\min} = 5$. But if your tokenizer drops the space token, it becomes `a a a` with two adjacent-duplicate pairs, $r = 2$, $T_{\min} = 5$ as well by coincidence here — the point is that whether the space is a token changes the count, and you must compute $r$ on exactly the id sequence you pass to the loss. When you write the precondition-4 assert, compute `repeats` on the *padded-and-tokenized targets tensor*, never on the original string, or your guard will be checking the wrong number.

#### Worked example: the inf that only appears at a longer max-length

A team trained a character-level OCR model — text recognition on cropped word images, the visual cousin of speech ASR, same CTC loss — at an input image width that gave $T = 24$ output columns. Everything was clean for weeks. Then they raised the dataset's `max_label_length` filter from 20 to 30 characters to include longer words, and the run started throwing `inf` intermittently. The mechanism is exactly the length constraint: at the old cap, the longest target was 20 characters with at most a few duplicate pairs, so $T_{\min} \leq 23 < 24$ and every sample fit. At the new cap, a 28-character word with three doubled letters needs $T_{\min} = 28 + 3 = 31$ columns, but the images still produced only 24. The fix was not to revert the cap (the long words were legitimate) but to widen the input image resolution so the CNN produced more output columns ($T = 40$), restoring $T \geq T_{\min}$ for the new label distribution. The lesson: the length constraint is a *joint* property of your downsampling and your label-length distribution, and a change to either side — a longer label cap, a more aggressive downsample — can silently push the tail of your distribution below $T_{\min}$. Re-run the dataset-wide assert whenever you touch either.

## 3. The blank token, the collapse rule, and the all-blank phase

The blank token is the second great source of CTC bugs, and they come in two flavors: an **index mismatch** (the blank is at a different integer in different parts of your code) and the **all-blank collapse** (the model emits only blank). The first is a bookkeeping bug; the second is a numerics-and-optimization phenomenon you need to recognize so you do not chase it as if it were a bug when it is a phase.

Start with the index. `nn.CTCLoss` takes a `blank` argument that defaults to `0`. That single integer has to mean the same thing in three places that are often written by three different people (or three different libraries):

1. **The model's output layer.** Your final linear projects to `C` classes; one of those columns is the blank class. Which column? Whatever you decided. wav2vec2's CTC head, following the convention of the Hugging Face tokenizer, typically puts the pad/blank token at a specific index defined by the tokenizer's vocabulary, often index 0 or a dedicated `<pad>` id.
2. **The loss.** `nn.CTCLoss(blank=K)` treats column `K` of your log-probs as the blank class. If you leave it default, `K = 0`.
3. **The decoder.** When you greedily decode, you take the argmax per frame and then run the collapse rule, which *removes the blank index*. The decoder needs to know which index to remove.

If these three disagree, the failure is silent and total. Suppose your tokenizer maps `c → 0, a → 1, t → 2`, and your model reserves the *last* index `C-1` for blank — but you call `nn.CTCLoss()` with the default `blank=0`. Now the loss treats your character `c` (index 0) as the blank token. It will happily train: it pushes up the probability of "blank" (your `c` column) on the frames between labels, and it tries to spell your target using indices `1..C-1` as if they were the real alphabet shifted by one. The result is a model whose outputs are internally consistent with a wrong alphabet, so when you decode by removing index `C-1` (your actual blank) instead of index 0, you get garbage. The training loss can even *descend* — the optimization is solving a coherent but wrong problem. Figure 6 lays this out as a before–after: model blank at `C-1`, loss assuming `blank=0`, the wrong column trained, WER 100%; then both set to blank `=0` with labels shifted to `1..C`, alignment real, WER 9.1%.

![A before and after figure contrasting a model that reserves blank at the last index against a loss assuming blank at index zero with the corrected version where both agree on index zero](/imgs/blogs/debugging-ctc-and-alignment-6.png)

A closely related bug: **the blank symbol appearing inside your label sequence.** Your targets are supposed to be real labels only — indices in `1..C-1` if blank is 0, never the blank index itself. If a preprocessing step accidentally inserts the blank id into a target (for example, a tokenizer that emits a `<pad>` id mid-sequence, or a `0` that should have been a real character), CTC's accounting breaks: the extended-label construction assumes your raw labels contain no blanks, and a blank inside the target makes the dynamic program meaningless. The fix is an assert that no target contains the blank index; the bug is easy to introduce when blank and pad share an id.

Now the all-blank collapse, which is *not* an index bug — it is what a correctly-wired CTC model does early in training. Ask yourself: at step 0, with random weights, what is the single safest thing the network can output to get a not-terrible loss? Consider the loss landscape. The target probability $P(\mathbf{y} \mid X)$ is a sum over alignments, and *every* alignment is mostly blank — for a 100-frame clip of a 5-character word, even the "busiest" alignment has only a handful of non-blank frames and ~95 blank frames. So the gradient, averaged over all the alignments CTC is summing, points strongly toward "emit blank on most frames." Worse, blank is the one class that appears in *every* target's alignments, so it gets a consistent upward push from every sample, while each real character gets push from only the samples that contain it. The rational early behavior is therefore: **drive the blank probability up everywhere first, then slowly carve out the non-blank spikes where the labels actually are.** Figure 7 shows this as a timeline — random at step 0, all-blank by step 200, a single repeated token by step 800, first real spikes by step 2k, sharp alignment by step 8k — with the failure branch (stuck all-blank, LR too high) called out in red.

![A left-to-right timeline of CTC training phases moving from random output to all-blank to a single repeated token to real spikes to sharp alignment with a failure branch stuck at all-blank](/imgs/blogs/debugging-ctc-and-alignment-7.png)

This is the single most important fact for not panicking: **an all-blank output for the first few hundred to few thousand steps is normal CTC behavior, not a bug.** The bug is when it *stays* all-blank — when at step 10,000 your argmax is still uniformly blank and the loss has plateaued near $-\log(\text{prior})$. That stuck state has a small set of causes: a learning rate so high that the non-blank logits never get a stable foothold (the most common), a blank-index mismatch (you are training the wrong column), inputs that are not log-probabilities (so the loss is reading garbage), or a length problem that makes most batches `inf` and the survivors uninformative. The "single repeated token" failure — the model emits, say, `e e e e e e` and collapses to a single `e` — is the same phenomenon one notch along: the network found the single most frequent character and is hedging on it instead of blank. Both are escaped by the same first move: lower the learning rate and add warmup, then re-check the index and the log-softmax.

Let me make the "blank dominates the gradient" claim rigorous instead of hand-waving it, because once you have seen the gradient you will never mistake the all-blank phase for a bug again. The CTC gradient with respect to the pre-softmax logit $a_t^k$ (class $k$ at frame $t$) has a beautifully clean form. Define the per-frame, per-class *occupancy* — the posterior probability, under the CTC model, that frame $t$ is assigned to symbol $k$ summed over all alignments — as $\gamma_t(k)$, computed from the forward and backward variables as $\gamma_t(k) = \frac{1}{P(\mathbf{y} \mid X)}\sum_{s : \mathbf{y}'_s = k} \alpha_t(s)\beta_t(s)$. Then the gradient of the loss with respect to the logit is exactly:

$$
\frac{\partial \mathcal{L}_{\text{CTC}}}{\partial a_t^k} = p_t(k) - \gamma_t(k)
$$

This is the same softmax-minus-target form as ordinary cross-entropy, except the "target" $\gamma_t(k)$ is not a hard one-hot label — it is the *soft, alignment-marginalized occupancy* that CTC computes on the fly from the lattice. Now reason about $\gamma_t(\text{blank})$ at step 0 with random weights. Every alignment in $\mathcal{B}^{-1}(\mathbf{y})$ is mostly blank (a 100-frame clip of a 5-character word has at least 95 blank frames in every legal path), so summed over alignments, the blank occupancy $\gamma_t(\text{blank})$ is large for the vast majority of frames. The gradient $p_t(\text{blank}) - \gamma_t(\text{blank})$ is therefore strongly negative (push the blank logit *up*) on most frames, for *every* sample in the batch. By contrast, $\gamma_t(c)$ for a real character $c$ is non-negligible only on the few frames and the few samples where $c$ actually occurs, so its push is sparse and inconsistent across the batch. Averaged over a batch, the blank logit gets a coherent, reinforcing upward gradient while each content logit gets a weak, intermittent one. **The all-blank attractor is not a pathology; it is the literal direction of steepest descent at initialization.** A model escapes it only once the content occupancies $\gamma_t(c)$ at the right frames grow enough to overcome the blank push — which a moderate, warmed-up learning rate allows and an excessive one prevents (the content logits get knocked around faster than they can stably climb). This is *why* the cure for stuck-all-blank is almost always "lower the LR," and now you can see it is not a folk remedy but a consequence of the gradient's structure.

## 4. The diagnostic: assert the CTC contract before the loss

Here is the discipline that turns CTC from a source of mystery runs into a tractable one. `nn.CTCLoss` returns a meaningful number only when *five* preconditions all hold, and every one of them is a one-line assertion you can run before the loss. Figure 5 stacks them: inputs are log-probs, shape is `(T, N, C)`, lengths are true not padded, $T \geq U + r$ for every sample, and the blank index agrees across loss and decode (plus targets exclude the blank). Install these once and you will never lose a run to a silent CTC bug again.

![A vertical stack of five CTC preconditions from log-probabilities through time-first shape true lengths the length assertion and a consistent blank index leading to a finite trainable loss](/imgs/blogs/debugging-ctc-and-alignment-5.png)

Let me take them one at a time, with the exact PyTorch API.

**Precondition 1: the inputs are log-probabilities.** `nn.CTCLoss` expects `log_probs`, not logits and not probabilities — it does *not* apply a softmax internally. If you pass raw logits, the loss is computed on un-normalized scores and is silently wrong (often it still descends, which is the trap). You must apply `F.log_softmax` over the class dimension. This is the single most common "my CTC loss is weird but not inf" bug.

**Precondition 2: the shape is `(T, N, C)`.** PyTorch's `nn.CTCLoss` wants the log-probs as `(input_length, batch, n_classes)` — time first, batch second. Almost every other PyTorch loss and almost every model output is batch-first `(N, T, C)`. If you forget the transpose, you do not get an error (the shapes are often compatible enough to run); you get a loss that is treating your batch dimension as time and your time dimension as batch, which is nonsense that may or may not be `inf`.

**Precondition 3: lengths are true, not padded.** You pass `input_lengths` (the real number of valid frames per sample, before padding) and `target_lengths` (the real number of labels per sample, before padding). If you pass the padded length, CTC counts pad frames and pad labels as real, which both breaks the length math and trains the model to align against padding.

**Precondition 4: $T \geq U + r$ for every sample.** This is the inf-loss guard from section 2.

**Precondition 5: the blank index is consistent**, and targets never contain it.

Here is a single setup-and-guard block you can drop into a training step.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

BLANK = 0  # one source of truth, used by loss AND decoder

ctc_loss = nn.CTCLoss(blank=BLANK, zero_infinity=False, reduction="mean")

def ctc_step(logits, targets, input_lengths, target_lengths):
    """
    logits:         (N, T, C) raw model outputs, batch-first
    targets:        (N, Lmax) padded label ids, or a 1D concatenated tensor
    input_lengths:  (N,) true frames per sample (after the encoder's downsampling)
    target_lengths: (N,) true labels per sample (before padding)
    """
    N, T, C = logits.shape

    # --- Precondition 5: targets must not contain the blank id ---
    assert (targets != BLANK).all() or target_lengths.sum() == 0, \
        "A target contains the blank index; blank must never appear in labels."

    # --- Precondition 4: T >= U + repeats, per sample (the inf-loss guard) ---
    for i in range(N):
        y = targets[i, : target_lengths[i]]
        repeats = (y[1:] == y[:-1]).sum().item() if len(y) > 1 else 0
        t_min = int(target_lengths[i].item()) + repeats
        assert input_lengths[i].item() >= t_min, (
            f"sample {i}: T={input_lengths[i].item()} < U+reps={t_min} "
            f"(U={target_lengths[i].item()}, reps={repeats}); CTC loss will be inf. "
            f"Reduce downsampling or filter/pad this sample."
        )

    # --- Precondition 1: log-probabilities, not logits, not probs ---
    log_probs = F.log_softmax(logits, dim=-1)   # over the class axis

    # --- Precondition 2: nn.CTCLoss wants (T, N, C), time-first ---
    log_probs = log_probs.transpose(0, 1).contiguous()  # (T, N, C)

    # --- Precondition 3: lengths are the TRUE lengths, dtype long ---
    input_lengths = input_lengths.to(torch.long)
    target_lengths = target_lengths.to(torch.long)

    loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
    assert torch.isfinite(loss), "CTC loss is non-finite even after the guards."
    return loss
```

Notice the deliberate choice `zero_infinity=False`. The `zero_infinity` flag in `nn.CTCLoss`, when set to `True`, replaces `inf` losses (and their `nan` gradients) with zero for the offending samples. That is a *band-aid*: it stops the run from dying, but it silently drops those samples from training, so a chunk of your hard, fast-speech examples never contribute a gradient and your model quietly underperforms on exactly them. The right discipline is to debug with `zero_infinity=False` first, so the bad samples loudly throw and you *find* them with the assert above, fix the real cause (less downsampling, or filter the truly-impossible samples), and only then — if you have a small irreducible tail of bad samples — consider `zero_infinity=True` as a production safety net with the count of zeroed samples logged so you know how many you are dropping.

#### Worked example: the assert that saved a 14-hour run

A team finetuning a Conformer-CTC model on a 2,000-hour corpus kept getting `nan` loss "randomly" — sometimes at step 800, sometimes at step 3,000, never reproducible because the dataloader shuffled. Each crash burned several GPU-hours of wasted compute before someone noticed the dashboard had flatlined, plus the babysitting cost. They added the precondition-4 assert above and ran one epoch with `zero_infinity=False`. The assert fired on 41 samples out of 1.1 million — fast-speech clips with long transcripts where the 4× downsampling left $T < U + r$. The "random" NaN was these 41 samples being shuffled into different positions each run. The fix was two lines: filter the 41 samples at dataset-build time (they were transcription-misaligned anyway, a few were truncated audio) and lower one downsampling stage from stride 2 to stride 1 for safety. The run went from "NaN at a random step every few hours" to clean for the full 14-hour epoch, and final WER improved by 0.6 points because the formerly-`inf` batches now contributed real gradients instead of dying. The assert cost microseconds per step; the bug had been costing the team a day a week.

## 5. Visualizing the alignment to catch the collapse

The length assert catches `inf`. To catch the *all-blank collapse* and the *garbage transcript* you need to look at the alignment directly, because the loss number alone cannot distinguish "healthy, still in the all-blank phase" from "stuck forever in all-blank." The instrument is the **per-frame argmax**: for each frame, which class does the model think is most likely? Plotting or printing that sequence tells you immediately whether the model is emitting only blank, only one repeated token, or real, sparse spikes of content.

```python
@torch.no_grad()
def inspect_ctc_alignment(model, batch, idx2char, blank=0, n_show=2):
    """Print per-frame argmax and the greedy-collapsed decode for a few samples."""
    model.eval()
    logits = model(batch["input_values"])          # (N, T, C)
    log_probs = F.log_softmax(logits, dim=-1)
    argmax = log_probs.argmax(dim=-1)              # (N, T) per-frame class id

    for i in range(min(n_show, argmax.size(0))):
        frames = argmax[i].tolist()
        # fraction of frames that are blank: the collapse-health signal
        blank_frac = sum(f == blank for f in frames) / len(frames)
        # number of distinct non-blank classes the model is actually using
        non_blank = sorted({f for f in frames if f != blank})

        # greedy CTC decode: merge repeats, then drop blanks
        collapsed, prev = [], None
        for f in frames:
            if f != prev:            # merge consecutive repeats
                if f != blank:       # then remove blanks
                    collapsed.append(f)
                prev = f
        text = "".join(idx2char[c] for c in collapsed)

        print(f"[sample {i}] blank_frac={blank_frac:.2f}  "
              f"distinct_nonblank={len(non_blank)}  decode={text!r}")
        # a healthy run: blank_frac ~0.7-0.9, distinct_nonblank grows, decode is real text
        # all-blank collapse: blank_frac == 1.00, distinct_nonblank == 0, decode == ''
        # single-token collapse: distinct_nonblank == 1, decode is one repeated char
    model.train()
```

This 25-line function is the highest-leverage CTC instrument there is. Run it every few hundred steps (or log `blank_frac` and `distinct_nonblank` to W&B as scalars) and the three failure modes become visually obvious:

- **All-blank collapse:** `blank_frac == 1.00`, `distinct_nonblank == 0`, `decode == ''`. Early it is fine; if it persists past a few thousand steps, your LR is too high or your inputs/index are wrong.
- **Single-token collapse:** `distinct_nonblank == 1`, decode is one character repeated. The optimizer found the most frequent character and is hedging; same fixes.
- **Healthy alignment:** `blank_frac` settling around 0.7–0.9 (most frames are blank by design — that is correct), `distinct_nonblank` climbing toward your real vocabulary size, decode producing recognizable text.

The reason `blank_frac` around 0.8 is *healthy* and not a bug deserves emphasis, because people see "80% blank" and panic. In a typical speech setup with one frame every 20 ms, a spoken character occupies maybe 60–120 ms, so the network legitimately emits a content spike for a couple of frames and blank for the rest. A correctly-trained CTC model is *supposed* to be mostly blank with sharp, sparse spikes — that "peaky" behavior is well documented and is a known property of CTC, not a defect. The diagnostic signal is not the absolute blank fraction; it is whether `distinct_nonblank` is growing and the decode is converging to real text. A flat `distinct_nonblank == 0` is the bug; a high-but-stable `blank_frac` with growing content is health.

For a richer view, save the full posterior matrix `log_probs[i]` of shape `(T, C)` and render it as a heatmap (frames on the x-axis, classes on the y-axis). A healthy CTC model produces a near-solid bright row for the blank class with sharp vertical spikes of content where each label is emitted — the "picket fence" that gives CTC its peaky reputation. A collapsed model produces a single solid bright blank row and nothing else. You do not need a fancy plot to debug; the `blank_frac`/`distinct_nonblank`/`decode` triple from the function above is enough to tell health from collapse in one glance at the log.

### A complete, instrumented torchaudio CTC setup

It helps to see all five preconditions plus the alignment instrument wired into one small, real training step on a torchaudio model, so you can copy the *shape* of a debuggable CTC loop rather than re-deriving it. The key API facts: torchaudio ships pretrained ASR bundles (for example `torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H`) whose model exposes the encoder's length transform, and PyTorch's `nn.CTCLoss` is the loss. The single most important practical detail is computing the *post-encoder* output length per sample — the encoder downsamples, so the `input_lengths` you pass the loss are not the raw frame counts but the encoder's output frame counts, which the model can tell you.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

BLANK = 0  # one constant: model head reserves col 0, loss + decoder agree

class CTCTrainer:
    def __init__(self, encoder, n_classes, lr=3e-5):
        self.encoder = encoder                       # outputs (N, T, C) logits
        self.ctc = nn.CTCLoss(blank=BLANK, zero_infinity=False, reduction="mean")
        self.opt = torch.optim.AdamW(encoder.parameters(), lr=lr)

    def output_lengths(self, raw_lengths):
        """Map raw input lengths to post-downsampling frame counts.
        Use the model's documented transform; here a generic stacked-stride form."""
        lengths = raw_lengths.clone()
        for kernel, stride in [(10, 5), (3, 2), (3, 2)]:   # example conv front end
            lengths = (lengths - kernel) // stride + 1
        return lengths.clamp_min(1)

    def step(self, waveforms, raw_lengths, targets, target_lengths, train=True):
        logits = self.encoder(waveforms)                    # (N, T, C) batch-first
        N, T, C = logits.shape
        input_lengths = self.output_lengths(raw_lengths)    # TRUE post-encoder T

        # length guard: assert T_i >= U_i + repeats for every sample (catch inf early)
        for i in range(N):
            y = targets[i, : target_lengths[i]]
            reps = (y[1:] == y[:-1]).sum().item() if len(y) > 1 else 0
            assert input_lengths[i].item() >= int(target_lengths[i]) + reps, (
                f"sample {i}: T={input_lengths[i].item()} "
                f"< U+reps={int(target_lengths[i]) + reps}; CTC would be inf"
            )

        log_probs = F.log_softmax(logits, dim=-1)           # log-probs, not logits
        log_probs = log_probs.transpose(0, 1).contiguous()  # (T, N, C) time-first
        loss = self.ctc(log_probs, targets,
                        input_lengths.long(), target_lengths.long())
        assert torch.isfinite(loss), "non-finite CTC loss after guards"

        if train:
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.encoder.parameters(), 5.0)
            self.opt.step()
        return loss.item(), log_probs.detach()
```

Three things in this loop are doing real debugging work and are worth calling out. First, `output_lengths` is the part people get wrong: it does *not* return the raw lengths, it pushes them through the same integer arithmetic the encoder's strided convolutions apply, so the `input_lengths` handed to the loss are the true post-downsampling frame counts. Get this transform wrong and you reproduce the padded-length bug from section 6 even though you "computed a length." Second, the length guard runs *before* the loss, so a too-short sample throws a readable assert that names the sample and the numbers, not an `inf` three operations later. Third, `clip_grad_norm_` at 5.0 is cheap insurance against the gradient spike that can accompany the transition out of the all-blank phase; CTC gradients can jump when content occupancies first appear, and clipping keeps that transition from destabilizing a hot learning rate. None of this is exotic — it is the generic "read the instruments, guard the preconditions" discipline specialized to the CTC contract.

## 6. Length bookkeeping and the shape order

Two of the five preconditions — the `(T, N, C)` shape and the true lengths — deserve their own section, because they are the bugs that do *not* throw `inf` and do *not* collapse to blank. They produce a loss that looks plausible, descends a little, and yields a model that is subtly wrong. They are the hardest CTC bugs precisely because nothing screams.

**The shape order.** `nn.CTCLoss` is the odd one out in PyTorch: it wants `log_probs` as `(T, N, C)`, time-first. Your model almost certainly outputs `(N, T, C)`, batch-first, like every transformer and CNN in the ecosystem. The fix is a single `transpose(0, 1)`, but the bug when you forget it is insidious. If $T \neq N$, you will usually get a shape error somewhere and find it fast. But if your batch size happens to equal your sequence length — which is not as rare as it sounds in small debugging runs where you set both to, say, 16 — the loss runs without error and computes complete nonsense, treating each of your 16 batch elements as a time step of a single 16-frame sequence. The guard is to make the transpose explicit and assert the shape right before the loss, as in the setup block in section 4. Never rely on "it ran without an error" for CTC shapes.

**True lengths versus padded lengths.** Batches are rectangular, so short utterances get padded with zeros to the length of the longest in the batch, and short transcripts get padded labels. But CTC must know the *real* length of each sample — `input_lengths[i]` is the count of valid frames *after* the encoder's downsampling, and `target_lengths[i]` is the count of real labels before padding. If you pass the padded lengths instead (a very common copy-paste error where someone passes `torch.full((N,), T)`), three things go wrong at once: the length constraint check (CTC's internal one) uses wrong numbers, the model is trained to align real content against pad frames, and the gradient is computed over alignments that include the padding region. The loss will be finite and will descend — the model learns *something* — but it learns to spread content into the padded tail, and at inference, where there is no padding, the alignment is shifted and the WER is mysteriously high. The table below is the field guide for these "finite but wrong" bugs.

| Bug | Throws inf or NaN? | Loss looks normal? | Confirming test | Fix |
|---|---|---|---|---|
| `T < U + r` (downsampling too aggressive) | Yes (inf → NaN) | No, dies | Precondition-4 assert fires | Less downsampling, or filter/pad sample |
| Inputs are logits, not log-probs | Usually no | Yes, descends slowly | Check whether `F.log_softmax` was applied | Apply `F.log_softmax(dim=-1)` |
| Shape passed as `(N, T, C)` | Sometimes (random) | Sometimes | Assert shape is `(T, N, C)` before loss | `log_probs.transpose(0, 1)` |
| Padded lengths instead of true lengths | No | Yes, descends | Print `input_lengths` vs `T`; should differ | Pass true per-sample lengths |
| Blank index mismatch (loss vs model) | No | Yes, can descend | Decode is garbage despite low loss | One `BLANK` constant everywhere |
| Blank id appears inside targets | Sometimes | No | Assert `(targets != BLANK).all()` | Remove blank id from label construction |
| LR too high (stuck all-blank) | No | Plateaus high | `distinct_nonblank == 0` past 5k steps | Lower LR, add warmup |
| Decode collapse ≠ train collapse rule | No (only at eval) | Yes | Re-decode with merge-then-blank rule | Match the canonical collapse rule |

Figure 4 renders the core of this table as a matrix — symptom row, confirming-test column, fix column — so you can route from "loss = inf" or "all-blank output" or "garbage transcript" to the one test and the one fix without re-reading the prose.

![A matrix mapping five CTC symptoms each to a confirming test and a single fix covering inf loss all-blank output garbage transcript repeated token and high WER](/imgs/blogs/debugging-ctc-and-alignment-4.png)

#### Worked example: the padded-length bug that cost 4 WER points

A model trained cleanly — finite loss, descending nicely from 3.4 to 0.9 over 20k steps, alignment heatmap showing real spikes — but its test WER was 13.2% while a published baseline on the same data was around 9%. No crash, no collapse, nothing obviously wrong. The bisection: overfit a single batch (it overfit fine, ruling out the model), then look at the instruments. The smoking gun was printing `input_lengths` next to the actual frame count `T`: someone had set `input_lengths = torch.full((N,), T)` — the padded length for every sample — instead of computing the true post-downsampling length per sample. The model had been trained to consider alignments that ran into the zero-padded tail of every short utterance, learning to emit content late. The fix was to compute the real length: take each sample's true frame count before padding, apply the same integer downsampling the encoder applies (`(raw_len - kernel) // stride + 1` per conv stage, or the model's documented length-transform), and pass that. After the fix, the alignment spikes moved earlier and tightened, and WER dropped from 13.2% to 9.1% — 4.1 points — with no change to the model or the data, only to four characters of length bookkeeping. The loss curve before and after was nearly identical; this bug was invisible in the loss and visible only in the WER and the alignment timing.

## 7. The full bisection on the running example

Let us run the whole method on the failing run from the introduction: the wav2vec2 finetune that printed `inf` then `nan`, and decoded every clip to the empty string. We will bisect through the six places — but for CTC the live suspects are *numerics* (the inf) and *model code / data bookkeeping* (the blank, lengths, log-softmax, decode), so we go there first. Figure 8 is the decision tree we are walking: from "CTC run broken" split on inf-versus-finite, then on blank-versus-content, down to one root cause and one fix.

![A decision tree starting from a broken CTC run splitting on inf versus finite loss then blank versus wrong content down to length fixes and learning-rate or blank-index fixes](/imgs/blogs/debugging-ctc-and-alignment-8.png)

**Step 1 — Is the loss inf or finite?** The log shows `inf` starting at step 1,200, before the `nan`. That immediately routes us to the *length* branch of the tree. (The `nan` at step 1,210 is downstream: the `inf` loss produced a `nan` gradient, which the optimizer wrote into the weights, after which every forward pass is `nan`. The `inf` is the original sin; the `nan` is the spread. This matches the general NaN-spread mechanism — find the *first* non-finite value, not the corpse.)

**Step 2 — Confirm the length constraint.** We add the precondition-4 assert with `zero_infinity=False` and run with shuffling off so it is reproducible. The assert fires: a batch around step 1,200 contains a 0.7-second clip of a fast speaker saying a nine-word phrase, downsampled by the encoder to $T = 34$ frames against a transcript that, extended, needs $U + r = 39$. There it is: $34 < 39$, no valid path, `inf`. The "random" timing was the shuffle placing this sample in different batches each run.

**Step 3 — Fix the length, re-run.** Two changes: (a) at dataset-build time, compute $T_{\min} = U + r$ for every sample using the encoder's true length transform, and drop or flag the handful where even un-downsampled audio cannot satisfy it (these turn out to be misaligned transcripts); (b) reduce one downsampling stage so typical fast-speech clips keep enough frames. The `inf` is gone, the loss is finite at 3.4, the run no longer dies. But — and this is why you do not stop at the first fix — the decode is *still* the empty string. The model trains without crashing and outputs all-blank. We are now on the *finite-loss* branch of the tree.

**Step 4 — Blank or wrong content?** We run `inspect_ctc_alignment` from section 5. Output: `blank_frac=1.00, distinct_nonblank=0, decode=''` at step 3,000. All-blank, well past the normal early phase. The tree now splits between "LR too high" and "blank-index mismatch." We check the cheap one first.

**Step 5 — Check the blank index.** We grep for every place blank is referenced. The tokenizer's `<pad>` id (used as blank) is at index `0`. The model's CTC head outputs `C` classes with — we trace it — blank intended at index `0` too. But the `nn.CTCLoss` was constructed as `nn.CTCLoss()` with the default `blank=0`, which *happens to match*, so index is not the bug here. Good — eliminate it. (Had they disagreed, this would have been the fix: one `BLANK = 0` constant feeding the head, the loss, and the decoder.)

**Step 6 — Check log-softmax and then the LR.** We confirm `F.log_softmax(dim=-1)` is applied before the loss (it is — precondition 1 holds). That leaves the LR. The finetune used a constant `5e-4`, which for a pretrained wav2vec2 backbone is far too hot — finetuning these models typically wants `1e-5` to `5e-5` with warmup, often 50–100× lower than a from-scratch run. A too-high LR keeps the non-blank logits from ever getting a stable foothold above the dominant blank gradient, so the model parks at the all-blank optimum and never leaves. We drop the LR to `3e-5` with 500 steps of linear warmup.

**Step 7 — Re-run and read the instruments.** Now `inspect_ctc_alignment` tells the healthy story: step 200 `blank_frac=1.00` (the normal early all-blank phase), step 800 `distinct_nonblank=1` (a single repeated token — the next phase), step 2,000 `distinct_nonblank=18` with `decode='the cat sat'`-style real text, step 8,000 `blank_frac=0.82, distinct_nonblank=29, decode` matching the reference closely. The held-out WER lands at 9.1%. The run that was `inf`-ing and decoding to nothing is now a working ASR model. Two root causes (a length-constraint violation and a too-high LR), two confirming tests (the length assert and the alignment inspector), two fixes — found by bisection in the order the tree prescribes, not by randomly changing hyperparameters.

**Stress-testing the diagnosis.** What if it had been data, not numerics? If the assert had *not* fired and the loss had been finite from the start but still all-blank, the suspect shifts to the index or the log-softmax (model code) or the LR (optimization), exactly as the tree routes. What at fp16? CTC under AMP is a real hazard: the forward-backward involves log-sum-exp over many small probabilities, and fp16's narrow range makes underflow more likely, which can manufacture `inf`/`nan` independent of the length constraint — prefer bf16 for CTC, or keep the loss computation in fp32 even under autocast. What if the batch is tiny? A batch of one makes the all-blank phase look like a permanent collapse for longer (less gradient signal per step); do not judge collapse-versus-phase on a batch of one — use the step count, not the batch. What if it only fails on multi-GPU? Then suspect length-bucketing or a per-rank data shard that concentrates the short-and-dense samples on one rank; the `inf` would appear only on that rank and desync the run.

### Why CTC finetuning fails differently from training from scratch

Finetuning a pretrained CTC backbone — wav2vec2, HuBERT, a Conformer checkpoint — fails in a few CTC-specific ways that a from-scratch run does not, and they are worth naming because the brief for this whole series is that finetuning deserves first-class treatment. The headline difference is the learning rate. When you train CTC from scratch, the whole network is random and a moderately warm LR is fine. When you finetune, the *encoder* already produces excellent speech representations and only the new CTC head is random, so a too-high global LR does something specific and destructive: it lets the large, random-headed gradients flow back and *destroy the pretrained features* before the head has learned to use them. The signature is a run that decodes worse after 2,000 steps of finetuning than the zero-shot pretrained model did — you have un-learned the representation. The fixes are the standard transfer-learning ones, but CTC sharpens them: use an LR around `1e-5`–`5e-5` (often 50–100× lower than a from-scratch CTC run), warm up over a few hundred steps so the random head settles before big updates reach the encoder, and — the wav2vec2-specific trick — *freeze the feature encoder* (the CNN front end) for the entire finetune or at least the first several thousand steps, since those low-level features rarely need to move and freezing them removes a whole class of instability. If your finetune gets stuck all-blank, the order of suspects is: LR too high, then feature encoder not frozen, then blank index, then length constraint.

A second finetuning-specific trap is the **vocabulary / blank-index mismatch between the checkpoint and your data.** A pretrained CTC model was trained with a specific vocabulary and a specific blank index baked into its head. If you swap in your own character set but reuse the old head, or you build a new head but keep the old tokenizer's blank id, you reintroduce the index mismatch from section 3 — and it is easy to do by accident because two different config files (the model's and the tokenizer's) now have to agree. The defensive move is the same one source of truth: derive `BLANK` from the tokenizer, pass that same integer to `nn.CTCLoss(blank=...)` and to the decoder, and assert at setup time that the model head's output dimension equals `len(tokenizer.vocab)` including blank. A third trap, subtle and common: finetuning often runs with `gradient_checkpointing=True` to save memory, and gradient checkpointing re-runs the forward pass during backward — which interacts badly with any in-place op or any non-determinism in the encoder and can, on some stacks, change which frames receive gradient. If your finetune trains fine without checkpointing and goes all-blank with it, that interaction is the suspect; verify by toggling checkpointing off for a short run.

#### Worked example: the finetune that un-learned its features

A team took a wav2vec2 model pretrained on 960 hours and finetuned it on 50 hours of in-domain medical dictation, expecting a WER below the zero-shot baseline of about 18%. After an epoch, WER was 41% — far *worse* than zero-shot — and the alignment inspector showed `distinct_nonblank` had spiked early then *collapsed back* toward all-blank. The loss had descended (the head was fitting the training set), but generalization had cratered. The bisection: overfit-one-batch passed (the model could memorize a batch), ruling out wiring; the length assert passed, ruling out `inf`; the blank index was consistent, ruling out the mismatch. That left optimization, and the LR was `1e-4` — ten times too hot for this backbone, with no freezing of the feature encoder. The big early gradients from the random head had flowed into the CNN front end and degraded the pretrained features faster than the head could exploit them, an effect sometimes called catastrophic forgetting of the representation. The fix was three changes: drop the LR to `2e-5`, add 500 steps of warmup, and freeze the feature encoder. WER went from 41% to 11.3% — better than the 18% zero-shot baseline, as a finetune should be — with no change to the data. The loss curves looked similar in both runs; the difference lived entirely in the eval WER and in the alignment inspector's collapse signal, which is exactly why you watch the alignment and the held-out metric, not just the training loss.

## 8. Decode-time bugs: greedy, beam, and the collapse rule

Everything so far is training-side. There is a final class of CTC bug that lives entirely at *decode* time and produces the signature "training metrics look great, deployed transcription is wrong" — a train–infer mismatch specific to CTC. The cause is that the collapse rule used at decode must match the rule CTC used during training, and there are a few ways to get it subtly wrong.

**Greedy decoding** is the simplest: take the argmax class per frame, then apply the collapse rule — *merge consecutive repeats, then remove blanks*, in that order. The order matters, and getting it backwards is a real bug. If you remove blanks *first* and then merge repeats, the word `hello` emitted as `h e l - l o` becomes (blanks removed) `h e l l o` then (merge repeats) `h e l o` — you have lost the double `l` that the blank was specifically there to preserve. The correct order keeps the blank between the two `l`s as a separator during the merge step, so they survive. Here is the canonical greedy collapse, the same logic as in the inspector:

```python
def ctc_greedy_decode(log_probs, idx2char, blank=0):
    """log_probs: (T, C) for ONE sample. Returns the decoded string."""
    ids = log_probs.argmax(dim=-1).tolist()   # per-frame argmax
    out, prev = [], None
    for k in ids:
        if k != prev:        # 1) merge consecutive repeats
            if k != blank:   # 2) then drop blanks  (ORDER MATTERS)
                out.append(k)
            prev = k
    return "".join(idx2char[k] for k in out)
```

**Beam search** with a language model (the standard for production ASR) introduces its own pitfalls. The CTC beam search must correctly merge the probabilities of paths that collapse to the same prefix — the prefix-beam-search algorithm tracks two probabilities per prefix (ending in blank versus ending in a non-blank) precisely so that repeated characters are handled correctly. A naive beam search that treats CTC outputs like a normal autoregressive softmax will mis-handle the blank and the repeats, and you will see beam search *underperform* greedy decoding, which is the tell-tale sign the beam decoder's collapse handling is wrong (a correct CTC beam search with an LM should beat greedy, not lose to it). If your beam search is worse than greedy, do not blame the beam width; check the collapse logic and the blank handling first.

A third decode-time mismatch: **the blank index again.** Your decoder removes "the blank index" — if it removes the wrong index (because the decoder was written assuming `blank=0` but your tokenizer put pad/blank elsewhere), it strips a real character on every frame and inserts blanks into the output. This is the same one-source-of-truth discipline as training: the `BLANK` constant that the loss uses must be the constant the decoder uses. The garbage-transcript-despite-low-loss signature can come from either the training-side index bug (section 3) or this decode-side one; the way to tell them apart is to decode with a known-correct blank index and see if the transcript becomes sensible — if it does, the bug was decode-side only.

| Decode bug | Signature | Test | Fix |
|---|---|---|---|
| Collapse order reversed (drop-then-merge) | Doubled letters lost (`hello`→`helo`) | Decode a word with a double letter | Merge repeats first, then drop blanks |
| Beam search mishandles blank/repeats | Beam WER worse than greedy | Compare beam vs greedy WER | Use prefix-beam-search with blank/non-blank tracking |
| Decoder strips wrong blank index | Garbage despite low train loss | Re-decode with correct blank id | One `BLANK` constant for loss and decode |
| LM weight too high | Fluent but wrong words (over-correction) | Lower LM weight, check WER curve | Tune LM weight / insertion bonus |

## 9. Case studies and real signatures

A few well-known CTC patterns, to calibrate your intuition against real systems rather than only the toy run.

**The original CTC paper (Graves et al., 2006), "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks."** This is the source of everything above — the blank token, the collapse map $\mathcal{B}$, the forward-backward recursion, and the marginalization over alignments. The paper's framing is worth internalizing: CTC is the sequence analogue of the forward algorithm in HMMs, and the inf-loss condition is the direct consequence of the lattice having no path to the corner. Reading the recursion in the paper makes the $T \geq U + r$ constraint feel inevitable rather than mysterious. The paper trained RNNs on TIMIT phoneme recognition; the same loss now powers most non-attention ASR and a great deal of OCR and handwriting recognition.

**The peaky-behavior literature.** Multiple analyses (and anyone who has rendered a CTC posterior heatmap) have documented that CTC outputs become *peaky*: the model emits blank on the vast majority of frames and concentrates each label into one or two sharp spikes. This is not a bug and not a sign of under-training; it is a property of the loss. The practical debugging consequence is the one we leaned on: a high blank fraction (0.7–0.9) is *healthy*, and the collapse diagnostic must look at whether content classes are being used and the decode is converging, not at the absolute blank fraction. Engineers who do not know this routinely "fix" a perfectly healthy model into a worse one by fighting its blank fraction.

**wav2vec2 / HuggingFace CTC finetuning.** The standard recipe (the `Wav2Vec2ForCTC` head plus `Wav2Vec2CTCTokenizer`) bundles the blank/pad token at a defined vocabulary index and computes `nn.functional.ctc_loss` with that index. The two most common community bugs reported against this recipe are exactly the ones in this post: (1) a learning rate set 10–100× too high for the pretrained backbone, producing a model stuck at all-blank (the fix is `1e-5`–`5e-5` with warmup and often freezing the feature encoder for the first steps), and (2) length mismatches when custom audio is shorter than the transcript needs after the encoder's 320× time reduction, producing `inf` loss. If you finetune wav2vec2 and get all-blank, lower the LR before anything else; if you get `inf`, check the length constraint before anything else.

**The fp16 CTC underflow.** Teams enabling AMP for an ASR run sometimes see `nan` that the length assert does not explain. The cause is numerical, not structural: the CTC forward-backward sums products of many small probabilities, and fp16's limited range underflows intermediate terms to zero, after which a `log(0)` produces `-inf` and the gradient goes `nan`. The fix is to compute the CTC loss in fp32 even under autocast (wrap the loss in an `autocast(enabled=False)` region and cast the log-probs up), or to use bf16 whose wider exponent range tolerates the small magnitudes. This is the CTC-specific instance of the general mixed-precision rule: keep reductions and log-domain math in higher precision.

**The OCR / handwriting-recognition lineage.** It is easy to think of CTC as a speech-only loss, but its single largest deployment outside ASR is text recognition in images — printed-text OCR, scene-text recognition, and handwriting recognition, all of which feed a CNN's column-wise features into a recurrent or transformer encoder and train with CTC against the character sequence. Every bug in this post has an OCR twin: the inf-loss trap appears when the input image is too narrow for a long word (the column count $T$ is the analogue of audio frames, as in the OCR worked example above); the all-blank collapse appears identically early in training; the blank-index mismatch is just as silent. The reason this matters for debugging is that the OCR community independently rediscovered the same fixes (wider input resolution, lower LR, careful blank handling), so when you are stuck on a speech CTC bug, the OCR literature is a second corpus of the same lessons under different vocabulary. If you internalize the lattice and the length constraint once, you debug both modalities with the same checklist.

**The "decode-beats-loss" mismatch in production.** A recurring real-world signature is a model whose validation loss kept improving across a finetune but whose deployed WER got *worse* over the same checkpoints. The cause is usually not the loss at all but the decoding stack drifting out of sync with the model: a language-model weight tuned for the old checkpoint, a beam search whose blank handling silently regressed after a library upgrade, or a collapse rule that diverged from the training-time rule when someone "optimized" the decoder. The lesson is to treat the decoder as part of the model under test: pin the decode configuration, compare greedy and beam WER on every checkpoint, and never trust the loss curve as a proxy for end-task quality in a CTC system, because the loss is computed on the marginalized probability and the WER is computed on the collapsed argmax, and those two can move in opposite directions when the decode stack is wrong.

## 10. When this is (and isn't) your CTC bug

CTC symptoms overlap with general training bugs, so it pays to be decisive about when to reach for the CTC-specific tools versus the general ones.

It *is* a CTC bug — go to the length/blank/decode tools — when: the loss is `inf` from early on and you use CTC (the length constraint is the first suspect, before any numerics hunt); the output is all-blank or a single repeated token past the early phase (the LR and the blank index are the first suspects); training metrics are good but deployed transcription is garbage and you decode with a collapse rule (the decode-time blank/collapse logic is the first suspect); or the WER is mysteriously high with a clean-looking loss curve (length bookkeeping or the alignment timing).

It is probably *not* a CTC-specific bug — use the general tools — when: the loss is `nan` *without* a preceding `inf` and you can reproduce it at step 1 (that smells like a data or masking bug, hunt it as a general NaN); the loss never descends at all even with a sane LR and the length assert passes (suspect the data pipeline or the model wiring, and run the overfit-one-batch test — if the model cannot drive CTC loss toward zero on a single batch, the problem is upstream of CTC); the GPUs are idle and throughput is the complaint (that is a systems bug, not a loss bug); or the model trains and evaluates fine offline but degrades only in streaming (that is a chunking/lookahead mismatch, a different post). The cleanest discriminator is the one from the tree: **inf before nan points at the CTC length constraint; nan without inf points at general numerics; finite-but-all-blank points at LR or blank index; finite-but-garbage-decode points at the collapse rule or the blank index.** Route on that first and you will rarely chase the wrong place.

One more honesty check, the overfit-one-batch test specialized to CTC: take a single batch, fix it, and train only on it for a few hundred steps. A correctly-wired CTC setup will drive that batch's loss toward zero and its decode toward the exact transcript. If it cannot overfit one batch — loss stalls, decode stays all-blank — then the bug is *not* a subtle data or LR issue; it is structural (blank index, log-softmax, shape, or lengths), and you should re-walk the five preconditions before touching anything else. Overfit-one-batch is as diagnostic for CTC as it is everywhere else in this series, with the bonus that the decode gives you a second, human-readable signal beyond the loss number.

## Key takeaways

- **CTC sums over all alignments; it never picks one.** The loss is $-\log$ of the summed probability of every blank-padded frame sequence that collapses (merge repeats, then drop blanks) to your target. Almost every CTC bug is a violation of one of those mechanics.
- **`inf` loss means no valid alignment exists**, because $T < U + r$ (frames below labels-plus-adjacent-duplicates). It is the dynamic program reporting an empty path set, not floating-point overflow. The fix is more frames / less downsampling; the guard is a per-sample `T >= U + reps` assert.
- **Use `zero_infinity=False` to *find* bad samples, then fix the cause.** `zero_infinity=True` is a band-aid that silently drops your hardest examples; debug loud first.
- **The blank index must be one source of truth** across the model head, `nn.CTCLoss(blank=K)`, and the decoder. A mismatch trains the wrong column and decodes garbage *while the loss descends* — the most deceptive CTC bug.
- **`nn.CTCLoss` wants log-probs in `(T, N, C)` with true per-sample lengths.** Apply `F.log_softmax(dim=-1)`, transpose to time-first, and pass real (not padded) `input_lengths`/`target_lengths`. None of these throw; all of them silently corrupt training.
- **All-blank early is a phase, not a bug.** Healthy CTC goes random → all-blank → single repeated token → real spikes. Stuck all-blank past a few thousand steps means LR too high (lower it, add warmup), or a blank-index / log-softmax / length problem.
- **A high blank fraction (0.7–0.9) is healthy peaky behavior.** Diagnose collapse by whether `distinct_nonblank` grows and the decode converges, not by the absolute blank fraction.
- **At decode, merge repeats *then* drop blanks.** Reversing the order silently deletes doubled letters; a wrong blank index strips real characters; a beam search worse than greedy means broken blank/repeat handling, not too small a beam.
- **Under AMP, compute CTC loss in fp32 or use bf16.** The log-domain forward-backward underflows in fp16 and manufactures `nan` independent of the length constraint.

## Further reading

- Graves, Fernández, Gomez, Schmidhuber (2006), *Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks*, ICML — the original CTC paper; the blank token, the collapse map, and the forward-backward recursion that makes the length constraint inevitable.
- PyTorch documentation, [`torch.nn.CTCLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html) — the exact contract: `(T, N, C)` log-probs, `input_lengths`/`target_lengths`, the `blank` argument, and `zero_infinity`.
- Hannun (2017), *Sequence Modeling with CTC*, Distill — a clear visual treatment of the collapse rule, the lattice, and the peaky-behavior intuition.
- Baevski, Zhou, Mohamed, Auli (2020), *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*, NeurIPS — the pretrained backbone whose CTC finetuning produces the LR-too-high and length bugs in this post.
- The series taxonomy and decision tree, [A taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) — the symptom → suspect → confirming test → fix frame this post instantiates for CTC.
- The capstone, [The training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) — the full bisection method and the printable checklist that the CTC contract slots into.
- Companion numerics post, [Hunting NaNs and Infs: a systematic method](/blog/machine-learning/debugging-training/hunting-nans-and-infs) — the general method for the `inf`-then-`nan` spread that a CTC length violation triggers.
- Companion shape post, [Shape bugs and silent broadcasting](/blog/machine-learning/debugging-training/shape-bugs-and-silent-broadcasting) — the `(N, T, C)` vs `(T, N, C)` transpose bug generalizes to the whole class of silent shape errors.
- Companion audio posts, [Audio input bugs](/blog/machine-learning/debugging-training/audio-input-bugs) and [Debugging ASR finetuning](/blog/machine-learning/debugging-training/debugging-asr-finetuning) — the feature-extractor and WER-metric issues that sit on either side of the CTC loss in a speech pipeline.
