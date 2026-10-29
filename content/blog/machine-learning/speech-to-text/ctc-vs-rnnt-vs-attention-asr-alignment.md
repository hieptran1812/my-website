---
title: "CTC vs RNN-T vs Attention Encoder-Decoder: The Three Ways ASR Models Align Audio to Text"
date: "2026-10-29"
publishDate: "2026-10-29"
description: "Every speech recognizer is an answer to one question: which audio frames produced which tokens. Here is how CTC, RNN-T and attention decoders each answer it, what that choice costs, and how to explain it in an interview."
tags: ["asr", "speech-recognition", "ctc", "rnn-t", "transducer", "attention", "whisper", "speech-to-text", "interview"]
category: "machine-learning"
subcategory: "Speech-to-Text"
author: "Hiep Tran"
featured: true
readTime: 13
---

Ask a speech engineer "CTC or RNN-T?" in an interview and you learn within a minute whether they have trained a model or only called one. The question sounds like trivia about loss functions. It is really a question about **alignment**: the transcript says *what* was said, but nothing in the training data says *when* each token was said, and every ASR architecture is a different bet on how to recover that missing mapping.

![One second of audio becomes 25 encoder frames, while the transcript is only 3 tokens](/imgs/blogs/ctc-vs-rnnt-vs-attention-asr-alignment-1.webp)

The diagram above is the mental model: a long sequence of frames on one side, a short sequence of tokens on the other, and no labels connecting them. CTC, RNN-T and attention encoder-decoders (AED) are three answers to that gap. Hybrids and LLM-based recognizers are combinations of those answers. Once you see them that way, their failure modes, their streaming behavior and their memory bills all fall out of the same picture. Let's build up the intuition step by step.

## The alignment problem

A typical modern pipeline turns audio into 80-dimensional log-mel features every 10 ms (the details are in [representing sound as waveforms and spectrograms](/blog/machine-learning/audio-generation/representing-sound-waveforms-spectrograms-and-perception)). One second is 100 feature frames. A Conformer encoder subsamples by 4 (FastConformer by 8), so the model sees $T = 25$ encoder frames per second. The transcript "hello world" might be $U = 3$ BPE tokens.

Training data is a pair (audio, text). There is no frame-level label telling you that frames 4 to 9 are "hel". You could get those labels with a forced aligner from an older HMM system, and that is exactly what pre-2015 hybrid systems did. End-to-end ASR exists to remove that dependency.

Think of it like subtitling a film you have the script for but no timecodes. Say the script is "turn on the lights" and the film as 2.6 seconds of speech, or 65 encoder frames. The job is to decide, for every frame, which word (if any) it belongs to. There are three broad strategies:

1. Label every frame, allow "nothing here", then clean up. That is **CTC**.
2. Walk frame by frame, deciding at each step whether to emit or advance. That is **RNN-T**.
3. Read the whole film, then write the script, glancing back at whatever scene you need. That is **attention**.

Each option makes the model learn the alignment implicitly. They differ in what they assume, and those assumptions are the whole interview answer.

## CTC: predict every frame, then collapse

Connectionist Temporal Classification (Graves et al., 2006) solves the length mismatch by forcing the output to be as long as the input. The encoder emits one distribution per frame over the vocabulary plus one extra symbol, **blank** (written $\varnothing$). A frame-level string is mapped to text by a fixed collapse function: merge consecutive repeats, then delete blanks.

![CTC collapse: merge repeats, then drop blanks; blank is what lets a doubled letter survive](/imgs/blogs/ctc-vs-rnnt-vs-attention-asr-alignment-2.webp)

Notice the second row of the figure. Without blank, "aa" and a single "a" held for two frames would be indistinguishable. Blank serves two jobs: it fills frames where nothing new is happening, and it separates genuine repeats.

The loss is the negative log of the total probability of every frame string that collapses to the target:

$$
\mathcal{L}_{\text{CTC}} = -\log \sum_{\pi \in \mathcal{B}^{-1}(y)} \prod_{t=1}^{T} p(\pi_t \mid x)
$$

Here $x$ is the audio, $y$ the target tokens, $\pi$ one frame-level path of length $T$, $\pi_t$ its symbol at frame $t$, and $\mathcal{B}$ the collapse function, so $\mathcal{B}^{-1}(y)$ is every path that collapses to $y$. The number of paths is exponential, but they share prefixes, so the forward-backward algorithm sums them in $O(T \cdot U)$ time on a trellis of $2U+1$ states (each token with a blank on either side).

Two properties of that product drive everything else about CTC.

**Conditional independence.** The product treats every frame as independent given the audio. The output at frame 12 never sees what was emitted at frame 11. CTC therefore has no internal language model. It will happily output "their" and "there" with whatever spelling the acoustics suggest, and it cannot learn that "New York" makes "Yorke" unlikely. This is a weakness for accuracy and a strength for engineering: you can bolt on an external n-gram or neural LM with **shallow fusion**, scoring beam hypotheses with $\log p_{\text{CTC}} + \alpha \log p_{\text{LM}} + \beta \cdot |y|$, and swap that LM per customer without retraining the acoustic model. `pyctcdecode` with a KenLM model is the standard way to do this.

**The length constraint.** Each token needs at least one frame, plus a blank between repeats. With 8x subsampling (12.5 frames per second), fast speech or character-level vocabularies can hit $T \lt U$, and the loss goes to infinity. That is the first thing to check when CTC returns `inf`, a story covered in [debugging CTC and alignment](/blog/machine-learning/debugging-training/debugging-ctc-and-alignment).

![Trained CTC posteriors are peaky: blank owns almost every frame and each token is a one-frame spike](/imgs/blogs/ctc-vs-rnnt-vs-attention-asr-alignment-3.webp)

The figure shows what a trained CTC model actually outputs. Blank wins nearly every frame and each token fires as a single spike, one 40 ms frame wide. This **peaky** behavior is a known consequence of maximum-likelihood training on the CTC trellis: putting all mass on blank is the cheapest way to be correct on most frames. Practically, it means greedy decoding is very fast (argmax, collapse, done) and CTC timestamps mark *when the model was sure*, not when the word began. Use a separate forced aligner if you need word boundaries for subtitles.

```python
import torch
import torch.nn as nn

T, B, V = 25, 2, 32          # frames, batch, vocab (index 0 = blank)
log_probs = torch.randn(T, B, V).log_softmax(-1).requires_grad_()
targets = torch.tensor([[3, 1, 20], [8, 5, 0]])  # "cat", "he" (padded)
input_lengths = torch.tensor([25, 25])
target_lengths = torch.tensor([3, 2])

ctc = nn.CTCLoss(blank=0, zero_infinity=True)    # zero_infinity hides T < U cases
loss = ctc(log_probs, targets, input_lengths, target_lengths)
loss.backward()

def greedy_ctc(lp: torch.Tensor, blank: int = 0) -> list[int]:
    best = lp.argmax(-1).tolist()                # one symbol per frame
    out, prev = [], None
    for s in best:
        if s != prev and s != blank:             # merge repeats, then drop blanks
            out.append(s)
        prev = s
    return out

print(loss.item(), greedy_ctc(log_probs[:, 0]))
```

## RNN-T: put a language model inside

The Recurrent Neural Network Transducer (Graves, 2012) keeps CTC's blank but removes the independence assumption. It adds two pieces: a **prediction network** that reads the tokens emitted so far (a small LSTM or a stateless embedding of the last one or two tokens), and a **joiner** that combines one encoder frame $f_t$ with one prediction state $g_u$:

$$
z_{t,u} = W \tanh(f_t + g_u), \qquad p(k \mid t, u) = \operatorname{softmax}(z_{t,u})_k
$$

where $k$ ranges over the vocabulary plus blank. Decoding is a walk. At position $(t, u)$ the model either emits blank and advances to the next frame, or emits a token and stays on the same frame, ready to emit another.

<figure class="blog-anim">
<svg viewBox="0 0 720 380" role="img" aria-label="RNN-T alignment lattice: a path crosses a 6 by 3 grid, moving right on blank and up on each emitted token" style="width:100%;height:auto;max-width:760px">
<style>
.r1-dot{fill:var(--border,#d1d5db)}
.r1-ax{font:500 14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.r1-tok{font:700 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.r1-bl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.r1-em{font:700 16px ui-sans-serif,system-ui;fill:var(--accent,#6366f1);text-anchor:start}
.r1-leg{font:500 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.r1-path{fill:none;stroke:var(--accent,#6366f1);stroke-width:5;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:630;stroke-dashoffset:630}
@keyframes r1-draw{0%{stroke-dashoffset:630}75%,100%{stroke-dashoffset:0}}
.r1-anim{animation:r1-draw 10s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.r1-anim{animation:none;stroke-dashoffset:0}}
</style>
<circle class="r1-dot" cx="80" cy="300" r="6"/>
<circle class="r1-dot" cx="150" cy="300" r="6"/>
<circle class="r1-dot" cx="220" cy="300" r="6"/>
<circle class="r1-dot" cx="290" cy="300" r="6"/>
<circle class="r1-dot" cx="360" cy="300" r="6"/>
<circle class="r1-dot" cx="430" cy="300" r="6"/>
<circle class="r1-dot" cx="500" cy="300" r="6"/>
<circle class="r1-dot" cx="80" cy="230" r="6"/>
<circle class="r1-dot" cx="150" cy="230" r="6"/>
<circle class="r1-dot" cx="220" cy="230" r="6"/>
<circle class="r1-dot" cx="290" cy="230" r="6"/>
<circle class="r1-dot" cx="360" cy="230" r="6"/>
<circle class="r1-dot" cx="430" cy="230" r="6"/>
<circle class="r1-dot" cx="500" cy="230" r="6"/>
<circle class="r1-dot" cx="80" cy="160" r="6"/>
<circle class="r1-dot" cx="150" cy="160" r="6"/>
<circle class="r1-dot" cx="220" cy="160" r="6"/>
<circle class="r1-dot" cx="290" cy="160" r="6"/>
<circle class="r1-dot" cx="360" cy="160" r="6"/>
<circle class="r1-dot" cx="430" cy="160" r="6"/>
<circle class="r1-dot" cx="500" cy="160" r="6"/>
<circle class="r1-dot" cx="80" cy="90" r="6"/>
<circle class="r1-dot" cx="150" cy="90" r="6"/>
<circle class="r1-dot" cx="220" cy="90" r="6"/>
<circle class="r1-dot" cx="290" cy="90" r="6"/>
<circle class="r1-dot" cx="360" cy="90" r="6"/>
<circle class="r1-dot" cx="430" cy="90" r="6"/>
<circle class="r1-dot" cx="500" cy="90" r="6"/>
<text class="r1-tok" x="80" y="332">0</text>
<text class="r1-tok" x="150" y="332">1</text>
<text class="r1-tok" x="220" y="332">2</text>
<text class="r1-tok" x="290" y="332">3</text>
<text class="r1-tok" x="360" y="332">4</text>
<text class="r1-tok" x="430" y="332">5</text>
<text class="r1-tok" x="500" y="332">6</text>
<text class="r1-ax" x="215" y="362">encoder frame t</text>
<text class="r1-tok" x="40" y="305">start</text>
<text class="r1-tok" x="40" y="235">c</text>
<text class="r1-tok" x="40" y="165">a</text>
<text class="r1-tok" x="40" y="95">t</text>
<polyline class="r1-path r1-anim" points="80,300 150,300 150,230 290,230 290,160 360,160 360,90 500,90"/>
<text class="r1-bl" x="115" y="324">&#8709;</text>
<text class="r1-bl" x="185" y="254">&#8709;</text>
<text class="r1-bl" x="255" y="254">&#8709;</text>
<text class="r1-bl" x="325" y="184">&#8709;</text>
<text class="r1-bl" x="395" y="114">&#8709;</text>
<text class="r1-bl" x="465" y="114">&#8709;</text>
<text class="r1-em" x="160" y="270">c</text>
<text class="r1-em" x="300" y="200">a</text>
<text class="r1-em" x="370" y="130">t</text>
<text class="r1-leg" x="560" y="140">&#8594; blank: next frame</text>
<text class="r1-leg" x="560" y="170">&#8593; token: emit,</text>
<text class="r1-leg" x="578" y="192">stay on frame</text>
<text class="r1-leg" x="560" y="240">6 blanks + 3 tokens</text>
<text class="r1-leg" x="560" y="262">= one alignment</text>
</svg>
<figcaption>One RNN-T alignment of "cat" over 6 frames: the path moves right on a blank and up on a token. Training sums the probability of every such path through the T by U grid.</figcaption>
</figure>

The animation traces one path. Training, like CTC, sums over all of them with forward-backward on the $T \times U$ lattice. Three consequences follow, and each is a likely follow-up question.

**It can emit several tokens per frame.** Because "emit" does not advance $t$, RNN-T has no $T \ge U$ constraint. That is why it tolerates aggressive 8x subsampling better than CTC.

**It conditions on its own history.** The prediction network is a small internal LM, so RNN-T is usually more accurate than CTC on the same encoder. The cost is that external LM fusion is less clean, because the internal LM is already in the score. Production systems often subtract an estimate of it (internal LM estimation) before adding the domain LM.

**It is naturally streaming.** Each decision depends on frames up to $t$ and tokens so far, never on the future. Give it a causal or chunked encoder and you have the architecture Google shipped on-device for Pixel phones in 2019. This is why most real-time products run a transducer.

The bill comes due in memory. The joiner produces a full distribution for every $(t, u)$ pair.

![RNN-T joint tensor memory dwarfs CTC logits; pruning the lattice brings it back down](/imgs/blogs/ctc-vs-rnnt-vs-attention-asr-alignment-4.webp)

With batch 8, 250 frames, 100 target tokens and a 1,024-token vocabulary, the joint tensor holds $B \cdot T \cdot U \cdot V$ ≈ 205 million floats, about 819 MB in fp32 before gradients. CTC's output for the same batch is about 8.2 MB. That 100x gap is why naive RNN-T training runs out of memory at batch sizes CTC handles easily. The fixes are worth naming in an interview: **function merging** (compute the loss gradient without materializing the softmax twice), **pruned RNN-T** in k2, which first uses a cheap "trivial joiner" to find a narrow band of plausible $u$ for each $t$ and evaluates the real joiner only there, and fused CUDA kernels like those in NeMo and torchaudio.

```python
import torch
import torchaudio.functional as F

B, T, U, V = 2, 25, 3, 32                     # U target tokens, V includes blank
logits = torch.randn(B, T, U + 1, V, requires_grad=True)  # 4-D: the joint tensor
targets = torch.tensor([[3, 1, 20], [8, 5, 1]], dtype=torch.int32)
logit_lengths = torch.tensor([25, 20], dtype=torch.int32)
target_lengths = torch.tensor([3, 3], dtype=torch.int32)

loss = F.rnnt_loss(logits, targets, logit_lengths, target_lengths,
                   blank=0, reduction="mean")
loss.backward()
print(loss.item(), logits.shape)               # (B, T, U+1, V): note the extra U axis
```

NVIDIA's **Token-and-Duration Transducer** (TDT, 2023) attacks the other cost, inference time. Instead of stepping one frame per blank, the joiner also predicts how many frames to skip. Parakeet-TDT models spend most of their decoding loop jumping over silence and long vowels, which is a large part of why they top throughput leaderboards; the architecture is unpacked in [Canary, Parakeet and FastConformer](/blog/machine-learning/speech-to-text/canary-parakeet-fastconformer-asr).

## Attention encoder-decoder: let attention find it

The third family drops explicit alignment entirely. An encoder reads the whole utterance; an autoregressive decoder generates tokens one at a time, and cross-attention decides which frames to look at for each token. Listen, Attend and Spell (2016) introduced the pattern for ASR, and Whisper scaled it to 680k hours of weakly supervised audio (covered in [Whisper under the hood](/blog/machine-learning/speech-to-text/whisper-under-the-hood)).

The decoder is a full conditional language model, so AED models have the strongest built-in LM of the three and produce the best-formatted text: punctuation, casing, numbers written as digits, even translation, all from one decoder. With large data they are typically the most accurate offline.

The freedom is also the failure mode. Nothing forces cross-attention to move monotonically forward through the audio. On silence, music or noise, the decoder can generate fluent text that was never spoken, or attend to the same region twice and loop ("thank you, thank you, thank you"). Whisper's decoding wrapper exists largely to catch this: it re-decodes at higher temperature when the output's gzip compression ratio exceeds 2.4 (a sign of repetition) or average log-probability drops below -1.0, and it drops segments the no-speech token flags. CTC and RNN-T cannot hallucinate a sentence over silence in the same way, because their monotonic lattice ties every emitted token to a frame.

Two more practical points. AED models usually train with label smoothing (0.1 is common), which helps the decoder's calibration but further weakens the link between scores and true confidence. And they are offline by construction: the encoder wants the full input, and Whisper processes fixed 30-second windows.

## Streaming changes the answer

The single most decisive product question is whether users see words while they are still talking.

![Streaming models emit each word shortly after it is spoken; the attention decoder emits everything after the utterance ends](/imgs/blogs/ctc-vs-rnnt-vs-attention-asr-alignment-5.webp)

The lanes make the gap concrete. A chunked transducer emits "turn" about 200 ms after it is spoken. The attention decoder waits for the endpoint at 2.6 s, then emits the whole sentence. For a voice agent that must start thinking before the user finishes, or for live captions, that difference is the product. Streaming AED variants exist (monotonic chunkwise attention, chunked Whisper with local agreement), but they trade accuracy for latency and add machinery that transducers get for free.

## Hybrids and the fourth family

Real systems rarely pick one loss. The most common combination is **joint CTC/attention** training (Watanabe et al., 2017), the default recipe in ESPnet.

![Hybrid training shares one encoder between a CTC head and an attention decoder](/imgs/blogs/ctc-vs-rnnt-vs-attention-asr-alignment-6.webp)

A shared encoder feeds both a CTC head and an attention decoder, and the loss mixes them, typically $\mathcal{L} = 0.3\,\mathcal{L}_{\text{CTC}} + 0.7\,\mathcal{L}_{\text{att}}$. The CTC branch pushes the encoder toward monotonic, frame-aligned representations, which speeds convergence and makes the attention decoder far less prone to looping. At decode time, CTC prefix scores rescore the attention beam, vetoing hypotheses whose length or content does not fit the audio. NeMo applies the same idea to transducers with hybrid RNNT-CTC FastConformer models: one encoder, two decoders, and you choose the fast CTC head or the more accurate transducer head at inference.

The **fourth family** is LLM-based ASR: an audio encoder, a projector that maps frames into the LLM's embedding space, and a pretrained LLM as the decoder. Qwen3-ASR (see [the technical report breakdown](/blog/machine-learning/speech-to-text/qwen3-asr-technical-report)) is a current example. Architecturally this is an attention encoder-decoder with a much larger, text-pretrained decoder. It inherits AED's strengths (formatting, context biasing through a text prompt, multilingual reach) and AED's weaknesses (hallucination, non-streaming by default, slow autoregressive decoding), plus a new one: the LLM can follow instructions hidden in the audio.

## Side by side

| | CTC | RNN-T / TDT | Attention enc-dec | LLM-based |
| --- | --- | --- | --- | --- |
| Alignment | Monotonic, one symbol per frame | Monotonic lattice, many tokens per frame | Learned by attention, unconstrained | Learned by attention |
| Internal LM | None (conditionally independent) | Small (prediction net) | Strong (decoder) | Very strong (pretrained LLM) |
| Streaming | Natural | Natural, the industry default | Hard | Hard |
| Decode speed | Fastest (non-autoregressive) | Fast; TDT skips frames | Slow (token by token) | Slowest |
| External LM | Easy shallow fusion | Works, needs internal-LM correction | Possible, rarely needed | Prompt-based biasing instead |
| Training memory | Low ($B \cdot T \cdot V$) | High ($B \cdot T \cdot U \cdot V$), pruning needed | Moderate | High |
| Signature failure | Spelling errors, no context | Deletions at chunk boundaries | Hallucination, loops on silence | Hallucination, prompt injection |
| Examples | wav2vec 2.0 fine-tunes, Parakeet-CTC | Parakeet-TDT, on-device Google ASR | Whisper, Canary | Qwen3-ASR |

## How to answer in an interview

A strong 60-second answer follows the structure of this post: name the problem, then give each family's assumption and its consequence.

> "ASR has an alignment problem: many frames, few tokens, no frame labels. CTC adds a blank, predicts every frame independently and sums over all collapsible paths with forward-backward. It is fast and streams, but it has no internal LM, so we add one at decode time. RNN-T adds a prediction network and a joiner, so each output conditions on previous tokens; it streams, it is more accurate, but the T by U by V joint tensor is expensive to train, which is why pruned losses exist. Attention encoder-decoders let cross-attention learn the alignment; they are the most accurate offline and produce the best formatting, but they are not streaming and can hallucinate on silence. In practice we combine them: CTC as an auxiliary loss for attention models, or hybrid CTC and transducer heads."

Then expect follow-ups. The ones I see most:

- **Why does CTC need blank?** To represent frames with no new token and to separate true repeats like "ll" in "hello".
- **Why are CTC outputs peaky?** Maximum likelihood over the trellis rewards concentrating mass on blank; tokens become one-frame spikes, so timestamps are imprecise.
- **When does CTC loss become infinite?** When the target needs more frames than exist: too much subsampling, character vocabularies, or mislabeled data. Check $T$ against $U$ plus repeats.
- **Why does RNN-T use so much memory, and how do you fix it?** A full softmax for every frame-token pair; use function merging, pruned RNN-T, or fused kernels.
- **Why does Whisper hallucinate, and why doesn't CTC?** Unconstrained attention plus a strong decoder LM; CTC's monotonic frame-by-frame output has nowhere to put an invented sentence. Mitigate with VAD, compression-ratio and log-prob fallbacks, and CTC-based rescoring.
- **What would you ship for a real-time voice agent?** A streaming transducer (chunked FastConformer with RNN-T or TDT), with an optional second-pass AED or LLM rescoring on the final segment if accuracy on entities matters.

If the role involves training rather than serving, be ready to discuss data, since alignment-based losses are unforgiving about bad transcripts; [training ASR models](/blog/machine-learning/speech-to-text/training-asr-models) covers that end of the stack.

## Summary

- Every ASR architecture answers the same question: which frames produced which tokens, given that training data never says.
- CTC predicts a symbol per frame, uses blank to absorb the length gap, and is fast, streaming and LM-free by design.
- RNN-T adds a prediction network so outputs condition on history, keeps streaming, and pays for it with a $T \times U$ joint tensor that needs pruning to train at scale.
- Attention encoder-decoders learn alignment through attention, win offline accuracy and formatting, and lose streaming and monotonicity, which is where hallucination comes from.
- Hybrids use CTC to keep attention honest; LLM-based ASR is an attention decoder with a far bigger language model attached.
- The question to ask before choosing is not "which is most accurate" but "does the user need words before they stop talking". That one answer usually decides between a transducer and everything else.
