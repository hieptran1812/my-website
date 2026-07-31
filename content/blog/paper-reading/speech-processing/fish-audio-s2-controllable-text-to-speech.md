---
title: "Fish Audio S2: Dual-AR, Multi-Reward RL, and the Data Flywheel Behind Controllable TTS"
date: "2026-07-31"
description: "A technique-by-technique explanation of Fish Audio S2, from RVQ audio tokens and Dual-AR generation to instruction-following data, GRPO alignment, and low-latency SGLang serving."
tags: ["paper-reading", "fish-audio-s2", "text-to-speech", "speech-synthesis", "audio-language-model", "dual-ar", "grpo", "reinforcement-learning", "streaming-inference"]
category: "paper-reading"
subcategory: "Speech Processing"
author: "Hiep Tran"
featured: true
readTime: 33
paper:
  title: "Fish Audio S2 Technical Report"
  authors: "Fish Audio Team"
  venue: "arXiv 2026"
  url: "https://arxiv.org/abs/2603.08823"
---

> [!tldr]
> - Fish Audio S2 turns expressive TTS into language modeling over discrete audio tokens. Its tokenizer produces ten RVQ codebooks per frame: one semantic stream and nine acoustic streams.
> - The key architectural move is Dual-AR: a large Slow AR model plans semantic tokens over time, while a small Fast AR model fills in acoustic detail across codebook depth.
> - The data pipeline is also a reward pipeline. A speech-quality model and a rich-transcription ASR model first filter and annotate training data, then score generated speech during RL post-training.
> - The paper reports strong results: 0.54 Chinese WER and 0.99 English WER on Seed-TTS-Eval, 81.88% win rate on EmergentTTS-Eval, RTF 0.195, and TTFA below 100 ms.
> - The important caveat is evaluation dependence: several headline claims use Gemini-based judges or newly introduced benchmarks, and the report gives limited ablations isolating each contribution.

![Figure 1 from Fish Audio Team (2026): Fish Audio S2 capability overview](/imgs/blogs/fish-audio-s2-controllable-text-to-speech-fig1.webp)

The diagram above is the mental model: Fish Audio S2 is not merely a text-to-waveform box. It is a language model that plans a conversation, emits a compact semantic audio stream, refines that stream into acoustic detail, and then decodes the result quickly enough to stream. The rest of this post unpacks why those separations matter.

## The problem: TTS has become a systems problem

The old mental model of text-to-speech is pleasantly simple: take text, predict an acoustic representation, and run a vocoder. The difficulty is that every word in that sentence hides a different failure mode.

The system must preserve what was written, sound like the requested speaker, express the requested emotion, place a laugh or whisper at the correct word, keep multiple speakers separated, remain coherent for minutes, and start producing audio quickly. Optimizing one property can damage another. A model that copies a voice accurately may ignore `[laugh]`; a model that follows expressive instructions may hallucinate or skip words; a model that generates long audio may drift in timbre.

There is also a representation problem. A waveform sampled at 44.1 kHz contains 44,100 scalar samples per second. Feeding that raw sequence to an autoregressive Transformer is wasteful. Neural audio codecs solve this by turning waveform segments into discrete tokens. But high-fidelity audio needs more than one token per frame. Fish Audio S2 uses ten residual vector-quantization codebooks, so naively flattening the representation makes the language-model sequence ten times longer.

Finally, the data problem is just as important as the architecture. The model needs transcripts that say more than “the speaker said these words.” It needs descriptions of prosody, emotion, speaker turns, breathing, laughter, emphasis, and recording quality. Human annotation at the scale claimed in the report is expensive. The authors therefore build models that automatically filter and caption audio, then reuse those same models as reward functions.

This places Fish Audio S2 in the family of codec language models: text and audio are represented as tokens, and a decoder-only Transformer learns to predict the next token. The interesting contribution is not one exotic layer. It is the way representation, supervision, alignment, and serving are made to fit together.

## Contributions

The report makes four closely connected claims.

1. **A Dual-AR codec language model** separates temporal semantic planning from depth-wise acoustic generation. The Slow AR backbone is a pretrained Qwen3-4B; the Fast AR is a four-layer Transformer.
2. **A multi-purpose data pipeline** uses a speech-quality model and a rich-transcription ASR model to clean and annotate data. Those evaluators are later reused as RL rewards.
3. **Multi-reward RL alignment** uses a GRPO/Dr.GRPO-style objective to balance semantic accuracy, acoustic preference, and speaker similarity without a value network.
4. **An LLM-native serving path** adapts SGLang with audio-token I/O, multi-token RadixCache keys, and GPU co-scheduling of language-model decoding and waveform decoding.

The rest of the method is easier to understand if we keep one question in view: which subproblem is each design choice moving somewhere else?

## Method

### 1. From waveform to ten discrete streams

#### The problem it solves

A waveform is a high-rate continuous signal. If we ask a Transformer to predict each sample, the model spends most of its capacity tracking local periodic structure instead of planning language and prosody. We need a compact sequence where each timestep still contains enough information to reconstruct natural speech.

#### Intuition: a layered audio file

Imagine compressing a song into a stack of descriptions. The first layer says roughly what is happening: phonetic content, syllable timing, and coarse prosody. Additional layers add progressively finer detail: exact timbre, harmonics, noise texture, and phase. A listener can recognize the sentence from the first layer, but the sound becomes convincing only when the layers are combined.

Residual vector quantization (RVQ) implements that stack. The first codebook quantizes the input. The second codebook quantizes the residual error left over after the first approximation. Each subsequent codebook explains what the previous codebooks missed.

#### Mechanism

Fish Audio S2 uses a DAC-inspired tokenizer operating at 44.1 kHz. It is modified for streaming with causal convolutions and causal sliding-window Transformer blocks. The encoder downsamples by `512×` in the standard DAC path and adds `4×` ConvNeXt V2 downsampling, giving `2048×` total downsampling and a frame rate of about 21 Hz.

At each audio frame $t$, the tokenizer emits $N=10$ discrete indices:

- $q_t^{(0)}$ is the semantic codebook token.
- $q_t^{(1)},\ldots,q_t^{(9)}$ encode progressively finer acoustic detail.

The decoder receives all ten tokens and reconstructs waveform samples. The paper replaces the original DAC decoder with an EVA-GAN-style generator for parameter efficiency and synthesis quality.

#### The math

Let $z_t$ be the continuous encoder representation at frame $t$, with $z_t\in\mathbb{R}^{d}$. Let $e_k(c)$ be the embedding vector for codebook $k$ and code index $c$. RVQ produces a sequence of residuals. At the first level:

$$
q_t^{(0)}=\arg\min_{c\in\{1,\ldots,C_0\}}\left\|z_t-e_0(c)\right\|_2^2,\qquad
r_t^{(1)}=z_t-e_0\!\left(q_t^{(0)}\right).
$$

Here $C_0$ is the size of the first codebook, $q_t^{(0)}$ is its selected index, and $r_t^{(1)}$ is the residual passed to the next codebook. For codebook $k$:

$$
q_t^{(k)}=\arg\min_{c\in\{1,\ldots,C_k\}}\left\|r_t^{(k)}-e_k(c)\right\|_2^2,\qquad
r_t^{(k+1)}=r_t^{(k)}-e_k\!\left(q_t^{(k)}\right).
$$

The reconstructed latent is the sum of the selected embeddings:

$$
\hat z_t=\sum_{k=0}^{N-1}e_k\!\left(q_t^{(k)}\right).
$$

The exact production tokenizer contains adversarial training and three discriminators: multi-period, multi-resolution, and multi-scale STFT discriminators. These encourage periodic structure, spectral consistency, high-frequency detail, and phase coherence.

#### Worked micro-example

Suppose a one-dimensional encoder frame is $z_t=1.30$. A first codebook contains values `{0.0, 1.0, 2.0}`. The closest value is `1.0`, so $q_t^{(0)}=1$ and the residual is `0.30`. A second codebook contains `{0.0, 0.25, 0.5}`. It selects `0.25`, leaving residual `0.05`. A third codebook might select `0.05` exactly. The sum is `1.30`.

```python
import torch

def rvq_one_frame(z, codebooks):
    """Toy RVQ: z is [D], codebooks is a list of [C_k, D]."""
    residual = z.clone()
    indices = []
    selected = []
    for codebook in codebooks:
        distances = ((codebook - residual[None, :]) ** 2).sum(dim=-1)
        index = distances.argmin()
        vector = codebook[index]
        indices.append(index.item())
        selected.append(vector)
        residual = residual - vector
    return torch.tensor(indices), torch.stack(selected).sum(dim=0), residual

z = torch.tensor([1.30])
books = [torch.tensor([[0.0], [1.0], [2.0]]),
         torch.tensor([[0.0], [0.25], [0.5]]),
         torch.tensor([[0.0], [0.05], [0.10]])]
indices, reconstruction, residual = rvq_one_frame(z, books)
print(indices, reconstruction, residual)
```

#### Why it works, and when it fails

RVQ gives the language model a useful division of labor: the first stream can carry linguistic information while later streams preserve fidelity. It also converts a 44.1 kHz waveform into roughly 21 frames per second, which is a manageable starting point for autoregressive generation.

The failure mode is rate-distortion mismatch. If the codebooks are too small, the decoder cannot recover pitch and timbre. If the semantic stream is not semantically meaningful, the language model must learn linguistic planning from a noisy acoustic code. And because the final quality still depends on the decoder, good token prediction does not automatically imply a good waveform.

### 2. Causality and semantic distillation

#### The problem it solves

Offline audio models can use future context. A streaming TTS service cannot. If the encoder or decoder looks ahead, the first audio chunk must wait for future waveform samples. Separately, ordinary reconstruction training may encourage the first codebook to preserve whatever is easiest for the codec, not whatever is easiest for a language model to plan.

#### Intuition: a radio announcer with a short memory

For streaming, imagine a radio announcer who can hear everything already spoken but not the next sentence. Causal convolutions enforce this rule locally. A sliding-window Transformer gives the announcer a bounded working memory instead of an unlimited archive.

For semantic distillation, imagine a senior listener whispering a phonetic summary to the codec. The first codebook is trained not only to reconstruct sound but also to reproduce the representation learned by a pretrained speech model. Fish Audio S2 regresses the 16th-layer activations of w2v-BERT 2.0.

![Figure 2 from Fish Audio Team (2026): the causal tokenizer output feeding the Dual-AR architecture](/imgs/blogs/fish-audio-s2-controllable-text-to-speech-fig2.webp)

#### Mechanism

Standard convolutions are replaced with masked causal convolutions. Transformer bottlenecks before and after RVQ use causal sliding-window attention, so memory remains bounded during long-form inference. The first quantized stream is passed through an auxiliary prediction head. That head tries to match w2v-BERT 2.0 activations, while the main decoder still reconstructs waveform audio.

#### The math

Let $h_t^{(0)}$ be the quantized representation associated with the first codebook, and let $s_t$ be the frozen target activation from w2v-BERT at the same or aligned frame. Let $g_\phi$ be the auxiliary semantic head. A simple semantic distillation objective is:

$$
\mathcal{L}_{\mathrm{sem}}=\frac{1}{T}\sum_{t=1}^{T}\left\|g_\phi\!\left(h_t^{(0)}\right)-s_t\right\|_2^2.
$$

The actual tokenizer also uses a composite GAN reconstruction objective. The conceptual total can be written as:

$$
\mathcal{L}_{\mathrm{tokenizer}}=\mathcal{L}_{\mathrm{reconstruction}}+\lambda_{\mathrm{GAN}}\mathcal{L}_{\mathrm{GAN}}+\lambda_{\mathrm{sem}}\mathcal{L}_{\mathrm{sem}},
$$

where the reconstruction term preserves the waveform, the adversarial term improves perceptual realism, and the semantic term makes the first codebook useful to the downstream language model. The paper does not provide every scalar coefficient in the technical report, so we should treat this decomposition as an explanatory abstraction rather than a reproduction recipe.

#### Worked example

Consider two candidate first-codebook representations for the same phoneme. Candidate A reconstructs a clean waveform but maps to a vector far from w2v-BERT’s target. Candidate B has a tiny high-frequency reconstruction error but its representation is close to the target activation. With $\lambda_{\mathrm{sem}}>0$, training can prefer B because the downstream Slow AR can more reliably associate it with the phonetic content.

```python
import torch
import torch.nn.functional as F

quantized_semantic = torch.randn(8, 128)      # [frames, codec_dim]
target_w2vbert = torch.randn(8, 768)          # [frames, teacher_dim]
semantic_head = torch.nn.Linear(128, 768)

predicted = semantic_head(quantized_semantic)
semantic_loss = F.mse_loss(predicted, target_w2vbert)
semantic_loss.backward()
```

#### Why it works, and when it fails

The semantic loss gives the first codebook a stable interface to the language model. Causality and sliding windows make that interface usable for low-latency, long-form generation. The tradeoff is that a causal tokenizer has less information than an offline tokenizer and may produce boundary artifacts when its window is too short. Distillation can also overfit the teacher’s representation: a speech model trained for recognition is not a perfect description of every expressive acoustic detail.

### 3. Dual-AR generation: time first, detail second

#### The problem it solves

If ten codebooks are flattened along the time axis, a waveform segment of $T$ frames becomes roughly $10T$ autoregressive positions. Long context becomes expensive, and the model must spend every generation step deciding both “what comes next in the sentence?” and “what exact spectral detail belongs to this frame?”

![Redrawn diagram: Dual-AR separates temporal semantic planning from codebook-depth acoustic refinement](/imgs/blogs/fish-audio-s2-controllable-text-to-speech-1.webp)

#### Intuition: a film director and a sound designer

The Slow AR is the director. It decides the next semantic beat: which phonetic content, speaker turn, and coarse prosody should happen next. The Fast AR is the sound designer. Once the beat is chosen, it fills in the fine acoustic layers needed to render that frame.

This analogy maps directly to the factorization. Slow AR runs over the full interleaved sequence of system prompt, reference audio, target text, and generated audio. At each audio timestep it predicts the semantic token $q_t^{(0)}$. Fast AR receives the Slow-AR hidden state and that semantic token, then predicts $q_t^{(1)}$ through $q_t^{(9)}$.

#### Mechanism

The Slow AR is a pretrained Qwen3-4B decoder-only Transformer. It autoregresses over time and predicts the first codebook token. Its hidden state $h_t^{slow}$ is projected into the Fast AR dimension. The semantic token is embedded as a seed. The Fast AR has four Transformer layers and generates the remaining nine tokens autoregressively over codebook depth.

The Fast AR shares one embedding table across codebook layers. RoPE positions encode the layer identity. This is an asymmetric design: approximately four billion parameters model the long temporal dependency, while only four Transformer layers handle the nine-token local refinement.

#### The math

The naive factorization treats each pair $(t,k)$ as one long sequence:

$$
P\!\left(q_{1:T}^{(0:N-1)}\mid x\right)=\prod_{t=1}^{T}\prod_{k=0}^{N-1}P\!\left(q_t^{(k)}\mid x,q_{<t}^{(0:N-1)},q_t^{(<k)}\right).
$$

Dual-AR groups the factorization into a temporal model and a depth model:

$$
P\!\left(q_{1:T}^{(0:N-1)}\mid x\right)=\prod_{t=1}^{T}\left[P\!\left(q_t^{(0)}\mid x,q_{<t}^{(0:N-1)}\right)\prod_{k=1}^{N-1}P\!\left(q_t^{(k)}\mid h_t^{slow},q_t^{(<k)}\right)\right].
$$

Here $x$ denotes text, system instructions, and reference-audio context; $T$ is the number of audio frames; $N=10$ is the codebook count; $q_t^{(<k)}$ denotes acoustic tokens already generated at timestep $t$; and $h_t^{slow}$ is the Slow-AR hidden state. The first product is long but semantically meaningful. The second is short and local.

#### Worked micro-example

Suppose a 5-second clip has $T=105$ frames at 21 Hz. Flattening ten codebooks produces 1,050 audio-token positions before adding text and reference context. Dual-AR still generates ten tokens per frame, but only one token per frame extends the long temporal chain. The other nine decisions are handled by the small Fast AR using the current frame’s condition.

```python
import torch

class ToyDualAR(torch.nn.Module):
    def __init__(self, semantic_vocab, acoustic_vocab, d_model=256):
        super().__init__()
        self.slow = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True),
            num_layers=4)
        self.semantic_head = torch.nn.Linear(d_model, semantic_vocab)
        self.fast = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True),
            num_layers=2)
        self.acoustic_heads = torch.nn.ModuleList(
            [torch.nn.Linear(d_model, acoustic_vocab) for _ in range(9)])

    def forward(self, temporal_inputs, acoustic_prefix):
        # temporal_inputs: [batch, time, d_model]
        slow_hidden = self.slow(temporal_inputs)
        semantic_logits = self.semantic_head(slow_hidden)  # [B, T, V_sem]
        # acoustic_prefix contains projected slow hidden + q^(0)_t seed.
        fast_hidden = self.fast(acoustic_prefix)            # [B, T, 10, d_model]
        acoustic_logits = [head(fast_hidden[:, :, k + 1])
                           for k, head in enumerate(self.acoustic_heads)]
        return semantic_logits, acoustic_logits
```

#### Why it works, and when it fails

The architecture spends expensive context capacity on the dependency that really needs it: language, speaker turns, and long-range prosody. Fine acoustic detail is generated with a cheap conditional model. The paper reports this as the foundation for both long-form stability and efficient inference.

The boundary is important. Fast AR is not independent of Slow AR; it inherits errors from the semantic token and hidden state. If the Slow AR skips a word, the Fast AR can make the wrong word sound beautiful. Conversely, if the Fast AR is too weak, semantic accuracy may remain excellent while timbre or high-frequency detail degrades.

### 4. Multi-Codebook Fusion

#### The problem it solves

After timestep $t$ is complete, the Slow AR must decide what comes next. If it receives only $q_t^{(0)}$, it cannot directly see whether the acoustic decoder produced an unusual prosodic or timbral realization. If it receives ten independent tokens without a common vector space, the temporal Transformer has no convenient input representation.

#### Intuition: a production report with two summaries

Each frame produces a semantic summary and a detailed acoustic report. Before the next decision, we add all reports into one vector. The semantic token is represented twice: once through the language model’s normal token embedding and once through the dedicated codebook embedding. They are related but independently learned views.

#### Mechanism

Every codebook index $q_t^{(k)}$ passes through a dedicated embedding table $E^{(k)}$ whose output has the Slow AR’s embedding dimension. The language model embedding $e_t^{LM}$ for the semantic token is added as well. The sum becomes $x_{t+1}$, the input embedding for the next temporal step.

![Redrawn diagram: Multi-Codebook Fusion adds all codebook embeddings into the next temporal input](/imgs/blogs/fish-audio-s2-controllable-text-to-speech-2.webp)

#### The math

The paper defines:

$$
x_{t+1}=e_t^{LM}+\sum_{k=0}^{N-1}E^{(k)}\!\left[q_t^{(k)}\right],\qquad N=10.
$$

Here $e_t^{LM}\in\mathbb{R}^{d_{slow}}$ is the ordinary Slow-AR token embedding, and each $E^{(k)}$ maps a discrete codebook index to $\mathbb{R}^{d_{slow}}$. The sum is therefore well-defined in the same hidden space. The fact that $q_t^{(0)}$ appears in both $e_t^{LM}$ and $E^{(0)}[q_t^{(0)}]$ is intentional: one embedding is optimized as a language token representation, the other as a codec representation.

#### Worked example

With a toy embedding size of three, suppose:

```python
import torch

e_lm = torch.tensor([1.0, 0.0, 0.0])
codebook_embeddings = [
    torch.tensor([0.0, 1.0, 0.0]),  # semantic codebook
    torch.tensor([0.0, 0.0, 1.0]),  # acoustic codebook 1
    torch.tensor([0.1, 0.1, 0.0]),  # acoustic codebook 2
]
x_next = e_lm + torch.stack(codebook_embeddings).sum(dim=0)
print(x_next)  # tensor([1.1, 1.1, 1.0])
```

The next timestep sees a mixture of linguistic and acoustic evidence. In a real model the embeddings are learned vectors, not interpretable basis axes, but the addition has the same shape logic.

#### Why it works, and when it fails

Fusion is cheap: it adds ten vectors instead of extending the Slow AR sequence by nine extra positions. It also gives the temporal model feedback about the complete frame. The risk is interference. If one codebook’s embedding norms become much larger than the others, it can dominate the sum. A production implementation should monitor per-codebook norms and may need normalization or careful initialization, although the report does not claim such a normalization step.

### 5. Natural-language inline control and multi-speaker dialogue

#### The problem it solves

A global instruction such as “read this sadly” does not say where an inhale, laugh, whisper, or emphasis should happen. A separate control-token vocabulary is precise but expensive to enumerate and hard to extend. Multi-speaker generation adds another alignment problem: the model must associate each turn with the correct voice.

#### Intuition: stage directions embedded in a script

A screenplay can write “the character whispers” immediately before one line, “laughs” in the middle of another, and switch characters with a speaker label. The direction is local to the words it modifies. Fish Audio S2 trains on transcripts that use natural-language tags such as `[prolonged laugh]`, `[inhale]`, `[angry]`, `[emphasis]`, and `[in a hurry]`, together with speaker markers such as `<|speaker:0|>`.

The important mapping is not “special tag ID → fixed acoustic template.” It is “textual description at a position → learned acoustic variation at that position.”

#### Mechanism

The rich-transcription ASR produces spoken words, speaker turns, vocal events, prosody, and natural disfluencies in one text stream. During pre-training and SFT, the model learns to predict audio tokens conditioned on that stream. At inference time, a user can place a description near a word or phrase instead of providing one global style prompt.

```python
dialogue = """<|speaker:0|> I thought you were coming.
<|speaker:1|> [inhale] I was, but then—[prolonged laugh]—the train stopped.
<|speaker:0|> [whisper] Are you serious?"""

# The actual repository API may wrap this in a chat template. The important
# invariant is that speaker IDs and inline instructions are in the same
# autoregressive context as the target text.
prompt = {
    "system": "Generate a natural multi-speaker dialogue.",
    "reference_audio": "speaker_0.wav",
    "text": dialogue,
}
```

#### Why it works, and when it fails

Inline control is expressive because it shares the model’s main language interface. It can also compose: speaker tags, emotion descriptions, and local events coexist in one sequence. The failure mode is ambiguity. “Sound excited” is not a deterministic acoustic specification, and a tag placed too far from its target may be ignored. A rich-transcription model can also hallucinate an event in the training caption; those errors become supervision rather than mere metadata noise.

### 6. The multi-purpose data pipeline

#### The problem it solves

Large-scale speech collections contain background music, overlapping speakers, silence, bad microphones, inaccurate transcripts, and subtle vocal events that are not labeled. Filtering only by waveform quality leaves semantic and speaker-turn errors. Annotating every event by hand does not scale.

There is a second problem in RL. If the reward model is trained on a distribution unrelated to the pre-training data, the policy may optimize quirks of the reward rather than the intended speech quality. The paper calls this distribution mismatch.

#### Intuition: the same inspector at intake and at the end of the factory

Imagine a factory where the same inspector checks incoming parts and tests the finished product. The factory’s definition of “acceptable” stays consistent. Fish Audio S2 uses that pattern: the speech-quality model and rich-transcription ASR first curate data, then evaluate generated audio during RL.

![Figure 3 from Fish Audio Team (2026): the three-stage data pipeline](/imgs/blogs/fish-audio-s2-controllable-text-to-speech-fig3.webp)

#### Mechanism

**Stage 1: source separation and segmentation.** A vocal separation module isolates speech. Voice Activity Detection slices continuous audio into utterance-level segments.

**Stage 2: quality filtering.** A w2v-BERT 2.0 backbone plus an MLP head scores signal-to-noise ratio, speaker consistency, recording quality, and intelligibility. It is trained on thousands of hours with human labels using MSE and focal loss. Low-quality segments are removed.

**Stage 3: rich transcription.** A fine-tuned Qwen3-Omni-30B-A3B produces text, speaker turns, and vocal-event descriptions. The resulting transcript is a natural-language control surface for training.

In RL, the quality model becomes the acoustic preference reward. The ASR model re-transcribes generated speech and compares it with the requested text and tags, providing semantic and instruction-following reward.

![Redrawn diagram: the same evaluation stack serves data curation and RL reward scoring](/imgs/blogs/fish-audio-s2-controllable-text-to-speech-3.webp)

#### The math

Let $a$ be an audio segment, $Q(a)$ the quality score, and $C(a)$ the rich caption. The curation process can be represented abstractly as:

$$
\mathcal{D}_{clean}=\left\{(a,C(a))\;\middle|\;Q(a)\geq\tau_Q\right\}.
$$

Here $\tau_Q$ is a filtering threshold. For a generated waveform $\tilde a$, the same functions produce reward components:

$$
R_{\mathrm{Pref}}(\tilde a)=Q(\tilde a),\qquad R_{\mathrm{STT}}(\tilde a)=\operatorname{match}\!\left(C(\tilde a),\text{requested text/tags}\right).
$$

The key design property is not that the two functions are perfect. It is that the policy is trained on examples selected and described by measurements related to the measurements used to optimize it.

![Figure 3 from Fish Audio Team (2026): data filtering and rich transcription before model training](/imgs/blogs/fish-audio-s2-controllable-text-to-speech-fig3.webp)

#### Worked example

Suppose three candidate clips contain the same sentence. Clip A has clean audio but misses `[laugh]`. Clip B includes the laugh at the right position but has background music. Clip C includes the laugh and is clean but changes the speaker’s timbre. A transcript reward favors B and C; a quality reward favors A and C; a similarity reward favors A and B. Reusing multiple evaluators makes C the likely winner rather than allowing any one defect to dominate.

#### Why it works, and when it fails

The pipeline turns unstructured audio into paired supervision: words plus instructions. Reusing evaluators makes the data and RL stages operationally coherent. It does not eliminate bias. Human labels used to train the quality model may encode a particular recording style; the ASR may be weaker in low-resource languages; and a model judging its own data can amplify systematic mistakes. The paper reports roughly 80 languages and dialects in pre-training, but evaluator quality is unlikely to be uniform across all of them.

### 7. Pre-training, SFT, and masking

#### The problem it solves

The model must learn several distributions at once: ordinary text, reference-audio conditioning, semantic audio tokens, and acoustic refinement. Training every position equally can teach the model to copy reference audio or over-invest in codebooks that are not sampled the same way at inference.

#### Intuition: rehearsal versus performance

During rehearsal, an actor may practice every line, including stage directions and timing cues. During performance, we care most about the parts the audience hears. Fish Audio S2 uses broad objectives during pre-training, then changes the masking and weighting in SFT to better match generation.

The reference audio is prepended to the system prompt rather than appended to the user input as in Fish Audio S1. Its loss is masked, preventing verbatim memorization. Fine-grained instructions are inserted at word or phrase positions.

#### The math

For the Slow AR, the report defines:

$$
\mathcal{L}_{slow}=-\sum_{t=0}^{T-1}m_t\lambda_t\log P(x_t\mid x_{<t}).
$$

Here $x_t$ is the target token, $x_{<t}$ is the previous context, $m_t\in\{0,1\}$ is the reference mask, and $\lambda_t$ is a position weight. Reference prompt and reference-audio positions have $m_t=0$; supervised target positions have $m_t=1$.

The Fast AR loss is:

$$
\mathcal{L}_{fast}=-\frac{1}{\sum_{k=1}^{N-1}w^{(k)}}
\sum_{k=0}^{N-1}w^{(k)}\log P\!\left(q_t^{(k)}\mid h_t^{slow},q_t^{(<k)}\right).
$$

The weight $w^{(k)}$ controls the importance of codebook $k$. During pre-training, the report uses uniform weights and includes semantic-token prediction as an auxiliary task. During SFT, semantic-token prediction is removed from the Fast AR and later codebooks receive progressively decayed weights, concentrating capacity on perceptually important coarse acoustic layers.

The combined objective is:

$$
\mathcal{L}_{total}=\lambda_{slow}\mathcal{L}_{slow}+\lambda_{fast}\mathcal{L}_{fast}.
$$

#### Worked example

Consider a sequence `[system, reference-audio, target-word-1, target-word-2]`. The mask is `[0, 0, 1, 1]`. The model still reads the first two positions as context, but gradients from their next-token losses are suppressed. This lets the reference voice condition the model without rewarding it for reproducing the reference token sequence.

```python
import torch
import torch.nn.functional as F

logits = torch.randn(4, 32)                 # [sequence, vocabulary]
targets = torch.tensor([4, 8, 11, 7])
mask = torch.tensor([0.0, 0.0, 1.0, 1.0])  # hide reference positions

token_loss = F.cross_entropy(logits, targets, reduction="none")
slow_loss = (token_loss * mask).sum() / mask.sum().clamp_min(1.0)
slow_loss.backward()
```

#### Why it works, and when it fails

Masking separates conditioning from imitation. Differential learning rates further protect the text foundation model while adapting audio modules. The tradeoff is optimization complexity: a poor mask can remove useful supervision, while an aggressive acoustic-weight decay can leave fine detail undertrained. The report also uses FSDP and Warmup-Stable-Decay scheduling, so reproducing the result requires more than copying the loss equations.

### 8. GRPO and Dr.GRPO for long speech

#### The problem it solves

Supervised likelihood rewards matching the dataset, but it does not directly penalize skipped words, timbre drift, or ignored instructions in generated audio. RL can optimize those properties, yet PPO normally needs a value model to estimate advantages. For long audio, maintaining another large model and computing value estimates is expensive.

#### Intuition: a panel audition

Instead of asking a critic to predict the absolute quality of one performance, audition several performances of the same script. If one is better than the group average, reinforce it. If another is worse, reduce its probability. The group mean is a cheap local baseline.

![Redrawn diagram: group-relative advantage replaces a separate value model in the RL loop](/imgs/blogs/fish-audio-s2-controllable-text-to-speech-5.webp)

#### Mechanism

For one prompt, Fish Audio S2 samples $G$ candidate outputs. Each is scored by the composite reward system. The advantage is the candidate reward minus the intra-group mean. The same advantage supervises the Slow AR sequence and the Fast AR acoustic tokens. A KL penalty keeps the updated policy near a reference policy.

The report follows Dr.GRPO in removing division by intra-group standard deviation. The stated motivation is to avoid sample-level difficulty bias: when rewards have low variance, standard-deviation normalization can make gradients disproportionately large.

#### The math

For candidates $y_1,\ldots,y_G$ with rewards $R_1,\ldots,R_G$:

$$
\bar R=\frac{1}{G}\sum_{j=1}^{G}R_j,qquad A_i=R_i-\bar R.
$$

The Slow AR policy loss is described as:

$$
\mathcal{L}_{slow}^{RL}=-\frac{1}{|T|}\sum_{t=1}^{|T|}A_i\log\pi_\theta(x_t\mid x_{<t})+\beta D_{KL}^{(t)}.
$$

The Fast AR follows the same pattern over acoustic tokens:

$$
\mathcal{L}_{fast}^{RL}=-\frac{1}{C^{(k)}}\sum_{t,k}A_i\log\pi_\theta^{FA}\!\left(q_t^{(k)}\mid q_t^{(<k)}\right)+\beta D_{KL}^{(t,k)}.
$$

Here $\pi_\theta$ is the trainable policy, $\pi^{FA}_\theta$ is the Fast AR policy, $C^{(k)}$ is the codebook size used for normalization in the paper’s notation, and $\beta$ controls the KL penalty. The total objective is:

$$
\mathcal{L}_{RL}=\mathcal{L}_{slow}^{RL}+\gamma\mathcal{L}_{fast}^{RL}.
$$

#### Worked micro-example

Let a prompt produce four rewards `{0.8, 0.5, 0.4, 0.3}`. The group mean is `0.5`, so the advantages are `{+0.3, 0, -0.1, -0.2}`. The first sequence gets a positive likelihood gradient, the second is neutral, and the last two are pushed down. No absolute calibration of “good speech” is required for that update.

```python
import torch

rewards = torch.tensor([0.8, 0.5, 0.4, 0.3])
advantages = rewards - rewards.mean()
log_probs = torch.tensor([-12.0, -11.0, -13.0, -10.0], requires_grad=True)

# A simplified REINFORCE-style group loss.
loss = -(advantages.detach() * log_probs).mean()
loss.backward()
print(advantages)  # tensor([ 0.3000, 0.0000, -0.1000, -0.2000])
```

#### Why it works, and when it fails

The group baseline removes a large amount of variance without a value network. The downside is that candidates in the same group are not independent: a bad group can still provide a bad baseline. Reward gaps can also be noisy, especially when audio judges disagree. The shared advantage for every token is another approximation: a single output-level score does not identify which word or acoustic frame caused the quality difference.

### 9. Three rewards and anti-hacking

#### The problem it solves

Speech quality is multidimensional. Optimizing only ASR accuracy can produce intelligible but flat or robotic speech. Optimizing only acoustic preference can produce a beautiful clip that skips words. Optimizing only voice similarity can preserve timbre while ignoring an instruction.

#### Intuition: a three-legged stool

A stool with one strong leg still falls. Fish Audio S2 combines three legs:

- semantic accuracy and instruction following,
- acoustic quality and preference,
- speaker/timbre similarity.

Each evaluator catches a different shortcut.

#### Mechanism and math

The composite reward is:

$$
R_{total}=\lambda_{STT}R_{STT}+\lambda_{Pref}R_{Pref}+\lambda_{SIM}R_{SIM}.
$$

The ASR-based reward uses per-token confidence and stronger penalties for incorrect speaker IDs or missed vocal instructions. The preference reward comes from the speech-quality model. The similarity reward compares voiceprint features using cosine similarity. The weights $\lambda_{STT}$, $\lambda_{Pref}$, and $\lambda_{SIM}$ define the operating point.

The authors also decouple scoring asynchronously and cache generated waveforms. For KL computation, a full reference model need not remain in GPU memory: a LoRA backup is kept in CPU memory and swapped in for gradient-free reference-policy passes. They use rsLoRA with rank 16 and $\alpha=64$, updating only MLP layers.

![Redrawn diagram: semantic, acoustic, and speaker rewards are combined before policy updates](/imgs/blogs/fish-audio-s2-controllable-text-to-speech-4.webp)

#### Worked example

Suppose a candidate receives $R_{STT}=0.95$, $R_{Pref}=0.60$, and $R_{SIM}=0.85$, with weights `{0.5, 0.3, 0.2}`. Then:

$$
R_{total}=0.5(0.95)+0.3(0.60)+0.2(0.85)=0.795.
$$

Another candidate with scores `{0.80, 0.90, 0.70}` gets `0.81`. The second candidate wins despite lower semantic accuracy because it sounds substantially cleaner. The correct weights depend on product goals; the paper does not present a full sensitivity analysis over them.

#### Why it works, and when it fails

Orthogonal rewards make common hacks more difficult. They do not make hacking impossible. A reward model can be fooled by artifacts, ASR confidence can be over-trusted, and a voiceprint model may reward similarity that human listeners perceive as unpleasant. The most important operational detail is asynchronous scoring: expensive judges must not leave the training GPU idle while rollouts wait.

### 10. Inference with SGLang

#### The problem it solves

An excellent model can still be unusable if it starts too slowly or wastes GPU bandwidth. TTS has an additional constraint compared with ordinary text generation: the user wants the first audible chunk quickly, and the decoder must keep producing chunks in real time.

#### Intuition: an LLM server with an audio exit ramp

SGLang already knows how to schedule autoregressive language models, batch requests, replay CUDA graphs, and cache shared prefixes. Fish Audio S2 keeps that machinery. It bypasses text tokenization at the API boundary, accepts mixed semantic/acoustic token prompts, extends RadixCache keys to multiple tokens, and sends generated audio token IDs to the tokenizer decoder.

![Redrawn diagram: reference-prefix caching and co-scheduled decoding create the streaming path](/imgs/blogs/fish-audio-s2-controllable-text-to-speech-6.webp)

Reference audio is placed in the system prompt. When the same voice is reused, its deterministic token prefix can hit the Radix tree cache. The report measures an average prefix-cache hit rate of 86.4% and over 90% at peak. It also uses MPS to co-schedule vocoder decoding with LLM decoding on the same GPU because LLM decoding is primarily memory-bandwidth bound in this setting.

#### The math of the serving metrics

The real-time factor is conventionally:

$$
RTF=\frac{\text{wall-clock generation time}}{\text{duration of generated audio}}.
$$

An RTF below 1 means faster-than-real-time generation. The report’s `0.195` means, under its stated H200 serving setup, five seconds of audio takes roughly 0.975 seconds of generation time in the idealized ratio. Time-to-first-audio (TTFA) measures the delay until the first playable audio arrives; the paper reports below 100 ms.

```python
import time

def measure_rtf(generate_audio, seconds_expected):
    start = time.perf_counter()
    audio = generate_audio()
    elapsed = time.perf_counter() - start
    rtf = elapsed / seconds_expected
    return audio, {"elapsed_s": elapsed, "rtf": rtf}

first_audio_time = None
for chunk in stream_tts(prompt):
    if first_audio_time is None:
        first_audio_time = time.perf_counter()
    play(chunk)
```

#### Why it works, and when it fails

The serving design inherits mature LLM infrastructure. Prefix caching is especially useful when many requests reuse the same voice. The tradeoff is coupling: an audio model must obey the scheduler’s assumptions, cache-key semantics, and memory layout. Cache hits also depend on exact prefix reuse; small differences in reference-audio preprocessing or system prompts can destroy them.

## Training recipe in one pass

Fish Audio S2’s training pipeline has four stages.

1. **Tokenizer training:** a 446M-parameter audio tokenizer is trained for one million steps with a composite GAN loss.
2. **Pre-training:** the audio tokens are aligned with Qwen3-4B. One stage uses maximum context 8,192; another extends to 16,384 for long-form and multi-turn generation. The report states over ten million hours of raw audio across approximately 80 languages and dialects.
3. **SFT:** curated, high-quality labeled data improves expressiveness and controllability. The vocabulary adds structural control tokens and 4,096 semantic tokens. New embeddings are initialized from a multivariate normal distribution using the existing text embedding mean and covariance.
4. **RL post-training:** group rollouts, multi-dimensional rewards, KL-regularized policy updates, asynchronous scoring, and LoRA reference-policy swapping address hallucinations, token skipping, and timbre drift.

The details matter because the headline architecture is only one component. A smaller implementation with the same Dual-AR split but weaker data, tokenizer, or reward models should not be expected to reproduce the report’s results.

## Experiments and results

The report evaluates both objective correctness and higher-level subjective behavior. WER and CER are useful because speech systems must say the requested words. Speaker similarity measures voice preservation. LLM-as-a-Judge tests whether the result sounds natural and follows expressive instructions.

| Benchmark | Fish Audio S2 result | Comparison or interpretation |
|---|---:|---|
| Seed-TTS-Eval, Chinese test-zh | WER 0.54% | Ties Fish Audio S1 and beats the listed Qwen3-TTS result of 0.77% |
| Seed-TTS-Eval, English test-en | WER 0.99% | Better than Fish Audio S1 at 1.07% and Seed-TTS at 2.25% |
| Seed-TTS-Eval, zh-hard | WER 5.99% | Competitive with CosyVoice 3 at 5.83%; better than Fish Audio S1 at 17.00% |
| CV3-Eval average | WER 3.01% | Down from Fish Audio S1’s 3.96%, a reported 23.9% relative reduction |
| Long-TTS-Eval English | WER 4.38%; SIM-Mean 0.523; SIM-Std 0.0761 | Better WER and speaker mean than Fish Audio S1 |
| Long-TTS-Eval Chinese | CER 5.95%; SIM-Mean 0.557; SIM-Std 0.0923 | Better CER and speaker mean than Fish Audio S1 |
| Audio Turing Test | posterior mean 0.483 | Improves to 0.515 with instruction rewriting |
| EmergentTTS-Eval | 81.88% overall win rate | Compared against the benchmark baseline; strongest listed dimensions include paralinguistics at 91.61% |
| Fish Instruction Benchmark, Chinese | TAR 0.984; naturalness 4.40; expressiveness 4.94 | All improve over Fish Audio S1 |
| Fish Instruction Benchmark, English | TAR 0.881; naturalness 4.21; expressiveness 4.50 | Larger gains over Fish Audio S1 than on the Chinese set |
| Production serving | RTF 0.195; TTFA <100 ms | Measured on one NVIDIA H200 according to the report |

The multilingual table is particularly revealing. On the 24-language MiniMax test set, Fish Audio S2 has the lowest WER in 11 languages and the highest speaker similarity in 17. It is not uniformly best: the report notes that MiniMax Speech and ElevenLabs retain advantages in some low-resource languages, often those with fewer than 1,000 hours of training data.

The long-form table also shows why multiple metrics matter. Fish Audio S2 has better English WER than VibeVoice, but VibeVoice has lower SIM-Std, meaning more stable speaker embeddings under that metric. A system can be more intelligible while another is more consistent in timbre. That is exactly the kind of trade-off a single score hides.

### What is load-bearing and may not transfer?

Several factors are likely load-bearing:

- The scale and diversity of the data: more than ten million hours and approximately 80 languages are not a small implementation detail.
- Rich transcription quality: inline control only works if the annotations place events at useful positions.
- Evaluator quality: RL follows the speech-quality model, ASR, and voiceprint model as much as it follows the nominal objective.
- The reference-audio placement and cache design: the low TTFA result assumes repeated prefixes can actually hit RadixCache.
- The benchmark prompt protocol: EmergentTTS-Eval uses Gemini 3 Pro to rewrite prompts and Gemini 2.5 Pro as judge. That measures a particular instruction pipeline, not only raw model behavior.

### The new instruction benchmark

The Fish Audio Instruction Benchmark evaluates inline tags such as `[laugh]`, `[whispers]`, `[inhale]`, `[exhale]`, and `[emphasis]` at specific word positions. It uses roughly 500 utterances for each of English and Chinese settings, with MELD supplying the English dialogue data and a game-character voice dataset supplying Chinese examples.

The benchmark reports Tag Activation Rate, Acoustic Naturalness, and Global Expressiveness. The authors also compare Gemini 3 Pro with human annotators on 200 clips. Agreement for event detection is 76.2% with Cohen’s $kappa=0.47$. Pearson correlations for naturalness and expressiveness are 0.55 and 0.42, with QWK scores of 0.36 and 0.47.

Those validation numbers are useful precisely because they are imperfect. Gemini is a practical screening judge, but its score should not be interpreted as a noiseless substitute for a human listening panel.

## Critique

### What is genuinely strong

**The representation and serving story fit.** Dual-AR makes the audio model look like an autoregressive language model at the temporal level. That lets the team reuse SGLang’s batching, KV caching, and prefix caching instead of creating a completely separate serving stack.

**The data pipeline treats annotation as a model capability.** A clean transcript is not enough for expressive TTS. Building rich transcription into the curation process is a practical answer to the annotation bottleneck.

**Reward reuse is a sensible systems choice.** The same quality vocabulary is used to select data and score outputs. This does not remove evaluator bias, but it makes the source of the bias more inspectable than an unrelated black-box reward model.

**The evaluation is broader than WER.** The paper includes long-form, multilingual, speaker similarity, instruction following, naturalness, and serving latency. For a production TTS system, that breadth is more meaningful than one leaderboard number.

### What is weak or under-specified

**The technical report is not a full reproducibility paper.** It names major models, data scale, tokenizer structure, and several optimization choices, but many details needed for a faithful reproduction are absent: exact codebook sizes, context-window sizes, reward weights, learning rates, rollout group size, KL coefficients, and filtering thresholds.

**Ablations are limited.** We want to see at least the following comparisons: Dual-AR versus flattened AR at matched compute; data filtering without rich transcription; rich transcription without reward reuse; each reward alone versus the full reward; and GRPO with versus without standard-deviation normalization. Without these, the report shows that the whole recipe works, not how much each ingredient contributes.

**The new benchmark is model-adjacent.** The Fish Instruction Benchmark is valuable, but it is built and judged by the same organization that builds the model. The paper acknowledges limited diversity, imbalanced tag distributions, and early-stage human-model alignment. External raters and a held-out tag taxonomy would make the claim stronger.

**LLM judges can create protocol leakage.** EmergentTTS-Eval uses prompt rewriting with Gemini 3 Pro and judging with Gemini 2.5 Pro. Fish Audio S2 may indeed follow rewritten instructions well, but the result reflects the whole rewrite–synthesis–judge pipeline. Direct human preference tests and judge swaps would test robustness.

**The report’s “stable long-form” claim needs stress tests.** The Long-TTS-Eval samples are truncated at context limits and cap expected audio around 185 seconds. That is a useful benchmark, but it does not establish stability for arbitrary multi-minute conversations, very long speaker alternations, or context lengths beyond the training regime.

**Safety and consent are not central in the evaluation.** A multi-speaker voice-cloning system should be tested for impersonation misuse, speaker consent, watermarking, and abuse-resistant serving. Those concerns are outside the report’s main technical scope, but they matter for production deployment.

### What would change my mind?

What would change my mind is a public, matched-compute ablation suite showing that Dual-AR, evaluator reuse, and each reward component independently improve held-out human ratings and objective metrics across languages, plus an external human study that reproduces the instruction-following gains without prompt rewriting or a Gemini judge.

## What I’d build with this

These are extrapolations, not claims made by the paper.

1. **A reward dashboard with per-event attribution.** Instead of returning only one group reward, log word-level ASR confidence, tag activation, speaker similarity by 3-second chunk, and acoustic quality over time. This would reveal whether an RL update improved laughter while damaging the preceding phoneme.

2. **A constrained inline-control decoder.** The model could maintain a small control state for pending tags: a `[whisper]` tag remains active until a boundary, while `[laugh]` is consumed once. That would make scope explicit without replacing natural-language instructions.

3. **External evaluator ensembles.** Use independent ASR, speech-quality, and voiceprint models during validation. Training can use the Fish evaluators, but a held-out evaluator family would make reward hacking easier to detect.

4. **Adaptive Fast AR depth.** Easy frames such as silence or stable vowels may not require all nine acoustic refinement steps. A confidence-aware Fast AR could spend more computation on fricatives, laughter, and speaker transitions.

5. **Streaming-aware RL.** Add TTFA and chunk-boundary continuity to the reward. A waveform can sound excellent offline but click at chunk boundaries or delay the first audio. Production behavior should be part of the optimization target, not only an inference benchmark.

## A compact implementation checklist

If we wanted to build a smaller research prototype inspired by Fish Audio S2, the order matters:

1. Train or adopt a causal RVQ tokenizer and verify reconstruction at the target sample rate.
2. Measure the first codebook’s linguistic information with a frozen speech-model probe.
3. Implement Slow AR semantic generation and a small Fast AR acoustic decoder separately.
4. Add Multi-Codebook Fusion and verify all embeddings have the same hidden dimension.
5. Create rich transcripts with speaker tags and inline events, then inspect them manually before scaling.
6. Train quality, ASR, and speaker-similarity evaluators independently of the policy.
7. Start with supervised training and masked reference-audio loss.
8. Add group-relative RL only after reward correlations and failure cases are visible.
9. Measure quality and streaming latency separately. Do not infer one from the other.

The most important engineering lesson is that the model is not the whole system. Tokenizer semantics, annotation quality, reward design, rollout throughput, cache reuse, and evaluator validity all sit on the critical path.

## References

- Fish Audio Team. [Fish Audio S2 Technical Report](https://arxiv.org/abs/2603.08823).
- [Fish Speech source code](https://github.com/fishaudio/fish-speech).
- [Fish Audio S2 Pro weights](https://huggingface.co/fishaudio/s2-pro).
- [MiniMax Speech: Speaker Encoder and Flow-VAE](/blog/paper-reading/speech-processing/minimax-speech-speaker-encoder-flow-vae).
- [Kimi-Audio](/blog/paper-reading/speech-processing/kimi-audio).
- [Qwen3-Omni Technical Report](/blog/paper-reading/multimodal/qwen3-omni-technical-report).
