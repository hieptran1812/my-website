---
title: "VALL-E: Text-to-Speech as Codec Language Modeling"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "How VALL-E turned text-to-speech into next-token prediction over EnCodec tokens, cloned an unseen voice from three seconds, and seeded the entire zero-shot TTS wave."
tags:
  [
    "audio-generation",
    "audio-synthesis",
    "text-to-speech",
    "voice-cloning",
    "neural-audio-codec",
    "language-models",
    "generative-ai",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Audio Generation"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/neural-codec-language-model-tts-vall-e-1.png"
---

The first time I cloned a voice from three seconds of audio, I did not quite believe the file I was listening to. I had recorded myself saying one short sentence on a laptop microphone in a slightly echoey room, fed those three seconds into a model, typed a completely different sentence, and the model read my new sentence back to me in my own voice, with my accent, my room's faint reverb, even the way I tend to trail off at the end of a clause. I had never trained anything. There was no fine-tuning step, no speaker enrollment, no "please read these twenty calibration sentences." The model had never heard my voice before that one clip and would forget it the moment the process ended. It had simply *continued* the voice it was given, the way a language model continues a prompt.

That experience is the whole subject of this post, and the model that first made it routine was **VALL-E**, from a Microsoft team in early 2023 (Wang and colleagues, "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers"). VALL-E did something that, in retrospect, looks obvious and at the time looked slightly mad. For two decades, text-to-speech meant *regression*: predict a mel-spectrogram from text, frame by frame, with a model trained to minimize a reconstruction loss, then hand that spectrogram to a vocoder. VALL-E threw that framing out. Instead of regressing a continuous spectrogram, it treated the discrete tokens of a [neural audio codec](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound) as a *language* and trained a plain autoregressive transformer to predict the next acoustic token, exactly the way you train GPT to predict the next word. Condition that language model on the phonemes of the text you want spoken plus a three-second acoustic prompt of a target speaker, and it generates the codec tokens of that speaker saying that text. Decode the tokens with the codec, and you have a waveform.

![A dataflow graph showing phonemes and a three-second acoustic prompt feeding an autoregressive model for the first codebook and a non-autoregressive model for the rest, producing a codec token grid that EnCodec decodes to a waveform](/imgs/blogs/neural-codec-language-model-tts-vall-e-1.png)

The figure above is the entire architecture on one slide, and it is worth holding in your head for the rest of this post. Two inputs go in: the phoneme sequence of the target text, and a short acoustic prompt that *is* the speaker. Two transformers do the work: an autoregressive model that generates the first, coarse codebook of [residual-vector-quantization tokens](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq) one frame at a time, and a non-autoregressive model that fills in the remaining seven fine codebooks in parallel. The codec decodes the resulting token grid into 24 kHz speech. By the end of this post you will understand exactly why this factorization works, why an in-context acoustic prompt is enough to clone a voice with no training, why the coarse-to-fine split between the two models follows directly from the structure of residual VQ, why scale rather than architecture was the real unlock, and what breaks when you push it. You will also see runnable code for the inference flow and an open stand-in you can actually execute today.

This sits squarely on the series spine: the **audio stack** of waveform to codec tokens to generative model and back to waveform, under the tension of **fidelity, controllability, speed, and length**. VALL-E is a controllability story above all. It buys you something no prior TTS system had, the ability to specify *any* voice at inference time with three seconds of reference audio, and it pays for that with the instabilities autoregressive sampling always brings. If you have not read [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard), the framing there about audio being short in wall-clock time but enormous in token count is exactly the pressure that shapes everything VALL-E does. And if you want the prior chapter of this exact story, [from Tacotron to VITS](/blog/machine-learning/audio-generation/text-to-speech-from-tacotron-to-vits) covers the regression-based TTS that VALL-E was reacting against.

## What "TTS as language modeling" actually means

Let me make the reframing precise, because the whole post turns on it, and the precise version is more interesting than the slogan.

Classical neural TTS is a *conditional regression* problem. You have text, you want a spectrogram, and you train a model $f_\theta$ to map one to the other by minimizing a per-frame reconstruction loss, typically an L1 or L2 distance between the predicted mel-spectrogram $\hat{Y}$ and the ground-truth mel $Y$:

$$
\mathcal{L}_\text{reg} = \frac{1}{T}\sum_{t=1}^{T} \lVert \hat{Y}_t - Y_t \rVert_1, \qquad \hat{Y} = f_\theta(\text{text}).
$$

This is the Tacotron and FastSpeech lineage. It works, and for a fixed single speaker it works very well, but it has a built-in flaw that matters enormously for voice cloning. A regression loss is minimized by predicting the *conditional mean* of the target. When a given phoneme sequence could be spoken many valid ways, with different pitch contours, different pacing, different emphasis, the L1-optimal prediction is the *average* of all of them, which is a blurry, over-smoothed, slightly lifeless spectrogram. This is why early neural TTS sounded subtly robotic even when it was intelligible: the regression objective washed out the natural variability of real speech. The field papered over this with adversarial losses, flow-based posteriors (VITS), and duration predictors, but the underlying tension never fully went away.

VALL-E sidesteps the whole thing by changing the *target representation* and the *loss*. Instead of regressing a continuous spectrogram, it predicts *discrete codec tokens* with a categorical cross-entropy loss, which is the natural objective for a language model. Concretely, let the EnCodec codec turn a waveform into a sequence of discrete token vectors $C = [c_1, c_2, \ldots, c_T]$, where each $c_t$ is itself a length-8 vector of codebook indices (one index per RVQ codebook). VALL-E models the conditional distribution

$$
p(C \mid x, \tilde{C}) = \prod_{t} p\big(c_t \mid c_{\lt t}, x, \tilde{C}\big),
$$

where $x$ is the phoneme sequence and $\tilde{C}$ is the acoustic prompt (the codec tokens of the three-second reference). This is *exactly* the factorization a language model uses, $p(\text{sequence}) = \prod_t p(\text{token}_t \mid \text{earlier tokens})$, except the "tokens" are slices of an audio codec grid rather than subwords of text. The training objective is the standard next-token cross-entropy:

$$
\mathcal{L}_\text{AR} = -\sum_t \log p_\theta\big(c_t^{(1)} \mid c_{\lt t}^{(1)}, x, \tilde{C}\big),
$$

written here for the first codebook, which is the part the autoregressive model handles. We will get to why only the first codebook is autoregressive in a moment.

The shift from regression to categorical prediction is the single most important conceptual move, so let me dwell on *why* it fixes the over-smoothing. A categorical distribution over a codebook can represent *multimodality* directly: if a phoneme can be realized as either a rising or a falling pitch, the model can put probability mass on the tokens for both, and *sampling* picks one. There is no averaging. The model never has to commit to a blurry compromise; it commits to one concrete realization at sampling time, the way GPT commits to one concrete next word even though many are plausible. This is precisely the same argument that makes autoregressive image models produce sharp, varied images where a naive regression-to-pixels would produce mush, and it is exactly the parallel drawn in [autoregressive image models](/blog/machine-learning/image-generation/autoregressive-image-models) over in the image series. Discreteness plus sampling buys you back the natural variability that a regression loss destroys.

There is a second, subtler payoff that turns out to be the real prize. Once TTS is *language modeling*, it inherits the property that made large language models magical: **in-context learning**. A language model conditioned on a prompt continues *in the style of that prompt*. Give GPT a few lines of Shakespeare and it continues in iambic pentameter; give it a Python function signature and it writes Python. VALL-E conditions on an acoustic prompt, and so it continues *in the voice of that prompt*. The speaker is not a parameter the model learned during training. It is part of the context, supplied fresh at inference. That is the entire mechanism of zero-shot cloning, and it is why the reframing matters far beyond fixing over-smoothing.

## Why an in-context acoustic prompt clones a voice

Here is the question that trips up everyone the first time: how can a model clone a voice it has never heard, with no training, just from three seconds of audio? The answer is the most beautiful part of VALL-E, and it follows directly from the language-modeling framing.

Think about what the autoregressive objective forces the model to learn during training. The model sees millions of utterances. For each one, it is given some early portion of the codec tokens as context and asked to predict the continuation. The crucial fact about speech is that **acoustic identity is roughly constant within an utterance**. The speaker does not change mid-sentence. The recording conditions, the microphone, the room, the timbre, the average pitch register, all of these persist from the first frame to the last. So when the model is predicting frame $t$ from frames $c_{\lt t}$, the single most reliable signal in that context for *how the next frame should sound* is the *acoustic character of the frames it has already seen*. To minimize its loss, the model is forced to learn a function that reads "what voice is this, what room is this, what register is this" out of the preceding tokens and continues consistently. It has to, because an utterance where the voice flips halfway through is wildly improbable in the training data and would be heavily penalized.

So the model learns, as a *side effect of next-token prediction on consistent utterances*, to **continue the acoustic identity of its context**. Nobody told it to. There was no speaker-similarity loss, no speaker embedding network, no enrollment. The skill of "keep speaking in the voice you have been hearing" falls out of the prediction problem, exactly the way "keep writing in the style you have been reading" falls out of language modeling on text.

Now the cloning recipe is obvious. At inference, you *seed* the context with the three-second prompt of an unseen speaker. The model does what it always does, it continues the acoustic identity of its context, and that identity is now the target speaker. It does not know or care that this speaker was not in its training set. It is not retrieving a learned voice; it is *continuing* the one in front of it. This is why the cloning is genuinely zero-shot: the model's parameters never change, and the speaker enters purely through the context window.

![A before and after comparison contrasting classic multi-speaker TTS that needs a trained speaker embedding against VALL-E that clones an unseen voice from a three-second in-context prompt](/imgs/blogs/neural-codec-language-model-tts-vall-e-2.png)

The contrast above is the thesis stated as an engineering result. On the left is the old way: a *trained speaker table*. Multi-speaker Tacotron, FastSpeech, and VITS all learn a fixed embedding per speaker, or train a separate speaker-encoder, and to add a new voice you need minutes of clean audio and a training or adaptation pass. An *unseen* voice at inference is simply impossible; the model can only produce voices whose embeddings it learned. On the right is VALL-E: the speaker is supplied as a three-second prompt at inference, no retraining, and the model clones a voice it has never encountered. This is not a quantitative improvement on the old paradigm. It is a different paradigm.

It is worth being precise about *what* the prompt transfers, because it is more than just "timbre." Because the prompt is raw codec tokens of real audio, it carries everything the codec preserves: the speaker's vocal-tract timbre, yes, but also their prosodic tendencies, the emotional coloring present in the prompt, and crucially the *acoustic environment*, the room reverb, the microphone's coloration, the noise floor. If you prompt VALL-E with three seconds recorded over a phone line, the output sounds like it was recorded over a phone line. If you prompt it with three seconds of someone speaking angrily, the output inherits some of that energy. The model is continuing the *whole acoustic situation*, not a disentangled "voice" vector. This is a strength (rich, faithful cloning) and a weakness (you cannot easily clone the voice without also cloning the noisy room), and we will return to that trade.

#### Worked example: cloning a voice in your head, step by step

Let me run the mechanism concretely so it is not abstract. Suppose I want VALL-E to say "the quick brown fox jumps over the lazy dog" in a specific person's voice, and I have three seconds of them saying "thanks for calling, how can I help."

First, the text front-end converts my target sentence to phonemes: roughly `DH AH K W IH K B R AW N F AA K S ...`. Call this $x$. Second, the codec encodes the three-second reference into its token grid: about $3 \times 75 = 225$ frames, each with 8 codebook indices, so a $225 \times 8$ grid. Call this $\tilde{C}$, the acoustic prompt. Third, the autoregressive model is fed the concatenation of the phonemes and the prompt's first-codebook row, and it generates the first-codebook tokens for the *new* sentence, frame by frame, until it emits an end-of-sequence token that determines the output length, say about 150 frames for a two-second utterance. Fourth, the non-autoregressive model takes the phonemes, the full prompt grid, and the freshly generated first-codebook row, and produces the remaining seven codebooks for all 150 frames at once. Fifth, the codec decoder turns the completed $150 \times 8$ grid into a 24 kHz waveform.

The output is two seconds of new speech, in the reference speaker's voice, saying a sentence they never said. The model never updated a weight. The "voice" lived entirely in those 225 prompt frames. That is the whole trick, and once you have seen it work the magic dissolves into the same in-context-learning mechanism that makes prompting an LLM work.

## The two-stage design: AR for codebook one, NAR for the rest

Now to the part of VALL-E that looks like an arbitrary architectural choice but is actually forced by the structure of residual VQ: why generate the first codebook autoregressively and the remaining seven non-autoregressively? Why not one model for everything?

Recall how a residual vector quantizer encodes a frame, which we derived in detail in [residual vector quantization](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq). The first codebook quantizes the encoder's continuous vector to its nearest entry, leaving a residual. The second codebook quantizes *that residual*, leaving a smaller residual. The third quantizes the next residual, and so on for all eight codebooks. The key structural fact is **coarse-to-fine importance**: the first codebook carries the bulk of the energy and the most perceptually important information (the overall spectral shape, pitch, loudness, broad timbre), and each subsequent codebook adds a smaller, finer correction. The eighth codebook is the last little increment of texture that you would barely miss if it were dropped.

![A vertical stack showing conditioning feeding the autoregressive first codebook that sets timing and prosody, then the non-autoregressive codebooks that add coarse and fine detail in parallel](/imgs/blogs/neural-codec-language-model-tts-vall-e-3.png)

The stack above shows how VALL-E maps its two models onto that coarse-to-fine structure, and the mapping is the clever bit. The **first codebook is where the hard decisions live**. It sets the timing (how long the utterance is, where the pauses fall), the prosody (the pitch contour, the rhythm), and the coarse content. These are *sequential* decisions: where the next pause falls depends on where the previous words ended, the pitch contour unfolds over time, the length is determined as you go. This is exactly the kind of left-to-right dependency an autoregressive model is built for, and crucially it is where *length* is decided, because the AR model emits tokens until it chooses to stop. So VALL-E generates the first codebook with a proper autoregressive transformer, one frame at a time, conditioned on the phonemes and the prompt.

Once the first codebook is fixed, the remaining seven codebooks are a *different kind of problem*. They are corrections layered on top of an already-decided coarse signal. Given the first codebook for all frames, the fine codebooks for frame $t$ depend mostly on the first codebook *at that same frame* and its neighbors, not on a long sequential history of fine tokens. The hard, length-determining, prosody-setting work is already done. So VALL-E generates them **non-autoregressively**: a separate transformer takes the phonemes, the prompt, and the complete first-codebook row, and predicts codebooks 2 through 8 for *all frames at once*, in a handful of forward passes (one per codebook level, conditioning each level on the levels below it). The NAR model does not need to be autoregressive in time because the temporal structure is inherited from the first codebook.

This split is not a minor optimization. It is the difference between a system that is plausibly fast enough to use and one that is not. If VALL-E autoregressed all eight codebooks frame by frame, a one-second utterance at 75 frames per second would need $75 \times 8 = 600$ sequential forward passes. With the split, the AR model does 75 sequential passes (one per frame, first codebook only) and the NAR model does 7 forward passes total (one per remaining codebook), each producing all 150-or-so frames in parallel. The sequential bottleneck drops by nearly an order of magnitude. The split exploits the fact that *only the coarse structure is genuinely sequential*; the fine detail is conditionally parallelizable. That insight, that the RVQ hierarchy lets you autoregress only the part that needs it, is the engineering heart of VALL-E.

There is one more subtlety in the NAR stage worth naming because it explains a code detail later. The NAR model predicts codebook 2 conditioned on codebook 1, then codebook 3 conditioned on codebooks 1 and 2 (it sums their embeddings), and so on up to codebook 8. So within a frame the fine codebooks are still generated in a coarse-to-fine *order*, just not autoregressively in *time*. Each level is a single parallel forward pass over all frames, conditioned on the sum of all coarser levels. This respects the residual structure: codebook $k$ quantizes the residual after codebooks $1$ through $k-1$, so to predict it well you must know all of them.

### The exact factorization, written out

It pays to write the joint distribution VALL-E models explicitly, because the AR-versus-NAR split is precisely a *choice of factorization*, and seeing the factorization makes the architecture inevitable rather than arbitrary. Let the codec tokens for an utterance be $C \in \{1,\ldots,V\}^{T \times 8}$, a grid of $T$ frames by 8 codebooks, where $c_{t,k}$ is the index in codebook $k$ at frame $t$. We condition on phonemes $x$ and the prompt grid $\tilde{C}$. The full joint $p(C \mid x, \tilde{C})$ could be factored a thousand ways; VALL-E picks a specific one:

$$
p(C \mid x, \tilde{C}) = \underbrace{\prod_{t=1}^{T} p\big(c_{t,1} \mid c_{\lt t,1},\, x,\, \tilde{C}\big)}_{\text{AR over codebook 1}} \;\cdot\; \underbrace{\prod_{k=2}^{8} p\big(c_{:,k} \mid c_{:,\lt k},\, x,\, \tilde{C}\big)}_{\text{NAR over codebooks 2..8}}.
$$

Read the two factors carefully, because the whole design is in them. The **first factor** is autoregressive *in time*, $t$ ranges over frames, and each frame's first-codebook token depends on all *earlier frames'* first-codebook tokens $c_{\lt t,1}$. That is a left-to-right chain, exactly a language model, and it is where length is decided (the product runs until the model emits EOS). The **second factor** is autoregressive *in codebook depth* but *parallel in time*, the notation $c_{:,k}$ means "all frames of codebook $k$ at once," and it is conditioned on $c_{:,\lt k}$, all frames of all *coarser* codebooks. There is no $c_{\lt t}$ inside the second factor: given the coarser codebooks, the fine token at frame $t$ is conditionally independent of the fine tokens at other frames. *That conditional independence is the entire justification for non-autoregressive parallel generation.* The model is not assuming the fine codebooks are unconditionally independent across time (they are not); it is assuming they are independent *given the coarse codebooks*, which the residual structure makes approximately true because the coarse codebooks already carry the temporal correlation.

This is worth pausing on because it is the reusable lesson. Whenever you want to parallelize an autoregressive generator, you look for a conditional-independence structure to exploit: find a coarse variable that, once fixed, makes the fine variables independent, generate the coarse variable sequentially, then generate the fine variables in parallel. VALL-E found that structure handed to it by residual VQ, the first codebook is the coarse variable that decorrelates the rest. The same move powers parallel decoding in many modalities; here it is especially clean because the codec was *designed* to put the temporal-structure-bearing information in the early codebooks.

There is a cost to the conditional-independence assumption, and being honest about it matters. The fine codebooks are not *perfectly* independent given the coarse one; there is some residual temporal correlation the NAR model's per-frame parallelism cannot capture. In practice this is a minor quality loss, the fine codebooks are correcting small residuals where the perceptual stakes are low, so a slight independence error there costs little. But it is the reason a fully autoregressive model over all eight codebooks would, in principle, be marginally higher fidelity at vastly higher cost, and it is why some later systems revisit exactly how many codebooks to treat as parallel. The AR-one-NAR-rest split is a rate-distortion-flavored engineering compromise: it spends sequential compute only where the conditional dependence is strong (the coarse codebook) and parallelizes where it is weak (the fine codebooks), and the small fidelity it gives up there is a bargain for the order-of-magnitude speedup.

## How the tokens are laid out, and what the model actually sees

It helps to picture the literal grid the model is filling, because the AR-versus-NAR split is really a statement about *which cells get generated when*.

![A grid showing the codec token layout with prompt frames given, the first codebook row generated left to right by the autoregressive model, and the lower codebooks filled in parallel by the non-autoregressive model](/imgs/blogs/neural-codec-language-model-tts-vall-e-5.png)

The grid above is the codec token matrix: rows are codebooks (1 at the top through 8 at the bottom), columns are time frames. The leftmost columns are the *prompt*, which is given, not generated. The rest is what VALL-E produces. The top row, codebook 1, is filled left to right by the autoregressive model, each new cell depending on all cells to its left in that row. The lower rows, codebooks 2 through 8, are filled by the non-autoregressive model, which writes an entire row in one parallel pass conditioned on the rows above it. So the generation order is: walk the top row left to right (AR), then sweep down the remaining rows one at a time (NAR), each sweep parallel across all columns.

This picture makes the cost model obvious. The AR cost is proportional to the *number of frames* (the width of the grid), because that is how many sequential steps the top row takes. The NAR cost is proportional to the *number of codebooks minus one* (the height minus the AR row), because that is how many parallel sweeps you do, and each sweep is one forward pass regardless of width. For a typical configuration, 75 frames per second of output and 8 codebooks, a two-second utterance is a $150 \times 8$ grid: 150 sequential AR steps plus 7 parallel NAR sweeps. The AR steps dominate wall-clock time, which is why everything about VALL-E's speed and its instabilities comes down to the autoregressive top row.

It also makes the prompt's role concrete. The prompt is the leftmost block of *all eight rows*, fully given. The AR model sees the prompt's top row as the start of its sequence and literally continues it. The NAR model sees the prompt's full grid as context for predicting the fine rows of the generated region. So the speaker identity enters through *both* models: the AR model continues the prompt's coarse acoustic character (register, timbre, pacing), and the NAR model matches the prompt's fine texture (the exact spectral fingerprint, the room). That is why the cloning is faithful down to the recording environment, the fine codebooks copy the texture of the prompt's room because that texture is right there in the context.

## A matrix view: VALL-E against classic TTS

Before the code, let me put VALL-E side by side with the systems it displaced, because the comparison clarifies exactly what is new and what is borrowed.

![A matrix comparing Tacotron 2, VITS, and VALL-E across their generation target, zero-shot cloning ability, speaker input, and whether they are autoregressive](/imgs/blogs/neural-codec-language-model-tts-vall-e-4.png)

The matrix above lines up three representative systems on four axes that matter for cloning. **Tacotron 2** (Shen and colleagues, 2018) regresses a mel-spectrogram with an attention-based decoder, then a separate vocoder turns the mel into a waveform; it is autoregressive over mel frames, cannot do zero-shot cloning, and needs a trained speaker ID for multi-speaker use. **VITS** (Kim and colleagues, 2021) is end-to-end, going from text straight to waveform through a conditional VAE with a normalizing-flow prior and an adversarial decoder; it is non-autoregressive (it uses a duration predictor and parallel decoding), produces excellent single-speaker quality, but its multi-speaker variant still relies on learned speaker embeddings and clones unseen voices only weakly. **VALL-E** targets codec tokens, is autoregressive on the first codebook, takes a three-second prompt as its speaker input, and does genuine zero-shot cloning.

The axis that jumps out is the speaker input. Tacotron and VITS both encode the speaker as something *learned and fixed*. VALL-E encodes the speaker as something *given and arbitrary*. That single change, from a trained parameter to a runtime context, is the whole revolution, and everything else (the codec target, the autoregression, the cross-entropy loss) is in service of making it work. Note also the trade hiding in the "autoregressive" column: VITS is non-autoregressive and therefore stable and fast, but it gave up zero-shot cloning to get there; VALL-E recovered cloning by going back to autoregression and paid for it with the instabilities we will dissect later. There is no free lunch in this table; each system bought one property by spending another.

## The scale argument: data, not architecture, was the unlock

Here is the part of the VALL-E story that is easy to miss and most important to internalize. The architecture is not exotic. An autoregressive transformer over discrete tokens is the most standard thing in modern machine learning; a non-autoregressive transformer for the residual codebooks is a small variation. If you had described VALL-E's architecture to a researcher in 2020, nothing would have looked surprising. So why did zero-shot cloning suddenly work in 2023 and not before?

The answer is **scale of data**. Prior TTS systems trained on hundreds of hours of clean, studio-recorded, single-or-few-speaker audio (LJSpeech is about 24 hours; the multi-speaker VCTK is about 44 hours; LibriTTS is around 585 hours). VALL-E trained on **LibriLight**, roughly **60,000 hours** of English audiobook speech from over 7,000 speakers, two-plus orders of magnitude more data and speakers than the norm. This is the "GPT moment" for TTS in the most literal sense: the same thing that happened when language models scaled from BookCorpus to the open web happened to TTS when it scaled from LJSpeech to LibriLight. In-context learning *emerged* with scale.

Why does in-context cloning need that much data? Because the skill VALL-E relies on, "read the speaker out of the context and continue in that voice," only generalizes to *unseen* speakers if the training distribution covered enough speaker variation that the model learned the general operation rather than memorizing specific voices. With 44 hours and a handful of speakers, a model can memorize those speakers but has no pressure to learn the *abstract* skill of continuing an arbitrary voice, because it never sees enough different voices to need it. With 60,000 hours and 7,000+ speakers, the only way to fit the data is to learn the general skill: *given any preceding voice, continue it*. The speaker variation in the training set is what forces the model to treat the voice as a continuation problem rather than a lookup. This is the same reason large language models generalize their in-context skills, diversity in the training distribution converts a memorization problem into a generalization one.

This is the single most important lesson of VALL-E, and it is a strategic one, not a technical one: **the architecture was a known quantity; the unlock was deciding to train a standard codec language model on two orders of magnitude more speech than anyone had before.** It reframes how you should think about progress in audio generation. The bottleneck was rarely a clever module; it was usually data scale and the willingness to treat audio as just another token stream. Once you accept that, the path to better TTS looks a lot like the path to better LLMs: more data, more speakers, bigger models, better tokenizers.

#### Worked example: why 60k hours and not 600

Let me put rough numbers on the scale argument to make it concrete, with the caveat that these are illustrative orders of magnitude, not exact thresholds from an ablation.

Picture the model learning the skill "continue an arbitrary speaker." If your training set has $N$ distinct speakers, the model can succeed on *training* utterances by, in the worst case, memorizing $N$ voice profiles. To force *generalization* to a new speaker, you need $N$ large enough that memorization is no longer the cheapest way to fit the data, the model is better off learning the general continuation operation than storing thousands of individual profiles. Empirically, LibriTTS with roughly 2,400 speakers and 585 hours produced decent multi-speaker TTS but weak zero-shot cloning; LibriLight with 7,000+ speakers and 60,000 hours produced strong zero-shot cloning. The jump from hundreds to tens of thousands of hours, and from low-thousands to many-thousands of speakers, is where the in-context skill crossed from "memorized" to "generalized." If you only had 600 hours, you would get a competent multi-speaker model that clones *training* voices well and *unseen* voices poorly, which is exactly what the pre-VALL-E literature reported. The hundredfold data increase was the difference, and that is the number to carry away.

## The text front-end: from characters to phonemes

Before any of the neural machinery runs, VALL-E has to turn your text into phonemes, and this unglamorous front-end matters more than people expect. VALL-E conditions on *phonemes*, not raw characters, because phonemes are a far more direct encoding of *what sounds to make*. The mapping from spelling to sound in English is notoriously irregular ("though," "through," "tough," "cough" share letters but not sounds), so handing the model phonemes instead of letters removes a whole layer of ambiguity the acoustic model would otherwise have to learn. The component that does this is a **grapheme-to-phoneme** (G2P) converter, and in practice it is a small, separate, often rule-plus-lexicon system, not part of the transformer at all.

```python
# pip install phonemizer ; needs espeak-ng installed on the system
from phonemizer import phonemize

text = "the quick brown fox jumps over the lazy dog"
phones = phonemize(text, language="en-us", backend="espeak",
                   strip=True, preserve_punctuation=True,
                   with_stress=True)
print(phones)
# -> "ð ə  k w ɪ k  b ɹ aʊ n  f ɑː k s  ..."  (IPA-style phonemes)
```

A few practical notes that bite if you ignore them. First, the G2P system is **language-specific**: an English G2P will mangle French text, which is one concrete reason the original VALL-E was monolingual, the front-end, not just the acoustic model, was English-only. Multilingual successors like XTTS carry a multilingual phonemizer. Second, **out-of-vocabulary words and proper nouns** are where G2P fails most, a rule-based system guesses "VALL-E" or a rare surname phonetically and sometimes guesses wrong, which surfaces as a mispronunciation no amount of acoustic-model quality can fix. Third, the phoneme set is a *design choice*: some systems use IPA, some use ARPABET (the `DH AH K W IH K` style), and the choice has to match between training and inference or every prediction is garbage. The acoustic model only ever sees phoneme *ids*, so the phoneme inventory is effectively the model's input vocabulary, and it is fixed at training time. When a VALL-E-style system mispronounces a word, the bug is very often in this front-end and not in the billion-parameter transformer everyone instinctively blames.

The reason this matters for the rest of the post is that it cleanly separates *content* from *voice* in the conditioning. The phonemes carry the content ("what to say"), supplied by the front-end; the acoustic prompt carries the voice ("how it should sound"), supplied by the reference clip. The transformer's job is to fuse them: produce codec tokens that say the *phonemes' content* in the *prompt's voice*. Holding those two conditioning streams separate in your head is the cleanest way to reason about VALL-E, and it maps exactly onto the two inputs in the architecture figure from the intro.

## The inference flow in code

Now the practical part. Let me sketch the VALL-E inference flow in PyTorch so the architecture stops being a diagram and becomes operations you could implement. This is a faithful sketch of the control flow, not a drop-in library call; the open weights ecosystem for VALL-E itself is a community reimplementation, but the *shape* of the code below matches what any codec-LM TTS does.

```python
import torch
from encodec import EncodecModel  # pip install encodec

# --- 0. The frozen codec: our acoustic tokenizer / detokenizer ---
codec = EncodecModel.encodec_model_24khz()
codec.set_target_bandwidth(6.0)  # 6 kbps -> 8 RVQ codebooks at 24 kHz
codec.eval()

@torch.no_grad()
def encode_to_tokens(wav_24k):
    """wav_24k: (1, 1, samples) float tensor at 24 kHz.
    Returns codes: (n_codebooks, n_frames) long tensor."""
    encoded = codec.encode(wav_24k)               # list of (frames) per chunk
    codes = torch.cat([c[0] for c in encoded], dim=-1)  # (1, n_q, n_frames)
    return codes[0]                               # (n_q=8, n_frames)

@torch.no_grad()
def decode_from_tokens(codes):
    """codes: (n_codebooks, n_frames) long. Returns wav (1, 1, samples)."""
    frames = [(codes.unsqueeze(0), None)]
    return codec.decode(frames)
```

That is the codec round-trip, the tokenizer and detokenizer that bracket the whole system. The codec is *frozen*; VALL-E never trains it. Note the configuration: 6 kbps gives 8 codebooks, and at 24 kHz the frame rate is 75 frames per second. Those two numbers, 8 and 75, are the height and the per-second width of the grid the language models fill.

Now the AR model for the first codebook. The interface is a transformer that, given the phoneme embeddings, the prompt's first-codebook tokens, and the first-codebook tokens generated so far, predicts the next first-codebook token:

```python
import torch.nn.functional as F

@torch.no_grad()
def ar_generate_codebook1(ar_model, phonemes, prompt_codes,
                          max_frames=750, temperature=1.0, top_p=0.9,
                          eos_id=1024):
    """Autoregressively generate the first-codebook tokens for new speech.
    phonemes:     (P,) long  -- phoneme ids of the target text
    prompt_codes: (8, Tp) long -- EnCodec tokens of the 3s reference
    Returns generated codebook-1 ids: (Tg,) long."""
    device = phonemes.device
    prompt_c1 = prompt_codes[0]                     # (Tp,) first codebook of prompt
    generated = []                                  # the new first-codebook row
    for _ in range(max_frames):
        # The model attends over: phonemes, prompt c1, generated-so-far.
        logits = ar_model(phonemes, prompt_c1,
                          torch.tensor(generated, device=device).long())
        logits = logits[-1] / temperature           # next-token logits
        probs = top_p_filter(F.softmax(logits, dim=-1), top_p)
        next_id = torch.multinomial(probs, 1).item()
        if next_id == eos_id:                        # the model decides length
            break
        generated.append(next_id)
    return torch.tensor(generated, device=device).long()
```

The thing to notice is the `eos_id` check. The autoregressive model *decides the length of the output by choosing when to emit end-of-sequence*. This is the AR model's superpower (it sets timing implicitly, no separate duration predictor needed) and its Achilles' heel (if it never decides to stop, you get a run-on; if it stops early, you get a truncated utterance). Length, prosody, and pacing all live in this loop.

Next the NAR model fills codebooks 2 through 8. It is *not* a loop over frames; it is a loop over *codebook levels*, each one a single parallel forward pass:

```python
@torch.no_grad()
def nar_generate_codebooks_2_to_8(nar_model, phonemes, prompt_codes, codes_c1):
    """Non-autoregressively fill codebooks 2..8 for all frames at once.
    codes_c1: (Tg,) long -- the AR-generated first codebook.
    Returns full grid: (8, Tg) long."""
    Tg = codes_c1.shape[0]
    grid = torch.zeros(8, Tg, dtype=torch.long, device=codes_c1.device)
    grid[0] = codes_c1
    for level in range(1, 8):                        # codebooks 2..8 (index 1..7)
        # Condition on phonemes, the full 8-codebook prompt, and the sum of
        # all coarser generated levels 0..level-1. One parallel pass over Tg.
        logits = nar_model(phonemes, prompt_codes, grid[:level], level)
        grid[level] = logits.argmax(dim=-1)          # (Tg,) for this codebook
    return grid
```

And the full pipeline ties them together exactly as the figure at the top promised:

```python
@torch.no_grad()
def vall_e_tts(ar_model, nar_model, text, reference_wav_24k, g2p):
    phonemes = g2p(text)                             # text -> phoneme ids (P,)
    prompt_codes = encode_to_tokens(reference_wav_24k)   # (8, Tp)
    c1 = ar_generate_codebook1(ar_model, phonemes, prompt_codes)   # (Tg,)
    full_grid = nar_generate_codebooks_2_to_8(           # (8, Tg)
        nar_model, phonemes, prompt_codes, c1)
    wav = decode_from_tokens(full_grid)              # (1, 1, samples)
    return wav
```

Five functions, and the whole system is visible: phonemize the text, tokenize the reference, autoregress the first codebook, parallel-fill the rest, decode. The AR model is the only sequential part; everything else is a few parallel passes or a frozen codec call. If you have read this far, you now understand every line of what a codec-LM TTS system does at inference, which is more than most people who use them.

## A runnable open stand-in you can execute today

The VALL-E weights are not officially open, but you can run the two halves of its idea independently right now with off-the-shelf tools, which is the best way to build intuition. The codec round-trip is real and runnable, and a token-LM sampling loop over codec tokens is exactly the AR half of VALL-E in miniature. Here is a self-contained demonstration of both.

First, the EnCodec round-trip, which proves the codec is a clean tokenizer and detokenizer of audio. If this sounds good, the ceiling of any codec-LM TTS is set by it:

```python
import torch, torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio

codec = EncodecModel.encodec_model_24khz()
codec.set_target_bandwidth(6.0)          # 8 codebooks
codec.eval()

wav, sr = torchaudio.load("reference.wav")
wav = convert_audio(wav, sr, codec.sample_rate, codec.channels)  # to 24k mono
wav = wav.unsqueeze(0)                    # (1, 1, samples)

with torch.no_grad():
    encoded = codec.encode(wav)
    codes = torch.cat([c[0] for c in encoded], dim=-1)  # (1, 8, n_frames)
    print("token grid:", codes.shape,     # e.g. (1, 8, 225) for ~3 s
          "bitrate ~6 kbps, 75 frames/s")
    recon = codec.decode([(codes, None)])

torchaudio.save("reconstructed.wav", recon[0], codec.sample_rate)
```

Listen to `reconstructed.wav`. It will sound nearly identical to the input, which is the point: the codec preserves the voice, the room, the texture, all in those 8 discrete rows. That preserved fidelity is the *upper bound* on what any codec-LM can clone, the model can only ever produce tokens the codec can decode, so the codec's reconstruction quality is the quality ceiling.

Second, a minimal token-LM sampling loop, which is the AR half of VALL-E reduced to its skeleton. Here we use a tiny untrained model just to show the *control flow* of generating codec tokens with a transformer and decoding them; swap in trained weights and conditioning and you have VALL-E's first stage:

```python
import torch, torch.nn as nn, torch.nn.functional as F

class TinyTokenLM(nn.Module):
    """A toy AR LM over first-codebook tokens (vocab = codebook size + EOS)."""
    def __init__(self, vocab=1025, d=256, n_layer=4, n_head=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(1, 2048, d))
        layer = nn.TransformerEncoderLayer(d, n_head, batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layer)
        self.head = nn.Linear(d, vocab)
    def forward(self, ids):                      # ids: (B, T)
        T = ids.shape[1]
        x = self.emb(ids) + self.pos[:, :T]
        mask = torch.triu(torch.ones(T, T), 1).bool()   # causal
        x = self.blocks(x, mask=mask)
        return self.head(x)                      # (B, T, vocab)

@torch.no_grad()
def sample_tokens(model, prompt_ids, max_new=200, temperature=0.9,
                  top_p=0.9, eos_id=1024):
    ids = prompt_ids.clone()                     # (1, Tp) seed with prompt tokens
    for _ in range(max_new):
        logits = model(ids)[0, -1] / temperature
        probs = F.softmax(logits, dim=-1)
        # nucleus (top-p) filtering
        s, idx = torch.sort(probs, descending=True)
        keep = (torch.cumsum(s, 0) - s) < top_p
        s = s * keep; s = s / s.sum()
        nxt = idx[torch.multinomial(s, 1)]
        if nxt.item() == eos_id:
            break
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
    return ids
```

This little loop *is* VALL-E's first stage in structure: seed with prompt tokens, sample the next first-codebook token under temperature and nucleus sampling, stop on EOS. The only things separating it from the real thing are scale (4 layers versus a large transformer), conditioning (no phonemes here), and 60,000 hours of training. Wire `sample_tokens` to produce a first-codebook row, fill the rest with a NAR model, and `codec.decode` the grid, and you have rebuilt the pipeline. Running this end to end with a trained checkpoint is exactly what community VALL-E reimplementations do.

## Results: VALL-E against the prior art

Now the numbers, with the standing caveat from the kit that I will mark anything approximate as approximate and never fabricate a precise figure. VALL-E's headline results come from its paper, evaluated on LibriSpeech test-clean and the VCTK speakers, using a three-second enrolled prompt.

| System | Zero-shot clone | Prompt length | Speaker sim (SMOS) | Naturalness (MOS/CMOS) | WER | Autoregressive |
|---|---|---|---|---|---|---|
| Tacotron 2 + voc. | no | n/a (trained ID) | low for unseen | good in-domain | low | yes (mel) |
| VITS (multi-spk) | weak | n/a (trained ID) | moderate | high | low | no |
| YourTTS | yes (cross-spk) | several seconds | moderate | good | moderate | no |
| VALL-E | yes | ~3 seconds | high, approaches GT | matches/exceeds baseline | competitive | yes (codebook 1) |
| Ground truth | --- | --- | reference ceiling | reference ceiling | speaker's own | --- |

The pattern is what matters more than the exact cell values. VALL-E reported **speaker similarity (SMOS) approaching the ground-truth reference** and **naturalness (CMOS) that matched or modestly beat** a strong baseline (the paper compared against YourTTS), from only a three-second prompt, on speakers not seen in training. That combination, near-ground-truth similarity on *unseen* speakers from *three seconds*, is what made the result land. For intelligibility, VALL-E's word error rate (measured by running an ASR over the synthesized speech and comparing to the target text) was competitive with baselines on clean prompts, though, as we will see, it degraded on hard prompts in a way the regression-based systems did not.

Two honest caveats on these numbers. First, MOS and SMOS are *human* ratings with all the noise that implies; the paper used crowd raters, and a few-tenths-of-a-MOS-point gap is within the range where rater pools and instructions matter as much as the model. Second, WER from an ASR is a *proxy* for intelligibility, and the choice of ASR matters; a strong ASR can "correct" mildly degraded audio and flatter the system. When you read any zero-shot TTS result, including this one, ask which ASR computed the WER, how many raters scored the MOS, and whether the prompt set was cherry-picked. The honest way to *measure* VALL-E yourself is: fix a held-out speaker set the model never trained on, fix the prompt to exactly three seconds, run a *named* ASR (say, Whisper-large) for WER, recruit a stated number of raters (15+ per sample is a reasonable floor) for MOS and SMOS, and report variance, not just means.

There is also an *automatic*, reproducible way to measure speaker similarity that you should run alongside the human SMOS, because it removes the rater noise and lets you iterate quickly. Take a pretrained speaker-verification model (a model trained to embed a voice into a vector such that same-speaker clips land close together, the kind behind speaker-ID systems), embed the synthesized clip and the reference clip, and compute the cosine similarity of the two embeddings. A high cosine means the verification model thinks they are the same speaker; that number correlates with human SMOS and is fully reproducible:

```python
import torch, torchaudio
# a speaker-verification model that maps a waveform to an embedding vector
# (e.g. an ECAPA-TDNN from speechbrain); spk_embed(wav) -> (D,) tensor

def speaker_similarity(synth_wav, ref_wav, spk_embed):
    """Cosine similarity of speaker embeddings; higher = more similar voice."""
    e_synth = torch.nn.functional.normalize(spk_embed(synth_wav), dim=-1)
    e_ref   = torch.nn.functional.normalize(spk_embed(ref_wav),   dim=-1)
    return (e_synth * e_ref).sum().item()   # in [-1, 1], want close to 1
```

This is the metric I trust most for *fast* iteration on a cloner, because it is deterministic and does not need a crowd. The honest protocol is to report *both*: the automatic cosine similarity (reproducible, cheap, run on every checkpoint) and a human SMOS on a final candidate (the ground truth, expensive, run rarely). VALL-E's paper did exactly this kind of dual reporting, and a high automatic speaker-similarity from three seconds was the quantitative backbone behind the subjective "it really sounds like them" reaction.

#### Worked example: the three-second clone, with rough timings

Let me ground the result with an order-of-magnitude latency picture on a named device, marked approximate because exact figures depend on the implementation and the model size.

Suppose you synthesize a five-second utterance from a three-second prompt with a VALL-E-scale model on an **A100 80GB**. The grid to fill is about $5 \times 75 = 375$ frames by 8 codebooks. The AR stage does roughly 375 sequential forward passes for the first codebook; at a few milliseconds per step for a several-hundred-million-parameter transformer with KV-caching, that is on the order of **one to two seconds** of wall-clock. The NAR stage does 7 parallel passes over all 375 frames, which is **a fraction of a second** total because each pass is parallel across time. The EnCodec decode of a $375 \times 8$ grid is **tens of milliseconds**. So end-to-end you are looking at very roughly **1.5 to 3 seconds** to generate 5 seconds of audio, a real-time factor (generation time over audio duration) somewhere around **0.3 to 0.6**, i.e. faster than real time but not dramatically so, and dominated entirely by the AR top row. Treat these as ballpark; the exact RTF moves with model size, batch, sequence length, and whether you compiled the model. The structural takeaway is solid: the AR first codebook is the bottleneck, the NAR fill and the codec decode are cheap, and that is why every efficiency effort on codec-LM TTS targets the autoregressive stage.

## When it breaks: stability, hallucination, and prompt sensitivity

VALL-E is a beautiful idea with a real dark side, and being honest about the failure modes is the most useful thing this post can do, because the failures are *structural*, they follow from the autoregressive framing, not from a bug. If you have ever watched a long autoregressive generation slowly come apart, you already know the shape of these problems.

The root issue is that **VALL-E inherits every pathology of autoregressive sampling**. A language model that samples one token at a time, conditioned on its own previous outputs, can drift, loop, and run away, and there is no ground-truth teacher at inference to pull it back. In text this gives you repetition loops and topic drift; in codec-token speech it gives you three named failures.

![A matrix mapping VALL-E's hallucination, repeat-and-skip, and prompt-sensitivity failures to their root cause in autoregressive sampling and the fix adopted by successor models](/imgs/blogs/neural-codec-language-model-tts-vall-e-7.png)

The matrix above names the three failures, their causes, and their fixes. **Hallucination** is the model speaking words that were not in the input text, or mangling them, the acoustic analogue of an LLM confabulating. Its root cause is AR drift: a small error in an early token shifts the context, which makes the next token slightly more wrong, which compounds. **Repeats and skips** are the model looping a syllable or dropping a word entirely. The cause is the lack of a hard alignment constraint: classical TTS used a monotonic attention or an explicit duration predictor to *guarantee* every input phoneme is spoken once, in order; VALL-E has no such guarantee, the AR model is free to re-attend to a phoneme (repeat) or skip past one (drop). **Prompt sensitivity** is the clone degrading when the three-second prompt is noisy, very short, or atypical; because the speaker enters purely through the in-context prompt, a bad prompt directly poisons the output, there is no learned speaker prior to fall back on.

These are not edge cases you can ignore. In the original VALL-E, you would occasionally get an utterance that was perfect and occasionally one that repeated a word three times or trailed off into a different sentence, and you could not always predict which from the input. That unpredictability, *most* outputs are great but a meaningful fraction are broken, is exactly what blocks autoregressive TTS from high-stakes production use without a safety net. The standard mitigation in practice is *sampling several candidates and re-ranking*: generate, say, five outputs, run an ASR over each, and keep the one whose transcript best matches the target text. That turns a model that is great 85 percent of the time into a system that is great 99 percent of the time, at five times the compute. It is a band-aid, but a load-bearing one.

Here is the re-ranking loop, because it is genuinely the thing you ship around an autoregressive cloner:

```python
import torch
from jiwer import wer  # pip install jiwer ; word error rate
# whisper or any ASR; here sketched as an asr(wav) -> transcript callable

def best_of_n_tts(synth_fn, asr, target_text, n=5):
    """Generate n clones, transcribe each, keep the one closest to target.
    synth_fn() -> wav (one sampled generation); asr(wav) -> str transcript."""
    candidates = []
    for _ in range(n):
        wav = synth_fn()                       # one VALL-E sample (stochastic)
        hyp = asr(wav).lower().strip()
        score = wer(target_text.lower(), hyp)  # lower is better
        candidates.append((score, wav, hyp))
    candidates.sort(key=lambda c: c[0])        # ascending WER
    best_wer, best_wav, best_hyp = candidates[0]
    return best_wav, best_wer
```

This little loop is the difference between a research demo and a product. It costs you $n$ times the generation compute and one ASR pass per candidate, and it converts the model's *variance* into reliability. The deep reason it works is that the failures are *uncorrelated across samples*: a repeat or a skip happens on one stochastic draw and not another, so taking the best of five draws makes the probability that *all five* are broken vanishingly small. It only fails when the model is *systematically* wrong (a mispronounced proper noun from the G2P front-end, say), because then every candidate shares the error and there is nothing better to rank to.

#### Worked example: how much does best-of-N actually buy you

Let me put numbers on the re-ranking trade, marked approximate because the exact rates depend on the model and the text. Suppose a single VALL-E draw has a **15 percent chance of a serious error** (a repeat, skip, or hallucination bad enough to reject) on a given sentence, so an **85 percent** chance of being clean. If the errors are independent across draws, the chance that *all $n$* draws are bad is $0.15^n$. For $n=1$ that is 15 percent; for $n=3$ it is $0.15^3 \approx 0.3$ percent; for $n=5$ it is $0.15^5 \approx 0.008$ percent. So best-of-3 already drops your serious-error rate from 1-in-7 to roughly 1-in-300, and best-of-5 to roughly 1-in-12,000, *if* the ASR re-ranker reliably identifies the bad ones (it usually does for gross errors, less so for subtle prosody). The cost is linear: best-of-5 is 5 times the generation compute plus 5 ASR passes. On the A100 timing from earlier, a 5-second utterance at roughly 2 seconds per draw plus a fast ASR puts best-of-5 at very roughly **12 to 15 seconds** of wall-clock per final utterance. That is the real production cost of taming autoregressive instability, and it is why VALL-E 2's *built-in* repetition-aware sampling, which lowers the single-draw error rate directly, is such a meaningful improvement: it lets you drop $n$ and save most of that compute.

#### Stress-testing the prompt

Let me push on the prompt sensitivity specifically, because it is where the in-context-learning design has its sharpest trade. What happens as you vary the three-second prompt?

If the prompt is **clean studio audio**, cloning is excellent, exactly the demo case. If the prompt is **three seconds of noisy phone audio**, the model faithfully clones the noise: the in-context mechanism cannot separate "the voice" from "the room," so you get the target voice *in a noisy room*, because the fine codebooks copy the prompt's texture. If the prompt is **shorter than three seconds**, say one second, the model has less acoustic evidence to continue and the clone gets less stable, sometimes the voice drifts mid-utterance because the context did not pin it down firmly. If the prompt is **emotionally extreme** (someone shouting), the output inherits that energy, which is sometimes what you want and sometimes a surprise. And if the prompt is a **different language** from the target text, you hit VALL-E's monolingual limit: the original was English-only, and code-switching the speaker's accent onto another language was the explicit motivation for VALL-E X.

The general lesson is that the in-context prompt is a *double-edged sword*. It gives you faithful, training-free cloning of the *entire acoustic situation*, which is the magic, but it gives you no control to clone *only the voice* and not the room, and it makes output quality a direct function of prompt quality. A production system built on this paradigm spends real engineering effort on *prompt hygiene*: denoising the reference, trimming to clean speech, normalizing loudness, and rejecting prompts that are too short or too noisy. The model is only as good as the three seconds you feed it.

## The successors: VALL-E 2, VALL-E X, and the wave that followed

VALL-E was the spark, not the finished product, and the year after it landed saw a rapid wave of work that fixed its weaknesses and extended its reach. Understanding the lineage tells you where the paradigm went.

![A timeline of the codec-LM TTS lineage from AudioLM through VALL-E and VALL-E X to VALL-E 2 and the open zero-shot TTS frontier](/imgs/blogs/neural-codec-language-model-tts-vall-e-6.png)

The timeline above places VALL-E in its lineage. **AudioLM** (Borsos and colleagues, 2022) proved you could language-model audio over codec tokens at all, using the [semantic-then-acoustic hierarchy](/blog/machine-learning/audio-generation/semantic-vs-acoustic-tokens) we covered separately; VALL-E specialized that idea to TTS and added the in-context speaker prompt. **VALL-E X** (2023) extended it cross-lingually: prompt in one language, synthesize in another, preserving the speaker, which is exactly the code-switching the monolingual VALL-E could not do. **VALL-E 2** (2024) is the most important successor for the stability story: it introduced *repetition-aware sampling* and *grouped code modeling* that directly attack the repeat-skip-drift failures, and the paper reported reaching *human parity* on robustness and naturalness for the first time in a zero-shot codec-LM TTS, the failure modes that made VALL-E unpredictable were largely tamed by smarter sampling and a more constrained decoder rather than by abandoning the paradigm.

The deeper point is that VALL-E *seeded an entire research direction*. Once it showed that zero-shot TTS was a codec-language-modeling problem, a flood of systems adopted and varied the recipe. Some kept the AR-codec-LM core and improved sampling and conditioning, **XTTS** is essentially a productized GPT-over-codec-tokens TTS with multilingual cloning. Others kept the *goal* (zero-shot cloning) but swapped the *engine*: **F5-TTS** and Voicebox-style models replaced the autoregressive codec LM with non-autoregressive flow matching over a mel or latent, trading VALL-E's instability for the stability of a non-AR generator while keeping the in-context cloning idea. We cover that branch in depth in [zero-shot voice cloning and the TTS frontier](/blog/machine-learning/audio-generation/zero-shot-voice-cloning-and-the-tts-frontier), the direct sequel to this post. The common thread across all of them is the move VALL-E made first: *the speaker is a prompt, not a parameter.*

![A taxonomy tree of zero-shot TTS approaches showing the codec-LM, flow-matching, and end-to-end families with VALL-E founding the codec-LM branch](/imgs/blogs/neural-codec-language-model-tts-vall-e-8.png)

The tree above organizes the zero-shot TTS landscape that VALL-E opened up. The root is the shared goal: clone a voice from a short prompt. The **codec-LM** branch is VALL-E's own, autoregressive next-token prediction over codec tokens, with VALL-E itself and the GPT-plus-codec XTTS as members. The **flow-matching** branch (F5-TTS and kin) keeps in-context cloning but generates non-autoregressively, which is more stable and parallelizable; the connection to the image series' [flow matching](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow) is direct, it is the same objective applied to audio latents. The **end-to-end** branch (YourTTS, built on VITS) reached cross-speaker synthesis through a VAE-plus-flow architecture from the prior TTS generation. VALL-E did not invent the field, but it founded the branch that reframed the whole problem as language modeling, and that reframing is why "TTS is just next-token prediction now" became a sentence people say.

## When to reach for codec-LM TTS, and when not to

Time for the decisive part, because a technique you cannot place is a technique you will misuse.

**Reach for a VALL-E-style codec LM when zero-shot cloning is the requirement.** If your product is "the user uploads three seconds and we read arbitrary text in their voice," this is the paradigm, full stop. Nothing in the regression-based TTS world does training-free cloning of unseen voices; the in-context prompt is the only mechanism that delivers it. This covers personalization, dubbing-with-voice-preservation, accessibility (restoring someone's own voice from an old recording), and rapid prototyping of many voices.

**Reach for it when you want one model to cover many speakers and languages without per-voice training.** The marginal cost of a new voice is zero, no fine-tune, no enrollment pass, just a prompt, which is operationally transformative compared to a system that needs minutes of clean audio and a training run per speaker.

**Do not reach for it when you have one fixed speaker and need rock-solid reliability.** If you are building a single-voice IVR system or a fixed narrator, a well-trained single-speaker VITS or a FastSpeech-plus-HiFi-GAN pipeline will be *more stable, faster, and cheaper* than a codec LM, and you do not need the cloning at all. Autoregressive instability is a price you pay for cloning; do not pay it if you are not buying cloning. A non-AR system has no drift, no repeat-skip risk, and a predictable real-time factor.

**Do not reach for the original VALL-E specifically if stability is non-negotiable.** The first VALL-E's repeat-and-skip rate is too high for unsupervised production use without candidate re-ranking. Use a successor that fixed it (VALL-E 2-style repetition-aware sampling) or a non-AR flow-matching cloner (F5-TTS-style) if you want both cloning and stability. The paradigm is right; the original implementation's sampling is the part to upgrade.

**Do not feed it a bad prompt and expect magic.** Because quality tracks prompt quality directly, a codec-LM cloner with no prompt hygiene will ship noisy, unstable clones. If you cannot guarantee clean three-second references, budget engineering for denoising and prompt validation, or pick a system with a stronger learned speaker prior.

And the honest meta-point: a *mel-spectrogram regression pipeline still beats raw codec-LM for many single-speaker TTS jobs* on speed and reliability. The codec-LM paradigm earns its place specifically when *controllability over the speaker at inference time* is the thing you need. Match the tool to the axis of the [fidelity-controllability-speed-length](/blog/machine-learning/audio-generation/why-audio-generation-is-hard) frame you actually care about: VALL-E is a controllability play, and you should reach for it when controllability is your binding constraint.

## Case studies: real numbers from the literature

Let me anchor the discussion in named results, accurately and with approximations flagged.

**VALL-E (Wang et al., 2023), the headline clone.** Trained on roughly 60,000 hours of LibriLight English audiobook speech over 7,000+ speakers, using EnCodec at 6 kbps (8 codebooks, 75 Hz). It clones an unseen speaker from a three-second enrolled prompt with *speaker similarity approaching the ground-truth reference* and *naturalness matching or exceeding* a YourTTS baseline, while preserving the prompt's acoustic environment and emotion. The reported weakness is robustness: repeats, skips, and occasional hallucination from the AR sampler. This is the result that opened the zero-shot TTS era.

**VALL-E 2 (2024), human parity on robustness.** The same codec-LM core plus *repetition-aware sampling* (penalizing token repetition during decoding) and *grouped code modeling* (grouping codec frames to shorten the AR sequence and stabilize attention). The paper reported reaching human parity on the LibriSpeech and VCTK zero-shot benchmarks, the first codec-LM TTS to do so, by directly fixing the failure modes the original VALL-E exhibited. The architecture barely changed; the *sampling and sequence layout* did. That is a recurring lesson, autoregressive TTS is often fixed at the decoder, not the model.

**XTTS (Coqui), the productized codec LM.** A GPT-style autoregressive model over codec tokens, trained multilingually, with instant voice cloning from a few seconds of reference, shipped as an open library (`coqui-tts`). It is the most accessible way to actually run a VALL-E-shaped system today; it demonstrates that the paradigm productizes into a usable, multilingual cloning tool, and it reports strong cloning quality and intelligibility across many languages (treat specific per-language MOS/WER as version-dependent and check the model card).

**F5-TTS (2024), the non-AR alternative.** Not a codec LM at all, it is flow matching over a mel latent, but it targets the same zero-shot cloning goal VALL-E defined. It is the cleanest illustration of the "keep the goal, swap the engine" move: by replacing autoregression with a non-AR flow generator, it sidesteps the repeat-skip instability while retaining in-context cloning, and it reports competitive WER and speaker similarity with simpler, more stable inference. It is the model to compare against when you are deciding between the AR and non-AR branches of the tree above.

The throughline across all four: VALL-E proved the concept and set the benchmark, and the field then spent a year either *fixing its sampler* (VALL-E 2, XTTS) or *replacing its engine while keeping its goal* (F5-TTS). Every one of them inherited VALL-E's core reframing, the speaker is an in-context prompt.

## How this ties back to the audio stack

Step back to the series spine. The **audio stack** is waveform to codec tokens to generative model to decoder back to waveform, and VALL-E is a specific, opinionated instantiation of the middle of that stack. The tokenizer is a frozen EnCodec (the [codec post](/blog/machine-learning/audio-generation/encodec-dac-and-the-modern-codec) is the prerequisite for this one). The generative model is a *pair* of transformers exploiting the [RVQ coarse-to-fine structure](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq), one autoregressive for the coarse codebook and one non-autoregressive for the fine ones. The decoder is, again, the frozen codec. And the conditioning, the part that makes it TTS and makes it clone, is phonemes plus an in-context acoustic prompt.

On the **fidelity-controllability-speed-length** tension, VALL-E is unambiguously a *controllability* maximizer. Fidelity is whatever the codec preserves (good, capped by EnCodec). Speed is acceptable but bottlenecked by the AR first codebook (real-time factor well under one but not negligible). Length is bounded by AR stability (long utterances accumulate more drift, so very long single-pass generations are risky, the same length wall the [autoregressive audio models](/blog/machine-learning/audio-generation/autoregressive-audio-models-wavenet-to-audiolm) post discusses). But controllability over the *speaker*, the ability to specify any voice with three seconds at inference, is something no prior system offered, and that is the axis VALL-E pushed to a place the field had never been. Every later system in the lineage trades along these same four axes; VALL-E's particular trade, maximize speaker controllability, accept AR instability, is the one that founded zero-shot TTS.

When you assemble your own stack in the [capstone](/blog/machine-learning/audio-generation/building-an-audio-generation-stack), the decision of whether to put a codec LM in the generative slot comes down to exactly this question: do you need runtime control over the speaker badly enough to pay the autoregressive tax? If yes, VALL-E's descendants are your engine. If no, a non-AR regression or flow pipeline is simpler and steadier. That is the decision VALL-E forces, and forcing it clearly is part of why the paper mattered.

## Key takeaways

- **TTS became language modeling.** VALL-E predicts discrete EnCodec tokens with categorical cross-entropy instead of regressing a continuous mel, which restores the natural variability a regression loss averages away and unlocks in-context learning.
- **Zero-shot cloning is in-context continuation.** Because the model learns during training to continue the acoustic identity of its context, seeding the context with a three-second prompt of an unseen speaker makes it clone that voice with no fine-tuning, the speaker is a prompt, not a parameter.
- **The prompt transfers the whole acoustic situation.** Timbre, prosody, emotion, *and* the room and noise floor all come along, which is faithful but means you cannot easily clone the voice without the room; prompt hygiene matters.
- **AR for codebook one, NAR for the rest, follows from RVQ.** The first codebook is coarse, sequential, and length-determining, so it is autoregressive; the residual codebooks are parallel corrections, so they are non-autoregressive, cutting the sequential bottleneck by nearly an order of magnitude.
- **Scale, not architecture, was the unlock.** A standard codec language model trained on 60,000 hours and 7,000+ speakers (LibriLight) made in-context cloning *emerge*; the same data-diversity-to-generalization story as large language models.
- **It inherits every autoregressive pathology.** Hallucination, repeats, skips, and prompt sensitivity are structural costs of AR sampling, mitigated in practice by candidate re-ranking and, in VALL-E 2, by repetition-aware sampling and grouped codes.
- **It seeded the whole zero-shot TTS wave.** XTTS productized the AR codec LM, VALL-E X added cross-lingual cloning, VALL-E 2 reached robustness parity, and flow-matching systems like F5-TTS kept the goal while swapping the engine.
- **Reach for it when speaker controllability is your binding constraint**, and reach for a stable non-AR or single-speaker pipeline when it is not, the codec LM earns its instability only by buying you runtime control over the voice.

## Further reading

- Wang, Chen, et al., **"Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers" (VALL-E)**, 2023, the paper this post is about; read it for the AR-plus-NAR architecture and the three-second cloning results.
- The VALL-E 2 report, 2024, **for the repetition-aware sampling and grouped code modeling** that fixed the original's robustness and reached human parity on zero-shot benchmarks.
- Borsos, Marinier, et al., **"AudioLM: a Language Modeling Approach to Audio Generation"**, 2022, the codec-language-modeling foundation VALL-E specialized to TTS.
- Défossez, Copet, et al., **"High Fidelity Neural Audio Compression" (EnCodec)**, 2022, the codec that produces the acoustic tokens VALL-E models; the quality ceiling of the whole system.
- The F5-TTS report, 2024, **for the non-autoregressive flow-matching alternative** that pursues VALL-E's goal with a different engine; the contrast clarifies the AR-versus-non-AR trade.
- Within this series: [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard) (the foundation and the fidelity-controllability-speed-length frame), [from Tacotron to VITS](/blog/machine-learning/audio-generation/text-to-speech-from-tacotron-to-vits) (the regression-based TTS VALL-E reacted against), [semantic vs acoustic tokens](/blog/machine-learning/audio-generation/semantic-vs-acoustic-tokens) and [residual vector quantization](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq) (the token machinery), [zero-shot voice cloning and the TTS frontier](/blog/machine-learning/audio-generation/zero-shot-voice-cloning-and-the-tts-frontier) (the direct sequel), [audio deepfakes, watermarking, and voice safety](/blog/machine-learning/audio-generation/audio-deepfakes-watermarking-and-voice-safety) (the misuse and defense side of training-free cloning), and the [capstone stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack).
- Across series: [autoregressive image models](/blog/machine-learning/image-generation/autoregressive-image-models) for the in-context token-LM parallel, and [flow matching and rectified flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow) for the engine the non-AR TTS branch borrows.
