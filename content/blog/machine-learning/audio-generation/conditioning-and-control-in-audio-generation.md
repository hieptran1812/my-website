---
title: "Conditioning and Control in Audio Generation: Text, Speaker, and Melody"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "The steering wheel of every audio model: how text, a speaker's voice, a melody, and prosody are encoded and injected, and how classifier-free guidance turns a knob between adherence and diversity."
tags:
  [
    "audio-generation",
    "audio-synthesis",
    "conditioning",
    "classifier-free-guidance",
    "text-to-speech",
    "music-generation",
    "voice-cloning",
    "generative-ai",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Audio Generation"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/conditioning-and-control-in-audio-generation-1.png"
---

The first audio model I shipped could generate a beautiful eight seconds of acoustic guitar. It could not generate the *specific* eight seconds anyone wanted. A product manager would ask for "warm fingerpicked guitar, slow, in a minor key," and the model would hand back warm fingerpicked guitar that was fast, or in a major key, or that drifted into a different instrument halfway through. The generative core was fine. The codec was fine. What was missing was every wire that connects a human intention to the model's hidden state: the **conditioning**. A generative audio model with no conditioning is an engine with no steering wheel. It will happily produce plausible sound forever, in whatever direction it pleases. Conditioning is the entire apparatus by which you tell it where to go.

This post is about that apparatus, end to end. We have spent earlier posts on the *engines* of audio generation: the [autoregressive language model over codec tokens](/blog/machine-learning/audio-generation/autoregressive-audio-models-wavenet-to-audiolm), [diffusion on a mel or codec latent](/blog/machine-learning/audio-generation/diffusion-for-audio), and the [GAN vocoder](/blog/machine-learning/audio-generation/gan-vocoders-hifi-gan-and-fast-synthesis) that turns a spectrogram back into a waveform at a useful real-time factor. Every one of those engines has a slot, sometimes several, where conditioning plugs in. This post is the catalogue of what plugs in and how it is wired: **text** (a caption for music, phonemes for speech), a **speaker's identity** (a learned embedding or a three-second reference clip you want to clone), a **melody** (a pitch contour the music must follow), and the softer controls of **prosody, style, and emotion**. And once a model accepts a condition, there is a single, beautiful trick, **classifier-free guidance**, that lets you turn a knob at inference time to decide how hard the model should obey.

![A vertical stack diagram of the conditioning menu showing text, speaker, melody, and prosody signals feeding into a generative core that emits a steered waveform](/imgs/blogs/conditioning-and-control-in-audio-generation-1.png)

The figure above is the mental picture to keep for the whole post. The generative core in the middle is fixed: it is whatever engine you chose, AR LM or diffusion or flow. Stacked on top of it are the conditioning signals, each a different kind of instruction. By the end of this post you will know, for each signal, what it controls, how it is encoded into a vector or a sequence the model can read, and exactly where in the network it gets injected, whether that is cross-attention, a prefix of tokens, a FiLM modulation, or a per-frame addition to the hidden state. You will be able to read a `transformers` MusicGen call or a `diffusers` AudioLDM2 call and know precisely which argument is doing the steering and what the model does with it.

This sits on the series spine: the **audio stack** of waveform to latent to generative core to vocoder back to waveform, under the tension of **fidelity, controllability, speed, and length**. Conditioning is the *controllability* axis made concrete. And it trades against the others. More conditioning generally means more controllability and, paradoxically, sometimes *less* fidelity, because a strongly-constrained model has fewer degrees of freedom to make the output sound natural. That control-versus-fidelity tension is the running theme. If you have not read [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard), the foundation post, the framing there about audio being short in wall-clock time but enormous in token count is exactly why some conditioning signals must be *time-aligned* with the output rather than handed over once at the start.

## The shape of a conditioning signal

Before we go through the menu, it helps to set up two axes that organize the entire space, because almost every design decision falls out of them.

The first axis is **what the signal controls**: content (the words, the genre, the caption), identity (whose voice, which timbre), the musical line (the melody, the chords, the tempo), or the delivery (the prosody, the emotion, the style). These are not arbitrary categories; they correspond to genuinely different parts of the audio that a listener attends to separately. You can keep the words the same and change the voice; you can keep the melody and change the instrument; you can keep everything and change only the emotional reading. A good conditioning system gives you an independent handle on each.

The second axis, the one that drives the engineering, is **whether the signal is global or time-aligned**. A *global* signal is one vector for the whole clip: the speaker identity does not change from frame to frame, so you encode it once into a single embedding and broadcast it everywhere. A *time-aligned* signal is a sequence, one vector per output frame, that must line up with the audio in time: a melody is a sequence of pitches, and the pitch at second three must condition the audio at second three, not second one. This distinction decides the injection mechanism. Global signals get added once, often via FiLM or a prefix token or simple concatenation. Time-aligned signals get added at every position, in lockstep with the model's internal time axis.

![A taxonomy tree of conditioning signals branching into content, identity, and time-aligned musical groups, each with its own encoder leaf nodes](/imgs/blogs/conditioning-and-control-in-audio-generation-2.png)

The taxonomy above is the map for the rest of the post. At the top is the raw conditioning signal. It branches into three families: **content** (phonemes for speech, a free-text caption for music or sound effects), **identity** (a speaker embedding like an x-vector, or an acoustic prompt, the three-second clip you clone from), and **time-aligned musical** (a chromagram melody, chord symbols, a tempo). Each leaf has its own encoder: a grapheme-to-phoneme front-end, a T5 or CLAP text encoder, an ECAPA speaker encoder, a chromagram extractor. The encoder's job is always the same, to turn the human-meaningful signal into a tensor the model can attend to, but the encoders are wildly different because the signals are. We will walk the tree branch by branch, then come back to the unifying trick of guidance.

There is one more thing worth saying up front, because it dissolves a lot of confusion. Conditioning is not a property of the generative family. It is a property of how you *wire* the family. The same speaker embedding can condition an autoregressive token model, a diffusion model, and a flow model. What changes is the injection point. So instead of asking "how does VALL-E do conditioning" as if it were one thing, the sharper questions are "what signal" (an acoustic prompt) and "injected how" (as a prefix of codec tokens the AR model continues). Keep the *signal* and the *injection* separate in your head and the whole zoo becomes orderly.

## Text conditioning, part one: speech and the phoneme front-end

Text is the most common conditioning signal, but "text" means two very different things depending on whether you are doing speech or music, so we take them in turn. Speech first.

For text-to-speech, the text is not really the condition the model wants. The model wants **phonemes**, the actual speech sounds, because English spelling is a lossy and irregular encoding of pronunciation. The word "read" is two different sounds depending on tense; "lead" is a verb or a metal; "Worcestershire" is a practical joke. So the first stage of nearly every TTS system is **grapheme-to-phoneme** conversion, usually abbreviated G2P: a module that maps the written letters (graphemes) to a phoneme sequence, typically in the International Phonetic Alphabet or a system-specific phone set like ARPAbet. G2P is part dictionary lookup (a pronunciation lexicon handles the common and irregular words) and part learned model (a small sequence-to-sequence network handles out-of-vocabulary words and names). The output for "the quick brown fox" is something like `DH AH0 K W IH1 K B R AW1 N F AA1 K S`, a sequence of discrete phone symbols with stress markers.

Those phonemes are then embedded and fed to the **text encoder** of the acoustic model. In Tacotron 2 the text encoder is a stack of convolutions plus a bidirectional LSTM; in [VITS](/blog/machine-learning/audio-generation/text-to-speech-from-tacotron-to-vits) and the modern transformer TTS systems it is a transformer encoder. The encoder produces a sequence of hidden states, one per phoneme. The crucial subtlety is **alignment**: the phoneme sequence is short (a few dozen symbols) while the mel-spectrogram the model must produce is long (hundreds of frames), and the model has to learn which phonemes correspond to which audio frames, and for how long each phoneme is held. Tacotron solves this with attention; FastSpeech and the non-autoregressive systems solve it with an explicit **duration predictor** that says how many frames each phoneme should occupy, then upsamples the phoneme encodings to the frame rate. Either way, text conditioning in TTS is fundamentally about turning a short symbol sequence into a long, time-aligned acoustic plan.

Here is what the phoneme front-end and the encoder look like in practice with 🤗 `transformers`, using a VITS-family model where the whole pipeline is wrapped for you:

```python
import torch
from transformers import VitsModel, AutoTokenizer
import soundfile as sf

# A VITS model bundles G2P-adjacent text normalization, the text encoder,
# the flow-based acoustic model, and the vocoder into one forward pass.
model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

text = "the quick brown fox jumps over the lazy dog"
inputs = tokenizer(text, return_tensors="pt")  # text -> token ids

with torch.no_grad():
    # The text token ids ARE the conditioning. The model produces a waveform
    # whose content is determined entirely by these ids.
    output = model(**inputs).waveform  # shape: (1, num_samples)

sf.write("fox.wav", output.squeeze().numpy(), samplerate=model.config.sampling_rate)
print("sample rate:", model.config.sampling_rate, "Hz")
```

The thing to notice is that the text tokens are the *only* conditioning here. There is no speaker argument, no style argument, because this particular model has one fixed voice baked into its weights. To change the voice we need a second, independent conditioning signal, and that is the next branch of the tree.

A practical warning that has bitten me more than once: G2P quality is the silent ceiling on TTS quality. A state-of-the-art acoustic model fed a wrong phoneme will pronounce the wrong word with perfect fidelity and a confident voice. When a TTS system mangles a proper noun or a technical term, the bug is almost never in the neural network; it is in the lexicon or the G2P fallback. The fix is boring and effective: maintain a pronunciation lexicon for your domain's vocabulary and let it override the learned G2P.

## Text conditioning, part two: captions, T5, and CLAP

For music and sound-effect generation, text conditioning is a completely different animal. There are no phonemes; there is a free-text **caption** like "upbeat lo-fi hip hop with a mellow piano and vinyl crackle" and the model must understand the *meaning* of that English sentence and map it to an audio style. This is a semantic problem, not a phonetic one, and it is solved by a pretrained text encoder. There are two dominant choices, and the difference between them is the most important conceptual point in this section.

The first choice is a generic **T5 text encoder**. MusicGen uses exactly this: a frozen T5 encoder turns the caption into a sequence of hidden states, and the music model cross-attends to them. T5 is a strong language model, so it understands phrasing, negation, and compositional descriptions. The catch is that T5 was trained on *text only*. It has never heard a single sound. It knows that "piano" and "keyboard" are related words because they co-occur in text, but it has no grounding in what a piano actually sounds like. The music model has to learn the entire mapping from "the word piano" to "the sound of a piano" during its own training, using the paired caption-audio data. This works, and works well, but it puts the whole burden of grounding text in sound on the generative model.

The second choice is **CLAP**, and this is the idea worth slowing down for. CLAP, Contrastive Language-Audio Pretraining, is the audio analogue of CLIP from the image world. (If you have read [text encoders and prompt conditioning](/blog/machine-learning/image-generation/text-encoders-and-prompt-conditioning) from the image series, this is the same machinery transplanted to sound.) CLAP trains *two* encoders simultaneously, a text encoder and an audio encoder, on a large dataset of audio clips paired with their text descriptions. The training objective is **contrastive**: for a batch of paired (text, audio) examples, the model is rewarded for making the embedding of each text close to the embedding of its matching audio, and far from the embeddings of all the *other* audio clips in the batch. The same loss, symmetrically, pulls each audio toward its matching text. The math is the standard symmetric cross-entropy over a similarity matrix, identical in form to CLIP.

![A dataflow graph showing a caption and an audio clip passing through a text encoder and an audio encoder into a shared contrastive embedding space where the matching pair is pulled close](/imgs/blogs/conditioning-and-control-in-audio-generation-6.png)

The figure shows what this buys you. After contrastive training, the text encoder and the audio encoder map into a **shared embedding space** where, crucially, the embedding of the caption "lo-fi piano" lands near the embedding of an actual lo-fi piano recording. The text encoder is now *grounded in sound*. When you take the CLAP text embedding of a new caption and use it to condition a generator, you are handing the generator a vector that already lives in audio-aware space, a vector that "knows" what the words sound like, not just what they mean linguistically. This is why CLAP enables text-to-audio so cleanly: the hard part, bridging the gap between language and sound, was solved once during CLAP's contrastive pretraining, and every downstream generator gets to reuse it.

Let me make the contrastive objective precise, because the "why" is the whole point. Suppose a batch has $N$ pairs. The text encoder produces normalized embeddings $t_1, \dots, t_N$ and the audio encoder produces normalized embeddings $a_1, \dots, a_N$. Form the $N \times N$ similarity matrix $S_{ij} = t_i \cdot a_j / \tau$, where $\tau$ is a learned temperature. The diagonal entries $S_{ii}$ are the matching pairs; everything off-diagonal is a mismatch. The loss is

$$
\mathcal{L} = \frac{1}{2}\Big[ \text{CE}(S, \text{rows}) + \text{CE}(S^\top, \text{rows}) \Big],
$$

where each cross-entropy treats row $i$ as a classification problem whose correct answer is column $i$. Minimizing this pushes $S_{ii}$ up and the rest of row and column $i$ down. The geometric consequence is exactly the shared space: matching text and audio become each other's nearest neighbors. The temperature $\tau$ controls how sharply; a small $\tau$ makes the model very confident and the space very tightly clustered, which helps retrieval but can hurt the smoothness you want for *conditioning*, where you would like nearby captions to give nearby audio. That tension is one reason text-to-audio systems sometimes pick T5 over CLAP, or use both.

A clean way to summarize the trade: **T5 gives you compositional language understanding but no audio grounding; CLAP gives you audio grounding but, being a contrastive model, a coarser grip on complex compositional phrasing.** AudioLDM and AudioLDM2 lean on CLAP (AudioLDM2 actually builds a richer "language of audio" representation on top); MusicGen leans on T5. Stable Audio uses a CLAP-style text encoder trained on its own data. None of these is wrong; they are different points on the grounding-versus-compositionality trade.

## How text gets injected: cross-attention, prefix, and FiLM

We have the text encoded, by phonemes, by T5, or by CLAP, into either a single vector or a sequence of vectors. Now: where does it enter the model? There are three dominant injection mechanisms, and which one you use depends mostly on whether your generative core is a diffusion model or an autoregressive language model.

**Cross-attention** is the default for diffusion models, inherited directly from text-to-image diffusion. The text encoder produces a sequence of hidden states $C = [c_1, \dots, c_L]$. Inside the denoising network, at each layer, the audio latent's hidden states form the queries and the text hidden states form the keys and values:

$$
\text{Attn}(Q_\text{audio}, K_\text{text}, V_\text{text}) = \text{softmax}\!\left(\frac{Q_\text{audio} K_\text{text}^\top}{\sqrt{d}}\right) V_\text{text}.
$$

This lets every position in the audio latent attend to every word in the caption, and decide for itself how much each word matters at that point in the audio. It is flexible and expressive, and it handles the variable length of captions naturally. AudioLDM2 and Stable Audio condition on text exactly this way. The cost is that cross-attention adds parameters and compute at every layer.

**Prefix or prompt tokens** is the default for autoregressive language models. An AR audio model generates codec tokens one at a time; to condition it on text, you simply *prepend* the text to the sequence. In MusicGen the T5 caption hidden states are projected to the model's dimension and placed before the audio tokens, so when the model generates the first audio token it is already attending back over the whole caption through ordinary self-attention. There is no separate cross-attention machinery; the condition is just more context in the same stream. This is elegant and it is exactly the in-context-conditioning pattern that also underlies voice cloning, which we will see shortly. VALL-E's phoneme conditioning works the same way: the phonemes are a prefix, the codec tokens are the continuation.

**FiLM**, feature-wise linear modulation, is the default for *global* conditioning, the single-vector kind. A global condition vector $g$ (a speaker embedding, a style vector, a diffusion timestep) is passed through a small linear layer to produce a per-channel scale $\gamma$ and shift $\beta$, which then modulate a hidden activation $h$:

$$
\text{FiLM}(h \mid g) = \gamma(g) \odot h + \beta(g).
$$

This is cheap, it injects the condition everywhere at once, and it is the natural fit for "one vector for the whole clip" signals. You will see FiLM used for speaker conditioning, for global style tokens, and universally for the diffusion timestep. The reason FiLM suits global conditioning and cross-attention suits sequential conditioning is exactly the global-versus-time-aligned axis from earlier: FiLM broadcasts one instruction to all positions, while cross-attention lets different positions pull different things from a *sequence* of instructions.

![A dataflow graph showing text, a speaker clip, and a melody reference each passing through a dedicated encoder and entering a shared injection block before the generative core](/imgs/blogs/conditioning-and-control-in-audio-generation-3.png)

The figure consolidates the wiring. Three different raw signals, three different encoders, but they converge on an **injection block** that the generative core reads. The labels on the arrows are the mechanisms: text by cross-attention (a sequence the model attends to), speaker globally (one vector broadcast everywhere), melody time-aligned (a per-frame sequence added in lockstep). The generative core does not need to know where the conditioning came from; it just sees conditioned hidden states. This separation is what lets the same engine accept many signals: each signal brings its own encoder and its own injection wire, and the core stays the same.

There is a fourth injection mechanism worth naming, because it shows up the moment you combine several conditions, and it is the simplest of all: **concatenation**. When you have a global condition vector and you want it visible to the model, you can just concatenate it onto the input features at every position, or prepend it as an extra "token," and let the model's own layers figure out how to use it. Concatenation has no parameters of its own and imposes no structure; it is the lazy default that often works surprisingly well for a single global signal. Its weakness shows when you have *many* conditions: concatenating a speaker vector, a style vector, and a language ID all onto the input gives the model a soup it must learn to disentangle, whereas FiLM applies each as a clean multiplicative-plus-additive modulation, and cross-attention keeps each as a separately-addressable memory. A useful rule of thumb: concatenation for one global signal, FiLM for a few global signals you want cleanly separated, cross-attention for any signal that is itself a sequence, and per-frame addition for any signal that is time-aligned. Real systems mix all four, one per signal, which is exactly why the injection block in the figure is drawn as a single box absorbing several differently-wired arrows.

It is worth dwelling for a moment on *why* the injection mechanism is not just an implementation detail but a modeling decision with consequences. Cross-attention gives the model the most freedom: every output position can decide, independently, how much of each conditioning element it wants, which is why it handles long, structured captions so well, but that freedom is also capacity the model must spend learning what to attend to, and on small datasets it can attend to the wrong things. FiLM gives the model the least freedom: one global modulation, applied identically everywhere, which is exactly right when the condition genuinely is global and exactly wrong when it is not (FiLM cannot express "this word matters here but not there"). The per-frame addition sits in between for time-aligned signals: it provides the right information at the right time but does not let the model *choose* how much to use unless you add a learned gate. Matching the mechanism to the signal's true structure, sequential, global, or time-aligned, is what makes a conditioning system learn fast and behave predictably; mismatching it, for instance forcing a time-aligned melody through a single global FiLM vector, throws away the very structure that makes the signal useful.

#### Worked example: which injection costs what

Take a 300M-parameter MusicGen-small generating 10 seconds of 32 kHz audio at 50 codec frames per second, so 500 frame positions times 4 codebooks. The T5 caption is at most 64 tokens. With **prefix injection**, the 64 caption tokens simply extend the self-attention context from 500 to 564 positions, a roughly 13% increase in attention cost and zero new parameters, because the model reuses its existing self-attention. If instead you bolted on **cross-attention** to the same 64 tokens, you would add a cross-attention sublayer to every transformer block, on the order of 15 to 20% more parameters and a second attention operation per layer. For an AR model the prefix approach is strictly cheaper, which is one reason MusicGen chose it. For a diffusion model, where the denoiser is run dozens of times per sample, the calculus differs and cross-attention's flexibility usually wins. The lesson is that the *cheapest correct* injection depends on how many times you run the network: once-per-token (AR) favors prefix, dozens-of-times-per-sample (diffusion) favors cross-attention.

## Speaker and voice conditioning: embeddings versus prompts

Now the identity branch. You want the model to speak in a *particular* voice. There are two fundamentally different ways to specify a voice, and the field has largely shifted from the first to the second over the last few years.

The classic approach is a **speaker embedding**: a single fixed-length vector that summarizes a voice. You obtain it from a **speaker verification** model, a network trained on a separate task, namely deciding whether two utterances come from the same person. Such a model is trained with thousands of speakers and a metric-learning loss (a softmax over speaker identities, or a contrastive/angular-margin loss), and its penultimate layer becomes a compact, discriminative voice fingerprint. The lineage of these is worth knowing because the names show up everywhere: the **d-vector** (a deep speaker embedding from an LSTM verification net), the **x-vector** (a time-delay neural network with statistics pooling, long the standard), and the current state of the art, **ECAPA-TDNN**, which adds channel attention and multi-scale features. The output, by convention, is a 192- or 256-dimensional vector. That vector is the global conditioning signal: you compute it once from a reference clip, and inject it via FiLM or concatenation into the acoustic model.

```python
import torch, torchaudio
from speechbrain.inference import EncoderClassifier

# ECAPA-TDNN speaker encoder: a reference clip -> a 192-d voice embedding.
spk_encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
)

ref, sr = torchaudio.load("reference_voice.wav")   # any clean clip of the target speaker
if sr != 16000:
    ref = torchaudio.transforms.Resample(sr, 16000)(ref)

with torch.no_grad():
    spk_emb = spk_encoder.encode_batch(ref)         # shape: (1, 1, 192)
spk_emb = torch.nn.functional.normalize(spk_emb.squeeze(0), dim=-1)
print("speaker embedding:", spk_emb.shape)          # (1, 192)

# This single vector now conditions the TTS acoustic model for the whole utterance,
# typically broadcast across every output frame via FiLM or concatenation.
```

The strength of a speaker embedding is that it is compact, stable, and disentangled: it carries voice identity and very little else, so you can swap the voice without touching the content. The weakness is that a single 192-dimensional vector is a *bottleneck*. It captures the broad timbre of a voice but throws away the fine, idiosyncratic detail, the exact breathiness, the specific way someone's "s" sounds, the micro-prosody. Clones built purely from a speaker embedding sound recognizably like the target but rarely *uncannily* like them.

The modern approach throws out the bottleneck entirely. Instead of compressing the voice into a vector, you give the model the **raw reference audio itself** as an **acoustic prompt** and let it figure out the voice in context. This is the celebrated **VALL-E** trick: a three-second clip of the target speaker is encoded to codec tokens, and those tokens are placed as a *prefix* in front of the tokens the model is about to generate. The model is an autoregressive codec language model, so continuing from the prefix means continuing *in the same voice*, the same way a text LM completing a paragraph naturally continues in the same style. There is no speaker embedding, no verification network, no bottleneck. The entire richness of the three-second clip, timbre, prosody, recording environment, accent, is available to the model as in-context examples.

This is the same in-context-learning pattern that makes large language models do few-shot tasks from a prompt, transplanted to audio. It is why VALL-E and its descendants (XTTS, F5-TTS, and the rest of the [zero-shot voice cloning frontier](/blog/machine-learning/audio-generation/zero-shot-voice-cloning-and-the-tts-frontier)) can clone a voice from seconds of audio with no per-speaker fine-tuning. It also inherits the failure modes of in-context learning: if your three-second prompt is noisy, the clone inherits the noise; if the prompt is emotionally flat, the clone is flat; if the prompt clips, the clone has a tendency to clip. The prompt is not just identity, it is *everything about how that clip sounds*, and the model copies all of it.

Here is the conceptual sketch of injecting an acoustic prompt into an AR codec model, in the spirit of VALL-E:

```python
import torch
from transformers import EncodecModel, AutoProcessor

# Encode the speaker's reference clip into codec tokens (the acoustic prompt).
codec = EncodecModel.from_pretrained("facebook/encodec_24khz")
proc = AutoProcessor.from_pretrained("facebook/encodec_24khz")

ref_wav = torch.randn(1, 24000 * 3)                      # 3 s placeholder reference
enc = proc(raw_audio=ref_wav.squeeze().numpy(),
           sampling_rate=24000, return_tensors="pt")
with torch.no_grad():
    prompt_codes = codec.encode(enc["input_values"],
                                enc["padding_mask"]).audio_codes  # (1, 1, K, T)

prompt_codes = prompt_codes.squeeze(0).squeeze(0)        # (K codebooks, T frames)

# An AR codec LM then continues from BOTH the phoneme prefix AND the acoustic prompt.
# Sketch of the conditioning concatenation (model internals abstracted):
#   sequence = [ phoneme_tokens ] + [ flatten(prompt_codes) ] + [ <generate from here> ]
# Because the model just predicts the next codec token, "continuing" the acoustic
# prompt means continuing IN THE SAME VOICE, with no speaker embedding anywhere.
phoneme_tokens = torch.randint(0, 100, (1, 40))          # placeholder phoneme ids
print("prompt frames:", prompt_codes.shape[-1],
      "| phoneme tokens:", phoneme_tokens.shape[-1])
```

The trade between the two approaches is clean and worth stating as a rule. A **speaker embedding** is global, disentangled, compact, and lets you *interpolate* or *store* voices cheaply (a voice is just a vector you can average or look up), but it caps clone fidelity at the bottleneck. An **acoustic prompt** is high-fidelity and needs zero per-speaker setup, but it copies *everything* in the prompt, costs prompt-length context every generation, and gives you a less clean handle on identity in isolation. Production systems often use both: an embedding for a stable identity handle plus a short prompt for fine detail.

### A problem-solving narrative: the noisy three-second prompt

Let me walk a real engineering decision, because it shows how the choice between conditioning mechanisms is made under pressure. You are building a voice-cloning feature. Users will upload short clips of themselves, often recorded on a phone, often in a noisy room, and you must clone their voice well enough that they recognize themselves. You reach for the obvious frontier choice, an acoustic-prompt model in the VALL-E lineage, because it gives the best clone fidelity from seconds of audio. You ship it to a test group. The clean-room uploads sound great. The clips recorded in a cafe sound like a cafe: the clone inherits the background hum, the clipped consonants, the room reverb, because, as we established, *the prompt is everything about how that clip sounds*, and the model faithfully reproduces the noise along with the voice. Worse, the AR model's stability degrades on the noisy prompt, you see occasional repeated syllables, the WER climbs, because the noisy prefix is out of the model's comfortable distribution.

Now you reason through the options. Option one: denoise the prompt before encoding it. This helps, a speech-enhancement front-end cleans the obvious hum, but aggressive denoising introduces its own artifacts that the clone then inherits, so it is a partial fix that trades one problem for another. Option two: fall back to a speaker *embedding* for noisy uploads. An ECAPA embedding is far more robust to noise than a raw acoustic prompt, because the verification model was *trained* on noisy, in-the-wild speech (VoxCeleb is YouTube audio) and its embedding deliberately discards everything except identity, including the noise. The clone fidelity is lower (the bottleneck), but it is *stable* and *noise-free*, which on a bad upload beats a high-fidelity clone of the noise. Option three, the one production systems actually pick: a **quality gate**. Score the upload's signal-to-noise ratio, and route clean uploads to the high-fidelity acoustic-prompt path and noisy uploads to the robust speaker-embedding path, or ask the user to re-record. The decision is not "which mechanism is best" in the abstract; it is "which mechanism degrades gracefully on the input I will actually get," and the answer is *different mechanisms for different inputs*, gated by a measurable property of the input. This is the kind of trade the global-versus-prompt distinction lets you reason about cleanly: a prompt is a high-variance, high-ceiling channel, an embedding is a low-variance, capped channel, and a robust product uses the channel that suits the input quality.

## Melody conditioning: the time-aligned signal

Now the most interesting branch, because it is genuinely time-aligned and forces a different injection mechanism. You want to generate music that follows a *specific melody*, not just a specific genre. The text caption sets the style ("epic orchestral"); the melody sets the actual tune. These are independent controls, and being able to combine them, "render *this* melody in *that* style," is one of the most useful things a music model can do.

The signal MusicGen uses for melody is a **chromagram**. A chromagram is a representation of audio that, for each short time frame, gives the energy in each of the twelve pitch classes (C, C-sharp, D, and so on up the octave), collapsing all octaves together. So a chromagram is a sequence of 12-dimensional vectors, one per frame, that captures *which notes are sounding* over time while discarding timbre and octave. This is exactly the right representation for a melody control: it says "the tune goes C, then E, then G" without dictating *which instrument* plays those notes or in which octave, leaving the style to the text caption. You compute it with `librosa` in two lines:

```python
import librosa
import numpy as np

# Extract a chromagram melody from a reference tune.
y, sr = librosa.load("reference_melody.wav", sr=32000)
chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=640)  # (12, frames)

# MusicGen's melody conditioning keeps, per frame, only the DOMINANT pitch class,
# so the model follows the lead line, not the full harmony.
dominant = np.argmax(chroma, axis=0)                 # (frames,) one pitch class per frame
print("chromagram:", chroma.shape, "| dominant pitch per frame:", dominant.shape)
```

The chromagram is **time-aligned**: frame $t$ of the chromagram conditions frame $t$ of the generated audio. This is the defining feature and the reason melody conditioning cannot use the prefix trick that text and speaker prompts use. A prefix is read once, at the start; a melody must be read *continuously*, in lockstep with generation. So MusicGen injects the melody differently: the chromagram conditioning is projected and *added into the model's input at each frame position*, alongside the audio token embeddings, so that when the model decides frame $t$'s codec token it can see frame $t$'s target pitch. The text caption still rides as a prefix (global), but the melody rides per-frame (time-aligned). One model, two injection mechanisms, because it carries two kinds of signal.

Here is melody-conditioned generation end to end with 🤗 `transformers`, the call that combines a text caption *and* a melody audio:

```python
import torch, torchaudio
from transformers import MusicgenMelodyForConditionalGeneration, AutoProcessor

model = MusicgenMelodyForConditionalGeneration.from_pretrained(
    "facebook/musicgen-melody"
)
processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")

# The reference melody whose tune we want to follow.
melody, sr = torchaudio.load("reference_melody.wav")
melody = torchaudio.transforms.Resample(sr, model.config.audio_encoder.sampling_rate)(melody)

inputs = processor(
    audio=melody.numpy(),                 # the MELODY conditioning (time-aligned)
    sampling_rate=model.config.audio_encoder.sampling_rate,
    text=["90s rock song with electric guitar and drums"],  # the STYLE (global)
    padding=True,
    return_tensors="pt",
)

with torch.no_grad():
    audio = model.generate(**inputs, do_sample=True,
                           guidance_scale=3.0,    # classifier-free guidance, see below
                           max_new_tokens=500)    # ~10 s at 50 Hz frame rate

sr_out = model.config.audio_encoder.sampling_rate
torchaudio.save("rock_following_melody.wav", audio[0].cpu(), sr_out)
print("generated", audio.shape[-1] / sr_out, "s following the reference tune")
```

The `audio=` argument is the melody; the `text=` argument is the style; they are independent, and that is the whole point. You can hum a tune, pass it as the melody, and ask for it "as a jazz piano ballad" or "as an aggressive metal riff," and the *notes* stay while the *style* changes.

![A before and after comparison contrasting text-only MusicGen whose tune wanders against text-plus-melody conditioning that follows the given chromagram while keeping the genre from text](/imgs/blogs/conditioning-and-control-in-audio-generation-5.png)

The figure makes the effect concrete. On the left, text-only conditioning: the prompt fixes the genre, but the model is free to invent *any* melody consistent with that genre, and across two generations from the same prompt you get two completely different tunes. On the right, text-plus-melody: the chromagram pins the pitch contour, so the generated audio follows your specific tune, while the genre still comes from the text. The control you have gained is precisely the *melodic line*, and the freedom you have given up is the model's liberty to choose its own notes. That is the control-versus-fidelity trade in miniature: a melody-conditioned generation sometimes sounds slightly less natural than an unconstrained one, because the model is being forced through a pitch contour it might not have chosen, and reconciling "follow this tune" with "sound like great rock" is genuinely harder than either alone.

#### Worked example: when the melody and the clip lengths disagree

A trap that bites everyone the first time. You extract a chromagram from a 6-second reference melody at MusicGen's frame rate, so roughly 300 melody frames, and then ask the model to generate 10 seconds, 500 frames. Now you have 300 conditioning frames and 500 output positions, and the model has nothing to condition the last 4 seconds on. Different systems handle this differently, and knowing which yours does saves an afternoon. MusicGen-melody, in the `transformers` implementation, conditions only as far as the melody extends and then continues *unconditioned* for the remainder, so your generated audio follows the tune for 6 seconds and then *wanders* for the last 4, which sounds like a perfectly good song that suddenly loses the plot. The fix is to make the conditioning span match the request: either generate only as long as your melody (set `max_new_tokens` to about 300), or loop and repeat the chromagram to fill the full length if a repeating motif is what you want, or pad the melody with a held final note. The general principle for *any* time-aligned condition is that its length is a hard contract with the output length, and a silent mismatch produces audio that is correct where the condition exists and free everywhere else. Always assert that your conditioning-frame count matches your requested output-frame count before you hit generate.

A deeper point hides in that worked example, and it is the science of why time-aligned conditioning is fundamentally harder than global. A global condition has no alignment problem: there is exactly one vector and exactly one clip, and "where does it go" is answered by "everywhere." A time-aligned condition introduces an *alignment* between two sequences of possibly-different length and rate, and getting that alignment right is its own modeling problem. For melody the alignment is mercifully simple, because the chromagram is resampled to the model's frame rate and the correspondence is one-to-one. For phoneme conditioning in TTS the alignment is *hard*, because a short phoneme sequence must be stretched to a long frame sequence and the model has to decide how long to hold each phoneme. This is exactly why TTS architectures split into two camps: the **attention-based** systems (Tacotron 2) that learn the phoneme-to-frame alignment implicitly through an attention matrix, with the well-known risk that attention can fail catastrophically (skipping words, repeating words, babbling on a hard input), and the **duration-predictor** systems (FastSpeech, VITS) that predict an explicit duration per phoneme and upsample deterministically, which is more robust but needs ground-truth durations to train on. The alignment problem is the tax you pay for time-aligned conditioning, and the whole history of TTS architecture is, in one reading, the search for a robust way to pay it.

Tempo, rhythm, and timing are time-aligned signals of the same kind. Stable Audio's signature trick is **timing conditioning**: it conditions on the start time and total duration so it can produce variable-length audio that begins and ends cleanly, encoded as time embeddings the diffusion model reads. Chord-symbol conditioning (a sequence of chord labels, one per beat) is another per-frame signal injected the same way as melody. The common thread is that anything that varies *over* the clip must be supplied *along* the clip.

## Prosody, style, and emotion: the soft controls

The last branch is the subtlest. You have the words (phonemes), the voice (speaker), and for music the tune (melody). What is left is *how the line is delivered*: the prosody (pitch contour, rhythm, stress, intonation), the style (read versus conversational versus newscaster), and the emotion (happy, sad, urgent). These are real, perceptually huge, and notoriously hard to control, because there is no clean label for "deliver this sentence with gentle warmth and a slight rising question intonation at the end." We will go deep on this in the dedicated [prosody, emotion, and expressive speech](/blog/machine-learning/audio-generation/prosody-emotion-and-expressive-speech) post; here is the conditioning-mechanism preview.

The foundational idea is the **reference encoder** and its refinement, **global style tokens** (GST). A reference encoder is a small network that takes a *reference audio clip* carrying the desired style and compresses it into a **prosody embedding**, a vector that captures the delivery while being (ideally) blind to the words and the speaker. You then condition the TTS model on this prosody vector, globally, the same way you condition on a speaker embedding. The promise is "say *my* text with the *delivery* of *this* reference," transferring emotion from a reference utterance to new content. Global style tokens add a clever twist: instead of using the raw reference embedding, the model learns a small bank of **style tokens** (say, ten learned vectors) and represents any reference as an attention-weighted combination of them. This makes the style space discrete and interpretable, so that token 3 might come to mean "high arousal" and token 7 "low pitch," and you can drive style *without* a reference by setting the token weights directly.

The mechanism, then, is: encode a reference (or pick token weights) into a global style vector, inject it via FiLM or concatenation, exactly the global-conditioning machinery we already built for speaker. The reason prosody is hard is not the injection; it is the **disentanglement**. A reference clip carries words, speaker identity, *and* prosody all at once, and a naive reference encoder leaks the words and the speaker into the "style" vector, so transferring style accidentally transfers content or voice. The research on expressive TTS is largely a war on this leakage, with information bottlenecks, adversarial speaker classifiers, and careful architectural constraints. The modern frontier adds a fourth route entirely: **instruction-driven** style, where a free-text description like "speak slowly and sadly, almost whispering" is encoded by a text encoder and used as the style condition, reusing the very text-conditioning machinery from the start of this post.

There is also a per-frame, time-aligned face of prosody for fine control. Systems that expose explicit pitch and energy contours (FastSpeech 2 predicts per-frame pitch and energy, and lets you override them) treat prosody like melody: a sequence you supply along the clip. This gives surgical control, "raise the pitch on *this* word," at the cost of having to specify a contour rather than a single mood vector. Coarse mood is global; fine contour is time-aligned. Same axis, again.

![A before and after comparison contrasting global conditioning as one vector broadcast to every frame against time-aligned conditioning as a per-frame sequence that must line up with the output](/imgs/blogs/conditioning-and-control-in-audio-generation-7.png)

The figure crystallizes the organizing axis of the whole post. On the left, global conditioning: speaker and coarse style are one vector for the entire clip, broadcast to every frame, injected by FiLM or a prefix or simple concatenation. On the right, time-aligned conditioning: melody and fine prosody are one vector *per frame*, a sequence that must be in exact temporal lockstep with the output, injected by adding to the hidden state at each position. Almost every conditioning decision in audio reduces to which side of this figure your signal lives on. Global is easy to inject and easy to specify; time-aligned is more powerful and more demanding, because you have to produce, and align, a whole sequence.

## Classifier-free guidance for audio

We have wired in the conditions. Now the inference-time knob that decides how *hard* the model obeys them: **classifier-free guidance**, or CFG. This is the same technique that powers text-to-image diffusion, and rather than re-derive it I will state it, point you to the [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance) post in the image series for the full derivation, and then focus on what is *different* about its effect on audio.

The setup is this. During training, you randomly **drop the condition** some fraction of the time (say 10%), replacing it with a null condition, so the model learns to generate both *conditionally* and *unconditionally* with the same weights. At inference, you run the model twice, once with the condition and once with the null, and **extrapolate** away from the unconditional prediction in the direction of the conditional one:

$$
\hat{\epsilon} = \epsilon_\text{uncond} + w \,\big(\epsilon_\text{cond} - \epsilon_\text{uncond}\big).
$$

Here $\epsilon_\text{cond}$ is the model's noise prediction given the condition, $\epsilon_\text{uncond}$ given the null, and $w$ is the **guidance scale**. At $w = 1$ you recover the plain conditional model. As $w$ grows, you push the sample further in the direction the condition wants, amplifying the condition's influence beyond what the conditional model alone would produce. The exact same formula applies whether the condition is a text caption, a CLAP embedding, or a melody, and whether the model is a diffusion model (where $\hat\epsilon$ is a noise prediction) or, with the analogous logit-space version, an autoregressive model (MusicGen applies CFG by interpolating *logits* between the conditioned and unconditioned forward passes).

What is *different* for audio is the character of the trade and where it breaks. In images, high CFG over-saturates colors and produces a glossy, slightly unreal look. In audio, the failure modes are their own:

- **Low $w$ (around 1):** the sample is diverse and natural, but only loosely follows the prompt. Ask for "sad piano" and you might get neutral piano. For voice cloning, the voice drifts off the target.
- **Moderate $w$ (around 3 for MusicGen, 3 to 7 for AudioLDM):** the sweet spot. Strong prompt adherence with the audio still sounding natural. This is where you want to live.
- **High $w$ (above 7 to 10):** the sample rigidly matches the prompt but the audio quality *degrades*. You start to hear harshness, a metallic or over-sharpened timbre, clipping-like distortion, and a collapse in diversity, every sample from the prompt starts to sound the same. The model is being pushed so far past its learned distribution that the extrapolation lands in unnatural territory.

![A matrix showing guidance scale values against prompt match, diversity, and artifacts, with moderate guidance as the sweet spot and high guidance causing harsh distortion and collapsed diversity](/imgs/blogs/conditioning-and-control-in-audio-generation-8.png)

The matrix above is the practical guide. Read down the guidance-scale column and you see the arc: at $w=1$ prompt match is weak but diversity and quality are high; at $w=3$ everything is good, the genuine sweet spot; at $w=7$ prompt match is strong but diversity drops and artifacts creep in; at $w=15$ the sample is rigid, diversity has collapsed, and the audio is harsh and clipped. The actionable rule that falls out: **start at the model's recommended default (3 for MusicGen, around 3.5 for Stable Audio, 3 to 7 for AudioLDM2), and only raise it if the model is ignoring your prompt, watching for the artifacts that mark the ceiling.** Raising guidance is not free control; it is borrowed control you pay back in fidelity and diversity.

Here is CFG in a `diffusers` AudioLDM2 call, where the knob is a single argument:

```python
import torch, soundfile as sf
from diffusers import AudioLDM2Pipeline

pipe = AudioLDM2Pipeline.from_pretrained(
    "cvssp/audioldm2", torch_dtype=torch.float16
).to("cuda")

prompt = "a cheerful acoustic guitar melody with light percussion"
negative_prompt = "low quality, muffled, distorted"   # the "null" can be a NEGATIVE prompt

audio = pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=200,
    audio_length_in_s=10.0,
    guidance_scale=3.5,        # THE control-vs-fidelity knob. 1 = no guidance.
    generator=torch.Generator("cuda").manual_seed(0),
).audios[0]

sf.write("guitar.wav", audio, samplerate=16000)
print("generated 10 s at guidance_scale=3.5")
```

Two practical notes that the argument list hides. First, **negative prompting** is CFG with a twist: instead of a truly null condition, you replace $\epsilon_\text{uncond}$ with the model's prediction for a *negative* prompt ("low quality, distorted"), so the extrapolation pushes *away* from those qualities as well as *toward* the positive prompt. It is the same formula with a more useful baseline. Second, CFG **doubles inference cost**, because you run the network twice per step (once conditioned, once unconditioned). For a diffusion model at 200 steps that is 400 forward passes. Some systems amortize this, but the naive cost is 2x, which matters when you are optimizing the real-time factor.

#### Worked example: tuning guidance for a voice clone

Say you are cloning a voice with a CFG-capable TTS model and the clone sounds *close* but keeps slipping toward a generic voice. You raise the guidance scale from 3 to 5 and the clone snaps tighter to the target, the listener-rated similarity (a CMOS-style comparison) improves by perhaps half a point on a 5-point scale in my experience with these systems, treat that as illustrative, not a benchmarked constant. Encouraged, you push to 9. Now similarity barely improves over 5, but a fresh problem appears: the speech develops a faint metallic buzz on sustained vowels and the intelligibility drops, your ASR-measured word error rate creeps from 3% to 6% as the over-guided audio confuses the recognizer. You have walked off the cliff. The right operating point was 5: enough guidance to lock identity, not so much that fidelity and intelligibility pay for it. The general procedure is to **sweep guidance on a held-out prompt, measure both an adherence metric (similarity, or CLAP-score for text-to-audio) and a fidelity metric (FAD, or WER for speech), and pick the knee where adherence has saturated but fidelity has not yet fallen.**

## The full conditioning menu, side by side

Let me consolidate everything into the comparison the whole post has been building toward. Six conditioning signals, what each controls, how each is injected, and who uses it.

![A matrix comparing six conditioning signals across what they control, how they are injected, and which models use them](/imgs/blogs/conditioning-and-control-in-audio-generation-4.png)

The matrix is the post in one table, and it is worth reading row by row because each row is a different lesson. **Text caption** controls content and genre, injected by cross-attention (diffusion) or prefix (AR), used by MusicGen and AudioLDM2. **Phonemes** control the exact words, injected via the text encoder and attention/duration, used by Tacotron and VITS. **Speaker embedding** controls voice identity, injected globally via FiLM, used by YourTTS and XTTS. **Acoustic prompt** controls voice via in-context cloning, injected as prefix tokens, used by VALL-E and F5-TTS. **Melody** controls the musical line, injected time-aligned per frame, used by MusicGen-melody. **Prosody** controls delivery and emotion, injected via global style tokens or a reference encoder, used by expressive TTS like Tacotron-GST. Read the "how injected" column top to bottom and you see the two-mechanism story: global signals (speaker, coarse prosody) take FiLM or a prefix; time-aligned signals (melody, fine prosody) take a per-frame addition; sequential content (text) takes cross-attention or a prefix.

Here is the same information as a written table, with a bit more detail on the encoder, so you have it in copy-pasteable form:

| Signal | Controls | Encoder | Injection | Global/aligned | Used by |
| --- | --- | --- | --- | --- | --- |
| Text caption | Content, genre, mood | T5 or CLAP | Cross-attn (diffusion) / prefix (AR) | Sequential | MusicGen, AudioLDM2, Stable Audio |
| Phonemes | Exact words | G2P + text encoder | Attention / duration upsample | Sequential | Tacotron 2, FastSpeech, VITS |
| Speaker embedding | Voice identity | d/x-vector, ECAPA | FiLM / concat | Global | YourTTS, XTTS, multi-speaker TTS |
| Acoustic prompt | Voice (in-context clone) | Codec encoder | Prefix codec tokens | Global (as prefix) | VALL-E, F5-TTS, XTTS |
| Melody | Pitch contour, tune | Chromagram | Per-frame add | Time-aligned | MusicGen-melody |
| Chords / tempo / timing | Harmony, rhythm, length | Symbol/time embed | Per-frame add / time embed | Time-aligned | Stable Audio (timing), chord-cond. music |
| Prosody / style | Delivery, emotion | Reference encoder, GST | FiLM / concat (or per-frame contour) | Global or aligned | Tacotron-GST, expressive TTS |

The most useful thing to extract from this table is the **decision procedure** for a new conditioning need. Ask first: is the signal global (one fact about the whole clip) or time-aligned (a fact that changes over the clip)? That picks your injection family. Then ask: is the signal a single concept (use an embedding) or rich detail you want copied wholesale (use a raw-audio prompt)? That picks your encoder. Two questions, and the design is mostly determined.

## Measuring whether conditioning actually worked

A conditioning system you cannot measure is a conditioning system you cannot improve, and "it sounds about right" is not a metric. Each conditioning signal has a natural adherence metric, distinct from the overall fidelity metrics ([FAD, MOS, and friends](/blog/machine-learning/audio-generation/audio-quality-metrics) from the metrics post), and the discipline that separates a working pipeline from a hopeful one is measuring *adherence* and *fidelity* as two different numbers, because they trade against each other.

For **text-caption** conditioning, the adherence metric is the **CLAP-score**: encode the generated audio and the caption with CLAP, and take their cosine similarity in the shared space. A high CLAP-score means the audio matches what the caption asked for. It is imperfect (it inherits CLAP's blind spots and can be gamed by audio that is CLAP-typical but unmusical) but it is the standard, automatic text-to-audio adherence number, and it correlates reasonably with human judgments of prompt match. For **phoneme** conditioning in TTS, adherence is **intelligibility**, measured as **word error rate** (WER): run the generated speech through a strong ASR model (Whisper is the usual choice) and compare the transcript to the input text. A low WER means the words came out right; a rising WER under heavy guidance is the canary for over-guidance distortion. For **speaker** conditioning, adherence is **speaker similarity**: take the same ECAPA verification model you used to make the embedding, compute the cosine similarity between the embedding of the generated audio and the embedding of the target reference, and you have an automatic clone-fidelity number (often reported as SECS, speaker encoder cosine similarity). For **melody** conditioning, adherence is a **chroma/pitch accuracy**: extract the chromagram of the generated audio and compare it frame-by-frame to the target chromagram, scoring how often the dominant pitch class matches.

Here is the table that pulls adherence and fidelity together, the one I keep open when tuning a conditioned model:

| Signal | Adherence metric | How to compute | Fidelity metric (trades off) | Watch for |
| --- | --- | --- | --- | --- |
| Text caption | CLAP-score | cosine(CLAP-audio, CLAP-text) | FAD (VGGish or CLAP-FAD) | High guidance inflates CLAP-score but hurts FAD |
| Phonemes | WER | ASR transcript vs input text | MOS / naturalness | WER rises as guidance over-sharpens |
| Speaker embedding | SECS | cosine(ECAPA-gen, ECAPA-ref) | MOS / naturalness | Bottleneck caps SECS below prompt-based |
| Acoustic prompt | SECS | cosine(ECAPA-gen, ECAPA-prompt) | WER (stability) | Noisy prompt drops SECS and raises WER |
| Melody | Chroma accuracy | per-frame dominant-pitch match | FAD / naturalness | Tight melody can lower naturalness |
| Prosody | Style/emotion classifier acc. | classify gen vs intended style | Content WER (leakage) | Leakage shows as content/voice drift |

The single most important habit this table encodes: **always report adherence and fidelity together, never one alone.** A model that scores brilliantly on CLAP-score but terribly on FAD is over-guided and unmusical; a model with great FAD but low CLAP-score is making beautiful audio that ignores your prompt. The pair tells the story; either number alone lies. And when you sweep the guidance scale, plot *both* against $w$: adherence climbs and then plateaus, fidelity is flat and then falls, and the operating point you want is the knee where adherence has plateaued but fidelity has not yet dropped. That is the whole tuning procedure in one sentence, and it generalizes across every signal in the table.

A measurement honesty note, because this series is strict about it. SECS depends on *which* speaker encoder you use, and a clone that scores 0.85 against one ECAPA checkpoint may score 0.80 against another, so a SECS number is only comparable within a fixed encoder. CLAP-score depends on which CLAP checkpoint. WER depends on the ASR model and its own error rate on clean speech (Whisper is not a perfect oracle; it has a floor WER even on real human audio). So when you publish a conditioning number, name the encoder, the ASR, the CLAP checkpoint, the guidance scale, the prompt set, and the seed. Without those, the number is a vibe, not a measurement.

## The control-versus-fidelity tension, made precise

I have invoked the control-versus-fidelity trade several times; let me make it precise, because it is the deepest idea in conditioning and it is easy to hand-wave.

A generative model defines a distribution $p(x)$ over audio. Conditioning narrows that distribution to $p(x \mid c)$, the distribution of audio consistent with condition $c$. The more, and the more specific, the conditioning, the *narrower* this conditional distribution becomes. With a loose condition ("some music"), the conditional distribution is broad and contains many natural, high-quality samples. With a tight, multi-part condition ("this exact melody, in this exact voice, with this exact emotion, at this exact tempo"), the conditional distribution is *narrow*, and here is the catch: the narrow region of audio space that satisfies all your constraints simultaneously may contain *few* truly natural samples, because your constraints can pull in tensions the training data rarely resolved together. Forcing a bright cheerful melody through a deep mournful voice is a combination the model saw little of, so the best it can do in that narrow region is somewhat unnatural.

CFG is a *second*, orthogonal pressure on top of this. Even within $p(x \mid c)$, CFG with $w > 1$ does not sample from the true conditional; it samples from a *sharpened* distribution $\propto p(x \mid c)\, \big(p(c \mid x)\big)^{w-1}$ (this is the Bayesian reading of the guidance formula, derived in the image-series post). Sharpening concentrates mass on the most prompt-typical samples, which is why high $w$ improves adherence and kills diversity, and why, pushed far enough, it lands on samples that are *more prompt-typical than anything in the training data*, which is exactly where the artifacts live.

So there are two distinct fidelity costs, and a good engineer separates them. The first is **constraint cost**: even at $w=1$, a tight condition can force the model into a thin, unnatural region. The fix is not a knob; it is *better, more consistent training data* for the constraint combinations you care about, or *relaxing* a constraint. The second is **over-guidance cost**: pushing $w$ too high sharpens past the natural distribution. The fix *is* a knob, lower $w$. Misdiagnosing the first as the second (cranking guidance to fix a model that is struggling with a hard constraint) makes things strictly worse: you get the unnatural constrained region *and* over-guidance artifacts. When a conditioned generation sounds bad, the first question is which cost you are paying.

## Case studies: conditioning in shipped systems

Theory grounded in what real systems actually do. Four named cases, with the numbers I am confident about and honest hedging where I am not.

**MusicGen melody conditioning (Copet et al., 2023).** MusicGen ships in sizes of roughly 300M (small), 1.5B (medium), and 3.3B (large) parameters. Text conditioning is a frozen **T5-base** encoder, injected as a prefix to the single-stage codec language model over EnCodec tokens at 50 Hz with 4 codebooks. The melody variant adds **chromagram** conditioning, time-aligned, keeping the dominant pitch class per frame. The reported pattern, which matches my own use, is that melody conditioning meaningfully constrains the tune at a modest cost in raw audio quality, and that the default guidance scale of 3 is the practical sweet spot. The genuinely useful design lesson from MusicGen is the *single-stage* choice: rather than the multi-stage hierarchy of [AudioLM and MusicLM](/blog/machine-learning/audio-generation/music-generation-musiclm-and-musicgen), MusicGen interleaves the 4 codebooks with a delay pattern in one model, so all conditioning, text prefix and melody per-frame, plugs into one transformer.

**AudioLDM2 (Liu et al., 2023).** A latent diffusion model that conditions on a learned "language of audio" representation derived in part from CLAP plus a fine-tuned language model, injected by **cross-attention** into the denoising UNet, then decoded through a mel VAE and a HiFi-GAN-class vocoder. The default `guidance_scale` in the `diffusers` pipeline is around 3.5, and `num_inference_steps` in the low hundreds. The instructive bit is the *layered* text encoding: AudioLDM2 does not trust a single text encoder, it combines CLAP's audio-grounded embedding with an LM's compositional understanding, an explicit acknowledgment of the T5-versus-CLAP trade we discussed, resolved by using both.

**VALL-E (Wang et al., 2023).** The landmark for acoustic-prompt conditioning. VALL-E reframes TTS as codec-token language modeling on EnCodec tokens, with **phonemes as a text prefix** and a **3-second acoustic prompt** as a codec-token prefix. Its headline result is zero-shot voice cloning from those 3 seconds with no per-speaker training, and it reported strong speaker-similarity and naturalness gains over the prior cascaded approaches on LibriSpeech, with the well-known caveats that quality depends heavily on prompt cleanliness and that stability (occasional word repetitions or skips) is a real AR failure mode. The conditioning lesson is the purest in the post: *the prompt is the condition*, and in-context continuation is the injection.

**Stable Audio (Evans et al., 2024).** Latent diffusion for music and sound with a CLAP-style text encoder via cross-attention *and* the distinctive **timing conditioning**, start-second and total-duration embeddings, that lets it generate variable-length audio (up to minutes) that ends cleanly rather than fading arbitrarily. It is the clearest production example of a *time-related* global condition: duration is one fact about the whole clip, but it is a fact *about time*, and encoding it explicitly is what unlocked long, well-formed outputs. The takeaway is that not every useful condition is text or voice; sometimes the most valuable handle is *how long and where in the timeline*.

A note on honesty with these numbers, in the spirit of the series. FAD and MOS figures for conditioned generation are highly sensitive to the FAD embedding (VGGish versus a CLAP-based FAD give different absolute numbers), the rater pool for MOS, the exact prompts, and the guidance scale, so I have deliberately quoted *relationships and defaults* I am confident about rather than precise scores I am not. When you benchmark your own conditioning, fix the seed, the prompt set, the guidance scale, and the FAD embedding, and report all four alongside the number, or the number means nothing.

## When to reach for each conditioning signal (and when not to)

A decisive section, because the failure mode I see most is *over-conditioning*: bolting on every signal because you can, and ending up with a model that is hard to train, hard to control, and lower-fidelity than a simpler one.

**Reach for text-caption conditioning (T5 or CLAP)** whenever the user's intent is naturally expressed in words and you have caption-audio paired data to train on. This is the default for music and sound-effect generation. Prefer **CLAP** when your prompts are short, vocabulary-like descriptions and you want audio grounding cheaply; prefer **T5** (or both) when prompts are long, compositional, and full of negation and relationships. Do *not* reach for a caption encoder for TTS content, that is what phonemes are for, and a caption encoder has no notion of exact wording.

**Reach for a speaker embedding** when you need a *stable, storable, interpolatable* voice handle: a multi-speaker TTS product where users pick from a roster, or where you want to blend voices. Do *not* reach for a speaker embedding when the goal is *maximum-fidelity cloning of a specific person from a specific clip*; the 192-dimensional bottleneck will cap you below what an acoustic prompt achieves.

**Reach for an acoustic prompt (the VALL-E trick)** when you want zero-shot, high-fidelity cloning with no per-speaker setup and you can afford the prompt-length context cost every generation. Do *not* reach for it when your reference audio is noisy or clipped, the prompt copies the flaws, or when you need a clean identity handle in isolation, the prompt entangles identity with everything else in the clip.

**Reach for melody conditioning** when the *tune* matters and you want to decouple it from the *style*, the "render this melody as that genre" use case. Do *not* reach for it when you want the model's creativity over the melodic line; you would be throwing away the model's strongest ability and paying a fidelity cost for the privilege.

**Reach for prosody/style conditioning** when delivery and emotion are the product, audiobooks, characterful assistants, expressive narration. Do *not* expect clean control from a naive reference encoder; budget for the disentanglement work, or use an instruction-driven style interface that sidesteps reference leakage.

**Raise the guidance scale** only when the model is *ignoring* a condition it should be following, and stop the moment you hear artifacts or see diversity collapse. Do *not* use high guidance to paper over a hard constraint or thin training data; that is a constraint-cost problem, and guidance makes it worse, not better.

## Key takeaways

- Conditioning is the steering wheel of an audio model, and it has two organizing axes: **what it controls** (content, identity, musical line, delivery) and, more importantly for engineering, **whether it is global** (one vector for the clip) **or time-aligned** (one vector per frame).
- **Text means two different things.** For speech it is phonemes via grapheme-to-phoneme, a short symbol sequence turned into a long time-aligned acoustic plan. For music and sound it is a free-text caption encoded by **T5** (compositional, no audio grounding) or **CLAP** (audio-grounded via contrastive training, the audio CLIP).
- **CLAP enables text-to-audio** by contrastively aligning a text encoder and an audio encoder into one shared space, so a caption embedding already "knows" what the words sound like, handing the generator a grounded vector instead of a raw linguistic one.
- **Injection follows the global/aligned split:** sequential text uses cross-attention (diffusion) or a prefix (AR); global signals (speaker, coarse style) use FiLM or a prefix; time-aligned signals (melody, fine prosody) are added per frame in lockstep with the output.
- **Two ways to specify a voice:** a compact, disentangled, interpolatable **speaker embedding** (x-vector, ECAPA) that caps fidelity at its bottleneck, or a high-fidelity **acoustic prompt** (the VALL-E 3-second clip) that clones in-context but copies every flaw in the prompt.
- **Melody is the canonical time-aligned signal:** a chromagram (12 pitch classes per frame) pins the tune frame-by-frame while text keeps the style, which is why melody cannot ride as a prefix and must be added per position.
- **Classifier-free guidance is the same $\hat\epsilon = \epsilon_\text{uncond} + w(\epsilon_\text{cond} - \epsilon_\text{uncond})$ as in image generation,** but audio's failure mode at high $w$ is harshness, metallic timbre, clipping, and collapsed diversity; the sweet spot is moderate (around 3 for MusicGen, 3 to 7 for AudioLDM), and guidance doubles inference cost.
- **Control trades against fidelity in two distinct ways:** a tight condition narrows the distribution into possibly-unnatural regions (a data/constraint problem, not a knob), and over-guidance sharpens past the natural distribution (a knob problem); diagnose which before reaching for the guidance scale.
- **Do not over-condition.** Add the fewest signals that meet the intent; every extra condition is more to train, more to entangle, and usually less fidelity.

## Further reading

- **MusicGen** — Copet, Kreuk, Gat, et al., "Simple and Controllable Music Generation," 2023. The single-stage codec LM with T5 text prefix and chromagram melody conditioning; the cleanest reference for combining global and time-aligned conditioning in one model.
- **CLAP** — Wu, Chen, Zhang, et al., "Large-Scale Contrastive Language-Audio Pretraining," 2023 (and Elizalde et al., "CLAP," 2022). The contrastive text-audio alignment that grounds text encoders in sound.
- **VALL-E** — Wang, Chen, Wu, et al., "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers," 2023. The acoustic-prompt, in-context voice-cloning paradigm.
- **AudioLDM 2** — Liu, Tian, Yuan, et al., "AudioLDM 2: Learning Holistic Audio Generation with a Self-Supervised Pretraining," 2023. Cross-attention text conditioning with a layered CLAP-plus-LM text representation.
- **Stable Audio** — Evans, Carr, Taylor, et al., "Fast Timing-Conditioned Latent Audio Generation," 2024. Timing conditioning for clean variable-length generation.
- **Global Style Tokens** — Wang, Stanton, Zhang, et al., "Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis," 2018. The reference-encoder and learned-style-token mechanism for prosody.
- **Classifier-free guidance** — Ho and Salimans, "Classifier-Free Diffusion Guidance," 2022, and the [image-series derivation](/blog/machine-learning/image-generation/classifier-free-guidance).
- **Within this series:** the foundation [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard), the engines [autoregressive audio models](/blog/machine-learning/audio-generation/autoregressive-audio-models-wavenet-to-audiolm), [diffusion for audio](/blog/machine-learning/audio-generation/diffusion-for-audio), and [GAN vocoders](/blog/machine-learning/audio-generation/gan-vocoders-hifi-gan-and-fast-synthesis); the tokens [semantic vs acoustic tokens](/blog/machine-learning/audio-generation/semantic-vs-acoustic-tokens); the forward links [MusicLM and MusicGen](/blog/machine-learning/audio-generation/music-generation-musiclm-and-musicgen) and [prosody, emotion, and expressive speech](/blog/machine-learning/audio-generation/prosody-emotion-and-expressive-speech); and the capstone [building an audio generation stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack). Image-series cross-links: [text encoders and prompt conditioning](/blog/machine-learning/image-generation/text-encoders-and-prompt-conditioning) and [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance).
