---
title: "Text-to-Audio and Sound Effects: Foley for the AI Era"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "How to generate sound effects, foley, and environmental audio from a text prompt or a silent video, the third pillar of audio generation, with runnable AudioGen and AudioLDM2 code and the cross-modal sync trick that makes a soundtrack land on the frame."
tags:
  [
    "audio-generation",
    "audio-synthesis",
    "sound-effects",
    "foley",
    "video-to-audio",
    "text-to-audio",
    "neural-audio-codec",
    "generative-ai",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Audio Generation"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/text-to-audio-and-sound-effects-1.png"
---

The first time I tried to add sound to a silent clip with a model, I learned how unforgiving an ear is. The clip was three seconds of a wine glass tipping off a table and shattering on a tile floor. The model I had wired up could generate a perfectly convincing shatter, a crisp burst of high-frequency glass with exactly the right metallic ring. There was just one problem. The shatter arrived about 200 milliseconds after the glass visibly hit the floor. On paper, 200 milliseconds is nothing. In the clip it was a catastrophe. Your brain knows, with a precision it never tells you about, that the sound of an impact arrives the instant the impact happens. A fifth of a second of slack and the whole thing reads as fake, dubbed, wrong, the way a badly translated movie feels off even when you cannot say why. The timbre was perfect and the result was unusable, because the *timing* was off. That gap, between getting the sound right and getting the moment right, is the entire subject of this post.

This is the third pillar of audio generation. We have spent the series on **speech** (turning words into a voice) and **music** (turning a prompt into a song). The third domain is everything else a soundtrack is made of: the **sound effects**, the **foley**, the **environmental and ambient audio**. The footsteps on gravel, the door creak, the distant traffic, the rain on a window, the thunderclap, the dog that barks just before a car passes. These sounds are short, they are transient, they have no words and no melody, and yet getting them right is its own deep problem, because they live or die on two things our ears measure ruthlessly: the **temporal envelope** (when the energy arrives and how fast it decays) and the **timbre** (the spectral fingerprint that tells a crunch of gravel from a crunch of snow). By the end of this post you will be able to generate a sound effect from a text caption with two different open models, reason about why sound effects need a different modeling emphasis than speech or music, build the conceptual pipeline that generates a soundtrack synced to a silent video, and measure all of it honestly.

![A dataflow graph showing a text caption and a silent video both feeding a conditioner into a generative audio model that emits a short sound effect waveform](/imgs/blogs/text-to-audio-and-sound-effects-1.png)

The figure above is the shape of the whole field. There are two ways to ask a model for a sound. You can describe it in **text**, "a thunderclap," "footsteps on gravel," "a dog barking then a car passing," and the model invents a sound that matches the words. Or you can hand it a **silent video** and ask it to generate the soundtrack that belongs to those visuals, with every audio event landing on the frame where the visible action happens. The first is **text-to-audio**. The second is **video-to-audio**, also called **neural foley**, after the craft of the foley artist who records footsteps and punches and cloth rustles in a studio to lay under a film. Both feed the same kind of generative core, a codec language model or a diffusion model, but they differ in where the *timing* comes from: from your words in one case, from the visual stream in the other. That difference is what makes video-to-audio the harder and more interesting problem.

This sits on the series spine: the **audio stack** of waveform to neural-codec tokens or mel latent to generative model to vocoder back to waveform, under the tension of **fidelity, controllability, speed, and length**. Sound effects push hard on a corner of that frame the other domains touch lightly. They are *short* in length but *unforgiving* in fidelity, specifically the fidelity of the onset. And in the video-to-audio case they add a fifth axis the speech and music posts could largely ignore: **synchronization**, the requirement that the generated audio be aligned in time with an external signal. If you have not read [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard), the foundation post, its framing about audio being short in wall-clock time but enormous in sample count is exactly why a 200-millisecond drift is both perceptually fatal and, in token terms, a tiny error the loss barely notices. We will come back to that tension repeatedly.

## What makes a sound effect a different problem

Before any code, it is worth being precise about why sound effects are not just "short music" or "speech without words." They are a genuinely different modeling problem, and the difference is the key to everything else in this post.

Start with what speech and music have that sound effects do not: **long-range structure**. Speech has linguistic structure, phonemes assemble into words, words into a grammatical sentence with a meaning that constrains what can come next; a model that forgets the subject of a sentence by the time it reaches the verb produces nonsense. Music has tonal and rhythmic structure, a key, a chord progression, a beat that must stay coherent over minutes; a model that drifts out of key or loses the tempo produces something a listener rejects. Both demand that the model maintain a plan over a long horizon. That is why both lean on architectures, autoregressive language models and long-context diffusion, that are good at long-range coherence, and why the [autoregressive audio models](/blog/machine-learning/audio-generation/autoregressive-audio-models-wavenet-to-audiolm) and [diffusion for audio](/blog/machine-learning/audio-generation/diffusion-for-audio) posts spent so much energy on context length.

A sound effect has almost none of this. A thunderclap is not a sentence. It does not have a grammar. There is no "wrong next note" in a door creak. What a sound effect *does* have, and what it must get exactly right, is two things that operate on a much shorter timescale.

![A vertical stack showing the four things a sound effect must get right: a sharp onset envelope, correct timbre, a short window, and the absence of long-range linguistic structure](/imgs/blogs/text-to-audio-and-sound-effects-3.png)

The first is the **temporal envelope**: the shape of the sound's energy over time. A gunshot is a near-instantaneous spike, a few milliseconds of attack and then a fast decay. A door creak is a slow swell. Rain is a stationary texture, roughly constant energy with fine random fluctuation. A dog bark is a series of sharp bursts with gaps. The envelope, especially the **onset**, the moment the energy first arrives and how fast it rises, is the single most perceptually important feature of a transient sound. Our auditory system evolved to localize and identify sudden events (a snapping twig, a predator's footfall), so it is exquisitely sensitive to onsets. A sound with the right spectrum but a smeared or mistimed onset reads as wrong immediately. This is why my shattered-glass clip failed: the *spectrum* was a perfect shatter, but the *onset* landed 200 milliseconds late.

The second is **timbre**: the spectral content that distinguishes one sound source from another. Footsteps on gravel and footsteps on a wooden floor have nearly identical envelopes, a series of soft impacts at a walking cadence, but completely different timbres, the gravel has a high-frequency crunch from many small particles shifting, the wood has a hollow resonant thud. The model has to get the *spectral fingerprint* right, the distribution of energy across frequencies and how it evolves through the sound. Timbre is what makes a synthesized "fire" sound like fire and not like white noise or frying bacon (which, famously, is what fire often sounds like to a foley artist, but that is a different story).

What a sound effect does *not* need is the thing speech and music spend most of their capacity on: a coherent plan over seconds or minutes. This flips the modeling emphasis. For sound effects you want a model that is excellent at producing the right **local** spectro-temporal pattern, the right onset, the right timbre, the right short envelope, and you care comparatively little about long-range consistency, because there is little long-range structure to be consistent with. A two-second sound effect is genuinely a *short* generation problem, which is good news for latency and bad news for nothing, except that the bar for the short part is brutally high.

There is one more property worth naming because it shapes the data problem later: sound effects are **diverse and unbounded** in a way speech and even music are not. Speech is constrained, it is one species making vowel and consonant sounds with one vocal tract. Music is constrained by instruments and tonality. But "a sound" is anything: a zipper, a sword unsheathing, an alien spaceship, a heartbeat, a city at night, a single water drop in a cave. The space of sounds a text-to-audio model must cover is vast and the labeled data is thin, which is the central reason the field looked the way it did until recently. We will get to that.

It is worth pinning down the envelope idea quantitatively, because "the onset matters" sounds soft until you put numbers on it. A useful way to think about a sound is as an **amplitude envelope** $a(t)$ multiplying a faster-varying carrier: the envelope is the slow outline of the energy, the carrier is the fine texture inside it. For a percussive sound the envelope is often modeled as a fast attack and an exponential decay, $a(t) \approx (1 - e^{-t/\tau_{\text{atk}}})\, e^{-t/\tau_{\text{dec}}}$, where $\tau_{\text{atk}}$ is a few milliseconds for a crack or a click and tens of milliseconds for a soft thud, and $\tau_{\text{dec}}$ sets how long it rings. Two sounds with the *same* spectrum but different $\tau_{\text{atk}}$ are perceived as completely different events, a sharp $\tau_{\text{atk}}$ reads as an impact, a slow one as a swell. Crucially, the ear's temporal resolution for onsets is on the order of a few milliseconds, far finer than the roughly 20-millisecond frame the model usually works at, which is precisely why onset fidelity is a problem the codec frame rate can blur. The lesson for modeling is direct: a sound-effect model is in large part an *envelope* model, and any part of the stack that smooths the envelope, a low-frame-rate codec, a transient-averaging vocoder, will be heard.

The flip side is that because there is no long-range structure to maintain, a sound-effect model can be *smaller and shorter-context* than a comparable speech or music model and still do its job well. You are not asking it to remember the start of a sentence or hold a key for two minutes; you are asking it to nail a two-second local pattern. That is genuinely good news for deployment, the compute you would spend on long-context attention in a music model you can instead spend on the codec and vocoder fidelity that the onset and timbre actually need. The modeling budget *shifts* from "remember a lot" to "render the local detail exactly right," and recognizing that shift is half of building a good SFX system.

## The task, stated precisely

Let me state the two tasks crisply, because the rest of the post is about how to do each one well.

**Text-to-audio** takes a free-text caption $c$ (an English description like "a heavy rainstorm with distant thunder") and produces a waveform $x$ whose content matches the description. The model learns $p(x \mid c)$, the distribution over sounds consistent with the caption, and samples from it. There is no requirement that the output be aligned to anything external, the model is free to decide when the thunder happens; it just has to produce a believable rainstorm-with-thunder. This is the direct analogue of text-to-image and text-to-music: a caption in, a sample of the matching modality out.

**Video-to-audio** (foley) takes a silent video $v$, a sequence of image frames, and optionally a text caption $c$, and produces a waveform $x$ that is both *semantically* appropriate (it sounds like what is on screen) and *temporally aligned* (the audio events happen at the same instants as the visible events). The model learns $p(x \mid v, c)$, and the hard part is the alignment. A footstep visible at frame 30 of a 24-frames-per-second clip must produce an audio impact at the corresponding 1.25-second mark in the waveform, not a quarter-second early or late. This is the [audio-visual generation](/blog/machine-learning/video-generation/audio-and-joint-av-generation) problem from the video series, viewed from the audio side: there we cared about generating video and audio jointly; here we take the video as given and generate only the audio to match it.

The distinction matters for evaluation, too. For text-to-audio you ask: does it sound like the caption (a CLAP-score, a semantic-match metric), and does it sound real (a Fréchet Audio Distance against a reference distribution)? For video-to-audio you ask both of those *plus*: is it in sync (an audio-visual synchronization score that measures temporal alignment between the audio onsets and the video events)? Sync is a metric the text-only task does not even have, and it is the one that is hardest to satisfy. We will define all of these precisely in the results section.

It helps to keep one running example through the post. I will use **"footsteps on gravel"** as the text-to-audio thread, because it has a clear envelope (a series of soft impacts at a walking cadence) and a clear timbre (the high-frequency crunch of shifting small stones), and **a five-second clip of a person walking across a gravel courtyard** as the video-to-audio thread, because the very same sound now has to land each crunch on the frame where the foot lands. Same sound, two tasks, and the gap between them is the whole story.

## AudioGen: a codec language model for sound

The cleanest way into text-to-audio is **AudioGen** (Kreuk et al., Meta, 2022), because it is the sound-effect counterpart of MusicGen and reuses machinery we have already built in the series. If you understood [MusicGen](/blog/machine-learning/audio-generation/music-generation-musiclm-and-musicgen), you almost understand AudioGen already; the differences are instructive.

![A dataflow graph showing a T5 text encoder and past codec tokens both feeding a transformer language model that predicts EnCodec tokens, which a codec decoder turns into a sound-effect waveform](/imgs/blogs/text-to-audio-and-sound-effects-4.png)

The recipe is the codec-language-model recipe. First, a [neural audio codec](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound), an EnCodec-style encoder, turns the target waveform into a short sequence of discrete tokens via [residual vector quantization](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq). A second of 16 kHz audio that was 16,000 floating-point samples becomes, after the codec, something like 50 frames per second times a handful of codebooks, a few hundred integer tokens. That is the compression that makes language modeling of audio tractable at all. Second, a **transformer language model** is trained to predict the next codec token autoregressively, conditioned on a **text caption** encoded by a frozen **T5** text encoder. The caption hidden states are injected as conditioning (a prefix or via cross-attention, depending on the variant), exactly as in [conditioning and control](/blog/machine-learning/audio-generation/conditioning-and-control-in-audio-generation). Third, at generation time the model samples codec tokens one at a time, and the **codec decoder** turns the predicted tokens back into a waveform.

Why does this work well for sound effects specifically? Two reasons. First, the autoregressive token model is naturally good at *transients and onsets*, because predicting the next token at 50 Hz means the model decides, frame by frame, when energy arrives, it can place a sharp onset at exactly the frame it wants. Second, AudioGen made a data-and-augmentation choice that matters for SFX: it trained on a mixture of many audio-text datasets and used an augmentation that *mixes* pairs of audio clips and concatenates their captions, so the model learns compositional prompts like "a dog barking and a bird chirping," which is exactly the kind of layered scene a sound designer asks for. That compositional ability, generating two or three named sources in one clip, is a hallmark of a good text-to-audio model and it came partly from teaching the model on mixtures.

Here is text-to-audio with AudioGen through the `audiocraft` library, the original Meta implementation:

```python
import torchaudio
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write

# medium is ~1.5B params; there is also a small (~300M) checkpoint.
model = AudioGen.get_pretrained("facebook/audiogen-medium")

# duration in seconds; AudioGen generates short clips well, drifts on long ones.
model.set_generation_params(duration=5.0, top_k=250, temperature=1.0)

prompts = [
    "footsteps on gravel, steady walking pace",
    "a dog barking, then a car passing on a wet road",
    "heavy rain with distant thunder",
]

wavs = model.generate(prompts)  # (batch, channels, samples) at 16 kHz

for prompt, wav in zip(prompts, wavs):
    name = prompt[:24].replace(" ", "_").replace(",", "")
    # audio_write appends .wav and normalizes loudness sensibly.
    audio_write(name, wav.cpu(), model.sample_rate, strategy="loudness")
    print(f"wrote {name}.wav  ({wav.shape[-1] / model.sample_rate:.1f}s)")
```

A few things are worth flagging in this snippet. The sample rate is **16 kHz**, not 44.1 kHz, AudioGen targets a relatively low sample rate, which is fine for many sound effects (a lot of SFX energy is below 8 kHz) but is a real ceiling for bright, airy sounds like cymbals or high hiss; you will hear the band limit on those. The `top_k` and `temperature` are the usual sampling controls; lower temperature gives more typical, "safe" sounds, higher gives more variety and more risk of a weird artifact. And `duration=5.0` is in the comfort zone; ask AudioGen for 30 seconds and it will tend to lose the thread, because, like all autoregressive token models, it accumulates error over a long horizon, and a long ambient texture has no strong structure to pull it back on track. For a sustained ambience you are usually better off generating a few seconds and looping or stitching, a practical trick we will revisit.

The one-line summary to carry away: **AudioGen is to sound effects what MusicGen is to music and what VALL-E is to speech**, the same codec-language-model paradigm, retargeted at a different slice of the audio distribution, with data and augmentation choices tuned for that slice. The codec is the tokenizer, the transformer is the engine, the text encoder is the steering wheel. Everything you learned about that stack transfers.

#### Worked example: AudioGen sizes and what they cost

AudioGen ships in roughly two sizes: a small checkpoint around 300M parameters and a medium around 1.5B. On an RTX 4090 (24 GB), generating a 5-second clip with the medium model takes very roughly on the order of a few seconds of wall-clock once the model is warm, so the **real-time factor** (RTF = generation-time / audio-duration) is around 1 or below for short clips, fast enough to be interactive but not free. The small model is roughly 2 to 3 times faster and noticeably lower fidelity, you will hear more graininess and weaker timbres, which is the usual capacity-versus-quality trade. I am quoting these as order-of-magnitude figures because exact numbers depend on your `top_k`, sampling temperature, `torch.compile` status, precision (bfloat16 versus float32), and whether you batch; measure on your own box with a fixed prompt set and a warm-up pass before you trust any RTF number. The honest takeaway is the *relationship*: medium is the quality default, small is the latency escape hatch, and both are interactive for short SFX on a consumer GPU.

## Diffusion text-to-audio: AudioLDM, Make-An-Audio, and Tango

The codec-language-model route is one of two dominant families. The other is **latent diffusion**, the same machinery as text-to-image, run on an audio latent. Three closely related systems define this family, and it is worth seeing how they relate because they share a backbone.

**AudioLDM** (Liu et al., 2023) is the canonical one. It compresses a mel-spectrogram into a latent with a VAE, runs a diffusion model in that latent space conditioned on a **CLAP** text embedding, and decodes the generated mel-latent back to a waveform with a HiFi-GAN-class vocoder. If you read [diffusion for audio](/blog/machine-learning/audio-generation/diffusion-for-audio), this is exactly the latent-diffusion recipe described there; the sound-effect angle is mostly in the *data* and the *conditioning*. The clever bit in AudioLDM is that it leans hard on CLAP's audio-grounded text-audio space (covered in [conditioning and control](/blog/machine-learning/audio-generation/conditioning-and-control-in-audio-generation)): because CLAP maps text and audio into one shared space, AudioLDM can even be trained with *audio* embeddings as the condition and prompted with *text* embeddings at inference, since the two live in the same space. That trick stretches scarce paired data.

**Make-An-Audio** (Huang et al., 2023) is a contemporary system with the same latent-diffusion shape and a strong emphasis on a **data-augmentation pipeline** to manufacture more audio-text pairs, a distillation-and-captioning approach to the data scarcity we will discuss next. **Tango** (Ghosal et al., 2023) is the third, and its distinctive move is to swap CLAP for a **frozen large language model** (an instruction-tuned FLAN-T5) as the text encoder, on the bet that a strong LLM understands compositional and ordered prompts ("a dog barks *and then* a car passes") better than a contrastive encoder does. Tango reported competitive or better results on the standard benchmark with that single change, which is a nice illustration of the T5-versus-CLAP trade from the conditioning post, here resolved in favor of the language model's compositional grip.

Here is text-to-audio with AudioLDM2 through 🤗 `diffusers`, which is the most copy-and-run path:

```python
import torch
import soundfile as sf
from diffusers import AudioLDM2Pipeline

repo = "cvssp/audioldm2"  # there is also audioldm2-large
pipe = AudioLDM2Pipeline.from_pretrained(repo, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "footsteps on gravel, steady walking pace, close mic"
negative = "low quality, muffled, music"  # CFG steers away from these

generator = torch.Generator("cuda").manual_seed(0)  # fix the seed for A/B tests
audio = pipe(
    prompt,
    negative_prompt=negative,
    num_inference_steps=200,     # diffusion steps; quality vs speed knob
    guidance_scale=3.5,          # CFG strength; ~3-4 is the SFX sweet spot
    audio_length_in_s=5.0,
    generator=generator,
).audios[0]                       # numpy array, mono

sf.write("gravel_audioldm2.wav", audio, samplerate=16000)
print("samples:", audio.shape[0], "≈", audio.shape[0] / 16000, "s")
```

The arguments map directly onto the science. `num_inference_steps` is the number of denoising steps, the fidelity-versus-speed knob you know from image diffusion; 200 is high-quality-slow, you can drop to 50 or fewer with a faster scheduler and lose some detail. `guidance_scale` is [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance); for sound effects a value around 3 to 4 is the sweet spot, push it to 7 or 10 and you get the audio equivalent of an over-sharpened image, harsh, brittle, sometimes clipping, with collapsed diversity. The `negative_prompt` is a genuinely useful SFX control: putting "music" in the negative prompt is a practical way to stop a text-to-audio model from sneaking a musical bed under your sound effect, a real failure mode when your caption is ambiguous.

The trade between the two families, codec-LM (AudioGen) versus latent-diffusion (AudioLDM/Tango), mirrors the AR-versus-diffusion split from the rest of the series. The codec LM samples sequentially (one token at a time, naturally good at sharp onsets and streamable) while diffusion samples in a fixed number of parallel steps (good at full-context coherence and easy to guide). For short, transient SFX both work well; for longer or more textured ambiences, diffusion's full-context view often holds together better, while for tight, percussive onsets the AR model's frame-by-frame control is appealing. Neither dominates; pick by your sound and your latency budget.

It is worth being concrete about the compression that makes either family tractable, because it is the same number that bounds your latency. A 5-second clip at 16 kHz is 80,000 raw samples. No autoregressive model is going to emit 80,000 tokens per clip at a useful speed. The codec is what rescues this: an EnCodec-style codec at a 50 Hz frame rate turns those 5 seconds into 250 frames, and with, say, 4 residual codebooks that is 1,000 discrete tokens (or fewer if you interleave the codebooks cleverly, as MusicGen-style delay patterns do). One thousand tokens is a sequence a transformer generates comfortably. So the codec buys roughly an 80-to-1 reduction in the number of steps the generative model takes, and that ratio is *exactly* the thing standing between "a few seconds to generate" and "unusably slow." For diffusion the same logic applies to the latent: you diffuse a compressed mel or codec latent, not the 80,000-point waveform, so each of your 50-to-200 denoising steps operates on a small tensor. Whichever family you pick, the compression ratio of the codec is the single biggest lever on your real-time factor, which is why the [codec posts](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound) sit upstream of everything in this series.

One more practical note that bridges the two families: you can mix them. A common production pattern is to generate the *content* with whichever model you prefer at a modest sample rate, then run a separate **upsampling or bandwidth-extension** model to add the high-frequency detail and bring 16 or 24 kHz audio up to 44.1 or 48 kHz. That two-stage approach lets you keep a fast, well-understood generator and recover the brightness a low-rate model loses, at the cost of a second pass. It is not free and it is not magic, the extension model is *inventing* plausible high frequencies, not recovering real ones, but for many sound effects the invented top octave is convincing enough, and it is a pragmatic alternative to retraining your whole generator at a higher sample rate.

## The data problem: why audio-text pairs are scarce

Here is the thing that shaped this whole subfield until recently, and that you must understand to read the literature correctly: **paired audio-text data is scarce, noisy, and small** compared to the image and text worlds.

Think about why. Images come with captions all over the web, alt text, surrounding article text, dense human-captioned datasets. Text is, obviously, abundant. But a sound clip with a faithful text description of *what it sounds like* is rare. Most audio on the internet is either speech (transcribed, but a transcript describes the *words*, not the *sound*) or music (tagged with genre and artist, not "warm overdriven guitar with vinyl crackle") or it is the audio track of a video with no description of the sound at all. The clean, human-written caption "a glass shatters on a tile floor, followed by a soft sweep" is something almost nobody writes, because in the real world nobody narrates the sounds around them.

The community's main resources tell the story. **AudioSet** (Gemmeke et al., 2017) is huge, around two million ten-second YouTube clips, but it is *labeled*, not *captioned*: each clip carries one or more tags from an ontology of around 600 sound classes ("Dog," "Vehicle," "Rain"), not a free-text sentence. Labels are great for classification and weak for generation, because generation wants the rich, compositional description, not a bag of class tags. **AudioCaps** (Kim et al., 2019) is the workhorse *captioned* set, around 50,000 clips (a subset of AudioSet) with human-written captions, and it is the standard text-to-audio benchmark, but 50,000 pairs is tiny next to the hundreds of millions of image-text pairs that train text-to-image models. **Clotho** (Drossos et al., 2020) adds a few thousand more carefully captioned clips. That is roughly the well-curated supply, and it is small.

So the field did what you do when labels are scarce: it **manufactured captions**. The standard playbook, used in different forms by Make-An-Audio, AudioGen's data mixture, and others, is a captioning pipeline:

```python
# Conceptual: turn a labeled / unlabeled audio clip into a training caption.
# Real pipelines combine several of these signals.

def caption_clip(clip):
    # 1) If the clip has AudioSet tags, template them into a sentence.
    tags = clip.get("audioset_tags", [])          # e.g. ["Dog", "Bark"]
    templated = ", ".join(t.lower() for t in tags) # "dog, bark"

    # 2) If it is the audio track of a video, mine the title / description /
    #    ASR transcript for sound-relevant phrases (filter out speech content).
    video_text = mine_video_metadata(clip)         # noisy, web-scale

    # 3) Use an audio captioning model (trained on AudioCaps) to PREDICT a caption.
    predicted = audio_captioner(clip.waveform)      # "a dog barking outdoors"

    # 4) Optionally rewrite/clean with an LLM into fluent, varied phrasings.
    return llm_rewrite(predicted, hints=[templated, video_text])
```

Every line of that pipeline is *lossy and noisy*. Templated tags are stilted ("dog, bark" is not how anyone describes a sound). Video metadata is web-scale but mostly irrelevant to the sound. The audio captioner is itself trained on the same tiny AudioCaps and inherits its blind spots, so you are bootstrapping captions from a model trained on the very scarcity you are trying to fix. And LLM rewriting can hallucinate details not in the audio. The result is *more* pairs but *noisier* pairs, and managing that noise, filtering, weighting, deduplicating, is a large part of what separates a good text-to-audio model from a mediocre one. When you read that a model "scaled to X hours of audio-text data," read it skeptically: a lot of that data was captioned by a machine, and the quality of the captioning pipeline is doing quiet, heavy lifting.

This data scarcity is also *why video-to-audio is so attractive*, and it is the bridge to the second half of the post. Video with audio is *everywhere*, every video on the internet is a paired (silent-video, audio) example, with the sound already perfectly synced to the visuals, for free, at web scale, with no human captioning needed. The supervision the text-to-audio world has to manufacture, the video-to-audio world gets automatically: the original soundtrack of any video is the ground-truth answer to "what should this video sound like." That is an enormous data advantage, and it is a big reason the frontier moved toward video conditioning.

## Video-to-audio: the foley problem

Now the harder, more interesting task. We have a silent video and we want its soundtrack.

![A before-and-after comparison showing a silent video with no sound on the left and a foley-synced version on the right where impacts land on the visible actions](/imgs/blogs/text-to-audio-and-sound-effects-2.png)

The figure makes the goal concrete. On the left, the silent clip: you see the footsteps, you see the glass shatter, you see the action, but there is dead air, and dead air over visible action reads as deeply unreal, this is exactly why silent films had live piano. On the right, the foley-synced version: each footstep produces a gravel crunch on the frame the foot lands, the shatter onset is locked to the frame the glass hits, and a quiet ambient bed fills the scene. The model has to do two things at once: produce the *right* sounds (semantic correctness, gravel not carpet) and produce them at the *right times* (temporal sync, on the frame not 200 milliseconds late). The first is the same problem as text-to-audio. The second is new, and it is the crux.

The applications are immediate and large. **Film and game foley**: a foley artist's day of recording footsteps and cloth rustles is exactly the kind of laborious, high-skill work a model can assist or accelerate, generating a first-pass soundtrack a sound designer then refines. **Video dubbing and restoration**: silent or sound-damaged archival footage, or AI-generated video (which is born silent, the [video generation](/blog/machine-learning/video-generation/audio-and-joint-av-generation) models produce no audio) that needs a soundtrack. **Accessibility**: richer audio description for the visually impaired, or sound for content that lacks it. The single biggest near-term driver is the last point about generated video: as text-to-video models proliferate, every clip they make is silent, and somebody has to sound it. Video-to-audio is the natural partner to text-to-video, which is exactly why the frontier labs that build video models, Meta with Movie Gen, Google with its video stack, build a video-to-audio model right alongside.

Let me be honest about the difficulty before the methods. Semantic correctness, "this looks like rain, generate rain", is largely solved by the same conditioning machinery as text-to-audio; you swap the text encoder for a visual encoder and the model learns to map "wet street, falling streaks" to "rain sound." The genuinely hard, unsolved-until-recently part is the *timing*, and the reason it is hard is a beautiful, concrete mismatch we have to confront directly.

## The temporal-sync problem, precisely

Here is the heart of the post. Why is it hard to make a generated sound land on the right frame?

![A before-and-after comparison contrasting an unaligned video-to-audio model whose onsets drift with a sync-trained model whose onsets land on the visible frame](/imgs/blogs/text-to-audio-and-sound-effects-8.png)

The first reason is a literal **rate mismatch**. Video runs at a *frame rate*: 24, 25, or 30 frames per second is typical. Audio runs at a *sample rate*: 16,000, 24,000, or 48,000 samples per second. These are different clocks, off by three orders of magnitude, and the audio model rarely works at the raw sample rate anyway; it works in a *latent* or *token* rate, maybe 50 Hz for a codec, or some down-sampled feature rate for a diffusion latent. So a video-to-audio model has at least three time grids to reconcile: the video frame grid (say 24 Hz), the audio latent grid (say 50 Hz), and the final waveform grid (say 24,000 Hz). A footstep at video frame 30 (time 30/24 = 1.25 s) must produce audio energy at latent frame round(1.25 × 50) = 62 (or 63), which after decoding becomes a sample-domain onset at sample 1.25 × 24000 = 30,000. Every one of those conversions is an opportunity to be off by a frame, and one frame at 24 fps is about 42 milliseconds, already at the edge of audible misalignment. Get it wrong by a few frames and you are in the 100-to-300-millisecond drift zone that reads as dubbed.

The second reason is deeper and is the one that defeats naive approaches: the **loss function barely sees the timing**. Suppose you train a video-to-audio model with the usual generative loss, a reconstruction or diffusion loss on the audio (latent). That loss is, roughly, an average error over all the audio frames. A footstep is a *sparse, brief* event, maybe 50 milliseconds of energy in a 5-second clip. If the model places that footstep one frame too late, the loss penalty is tiny, it got 99% of the (mostly silent or ambient) clip right, and the misplaced 50-millisecond burst is a rounding error in the average. So the training signal is almost indifferent to exactly the thing the human ear cares about most. This is the cross-modal version of the onset-sensitivity point from earlier: *the perceptual cost of a timing error is enormous, the loss cost is negligible, and that gap is why sync does not come for free.* You have to *engineer* the model to care about timing, because the default loss will not make it care.

The third reason is that the *information* about timing has to flow from the video into the audio in a way the model can use at the right resolution. The visual events that cause sound, an impact, a contact, a sudden motion, are encoded in the *changes between frames* (a foot is in the air, then it is on the ground; a glass is whole, then it is in pieces). The model needs **motion-aware** or temporally-localized visual features, not just a single pooled "what is in this video" embedding, because a single pooled embedding tells you it is a "walking on gravel" video but throws away *when* each step happens. So good video-to-audio systems extract *frame-level* or *clip-level temporal* features and feed them to the audio model time-aligned, so the model knows not just *what* but *when*.

How do the real systems solve this? Three families of techniques recur:

- **Explicit temporal feature alignment.** Extract per-frame visual features (often from a video encoder like a CLIP-style image encoder applied per frame, plus motion features), then *resample* them to the audio latent rate so there is a one-to-one correspondence between a visual feature and an audio latent frame. Now the model can attend to "the visual feature at this audio frame" and place energy accordingly. This is the workhorse, and the resampling step, interpolating 24 Hz visual features up to 50 Hz audio latent frames, is the literal answer to the rate mismatch.

- **A synchronization objective or module.** Add a loss term, or a discriminator, or a pretrained audio-visual sync model, that explicitly rewards temporal alignment, so the training signal *does* see the timing instead of averaging it away. Some systems borrow a **pretrained AV-sync network** (the kind trained to detect lip-sync errors, e.g. SyncNet-style) and use its score as a training signal or a re-ranking criterion at inference, generate several candidates and keep the most in-sync one. This directly attacks the "loss barely sees timing" problem by adding a loss that does.

- **Joint multimodal training.** Train the audio generator on video features from the start, on the web-scale (silent-video, real-audio) pairs that are free, so the model learns the audio-visual *temporal* correlations directly from data rather than bolting them on. The frontier systems do this at scale, which is part of why they sync better, they have simply seen vastly more examples of "this visual motion co-occurs with this sound at this lag."

The honest summary: **sync is a real, hard, cross-modal problem and the gains of the last two years came from taking it seriously as a first-class objective, not from a single trick.** When you read that a new V2A model "improves synchronization," look for *which* of these three levers it pulled, almost always it is better temporal features, an explicit sync objective, more joint data, or some combination.

## The video-to-audio pipeline, end to end

Let me lay out the full pipeline as you would actually build it, then give the conceptual code.

![A vertical stack of the video-to-audio pipeline stages: silent video in, visual encoder, temporal alignment, conditioned audio model, and mux with ffmpeg](/imgs/blogs/text-to-audio-and-sound-effects-6.png)

The stack has five stages. First, **silent video in**, a sequence of frames at some frame rate. Second, a **visual encoder** turns the frames into features; in practice this is a per-frame image encoder (a CLIP-style ViT) for semantics, often augmented with a *temporal* or *motion* feature stream so the encoder captures *changes* (the events that cause sounds), not just static content. Third, **temporal alignment**, the resampling step that maps the video frame rate to the audio model's latent rate so visual features and audio latents line up one-to-one in time; this is the stage that fixes the rate mismatch, and it is where most sync bugs live. Fourth, the **conditioned audio model**, a diffusion or flow model (or a codec LM) that generates the audio latent conditioned on the time-aligned visual features (and optionally a text prompt for extra control). Fifth, **mux**, combine the generated audio with the original video into a single file, in practice with `ffmpeg`.

Here is the conceptual pipeline in code. I am explicit that the middle stage is a stand-in for whichever V2A model you use (MMAudio, FoleyCrafter, a Movie-Gen-style model); the *shape* of the pipeline is what to internalize, not a specific API:

```python
import subprocess
import torch
import torchaudio

# --- Stage 1: read the silent video and sample its frames ---
# (decord / torchvision give you frames; here we sketch the shapes.)
video_fps = 24
frames = load_frames("walk_on_gravel.mp4")     # (T_v, 3, H, W), T_v frames
duration_s = frames.shape[0] / video_fps        # e.g. 5.0 seconds

# --- Stage 2: visual encoder -> per-frame features ---
# A per-frame CLIP-style encoder for semantics + a motion stream for events.
visual_encoder = load_visual_encoder()           # frozen, e.g. CLIP ViT
vis_feats = visual_encoder(frames)               # (T_v, D), one vector per frame

# --- Stage 3: temporal alignment to the audio latent rate ---
audio_latent_hz = 50                             # the audio model's frame rate
T_a = round(duration_s * audio_latent_hz)        # number of audio latent frames
# Resample visual features from video_fps (24 Hz) up to 50 Hz so each audio
# latent frame has a matching visual feature. THIS is the rate-mismatch fix.
aligned_feats = torch.nn.functional.interpolate(
    vis_feats.transpose(0, 1).unsqueeze(0),      # (1, D, T_v)
    size=T_a, mode="linear", align_corners=False,
).squeeze(0).transpose(0, 1)                     # (T_a, D), aligned to audio

# --- Stage 4: conditioned generative audio model ---
# Stand-in for a real V2A model (MMAudio / FoleyCrafter / Movie-Gen-style).
# It generates the audio latent conditioned on the time-aligned visual features
# (and optionally a text prompt for extra control), then decodes to a waveform.
v2a_model = load_v2a_model()
waveform = v2a_model.generate(
    visual_cond=aligned_feats,                   # time-aligned: drives WHEN
    text_cond="footsteps on gravel",             # optional: nudges WHAT
    seconds=duration_s,
)                                                # (1, samples) at, say, 24 kHz
torchaudio.save("foley.wav", waveform.cpu(), 24000)

# --- Stage 5: mux the generated audio onto the original video ---
subprocess.run([
    "ffmpeg", "-y",
    "-i", "walk_on_gravel.mp4",                  # original (silent) video
    "-i", "foley.wav",                           # generated soundtrack
    "-c:v", "copy",                              # don't re-encode the video
    "-map", "0:v:0", "-map", "1:a:0",            # video from #0, audio from #1
    "-shortest", "foley_video.mp4",
], check=True)
```

The single most important line in that snippet is the `interpolate` call in stage 3. That is the literal answer to "how do you reconcile 24 frames per second with a 50 Hz audio latent": you resample the visual features onto the audio grid so the model can attend to "the visual feature at *this* audio frame." Get that resampling wrong, off by a frame, wrong rounding, an off-by-one in the frame count, and your sync drifts in exactly the perceptually-fatal way. In a real system the alignment is usually learned and more sophisticated than a linear interpolation (the model may have its own temporal attention across the two streams), but the *conceptual* job is this: put the visual events and the audio frames on the same clock.

The `ffmpeg` mux in stage 5 is mundane but worth getting right: `-c:v copy` avoids re-encoding the video (faster, lossless), `-map` selects the video stream from the original file and the audio stream from the generated wav, and `-shortest` trims to the shorter of the two so you do not get a trailing tail of silent video or audio. These are the practical details that bite when you ship.

#### Worked example: the cost of an off-by-N alignment

Suppose your video is 24 fps and your audio latent runs at 50 Hz, and you generate a 5-second clip of someone walking, one footstep roughly every 0.5 seconds, so about 10 steps. If your temporal alignment is off by **one audio latent frame**, that is 1/50 = 20 milliseconds, borderline; a careful listener might feel a slight looseness but most would accept it. Off by **two latent frames**, 40 milliseconds, and it starts to read as soft, dubbed footsteps. Off by **five frames**, 100 milliseconds, and every step is audibly disconnected from the foot; the clip is unusable. Now notice the asymmetry: a generative loss that scores this clip would see *almost no difference* between the 20-millisecond and 100-millisecond versions, both place the same ten footstep sounds with the same timbres; the average reconstruction error changes by a hair. The human verdict swings from "fine" to "broken." That asymmetry, large perceptual delta, tiny loss delta, is exactly why you cannot rely on the reconstruction loss alone and why AV-sync-aware training or re-ranking exists. The number to remember: roughly **one video frame (about 40 ms at 24 fps) is the budget**; spend more than a frame or two of drift and the ear notices.

## The systems: MMAudio, Movie Gen Audio, FoleyCrafter, V-AURA, and Google's V2A

Let me ground the methods in the named systems, because the design choices differ in instructive ways. I will be careful to flag what I am confident about and what is approximate, in the spirit of the series.

**MMAudio** (Cheng et al., 2024) is the open system I reach for first to explain the modern approach. Its central idea is **multimodal joint training**: rather than training only on (video, audio) pairs, it trains on *both* audio-text data *and* audio-video data jointly in a single flow-matching model with a shared representation, so the abundant text-audio data helps the scarcer video-audio task and vice versa. It uses a flow-matching generative core (see [flow matching for audio](/blog/machine-learning/audio-generation/flow-matching-and-consistency-for-audio)) and pays explicit attention to synchronization, reporting strong audio quality and AV-sync on the standard benchmarks while remaining open-weight. The lesson from MMAudio is the data-mixing one: the text-to-audio data scarcity and the video-to-audio data abundance are *complementary*, and training on both at once is a way to get the best of each.

**Meta Movie Gen Audio** (Meta, 2024) is the closed, frontier-scale foley-and-sound system that ships alongside the Movie Gen video model. It is a large diffusion-based model that generates high-fidelity, cinematic sound and music synced to video (and optionally text), at 48 kHz, designed to sound movie-grade. It is the clearest example of the "video model needs an audio model" pattern: Meta built a state-of-the-art video generator and then built a matching audio generator so the generated video would not be silent. I do not have audited public FAD numbers I would quote as precise, what I am confident about is the *positioning*, frontier-quality, video-and-text conditioned, 48 kHz, closed, and tuned for film-like results. Treat any specific score you see as approximate.

**FoleyCrafter** (Zhang et al., 2024) is an open system with a pragmatic, modular design: it takes a *pretrained text-to-audio model* (an AudioLDM-class model) and adds two adapters on top, a **semantic adapter** that conditions on the video's content and a **temporal controller** that conditions on the *timing* of detected onsets, so it cleanly separates the *what* (semantics) from the *when* (sync). That separation is a clean engineering pattern: reuse the big pretrained text-to-audio model for sound quality and bolt on a dedicated, lightweight module whose only job is timing. It is a good template if you want to build V2A on top of an existing T2A model rather than train from scratch.

**V-AURA** (Viertola et al., 2024) is an autoregressive video-to-audio system that emphasizes *high temporal alignment* via a fine-grained visual feature extractor and a high-frame-rate feature representation, an explicit bet that the path to better sync is *finer-grained, higher-rate visual features* so the model can localize events precisely. It is the codec-LM counterpart in the V2A world (most of the others are diffusion or flow), and a useful reminder that the AR-versus-diffusion split runs through video-to-audio too.

**Google's V2A** (DeepMind, announced 2024) is a closed system in the same family, video-and-optional-text to soundtrack, positioned for the same generated-video-needs-sound use case, with the same emphasis on synchronization and prompt control. As with Movie Gen Audio, I would treat specific public numbers as approximate; the confident statement is the *category*, a frontier-lab V2A model built to soundtrack generated and real video.

A compact comparison pulls the landscape together:

![A comparison matrix of AudioGen, AudioLDM and Tango, MMAudio, and Movie Gen Audio across whether they take text, whether they sync to video, whether they are open, and approximate quality](/imgs/blogs/text-to-audio-and-sound-effects-5.png)

The matrix is the buyer's map. The text-only models (**AudioGen**, **AudioLDM2/Tango**) are the mature, open, well-understood path for pure text-to-audio, you describe a sound, you get a sound, no video involved. The video-sync models (**MMAudio** open, **Movie Gen Audio** and Google's V2A closed) add the temporal-alignment machinery and the visual conditioning, at the cost of a more complex pipeline and, for the frontier ones, closed weights. The diagonal you trade along is the familiar one: openness and simplicity (text-only, open AudioGen/Tango/MMAudio) versus top-end cinematic quality (closed Movie Gen Audio). For most builders, an open model, AudioGen or Tango for text-to-audio, MMAudio or FoleyCrafter for video-to-audio, is the right starting point, and you reach for the closed frontier models only when movie-grade fidelity is the product.

The field's trajectory is worth seeing as a line:

![A timeline of sound generation from WaveNet raw-waveform synthesis through AudioGen and AudioLDM to MMAudio and Movie Gen Audio](/imgs/blogs/text-to-audio-and-sound-effects-7.png)

The progression is the same shape as the rest of generative audio: from **raw-waveform** models (WaveNet-style, slow, short) to **codec language models** (AudioGen, 2022, the tractable text-to-audio breakthrough) to **latent diffusion** (AudioLDM, Make-An-Audio, Tango, 2023, higher fidelity, easier to guide) to **video-synced foley** (MMAudio, V-AURA, FoleyCrafter, 2024, and Movie Gen Audio / Google V2A at the frontier, 2024 to 2025). Each step did not replace the last so much as add a capability: tokens made text-to-audio tractable, diffusion raised the fidelity ceiling, and video conditioning added the sync axis. The next frontier, full audio-visual *joint* generation where video and audio are produced together rather than audio-after-video, is the subject of the [video series' AV post](/blog/machine-learning/video-generation/audio-and-joint-av-generation); video-to-audio is the audio-first slice of that larger problem.

## Evaluating sound generation honestly

You cannot improve what you cannot measure, and sound-effect evaluation has its own traps. Here are the metrics that matter, what each one actually captures, and where each one lies.

**Fréchet Audio Distance (FAD)** is the FID of audio, covered in depth in [audio quality metrics](/blog/machine-learning/audio-generation/audio-quality-metrics). You embed a set of generated clips and a set of reference clips with a pretrained audio embedding model, fit a Gaussian to each set, and measure the Fréchet distance between the two Gaussians; lower is closer to the reference distribution, hence "more realistic." The critical caveat, and it is *the* honesty issue for SFX, is that **FAD depends entirely on the embedding model**. The classic FAD used a **VGGish** embedding trained on AudioSet; a VGGish-FAD and a **CLAP**-based FAD or a **PANNs**-based FAD give *different absolute numbers and sometimes different rankings*. So an FAD number is meaningless without naming the embedding and the reference set, and you cannot compare an FAD from one paper against an FAD from another unless they used the same embedding. When I quote FAD relationships below I mean "lower than the baseline on the same embedding," never an absolute.

**CLAP-score** measures *semantic match*: it is the cosine similarity in CLAP's shared text-audio space between the caption embedding and the generated audio's embedding. A high CLAP-score means the generated sound matches the caption's *meaning*, "footsteps on gravel" actually produced footsteps-on-gravel and not, say, rain. It is the text-to-audio analogue of CLIP-score for images. Its blind spot is that it measures *semantics*, not *fidelity* or *timing*: a low-quality, buzzy footsteps-on-gravel clip can score as well as a pristine one, because both are semantically "footsteps on gravel." So CLAP-score and FAD are complementary, CLAP-score for "is it the right sound," FAD for "is it a realistic sound," and you report both.

**AV-sync metrics** are the video-to-audio-specific ones, and they are the hardest to get right. The goal is to measure *temporal alignment* between the audio events and the video events. Approaches include using a pretrained audio-visual synchronization network (the kind trained on the lip-sync task, which learns to predict the temporal offset between an audio and video stream) and reporting its predicted offset or its confidence, or computing an onset-alignment score between detected audio onsets and detected visual events. These metrics are noisier and less standardized than FAD, which is part of why sync was under-measured for so long, *the metric for the thing that matters most was the least mature*. When you build V2A, invest in a sync metric early, even an imperfect one, because the generative loss, as we established, will not tell you about timing.

Here is a compact, honest table of where the families land. The numbers are deliberately *relational and approximate*, the absolute FAD depends on the embedding and reference set, and I will not fabricate precise scores:

| System | Conditioning | Sample rate | Sync metric | FAD (approx, same-embed) | Open? |
|---|---|---|---|---|---|
| AudioGen-medium | text | 16 kHz | n/a (no video) | moderate baseline | yes |
| AudioLDM2 | text (CLAP) | 16 kHz | n/a | competitive on AudioCaps | yes |
| Tango | text (FLAN-T5) | 16 kHz | n/a | strong on AudioCaps | yes |
| MMAudio | text + video | up to 44 kHz | strong (reported) | strong (low) | yes |
| FoleyCrafter | text + video | 16 kHz | good (dedicated module) | competitive | yes |
| Movie Gen Audio | text + video | 48 kHz | strong (reported) | top-tier (reported) | no |

Read that table as a *map of trade-offs*, not a leaderboard. The honest measurement protocol behind any cell would be: fix the prompt or video set, fix the random seed, name the FAD embedding (VGGish or CLAP or PANNs) and the reference distribution (usually AudioCaps for text-to-audio, the benchmark's own test set for V2A), generate a fixed number of clips per prompt, warm up the model before timing RTF, and report FAD *and* CLAP-score *and*, for V2A, a named sync metric, together. A single number in isolation is a number you should not trust.

#### Worked example: comparing two text-to-audio runs honestly

Say you are choosing between AudioGen-medium and Tango for a sound-effects feature. You take the AudioCaps test captions, generate, say, five clips per caption with a fixed seed, embed everything with the *same* CLAP model, and compute CLAP-FAD against the AudioCaps reference set and the mean CLAP-score against each caption. Suppose Tango comes out with a lower FAD (more realistic on this embedding) and a higher CLAP-score (better caption match), while AudioGen is faster (lower RTF on your RTX 4090) and generates the compositional "X and then Y" prompts a bit more reliably because of its mixing augmentation. That is a *real* decision: Tango if fidelity and semantic match are the priority, AudioGen if latency and compositional control matter more. Notice you only get a defensible answer because you fixed the seed, the prompt set, the embedding, and the reference, and you reported three numbers (FAD, CLAP-score, RTF) instead of one. If a colleague hands you "model A has FAD 2.1, model B has 3.4," your first question should be "which embedding and which reference set," and if they cannot answer, the comparison is void. I am not quoting specific FAD values here precisely *because* they are embedding-dependent; the protocol is the deliverable, not a leaderboard score.

## A practical stress test: the things that break

Theory is cheap; let me walk the failure modes I have actually hit, because they are where the understanding lives.

**The onset is smeared.** You generate a gunshot and it sounds like a soft "pop" instead of a sharp crack. The cause is almost always the *codec or the vocoder*, not the generative model: a low-bitrate codec or a vocoder optimized for speech smooths exactly the fast transients a gunshot is made of, because its training objective never rewarded sub-millisecond attack fidelity. The fix is a codec and vocoder with enough bandwidth and a training loss (a multi-resolution STFT loss, an adversarial loss with a fine-time-resolution discriminator) that preserves transients, the [GAN vocoder](/blog/machine-learning/audio-generation/gan-vocoders-hifi-gan-and-fast-synthesis) discriminators exist partly to keep onsets crisp. If your SFX onsets are mushy, suspect the codec before the model.

**The high frequencies vanish.** You ask for cymbals or hiss or a snake's rattle and the result is dull. The cause is the *sample rate*: a 16 kHz model has a Nyquist limit of 8 kHz and physically cannot represent energy above it, and a lot of the "air" and sparkle of bright sounds lives above 8 kHz. AudioGen at 16 kHz will *always* sound band-limited on bright sounds; there is no prompt that fixes a Nyquist wall. The fix is a higher-sample-rate model (24, 44.1, or 48 kHz), which is exactly why the frontier foley systems run at 48 kHz, cinematic sound needs the top octave. If your bright sounds are dull, check the sample rate before you blame the model.

**The ambience loses the thread at length.** You ask AudioGen for 30 seconds of "a busy cafe" and the first few seconds are great, then it drifts, the texture changes, a phantom voice appears, the energy wanders. The cause is the *autoregressive horizon*: with no long-range structure to anchor on, the token model accumulates error and wanders, exactly the length-fidelity tension on the series spine. The fixes are pragmatic: generate a few seconds and *loop* a stationary texture (crossfade the loop point to hide the seam), or generate in overlapping windows and crossfade, or use a diffusion model with a longer fixed context that holds the whole clip in view. For sustained ambience, do not autoregress a single long pass.

**The video-to-audio sync drifts late.** Your foley footsteps consistently land a beat *after* the visible footfall. The cause is almost always a *temporal-alignment bug*, an off-by-one in the frame-to-latent resampling, a wrong frame rate assumed (you treated a 30 fps clip as 24 fps), or a latency in the feature extractor that shifts everything. This is a *systems* bug, not a model-quality bug, and the tell is that the *drift is consistent* (every event off by the same amount), whereas a model-quality sync problem is *inconsistent* (some events sync, some do not). Consistent drift, check your rate conversion and your frame counts; inconsistent drift, the model genuinely needs better temporal features or a sync objective.

**The model adds music you did not ask for.** You request "rain on a window" and there is a faint cinematic drone or pad under it. The cause is the training data, a lot of "ambient" clips on the web have a musical bed, so the model associates ambience with light music. The fix is the negative prompt ("music, soundtrack, score" in the negative) and, if you train, cleaner data. This one surprised me the first time, the model was not wrong by its data, it was wrong by my intent.

Each of these maps to a specific part of the stack, the codec for transients, the sample rate for brightness, the AR horizon for length, the rate conversion for sync, the data for spurious music, which is the whole point of having the audio-stack frame in your head: when something is wrong, you know *which layer* to interrogate.

## Case studies: real systems and what they teach

Four named cases, with the claims I am confident about and explicit hedging on the numbers, in the series' style.

**AudioGen (Kreuk et al., Meta, 2022).** The codec-language-model breakthrough for text-to-audio. The two design lessons that endured: a single-stage transformer over EnCodec tokens conditioned on a T5 caption works for general sound the same way it works for music, and the *mixing augmentation* (concatenate two clips and their captions) teaches compositional, multi-source prompts. AudioGen targets a relatively low sample rate (16 kHz) and short clips, which is exactly its limitation for bright or long sounds. The confident takeaway is the *paradigm*: text-to-audio is a codec-LM problem, and the data augmentation matters as much as the architecture. Specific FAD figures are embedding-dependent; I quote the design, not a score.

**AudioLDM / AudioLDM2 (Liu et al., 2023).** The latent-diffusion route, compress a mel to a latent with a VAE, diffuse in latent space conditioned on CLAP, decode with a vocoder. The instructive trick is exploiting CLAP's shared space to train on audio embeddings and prompt with text embeddings, stretching the scarce paired data, and AudioLDM2's move to a *layered* text representation (CLAP plus a language model) for better compositional prompts. The lesson: latent diffusion plus a strong, audio-grounded text encoder is a robust text-to-audio recipe, and the text-encoder choice (CLAP vs LM) is the main quality lever, the same T5-versus-CLAP trade from the conditioning post.

**MMAudio (Cheng et al., 2024).** The open multimodal-joint-training V2A model. Its enduring lesson is the *data-mixing* insight: train on abundant text-audio data *and* scarcer video-audio data jointly in one flow-matching model, so each helps the other, and it pays explicit attention to synchronization. It is the open system I would start a serious video-to-audio project on today, and the one that best illustrates that the text-to-audio data scarcity and the video-to-audio data abundance are two sides of one coin you can train against jointly. The reported AV-sync and FAD are strong on the standard benchmarks; treat absolute numbers as embedding-and-benchmark-dependent.

**Meta Movie Gen Audio (2024).** The frontier, closed, 48 kHz diffusion foley-and-music model built to soundtrack the Movie Gen video model. The lesson is structural, not numerical: *a video generator needs an audio generator*, and at the frontier the two are built and shipped as a pair, because the world is increasingly full of generated, born-silent video that has to be sounded. The confident claim is the positioning, video-and-text conditioned, movie-grade, 48 kHz, closed; any specific score I would frame as approximate. The same structural lesson holds for Google's V2A: the frontier labs that build video build audio-for-video right beside it, which is the single clearest signal of where this subfield is going.

A note on honesty, again in the series' spirit. The reason I keep refusing to print precise FAD numbers is not vagueness for its own sake; it is that FAD is *embedding-dependent* and most cross-paper comparisons are not apples-to-apples, so a precise-looking number copied across systems would be *misinformation dressed as rigor*. The relationships, codec-LM versus diffusion, text-only versus video-synced, 16 kHz versus 48 kHz, open versus closed, are the durable, defensible content; the exact scores are benchmark artifacts you must reproduce yourself under a named protocol.

## When to reach for text-to-audio, and when not to

A decisive section, because the failure mode I see most is reaching for the wrong tool for the sound you actually need.

**Reach for text-to-audio** (AudioGen, AudioLDM2, Tango) when you want a *specific sound described in words*, with no requirement that it sync to anything, a sound-effect library, a notification chime, a placeholder ambience, a "give me a thunderclap" request. It is mature, open, fast enough to be interactive for short clips, and the right default for pure sound generation. Prefer **Tango** (LLM text encoder) when your prompts are compositional and ordered ("A, then B"); prefer **AudioGen** when latency matters and for multi-source mixtures; prefer **AudioLDM2** when you want diffusion's full-context coherence and easy CFG control. Do *not* reach for text-to-audio when the sound must *line up with a video*, it has no concept of the video's timing and will place events wherever it likes.

**Reach for video-to-audio** (MMAudio, FoleyCrafter for open; Movie Gen Audio, Google V2A for frontier) when the sound must be *synced to visible action*, soundtracking generated or silent video, foley for film and games, dubbing archival footage. Prefer the **open** systems (MMAudio, FoleyCrafter) for most work, they are strong and you control them; reach for the **closed frontier** models only when 48 kHz cinematic fidelity is the deliverable and you can live with a closed API. Do *not* reach for video-to-audio when you do not actually need sync, if you just want "some rain sound" and there is no specific visual timing to match, a text-to-audio model is simpler, faster, and entirely sufficient; the whole sync apparatus is dead weight you do not need.

**Do not autoregress a long ambience in one pass.** For sustained, stationary textures (rain, crowd, wind), generate a few seconds and loop-with-crossfade or stitch overlapping windows; a 30-second single AR generation will drift. The exception is a model with a long fixed context (a diffusion model that holds the whole clip), which can do length in one shot, at higher compute.

**Do not fight a Nyquist wall with prompts.** If you need bright, airy sounds (cymbals, sparkle, hiss, high detail), start with a high-sample-rate model (24, 44.1, or 48 kHz); no prompt engineering recovers energy a 16 kHz model physically cannot represent.

**Do not trust a single metric.** Report FAD (with a *named* embedding and reference set) *and* CLAP-score *and*, for video-to-audio, a named sync metric, together, with a fixed seed and prompt or video set; a lone FAD is a number you cannot defend.

## Key takeaways

- **Sound effects are a third, distinct audio domain.** Unlike speech (linguistic structure) and music (tonal/rhythmic structure), SFX have almost no long-range structure; they live or die on the **temporal envelope** (especially the onset) and the **timbre** over a short window, a different modeling emphasis that rewards local spectro-temporal accuracy over long-horizon planning.
- **Text-to-audio is a codec-LM or latent-diffusion problem.** **AudioGen** is the codec-language-model route (the sound-effect counterpart of MusicGen, a T5-conditioned transformer over EnCodec tokens). **AudioLDM / Make-An-Audio / Tango** are the latent-diffusion route, with a **CLAP** or LLM text encoder; the text-encoder choice is the main quality lever.
- **The data is the bottleneck.** Paired audio-text data is scarce and noisy, AudioSet is *labeled* not *captioned*, AudioCaps is the small captioned benchmark, so the field *manufactures* captions with lossy pipelines (tag templating, video-metadata mining, audio-captioning models, LLM rewriting); judge a model's "data scale" with that in mind.
- **Video-to-audio (foley) trades a data advantage for a sync problem.** Every video on the web is a free, perfectly-synced (silent-video, audio) pair, an enormous data advantage over text-to-audio, but it adds the hard requirement of **temporal synchronization**.
- **Sync is hard for three concrete reasons:** a literal **rate mismatch** (24 fps video versus a ~50 Hz audio latent versus a ~24 kHz waveform, reconciled by *resampling visual features onto the audio grid*); the **loss barely sees timing** (a misplaced 50 ms onset is a rounding error in the average loss but perceptually fatal); and the model needs **motion-aware, time-localized visual features**, not a single pooled embedding.
- **The sync budget is about one video frame (~40 ms at 24 fps).** Off by a frame is borderline; off by 100 ms reads as dubbed. Because the generative loss is nearly indifferent to this, you need AV-sync-aware training, a sync objective, a sync discriminator, or sync re-ranking, not just reconstruction loss.
- **The systems differ on the levers they pull:** MMAudio (joint text-audio + video-audio training, flow matching, open), FoleyCrafter (a pretrained T2A model plus separate semantic and *temporal* adapters), V-AURA (AR with fine-grained high-rate visual features), Movie Gen Audio and Google V2A (closed, frontier, 48 kHz, video-needs-audio).
- **Evaluate with named protocols.** FAD is **embedding-dependent** (VGGish vs CLAP vs PANNs give different numbers and rankings), CLAP-score measures *semantics* not fidelity or timing, and AV-sync metrics are the least mature; report all relevant ones together with a fixed seed and reference set, and never trust a lone FAD.
- **Match the tool to the sound.** Text-to-audio for described, un-synced sounds; video-to-audio only when sync to visuals is required; high sample rate for bright sounds; loop-or-stitch (not one long AR pass) for sustained ambience; the negative prompt to suppress spurious music.

## Further reading

- **AudioGen** — Kreuk, Synnaeve, Polyak, et al., "AudioGen: Textually Guided Audio Generation," 2022. The codec-language-model approach to text-to-audio and the mixing augmentation for compositional prompts.
- **AudioLDM / AudioLDM 2** — Liu, Chen, Yuan, et al., "AudioLDM: Text-to-Audio Generation with Latent Diffusion Models," 2023, and "AudioLDM 2," 2023. Latent diffusion with CLAP and a layered text representation.
- **Make-An-Audio / Tango** — Huang et al., "Make-An-Audio," 2023, and Ghosal, Majumder, Mehrish, et al., "Text-to-Audio Generation using Instruction-Tuned LLM and Latent Diffusion (Tango)," 2023. The data-augmentation pipeline and the LLM-text-encoder bet.
- **AudioCaps / AudioSet / Clotho** — Kim et al., "AudioCaps," 2019; Gemmeke et al., "AudioSet," 2017; Drossos et al., "Clotho," 2020. The captioning and labeling resources behind the whole subfield, and the scarcity at its core.
- **MMAudio** — Cheng, Ishii, Hayakawa, et al., "Taming Multimodal Joint Training for High-Quality Video-to-Audio Synthesis," 2024. The joint text-audio and video-audio training recipe with strong sync.
- **FoleyCrafter / V-AURA** — Zhang et al., "FoleyCrafter," 2024 (semantic + temporal adapters on a pretrained T2A model), and Viertola, Iashin, Rahtu, "V-AURA," 2024 (autoregressive V2A with fine-grained visual features).
- **Meta Movie Gen Audio** — Meta, "Movie Gen: A Cast of Media Foundation Models," 2024. The frontier 48 kHz video-and-text-conditioned foley-and-music model.
- **Within this series:** the foundation [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard); the codec [neural audio codecs](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound) and [residual vector quantization](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq); the engines [diffusion for audio](/blog/machine-learning/audio-generation/diffusion-for-audio), [flow matching for audio](/blog/machine-learning/audio-generation/flow-matching-and-consistency-for-audio), and [GAN vocoders](/blog/machine-learning/audio-generation/gan-vocoders-hifi-gan-and-fast-synthesis); the steering [conditioning and control](/blog/machine-learning/audio-generation/conditioning-and-control-in-audio-generation) (CLAP); the sibling [MusicLM and MusicGen](/blog/machine-learning/audio-generation/music-generation-musiclm-and-musicgen); the measurement [audio quality metrics](/blog/machine-learning/audio-generation/audio-quality-metrics); the forward links [the 2026 audio model landscape](/blog/machine-learning/audio-generation/the-2026-audio-model-landscape) and the capstone [building an audio generation stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack). Out to the video series: [audio and joint AV generation](/blog/machine-learning/video-generation/audio-and-joint-av-generation).
