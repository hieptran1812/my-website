---
title: "Suno, Udio, and the Commercial Music Frontier"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "A report on the closed commercial models that generate full songs with vocals and lyrics — what Suno and Udio do that open models cannot, the inferred recipe behind them, the lyrics-to-vocals alignment problem, and the copyright fight that follows."
tags:
  [
    "audio-generation",
    "audio-synthesis",
    "music-generation",
    "suno",
    "udio",
    "neural-audio-codec",
    "generative-ai",
    "deep-learning",
    "ai-copyright",
  ]
category: "machine-learning"
subcategory: "Audio Generation"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/suno-udio-and-the-commercial-music-frontier-1.png"
---

The first time I typed a few lines of lyrics into Suno and got back a two-minute pop song — verse, pre-chorus, a chorus that actually hooked, a voice that pronounced *my* words and stayed in key over a guitar arrangement I never asked for in detail — I sat there a little stunned. Not because it was perfect; it wasn't. The mix was slightly glassy, the consonants smeared in a couple of spots, and if you listened past the fourth bar of the bridge you could hear the model lose a little conviction. But I had spent the better part of two years getting open models to produce thirty seconds of *instrumental* loop that didn't drift out of time, and here was a closed product handing me a structured, sung, multi-minute **song** from a text box. The gap between what I could build with [MusicGen](/blog/machine-learning/audio-generation/music-generation-musiclm-and-musicgen) and what Suno shipped was not a quality gap on a ten-second clip. It was a *capability* gap: full songs, with vocals, that sing user-supplied lyrics, held together over minutes with real verse-chorus-bridge structure.

This post is a **report** on that frontier — Suno and Udio, the two closed commercial models that define it. I want to be honest up front about epistemics: these are closed models. Neither company has published a full technical report the way Meta did for [EnCodec](/blog/machine-learning/audio-generation/encodec-dac-and-the-modern-codec) or MusicGen. So everything here splits into two buckets, and I will keep them separate the whole way through: what is **disclosed** (product behavior, version notes, public statements, the court filings) versus what is **inferred** (the likely recipe, reasoned from the capabilities and from the mechanisms we have already built up in this series). When I say "Suno probably does X," I mean *the public behavior is most cheaply explained by X given what we know works* — not that I have inside knowledge. I don't.

By the end you will be able to reason from "full song + vocals + lyrics + structure over minutes" to the mechanisms that almost certainly have to be in there — a high-compression neural codec for length, a lyrics-and-phoneme-aligned conditioning path for the singing, section/structure conditioning for the arrangement, and heavy preference tuning for musicality — and you will be able to build a crude open **stand-in** that wires those components together so the pieces are concrete rather than magic. We will also walk through the lyrics-to-sung-vocals alignment problem (which is genuinely harder than text-to-speech), the product surface (extend, cover, stems, personas), and the copyright fight — the RIAA lawsuits and the "trained on copyrighted recordings" question — because that fight is now the single biggest force shaping where this technology goes. This all sits in the recurring frame of the series: the **audio stack** (waveform → neural-codec tokens / mel latent → generative model → vocoder/decoder → waveform) under the tension **fidelity × controllability × speed × length**. Suno and Udio push hard on the *length* and *controllability* axes in a way the open models have not matched.

![A vertical stack diagram of the inferred full-song pipeline showing a lyrics and style prompt feeding a conditioner, a large token model, a high-ratio codec, decoding, preference tuning, and a finished song](/imgs/blogs/suno-udio-and-the-commercial-music-frontier-1.png)

The figure above is the whole argument in one picture, and it is *inferred*, not disclosed — hold that caveat. A lyrics-plus-style prompt enters a conditioning path; a large token model generates a long sequence; a high-ratio codec decodes those tokens back to a 44.1 kHz waveform; preference tuning shapes the musicality; out comes a full song. Every box in that stack is a thing we have already studied in this series in some other form. The rest of this report is about which form, and why that arrangement is the cheapest explanation for what these products actually do.

## 1. What Suno and Udio actually do (the disclosed capabilities)

Let me start with what is *not* speculative: the observable product behavior. This is the disclosed layer, and it is worth being precise about because the inferred recipe later has to *explain exactly this list* and nothing fancier.

You type, at minimum, two things: a **style/description prompt** ("upbeat indie folk, female vocals, acoustic guitar, 120 BPM") and **lyrics** (either your own text, or you let the model write them). The product returns a **full song** — typically two clips per generation so you can pick — that is on the order of **two to four minutes** long, with:

- **Sung vocals that pronounce your lyrics.** This is the headline. The model does not just generate music; it generates a *voice singing specific words*, and those words are (mostly) intelligible and (mostly) in time with the music.
- **Song structure.** Intro, verse, chorus, sometimes a pre-chorus and a bridge and an outro. The chorus often *recurs* with the same melody and (roughly) the same arrangement, which means the model is holding a structural plan across the whole length, not just locally continuing.
- **Multiple genres and styles.** Pop, hip-hop, metal, lo-fi, EDM, country, orchestral, chiptune — the same model covers an enormous stylistic range from the text prompt alone.
- **Continuation and extension.** You can take a generated clip (or, in some product tiers, an uploaded clip) and **extend** it — the model continues from where it left off, keeping key, tempo, and arrangement. This is the explicit "give me another minute" button, and it is a strong hint about how length is achieved (more on this in section 6).
- **Covers and remixes.** Suno's "cover" feature re-renders an existing piece of audio in a new style; "remix" / "replace section" lets you regenerate part of a song.
- **Stems.** Both products can return separated **stems** — vocals, drums, bass, other — so you can remix in a DAW. That stems exist at all is interesting: it suggests either a model that natively generates separable components, or a source-separation model bolted on at the end (the second is far more likely and cheaper, and we will treat it that way).

The version history is also disclosed, at least at the marketing level, and it tracks a clear quality arc. **Suno** went **v3 → v3.5 → v4 → v4.5 (and v4.5+)**. v3 (early 2024) was the version that made the internet notice — coherent two-minute songs with vocals, but with a characteristic "Suno sheen," a slightly metallic, lossy-sounding high end and smeared sibilants that, once you heard it, you heard everywhere. v4 (late 2024) cleaned up the mix substantially — less of that codec haze, tighter low end, more believable vocals — and improved structural coherence. v4.5 (2025) pushed length, genre range, and vocal expressiveness further, and added more control over the arrangement. **Udio** launched (April 2024) close on Suno's heels with a notably *cleaner, more hi-fi* vocal sound out of the gate — many listeners felt Udio's vocals were more natural and its production more "finished," while Suno was faster and more prolific — and iterated through **v1 → v1.5 → v2**, with similar gains in fidelity, length, and control.

![A capability matrix comparing Suno, Udio, MusicGen, and Stable Audio across sung lyrics, song length, structure, stems, and open weights](/imgs/blogs/suno-udio-and-the-commercial-music-frontier-2.png)

The matrix above lines up the two commercial models against the two open workhorses we covered earlier in this track — MusicGen and [Stable Audio](/blog/machine-learning/audio-generation/latent-diffusion-for-music-stable-audio) — on the capabilities that matter. The crucial column is the first one: **sings lyrics?** MusicGen and Stable Audio are, at their core, **instrumental** generators. MusicGen produces a backing track from a text or melody prompt; Stable Audio produces high-fidelity audio (music and sound) up to a few minutes, conditioned on text and timing. Neither has a path that takes *your specific words* and *sings them*. That is the line that separates the commercial frontier from the open one, and it is not a line you cross by training the open models harder. It requires a different conditioning architecture. Everything in this report is, ultimately, about that one column.

One more disclosed fact, because it matters for the rest of the post: **both companies have been sued.** In June 2024 the major record labels, through the RIAA (UMG, Sony Music, Warner), filed copyright-infringement suits against Suno and Udio, alleging the models were trained on massive quantities of copyrighted sound recordings without license. That is disclosed — it is in the public court record. What the models were *actually* trained on is disputed and, as of this writing, unresolved. We will come back to it in section 8, because it is no longer a side issue; it is arguably the dominant force on this part of the field.

## 2. Why the open models can't do this (and what "this" really is)

Before we reason toward the recipe, I want to be very precise about *what capability* we are trying to explain, because it is easy to wave at "they're just better" and miss the actual hard part.

Take MusicGen. It is a single-stage autoregressive transformer over EnCodec tokens, with a clever codebook-interleaving pattern (the delay pattern) so one transformer can predict the four RVQ codebooks per frame. It is genuinely good at instrumental music. Give it "70s funk, wah guitar, tight drums" and it gives you a convincing groove. But three things cap it relative to a Suno song, and they are *structural*, not quality dials:

**First, no vocal-lyric path.** MusicGen has no input for "sing *these words*." Its conditioning is a text description (via a T5 encoder) and optionally a melody (via a chromagram). There is nowhere to put lyrics that the model would learn to *pronounce*. You can prompt "with vocals" and it will hum some vocal-like texture, but it is babble — it does not say words, because it was never trained on a lyrics-aligned objective. Adding singing is not a bigger-prompt problem; it is a new conditioning channel and a new kind of training data (audio paired with *time-aligned lyrics*).

**Second, length.** MusicGen generates around 30 seconds comfortably; longer generations drift — they lose the beat, wander out of key, forget the motif. The default open configurations simply weren't built to hold a four-minute structure. Stable Audio does better on length (it generates up to a few minutes via latent diffusion with timing conditioning) but still has no lyric path.

**Third, structure.** Neither open model has a strong notion of "now play the chorus *again*, the same one from before." They continue locally. A Suno song that brings the chorus back at 2:10 with the same melody it had at 0:45 is doing something the open models don't: holding a *global structural plan* and re-realizing a section.

![A before-and-after comparison showing an open model producing a short instrumental clip versus a commercial model producing a structured multi-minute song with sung lyrics](/imgs/blogs/suno-udio-and-the-commercial-music-frontier-3.png)

So "this" — the capability we are reasoning toward — is the conjunction of three things: **(1) sung vocals that pronounce specific lyrics, (2) coherence over multiple minutes, and (3) verse-chorus-bridge structure with recurring sections.** The commercial gap is that *conjunction* of vocals, length, and structure, and you cannot get there by scaling MusicGen's instrumental objective. You need a different machine. Let's reason out what that machine most likely is.

It is worth dwelling on point (3), the *recurring* structure, because it is the most underappreciated and the most diagnostic. When a song brings its chorus back at 2:10 with the *same* melody and roughly the same arrangement it had at 0:45, the model is doing something a local continuer cannot: it is realizing a **global plan** in which two distant spans of audio are *the same section*. A pure left-to-right model with a short window cannot do this — by 2:10 the first chorus is far out of context. So either the context is genuinely long (the whole song is in the window), or there is an explicit structural representation (a section plan the model conditions on at both 0:45 and 2:10), or both. The products' acceptance of `[Chorus]` tags in the lyric box is the public tell that an explicit structural representation exists. This is the strongest evidence that these are not "really good continuers" but planned generators, and it is the capability the open models most conspicuously lack.

## 3. Reasoning to the recipe (inferred, not disclosed)

Here is the core analytical move of this report. We do **not** know Suno's or Udio's architecture. But we know a great deal about *what mechanisms produce each of the observed capabilities*, because we built those mechanisms up over the previous posts in this series. So we can do something disciplined: for each disclosed capability, ask "what is the cheapest known mechanism that produces it?" and assemble the answer. The result is a *hypothesis*, clearly labeled, that is consistent with everything public.

Let me reason capability by capability.

**Capability: multi-minute length at high fidelity.** Audio at 44.1 kHz is ~44,100 samples per second. A four-minute song is ~10.5 million samples. You cannot autoregress over raw samples for minutes — that is the lesson from [WaveNet to AudioLM](/blog/machine-learning/audio-generation/autoregressive-audio-models-wavenet-to-audiolm). You need a **high-compression neural codec** that turns the waveform into a *short* sequence of discrete tokens. EnCodec at 50 fps with 4 codebooks is 200 tokens/second — four minutes is ~48,000 tokens per codebook, which is a lot. So the cheapest explanation for *length* is a codec with an even **higher compression ratio** than vanilla EnCodec — fewer frames per second, or a smarter token layout — so that minutes of audio fit in a context the model can actually attend over. This is the single most load-bearing inference in the whole report: **length forces a high-ratio codec.** Everything else hangs off having short token sequences.

**Capability: sung vocals pronouncing lyrics.** This requires a **lyrics conditioning path** that the model learns to *realize as pronounced, sung phonemes*. The cheapest known mechanism is the one TTS uses: convert lyrics to **phonemes** (grapheme-to-phoneme, g2p) and condition the generative model on that phoneme sequence, trained on audio paired with *time-aligned* lyrics so the model learns to place the right phoneme at the right musical moment. This is harder than TTS (section 4 is entirely about why), but it is the same *family* of mechanism. So: **a phoneme/lyrics conditioning channel, trained on lyric-aligned audio.**

**Capability: style/genre control from a text prompt.** This is exactly the text-conditioning machinery from [conditioning and control in audio generation](/blog/machine-learning/audio-generation/conditioning-and-control-in-audio-generation): a text encoder (T5-style, or a CLAP-style joint text-audio embedding) produces a conditioning vector that the generative model cross-attends to. We know this works because MusicGen and Stable Audio both do it. So: **a text/style conditioning channel**, almost certainly cross-attention over a frozen or jointly-trained text encoder.

**Capability: structure (verse/chorus/bridge, recurring sections).** The cheapest mechanism is **section conditioning** — the lyrics input is itself *structured* (Suno's lyric box accepts `[Verse]`, `[Chorus]`, `[Bridge]` tags, which is a strong public hint), so the model receives explicit section markers it learned to associate with arrangement changes and melodic recurrence. So: **structural tags in the conditioning**, learned from a corpus where songs are labeled by section.

**Capability: "it sounds like a hit, not like math."** This is taste, and taste does not fall out of a likelihood objective. The cheapest known mechanism for "make outputs that humans *prefer*" is **preference tuning** — RLHF / DPO-style fine-tuning on human ratings of which generation sounds better. This is exactly the post-training that turned base LLMs into chat assistants, and it is the most plausible explanation for the dramatic *musicality* jump from v3 to v4 that is much larger than you would expect from just more data. So: **a preference-tuning stage on human ratings.**

Now assemble it. The inferred recipe is: **a high-compression neural audio codec + a large generative model over its tokens (an AR language model or a diffusion model — we'll weigh both) + a lyrics/phoneme + style + structure conditioning path + heavy data + preference tuning.** That is figure 1, the stack at the top of this post. I want to stress one more time: this is the *cheapest hypothesis consistent with public behavior*, built entirely from mechanisms we have already validated in this series. It is not a leak. The companies may do something cleverer or simpler in places. But if you handed this spec to a competent audio team with enough compute and data, they could plausibly reproduce most of what Suno v3 did — which is, itself, a strong sign the hypothesis is roughly right.

### AR language model or diffusion?

The one box I left ambiguous is the generative model itself. Both families can produce what we see, and the public signal is genuinely mixed, so let me lay out the case for each rather than pretend certainty.

The **autoregressive-LM-over-codec-tokens** case (the [AudioLM](/blog/machine-learning/audio-generation/autoregressive-audio-models-wavenet-to-audiolm) / MusicGen / VALL-E lineage): AR models are excellent at *sequential structure* and *continuation*, which fits the strong "extend" feature perfectly — extension is literally "keep generating from these tokens," which is native to an AR model and awkward for a vanilla diffusion model. AR over a high-ratio codec is the most direct path to "song as a sequence of sound-words." The characteristic v3 "codec sheen" is also consistent with an RVQ-codec-token model: you are hearing the codec's reconstruction limit. I lean slightly toward AR-over-codec-tokens as the *primary engine* for at least one of these products, mostly because of how natural extend/continue is.

The **latent-diffusion** case (the Stable Audio / [diffusion-for-audio](/blog/machine-learning/audio-generation/diffusion-for-audio) lineage): diffusion over a continuous codec latent gives you very clean high fidelity and natural variable-length generation via timing conditioning, and the "cleaner, more hi-fi" character some attribute to Udio is consistent with a diffusion or flow approach over a continuous latent rather than discrete RVQ tokens. Diffusion also handles *global* structure gracefully because it denoises the whole clip at once rather than committing left-to-right.

It is entirely possible the two products made *different* choices here — and that the cleanest reading of "Udio sounds more hi-fi, Suno extends more naturally" is exactly that one leaned diffusion-ish and the other AR-ish. We cannot resolve it from outside. What we *can* say with confidence is the surrounding scaffolding — codec, lyric path, style conditioning, structure tags, preference tuning — because that scaffolding is forced by the capabilities almost regardless of which generative core sits in the middle.

There is also a real possibility that the architecture is **hybrid** — a coarse-to-fine arrangement that the audio field has converged on repeatedly. AudioLM itself is hierarchical: a semantic-token stage that captures the *what* (the long-range structure, the melody, the phonetic content) feeding an acoustic-token stage that captures the *how* (the timbre, the fine detail). MusicGen collapsed that into one stage; VALL-E split TTS into an AR stage for the first codebook and a non-AR stage for the rest. It would be unsurprising — even expected — if Suno or Udio used a coarse model (AR or diffusion) to lay down the *structure and melody and lyric timing* at a low token rate, and a second model (a codec-token model, a diffusion decoder, or a vocoder) to fill in the *high-fidelity detail*. This coarse-to-fine split is exactly how the field has solved "long-range coherence *and* fine detail" everywhere else, and a four-minute sung song needs both at once. If I had to bet on the single most likely overall shape, it would be: a high-ratio codec, a coarse structure/melody/lyric model, a fine acoustic model, and preference tuning over the whole thing — a song-scale version of the same hierarchy AudioLM introduced for general audio.

### The data pipeline you cannot see (but can reason about)

The recipe is not just the model — it is the *data pipeline that feeds it*, and that pipeline is where most of the real work (and the legal exposure) lives. Reasoning from the capabilities, the training corpus has to provide, per song, at least: the **audio** (the mixed waveform, and ideally separated stems), the **lyrics** (the text), a **time alignment** between the two (so the model learns which words land where), the **structure labels** (where the verse ends and the chorus begins), and **metadata** (genre, tempo, key, mood) for the style-conditioning channel. None of that is free. Lyrics have to be transcribed or scraped and then *aligned* to the audio — typically with a forced-alignment model (a Whisper-style ASR run in alignment mode, or a dedicated lyric-aligner) that produces word- or line-level timestamps. Structure has to be detected (or labeled), key and tempo estimated (with an off-the-shelf MIR tool), and the whole thing cleaned, deduplicated, and filtered for quality. Stems, if used, come from a source-separation model run over the mixes.

The reason this matters for the report is that **the quality of the alignment and the labels caps the model's musicality** just as much as the architecture does. A model trained on poorly-aligned lyrics will smear its consonants and drift off the beat — exactly the v3-era artifacts that v4 cleaned up, which is consistent with a story where the v3→v4 jump was as much about *better data and alignment* as about a bigger model. And it matters for the controversy because every stage of that pipeline — the audio, the lyrics, the alignment derived from copyrighted recordings — touches material the labels claim rights over. You cannot build the pipeline that produces the capability without ingesting the material the lawsuits are about. The data pipeline *is* the moat, and it is also the liability.

## 4. The lyrics-to-sung-vocals problem (the genuinely hard part)

If you take one piece of *science* away from this report, make it this section. The reason singing is harder than speech is specific and worth formalizing, because it is the part of the recipe with the least precedent and the most engineering risk.

In text-to-speech, the job is: given a phoneme sequence, produce audio that pronounces those phonemes intelligibly with natural-sounding prosody. The system has *latitude* over timing and pitch — there is no single correct duration for the word "fox," and the listener accepts a wide band of natural rhythms and intonations. The duration predictor in FastSpeech, or the implicit alignment in [VALL-E](/blog/machine-learning/audio-generation/neural-codec-language-model-tts-vall-e), just has to land somewhere in that acceptable band.

Singing removes the latitude. In a song, the phonemes are **bound to a melody** — a sequence of pitches with specific durations dictated by the rhythm of the music. The word "fox" might be held for a dotted half note at a specific pitch, or chopped into a sixteenth-note triplet, and there is a *right answer* relative to the backing track, because the vocal has to land *on the beat* and *in the key*. So the problem is not just "pronounce the phonemes"; it is "**align each phoneme to a melodic note — a (pitch, onset, duration) triple — such that the sung result is in time with the instrumental and in the right key.**"

Let me write that down. Given lyrics that g2p into a phoneme sequence $p_1, p_2, \ldots, p_N$, and a melody that is a sequence of notes $(\text{pitch}_k, t^{\text{on}}_k, t^{\text{dur}}_k)$, singing-voice generation must learn an **alignment** $a : \{1,\ldots,N\} \to \text{notes}$ assigning phonemes to notes (a syllable may span one note, a melisma may span several, a fast lyric may pack several phonemes into one note), and then render audio whose fundamental frequency $f_0(t)$ tracks the assigned pitch over each note's duration while the spectral envelope realizes the assigned phoneme. The loss the model is implicitly minimizing has *two* terms a TTS model does not face together: a **pitch term** ($f_0(t)$ must match the melody, so the singing is in tune) and a **timing term** ($t^{\text{on}}$ and $t^{\text{dur}}$ must match the rhythm, so the singing is on beat) — both **conditioned on the instrumental** so the vocal sits *with* the band rather than beside it. TTS has neither hard constraint; a sentence has no key and no metronome.

![A dataflow graph showing lyrics and a style prompt splitting into a phoneme path with melody alignment and an instrumental path, merging into joint codec tokens that produce a mixed song](/imgs/blogs/suno-udio-and-the-commercial-music-frontier-4.png)

The graph above is the data flow this implies, and it is the *crux* of why the open instrumental-only systems can't be retrofitted into Suno. Lyrics and style split into two branches: a **phoneme branch** that must be aligned to a melody (the caution-colored node — this is the hard part), and an **instrumental branch**. Both branches have to merge into a **single joint token stream** so that the vocal and the backing are generated *together* and *coherently* — same key, same tempo, sitting in the same mix. If you generate them independently and glue them, they fight: the vocal drifts off the beat, or sits in a slightly wrong key, or sounds pasted on. The reason a commercial song sounds *produced* rather than *assembled* is almost certainly that the vocal and instrumental share representation in the generative model — they are conditioned on each other, not concatenated after the fact.

This is also why the **training data** problem is acute and why the copyright fight has teeth. To learn the alignment $a$ and the pitch/timing terms, you need a very large corpus of **songs paired with time-aligned lyrics** — ideally with phoneme-level or at least line-level timing. That data is hard to get cleanly. Karaoke datasets, transcribed-and-aligned commercial recordings, lyric databases with timestamps — the obvious sources are exactly the sources that raise the "trained on copyrighted recordings" question. The capability and the controversy come from the same place: you cannot learn to sing real lyrics over real arrangements without a mountain of real songs with their words attached.

### How the alignment is actually learned

There is a deeper point worth making about *how* a model would learn the alignment $a$ without anyone hand-labeling which phoneme lands on which note. The honest answer is that you almost certainly do not supply explicit per-phoneme timing at all — you let the model learn a **soft, monotonic alignment** the way TTS systems do, and the singing constraint makes the monotonicity matter more, not less.

Two mechanisms from the TTS literature carry directly over. The first is **monotonic alignment search**, the trick VITS and Glow-TTS use: the alignment between the phoneme sequence and the audio frames is constrained to be *monotonic* (you sing the words in order — you never sing the second syllable before the first) and *surjective onto frames* (every audio frame is "covered" by exactly one phoneme), and the model searches over the space of such alignments for the most likely one during training. That monotonicity constraint is the single most important inductive bias, because it collapses an astronomically large alignment space into something learnable: an $N$-phoneme line over $T$ frames has only the monotonic non-decreasing assignments to consider, not all $N^T$ arbitrary maps. The second is the **CTC-style** marginalization, where instead of committing to one alignment you sum the likelihood over *all* valid monotonic alignments, which gives a smoother training signal. Either way, the model is never told "phoneme 7 is at frame 412"; it is told "these phonemes, this audio, and they line up monotonically," and it infers the rest.

What singing *adds* on top of the TTS alignment problem is the **pitch supervision**. In TTS the fundamental frequency $f_0$ is a free, model-chosen prosodic variable — the duration predictor picks durations and the model picks a natural intonation contour. In singing, $f_0$ is *supervised by the melody*: the model is conditioned on (or must infer) the note pitches, and the rendered $f_0(t)$ has a *target* it is graded against. Concretely, a singing-aware loss adds a term like $\mathcal{L}_{f_0} = \frac{1}{T}\sum_t \big(\log f_0(t) - \log f_0^{\text{target}}(t)\big)^2$ — a log-frequency error so that being a semitone flat costs the same everywhere on the keyboard (pitch perception is logarithmic). That single extra term, absent from TTS, is what forces the vocal to stay in tune. Whether Suno/Udio expose $f_0$ supervision explicitly or let the codec tokens carry pitch implicitly, *something* has to grade pitch against the music, or the singing would wander — and it audibly does not. This is the kind of mechanism you can be confident is *somewhere* in there even without disclosure, because the alternative (a model that magically stays in key with no pitch grounding) is not how these systems are known to work.

### The instrumental-conditioned vocal: the part with the least precedent

The hardest piece — the one with the *least* published precedent — is that the vocal must be generated *conditioned on the instrumental*, not in a vacuum. A standalone singing-voice synthesizer (DiffSinger, for instance) takes a musical score and renders a clean vocal in isolation; mixing it with a backing track is a separate, later step. Suno and Udio do not sound like that. Their vocals *react* to the arrangement — they sit in the pocket of the groove, they breathe where the music breathes, they ride the dynamics of the chorus. The cheapest explanation is that the generative model produces the vocal and the backing in the **same forward pass over a shared token stream**, so the vocal is conditioned on the band and the band leaves room for the vocal, jointly. This is genuinely beyond what any open pipeline does well, and it is the strongest part of the commercial moat. It is also the part most dependent on training data that *is* full mixed songs (where the vocal and band already sit together) rather than isolated stems — another thread connecting the capability to the data-provenance question.

#### Worked example: why singing needs the pitch term that TTS skips

Take the line "*hold on*" set to two notes: "hold" on an A3 (220 Hz) held for one beat, "on" rising to a C4 (~262 Hz) held for two beats, at 90 BPM (so a beat is 0.667 s). A TTS model asked to *say* "hold on" is free to pick any natural duration and a falling, declarative intonation — and it would sound *wrong* as singing precisely because it would not hold "on" for 1.33 s at 262 Hz over the chord. The singing model must place $f_0$ at 220 Hz for 0.667 s, then glide to 262 Hz and hold it for 1.33 s, with the vowel of "on" sustained the whole time, *while the backing track plays the chord underneath at the same tempo*. Get the pitch wrong by a semitone and it is audibly flat against the chord. Get the timing wrong by an eighth note and it drags behind the beat. TTS has no equivalent failure mode — there is no chord to be flat against. That extra (pitch, timing, in-context-with-the-band) constraint is the entire difficulty, and it is why I would expect a singing model to need far more careful data and conditioning than a comparable TTS model.

## 5. An open stand-in: building the components yourself

Suno and Udio are closed APIs — you cannot inspect them and you cannot run them locally. But you *can* build a crude open stand-in that wires the same components together, which is the best way to turn the inferred recipe from a diagram into something concrete. The stand-in will be worse — much worse — than the commercial products, and that gap is itself instructive: it shows you exactly which parts the commercial teams got right that an off-the-shelf assembly does not.

![A before-and-after diagram contrasting a thirty-second instrumental clip from an open model against a multi-minute commercial song with sung lyrics and held structure](/imgs/blogs/suno-udio-and-the-commercial-music-frontier-5.png)

The plan, shown above as the gap we are trying to close: generate a **backing track** with an open music model, generate a **vocal stem** with an open singing/TTS model, align them to a shared tempo grid, and mix. This is the "generate the parts separately and glue them" approach I just warned against — and doing it yourself is the fastest way to *feel* why the commercial models generate jointly instead.

First, the conceptual pipeline as code — this is the *shape* of the commercial system, written so the data flow is explicit. It is intentionally a sketch (the `JointSongModel` does not exist as an open checkpoint), to make the inferred architecture legible.

```python
# Conceptual pipeline (the INFERRED commercial shape — not a real checkpoint).
# Lyrics + style -> conditioning -> token model -> codec decode -> song.
from dataclasses import dataclass

@dataclass
class SongRequest:
    lyrics: str          # may contain [Verse] / [Chorus] / [Bridge] tags
    style: str           # "upbeat indie folk, female vocals, 120 BPM"
    duration_s: float = 180.0

def g2p(text: str) -> list[str]:
    # grapheme-to-phoneme; real systems use phonemizer / g2p_en
    ...

def conceptual_generate(req: SongRequest):
    phonemes = g2p(req.lyrics)                 # lyric path
    sections = parse_section_tags(req.lyrics)  # [Verse]/[Chorus] structure
    style_emb = text_encoder(req.style)        # T5/CLAP-style conditioning
    # A SINGLE model over high-ratio codec tokens, conditioned on all three.
    tokens = JointSongModel.generate(
        phonemes=phonemes, sections=sections, style=style_emb,
        n_tokens=int(req.duration_s * CODEC_FPS),  # high-ratio codec -> few fps
    )
    wav = codec.decode(tokens)                 # tokens -> 44.1 kHz waveform
    return wav                                  # vocals + backing, mixed jointly
```

That is the target. Now the *runnable* open approximation, which substitutes real open checkpoints for the parts that don't exist as open weights. Start with the backing track from MusicGen via 🤗 `transformers`:

```python
# Open stand-in, part 1: an instrumental backing track with MusicGen.
import torch, soundfile as sf
from transformers import AutoProcessor, MusicgenForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
model = MusicgenForConditionalGeneration.from_pretrained(
    "facebook/musicgen-medium"
).to(device)

style = "upbeat indie folk, acoustic guitar and light drums, 120 BPM, no vocals"
inputs = processor(text=[style], padding=True, return_tensors="pt").to(device)

# MusicGen's sampling rate is 32 kHz; max_new_tokens ~ 50/s of audio.
sr = model.config.audio_encoder.sampling_rate  # 32000
backing = model.generate(**inputs, do_sample=True, guidance_scale=3.0,
                         max_new_tokens=512)    # ~10 s at 50 tok/s
backing = backing[0, 0].cpu().numpy()
sf.write("backing.wav", backing, sr)
print("backing:", backing.shape[0] / sr, "s @", sr, "Hz")
```

Now the vocal. There is no great *open* singing-voice model with the quality of the commercial vocal path, so this is exactly where the stand-in is weakest — which is the point. The honest options are: (a) a singing-voice-synthesis system like DiffSinger or an RVC-style pipeline, which needs a musical score (a MIDI melody plus aligned lyrics), or (b) cheat with a TTS model (XTTS) and pitch-correct toward a melody afterward, which sounds robotic because TTS does not *sing*. I will show the TTS-cheat because it runs anywhere and makes the gap visceral:

```python
# Open stand-in, part 2: a "vocal" via TTS (it will NOT sing — that's the lesson).
from TTS.api import TTS  # Coqui XTTS v2

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
lyrics_line = "Hold on, the morning's coming back around"
tts.tts_to_file(text=lyrics_line, file_path="vocal_spoken.wav",
                speaker_wav="ref_voice.wav", language="en")
# vocal_spoken.wav is SPEECH. To approximate singing you would now have to:
#   1) estimate a target melody (MIDI),
#   2) time-stretch each word to the note durations,
#   3) pitch-shift f0 onto the note pitches (e.g. with pyworld / a phase vocoder).
# The result is auto-tuned speech, not singing — flat affect, wrong phrasing.
```

Finally, align to a tempo grid and mix. Even this "mix" step exposes the joint-generation problem: the backing has its *own* implicit tempo, the vocal has none, and reconciling them by hand is exactly the work the commercial model does internally and for free.

```python
# Open stand-in, part 3: align to a shared grid and mix (the brittle part).
import numpy as np, librosa

backing, sr = librosa.load("backing.wav", sr=32000)
vocal, _   = librosa.load("vocal_spoken.wav", sr=32000)
tempo, _   = librosa.beat.beat_track(y=backing, sr=sr)  # the band's tempo
print("estimated backing tempo:", float(tempo), "BPM")

# Time-stretch the vocal toward the grid (crude: one global rate).
target_len = len(backing)
rate = len(vocal) / target_len
vocal_aligned = librosa.effects.time_stretch(vocal, rate=max(rate, 1e-3))
vocal_aligned = librosa.util.fix_length(vocal_aligned, size=target_len)

mix = 0.8 * backing + 0.6 * vocal_aligned
mix = mix / (np.max(np.abs(mix)) + 1e-8)
sf.write("song_approx.wav", mix, sr)
# Listen: the vocal floats OFF the beat and OUT of key. That gap between
# "glued" and "generated jointly" is the whole commercial moat.
```

Run that and listen. The backing is decent; the "vocal" is robotic auto-tuned speech that floats off the beat and out of the key, and no amount of post-hoc pitch correction fixes the phrasing, because the phrasing was never *musical* to begin with. That is the lesson the stand-in teaches: the commercial moat is **joint generation of vocal and backing in a single model with a real lyric-to-melody alignment** — not any single off-the-shelf component. You can buy the codec, you can buy the text encoder, you can buy a music model and a TTS model; you cannot easily buy the *joint training on lyric-aligned songs* that makes the vocal sit in the track.

## 6. How length and structure are held together

Back to the most load-bearing inference: **length forces a high-ratio codec**, and structure forces explicit conditioning. Let me make both concrete and then stress-test them.

![A vertical stack diagram showing a high-ratio codec, a section plan, a long context window, a held global key and tempo, an extend mechanism, and a coherent multi-minute song](/imgs/blogs/suno-udio-and-the-commercial-music-frontier-6.png)

The stack above is how I think minutes of coherent music are actually held together. Start at the bottom of the reasoning: the **codec ratio** sets how many tokens a second of audio costs, and that is the difference between "a four-minute song fits in context" and "it doesn't." Do the arithmetic. If the codec emits 50 frames/second with 4 codebooks read, that is 200 tokens/second, and four minutes is 48,000 tokens *per codebook* or ~48k positions in the worst case — large but feasible for a modern long-context transformer, and much larger than what MusicGen's default config attends over. Push the codec to a lower frame rate — say a more aggressive ~12–25 frames/second design, which the Mimi/[modern-codec](/blog/machine-learning/audio-generation/encodec-dac-and-the-modern-codec) line shows is achievable — and the same four minutes drops to a far friendlier token budget. The cheaper each second is in tokens, the longer the song you can hold in one context. So the first thing the codec buys you is not fidelity; it is *length*.

On top of the codec sits the **section plan** (from the `[Verse]/[Chorus]` tags), a **long context window** so the model can attend back to the first chorus when it generates the second, a **held global key and tempo** so the song doesn't modulate randomly, and — crucially — an **extend/continue** mechanism. That extend feature is a strong public hint about the architecture. The cleanest way to make arbitrarily long songs without an arbitrarily long context is **windowed roll-out**: generate a chunk, then condition the next chunk on the tail of the previous one (and on the global plan), and stitch. The product "extend" button is almost certainly this same mechanism exposed to the user. It also explains a failure mode you can hear: extended sections sometimes lose a little of the original's character, because the global plan is being re-realized from a windowed context rather than the full history.

Why does coherence need *both* the codec and scale, not just one? Because they solve different halves. The codec solves the **representation** half — it makes minutes of audio short enough to model at all. Scale (a big model with a long context, trained on lots of full songs) solves the **dependency** half — it lets the model actually *use* the distant context to bring the chorus back in tune and in time. A high-ratio codec with a tiny model gives you short token sequences the model can't make coherent; a huge model over raw samples gives you a sequence too long to attend over. You need the codec to shorten it *and* the scale to hold it. That is the same fidelity × length tension that runs through the whole series, just at the scale of a song instead of a clip.

### The rate-distortion-length triangle (the science)

There is a clean way to see why the codec's frame rate is the master knob, and it is worth deriving because it makes the design trade quantitative rather than hand-wavy. A neural codec sits on a **rate-distortion curve**: for a given architecture, you trade bits-per-second (the rate $R$) against reconstruction error (the distortion $D$). More bits, lower distortion. We covered the shape of that curve in [residual vector quantization](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq) — each added RVQ codebook quantizes the residual of the ones before it, so distortion falls roughly geometrically while rate rises linearly, which is why the curve bends.

Now bolt on the *length* dimension the generative model cares about. The generative model does not see bits-per-second; it sees a **sequence of token positions**, and its cost (compute and its ability to hold context) scales with sequence length $L$. For a song of duration $\tau$ seconds, a codec at frame rate $r$ (frames per second) with $N_q$ codebooks read produces $L \approx \tau \cdot r \cdot N_q$ token positions in a flattened layout (less if you interleave cleverly, but the scaling is the same). So you have a *triangle* of three quantities you cannot all maximize:

$$ L \;\approx\; \underbrace{\tau}_{\text{length}} \cdot \underbrace{r \cdot N_q}_{\text{rate (fidelity)}} $$

For a *fixed* model context budget $L_{\max}$ (how many positions the transformer can attend over), the maximum song length is $\tau_{\max} = L_{\max} / (r \cdot N_q)$. To make $\tau_{\max}$ bigger — longer songs — you must *lower* $r \cdot N_q$, which means *lower fidelity per second*, unless you also raise $L_{\max}$ (a bigger, more expensive model). That is the whole trade in one line: **at a fixed model budget, length and per-second fidelity are in direct competition, and the codec's $r \cdot N_q$ is the exchange rate.** A codec that achieves the same distortion $D$ at a lower $r \cdot N_q$ — a *better* rate-distortion curve — is worth more to a song model than to a clip model, because it buys length *and* fidelity simultaneously. This is precisely why a high-ratio codec (the Mimi/DAC line that hits good quality at low frame rates) is the enabling technology for commercial-scale songs, and why the codec, not the transformer, is the thing I would invest in first if I were building this. Improve the codec's rate-distortion curve and *every* downstream number — length, fidelity, compute — improves at once.

#### Worked example: token budget for a 3.5-minute song

Suppose the codec runs at 25 frames/second with 8 RVQ codebooks (a plausible high-ratio design). That is 200 tokens/second across codebooks, but if the model predicts codebooks in a flattened or delayed-interleaved layout, the *sequence length* the transformer sees is what matters. A 3.5-minute song is 210 seconds × 25 frames = 5,250 frames. With an interleaved 8-codebook layout that is ~42,000 token positions — well within a long-context transformer's reach, and small enough that windowed extend can stretch it further. Now drop the codec to 12.5 frames/second (twice as aggressive): 2,625 frames, ~21,000 positions — half the budget, twice the headroom for length, at some cost to fidelity. *That* trade — frame rate versus fidelity versus how long a song fits — is the central design knob, and it is why the codec choice dominates everything downstream. Halve the frame rate and you can either double the song length or halve the compute; the codec, not the transformer, sets that ceiling.

## 7. The product surface, and what it implies

The features Suno and Udio expose are not just UX; each one is a hint about the machine underneath, and reading them is part of the report. The conceptual pipeline below is the open stand-in's data flow, but the *commercial* features map onto it cleanly.

![A dataflow graph of an open stand-in pipeline where a style prompt drives MusicGen and lyrics drive a singing TTS, both aligned to a grid and mixed into an approximate song](/imgs/blogs/suno-udio-and-the-commercial-music-frontier-7.png)

The graph above is the open stand-in from section 5 drawn as a pipeline — style → backing, lyrics → vocal, align, mix — and the commercial products expose user-facing buttons that correspond to operations on the *real* (joint) version of this flow:

- **Extend / Continue** maps to the windowed roll-out from section 6: condition on the tail plus the plan, generate the next window. Its existence is the strongest single hint that there is an autoregressive or autoregressive-flavored continuation mechanism in at least one of these products.
- **Cover** maps to *re-conditioning*: take an input audio, extract a representation (melody, structure, maybe the lyrics via ASR), and re-generate in a new style. The fact that "cover" preserves the *song* while changing the *production* implies the model can separate "what is sung/played" from "how it sounds" — content versus style — which is exactly what a good conditioning architecture gives you.
- **Replace section / Inpaint** maps to *masked generation*: hold most of the token sequence fixed, regenerate a span conditioned on its surroundings. This is audio inpainting, and it is much more natural for a diffusion model (mask and denoise) than for a strict left-to-right AR model — a small piece of evidence that diffusion is in the mix somewhere, at least for editing.
- **Stems** almost certainly maps to a *separate source-separation model* (a Demucs-style separator) run on the final mix, not to the generator natively emitting separated tracks. It is far cheaper to separate after the fact than to generate guaranteed-separable stems, and the separation quality you hear is consistent with a good off-the-shelf separator rather than ground-truth stems.
- **Personas / voice consistency** maps to a *speaker/style embedding* carried across generations — the same in-context conditioning idea as zero-shot voice cloning in [VALL-E and the cloning frontier](/blog/machine-learning/audio-generation/zero-shot-voice-cloning-and-the-tts-frontier), applied to a singing voice.

Reading the product this way is useful precisely because it constrains the hypothesis. A pure left-to-right AR model would make *replace-section* awkward; a pure diffusion model would make *extend* awkward; both features existing suggests either a hybrid, or different engines for generation versus editing, or a flexible model that supports both masked and continued generation. The honest conclusion is that the product surface is *consistent with* the inferred recipe and mildly *informative* about the generative core, without pinning it down.

## 8. The copyright fight, and why it now drives the field

This is the part that has moved from "interesting footnote" to "central force," so I am giving it real space. I will stick to what is disclosed in the public record and clearly mark anything inferred.

**Disclosed:** In June 2024, the major record labels — UMG, Sony Music Entertainment, Warner Records — filed copyright-infringement lawsuits against Suno (in Massachusetts federal court) and Udio (in New York), coordinated through the RIAA. The core allegation is that the companies trained their models on **massive quantities of copyrighted sound recordings without authorization**, and the complaints include examples where prompts allegedly elicited outputs that closely resembled specific recognizable recordings (as evidence of what was in the training set). The suits seek damages and injunctive relief. As of this writing the cases are ongoing and unresolved; there has been no final ruling on the merits.

**Disclosed (the defense posture):** The companies have broadly argued that training on recordings to learn general musical patterns is **fair use** — transformative, learning style and structure rather than copying works — analogous to the fair-use arguments made across the generative-AI industry for text and images. They have generally not conceded the specific contents of their training data in public.

That is the factual core. Now the analysis, clearly flagged as my reading rather than a legal opinion:

The copyright question is not incidental to the technology — it is *entailed* by it. Section 4 established that to make a model sing real lyrics over real arrangements, you need an enormous corpus of **songs paired with their words and structure**. The richest, most diverse such corpus in existence is the catalog of commercially released music. So there is a direct line from "we want a model that sings like the radio" to "we need data that looks like the radio." The capability and the legal exposure come from the *same* data requirement. This is different from, say, a TTS model trained on read audiobooks with permissive licensing, or MusicGen trained on a curated, licensed/cleared set (Meta has stated MusicGen was trained on licensed and owned music). The commercial vocal-music frontier is *harder to reach with clean data* than the instrumental open-music frontier, which is part of why the open models are instrumental and the controversial models sing.

What it means for the field, again as my reading: the outcome of these cases will shape *what the next generation of these models can be trained on*, and therefore *who can build them*. If training on copyrighted recordings without license is held to require licensing, the economics shift toward companies that can strike catalog deals with the labels (and, indeed, there have since been moves toward licensing arrangements between AI music companies and rights holders). If broad fair use prevails, the open frontier could catch up faster. Either way, the durable lesson for a *builder* is to treat training-data provenance as a first-class design constraint, not an afterthought — the same lesson the image generators learned the hard way, which we discussed in [safety, watermarking, and provenance](/blog/machine-learning/image-generation/safety-watermarking-and-provenance). If you are building on these APIs commercially, read the terms of service for *output ownership and license* carefully, because they are plan-bound and they change.

![A matrix mapping the IP and safety surface — training data, voice likeness, watermarking, and output license — across the concern, the commercial models, and the open models](/imgs/blogs/suno-udio-and-the-commercial-music-frontier-8.png)

The matrix above lays out the four risk axes a builder has to weigh, and how the commercial and open paths differ on each. **Training data**: the commercial models carry the disputed-provenance risk we just discussed; the open models (used on licensed/clean sets) sidestep it but give up vocal quality. **Voice likeness**: a singing model that can mimic a recognizable artist's voice is a likeness problem distinct from copyright — both products police this with filters, with mixed success. **Watermarking**: whether and how commercial outputs are marked for provenance is mostly *inferred* (audio watermarking like AudioSeal exists and is the obvious tool), and we will treat detection and watermarking properly in the safety post; with open weights *you* are responsible for adding a mark. **Output license**: what you are actually allowed to do with a generated song is plan-bound on the commercial side and governed by the model license on the open side. None of these are reasons not to use the technology; they are the dimensions you have to think through before you ship something built on it.

## 9. Case studies and real numbers

Let me ground the report in named, defensible numbers — clearly marking which are disclosed, which are widely-reported community measurements, and which are my approximations. I will not invent precise figures.

**MusicGen (disclosed, Copet et al., 2023).** MusicGen ships at **300M, 1.5B, and 3.3B** parameters, generates **32 kHz** audio over EnCodec tokens at 50 Hz with 4 codebooks (the delay-pattern interleaving), and the canonical configuration comfortably generates around **30 seconds** of music. On the MusicCaps benchmark the paper reports FAD and CLAP-score competitive with or better than prior open music models. It is **instrumental** (no lyric-pronouncing vocals) and **open** (MIT-licensed weights via `audiocraft` / 🤗 `transformers`). This is your open baseline.

**Stable Audio (disclosed, Evans et al., 2024).** Stable Audio Open and Stable Audio 2 use **latent diffusion** over an audio autoencoder with **timing conditioning** for variable-length generation, reaching up to roughly **3 minutes** of **44.1 kHz** audio in the 2.0 model. It reports strong FAD on its benchmarks and is notable for *length* and *fidelity* — but, again, **no lyric-pronouncing vocal path**. Stable Audio Open has openly released weights; the larger models are more restricted.

**Suno (disclosed at the product level; quality inferred from listening).** Public, generates **two-to-four-minute** songs with **sung vocals** and **structure**, sampled at a music-grade rate. The v3→v4→v4.5 arc is a clear, audible quality progression: v3's characteristic codec sheen, v4's much cleaner mixes and more believable vocals, v4.5's longer and stylistically broader output. I have *no* disclosed FAD/MOS for Suno — the company has not published a benchmark report — so any number I gave would be fabricated, and I won't. What I can say honestly: in informal listening, v4+ vocals cross the "I might not have known this was synthetic on a casual listen" line for many genres, while still betraying themselves on close listening (smeared sibilants, occasional pitch wobble, a slightly *too*-consistent production polish).

**Udio (disclosed at the product level).** Public, similar capability envelope to Suno — multi-minute songs with sung vocals and structure — with a community reputation for somewhat *cleaner, more hi-fi* vocal production at launch and through its v1.5/v2 iterations. As with Suno, no official FAD/MOS benchmark is published, so I will not quote one.

Here is the honest comparison table. The "quality (approx)" column is *my qualitative ranking from listening*, explicitly not a measured score, and I have marked it so:

| Model | Sings lyrics? | Song length | Structure | Stems | Open? | Access | Quality (approx, qualitative) |
|---|---|---|---|---|---|---|---|
| Suno v4.5 | Yes | ~2–4 min | Strong (verse/chorus/bridge) | Yes (likely post-hoc separation) | No | API / app, plan-bound | Very high (vocals + production) |
| Udio v2 | Yes | ~2–4 min | Strong | Yes | No | API / app, plan-bound | Very high (clean vocals) |
| MusicGen 3.3B | No | ~30 s (default) | Weak | No | Yes (MIT) | Local / 🤗 | High instrumental, no vocals |
| Stable Audio 2 | No | ~3 min | Weak | No | Partly (Open variant) | Local / API | High fidelity, no vocals |

The shape of this table *is* the open-versus-closed music gap. On a ten-second *instrumental* clip, the open models are genuinely competitive — MusicGen and Stable Audio sound great. The gap opens on exactly the three axes the commercial models invested in: **sung lyrics, multi-minute length, and held structure**. And critically, two of those three (vocals and length) are tied to the *data and conditioning* problems that are hardest to solve with clean, licensable data — which is why the gap has been stubborn and why it sits at the center of the copyright fight rather than off to the side.

### How you would measure this honestly

Because I refused to quote FAD/MOS numbers Suno and Udio have not published, it is worth saying what an *honest* evaluation would even look like — both so you can run one yourself and so you can be appropriately skeptical of the leaderboards that circulate. We covered the machinery in [audio quality metrics](/blog/machine-learning/audio-generation/audio-quality-metrics); here is how I would apply it to commercial vocal music specifically.

For **musicality and fidelity**, FAD (Fréchet Audio Distance) is the standard, but it is treacherous for this task. FAD compares the distribution of an embedding of your generated set against a reference set, and *which embedding* you use changes the answer dramatically — a VGGish FAD and a CLAP-based FAD can rank systems differently, and neither was designed for *sung vocal music*. So you must (a) fix the embedding and report it, (b) use a large enough sample (hundreds of clips, not dozens — FAD is biased and high-variance at small $n$), and (c) match the reference distribution to the genre you are testing, or the score measures "is this the right genre" more than "is this good." For **vocal intelligibility**, the right metric is borrowed from speech: run an ASR (Whisper) over the generated vocal stem and compute **word error rate against the input lyrics** — a song that sings garbled words scores a high WER even if it sounds musical. This is the single most useful objective number for the *lyric* capability, and you can compute it yourself on any commercial output by separating the vocal and transcribing it. For **human preference**, MOS (mean opinion score) on a 1–5 scale with enough raters (20+ per clip, screened, with a known-good and known-bad anchor in the set to catch inattentive raters) is the gold standard, and CMOS (comparative MOS, A/B) is more sensitive when two systems are close. For **the structural/length capability**, there is no clean automatic metric — you fall back to human judgments of "does the chorus recur coherently" and "does it hold the beat at 3 minutes," which is exactly why this axis is under-measured and over-claimed.

The most actionable of these — vocal intelligibility — you can compute on any commercial output yourself, and it is worth showing because it is the one number that directly measures the *lyric* capability rather than vague "quality." Separate the vocal stem, transcribe it with Whisper, and compare to the lyrics you typed:

```python
# Measure lyric intelligibility: WER of the SUNG vocal vs the input lyrics.
import torch
from transformers import pipeline
import jiwer  # pip install jiwer

# 1) isolate the vocal stem (e.g. with Demucs) -> "vocal.wav", then ASR it.
asr = pipeline("automatic-speech-recognition",
               model="openai/whisper-small",
               device=0 if torch.cuda.is_available() else -1)
hyp = asr("vocal.wav", chunk_length_s=30)["text"].lower().strip()

reference = "hold on the mornings coming back around"  # the lyrics you typed
transform = jiwer.Compose([jiwer.RemovePunctuation(),
                           jiwer.ToLowerCase(), jiwer.Strip()])
wer = jiwer.wer(transform(reference), transform(hyp))
print(f"sung WER vs input lyrics: {wer:.1%}")
# A model that sings garbled words scores a HIGH WER even if it sounds musical.
# This is the single most useful OBJECTIVE number for the lyric capability.
```

The takeaway for a skeptical reader: any single headline number about Suno-vs-Udio-vs-MusicGen should be read with the embedding, sample size, reference set, rater count, and exact prompt set in hand — and most that circulate online do not disclose those. The honest comparison is the *capability* matrix (figure 2), not a single FAD digit, because the capability differences are categorical and the quality differences are contested and methodology-dependent.

#### Worked example: estimating the cost gap honestly

Suppose you need 100 fully-produced songs with sung vocals for a project. With a commercial product, the *marginal* cost is a subscription tier — on the order of a few tens of dollars a month for hundreds of generations, so call it well under **\$1 per finished song** in subscription terms, with the model doing the vocals, structure, and mix for you. With the open stand-in from section 5, the *compute* is cheap (a few cents of GPU time per clip on a rented [GPU at a few dollars an hour](/blog/machine-learning/edge-ai)), but the *labor* is the real cost: you are now a producer, doing melody design, vocal alignment, pitch correction, and mixing by hand for each song, because no open component sings your lyrics in-track for you. So the open route is cheaper in dollars-per-GPU-second and far more expensive in *human time and quality variance*. For instrumental backing, beds, or sound design, the open route is often the right call. For *finished songs with vocals*, the commercial products are, today, dramatically more cost-effective per acceptable result — and that, more than any FAD number, is why they dominate that use case. The cost gap is real but it is a *labor* gap, not a compute gap.

## 10. How they compare to open models, decided

Let me convert the analysis into a recommendation, because a report should end somewhere actionable.

On **quality**: for *instrumental* music and sound, the open models (MusicGen, Stable Audio) are competitive and sometimes preferable — you control them, you can fine-tune them, you can run them offline. For *songs with sung vocals*, the commercial models are in a different class because nothing open closes the lyric-to-vocal gap yet.

On **control**: this cuts both ways. The commercial products give you *high-level* control (style prompt, lyrics, structure tags, extend, cover) and almost no *low-level* control (you cannot set a seed and get determinism, you cannot fine-tune on your own catalog, you cannot inspect or modify the model). The open models give you *low-level* control (seeds, fine-tuning, melody conditioning via chromagram, LoRA adaptation) and weaker high-level control (no lyric path). If you need *reproducibility* or *customization to a specific sound*, open wins despite lower vocal quality.

On **cost**: as the worked example showed, commercial wins on *finished songs with vocals* (the labor is in the model), open wins on *instrumental/utility audio at scale* (compute is cheap, no labor needed per clip).

On **licensing and provenance**: open wins decisively for *anything where you must control or warrant the training-data provenance and the output license* — a regulated product, a client who needs indemnification, a use case where the copyright uncertainty is unacceptable. The commercial products' plan-bound output terms and the unresolved training-data litigation are real risks for a serious commercial deployment, and you should weigh them like any other dependency risk.

## 11. When to reach for the commercial frontier (and when not to)

A decisive section, because "it depends" is not advice.

**Reach for Suno/Udio when** you need *finished songs with sung vocals*, quickly, for content where the licensing terms of the product are acceptable to you — a demo, a social clip, a scratch track, a personalized song, internal prototyping, a jingle where you have read and accept the output terms. Nothing open will get you there with comparable quality and effort today. If the deliverable is "a song that sings these words in this style," this is the tool.

**Do not reach for them when** any of these is true: you need *reproducibility* (no seeds, outputs vary run to run); you need to *fine-tune on your own catalog or voice* (closed, you can't); you need *guaranteed clean training-data provenance and clear output rights* for a regulated or high-stakes commercial deployment (the litigation is unresolved and the terms are plan-bound); you need *offline / on-device* operation (these are cloud APIs); or you only need *instrumental* audio — in which case MusicGen or Stable Audio give you control, reproducibility, and a permissive license at a fraction of the legal uncertainty.

And a flat rule: **do not treat a commercial-model output as automatically yours to ship.** Read the output license for your plan, understand the unresolved IP backdrop, and if you are building a real product on top, treat the API as a dependency with both quality risk *and* legal risk. That is not fearmongering; it is the same diligence you would apply to any third-party component your product cannot function without.

One more practical note on the *hybrid* path, because it is often the right answer and people miss it. You do not have to choose open-or-commercial wholesale; you can split by component. Use a commercial model for the *vocal-bearing* part of a track (where it is uniquely strong) and open models for the *instrumental beds, transitions, and sound design* (where you want control, reproducibility, and a clean license). Or use a commercial model to *prototype* a song quickly, decide it works, and then *re-produce* the keeper parts with tooling whose provenance and rights you control. The capability matrix is not a verdict that one approach wins; it is a map of where each is strong, and a serious pipeline routes each sub-task to whichever side of the map serves it best. The mistake is treating "AI music" as a single buy decision rather than a set of per-component decisions, each with its own quality, control, cost, and license profile.

And a forward pointer on the safety dimension, which I have only previewed here: if you ship anything built on generated vocals — especially anything that could be mistaken for a real artist — you inherit responsibility for **provenance and consent** that the model does not discharge for you. Watermarking the output so it can be identified as synthetic, avoiding generations that mimic identifiable living artists, and disclosing AI involvement where required are not optional niceties; they are becoming table stakes, and they are the subject of the safety post this report repeatedly defers to. Build with that obligation in mind from the start, not as a bolt-on after launch.

## 12. Key takeaways

- The commercial gap is not raw quality on a ten-second clip — it is the *conjunction* of **sung lyrics + multi-minute length + held structure**. You cannot reach it by scaling an instrumental-only generator.
- The inferred (not disclosed) recipe is a **high-compression codec + a large token model (AR or diffusion) + a lyrics/phoneme + style + structure conditioning path + heavy data + preference tuning**. Every piece is a mechanism we validated earlier in this series.
- **Length forces a high-ratio codec.** That single constraint dominates the design: the codec's frame rate sets how long a song fits in context, and that ceiling matters more than the size of the transformer on top.
- **Singing is harder than TTS** because phonemes must align to a melody — a (pitch, onset, duration) target — *in time with the band and in key*. TTS has no chord to be flat against; singing does. That extra constraint drives the data requirement.
- The capability and the **copyright controversy share a root cause**: learning to sing real lyrics over real arrangements wants a corpus of real songs with their words — exactly the data the RIAA suits are about.
- **Reading the product surface** (extend ≈ windowed roll-out, cover ≈ re-conditioning, replace-section ≈ masked generation, stems ≈ post-hoc separation) constrains the hypothesis without pinning the generative core.
- For *instrumental* audio, **open models are competitive and give you control, reproducibility, and a clean license**. For *finished songs with vocals*, the commercial products dominate today — at the cost of reproducibility, customization, and legal certainty.
- Treat **training-data provenance and output licensing as first-class design constraints**, not afterthoughts — the field's near future will be shaped as much by the courts as by the codecs.

## 13. Further reading

- **MusicGen** — Copet et al., "Simple and Controllable Music Generation," Meta AI, 2023. The open single-stage codec-LM baseline this whole report is measured against. See the [MusicLM and MusicGen post](/blog/machine-learning/audio-generation/music-generation-musiclm-and-musicgen).
- **Stable Audio** — Evans et al., "Long-form Music Generation with Latent Diffusion," Stability AI, 2024. The open latent-diffusion long-form approach, covered in the [Stable Audio post](/blog/machine-learning/audio-generation/latent-diffusion-for-music-stable-audio).
- **EnCodec** — Défossez et al., "High Fidelity Neural Audio Compression," Meta AI, 2022, and **DAC** — Kumar et al., "High-Fidelity Audio Compression with Improved RVQGAN," 2023 — the codec machinery behind the inferred high-ratio token layer; see [the modern codec post](/blog/machine-learning/audio-generation/encodec-dac-and-the-modern-codec).
- **AudioLM** — Borsos et al., "AudioLM: a Language Modeling Approach to Audio Generation," Google, 2022 — the AR-over-tokens lineage for the generative core, in [WaveNet to AudioLM](/blog/machine-learning/audio-generation/autoregressive-audio-models-wavenet-to-audiolm).
- **VALL-E** — Wang et al., "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers," Microsoft, 2023 — the lyric/phoneme-conditioning analogue for the vocal path; see [VALL-E](/blog/machine-learning/audio-generation/neural-codec-language-model-tts-vall-e).
- **The RIAA complaints** — the public court filings (UMG et al. v. Suno; UMG et al. v. Uncharted Labs/Udio, 2024) are the primary source for the disclosed copyright dispute; read the complaints themselves rather than coverage.
- **The image-generation provenance discussion** — the same data-provenance and watermarking lessons in [safety, watermarking, and provenance](/blog/machine-learning/image-generation/safety-watermarking-and-provenance), which the audio field is now reliving.

This report sits between the open music posts and the safety and landscape posts that close the series. If you have not read them, start from the series foundation, [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard), which sets up the audio stack and the fidelity × controllability × speed × length frame this whole report lives inside. The detailed treatment of singing-voice synthesis and music editing continues in the [singing voice and music editing](/blog/machine-learning/audio-generation/singing-voice-and-music-editing) post; the watermarking, deepfake-detection, and voice-safety machinery this report only previewed gets its full treatment in [audio deepfakes, watermarking, and voice safety](/blog/machine-learning/audio-generation/audio-deepfakes-watermarking-and-voice-safety); the head-to-head buyer's-and-builder's view of the whole field is [the 2026 audio model landscape](/blog/machine-learning/audio-generation/the-2026-audio-model-landscape); and everything composes in the capstone, [building an audio generation stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack), where the codec, the generative core, the conditioning, the serving, the eval, and the safety all come together into one end-to-end system you can actually ship.
