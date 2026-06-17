---
title: "Real-Time, Streaming, and Full-Duplex Speech: The Conversational Frontier"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "How speech generation becomes real-time and conversational — streaming chunked synthesis, the latency budget behind sub-300ms responses, and the full-duplex frontier where Moshi and GPT-4o-style models listen and speak at once, with runnable code and honest numbers."
tags:
  [
    "audio-generation",
    "audio-synthesis",
    "text-to-speech",
    "streaming",
    "full-duplex",
    "moshi",
    "real-time",
    "generative-ai",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Audio Generation"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/real-time-streaming-and-full-duplex-speech-1.png"
---

The first time I wired a "good" TTS model into a voice assistant, the audio was beautiful and the experience was unusable. The model produced studio-clean speech — I had measured its MOS, I trusted it — but every time I asked it a question there was a beat of silence, then a beat more, and then it started talking. Maybe seven hundred milliseconds of nothing. On paper that is fast. In a conversation it is an eternity. Humans take turns in spoken dialogue with gaps measured in the low hundreds of milliseconds; cross-linguistic studies of natural conversation put the median gap between one person stopping and the next starting at around 200 milliseconds, and a gap past roughly half a second reads as a hesitation, a bad line, or a system that did not hear you. My gorgeous model felt slow and dead, and no amount of extra fidelity would have fixed it, because the problem was not how good the audio was. It was *when* the first sample arrived.

That gap between "high quality" and "feels alive" is what this post is about. Real-time conversational speech is a different engineering problem from offline synthesis, and it is governed by a different number. Offline, you optimize total quality and throughput. In conversation, the metric that decides whether the system feels human is **time-to-first-audio** (TTFA): how long after the user stops talking before your first sample of speech reaches their ears. By the end of this post you will be able to reason about that number from first principles — decompose it into encode, generate, and vocode stages; understand why a *causal* codec and *chunked* decoding are non-negotiable; measure the real-time factor and TTFA of a streaming synthesizer in PyTorch; and follow the architecture of a true full-duplex model like **Moshi**, which listens and speaks at the same time. We will start from the latency budget, climb through streaming TTS, take apart the classic cascade and why it leaks both time and emotion, and finish at the speech-native frontier where the model ingests and emits audio directly.

![A layered stack diagram showing the streaming text-to-speech latency budget from text input through encode, first-chunk generation, causal vocoding, to first audio playing under a 300 millisecond target](/imgs/blogs/real-time-streaming-and-full-duplex-speech-1.png)

This sits squarely in the recurring frame of this series, the **audio stack** — waveform → neural-codec tokens / mel latent → generative model → vocoder/decoder → waveform — under the tension of **fidelity × controllability × speed × length**. Conversational speech is where the *speed* axis stops being a nice-to-have and becomes the whole game, and where it collides with the others: you want the tone and emotion (controllability) of a big model, the fidelity of a good codec, and a latency budget that a 1-billion-parameter model running on a single GPU can actually hit. If you have not read [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard), it sets up the 1D high-rate-signal problem this all stands on, and the [capstone on building an audio generation stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack) is where these latency decisions get assembled into a serving system. This post is the conversational corner of that map.

## 1. The latency problem: why time-to-first-audio is the metric

Let me be precise about the target, because vague targets lead to over-engineering the wrong thing. A natural spoken conversation has a turn-taking rhythm. When you finish a sentence and I respond, the silence between us is short — and critically, it is short *because I started planning my reply while you were still talking*. The human turn-taking system overlaps comprehension and production; we do not wait for end-of-turn to start formulating. That is a hint we will cash in later when we get to full-duplex. For now, the empirical fact: the modal gap between turns across many languages is in the **0 to 200 millisecond** range, and listeners start to perceive a response as delayed somewhere around **300 to 500 milliseconds**. So the working budget for a conversational agent is: get the first audible sample out the door in **under ~300 milliseconds** after the user stops, and ideally under 200.

Notice what that budget is *not*. It is not "generate the whole reply fast." A two-second spoken reply at, say, a real-time factor of 0.3 takes 600 milliseconds of wall-clock compute to produce in full — but if you stream it, the user does not wait 600 milliseconds. They wait for the *first chunk*, then the rest of the audio is produced and played concurrently while they are already listening. The total compute time still matters (you cannot fall behind real-time playback, or you get gaps and stutters), but it is a *throughput* constraint, not the latency constraint. TTFA is latency; sustained generation speed is throughput. Conflating the two is the single most common mistake I see in real-time audio systems, and it leads people to spend weeks shaving total generation time while the actual felt lag — TTFA — barely moves.

So let us decompose TTFA. From the moment the user's turn ends to the moment the first speech sample plays, the time is roughly:

$$\text{TTFA} = t_\text{detect} + t_\text{encode} + t_\text{gen}^{(1)} + t_\text{vocode}^{(1)} + t_\text{buffer}$$

where $t_\text{detect}$ is how long it takes to decide the user actually stopped (voice-activity detection or an end-of-turn model — itself a latency source we will return to), $t_\text{encode}$ is encoding the text and any conditioning into the model's input, $t_\text{gen}^{(1)}$ is generating the *first chunk* of acoustic tokens or mel frames, $t_\text{vocode}^{(1)}$ is turning that first chunk into a waveform, and $t_\text{buffer}$ is the audio-output buffering the playback device imposes (often 10 to 50 milliseconds you cannot avoid). The figure above lays out these stages as a budget. The whole design discipline of streaming speech is keeping every one of these terms small, and the two that dominate for a generative model are $t_\text{gen}^{(1)}$ and $t_\text{vocode}^{(1)}$ — which is exactly why the *first chunk* and the *causal vocoder* are where we spend our attention.

Here is the reframe that makes the rest of the post click: **a non-streaming pipeline sets $t_\text{gen}^{(1)}$ equal to the time to generate the *entire* utterance**, because it produces all the frames before any vocoding starts. A streaming pipeline sets $t_\text{gen}^{(1)}$ equal to the time to generate *one short window* of frames. That single difference is the order-of-magnitude win. Everything else — causal codecs, chunked vocoding, small fast models — exists to make that first-window time as small as possible without wrecking quality at the chunk boundaries.

#### Worked example: feeling the TTFA difference

Suppose your TTS model generates audio at a real-time factor (RTF) of **0.25** on an RTX 4090 — meaning it produces audio four times faster than playback — and the reply is **2.0 seconds** long. Non-streaming: you generate all 2.0 seconds, which costs $0.25 \times 2.0 = 0.5$ seconds of compute, then you vocode the whole clip (say another 30 milliseconds), then audio starts. TTFA ≈ **530 ms** plus VAD and buffering — call it **~650 ms** felt. Now stream it in 80-millisecond chunks: the first chunk costs roughly $0.25 \times 0.08 = 20$ ms of generation plus a fixed per-step overhead (KV-cache warmup, kernel launch — say 60 ms the first time), plus ~10 ms to vocode that chunk causally. TTFA ≈ **~90 ms** of model work plus VAD and buffering — call it **~180 ms** felt. Same model, same RTF, same total quality. Streaming turned a 650-millisecond lag into a 180-millisecond one. The user goes from "is it broken?" to "it answered me." That is the entire value proposition of streaming, in one example.

The catch, and there is always a catch, is that the fixed per-step overhead (kernel launches, the first forward pass that fills the KV cache, any Python overhead) does not shrink when you shrink the chunk. So there is a floor: chunks below a certain size spend more time on overhead than on useful generation, and your RTF *gets worse* even as your per-chunk latency gets better. The art is finding the chunk size that puts TTFA under budget while keeping sustained RTF comfortably under 1. We will quantify that trade in section 3.

There is a second piece of the streaming latency model worth making explicit, because it explains why streaming sustains rather than just starts fast: **pipelining**. Once the first chunk has played, generation and playback overlap. While the user is listening to chunk $c$ (which lasts $\Delta$ seconds of audio), you are generating chunk $c{+}1$ (which costs $\text{RTF} \cdot \Delta$ seconds of compute). As long as $\text{RTF} \cdot \Delta < \Delta$ — that is, $\text{RTF} < 1$ — you finish generating the next chunk *before* the current one runs out, and the stream never starves. If $\text{RTF} > 1$ you fall behind by $(\text{RTF} - 1)\Delta$ seconds per chunk, the gap compounds, and the user hears stutter. This is the mathematical reason RTF < 1 is the hard line for sustained streaming: it is the condition under which the producer keeps up with the consumer. The margin below 1 is your safety buffer against jitter — a GPU that occasionally takes longer on one chunk (garbage collection, a context switch, a thermal throttle) needs RTF headroom to recover before the playback buffer drains. In practice I target RTF ≤ 0.5 for a comfortable conversational system, not RTF ≤ 0.99, precisely so a single slow chunk does not cause an audible gap.

Put the two halves together and you have the complete streaming latency model. **TTFA** is set by the *first* chunk — encode plus first-chunk generate plus first-chunk vocode plus buffering — and it determines whether the system feels responsive. **Sustained RTF** is set by the *steady-state* per-chunk cost and it determines whether the stream holds together for the rest of the utterance. A good streaming system makes the first number small (chunk small enough, model fast enough, codec causal) and keeps the second number well under 1 (chunk large enough that overhead does not dominate, model small enough to keep up). The reason these two goals pull in opposite directions on chunk size is the whole tension of section 3. Everything else in this post is in service of getting both numbers right at once on real hardware.

## 2. Streaming generation: emit audio chunk-by-chunk

"Streaming" in audio means exactly what it means in video: you produce and emit the output in pieces, left to right, so the consumer can start consuming before you have finished producing. For a TTS model the pieces are time-chunks of audio. Concretely, instead of one call that returns a `[1, 32000]` waveform tensor (2 seconds at 16 kHz), you run a loop that yields `[1, 1280]` waveform tensors (80 ms each) one after another, and you pipe each into the audio device as it appears. The user hears chunk 1 while you compute chunk 2.

For this to work, three properties must hold all the way down the stack, and they are the through-line of this whole post:

1. **The generative model must produce frames left-to-right and be cheap to advance one step.** An autoregressive model over codec tokens is naturally streaming: it predicts token $t$ given tokens $1..t{-}1$, and with a KV cache each new token costs one incremental forward pass. A non-autoregressive model (a diffusion model that denoises the whole spectrogram at once, or a duration-predictor TTS that lays out the full mel in one shot) is *not* naturally streaming — you have to chunk it deliberately, and that is harder. This is one reason AR codec-token models dominate the real-time conversational space even though diffusion and flow models often win on offline quality. (For the diffusion/flow machinery itself, see [diffusion for audio](/blog/machine-learning/audio-generation/diffusion-for-audio) and [flow matching and consistency for audio](/blog/machine-learning/audio-generation/flow-matching-and-consistency-for-audio) — the latter's few-step consistency tricks are precisely what make a flow model fast enough to chunk.)

2. **The codec / vocoder must be causal.** This is the property people forget and then cannot figure out why their "streaming" model has 400 ms of latency. We will spend section 4 here.

3. **You must handle chunk boundaries** so the seams between chunks do not click, pop, or break prosody. Vocoders have receptive fields; naive chunking cuts across them and produces discontinuities. The fix is overlap and a little lookahead — also section 3 and 4.

There is a fourth requirement that is not about the schedule but about the model itself, and it is the constraint people underestimate: **the model has to be small and fast enough to hit RTF < 1 on the hardware you actually have.** A 70-billion-parameter model that produces gorgeous speech at RTF 3.0 cannot stream — it generates audio three times slower than playback, and no chunking schedule fixes that, because the underlying generation rate is the problem. This is why the conversational frontier is dominated by models in the **0.5 to 8 billion** parameter range, not the largest models. Moshi is a 7B model; many production streaming voices are far smaller. The discipline of real-time speech is partly a discipline of *model sizing*: you pick the smallest, fastest model that clears your quality bar, then you make it stream, rather than picking the best model and hoping it is fast enough. When the model is too big, the standard efficient-inference toolkit applies directly — quantization (int8 or int4 weights), KV-cache optimization, batching across requests, speculative decoding, and `torch.compile` to fuse kernels and shave the per-step overhead. These are exactly the techniques covered for language models in [efficient LLM inference techniques](/blog/machine-learning/large-language-model/efficient-llm-inference-techniques), and they transfer to an audio LM because an audio LM *is* an autoregressive transformer over tokens — the tokens are just codec codes instead of text. A streaming audio system is, in inference-engineering terms, a small autoregressive LM with a tight latency SLA and a causal codec bolted on each end.

The codec's frame rate compounds the model-size constraint in a way that is easy to miss. The model must predict every codec frame, so a codec running at 75 Hz forces the model to generate 75 frames of tokens per second of audio, while a codec at 12.5 Hz forces only 12.5 — a 6× difference in tokens-per-second, and therefore a 6× difference in how fast the model must run to stay real-time. This is the second reason Mimi's low 12.5 Hz frame rate matters (the first being causality): it makes real-time generation achievable for a 7B model that could not keep up at 75 Hz. Frame rate, model size, and quantization are the three levers that set whether your RTF clears 1, and they trade against fidelity — a lower frame rate and a smaller model are easier to run real-time but cap how much detail the audio can carry. Conversational speech sits at a deliberate point on that trade: low enough frame rate and small enough model to run live, high enough fidelity to sound like a person. Getting that point right is most of the systems work.

![A before and after comparison contrasting whole-utterance batch synthesis, which pays the full generation time as latency, against chunked streaming, which pays only the first chunk before audio plays](/imgs/blogs/real-time-streaming-and-full-duplex-speech-2.png)

The figure contrasts the two regimes. On the left, whole-utterance synthesis: you generate every frame, vocode the whole clip, and only then does audio start — so TTFA equals total synthesis time. On the right, chunked streaming: generate the first window, vocode it causally, play it, and produce the rest concurrently — so TTFA equals first-chunk time. The shapes are the same model; only the schedule changed.

Here is a streaming AR-TTS loop sketch in PyTorch. It is deliberately a *sketch* — the exact model API differs across XTTS, a VALL-E-style codec LM, or Moshi — but the control flow is the real thing: generate a small block of tokens, decode it to a waveform with a causal codec, yield it, repeat. I have written it to measure TTFA and RTF as it goes, because if you are not measuring those two numbers you are not really doing streaming, you are just hoping.

```python
import time
import torch

@torch.inference_mode()
def stream_tts(model, codec, text_ids, *,
               sr=24000, chunk_frames=6, max_frames=300,
               device="cuda"):
    """Stream audio chunk-by-chunk from an AR codec-token TTS model.

    model: AR transformer over codec tokens (predicts next-frame codes).
    codec: a CAUSAL neural codec with a streaming decoder (e.g. Mimi/EnCodec).
    chunk_frames: how many codec frames to generate before emitting audio.
                  With a 12.5 Hz codec, 6 frames is ~480 ms; with an 80 ms
                  frame codec, fewer frames hits a tighter latency budget.
    Yields: (waveform_chunk [1, n], stats_dict) per chunk.
    """
    model.eval(); codec.eval()
    past = None                         # KV cache, grows each step
    state = codec.get_decoder_state()   # streaming decoder state (causal)
    cond = model.prefill(text_ids.to(device))  # encode text -> initial state
    t_start = time.perf_counter()
    first_audio_at = None
    total_audio_s = 0.0
    produced = 0

    while produced < max_frames:
        # --- generate one chunk of codec frames, autoregressively ---
        frames = []
        for _ in range(chunk_frames):
            logits, past = model.step(cond, past)   # one incremental step
            codes = logits.argmax(dim=-1)           # [1, n_codebooks]
            cond = model.embed_codes(codes)
            frames.append(codes)
            produced += 1
            if model.is_end_of_speech(codes):
                break
        if not frames:
            break
        codes_chunk = torch.stack(frames, dim=1)    # [1, T_chunk, n_cb]

        # --- causal decode this chunk to a waveform (no future needed) ---
        wav_chunk, state = codec.decode_streaming(codes_chunk, state)
        n = wav_chunk.shape[-1]
        total_audio_s += n / sr

        # --- measure: when did the FIRST sample become available? ---
        if first_audio_at is None:
            first_audio_at = time.perf_counter() - t_start  # this is TTFA

        elapsed = time.perf_counter() - t_start
        rtf = elapsed / max(total_audio_s, 1e-9)            # gen time / audio
        yield wav_chunk, {"ttfa_s": first_audio_at,
                          "rtf": rtf,
                          "audio_s": total_audio_s}

# usage: pipe each chunk straight to the sound device
# for wav_chunk, stats in stream_tts(model, codec, text_ids):
#     sounddevice.play_blocking(wav_chunk.squeeze().cpu().numpy())
#     print(f"TTFA={stats['ttfa_s']*1e3:.0f} ms  RTF={stats['rtf']:.2f}")
```

Three things in that loop carry the whole idea. First, `model.step(...)` advances the autoregressive model by exactly one frame using a KV cache (`past`), so each step is cheap and the model never recomputes the prefix. Second, `codec.decode_streaming(codes_chunk, state)` decodes the chunk *without needing future frames*, threading a small `state` (the causal convolution buffers) from chunk to chunk — that is the causal codec doing its job. Third, the `first_audio_at` timestamp captured on the very first chunk *is your TTFA*. If you print these numbers and watch them, the whole abstract latency budget becomes a live readout you can optimize against. When I tune a streaming voice, I literally watch `TTFA` and `RTF` scroll by and adjust `chunk_frames` until TTFA is under budget and RTF is comfortably under 1.

## 3. The real-time factor and the chunk-size trade-off

The **real-time factor** (RTF) is the workhorse metric for streaming audio, so let us nail the definition. RTF is generation wall-clock time divided by the duration of audio produced:

$$\text{RTF} = \frac{t_\text{generate}}{t_\text{audio}}$$

RTF < 1 means you produce audio faster than it plays — necessary for streaming, because if you generate slower than playback you fall behind and the stream stalls. RTF = 0.25 means a 4-second clip takes 1 second to make. RTF = 2.0 means it takes twice as long as the audio lasts, which is fine offline but fatal for live conversation. Critically: **RTF < 1 is necessary but not sufficient for real-time conversation.** A model with a glorious RTF of 0.05 that generates the entire utterance before emitting anything still has a TTFA equal to the full generation time. RTF tells you whether you can *sustain* the stream; TTFA tells you whether the conversation *feels* responsive. You need both, and they are different knobs.

Measuring RTF honestly has gotchas worth stating, because RTF numbers in papers and READMEs are frequently optimistic:

- **Warm up first.** The first forward pass pays for CUDA kernel compilation, lazy initialization, and KV-cache allocation. Run the model on a throwaway input before timing, or your "RTF" includes one-time costs that do not recur.
- **Use `torch.cuda.synchronize()` around timers.** GPU calls are asynchronous; without a sync you are timing the Python launch, not the compute, and you will report absurdly low numbers.
- **Report the device and precision.** "RTF 0.2" is meaningless without "on an A100 80GB in bfloat16" or "on an M2 MacBook CPU." A 30× RTF gap between a 4090 and a laptop CPU is normal.
- **Measure steady-state, not the first chunk.** TTFA captures the first chunk; RTF should be measured over the sustained stream after warmup, because that is what determines whether you keep up.

Now the central streaming trade-off: **chunk size**. The chunk is how many frames you generate before emitting audio. Smaller chunks lower TTFA (you emit sooner) but raise overhead (more per-chunk fixed costs, worse sustained RTF) and risk boundary artifacts (each seam is a place prosody and waveform continuity can break). Larger chunks smooth quality and improve RTF but raise TTFA. And **lookahead** — letting the model or vocoder peek one chunk into the future before committing the current chunk — improves boundary quality at the direct cost of one chunk's worth of added latency. These are the three dials, and they fight each other.

![A matrix comparing twenty millisecond, eighty millisecond, and three hundred twenty millisecond chunk sizes plus lookahead across time-to-first-audio, boundary quality, and compute overhead](/imgs/blogs/real-time-streaming-and-full-duplex-speech-3.png)

The matrix lays the trade out. A 20-millisecond chunk gives the lowest TTFA but the worst seams and the highest overhead — you are paying the fixed per-step cost dozens of times per second. A 320-millisecond chunk gives smooth audio and low overhead but a TTFA that may blow your budget. The 80-millisecond row is where a lot of production systems land: low enough TTFA to feel responsive, large enough that boundary artifacts are minor and overhead is tolerable. Adding one chunk of lookahead buys the best seams (the join has context on both sides) at the cost of one chunk of latency — a trade you make when quality at the boundary matters more than the last 80 ms of responsiveness.

Here is how I actually find the right chunk size: sweep it and measure both numbers. This is the script I reach for.

```python
import time, torch

def benchmark_chunk_size(make_model, codec, text_ids, sr=24000,
                         chunk_options=(2, 4, 6, 12, 24), trials=5):
    """Sweep chunk_frames; report TTFA and steady-state RTF for each."""
    results = []
    for cf in chunk_options:
        ttfas, rtfs = [], []
        for _ in range(trials):
            model = make_model()                 # fresh KV cache each trial
            # warmup so we don't time kernel compilation
            _ = next(stream_tts(model, codec, text_ids, chunk_frames=cf))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            first_at, last_audio_s, last_elapsed = None, 0.0, 0.0
            for wav, stats in stream_tts(model, codec, text_ids,
                                         chunk_frames=cf, sr=sr):
                torch.cuda.synchronize()
                if first_at is None:
                    first_at = time.perf_counter() - t0   # TTFA
                last_audio_s = stats["audio_s"]
                last_elapsed = time.perf_counter() - t0
            ttfas.append(first_at * 1e3)                   # ms
            rtfs.append(last_elapsed / max(last_audio_s, 1e-9))
        results.append((cf,
                        sum(ttfas) / len(ttfas),
                        sum(rtfs) / len(rtfs)))
        print(f"chunk={cf:>2} frames  TTFA={results[-1][1]:6.0f} ms  "
              f"RTF={results[-1][2]:.3f}")
    return results
```

When you run a sweep like this, the shape is always the same: TTFA rises monotonically with chunk size, RTF *falls* as chunk size grows (fewer fixed-overhead hits) until it plateaus, and somewhere in the middle is the sweet spot where TTFA is under your budget and RTF is comfortably under 1. You cannot reason your way to the exact number — it depends on the model, the device, the codec, and the kernel overheads — so you measure. That is the whole point of building the measurement into the loop in section 2.

#### Worked example: the chunk-size sweet spot

On a hypothetical AR codec-LM at 12.5 Hz frame rate (each codec frame ≈ 80 ms of audio) running on an RTX 4090, a sweep might land like this. Chunk of 2 frames (160 ms of audio): TTFA ≈ 95 ms, sustained RTF ≈ 0.55 (overhead-heavy — you hit the fixed per-chunk cost ~6× per second). Chunk of 6 frames (480 ms): TTFA ≈ 220 ms, RTF ≈ 0.30. Chunk of 24 frames (~2 s): TTFA ≈ 640 ms, RTF ≈ 0.22. If your TTFA budget is 300 ms, the 2-frame and 6-frame chunks both qualify, and you would pick the 6-frame chunk because its RTF (0.30) gives you far more headroom against falling behind, and its boundaries are smoother. The 2-frame chunk only wins if you are desperate for the last 100 ms of TTFA and can tolerate the extra overhead. This is a fictional-but-representative sweep — your real numbers depend on the model and device, which is exactly why you run the benchmark rather than trust a table.

## 4. Why a causal codec is mandatory

This is the section people skip and then suffer for, so I want to be concrete. A neural codec or vocoder turns frames (codec tokens, or a mel-spectrogram) into a waveform using stacked convolutions. Every convolution has a **receptive field**: to compute output sample $n$, it reads a window of input around position $n$. In a *non-causal* (bidirectional) convolution, that window is centered — it reads both past and future inputs. Stack a dozen of those and the receptive field is wide, often hundreds of milliseconds. Which means: **to produce the waveform for the current chunk, a non-causal vocoder needs frames from the future.** And if it needs the future, it cannot run until the future has been generated, which means it cannot stream. You are forced to wait for the whole utterance — and your beautiful streaming AR model is bottlenecked by a vocoder that refuses to start.

A **causal codec** fixes this by padding only on the left. Each convolution reads only the current and past inputs, never the future. Formally, for a causal 1D convolution with kernel size $k$, output $y[n] = \sum_{i=0}^{k-1} w[i]\, x[n-i]$ — the index $n-i$ never exceeds $n$, so no future sample is ever read. Stack causal layers and the receptive field extends only backward in time. The consequence: you can decode chunk $c$ the instant the model has produced chunk $c$'s frames, threading the convolution's internal buffers (the "state" in our streaming loop) forward from chunk to chunk so the layers retain their needed past context across the boundary. No future, no waiting, true streaming.

This is exactly the design choice behind **Mimi**, the codec inside Moshi (covered in detail in [EnCodec, DAC, and the modern neural codec](/blog/machine-learning/audio-generation/encodec-dac-and-the-modern-codec)). Mimi is a fully *causal* streaming codec at a low frame rate (~12.5 Hz, so ~80 ms per frame), designed so that both its encoder and decoder run streaming with a small fixed latency. That low frame rate is itself a latency and throughput choice: fewer frames per second of audio means the language model on top has fewer tokens to predict per second of speech, which directly improves how much audio you can generate per forward pass. A causal codec at a low frame rate is the codec-side half of why Moshi can hit ~200 ms latency; the small fast model is the other half.

Contrast this with a high-quality *offline* vocoder. HiFi-GAN, in its standard configuration, is fast (very low RTF) but its quality recipe uses a receptive field that, run naively, wants context on both sides of each frame. You *can* make GAN vocoders causal and streaming — BigVGAN and streaming HiFi-GAN variants exist, and Vocos is built to be efficient — but it is a deliberate architectural commitment, not free. The lesson, learned the hard way more than once: **a streaming system is only as streaming as its least-causal component.** A perfectly streaming AR model feeding a non-causal vocoder has the latency of the non-causal vocoder, full stop. Audit every box in your stack for causality before you celebrate your TTFA.

The science here is a clean trade you can state in one sentence: **causality costs you the future context that a bidirectional model uses to smooth its output, so a causal codec at the same bitrate is, all else equal, slightly lower quality than a non-causal one — and that quality gap is the price of streaming.** In practice the gap is small for a well-trained causal codec (Mimi sounds excellent), and it is dwarfed by the conversational gain of low latency. But it is real, and it is why offline music codecs (where latency does not matter) often stay non-causal: they spend the future-context budget on fidelity because they can.

It helps to put a number on the "needs the future" cost. The **algorithmic latency** of a vocoder is the amount of future input its receptive field requires before it can emit the current output sample. For a non-causal model, that is roughly half the total receptive field — if the receptive field spans 400 ms, the model needs ~200 ms of *future* audio context, so the earliest it can emit the current sample is 200 ms after that sample's time, a 200 ms latency floor you cannot remove by buying a faster GPU because it is structural, not computational. For a causal model the future requirement is zero (look-back only), so the algorithmic latency is just the codec's frame size (~80 ms for Mimi) plus whatever lookahead you deliberately add. This is why a vocoder's RTF and its latency are *independent* properties: a non-causal HiFi-GAN can have a fantastic RTF (it computes fast) and a terrible algorithmic latency (it needs the future) at the same time. You can be fast and still not streamable.

You can feel this difference in code. Here is a minimal demonstration of why a centered (non-causal) convolution stack cannot stream while a left-padded (causal) one can — the same operation, two padding schemes, two latency profiles:

```python
import torch
import torch.nn.functional as F

def causal_conv1d(x, weight, bias=None):
    """Left-pad only: output[n] depends on x[..n], never the future."""
    k = weight.shape[-1]
    x = F.pad(x, (k - 1, 0))                 # pad LEFT by k-1, right by 0
    return F.conv1d(x, weight, bias)          # streamable: no future read

def centered_conv1d(x, weight, bias=None):
    """Symmetric pad: output[n] depends on x[n-k/2 .. n+k/2] -> needs future."""
    k = weight.shape[-1]
    x = F.pad(x, ((k - 1) // 2, k // 2))      # pad BOTH sides
    return F.conv1d(x, weight, bias)          # NOT streamable: reads future

# A causal decoder can run on a chunk the instant it arrives, threading the
# left-pad state across chunk boundaries. A centered decoder must wait for the
# right-side (future) context, so its earliest output for the current chunk is
# delayed by ~half the stacked receptive field -> structural streaming latency.
x = torch.randn(1, 1, 200)                     # 200 codec frames
w = torch.randn(8, 1, 7)                       # kernel size 7
print("causal out frames:", causal_conv1d(x, w).shape[-1])     # == 200, no lag
print("centered out frames:", centered_conv1d(x, w).shape[-1]) # == 200, but lagged
```

The shapes come out identical — that is the trap. Both produce the same number of output frames, so a naive benchmark sees no difference. The difference is *which inputs each output depends on*: the causal version's output at frame $n$ reads only frames $\le n$, so you can compute it the moment frame $n$ arrives; the centered version's output at frame $n$ reads frames up to $n + k/2$, so you must wait for the future before you can emit. Stack eight such layers and the centered version's wait is eight half-kernels deep. The tensor shapes hide the latency; only the data dependency reveals it. This is exactly the bug that bites people who "make their vocoder streaming" by chunking a non-causal model and cannot understand why the seams glitch and the latency will not drop — the architecture reads the future, and chunking it just means each chunk reads across its own right boundary into data that is not there yet.

## 5. The cascade: ASR → LLM → TTS, and why it leaks

For years, the only way to build a talking AI was the **cascade**: transcribe the user's speech with an automatic speech recognizer (ASR), feed the text to a language model (LLM), and synthesize the LLM's text reply with a TTS model. Three models in series. It works, it is modular, you can swap any component, and it is still how a large share of production voice agents are built. It also has two structural problems that no amount of optimizing the individual models can fully fix, and understanding them is the setup for why the frontier moved to speech-native models.

![A dataflow graph of the cascade pipeline showing user audio entering ASR, where tone and emotion are dropped, then the text path to the LLM, then TTS, summing three serial latencies into the output](/imgs/blogs/real-time-streaming-and-full-duplex-speech-4.png)

**Problem one: serial latency.** The three models run in sequence, and their latencies add. ASR needs to hear enough audio to transcribe (and, crucially, to decide the user *stopped* — end-of-turn detection adds its own delay); the LLM needs its time-to-first-token; the TTS needs its time-to-first-audio. Even if each is individually fast, the sum is the latency floor. A rough budget: ASR final transcript 150–400 ms after end-of-speech, LLM TTFT 200–600 ms (longer for a big model or a long context), TTS TTFA 100–300 ms. Add them and you are looking at **roughly 0.5 to 1.3 seconds** before the reply starts — already over the conversational budget before you have optimized anything. You can claw some of this back with clever overlap: stream the ASR partials into the LLM before end-of-turn, stream the LLM's tokens into the TTS as they generate, so the TTS starts synthesizing the first words while the LLM is still writing the rest. That *streaming cascade* is a real and worthwhile engineering pattern — it can get you into the 300–600 ms range — but it is intricate, fragile at the seams (the LLM might revise a word the TTS already spoke), and it still cannot solve problem two. (For the LLM half of this — TTFT, KV-cache, and how to make the language model itself fast — see [efficient LLM inference techniques](/blog/machine-learning/large-language-model/efficient-llm-inference-techniques) and [KV-cache optimization](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management).)

**Problem two: information loss at the text bottleneck.** This one is deeper and unfixable within the cascade. The moment ASR converts speech to text, it throws away everything that is not words: the user's tone, their emotion, whether they were laughing or on the verge of tears, their hesitations, their sarcasm, the music of their prosody, even *who* is speaking. The LLM reasons over a flat transcript — "I'm fine" — with no signal that the user said it through clenched teeth. And on the way out, the LLM hands the TTS flat text, so the TTS has to *guess* the emotional delivery from words alone, with no grounding in how the user actually sounded. The cascade is a lossy funnel: rich audio → impoverished text → rich audio, with the middle of the funnel discarding exactly the paralinguistic information that makes a conversation feel human. You cannot recover at the TTS stage what ASR deleted at the input stage. This is why a cascade-based assistant can be word-perfect and still feel robotic and emotionally deaf — it literally never had access to how you sounded.

That funnel is the graph in the figure: the user's audio carries tone and emotion, ASR splits it into a text path that continues and a tone-and-emotion path that simply ends, the LLM reasons on text alone, and the TTS re-inflates flat text into audio. Two serial model hops and one irreversible deletion. Hold this picture — the speech-native frontier exists to delete neither the time nor the tone.

It is worth being precise about how far you can push the streaming cascade before you hit its walls, because the answer determines whether you even need a speech-native model. The latency win comes from *overlapping* the three stages instead of running them strictly in series. A streaming ASR emits partial hypotheses as the user talks ("set a tim—", "set a timer", "set a timer for ten"); you can feed those partials into the LLM speculatively, so the LLM has begun reasoning before the user even stops. The LLM, in turn, emits tokens one at a time; you stream those tokens into a streaming TTS, so the TTS starts synthesizing "Sure, I've set" while the LLM is still generating "a timer for ten minutes." Done well, this collapses the serial sum into something closer to the *maximum* of the stages plus a little slack, which is how a streaming cascade reaches the 300–600 ms range. But it is genuinely hard to get right. The ASR partials are unstable — the model might revise "timer" to "diner" two words later, after the LLM already committed; the LLM might revise a phrase the TTS already spoke aloud, which you cannot un-speak. You end up building careful commit-point logic (only act on ASR text that is stable, only synthesize LLM tokens that will not be revised), and every such guard adds latency back. The streaming cascade is a real engineering achievement, but it is a pile of carefully tuned overlap machinery wrapped around a fundamentally serial, text-bottlenecked pipeline — and no amount of overlap recovers the tone that ASR deleted at the very first hop. That ceiling is what pushed the field to stop converting to text at all.

#### Worked example: the cascade tax in numbers

Picture a customer-support voice bot built as a streaming cascade on cloud GPUs. Best-case warm path: ASR returns a stable transcript ~250 ms after the caller stops, the LLM (a 7B model) returns its first token ~350 ms later, and the streaming TTS emits first audio ~150 ms after that. Serial TTFA ≈ **750 ms** — already a noticeable beat. Now the caller says something sarcastic. The bot, reasoning over the bare transcript, answers earnestly and gets it wrong, because the sarcasm lived entirely in the prosody that ASR discarded at millisecond zero. The cascade paid the latency tax *and* the information-loss tax on the same turn. No swap of a faster ASR or a smaller LLM fixes the second tax — it is architectural. This is the concrete pressure that pushed the field toward models that never leave the audio domain.

## 6. The speech-native frontier: one model, audio in and audio out

The alternative to the cascade is to stop converting speech to text and back. A **speech-native** (or speech-to-speech, S2S) model ingests audio directly and emits audio directly, with the language modeling happening over audio tokens (and, usually, a parallel text stream for reasoning) inside a single network. No ASR hop, no TTS hop, no text bottleneck on the input. The model hears how you sound and can speak how it means.

![A before and after comparison of the cascade pipeline against a single speech-native end-to-end model, showing fewer hops, preserved tone, and full-duplex capability](/imgs/blogs/real-time-streaming-and-full-duplex-speech-6.png)

The win is structural, as the figure shows. Collapse three serial models into one and you delete two model hops of latency. Keep the signal in the audio domain end-to-end and you stop deleting tone, laughter, and timing. And — this is the part that the cascade fundamentally cannot do — a single model running on one clock can *generate audio while still ingesting audio*, which is the door to full-duplex (section 7). The two flagship instances of this frontier are **GPT-4o**-style speech-to-speech (a single multimodal model that takes audio in and produces audio out, preserving and producing tone, laughter, and interruptions, with reported response latencies in the low hundreds of milliseconds — closed, API-only) and **Moshi** from Kyutai (open weights, the architecture we can actually inspect).

There is a real cost to this win, and it would be dishonest to skip it: speech-native models are harder to *control and audit* than a cascade. In a cascade, you can read the transcript, log it, filter it, run the LLM's text output through a moderation pass, and know exactly what was said. In a speech-native model the "thinking" happens partly in audio tokens you cannot trivially read. Moshi's answer to this — and it is a genuinely clever one — is the **inner monologue**: the model predicts a *text* stream a beat ahead of its *audio* stream, so the text both improves linguistic quality (the model "thinks in words" before it speaks them) and gives you a readable, loggable, moderatable transcript of its own speech. It is the best of both: speech-native end-to-end behavior with a text trace for control. We get into that next.

## 7. Full-duplex: modeling both audio streams at once

Here is the deepest idea in the whole post, and the one most worth slowing down for. Human conversation is **full-duplex**: both people can talk (and listen) at the same time. We interrupt, we backchannel ("mm-hmm", "right", "yeah") while the other person is still talking, we start our reply before they finish, we cut in. A **half-duplex** system — which is what a cascade and most TTS systems are — strictly alternates: it listens, *then* it speaks, *then* it listens. It cannot do anything useful while it is speaking, and it cannot speak while it is listening. That is why interrupting a half-duplex assistant is awkward: you have to wait for it to finish, or it has to detect your interruption, stop, and restart a separate listening turn. The rhythm is walkie-talkie, not conversation.

True full-duplex requires a property that sounds simple and is architecturally profound: **the model must generate its own audio while simultaneously modeling the user's audio, on the same timeline.** It cannot wait for end-of-turn to start processing the user, because in full-duplex there *is* no clean end-of-turn — the user might cut in at any moment, including mid-word while the model is speaking. So the model has to predict, at every time step, both *what it is going to say next* and *what it is hearing from the user next*, jointly. This is the core of Moshi's design: it models **two audio streams in parallel** — the model's own speech stream and the user's speech stream — as a single multi-stream sequence, advancing both on a shared ~12.5 Hz clock (the Mimi frame rate). At every frame the model has just heard the latest user frame and just produced its own next frame. It is always listening *and* always (potentially) speaking.

![A dataflow graph of full-duplex dual-stream modeling where the user audio stream and the model speak stream are predicted jointly by a transformer with an inner-monologue text stream and a barge-in detector feeding the output waveform](/imgs/blogs/real-time-streaming-and-full-duplex-speech-5.png)

The figure shows the shape. The user's audio stream is encoded by the causal Mimi codec and fed in continuously — it is *always modeled*, never gated off. The joint Transformer predicts, at each frame, the model's own audio tokens (its "speak" stream), the inner-monologue text token (which runs slightly ahead and grounds the linguistic content), and implicitly carries the information needed to notice when the user's stream becomes non-silent (a barge-in). All of these advance on the same clock and feed the output waveform. Notice there are no back-edges, no feedback arrows — the "feedback" of the user interrupting is not a loop in the architecture, it is just the next user frame arriving and being modeled like every other frame. That is the elegance: barge-in is not a special case bolted on, it is the default behavior of a model that never stops listening.

There is a real modeling subtlety hiding in "predict both streams every frame," and it is where the engineering gets interesting. Each frame is not a single token — a codec like Mimi produces *several* codebook tokens per frame (one semantic plus several acoustic, in an RVQ stack), and the model is tracking two audio streams plus a text stream. So at every 80 ms tick the model must emit a small *bundle* of tokens: the text token, and for each of the two audio streams its stack of codebook codes. Moshi handles this with a depth-wise architecture (an "RQ-Transformer" style design): a large temporal transformer advances the sequence one frame at a time across the shared clock, and a small depth transformer predicts the stack of codebook tokens *within* each frame. This keeps the expensive temporal model running at the low 12.5 Hz frame rate (cheap, fast, streamable) while a tiny inner model handles the per-frame codebook depth. It is the same trick that makes RVQ-token language models tractable in general, and here it is what keeps a model tracking three streams from exploding in cost. The acoustic tokens are also typically *delayed* by a frame or two relative to the semantic and text tokens, so the model commits to *what* it will say (semantics, text) slightly before it commits to exactly *how* it sounds (acoustics) — a small structural lookahead that improves coherence without adding user-visible latency.

Why must the two streams be modeled *jointly* rather than by two separate networks? Because the model's decision of what to say next depends on what the user is doing *right now*. If the user starts talking while the model is speaking, the model needs to decide — within a frame or two — whether to keep going (the user is just backchanneling "uh-huh"), to pause, or to stop and yield the turn. That decision is a function of *both* streams at the current instant. Two independent networks, one generating and one listening, would have to communicate that decision across a boundary every frame, with latency and coordination cost. A single model that predicts both streams has the joint state in one place and can condition its next spoken token directly on the latest user frame. Joint modeling is not an optimization; it is what makes the interruption decision *possible* at conversational speed.

Here is a conceptual full-duplex loop. Again, a sketch — the real Moshi inference is more involved — but the control flow is the genuine article: at each frame, ingest the latest user audio, advance the joint model one step, and emit (or suppress) the model's own audio, all on one clock.

```python
import torch

@torch.inference_mode()
def full_duplex_step_loop(model, codec, mic_stream, speaker_out, *,
                          frame_hz=12.5):
    """Conceptual full-duplex loop: model BOTH streams every frame.

    mic_stream: yields raw audio in real time (one Mimi frame's worth each tick).
    speaker_out: a sink that plays a waveform chunk immediately.
    The model never waits for 'end of turn' — it advances every frame and
    decides per-frame whether to speak, stay quiet, or stop on a barge-in.
    """
    state = model.init_state()
    dec_state = codec.get_decoder_state()
    speaking = False

    for user_pcm in mic_stream:                       # one frame per ~80 ms
        # 1) encode the latest USER frame with the causal codec
        user_codes = codec.encode_streaming(user_pcm)  # [1, n_cb]

        # 2) advance the JOINT model one frame: it consumes the user frame
        #    and predicts its OWN next audio tokens + inner-monologue text
        out = model.step(user_codes, state)
        state = out.state
        my_codes = out.audio_tokens                    # model's own speech
        my_text  = out.text_token                      # inner monologue (loggable!)

        # 3) per-frame turn decision, conditioned on BOTH streams' joint state
        user_active = out.user_speech_prob > 0.5       # user is talking now
        if speaking and user_active and out.should_yield:
            # BARGE-IN: user interrupted -> stop speaking within ~1 frame
            speaking = False
            speaker_out.flush()                        # drop queued audio fast
            continue

        # 4) if the model chose to speak this frame, decode + play it
        if out.is_speaking:
            speaking = True
            wav, dec_state = codec.decode_streaming(my_codes, dec_state)
            speaker_out.play(wav)                       # causal -> no future needed
        else:
            speaking = False                            # silence frame: just listen

        # the inner-monologue text gives a readable transcript for moderation/logs
        log_text_stream(my_text)
```

The thing to internalize from that loop: there is no `while listening: ... then while speaking: ...` structure. There is one loop, advancing one frame at a time, and every frame the model both hears and (maybe) speaks. The half-duplex/full-duplex distinction is not a feature you add — it falls out of *whether your loop has separate listen and speak phases or a single joint phase*. Moshi reports end-to-end latency around **160 to 200 ms** in its theoretical-and-practical framing, which is inside the human turn-taking budget. That is not an accident of fast hardware; it is the architecture removing the serial hops and the end-of-turn wait that a cascade cannot remove.

## 8. Barge-in, turn-taking, and interruption handling

Full-duplex gives you the *capability* to handle interruptions; turning that capability into a system that feels polite rather than chaotic is its own craft. Three behaviors matter.

![A layered stack diagram of barge-in handling showing the model speaking while still modeling the user stream, detecting user onset, deciding between backchannel and stop, yielding the turn, and resuming](/imgs/blogs/real-time-streaming-and-full-duplex-speech-7.png)

**Barge-in (the user interrupts).** When the user starts talking while the model is speaking, a good system stops *fast* — within a frame or two, not after the current sentence. Because the user stream is modeled continuously (the figure's top box), the onset is detected almost immediately; the work is in the *decision* and the *flush*. The decision: is this a genuine interruption ("actually, wait —") or just a backchannel ("mm-hmm") that the model should talk through? A continuously listening model can tell the difference from the content and prosody of the incoming frames, where a VAD-only system (which just detects *energy*) often cannot, and stops on every "uh-huh". The flush: you must drop the audio you have already generated and buffered but not yet played, fast, or the model keeps talking for half a second after it has "decided" to stop, which feels broken. In practice this means keeping the playback buffer short and making `flush()` immediate — a place where a small output buffer (low $t_\text{buffer}$) pays off twice.

**Backchanneling (the model interrupts, gently).** A truly conversational model emits its own "mm-hmm" and "right" while the *user* is talking, to signal it is following. Half-duplex systems cannot do this at all — they are mute while listening. A full-duplex model can, because it is always able to emit on its own stream. This is a subtle but enormous part of why full-duplex feels alive: the listening is *audible*.

**Turn-taking (who goes next).** The model has to decide when the user has actually finished and it is its turn — and do so without the long end-of-turn timeout that makes cascades feel sluggish. Because it has been modeling the user's prosody continuously, it can predict turn-ends from intonation and pacing (a falling pitch contour, a completed syntactic unit) rather than waiting for a fixed silence threshold. This is the same overlap-comprehension-with-production trick humans use, and it is only available to a model that has been listening the whole time, not one that starts processing at end-of-turn.

The stack figure orders these: the model is speaking while still modeling the user, it detects a user onset, it *decides* (backchannel and keep going, or stop and yield), it stops quickly if needed, it yields and attends to the user fully, and then it resumes or replans with the new context. Every one of those steps requires the user stream to have been modeled continuously — which is precisely the property a cascade lacks and full-duplex provides.

#### Worked example: barge-in latency budget

Say the model is mid-reply and the user cuts in. With a 12.5 Hz codec, the model sees the user's onset frame within ~80 ms (one frame). The joint model's per-frame decision adds one more frame (~80 ms) to classify it as an interruption rather than a backchannel. Then `flush()` must drop the queued playback audio — if your output buffer holds 200 ms of audio, the user keeps hearing the model for up to 200 ms after the decision, which feels laggy; if you keep the buffer to ~50 ms, the model goes quiet ~50 ms after deciding. Total felt barge-in latency ≈ 80 + 80 + 50 ≈ **210 ms** with a tight buffer, versus ≈ 80 + 80 + 200 ≈ **360 ms** with a fat one. The lesson: in full-duplex, your *output buffer size* is a first-class latency parameter for interruption responsiveness, not just for playback smoothness. Keep it small and flush hard.

## 9. The trade-offs and the open-vs-closed state

Let me put the approaches side by side and be honest about where each lives on the fidelity-latency-control axes and on the open-versus-closed question.

![A matrix comparing the cascade, streaming TTS cascade, Moshi, and GPT-4o-style speech-to-speech across time-to-first-audio, full-duplex capability, tone preservation, and openness](/imgs/blogs/real-time-streaming-and-full-duplex-speech-8.png)

The matrix is the summary you can act on. The **plain cascade** is the most controllable and auditable (you have the transcript at every stage), its parts are the most swappable, and much of it is open — but it is the slowest (~0.5–1.3 s serial) and the most emotionally lossy (text bottleneck). The **streaming cascade** claws back latency into the 0.3–0.6 s range with overlapped streaming and gets partial duplex via VAD-driven turns, but still routes through text and still loses tone. **Moshi** is the open speech-native full-duplex point: ~200 ms latency, true dual-stream duplex, tone preserved because it never leaves the audio domain, and open weights so you can actually run and study it. **GPT-4o-style** speech-to-speech sets the bar for latency and naturalness (reported low-hundreds-of-ms, expressive, handles laughter and interruptions) but is closed and API-only — you cannot inspect it, run it on your own hardware, or fully control it.

The trade-offs in plain language:

- **Latency vs control.** The cascade gives you maximum control (read and filter the text at every hop) at the cost of latency and tone. Speech-native gives you minimum latency and full tone at the cost of auditability — which Moshi partially restores with the inner-monologue text stream. There is no free lunch; pick what your application needs. A medical or legal voice agent that must log and moderate every word may *want* the cascade's transcript-everywhere property even at a latency cost.
- **Quality vs latency vs causality.** A causal, low-frame-rate codec and a small fast model are what make ~200 ms latency possible, and both cost a little fidelity versus a big non-causal offline stack. For conversation, that trade is almost always worth it — a slightly-less-pristine voice that responds in 200 ms beats a flawless voice that responds in 800.
- **Open vs closed.** Today the very best conversational latency-and-naturalness lives in closed models (GPT-4o-style), while the best *open* full-duplex model is Moshi. The gap is real but narrowing, and the open side has the enormous advantage that you can run it on-device, fine-tune it, audit it, and not ship your users' voices to someone else's API. For on-device latency specifically — keeping the whole loop local so you never pay network round-trips — see [making on-device LLMs fast](/blog/machine-learning/edge-ai/making-on-device-llms-fast) and [multimodal and speech at the edge](/blog/machine-learning/edge-ai/multimodal-and-speech-at-the-edge); the network hop is itself often 50–150 ms each way, which is a brutal tax on a 200 ms budget and a strong argument for local inference.

## 10. Case studies: real numbers from the conversational frontier

Numbers ground the argument. These are drawn from the public literature and model reports; where I am giving an order of magnitude rather than a benchmarked figure, I say so, and you should treat conversational-latency numbers especially skeptically because they depend enormously on hardware, network, and what exactly is being timed.

**Moshi / Mimi (Kyutai, 2024).** Moshi is a 7B-parameter speech-text foundation model built on the **Mimi** codec — a fully causal streaming neural codec running at **12.5 Hz** (so ~80 ms per frame) at a low bitrate (~1.1 kbps in its semantic-plus-acoustic configuration). Moshi models its own and the user's audio streams jointly and runs an inner-monologue text stream a beat ahead of its audio. Kyutai reports a theoretical latency around **160 ms** and a practical latency around **200 ms** on an L4-class GPU — inside the human turn-taking window. The crucial design facts for our purposes: the codec is *causal* (enabling streaming decode), the frame rate is *low* (fewer tokens per second for the LM to predict), and the duplex behavior comes from *joint multi-stream modeling*, not a bolt-on VAD. Moshi is open weights, which is why we can describe its architecture in this detail at all.

**GPT-4o speech-to-speech (OpenAI, 2024).** A single multimodal model that ingests and emits audio directly, reported to respond in roughly the low-hundreds-of-milliseconds range (comparable to human reaction time) and to preserve and produce tone, laughter, singing, and to handle interruptions. It is closed and API-only, so the exact architecture, latency methodology, and codec are not public — treat the latency figures as the vendor's reported numbers, not independently benchmarked ones. What it demonstrates unambiguously is that a single end-to-end speech-native model *can* deliver conversational latency with full paralinguistic preservation; it set the bar the open ecosystem is chasing.

**Streaming cascade in production (general industry practice).** A well-engineered streaming cascade — partial ASR streamed into an LLM, LLM tokens streamed into a streaming TTS — typically lands TTFA in the **300–600 ms** range on warm cloud GPUs, dominated by ASR end-of-turn detection and LLM time-to-first-token. It never crosses the tone-preservation barrier (text bottleneck), and its duplex behavior is VAD-driven turn management rather than true simultaneous modeling. This is the strong baseline that speech-native models have to beat, and on pure latency-plus-tone they do; on auditability and component-swappability the cascade still wins.

**Streaming TTS vocoder throughput (HiFi-GAN / Vocos family).** The vocoder is rarely the latency bottleneck if it is causal and efficient: GAN vocoders like HiFi-GAN and Vocos run at very low RTF (well under 0.05 on a modern GPU, often real-time even on CPU), so $t_\text{vocode}^{(1)}$ is typically single-digit-to-low-tens of milliseconds per chunk. The bottleneck in a streaming TTS is almost always the generative model's first chunk and the end-of-turn detection, not the vocoder — *provided* the vocoder is causal. A non-causal vocoder turns this from a non-issue into the dominant latency term, which is the whole reason section 4 exists. (Vocoder details live in [GAN vocoders, HiFi-GAN, and fast synthesis](/blog/machine-learning/audio-generation/gan-vocoders-hifi-gan-and-fast-synthesis).)

Here is a consolidated comparison table. The latency figures are approximate and hardware/network-dependent; the structural columns (duplex, tone, open) are the durable facts.

| Approach | TTFA (approx) | Full-duplex? | Tone preserved? | Open? | Best for |
|---|---|---|---|---|---|
| Cascade ASR→LLM→TTS (batch) | ~0.5–1.3 s | No | No (text bottleneck) | Mostly yes | Auditable, swappable, non-latency-critical |
| Streaming cascade | ~0.3–0.6 s | Partial (VAD turns) | No | Mostly yes | Production voice bots needing transcripts |
| Moshi (Kyutai) | ~200 ms | Yes (dual-stream) | Yes (speech-native) | Yes (weights) | Open full-duplex, on-device, research |
| GPT-4o-style S2S | ~230–320 ms reported | Yes | Yes (laughter+tone) | No (closed API) | Best-in-class naturalness, hosted |

And the latency anatomy, cascade versus native, as a second table — this is the comparison that drove the whole field:

| Latency term | Streaming cascade | Speech-native (Moshi-style) |
|---|---|---|
| End-of-turn / VAD | 150–400 ms (explicit) | ~0 (predicted continuously) |
| ASR transcription | 150–400 ms | none (no ASR hop) |
| LLM time-to-first-token | 200–600 ms | folded into the joint model |
| TTS time-to-first-audio | 100–300 ms | folded into the joint model |
| Text bottleneck (tone loss) | yes (irreversible) | none (stays in audio) |
| Typical total TTFA | ~0.5–1.3 s (less if overlapped) | ~160–320 ms |

## 11. When to reach for this (and when not to)

A decisive recommendation section, because "it depends" is not advice.

**Reach for a streaming TTS (chunked, causal codec) whenever a human is waiting for the audio.** Voice assistants, live narration, any interactive product. The TTFA win is enormous and the engineering cost is moderate. If you are building any interactive voice product and your TTS is non-streaming, that is the first thing to fix — before you chase a fancier model.

**Reach for a streaming cascade when you need auditability and component-swappability more than the last 300 ms of latency.** Customer support bots that must log and moderate every word, regulated domains, or systems where you want to swap the LLM independently of the voice. The transcript-at-every-stage property is genuinely valuable, and a well-overlapped streaming cascade can get TTFA into the 300–600 ms range — acceptable for many products even if not magical.

**Reach for a speech-native full-duplex model (Moshi-style or a hosted S2S) when the *interaction quality* is the product** — natural barge-in, backchanneling, emotional responsiveness, the sense of a real conversation rather than a query-response terminal. Companions, tutors, accessibility tools, anything where the feel of talking to it matters as much as what it says. Use Moshi (or its lineage) when you need open weights, on-device deployment, or to study and fine-tune the model; use a hosted S2S when you want best-in-class naturalness and can accept a closed API and sending audio off-device.

**Do NOT reach for full-duplex when half-duplex is fine and simpler.** Plenty of voice applications are genuinely turn-based — a voice search, a command interface ("set a timer for ten minutes"), an IVR menu. Full-duplex adds real complexity (joint modeling, barge-in logic, flush handling) for a benefit those applications do not need. Half-duplex streaming TTS is the right, simpler tool there.

**Do NOT chase sub-100 ms TTFA when 250 ms is under budget and free.** Below the human turn-taking floor, further latency reduction is imperceptible and you are spending engineering effort (smaller chunks, more overhead, worse RTF, harder boundaries) for a number nobody can feel. Hit the budget with comfortable margin, then stop optimizing latency and spend the effort on quality or barge-in feel.

**Do NOT put a non-causal vocoder anywhere in a streaming path.** It does not matter how fast it is in RTF terms; if it needs future frames, it serializes your whole stream behind the full utterance. Audit causality first, optimize speed second.

#### Stress-testing the design

The honest engineer asks what breaks. *What happens when the chunk is too small?* Overhead dominates, sustained RTF climbs toward and past 1, and the stream stalls — you fall behind playback and get gaps. *When the codec drops the high frequencies at low bitrate?* The voice gets dull and slightly muffled; for conversation this is usually acceptable (intelligibility holds), but for a singing or expressive application it is not — raise the bitrate or use more codebooks, trading tokens-per-second (and thus throughput) for fidelity. *When the user speaks over the model and the buffer is fat?* The model keeps talking for the buffer's duration after deciding to stop, which feels broken — shrink the output buffer and flush hard. *When the network adds 150 ms each way?* Your 200 ms on-device budget becomes 500 ms felt — which is the case for running the loop on-device whenever the model is small enough to fit. *When the inner-monologue text and the audio drift out of sync?* The model can say one thing and "mean" another in its log — a real failure mode for moderation, and a reason the text-audio alignment in Moshi-style models is a careful part of the design, not an afterthought. Knowing these failure modes is most of what separates a demo that works once from a system that holds up in real conversations.

## 12. Key takeaways

- **Time-to-first-audio (TTFA), not total throughput, is the metric for conversational speech.** The human turn-taking budget is ~200 ms, with perceptible lag past ~300–500 ms. Optimize when the first sample arrives, not how fast the whole utterance generates.
- **Streaming is the order-of-magnitude win.** Emitting audio chunk-by-chunk makes TTFA equal to first-chunk time instead of full-utterance time. It is the first thing to fix in any interactive voice system.
- **RTF < 1 is necessary but not sufficient.** It tells you whether you can sustain the stream; TTFA tells you whether it feels responsive. Measure both, with warmup, device, and `cuda.synchronize()`.
- **Chunk size is the central streaming dial.** Smaller chunks cut TTFA but raise overhead and seam artifacts; lookahead buys boundary quality at one chunk of latency. Sweep and measure — you cannot reason your way to the number.
- **A causal codec is mandatory for streaming.** A non-causal vocoder needs future frames and serializes your stream behind the whole utterance. Your system is only as streaming as its least-causal component. Mimi is causal by design.
- **The cascade leaks time and tone.** Three serial model latencies sum past the budget, and the text bottleneck irreversibly deletes tone, emotion, and timing at the ASR hop. No amount of optimizing the parts fixes the second problem.
- **Speech-native models keep the signal in audio end-to-end**, deleting two model hops and the tone bottleneck, and — uniquely — enabling generation while still listening.
- **Full-duplex requires joint modeling of both audio streams on one clock.** You must predict your own next frame and the user's next frame together, so barge-in is the default behavior, not a special case. This is Moshi's core design.
- **Open vs closed:** Moshi is the open full-duplex frontier (~200 ms, weights available); GPT-4o-style S2S sets the closed naturalness bar. The cascade trades latency and tone for auditability and swappability.

## 13. Further reading

- **Défossez, Mazaré, Orsini, et al. — "Moshi: a speech-text foundation model for real-time dialogue" (Kyutai, 2024).** The full-duplex, dual-stream, inner-monologue architecture and the Mimi codec; the primary source for everything in sections 6–8.
- **Défossez, Copet, Synnaeve, Adi — "High Fidelity Neural Audio Compression" (EnCodec, Meta AI, 2022).** The streaming causal codec template Mimi builds on; the causality discussion in section 4 traces here.
- **Borsos, Marinier, Vincent, et al. — "AudioLM: a Language Modeling Approach to Audio Generation" (Google, 2022).** The semantic-plus-acoustic token paradigm underneath speech-native audio LMs.
- **Wang, Chen, Wu, et al. — "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers" (VALL-E, Microsoft, 2023).** TTS as codec-token language modeling — the AR-streaming template for speech.
- **Kong, Kim, Bae — "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis" (2020).** The fast vocoder whose causality (or lack of it) decides whether your stream is real.
- **🤗 `transformers` audio docs and the Moshi / Mimi model cards** — runnable APIs for the streaming and codec pieces sketched here.
- Within this series: [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard) (the foundation), [EnCodec, DAC, and the modern codec](/blog/machine-learning/audio-generation/encodec-dac-and-the-modern-codec) (the Mimi causal codec), [flow matching and consistency for audio](/blog/machine-learning/audio-generation/flow-matching-and-consistency-for-audio) (fast few-step synthesis), [conditioning and control in audio generation](/blog/machine-learning/audio-generation/conditioning-and-control-in-audio-generation), [zero-shot voice cloning and the TTS frontier](/blog/machine-learning/audio-generation/zero-shot-voice-cloning-and-the-tts-frontier), and the [capstone on building an audio generation stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack). Looking ahead: [the 2026 audio model landscape](/blog/machine-learning/audio-generation/the-2026-audio-model-landscape).
