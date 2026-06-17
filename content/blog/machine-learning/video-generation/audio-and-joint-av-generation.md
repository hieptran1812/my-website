---
title: "Audio and Joint Audio-Video Generation: Giving Video a Soundtrack"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "The 2024-2026 leap from silent clips to synchronized sound: why audio sync is brutally unforgiving, how cross-modal attention on a shared timeline enforces it, the video-to-audio and joint-generation recipes, and a runnable foley pipeline you can adapt."
tags:
  [
    "video-generation",
    "diffusion-models",
    "audio-generation",
    "video-to-audio",
    "joint-generation",
    "lip-sync",
    "multimodal",
    "generative-ai",
    "deep-learning",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Video Generation"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/audio-and-joint-av-generation-1.png"
---

Play any silent generated clip from early 2024 next to the same scene from a model shipped in 2025, and the difference that hits you first is not resolution, not motion, not even prompt adherence. It is that the 2025 clip *makes a sound*. A glass tips off a table and you hear it shatter on the exact frame it touches the floor. A woman turns to the camera and says a sentence, and her lips form the words. Rain hits a window and the patter sits under everything like ambient room tone. None of those pixels are better than they were a year earlier. What changed is that the model learned to put sound on the same timeline as the picture, and the perceived-quality jump from that one capability was larger than any single improvement in the image quality itself.

That is the strange thing about audio in video generation: it is cheap in bits and enormous in impact. A 5-second 48 kHz mono soundtrack is 240,000 samples, which after a neural codec is a few hundred latent vectors — a rounding error next to the millions of latent voxels in the video. Yet humans are violently sensitive to whether that tiny stream of audio is *aligned*. We will tolerate a soft, slightly wrong-colored frame for a long time. We will not tolerate a footstep that lands two frames after the foot does. The whole engineering problem of audio-video generation is this asymmetry: a modality that costs almost nothing to generate has to be timed to within a few tens of milliseconds or the entire clip reads as fake.

![Graph of a joint audio-video model running a video branch and an audio branch that meet at a cross-modal attention block on a shared timeline before muxing into one synced clip](/imgs/blogs/audio-and-joint-av-generation-1.png)

This is the post where we give video a soundtrack. We will keep returning to one running example — a 5-second 720p clip of a dog running across a wooden deck and barking once, which at 24 frames per second is 120 frames and at 48 kHz is 240,000 audio samples — and watch what each approach does to the *timing* of that single bark. We will derive why a few-frame offset is perceptible (it is a sample-rate-mismatch problem, and the human auditory system resolves audio-visual asynchrony far below one video frame), and then spend our real depth on the three families that solve it: post-hoc **video-to-audio** (generate a soundtrack for a finished clip, the Movie Gen Audio and MMAudio line), single-pass **joint audio-video** (Veo 3's native synchronized dialogue, sound effects, and ambience — the headline capability of 2025), and the **cascaded** fallback that routes through a caption and pays for it in sync. We will look at **lip-sync and talking-head generation** as a distinct sub-problem with its own physics, build a runnable conceptual video-to-audio pipeline in PyTorch and mux it with `ffmpeg`, and end on how you actually *measure* sync honestly. By the end you will know which approach to reach for, what each costs in alignment, and where the open models stand against the proprietary frontier.

This sits at the conditioning edge of the series. If you have read [why video generation is hard](/blog/machine-learning/video-generation/why-video-generation-is-hard), the one-line version of the whole series is *video equals spatial generation times temporal coherence under a brutal compute budget* — and audio adds a second clock running at a thousand times the frame rate that has to stay locked to the first. The audio branch reuses the exact diffusion machinery we built for images; if you want the basis, read [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles) in the image series, because the audio denoiser is the same idea applied to a mel or codec latent. We treat conditioning a video on text, image, motion, and camera in its [sibling post](/blog/machine-learning/video-generation/conditioning-video-text-image-motion-camera); audio is the conditioning axis that turned out to be the hardest to time. We will forward-link the frontier audio capability of [Veo and cinematic generation](/blog/machine-learning/video-generation/veo-and-cinematic-generation) and the [capstone playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook) where you wire all of this into a real pipeline.

## 1. Why audio sync is the unforgiving part

Start with the numbers, because the entire difficulty falls out of a sample-rate mismatch. Our running clip is 120 video frames over 5 seconds — one frame every 41.7 milliseconds. The audio is 240,000 samples over the same 5 seconds — one sample every 20.8 *microseconds*. The two streams describe the same physical event, the dog's paw striking the deck, but they are quantized on clocks that differ by a factor of two thousand. The video has no idea, frame to frame, where inside a 41.7 ms window the strike actually happened; the audio knows it to the microsecond. When we generate both, we are asking the model to decide *which audio sample* the visual event lands on, and to do it consistently, when the visual signal only resolves the event to a 41.7 ms bin.

That would be fine if humans were as forgiving about timing as they are about, say, color. They are not. The perceptual literature on audio-visual synchrony is brutally specific: people reliably notice when audio *leads* video by more than about 45 ms and when audio *lags* video by more than about 125 ms; inside that window the brain fuses the two into a single event, and outside it the illusion breaks and you perceive two separate things. The asymmetry is real — we tolerate sound arriving late better than early, because in the physical world sound always arrives after light, so our priors expect a small lag. But 45 ms is roughly *one video frame*. The tolerance for a footstep landing early is on the order of a single frame at 24 fps. That is the whole problem in one sentence: the acceptable sync error is about the size of one quantization step of the slower modality, so there is no slack.

It gets worse for speech. For lip-sync specifically, the McGurk-effect literature shows the auditory and visual streams are integrated even more tightly — a viseme (the visible mouth shape) that does not match the phoneme (the heard sound) produces a *different perceived syllable*, not just a feeling of wrongness. A model that gets the mouth shape right but two frames late does not produce slightly-off dialogue; it produces dialogue that your brain actively re-interprets into the wrong words. This is why talking-head generation is a distinct sub-problem with a tighter tolerance than foley: a splash that is one frame late is a minor flaw, but a mouth that is one frame late changes what you think you heard.

So the design constraint is set before we choose any architecture. Whatever generates the audio must be conditioned on the video — or generated jointly with it — at a temporal resolution fine enough to place events within a single frame, and the conditioning has to be *frame-aligned*, not just "this clip contains a bark." The moment you route the timing through anything that does not preserve per-frame structure — a text caption, a global embedding, a pooled feature — you lose the clock, and the bark drifts. Hold that thought; it is the single most important fact in this post, and it is the reason cascaded approaches lose.

#### Worked example: how many samples is "one frame" of error?

Take the bark. Suppose the true onset is at $t = 2.000$ s, frame 48. A model that places it one frame late puts it at $t = 2.0417$ s — an offset of 41.7 ms, which is $0.0417 \times 48000 \approx 2000$ audio samples. That is well past the ~45 ms early / ~125 ms late fusion window on the early side and right at its edge on the late side. Now suppose we route through a 2 fps captioning model (a common cascaded design): its temporal resolution is 500 ms, so the *best* it can do is place the bark within a 500 ms bin — twelve frames, an offset that is unambiguously, jarringly wrong. The lesson is quantitative: your audio-conditioning clock has to tick at least as fast as the perceptual tolerance, ~40 ms, which means you need video features at roughly 24-25 Hz, i.e. per-frame. Anything coarser bakes in audible drift before the audio model does any work.

## 2. The shared-timeline idea, made precise

If the constraint is "audio must be placed to within one frame," the architecture that satisfies it most directly is one where audio and video share a *timeline* — a common temporal axis the model attends over — so that the network can literally point the audio at the frame where the event happens. This is the core mechanism behind every strong system, whether it generates audio after the video (V2A) or alongside it (joint), and it is worth making precise because the validator of whether an approach can possibly sync is "does it preserve a shared, per-frame timeline?"

Concretely, both modalities get encoded into latent *frames* on the same clock. The video latent is $z \in \mathbb{R}^{T \times H \times W \times C}$, where after the causal 3D-VAE (see [video autoencoders and spatiotemporal compression](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression)) $T$ might be 30 latent frames for our 120-pixel-frame clip — a 4× temporal compression. The audio latent is $a \in \mathbb{R}^{T_a \times D}$, where $T_a$ is the number of audio latent frames. The trick that makes everything work is choosing the audio codec's frame rate so that $T_a$ relates to $T$ by a clean integer ratio, and then attending across both with positional encodings that put a video latent frame and the audio latent frames that overlap it at the *same position*. When that alignment holds, a cross-modal attention layer can let audio frame $\tau$ attend to exactly the video frames around time $\tau / \text{fps}_a$, and the network has a direct, differentiable path to "make the splash sound loud at the audio frame that overlaps the splash video frame."

Cross-modal attention is the load-bearing mechanism, so let us write it out. In a joint block, you have video tokens $V \in \mathbb{R}^{N_v \times d}$ and audio tokens $A \in \mathbb{R}^{N_a \times d}$ living in a shared model dimension $d$. The audio branch attends to video with

$$
\text{CrossAttn}(A, V) = \text{softmax}\!\left(\frac{(A W_Q)(V W_K)^\top}{\sqrt{d}} + B\right) (V W_V),
$$

and the video branch symmetrically attends back to audio. The piece that enforces sync is the bias $B \in \mathbb{R}^{N_a \times N_v}$: a *temporal* bias that rewards attention between audio and video tokens that are close in time and penalizes far-apart ones. In the simplest form $B$ is a learned function of the time offset $|t_a - t_v|$, a relative-position bias on the time axis only. This is the same relative-position trick used in long-context transformers, repurposed so that "context" means "the same instant in the other modality." Give the network this bias and it does not have to *learn from scratch* that audio at 2.0 s should look at video at 2.0 s; the prior is baked into the attention pattern and only the content (how loud, what timbre) is learned.

The alternative to a learned bias is a hard *windowed* cross-attention: audio frame $\tau$ may only attend to video frames within $\pm w$ of its mapped time. This is cheaper (the attention is banded, not dense) and it makes the sync guarantee structural rather than learned — the audio physically cannot attend to a video frame more than $w$ frames away, so it cannot place an event there. The cost is that a genuinely non-local audio-visual relationship (an echo, a reverb tail that depends on the room geometry seen earlier) gets clipped. In practice systems use a wide-ish window for foley and let dialogue rely more on the learned bias, because speech timing is local but prosody is global.

![Stack diagram showing audio compressed from a 48 kHz waveform through a mel or neural codec encoder into a latent, modeled by a diffusion or autoregressive network, then decoded back to a waveform and muxed with the video](/imgs/blogs/audio-and-joint-av-generation-2.png)

How much does this cross-modal attention actually cost? This is the question that decides whether joint generation is affordable, and the answer is the reassuring part of the whole post. Self-attention within the video branch is $O(N_v^2 d)$, where $N_v$ is the number of video tokens — and $N_v$ is *enormous*, the dominant cost of the entire model. Cross-modal attention between audio and video is $O(N_a N_v d)$, and because the audio token count $N_a$ is two to three orders of magnitude smaller than $N_v$ (Section 3 makes this concrete: ~250 audio tokens against hundreds of thousands of video tokens), the cross term is a rounding error next to the video self-attention. Concretely, if $N_v = 432{,}000$ and $N_a = 250$, then $N_a N_v / N_v^2 = N_a / N_v \approx 0.0006$ — the cross-modal attention adds well under a tenth of a percent to the attention FLOPs. With a windowed temporal bias it is cheaper still, because each audio token only attends to a band of nearby video tokens rather than all of them. The arithmetic is the whole reason joint generation took off: *sync is nearly free to compute once you have decided to generate the video at all*. The cost is in the data and the training, not the inference flops.

There is one more piece the shared-timeline picture needs: *timestep-aligned conditioning*. In a diffusion model both branches denoise over the same schedule of noise levels $t \in [0, 1]$. If you let the audio branch run on a different denoising schedule than the video branch, their intermediate representations are at different noise levels and the cross-attention compares apples to oranges — a clean audio token attending to a still-very-noisy video token gets garbage. The fix is to share the diffusion timestep across both branches so that at every step they are at the *same* noise level, and the cross-modal attention is always comparing representations of comparable cleanliness. This is subtle and easy to get wrong: I have seen a joint model where the audio quality was fine and the video quality was fine but the *sync* was mush, and the cause was that the two branches had been given independently sampled timesteps during training, so the model never learned to align clean-to-clean. Tie the timestep and the sync snaps into place.

## 3. How audio becomes a latent the model can generate

We keep saying "audio latent" — let us be concrete about what it is, because the representation choice is exactly as load-bearing for audio as the 3D-VAE choice is for video, and for the same reason: it sets the token budget and therefore what is feasible.

Raw audio is a wall of samples. Our 5-second clip is 240,000 mono samples at 48 kHz; stereo doubles it. You cannot generate that with a diffusion transformer directly any more than you can diffuse raw pixels — the sequence is too long and the redundancy too high. So, exactly as with video, you compress first. There are two dominant families.

The first is the **mel spectrogram**. You take the waveform, run a short-time Fourier transform with, say, a 1024-sample window hopping every 256 samples, and map the magnitudes onto ~80-128 mel-frequency bins. The result is a 2D image: time on one axis (now at ~188 frames per second for a 256-sample hop at 48 kHz, or whatever the hop gives you), mel-frequency on the other. This is a *picture of the sound*, and the beautiful consequence is that you can diffuse it with almost the same 2D U-Net or DiT you use for images — a mel spectrogram is just a single-channel image with a strong axis structure. The catch is that a mel spectrogram throws away phase; to get audio back you need a *vocoder* (HiFi-GAN, BigVGAN, or a diffusion vocoder) that hallucinates plausible phase from the magnitudes. Mel-plus-vocoder was the workhorse of text-to-audio for years and is still common in V2A.

The second, and now more common at the frontier, is the **neural audio codec latent** — the audio analog of the 3D-VAE. A model like EnCodec, DAC (Descript Audio Codec), or SoundStream learns an encoder that maps the waveform to a sequence of latent frames at a low rate (commonly 50-75 Hz) and a decoder that reconstructs the waveform from them, trained with a reconstruction loss plus an adversarial loss for perceptual quality. The latent can be continuous (good for diffusion) or discretized into a stack of codebook tokens via residual vector quantization (good for autoregressive models). A 48 kHz waveform at a 50 Hz codec rate is compressed by roughly 960× in frame count along the time axis before the channel dimension is even counted — the same kind of brutal-but-necessary compression the video VAE buys you. For our 5-second clip, a 50 Hz codec gives ~250 audio latent frames, which is tiny and sits comfortably next to the ~30 video latent frames. The codec is what makes joint generation affordable.

#### Worked example: the audio token budget is negligible

Put the two budgets side by side. The video latent for our clip, after a 4×8×8 causal 3D-VAE, is roughly $30 \times 90 \times 160 = 432{,}000$ latent voxels — millions of values once you count channels, and it is the dominant cost of the whole generation. The audio latent at a 50 Hz codec with, say, a 128-dim continuous latent is $250 \times 128 = 32{,}000$ values. That is under one percent of the video latent. The practical takeaway is stark and worth internalizing: *audio is almost free to generate*. The entire challenge is not the cost of the audio branch — it is the cost of keeping it *aligned*, which is paid in cross-attention between the two branches, and even that is cheap because the audio token count is so small. When people say "Veo 3 generates audio in the same pass," the surprising part is not that it is expensive; it is that it is nearly free once you have decided to pay for the cross-modal attention.

There is a second axis to the representation choice that interacts with how you *generate*: diffusion versus autoregressive. A diffusion audio model denoises a *continuous* latent in parallel across all audio frames at once, which makes it natural to keep on the same parallel diffusion schedule as the video branch — both denoise the whole clip together, step by step, which is exactly what timestep-aligned joint generation wants. An autoregressive audio model instead predicts *discrete* codec tokens one (or one stack) at a time, left to right, the way a language model emits subword tokens; this is the AudioLM and MusicGen lineage. The autoregressive route has one elegant property for joint generation: because both video patches and audio tokens can be serialized into a *single* sequence, you can interleave them and let one transformer attend over the merged stream, so sync becomes "tokens that are near each other in the sequence are near each other in time" — no separate cross-modal-attention machinery needed, just careful interleaving and positional encoding. The cost is that autoregressive generation is sequential and slow for the long token streams audio produces, and it accumulates error along the sequence the same way long-video autoregressive rollout does. Most 2025 frontier *joint* systems lean diffusion (or flow matching) for the parallel, timestep-aligned property; the autoregressive interleaving idea is elegant and is where some research is heading, but the diffusion two-branch design is what is shipping at quality today.

Which representation should you reach for? For a diffusion-based V2A or joint model, a continuous neural-codec latent is the modern default — it gives you a compact, smooth space to diffuse in and a high-quality learned decoder, and it sidesteps the phase-reconstruction problem that plagues mel. For an autoregressive model (the AudioLM / MusicGen lineage), discrete RVQ codec tokens are natural — the model predicts codebook indices the same way a language model predicts subword tokens, and you can interleave them with text or video tokens in one sequence. Mel-plus-vocoder still wins when you want to lean on a mature image-diffusion stack with minimal new machinery, which is part of why several early V2A systems used it. The trend at the 2025 frontier is decisively toward neural-codec latents for both joint and V2A, because the codec's learned decoder produces cleaner audio than a vocoder reconstructing from lossy mel magnitudes.

## 4. Approach one: video-to-audio (add the soundtrack after)

The most pragmatic way to give a finished clip a soundtrack is to treat audio generation as a *conditioning* problem: you have the video, it is done, and you want a sound track that matches it. This is **video-to-audio** (V2A), and it is the approach behind Meta's Movie Gen Audio and the open MMAudio line. The appeal is modularity — your video model and your audio model are separate, you can swap either, and you can run V2A on *any* video, including footage you did not generate. The constraint, from Section 1, is that the audio model has to be conditioned on *frame-level* video features, not a global summary, or it cannot sync.

The pipeline has a clean shape. Read visual features off every frame of the clip with a frozen image encoder (CLIP is the common choice, sometimes a video encoder like a frozen VideoMAE). Optionally compute an explicit *motion* signal — frame-difference energy, optical-flow magnitude, or a learned onset detector — because the single most useful cue for "when does a sound happen" is "when does something move sharply." Feed those per-frame features as timestep-aligned conditioning into an audio diffusion (or autoregressive) model that generates the audio latent, decode to a waveform, and mux. MMAudio's contribution was to train this V2A model *jointly* on audio-visual *and* text-audio data in a shared representation, so the same model can be conditioned on the video, on a text prompt ("rain, distant thunder"), or both — which is how you get controllability on top of sync.

![Graph of a video-to-audio pipeline reading visual and motion features off a finished silent clip, feeding timestep-aligned conditioning into an audio diffusion model, and muxing the generated waveform back with the video](/imgs/blogs/audio-and-joint-av-generation-3.png)

Here is a conceptual V2A pipeline sketch. It is deliberately framework-shaped — the structure mirrors how a real MMAudio-style model is wired, with the places a public checkpoint plugs in marked. It will not run end-to-end without the actual model weights, but every API call and tensor shape is real, and you can drop in a published V2A checkpoint where indicated.

```python
import torch
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor
from diffusers import DDPMScheduler
import decord  # frame I/O; pip install decord

device = "cuda"
dtype = torch.float16

# --- 1. Read the finished silent clip, one tensor per frame -------------
vr = decord.VideoReader("dog_on_deck_silent.mp4")
fps = vr.get_avg_fps()              # ~24
n_frames = len(vr)                  # ~120 for our 5 s clip
frames = vr.get_batch(range(n_frames)).asnumpy()   # (T, H, W, 3) uint8

# --- 2. Per-frame visual features (the sync clock) ----------------------
proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
vis = CLIPVisionModel.from_pretrained(
    "openai/clip-vit-large-patch14", torch_dtype=dtype
).to(device).eval()

with torch.no_grad():
    px = proc(images=list(frames), return_tensors="pt").pixel_values.to(device, dtype)
    # pooled CLS per frame -> (T, d_clip); keep per-frame, never pool over T
    vfeat = vis(px).pooler_output            # (T, 1024)

# --- 3. An explicit motion signal: where does something move sharply? ---
gray = torch.tensor(frames, dtype=torch.float32, device=device).mean(-1)  # (T,H,W)
motion = (gray[1:] - gray[:-1]).abs().mean(dim=(-1, -2))                   # (T-1,)
motion = F.pad(motion, (1, 0))                                            # (T,) align to frames
motion = (motion - motion.mean()) / (motion.std() + 1e-6)                 # z-score

# Stack the two per-frame conditions on a shared timeline (length T).
cond_per_frame = torch.cat([vfeat, motion[:, None].to(dtype)], dim=-1)    # (T, 1025)
```

That gives you the conditioning stream — a per-frame feature vector that ticks at the video frame rate and carries both *what is on screen* and *when it moves*. The next stage is the audio diffusion model itself. The structure below shows where a real V2A checkpoint slots in; the denoise loop is the standard diffusion sampler you already know from the image series, applied to an audio latent and conditioned on `cond_per_frame`.

```python
# --- 4. Audio diffusion over a neural-codec latent ----------------------
# In a real system, `audio_model` is an MMAudio-style V2A denoiser and
# `codec` is a neural audio codec (e.g. DAC / EnCodec). Load published
# weights here; we show the shapes and the conditioning path.
from your_v2a_pkg import load_v2a_model, load_audio_codec   # placeholder import

audio_model = load_v2a_model().to(device, dtype).eval()     # the V2A denoiser
codec = load_audio_codec().to(device).eval()                # waveform <-> latent

sched = DDPMScheduler(num_train_timesteps=1000)
sched.set_timesteps(50)                                     # 50 sampling steps

latent_fps = 50                                             # codec frame rate
T_a = int(round(n_frames / fps * latent_fps))              # ~250 audio frames
d_a = 128                                                   # codec latent dim

# Upsample the per-frame condition from video-fps to audio-latent-fps so the
# two streams share the SAME clock before the model ever sees them.
cond_a = F.interpolate(
    cond_per_frame.t()[None], size=T_a, mode="linear", align_corners=False
)[0].t()                                                    # (T_a, 1025)

a = torch.randn(1, T_a, d_a, device=device, dtype=dtype)   # noisy audio latent
for t in sched.timesteps:
    with torch.no_grad():
        # SAME timestep drives the denoiser; condition is frame-aligned.
        eps = audio_model(a, t.to(device), cond=cond_a[None])
    a = sched.step(eps, t, a).prev_sample

with torch.no_grad():
    waveform = codec.decode(a)                              # (1, samples) at 48 kHz
```

Two details in that loop matter more than they look. First, the `F.interpolate` step is the shared-timeline enforcement from Section 2 in code: we resample the per-frame condition to the *audio* latent rate so that condition frame and audio latent frame index the same instant. Get the resampling wrong — off-by-one, wrong mode, a stray pooling — and you reintroduce the drift you worked so hard to avoid. Second, the condition is passed at *every* denoising step, not just the first; the audio model needs the visual clock present throughout the trajectory, because the alignment is refined as the audio sharpens.

Finally, you mux. The audio waveform and the original video are two separate streams that need to be combined into one container with their timelines locked. `ffmpeg` does this without re-encoding the video, which keeps it fast and lossless:

```bash
# Write the generated waveform to a wav, then mux with the ORIGINAL video.
# -c:v copy avoids re-encoding the frames (fast, lossless); -shortest trims
# to the shorter stream so a/v stay aligned at the tail.
ffmpeg -i dog_on_deck_silent.mp4 -i generated_audio.wav \
       -c:v copy -c:a aac -b:a 192k -map 0:v:0 -map 1:a:0 \
       -shortest dog_on_deck_synced.mp4
```

If you would rather stay in Python, the `av` library (PyAV, a binding over FFmpeg) gives you frame-level control over the mux, which is what you want when you are programmatically aligning a generated waveform to a generated video and need to set the audio start timestamp exactly:

```python
import av, numpy as np

inp = av.open("dog_on_deck_silent.mp4")
out = av.open("dog_on_deck_synced.mp4", "w")

vstream = out.add_stream(template=inp.streams.video[0])   # copy video as-is
astream = out.add_stream("aac", rate=48000)               # new audio stream

# Remux video packets untouched (preserves frame timing).
for packet in inp.demux(inp.streams.video[0]):
    if packet.dtype is None:
        continue
    packet.stream = vstream
    out.mux(packet)

# Encode the generated waveform (numpy float32, shape (samples,)) as audio frames.
samples = (waveform.float().cpu().numpy() * 32767).astype(np.int16)
aframe = av.AudioFrame.from_ndarray(samples[None], format="s16", layout="mono")
aframe.rate = 48000
for p in astream.encode(aframe):
    out.mux(p)
for p in astream.encode(None):   # flush
    out.mux(p)
out.close()
```

The honest engineering note on V2A: its ceiling is set by how much the audio model can infer from the video alone. It can nail foley — impacts, footsteps, splashes, ambient texture — because those are tightly determined by visible motion. It cannot produce *dialogue with the right words*, because the words are not in the pixels; a V2A model watching a person's mouth move can generate mouth-noise that is rhythmically plausible but it is not speaking your script. That ceiling is exactly the gap that joint generation closes.

It is worth being precise about *what kinds of sound* a video-conditioned model can and cannot get right, because it is the cleanest way to understand the approach's reach. The table below sorts common audio events by how strongly the video determines them, which is the same as how well V2A handles them.

| Sound type | Visual determinacy | V2A handles it? | Why |
| --- | --- | --- | --- |
| Impact / footstep / splash | High — a visible onset | Yes, well | Sharp motion onset directly cues the sound's timing and rough character |
| Ambient texture (rain, room tone) | Medium — visible context | Yes | The scene determines the texture; timing is loose so small offsets are inaudible |
| Material-specific foley (glass vs wood) | Medium — appearance cues | Often | The model infers material from texture, but can confuse visually similar materials |
| Off-screen events (door behind camera) | None — not in frame | No | Nothing to condition on; the model omits or mis-places it |
| Music with a specific theme | None — not in pixels | No | Music is a creative choice the video does not specify; use a text prompt instead |
| Dialogue with specific words | None — words not in pixels | No | The words live in a script, not the mouth motion; this is the joint-generation gap |

The pattern is a clean gradient: the more tightly the *visible* signal determines the sound, the better V2A does, and it falls off a cliff exactly where the sound is a *choice* (music, dialogue) rather than a *consequence* of visible physics. That gradient is the single most useful mental model for deciding when V2A is enough — and it is why the next section exists.

## 5. Approach two: joint audio-video generation (one model, one pass)

The headline capability of 2025 was Veo 3 generating *synchronized dialogue, sound effects, and ambient sound in a single pass* alongside the video — not stitched on afterward, but produced by one model that decides the pixels and the audio together. This is **joint audio-video generation**, and it is the approach that closes the dialogue gap, because when the model is generating both the mouth shapes and the speech from the same prompt, it can make them agree by construction.

The architecture is the shared-timeline picture from Section 2 taken to its conclusion: two branches — a video DiT and an audio diffusion (or AR) model — that run on a common diffusion timestep and exchange information through cross-modal attention at multiple depths. Several published joint designs (the MM-Diffusion line is the clearest open example) interleave video blocks and audio blocks and insert cross-modal attention so that, at each layer, video tokens can read the audio being generated and audio tokens can read the video. Because both are sampled from the same noise on the same schedule, the model can *negotiate* a consistent audio-visual event: it can decide "there is a bark at frame 48" once and have both branches commit to it, the video showing the open mouth and the audio producing the bark sound at the overlapping audio frames.

![Matrix comparing video-to-audio, joint audio-video, cascaded, and lip-sync approaches across sync quality, controllability, openness, and example models](/imgs/blogs/audio-and-joint-av-generation-4.png)

Why does joint beat V2A on dialogue and tie-or-beat it on foley? Two reasons. First, *information flow is bidirectional*. In V2A the video is frozen — the audio adapts to it but cannot change it. In joint generation, if the prompt says "she says 'hello' and waves," the model can shape the *mouth motion* to match the *speech it is generating*, because both are still being decided. The visemes and phonemes are co-designed. Second, the model has the *text prompt* for both modalities, so it knows the words. A V2A model only has the silent video; a joint model has "a woman says 'the package arrived' " and generates both the audio of that sentence and the lips forming it. The words live in the prompt, the timing lives in the shared timeline, and the model binds them.

The trade-off is the mirror image of V2A's modularity. A joint model is one big coupled system: you cannot swap the audio model without retraining the joint attention, you cannot run it on external footage (it only does sound for video *it* generates), and it is more expensive to train because you need a large corpus of *aligned* audio-video data — clips where the soundtrack genuinely matches the picture, which is harder to curate than it sounds (stock footage is full of clips with added music that has nothing to do with the visible events, and that data actively teaches the model the wrong thing). The frontier proprietary models (Veo 3, and Sora 2's audio) have the data and the compute to pay for this; the open ecosystem is a step behind on joint generation specifically because the aligned-data and compute bar is high.

Here is a sketch of the joint-block forward pass — the part that distinguishes a joint model from two independent ones. It shows the shared timestep, the per-branch self-attention, and the cross-modal attention with a temporal bias that ties them on the timeline.

```python
import torch
import torch.nn as nn

class JointAVBlock(nn.Module):
    """One layer of a joint audio-video diffusion model.
    Video and audio each self-attend, then attend to each other on a
    shared timeline. The SAME diffusion timestep conditions both branches."""
    def __init__(self, d, n_heads):
        super().__init__()
        self.v_self = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.a_self = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.v2a   = nn.MultiheadAttention(d, n_heads, batch_first=True)  # video reads audio
        self.a2v   = nn.MultiheadAttention(d, n_heads, batch_first=True)  # audio reads video
        self.t_embed = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))

    def forward(self, v, a, t_emb, time_bias):
        # t_emb: shared timestep embedding -> AdaLN-style scale/shift for both.
        sv, bv = self.t_embed(t_emb).chunk(2, dim=-1)
        v = v + sv.unsqueeze(1) * self.v_self(v, v, v, need_weights=False)[0]
        a = a + bv.unsqueeze(1) * self.a_self(a, a, a, need_weights=False)[0]

        # Cross-modal attention with a temporal bias that rewards same-instant
        # audio<->video pairs. `time_bias` has shape (N_a, N_v) (and its
        # transpose for the other direction); it encodes |t_audio - t_video|.
        a = a + self.a2v(a, v, v, attn_mask=time_bias,   need_weights=False)[0]
        v = v + self.v2a(v, a, a, attn_mask=time_bias.t(), need_weights=False)[0]
        return v, a
```

The `time_bias` is the whole game. It is a $(N_a \times N_v)$ matrix, large-negative where audio and video tokens are far apart in time and near-zero where they overlap, so the softmax concentrates each audio token's attention on the video tokens at the same instant. With this bias, "sync" is not something the model has to discover from scratch in the data — it is a structural prior, and the data only has to teach *what* sound goes with *what* motion. Without it, a joint model can still learn sync from a huge aligned corpus, but it is slower to converge and more fragile, which is one reason the open joint models lag: they have neither the proprietary data scale nor, in some early versions, this kind of inductive bias.

## 6. Approach three: cascaded (and why it loses on sync)

For completeness — and because it is a tempting shortcut — there is the **cascaded** approach: run the video, *caption* what happens ("a dog runs across a deck and barks"), and feed that caption to a text-to-audio model. It is appealing because both halves already exist: video captioning is mature, text-to-audio (AudioGen, MusicGen, Stable Audio) is mature, and you avoid training anything new. For *ambient* audio with no sharp events — wind, a city hum, ocean — cascaded is fine, because the timing tolerance is loose; nobody can tell you the wind started 200 ms late.

It falls apart on anything with sharp onsets, and Section 1 already told us why: *the caption has no clock*. "A dog barks" does not say *when*, to the frame, the bark happens. The text-to-audio model places the bark wherever it likes within the clip, and the odds it lands within one frame of the visual bark are essentially zero. You can patch this with extra machinery — emit timestamped captions ("bark at 2.0 s"), detect onsets in the video and force the audio model to align to them — but every patch is reintroducing the per-frame timeline that V2A and joint generation have for free, and doing it through a lossy text bottleneck. The before/after below makes the failure mode concrete: the cascaded path drops the timing on the floor at the captioning step, while the joint path keeps both modalities on one clock end to end.

![Before-and-after comparison contrasting cascaded video-to-text-to-audio generation that loses per-frame timing against joint generation that keeps both modalities on one shared timeline](/imgs/blogs/audio-and-joint-av-generation-6.png)

So the cascaded approach is the right tool for exactly one job: generating *non-synchronous* audio — a music bed, a continuous ambience — where you want maximal *controllability* (you can write and edit the caption precisely) and you do not need frame-level sync. For that, it is actually the *best* approach, because routing through text gives you a clean, editable control surface. For foley and dialogue, it is the wrong tool, and the reason is structural, not a matter of model quality: text is a lossy representation of timing, and you cannot recover from the frame to the microsecond what you threw away at the word.

## 7. Lip-sync and talking heads: a distinct sub-problem

Everything so far treats audio as something you generate *for* or *with* a scene. Talking-head generation flips the conditioning: you are *given* the audio — a recorded or synthesized speech track — and a portrait, and you must generate video where the face says exactly that audio. This is **audio-driven portrait video**, the "say this" use case (a still photo or a reference clip plus a voice track, out comes a video of that person speaking those words), and it is a distinct sub-problem with its own tolerances and its own line of models: Wav2Lip, SadTalker, the diffusion-based EMO and Hallo line, and proprietary systems like VASA-1.

It is distinct for three reasons. First, the *tolerance is tighter*, as Section 1 argued: a viseme one frame off the phoneme does not read as "slightly wrong," it reads as a *different syllable* via the McGurk effect, so lip-sync needs sub-frame alignment, tighter than foley. Second, the *conditioning is dense and structured*: you are not conditioning on "there is speech," you are conditioning on the *phonetic content over time* — the audio has to drive the precise mouth shape (the viseme) for each phoneme, which means the model needs a representation of the audio that exposes phonetic structure, typically features from a self-supervised speech model (Wav2Vec 2.0, HuBERT) rather than a raw codec latent. Third, *identity must be preserved* while only the mouth (and ideally the whole face, for expressiveness) moves — the rest of the person has to stay the same person, which is an identity-preservation problem layered on top of the sync problem.

The modern recipe is a diffusion model conditioned on three things: a reference image or video for identity, a sequence of speech features for the audio content, and often an explicit or implicit *motion* representation (head pose, facial landmarks, or a learned expression code) to control how much the head moves versus how much only the lips move. The EMO/Hallo line of diffusion talking-head models is the clearest recent example — they condition a video diffusion model on audio features and a reference frame and generate expressive talking-head video where the lips, and also the eyebrows and head, move with the prosody, not just the phonemes. The result is far past the old "paste a synced mouth onto a static face" look of Wav2Lip and into territory where the whole face is alive.

Here is the conditioning shape for an audio-driven talking head, sketched to show where the speech features and the identity reference enter. The diffusion loop is the same one you know; what is specific is the *audio feature extraction* that exposes phonetic structure and the *identity* conditioning that keeps it the same person.

```python
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

device = "cuda"
# --- Speech features that expose PHONETIC structure (not a codec latent) ---
sp_proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
sp_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()

speech, sr = torch.load("say_this.pt"), 16000        # 16 kHz mono speech
inp = sp_proc(speech.numpy(), sampling_rate=sr, return_tensors="pt").input_values
with torch.no_grad():
    # (1, T_a, 768): one feature vector per ~20 ms of audio -> ~50 Hz.
    phon = sp_model(inp.to(device)).last_hidden_state

# --- Identity: a single reference portrait (or a short reference clip) -----
ref_img = load_reference("portrait.png")             # (1, 3, H, W), the person

# --- Talking-head diffusion: audio drives the face, ref fixes identity -----
# `talking_head` is a SadTalker/EMO/Hallo-style diffusion model; we resample
# the phonetic features to the VIDEO frame rate so each frame gets the right
# mouth shape, then generate.
phon_at_fps = torch.nn.functional.interpolate(
    phon.transpose(1, 2), size=num_video_frames, mode="linear"
).transpose(1, 2)                                    # (1, T_video, 768)

video = talking_head(reference=ref_img, audio_feats=phon_at_fps,
                     motion_scale=0.4)               # 0 = lips only, 1 = expressive
```

The `motion_scale` knob is worth dwelling on because it is where talking-head models trade realism for safety. Crank it to expressive and the head bobs, the eyebrows move, the whole face emotes with the prosody — gorgeous when it works, but if the model misjudges, the head does something uncanny. Keep it near zero and only the lips move on a still face — safer, more robust, less alive. The frontier diffusion models (EMO, VASA-1) push the expressive end and largely get away with it; the older landmark-driven models (Wav2Lip) live at the safe end by construction. This is the same realism-versus-stability tension that runs through the whole series, here playing out on a single face.

A safety note, because talking-head generation is the most directly misusable capability in this whole post: audio-driven portrait video of a real person saying words they never said is the literal definition of a deepfake. The provenance and watermarking machinery the series covers in the [image-series safety post](/blog/machine-learning/image-generation/safety-watermarking-and-provenance) — C2PA content credentials, invisible watermarks — is not optional decoration here; it is the difference between a useful tool and a weapon, and any production talking-head pipeline should embed provenance at generation time and gate consent at the reference-image step.

## 8. Measuring sync honestly

You cannot ship audio-video generation on vibes, and audio sync is exactly the kind of thing that looks fine in the demos you cherry-picked and falls apart on the average clip. So how do you measure it honestly? Three families of metric, each catching a different failure.

The first is **audio quality in isolation**, ignoring sync entirely. Borrow the audio-generation toolkit: **Fréchet Audio Distance (FAD)**, the audio analog of FVD and FID — embed real and generated audio with a pretrained audio network (VGGish or a more modern PANNs/CLAP encoder) and compute the Fréchet distance between the two feature distributions. Low FAD means your audio *sounds* like real audio of that type. It says nothing about whether it is synced — a beautifully realistic bark that lands two frames late scores great on FAD. This is the audio twin of the lesson from [the metrics of video generation](/blog/machine-learning/video-generation/the-metrics-of-video-generation): a distribution metric measures realism, not correctness, and you must not read it as a sync score.

The second family is **alignment metrics**, which are the ones that actually matter for this post and the hardest to get right. The dominant automatic approach uses a pretrained audio-visual *synchronization* model — the SyncNet lineage, originally built to detect lip-sync errors in TV broadcasts. SyncNet learns a joint embedding of short audio and video windows such that *temporally aligned* windows embed close together and misaligned ones far apart; you slide it over the clip and read off the *audio-visual offset* (in frames) and a *confidence*. For lip-sync you report the **LSE-D** (lip-sync error distance, the embedding distance at the best offset) and **LSE-C** (lip-sync error confidence). For general foley sync, people use the same idea with a foley-trained sync model or report the distribution of estimated offsets — a model that syncs well has offsets tightly clustered near zero, and a model that does not has a fat tail. The honest version of this metric reports the *distribution* of offsets across many clips, not the mean, because the mean can be near zero while half the clips are a frame early and half a frame late.

It is worth seeing how SyncNet actually recovers the offset, because the mechanism explains both its power and its blind spot. The network produces, for each short window, an audio embedding $f_a(\tau)$ and a video embedding $f_v(\tau)$ in a shared space, trained with a contrastive loss so that the matched pair at the *true* alignment has small distance. To find a clip's offset $\delta$ you slide the audio embeddings against the video embeddings and pick the lag that minimizes the average distance:

$$
\hat{\delta} = \arg\min_{\delta} \; \frac{1}{T}\sum_{\tau} \big\lVert f_a(\tau + \delta) - f_v(\tau) \big\rVert_2 .
$$

The minimizing $\delta$ is the estimated audio-visual offset in frames, and the *sharpness* of the minimum (the gap between the best lag's distance and the runner-up's) is the confidence. The power is that this is fully automatic and correlates well with human sync judgments for clips that *have* a clear audio-visual event. The blind spot follows directly from the formula: if the clip has no sharp event — a continuous ambient wash with no onset — every lag gives roughly the same distance, the minimum is flat, the confidence is near zero, and the offset estimate is meaningless. That is exactly the regime where a lazy model hides, which is why you cannot read SyncNet confidence alone as "this model syncs well"; a model that only ever produces flat ambient audio gets an uninformative-but-not-bad score and skates by. The metric has to be paired with event recall, computed only over clips that actually contain a labeled sharp event, or it rewards the wrong behavior.

![Before-and-after comparison showing the same generated frames reading as a muted tech demo when silent versus reading as real footage once on-frame foley and ambient sound are added](/imgs/blogs/audio-and-joint-av-generation-5.png)

The third family is **human evaluation**, and for sync it is non-negotiable, because the perceptual tolerance window from Section 1 *is* a fact about human perception, so the ground truth is literally a human judgment. The honest protocol is a forced-choice or rating study where annotators watch clips and rate "does the sound match the picture?" on a fixed scale, with the clips presented at the *intended* frame rate and audio latency (a study run on a laggy player measures the player, not the model), and with attention checks (a deliberately desynced clip that any honest rater must flag). You report inter-rater agreement, not just the mean, because sync judgments are noisy near the tolerance edge.

The trap to call out, the audio analog of the dynamic-degree-gaming problem from the metrics post: a model can win FAD and even win the SyncNet offset metric while producing audio that is *generic*. If your V2A model learns to emit a soft, plausible, low-energy ambient bed for every clip, it will sync trivially (there are no sharp onsets to misplace) and sound fine in isolation (FAD is happy) — and it will be useless, because it never produces the *specific* sound the scene calls for. The fix is to measure *event recall* too: for clips with known sharp events (the bark, the splash), did the model produce a sharp sound at the right time, or did it smear an ambient wash over the whole clip? A model that aces FAD and sync but flunks event recall has learned to cheat by being boring.

#### Worked example: a sync scorecard for our dog clip

Suppose you run two V2A models on a test set of 500 clips with hand-labeled event times. Model A reports FAD 4.1, median SyncNet offset 0.3 frames, offset standard deviation 1.1 frames, event recall 78%. Model B reports FAD 2.9 (better audio realism), median offset 0.2 frames, offset standard deviation 3.4 frames, event recall 41%. Read naively, B looks better — lower FAD, lower median offset. Read honestly, A is the one you ship: B's offset *standard deviation* of 3.4 frames means a third of its clips are audibly desynced even though the median is fine, and its event recall of 41% means it is producing the right sound less than half the time — it is winning FAD by emitting safe ambient washes. The scorecard that matters is the *spread* of the offset and the *event recall*, not the headline FAD. This is the same "read the metric that catches the failure, not the one that flatters the model" discipline the whole series insists on.

## 9. Case studies: the 2025 audio-video landscape

Let us ground the three approaches in named, shipped systems and be specific about what each actually does. Numbers are as reported in the respective technical reports and model cards; where a figure is approximate or I am inferring from the public description, I say so.

**Veo 3 (Google DeepMind, 2025) — the joint-generation frontier.** Veo 3 was the model that made "native synchronized audio" a headline capability. From a single text (or image) prompt it generates the video *and* a synchronized soundtrack that includes dialogue (characters speaking lines from the prompt with lip-sync), sound effects tied to on-screen events, and ambient sound, all in one generation pass. The reported quality bar is that the dialogue is plausibly lip-synced and the effects land on-frame — the things Sections 5 and 7 say require a shared timeline and bidirectional video-audio information flow. The exact architecture is not public, but the *capability* (one pass, dialogue, sync) is consistent only with a joint model, not a V2A post-process, because V2A cannot produce the right dialogue words. Veo 3 is proprietary and API-gated; it sets the bar the open ecosystem is chasing. We cover it more fully in [Veo and cinematic generation](/blog/machine-learning/video-generation/veo-and-cinematic-generation).

**Meta Movie Gen Audio (2024) — the V2A frontier.** Movie Gen's audio component is a large video-to-audio (and text-to-audio) model: given a generated (or real) video and an optional text prompt, it produces high-quality synchronized sound effects, foley, and ambient audio, plus music, conditioned on the video. It is the strongest *post-hoc* V2A system reported, and it makes the V2A trade-off concrete — it does *not* generate dialogue with specific words (that is not in its scope), but for everything-except-speech it produces cinematic-quality, on-event sound. Architecturally it is the Section 4 picture at scale: a flow-matching audio generator conditioned on video and text features. It is proprietary; the report is the public artifact.

**MMAudio (2024-2025) — the open V2A reference.** MMAudio is the open model the community actually runs for video-to-audio. Its contribution is the *multimodal joint training*: it trains on both audio-visual and text-audio data in a shared representation, so one model handles video-conditioned foley, text-conditioned sound, or both, and the joint training improves sync and quality over training on video-audio alone. It is small enough to run on a single consumer GPU (a 24 GB RTX 4090 comfortably), which is why it is the reference implementation for "I want to add sound to my generated clips today." It does foley and ambient well; like all V2A it does not produce scripted dialogue. MMAudio is the clearest evidence that the open V2A gap to proprietary is *narrow* and closing — the open frontier is roughly one generation behind on foley quality, not categorically behind.

**The talking-head line (Wav2Lip → SadTalker → EMO/Hallo → VASA-1).** This is the audio-driven-portrait lineage. Wav2Lip (2020) established the "paste a synced mouth" baseline with strong lip-sync (high SyncNet confidence) but a frozen, lifeless rest-of-face. SadTalker (2023) added head pose and expression for a more natural result. The diffusion line — EMO (2024), Hallo, and the proprietary VASA-1 (2024) — pushed to *expressive* talking heads where the whole face emotes with the audio prosody, generated by audio-conditioned video diffusion. The open models (SadTalker, Hallo variants) are runnable; VASA-1 is a research demo, not released, explicitly because of deepfake-misuse concerns. The trajectory mirrors the rest of the field: landmark/GAN methods gave way to diffusion, and quality jumped, with the safety stakes rising in lockstep.

For a side-by-side you can read at a glance, here are the four systems against the axes that actually differentiate them. Numbers and capabilities are from the respective reports and model cards; "approx." marks figures I am inferring from public description rather than a stated benchmark.

| System | Year | Mode | Dialogue + lip-sync | Best at | Open? | Runs on |
| --- | --- | --- | --- | --- | --- | --- |
| Veo 3 | 2025 | Joint A/V, one pass | Yes (native) | Synced dialogue + SFX | No (API) | Google cloud |
| Movie Gen Audio | 2024 | Video-to-audio | No | Cinematic foley + music | No (report) | Meta infra |
| MMAudio | 2024-25 | Video-to-audio | No | Open foley + text control | Yes (weights) | One 24 GB GPU (approx.) |
| SadTalker / EMO | 2023-24 | Audio-to-face | Lip-sync only (input speech) | Talking-head portrait | Yes / research | One consumer GPU |

The columns that matter are "Dialogue + lip-sync" and "Open?", because they are correlated: the only systems that produce scripted dialogue with lip-sync in one pass are the closed joint models, and the open systems are all either V2A (no dialogue) or audio-driven faces (dialogue is an *input*, not generated). That correlation is the open-vs-proprietary gap stated as a table.

The matrix below stacks these four against the capabilities that distinguish them, and the tree after it places the whole family in one taxonomy so you can see where any new model fits.

![Matrix placing Veo 3, Movie Gen Audio, MMAudio, and the lip-sync line against mode, sync, dialogue support, and availability](/imgs/blogs/audio-and-joint-av-generation-8.png)

A reasonable order-of-magnitude reading of the open-vs-proprietary gap as of mid-2026: on *foley sync and quality*, open V2A (MMAudio) is close — within a generation — of proprietary V2A (Movie Gen Audio). On *joint dialogue generation with lip-sync in one pass*, the gap is wider, because that capability (Veo 3, Sora 2 audio) needs both the aligned-data scale and the compute to train a coupled joint model, and the open ecosystem has not yet shipped a strong open joint A/V-with-dialogue model. So the honest summary is: *you can add great foley to your open video pipeline today with MMAudio; you cannot yet generate Veo-3-quality synchronized dialogue with open weights.* That is the frontier as it stands, and it is the one capability gap in this post most likely to close in the next year.

![Tree diagram organizing the audio-for-video family into post-hoc video-to-audio, single-pass joint generation, and the separate audio-driven talking-head branch](/imgs/blogs/audio-and-joint-av-generation-7.png)

## 10. When to reach for each approach (and when not to)

Decisions, stated plainly, because the three approaches are genuinely different tools and the right choice is usually obvious once you name the constraint.

**Reach for video-to-audio (MMAudio-style) when** you have a finished clip — generated or real — and want foley, sound effects, and ambient audio that sync to the visible action, and you do not need scripted dialogue. This is the default for *adding sound to an existing video pipeline*, it is open and runnable on a single GPU today, and it is modular — you can swap your video model and your audio model independently. It is also the *only* option when the video is external footage you did not generate. Do not reach for V2A when you need a character to *say specific words* — V2A cannot put words in a mouth, because the words are not in the pixels.

**Reach for joint audio-video generation (Veo-3-style) when** you need *dialogue* — a character speaking scripted lines with correct lip-sync — or when you want the tightest possible event sync and you are generating the video anyway. Joint is the only approach that closes the dialogue gap, because it co-designs the visemes and phonemes. The catch is availability: as of mid-2026 strong joint-with-dialogue generation is proprietary and API-gated, so "reach for it" in practice means "use the API," not "self-host." Do not reach for joint generation when you only need foley on existing footage — that is a V2A job, and V2A is open and cheaper.

**Reach for the cascaded (video → caption → text-to-audio) approach when** you want a *non-synchronous* audio layer — a music bed, a continuous ambience, a mood — where timing tolerance is loose and you want maximal editable control over *what* the audio is via the caption. It is the best tool for that specific job. Do not reach for cascaded for anything with sharp onsets — foley or dialogue — because the text bottleneck destroys the frame-level timing and you will hear the drift.

**Reach for a talking-head / lip-sync model when** the task is specifically *audio-driven portrait video* — you have a voice track and a face and want the face to say it. This is its own model family (SadTalker, Hallo, EMO), with sub-frame tolerance and identity preservation as first-class concerns. Do not use a general joint A/V model for a pure "make this photo say this audio" task — the talking-head models are purpose-built for it and handle identity and phonetic conditioning that a general model does not expose. And do not ship any talking-head pipeline without provenance and consent gating; this is the most misusable capability in the post.

#### Worked example: choosing for the dog clip

Back to the running example, the 5-second clip of the dog running across a deck and barking once. What do you reach for? You generated the video, you want the *bark to land on-frame* and the *paw-falls and ambient deck creak* to sync, and there is no dialogue. That is a textbook V2A job: run MMAudio (or a similar open V2A model) conditioned on the clip, get foley that syncs to the motion, mux with `ffmpeg`. Cost is negligible — a few seconds of audio inference on the same GPU that rendered the video, the audio token budget being under one percent of the video's. If instead the brief were "the owner turns and says 'good boy' " you would need *dialogue*, which V2A cannot produce, so you would reach for a joint model (the Veo 3 API) and accept the proprietary dependency to get the words and the lip-sync. The constraint — foley versus dialogue — picks the tool, and it picks it cleanly.

## 11. Stress-testing the design

A good design survives being pushed past where it was meant to work. Push audio-video generation on the same axes the rest of the series pushes video, and the failure modes are instructive.

**What happens when motion is large and fast between frames?** The visual sync clock gets coarse. If the dog crosses the whole frame in a few frames, the per-frame visual features change so fast that "which frame is the paw-strike" becomes ambiguous even for the video encoder, and the audio onset can land a frame or two off because the conditioning itself is uncertain. The mitigation is the explicit motion signal from Section 4 — frame-difference or optical-flow energy resolves *when* the sharp change happens more precisely than CLIP features do, which is exactly why V2A models add it. Past a point, though (genuine motion blur, sub-frame events), the video itself does not resolve the event time, and no audio model can sync to a time the video does not encode.

**What happens when the audio event has no visible cause?** An off-screen sound — a door slamming behind the camera, thunder from a sky not in frame — has *no visual onset to sync to*. V2A cannot generate it correctly because there is nothing in the pixels to condition on; it will either omit it or place it arbitrarily. Joint generation can do better *if the prompt mentions it*, because then the text supplies the event the video lacks, and the model can place it plausibly even without a visual cue. This is a real, structural limit of V2A and a genuine advantage of joint+text: video-only conditioning can only sync to *visible* events.

**What happens when you roll out to a long video?** Sync has to hold not just within a clip but across the chunk boundaries of a long generation (see [long-video and autoregressive rollout](/blog/machine-learning/video-generation/long-video-and-autoregressive-rollout), if shipped, or the general rollout discussion). The risk is a *discontinuity at the seam* — the audio's ambient bed jumps, or a sound that started in one chunk gets cut off when the next chunk regenerates it. The honest engineering answer is to generate audio with overlap-and-crossfade at chunk boundaries, the audio analog of the latent-overlap trick used for video chunks, so the ambient texture is continuous even though the chunks were generated separately. Get this wrong and the soundtrack stutters at every seam even though each chunk is internally fine.

**What happens when the diffusion timesteps drift apart between branches?** We flagged this in Section 2: if the audio and video branches are not on the *same* denoising timestep, cross-modal attention compares representations at different noise levels and sync turns to mush even though each modality is individually clean. The stress test is to deliberately desync the timesteps and watch the sync metric collapse while FAD and FVD stay fine — a diagnostic that isolates the timestep-sharing bug from a data or capacity problem. The fix is structural: tie the timestep, full stop.

**What happens when the training data is mis-aligned?** The quietest and nastiest failure. Stock video with added music teaches a joint model that "audio" means "unrelated background music," and a model trained on too much of that will *generate* unrelated background music instead of synced foley, scoring fine on FAD (the music is realistic) and terribly on event recall and sync. This is the data-curation lesson the SVD curation work taught for video, transposed to audio: the *alignment* of your training pairs is more important than their volume, and a smaller corpus of genuinely-synced clips beats a huge corpus of music-over-footage. If your joint model produces lovely irrelevant music, suspect the data before the architecture.

## 12. Key takeaways

- **The constraint is timing, not cost.** Audio is under one percent of the video token budget; it is nearly free to generate. The entire difficulty is *alignment* — the human audio-visual fusion window is about 45 ms early to 125 ms late, roughly one video frame, so there is no slack on sync.
- **A shared, per-frame timeline is the validator of whether an approach can sync.** Anything that routes timing through a representation without a clock — a text caption, a pooled embedding — loses the frame-level alignment and drifts audibly. V2A and joint generation keep the timeline; cascaded throws it away.
- **Cross-modal attention with a temporal bias is the mechanism.** A learned (or windowed) bias on $|t_{\text{audio}} - t_{\text{video}}|$ bakes sync in as a structural prior, so the data only has to teach *what* sound goes with *what* motion, not *that* they should align.
- **Tie the diffusion timestep across branches.** If audio and video denoise on different schedules, cross-modal attention compares different noise levels and sync turns to mush even when each modality is individually clean.
- **Video-to-audio is the open, modular default for foley** — runnable on a single 4090 (MMAudio), works on external footage, but cannot produce scripted dialogue because the words are not in the pixels.
- **Joint generation is the dialogue frontier** — one pass, lip-synced speech, tightest sync (Veo 3, Sora 2 audio), but proprietary and data-hungry because it needs genuinely aligned audio-video at scale.
- **Lip-sync is a distinct sub-problem** with a tighter sub-frame tolerance (McGurk effect), dense phonetic conditioning (Wav2Vec/HuBERT features, not a codec latent), and identity preservation as a first-class concern — and the most misusable capability here, so provenance and consent are mandatory.
- **Measure the spread, not the mean.** FAD measures audio realism, not sync; SyncNet offset *distribution* and *event recall* catch the failures a mean offset and a low FAD hide — a model that emits safe ambient washes aces both and is useless.
- **The open gap is narrow on foley, wide on dialogue.** Open V2A is within a generation of proprietary on foley; open joint-with-dialogue does not yet exist at frontier quality, and that is the gap most likely to close next.

## Further reading

- **MMAudio: Taming Multimodal Joint Training for High-Quality Video-to-Audio Synthesis** (2024-2025) — the open V2A reference; the multimodal-joint-training idea (audio-visual + text-audio in one representation) that improves sync and quality, and the model you can actually run.
- **Meta Movie Gen** technical report (2024) — the Movie Gen Audio component: large-scale video-to-audio and text-to-audio generation of synchronized foley, effects, ambience, and music; the proprietary V2A frontier.
- **Google DeepMind Veo 3** model card / report (2025) — native synchronized audio, dialogue with lip-sync, and sound effects in a single generation pass; the joint-generation frontier and the bar the open ecosystem is chasing.
- **MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation** (Ruan et al., 2023) — the clearest open exposition of the coupled two-branch joint-diffusion architecture with cross-modal attention.
- **EMO: Emote Portrait Alive** (2024) and **VASA-1** (2024) — diffusion-based expressive talking-head generation; audio-driven portrait video where the whole face emotes with prosody, and the deepfake-safety reasoning behind not releasing weights.
- **Wav2Lip / SyncNet** (Prajwal et al. 2020; Chung & Zisserman 2016) — the lip-sync baseline and the synchronization-embedding network behind LSE-D / LSE-C, the standard automatic sync metric.
- **High Fidelity Neural Audio Compression (EnCodec)** (Défossez et al., 2022) and **DAC** (Kumar et al., 2023) — the neural audio codecs that produce the latents joint and V2A models generate; the audio analog of the 3D-VAE.
- Within this series: [why video generation is hard](/blog/machine-learning/video-generation/why-video-generation-is-hard), the [conditioning sibling post](/blog/machine-learning/video-generation/conditioning-video-text-image-motion-camera), [the metrics of video generation](/blog/machine-learning/video-generation/the-metrics-of-video-generation), [Veo and cinematic generation](/blog/machine-learning/video-generation/veo-and-cinematic-generation), and the [capstone playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook). For the diffusion basis the audio branch reuses, see [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles) and [safety, watermarking, and provenance](/blog/machine-learning/image-generation/safety-watermarking-and-provenance) in the image series.
