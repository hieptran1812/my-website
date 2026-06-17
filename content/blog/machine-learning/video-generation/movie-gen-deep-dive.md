---
title: "Movie Gen Deep-Dive: Meta's Recipe for Video and Audio at Scale"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "A page-by-page read of Meta's Movie Gen paper: the 30B flow-matching video model on a TAE latent, the separate 13B audio model that syncs sound to the picture, personalization from one reference face, instruction-based editing, and the tiled inference recipe that makes a 16-second 1080p clip tractable."
tags:
  [
    "video-generation",
    "diffusion-models",
    "movie-gen",
    "flow-matching",
    "video-diffusion",
    "text-to-video",
    "audio-generation",
    "generative-ai",
    "deep-learning",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Video Generation"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/movie-gen-deep-dive-1.png"
---

Most frontier video models hand you a glossy demo reel and a one-paragraph "technical report" that tells you almost nothing about how the thing was built. Movie Gen is the exception. When Meta published the Movie Gen work in late 2024, it came with a paper north of ninety pages — close to the most transparent disclosure any lab has made about a production-scale video-and-audio generation system. You get parameter counts, the compression ratio of the autoencoder, the training curriculum stage by stage, the conditioning encoders, the inference tiling strategy, the human-eval protocol, and a frank limitations section. For someone trying to actually understand how these systems are put together rather than admire the output, it is the single richest document in the field. This post is a careful read of it.

The headline numbers set the scale. Movie Gen Video is a **30-billion-parameter** flow-matching Transformer that generates up to **16-second** clips at up to **16 frames per second**, at resolutions reaching **1080p**. Alongside it is **Movie Gen Audio**, a **13-billion-parameter** model that generates synchronized sound effects, ambient sound, and instrumental music conditioned on the video and a text prompt. On top of the base text-to-video model sit two extra capabilities that most competitors did not have in one stack: **personalization**, which keeps a specific person's face across a generated clip from a single reference image, and **precise instruction-based editing**, which applies a text instruction like "make it rain" to an existing video while leaving everything else alone. None of it is openly released — there are no weights to download — but because the recipe is so thoroughly documented, you can rebuild the *shape* of it from open parts, which is exactly what we will do.

![Stack diagram of the Movie Gen pipeline showing a TAE latent feeding a 30B flow-matching DiT, a separate spatial upsampler to 1080p, a 13B audio model, and a final mux into a 16-second clip with sound](/imgs/blogs/movie-gen-deep-dive-1.png)

This post sits in the Frontier Model Reports track of the series, and it leans hard on machinery we built earlier. The video model is a flow-matching Transformer on a learned spatiotemporal latent — the exact pattern from [flow matching for video](/blog/machine-learning/video-generation/flow-matching-for-video) operating on the kind of latent we dissected in [video autoencoders and spatiotemporal compression](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression). The conditioning story is the one from [conditioning video on text, image, motion, and camera](/blog/machine-learning/video-generation/conditioning-video-text-image-motion-camera), and the audio half is the video-to-audio problem we framed in [audio and joint audio-video generation](/blog/machine-learning/video-generation/audio-and-joint-av-generation). If you have read [why video generation is hard](/blog/machine-learning/video-generation/why-video-generation-is-hard), the one-line frame still holds: video equals spatial generation times temporal coherence under a brutal compute budget, and Movie Gen's design is best understood as a series of decisions made to keep that compute budget survivable while still reaching cinematic quality. We will forward-link the comparative [2026 video model landscape](/blog/machine-learning/video-generation/the-2026-video-model-landscape) and the [capstone playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook) where a pipeline like this becomes something you actually run. For the pure flow-matching objective, we link out to the image series at [flow matching and rectified flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow) rather than re-derive it here.

By the end you will be able to read the Movie Gen architecture off a diagram and know why each piece exists, write a flow-matching training step over a video latent, explain why the autoencoder decode rather than the denoiser is the memory wall and how tiling defeats it, describe how the audio model conditions on video features to stay in sync, and place Movie Gen against Sora, Veo, and Kling on the axes that actually differentiate them.

## 1. The shape of the system: five trained components

The first thing the paper makes clear is that "Movie Gen" is not a model. It is a family of trained components stitched into a pipeline, and conflating them is the fastest way to misunderstand the system. There are five pieces worth naming up front, and the rest of this post takes them one at a time.

The first is the **Temporal Autoencoder**, the TAE. This is the learned spatiotemporal compressor that turns pixel video into a compact latent and back. Everything else operates in that latent space, never on pixels, which is the whole reason a 30B model on 16-second clips is even thinkable. The second is **Movie Gen Video** itself, the 30B flow-matching Transformer that denoises TAE latents conditioned on text. The third is a **spatial upsampler** — a separate, smaller model that takes the foundation model's output and increases its resolution, because generating directly at full 1080p would be ruinously expensive. The fourth is **Movie Gen Audio**, the 13B model that scores the finished video with sound. The fifth is really two capabilities layered onto the video model: **personalization** (Personalized Movie Gen Video, which conditions on a reference face) and **precise video editing** (a trained edit model driven by text instructions).

![Matrix table listing the five Movie Gen components with what each one does, its disclosed scale detail, and the output it produces](/imgs/blogs/movie-gen-deep-dive-3.png)

The reason this decomposition matters is economic. If you tried to build one monolithic network that took text and produced a 1080p clip with sound, you would be paying full-resolution, full-length compute through every layer of a giant model, and the autoencoder would have to round-trip raw pixels at the same time. By factoring the problem — generate at a manageable resolution in a compressed latent, upsample with a cheaper dedicated model, add audio in a separate pass — Movie Gen keeps each stage inside a budget that a realistic GPU cluster can serve. This is the same instinct behind latent diffusion in the image world (covered in the image series at [latent diffusion and stable diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion)), pushed harder because the time axis makes everything more expensive. We will keep one running example throughout: a **16-second 1080p clip of a chef searing a steak in a loud kitchen**, generated from text, with the same chef's face supplied as a reference, and the sizzle and clatter generated to match. Watch what each component does to that one clip.

## 2. The Temporal Autoencoder: where the budget is won

Start where the savings come from. The TAE is a variational autoencoder with 3D structure — it compresses jointly across space *and* time. The paper discloses an **8×8×8** compression: a factor of 8 in time, 8 in height, 8 in width. That single ratio is the most important number in the whole system, so it is worth making concrete.

Take our 16-second clip at 16 fps and 1080p. In pixels that is 256 frames of 1920×1080×3 values, which is roughly 1.6 billion numbers. After the TAE compresses by 8 temporally and 8×8 spatially, the temporal dimension shrinks from 256 to 32 latent frames, each spatial dimension shrinks by 8, and the channels go up to whatever latent width the TAE uses (on the order of a dozen). The latent tensor has on the order of a few million values rather than 1.6 billion — a compression of several hundred times in raw element count. Every forward pass of the 30B denoiser runs on *that* tensor, not on pixels. The cost of attention scales with the square of the token count, so a few-hundred-times reduction in tokens is the difference between a model you can train and one you cannot.

Let me make the token-count argument precise, because it is the lever the whole field pulls. If the latent is patchified into spacetime tokens of size $p_t \times p_h \times p_w$, the number of tokens for a latent of shape $T' \times H' \times W'$ is

$$
N = \frac{T'}{p_t} \cdot \frac{H'}{p_h} \cdot \frac{W'}{p_w}.
$$

The dominant cost in the Transformer is full spatiotemporal self-attention, which is $\mathcal{O}(N^2 d)$ for hidden width $d$. Now substitute the TAE's compression. Without compression, $T' = T$, $H' = H$, $W' = W$. With $8 \times 8 \times 8$ compression, each of those is divided by 8, so $N$ is divided by $8^3 = 512$, and the attention cost — being quadratic — is divided by $512^2 \approx 2.6 \times 10^5$. That is the entire game. The autoencoder is not a preprocessing convenience; it is the component that converts an intractable attention bill into a tractable one. This is exactly the argument we made in detail in [video autoencoders and spatiotemporal compression](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression), and Movie Gen is a clean instance of it.

The "Temporal" in the name is doing real work. A naive approach would compress each frame independently with an image VAE and stack the latents, which throws away temporal redundancy — the fact that consecutive frames are mostly the same. The TAE compresses *across* frames, so a slow pan or a static background costs almost nothing in latent terms because the temporal convolutions and the temporal downsampling absorb the redundancy. The paper trains the TAE with the usual VAE machinery (a reconstruction loss, a KL regularizer to keep the latent space well-behaved, and a perceptual/adversarial term so decoded video looks sharp rather than blurry), plus care around the temporal boundaries so that the encoder is effectively causal-friendly and clips can be tiled at inference. We will come back to that tiling because it is where the memory wall actually lives.

#### Worked example: how many tokens does the chef clip cost?

Take the latent at $32 \times 135 \times 240$ (the 256-frame, 1080p clip after $8\times8\times8$ compression, before any further patching). Suppose the denoiser patchifies with $p_t=1, p_h=2, p_w=2$. Then $N = 32 \cdot \frac{135}{2} \cdot \frac{240}{2} \approx 32 \cdot 67 \cdot 120 \approx 257{,}000$ tokens. That is already a quarter-million-token sequence with full attention — which is why Movie Gen does **not** generate at 1080p directly. It generates at a lower spatial resolution (the paper's foundation model targets around 768 pixels on the long side) and leaves the final climb to 1080p to the separate upsampler. Run the same arithmetic at 768px and the token count drops by roughly the square of the resolution ratio, into the tens of thousands, which is a sequence a 30B Transformer can attend over for dozens of denoising steps without the cost exploding. The decision to upsample separately falls directly out of this token-count math.

## 3. Movie Gen Video: a Llama-style flow-matching Transformer

Now the denoiser. Movie Gen Video is a Transformer, and a notable design choice the paper highlights is that the backbone borrows heavily from the **Llama** Transformer architecture rather than inventing a bespoke video block. Meta had a battle-tested, well-optimized large-language-model Transformer in house, and they adapted it to operate on spacetime tokens. That means the familiar ingredients — RMSNorm, SwiGLU feed-forward layers, rotary-style positional handling adapted to the spatiotemporal grid — show up in a video model. The practical payoff is that all the systems work that went into training Llama efficiently at scale (tensor and sequence parallelism, attention kernels, mixed precision) transfers, which matters enormously when you are training a 30B model on sequences of tens of thousands of tokens.

![Graph showing text encoders and an optional first-frame image both conditioning a 30B Llama-style flow-matching backbone that produces a velocity field, solved to a clean latent and decoded by the TAE](/imgs/blogs/movie-gen-deep-dive-2.png)

The training objective is **flow matching**, in the rectified-flow style. We covered this thoroughly in [flow matching for video](/blog/machine-learning/video-generation/flow-matching-for-video) and the underlying derivation lives in the image series at [flow matching and rectified flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow), so here is just the shape of it. You define a straight-line interpolation between a clean latent $x_1$ (real video, encoded by the TAE) and a noise sample $x_0 \sim \mathcal{N}(0, I)$:

$$
x_t = (1-t)\, x_0 + t\, x_1, \qquad t \in [0, 1].
$$

The velocity along that straight path is constant and equal to $v = x_1 - x_0$. The model $u_\theta(x_t, t, c)$ — conditioned on the text/image condition $c$ — is trained to regress that velocity:

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t,\, x_0,\, x_1,\, c}\left[\,\big\| u_\theta\big((1-t)x_0 + t x_1,\ t,\ c\big) - (x_1 - x_0)\big\|^2\,\right].
$$

That is the whole training signal. No trajectory is ever simulated during training; you sample a random $t$, form $x_t$ by linear interpolation, and regress the constant velocity. At inference you run an ODE solver from $t=0$ to $t=1$ along the learned velocity field, which because the paths are nearly straight needs relatively few steps. The reason flow matching, rather than classic DDPM noise-prediction, is the objective of choice for video is laid out in the sibling post: the objective is indifferent to how large the latent is, it trains stably at the absurd token counts video produces, and its straight paths let you spend fewer of the very expensive sampling steps. Movie Gen is one of the models that made this the consensus.

A subtle but important detail the paper raises is the **timestep schedule shift**. High-resolution, long-duration latents have more tokens and more signal, and you have to push the training timestep distribution toward higher noise levels to compensate, otherwise the model spends too much of its capacity on the easy, low-noise end. This is the same shift Stable Diffusion 3 introduced for images and that scales with token count for video. Movie Gen applies it as part of training the foundation model at its target resolution. The reader who wants the derivation of why the shift scales with token count will find it in [flow matching for video](/blog/machine-learning/video-generation/flow-matching-for-video); the takeaway here is that it is not a free hyperparameter, it is a consequence of the latent size.

Here is a flow-matching training step over a video latent, in the idiom you would actually write it in PyTorch. It is deliberately close to what the real training loop does, minus the distributed-training scaffolding.

```python
import torch
import torch.nn.functional as F

def flow_matching_step(model, tae, batch, text_embeds):
    # batch: pixel video [B, C, T, H, W]; text_embeds: [B, L, D]
    with torch.no_grad():
        # Encode pixels to the TAE latent. This is x1, the clean target.
        x1 = tae.encode(batch).latent_dist.sample()        # [B, c, T', H', W']

    B = x1.shape[0]
    x0 = torch.randn_like(x1)                              # noise sample

    # Sample a timestep per example. A shifted schedule pushes mass
    # toward higher noise for big latents; here we draw from a shifted
    # logit-normal as the SD3 / Movie Gen style training does.
    t = torch.sigmoid(torch.randn(B, device=x1.device))   # in (0, 1)
    t_ = t.view(B, 1, 1, 1, 1)

    x_t = (1.0 - t_) * x0 + t_ * x1                        # straight-path point
    target_v = x1 - x0                                     # constant velocity

    # Predict the velocity; condition on text (and optionally a first frame).
    pred_v = model(x_t, t, encoder_hidden_states=text_embeds)

    return F.mse_loss(pred_v, target_v)
```

Two things to notice. First, the TAE encode is under `no_grad` — the autoencoder is frozen while the denoiser trains, which is standard latent-diffusion practice. Second, the loss is a plain MSE on the velocity. All the difficulty of video — the temporal coherence, the motion, the length — is carried by the model's spacetime attention and by the data, not by a complicated objective. The objective is the simplest part. That is by design.

A detail worth surfacing is how the condition $c$ actually enters the backbone. Movie Gen does not concatenate text tokens into the same self-attention as the video tokens (that would inflate the already-large sequence); it injects the text condition through **cross-attention**, where the video tokens are the queries and the encoded text tokens (from T5 and the long-context MetaCLIP) are the keys and values. The timestep $t$ enters through a separate path — a learned embedding that modulates the normalization layers, the adaptive-layernorm trick from DiT — so the model knows how much noise it is currently looking at. This is the standard DiT-with-cross-attention recipe ([diffusion transformers DiT](/blog/machine-learning/image-generation/diffusion-transformers-dit) in the image series), and Movie Gen's contribution is not a novel block but the scaling of a known-good block to 30B on spacetime tokens. The lesson is one the series keeps returning to: the architectural novelty in frontier video models is modest; the engineering — the data, the curriculum, the systems work, the inference tiling — is where the quality comes from.

## 4. Spatiotemporal attention: where the FLOPs actually go

Before the curriculum, it is worth being precise about the most expensive operation in the whole denoiser: attention over spacetime tokens. This is the part of the model that does the real work of temporal coherence — it is how a token representing one patch at frame 10 learns what is happening at the same location at frame 30 — and it is also where the compute bill is dominated. Understanding it is what lets you reason about why Movie Gen makes the resolution and length choices it does.

Recall from section 2 that the patchified latent yields $N$ spacetime tokens, and full self-attention over them costs $\mathcal{O}(N^2 d)$. The question that decides architectures is whether you do **full** spatiotemporal attention — every token attends to every other token, across both space and time at once — or you **factorize** it into a spatial pass (tokens attend within their frame) followed by a temporal pass (tokens attend across frames at the same spatial location). We dissected this trade in detail in [spatiotemporal attention patterns](/blog/machine-learning/video-generation/spatiotemporal-attention-patterns); here is the cost arithmetic that matters for reading Movie Gen.

Let the latent have $T'$ temporal positions and $S = H' \cdot W'$ spatial positions per frame, so $N = T' \cdot S$. Full attention costs

$$
C_{\text{full}} = \mathcal{O}\big((T' S)^2 \, d\big) = \mathcal{O}\big(T'^2 S^2 \, d\big).
$$

Factorized attention splits into a spatial term and a temporal term:

$$
C_{\text{fact}} = \underbrace{\mathcal{O}\big(T' \, S^2 \, d\big)}_{\text{spatial, per frame}} + \underbrace{\mathcal{O}\big(S \, T'^2 \, d\big)}_{\text{temporal, per column}}.
$$

The ratio of full to factorized is roughly $\frac{T' S}{T' + S}$, which for a clip with many frames and many spatial tokens is a large number — factorizing can cut the attention cost by an order of magnitude or more. The price is coherence: full attention lets a token at frame 10 directly attend to a *different spatial location* at frame 30 (an object that moved), while factorized attention only connects same-location-across-time and same-frame-across-space, so it has to route cross-location-cross-time information through multiple layers. For large, fast motion this is a real quality cost; for the moderate motion of most clips it is an acceptable trade for the FLOP savings.

Movie Gen, at 30B with a generation resolution kept modest precisely so that attention stays affordable, leans toward the full-attention end where coherence is best, and pays for it by *not* generating at 1080p directly — the upsampler, not factorization, is how it controls the high-resolution cost. That is the key design read: Movie Gen buys coherence with full spacetime attention at a tractable resolution, and buys resolution with a separate cheap upsampler, rather than buying both at once by factorizing attention. Different models make this trade differently, and knowing the arithmetic above lets you see which lever each one pulled.

Here is what a factorized spatiotemporal attention block looks like in PyTorch, so the trade is concrete rather than abstract. This is the block Movie Gen *could* have used and that many open models do use; reading it makes clear exactly which connections factorization keeps and which it drops.

```python
import torch
import torch.nn as nn

class FactorizedSpacetimeAttn(nn.Module):
    """Spatial attention within each frame, then temporal across frames."""
    def __init__(self, dim, heads):
        super().__init__()
        self.spatial = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.temporal = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x):                       # x: [B, T, S, D]
        B, T, S, D = x.shape

        # 1) Spatial pass: each frame attends within itself.
        xs = x.reshape(B * T, S, D)             # batch the frames
        xs, _ = self.spatial(xs, xs, xs)
        x = xs.reshape(B, T, S, D)

        # 2) Temporal pass: each spatial column attends across time.
        xt = x.permute(0, 2, 1, 3).reshape(B * S, T, D)
        xt, _ = self.temporal(xt, xt, xt)
        x = xt.reshape(B, S, T, D).permute(0, 2, 1, 3)
        return x
```

Notice what the temporal pass connects: position $(h, w)$ at time $t$ to position $(h, w)$ at time $t'$ — the *same* spatial location across frames. An object that moves from $(h_1, w_1)$ at frame 10 to $(h_2, w_2)$ at frame 30 is never directly connected in one block; the information has to flow spatially in one layer and temporally in the next, taking two layers to cross what full attention crosses in one. That two-layer detour is exactly the coherence cost of factorization, and it is why models that can afford full attention — like Movie Gen at its chosen resolution — tend to use it.

## 5. The training curriculum: text-to-image, then text-to-video

A 30B video model is not trained in one shot on video. The paper describes a multi-stage curriculum that is worth internalizing because it is how nearly every serious video model is now trained, and Movie Gen documents it unusually clearly.

The first stage is **text-to-image**. Before the model ever sees a moving frame, it is pre-trained as an image generator on a large image corpus. This is enormously efficient: images are single frames, so the token count per example is small, and the model learns the hard part — how to render a coherent, prompt-aligned still — cheaply, on far more examples than you could afford in video. A frame of a steak searing is the same rendering problem whether or not it is part of a clip. By the time the model has mastered text-to-image, it already knows objects, materials, lighting, and composition.

The second stage is **text-to-video**, where the temporal dimension is switched on and the model learns motion and coherence on top of the spatial competence it already has. The paper runs this as a curriculum over resolution and length: **low resolution before high, short before long, image before video**. Early in training the model sees small, short clips, which are cheap, and it learns the basics of motion. As training proceeds the resolution climbs and the clip length grows, concentrating the expensive high-token-count compute on a model that already knows what it is doing. This staged approach is not optional polish — it is the difference between a training run that converges within budget and one that wastes its most expensive compute teaching a model things it could have learned for a hundredth of the cost on images.

Data quality is the other half of the curriculum. The paper emphasizes heavy **curation and captioning**. Raw web video is mostly useless for training a high-quality generator — it is full of cuts, low motion, watermarks, text overlays, and junk. Movie Gen filters aggressively (scene-cut detection to get single continuous shots, motion filtering to remove static clips, aesthetic and resolution thresholds) and then re-captions the surviving clips with a dedicated captioning model so the text-video pairs are dense and accurate rather than relying on noisy alt-text. The conditioning encoders are themselves notable: the paper uses **T5** for fine-grained text understanding and a long-context **MetaCLIP** variant, combining a language-model text encoder's compositional grasp with a contrastively-trained encoder's visual grounding. Better captions and better text encoders are, empirically, among the highest-leverage things you can do for prompt adherence, and Movie Gen spends real effort there.

It is worth dwelling on *why* the captioning matters so much, because it is the least glamorous and most decisive part of the recipe. A model can only learn the mapping from text to video as densely as its captions describe the video. If a clip of our chef is captioned "cooking," the model learns to associate that one word with everything in the frame and learns nothing precise about searing, steam, the close-up framing, or the kitchen clatter implied by the scene. If the same clip is re-captioned "a close-up of a chef searing a steak in a cast-iron pan, steam rising, busy restaurant kitchen behind," the model gets a dense supervision signal that ties specific words to specific visual content, and at inference it can be controlled with that same vocabulary. The scene-cut filtering is equally load-bearing: a training clip that contains a hard cut teaches the model that the scene can teleport mid-clip, which is exactly the incoherence you are trying to eliminate. Filtering to single continuous shots is how you teach the model that within a clip the world is continuous. None of this is novel research — it is data engineering — but it is where a large fraction of the perceived quality gap between models actually comes from, and Movie Gen's willingness to document it is part of what makes the paper valuable.

The **motion filtering** has a subtler purpose than just removing static clips. Video models have a well-known failure mode where they learn to minimize the training loss by *barely moving* — a nearly-static clip is easy to predict and scores low reconstruction error, so a model trained on a corpus heavy with low-motion footage learns to produce the visual equivalent of a slideshow. This is the same dynamic-degree-versus-stability tension that plagues VBench scoring. Curating for adequate motion in the training data is how you push the model toward generating real dynamics rather than coasting on near-static frames. It is a data-side fix for a problem that looks like a modeling problem, which is a recurring theme in this field: the lever you reach for is often the dataset, not the architecture.

#### Worked example: why image pre-training pays for itself

Suppose a video example costs, in attention FLOPs, roughly the square of its token count, and a 16-frame clip latent has about 16 times the tokens of a single-frame latent at the same spatial resolution. Then one short-clip step costs on the order of $16^2 = 256$ times a single image step in attention. If your model needs, say, a billion image-equivalent gradient steps' worth of rendering competence before it is any good, paying for that competence in *video* examples would cost two-plus orders of magnitude more compute than paying for it in *images*. The text-to-image-first curriculum is how you buy the spatial competence at image prices and reserve the expensive video compute purely for learning motion. The numbers are illustrative, but the ratio is real, and it is why every frontier video model starts from an image model.

## 6. The decode wall: why the TAE decodes in tiles

Here is a failure mode that surprises people the first time they hit it, and the Movie Gen paper is honest about it. The component that runs out of memory at inference is usually **not the 30B denoiser** — it is the **TAE decoder**. The denoiser works in the compressed latent, where the tensors are small. But to produce pixels you have to decode the *entire* latent back to full-resolution video, and a 16-second clip's worth of pixel activations through the decoder's convolutions is a huge tensor. Decode the whole thing in one pass and you OOM, even on an 80GB GPU, while the denoiser that produced the latent ran comfortably.

![Before and after comparison contrasting a full one-pass TAE decode that runs out of memory with a tiled space-time decode that uses constant memory and blends overlapping tiles into a seamless result](/imgs/blogs/movie-gen-deep-dive-4.png)

The fix is **tiling**, in both space and time. Instead of decoding the latent as one block, you cut it into overlapping tiles along the temporal axis (and, if needed, the spatial axes), decode each tile independently with bounded memory, and blend the overlapping margins back together so there is no visible seam. The overlap is essential: a hard cut between tiles would produce a discontinuity at the boundary — a flicker every few frames where one tile ends and the next begins — so you decode a margin of extra frames on each side and cross-fade. This converts decode from an $\mathcal{O}(\text{clip length})$ memory cost into an $\mathcal{O}(\text{tile size})$ cost, which is constant regardless of how long the clip is. It is the single technique that lets a fixed GPU decode an arbitrarily long video.

In `diffusers` this is exposed directly on video VAEs, and it is the first knob you reach for when a video pipeline OOMs at decode. The idiom is the same across the open video models:

```python
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()      # keep only the active module on the GPU

# These two are the decode-wall fix: the VAE decodes in tiles/slices
# instead of one giant pass, trading a little wall-clock for bounded VRAM.
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

video = pipe(
    prompt="a chef searing a steak in a loud kitchen, close up, steam rising",
    num_frames=49,
    guidance_scale=6.0,
    num_inference_steps=50,
).frames[0]

export_to_video(video, "steak.mp4", fps=16)
```

The structural lesson generalizes well beyond Movie Gen: when you profile a video pipeline's peak memory, look at the decode, not just the denoise. The denoiser's memory is bounded by the latent and the attention window; the decoder's memory is bounded by the *pixel* tensor, which is hundreds of times larger. Tiling the decode is almost always how you fit a long clip on the GPU you have, and Movie Gen's inference recipe builds it in.

## 7. The spatial upsampler: reaching 1080p without paying for it

We already saw the reason in section 2: generating directly at 1080p means a quarter-million-token sequence with quadratic attention, which is not affordable for a 30B model across dozens of denoising steps. Movie Gen's answer is to generate at a manageable resolution and then climb to 1080p with a **separate spatial upsampling model**. This is a dedicated network — the paper describes a spatial super-resolution model in the few-billion-parameter range — that takes the foundation model's lower-resolution output and produces a high-resolution version, conditioned on the low-resolution frames so it adds plausible detail rather than hallucinating something inconsistent.

Why a separate model rather than just generating bigger? Because the two jobs have different cost profiles. The expensive, slow, high-parameter foundation model does the hard creative work — deciding *what* happens, the motion, the scene — at a resolution where attention is affordable. Upsampling is a comparatively local, well-conditioned problem: you already have the content, you just need more pixels. A smaller, cheaper model conditioned on the low-resolution video can do that climb at a fraction of the cost of running the full 30B model at native 1080p. The separation is the same compute-tractability logic that drove the TAE and the audio split: factor the problem so each stage runs inside a budget.

There is a temporal-consistency subtlety here that separates a good video upsampler from a naive one. If you upsampled each frame independently with an image super-resolution model, the added high-frequency detail would *flicker* — the model would invent slightly different fine texture on each frame, and that per-frame inconsistency reads as shimmer on edges and surfaces, one of the most distracting artifacts in generated video. A video upsampler has to add detail *consistently across frames* so that the high-frequency content it hallucinates is stable through the clip. This is why the upsampler is conditioned on the video as a spatiotemporal whole rather than frame-by-frame, and why it cannot simply be an off-the-shelf image SR model bolted on. The same temporal-coherence demand that drives the rest of the stack applies to the last stage too.

The upsampler also tiles, for the same memory reason the decoder does — at 1080p the activation tensors are large, so it processes the video in chunks, with overlap to keep tile boundaries seamless. The net inference path for a full-quality clip is therefore: flow-matching sampling in the TAE latent, tiled TAE decode to lower-resolution pixels, spatial upsample to 1080p with its own tiling, and finally the audio pass. Each step is a separate model with its own memory profile, and that is precisely what makes the whole thing serveable.

```python
# Conceptual cascade mirroring the Movie Gen video path.
# (Open stand-ins; Movie Gen's own weights are not released.)
import torch

def generate_clip(prompt, fm_dit, tae, upsampler, audio_model, ref_face=None):
    # 1) Flow-matching sample in the compressed TAE latent space.
    latent = fm_dit.sample(
        prompt=prompt,
        ref_face=ref_face,          # None -> plain T2V; image -> personalized
        num_steps=50,
        shape=(32, 96, 96),         # T' x H' x W' at the generation resolution
    )

    # 2) Tiled decode back to pixels at the generation resolution.
    tae.enable_tiling()
    frames_lo = tae.decode(latent)  # e.g. ~768px on the long side

    # 3) Separate spatial upsampler climbs to 1080p (also tiled).
    frames_hi = upsampler(frames_lo, tile=True)

    # 4) Audio is a separate conditioned pass (next section).
    audio = audio_model(frames_hi, prompt=prompt)

    return mux(frames_hi, audio)     # ffmpeg-style mux into one file
```

That `generate_clip` function is the whole system in miniature. It is conceptual because the real weights are not public, but the *shape* — sample in latent, tiled decode, separate upsample, separate audio, mux — is exactly the disclosed recipe, and each stage maps onto an open stand-in you can run today.

## 8. Movie Gen Audio: scoring the picture

Now the second model. **Movie Gen Audio** is a 13B model whose job is to generate a soundtrack for a finished video: synchronized sound effects, ambient sound, and instrumental music, conditioned on the video and an optional text prompt. This is the **video-to-audio** (V2A) problem, and we framed why it is uniquely unforgiving in [audio and joint audio-video generation](/blog/machine-learning/video-generation/audio-and-joint-av-generation). The short version: audio and video run on clocks that differ by a factor of thousands — 16 fps video is one frame every 62 milliseconds, while 48 kHz audio is one sample every 21 microseconds — and humans are violently sensitive to misalignment. A footstep two frames late reads as fake even when every other aspect is perfect. The engineering problem is timing.

![Graph showing a finished video and an audio text prompt both encoded, the video into per-frame visual features, both feeding a 13B audio flow-matching model that produces a 48 kHz waveform muxed back onto the original video](/imgs/blogs/movie-gen-deep-dive-5.png)

Movie Gen Audio solves the timing problem the way V2A models do: it conditions on **per-frame visual features** extracted from the video, so the audio model can see *when* events happen and place the corresponding sound on the right frame. The visual condition is not a single global summary of the clip — that would lose all timing — but a sequence of features aligned to the video timeline, so the audio model knows the paw hits the deck at frame 40, not frame 50. Combine that with a text prompt that specifies *what* the sound should be ("sizzling steak, kitchen clatter, low background music"), and the model generates a 48 kHz waveform that is both correct in content and aligned in time. The paper also handles the **ambient + SFX + music** mix, which is why a single pass can produce a soundtrack that feels like a real scene rather than a single sound effect.

Architecturally, the audio model is the same family as the video model: it operates on a compressed audio latent (a neural codec or learned audio autoencoder), and it is trained with a diffusion/flow-matching objective to denoise that latent conditioned on the video features and text. It is a separate model from the video generator, which keeps each tractable and lets the audio model be trained on its own large audio-video corpus. The price of separation is that audio is generated *after* the video is final — a video-to-audio cascade rather than a single joint pass — but for the kind of sound Movie Gen targets (SFX, ambience, instrumental scoring rather than spoken dialogue tightly locked to lips), conditioning on dense per-frame visual features gets the sync tight enough.

Why a 13B model at all, when the audio latent is tiny compared to the video latent? The answer is that the *mapping* is hard even when the output is small. To score our kitchen clip correctly the model has to recognize, from the visual features alone, that the steak hits the pan at one moment (sizzle starts there), that a pan is set down at another (a clatter there), and that the whole scene wants a low ambient kitchen hum underneath — and it has to render all three as a single coherent waveform whose every transient lands on the right frame. That is a rich audio-visual understanding problem, and understanding takes parameters even when the thing you emit is a few hundred audio latent vectors. The asymmetry between the 30B video model and the 13B audio model is therefore not "audio is half as important" but "audio output is cheap, audio-visual understanding is not."

#### Worked example: how you would measure sync honestly

Suppose you want to claim Movie Gen Audio syncs to within one frame. You cannot eyeball it on a few demos. The honest protocol is an **onset-alignment** measurement: take a held-out set of clips with clear visual events (a ball bounce, a hand clap, a door slam), run V2A, then for each clip detect the audio onset (an energy-derivative peak) and the visual event frame (a motion or contact detector), and compute the signed offset in milliseconds. Report the distribution — median offset and the fraction within $\pm 1$ frame ($\pm 62$ ms at 16 fps) — not a single number, because a model can have a tiny median offset and still have a long tail of badly-missed events. A model that is on-frame for 90% of clear onsets and within two frames for the rest is genuinely synced; a model that is on-frame on average but has a wide spread is not, and only the distribution reveals the difference. This is the same "measure the distribution, not the mean" discipline that the [metrics post](/blog/machine-learning/video-generation/the-metrics-of-video-generation) argues for on the video side.

Here is a conceptual V2A pass with an open stand-in, and the `ffmpeg` mux that puts the sound back on the picture. The open model here is a placeholder for the kind of V2A model you would actually wire in; the *interface* — frames in, text in, waveform out, then mux — mirrors Movie Gen Audio.

```python
import subprocess
import torch

def add_audio(video_path, prompt, v2a_model, sr=48000):
    # Extract per-frame visual features the audio model conditions on.
    frames = read_frames(video_path)                  # [T, C, H, W]
    vis_feats = v2a_model.encode_video(frames)        # [T, D] aligned to time

    # Generate a synced waveform conditioned on visuals + text.
    waveform = v2a_model.generate(
        video_features=vis_feats,                     # carries the timing
        text=prompt,                                  # carries the content
        sample_rate=sr,
    )                                                  # [num_samples]
    save_wav("score.wav", waveform, sr)

    # Mux audio onto the original video without re-encoding the picture.
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-i", "score.wav",
        "-c:v", "copy", "-c:a", "aac", "-shortest", "out_with_audio.mp4",
    ], check=True)
    return "out_with_audio.mp4"
```

The `-c:v copy` flag matters: you do not re-encode the video, you only attach the audio stream, so the picture is byte-for-byte the one the generator produced and only the soundtrack is new. That is the entire V2A mux pattern, and it is what sits at the end of the Movie Gen pipeline.

## 9. Personalization: keeping a face across a clip

The first capability that pushes Movie Gen past plain text-to-video is **personalization** — Personalized Movie Gen Video, PT2V. The problem it solves is concrete: text-to-video gives you a *plausible* person, but a different one every generation, and you cannot ask "make a video of *this specific* person" from text alone. Personalization conditions the generator on a **single reference image of a face** and produces a video in which that identity is preserved while the rest of the clip follows the text prompt.

![Before and after comparison showing plain text-to-video producing a random identity that changes every reprompt versus a personalized and edited path that locks a reference face and applies an instruction while preserving the rest](/imgs/blogs/movie-gen-deep-dive-6.png)

The mechanism, as the paper describes it, is a fine-tuned variant of the foundation model that takes the reference face as an additional conditioning input alongside the text. The training setup is the interesting part: you need pairs where the *same* identity appears, so that the model learns to copy identity from the reference into a *different* generated context rather than just regurgitating the reference frame. Done naively, an identity-conditioned model degenerates into copy-paste — it reproduces the reference pose and background instead of generating a new scene with that face. The training has to encourage disentangling *who* (from the reference) from *what they are doing and where* (from the text), so that the reference contributes identity and the prompt contributes everything else. The result is a model where you supply a face and a prompt — "this person cooking a steak in a loud kitchen" — and get a clip that is recognizably that person doing the new thing.

This is the personalization analog of the subject-driven generation we know from images (DreamBooth-style identity injection), but applied to video and built into the foundation model rather than bolted on per-subject. The key practical difference from image personalization is that you only need *one* reference image at inference, not a fine-tuning run per identity, because the personalization capability is trained into the model. From the user's side it is a single extra input. From the training side it is a substantial effort to assemble identity-consistent data and a conditioning path that copies identity without copying context.

The hard part is the *temporal* dimension of identity, which is what makes video personalization meaningfully harder than image personalization. In a still image you have to get the face right once. In a sixteen-second clip you have to keep it right across hundreds of frames while the head turns, the lighting changes, expressions shift, and the person moves through the scene. An identity that is perfect on frame one but drifts by frame two hundred — slowly morphing into a different person, the classic identity-drift failure — is worse than useless, because the drift is exactly what the human eye is tuned to catch on a face. So the personalization conditioning cannot just inject the reference once at the start; it has to keep the identity anchored throughout the clip, which is why it is built into the conditioning path of the foundation model rather than applied as a one-time initialization. This connects to the long-video identity-drift problem we treat in [long video and autoregressive rollout](/blog/machine-learning/video-generation/long-video-and-autoregressive-rollout) — keeping an identity stable over time is a coherence problem, and it is the same coherence problem whether the identity is supplied by a reference or invented by the model.

## 10. Precise editing: instructions, not masks

The second capability beyond plain T2V is **precise video editing**. The goal: take an existing video and a text instruction — "add fireworks to the sky," "make it rain," "turn the chef's apron blue" — and apply exactly that change while leaving everything else untouched. The hard constraint is *locality*: a good edit changes only what the instruction asks and preserves the rest of the clip frame-for-frame, including the parts the edit does not mention and the temporal coherence of the whole thing.

The naive approaches all fail in instructive ways. Re-prompting from scratch gives you a different video, not an edit. Mask-based editing requires the user to draw a mask, which is tedious and brittle for video because the mask has to track the moving region across every frame. Movie Gen instead trains a dedicated **instruction-based edit model**: it learns, from data, to take a source video plus a text instruction and produce the edited video, with no mask required. The model has to infer *where* the edit applies from the instruction and the content, apply it consistently across all frames so the change does not flicker or drift, and leave the untouched regions pixel-faithful.

The training challenge here is data, again. There is no large natural corpus of (source video, instruction, edited video) triples — you cannot scrape "before and after" video edits at scale. The paper describes building this supervision, including leveraging image-editing capabilities and the model's own generation to construct training pairs where the instruction, the source, and the correct edited result are all known. The payoff is an editing interface that feels like talking: you describe the change in words and the model applies it, preserving identity, motion, and the rest of the scene. Combined with personalization, you get the two-step our running example wants — supply the chef's face, generate the kitchen clip, then issue "make the steak flames bigger" as an edit — and each step composes without redoing the whole clip.

#### Worked example: the cost of an edit vs a regenerate

Say a full generate-and-upsample-and-score pass for our 16-second clip costs roughly one unit of compute. A re-prompt to change one detail costs another full unit *and* gives you a different clip — the chef's pose, the kitchen layout, everything shifts, because text-to-video is not deterministic across prompts. The trained edit model, by contrast, conditions on the *existing* clip and changes only the targeted region, so it (a) costs on the order of a single conditioned pass rather than a fresh generation plus the risk of regenerating everything wrong, and (b) preserves the parts you wanted to keep. For any iterative workflow — generate once, then refine — the edit model is the difference between converging on the clip you want and spinning on a slot machine. That is the practical argument for training a dedicated edit head rather than telling users to re-prompt.

## 11. The inference recipe, end to end

Pull the pieces together and the full inference path is a fixed ladder, not a single forward pass. The paper's recipe, in order: flow-matching sampling in the TAE latent space (an ODE solve over the learned velocity field, on the order of tens of steps), a tiled TAE decode to pixels at the generation resolution, a separate spatial upsample to 1080p with its own tiling, and finally the Movie Gen Audio pass conditioned on the upsampled video, muxed into one file.

![Stack diagram of the inference ladder showing flow-matching sampling, tiled TAE decode, separate spatial upsampling to 1080p, the audio model pass, and a final mux and export step](/imgs/blogs/movie-gen-deep-dive-8.png)

Two efficiency techniques deserve a name. **Temporal tiling** at decode (section 5) bounds the memory of the autoencoder so clip length does not blow up VRAM. **Model tiling** — running the upsampler and decoder in spatial/temporal chunks — does the same for the high-resolution stages. There is also the question of *how many* flow-matching steps you need, which the sibling [flow matching for video](/blog/machine-learning/video-generation/flow-matching-for-video) post covers in depth: the straight-path objective is what lets you cross the distribution in relatively few steps, and step-distillation can push it lower still, at the usual quality cost. For a model where a single step is a forward pass over a 30B Transformer on tens of thousands of tokens, halving the step count roughly halves the wall-clock and the bill, so the sampler choice is a first-order cost lever, not a detail.

The sampling step itself is the same ODE-solve you would write for any flow-matching model. Here is the Euler solver over the velocity field, the inference counterpart to the training step in section 3:

```python
import torch

@torch.no_grad()
def fm_sample(model, shape, text_embeds, num_steps=50, ref_face=None, device="cuda"):
    # Start from pure noise in the TAE latent space.
    x = torch.randn(shape, device=device)                  # [B, c, T', H', W']
    ts = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

    for i in range(num_steps):
        t = ts[i].expand(shape[0])
        dt = ts[i + 1] - ts[i]
        # Predict velocity, optionally conditioned on a reference face
        # for personalized generation.
        v = model(x, t, encoder_hidden_states=text_embeds, ref=ref_face)
        x = x + v * dt                                     # Euler step along the field

    return x                                               # clean latent, ready to decode
```

Swap the Euler step for a higher-order solver and you can often cut the step count; that trade is exactly the samplers-deep-dive discussion from the image series ([samplers deep dive](/blog/machine-learning/image-generation/samplers-deep-dive)) applied to a much more expensive per-step cost. The structure — noise in, integrate the velocity field, clean latent out — is identical to the image case; only the latent is spatiotemporal and each step is far heavier.

## 12. Results, benchmarks, and how Meta measured them

Movie Gen's evaluation is, like the rest of the paper, more transparent than most. The headline claim from the report is that on human preference evaluations, Movie Gen Video was rated favorably against the leading systems of its time — the report compares against the contemporaneous frontier (including Sora, Kling, and other top models) and reports net-positive human preference on overall quality and on specific axes like motion and realism for the foundation model, and strong results for the audio and personalization capabilities. The honest framing matters here: these are **human-preference** results on a curated prompt set, not a single automatic score, and human preference is both the most meaningful and the most gameable metric in this field.

This is the right moment to be precise about how you measure video generation honestly, which we covered in [the metrics of video generation](/blog/machine-learning/video-generation/the-metrics-of-video-generation). Two pitfalls dominate. First, **FVD** (Fréchet Video Distance — the Fréchet distance between distributions of I3D features for real and generated clips) is noisy and sensitive to the sample set, the frame count, and preprocessing, so a small FVD gap is not a reliable ranking. Second, **VBench**'s dimensions can be gamed against each other: a model can score high on motion-smoothness and subject-consistency by *barely moving*, while a model with real dynamics scores lower on stability — the dynamic-degree-versus-stability tension. A single VBench number hides that trade-off. The credible way to compare, and the way Movie Gen leans on, is **human preference on a fixed, public-ish prompt set with a defined protocol** (forced choice, multiple raters, reported win rates), reported alongside the component scores, with enough detail that someone could in principle reproduce it.

Here is a comparison table of the Movie Gen components against the disclosed details and what each one is evaluated on. Numbers are from the paper where stated and marked approximate where the disclosure is qualitative.

| Component | Params | Job | Output ceiling | Evaluated on |
| --- | --- | --- | --- | --- |
| Movie Gen Video | 30B | T2V + I2V foundation | 16s, up to 16 fps, up to 1080p | Human preference vs frontier; motion, realism |
| Spatial upsampler | ~7B (approx) | Low-res to 1080p | 1080p frames | Detail fidelity, no artifacts |
| Movie Gen Audio | 13B | V2A: SFX + ambient + music | 48 kHz, up to ~45s | Sync + audio quality, human preference |
| Personalization (PT2V) | foundation variant | Identity from one ref face | Same as video model | Identity preservation + prompt adherence |
| Precise editing | trained edit model | Instruction-based edit | Same as video model | Edit accuracy + background preservation |

And here is where the capabilities differentiate Movie Gen from the other frontier systems. The point is not that Movie Gen wins on raw fidelity — Sora, Veo, and Kling are all extremely strong on that axis — but that it bundles **native audio, identity personalization, and trained instruction editing** into one documented stack.

![Matrix comparing Movie Gen against Sora 2, Veo 3.1, and Kling 3.0 on native audio, personalization, instruction editing, and openness](/imgs/blogs/movie-gen-deep-dive-7.png)

| Capability | Movie Gen | Sora 2 | Veo 3.1 | Kling 3.0 |
| --- | --- | --- | --- | --- |
| Native synchronized audio | Yes (13B V2A) | Yes (joint) | Yes (native dialogue) | Yes (SFX + lip-sync) |
| Personalization from a face | Yes (PT2V) | Cameo / likeness opt-in | Limited | Limited |
| Trained instruction editing | Yes (edit model) | Remix-style | Ingredients / remix | Start-end frames |
| Max documented length | ~16s | tens of seconds | extended | extended |
| Openly released weights | No (paper only) | No | No | No |
| Recipe transparency | Very high (90+ pp paper) | Low (short report) | Low | Low |

The last row is the real reason this post exists. Every model in that table is closed, but Movie Gen is the only one whose recipe is documented in enough depth to learn from. You cannot run its weights, but you can read exactly how it was built, which for an engineer is more valuable than another inaccessible API.

#### Worked example: estimating a render's cost profile

Reason about where the time goes for our 16-second clip. The flow-matching sampling dominates *if* you run many steps: 50 steps times a 30B forward pass over tens of thousands of tokens is the bulk of the compute. The tiled TAE decode is cheaper per element but runs on a large pixel tensor, so on a memory-constrained GPU it is the part most likely to OOM if you forget to tile. The spatial upsample to 1080p, on a few-billion-parameter model, adds a meaningful but smaller slice. The audio pass on a 13B model over a short audio latent is comparatively light. So the cost ranking is roughly *sampling > upsample > decode > audio* in compute, but *decode > sampling* in peak memory — which is exactly why the engineering attention in the paper goes to flow-matching step count (for speed) and to tiling the decode and upsample (for memory). If you were building this and had to cut wall-clock, you would attack the sampler step count first; if you had to fit it on a smaller GPU, you would attack the tiling first. Different bottlenecks, different levers.

## 13. The honest limitations Meta noted

A paper this transparent does not hide its failure modes, and the limitations section is worth reading as carefully as the results. Several are characteristic of the whole field, and Movie Gen states them plainly. The honesty itself is instructive: a model this strong listing its failures is a useful calibration for how far the frontier actually is from "solved," and it sets a standard the shorter technical reports from other labs do not meet.

**Physics and complex motion.** Like every model in this generation, Movie Gen can violate physical plausibility in complex dynamics — object permanence under occlusion, fluid and granular motion, intricate multi-object interactions. It is a learned pattern-matcher, not a simulator, a point we develop in [physics and the limits of learned simulation](/blog/machine-learning/video-generation/physics-and-the-limits-of-learned-simulation). It can produce a clip that looks right frame-by-frame but is subtly impossible if you track an object through an occlusion.

**Length.** The foundation model targets up to 16 seconds. Beyond its trained length you are into the autoregressive-rollout regime where identity and scene coherence drift, which is a separate hard problem we cover in [long video and autoregressive rollout](/blog/machine-learning/video-generation/long-video-and-autoregressive-rollout). Movie Gen does not claim minute-long coherent generation; it claims excellent short clips.

**Audio scope.** Movie Gen Audio targets sound effects, ambient sound, and *instrumental* music. Tightly lip-synced spoken dialogue — where the audio has to match mouth shapes frame-for-frame — is a different and harder problem than scoring a scene with foley and ambience, and the V2A framing (audio after final video) is well suited to the latter but not the most demanding dialogue cases. This is the boundary between the V2A approach and the joint-generation approach we contrast in [audio and joint audio-video generation](/blog/machine-learning/video-generation/audio-and-joint-av-generation).

**Editing and personalization edge cases.** Instruction editing can fail to localize correctly on ambiguous instructions or hard compositional changes, and personalization can struggle with extreme poses or identities far from the training distribution. These are capability frontiers, not solved problems.

**Compute and access.** A 30B video model plus a 13B audio model plus an upsampler is not something you serve cheaply, and the weights are not released. The transparency of the paper is the gift; the model itself is not in your hands. Concretely, a 30B model alone wants tens of gigabytes just to hold its weights in half precision, before you add the activation memory for a tens-of-thousands-token spacetime sequence, the tiled decode, the upsampler, and the 13B audio model — this is a multi-accelerator serving problem with a real per-second cost, not something that fits on a single consumer GPU. The honest framing for a builder is that Movie Gen is a research demonstration of what a well-funded lab can do, and the value you extract from it is the *recipe*, which is portable to smaller open models, rather than the artifact, which is not. For the deployment-cost reality of serving models in this class, see [efficient video inference and serving](/blog/machine-learning/video-generation/efficient-video-inference-and-serving).

## 14. Rebuilding the shape with open parts

Because Movie Gen is closed, the practical question is: what is the closest thing you can actually run, and how does it map onto the recipe? The good news is that the *recipe* Movie Gen documents is the converged open recipe — causal/temporal 3D-VAE plus a flow-matching DiT plus a separate decode/upsample path — so every component has an open stand-in.

For the **video model**, reach for a flow-matching open video pipeline. CogVideoX, Wan, HunyuanVideo, and Mochi all instantiate the same TAE-plus-flow-matching-DiT pattern, and `diffusers` exposes them with the tiling knobs you need. For the **TAE**, these pipelines ship their own causal video VAE (`AutoencoderKLCogVideoX` and relatives) with `enable_tiling()` for the decode wall. For **audio**, run a separate open V2A model conditioned on the generated frames and mux with `ffmpeg`, exactly as in section 7. For **personalization and editing**, the open analogs are subject-driven and instruction-editing adapters (IP-Adapter-style identity conditioning, instruction-edit pipelines), which are weaker than a foundation-trained capability but follow the same conditioning logic.

```python
# An open stand-in pipeline that mirrors the Movie Gen shape:
# flow-matching video DiT -> tiled VAE decode -> (separate) audio -> mux.
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()                      # the decode-wall fix

frames = pipe(
    prompt="a chef searing a steak in a loud kitchen, steam rising, close up",
    num_frames=49,
    num_inference_steps=50,                   # flow-matching ODE steps
    guidance_scale=6.0,
).frames[0]

export_to_video(frames, "steak_silent.mp4", fps=16)
# Then: add_audio("steak_silent.mp4", "sizzling steak, kitchen clatter", v2a)
# from section 7 to score it, mirroring Movie Gen Audio's V2A pass.
```

This will not match Movie Gen's quality — it is a 5B open model standing in for a 30B closed one, without the upsampler or the trained personalization and edit heads — but it lets you *run the recipe*, profile where the time and memory go, and understand the system by operating it. That is the most useful thing you can do with a closed model's open documentation.

## 15. When to reach for this design (and when not to)

Movie Gen is a research system you cannot deploy, so "reaching for it" really means reaching for its *design decisions* when you build your own stack. Here is the decisive version.

**Factor the pipeline; do not build a monolith.** The single most reusable lesson is the decomposition: latent generation, separate decode, separate upsample, separate audio. If you are building a video pipeline, resist the urge to make one model do everything. Generate in a compressed latent at a manageable resolution, upsample with a cheaper dedicated model, add audio in a separate conditioned pass. Each stage stays inside a budget, and you can improve or swap stages independently.

**Pre-train on images before video.** Do not spend your most expensive video compute teaching the model to render. Master text-to-image first on cheap single-frame examples, then switch on the time axis. The curriculum — low res before high, short before long, image before video — is not optional polish; it is how the training run converges within budget.

**Tile the decode and upsample, always.** The autoencoder decode, not the denoiser, is your memory wall on long clips. Build tiling in from the start rather than discovering the OOM in production.

**Use video-to-audio when dialogue is not the point.** Movie Gen's V2A approach — generate audio after the final video, conditioned on per-frame features — is the right call for SFX, ambience, and scoring, and it keeps the audio model separate and trainable on its own data. If your product needs tightly lip-synced spoken dialogue, the V2A-after-video approach is the wrong tool; reach for joint generation or a dedicated lip-sync model instead (the contrast is in [audio and joint audio-video generation](/blog/machine-learning/video-generation/audio-and-joint-av-generation)).

**Train a dedicated edit head; do not tell users to re-prompt.** If iterative refinement matters to your product, an instruction-based edit model that conditions on the existing clip beats re-prompting, both on cost and on preserving what the user already liked. The hard part is the data, not the architecture.

**When not to imitate Movie Gen:** if you only need short, silent, single-shot clips and have no personalization or editing requirement, you do not need five components — a single open flow-matching pipeline with tiled decode is enough, and the rest of Movie Gen's machinery is overhead you will not use. Match the complexity of the stack to the capability you actually ship.

## 16. Case studies: three numbers from the recipe

**The compression ratio is the budget.** The TAE's $8\times8\times8 = 512\times$ element compression, fed into the quadratic attention cost, is a $512^2 \approx 2.6\times10^5$ reduction in attention FLOPs versus pixel-space. This is not a Movie Gen quirk; it is the load-bearing decision behind every latent video model, and Movie Gen's clean disclosure of the ratio makes it the best worked example of *why* the autoencoder is the real enabler. Cross-check it against the general argument in [video autoencoders and spatiotemporal compression](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression).

**The 30B / 13B split is a deliberate asymmetry.** Meta put more than twice the parameters into video as into audio (30B vs 13B), which tracks the difficulty: spatiotemporal generation with coherent motion is a far harder learning problem than scoring a finished clip with synced sound. The split tells you where Meta judged the hard problem to be, and it is a useful prior when you allocate your own parameter budget across a multi-component system.

**The length ceiling is honest.** Movie Gen claims up to 16 seconds and does not pretend to minute-long coherence. Compare that to the autoregressive-rollout systems we cover in [long video and autoregressive rollout](/blog/machine-learning/video-generation/long-video-and-autoregressive-rollout): a model that generates a fixed-length clip in one shot avoids the identity-drift and error-accumulation problems that plague rollout, at the cost of a hard length limit. The 16-second ceiling is the *price* of that one-shot coherence, and stating it plainly is part of what makes the paper trustworthy.

## 17. Key takeaways

- **Movie Gen is five trained components, not a model**: the TAE latent, the 30B flow-matching video DiT, a separate spatial upsampler, the 13B V2A audio model, and the personalization plus editing capabilities. Reading it as a pipeline is the key to understanding it.
- **The TAE's $8\times8\times8$ compression is the load-bearing decision.** It cuts attention FLOPs by five orders of magnitude versus pixel space and is the only reason a 30B model on 16-second clips is trainable.
- **Flow matching on the latent is the objective**, in the rectified-flow style with a token-count-dependent schedule shift — straight paths regressed with a plain velocity MSE, sampled with an ODE solver in tens of steps.
- **The curriculum is text-to-image then text-to-video**, low-res before high, short before long, so the expensive video compute is reserved for learning motion, not rendering.
- **The decode, not the denoiser, is the memory wall.** Tiling the TAE decode in space and time turns an OOM into a constant-memory pass and is the technique that makes long clips fit.
- **A separate spatial upsampler reaches 1080p** because generating natively at 1080p means a quarter-million-token quadratic-attention sequence the 30B model cannot afford across dozens of steps.
- **Movie Gen Audio syncs by conditioning on per-frame visual features**, which carry the timing, plus a text prompt, which carries the content — the standard V2A recipe, scoped to SFX, ambience, and instrumental music rather than tight dialogue.
- **Personalization and instruction editing are the differentiators**, not raw fidelity. One reference face preserves identity; a trained edit head applies instructions without masks while preserving the rest. Both are bounded by the difficulty of assembling their training data.
- **The real gift is transparency.** The weights are closed, but the 90-plus-page paper documents the recipe in enough depth to rebuild its shape from open parts — which is more useful to an engineer than another inaccessible API.

## 18. Further reading

- **Movie Gen: A Cast of Media Foundation Models** — Polyak et al. (Meta), 2024. The primary source: the 90-plus-page paper with the parameter counts, TAE compression ratio, training curriculum, conditioning encoders, inference recipe, and limitations covered above. Read it directly.
- **Flow Matching for Generative Modeling** — Lipman et al., 2023. The objective Movie Gen trains under; the rectified-flow variant and the straight-path velocity target.
- **Scalable Diffusion Models with Transformers (DiT)** — Peebles and Xie, 2023. The Transformer-as-denoiser backbone that Movie Gen adapts in the spacetime regime.
- **The Llama family papers** — Touvron et al. and successors, 2023 onward. The Transformer architecture Movie Gen Video borrows for its backbone, and the systems work that makes 30B-scale training tractable.
- **Within this series**: [why video generation is hard](/blog/machine-learning/video-generation/why-video-generation-is-hard) for the coherence-times-motion-times-length-times-cost frame; [flow matching for video](/blog/machine-learning/video-generation/flow-matching-for-video) for the objective; [video autoencoders and spatiotemporal compression](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression) for the TAE; [conditioning video on text, image, motion, and camera](/blog/machine-learning/video-generation/conditioning-video-text-image-motion-camera) for personalization and editing as conditioning; [audio and joint audio-video generation](/blog/machine-learning/video-generation/audio-and-joint-av-generation) for the V2A audio half; the comparative [2026 video model landscape](/blog/machine-learning/video-generation/the-2026-video-model-landscape); and the [capstone playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook) where a pipeline like this becomes something you run.
- **Out to the image series**: [flow matching and rectified flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow) for the full objective derivation, [latent diffusion and stable diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion) for the latent-space generation pattern, and [samplers deep dive](/blog/machine-learning/image-generation/samplers-deep-dive) for the ODE-solver step-count trade.
