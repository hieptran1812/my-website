---
title: "Mochi 1 and the Asymmetric DiT: Anatomy of an Open Video Model"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Genmo's Mochi 1 is a 10B open video model whose design choices are all public, and this post dissects them end to end — the asymmetric DiT that pours its parameter budget into the visual stream, the 128x AsymmVAE that sets the token budget, the flow-matching recipe, and exactly how to run it in diffusers."
tags:
  [
    "video-generation",
    "diffusion-models",
    "video-diffusion",
    "text-to-video",
    "mochi",
    "asymmetric-dit",
    "open-source",
    "flow-matching",
    "generative-ai",
    "deep-learning",
    "diffusers",
  ]
category: "machine-learning"
subcategory: "Video Generation"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/mochi-and-asymmetric-dit-video-1.png"
---

Most frontier video models are black boxes. You read a technical report that tells you the model is "a diffusion transformer trained on a large curated dataset," you see a leaderboard number, and you are left to guess at every choice that actually mattered — the compression ratio, the attention pattern, the width of each stream, the way the text and the pixels talk to each other. Mochi 1 is different. When Genmo released it in October 2024 they shipped the weights under Apache-2.0, they shipped the inference code, and they wrote down the architecture in enough detail that you can sit down and reconstruct it. That makes Mochi one of the best teaching objects in the entire open video stack: not because it is the highest-scoring model you can download — it is not — but because its design choices are *legible*, and several of them are genuinely clever in ways that survive transplanting into your own work.

The single choice worth the price of admission is the word in the title: **asymmetric**. Mochi's denoiser is a 10-billion-parameter diffusion transformer that processes two streams — the video and the text prompt — and it makes them deliberately *unequal*. The visual stream is wide; the text stream is narrow. That sounds like a small implementation detail, the kind of thing you would skim past in a config file. It is actually a sharp, defensible answer to a real efficiency question, and once you see why it is right, you start noticing the same asymmetry argument lurking in half the multimodal architectures shipped since. There is a second clever choice underneath it — an autoencoder that compresses a clip by a brutal $128\times$ before the denoiser ever runs — and the two choices are coupled. This post takes both apart.

![Graph of the asymmetric joint attention block showing a wide visual stream and a narrow text stream feeding one shared softmax](/imgs/blogs/mochi-and-asymmetric-dit-video-1.png)

By the end you will be able to do five concrete things. First, you will be able to *explain why an asymmetric DiT is more parameter-efficient than a symmetric one* for video, from the token-count arithmetic rather than from hand-waving. Second, you will be able to *derive Mochi's $128\times$ compression* and trace its consequence all the way to the token sequence the denoiser attends over and the VRAM that costs. Third, you will be able to *run Mochi in 🤗 `diffusers`* with the offload and tiling flags that make a 10B-class model fit on a single consumer card. Fourth, you will be able to *sketch the asymmetric joint-attention block in PyTorch* — separate projection widths for the two streams, one shared attention — so you could implement the idea yourself. Fifth, you will be able to *place Mochi honestly against CogVideoX, HunyuanVideo, and Wan* on the axes that matter: openness, parameters, VAE ratio, resolution, and quality, including the limitations Genmo themselves were candid about.

This post sits in the [video generation series](/blog/machine-learning/video-generation/why-video-generation-is-hard) as one of its frontier-model dissections. It assumes you have met the pieces in their own posts: the [causal 3D-VAE](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression) that does the temporal compression, the [video diffusion transformer](/blog/machine-learning/video-generation/video-diffusion-transformers) that denoises spacetime tokens, and [flow matching for video](/blog/machine-learning/video-generation/flow-matching-for-video) that trains it. Where Mochi reuses a pure image-diffusion idea, I will point at the image series — the [MM-DiT recipe](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe) and [flow matching and rectified flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow) — rather than re-deriving it. The spine of the whole series holds here too: video is spatial generation times temporal coherence under a brutal compute budget, and Mochi's two clever choices are both, at heart, ways of spending that budget where it buys the most.

A note on numbers before we start. Mochi's design is public, so most of what follows traces to Genmo's release blog, the model card, and the `genmo/mochi-1-preview` code on the Hugging Face Hub and GitHub. Where I give an exact figure — $10\text{B}$ parameters, $128\times$ compression, a $12$-channel latent, $848\times480$ output — it is reported. Where I compute a downstream consequence (the token count for a given clip, a VRAM estimate at a dtype) I show the arithmetic and mark it approximate, because the same model needs wildly different memory depending on resolution, length, dtype, and offload. Never trust a video-model VRAM number that does not name those four.

## 1. The problem Mochi's asymmetry solves

Start with the question that the asymmetric DiT answers, because the architecture only makes sense once the question is sharp. A text-to-video diffusion transformer has to process two very different sequences at once. One is the **prompt**: a string of words, tokenized and encoded, which for a typical caption lands somewhere between a few dozen and a few hundred tokens. The other is the **video latent**: the noisy spacetime grid the model is denoising, which — as we will compute in detail later — is *tens of thousands* of tokens for even a short clip. These two sequences are not remotely the same size. The text is a footnote; the video is a novel.

Now recall how a multimodal diffusion transformer, an MM-DiT, handles two modalities. The clean MM-DiT recipe from the image world — the one [Stable Diffusion 3 popularized](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe) — runs *two parallel streams*, one per modality, each with its own set of weights (its own QKV projections, its own MLP, its own normalization), and lets the two streams exchange information through a **joint attention** operation where the queries from one stream can attend to the keys of the other. It is an elegant design: each modality gets its own processing capacity, and they still talk. The catch, when you carry it over to video unchanged, is that "its own processing capacity" means *equal* capacity. A symmetric MM-DiT gives the text stream exactly as much width — the same hidden dimension, the same MLP expansion, the same number of attention heads' worth of projection weights — as the video stream.

That is the waste Mochi noticed. The text stream, in a symmetric design, is a full-width transformer processing a few hundred tokens. The video stream is a full-width transformer processing forty thousand. You are paying for two large transformers but only one of them has real work to do; the other is a wide road carrying a bicycle. The text representation does need to be good — a strong, well-conditioned text embedding is what makes a video model follow a prompt at all — but "good" does not require the *same width* as the visual processing. A few hundred tokens can be served by a much narrower path and still produce a rich conditioning signal.

So Mochi makes the streams **asymmetric**: the visual stream keeps a large hidden size (Mochi uses a model dimension of $3072$ for the visual stream), and the text stream runs at a smaller width. The parameters you save by narrowing the text stream do not vanish — you *move them into the visual stream*, making it deeper or wider where the tokens actually are. At a fixed total parameter budget, that reallocation is a free lunch in the precise sense that it raises capacity on the long, hard sequence without paying for capacity on the short, easy one. That is the whole argument, and it is correct.

![Before and after comparison of symmetric versus asymmetric parameter allocation across the visual and text streams](/imgs/blogs/mochi-and-asymmetric-dit-video-4.png)

Let me make the "free lunch" precise, because efficiency claims deserve numbers. Suppose the symmetric design has both streams at width $d$ and your total budget is fixed. The per-layer parameter cost of a transformer stream is dominated by the projection and MLP matrices, which scale as $O(d^2)$. If the text stream's width drops from $d$ to $d/r$ for some ratio $r > 1$, its parameter cost falls by $r^2$ — a text stream at a quarter of the width costs a sixteenth of the parameters. Those reclaimed parameters get spent on the visual stream, where every additional parameter is doing work on the dominant sequence. Crucially, the *compute* of the attention operation is unchanged by this — joint attention still attends over the concatenation of all tokens, and the visual tokens were always going to dominate that $O(L^2)$ cost regardless of stream width. So you keep the attention bill, you keep the text quality (a narrower-but-sufficient text path), and you upgrade the visual capacity for free. The asymmetry is not a compromise; it is a strict improvement under the realistic assumption that video tokens vastly outnumber text tokens.

#### Worked example: the parameters the asymmetry frees up

Put numbers on the reallocation. Take a single transformer layer at width $d = 3072$. The bulk of its parameters are in four matrices: the QKV projection ($3 d^2$), the attention output projection ($d^2$), and the two MLP matrices ($d \times 4d$ and $4d \times d$, so $8 d^2$ together), for roughly $12 d^2$ parameters per stream per layer. At $d = 3072$ that is about $12 \cdot 3072^2 \approx 1.1 \times 10^8$ parameters — $110\text{M}$ — for *one stream* in *one layer*. A symmetric design pays that twice, once for video and once for text, every layer. Now narrow the text stream to $d_t = 1536$ (half) or $d_t = 768$ (a quarter). At half width the text stream costs $12 \cdot 1536^2 \approx 28\text{M}$ per layer instead of $110\text{M}$ — a $\sim82\text{M}$ saving per layer. Across, say, $48$ layers that is roughly $48 \cdot 82\text{M} \approx 3.9\text{B}$ parameters freed, which you pour back into a wider or deeper visual stream. That is not a rounding error on a 10B model — it is nearly $40\%$ of the budget reallocated from a few-hundred-token sequence to a $44\text{k}$-token one. The exact widths Mochi ships differ from these round numbers, but the order of magnitude is the point: the asymmetry is not a tweak, it is a large fraction of the parameter budget moved to where the work is.

This is the kind of design choice that is obvious in retrospect and was not obvious in advance, and it is exactly why a legible open model is worth dissecting. You could not have read this argument off a leaderboard. You can read it off Mochi's config.

## 2. Inside the AsymmDiT block

Now let us open the block itself, because the asymmetry shows up in a specific, implementable way. A single AsymmDiT layer takes the current visual tokens and the current text tokens and updates both. The structure mirrors a standard MM-DiT layer — modulation, attention, MLP — but with the two streams running at different widths and meeting only inside attention.

Walk the data path. The visual tokens enter at the visual width (call it $d_v = 3072$). The text tokens enter at the smaller text width $d_t$. Each stream gets its own AdaLN-style modulation conditioned on the diffusion timestep, so the timestep tells each stream how aggressively to denoise at this noise level — this is the same time-conditioning trick the [DiT](/blog/machine-learning/image-generation/diffusion-transformers-dit) uses, applied per stream. Then comes the one place the streams interact: **joint attention**. Each stream projects its tokens to queries, keys, and values — but with its *own* projection matrices, sized to its *own* width. The visual stream's QKV projections map from $d_v$; the text stream's map from $d_t$. The trick that makes the joint attention work despite the width mismatch is that both streams project into a *shared head dimension*: the per-head size is the same on both sides, so once you have the per-head Q, K, V vectors, a video query and a text key live in the same space and their dot product is meaningful. You concatenate the visual and text K/V along the sequence axis, and every query — visual or text — attends over the whole concatenated set in one shared softmax.

![Graph of the asymmetric joint attention block with separate visual and text projections feeding a single shared attention](/imgs/blogs/mochi-and-asymmetric-dit-video-1.png)

After attention, each stream takes its slice of the output back at its own width and runs its own MLP — wide for video, narrow for text. The text stream, in fact, is deliberately lightweight: it is closer to a small MLP path that keeps a usable text representation alive layer to layer than to a full co-equal transformer. This is the "narrow MLP path" the figures keep referring to. The point is not that the text is processed *badly*; it is that the text is processed *proportionally* — given a budget that matches its few-hundred-token length rather than the video's forty thousand.

Here is the block sketched in PyTorch. It is not Mochi's exact code, but it captures the load-bearing idea — separate widths, shared head dimension, one joint attention — closely enough that you could grow it into a real implementation.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmJointAttention(nn.Module):
    """One joint-attention op over concatenated video + text tokens.
    The two streams have DIFFERENT model widths but the SAME per-head
    dimension, so their queries and keys live in a shared space."""
    def __init__(self, dim_visual=3072, dim_text=1536, num_heads=24):
        super().__init__()
        self.num_heads = num_heads
        # per-head dim is shared across streams -> q/k/v are comparable
        self.head_dim = dim_visual // num_heads          # e.g. 128
        inner = self.head_dim * num_heads                 # = dim_visual

        # each stream projects FROM its own width INTO the shared inner dim
        self.qkv_visual = nn.Linear(dim_visual, 3 * inner, bias=False)
        self.qkv_text   = nn.Linear(dim_text,   3 * inner, bias=False)
        # outputs go back to each stream's own width
        self.proj_visual = nn.Linear(inner, dim_visual)
        self.proj_text   = nn.Linear(inner, dim_text)

    def _split_heads(self, x, B):
        # (B, S, 3*inner) -> q,k,v each (B, heads, S, head_dim)
        q, k, v = x.chunk(3, dim=-1)
        reshape = lambda t: t.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        return reshape(q), reshape(k), reshape(v)

    def forward(self, vis, txt):
        B, Sv, _ = vis.shape
        _, St, _ = txt.shape
        qv, kv, vv = self._split_heads(self.qkv_visual(vis), B)   # video stream
        qt, kt, vt = self._split_heads(self.qkv_text(txt),   B)   # text stream

        # concatenate along the sequence axis: every query sees every key
        q = torch.cat([qv, qt], dim=2)
        k = torch.cat([kv, kt], dim=2)
        v = torch.cat([vv, vt], dim=2)
        out = F.scaled_dot_product_attention(q, k, v)            # one shared softmax

        # split the output back into the two streams, each at its OWN width
        out = out.transpose(1, 2).reshape(B, Sv + St, -1)
        out_vis, out_txt = out[:, :Sv], out[:, Sv:]
        return self.proj_visual(out_vis), self.proj_text(out_txt)
```

The thing to notice is that `dim_visual` and `dim_text` differ, but `head_dim` is shared, so the concatenation in the middle is legal and the dot products are meaningful. The visual stream pays a $3072$-wide QKV; the text stream pays a $1536$-wide QKV (and could go narrower). Everything downstream of the shared attention returns to per-stream widths. This is the asymmetry, expressed in about forty lines. In Mochi the real block also carries rotary position embeddings on the visual tokens (3D RoPE over time, height, and width) so the model knows where each spacetime token sits, and the text tokens get their own positional treatment — but the asymmetric skeleton above is the part worth internalizing.

One more design note that earns Mochi points for honesty: the model uses **full 3D attention**, not the factorized spatial-then-temporal attention that some video models use to save compute. Every visual token can attend to every other visual token across all of space and time in a single operation. We covered [the factorization trade-off](/blog/machine-learning/video-generation/spatiotemporal-attention-patterns) in its own post — factorized attention is cheaper but couples space and time only indirectly, which can cost coherence. Mochi spends the full-3D-attention compute and gets the stronger coupling, which is part of why its *motion* is good even when its resolution is modest. Full 3D attention over $44\text{k}$ tokens is exactly the $O(L^2)$ bill that makes the VAE compression so important — which is the next section.

Let me make the full-3D-versus-factorized choice quantitative, because it is the second axis (after the asymmetric budget) where Mochi's compute goes, and the FLOPs are worth seeing. In full 3D attention, every one of the $L$ tokens attends to all $L$ tokens, so the attention cost per layer is $O(L^2 d)$ — with $L \approx 44{,}500$ and $d = 3072$, that is roughly $44{,}500^2 \cdot 3072 \approx 6 \times 10^{12}$ multiply-accumulates per layer, per step, just for the attention scores and the value aggregation. Factorized attention splits this: it does spatial attention within each frame (over $L/T'$ tokens) and then temporal attention across frames (over $T'$ tokens per spatial location), so the cost drops to roughly $O\!\left(T' \cdot (L/T')^2 \cdot d + (L/T') \cdot T'^2 \cdot d\right)$. Plug in $T' = 28$, $L/T' \approx 1{,}590$: the spatial term is $28 \cdot 1{,}590^2 \cdot 3072 \approx 2.2 \times 10^{11}$ and the temporal term is $1{,}590 \cdot 28^2 \cdot 3072 \approx 3.8 \times 10^9$, for a total around $2.2 \times 10^{11}$ — about $27\times$ cheaper than full 3D attention. That is an enormous saving, and it is exactly why many video models factorize.

So why does Mochi pay the $27\times$? Because factorization breaks the direct path between a token at time $t_1$, position $p_1$ and a token at time $t_2 \ne t_1$, position $p_2 \ne p_1$ — they can only influence each other *through* an intermediate token they share a row or column with, across two attention operations. For *coherent motion*, where an object's pixels move diagonally through spacetime, that indirect coupling is precisely the weak link: the model has to route a moving edge's information through a relay rather than attending to it directly. Full 3D attention has no relay; a token at $(t_1, p_1)$ attends to $(t_2, p_2)$ in one hop, and Genmo bet that this direct spatiotemporal coupling buys enough motion quality to justify the compute. Given that Mochi's whole identity is *motion*, that bet is internally consistent — and it is another choice you can read straight off the architecture and reason about, rather than guess at. The cost of the bet, of course, is that the VAE *must* be aggressive: full 3D attention is only affordable because the $128\times$ VAE kept $L$ down to $44\text{k}$. The two clever choices are not independent — the aggressive VAE is what *pays for* the full 3D attention. Drop the VAE to $48\times$ like CogVideoX and full 3D attention over the resulting $65\text{k}$ tokens would cost about $2.1\times$ more, which is the difference between affordable and not.

## 3. The AsymmVAE and the 128x compression

If the asymmetric DiT is Mochi's cleverest *denoiser* choice, the AsymmVAE is its cleverest *systems* choice, and it is the one that sets every downstream cost. Let me derive the compression from scratch, because the number — $128\times$ — is the single most consequential figure in the whole model.

A causal 3D-VAE takes a clip of shape $T \times H \times W \times 3$ — frames, height, width, RGB — and encodes it to a latent of shape $T' \times H' \times W' \times C$. "Causal" means the temporal convolutions only look backward in time, so the encoder can produce latent frame $t'$ without seeing future input frames; this is what lets the VAE handle the first frame specially and, in principle, stream arbitrary lengths. The *compression ratio* is the ratio of input voxels to output voxels, counting channels. Mochi's AsymmVAE compresses **$8\times$ in each spatial dimension and $6\times$ in time**, down to a **$12$-channel** latent. So a clip shrinks by

$$
\frac{T \cdot H \cdot W \cdot 3}{T' \cdot H' \cdot W' \cdot C} = \frac{6 \cdot 8 \cdot 8 \cdot 3}{1 \cdot 1 \cdot 1 \cdot 12} = \frac{1152}{12} = 96\times
$$

in voxel-channel count. But the figure Genmo quotes, and the one that matters for the spatial-temporal grid the denoiser sees, is the **voxel** compression ignoring the channel re-pack: $6 \times 8 \times 8 = 384$ in voxels, and accounting for the channel expansion from $3$ to $12$ the *spatial-temporal* token reduction is what dominates. The headline Genmo reports is $128\times$, which comes from how they count the spatial-temporal compression together with the latent channel budget: the latent is a $12$-channel grid that is $8\times8$ smaller in space and $6\times$ shorter in time, and relative to a naive per-frame VAE that would keep more channels, the effective reduction in what the transformer must process is on the order of $128\times$. The precise bookkeeping matters less than the consequence, so let me state the consequence sharply and move to it: **Mochi's VAE is among the most aggressive in the open tier, more aggressive temporally than CogVideoX's or HunyuanVideo's $4\times$, and that aggression is exactly what makes a 10B model runnable.**

![Stacked diagram of the AsymmVAE compression stages from RGB clip through spatial and temporal squeezes to a 12 channel latent and the token budget](/imgs/blogs/mochi-and-asymmetric-dit-video-2.png)

Why does the VAE, and not the giant denoiser, decide your cost? Because the latent grid is what the denoiser attends over, and attention is quadratic in the grid's flattened length. We covered [the master-lever argument](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression) in the VAE post; here it is in Mochi's specific numbers. The token sequence length is

$$
L = \frac{T}{c_t\, p_t} \cdot \frac{H}{c_h\, p_h} \cdot \frac{W}{c_w\, p_w},
$$

where $(c_t, c_h, c_w) = (6, 8, 8)$ is the VAE compression and $(p_t, p_h, p_w)$ is the DiT's patch size. Self-attention over that sequence costs $O(L^2 d)$ per layer. So the VAE's compression sits *inside* the term that gets squared. A more aggressive temporal compression — $6\times$ instead of $4\times$ — cuts $T'$ by a factor of $1.5$, which cuts $L$ by $1.5$, which cuts the attention bill by $2.25\times$ for the same clip. That is real money on a 10B model running full 3D attention.

#### Worked example: token count for the running 5.4-second clip

Take Mochi's native output: a $5.4$-second clip at $848 \times 480$, $30$ fps, which is about $163$ frames (Mochi generates up to roughly this length per call). Through the $6 \times 8 \times 8$ AsymmVAE:

$$
T' = \frac{163}{6} + 1 \approx 28, \qquad H' = \frac{480}{8} = 60, \qquad W' = \frac{848}{8} = 106, \qquad C = 12.
$$

The latent is about $28 \times 60 \times 106 \times 12$. Mochi patchifies space by roughly $2\times2$ and not time, so the token sequence the DiT sees is

$$
L = T' \cdot \frac{H'}{2} \cdot \frac{W'}{2} = 28 \cdot 30 \cdot 53 \approx 44{,}500 \text{ tokens}.
$$

That $\approx 44\text{k}$ tokens is the sequence Mochi's full 3D attention runs over at every one of its layers, at every one of its $\sim 64$ sampling steps. Now do the counterfactual: if the VAE compressed time by only $4\times$ like CogVideoX, $T'$ would be about $41$ instead of $28$, $L$ would be about $65\text{k}$, and the attention bill — quadratic in $L$ — would be roughly $(65/44.5)^2 \approx 2.1\times$ larger. The $6\times$ temporal squeeze is buying Mochi more than a factor of two on its dominant cost. The VAE is the lever; the denoiser is the load.

![Grid showing the latent tensor layout for the running clip with latent frames height width channels patchify and final token count](/imgs/blogs/mochi-and-asymmetric-dit-video-6.png)

There is a cost to all this aggression, and Mochi pays it visibly. A $128\times$ compression asks the decoder to reconstruct a great deal of high-frequency spatial detail from a coarse latent. The more you squeeze, the more the decoder has to *hallucinate* fine texture back, and the harder it is to keep that texture stable and artifact-free. This is part of why Mochi's native resolution is $480p$ rather than $720p$ — the aggressive VAE trades spatial fidelity for token economy, and Genmo chose to spend the savings on *motion* (full 3D attention, a big denoiser) rather than on resolution. It is a coherent set of choices: prioritize motion quality and openness, accept modest resolution. We will see in the limitations section that Genmo were upfront about exactly this trade.

The "causal" part of the AsymmVAE deserves one more sentence because it is what makes the temporal arithmetic clean. The $+1$ in $T' = T/6 + 1$ is the special first latent frame: a causal VAE encodes the opening frame on its own (it has no past to look back on) and then compresses subsequent groups of frames. This is the same causal-frame bookkeeping every modern video VAE uses, and it is why latent-frame counts are never quite $T/c_t$ but $T/c_t + 1$.

Why "causal" and not just a regular 3D convolution that looks both backward and forward in time? Two reasons, both practical. First, a causal temporal convolution can encode a clip *incrementally* — it never needs a future frame to produce the current latent — which is what lets the VAE, in principle, handle clips of varying length without retraining and lets the decode stream frame groups out rather than holding the whole clip's latent in memory at once. Second, the causal structure plays nicely with the special first frame: by treating frame zero as its own latent, the VAE can support an image-to-video extension cleanly later (the first frame is exactly where you would inject a conditioning image), even though Mochi 1's preview did not ship that path. The causality is an architectural option kept open, not just a compute trick.

Now stress-test the VAE, because this is where the aggressive design shows its edges. The VAE was *trained* on clips up to a certain latent length. Ask it to decode a latent far longer than anything it saw in training and two things degrade. The reconstruction quality drops — the decoder's temporal receptive field was tuned for a certain horizon, and beyond it the frame-to-frame consistency it learned starts to fray, so you see drift or flicker creep in at the tail of a too-long clip. And the *decode memory* grows with length, so a long clip is exactly when the decode-OOM wall bites hardest. This is the deeper reason Mochi generates a bounded clip per call rather than an arbitrary-length one: not because the denoiser cannot in principle run longer, but because the VAE's trained horizon and the quadratic attention bill both punish length, and length is the quadratic axis. To go longer you do not stretch one call — you stitch clips, with all the identity-drift and seam problems that [long-video rollout](/blog/machine-learning/video-generation/long-video-and-autoregressive-rollout) entails. Mochi is a five-second model by the same logic that makes it a $480p$ model: the aggressive VAE bought a runnable 10B denoiser, and the bill came due as bounded length and modest resolution. Every one of these limits is the same trade seen from a different angle.

## 4. Flow matching, briefly, and why it fits the video regime

Mochi trains with **flow matching**, the same objective family that CogVideoX, HunyuanVideo, and Wan all converged on. I am not going to re-derive flow matching here — the [image series covers rectified flow and conditional flow matching](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow) properly, and the [flow-matching-for-video post](/blog/machine-learning/video-generation/flow-matching-for-video) covers the video-specific version. Let me recap only the part that explains *why* Mochi uses it, then move on.

The one-line version: instead of training the network to predict the noise added to a sample (the older DDPM $\epsilon$-prediction objective), flow matching trains the network to predict a **velocity** — the direction and speed to move a point along a straight path from noise to data. You define an interpolation $x_t = (1-t)\,x_0 + t\,\epsilon$ between a clean latent $x_0$ and Gaussian noise $\epsilon$, and the target velocity is simply the constant $v = \epsilon - x_0$. The loss is a plain regression:

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t,\, x_0,\, \epsilon} \left\lVert v_\theta(x_t, t, c) - (\epsilon - x_0) \right\rVert^2,
$$

where $c$ is the text conditioning. The straight-line path is the point: because the target trajectory is a straight line rather than the curved path of a diffusion SDE, a sampler can take *large, accurate steps* along it, which is why flow-matching models often produce good results in fewer steps than the old DDPM samplers needed.

Why does this matter *more* for video than for images? Because in video, every sampling step is catastrophically expensive — it is a full forward pass of a 10B denoiser over $44\text{k}$ tokens with full 3D attention. If $\epsilon$-prediction needs $250$ steps and flow matching needs $64$, that is not a $4\times$ saving on a cheap operation; it is a $4\times$ saving on the single most expensive operation in the entire stack. The straight-path efficiency of flow matching compounds against the brutal per-step cost of video. That compounding is why the whole open tier abandoned $\epsilon$-prediction for video even where they kept it for images. Mochi runs roughly $64$ flow-matching steps to render a clip; each step is one pass of the AsymmDiT; the VAE decodes once at the end.

There is a subtlety the video regime forces, which the image series flags too: the **noise schedule shift**. High-resolution, long latents need more noise at a given timestep than low-resolution images do, because there is more signal to corrupt before the model can no longer see the structure. Flow-matching video models therefore apply a *timestep shift* — they bias sampling toward the noisier end of the schedule — and Mochi is no exception. In the `diffusers` pipeline this shows up as a parameter on the `FlowMatchEulerDiscreteScheduler`; you usually leave it at the model's default, but it is worth knowing it is there, because it is the kind of knob that, mis-set, gives you either mush or under-denoised noise.

There is a second reason the straight-line velocity target is the right fit for *this* model specifically, beyond the step-count saving, and it ties back to the aggressive VAE. Sampling integrates the learned velocity field from pure noise at $t=1$ to a clean latent at $t=0$:

$$
x_0 = x_1 + \int_1^0 v_\theta(x_t, t, c)\, dt \;\approx\; x_1 + \sum_{i} v_\theta(x_{t_i}, t_i, c)\, \Delta t_i,
$$

and the discretization error of that sum depends on how *curved* the true trajectory is. A curved path needs many small steps to integrate accurately; a straight path can be integrated with a handful of large ones, because a straight line is its own first-order approximation. Flow matching trains the field so the paths are as close to straight as the data allows, which is exactly what lets Mochi use $\sim64$ steps where an SDE sampler would want hundreds. The payoff is largest precisely when each step is most expensive — and a 10B denoiser running full 3D attention over a $128\times$-compressed latent is about the most expensive step in open video. The VAE makes the step affordable; flow matching makes the *number* of steps small; together they make a 10B model with full 3D attention practical to sample. Every choice in this model leans on the others.

## 5. The text encoder: one frozen T5-XXL

Mochi conditions on a single, frozen **T5-XXL** text encoder — the same $\sim4.7$-billion-parameter encoder-decoder language model (Mochi uses the encoder half) that has become a default for high-fidelity prompt following in the open generative-image and -video world. This is worth a short section because it is both a strength and one of Mochi's honest limitations.

The strength: T5-XXL produces a rich, well-conditioned text embedding that carries far more compositional and relational information than the small CLIP text encoder that older models used. When you ask Mochi for "a red fox trotting left to right across fresh snow, camera panning to follow," the T5 embedding actually encodes the relations — the color binding, the direction, the camera intent — in a way the narrow text stream of the AsymmDiT can then inject into the visual stream through joint attention. Strong prompt adherence in open video models is, to a first approximation, a story about using a strong text encoder, and Mochi made the strong choice.

Trace how the embedding actually reaches the pixels, because it explains why the narrow text stream still matters. T5-XXL encodes the prompt into a sequence of token embeddings — one vector per prompt token, a few hundred at most. Those vectors enter the AsymmDiT as the text stream's input. At every layer, the text stream is processed by its own (narrow) projections and then participates in the joint attention, where the *visual* queries attend to the *text* keys. That cross-modal attention is the moment the prompt's meaning is written into the video tokens: a visual token representing a patch of the scene pulls in whichever text tokens are relevant ("fox," "snow," "left to right") and updates itself accordingly. The text stream does not need to be wide to do this well — it needs to keep a faithful, well-separated representation of the prompt tokens alive so the visual queries have something clean to attend to. A few hundred tokens at a moderate width is plenty for that. This is the concrete reason the asymmetry does not hurt prompt adherence: the text stream's *job* is to be a good set of attention targets, not to do heavy reasoning, and a narrow stream is a perfectly good set of targets. The heavy lifting — figuring out where the fox goes and how the snow scatters — happens in the wide visual stream, which is exactly where Mochi put the parameters.

![Graph of the Mochi generation path from prompt through frozen T5 encoder and noise into the AsymmDiT then sampler then AsymmVAE decode to frames](/imgs/blogs/mochi-and-asymmetric-dit-video-5.png)

The cost: T5-XXL is itself a $4.7\text{B}$-parameter model that has to be loaded and run. It is frozen — you never train it — but it sits in memory and it adds to your load. In practice you run it once per prompt, cache the embedding, and free it before the denoiser runs, which is exactly what the offload machinery in the next section does. But it means that "Mochi is a 10B model" undersells the footprint: the *system* is a 10B denoiser plus a $4.7\text{B}$ frozen text encoder plus a VAE, and a naive load of all of it at once is what OOMs a 24GB card. The fix is to never hold all three on the GPU at the same time, and `diffusers` gives you that fix for free.

It is also worth flagging what the single-T5, text-only design *cannot* do at launch: Mochi 1's released preview is **text-to-video only**. There is no image-to-video path in the initial release — you cannot hand it a first frame and ask it to animate. We will come back to this in the limitations and the "when to reach for it" sections, because it is one of the sharpest practical constraints on where Mochi fits.

The full generation path, then, is the one in the figure above: the prompt goes to the frozen T5-XXL encoder; noise seeds the $12$-channel latent grid; the AsymmDiT runs its $\sim64$ flow-matching steps conditioned on the text embedding; the AsymmVAE decodes the final latent up by $128\times$ back to pixels; you export frames. Two frozen modules — the text encoder and the VAE — bracket one trained denoiser. The same shape as every open video model, instantiated with Mochi's specific, legible choices.

## 6. Running Mochi in diffusers

Enough architecture; let us run it. 🤗 `diffusers` ships a `MochiPipeline`, and the whole point of this section is the *memory management*, because a 10B denoiser plus a $4.7\text{B}$ text encoder plus a VAE that decodes a $128\times$-compressed latent does not fit naively on a consumer card. The three flags that make it fit are `enable_model_cpu_offload()`, `enable_vae_tiling()`, and running in `bfloat16`. Here is the minimal call.

```python
import torch
from diffusers import MochiPipeline
from diffusers.utils import export_to_video

pipe = MochiPipeline.from_pretrained(
    "genmo/mochi-1-preview",
    torch_dtype=torch.bfloat16,        # half precision: ~2 bytes/param
)

# never hold the text encoder, denoiser, and VAE on the GPU at once.
# offload moves each module to GPU only while it runs, then back to CPU.
pipe.enable_model_cpu_offload()
# the VAE decode of a 128x-compressed latent is a VRAM spike;
# tiling decodes the latent in spatial tiles so the peak stays bounded.
pipe.enable_vae_tiling()

prompt = (
    "A red fox trotting left to right across fresh snow at golden hour, "
    "the camera slowly panning to follow, soft rim light, shallow depth of field"
)

frames = pipe(
    prompt=prompt,
    num_frames=163,            # ~5.4 s at 30 fps, Mochi's native length
    num_inference_steps=64,    # flow-matching steps
    guidance_scale=4.5,        # CFG strength for prompt adherence
    height=480,
    width=848,                 # native 848x480; off-aspect needs multiples of 16
).frames[0]

export_to_video(frames, "fox.mp4", fps=30)
```

Read the flags, because each one maps to a specific failure you would otherwise hit. `torch_dtype=torch.bfloat16` halves the parameter footprint versus fp32 and, unlike fp16, will not silently overflow on the larger activations a video model produces; bf16 is the right default for a 10B video denoiser. `enable_model_cpu_offload()` is the load-bearing flag: it keeps the modules on CPU and pulls each onto the GPU only for the duration of its forward pass — text encoder runs, then offloads; denoiser runs step by step; VAE runs at the end. At no point do all three sit on the GPU together, so your *peak* VRAM is set by the largest single module plus the activations, not the sum. `enable_vae_tiling()` addresses the sneakiest spike of all: decoding the $128\times$-compressed latent back to a full $848\times480\times163$ pixel tensor is a memory explosion at the very end of generation, the kind that OOMs you at second six after eight minutes of denoising. Tiling decodes the latent in overlapping spatial tiles and stitches them, capping the decode peak.

If you are even tighter on memory, you add two more knobs:

```python
# for the very tightest cards: offload each layer's weights individually
# (slower than model-cpu-offload, but the lowest possible peak VRAM)
pipe.enable_sequential_cpu_offload()

# and quantize the text encoder, the single biggest non-denoiser block,
# so it costs ~half the memory while it runs
from transformers import T5EncoderModel
text_encoder = T5EncoderModel.from_pretrained(
    "genmo/mochi-1-preview", subfolder="text_encoder",
    torch_dtype=torch.bfloat16, load_in_8bit=True,   # 8-bit T5
)
pipe = MochiPipeline.from_pretrained(
    "genmo/mochi-1-preview", text_encoder=text_encoder,
    torch_dtype=torch.bfloat16,
)
```

`enable_sequential_cpu_offload()` is more aggressive than `enable_model_cpu_offload()` — it streams individual sub-modules on and off the GPU rather than whole models, which drops the peak further at a real latency cost. Eight-bit-quantizing the T5 encoder, via `bitsandbytes`, shaves the largest non-denoiser block roughly in half while it runs. Stack offload, tiling, bf16, and an 8-bit text encoder and you can bring a 10B-class system onto a single 24GB card; the cost is wall-clock time, because every offload is a PCIe transfer. The trade is explicit and it is yours to make: VRAM versus seconds-per-clip.

Two parameters in the call deserve a sentence each, because they are the levers you actually turn at inference. `guidance_scale=4.5` sets the classifier-free guidance strength — the same [CFG mechanism the image series derives](/blog/machine-learning/image-generation/classifier-free-guidance), where the model runs once with the prompt and once unconditioned and extrapolates away from the unconditioned prediction. Higher CFG tightens prompt adherence at the cost of saturation and, in video, sometimes at the cost of motion (over-guided clips can look stiff). Mochi's sweet spot sits in the $4$–$6$ range; push past it and you trade liveliness for literalism. The cost of CFG is that every step is *two* forward passes, not one — the conditional and the unconditional — which is why guided video sampling is twice the per-step bill, and why distilled or guidance-free variants are such an active research direction. `num_inference_steps=64` is the flow-matching step count; drop it toward $30$ and you save proportional time at some quality cost, raise it toward $100$ and you spend time for diminishing returns. The straight-line paths of flow matching are what make $64$ enough; on a curved DDPM schedule you would need far more.

If you want to render several clips, do *not* reload the pipeline each time — load once, keep it warm, and loop over prompts, reusing the offload hooks. The first generation pays a warm-up cost (kernel compilation, cache allocation); subsequent ones are faster. If you have the VRAM to skip offload entirely, a resident pipeline on an A100 $80\text{ GB}$ will outrun an offloaded one on a 24GB card by a wide margin per clip, because you stop paying the PCIe transfer on every module swap. Batching multiple clips in one forward pass is possible but rarely worth it on a single card — the $44\text{k}$-token activations are already near the memory ceiling for one clip, so a batch of two often OOMs where two sequential single-clip runs fit. On video, sequential beats batched on memory-bound hardware.

One ergonomic note on the export. `diffusers.utils.export_to_video` wants a list of frames and an `fps`; Mochi's native frame rate is $30$, so pass `fps=30` to keep motion at the speed the model was trained for. If you re-time the clip in post you will get either slow-motion or judder, depending on which way you stretched it — match the export fps to the model's native fps unless you are deliberately re-timing.

## 7. The cost ledger: VRAM, seconds, and the decode wall

Let me put concrete numbers on the cost, with the honesty caveat that every figure is a defensible estimate keyed to a named setup, not a guarantee. The point is to show you *where* the cost lives so you can reason about your own hardware.

The naive footprint first. The 10B denoiser in bf16 is about $20\text{ GB}$ of weights ($10\text{B} \times 2$ bytes). The T5-XXL encoder in bf16 is about $9.5\text{ GB}$ ($4.7\text{B} \times 2$). The VAE is small by comparison, a few hundred megabytes of weights, but its *activations* during decode are the spike. Add it up and a naive all-on-GPU load is north of $30\text{ GB}$ of weights before you have allocated a single activation — which is why an unoffloaded Mochi does not fit on anything short of an A100/H100-class card, and why on a 24GB consumer card you *must* offload.

With `enable_model_cpu_offload()`, the peak is governed by the largest single resident module plus its activations rather than the sum. During denoising that is the 10B denoiser ($\sim20\text{ GB}$ bf16) plus the activations for $44\text{k}$ tokens of full 3D attention — and the attention activations are themselves substantial because the attention matrix is $O(L^2)$ in the worst case (SDPA/FlashAttention keeps this from being fully materialized, which is why it matters that the pipeline uses a memory-efficient attention backend). With offload plus tiling plus bf16, community reports put a full $848\times480$, $163$-frame Mochi generation in roughly the $20$–$24\text{ GB}$ range on a single high-end consumer card, and well within an A100 $40\text{ GB}$ or $80\text{ GB}$ with headroom to spare. Treat those as order-of-magnitude — your exact peak depends on the attention backend, the offload mode, and whether you quantized the text encoder.

#### Worked example: the decode wall on a 24GB card

Here is the failure mode the VAE-tiling flag exists to prevent, made concrete. Suppose you run Mochi on a 24GB card with `enable_model_cpu_offload()` but you *forget* `enable_vae_tiling()`. Denoising proceeds fine — the 10B denoiser at $\sim20\text{ GB}$ plus memory-efficient attention activations sits under $24\text{ GB}$, and you watch $64$ steps complete over several minutes. Then the VAE decode fires. It has to expand a $28 \times 60 \times 106 \times 12$ latent into an $848 \times 480 \times 163 \times 3$ pixel tensor — that output alone is $848 \cdot 480 \cdot 163 \cdot 3 \approx 199$ million values, and the decoder's intermediate activations are several times that. The decode peak blows past $24\text{ GB}$ and you OOM *at the very end*, after paying for all the denoising. This is the single most demoralizing failure in video inference: the model did all the hard work and died on the last step. The fix is one line — `enable_vae_tiling()` — which decodes the latent in spatial tiles so the decode peak stays bounded by the tile size, not the full frame. The lesson generalizes past Mochi: in aggressive-VAE video models, the *decode*, not the denoiser, is frequently the actual VRAM ceiling, and it is the part people forget to budget.

On wall-clock time: a full $5.4$-second Mochi clip at native resolution is a multi-minute render on a single GPU — the dominant cost is $64$ steps $\times$ a 10B forward pass $\times$ full 3D attention over $44\text{k}$ tokens, plus the PCIe transfers that offload imposes. On an A100 $80\text{ GB}$ without offload (everything resident), it is faster because you pay no transfer cost; on a 24GB card with sequential offload, it is slower because every layer is streamed. The trade is the recurring one: VRAM down, seconds up. If you need many clips, the right move is to run on a card with enough memory to skip offload entirely and let the denoiser stay resident.

## 8. Mochi against the open peers

Now place Mochi honestly. The most useful comparison is against the three open models that define the tier — CogVideoX, HunyuanVideo, and Wan — on the axes a practitioner actually weighs: parameters, VAE compression, latent channels, license, resolution, and quality. The headline table:

| Model | Denoiser | VAE compression | Latent ch | License | Native res / length | Notes |
|---|---|---|---|---|---|---|
| **Mochi 1** | 10B AsymmDiT, full 3D attn | 8×8 spatial, 6× temporal (~128×) | 12 | **Apache-2.0** | 848×480, ~5.4 s | Strong motion, T2V only, 480p |
| CogVideoX-5B | 5B MM-DiT, expert AdaLN | 8×8 spatial, 4× temporal (48×) | 16 | Apache-2.0 | 720×480, ~6 s | T2V + I2V, lightest to run |
| HunyuanVideo | 13B dual→single stream | 8×8 spatial, 4× temporal (48×) | 16 | Custom (restricted) | 720p, ~5 s | Top open fidelity, heavy |
| Wan 2.1 | 14B DiT, flow matching | 16×16 spatial, 4× temporal (192×) | 16 | Apache-2.0 | 480p / 720p, ~5 s | Aggressive VAE, multilingual T5 |

![Matrix comparing Mochi CogVideoX HunyuanVideo and Wan across denoiser size VAE ratio latent channels and license](/imgs/blogs/mochi-and-asymmetric-dit-video-3.png)

Read the table as a set of deliberate trades rather than a ranking. Mochi's distinctive cell is the **license**: Apache-2.0, fully permissive, which HunyuanVideo's custom license is not. If you are building a product on top of a base model and you need to fine-tune it, redistribute it, or ship it commercially without a license dance, Mochi (with CogVideoX and Wan) is in the clean tier and HunyuanVideo is not. That alone decides the model for some teams.

Mochi's second distinctive cell is the **VAE temporal compression**: $6\times$ where CogVideoX and HunyuanVideo use $4\times$. Wan goes further still — $16\times$ *spatial* compression — but Wan trades that for needing a more powerful decoder to recover detail. The pattern across the four models is that VAE aggression and resolution are in tension: the more you compress, the harder it is to hold fine spatial detail, and Mochi spends its compression budget on motion-via-cheaper-tokens rather than on resolution, landing at $480p$. CogVideoX and HunyuanVideo compress time less and reach $720p$; HunyuanVideo at $13\text{B}$ holds the open-fidelity crown but costs the most to run.

On **quality**, be honest and specific. Mochi at launch scored well on *motion quality and prompt adherence* — Genmo's framing, and the community consensus, was that Mochi's motion was a standout among open models, a direct payoff of full 3D attention and a big denoiser. Where Mochi lagged was *resolution and occasional warping*: $480p$ native, and the aggressive VAE plus large motion sometimes produced geometric warping or instability, especially on fine textures and faces. On a benchmark like VBench, Mochi competes on the motion-smoothness and dynamic-degree dimensions while ceding ground on imaging-quality and aesthetic dimensions that reward crisp high resolution. The honest one-liner: **Mochi is a motion-and-openness model, not a resolution model.** If you score it on the axis it was built for, it looks strong; if you score it on raw pixel sharpness, it looks mid-tier — and both are true at once.

A word on measuring this honestly, because video benchmarks are gameable. [VBench](/blog/machine-learning/video-generation/the-metrics-of-video-generation) decomposes quality into dimensions, and the *dynamic-degree-versus-stability* tension is the classic gaming hole: a model can score high on "subject consistency" by barely moving — a near-still clip is trivially consistent — and a model with bold motion pays a consistency penalty for the same boldness that makes it good. Mochi sits on the bold-motion side of that trade, so a naive single-number comparison undersells it. If you benchmark Mochi against a low-motion model, fix the *dynamic degree* across both before you compare consistency, or you are measuring timidity, not quality. The same caution applies to FVD: report it on a fixed sample set, a fixed seed, and the same frame count, or the noise swamps the signal.

## 9. The limitations Genmo named, and why they are instructive

The most respectable thing about Mochi's release is that Genmo wrote down its limitations plainly, and those limitations are *legible consequences of the design choices we just dissected* — which is exactly why they teach. Let me take the three they called out.

**Resolution.** Mochi 1's preview is $480p$ native. This is not an oversight; it is the downstream cost of the $128\times$ VAE. A more aggressive compression buys a smaller token budget, which buys a runnable 10B model with full 3D attention, at the price of spatial fidelity. Genmo chose motion and openness over resolution, and they said so. The instructive part: you can *see* the trade in the architecture. The VAE ratio and the native resolution are two ends of the same lever, and Mochi's position on that lever is a coherent choice, not a flaw to be apologized for. (Higher-resolution paths and upscalers were on the roadmap; the *base* preview is a $480p$ model by design.)

**Occasional warping and instability under large motion.** Mochi sometimes produces geometric warping — a limb that bends wrong, a texture that swims — especially when motion between frames is large. This too traces to the design. Large motion means large displacement in pixel space, which the aggressive VAE has to encode and decode through a coarse latent, and the decoder's job of hallucinating fine detail back gets harder when that detail is *moving fast*. Full 3D attention helps coherence, but it cannot fully compensate for a latent that is geometrically coarse. This is the canonical aggressive-VAE failure: great global motion, occasional local warping. If you stress-test Mochi with a high-motion prompt — "a hummingbird's wings, extreme slow motion" — you will find the edge faster than with a slow pan.

**Photorealistic focus.** Mochi was tuned primarily for *photorealistic* generation; it is comparatively weaker on stylized or animated content. This is a *data and tuning* choice rather than an architecture one, but it is the same kind of honest scoping: the model is good at what it was pointed at and weaker outside that. If you need anime, claymation, or heavy stylization, Mochi is not the base to start from — a model trained on a broader stylistic distribution, or one you fine-tune yourself (which Apache-2.0 lets you do freely), will serve better.

There is a fourth limitation that Genmo's framing implies even where the release notes are gentler about it, and it is worth naming because it is the one that bites in production: **prompt sensitivity at the long tail.** A motion-strong model with an aggressive VAE has a wider variance in output quality than a conservative one. The same prompt rendered with two seeds can give you a clean clip and a warped one, because the model is reaching for bold motion and bold motion is exactly where the coarse latent occasionally fails. In practice this means a Mochi pipeline wants a *selection* stage — render several seeds, keep the best — rather than trusting a single generation, and it means your effective cost-per-usable-clip is some multiple of the cost-per-clip. This is not unique to Mochi, but it is sharper for a model that lives on the bold-motion side of the trade. Budget for it: if one in three renders has a warping artifact you would not ship, your real throughput is a third of your nominal throughput, and a 10B model is expensive enough per render that the multiplier matters.

Walk one more stress test to its end, because it is the most instructive failure. Push Mochi past its native clip length by stitching two clips and conditioning the second on the last frame of the first (a poor-man's I2V, since the preview has no real I2V). What breaks is *identity*: the fox in clip two is recognizably a fox, but it is not quite the *same* fox — its coat shade drifts, its proportions shift, the snow texture resets. This is the error-accumulation problem that [autoregressive rollout](/blog/machine-learning/video-generation/long-video-and-autoregressive-rollout) makes rigorous, and Mochi is not built to fight it — its design horizon is one bounded clip. The lesson for a practitioner is decisive: do not try to make Mochi a long-video model by stitching; if you need minutes, reach for a model built for rollout. Mochi is a five-second instrument played at its native length, and it is excellent there and frustrating outside it. Knowing *where* the instrument ends is half of using it well, and the architecture tells you where: the VAE's trained horizon, the quadratic attention bill, and the bounded-clip design all point to the same five-second edge.

Notice the through-line: each named limitation is a *predictable consequence* of a choice you can read off the architecture. That is the gift of an open, legible model. A black-box model with the same limitations would leave you guessing whether the $480p$ was a VAE choice or a training-budget choice or a deliberate product gate; with Mochi you know it is the VAE, because Genmo told you the VAE is $128\times$ and you can do the token arithmetic yourself.

#### Worked example: budgeting a fine-tune of Mochi

Suppose you want to adapt Mochi to your own domain — say, product-demo clips of furniture — and you are deciding whether it is the right base. Walk the budget. The license is Apache-2.0, so you can fine-tune and ship without restriction; that clears the legal gate that HunyuanVideo would not. A video-LoRA fine-tune, rather than a full fine-tune, keeps the trainable parameters tiny: you train low-rank adapters on the AsymmDiT's attention and MLP projections — perhaps $20$–$80\text{M}$ trainable parameters against the $10\text{B}$ frozen base — using `peft`, so the optimizer state and gradients fit where a full fine-tune's would not. Your data is a few hundred to a few thousand clips at Mochi's native $848\times480$, $30$ fps. The catch you must budget for: training needs the *forward and backward* pass over $44\text{k}$ tokens, so even a LoRA fine-tune wants a $40$–$80\text{ GB}$ card (an A100) with gradient checkpointing on, because activations dominate. The decision falls out cleanly: if your target is photoreal, motion-heavy, $480p$-acceptable, and you value a clean license — furniture demos qualify — Mochi is a strong base and a video-LoRA on an A100 is a tractable adaptation. If your target were $1080p$ stylized animation, you would start elsewhere. The architecture tells you which world you are in.

```python
# sketch: video-LoRA over Mochi's AsymmDiT with peft
from peft import LoraConfig, get_peft_model

lora = LoraConfig(
    r=64, lora_alpha=64,
    # adapt the attention + MLP projections of the visual stream
    target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
    lora_dropout=0.0,
)
pipe.transformer = get_peft_model(pipe.transformer, lora)
pipe.transformer.print_trainable_parameters()   # ~tens of millions, base frozen
# then a standard flow-matching training loop with gradient checkpointing on
```

## 10. Why this asymmetry argument generalizes

Step back from Mochi specifically, because the asymmetric-budget idea is the part most worth carrying away, and it is bigger than one model. The general principle is: **when a model processes two sequences of very different lengths, give them proportionally different capacity.** A symmetric architecture is the right default only when the two sequences are comparable in length and difficulty. The moment one sequence dwarfs the other — as video tokens dwarf text tokens — equal capacity is equal *waste*, and reallocating the saved parameters toward the dominant sequence is a strict win at fixed budget.

You see versions of this everywhere once you are looking for it. Encoder-decoder language models give different depths to encoding and decoding. Vision-language models run a small text adapter against a large vision backbone, or the reverse, depending on which modality carries the load for the task. Retrieval-augmented systems spend cheaply on the query and richly on the documents. The pattern is the same: *match capacity to load*. Mochi's contribution to this folklore is a clean, legible instance in the video-diffusion setting, with the numbers public enough that you can verify the argument rather than take it on faith — the visual stream at $3072$, the text stream narrower, the parameters moved where the $44\text{k}$ tokens are.

The honest caveat is that asymmetry is a *budget* optimization, not a *quality* miracle. It lets you spend a fixed parameter budget better; it does not, by itself, make a model better than a larger symmetric model with more total parameters. HunyuanVideo at $13\text{B}$ with a symmetric-ish dual-stream design outscores Mochi at $10\text{B}$ on raw fidelity — partly because it is bigger, partly because it compresses less and reaches higher resolution. Asymmetry is the right move *at a given budget*; it is not a substitute for budget. The clean way to think about it: Mochi got more visual capacity per parameter than a symmetric 10B model would have, and then spent the resolution budget on motion. Both decisions are defensible; neither is magic.

## Case studies: real numbers from the open tier

A few grounded data points to anchor the comparisons, cited to where they come from.

**Mochi 1's launch claim (Genmo, October 2024).** Genmo released Mochi 1 as a $10\text{B}$-parameter diffusion transformer under Apache-2.0 — at release, described as the largest openly released video generation model — with the AsymmDiT architecture (asymmetric visual/text streams, full 3D attention) and the AsymmVAE ($8\times8$ spatial, $6\times$ temporal, $12$-channel latent, $\sim128\times$ compression). Native output $848\times480$ at $30$ fps, roughly $5.4$ seconds per clip, text-to-video only in the preview, conditioned on a frozen T5-XXL. Genmo positioned it on *motion quality and prompt adherence* and named *resolution* and *occasional warping* as known limitations.

**CogVideoX (Zhipu AI / THUDM, August 2024).** The open DiT-T2V model that immediately preceded Mochi in the timeline, at $2\text{B}$ and $5\text{B}$ scales, with a $4\times8\times8$ causal VAE ($48\times$, $16$-channel) and an expert-AdaLN MM-DiT, also Apache-2.0, and — unlike Mochi's preview — *with* an image-to-video variant (`CogVideoXImageToVideoPipeline`). CogVideoX-5B is the lightest of the four to run, which is why it is the usual on-ramp for a 24GB card.

**HunyuanVideo (Tencent, December 2024).** A $13\text{B}$ open model that arrived just after Mochi and took the open-fidelity lead, with a dual-stream-then-single-stream MM-DiT, the same $4\times8\times8$-class VAE, and an MLLM-style text encoder, under a more restrictive custom license. The relevant contrast: HunyuanVideo bought top quality with more parameters and a less permissive license, where Mochi bought openness and motion at $480p$.

**Wan 2.1 (Alibaba, 2025).** A $14\text{B}$ open, Apache-2.0 model whose Wan-VAE pushes spatial compression to $16\times$ (the most aggressive spatial squeeze of the tier), with a multilingual umT5 encoder. Wan is the model that takes the *VAE-as-master-lever* argument furthest, and it is the natural comparison when you want to see what a different point on the compression-versus-fidelity curve looks like.

**How to benchmark Mochi against these peers honestly.** If you actually want to compare Mochi to CogVideoX or Wan rather than take launch claims on faith, the measurement discipline matters more than the numbers. Three rules. First, *fix the dynamic degree* before you compare consistency dimensions: VBench's subject- and background-consistency scores reward stillness, so a fair comparison filters to prompts that demand comparable motion across the models, or it reports consistency *alongside* dynamic degree rather than in isolation. Comparing Mochi's consistency to a near-static model's is comparing courage to caution. Second, *fix the sampling budget*: render every model at the same number of steps and the same CFG scale, or you are measuring sampler tuning, not model quality. Third, for FVD, *fix the reference set, the seed, and the frame count* — FVD on a different sample size or a different reference distribution is not comparable across runs, and FVD is noisy enough that a single comparison can flip on the seed. The honest report is not a single leaderboard number; it is a small table where every cell names the resolution, the length, the step count, and the dtype, and where motion and consistency are reported together so neither hides behind the other. Mochi looks strong under that discipline on the motion axis and mid-tier on the resolution axis, and saying both is the accurate summary.

Read together, the four models trace the open tier's design space: Mochi (motion + temporal compression + Apache-2.0), CogVideoX (light + I2V), HunyuanVideo (fidelity + heavy + restricted), Wan (aggressive spatial VAE + multilingual). Mochi's spot is specific and defensible, and it is the spot you should reach for under specific conditions — which is the next section.

## When to reach for Mochi (and when not to)

Be decisive. Mochi is the right choice when several of these hold:

- **You need a clean Apache-2.0 base you can fine-tune and ship commercially.** This is Mochi's strongest case. If your product fine-tunes a video model and redistributes it, Mochi's license is a feature, not a footnote — and it is the differentiator versus HunyuanVideo.
- **Motion quality matters more than resolution.** Mochi's full 3D attention and big denoiser give it strong, coherent motion. If your clips are motion-forward and $480p$ (or $480p$-then-upscale) is acceptable, Mochi is excellent.
- **Your content is photorealistic.** Mochi was tuned for photoreal; it is in its element there and weaker on heavy stylization.
- **You want a legible architecture to learn from or build on.** No other open video model documents its choices this clearly. If you are doing research that needs a transparent base, Mochi is the teaching object.

![Tree of when to reach for Mochi branching on license needs versus resolution and image to video needs](/imgs/blogs/mochi-and-asymmetric-dit-video-7.png)

Mochi is the *wrong* choice when:

- **You need native high resolution.** At $480p$ native, Mochi is not your $1080p$ model. CogVideoX or HunyuanVideo reach higher native resolution; reach for them, or plan an upscaling stage.
- **You need image-to-video.** The Mochi 1 preview is T2V only. If you must supply a first frame, use CogVideoX's I2V pipeline or another model with a first-frame conditioning path. We covered [why I2V often beats T2V when you can supply a frame](/blog/machine-learning/video-generation/conditioning-video-text-image-motion-camera) in the conditioning post — and if you need it, Mochi's preview cannot give it to you.
- **You need stylized or animated output.** Mochi's photoreal tuning makes it a weak base for anime or claymation; start from a broader model or fine-tune one.
- **You are running on a small card and want it easy.** A 10B-plus-T5 system needs aggressive offload to fit a 24GB card. CogVideoX-5B is dramatically easier to run if your priority is "works on my GPU without ceremony."

The decision, stated as one rule: **reach for Mochi when openness and motion are the constraints and $480p$-photoreal-T2V is acceptable; reach past it when resolution, I2V, stylization, or easy small-card inference are the constraints.**

## How Mochi fits the open line

A final piece of context: Mochi did not appear in a vacuum, and its place in the open text-to-video timeline is part of why it mattered. The open line ran from the flickery U-Net era — ModelScope and Zeroscope, $\sim1.7\text{B}$ U-Nets producing two-second $256p$ clips — through the first DiT-based open T2V models, and Mochi landed as a high-water mark for *open motion quality* at $10\text{B}$ scale under a permissive license, weeks before HunyuanVideo raised the fidelity bar.

![Timeline of open text to video models from ModelScope through CogVideoX to Mochi and HunyuanVideo](/imgs/blogs/mochi-and-asymmetric-dit-video-8.png)

What Mochi contributed to that line was not a leaderboard crown — HunyuanVideo took fidelity, Wan took benchmarks and multilingual reach, Kling and the closed models stayed ahead on resolution. What Mochi contributed was a *legible, permissively-licensed, motion-strong* base with two transplantable ideas: the asymmetric stream budget and the aggressive-but-coherent $128\times$ VAE. Those ideas outlast the specific weights. For where the whole open-and-closed field stands by 2026, see the [landscape synthesis](/blog/machine-learning/video-generation/the-2026-video-model-landscape); for assembling Mochi or any open model into a real pipeline — model selection, LoRA fine-tuning, sampler and offload choices end to end — see the [capstone playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook).

## Key takeaways

- **Asymmetric is the headline.** Mochi's AsymmDiT gives the visual stream a much wider hidden size than the text stream, because video tokens ($\sim44\text{k}$) vastly outnumber text tokens ($\sim$ hundreds). At a fixed parameter budget, narrowing the text stream and moving the saved parameters into the visual stream is a strict win — capacity matched to load.
- **The implementation trick is a shared head dimension.** The two streams project to QKV from *different* widths but into the *same* per-head size, so a video query and a text key are comparable and one joint attention over the concatenated sequence is well-defined.
- **The VAE sets every cost.** The AsymmVAE compresses $8\times8$ spatially and $6\times$ temporally to a $12$-channel latent ($\sim128\times$), turning the $5.4$-second running clip into $\sim44\text{k}$ tokens. The $6\times$ temporal squeeze (vs $4\times$ for CogVideoX/HunyuanVideo) cuts the quadratic attention bill by more than $2\times$ — the VAE, not the denoiser, is the master lever.
- **Flow matching pays off more in video than in images** because each sampling step is a full 10B forward pass over $44\text{k}$ tokens; cutting $250$ steps to $\sim64$ saves on the most expensive operation in the stack.
- **The decode is the VRAM wall.** With offload, the denoiser fits; the $128\times$ VAE decode back to full-resolution pixels is the spike that OOMs you at the last step. `enable_vae_tiling()` is not optional on a 24GB card.
- **Run it with bf16 + `enable_model_cpu_offload()` + `enable_vae_tiling()`**, and 8-bit the T5 encoder and use sequential offload if you are tighter. The trade is always VRAM down, seconds up.
- **Score it on the axis it was built for.** Mochi is a motion-and-openness model: strong motion, Apache-2.0, $480p$, photoreal, T2V-only. It cedes resolution and I2V to peers, and its known limitations (resolution, warping under large motion, photoreal focus) are legible consequences of its design.
- **The asymmetry idea generalizes.** Whenever a model processes two sequences of very different lengths, match capacity to load. Mochi is the cleanest open instance of that principle in video diffusion.

## Further reading

- **Genmo, "Mochi 1" release and model card (2024)** — the primary source for the AsymmDiT, AsymmVAE ($128\times$, $12$-channel), Apache-2.0 license, $480p$ native output, T5-XXL conditioning, and the named limitations; `genmo/mochi-1-preview` on the Hugging Face Hub and the `genmoai/models` GitHub repo for the inference code.
- **Peebles & Xie, "Scalable Diffusion Models with Transformers" (DiT), 2023** — the diffusion-transformer backbone and AdaLN time-conditioning that Mochi's per-stream modulation builds on.
- **Esser et al., "Scaling Rectified Flow Transformers" (Stable Diffusion 3 / MM-DiT), 2024** — the multimodal-DiT joint-attention recipe whose *symmetric* version Mochi makes asymmetric.
- **Lipman et al., "Flow Matching for Generative Modeling," 2023** — the flow-matching objective Mochi trains with; pair it with the [image series treatment](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow).
- **CogVideoX, HunyuanVideo, and Wan technical reports (2024–2025)** — the open peers in the comparison table; read them alongside this post to see the same recipe at different points on the compression-versus-fidelity curve.
- **🤗 `diffusers` documentation, `MochiPipeline`** — the canonical reference for the offload, tiling, and scheduler flags used in the code above.
- Within the series: [why video generation is hard](/blog/machine-learning/video-generation/why-video-generation-is-hard), [video autoencoders and spatiotemporal compression](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression), [video diffusion transformers](/blog/machine-learning/video-generation/video-diffusion-transformers), [flow matching for video](/blog/machine-learning/video-generation/flow-matching-for-video), [the open video frontier](/blog/machine-learning/video-generation/the-open-video-frontier-wan-hunyuanvideo-cogvideox), and the [capstone playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook).
