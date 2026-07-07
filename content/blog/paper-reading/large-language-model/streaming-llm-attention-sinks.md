---
title: "Attention Sinks: How StreamingLLM Lets a Fixed-Window LLM Read Four Million Tokens"
date: "2026-07-07"
description: "A deep read of StreamingLLM: why window attention collapses, what an attention sink actually is, and how keeping four early tokens plus a sliding window turns a 4K-context model into an endless streaming engine — no fine-tuning."
tags: ["paper-reading", "streaming-llm", "attention-sink", "kv-cache", "long-context", "llm-inference", "transformer", "rope", "alibi", "softmax"]
category: "paper-reading"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 30
paper:
  title: "Efficient Streaming Language Models with Attention Sinks"
  authors: "Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis"
  venue: "ICLR 2024"
  url: "https://arxiv.org/abs/2309.17453"
---

> [!tldr]
> - **The claim.** A pretrained LLM with a *finite* attention window (Llama-2's 4K, MPT's 2K) can generate over **4 million tokens** of stable, low-perplexity text — with **no fine-tuning** — if you keep the Key/Value states of just **four initial tokens** alongside a sliding window of recent tokens.
> - **The mechanism.** Transformers dump a huge, semantically-meaningless slice of their attention onto the first few tokens. Xiao et al. call these **attention sinks**. Evicting them (as naive window attention does) tears a dominant term out of the softmax denominator and the model collapses; keeping them anchors the distribution.
> - **The surprise.** The sink tokens carry *almost no information*. Replace the first four tokens with literal line-break characters and perplexity still recovers. What matters is their **absolute position**, not their content.
> - **The payoff.** Against the only quality-competitive baseline (sliding window with re-computation), StreamingLLM is up to **${22.2\times}$ faster** per token at the same memory footprint.
> - **Where it fails.** StreamingLLM does **not** extend the context window. Anything older than the cache is gone forever — it is a *streaming* engine, not a long-memory one.

The extracted figure below is the entire paper in one picture. The rest of this post unpacks each of its four panels, then goes underneath them to the softmax algebra, the position-encoding trick that actually makes it run, and the pre-training tweak that makes it cheaper still.

![Figure 1 from Xiao et al. (2024): four ways to attend over a stream that is far longer than the pre-training window — dense, window, sliding-window-with-recompute, and StreamingLLM.](/imgs/blogs/streaming-llm-attention-sinks-fig1.webp)

## The problem

Imagine a chat assistant that has been running for a week. Every turn appends more tokens to the conversation. A vanilla Transformer decoder caches the Key and Value vectors (the "KV cache") of **every** token it has ever seen, because self-attention lets the current token look back at all of them. Two things break as the stream grows.

**First, memory and latency grow without bound.** After $T$ tokens, the KV cache holds $T$ Key vectors and $T$ Value vectors per layer per head. Decoding token $T+1$ attends over all $T$ of them, so per-token compute grows like $O(T)$ and the cumulative cost of generating the stream grows like $O(T^2)$. A day-long conversation is millions of tokens; the cache alone would swamp the GPU. This is the classic KV-cache blowup that every serving stack fights — see the companion survey on [KV-cache management](/blog/paper-reading/large-language-model/a-survey-on-large-language-model-acceleration-based-on-kv-cache-management) for the zoo of eviction, quantization, and sharing tricks people have thrown at it.

**Second, and more fundamentally, the model cannot generalize past its pre-training length.** Llama-2 was trained on sequences of 4,096 tokens. Feed it a 20,000-token prompt and perplexity explodes, even if you *could* afford the memory — the model has simply never seen relative positions that large, and its position encoding does not extrapolate. Years of work on RoPE interpolation, ALiBi, and context-window extension (surveyed in [efficient attention mechanisms](/blog/paper-reading/large-language-model/efficient-attention-mechanisms-for-large-language-models-a-survey)) push that ceiling higher, but they never make it *infinite*, and they usually require fine-tuning.

The obvious fix is **window attention**: keep only the most recent $L$ tokens in the cache and throw the rest away. Memory is now constant at $L$, per-token cost is constant at $O(L)$, and you never see a relative distance bigger than $L$. It should be perfect. It is a disaster. The moment the very first token slides out of the window, perplexity jumps by three orders of magnitude — from a healthy ${\sim}5$ to over 5,000. Panel (b) of Figure 1 shows exactly this: `O(TL)` cost (good) but `PPL: 5158` (catastrophic).

The only method that keeps quality *and* bounds cost is **sliding window with re-computation** (panel c): for every new token, rebuild the KV cache from scratch over the last $L$ tokens. Quality is excellent (`PPL: 5.43`) but you pay quadratic attention *inside every window on every step*, giving `O(TL²)` — far too slow to deploy.

It helps to place this against the three research directions the paper explicitly distinguishes, because StreamingLLM is often misfiled into the wrong one. **Length extrapolation** (RoPE variants, ALiBi) tries to make a model trained short work when tested long — but current methods still break well before infinity. **Context-window extension** (position interpolation, YaRN, FlashAttention-enabled long training) enlarges the window the model can process in one pass — but a larger window is still finite, and, as the authors note, "extending the context size of LLMs doesn't improve the model's performance beyond the context size." **Improving long-text utilization** tackles whether a model actually *uses* the tokens it can see. Progress in one direction does not imply progress in another. StreamingLLM sits squarely in the first bucket — applying a model to text far exceeding its window — and pointedly does *not* claim the other two: it neither widens the window nor improves how well the model exploits its context. Keeping that boundary in view is the difference between using StreamingLLM correctly and being disappointed by it.

StreamingLLM's contribution is to explain *why* window attention collapses, and then to fix it with a change so small it fits in a sentence: **keep the first few tokens' KV forever.** That gap — between "window attention is broken and nobody quite knew why" and "here is the one-line patch" — is the whole paper.

## Contributions

Tightened into four claims, each of which the rest of the post will earn:

1. **The attention-sink phenomenon.** Autoregressive LLMs allocate a large, persistent fraction of their attention to the first handful of tokens, *regardless of whether those tokens are semantically relevant*. This is an empirical observation about trained Transformers, visualized across every layer and head.
2. **A mechanistic explanation for window-attention failure.** Because those initial tokens carry a dominant chunk of the softmax denominator, evicting them mathematically deforms the attention distribution — the model isn't losing *information*, it is losing its *normalization anchor*.
3. **StreamingLLM.** A training-free inference recipe: cache the KV of four attention-sink tokens plus a rolling window of recent tokens, and assign positions *within the cache* rather than within the original text. It works on Llama-2, MPT, Falcon, and Pythia, across RoPE and ALiBi, out to 4M+ tokens.
4. **A pre-training remedy.** Prepend a single dedicated, learnable "sink token" to every training sample. A model trained this way needs *only that one token* to stream stably, instead of scavenging four arbitrary initial tokens.

## Method

### 1. The attention sink: a grounding rod for leftover attention

**The problem it solves.** Before we can fix window attention, we need to know what the initial tokens are *doing*. If they held crucial information, eviction breaking the model would be unsurprising. The authors show it is stranger than that.

**Intuition.** Picture softmax attention as a voting rule where the votes are forced to sum to exactly one. Each query token must distribute a total "attention budget" of 1.0 across all the tokens it can see. But here is the catch: sometimes a token's own representation already contains everything it needs to predict the next word, and it has *no one it actually wants to attend to*. The budget still has to go somewhere — softmax will not let it vote for "nobody". So the model learns a **designated dumping ground**: a token it can pour the unwanted budget into without side effects. That dumping ground is a *grounding rod* — like the earth wire in household electrics that harmlessly carries away current that has nowhere useful to go. The initial tokens become the grounding rod for surplus attention. Xiao et al. name them **attention sinks**.

**The mechanism, step by step.** Look at the attention maps the paper extracts from Llama-2-7B, averaged over 256 short sentences:

![Figure 2 from Xiao et al. (2024): average attention logits in Llama-2-7B across layers and heads. Layers 0–1 attend locally; every layer above them dumps attention onto the very first token (the bright left column).](/imgs/blogs/streaming-llm-attention-sinks-fig2.webp)

Read it left to right. In the bottom two layers (Layer 0, Layer 1) attention is *local* — recent tokens light up, which is what you would naively expect. But from Layer 2 onward, a single bright vertical stripe dominates every head: **the first token's column**. Every query, no matter how far away, sends a slug of attention back to position 0. The model has universally elected the first token as its sink.

Why the *first* token and not some token in the middle? Because of the causal mask. In autoregressive decoding, token 0 is visible to *every* subsequent token, whereas a token at position 500 is only visible to tokens 500 and later. During training, if the model wants a reliable place to dump surplus attention, the only positions guaranteed to be in-view for all queries are the earliest ones. So gradient descent naturally trains the initial tokens into sinks — they are the only universally-reachable real estate.

**The math.** Attention over $N$ visible tokens turns a vector of logits (pre-softmax scores) $x = (x_1, \dots, x_N)$ into weights via

$$
\text{SoftMax}(x)_i = \frac{e^{x_i}}{e^{x_1} + \sum_{j=2}^{N} e^{x_j}}, \qquad x_1 \gg x_j \text{ for } j \in \{2, \dots, N\}.
$$

Here $x_i$ is the attention logit from the current query to the $i$-th key (a scalar; the query·key dot product scaled by ${1/\sqrt{d_k}}$), and $\text{SoftMax}(x)_i \in (0,1)$ is the resulting attention weight on token $i$. The condition $x_1 \gg x_j$ is the attention-sink phenomenon written as an inequality: the logit to the first token is far larger than to any other, so $e^{x_1}$ dominates the denominator. That single term can be more than half of the entire sum — the appendix measures the first token absorbing **over 50% of all attention** in most layers of Llama-2-7B on 4,096-token inputs.

**A worked micro-example.** Take four tokens with logits $x = (6, 1, 0.5, 0)$, where token 1 is the sink. Exponentiate: $e^{6} \approx 403.4$, $e^{1} \approx 2.72$, $e^{0.5} \approx 1.65$, $e^{0} = 1$. The denominator is ${403.4 + 2.72 + 1.65 + 1 = 408.8}$. So the weights are $(0.987, 0.0067, 0.0040, 0.0024)$. The sink alone eats 98.7% of the budget; the three "real" tokens split the remaining 1.3% in the *ratio* ${2.72 : 1.65 : 1}$. That ratio — the relative preference among the content tokens — is what actually drives the prediction. The sink is just soaking up the slack.

**Why it works / when it's surprising.** The decisive experiment is a substitution. The authors replace the first four tokens with the literal line-break character `\n` and re-measure. On Llama-2-13B (first 65K-token book of PG-19), the numbers from Table 1 are:

| Cache config | Perplexity ↓ |
|---|---|
| `0 + 1024` (pure window) | 5158.07 |
| `4 + 1020` (original first 4 tokens kept) | 5.40 |
| `4"\n" + 1020` (first 4 replaced by line-breaks) | 5.60 |

Swapping the informative opening tokens for meaningless newlines barely moves perplexity (5.40 → 5.60). The sink's *value* is irrelevant; its *position* is everything. That is the counter-intuitive core of the paper — and it is why the fix can be so cheap.

There is a second, quantitative confirmation buried in the appendix worth surfacing, because it turns "the sink gets a lot of attention" into a hard number. The authors take the 4,096th token of a 4,096-token sequence and measure how much of *its* attention lands on the very first token, averaged over 256 sequences and broken out per layer. In all but the bottom two layers, the first token receives attention scores "often exceeding half of the total attention." Half. A single token, thousands of positions away, carrying no task-relevant content, routinely commands more attention than the other 4,095 tokens combined. When you delete a term that large from the denominator, of course the distribution shatters — the surprise is not that eviction breaks the model but that anyone expected window attention to work at all.

The phenomenon also is not a quirk of one architecture. The paper shows the same bright-first-token stripe in Llama-2-70B, and — pushing further — an analogous pattern in *encoder* Transformers: BERT-base concentrates disproportionate attention on the omnipresent `[SEP]` token. This rhymes with the independent discovery that Vision Transformers "need registers" (Darcet et al., 2023) — extra dummy patch tokens that soak up global attention the model would otherwise dump onto random background patches. Attention sinks, `[SEP]` reliance, and ViT registers appear to be three faces of the same softmax-induced behavior: whenever a normalized attention distribution must sum to one, the model manufactures somewhere to park the leftover mass. StreamingLLM is the decoder-only, streaming-inference reading of that universal fact.

### 2. Why window attention collapses

**The problem it solves.** We now have the tool to explain the 5,158 perplexity. Window attention isn't losing important content when it evicts token 0 — Table 1 just told us token 0's content barely matters. So what breaks?

**Intuition.** Go back to the votes-summing-to-one picture. The sink was absorbing, say, 98% of every token's attention budget. Now delete the sink from the cache. The remaining tokens *still* have to receive a total weight of exactly 1.0 — softmax renormalizes over whatever is left. So the 98% that used to harmlessly park on the sink is suddenly force-fed to the recent content tokens. Their attention weights balloon by roughly ${50\times}$. The model is now attending to ordinary tokens with an intensity it never saw during training, and its internal activations go off-distribution. It doesn't degrade gracefully; it detonates.

![The softmax denominator, before and after eviction. Deleting the sink's exp-term does not remove information — it removes the anchor that kept every other weight small.](/imgs/blogs/streaming-llm-attention-sinks-1.webp)

**The mechanism.** The redrawn figure above traces both paths. On the left (window attention), the denominator loses its dominant term $e^{x_1}$. Because softmax weights must still sum to one, the surplus floods the recent tokens; the attention distribution shifts far from anything seen in training, and the model collapses. On the right (StreamingLLM), the sink term stays in the denominator, anchors the sum, and the recent-token weights remain calibrated.

**The math.** Write the full denominator as $Z = e^{x_1} + \sum_{j=2}^{N} e^{x_j}$ and split it into the sink term and the rest, $Z = e^{x_1} + R$ where $R = \sum_{j=2}^{N} e^{x_j}$ is the "residual mass" from the non-sink tokens. The weight on a recent token $k \ge 2$ is:

$$
w_k^{\text{full}} = \frac{e^{x_k}}{e^{x_1} + R} \qquad\text{vs.}\qquad w_k^{\text{evicted}} = \frac{e^{x_k}}{R}.
$$

The inflation factor is their ratio:

$$
\frac{w_k^{\text{evicted}}}{w_k^{\text{full}}} = \frac{e^{x_1} + R}{R} = 1 + \frac{e^{x_1}}{R}.
$$

If the sink held half the mass, then $e^{x_1} \approx R$ and every recent weight *doubles*. If it held 98% ($e^{x_1} \approx 49R$), every recent weight is multiplied by ${\sim}50$. Eviction doesn't remove a token; it removes the large constant in the denominator that was keeping all the other weights small.

**A worked micro-example.** Reuse $x = (6, 1, 0.5, 0)$ but evict the sink (drop $e^6$). The new denominator is $R = 2.72 + 1.65 + 1 = 5.37$. The weights become $(-, 0.506, 0.307, 0.186)$ — the three content tokens now carry the *entire* budget, up from a combined 1.3%. Token 2's weight jumped from 0.0067 to 0.506, a ${75\times}$ increase. Same logits, same content, wildly different distribution. The relative ratios (${2.72 : 1.65 : 1}$) are preserved, but the *absolute* magnitudes the downstream layers consume are unrecognizable.

**When it fails / the fix in one line.** This tells us the cure precisely: we don't need the sink's *content*, we need its *slot in the denominator*. Keep a few initial tokens' KV so $e^{x_1}$ (and its friends) stay in $Z$, and the recent-token weights never inflate. That is StreamingLLM.

### 3. StreamingLLM: rolling KV cache with attention sinks

**The problem it solves.** Combine everything: we want window attention's constant memory and $O(L)$ per-token cost, but without the denominator collapse. The sink analysis says we can have both.

**Intuition.** Split the KV cache into two parts with two different jobs. A tiny, *pinned* set of the earliest tokens does one thing — hold the attention distribution's normalization anchor in place. A larger, *rolling* window of the most recent tokens does the other — carry the actual local context you need to predict the next word. The pinned sinks never move; the rolling window evicts its oldest entry each step like a conveyor belt.

**The mechanism, step by step.** The redrawn cache below (a cleaner take on the paper's Figure 4) shows the layout at the moment we decode token 9, after tokens 4 and 5 have already rolled off:

![The StreamingLLM KV cache: four pinned attention-sink tokens plus a rolling window of recent tokens. Positions are numbered by cache slot, not by original text index — so RoPE sees a contiguous 0–7 window.](/imgs/blogs/streaming-llm-attention-sinks-2.webp)

Concretely, per decoding step:

1. The cache holds `[sink_0, sink_1, sink_2, sink_3]` (pinned) followed by the last few tokens `[…, tok_6, tok_7, tok_8]` (rolling).
2. Compute the new token's query and attend over exactly those cached KVs — a constant ${4 + L}$ of them.
3. Append the new token's KV to the rolling window; if the window is full, evict its oldest entry. The four sinks are *never* evicted.

The paper's default is **four sink tokens**. That number is not arbitrary — it falls out of an ablation we'll see below. The whole method "can be seamlessly incorporated into any autoregressive language model that employs relative positional encoding, such as RoPE and ALiBi."

**The math.** Let the sink budget be $S$ (default $S=4$) and the rolling window be $L$. The cache size is fixed at $S + L$ for the entire stream, so:

$$
\text{memory} = O(S + L) = O(L), \qquad \text{per-token compute} = O(S + L) = O(L),
$$

independent of the total stream length $T$. That is the same asymptotic profile as plain window attention (panel b of Figure 1), and dramatically better than dense $O(T)$-per-token or recompute's $O(L^2)$-per-token. You buy the quality of the recompute baseline at the price of window attention.

**A worked micro-example (pseudocode).** The entire inference-time change, in the shape of real code:

```python
# S = number of attention-sink tokens to pin (paper default: 4)
# L = rolling-window size (recent tokens)
sink_kv   = []   # filled once, from the first S tokens; never evicted
recent_kv = []   # deque of the most recent L tokens' (K, V)

def step(token):
    q, k, v = attn_projections(token)          # query/key/value for this token
    if len(sink_kv) < S:
        sink_kv.append((k, v))                 # still filling the sink slots
    else:
        recent_kv.append((k, v))
        if len(recent_kv) > L:
            recent_kv.popleft()                # evict oldest recent token

    cache = sink_kv + list(recent_kv)          # attend over S + L keys, always
    keys   = stack([k for k, _ in cache])
    values = stack([v for _, v in cache])
    logits = (q @ keys.T) / sqrt(d_k)          # shape [S + L]
    weights = softmax(assign_positions(logits, cache))  # see technique #4
    return weights @ values                    # context vector, shape [d]
```

Note what is *absent*: no re-computation, no growing tensors, no fine-tuning, no new parameters. The only non-obvious call is `assign_positions`, which is technique #4 and is where a naive implementation quietly fails.

**Why it works / when it fails.** It works because — per techniques #1 and #2 — the four sinks hold the softmax anchor in place, so recent-token weights stay in-distribution no matter how long the stream runs. It fails to do something it never promised: it cannot answer a question about token 12 once token 12 has rolled out of the window. The cache is your entire accessible past. We'll quantify that hard boundary in the results.

### 4. Assigning positions inside the cache (the detail that makes it run)

**The problem it solves.** There is a landmine in step 2 above. When you attend over `[sink_0..3, tok_6, tok_7, tok_8]`, what *positions* do you feed the position encoding? The tokens' original text indices are `[0, 1, 2, 3, 6, 7, 8]` — there is a **gap** where 4 and 5 used to be, and the numbers keep climbing past the pre-training length. Feed those raw indices to RoPE and you have reintroduced both problems StreamingLLM was supposed to kill: an out-of-distribution large position, and a discontinuity.

**Intuition.** The position encoding should describe *where a token sits in the cache the model is actually looking at*, not where it once sat in a text the model has mostly forgotten. Renumber the cache from zero every step, contiguously. The sinks and the recent window collapse into one gap-free ruler `[0, 1, 2, …, S+L-1]`.

**The mechanism.** This is exactly what the redrawn cache figure above depicts: the four sinks keep positions 0–3, and the recent tokens `tok_6, tok_7, tok_8` are re-labeled to the *next contiguous slots* 4, 5, 6 — sliding left to fill the hole left by the evicted 4 and 5. In the paper's own words: "if the current cache has tokens `[0, 1, 2, 3, 6, 7, 8]` and is in the process of decoding the 9th token, the positions assigned are `[0, 1, 2, 3, 4, 5, 6, 7]`, rather than the positions in the original text, which would be `[0, 1, 2, 3, 6, 7, 8, 9]`." The distances the model sees are always small and always contiguous.

**The math.** Let the cache, in order, be tokens $c_0, c_1, \dots, c_{m-1}$ (here $m = S + L$). Regardless of each token's original text index $\text{idx}(c_r)$, StreamingLLM assigns the encoding position

$$
\text{pos}(c_r) = r, \qquad r = 0, 1, \dots, m-1.
$$

For RoPE specifically, the implementation detail matters: you **cache the Keys before the rotary transformation is applied**, then rotate by the *cache-relative* position $r$ at each decoding step. If you instead cached the already-rotated Keys, each key would be frozen at its original text angle and you could not re-index it. For ALiBi it is even simpler — the linear distance bias is applied over the contiguous cache positions, so the "jumping" bias that the text-index gap would create becomes a smooth contiguous ramp.

**A worked micro-example.** Cache contents (original text indices) and their assigned encoding positions, one step of decoding:

| cache slot $r$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| token (text index) | 0 | 1 | 2 | 3 | 6 | 7 | 8 |
| **assigned position** | 0 | 1 | 2 | 3 | **4** | **5** | **6** |

Token 6 is told it lives at position 4; the relative distance between the query and token 8 is $r_\text{query} - 6$, never the text distance. No gap, no runaway index.

**Why it works / when it fails.** This is the quiet reason StreamingLLM extrapolates: it never lets the position encoding see a distance larger than the cache, so RoPE/ALiBi always operate inside their trained regime. The paper flags it emphatically — "this method of assigning positional embedding within the cache is crucial to StreamingLLM's functionality." Skip it and the method silently reverts to broken length-extrapolation. It also quietly bounds what StreamingLLM can *mean*: since positions only describe the cache, the model has no representation of *how long ago* an evicted token was — absolute time is not recoverable.

### 5. Pre-training a dedicated sink token

**The problem it solves.** Everything so far is training-free — it rescues models that already exist. But *why* do those models need *four* sinks instead of one? And can a model be trained to need fewer, so streaming is even cheaper?

**Intuition.** A vanilla model was never given a consistent, always-present "dump here" token, so it improvised — it recruited several of the earliest tokens as a committee of sinks, because no single one was reliably present across all training samples (Llama-2 prepends `<s>` before chunking, so the token at position 0 is effectively random). Give the model one dedicated, always-there sink from the start of pre-training, and it will concentrate all its surplus attention on that single token — no committee needed.

**The mechanism.** Xiao et al. compare three pre-training regimes, redrawn below:

![Three ways to supply an attention sink during pre-training. Vanilla recruits four arbitrary initial tokens; Zero Sink helps only slightly; a single learnable Sink Token makes one token suffice.](/imgs/blogs/streaming-llm-attention-sinks-3.webp)

- **Vanilla** — standard softmax, no designated sink. Streams stably only after re-introducing ${\sim}4$ initial tokens.
- **Zero Sink** — replace softmax with **SoftMax-off-by-one**, which adds a $+1$ to the denominator. This is mathematically equivalent to prepending a virtual token whose Key and Value are all-zero (its logit is $e^0 = 1$, i.e. the added ${1}$). It gives the surplus attention a place to go *without consuming a real token*.
- **Learnable Sink Token** — prepend one real, *trainable* placeholder token to every single training sample. Its KV is learned; it becomes the committed sink.

**The math.** Standard softmax forces the weights to sum to one over the content tokens. SoftMax-off-by-one relaxes that:

$$
\text{SoftMax}_1(x)_i = \frac{e^{x_i}}{1 + \sum_{j=1}^{N} e^{x_j}}.
$$

The lone $+1$ in the denominator is a constant that is *not* attached to any content token. Now the weights over real tokens sum to $\frac{\sum_j e^{x_j}}{1 + \sum_j e^{x_j}} < 1$, and the leftover slack — the model's "I don't want to attend to anything" mass — is absorbed by that ${1}$ instead of being forced onto content. The all-zero-KV "Zero Sink" is the concrete realization: a phantom token contributing exactly $e^0 = 1$ to $Z$ and a zero vector to the output.

**A worked micro-example (the payoff table).** From Table 3 of the paper — three 160M-parameter models pre-trained identically on the de-duplicated Pile, streaming perplexity on the first PG-19 sample, where cache config `x+y` means $x$ re-introduced initial tokens plus $y$ recent tokens:

| Method | `0+1024` | `1+1023` | `2+1022` | `4+1020` |
|---|---|---|---|---|
| Vanilla | 27.87 | 18.49 | 18.05 | 18.05 |
| Zero Sink | 29214 | 19.90 | 18.27 | 18.01 |
| **Learnable Sink Token** | 1235 | **18.01** | 18.01 | 18.02 |

Read the `1+1023` column: with just *one* token re-introduced, the Learnable Sink model is already at its floor (18.01), while Vanilla still needs to climb to `4+1020` to bottom out (18.05). The dedicated sink concentrates the sinking behavior into a single token. Interestingly, Zero Sink *alone* (`0+1024`) is the worst of all (29214) — the off-by-one helps only once you also add real initial tokens; it "alleviates the attention sink problem to some extent" but does not eliminate the model's reliance on real initial tokens the way a learned token does.

**Why it works / when it fails.** It works because a consistently-present token is the ideal gradient target for "dump surplus here" — the model learns to route all its off-by-one mass to that one KV. Crucially, it is *free at deployment*: Table 4 shows the sink-token model matches the vanilla model on seven zero-shot NLP benchmarks (ARC, HellaSwag, LAMBADA, OpenBookQA, PIQA, Winogrande), and Figure 6 shows identical pre-training loss curves. The catch: it only helps *future* models you train yourself. Everyone deploying an existing checkpoint is stuck with the four-sink workaround — which, happily, works fine.

## Experiments & results

The evaluation spans four model families deliberately chosen to cover both dominant position encodings: **Llama-2, Falcon, Pythia** (RoPE) and **MPT** (ALiBi). Unless stated, StreamingLLM uses four sink tokens.

### Language modeling stays flat to 4M+ tokens

The headline stability plot compares four methods on 20K-token texts across all four families:

![Figure 3 from Xiao et al. (2024): log-perplexity vs input length. Dense attention breaks past its pre-training window; window attention collapses once the cache fills and initial tokens evict; StreamingLLM tracks the recompute oracle flat.](/imgs/blogs/streaming-llm-attention-sinks-fig3.webp)

Three trends repeat across every model: **dense attention** spikes once the input passes the pre-training window; **window attention** collapses the instant the cache fills and the first token is evicted (the orange curve's cliff); **StreamingLLM** stays flat and nearly overlaps the sliding-window-with-recompute oracle. Extending the x-axis, the paper reports stable perplexity out to **4 million tokens** across Llama-2-[7,13,70]B, Falcon-[7,40]B, Pythia-[2.8,6.9,12]B, and MPT-[7,30]B — the point being not that 4M is a magic number but that the curve has no upward trend at all.

### How many sink tokens? Four.

The ablation that justifies the default (Table 2), streaming perplexity over 400K concatenated PG-19 tokens, config `x+y`:

| Model | `0+2048` | `1+2047` | `2+2046` | `4+2044` | `8+2040` |
|---|---|---|---|---|---|
| Falcon-7B | 17.90 | 12.12 | 12.12 | 12.12 | 12.12 |
| MPT-7B | 460.29 | 14.99 | 15.00 | 14.99 | 14.98 |
| Pythia-12B | 21.62 | 11.95 | 12.09 | 12.09 | 12.02 |
| Llama-2-7B* | 3359.95 | 11.88 | 10.51 | 9.59 | 9.54 |

<small>*Llama-2-7B row uses `0+4096 … 8+4088`.</small>

Two reads. First, going from zero to one sink is the giant leap — MPT falls from 460 to 15, Llama-2 from 3360 to 11.88. Second, most models are basically done by one or two sinks, but **Llama-2 keeps improving to four** (11.88 → 10.51 → 9.59) before flattening. Four is the smallest number that satisfies the greediest model in the suite; it is a floor, not a universal constant.

### Streaming question-answering: window attention returns garbage

The most striking table is real multi-round QA. All ARC question–answer pairs are concatenated into one long stream and fed to instruction-tuned Llama-2-Chat models, scoring each answer by exact match (Table 5, cache size 1024):

| Method | Llama-2-7B-Chat (ARC-E / ARC-C) | 13B (E / C) | 70B (E / C) |
|---|---|---|---|
| One-shot (sample-by-sample) | 71.25 / 53.16 | 78.16 / 63.31 | 91.29 / 78.50 |
| Dense | OOM | OOM | OOM |
| Window attention | 3.58 / 1.39 | 0.25 / 0.34 | 0.12 / 0.32 |
| **StreamingLLM** | **71.34 / 55.03** | **80.89 / 65.61** | **91.37 / 80.20** |

Dense attention runs out of memory. Window attention scores essentially **zero** — once the stream exceeds the cache, it emits random tokens. StreamingLLM *matches or slightly beats* the one-shot baseline where each question is asked in isolation, meaning the streaming format costs nothing. This is the practical headline: the same model, run as an endless stream, answers as well as it does one prompt at a time.

### StreamEval: where the cache boundary bites

To probe the *recall* limits honestly, the authors build **StreamEval**, a benchmark inspired by LongEval but designed for the streaming setting. New information arrives continuously ("the REGISTER_CONTENT in line 20 is `<45603>`"), and every 10 lines the model is queried about a value that appeared **20 lines earlier** — deliberately mimicking a chat where questions concern *recent* context, not the distant past. With StreamingLLM, Llama-2 variants hold reasonable accuracy even as the running input approaches 120K tokens, while dense attention dies at the pre-training length and window attention dies at the cache size. Crucially, the authors also show StreamingLLM *composes* with context-extension methods: swapping in LongChat-7B-32K or Llama-2-7B-32K-Instruct simply widens the cache, letting the streaming window capture more local context.

But the appendix draws the boundary with a scalpel. Table 7 sweeps the query-to-answer distance on StreamEval for Llama-2-7B-32K-Instruct across cache configs. Accuracy is high while the answer sits inside the cache and falls off a cliff the instant it does not:

| Line distance (≈ tokens) | cache `4+2044` | `4+4092` | `4+8188` |
|---|---|---|---|
| 20 (≈460) | 85.80 | 84.60 | 81.15 |
| 100 (≈2300) | 0.00 | 61.60 | 50.10 |
| 200 (≈4600) | 0.00 | 0.00 | 62.75 |
| 400 (≈9200) | 0.00 | 0.00 | 0.00 |

Read the `4+2044` column: the moment the answer is more than ~2K tokens back (past the cache), accuracy is a flat **0.00%**. Widening the cache pushes the cliff out but never removes it. This is the empirical statement of the limitation baked into technique #4: positions describe the cache, so anything evicted is *gone*, not merely far. StreamingLLM streams; it does not remember. LongBench results (Appendix Table 8) tell the same story from the other side — a `4+3496` cache underperforms a middle-truncation baseline on long-document QA, and only matching the sink budget to the truncation size restores parity, "demonstrating that StreamingLLM's effectiveness is contingent on the information within its cache."

### Efficiency: up to 22.2× faster at matched memory

Against the only quality-competitive baseline (recompute), the efficiency plot:

![Figure 10 from Xiao et al. (2024): per-token latency and memory vs cache size for Llama-2-7B and 13B. StreamingLLM's latency grows linearly with cache size; the recompute baseline grows quadratically, at nearly identical memory.](/imgs/blogs/streaming-llm-attention-sinks-fig4.webp)

Recompute's per-token latency rises **quadratically** with cache size (rebuilding an $L$-token window is $O(L^2)$); StreamingLLM's rises **linearly**. At a 4096 cache on Llama-2-13B, recompute takes 2355 ms/token versus StreamingLLM's 106 ms/token — a **${22.2\times}$** speedup — while memory stays essentially equal (both hold a comparable window). You are not trading memory for speed; you are getting speed for free by not recomputing.

### The ablation that undercuts the hype: bigger cache ≠ better

Table 6 is the paper's most honest result. Increasing the rolling window does **not** monotonically lower perplexity:

| Model | `4+252` | `4+508` | `4+1020` | `4+2044` |
|---|---|---|---|---|
| Falcon-7B | 13.61 | 12.84 | 12.34 | 12.84 |
| MPT-7B | 14.12 | 14.25 | 14.33 | 14.99 |
| Pythia-12B | 13.17 | 12.52 | 12.08 | 12.09 |

Falcon *worsens* from `4+1020` to `4+2044`; MPT worsens monotonically. As the authors put it, "increasing the cache size doesn't consistently yield a decrease in perplexity, showing these models may not fully utilize the provided context." A bigger window is more tokens the model was never good at using — a limitation StreamingLLM inherits rather than fixes.

**What's load-bearing that might not transfer.** The four-sink default is tuned on this model suite; a future architecture with a different sink structure (or a trained-in sink token) could need a different budget. The perplexity numbers are language-modeling on PG-19 and streamed ARC — they say nothing about tasks that genuinely need the evicted middle. And every result assumes standard softmax attention; a model built on an [attention-sink-free design](/blog/paper-reading/large-language-model/gated-attention-for-large-language-models-non-linearity-sparsity-and-attention-sink-free) — gated attention that removes the sink outright — sidesteps this entire failure mode and would not benefit from the patch.

## Critique

**What's genuinely strong.** The paper does the rarest thing in systems-ML: it *explains* before it *fixes*. The softmax-denominator argument is falsifiable, and they falsify the obvious alternative (that initial tokens carry information) with the line-break substitution. The fix is a few lines of code, needs no fine-tuning, and was adopted almost immediately by TensorRT-LLM, HuggingFace Transformers, and MLC-LLM — real-world validation that the underlying model of the phenomenon is right. The position-within-cache detail (technique #4) is the kind of thing that separates a method that works in a notebook from one that ships.

**What's weak or oversold.** The framing invites a misread the paper mostly resists but the discourse did not: StreamingLLM is repeatedly described near "4 million tokens" and "infinite length", but Table 6 and Appendix C make clear it has **no more usable memory than its cache**. Appendix Table 7 is brutal about this — on StreamEval, accuracy is high while the query's answer sits within the cache and drops to **0.00%** the moment the answer is older than the cache size. "4 million tokens" means "generates coherently while streaming across 4M tokens", not "remembers 4M tokens." That distinction is technically stated but easy to lose, and a lot of downstream excitement lost it.

**What ablation is missing.** The sink-token pre-training story rests on **160M-parameter** models. That is small; whether a single learned sink still suffices at 7B–70B scale (where vanilla models recruit four sinks) is exactly the question left open, and it is the expensive one. There is also no ablation on *which* four tokens — always the first four, but no test of, say, four tokens sampled from the first 64, which would separate "absolute position 0–3" from "any early token". The linebreak substitution hints the answer, but a positional sweep would nail it.

**What would change my mind.** If someone trained a 7B+ model with a single learnable sink token from scratch and showed it *still* needs only one sink to stream stably — matching vanilla downstream quality — I would upgrade "promising 160M result" to "this is how all LLMs should be pre-trained." Conversely, if the four-sink patch turned out to degrade on a task that depends on genuine long-range recall in a way recompute does not, I would downgrade StreamingLLM from "general streaming solution" to "great for chat, unsafe for long-document work" — which is roughly where the LongBench appendix (Table 8) already gently points.

## What I'd build with this

These are my extrapolations, not the paper's claims.

1. **Sink-aware KV-cache quantization.** Since a handful of sink tokens absorb >50% of attention mass and their *content* barely matters, quantize the entire rolling window aggressively (int4) but keep the four sinks in higher precision — the calibration anchor is exactly where precision is worth spending. Pairs naturally with the tricks in the [KV-cache management survey](/blog/paper-reading/large-language-model/a-survey-on-large-language-model-acceleration-based-on-kv-cache-management).
2. **A "sink + landmark" hybrid.** StreamingLLM forgets the middle entirely. Combine its pinned sinks with a sparse set of *retrieved* landmark tokens (summaries of evicted spans) to buy back some long-range recall without paying dense attention — explicitly attacking the Appendix-C failure where the answer has scrolled out of the window.
3. **Sink tokens as an interpretability probe.** The bright first-token column is a clean, reproducible signal. Track how sink mass redistributes when you inject a dedicated sink token mid-training, as a diagnostic for when a model has "finished" learning its attention-dumping behavior.
4. **Trained-in sinks for long-context MoE models.** Million-token-context designs like [DeepSeek-V4](/blog/paper-reading/large-language-model/deepseek-v4-million-token-context-moe) still pay for every cached token; a dedicated sink token plus StreamingLLM-style pinning could bound the *effective* attention span while keeping the trained context available for the cases that need it.
5. **Auto-tuning the sink budget.** Rather than hard-coding four, measure the first-token attention mass per model at load time and set $S$ to cover, say, 95% of the sink mass — turning the Table 2 ablation into a one-line calibration step.

## References

- **Paper.** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR 2024. [arXiv:2309.17453](https://arxiv.org/abs/2309.17453).
- **Code.** [github.com/mit-han-lab/streaming-llm](https://github.com/mit-han-lab/streaming-llm).
- **Related work referenced here.** Su et al., *RoFormer / RoPE* (arXiv:2104.09864); Press et al., *ALiBi* (ICLR 2022); Miller, *Attention Is Off By One* (2023); Darcet et al., *Vision Transformers Need Registers* (2023).
- **Sibling posts on this blog.** [A survey on KV-cache management](/blog/paper-reading/large-language-model/a-survey-on-large-language-model-acceleration-based-on-kv-cache-management) · [Efficient attention mechanisms survey](/blog/paper-reading/large-language-model/efficient-attention-mechanisms-for-large-language-models-a-survey) · [Gated attention: attention-sink-free](/blog/paper-reading/large-language-model/gated-attention-for-large-language-models-non-linearity-sparsity-and-attention-sink-free) · [DeepSeek-V4 million-token context](/blog/paper-reading/large-language-model/deepseek-v4-million-token-context-moe)
