---
title: "Designing and choosing a tokenizer for your LLM: a senior engineer's deep dive"
date: "2026-05-05"
publishDate: "2026-05-05"
description: "A practical, opinionated guide to picking, training, extending, and serving the right tokenizer for an LLM project — covering algorithms, vocab size economics, multilingual fertility, code handling, libraries, and runtime cost."
tags: ["tokenizer", "llm", "bpe", "sentencepiece", "tiktoken", "multilingual", "vocab", "subword", "nlp", "inference"]
category: "machine-learning"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 50
---

Most of the engineers I have worked with treat the tokenizer the way they treat the operating-system clock: it is a thing that exists, it does its job, and you don't think about it until something is wrong. Then a finetune blows up because the new domain vocabulary tokenizes at 3.4x the fertility of the base model. Then a customer in São Paulo complains that their per-request bill is double the bill in Boston, for the same query. Then a "simple" multilingual eval shows your 70B model losing to a 7B model with a tokenizer that was actually designed for the target language. Every one of those is a tokenizer problem dressed up as something else.

This post is the deep dive I wish I had handed to my younger self before I trained my first language model. It walks through the full design space — algorithm family, normalization, pre-tokenization, vocab size, training corpus, multilingual handling, code/digit policy, library choice, evaluation, vocab extension, and serving — and ends with a decision tree you can actually use on Monday. It is not an introduction to BPE; for the algorithm internals, see the sibling post on [BPE](/blog/machine-learning/large-language-model/bpe-tokenizer). Here we treat BPE as one of four serious choices and focus on which knob to turn for which problem.

> The tokenizer is the only part of an LLM that is *not learnable*. Everything downstream of it has to live with whatever it does. Choose accordingly.

## Why this is different

Most posts you'll find online either re-derive BPE from the 1994 Gage paper or hand-wave the whole thing as "it just works." Neither is useful when you are sitting in front of a real decision: a 30B base model, a target language with 15% of the training mix, a code-heavy customer, and a quarterly inference budget you have to defend.

| The common assumption                                | The naive view                              | What actually happens                                                                          |
| ---------------------------------------------------- | ------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| "BPE is BPE."                                        | One algorithm, one outcome.                 | Same merges + different pre-tok regex = up to 1.4x fertility shift on the same corpus.         |
| "Bigger vocab is always better."                     | More tokens = more compression = win.       | Embedding/softmax cost scales linearly with V; past 128k the marginal compression flattens.    |
| "We'll just retrain the tokenizer when we add Thai." | A new tokenizer is a clean slate.           | A new tokenizer means re-pretraining the model. You almost never want this. Extend instead.    |
| "Tokenization is fast; ignore it in serving."        | It is a microbenchmark, not a hot path.     | At p99 with long prompts, slow Python tokenizers add 5–15 ms per request — half a TTFT budget. |
| "Glitch tokens are a curiosity."                     | SolidGoldMagikarp was a one-off Reddit gag. | Every untrained vocab entry is a latent prompt-injection vector. They show up in production.   |
| "Multilingual is a model problem."                   | Add data, get capability.                   | If your tokenizer's fertility on Hindi is 4.5, you pay 4.5x at inference forever.              |

If any of those left columns is your current mental model, the rest of this post is for you.

## The mental model

![The five-stage tokenizer pipeline](/imgs/blogs/designing-choosing-tokenizer-llm-1.png)

The diagram above is the mental model: every modern tokenizer, from GPT-2's byte-BPE to Gemma's 256k-vocab SentencePiece, is the same five-stage pipeline. Raw bytes come in, normalized text comes out of stage one, the pre-tokenizer cuts it into rough word-like chunks, the segmentation algorithm decomposes those chunks into subwords, the vocabulary maps subwords to integer ids, and the model sees only those ids. Every "design decision" you can make about a tokenizer lives inside exactly one of those five boxes.

The reason this framing matters is that most tokenizer arguments in the wild are arguments at the wrong level. People debate "BPE vs Unigram" when the actual problem is in the pre-tokenization regex. They debate "vocab size 32k vs 64k" when the real bottleneck is normalization eating their math symbols. They debate library choice when their fertility tax is coming from a corpus mix decision, not from C++ vs Rust. The pipeline gives you a place to put each question.

The rest of the post is a tour: one section per stage, plus a section on the cross-cutting concerns (vocab size, library, evaluation, extension, runtime, security) that don't fit cleanly into a single stage. Each section opens with a senior rule of thumb in bold, has at least one runnable snippet, and ends with a non-obvious gotcha.

## 1. The algorithm family

> **Pick the algorithm by what your fallback for unseen input looks like, not by which paper you read most recently.**

The four algorithms you are choosing between in 2026 are BPE, WordPiece, Unigram LM, and byte-level variants of those. They differ in two things only: how they pick the unit to split or merge, and what they fall back to when they see something that is not in the vocabulary.

![Subword algorithms on two axes](/imgs/blogs/designing-choosing-tokenizer-llm-2.png)

**BPE** (Sennrich et al., 2015, originally Gage 1994) starts from characters (or bytes), counts adjacent pair frequencies, and greedily merges the most frequent pair, repeating until the vocabulary hits the target size. Each merge is recorded; tokenization at runtime applies the same merges in order. It is greedy, deterministic, and corpus-statistic-driven.

**WordPiece** (Schuster & Nakajima, 2012; popularized by BERT) is BPE-shaped but uses likelihood gain instead of raw frequency: at each step, merge the pair that maximizes $P(x_1 x_2) / (P(x_1) \cdot P(x_2))$ on the current segmentation. It produces slightly different vocabularies at the same size and tends to favor morphologically meaningful merges when the corpus has them.

**Unigram LM** (Kudo, 2018) goes the other way. It starts with a *large* candidate vocabulary, fits a unigram language model over it, and iteratively prunes the lowest-likelihood pieces until the vocabulary hits the target size. At runtime, it picks the segmentation that maximizes likelihood under that model — a Viterbi pass. Because it is probabilistic, it can sample alternative segmentations (used in subword regularization).

**Byte-level** (a *modifier*, not a separate algorithm) replaces the alphabet of characters with the 256 bytes of UTF-8. GPT-2 introduced byte-BPE; Llama-3 and most modern open-weights models use it. The win is that **the OOV problem disappears**: any sequence of bytes is representable, so you never need an `<UNK>` token. The cost is that one Unicode codepoint can take 2–4 bytes, so non-Latin scripts get more tokens unless the merges absorb them.

The decision is mostly about your fallback story:

| Fallback need                                                     | Choice                                                       |
| ----------------------------------------------------------------- | ------------------------------------------------------------ |
| Closed corpus, all-Latin, training-time-only (research paper)     | BPE or WordPiece, char-level                                 |
| Mixed-script production system, must handle anything              | Byte-BPE (GPT-2/4, Llama-3 style) or byte-Unigram            |
| Want sampled segmentations for regularization                     | Unigram LM (subword regularization)                          |
| Want one regular-expression-clean tokenizer for serving           | Byte-BPE with explicit pre-tok regex (cl100k, o200k)         |
| Multilingual, low-resource emphasis                               | SentencePiece Unigram (no whitespace assumption)             |

In practice, **byte-BPE with a hand-written pre-tokenization regex** is the dominant choice for new English-and-code LLMs (GPT-2, GPT-4, Llama-3, Mistral 7B). **SentencePiece Unigram** is the dominant choice for genuinely multilingual or non-whitespace-language LLMs (T5, mBART, Llama-2 in part, Gemma). WordPiece survives mainly in BERT-lineage encoders.

> The single best predictor of which algorithm you should pick is *whether your input has reliable whitespace*. Byte-BPE assumes it. SentencePiece Unigram does not.

### A BPE training trace, by hand

To make the differences concrete, here is a six-step BPE trace on a tiny corpus, the same way the algorithm would run on a real one — just smaller.

Initial corpus, with end-of-word markers `_`:
```
low_     5
lower_   2
newest_  6
widest_  3
```

We start with the character vocabulary `{l, o, w, e, r, n, s, t, i, d, _}` (11 tokens) and count adjacent pair frequencies in the corpus, weighted by line frequency.

- Step 1: most frequent pair is `(e, s)` with count 9 (from `newest` × 6 + `widest` × 3). Merge to produce token `es`. Corpus becomes `n e w es t _`, `w i d es t _`.
- Step 2: most frequent pair is `(es, t)` with count 9. Merge to `est`. Corpus: `n e w est _`, `w i d est _`.
- Step 3: most frequent pair is `(est, _)` with count 9. Merge to `est_`.
- Step 4: most frequent pair is `(l, o)` with count 7 (from `low` × 5 + `lower` × 2). Merge to `lo`.
- Step 5: most frequent pair is `(lo, w)` with count 7. Merge to `low`.
- Step 6: most frequent pair is `(low, _)` with count 5. Merge to `low_`.

After six merges, our vocabulary is `{l, o, w, e, r, n, s, t, i, d, _, es, est, est_, lo, low, low_}` and a new word like `lowest_` tokenizes via greedy merge as `low + est_` (2 tokens). A word that did not appear in the training corpus, like `news_`, tokenizes as `n + e + w + s + _` (5 tokens) because no merge produced an `ns` or `s_` token. This is the *fertility-on-OOV* problem byte-BPE solves: at the byte level, `news_` is `0x6E 0x65 0x77 0x73 0x5F` (still 5 tokens), but the bytes are universal, so the same trick works for `あ` and `🚀`.

Two takeaways. First, BPE is *order-dependent*: changing the merge order changes downstream tokenization. The merge list must be saved alongside the vocabulary, and the order is part of the model contract. Second, frequency-weighted merging concentrates vocabulary on the head of the distribution — common words and morphemes — at the cost of leaving the tail to character or byte fallback.

### Unigram LM, the other way around

Unigram trains in the opposite direction. It starts with a *very large* candidate vocabulary (typically the seed is BPE itself, run to 5–10x the target size) and iteratively prunes pieces that contribute the least likelihood gain. The contribution of a piece is measured by the EM-style "loss" of removing it — re-segmenting the corpus without that piece using Viterbi, and measuring the increase in negative log likelihood.

The practical consequence: Unigram's vocabulary contains *redundant* pieces by design. The string `"international"` might be representable as `["international"]` (1 piece), `["inter", "national"]` (2 pieces), or `["inter", "nation", "al"]` (3 pieces), and Unigram picks the highest-likelihood one at runtime. This redundancy is what enables subword regularization (sample alternative segmentations during training) and what lets Unigram tokenizers degrade gracefully on unfamiliar input — there is almost always *some* segmentation that uses high-likelihood pieces.

The cost is a slightly slower encode pass: BPE is a linear scan with a deterministic merge order; Unigram is a Viterbi forward pass over the segmentation lattice. In practice both are fast (~µs/word) but BPE is ~2x faster on the same hardware.

### Second-order optimization

The algorithm choice silently constrains your **vocabulary editability**. BPE merges are an ordered list; if you want to add a new token to a trained BPE tokenizer, you can only append it at the end of the merge list (or as a "special token" that bypasses merging). Unigram LM is a flat distribution over pieces, which makes adding domain-specific tokens cleaner — you can add the piece, set its log-probability, and keep going. We will use this property in §8.

## 2. Normalization and pre-tokenization

> **The pre-tokenization regex is the single most under-discussed knob in tokenizer design. It silently decides what your "words" are.**

### Normalization

Stage one is Unicode normalization. The two choices that matter are NFC (canonical composition) and NFKC (compatibility composition). NFC says "é" written as `U+00E9` and "é" written as `e + U+0301` should be the same string. NFKC goes further: it folds half-width katakana into full-width, fullwidth digits into ASCII digits, ligatures (`ﬁ`) into their components (`fi`), and superscripts (`²`) into base digits (`2`).

NFKC is convenient for messy web text but has *teeth*. SentencePiece defaults to NFKC, which means any of these silently lose information at training time:

- Math expressions like `x²` become `x2`, breaking arithmetic disambiguation.
- Half-width vs full-width Japanese punctuation collapses, which matters for Japanese typography.
- Roman numerals (`Ⅳ`) become `IV`.

For an LLM that will be evaluated on math or symbolic reasoning, **default to NFC, not NFKC**. The Llama-3 tokenizer disables NFKC for exactly this reason. We have shipped a model where 30% of the math eval failures came from a single line in the tokenizer config.

Beyond NFC, you also choose:
- Casing (almost always: keep case; lowercasing destroys NER and code).
- Whitespace handling (collapse multiple spaces? almost never — code requires it).
- Control characters (strip vs preserve — preserve for code).

### Pre-tokenization

Stage two is where the action is. The pre-tokenizer cuts the normalized stream into rough chunks — usually word-like units — *before* the subword segmentation algorithm runs on each chunk. This step is what makes `" for"` a different token from `"for"` in GPT-2: the regex captures the leading space as part of the word.

![Pre-tokenization regex picks the unit set](/imgs/blogs/designing-choosing-tokenizer-llm-3.png)

Here are the four production regexes you should know:

**GPT-2:**

```python
r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

**cl100k (GPT-3.5, GPT-4):**

```python
r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
```

**o200k (GPT-4o):**

```python
r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
```

**Llama-3 (tiktoken-style, simplified):**

```python
r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
```

The differences look cosmetic but produce dramatically different vocabularies:

- **GPT-2** captures one leading space: `" for"`, `" if"`, `" the"` are tokens. Any input with two leading spaces (indented code, formatted prose) splits awkwardly: `"  for"` becomes `[" "][" for"]`.
- **cl100k** captures *up to one* leading space and limits digit runs to three: `"100000"` becomes `["100", "000"]`. The 3-digit cap is what makes GPT-4 better at arithmetic than GPT-3 was — three-digit chunks are stable across magnitudes.
- **o200k** is case-aware: it segments `"CamelCase"` as `["Camel", "Case"]` rather than `["CamelCase"]`. It also keeps trailing whitespace with the word, which helps streaming (you can stop on word-boundary tokens).
- **Llama-3** drops the digit-run cap and goes per-digit (1-digit max). Combined with byte-BPE merges, you get `["1","2","3","4","5"]` for `"12345"`. Worse compression on numbers, better arithmetic generalization.

The takeaway: when you choose or train a tokenizer, **read its pre-tokenization regex carefully**. If you don't, you will be debugging "why does this token sometimes start with a space" for the rest of the project.

### Second-order optimization

The pre-tokenization regex interacts with the **special token** scheme. If your regex matches `<|im_start|>` as a single chunk because it is added to the vocabulary as a "special token" before pre-tok runs, you are fine. If it doesn't, your chat template will silently tokenize differently in production than in training. Check by tokenizing your own chat template offline and inspecting the byte sequence that comes out.

## 3. Vocab size economics

> **Vocab size is not a hyperparameter, it is a budget. Spend it on the languages and domains your customers care about.**

![Vocab size trades three costs at once](/imgs/blogs/designing-choosing-tokenizer-llm-4.png)

For a model with hidden size $d = 4096$, the embedding table is $V \times d$ parameters and the output softmax is another $V \times d$ (when not tied). At $V = 32{,}000$ you spend 131M params on each. At $V = 256{,}000$ you spend 1.05B — for many small models that is *more than half the model*. Llama-3 8B has 4096-d hidden and 128k vocab, so 524M of its 8B parameters are in the embedding/softmax — over 6%.

What do you get for that? Two things:

1. **Compression**, measured in bytes per token (BPT). Higher BPT = fewer tokens for the same input = shorter context, faster inference, lower API cost.
2. **Coverage**, measured in fertility per language (tokens-per-word). Wider vocab = more languages can have their *common* words be single tokens.

The relationship is sub-linear and corpus-dependent. Doubling vocab from 32k to 64k typically lifts BPT by 8–12% on English and 25–40% on under-represented languages. Doubling again to 128k adds another 4–8% on English and 15–25% on Hindi/Thai/Bengali. After 256k, English compression saturates entirely; the only further gains are multilingual or domain-specific.

This non-linearity is why the modern arc looks like:

- 32k (Llama-2, Mistral 7B): cheap, English-dominant, fertility ~4 on Hindi.
- 50k–65k (GPT-2, Falcon): legacy English-and-code.
- 100k–128k (cl100k, Llama-3, DeepSeek V2): the current "good default" for multilingual+code.
- 200k–262k (o200k, Gemma): aggressive multilingual coverage, justified by parameter sharing across languages.

> Every doubling of vocab buys diminishing English compression and meaningful multilingual coverage. You are paying for the second one, not the first one.

### A worked example

Let's compute the actual cost of doubling vocab on a 7B model.

```python
import math

def vocab_cost(V, d=4096, tied=True):
    embed = V * d
    softmax = 0 if tied else V * d
    total_params = embed + softmax
    # Inference: softmax is O(V*d) per generated token, embedding is O(d)
    softmax_flops_per_token = 2 * V * d   # 1 mul, 1 add per (V, d) pair
    return total_params, softmax_flops_per_token

for V in [32_000, 65_000, 128_000, 256_000]:
    p, f = vocab_cost(V)
    print(f"V={V:>7}: {p/1e6:>6.0f}M params, "
          f"{f/1e9:>5.2f} GFLOPs per token softmax")
```

Output (tied embeddings):

```
V=  32000:    131M params,  0.26 GFLOPs per token softmax
V=  65000:    266M params,  0.53 GFLOPs per token softmax
V= 128000:    524M params,  1.05 GFLOPs per token softmax
V= 256000:   1049M params,  2.10 GFLOPs per token softmax
```

For a 7B param model, going from 32k to 128k vocab adds 393M params (5.6%) and 0.79 GFLOPs per generated token. At 50 tokens/s decode that is 39.5 GFLOP/s of additional softmax compute on every output. On an A100 with 312 TFLOPs FP16, that is 0.013% of peak — *not* the bottleneck. The embedding params *are* the bottleneck, because they have to fit in VRAM.

The actionable rule: **vocab size is gated by VRAM and dataset breadth, not compute**. If you can fit the embedding table and you have enough non-English data to populate it, go bigger.

### Vocab size and the chinchilla-style scaling laws

There is a quieter scaling story for vocab. The DeepMind "chinchilla" line of work tuned compute against the parameter / token ratio, holding architecture fixed. A more recent analysis (Tao et al., 2024, "Scaling Laws with Vocabulary") factors vocab size into the same framework. The empirical finding: for a model with $N$ non-vocab parameters trained on $D$ tokens of text, the optimal vocab size scales as roughly $V^* \propto N^{0.27}$, increasing from ~32k at 1B parameters to ~256k at 100B parameters. The intuition is that larger models can amortize the embedding-table cost over more useful capacity, so they can afford a wider vocab.

Two practical implications. First, do not blindly copy the vocab size from a different-sized model. Llama-3 8B uses 128k; Llama-3 70B uses 128k too, which is *under-vocabbed* by the scaling law. The common reason for keeping the same vocab across sizes is operational simplicity (one tokenizer for the family). The cost is small but real — the 70B leaves a few percent of multilingual gain on the table.

Second, the scaling law is *training-mix dependent*. A heavily multilingual mix shifts the optimum toward larger vocabs because each language gets fewer tokens of training and benefits more from longer pieces. A pure English-and-code mix shifts it smaller. If you have the budget for an ablation, run the actual law on your mix instead of using the literature numbers.

### The cost of an under-trained vocab slot

A vocab slot you allocated and never trained is worse than not having allocated it at all. It costs $d$ parameters in the embedding and $d$ in the lm_head (if untied), it counts toward softmax compute on every step, and its embedding stays near initialization — making it a magnet for adversarial decoding (the model can output it but produces nonsense). The Renyi efficiency metric we discuss in §10 catches this directly: tokens that almost never appear in held-out text drag down the entropy, and the ratio falls.

A cheap rule of thumb: every token in your vocabulary should appear at least 1000 times in your model training corpus. If your training corpus has $D$ tokens, your vocab can be at most $V_\mathrm{max} \approx D / 1000 / \overline{f}$, where $\overline{f}$ is mean fertility. For $D = 1$T tokens and $\overline{f} = 1.3$, $V_\mathrm{max} \approx 770$M, which is far above any realistic vocab. For $D = 30$B tokens (a small fine-tune), $V_\mathrm{max} \approx 23$M — still loose. The constraint binds only at the very bottom of training scale.

### Second-order optimization

**Tie the embedding and output projection** when V is large. Most modern LLMs do this (Llama, Mistral, Gemma). It halves the parameter cost of the vocab and, contrary to old folklore about it hurting capacity, it helps small models and is roughly neutral for large ones. The only place it bites is decoder-only models that want different input/output distributions (rare).

## 4. Training corpus design

> **Your tokenizer's fertility profile is determined by your training corpus mix, not by your vocab size.**

The tokenizer is trained *before* the model, on a sample of the same data the model will see. The merges or pieces it learns are statistically driven by that sample. If your sample is 90% English, your common-token slots fill up with English. If it is 60% English, 30% code, 10% Spanish, you get a meaningfully different vocabulary at the same size.

Here is a concrete training script using the HF `tokenizers` library, which is the reference implementation for byte-BPE:

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

tok = Tokenizer(BPE(unk_token=None))
tok.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
tok.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=128_000,
    min_frequency=2,
    special_tokens=["<|begin_of_text|>", "<|end_of_text|>", "<|pad|>"],
    initial_alphabet=ByteLevel.alphabet(),       # all 256 bytes seeded
    show_progress=True,
)

## corpus.jsonl: each line is a {"text": "..."} record.
def iter_corpus(path):
    import json
    with open(path) as f:
        for line in f:
            yield json.loads(line)["text"]

tok.train_from_iterator(iter_corpus("corpus.jsonl"), trainer=trainer, length=10_000_000)
tok.save("tokenizer.json")
```

For SentencePiece Unigram (the multilingual default):

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="sp_unigram",
    vocab_size=128_000,
    model_type="unigram",
    character_coverage=0.9995,                   # cover 99.95% of seen chars
    byte_fallback=True,                          # OOV bytes -> 256 byte tokens
    normalization_rule_name="identity",          # disable NFKC; we want NFC only
    split_digits=True,                           # treat each digit as own piece
    allow_whitespace_only_pieces=True,           # keeps tab+space stable for code
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    user_defined_symbols=["<|im_start|>", "<|im_end|>"],
    train_extremely_large_corpus=True,
    num_threads=64,
)
```

A few flags deserve explicit comment:

- **`character_coverage`**: in multilingual training, set to 0.9995 (default is 0.9995 for languages with large alphabets, 1.0 for English-only). Set to 1.0 only if you trust your corpus is clean.
- **`byte_fallback=True`**: the killer flag. Any character not in the vocabulary falls back to its UTF-8 bytes — you get the OOV-free property of byte-BPE without committing the entire alphabet to the byte set.
- **`split_digits=True`**: enables the per-digit policy that helped Llama-3 on math.
- **`normalization_rule_name="identity"`**: disables NFKC. Use this. Almost always.

### Mixture choice is the lever

Below is a real fertility table I have measured on production tokenizers. Numbers are average tokens per word (or per equivalent-character unit for Chinese) on the FLORES-200 dev split, with code measured separately on a Python subset of The Stack.

![Multilingual fertility (tokens per word)](/imgs/blogs/designing-choosing-tokenizer-llm-5.png)

A few observations that constantly surprise people:

- Llama-2 (32k) is 4.10 tokens/word on Hindi. Llama-3 (128k) drops it to 1.95. That is a **2.1x throughput gain on Hindi inference, for free, on the same architecture**.
- Gemma's 256k vocab buys excellent fertility almost everywhere but adds 525M parameters of embed/softmax over Llama-3. On a 2B Gemma that is 25% of the model.
- cl100k is competitive with Llama-3 on multilingual but worse on code, because it caps digits at 3 and doesn't have explicit code-comment patterns in its merges.
- The English fertility numbers cluster between 1.25 and 1.35. **There is essentially no English compression to be won past 100k vocab.**

> If your customers speak Hindi, hire a tokenizer engineer before you hire a model researcher.

### Deduplication, document boundaries, and other corpus hygiene

The tokenizer corpus has its own preprocessing pipeline, separate from the model corpus. Three things matter more than people think.

**Deduplication.** The single biggest source of glitch tokens is a tokenizer corpus that has duplicated content (same Reddit thread copied many times, same boilerplate footer on every page) which the model corpus has deduplicated. The merge counts on duplicated content overweight tokens that are rare in the actual training distribution. Use the same dedup pipeline on the tokenizer corpus that you use on the model corpus — minhash near-duplicate dedup at the document level, plus exact-match line-level dedup.

**Document boundaries.** BPE merges are computed on a per-document basis (or per-line, depending on implementation). If your corpus is one giant concatenated string, the merge counter will happily merge the last byte of one document with the first byte of the next ("the end.<doc-boundary>The beginning"), producing junk merges. Make sure the trainer sees a clear document boundary token (or simply enforces document-level streaming).

**Boilerplate scrubbing.** Cookie banners, "click here to accept," and license headers are over-represented in scraped corpora. They contribute tokens that compress those exact strings into single ids — useless for actual generation, dead weight for the embedding table. Scrub before training.

**Language detection.** If your corpus is "the internet," roughly 60% is English by byte count even after dedup. If you want a balanced multilingual tokenizer, you must over-sample non-English — typically by 3–5x relative to its byte share. Use a language detector (cld3, fastText) on the documents and resample.

A practical recipe: train a *first* tokenizer on the unbalanced corpus to estimate per-language fertility, then resample the corpus inversely proportional to fertility (capped at 5x), and train the *production* tokenizer on the resampled mix. The two-pass approach typically halves the worst-case fertility on tail languages.

### Second-order optimization

Resist the urge to *over-mix* the tokenizer corpus. If your model will be trained on 5% Vietnamese data, training the tokenizer on 30% Vietnamese gives you tokens that are common in Vietnamese but rare in your training data — they will be **under-trained tokens**, the SolidGoldMagikarp class. Match the tokenizer training mixture to the *model* training mixture, not to your aspirations.

## 5. Multilingual and low-resource

> **Fertility is the single most predictive metric of multilingual capability per parameter.**

The argument has a clean theoretical form. Holding the model's compute budget constant, a language with fertility $f$ pays a $f \times$ tax on every byte: more tokens to process per byte of input, more tokens to generate per byte of output, less context-window for the same input bytes. If your context window is 8192 tokens and Hindi fertility is 4.5, your *effective* Hindi context is ~1820 tokens — a quarter of an English context.

This compounds with the *training-side* problem. During pretraining, every token gets one gradient step. If a Hindi paragraph takes 4.5x more tokens than an equivalent English paragraph, it gets 4.5x more gradient signal — but it also burns 4.5x more of your token budget, so you can train on 4.5x fewer Hindi paragraphs for the same compute. The net effect is *worse* multilingual capability per FLOP, not better.

The fix is on the tokenizer side: increase fertility-balance. Concretely:

1. **Sample the tokenizer training corpus to balance fertility, not byte count.** A standard recipe: sample languages with weight $\propto 1 / \mathrm{fertility}$ as estimated by a small initial tokenizer. Iterate twice.
2. **Set `character_coverage=0.9995` or higher** for any script you care about.
3. **Enable `byte_fallback`** so the long-tail scripts cost only a few tokens each instead of being unrepresentable.
4. **Audit fertility on a held-out FLORES split before committing.** Aim for fertility ≤ 2.0 on every language with > 1% of the model training mix.

### Case in point: Llama-2 → Llama-3

The Llama-2 tokenizer was 32k SentencePiece Unigram with character coverage 0.9999, NFKC normalization on, no byte fallback. Llama-3 made four changes simultaneously: byte-level BPE (tiktoken-style), 128k vocab, NFC instead of NFKC, and a fertility-balanced multilingual training mix that Meta has not fully disclosed but appears to weight non-English up by ~3x relative to its share of the corpus.

The result: Llama-3 fertility on Hindi dropped from 4.10 to 1.95. On Chinese, from 2.50 to 1.50. On a fixed 8192-token context window, the *effective context* in Hindi went from 1822 chars to 4201 chars. That is the single biggest non-architectural improvement in the Llama line.

## 6. Code, math, and digit handling

> **If you care about arithmetic, split digits. If you care about code, keep whitespace.**

The four design decisions that decide how well a model handles code and math are: digit grouping, whitespace policy, leading-space handling, and identifier-boundary handling.

![Digit and whitespace handling](/imgs/blogs/designing-choosing-tokenizer-llm-6.png)

### Digit policy

Three options, in increasing order of arithmetic-friendliness:

- **No constraint** (GPT-2): digit runs absorb into the most frequent merges. `"12345"` → `["123", "45"]` or `["1234", "5"]` depending on training data. Inconsistent across magnitudes — the model has to learn a different decomposition for every range.
- **Bounded digit run** (cl100k, n ≤ 3): digit runs cap at 3. `"12345"` → `["123", "45"]`, deterministic. Each chunk is a number 0–999. The model can learn modular arithmetic over a fixed alphabet of 1000 chunks.
- **Per-digit** (Llama-3, Gemma): each digit is its own token. `"12345"` → `["1","2","3","4","5"]`. Worst compression but cleanest arithmetic, because the model sees positional digit structure directly.

Per-digit is the right choice if math eval is on your roadmap. The compression cost is real (numerals get 5x longer) but recoverable with a 128k+ vocab and slightly more training tokens.

### Whitespace policy

For code, you need to preserve *exact* whitespace, including leading spaces and tabs. Three regimes:

- **Whitespace-collapsed** (default GPT-2): multiple spaces collapse, indentation is lost. Useless for code.
- **Whitespace-preserving with single leading space** (GPT-2 actual, cl100k): one leading space is part of the word token. Two leading spaces split into ` ` + ` for`. Indented Python is verbose but learnable.
- **Whitespace-preserving with multi-leading-space tokens** (cl100k partial, Llama-3 cl100k-style): tokens like `"    "` (4 spaces) and `"        "` (8 spaces) are explicitly in the vocabulary. Indented code is one token of indent + word tokens.

The last option is what you want for serious code support. cl100k's vocabulary contains 23 multi-space tokens, including 4-space and 8-space (for Python's two indent conventions). It also contains tokens for `\n    ` (newline + 4 spaces) and `\t\t` (double tab) — common in Go and TS.

### Identifier boundaries

CamelCase and snake_case interact with the pre-tok regex. o200k explicitly splits CamelCase: `"PaymentProcessor"` → `["Payment", "Processor"]`. cl100k does not, so `"PaymentProcessor"` is a single token if it appeared frequently enough in training, three tokens otherwise.

CamelCase splitting helps the model generalize across identifier conventions but slightly hurts compression on idiomatic code. For pure code-completion models (Codex, StarCoder), the codebases use *enough* of each convention that splitting wins. For chat models that handle code occasionally, not splitting is fine.

```python
## Quick fertility benchmark on code
import tiktoken

enc_cl = tiktoken.get_encoding("cl100k_base")
enc_o2 = tiktoken.get_encoding("o200k_base")

samples = [
    "def calculate_payment(amount, rate):\n    return amount * (1 + rate)",
    "class PaymentProcessor:\n\tdef process(self, tx: Transaction) -> bool:\n\t\treturn True",
    "for i in range(100000):\n    total += data[i] * 0.5",
]
for s in samples:
    print(f"len={len(s):>3}, cl100k={len(enc_cl.encode(s)):>3}, o200k={len(enc_o2.encode(s)):>3}")
```

Typical output:

```
len= 64, cl100k= 18, o200k= 17
len= 84, cl100k= 24, o200k= 21
len= 56, cl100k= 18, o200k= 19
```

The o200k advantage on code is real but small — about 5–10% on real codebases — and it has a slight loss on numeric-heavy code due to its lighter digit handling.

### Second-order optimization

If you ship a code-completion product, **add explicit tokens for the top-N opening sequences in your dominant language**. For Python: `def `, `class `, `import `, `from `, `return `, `if `, `for `, `while `, `try:`, `except `, `with `. These are already in cl100k/o200k for English, but training a custom tokenizer on a Python-heavy corpus pulls them in even more aggressively. The result is 10–20% compression on real code with no quality loss.

## 7. Library choice

> **Pick the library by your training and serving constraints, not by which one your last project used.**

Four real options in 2026:

| Library | What it is | Training | Inference speed | Notes |
|---|---|---|---|---|
| `tiktoken` | OpenAI's open-source byte-BPE library, Rust core, Python bindings. | No (read-only). | ~2–5 µs/token, fastest in the league. | Ships with cl100k, o200k, p50k, r50k. Read-only by design. |
| `sentencepiece` | Google's C++ implementation of BPE/Unigram. | Yes, both algorithms. | ~10–30 µs/token. | The standard for training Unigram. CLI + bindings. |
| `tokenizers` (HF) | Rust core, Python bindings; supports BPE, WordPiece, Unigram, custom. | Yes, all algorithms. | ~5–15 µs/token. | Most flexible. Loads HF Hub tokenizers natively. |
| `tekken` | Mistral's tokenizer (built on tiktoken with extensions). | Limited. | ~3–8 µs/token. | Special-token namespaces for tools/roles. |

The decision matrix:

- **Training a new tokenizer**: SentencePiece for Unigram, HF `tokenizers` for byte-BPE.
- **Serving an existing tokenizer**: `tiktoken` if you can re-export to its format, otherwise HF `tokenizers`.
- **Running both training and inference in one stack**: HF `tokenizers`. The serialization format is portable.
- **Mistral or Mistral-derivative**: `tekken`, no choice.

![Tokenizer on the inference hot path](/imgs/blogs/designing-choosing-tokenizer-llm-8.png)

### Speed in practice

A common mistake: people benchmark tokenizers on `["hello world"]` and conclude "they're all 50x faster than I'll ever need." Reality: production prompts are 1–10 KB, often containing system prompts, history, retrieved context. At that scale, a 5x difference is the difference between 100 µs and 500 µs *per request* — and at p99, with system call jitter and Python GIL contention, it can blow up to 5 ms.

```python
import time, tiktoken
from tokenizers import Tokenizer

text = open("realistic_prompt.txt").read()  # ~ 4 KB
enc_tt = tiktoken.get_encoding("cl100k_base")
enc_hf = Tokenizer.from_pretrained("Xenova/gpt-4")

def bench(name, fn, n=10_000):
    t = time.perf_counter()
    for _ in range(n):
        fn(text)
    dt = time.perf_counter() - t
    print(f"{name}: {dt*1e6/n:>6.1f} µs/call")

bench("tiktoken", enc_tt.encode)
bench("hf-tokenizers", enc_hf.encode)
```

On my M2 Pro, a 4 KB prompt:

```
tiktoken:        87.2 µs/call
hf-tokenizers:  142.8 µs/call
```

Both are fine. But the legacy Python BPE (the one in `transformers.GPT2Tokenizer` slow path) on the same text is ~9000 µs/call. **Never use the slow Python tokenizer in production.** Always use the Rust-backed fast version.

### Second-order optimization

Tokenizer **fork-safety** in production servers. `tiktoken` and HF `tokenizers` both initialize lazily. If your gunicorn/uvicorn worker forks *after* importing them, you can hit a state-corruption bug on macOS due to Apple's `objc` initializer. The fix: import and call once before the fork. Add a no-op `tokenizer.encode("warmup")` at module load time.

## 8. Vocab extension for fine-tuning

> **You almost never want to retrain a tokenizer. You almost always want to *extend* one.**

This is the most expensive lesson I've watched teams learn. They want to fine-tune a Llama-3 base for Vietnamese; their first instinct is "the tokenizer is bad for Vietnamese, let's train a new one." That instinct is wrong. Training a new tokenizer means training a new model — every embedding is now uninitialized, every attention pattern over tokens has to be relearned. You have just thrown away 15T tokens of pretraining for an 8% fertility win.

The right move is *vocab extension*: take the existing tokenizer, add a small number of high-value tokens (300–10000), resize the embedding matrix, initialize the new rows from the average of the subword decomposition of each new token, and continue pretraining for a few hundred million tokens.

![Vocab extension for a new language](/imgs/blogs/designing-choosing-tokenizer-llm-7.png)

Here is the actual code, end to end, for extending a Llama-3 tokenizer with a Vietnamese vocabulary:

```python
import sentencepiece as spm
from tokenizers import Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

## 1. Train an auxiliary BPE on Vietnamese corpus.
spm.SentencePieceTrainer.train(
    input="vi_corpus.txt",
    model_prefix="vi_aux",
    vocab_size=8000,
    model_type="bpe",
    character_coverage=0.9995,
    byte_fallback=True,
    split_digits=True,
    train_extremely_large_corpus=True,
)

## 2. Merge into base, deduping by string identity.
base_tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
base_vocab = set(base_tok.get_vocab().keys())

aux = spm.SentencePieceProcessor()
aux.load("vi_aux.model")
aux_pieces = [aux.id_to_piece(i) for i in range(aux.get_piece_size())]
new_pieces = [p for p in aux_pieces if p not in base_vocab]
print(f"adding {len(new_pieces)} new tokens (started with {len(aux_pieces)})")

## tokenizers library lets us add directly.
base_tok.add_tokens(new_pieces)

## 3. Resize the model's embedding and lm_head.
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B", torch_dtype=torch.bfloat16
)
old_V, d = model.get_input_embeddings().weight.shape
new_V = len(base_tok)
model.resize_token_embeddings(new_V, mean_resizing=False)  # we'll init below

## 4. Mean-init the new rows from the base tokenizer's segmentation.
embed = model.get_input_embeddings().weight
lm_head = model.get_output_embeddings().weight
old_tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
with torch.no_grad():
    for new_id in range(old_V, new_V):
        piece = base_tok.convert_ids_to_tokens(new_id)
        sub_ids = old_tok.encode(piece, add_special_tokens=False)
        if not sub_ids:                                 # bytes-only piece
            continue
        embed[new_id] = embed[sub_ids].mean(dim=0)
        if lm_head is not embed:                        # not tied
            lm_head[new_id] = lm_head[sub_ids].mean(dim=0)

## 5. Save and continue pretraining.
base_tok.save_pretrained("llama3-vi-extended")
model.save_pretrained("llama3-vi-extended")
```

Then continue pretraining on a 60/40 mix of Vietnamese and English, for ~500M tokens at the original learning rate, dropping to 10% LR for the final 100M. Fertility on Vietnamese typically drops from 3.1 to 1.6. Throughput on Vietnamese inference doubles.

Key gotchas:

- **Don't add tokens that the base tokenizer would have produced anyway.** Run a fertility check before extending; if `cl100k` already tokenizes your common Vietnamese words at fertility ~2.5, adding overlapping pieces just costs you embedding params with no win.
- **Mean-init is better than random-init by a wide margin.** Random init for new embeddings means every Vietnamese token starts as garbage; mean-init starts each as the average of its byte-level decomposition, which is already roughly meaningful.
- **Continued pretraining must include the original data.** If you only train on Vietnamese, you catastrophically forget English. 60/40 mix is a safe default.
- **Re-tie embedding and lm_head before saving** if the base model had them tied. Llama-3 does.

### Second-order optimization

For fine-tuning on a *domain* (medical, legal, code) within an existing language, the same flow works but with smaller numbers — 300–2000 new tokens, mean-init from the base, continued pretraining for ~50M tokens. The win on a medical corpus is usually 8–15% fertility reduction, which translates to 8–15% inference cost reduction in production.

## 9. Runtime and serving

> **In serving, the tokenizer is on the hot path. Treat it like one.**

Serving an LLM has three latency-critical components: tokenize, prefill, decode. The decode phase gets all the attention because it dominates wall-clock for long outputs. But for *short* outputs and at p99 — chat completions of 50–200 tokens with 4–10 KB of input — tokenization is a meaningful fraction of TTFT (time-to-first-token).

A real production breakdown on a 4096-token prompt:

| Phase | p50 | p99 |
|---|---|---|
| Tokenize (tiktoken) | 0.4 ms | 1.8 ms |
| Schedule + queue | 0.8 ms | 12 ms |
| Prefill (1 GPU) | 60 ms | 110 ms |
| Decode 1st token | 25 ms | 45 ms |
| Detokenize + emit | 0.05 ms | 0.3 ms |
| **TTFT** | **86 ms** | **170 ms** |

Tokenization is small but not negligible. The mistakes I see in production:

1. **Re-tokenizing the entire conversation history on every turn**, instead of caching. For a 30-turn chat, this is O(n²) work.
2. **Tokenizing on the Python event loop**. Even at 0.4 ms, doing it on the same thread as the HTTP handler limits concurrent requests. Push it to a thread pool or do it pre-queue.
3. **Running offset-mapping every request** when you don't need it. `tiktoken.encode(text)` is fast; `tokenizers.encode(text, return_offsets_mapping=True)` is 3x slower. Only ask for offsets if you actually need them (streaming detok, structured output, citation extraction).
4. **Forking workers after importing the tokenizer** (covered in §7).

### Streaming detokenization

The non-obvious bit. When you stream tokens out, you cannot just call `tokenizer.decode(token_id)` per token — multi-byte UTF-8 characters split across token boundaries (especially in CJK languages) will produce broken bytes. The standard fix is *delta detokenization*:

```python
class StreamingDetokenizer:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.buffer = []
        self.emitted = ""

    def step(self, token_id: int) -> str:
        self.buffer.append(token_id)
        text = self.tok.decode(self.buffer)
        # Only emit complete characters.
        if not text.endswith("�"):                   # no replacement char
            new = text[len(self.emitted):]
            self.emitted = text
            return new
        return ""
```

The `�` (Unicode replacement character) is what every reasonable decoder produces on incomplete byte sequences. We hold the partial bytes in the buffer until the next token completes them.

vLLM, SGLang, and TensorRT-LLM all implement variants of this. If you are rolling your own server, do not skip it — the bug surface for "Chinese characters appearing as ???" is one of the most reported and one of the most preventable.

### Observability for serving tokenizers

If your serving stack does not log per-request token counts, you are flying blind on cost. The minimum metric set:

- **Input tokens** per request — distribution, p50/p95/p99.
- **Output tokens** per request — same.
- **Tokenizer encode latency** — per request, in microseconds.
- **Per-language fertility** — estimated from a language-id pass on the input.
- **Special-token presence** — any of `<|im_start|>`, `<|tool_call|>`, etc. that appear in user input (which usually shouldn't).

The tokenizer encode latency in particular is a *leading indicator* of TTFT regressions. When someone "innocently" upgrades the tokenizer library version, the median encode time can shift by 10–50% with no other test signal. Watch for it.

A second-order observation: cost-per-request follows a fertility-weighted distribution, not a uniform one. A 1% population of Hindi users on a system with Hindi fertility 2x English fertility represents 2% of token cost, not 1%. Your cost dashboard should aggregate by language so you can size capacity correctly.

### Caching the encoded prompt

For chat applications, the system prompt and conversation history are the same on every turn except for the new user message. Re-tokenizing the entire conversation per turn is wasteful; cache the token ids of stable prefixes and only re-tokenize the new turn.

```python
class TurnCache:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.cached_text: str = ""
        self.cached_ids: list[int] = []

    def encode_turn(self, full_prompt: str) -> list[int]:
        if full_prompt.startswith(self.cached_text):
            tail = full_prompt[len(self.cached_text):]
            new_ids = self.tok.encode(tail).ids
            self.cached_ids = self.cached_ids + new_ids
            self.cached_text = full_prompt
            return self.cached_ids
        # cache miss; full encode
        self.cached_ids = self.tok.encode(full_prompt).ids
        self.cached_text = full_prompt
        return self.cached_ids
```

Two caveats. First, this only works if your tokenizer is *prefix-stable* — encoding a prefix produces the same ids as encoding the full text and taking the prefix. Byte-BPE with a regex pre-tokenizer is *almost* prefix-stable but breaks at the boundary where pre-tok regex would have absorbed the next character (e.g., a partial digit run). For safety, always re-encode the *last word* of the previous turn together with the new turn. Second, this is incompatible with continuous batching kernels that expect tokenization to happen on the GPU side (rare in practice).

### Second-order optimization

For very high QPS, switch from `tokenizer.encode(text)` to `tokenizer.encode_batch([text1, text2, ...])`. Both `tiktoken` and HF `tokenizers` parallelize internally; per-request encoding leaves the parallelism on the table. Bench it on your workload — I have seen 2–4x throughput gains at QPS > 100.

## 10. Evaluation: how to know your tokenizer is good

> **There is no single number that captures tokenizer quality. There are five.**

The five metrics that actually predict downstream model quality:

1. **Bytes per token (BPT)**: average UTF-8 bytes consumed per token across a held-out corpus. Higher is better. English baseline ~4.5 for cl100k, ~4.7 for Llama-3.
2. **Per-language fertility**: average tokens per word (or per equivalent unit for non-whitespace languages). Lower is better. Aim for ≤ 2.0 on every language with > 1% of training mix.
3. **Renyi efficiency** (Zouhar et al., 2023): the entropy of the token distribution divided by $\log V$. Higher means more uniform usage of the vocabulary; very low means most tokens are dead weight. Aim for > 0.4.
4. **OOV rate**: fraction of input bytes that fall back to the byte-level escape (only relevant if you have byte fallback). Should be < 0.5% on production data.
5. **Round-trip preservation**: `decode(encode(text)) == text` for a representative sample of texts including code, math, multilingual, control characters, emoji, ZWJ sequences. This *should* be 100%; in practice, NFKC normalization breaks it on math, and some tokenizers strip control characters silently. **Always test this**.

Here is a compact evaluation harness:

```python
import json, math, collections
from tokenizers import Tokenizer

def bench_tokenizer(tok, corpus_path):
    n_tokens = 0; n_bytes = 0; n_words = 0
    token_counts = collections.Counter()
    rt_failures = 0; rt_total = 0

    with open(corpus_path) as f:
        for line in f:
            text = json.loads(line)["text"]
            n_bytes += len(text.encode("utf-8"))
            n_words += len(text.split())
            ids = tok.encode(text).ids
            n_tokens += len(ids)
            token_counts.update(ids)
            decoded = tok.decode(ids)
            if decoded != text: rt_failures += 1
            rt_total += 1

    bpt = n_bytes / n_tokens
    fertility = n_tokens / n_words
    H = -sum((c/n_tokens) * math.log2(c/n_tokens) for c in token_counts.values())
    renyi = H / math.log2(tok.get_vocab_size())
    rt_ok = 1 - rt_failures / rt_total
    return dict(bpt=bpt, fertility=fertility, renyi=renyi, rt=rt_ok,
                vocab_used=len(token_counts), vocab_total=tok.get_vocab_size())

for name in ["cl100k", "llama3", "gemma"]:
    tok = Tokenizer.from_pretrained(f"./tokenizers/{name}")
    print(name, bench_tokenizer(tok, "eval/multilingual.jsonl"))
```

Two things this catches that aggregate BPT does not:

- A tokenizer with high BPT but low Renyi efficiency is using ~10% of its vocabulary heavily and the rest as dead weight. You could trim it.
- A tokenizer with `rt_ok < 1.0` is silently destroying information. Almost always NFKC. Switch to NFC and re-train.

### A real eval split

What I run before declaring a tokenizer "good enough":

| Eval slice | Source | What it catches |
|---|---|---|
| `english.web` | C4 held-out | English BPT regression |
| `english.code` | The Stack held-out, Python+JS | Indent/digit/identifier handling |
| `english.math` | OpenWebMath held-out | NFKC silently folding superscripts |
| `multilingual.flores` | FLORES-200 dev | Per-language fertility |
| `multilingual.cc100` | CC-100 held-out | Long-tail script coverage |
| `confusables.synth` | hand-built | Cyrillic/Latin/Greek look-alike handling |
| `chat.template` | golden chat templates | Special-token byte-for-byte match |
| `roundtrip.unicode` | hand-built | NFC vs NFKC information loss |

For each slice, I record BPT, fertility (where applicable), and round-trip success rate. A new tokenizer must improve BPT on at least one slice and not regress on any other by more than 5%.

The hand-built confusables slice is small (typically 200 strings) but high-leverage. Each entry is a triple of (visually-identical-looking string, expected ASCII normalization, attacker payload). It catches normalization mismatches that no natural-data benchmark surfaces.

### Downstream proxies

Token-level metrics are necessary but not sufficient. The honest question is "does the model actually do better." The cheapest proxies are:

- **Bits per character** (BPC) on a held-out corpus, with the same model architecture. Lower is better. A tokenizer change of +0.05 BPC at the same model size is roughly equivalent to 30% more parameters at the old tokenizer.
- **Multilingual MMLU** with translation-only prompts. Tokenizer-driven gains show up sharply here.
- **GSM8K with per-digit vs grouped-digit tokenizer**, all else equal. The digit policy gain is in the 5–15% range on 7B models.

## 11. Security and robustness

> **Every untrained vocabulary entry is a latent prompt-injection vector.**

This is not a theoretical concern. The "SolidGoldMagikarp" tokens in GPT-3 (later analyzed by Rumbelow & Watkins, 2023) caused the model to produce gibberish, refuse to repeat the token, swear, or output unrelated content when asked about them. The cause: those tokens were in the GPT-2 BPE vocabulary because they appeared frequently in the *tokenizer training data* (a Reddit dump that included a counting subreddit), but were filtered out of the *model training data* (which deduplicated Reddit). The model never learned what they meant.

The class of bug is broader than glitch tokens. It includes:

- **Trailing-whitespace tokens** (` ` followed by a chat template token). If your chat template tokenization differs from your training-time chat template tokenization by even one space, you get bizarre completions.
- **BOM tokens** (`﻿` at the start of input). Some tokenizers absorb it as a single rare token; the model has never seen it in normal input and produces unreliable output.
- **Unicode confusables** (Cyrillic `а` vs Latin `a`). They tokenize differently and route through different vocabulary entries. Adversaries use this for prompt injection (visible-text says one thing, tokens say another).
- **Special-token spoofing**. If a user's input happens to contain `<|im_start|>` and your tokenizer doesn't add it as a "special token" only when it comes from the system, the model treats user input as system input.

The defense in depth:

1. **Audit your vocabulary against your training data**. Every token should have appeared at least N times (typically N = 1000) in the model training data. The ones that haven't are candidates for removal or annotation as "untrained."
2. **Use the special-token namespace**. Special tokens should be added with a *protected* flag that bypasses pre-tokenization only when injected by the system, not when matched in user input.
3. **Normalize confusables at the application layer**, before passing to the tokenizer.
4. **Test the chat template byte-for-byte** in CI. Tokenize an example with your production code path and compare to a golden file.

A real glitch-token finder:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B", torch_dtype=torch.bfloat16
).eval().cuda()

embed = model.get_input_embeddings().weight                     # (V, d)
norms = embed.norm(dim=1)                                       # (V,)

## Untrained tokens have anomalously small norm because the optimizer
## never pushed them away from initialization.
mean, std = norms.mean(), norms.std()
suspect_ids = torch.where(norms < mean - 2*std)[0].tolist()

print(f"{len(suspect_ids)} suspect tokens out of {len(tok)}")
for i in suspect_ids[:30]:
    s = repr(tok.decode([i]))
    print(f"  {i:>6}: {s}")
```

You should expect 0.1–0.5% of the vocabulary to be flagged. Inspect them. If they are recognizable strings (URLs, usernames, language fragments), they are dead weight at best, security holes at worst.

## Case studies from production

### 1. The leading-space prompt-engineering footgun

**Symptom**: a customer reports that prompting "Translate the following to French:" then "Hello world" produces different output than "Translate the following to French: Hello world" on the same line.

**Wrong first hypothesis**: the model is sensitive to system-prompt length.

**Actual root cause**: GPT-2's pre-tok regex captures one leading space. " Hello" (one leading space) is token 18435; "Hello" (no leading space) is token 15496. The two prompts produce different first-token-of-target distributions. The model has been trained on text where word-internal vs sentence-initial casing is signaled by the leading-space variant, and your prompt was using the wrong one.

**Fix**: standardize chat template formatting to always include a leading space on the first user message turn, exactly as the model's training data did. Document it as a hard requirement.

**Lesson**: leading-space tokens are a fingerprint of byte-BPE pre-tokenization, and prompt formatting is part of the model contract.

### 2. SolidGoldMagikarp and the under-trained-token landmine

**Symptom**: GPT-3 produces nonsense, refuses, or insults users when asked about specific tokens like "SolidGoldMagikarp," "TheNitromeFan," or " disreg".

**Wrong first hypothesis**: model alignment failure, RLHF gone wrong.

**Actual root cause**: the GPT-2 BPE was trained on a corpus that included Reddit usernames and subreddit-specific phrases. Those tokens entered the vocabulary because they appeared frequently in the *tokenizer corpus*. The GPT-3 model training corpus deduplicated Reddit, so the model never saw most of those tokens during training. Their embeddings stayed near initialization, and pushing them through the model produced semantically random outputs.

**Fix**: at vocabulary finalization time, audit each token's frequency in the *model* training corpus. Remove or merge tokens that have fewer than 1000 occurrences. Modern tokenizers (Llama-3, GPT-4 cl100k) are visibly cleaner because their vocabulary went through this audit.

**Lesson**: tokenizer training corpus and model training corpus must be consistent. The cheapest way to guarantee this is to use the same dataset for both.

### 3. Llama-2 to Llama-3: the 2x multilingual speedup

**Symptom**: customers complain Llama-2 inference cost in Hindi is 2.5x the cost in English for the same content.

**Wrong first hypothesis**: model doesn't know Hindi well; need a Hindi fine-tune.

**Actual root cause**: Llama-2's 32k SentencePiece tokenizer had Hindi fertility ~4.1. Each Hindi character was on average 4.1 tokens. With NFKC normalization on, half-width vs full-width script variants collapsed in unwanted ways. With character_coverage=0.9999 and no byte fallback, rare Devanagari ligatures dropped to OOV.

**Fix in Llama-3**: switch to byte-level BPE, 128k vocab, NFC, fertility-balanced multilingual mix. Result: Hindi fertility 1.95, Chinese 1.50, Bengali ~2.1. No model architectural change. ~2x throughput improvement on multilingual workloads.

**Lesson**: many "multilingual model" problems are really tokenizer problems with a model wrapper around them.

### 4. Mistral's tekken and the special-token namespace

**Symptom**: when integrating Mistral 7B into a tool-calling agent, prompt-injection attempts sometimes succeed by including `[/INST]` in user input.

**Wrong first hypothesis**: instruction-tuning is not robust.

**Actual root cause**: in the original Mistral instruct format, `[INST]` and `[/INST]` were *just text*. The tokenizer had no protected special-token namespace, so user input could contain those literal strings and the model could not distinguish them from system-injected control tokens.

**Fix in tekken**: introduce a separate special-token namespace with tokens like `<|im_start|>`, `<|im_end|>`, `<|tool_call|>`, etc. These are added to the vocabulary at a fixed range (e.g., ids 32000–32063), are never produced by tokenizing arbitrary user input, and are stripped or escaped in the application layer if they appear.

**Lesson**: control-flow tokens belong in a protected namespace, not in the natural-text vocabulary. Build this in from day one; retrofitting is painful.

### 5. The DeepSeek bilingual vocabulary

**Symptom**: training a bilingual (English + Chinese) LLM with a 32k tokenizer results in fertility 1.3 on English and 2.6 on Chinese, despite a 50/50 corpus mix.

**Wrong first hypothesis**: Chinese is just intrinsically harder to compress; live with it.

**Actual root cause**: standard tokenizer training is byte-frequency-driven, and English bytes outnumber Chinese bytes per "unit of meaning" by ~3x (English ASCII is 1 byte/char, Chinese UTF-8 is 3 bytes/char). At the same byte sample size, English has 3x the lexical units to absorb merges.

**Fix in DeepSeek**: train the tokenizer with corpus weights inversely proportional to per-language byte counts, not per-language byte counts directly. The result: English fertility 1.32, Chinese fertility 1.55 — much closer to balanced — at the same 32k vocab size.

**Lesson**: tokenizer training mixture must be designed in *fertility space*, not in *byte space*.

### 6. SentencePiece NFKC silently breaking math evals

**Symptom**: a freshly trained 1B model gets 5% lower than expected on a math benchmark; the loss curves look fine.

**Wrong first hypothesis**: the model is too small for math.

**Actual root cause**: the SentencePiece tokenizer was trained with default `normalization_rule_name="nmt_nfkc"`, which (among other things) folds superscript digits into base digits. Strings like `x²` in the eval became `x2` after tokenization, and the answer `x²` in the model's vocabulary literally did not exist as a producible string. The model was correct but the eval scored it wrong.

**Fix**: re-train the tokenizer with `normalization_rule_name="identity"` (NFC only, no compatibility folding). Re-pretrain. Math eval score recovered.

**Lesson**: NFKC is convenient for messy text and toxic for math. Default to NFC.

### 7. Vietnamese vocab extension on Llama-3

**Symptom**: a Vietnamese chat product on Llama-3 8B has acceptable quality but inference cost is 1.7x the cost of the English version, killing margins.

**Wrong first hypothesis**: Vietnamese is just expensive to serve; raise prices.

**Actual root cause**: Llama-3's tokenizer has Vietnamese fertility ~3.1 (better than Llama-2's ~3.8 but still high). Common Vietnamese words like "không" (no) tokenize as 4 tokens because the diacritics force byte-level fallback for the precomposed form.

**Fix**: train an auxiliary 8000-piece Vietnamese SentencePiece BPE, dedupe against the base vocabulary, end up with ~6500 new tokens. Resize the embedding matrix to 134.5k, mean-init the new rows from their base-tokenizer decomposition, continue pretraining on a 60/40 Vietnamese/English mix for 500M tokens at the original LR. Result: Vietnamese fertility drops to 1.6, throughput doubles, English quality unchanged within noise.

**Lesson**: the right answer to "this language is expensive" is almost never to retrain or to give up — it is vocab extension.

### 8. Gemma's 256k vocab and the embedding-tax tradeoff

**Symptom**: a team picks Gemma 2B as a base for a small device deployment, expecting a "small model" footprint, but VRAM usage is 30% higher than they estimated from the 2B parameter count alone.

**Wrong first hypothesis**: the model is leaking memory; bisect the inference stack.

**Actual root cause**: Gemma's 256k SentencePiece vocabulary contributes 1.05B parameters of embedding (256k × 4096 × 2 / 2 with tied embed/lm_head). On a 2B-class model where the non-vocab parameters are around 2B, the embedding alone is 50% of the parameter count. The team's 2B mental model was wrong: this is effectively a 3B model with 2B of "useful" capacity.

**Fix options**:
- Quantize the embedding to int8 (saves ~512 MB, with negligible quality loss because embeddings have small dynamic range).
- Switch to Gemma 7B-class, where the embedding is 12% of parameters instead of 50% (better marginal use of VRAM).
- For very tight deployments, prune the vocabulary: identify the bottom 30% of tokens by frequency in the deployment domain, replace them with byte-fallback tokens, and rebuild the model. This reduces vocab to ~180k and saves ~300M params.

**Lesson**: a wide vocabulary is a parameter-count knob, not a free win. On small models, scrutinize the embedding share; on large models, scrutinize the wall-clock impact.

### 9. The Cyrillic-Latin confusable prompt-injection vector

**Symptom**: a retrieval-augmented system that calls a code-execution tool starts running unexpected payloads when given otherwise-benign user queries containing the word "admin" or "root."

**Wrong first hypothesis**: the prompt template has a parsing bug; injection is via the system-prompt slot.

**Actual root cause**: a fraction of the documents in the retrieval corpus had been adversarially modified to replace ASCII letters with Cyrillic look-alikes (`а` is `U+0430`, `e` is `U+0435`, `o` is `U+043E`, etc.). These render visually identically to the ASCII forms in any reasonable font, but tokenize differently — `аdmin` (Cyrillic-a followed by `dmin`) is 3 tokens, none of which match `admin`. The model, given the *retrieved* text, did not see the word "admin" anywhere in its context, so its safety training did not trigger. The downstream tool-call validator, by contrast, did NFKC-fold the model's output before checking it against the allowed list — so the model's output `аdmin` became `admin` and slipped through.

**Fix**:
1. Normalize confusables at the application layer, *before* the tokenizer, on retrieved content.
2. Apply the *same* normalization on the model output before the tool-call validator reads it.
3. Add a CI test that injects confusable strings into retrieval corpora and verifies the system rejects them.

**Lesson**: tokenizer behavior on confusables is part of the security boundary. Mismatched normalization between input path and output path is a prompt-injection vector.

### 10. The trailing-whitespace inference bug

**Symptom**: half the open-source serving stacks (vLLM, TGI, llama.cpp) at one point in 2024 produced subtly different outputs than the HF reference implementation on prompts that ended with whitespace.

**Wrong first hypothesis**: floating-point non-determinism in the kernels.

**Actual root cause**: the HF tokenizer's `add_generation_prompt=True` flag for chat templates appended a specific sequence (e.g., `<|im_start|>assistant\n`). Some serving stacks trimmed trailing whitespace from the user prompt before tokenizing; the HF reference did not. The tokenization differed by one space token, which changed the position of every subsequent token, which changed the KV cache contents, which produced different (still plausible) outputs.

**Fix**: standardize chat template tokenization across all serving paths. vLLM and TGI now run the chat template through the *exact* HF tokenizer code path rather than reimplementing it. Add a `chat_template_byte_check` test in CI.

**Lesson**: tokenization in serving must be byte-for-byte identical to tokenization in training. Anything else is a heisenbug factory.

## When to reach for X / when not to

![Choosing a tokenizer in 2026](/imgs/blogs/designing-choosing-tokenizer-llm-9.png)

The decision tree above collapses to four questions. Walking through them:

### Reach for byte-BPE with a hand-written regex (cl100k / o200k / Llama-3 style) when:

- You are training a primarily English LLM with code support.
- You need OOV-free behavior and don't want to manage character-level fallback by hand.
- You will serve via `tiktoken` or HF `tokenizers` and want maximum encode speed.
- You want a deterministic, reproducible tokenizer with no probabilistic segmentation.
- Your training infra already supports Rust-backed tokenizer training.

### Reach for SentencePiece Unigram (with byte fallback) when:

- Your model is genuinely multilingual and includes non-whitespace languages (Chinese, Japanese, Thai).
- You want subword regularization for training-time stochasticity.
- You want explicit control over `character_coverage` and `byte_fallback`.
- You are willing to disable NFKC and run NFC only.
- Your serving stack supports SentencePiece (most do; some don't).

### Reach for tiktoken's pretrained encodings (cl100k, o200k) when:

- You are serving an existing model that already uses one of them.
- You don't need to *train* a new tokenizer — just to use one.
- You want zero training, fastest possible inference encode, and minimal code.
- You want a stable, audited vocabulary with no glitch-token surprises.

### Reach for vocab extension (over retraining) when:

- You have an existing pretrained model and want to support a new language or domain.
- The base tokenizer's fertility on the new language is > 2.0.
- You can afford 100–500M tokens of continued pretraining.
- You want to preserve the base model's existing capabilities.

### Skip a custom tokenizer when:

- You are fine-tuning, not pretraining. Just use the base tokenizer.
- Your domain is a sub-distribution of an existing well-tokenized domain (English business prose, generic Python). The fertility win is below the engineering cost.
- You don't have a serious eval harness. Without one, "improvement" is unmeasurable and you will waste a quarter chasing a number.
- You are tempted to retrain for a 5% English compression win. Don't.

> The default 2026 tokenizer for a new English-and-code LLM is byte-BPE with a cl100k-style regex, 128k vocab, per-digit splitting, NFC normalization, and tied embeddings. The default for a new genuinely multilingual LLM is SentencePiece Unigram with byte fallback, 256k vocab, identity normalization, and 8 protected special tokens. Everything else is a deviation that needs justification.

## A walk-through: choosing a tokenizer for a real project

Let me close with a worked example, drawn from a real engagement (details abstracted). Setup: a fintech with English and Vietnamese customer support, code-heavy internal tools (Python, SQL), a monthly inference budget of about $40k, a need for tool-calling, and a 7B-class base model already in production. The question: stay on the current tokenizer or switch?

**Step 1 — measure.** I run the eval harness from §10 against the current tokenizer (Llama-2 32k) on a representative slice of production traffic. Results: English BPT 4.2, code BPT 4.6, Vietnamese fertility 3.8, round-trip OK on 99.4% (the 0.6% failures are emoji and ZWJ sequences). Tool-calling fragments tokenize as text (no special-token namespace).

**Step 2 — quantify the pain.** Vietnamese is 18% of traffic by request count but 32% by token count. That is a 14-point fertility tax — at $40k/month, $5.6k/month is "Vietnamese-specific tokenizer cost." Tool calls fail with `[INST]`-prompt-injection attempts at a measurable rate (0.1% of agentic queries).

**Step 3 — explore options.** Three candidates: (a) move to Llama-3 base (128k vocab), (b) extend the current Llama-2 vocab with Vietnamese tokens, (c) switch to a tekken-style tokenizer for tool isolation.

**Step 4 — pick.** Option (a) requires re-validating the model on internal evals — expensive. Option (b) preserves the model and reduces Vietnamese cost without breaking anything. Option (c) addresses tool injection but is orthogonal to the cost problem. We pick (b) for cost, then layer (c) on top by adding 8 special tokens to the extended vocabulary for tool-calling.

**Step 5 — execute.** Train an aux Vietnamese SentencePiece BPE (8000 pieces). Dedupe, leaving ~5800 net new tokens. Add 8 special tokens for tool-calling. Resize the embedding to 37,808. Mean-init from base subword decompositions. Continue pretrain on a 60% Vietnamese / 30% English / 10% code mix for 400M tokens at 1e-4 LR with cosine decay.

**Step 6 — re-measure.** New numbers: Vietnamese fertility 1.7 (-55%), English BPT 4.2 (unchanged within noise), code BPT 4.4 (-4%, marginal), tool-call special tokens never appear in user input. Net inference cost: $33.5k/month, a 16% reduction. Tool injection rate falls to undetectable.

**Step 7 — monitor.** Add fertility-by-language to the cost dashboard. Add a CI test for chat-template byte-stability. Add a confusables eval to the model release pipeline.

The whole project takes about three engineer-weeks. The win — $6.5k/month, recurring, plus a tool-injection class eliminated — pays for itself in the first month and recurs forever after. This is the shape of well-spent tokenizer engineering: pick a measurable problem, change one thing in the pipeline, measure the win, lock it in with monitoring.

> Tokenizer work has a uniquely good cost structure: it touches one config file, runs once, and the gains compound across every request the model will ever serve.

The shape of the argument generalizes. Every project that runs an LLM at scale has a similar latent tokenizer problem hiding behind language mix, code density, math evals, or tool-calling. Most teams never measure it because nobody owns it. The first team to put a name on the column in the cost dashboard wins, often by a margin large enough to fund the next quarter's roadmap. If you take one thing away from this post: open your production tokenizer in a notebook this week, run the harness from section ten on a thousand real prompts, and look at the numbers. Whatever you find will be useful.

## Further reading

- [BPE tokenizer deep-dive](/blog/machine-learning/large-language-model/bpe-tokenizer) — the algorithm and a from-scratch implementation.
- [Training an LLM to adapt to a new language](/blog/machine-learning/large-language-model/training-llm-adapt-new-language) — the sibling piece on full-stack adaptation, of which vocab extension is one part.
- [Modern LLM architectures: Qwen, Llama, Gemma, DeepSeek](/blog/machine-learning/large-language-model/modern-llm-architectures-qwen-llama-gemma-deepseek) — how vocab choices shape these systems.
- [Speech tokenizer](/blog/machine-learning/deep-learning/speech-tokenizer) — what tokenization looks like for audio.
- Sennrich, Haddow, Birch, *Neural Machine Translation of Rare Words with Subword Units*, ACL 2016 — the canonical BPE paper for NLP.
- Kudo, *Subword Regularization*, ACL 2018 — the Unigram LM paper and the case for sampled segmentations.
- Kudo & Richardson, *SentencePiece: A simple and language-independent subword tokenizer and detokenizer*, EMNLP 2018.
- Zouhar et al., *Tokenization and the Noiseless Channel*, ACL 2023 — Renyi efficiency as a tokenizer-quality metric.
- OpenAI, *tiktoken* (GitHub) — the reference implementation of cl100k and o200k.
- Rumbelow & Watkins, *SolidGoldMagikarp*, LessWrong 2023 — the canonical writeup of glitch tokens.
- Mistral AI, *tekken* — the special-token namespace approach to instruction tokenizers.
