---
title: "Inside Gigatoken: SIMD, Cache Hierarchies, and a 1000x Faster BPE Tokenizer"
date: "2026-08-11"
publishDate: "2026-08-11"
description: "A source-level tour of Gigatoken: why pretokenization and not the BPE merges owns half the wall clock, how SWAR and 64-byte mask scanning replace a regex engine, why the pretoken cache is designed around one cache line and the dTLB, and what the 1000x headline actually decomposes into."
tags:
  [
    "gigatoken",
    "tokenization",
    "bpe",
    "simd",
    "swar",
    "rust",
    "cpu-optimization",
    "cache-hierarchy",
    "profiling",
    "training-data",
    "pyo3",
  ]
category: "machine-learning"
subcategory: "Open Source Library"
author: "Hiep Tran"
featured: true
readTime: 59
---

Here is a benchmark line that should make you suspicious. On an AMD EPYC 9565 with 144 cores, tokenizing 11.9 GB of OpenWebText with the GPT-2 vocabulary:

```
gigatoken:  0.486 s | 11920.51 MB at 24532.45 MB/s | 2701.65 Mtok at 5564.94 Mtok/s
       hf:  4.033 s |   100.00 MB at    24.80 MB/s |   22.76 Mtok at    5.63 Mtok/s
gigatoken is 989.21x faster than hf
validation OK: 20401 documents match
```

Roughly a thousand times faster, with a token-identity check confirming the outputs match document for document. The reflex when you see a number like this is to assume someone rewrote a Python library in a fast language. That reflex is wrong here, and the reason it is wrong is what makes [Gigatoken](https://github.com/marcelroed/gigatoken) worth reading.

HuggingFace `tokenizers` is already Rust. `tiktoken` is already Rust. Both already run multithreaded. The baseline in that table is not a slow interpreter; it is compiled, parallel, production code that every major lab runs. So the thousand-fold gap has to come from somewhere other than language choice, and it does: it comes from three separate mechanisms, each worth understanding on its own, layered on top of each other.

![Where a cold 10 GB encode spends its time: the pretokenization walker owns 49.4 percent, the BPE merges only 15.4 percent](/imgs/blogs/inside-gigatoken-simd-cache-hierarchies-1000x-faster-bpe-1.webp)

The diagram above is the mental model, and it already contains the first surprise. When you profile a BPE tokenizer, the byte-pair merge loop, the part the algorithm is named after, is not where the time goes. Splitting the text into words before you ever merge anything is where the time goes: 49.4 percent of a cold 10 GB encode, against 15.4 percent for the merges. Almost every tokenizer in production outsources that splitting step to a regular expression engine, and that decision, made once, quietly sets the ceiling for everyone.

This post is a source-level tour of how Gigatoken removes that ceiling and then keeps going: the SWAR and SIMD byte classification that replaces the regex, the pretoken cache built around a single cache line and the translation lookaside buffer, the multithread orchestration work that turned out to matter more than the encoding itself, and an honest accounting at the end of what the 1000x really decomposes into. Several of the most interesting parts are the things that did not work, because the repository keeps them, with the numbers that killed them.

## 1. The number that should not exist

Gigatoken is a Rust library with Python bindings by Marcel Rød, MIT licensed, that describes itself in seven words: language model tokenization at GB/s. It loads the tokenizers you already use, by HuggingFace Hub repo id, by `tokenizer.json` path, by `.tiktoken` vocabulary file, or by wrapping an already-initialized HuggingFace tokenizer object, and produces byte-identical token ids.

The claim is not narrow. The published benchmark table covers more than twenty distinct tokenizer families across three machines, and the pattern holds across all of them:

| Tokenizer | gigatoken | HF tokenizers | Speedup |
| --- | ---: | ---: | ---: |
| GPT-2 | 24.53 GB/s | 24.8 MB/s | 989x |
| Phi-4 | 24.00 GB/s | 29.9 MB/s | 801x |
| GPT-OSS | 23.96 GB/s | 49.7 MB/s | 482x |
| Qwen 3 | 22.16 GB/s | 34.2 MB/s | 648x |
| Llama 3 / 3.1 / 3.2 | 22.15 GB/s | 48.5 MB/s | 457x |
| DeepSeek V3 / R1 / V4 | 19.69 GB/s | 26.2 MB/s | 750x |
| Kimi K2 | 18.85 GB/s | not published | not published |
| Gemma 3 (SentencePiece) | 3.43 GB/s | 357.2 MB/s | 9.6x |
| Gemma 1 (SentencePiece) | 2.51 GB/s | 342.2 MB/s | 7.3x |

All rows measured on the same AMD EPYC 9565, 144 cores across two sockets, on the same 11.9 GB corpus.

Two things in that table are worth noticing before we go anywhere near the implementation. First, the byte-pair encoding families cluster tightly between roughly 19 and 25 GB/s, which tells you the win is structural rather than a single tokenizer being overfit. Second, the SentencePiece rows are an order of magnitude slower in absolute terms and two orders of magnitude smaller in ratio. That is not a footnote; it is a signpost pointing at exactly which mechanism is doing the work, and we will come back to it in section 16.

There is also a detail in the benchmark methodology that cuts against the author's own headline, and the README states it plainly rather than hiding it. Gigatoken encodes the entire 11.9 GB file un-split, meaning it has to find document boundaries itself and parallelize itself. HuggingFace gets the first 100 MB, pre-split on `<|endoftext|>`, and tiktoken gets the first 1 GB, also pre-split. The comparison is fair only because neither comparison library caches, so their throughput is roughly uniform through the file and a prefix is representative. But it does mean Gigatoken is doing strictly more work per byte in that measurement, not less.

To make the rest of this post concrete, here is the entire installation and usage story:

```bash
pip install gigatoken
```

```python
import gigatoken as gt

# Accepts a HuggingFace Hub repo id, a path to a tokenizer.json or a directory
# containing one, a .tiktoken file, or an already-initialized HF tokenizer.
tokenizer = gt.Tokenizer("openai-community/gpt2")

ids = tokenizer.encode("Tokenize your text data at GB/s!")
print(ids)                                    # numpy uint32 array
print(tokenizer.decode(ids).decode("utf-8"))  # decode returns bytes

# Batches encode in parallel and return a ragged awkward Array,
# one row of token ids per document.
tokens = tokenizer.encode_batch(["The first document.", "And a second one."])

# For model input, assemble the padded matrix directly in Rust.
matrix, lengths = tokenizer.encode_batch_padded(["doc one", "doc two"], pad_id=0)
```

That is the native API. There are also two compatibility wrappers, `as_hf()` and `as_tiktoken()`, which we will look at in section 19, and a files API that is the fast path and which we will look at in section 13.

## 2. What a BPE tokenizer actually does

To see why the regex matters, you need the two-stage structure clearly in mind. If you want the full derivation, the [BPE tokenizer](/blog/machine-learning/large-language-model/bpe-tokenizer) post covers the algorithm and its training procedure; what follows is the minimum needed to read a profile.

Byte-pair encoding at inference time is a greedy merge procedure. You start with a sequence of bytes, you look up every adjacent pair in a table of learned merge ranks, you apply the lowest-ranked merge, and you repeat until no pair in the sequence has a rank. A word like `tokenization` might start as twelve single-byte symbols and end as three tokens.

If you ran that procedure on an entire document at once, you would get the wrong answer and it would be slow. Wrong, because merges would happily cross word boundaries and produce tokens spanning spaces and punctuation in ways the vocabulary was never trained on. Slow, because the merge loop is quadratic-ish in the length of the sequence you feed it.

So every BPE tokenizer first splits the document into small chunks called pretokens, using a fixed rule, and runs the merge loop independently inside each one. For GPT-2, that rule is this regular expression:

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

Read as English, it says: an apostrophe contraction, or an optional space followed by a run of letters, or an optional space followed by a run of digits, or an optional space followed by a run of things that are neither, or a run of whitespace not followed by a non-space, or a run of whitespace. The sentence `The cat sat.` becomes the pretokens `The`, ` cat`, ` sat`, `.`, and each of those four goes through the merge loop separately.

This is a fixed, known, tiny rule. It has no user input, it never changes at runtime, it is baked into the tokenizer definition, and there are only about a dozen distinct variants of it across the entire industry. Gigatoken ships hand-written scanners for the common ones, which you can see directly in the module layout:

```
src/pretokenize/fast/
    r50k.rs           GPT-2
    cl100k.rs         GPT-3.5 / GPT-4
    o200k.rs          GPT-4o / GPT-OSS
    qwen2.rs          Qwen 2 / 2.5
    qwen3_5.rs        Qwen 3.5 / 3.6
    deepseek_v3.rs    DeepSeek V3 / R1 / V4
    kimi.rs           Kimi K2 family
    olmo3.rs          OLMo 2 / 3
    nemotron.rs       Nemotron 3
    mask.rs           shared SIMD mask-scanner infrastructure
```

The critical observation is the mismatch. A regular expression engine is a general machine: it accepts an arbitrary pattern at runtime, compiles it, and walks it with a backtracking or automaton-based matcher that must handle alternation, captures, and lookahead in full generality. Gigatoken's position is that you should not pay for generality you do not use, twelve times, on every byte of your training corpus.

The second stage, the merges, gets its own treatment later. For now note only its shape: the merge loop is per pretoken, and pretokens repeat. The word ` the` appears in an English corpus a very large number of times, and it merges to exactly the same tokens every single time. Hold that thought until section 8.

## 3. The profile says the regex, not the merges

The repository contains something unusual for an open source project: a full profiling report, checked in, with the trace methodology, the sanity checks, and the raw numbers. It is worth reading `profiling/report.md` in its entirety if you care about how to measure anything on a modern CPU, but here is the headline decomposition.

The workload is one cold pass over 10 GB of OpenWebText, GPT-2 tokenizer, on an Apple M4 Max, producing 2,279,617,884 tokens from about 2.08 billion pretokens. Baseline: 10.48 to 10.74 seconds, around 950 MB/s single-threaded. Sampled with samply at 4 kHz, with inline frames resolved through `atos` against a dSYM so that attribution survives fat link-time optimization and aggressive inlining.

| Bucket | Percent of encode | Seconds | ns per pretoken |
| --- | ---: | ---: | ---: |
| Pretokenizer walker (span and mask loops) | 49.4% | 5.28 s | 2.54 |
| BPE merge, cache-miss fallback | 15.4% | 1.65 s | 0.79 |
| Encode driver (probe loop, output callback) | 14.6% | 1.56 s | 0.75 |
| Pretoken cache probe and insert | 14.2% | 1.52 s | 0.73 |
| Cache key packing | 5.6% | 0.60 s | 0.29 |
| memcpy, malloc, syscalls, other | ~0.9% | ~0.1 s | |

Half the time is spent deciding where words begin and end. The algorithm the library is named after gets a sixth.

There is a second number in that report that deserves its own moment. Outside the encode phase entirely, the input `String::from_utf8` validation costs 0.8 to 1.6 seconds of every run, showing up as 7 percent of total process samples under `run_utf8_validation`. The encoder already handles raw bytes. Validating the corpus as UTF-8 before handing it over was pure waste, and the report flags it as a free end-to-end win. This is the kind of thing you only find by profiling the whole process rather than the function you suspect.

![The same pretokenization rule run by a general-purpose regex engine versus a purpose-built byte classifier](/imgs/blogs/inside-gigatoken-simd-cache-hierarchies-1000x-faster-bpe-2.webp)

So how bad is the regex, specifically? The repository's optimization log measures it directly. On Apple Silicon, single-threaded, on 100 MB of OpenWebText, running the GPT-2 pretokenization pattern:

| Implementation | Throughput |
| --- | ---: |
| `fancy-regex` | ~47 MiB/s |
| Hand-rolled state machine | ~380 MiB/s |
| `winnow` parser combinators plus NEON SIMD | ~462 MiB/s |

Forty-seven mebibytes per second. That is the floor a general-purpose regex engine imposes, and it is roughly eight times slower than a hand-written state machine doing the identical job. The `winnow` plus NEON version, which was the pre-existing best in the project, uses real SIMD intrinsics to scan sixteen bytes at a time inside letter and digit runs, and still only reaches 462 MiB/s, because it pays parser-combinator framework overhead at every token start.

The target the author set was 1 GiB/s single-threaded. Getting there took eight documented steps, four of which worked.

## 4. Killing the regex, part 1: the class table and SWAR

The first rewrite threw away both the regex and the combinator framework, and replaced the SIMD intrinsics with something more portable and, surprisingly, faster.

Three changes landed together. First, the `winnow` combinators, with their `alt()`, `trace()`, `backtrack()` and `ModalResult` machinery, were replaced by a direct `Iterator` implementation with zero framework overhead per token. Second, a 256-byte lookup table gave constant-time classification of the first byte of every token, dispatching straight to the correct scan function instead of cascading through alternatives until one matched. Third, and most interestingly, the NEON intrinsics inside the scan loops were replaced by SWAR.

SWAR stands for SIMD Within A Register. The idea is old and underused: load eight bytes into an ordinary 64-bit integer, then apply branchless arithmetic that operates on all eight lanes simultaneously, using nothing but the scalar ALU. No intrinsics, no target-feature gates, no architecture-specific code paths, and it compiles to the same handful of instructions everywhere.

Here is the letter scan, which is the technique in its clearest form:

```rust
// Adapted from src/pretokenize, simplified for exposition.
const HI: u64 = 0x8080_8080_8080_8080;

/// Returns a mask with bit 7 of each lane set where that byte is an ASCII letter.
#[inline(always)]
fn letter_mask(word: u64) -> u64 {
    // Case-fold all eight bytes at once: 'A' | 0x20 == 'a'.
    let lowered = word | 0x2020_2020_2020_2020;

    // Two branchless range comparisons across all eight lanes.
    let ge_a = (lowered | HI).wrapping_sub(0x6161_6161_6161_6161); // >= 'a'
    let le_z = 0x7A7A_7A7A_7A7A_7A7A_u64.wrapping_sub(lowered);    // <= 'z'

    ge_a & le_z & HI
}

/// Index of the first non-letter in the next eight bytes, or 8 if all are letters.
#[inline(always)]
fn first_non_letter(word: u64) -> u32 {
    let mask = letter_mask(word);
    ((!mask) & HI).to_le().trailing_zeros() / 8
}
```

Six arithmetic operations classify eight bytes. One `trailing_zeros`, which is a single instruction on every CPU worth targeting, finds the first byte that breaks the run. Compare that with a regex engine, which for these same eight bytes would perform eight separate character-class tests, each with its own conditional branch, each of which the branch predictor must guess.

![The SWAR letter scan: case-fold, two range comparisons, and one mask, across eight lanes at once](/imgs/blogs/inside-gigatoken-simd-cache-hierarchies-1000x-faster-bpe-3.webp)

The dispatch side matters too. Rather than a chain of `if` tests or a combinator alternation, the first byte of each token indexes a 256-entry class table, and inside the scan loops the predicates are arithmetic rather than table lookups:

```rust
#[inline(always)]
fn is_letter(b: u8) -> bool {
    (b | 0x20).wrapping_sub(b'a') < 26
}

#[inline(always)]
fn is_digit(b: u8) -> bool {
    b.wrapping_sub(b'0') < 10
}
```

Both of these are two instructions and a comparison, with no memory access at all. That last property is the point. A table lookup inside a hot scan loop is a data-dependent load, which means the CPU cannot proceed until it arrives from L1, and L1 latency is several cycles. The arithmetic form issues in the ALU alongside everything else. The class table stays, because at the token start you genuinely have many-way dispatch and a jump table is the right structure; inside the runs, where you are asking one yes-or-no question millions of times, arithmetic wins.

The result: 462 MiB/s to 830 MiB/s. Adding `unsafe { *bytes.get_unchecked(pos) }` in the scan loops, which removes a redundant bounds check and therefore a predictable-but-real branch per byte, took it to 840 MiB/s. Separating an `advance()` method that moves the cursor from the `next()` that constructs a slice, so that pure counting never builds an `Option<Pretoken>`, added another 8 MiB/s.

That is 848 MiB/s, and it is where the straightforward ideas ran out.

## 5. Killing the regex, part 2: the mask scanner

The scalar SWAR scanners are not what ships as the fast path today. They remain in the tree as the ground truth and the fallback for CPUs without SIMD, but the production pretokenizers are mask scanners, and they work at a different granularity: 64 bytes at a time.

The module documentation in `src/pretokenize/fast/mask.rs` lays out the design in four layers:

1. Platform SIMD primitives. `movemask64` and `ascii_masks` on NEON, `ascii_masks_avx512` and `ascii_masks_avx2` on x86-64. This is the only per-platform code in the system.
2. Bit-domain helpers shared across schemes: platform-independent `u64` algebra and per-character table classification, parameterized by each scheme's codepoint classifier.
3. Per-scheme boundary algebra, living in each tokenizer's own module.
4. The scheme-agnostic batch walker that ties it together.

The flow for one 64-byte batch is: SIMD classifies every byte into letter, digit, whitespace, or other; a movemask reduces those four 16-lane vectors to a single 64-bit integer where bit $i$ corresponds to byte $i$; then plain `u64` bit algebra derives which of those positions are token starts.

![The mask scanner: classify wide, reduce to bits, then derive token boundaries with u64 algebra](/imgs/blogs/inside-gigatoken-simd-cache-hierarchies-1000x-faster-bpe-4.webp)

The elegant part is how the scheme handles the cases where bit algebra is not trustworthy. `batch_masks` returns two masks, not one. `usable` marks positions that are definitely token starts. `bad` marks zones the scheme cannot classify in-mask, meaning non-ASCII sequences and ambiguities at the batch edges. The walker never emits a token across an unresolved zone; it falls back to the scalar `advance` function, the same one from section 4, to re-derive those regions exactly. The fast path handles the 99-plus percent of English text that is ASCII, and correctness for everything else is guaranteed by the slow path that was already written and already tested.

Runtime dispatch decides which tier you get. On x86-64 the library checks for the full AVX-512 set, `avx512f`, `avx512bw`, `avx512vl`, plus `bmi1`, `bmi2`, `lzcnt` and `popcnt`, and falls back to an AVX2 tier requiring `avx2` and the same bit-manipulation features, and finally to pure scalar. Those scalar-visible bit features are requested explicitly, not assumed, because the boundary algebra inlined into the SIMD functions needs to compile to `tzcnt` and `blsr` rather than baseline-x86 `bsf` sequences. There is a separate check for `avx512vbmi2`, which provides native 512-bit `vpcompressb`, gating a further tier. Skylake-X has AVX-512 but lacks VBMI2, so the bit is detected rather than inferred.

On top of that sits the two-phase walker, and this is where the design gets genuinely clever.

<figure class="blog-anim">
<svg viewBox="0 0 800 344" role="img" aria-label="A 64-byte window advances along a text buffer; phase A pops boundary bits into a flat buffer, then phase B drains that buffer into emitted spans" style="width:100%;height:auto;max-width:860px">
<style>
.w1-cell{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1}
.w1-bnd{fill:var(--accent,#6366f1);stroke:none;opacity:.55}
.w1-win{fill:none;stroke:var(--accent,#6366f1);stroke-width:3}
.w1-slot{fill:none;stroke:var(--border,#d1d5db);stroke-width:1.5}
.w1-fill{fill:var(--accent,#6366f1)}
.w1-span{fill:#10b981}
.w1-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.w1-sub{font:400 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.w1-in{font:600 12px ui-sans-serif,system-ui;fill:white;text-anchor:middle}
@keyframes w1-step{0%{transform:translateX(0)}100%{transform:translateX(528px)}}
@keyframes w1-harvest{0%{opacity:.08}6%{opacity:1}50%{opacity:1}54%,100%{opacity:.08}}
@keyframes w1-emit{0%,50%{opacity:0}56%{opacity:1}100%{opacity:1}}
.w1-move{animation:w1-step 20s steps(4,jump-none) infinite}
.w1-h{animation:w1-harvest 5s ease-in-out infinite}
.w1-e{animation:w1-emit 5s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.w1-move{animation:none}.w1-h{animation:none;opacity:1}.w1-e{animation:none;opacity:1}.w1-last{opacity:.25}}
</style>
<text class="w1-lbl" x="40" y="24">phase A: harvest every boundary in the batch, branch-free</text>
<g>
<rect class="w1-cell" x="40" y="40" width="18" height="44" rx="3"/>
<rect class="w1-cell" x="62" y="40" width="18" height="44" rx="3"/>
<rect class="w1-cell" x="84" y="40" width="18" height="44" rx="3"/>
<rect class="w1-cell" x="106" y="40" width="18" height="44" rx="3"/>
<rect class="w1-cell" x="128" y="40" width="18" height="44" rx="3"/>
<rect class="w1-cell" x="150" y="40" width="18" height="44" rx="3"/>
<rect class="w1-cell" x="172" y="40" width="18" height="44" rx="3"/>
<rect class="w1-cell" x="194" y="40" width="18" height="44" rx="3"/>
<rect class="w1-cell" x="216" y="40" width="18" height="44" rx="3"/>
<rect class="w1-cell" x="238" y="40" width="18" height="44" rx="3"/>
<rect class="w1-cell" x="260" y="40" width="18" height="44" rx="3"/>
<rect class="w1-cell" x="282" y="40" width="18" height="44" rx="3"/>
<rect class="w1-cell" x="304" y="40" width="18" height="44" rx="3"/>
<rect class="w1-cell" x="326" y="40" width="18" height="44" rx="3"/>
<rect class="w1-cell" x="348" y="40" width="18" height="44" rx="3"/>
<rect class="w1-cell" x="370" y="40" width="18" height="44" rx="3"/>
<rect class="w1-cell" x="392" y="40" width="18" height="44" rx="3"/>
<rect class="w1-cell" x="414" y="40" width="18" height="44" rx="3"/>
<rect class="w1-cell" x="436" y="40" width="18" height="44" rx="3"/>
<rect class="w1-cell" x="458" y="40" width="18" height="44" rx="3"/>
<rect class="w1-cell" x="480" y="40" width="18" height="44" rx="3"/>
<rect class="w1-cell" x="502" y="40" width="18" height="44" rx="3"/>
<rect class="w1-cell" x="524" y="40" width="18" height="44" rx="3"/>
<rect class="w1-cell" x="546" y="40" width="18" height="44" rx="3"/>
<rect class="w1-cell" x="568" y="40" width="18" height="44" rx="3"/>
<rect class="w1-cell" x="590" y="40" width="18" height="44" rx="3"/>
<rect class="w1-cell" x="612" y="40" width="18" height="44" rx="3"/>
<rect class="w1-cell" x="634" y="40" width="18" height="44" rx="3"/>
<rect class="w1-cell" x="656" y="40" width="18" height="44" rx="3"/>
<rect class="w1-cell" x="678" y="40" width="18" height="44" rx="3"/>
<rect class="w1-cell" x="700" y="40" width="18" height="44" rx="3"/>
<rect class="w1-cell" x="722" y="40" width="18" height="44" rx="3"/>
</g>
<g>
<rect class="w1-bnd" x="62" y="40" width="18" height="44" rx="3"/>
<rect class="w1-bnd" x="128" y="40" width="18" height="44" rx="3"/>
<rect class="w1-bnd" x="172" y="40" width="18" height="44" rx="3"/>
<rect class="w1-bnd" x="238" y="40" width="18" height="44" rx="3"/>
<rect class="w1-bnd" x="304" y="40" width="18" height="44" rx="3"/>
<rect class="w1-bnd" x="348" y="40" width="18" height="44" rx="3"/>
<rect class="w1-bnd" x="414" y="40" width="18" height="44" rx="3"/>
<rect class="w1-bnd" x="480" y="40" width="18" height="44" rx="3"/>
<rect class="w1-bnd" x="524" y="40" width="18" height="44" rx="3"/>
<rect class="w1-bnd" x="590" y="40" width="18" height="44" rx="3"/>
<rect class="w1-bnd" x="656" y="40" width="18" height="44" rx="3"/>
<rect class="w1-bnd" x="700" y="40" width="18" height="44" rx="3"/>
</g>
<rect class="w1-win w1-move" x="36" y="34" width="184" height="56" rx="8"/>
<text class="w1-sub" x="40" y="108">the 64-byte window advances one batch per cycle; shaded cells are token starts</text>
<text class="w1-lbl" x="40" y="146">flat boundary buffer, written without a data-dependent branch</text>
<rect class="w1-slot" x="40" y="160" width="88" height="44" rx="6"/>
<rect class="w1-slot" x="148" y="160" width="88" height="44" rx="6"/>
<rect class="w1-slot" x="256" y="160" width="88" height="44" rx="6"/>
<rect class="w1-slot" x="364" y="160" width="88" height="44" rx="6"/>
<rect class="w1-slot" x="472" y="160" width="88" height="44" rx="6"/>
<rect class="w1-slot" x="580" y="160" width="88" height="44" rx="6"/>
<g class="w1-h">
<rect class="w1-fill" x="40" y="160" width="88" height="44" rx="6"/>
<text class="w1-in" x="84" y="188">off 1</text>
</g>
<g class="w1-h" style="animation-delay:.5s">
<rect class="w1-fill" x="148" y="160" width="88" height="44" rx="6"/>
<text class="w1-in" x="192" y="188">off 4</text>
</g>
<g class="w1-h" style="animation-delay:1s">
<rect class="w1-fill" x="256" y="160" width="88" height="44" rx="6"/>
<text class="w1-in" x="300" y="188">off 6</text>
</g>
<text class="w1-sub" x="40" y="228">phase B pops one entry and emits one span, then the next: a counted loop, no branch on the data</text>
<text class="w1-lbl" x="40" y="262">spans emitted</text>
<g class="w1-e">
<rect class="w1-span" x="180" y="274" width="150" height="40" rx="8"/>
<text class="w1-in" x="255" y="299">span 1</text>
</g>
<g class="w1-e" style="animation-delay:.5s">
<rect class="w1-span" x="342" y="274" width="110" height="40" rx="8"/>
<text class="w1-in" x="397" y="299">span 2</text>
</g>
<g class="w1-e w1-last" style="animation-delay:1s">
<rect class="w1-span" x="464" y="274" width="190" height="40" rx="8"/>
<text class="w1-in" x="559" y="299">span 3</text>
</g>
<text class="w1-sub" x="40" y="336">harvest is batched and branch-free; emission is a straight counted loop over what was harvested</text>
</svg>
<figcaption>Two-phase walking: phase A fills a flat boundary buffer, phase B drains it without a single data-dependent branch.</figcaption>
</figure>

Phase A harvests every span boundary in a 64-byte block, branchlessly, into a small flat buffer. Phase B consumes that buffer in a straight counted loop, emitting one span per entry. The alternative, and what the code did before, was a single fused loop that pulled one span at a time with a ladder of data-dependent branches: is there a remainder, do we need a segment refill, is this the scalar or the mask path. The comment in the source is blunt about why that ladder had to go: those branches "were the largest single source of encode's discarded issue bandwidth."

The function carries `#[inline(never)]`, deliberately, with a stated reason: each monomorphization becomes its own out-of-line loop, keeping its register allocation away from the register-hungry encode loop that calls it. And there is a measurement attached to the decision to have this interface at all. Routing spans through `Iterator::next` instead of the fused, always-inlined walker body measured about 23 percent of warm encode time in un-inlined call overhead. Twenty-three percent, for using the standard iterator protocol.

## 6. Latency, not throughput: the dual-cursor trick

At 848 MiB/s the scalar pretokenizer hit a wall, and the diagnosis of that wall is my favorite thing in the entire repository.

The instinct when a loop stops getting faster is that it is doing too much work. The author checked, and it was not. The bottleneck was that the work had nowhere to go. Every token has a serial dependency chain:

```
find_end(token N) -> pos_N -> load byte at pos_N -> classify -> scan -> find_end(token N+1)
```

You cannot start locating token $N+1$ until you know where token $N$ ended. That chain measures roughly 25 to 27 cycles on a modern core. Meanwhile an Apple M4 P-core is 8-wide: it can issue eight micro-operations per cycle. During those 25 cycles of waiting, the out-of-order engine has execution ports sitting completely idle, because there is no independent work available to fill them.

The fix is to manufacture independent work. Split the input at a safe boundary and run two cursors at once:

```rust
// Find a split point near the midpoint that is guaranteed to be a token
// boundary: a '\n' followed by a non-whitespace ASCII byte always splits.
let split = find_split(bytes);

let (mut p1, mut p2) = (0usize, split);
let mut count = 0usize;

while p1 < split && p2 < len {
    p1 = advance_pos(bytes, p1);   // chain 1
    p2 = advance_pos(bytes, p2);   // chain 2, fully independent
    count += 2;
}
```

Those two calls look sequential in the source. They are not, to the hardware. Different positions, different memory addresses, different registers, no data dependency between them. The out-of-order engine interleaves their micro-operations across execution ports, so while cursor one waits on a SWAR comparison, cursor two's loads and ALU operations execute on ports that were previously idle.

<figure class="blog-anim">
<svg viewBox="0 0 800 330" role="img" aria-label="Two tracks of CPU issue slots: with one cursor the dependent chain leaves two slots in three idle, with two cursors the second cursor fills exactly those gaps" style="width:100%;height:auto;max-width:860px">
<style>
.w2-slot{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1}
.w2-c1{fill:var(--accent,#6366f1)}
.w2-c2{fill:#10b981}
.w2-op{font:700 11px ui-monospace,SFMono-Regular,monospace;fill:white;text-anchor:middle}
.w2-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.w2-sub{font:400 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.w2-head{fill:var(--accent,#6366f1);opacity:.14}
@keyframes w2-sweep{0%{transform:translateX(0)}100%{transform:translateX(720px)}}
@keyframes w2-fire{0%{opacity:.12}2%{opacity:1}55%{opacity:1}60%,100%{opacity:.12}}
.w2-mv{animation:w2-sweep 9s linear infinite}
.w2-f{animation:w2-fire 9s linear infinite}
@media (prefers-reduced-motion:reduce){.w2-mv{animation:none}.w2-f{animation:none;opacity:1}}
</style>
<text class="w2-lbl" x="40" y="26">one cursor: 840 MiB/s</text>
<g>
<rect class="w2-slot" x="60" y="42" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="100" y="42" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="140" y="42" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="180" y="42" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="220" y="42" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="260" y="42" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="300" y="42" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="340" y="42" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="380" y="42" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="420" y="42" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="460" y="42" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="500" y="42" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="540" y="42" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="580" y="42" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="620" y="42" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="660" y="42" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="700" y="42" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="740" y="42" width="34" height="44" rx="4"/>
</g>
<g class="w2-f"><rect class="w2-c1" x="60" y="42" width="34" height="44" rx="4"/><text class="w2-op" x="77" y="70">ld</text></g>
<g class="w2-f" style="animation-delay:1.5s"><rect class="w2-c1" x="180" y="42" width="34" height="44" rx="4"/><text class="w2-op" x="197" y="70">cl</text></g>
<g class="w2-f" style="animation-delay:3s"><rect class="w2-c1" x="300" y="42" width="34" height="44" rx="4"/><text class="w2-op" x="317" y="70">sc</text></g>
<g class="w2-f" style="animation-delay:4.5s"><rect class="w2-c1" x="420" y="42" width="34" height="44" rx="4"/><text class="w2-op" x="437" y="70">ld</text></g>
<g class="w2-f" style="animation-delay:6s"><rect class="w2-c1" x="540" y="42" width="34" height="44" rx="4"/><text class="w2-op" x="557" y="70">cl</text></g>
<g class="w2-f" style="animation-delay:7.5s"><rect class="w2-c1" x="660" y="42" width="34" height="44" rx="4"/><text class="w2-op" x="677" y="70">sc</text></g>
<text class="w2-sub" x="40" y="108">every step waits on the one before it, so two issue slots in three sit empty</text>
<text class="w2-lbl" x="40" y="150">two cursors: 1049 MiB/s</text>
<g>
<rect class="w2-slot" x="60" y="166" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="100" y="166" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="140" y="166" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="180" y="166" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="220" y="166" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="260" y="166" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="300" y="166" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="340" y="166" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="380" y="166" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="420" y="166" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="460" y="166" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="500" y="166" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="540" y="166" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="580" y="166" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="620" y="166" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="660" y="166" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="700" y="166" width="34" height="44" rx="4"/>
<rect class="w2-slot" x="740" y="166" width="34" height="44" rx="4"/>
</g>
<g class="w2-f"><rect class="w2-c1" x="60" y="166" width="34" height="44" rx="4"/><text class="w2-op" x="77" y="194">ld</text></g>
<g class="w2-f" style="animation-delay:.5s"><rect class="w2-c2" x="100" y="166" width="34" height="44" rx="4"/><text class="w2-op" x="117" y="194">ld</text></g>
<g class="w2-f" style="animation-delay:1.5s"><rect class="w2-c1" x="180" y="166" width="34" height="44" rx="4"/><text class="w2-op" x="197" y="194">cl</text></g>
<g class="w2-f" style="animation-delay:2s"><rect class="w2-c2" x="220" y="166" width="34" height="44" rx="4"/><text class="w2-op" x="237" y="194">cl</text></g>
<g class="w2-f" style="animation-delay:3s"><rect class="w2-c1" x="300" y="166" width="34" height="44" rx="4"/><text class="w2-op" x="317" y="194">sc</text></g>
<g class="w2-f" style="animation-delay:3.5s"><rect class="w2-c2" x="340" y="166" width="34" height="44" rx="4"/><text class="w2-op" x="357" y="194">sc</text></g>
<g class="w2-f" style="animation-delay:4.5s"><rect class="w2-c1" x="420" y="166" width="34" height="44" rx="4"/><text class="w2-op" x="437" y="194">ld</text></g>
<g class="w2-f" style="animation-delay:5s"><rect class="w2-c2" x="460" y="166" width="34" height="44" rx="4"/><text class="w2-op" x="477" y="194">ld</text></g>
<g class="w2-f" style="animation-delay:6s"><rect class="w2-c1" x="540" y="166" width="34" height="44" rx="4"/><text class="w2-op" x="557" y="194">cl</text></g>
<g class="w2-f" style="animation-delay:6.5s"><rect class="w2-c2" x="580" y="166" width="34" height="44" rx="4"/><text class="w2-op" x="597" y="194">cl</text></g>
<g class="w2-f" style="animation-delay:7.5s"><rect class="w2-c1" x="660" y="166" width="34" height="44" rx="4"/><text class="w2-op" x="677" y="194">sc</text></g>
<g class="w2-f" style="animation-delay:8s"><rect class="w2-c2" x="700" y="166" width="34" height="44" rx="4"/><text class="w2-op" x="717" y="194">sc</text></g>
<rect class="w2-head w2-mv" x="56" y="38" width="42" height="176" rx="6"/>
<text class="w2-sub" x="40" y="232">cursor 2 issues into exactly the slots cursor 1 left empty: the core was already waiting</text>
<rect class="w2-c1" x="40" y="258" width="16" height="16" rx="3"/>
<text class="w2-sub" x="64" y="271">cursor 1</text>
<rect class="w2-c2" x="150" y="258" width="16" height="16" rx="3"/>
<text class="w2-sub" x="174" y="271">cursor 2</text>
<text class="w2-sub" x="270" y="271">ld = load, cl = classify, sc = scan for the next boundary</text>
<text class="w2-lbl" x="40" y="308">the gaps got filled: same work per op, 25% more throughput</text>
</svg>
<figcaption>Dual-cursor ILP: the second cursor costs nothing because the core was already waiting.</figcaption>
</figure>

The result: 840 MiB/s to 1,049 MiB/s. A 25 percent speedup, achieved by doing exactly the same total work in exactly the same way, and crossing the 1 GiB/s target that had been the goal.

Note the enabling detail, because it is the part that generalizes. The trick needs a split point that is provably a token boundary, otherwise the second cursor might start mid-token and produce garbage. A newline followed by a non-whitespace ASCII byte always splits under the GPT-2 rule, so `find_split` searches near the midpoint for that pattern. Knowing your grammar well enough to name a guaranteed-safe cut point is what buys the parallelism, and the same idea, at a much coarser grain, is how the library parallelizes an entire 12 GB file in section 13.

The general lesson is worth stating on its own. When a tight loop stops responding to work reduction, stop counting instructions and start counting dependencies. Instruction-level parallelism is free performance that is invisible in an instruction count and invisible in a flame graph; it shows up only when you ask what each instruction is waiting for. The [roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) covers the compute-versus-memory version of this question; dependency-chain latency is a third axis that neither ceiling captures.

## 7. Eight steps, and the four that did not work

The repository ships `pretokenizer_optimization_log.md`, which records all eight steps of that campaign, including the ones that regressed. Keeping negative results, with numbers, is rare enough that it is worth spending a section on what they teach.

![The optimization log's full ledger: four steps moved the number, two made it worse, two changed nothing](/imgs/blogs/inside-gigatoken-simd-cache-hierarchies-1000x-faster-bpe-7.webp)

**Step 4, the hot/cold split, regressed from 830 to 580 MiB/s.** The idea was textbook: English text is more than 99 percent ASCII, so move all the Unicode continuation handling into separate functions marked `#[cold]` and `#[inline(never)]`, shrinking the hot path's instruction footprint and improving instruction-cache utilization. It cost 30 percent. The `#[inline(never)]` barrier prevented LLVM from optimizing the combined ASCII-plus-Unicode loop as a unit, and the roughly 5-cycle function call on each Unicode encounter outweighed the instruction-cache benefit. Reverted.

**Step 6, the two-pass classification buffer, reached only 354 MiB/s against 848 for one-pass.** This one is the most instructive, because it was algorithmically correct, verified on 5 MB, and built on a genuinely good idea: classify every byte into a buffer, then XOR adjacent classes with SWAR to detect transitions and `count_ones()` them, which removes branch mispredictions from boundary detection entirely. It lost by a factor of 2.5. Writing all the classes and reading them back doubles memory traffic, and three passes over the data read more total bytes than one pass. Branch-free is not free when it costs you bandwidth.

The failure also generated a small catalogue of correctness hazards that is instructive on its own: whitespace and space are different class values that must be merged before the XOR or they generate false transitions; apostrophe and other need merging too, since a non-contraction apostrophe scans as "other"; contractions like `c'mon` where a letter follows require subtracting the apostrophe-to-letter transition only when the byte after the contraction is a different class; and multi-byte UTF-8 characters spanning chunk boundaries need chunk-end alignment. Every one of those is a real bug someone would have shipped.

**Step 7, profile-guided optimization, did nothing, and hurt the state machine by 13 percent.** Build with `-Cprofile-generate`, run the benchmark, rebuild with `-Cprofile-use`, measure 842 MiB/s against an 847 MiB/s baseline. The explanation is precise: the SWAR inner loop is already branchless, so there are no branch probabilities left to optimize; the word-boundary branch is fundamentally unpredictable because its outcome depends on the input data rather than on any static code pattern; and the class-table dispatch compiles to a jump table, where PGO cannot improve indirect branch prediction. PGO helps when your branches have stable biases. Data-dependent branches have none.

**Step 3, arithmetic space dispatch, was a wash at 840 MiB/s.** It replaced a second class-table lookup with direct arithmetic when the first byte is a space, which is the most common token-start pattern in English. No measurable change, but it was kept anyway, on the reasoning that it eliminates a latency risk on architectures with higher L1 latency than the test machine. That is a defensible reason to keep a neutral change, and it is stated rather than assumed.

Four steps worked: the SWAR rewrite (+1.80x), `get_unchecked` (+1.01x), the advance/count split (+1.01x), and dual-cursor ILP (+1.25x). Total: 2.27x over the `winnow` plus NEON baseline, 22.3x over the regex.

The pattern across the failures is that each one optimized a resource the core was not short of. Instruction cache, when the core had icache headroom. Branch predictability, at the cost of bandwidth. Static branch profiles, for data-dependent branches. Knowing which resource is actually scarce is the entire game, which is precisely what section 12 is about.

## 8. Zipf is the whole game

Everything so far has been about the pretokenizer, which is half the profile. The other mechanism, the one that produces the truly outsized numbers, is a cache, and its design starts from a property of language rather than a property of CPUs.

Pretoken frequency is Zipf-distributed. The word ` the` is astronomically more common than the word ` antidisestablishmentarianism`, and the distribution between them follows a power law over many orders of magnitude. This has two consequences that pull in opposite directions.

![Pretoken frequency follows Zipf, so a bounded cache absorbs 99.4 percent of lookups](/imgs/blogs/inside-gigatoken-simd-cache-hierarchies-1000x-faster-bpe-8.webp)

The good consequence: caching works spectacularly. Measured on 1 GB of OpenWebText with the GPT-2 tokenizer, the table holds about 1.3 million unique short pretokens and achieves roughly a 99.4 percent hit rate. If you have already encoded ` the`, you never need to run the merge loop for it again; you look up its token ids. Nearly all of the BPE algorithm evaporates.

The bad consequence, which the README states directly, is that "caching is a very hard problem in this domain since the cache grows very quickly, and pretoken distributions are very long-tailed." A 99.4 percent hit rate still means six lookups in a thousand miss, and at 2 billion pretokens that is 12 million trips through the merge loop. More importantly, 1.3 million entries is far beyond L2 or L3, so a hit in the Zipf tail is a random access to DRAM. You have replaced a compute problem with a memory-latency problem, which is progress only if you then engineer the memory access properly.

The growth rate follows Heaps' law, and the repository calibrates it on real data: distinct short pretokens number about 1.3 million at 1 GB and about 5.5 million at 10 GB of OpenWebText-like text, giving $\text{distinct}(n) \approx 3.45 \, n^{0.62}$. That sublinear exponent is what makes the whole approach viable. Doubling your corpus does not double your cache; it multiplies it by about $2^{0.62} \approx 1.54$. This is not a curiosity, either: that fitted exponent is used at runtime to pre-size each parallel worker's table, which we will get to in section 14.

What happens on a miss is worth a note, because Gigatoken has three separate merge implementations with disjoint domains rather than one general one. Pretokens of 15 symbols or fewer go to `bpe_merge_symbols_short_scalar` or a NEON variant. Sequences from 16 up to `SMALL_MERGE_MAX = 32` use a linear-scan merge in the style of tiktoken's `byte_pair_merge`: keep a per-position rank, find the minimum by linear scan, merge, recompute only the two affected neighbors. The scan over a few stack-resident `u32` values beats a `BinaryHeap`'s sift traffic at those sizes, while producing identical merge priority. Anything longer uses the heap. The source notes that unifying these three "was measured as a regression risk to the tuned short-miss path," which is an honest way of saying the duplication is deliberate and earns its keep.

## 9. A hash table designed around one cache line

Given the diagnosis in section 8, the cache lookup is a random DRAM access on the tail, and it happens roughly two billion times. Everything about the table's layout follows from making that one access as cheap as possible.

The module documentation states the reasoning up front, and it is a model of how to write a design comment:

> The table holds ~1.3M unique pretokens (~99.4% hit rate), far beyond L2/L3, so a lookup in the Zipf tail is a random DRAM access. hashbrown spends two cache lines per probe (control bytes + entry); this table's 32-byte entries are self-contained and bucketed into line-aligned pairs, so a probe touches exactly one line.

Here is the entry:

```rust
/// One slot: the packed pretoken key plus its packed encoding.
/// Exactly 32 bytes: two slots per cache line, never straddling one.
#[derive(Clone, Copy)]
#[repr(C)]
struct Entry {
    key: u128,
    val: u64,
    ext: u64,
}

const _: () = assert!(std::mem::size_of::<Entry>() == 32);
const EMPTY_KEY: u128 = 0;
```

That `const` assertion is the whole design defended at compile time. Thirty-two bytes, exactly, so that two entries fit in one 64-byte cache line and neither straddles a boundary.

![A 64-byte cache line holds one bucket of two probe slots, so a lookup touches exactly one line](/imgs/blogs/inside-gigatoken-simd-cache-hierarchies-1000x-faster-bpe-9.webp)

Linear probing then operates over aligned pairs: a bucket is slots `idx` and `idx + 1` with `idx` even, so both share one line, and `probe_pair` resolves the overwhelmingly common displacement-0 or displacement-1 hit branch-free from that single fetch. Compare with `hashbrown`, Rust's excellent general-purpose hash table, which stores control bytes separately from entries and therefore touches two cache lines per probe. On an in-cache workload that costs nothing. On a two-billion-lookup random-DRAM workload it is the difference between one memory stall and two.

The key packing is equally deliberate. A pretoken of 15 bytes or fewer packs into a `u128`: the bytes occupy the low 15 lanes, and the length goes in the top byte. Two properties fall out for free. Keys of different lengths can never collide, since the length is part of the key. And a real key is never zero, because a real pretoken has a nonzero length in that top byte, which means zero is available as the empty-slot sentinel and a freshly zeroed allocation is a valid empty table.

The packing itself avoids the obvious implementation:

```rust
#[inline(always)]
pub(crate) fn pack_pretoken_key(bytes: &[u8]) -> Option<u128> {
    let n = bytes.len();
    if n > 15 {
        return None;  // long pretokens use the slice-keyed fallback map
    }
    if n == 0 {
        return Some(0);
    }
    let p = bytes.as_ptr();
    let low = if (p as usize) & 4095 <= 4096 - 16 {
        // SAFETY: the offset within the page is <= 4096 - 16, so a 16-byte
        // read stays inside the page holding `p`, which is mapped because
        // `p` points to at least one valid byte.
        unsafe { (p as *const u128).read_unaligned() }
    } else {
        // Rare near-page-boundary case: plain copy. Identical key either way.
        let mut buf = [0u8; 16];
        buf[..n].copy_from_slice(bytes);
        u128::from_le_bytes(buf)
    };
    let (lo, hi) = pack_mask_halves(n);
    Some((low & (lo as u128 | ((hi as u128) << 64))) | ((n as u128) << 120))
}
```

The common path is one unaligned 16-byte load plus a mask, instead of a variable-length `memcpy` or a per-byte loop. It reads past the end of the pretoken, which is safe because the read cannot cross a page boundary, and a page that holds one valid byte is mapped in its entirety. Both paths produce identical keys; the fallback exists purely so the fast path can be unconditional.

Even the mask computation was optimized. It used to be a 16-entry table, `PACK_MASK[n]`, until profiling showed that dependent L1 load was 2.43 percent of total process samples, sitting directly on the length-to-key-to-store chain. It became arithmetic:

```rust
#[inline(always)]
pub(crate) const fn pack_mask_halves(n: usize) -> (u64, u64) {
    debug_assert!(n >= 1 && n <= 15);
    let s = (n * 8) as u32;
    let lo = if n < 8 { u64::MAX >> (64u32.wrapping_sub(s) & 63) } else { u64::MAX };
    let hi = if n > 8 { u64::MAX >> (128u32.wrapping_sub(s) & 63) } else { 0 };
    (lo, hi)
}
```

Two independent 3-deep chains of single-cycle shifts, and no load port used at all. The comment notes that a `u128` shift would lower to a multi-instruction sequence, which is why it is computed as two halves. It is also `const`, so the emission loop's table form is generated from this same function at compile time, guaranteeing the two forms cannot drift apart. That is a nice piece of defensive engineering: the optimization did not delete the table, it made the table derived.

One more detail, easy to miss and impossible to debug later: the file contains a `compile_error!` for big-endian targets. The key packing and the token-lane stores both read and write native-endian words, so a big-endian build would silently produce wrong keys and swapped tokens. Refusing to compile is the correct response.

## 10. Packing the value: ninety percent of pretokens are one token

Finding the entry in one memory access is only half the problem. If the entry then contains a pointer to the token ids, you have earned yourself a second random access, and you are back where you started.

The distribution saves you again. From the module docs: 228 million output tokens from 208 million pretokens, meaning about 90 percent of pretokens encode to exactly one token and about 98 percent to at most two. The tokenizer's vocabulary was trained on this corpus's distribution, so common words are single tokens by construction.

![The packed cache value: up to four token ids inline, with the rare long encoding spilled off the hot path](/imgs/blogs/inside-gigatoken-simd-cache-hierarchies-1000x-faster-bpe-10.webp)

So the value is packed inline:

```rust
/// Cache-value packing. `val` low byte: token count in bits 0-6 plus a
/// "spilled" flag in bit 7. Inline values (1-4 tokens; only the first ID
/// must fit 24 bits, true of every real vocab) carry tokens 1-2 in `val`
/// bits 8-31 and 32-63, and tokens 3-4 in `ext`'s two u32 lanes; spilled
/// values carry the token-arena offset in `val`'s high 32 bits.
const VAL_SPILL: u64 = 0x80;

#[inline(always)]
fn pack_val_inline(symbols: &[TokenId]) -> Option<(u64, u64)> {
    match *symbols {
        [a] if a.0 < (1 << 24) => Some((1 | ((a.0 as u64) << 8), 0)),
        // ... two, three and four token cases
        _ => None,  // spill to the arena
    }
}
```

One dependent load gets you the entry, and the entry already contains the answer for 98 percent of lookups. No arena, no pointer chase, no second cache miss.

The 24-bit constraint on the first token id is the kind of assumption that deserves scrutiny, and the comment addresses it: 24 bits holds 16.7 million ids, and the largest production vocabularies are around 200,000, so the constraint is true of every real vocabulary with four orders of magnitude to spare. It is checked rather than assumed, and the `None` return routes any violation to the spill path, which is correct if slower.

The spill path itself was moved off the hot loop for a measured reason. The first profile found the `VAL_SPILL` branch alone burning 2.7 percent of total process samples, because it is an unpredictable branch taken per pretoken. The fix, from the round-one `probe-emit` work, was to emit the two-token inline pair unconditionally into a local buffer and advance the write cursor branchlessly by the length, deferring the rare arena case entirely. You write tokens you might not need, and it is faster than asking whether you need them. That change measured +6.2 percent multithreaded and +27.6 percent single-threaded when materializing tokens.

## 11. The TLB is a cache too

At around 64 MB, the pretoken table hits a limit most application programmers never think about.

Virtual memory is translated to physical memory through page tables, and because walking those tables on every access would be ruinous, the CPU caches recent translations in the translation lookaside buffer. The dTLB has on the order of a few thousand entries. With standard 4 KiB pages, a 64 MB table spans about 16,000 pages, so a random probe into it very likely misses the dTLB and triggers a page walk, on top of the DRAM access you were already paying for.

With 2 MiB huge pages, that same table spans 32 pages. The entire thing fits in a few dozen dTLB entries.

![Same allocation and same madvise call: only the ordering against first touch decides whether it does anything](/imgs/blogs/inside-gigatoken-simd-cache-hierarchies-1000x-faster-bpe-11.webp)

So the table is allocated 2 MiB-aligned and hinted with `MADV_HUGEPAGE`. The interesting part is the ordering, and the comment explaining it is the single most useful paragraph in the codebase:

```rust
fn new_zeroed(cap: usize) -> Self {
    let layout = Self::layout(cap);
    let raw = unsafe { alloc(layout) };
    let Some(ptr) = NonNull::new(raw as *mut Entry) else {
        handle_alloc_error(layout)
    };
    // Hint huge pages BEFORE first touch. `alloc_zeroed` on a 2 MiB-aligned
    // layout is aligned_alloc + an explicit memset that faults the whole
    // fresh mapping in as 4 KiB pages, after which the hint is a no-op for
    // this run (khugepaged collapses far too slowly to matter): the table
    // then walks the dTLB on every probe, and Zen drops software prefetches
    // that miss the dTLB. Measured +15% cold / +7% warm encode from this
    // ordering alone.
    super::madvise_hugepage(raw, layout.size());
    unsafe { std::ptr::write_bytes(raw, 0, layout.size()) };
    Self { ptr, cap }
}
```

Use `alloc_zeroed` and you lose 15 percent, because the zeroing happens inside the allocator, before you get a chance to hint, and it faults the mapping in as 4 KiB pages. Once faulted, the hint is dead for the lifetime of the process, since Linux's `khugepaged` collapses pages far too slowly to help a batch job. Allocate uninitialized, hint, then zero it yourself, and the zeroing write faults it in as 2 MiB pages. Same three operations, different order, 15 percent.

The trailing clause matters independently: Zen drops software prefetches that miss the dTLB. All the careful prefetching in the probe loop, staged one chunk ahead into L2 and a few probes ahead into L1, silently does nothing if the translation is not resident. Two optimizations that look unrelated in the source are coupled through the hardware.

There is a second-order bug here that is even better, and it lives in `madvise_hugepage` itself:

```rust
// madvise demands a page-aligned start, and Vec/malloc pointers to
// mmap-served allocations sit 16 bytes past the page boundary (the
// allocator header) — passed through raw, every Vec-backed call here
// returned EINVAL and the hint was silently dead (only the 2 MiB-
// aligned_alloc table pointer ever worked). Align inward.
const PAGE: usize = 4096;
let start = (ptr as usize + PAGE - 1) & !(PAGE - 1);
```

`madvise` requires a page-aligned address. A `Vec`'s data pointer sits 16 bytes past the page boundary because of the allocator header. So every `madvise` call on a `Vec`-backed buffer returned `EINVAL`, and since nobody checks the return value of an advisory hint, the optimization was silently absent everywhere except the one allocation that happened to be over-aligned. The fix is to round the start up, which is safe because trimming a sub-page head is harmless and the kernel flags whole memory regions anyway.

This is a failure mode worth internalizing. A hint that fails silently is worse than one that fails loudly, because it looks exactly like an optimization that did not help, and the natural response is to remove it.

## 12. The counter-intuitive verdict from the performance counters

The sampled profile tells you where time goes. It does not tell you why. For that the author went to the hardware performance counters, using Instruments' CPU Bottlenecks mode, and the answer redirected the entire optimization program.

Over the encode window, 41.8 billion cycles at a sustained 4.0 GHz:

| Component | Fraction of issue bandwidth |
| --- | ---: |
| Useful (retiring) | 54.9% |
| **Discarded (bad speculation, mispredicted paths)** | **25.1%** |
| Backend bottleneck (data dependencies, memory latency) | 11.4% |
| Frontend bottleneck (instruction delivery) | 7.4% |

![A quarter of the core's issue bandwidth was thrown away on mispredicted branches, not stalled on memory](/imgs/blogs/inside-gigatoken-simd-cache-hierarchies-1000x-faster-bpe-12.webp)

A quarter of the core's issue bandwidth was thrown away executing instructions on mispredicted paths. Memory latency, the thing everyone assumes is the bottleneck in a hash-table-heavy workload, was 11.4 percent. Instruments raised the "High Discarded" remark 151 times against 26 for "High Processing."

This is the opposite of what the code looked like it should be limited by. It has a 64 MB hash table, two billion random lookups, and a working set that blows every level of cache. Every instinct says memory-bound. The counters say the core was mostly busy running instructions it then had to throw away.

And the mispredicts were traceable to specific source constructs. The discarded-bandwidth weight by function put `fill_spans_keyed_with` at 17.7 percent, `MaskState::next_span` at 10.5 percent, `ShortPretokenCache::get` at 9.7 percent, and `memoized_encode` at 9.3 percent, concentrating in three per-pretoken branch ladders: the walker's `while rem != 0` bit-walk and segment-refill exits, the key packer's length and page-boundary branches, and the probe loop's hit-versus-spill-versus-miss triad.

Every one of those branches depends on the data. Is this pretoken 3 bytes or 11? Does this span cross a page boundary? Did this key hit? The branch predictor has no pattern to learn, because English text has no pattern to learn at that granularity, and it is wrong often enough to burn a quarter of the machine.

So the de-branching program that followed attacked exactly those three sites, and it is worth listing because each fix is a different technique:

- **The walker** became two-phase. Phase A extracts all span boundaries of a 64-byte block branchlessly into a buffer; phase B consumes them with a table-based branchless key pack.
- **The key pack and hash** replaced the dependent table load with arithmetic masks, and adopted a hardware CRC32 hash.
- **The probe and emit loop** emits 4-token inline entries unconditionally into a flat output buffer with a branchless length advance, uses paired probe compares, stages prefetches, and defers the rare spill path.
- **The miss path** replaced `hashbrown` pair-rank probes with a flat `PairRankTable`, seeded the short cache from the vocabulary so every short vocabulary word hits on first touch, and used a stack array plus NEON for short merges.

The final microarchitectural verdict on the walker is the part I find most satisfying, because it is a proof rather than an observation. After de-branching, three separate attempts to make the walker faster all measured zero or negative, and the round-five analysis explains all three at once by decomposing the loop:

Phase B, the emission loop, is issue-bound. Exactly 25 instructions per span with no waste, running at an instructions-per-cycle of 6.5 to 7 against the core's 8-wide issue width, which puts it within about 15 percent of the theoretical floor for its mandatory instruction stream. Phase A, the harvest, is chain-latency-bound: about 190 dynamic instructions per batch at 56 to 65 cycles, an IPC of about 3, because the per-batch critical chain runs load-pair, classify, weighted-and, four `addp` reductions, a vector-to-scalar `fmov`, scalar algebra, a SWAR multiply, a table load, and a store, for an irreducible roughly 50 cycles, with only weak cross-batch overlap.

Which explains why restructuring it did nothing, and why cutting instructions did nothing. Under predicted branches, the dynamic operation stream of a restructured loop is identical, and a latency-bound loop does not care how many instructions you remove from it. The report concludes that a hypothetical 128-byte walker would remove roughly 30 to 40 of about 380 instructions per 128 bytes, which is more instruction cuts against a chain-latency bound, and retires the idea permanently. Knowing when to stop optimizing, with evidence, is rarer than knowing how to start.

## 13. Scaling to 144 cores: orchestration was the bottleneck

Everything so far is single-threaded. The headline number is multithreaded, and the multithreading story has its own arc, in which the encoding itself was never the problem.

The first question is how you parallelize an 11.9 GB text file that has not been pre-split. You cannot cut it at arbitrary byte offsets, because a cut in the middle of a word changes that word's tokenization and your output stops matching the reference. So the library finds document boundaries itself:

![How a single flat corpus file becomes concurrent encoders and one flat token buffer](/imgs/blogs/inside-gigatoken-simd-cache-hierarchies-1000x-faster-bpe-13.webp)

```rust
/// Cut `bytes` into ranges of roughly `target` bytes, each ending on a
/// document boundary so no document spans two chunks. A single range is
/// returned when the input is smaller than `target` or has no boundaries
/// (plain text without a separator is one document, which cannot be split
/// without changing tokenization).
pub fn chunk_ranges(
    bytes: &[u8],
    format: &DocFormat,
    target: usize,
) -> Vec<std::ops::Range<usize>>
```

The chunking rules are tuned rather than arbitrary. Chunks are at least 1 MiB, because a chunk that size encodes for tens of milliseconds and makes worker acquisition and work-stealing overhead into noise. The target is total bytes divided by sixteen times the thread count, giving roughly sixteen chunks per thread for load balancing. Inputs that do not fill more than one chunk encode serially, because for small inputs the thread fan-out costs more than it saves.

Then the campaign began, and it is documented round by round in `profiling/campaign_report.md` with a methodology I would happily adopt wholesale: one technique per branch per worktree, strictly sequential interleaved A/B measurement because the benchmark varies plus or minus 8 percent run to run, token-identity gates before any performance measurement, and a rule that implementation agents were forbidden from benchmarking their own work so that all numbers come from one central harness on an idle machine. That last rule removes both machine contention and motivated measurement.

![Three traced multithread rounds removed the gather, the teardown and the straggler tail, never the encoding](/imgs/blogs/inside-gigatoken-simd-cache-hierarchies-1000x-faster-bpe-14.webp)

The first multithread trace decomposed a 1462 ms window: 71.9 percent steady-state encode, and then three orchestration losses. A 163 ms gather copy. A 114 ms serial free of about 9.1 GB of chunk buffers on the main thread. And a 104 ms straggler tail, because rayon's range splitting had turned the longest-processing-time-first chunk ordering into a hint rather than a guarantee, with a 78 MB head chunk observed starting after other threads had already reached the tail.

Fixing those exposed the next layer, which is the pattern that repeats. The round-three fix, a fused parallel copy-and-drop gather plus an atomic-counter strict in-order chunk handout, cut the straggler spread from 104 to 23 ms and gather-plus-free from 277 to 214 ms. The round-four trace then showed the fused gather running at only 7.2 of 16 threads busy, and diagnosed a classic reader/writer convoy: first-touch zero-fill faults on the 9.1 GB output buffer take the virtual memory map read lock, while the interleaved chunk-buffer unmaps take the write lock at about 0.7 ms each, 307 times, each one stalling every concurrently faulting thread. Two hundred and five milliseconds of actual teardown work was costing about 1.4 seconds of blocked thread time.

Deferring the teardown to a detached background task removed every unmap from the timed window and took threads-busy from 7.2 to 12.2 of 16. Which exposed the next layer: with the unmaps gone, the gather's CPU time nearly doubled, from 1123 to 2079 ms, even as its wall time shrank. The threads had previously been sleeping on a lock; now sixteen of them fault concurrently on a 9.1 GB buffer and the cost converts into in-kernel fault-path contention.

The round-five fix folds the gather into the encode phase entirely. Reserve the output buffer up front at a strict upper bound, using the observation that one token consumes at least one input byte, so `total_bytes` tokens is always enough and untouched pages cost only virtual address space. Then a worker that finishes a chunk try-locks a commit cursor and copies its ready prefix at exact offsets while other chunks are still encoding. No dedicated committer thread; a worker that fails the try-lock goes back to encoding. Single-chunk inputs skip the gather copy altogether by returning the chunk's buffer directly.

Final: 8792 MB/s multithreaded on the M4 Max, against 5538 for the previous release and 2826 for the original baseline. A 3.11x improvement, essentially all of it in orchestration rather than encoding.

## 14. Cold caches, sixteen times over

There is one multithread cost that orchestration cannot remove, and it falls directly out of the Zipf story from section 8.

Each worker keeps its own pretoken cache. That is the right call, because sharing one table across threads would mean synchronization on the hottest data structure in the program, and the coherence traffic on a table that is written on every miss would be brutal. But it means each worker independently rediscovers the head of the Zipf distribution. The word ` the` gets merged from scratch once per worker.

The report quantifies it: encode CPU is about 14.7 seconds multithreaded against about 11 seconds single-threaded for the same input, and the sum of per-worker distinct pretokens is roughly 16 million against 5.5 million single-threaded. Sixteen workers each pay the miss path until their table warms, and the aggregate work is nearly three times what one thread would do.

Two mitigations ship. The first is vocabulary seeding: the short cache is pre-populated from the tokenizer's own vocabulary, so every short vocabulary word hits on first touch rather than being merged once per worker. The second is pre-sizing, and this is where the Heaps' law fit earns its keep:

```rust
/// `fork` with the caches pre-sized for a worker expected to encode roughly
/// `expected_bytes` of input. On a cold parallel run a default-sized worker
/// rehashes its pretoken table through 6-7 doublings, random scatter writes
/// into a fresh zeroed allocation each time, on every worker at once; sizing
/// from the input share pays for the table exactly once.
pub(crate) fn fork_sized(&self, expected_bytes: usize) -> Self {
    // Distinct short pretokens follow Heaps' law: ~1.3M at 1 GB and
    // ~5.5M at 10 GB of OWT-like text gives distinct(n) ~ 3.45 n^0.62.
```

Without this, sixteen workers each perform six or seven rehashes, and each rehash is a full random-scatter rewrite of a growing table into a freshly zeroed allocation, all sixteen at once, all competing for the same memory bandwidth. With it, each table is constructed exactly once at the size it will actually need. Round four measured the fork ramp dropping from 32 to 19 ms from this plus halving 2 GB of zeroing.

The remaining gap is acknowledged as open. The report's stated candidate is a shared warm-head structure, holding just the Zipf head that every worker needs, with the usual coherence-traffic risk. There is also a rejected experiment in the ledger, `opt/split-table`, at minus 22 percent multithreaded and minus 32 percent single-threaded, which suggests splitting the table is not the answer.

## 15. What the 1000x is, and what it is not

Now the honest accounting, because a 989x headline deserves scrutiny and the underlying data supports a more textured story.

![The same library, from 7.3x to 1268x depending on machine and tokenizer family](/imgs/blogs/inside-gigatoken-simd-cache-hierarchies-1000x-faster-bpe-15.webp)

Start with the ratios that Gigatoken measures against itself, which are the cleanest numbers available because the same harness measures both sides on the same machine:

- Pretokenization alone, single-threaded: 47 MiB/s with `fancy-regex` to 1,049 MiB/s. **22.3x.**
- Full encode, single-threaded, across the optimization campaign: 605 MB/s counting-only to 1,039 MB/s materializing every token. Comparing like for like, the mainline materializes all 2.28 billion tokens faster than its predecessor merely counted them.
- Full encode, multithreaded, across the campaign: 2,826 to 8,792 MB/s. **3.11x.**

Those are the parts attributable to the code, and they are excellent. Now the cross-library comparison, where things get more interesting. Here is HuggingFace `tokenizers` on GPT-2, the same benchmark, across three machines:

| Machine | Cores | gigatoken | HF tokenizers | Ratio |
| --- | ---: | ---: | ---: | ---: |
| AMD EPYC 9565 | 144 | 24.53 GB/s | 24.8 MB/s | 989x |
| Apple M4 Max | 16 | 8.79 GB/s | 6.9 MB/s | 1268x |
| AMD Ryzen 9800X3D | 16 | 6.27 GB/s | 59.0 MB/s | 106x |

Look at the HuggingFace column rather than the ratio column. It reads 24.8, 6.9, 59.0. HuggingFace is more than twice as fast on a 16-core desktop Ryzen as on a 144-core dual-socket EPYC, and nearly nine times faster there than on the M4 Max. The ratio is not tracking Gigatoken's performance; it is tracking how badly the baseline behaves on that particular machine.

Compare the SentencePiece rows, where HuggingFace does not have this problem. Gemma 1, same three machines: 342.2, 85.7, 84.9 MB/s. There the EPYC is four times faster than the 16-core machines, which is what you would expect from a library that scales. So HuggingFace's `encode_batch_fast` BPE path behaves anomalously on high-core-count and Apple Silicon machines in these runs, while its SentencePiece path does not.

I cannot tell you from the published data whether that is thread oversubscription, NUMA effects across two sockets, allocator contention, or something in the benchmark harness. What I can tell you is that the honest reading of the 989x is: a large real speedup, multiplied by a baseline having a bad day on that specific machine. The 106x on the Ryzen, where HuggingFace behaves normally, is the number I would quote to a colleague, and it is still a two-orders-of-magnitude win that would be the headline of most projects.

The range across the full published table runs from 7.3x, for Gemma 1's SentencePiece tokenizer on the EPYC, to 1268x. Any single number pulled from that spread is a choice about which cell to quote.

None of this makes the engineering less impressive. It makes the claim more legible: Gigatoken is roughly 20x faster at pretokenization through better algorithms, several times faster again through caching that its competitors do not do at all, and scales cleanly to many cores where at least one competitor does not. Those three multiply, and on an unlucky baseline configuration they multiply to about a thousand.

## 16. Where Gigatoken does not win

The README's Known Issues section is unusually candid, and reading the limitations tells you more about the mechanism than the wins do.

**SentencePiece is 4 to 8 times slower than the BPE path** in absolute terms, at 2.5 to 4.8 GB/s against 19 to 25. The author calls it "not nearly as optimized" and explicitly deprioritizes it, since mostly Google and BERT-style models use it. But notice what this tells you: SentencePiece tokenizers do not decompose into the same fixed pretokenization rule, so the entire section-4-and-5 mechanism does not apply cleanly. The cache still works, which is why it is still 7 to 21 times faster than HuggingFace, but the SIMD pretokenizer, the part that owns half the profile, has much less to bite on.

**CJK-heavy data is much slower**, and the stated reason is that "most pretokenizers make caching a challenge in this setting." This is the Zipf argument running in reverse. Chinese, Japanese and Korean text has different pretoken boundary behavior and a different distribution over pretokens, so the 99.4 percent hit rate that carries the English benchmark does not hold, and more lookups fall through to the merge loop. It is an active work item, not a fundamental limit, but if your corpus is majority CJK you should benchmark rather than assume.

**WordPiece is not supported at all.** If you are tokenizing for BERT-family models, this library has nothing for you today.

**Python iteration carries avoidable overhead.** The bindings use ABI3, the stable Python ABI, which is slower than version-specific CPython APIs. The author reports early experiments showing a 2x improvement for overhead-bound cases from specializing per Python version. This is a real tradeoff rather than an oversight: ABI3 means one wheel works across Python 3.10 through 3.14, which is why installation is a single `pip install` with no compilation.

**File sinks are not implemented** in the native API, so you can read files at full speed but you cannot yet write token output straight back to disk from Rust.

**Windows is barely tested**, with WSL recommended.

And one caveat the README does not list but which follows directly from the design: because the cache is what produces the extreme numbers, throughput is now a function of your data's redundancy. A corpus of near-duplicate documents will fly. A corpus of high-entropy text, random identifiers, base64 blobs, code with unique hashes, will run much closer to the uncached speed. If you are planning a pipeline around a throughput figure, measure on your data.

## 17. What this changes if you build training data

Tokenization has always sat in the annoying middle of the data pipeline: not slow enough to be the thing you fix, not fast enough to ignore. Moving it by two orders of magnitude changes which experiments are affordable.

The concrete framing from the README: at the EPYC's 24.53 GB/s, you could tokenize all of Common Crawl, roughly 130 trillion tokens by [one widely cited estimate](https://arxiv.org/pdf/2211.04325), in just under 6.5 hours. A 500 GB corpus goes from something you launch and come back to tomorrow to something that finishes during a coffee break.

The second-order effects are where it matters:

**Tokenizer ablations become cheap.** Choosing a vocabulary is a real decision with real downstream consequences, covered in [designing and choosing a tokenizer](/blog/machine-learning/large-language-model/designing-choosing-tokenizer-llm). Historically you made that decision once, early, on a sample, because re-tokenizing the corpus to compare candidates cost days. At GB/s you can tokenize the full corpus under four candidate vocabularies and compare actual compression ratios and actual sequence-length distributions on real data.

**Data mixture iteration gets faster.** Every change to filtering, deduplication or domain weighting invalidates your tokenized artifacts. When re-tokenization is nearly free, the tokenized form stops being a precious cached artifact you are reluctant to invalidate.

**Token counting becomes a routine query.** Wanting to know how many tokens a candidate data source contributes, per domain, per language, is a natural question that was previously expensive enough to discourage asking.

Here is the shape of a real pipeline using the fast path:

```python
import numpy as np
import awkward as ak
import gigatoken as gt

tokenizer = gt.Tokenizer("Qwen/Qwen3-8B")

# The fastest path: Rust reads, splits and encodes the files in parallel.
# No text passes through Python at all.
source = gt.TextFileSource(
    ["/data/corpus/shard-0000.txt", "/data/corpus/shard-0001.txt"],
    separator="<|endoftext|>",
)
tokens = tokenizer.encode_files(source)   # ragged awkward Array

print(f"{len(tokens)} documents, {ak.sum(ak.num(tokens))} tokens")

# JSON Lines works the same way, taking text from a named field.
tokens = tokenizer.encode_files(
    gt.JsonlFileSource(["/data/corpus/docs.jsonl"], field="text")
)

# Flatten to a single contiguous uint32 stream for a memmap-backed loader.
flat = np.asarray(ak.flatten(tokens)).astype(np.uint32)
flat.tofile("/data/tokenized/qwen3-shard-0000.bin")
```

The important line is `encode_files`. Passing a list of Python strings makes Rust cross the Python boundary per document, and the README is explicit that this "still incurs the overhead of reading from Python." Handing over file paths, or a `BytesSource` with a separator for in-memory data, lets Rust do the reading, the splitting and the parallelizing. That is the difference between the compatibility-mode speedup and the headline speedup.

## 18. Ten incidents from the optimization campaign

The campaign report reads like an incident log, and the incidents generalize well beyond tokenization.

**1. The streaming-gather committer thread, minus 5.9 percent.** Round one bundled four multithread changes into one branch, `opt/mt-alloc`: pre-sized worker caches, a streaming gather, longest-processing-time chunk ordering, and `Arc`-shared immutable tables. As a unit it regressed 5.9 percent and was rejected. Rather than discarding the work, the author bisected it with environment-variable kill switches and found the streaming-gather committer thread was solely responsible. The other three components were salvaged into `opt/mt-salvage` and merged. The lesson is procedural: bundling four changes into one branch cost a round, and the only reason it was recoverable was that each component could be independently disabled at runtime.

**2. The virtual-memory-map convoy.** The round-four trace showed the fused gather using 7.2 of 16 threads. The cause was lock asymmetry rather than anything in the gather's own code: page faults on the output buffer take the vm-map read lock, unmapping the finished chunk buffers takes the write lock, and 307 unmaps at roughly 0.7 ms each stalled every faulting thread. Two hundred and five milliseconds of teardown work cost 1.4 seconds of blocked thread time. Deferring the teardown outside the measured window fixed it. Freeing memory is not free, and on a multithreaded fault-heavy workload it can cost seven times what it appears to.

**3. The fix that made the profile look worse.** After deferring teardown, the gather's CPU time nearly doubled, from 1123 to 2079 ms, while its wall time shrank. Nothing regressed. The threads had previously been asleep on a lock, invisible to CPU accounting, and removing the lock converted that invisible waiting into visible in-kernel fault-path work. If you measure only CPU time you would have reverted a change that made the program faster.

**4. The silently dead `madvise`.** Every `MADV_HUGEPAGE` hint on a `Vec`-backed buffer returned `EINVAL`, because `madvise` requires page alignment and a `Vec` data pointer sits 16 bytes past the boundary. Only the one over-aligned allocation ever worked. Nobody checks the return value of an advisory hint, so the optimization was absent for an unknown period while appearing present in the source.

**5. The ordering bug worth 15 percent.** Allocating with `alloc_zeroed` and then hinting huge pages is a no-op, because the allocator's internal memset has already faulted the mapping in as 4 KiB pages and `khugepaged` will not collapse them in time. Allocate uninitialized, hint, then zero it yourself. Fifteen percent cold, seven percent warm, from three operations in a different order.

**6. The token-identity bug that GPT-2 could not see.** The round-one vocabulary-seeding optimization was correct for GPT-2, OLMo 3, Qwen 2 and DeepSeek V3, and wrong for Qwen 3.5, which has 201 vocabulary entries unreachable through merges. It passed every gate the campaign had, because every gate was GPT-2-based. It was caught only when the verification round extended the differentials to multiple tokenizers and the full corpus. A correctness gate that runs on one configuration is a correctness gate for that configuration.

**7. The adversarial review that found what the gates could not.** After round four, five parallel review agents audited the roughly 2,300-line diff through five different lenses: unsafe and memory, portability, API soundness, and others. It found an API soundness regression, a big-endian portability gap, and an added-token edge case, with hot-path code generation verified bit-identical afterward. Note the division of labor: the differentials caught the data-dependent bug that reading could not, and the review caught the soundness issues that testing did not cover.

**8. `opt/emit3`, minus 4 to minus 7 percent.** Two variants of the same idea, staging token stores into an L1 buffer then flushing with `memcpy`, and adding store prefetches. Both lost. The emit loop's stores were not the bottleneck, so adding a staging hop and extra prefetch traffic was pure cost. Optimizing a dimension the core has slack in is not neutral; it is negative.

**9. `opt/walker4`, zero percent despite cutting instructions by 22 percent.** A dual-nibble table-lookup classify plus assembly pins cut the walker from 113 to 88 instructions per 64-byte batch and changed nothing. This is the clearest possible demonstration that instruction count is not a performance metric on a latency-bound loop. The dependency-structure analysis in round five explains it: phase A is bound by a roughly 50-cycle critical chain, and removing instructions that were already overlapping with that chain removes nothing.

**10. `opt/split-table`, minus 22 percent multithreaded, minus 32 percent single-threaded.** Splitting the pretoken cache was tried and lost badly. It stays in the ledger with the numbers, which is the point of keeping rejected branches: the next person to have that idea can read why it did not work instead of spending a week rediscovering it.

Two meta-observations from the campaign. First, of roughly 17 techniques attempted, 11 merged and 6 were rejected or held, which is a healthy hit rate for genuine optimization work and a strong argument for cheap, isolated, measurable experiments. Second, the author's AI use disclosure states that most of the codebase was hand-written, with AI assistance in the later stages for the user-facing API, porting SIMD strategies between AVX-512, AVX2 and NEON, widening tokenizer compatibility, and the final roughly 4x from branch elimination and cache-hierarchy work. Disclosing that with specificity, rather than either hiding it or waving at it, is a standard more projects should meet.

## 19. When to reach for Gigatoken, and when not

**Reach for it when you are tokenizing a corpus.** Pretraining data preparation, dataset ablations, re-tokenizing under a new vocabulary, token accounting across sources. This is the design center and where the numbers are real. Use `encode_files` with a `TextFileSource`, `JsonlFileSource` or `ParquetFileSource` so that Rust owns the reading and splitting, and no text crosses the Python boundary.

**Reach for it when you want a drop-in with no code changes.** Both compatibility modes preserve exact output:

```python
import gigatoken as gt
from transformers import AutoTokenizer

hf_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
tokenizer = gt.Tokenizer(hf_tokenizer).as_hf()

# Use it exactly where the transformers tokenizer was used before.
out = tokenizer(["This is a test string", "Here is another"],
                return_tensors="np", padding=True)

# And the tiktoken shape, for code written against that API.
encoding = gt.Tokenizer("openai-community/gpt2").as_tiktoken()
ids = encoding.encode("Tokenize your text data at GB/s!")
```

The README is straightforward that compatibility mode costs performance: exact HuggingFace parity requires reproducing behavior that the native path skips, so you get a large speedup but not the headline one. That is the correct tradeoff for existing code, and the native API is there when a pipeline is worth rewriting.

**Do not bother for inference-time prompt tokenization.** Tokenizing a 2,000-token prompt takes microseconds under any implementation. The bottleneck is the model, and a tokenizer that is a thousand times faster on a workload that was never measurable does nothing for your latency.

**Do not assume the numbers transfer to your data.** Benchmark if your corpus is CJK-heavy, if it is dominated by high-entropy text such as code with unique identifiers or encoded blobs, or if you use a SentencePiece tokenizer. The cache is the mechanism, and redundancy is what feeds it.

**Check your tokenizer is supported before planning around it**, which takes one command and no installation:

```bash
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz

uvx --with tokenizers gigatoken bench 'openai-community/gpt2' owt_train.txt \
    --validate --doc-separator "<|endoftext|>"
```

That runs both implementations, prints throughput for each, and validates that the token ids match document for document. The `--validate` flag is the part I would insist on: a tokenizer that is fast and subtly different is worse than one that is slow, because the failure shows up as degraded model quality weeks later rather than as an error. On macOS, run it twice; the first run pays for a security scan that slows the Rust binary.

The deeper reason to read this codebase, though, has little to do with tokenization. It is one of the clearest worked examples I know of a specific methodology: profile the whole process rather than the function you suspect, use hardware counters to find out *why* rather than *where*, distrust your instinct about which resource is scarce, verify correctness before you measure speed, and keep the experiments that failed along with the numbers that killed them. The 22.3x on pretokenization came from knowing that a fixed rule does not need a general engine. The rest came from being willing to be wrong in public, sixteen times, with receipts.

If you want to go deeper on the Rust-and-Python boundary that makes this shippable as a single `pip install`, [PyO3 and maturin](/blog/software-development/python-performance/rust-for-python-pyo3-and-maturin) covers the packaging side of exactly this pattern.
