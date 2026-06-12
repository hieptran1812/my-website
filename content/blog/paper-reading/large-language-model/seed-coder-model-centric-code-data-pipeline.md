---
title: "Seed-Coder: letting the code model curate its own training data"
date: "2026-06-12"
publishDate: "2026-06-12"
description: "How ByteDance Seed's 8B Seed-Coder replaces 130+ hand-written filtering rules with an LLM-as-judge data pipeline — and beats Qwen2.5-Coder and OpenCoder at the same scale."
tags: ["seed-coder", "code-llm", "data-curation", "pretraining-data", "model-centric-data", "open-weights", "code-generation", "bytedance-seed", "reasoning-models", "fill-in-the-middle"]
category: "paper-reading"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 50
---

Spend a week reading the data sections of open code-LLM papers and a pattern jumps out: nobody actually talks about the model. They talk about *rules*. StarCoder published a filtering recipe that keys on file size, line length, the ratio of alphanumeric characters, and the percentage of lines that look like comments. DeepSeek-Coder layered repository-level dependency parsing on top of a comparable rule set. OpenCoder went furthest and shipped **more than 130 hand-crafted filtering rules with custom per-language weights** — a small expert system whose only job is to decide which `.py` files are worth training on. Every one of those rules encodes a human belief about what good code looks like, and every one of them is wrong somewhere.

Seed-Coder (ByteDance Seed, [arXiv:2506.03524](https://arxiv.org/abs/2506.03524), June 2025) takes the opposite bet. Its thesis fits on a bumper sticker — *let the code model curate data for itself* — and the engineering underneath it is the most interesting part. Instead of asking a committee of engineers to write down what clean code is, Seed-Coder asks an LLM to **read each file and score it**, the same way a senior reviewer would in a pull request. The rule set shrinks from 130-plus heuristics to a tiny handful of structural filters (dedup, syntax-check, language-ID), and an LLM-as-judge does the rest at a scale of billions of files. The result is a family of three 8B models — Base, Instruct, and Reasoning — released under the **MIT license**, trained on roughly **6 trillion tokens**, that posts state-of-the-art numbers among open models at the ~8B scale and reaches up to beat several larger ones.

![Rule-based heuristic filtering versus LLM-as-judge model-centric filtering for code pretraining data](/imgs/blogs/seed-coder-model-centric-code-data-pipeline-1.png)

<!-- FIGSPEC 1
kind: before-after
claim: Rule-based heuristics drop good code and keep broken code while an LLM judge scores semantics across billions of files.
caption: The core swap: 130+ brittle heuristics out, a single LLM-as-judge scorer in.
nodes:
  - id: r1 | label: "RULE: file-size cap" | color: amber
  - id: r2 | label: "RULE: comment-ratio" | color: amber
  - id: r3 | label: "RULE: alnum-ratio drops Pikachu LED art" | color: red
  - id: r4 | label: "RULE: stars / license" | color: amber
  - id: r5 | label: "130+ rules, per-lang weights" | color: red
  - id: m1 | label: "LLM judge: readability" | color: blue
  - id: m2 | label: "LLM judge: correctness catches if t<0 and t>1" | color: green
  - id: m3 | label: "LLM judge: modularity" | color: blue
  - id: m4 | label: "LLM judge: self-contained" | color: blue
  - id: m5 | label: "1.3B distilled scorer, billions of files" | color: green
notes: left column = brittle rules (amber/red), right column = LLM judge dimensions (blue/green); two stacked columns, vertical
-->

The diagram above is the mental model for the whole paper. On the left, a stack of human-written rules that are individually reasonable and collectively brittle: a numeric-ratio filter that throws away a perfectly good Python script because it draws a Pikachu on an LED matrix (lots of digits), a comment-ratio filter that can't tell a docstring from dead code, a star-count filter that conflates popularity with quality. On the right, an LLM that reads the file and notices the thing no rule will ever catch — `if temp < 0 and temp > 1:` is a logically impossible condition, so the file is broken no matter how clean it looks. The rest of this post is a tour of how Seed-Coder turns that single idea into 6T tokens of curated corpus, three trained models, and a benchmark sheet that embarrasses some 14B competitors.

## 1. Why "model-centric" is the actual contribution

Let me be blunt about what is and isn't novel here. The 8B dense transformer is a Llama-3-shaped model; there is nothing architecturally surprising about it. The SFT-then-DPO recipe for the Instruct model is standard. The GRPO long-CoT RL for the Reasoning model is a well-trodden path by mid-2025. **The contribution is the data pipeline**, and specifically the claim that you should stop writing rules and start trusting a model to read.

Why does this matter? Because data filtering is where most of the quality of a code LLM is actually decided, and it is also where the field has been quietly stuck. The standard pipeline — used by StarCoder, StarCoder2, DeepSeek-Coder, CodeLlama, and OpenCoder in various combinations — is a sieve of cheap, surface-level signals:

| Heuristic signal | What it's trying to proxy | How it fails |
|---|---|---|
| File size / line length caps | "Not autogenerated, not a blob" | Drops legitimately large modules; keeps short broken files |
| Alphanumeric / numeric ratio | "Real code, not data dumps" | Drops LED-art, lookup tables, embeddings, valid hex constants |
| Comment-line percentage | "Documented, human-written" | Can't distinguish docstrings from commented-out dead code |
| GitHub stars / forks | "Community-validated quality" | Rewards popularity, not correctness; new good code has zero stars |
| License allow-list | Legal safety | Necessary, but says nothing about quality |
| Per-language custom weights | "Tune strength per ecosystem" | 130+ knobs that nobody can globally reason about |

The Seed-Coder authors make a sharp argument against this whole edifice: hand-crafted rules "lack flexibility in adjusting filtering strength" and rely on "subjective consensus among a small group of experts." They go further and say the human-centric approach, intuitive as it is to programmers, "ultimately impedes the advancement of code LLMs in the long run." That is a strong claim and they back it with an ablation we'll get to in section 7.

> The rule-based pipeline optimizes for what is *easy to measure* about code. The model-centric pipeline optimizes for what actually *matters* about code. Those two sets overlap less than you'd hope.

The mechanism that makes the swap possible is **distillation of judgment**. You cannot run a frontier LLM over billions of files; the inference bill is absurd. So Seed-Coder uses a strong oracle (DeepSeek-V2-Chat) to score a sample of files, then trains a tiny **1.3B Llama-2-shaped regression model** to imitate those scores, and runs *that* over the corpus. The expensive judgment happens once, on a sample; the cheap imitation of that judgment happens at scale. This is the same trick reward-model distillation uses in RLHF, applied to data curation. If you've read the [DeepSeekMath data pipeline writeup](/blog/machine-learning/training-techniques/deepseekmath-data-pipeline-and-grpo-origin), the iterative-recall pattern there is a cousin of this — a model deciding what's worth keeping — but Seed-Coder pushes it from "find more in-domain documents" to "grade every document on a quality rubric."

The rest of the model family rides on top of that corpus. Get the data right and an 8B model trained on 6T tokens of it can punch above its weight. Get it wrong and no amount of post-training rescues you. So we start where the paper starts: the four kinds of code data and how each one is built.

## 2. The four data categories

Seed-Coder's pretraining corpus decomposes into four sources, and the per-category engineering differs enough that it's worth treating them separately. The headline is **~6 trillion tokens total**: roughly 5T in regular pretraining and ~1T in continued pretraining, drawn from these wells.

![Tree of the four Seed-Coder data categories with token scale per branch](/imgs/blogs/seed-coder-model-centric-code-data-pipeline-2.png)

<!-- FIGSPEC 2
kind: tree
claim: Four data sources feed the 6T-token corpus with very different scales and construction methods per branch.
caption: File-level and web data dominate the token budget; commits and repo-level add structure.
nodes:
  - id: root | label: "Seed-Coder corpus ~6T tokens" | color: blue
  - id: file | label: "File-level GitHub ~1T unique tokens" | color: green
  - id: repo | label: "Repo-level ~1T (long-context)" | color: blue
  - id: commit | label: "Commits ~100B tokens, 74M commits" | color: amber
  - id: web | label: "Code-related web ~1.2T tokens" | color: blue
  - id: f1 | label: "89 languages, bottom 10% dropped" | color: gray
  - id: r1 | label: "Topological dep concat" | color: gray
  - id: c1 | label: "140K repos, BM25 ctx" | color: gray
  - id: w1 | label: "Common Crawl extract" | color: gray
edges:
  - root -> file
  - root -> repo
  - root -> commit
  - root -> web
  - file -> f1
  - repo -> r1
  - commit -> c1
  - web -> w1
notes: root at top, four category branches, each with one detail leaf; vertical tree
-->

**File-level code** is the bread and butter: individual source files scraped from GitHub. After preprocessing reduces the raw GitHub volume by roughly 98%, what's left is **~1 trillion unique tokens** spanning **89 programming languages**. This is the category where the LLM quality filter does its most consequential work — the bottom ~10% of files by quality score get dropped before training. We'll dissect that filter in section 4.

**Repository-level code** is the same GitHub source, but instead of treating each file as an independent unit, entire repositories are stitched into single long sequences so the model can learn cross-file structure: how a function defined in `utils.py` gets called in `train.py`, how a class hierarchy spans modules. The construction is dependency-aware. For mainstream languages (Python, Java, C), files are concatenated in **topological order based on import/dependency edges** so that a symbol's definition precedes its uses. For languages where dependency graphs are messier — HTML, SQL, Shell — files are concatenated in random order. Each repository becomes one string; gigantic repos like PyTorch get decomposed into multiple independent dependency subgraphs so a single sequence doesn't blow past the context window. This data is the backbone of the continued-pretraining long-context phase, contributing on the order of **1T training tokens** alongside the file-level long-context data.

**Commits data** is the most underused signal in code pretraining and Seed-Coder leans into it hard. They collect **74 million commits from 140K high-quality repositories**, where "high-quality" here is one of the few places a rule survives: a repo qualifies if it has **≥100 stars, ≥10 forks, ≥100 commits, and ≥100 days of maintenance activity**. Each commit is reframed as a **code-change prediction task**: given a commit message and surrounding context, predict which files change and what the diff is. The context window for each sample includes the pre-commit README, the directory structure, and the **top-5 relevant files retrieved via BM25**. This yields **~100 billion tokens** of commit data and directly teaches the model the SWE-bench-shaped skill of "here's an issue, here's a repo, produce the patch."

**Code-related web data** is the largest single slice: **~1.2 trillion tokens** mined from Common Crawl. This is the documentation, the StackOverflow answers, the tutorials, the blog posts with embedded snippets — the natural-language scaffolding that turns a model that can autocomplete syntax into one that can follow an instruction like "write me a rate limiter." Extracting clean code-bearing text from web HTML is its own filtering problem, also handled by model-based scoring rather than rules.

Notice the shape of the budget. File-level and web data dominate (~1T and ~1.2T), repo-level adds the long-context structure (~1T), and commits are comparatively small (~100B) but disproportionately valuable for software-engineering tasks. The corpus is not "all of GitHub"; it's a deliberately weighted blend where the weights reflect what each source teaches.

## 3. The preprocessing pipeline before the model ever sees a file

The LLM judge is the star, but it sits at the *end* of a preprocessing chain that does the cheap, unambiguous filtering first. You do not want to spend a 1.3B forward pass scoring a file that is a byte-for-byte duplicate of one you already scored, or that doesn't parse. So the pipeline is ordered cheapest-first.

![Graph of the Seed-Coder data pipeline from raw GitHub to curated corpus](/imgs/blogs/seed-coder-model-centric-code-data-pipeline-3.png)

<!-- FIGSPEC 3
kind: graph
claim: The pipeline applies cheap structural filters first and the expensive LLM quality score last to cut cost.
caption: Raw GitHub shrinks ~98% through dedup, syntax, and language filters before the LLM scores what remains.
nodes:
  - id: raw | label: "Raw GitHub repos" | color: gray
  - id: exact | label: "Exact dedup SHA256" | color: blue
  - id: near | label: "Near dedup MinHash" | color: blue
  - id: syntax | label: "Tree-sitter syntax check" | color: blue
  - id: lang | label: "Language ID, minimal rules" | color: blue
  - id: score | label: "LLM quality score 0-10" | color: green
  - id: thresh | label: "Drop bottom ~10%" | color: amber
  - id: corpus | label: "~1T unique tokens" | color: green
edges:
  - raw -> exact
  - raw -> near
  - raw -> syntax
  - exact -> lang
  - near -> lang
  - syntax -> lang
  - lang -> score
  - score -> thresh
  - thresh -> corpus
notes: first layer has 3 parallel structural filters (exact/near/syntax) feeding language-ID, then LLM score, threshold, corpus; left-to-right with a 3-wide layer
-->

The order matters and the reduction is dramatic — **~98% of raw volume is discarded** before training. Step by step:

1. **Exact deduplication** via SHA256 hashing of file contents, at both file and repository level. GitHub is full of forks, vendored copies, and mirror repos; exact dedup removes the literal copies first because it's nearly free.
2. **Near-deduplication** via MinHash with locality-sensitive hashing. This catches the not-quite-identical copies — files that differ by a license header, a renamed variable, a reformatting pass. Near-dup removal is the difference between a model memorizing a popular utility function 4,000 times and seeing it a handful of times.
3. **Syntax validation** with Tree-sitter parsers. Files that don't parse in their declared language are discarded outright. This is a strong, cheap correctness signal: a `.py` file that throws a `SyntaxError` teaches the model nothing good.
4. **Language identification and minimal rule filters** to strip out irrelevant and non-code data — the stuff that slipped through, files with the wrong extension, binary-ish blobs. This is the *entire* surviving rule set, and it's deliberately tiny.
5. **LLM quality scoring** on what remains, dropping the **bottom ~10%** by predicted quality.

Only after the first four steps — which are deterministic, parallelizable, and cheap — does the expensive model-based scoring run. This ordering is the whole reason model-centric filtering is economically viable at trillion-token scale. If you scored first and deduped later you'd waste an enormous fraction of your inference budget grading duplicates.

There's a subtle design lesson here for anyone building their own pipeline: **keep the rules you can't argue with, kill the rules you can.** SHA256 dedup, MinHash, and syntax-parse are not "subjective consensus among experts" — they're objective facts about the data. A file either parses or it doesn't; it either is or isn't a duplicate. Those rules stay. It's the *quality* judgments — readability, correctness, modularity — that humans encode badly and that you should hand to a model. Seed-Coder draws that line cleanly.

## 4. The LLM filter: what a model judge actually scores

This is the heart of the paper, so we go slow. The question is: how do you turn "is this good code?" — a fuzzy, holistic judgment — into a number you can threshold at scale?

The answer is a two-stage build. First, an **oracle**: DeepSeek-V2-Chat is prompted to read a code file and assign an overall quality score from **0 to 10**, with a written justification, evaluated against four named dimensions. Second, a **distilled scorer**: the oracle's scores on **222,066 sampled files** (across the most common languages) become training labels, rescaled to `[0, 1]`, and a 1.3B Llama-2-architecture model with a regression head is fine-tuned for one epoch to predict them. That 1.3B scorer is what actually runs over the billions of files.

![Grid of the four LLM-filter scoring dimensions with what each rewards and penalizes](/imgs/blogs/seed-coder-model-centric-code-data-pipeline-4.png)

<!-- FIGSPEC 4
kind: grid
claim: The judge rates four orthogonal dimensions that no surface heuristic can capture, each with a concrete reward and penalty.
caption: Readability, correctness/clarity, modularity, and reusability/self-containedness, scored 0-10 then distilled.
nodes:
  - id: h1 | label: "Dimension" | color: gray
  - id: h2 | label: "Rewards" | color: gray
  - id: h3 | label: "Penalizes" | color: gray
  - id: d1 | label: "Readability" | color: blue
  - id: d1a | label: "clear names, docstrings, format" | color: green
  - id: d1b | label: "cryptic 1-letter vars" | color: red
  - id: d2 | label: "Correctness / Clarity" | color: blue
  - id: d2a | label: "valid logic, explicit intent" | color: green
  - id: d2b | label: "if t<0 and t>1 dead code" | color: red
  - id: d3 | label: "Modularity" | color: blue
  - id: d3a | label: "small focused functions" | color: green
  - id: d3b | label: "500-line god-function" | color: red
  - id: d4 | label: "Reusability / self-contained" | color: blue
  - id: d4a | label: "few deps, no hard-coded paths" | color: green
  - id: d4b | label: "env-specific magic constants" | color: red
notes: 4 rows x 3 cols table; header row gray; first col blue dimension names, second col green rewards, third col red penalties
-->

The four dimensions, as the paper describes them:

- **Readability** — does the code use clear naming conventions, sensible comments, and consistent formatting? A file with single-letter variables and no structure scores low even if it runs.
- **Modularity** — is the code organized into focused, composable functions, or is it one 500-line procedure that does everything? The judge rewards structure that a human would find maintainable.
- **Clarity** — is intent explicit and redundancy minimized? This is where logical errors get caught: the `if temp < 0 and temp > 1` example is a clarity/correctness failure because the condition can never be true, and the judge flags it where no rule could.
- **Reusability** — could this code be lifted into another project? The paper specifically notes the judge penalizes "excessive hard-coded data" and rewards code "free of syntax and logical errors." A script hard-coded to `/home/alice/data/run37/` is less reusable than one that takes a path argument.

These four are doing real work precisely because **they are uncorrelated with the surface features rules measure.** A file can be short (passes size rules) and unreadable. A file can be heavily commented (passes comment-ratio rules) and logically broken. A file can have 10,000 stars (passes popularity rules) and be a tangled god-object. The whole point of the LLM judge is to score the axes humans care about *directly*, instead of correlating with cheap proxies.

Here's a sketch of the scoring prompt — not the verbatim ByteDance prompt, which isn't fully published, but a faithful reconstruction of the rubric the paper describes. This is the kind of prompt you'd hand to the oracle:

```python
# ----- LLM-as-judge scoring prompt (rubric reconstruction) -----
# NOTE: column-0 comments use ## per house style, but inside a Python
## string the # is fine; we mark section headers with ## below.

JUDGE_SYSTEM = """You are a senior code reviewer. You will be shown the
full contents of a single source file. Rate its quality on a 0-10 scale,
where 10 is exemplary production code and 0 is unusable.

Score the file against FOUR dimensions, then give one overall score:

## Readability
  - Clear, descriptive names; consistent formatting; helpful comments.
  - Penalize cryptic single-letter variables and dead/commented-out code.

## Modularity
  - Small, focused, composable functions; sensible separation of concerns.
  - Penalize monolithic god-functions and deeply tangled control flow.

## Clarity / Correctness
  - Explicit intent; minimal redundancy; NO logical errors.
  - Penalize impossible conditions (e.g. `x < 0 and x > 1`), unreachable
    branches, and obvious bugs even if the file parses.

## Reusability / Self-containedness
  - Few external assumptions; parameterized I/O; no hard-coded absolute
    paths or environment-specific magic constants.

Return strict JSON: {"readability": int, "modularity": int,
"clarity": int, "reusability": int, "overall": int, "reason": str}.
The overall score must reflect the WEAKEST critical dimension, not the mean.
"""

def score_file(client, source_code: str) -> dict:
    resp = client.chat.completions.create(
        model="deepseek-v2-chat",          # the oracle
        temperature=0.0,                    # deterministic scoring
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"```\n{source_code}\n```"},
        ],
    )
    return parse_json(resp.choices[0].message.content)
```

Two design choices in that prompt are worth calling out. First, **temperature 0** — you want the oracle's scores to be reproducible so the distilled regressor has a stable target. Second, **"the overall score must reflect the weakest critical dimension, not the mean"** — this is the difference between a judge that's fooled by pretty-but-broken code and one that isn't. A file that's beautifully readable but logically wrong should score low overall, because correctness is non-negotiable. Averaging the four dimensions would let readability paper over a bug. The paper's emphasis on catching the `temp < 0 and temp > 1` case tells us the real rubric weights correctness heavily.

The distillation step is what makes this affordable. Running DeepSeek-V2-Chat over a trillion tokens would cost more than the pretraining run itself. Running a 1.3B regression model — roughly the size of a small embedding model — over the same corpus is a rounding error on the compute budget. You pay the oracle once for 222K files, capture its judgment in the regressor's weights, and amortize that judgment across the whole dataset. The 222K sample size is large enough to cover the distribution of common-language code but small enough to score with a frontier model in a reasonable window.

### A worked scoring example: two files, four dimensions

Make the rubric concrete. Imagine two Python files land in the pipeline. File A is a 40-line module with a clear module docstring, typed function signatures, three small functions each doing one thing, and no imports beyond the standard library. File B is a single 300-line function with one-letter variables, a commented-out block of an earlier attempt, a hard-coded `/home/alice/data/` path, and a branch guarded by `if n > 0 and n < 0`. A rule-based filter scores these almost identically on the metrics it can compute: both are valid UTF-8, both parse, both have a reasonable alphanumeric ratio, neither is auto-generated. On stars and license they might even rank File B *higher* if it happens to live in a popular repo. The LLM judge, asked to rate each 0–10 on readability, correctness/clarity, modularity, and reusability, separates them immediately: File A scores something like 9/9/9/8 — the one point off reusability because a real-world consumer still has to know the module's purpose — while File B scores roughly 3/2/2/2: unreadable names, an impossible branch that signals the author lost track of the logic, a monolith that can't be reused in pieces, and an environment-specific path that breaks the moment anyone else runs it. The 0–10 scale matters here: a binary keep/drop filter would have to pick a single line and would likely keep both (File B is not *garbage*, just *bad*), whereas the continuous score lets the pipeline rank File B into the bottom decile and drop it while keeping genuinely marginal-but-useful files. Multiply this judgment across billions of files and the corpus quality difference compounds — every File B you train on is a small nudge toward writing one-letter variables and dead branches, and every File A is a nudge toward the idioms you actually want the model to internalize. This is the whole thesis in one example: form is what rules measure, and form is exactly where a pretty-but-broken file scores well.

### Why a regressor and not a classifier

A subtle but smart choice: the distilled scorer has a **regression head**, predicting a continuous quality score in `[0, 1]`, rather than a binary keep/drop classifier. This buys two things. It lets you **tune the filtering strength after the fact** by moving the threshold — drop the bottom 10%, or the bottom 5%, or the bottom 20% — without retraining the scorer. And it preserves a *gradient* of quality you could use for curriculum or upsampling decisions later. A binary classifier throws that information away. This is exactly the flexibility the paper accuses rule-based pipelines of lacking: a continuous score with a movable threshold is far more controllable than 130 fixed boolean rules.

## 5. The 8B architecture, briefly

You came for the data pipeline and that's where the novelty is, so the architecture section is short — but precise, because the published `config.json` is the ground truth and a couple of secondary sources get it wrong.

Seed-Coder is a **dense decoder-only transformer in the Llama-3 mold**. From the released configs:

| Spec | Value |
|---|---|
| Total parameters | ~8.2B |
| Layers | 32 |
| Hidden size | 4,096 |
| FFN / intermediate size | 14,336 |
| Attention query heads | 32 |
| KV heads (GQA groups) | 8 |
| Head dimension | 128 |
| Activation | SiLU (SwiGLU FFN) |
| Normalization | RMSNorm, eps 1e-6 |
| Position encoding | RoPE, θ = 500,000 |
| Vocabulary | 155,136 |
| Tied embeddings | No |
| Context (Base / Instruct) | 32,768 |
| Context (Reasoning) | 65,536 |

A few things to notice. The **8-group GQA** (32 query heads sharing 8 KV heads) is the standard inference-efficiency move — it cuts the KV-cache footprint 4x versus full multi-head attention, which matters a lot for repository-level code completion where you're streaming long contexts. The **RoPE θ of 500,000** is the large base you need to make 32K-plus context behave; a small θ aliases at long range. The **155K vocabulary** is notably larger than a general-purpose Llama tokenizer (~128K) — the extra tokens buy better compression on code, where whitespace, operators, and common identifiers benefit from dedicated tokens. And the **untied embeddings** (input and output embedding matrices are separate) is a small capacity bump that's cheap at this scale.

The training schedule has two phases. **Regular pretraining** runs at 8K context over the bulk of the corpus. **Continued pretraining** then does two things: a high-quality phase (~400B tokens, learning rate dropped by √10) and a **long-context extension phase** (~600B tokens at LR 3e-5) that pushes context to 32K using the repo-level long sequences. The Reasoning variant later extends context further to 64K during RL. This is a textbook curriculum — broad-then-narrow, short-then-long — and it's the same pattern you'll see in the [OLMo 3 open-training writeup](/blog/paper-reading/large-language-model/olmo-3-training-finetuning-techniques), which is worth reading alongside this for the contrast between fully-open-data and model-curated-data philosophies.

## 6. Base → Instruct → Reasoning: three models, one corpus

The three released variants are not three separate trainings from scratch; they're a stack. Base is the pretrained foundation, Instruct adds alignment, Reasoning adds long-chain-of-thought RL on top. Understanding the stack tells you which model to reach for.

![Layered stack of Base, Instruct, and Reasoning training stages](/imgs/blogs/seed-coder-model-centric-code-data-pipeline-6.png)

<!-- FIGSPEC 6
kind: layered-stack
claim: Each variant adds one training stage on top of the previous, from 6T-token pretraining to GRPO long-CoT RL.
caption: Base is the curated-data foundation; Instruct adds SFT+DPO; Reasoning adds GRPO RL with 64K context.
nodes:
  - id: l0 | label: "Model-centric corpus ~6T tokens, 89 langs" | color: gray
  - id: l1 | label: "BASE: pretrain 8K then continued 32K context" | color: blue
  - id: l2 | label: "INSTRUCT: SFT ~3M pairs + DPO ~20K pairs" | color: green
  - id: l3 | label: "REASONING: GRPO long-CoT, verl+DAPO, 64K ctx" | color: amber
notes: four horizontal layers stacked bottom-to-top; corpus at base (gray), Base, Instruct, Reasoning on top; each layer full width
-->

**Base** is what comes out of the pretraining curriculum in section 5: a 6T-token model that's strong at raw code generation and completion but speaks "continuation," not "conversation." You prompt it with a prefix and it continues. It's the right model for fill-in-the-middle and code-completion harnesses where you control the prompt format. It tops the open 8B base models on HumanEval, MBPP, and MultiPL-E (numbers in section 8).

**Instruct** is Base plus a two-stage post-training. **SFT** on roughly **3 million high-quality instruction-response pairs** (public datasets plus synthetic data), trained for 3 epochs at LR 2e-5. Then **DPO** (Direct Preference Optimization) on roughly **20,000 high-quality preference pairs** to sharpen the model toward responses humans prefer. The SFT teaches the model to follow instructions and emit clean, complete code in a chat format; the DPO trims the rough edges — verbosity, format drift, the small ways a model annoys you. This is the model you'd wire into an IDE assistant or an agent.

**Reasoning** is Instruct's foundation plus **long-chain-of-thought reinforcement learning**. The algorithm is **GRPO** (Group Relative Policy Optimization), run on the open-source **verl** framework, with optimization tricks borrowed from **DAPO** — if you want the deep mechanics of VAPO/DAPO-style RL, the [Seed1.5-Thinking RL writeup](/blog/paper-reading/large-language-model/seed1-5-thinking-rl-reasoning-vapo-dapo) covers them. The reward is **execution-based**: solutions are run in a sandbox against test cases, and only correct generations survive rejection sampling during data collection. The training uses a progressive context schedule — first 90 steps at 16K sequence length with 16 samples per prompt, then 160 steps at 32K with 32 samples per prompt — with batch size 128, LR 1e-6, temperature 0.6, clip ratio 0.28, and KL loss removed (a DAPO-style choice that lets the policy move further from the reference). The payoff is a model that *thinks before it codes* on hard competitive-programming problems, and the Codeforces and IOI numbers in section 9 show it works.

The reason this stack matters operationally: **the Reasoning model is slower and chattier** because it generates a long CoT before the answer. For a one-line autocomplete you do not want it; for a Codeforces Div-1 problem you absolutely do. Picking the wrong variant for the task is the most common way to be disappointed by this family.

Here's the canonical way to load and run each variant. The Base model for raw completion and FIM, the Instruct model for chat:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

## ---- Instruct: chat-style code generation ----
model_id = "ByteDance-Seed/Seed-Coder-8B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True,
)

messages = [{"role": "user",
             "content": "Write a Python LRU cache with O(1) get/put."}]
inputs = tok.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

out = model.generate(
    inputs, max_new_tokens=512,
    temperature=0.6, top_p=0.8, repetition_penalty=1.05,
)
print(tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True))
```

And the Base model's native fill-in-the-middle, which uses dedicated sentinel tokens — this is the mode an IDE plugin uses when the cursor sits between existing code:

```python
import torch, transformers

## ---- Base: fill-in-the-middle (FIM) infilling ----
pipe = transformers.pipeline(
    "text-generation",
    model="ByteDance-Seed/Seed-Coder-8B-Base",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

prefix = "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n"
suffix = "\n    return quicksort(left) + middle + quicksort(right)\n"

## Seed-Coder FIM sentinels: prefix / suffix / middle
prompt = f"<[fim-prefix]>{prefix}<[fim-suffix]>{suffix}<[fim-middle]>"
print(pipe(prompt, max_new_tokens=128, do_sample=False)[0]["generated_text"])
```

The FIM token order is the important detail. Seed-Coder uses the **PSM (prefix-suffix-middle)** arrangement with `<[fim-prefix]>`, `<[fim-suffix]>`, `<[fim-middle]>` sentinels: you give it the code before the cursor and the code after the cursor, and it fills the gap. Getting the sentinel order wrong is the single most common way to get garbage out of any FIM model, and Seed-Coder is no exception. The repo-level pretraining is what makes this strong — the model has seen millions of real files where the middle was learned in the context of both directions.

## 7. The ablation: does the model filter actually pay off?

A strong thesis deserves a clean experiment, and this is where I expected the paper to be weakest — data-quality ablations are notoriously easy to game. Seed-Coder's evidence is a controlled pretraining comparison: train the same architecture on **data filtered with minimal rules only** versus **data filtered with the LLM-based quality scorer**, holding everything else fixed, and watch downstream benchmark pass@1 as pretraining progresses.

![Before-after of rule-filtered versus model-filtered pretraining on downstream pass@1](/imgs/blogs/seed-coder-model-centric-code-data-pipeline-5.png)

<!-- FIGSPEC 5
kind: before-after
claim: Holding architecture fixed, LLM-filtered data lifts downstream pass@1 above rule-filtered data throughout pretraining.
caption: The ablation that justifies the whole approach: model-curated data wins on every downstream code benchmark tracked.
nodes:
  - id: a1 | label: "RULE-FILTERED: minimal rules only" | color: amber
  - id: a2 | label: "HumanEval: lower curve" | color: amber
  - id: a3 | label: "MBPP: lower curve" | color: amber
  - id: a4 | label: "same 8B arch, same tokens" | color: gray
  - id: b1 | label: "MODEL-FILTERED: LLM judge" | color: green
  - id: b2 | label: "HumanEval: higher curve" | color: green
  - id: b3 | label: "MBPP: higher curve" | color: green
  - id: b4 | label: "drop bottom 10% by quality" | color: blue
notes: left column rule-filtered (amber/gray), right column model-filtered (green/blue); two stacked columns; emphasize the curve gap grows over training
-->

The result, in the paper's Figure 6: across pretraining, the model trained on LLM-filtered data tracks **consistently above** the rule-filtered baseline on downstream code benchmarks. The paper states plainly that "code data curated by LLM filters significantly improves the coding performance of the pretrained models." The gap is present early and persists — this isn't a transient that washes out by the end of training.

I'll be honest about what the public materials *don't* give us: a single clean number like "+X points pass@1 at fixed compute." The evidence is the curves and the qualitative claim of a significant, persistent gap, plus the end-to-end result that the fully-trained Base model beats every open 8B base model on the headline benchmarks. If you're skeptical of data-ablation papers — and you should be — the most convincing evidence is the *transfer*: the same recipe produces a Base that wins, an Instruct that wins, and a Reasoning model that beats 32B competitors. A flawed data pipeline does not produce three strong models across three different post-training regimes by accident.

There's a second-order point worth stating. The rule-filtered baseline here is not a strawman — "minimal rules" includes the dedup, syntax-check, and language-ID steps that everyone agrees are necessary. So the ablation isolates *exactly* the contribution of the LLM quality scorer: the delta between "structurally-clean data" and "structurally-clean AND quality-scored data." That the curve moves at all means the quality dimension carries real signal that structural filters miss. That it moves *persistently* means it's not just removing a few toxic files — it's reshaping the whole quality distribution of what the model trains on.

> The cleanest reading of the ablation: structural filters decide what is *valid* code; the LLM judge decides what is *good* code; and training on good code, not merely valid code, is worth measurable downstream points.

## 8. Base and Instruct benchmarks against the field

Now the scoreboard. The claim is SOTA among open ~8B code models, with reach into larger ones. Let's check it on the standard suite.

![Matrix of Seed-Coder versus peer code LLMs on HumanEval, MBPP, BigCodeBench, LiveCodeBench](/imgs/blogs/seed-coder-model-centric-code-data-pipeline-7.png)

<!-- FIGSPEC 7
kind: matrix
claim: Seed-Coder-8B-Instruct leads peers on MBPP, MHPP, BigCodeBench, and LiveCodeBench at the 8B scale.
caption: Across four benchmarks Seed-Coder tops Qwen2.5-Coder, OpenCoder, and DeepSeek-Coder, trailing only on HumanEval.
nodes:
  - id: c1 | label: "rows: models / cols: benchmarks" | color: gray
notes: matrix with rows = DeepSeek-Coder-6.7B-Inst, OpenCoder-8B-Inst, Qwen2.5-Coder-7B-Inst, Qwen3-8B, Seed-Coder-8B-Inst; cols = HumanEval, MBPP, BigCodeBench-Full, LiveCodeBench; bold/green the Seed-Coder leads (MBPP 85.2, BCB 53.3, LCB 24.7), amber where Qwen2.5-Coder leads HumanEval 88.4
-->

**Base models** (pretrained, no instruction tuning), pass@1:

| Benchmark | DeepSeek-Coder-6.7B | OpenCoder-8B | Qwen2.5-Coder-7B | **Seed-Coder-8B-Base** |
|---|---|---|---|---|
| HumanEval | 47.6 | 66.5 | 72.0 | **77.4** |
| HumanEval+ | — | — | — | **82.0** |
| MBPP | 70.2 | 79.9 | 79.4 | 68.3* |
| MultiPL-E (avg) | 44.7 | 61.0 | 58.8 | **67.6** |
| CRUXEval-O | 41.0 | 43.9 | **56.0** | 54.8 |

*The MBPP figures shift between the MBPP and MBPP-sanitized splits across sources; the paper's Table 1 reports Seed-Coder-8B-Base at 68.3 MBPP / 69.0 MBPP+, while a separate model-card table lists 82.0 on a different MBPP variant. Treat the MBPP cell as version-dependent and trust the relative ordering more than the absolute. The MultiPL-E win (**67.6** average across 8 languages) and the **77.4 / 82.0** HumanEval/HumanEval+ are the robust headline numbers for Base.

The Base MultiPL-E per-language breakdown is worth seeing because it shows Seed-Coder isn't a Python-only model:

| Language | Python | C++ | Java | PHP | TypeScript | C# | Bash | JavaScript |
|---|---|---|---|---|---|---|---|---|
| Seed-Coder-8B-Base | 77.4 | 69.6 | 72.8 | 63.9 | 77.4 | 53.8 | 48.1 | 77.6 |

Strong across the curly-brace languages and TypeScript/JavaScript; weaker on Bash and C# (Bash being notoriously hard for every model). **Instruct models**, pass@1, the table the paper actually leads with:

| Model | HumanEval | MBPP | MHPP | BigCodeBench-Full | BigCodeBench-Hard | LiveCodeBench (2410-2502) |
|---|---|---|---|---|---|---|
| CodeLlama-7B-Instruct | 40.9 | 54.0 | 6.7 | 25.7 | 4.1 | 3.6 |
| DeepSeek-Coder-6.7B-Instruct | 74.4 | 74.9 | 20.0 | 43.8 | 15.5 | 9.6 |
| CodeQwen1.5-7B-Chat | 83.5 | 77.7 | 17.6 | 43.6 | 15.5 | 3.0 |
| Yi-Coder-9B-Chat | 82.3 | 82.0 | 26.7 | 49.0 | 17.6 | 17.5 |
| Llama-3.1-8B-Instruct | 68.3 | 70.1 | 17.1 | 40.5 | 13.5 | 11.5 |
| OpenCoder-8B-Instruct | 83.5 | 79.1 | 30.5 | 50.9 | 18.9 | 17.1 |
| Qwen2.5-Coder-7B-Instruct | **88.4** | 83.5 | 26.7 | 48.8 | 20.3 | 17.3 |
| Qwen3-8B | 84.8 | 77.0 | 32.8 | 51.7 | 23.0 | 23.5 |
| **Seed-Coder-8B-Instruct** | 84.8 | **85.2** | **36.2** | **53.3** | **26.4** | **24.7** |

Read this carefully because it's the real story. Seed-Coder-8B-Instruct **leads on MBPP (85.2), MHPP (36.2), BigCodeBench-Full (53.3), BigCodeBench-Hard (26.4), and LiveCodeBench (24.7)** — and it's the hard benchmarks where it wins biggest. It *trails* Qwen2.5-Coder-7B on plain **HumanEval (84.8 vs 88.4)**, and that's worth being honest about: on the saturated, contamination-prone benchmark, the Qwen tuning edges it. But HumanEval is the least informative benchmark on the sheet in 2025 — it's nearly solved and heavily memorized. The benchmarks that actually discriminate — BigCodeBench-Hard, which demands real library usage, and LiveCodeBench, which is contamination-resistant by design (it uses problems published after model training cutoffs) — are exactly where Seed-Coder pulls ahead, even past Qwen3-8B. The paper's claim that it "surpasses Qwen2.5-Coder-14B-Instruct on LiveCodeBench despite its smaller size" is the kind of result that justifies the data thesis: better data, not more parameters.

On the structural side, the Base model's repository-aware completion is genuinely strong. On **CrossCodeEval** (cross-file completion) it averages **53.7 exact-match / 85.1 edit-similarity**; on **RepoEval** it averages **50.8 EM / 75.6 ES**; on single-line **MultiPL-HumanEval FIM** it hits **82.4% average exact match**. Those FIM and repo numbers are a direct dividend of the repo-level topological-concatenation pretraining — the model learned to complete code *in the context of the surrounding project*, not just in the context of a single file.

## 9. The Reasoning model and competitive programming

The Reasoning variant is where the "8B beats 32B" headline gets loud. After GRPO long-CoT RL, Seed-Coder-8B-Reasoning is a competitive-programming machine well out of proportion to its size.

![Grid of the three Seed-Coder variants against their target benchmarks](/imgs/blogs/seed-coder-model-centric-code-data-pipeline-8.png)

<!-- FIGSPEC 8
kind: grid
claim: Each variant targets a distinct benchmark family and the Reasoning model rivals 32B reasoning models on contests.
caption: Base for FIM/MultiPL-E, Instruct for BigCodeBench/LiveCodeBench, Reasoning for Codeforces and IOI.
nodes:
  - id: h1 | label: "Variant" | color: gray
  - id: h2 | label: "Primary task" | color: gray
  - id: h3 | label: "Headline result" | color: gray
  - id: h4 | label: "Notably beats" | color: gray
  - id: r1 | label: "Base" | color: blue
  - id: r1a | label: "completion / FIM / repo" | color: gray
  - id: r1b | label: "82.4 FIM EM, 67.6 MultiPL-E" | color: blue
  - id: r1c | label: "every open 8B base" | color: gray
  - id: r2 | label: "Instruct" | color: blue
  - id: r2a | label: "instruction codegen" | color: gray
  - id: r2b | label: "53.3 BCB, 24.7 LiveCodeBench" | color: blue
  - id: r2c | label: "Qwen2.5-Coder-14B on LCB" | color: green
  - id: r3 | label: "Reasoning" | color: green
  - id: r3a | label: "competitive / algorithmic" | color: gray
  - id: r3b | label: "53.3 LCB, CF ELO 1553, IOI 146.5" | color: green
  - id: r3c | label: "QwQ-32B + DeepSeek-R1 on IOI" | color: green
notes: 4 rows x 4 cols table; header row gray; first col variant names; Reasoning row emphasized green
-->

The numbers that matter:

- **LiveCodeBench (2410-2502): 53.3% pass@1.** That is a *huge* jump from the Instruct model's 24.7% on the same benchmark window — long-CoT RL more than doubles the contest-coding pass rate. For scale, this lands the 8B Reasoning model in the territory of much larger general reasoning models.
- **Codeforces: ELO ~1,553**, which the paper describes as approaching **o1-mini** on Codeforces contests. For an 8B open model to reach o1-mini-class contest ELO is the standout result of the paper.
- **IOI'2024: 146.5**, which **beats QwQ-32B (144.0) and DeepSeek-R1 (137.0)** — both of which are 4x-plus the parameter count. The International Olympiad in Informatics is about as hard as algorithmic problem-solving gets, and an 8B model topping a 32B reasoning model there is a genuine result, not a benchmark artifact.
- **SWE-bench Verified: 19.2% resolved.** This is the software-engineering end of the spectrum — resolve a real GitHub issue with a patch — and it's the weakest area relative to the contest numbers, which makes sense: SWE-bench rewards repo navigation and tool use more than pure algorithmic reasoning, and at 8B you're competing against models with far more general capability.

Put the three variants side by side and the division of labor is clean:

| Variant | Primary task | Headline result | Notably beats |
|---|---|---|---|
| **Base** | Completion, FIM, repo-level | 82.4 FIM EM · 67.6 MultiPL-E · 77.4 HumanEval | every open 8B base model |
| **Instruct** | Instruction-following codegen | 53.3 BigCodeBench-Full · 24.7 LiveCodeBench | Qwen2.5-Coder-14B on LiveCodeBench |
| **Reasoning** | Competitive / algorithmic | 53.3 LiveCodeBench · CF ELO 1553 · IOI 146.5 | QwQ-32B & DeepSeek-R1 on IOI'2024 |

The thing I'd underline: these three results come from **one corpus and one base model**, differentiated only by post-training. That's the strongest possible evidence for the data thesis. If your pretraining data is good enough, the same foundation supports completion, instruction-following, and Olympiad-level reasoning. The data is the moat; the post-training is the configuration.

## 10. Acknowledged limitations

The paper is refreshingly direct about where Seed-Coder is weak, and the limitations follow logically from the design choices.

**It is a code specialist, not a generalist.** Trained on ~6T code-heavy tokens, its general natural-language understanding and broad-task ability lag behind general-purpose models trained on far more diverse data. The paper explicitly contrasts itself with **Qwen3, pretrained on ~36T tokens**: that's 6x the token budget and a vastly broader mix. If you ask Seed-Coder to write a marketing email or reason about history, you are using the wrong tool. The 6T-token budget was spent buying code quality, and that trade-off is visible the moment you step outside code.

**The 6T budget is small by frontier standards.** Six trillion tokens is a deliberate, focused budget, not a frontier-scale one. The model's ceiling on tasks that benefit from sheer data exposure — obscure languages, very long-tail libraries, multilingual natural language — is correspondingly lower. The MultiPL-E table shows this: strong on Python/JS/TS/Java, weaker on Bash and C#.

**The LLM-filter inherits the oracle's biases.** This one the paper doesn't dwell on but it's the most important caveat for practitioners. The quality scorer is distilled from DeepSeek-V2-Chat's judgments. Whatever that oracle systematically over- or under-values — a coding style it prefers, a paradigm it's unfamiliar with, a language it's weaker in — propagates into what gets kept and dropped at trillion-token scale. A model-centric pipeline is only as good as its judge. If the oracle has a blind spot, the entire corpus inherits it, silently, and no rule will flag it because there are no rules anymore. The flip side of "let the model decide" is "you've delegated your data taste to a model whose taste you didn't fully audit."

**SWE-bench is the soft spot.** At 19.2% Verified, the agentic software-engineering capability trails the contest capability by a wide margin. Real-world issue resolution needs repo-scale reasoning, tool use, and broad world knowledge — exactly the generalist muscles a code specialist underbuilds. For agentic SWE work you'll likely still reach for a larger general model. For a fuller map of where Seed-Coder sits among ByteDance's other models, the [ByteDance Seed model universe writeup](/blog/machine-learning/large-language-model/bytedance-seed-model-universe-by-use-case) places it in context.

## 11. Porting the recipe to your own corpus

The reason this paper matters beyond Seed-Coder itself is that the pipeline is a *reusable recipe*, and most teams curating a domain corpus — internal code, legal text, scientific papers, support transcripts — can lift it almost verbatim. Here is the recipe stripped to its load-bearing parts, with the design decisions that actually matter.

**Step 1 — keep the cheap structural filters; you cannot afford to skip them.** Exact dedup (SHA-256), near-dedup (MinHash/LSH), a syntax or format check, and language identification are all O(cheap) and they shrink the raw corpus by something like 98% before any model touches a file. This ordering is not incidental — it is the entire cost argument. If you run an LLM judge over raw, un-deduplicated GitHub, you pay to score the same vendored `jquery.min.js` ten thousand times. Dedup first, score second. The structural filters are also the ones you *want* to keep as explicit rules, because "is this valid UTF-8" and "does this parse" are exactly the kind of unambiguous, auditable checks that rules are good at. The model-centric insight is narrow: replace the *quality* heuristics, not the *structural* ones.

**Step 2 — score a sample with a strong oracle, not the whole corpus with a cheap one.** Seed-Coder scored ~222K files with a capable model (the paper uses a strong general LLM as the oracle), reading each file and rating it 0–10 on the four dimensions. The sample size is the knob: you need enough labeled examples to train a regressor that generalizes, not enough to cover the corpus. This is the step that makes the whole thing affordable — 222K oracle calls is a rounding error next to scoring billions of files directly. Write the rubric explicitly (Seed-Coder's is readability, correctness/clarity, modularity, reusability), give the oracle few-shot anchors for what a 2 and a 9 look like, and have it emit a number per dimension rather than a single gestalt score, because the per-dimension breakdown is what lets you re-weight later.

**Step 3 — distill the oracle's judgment into a small regressor.** Train a small model (Seed-Coder uses a ~1.3B scorer) to predict the oracle's scores from the file, then run *that* over the full corpus. The regressor only has to imitate the oracle's numbers, which is a far easier task than reproducing its reasoning, so a model two orders of magnitude smaller suffices. This is the move that turns "model-centric filtering" from a slogan into something you can run on a budget: you capture the expensive judgment once, in the labels, and amortize it through a cheap model. The cost falls from "frontier-model inference over billions of files" to "frontier-model inference over a few hundred thousand, plus small-model inference over the rest."

**Step 4 — output a continuous score and make the threshold a runtime knob.** This is the regression-vs-classification decision from §4, and it is worth restating as a recipe rule because it is nearly free and saves a re-training run the first time you want to filter harder or softer. A binary keep/drop classifier bakes the decision boundary into the weights; a regressor that outputs `[0,1]` lets you sweep the threshold (drop bottom 10%, then try 20%) without touching the model. Continuous scores also let you do quality-weighted sampling — upweight the 9s in the mix rather than merely discarding the 2s — which a binary filter cannot express.

**Step 5 — re-balance per domain, and audit the drops, not the rules.** The two failure modes from the case studies and the limitations section are the ones to design against from day one. First, the oracle's "good code" is whatever its training data considered good, so a dimension that is right for general-purpose library code (self-containedness, few dependencies) can be wrong for your domain (framework code, where dependency *is* the design); re-weight the offending dimension for the directories where it misfires. Second, you have traded 130 rules you could read for one model you cannot, so the audit surface moves from the *rules* to the *outputs* — periodically sample the files your filter dropped and have a human check whether the drops are defensible. If your filter is quietly discarding an idiom your team writes constantly, you want to learn that from a spot-check, not from a model that has mysteriously never seen your house style.

The honest caveat: this recipe buys quality at the cost of auditability and adds an oracle-bias dependency you did not have with rules. For most teams that trade is worth it — the case studies show rule filters silently discarding correct-but-unusual code and keeping pretty-but-broken code, and at corpus scale those errors compound into a measurably worse model. But if you are in a regulated setting where you must point at the exact rule that dropped a file, model-centric filtering is the wrong tool, and that is a real constraint, not a corner case. Everyone else should treat Seed-Coder's pipeline as the new default for data curation and the 130-rule status quo as the thing it replaces.

## Case studies from production

### 1. The Pikachu LED script the rules killed

The paper's own motivating example, and the cleanest illustration of why surface heuristics fail. A perfectly valid Python script that lights up a Pikachu on an LED matrix contains a large block of numeric pixel data, which pushes its numeric-character ratio above a rule-based filter's threshold. The rule filter drops it as "probably a data dump." The LLM judge reads it, sees a well-structured program with a clear purpose that happens to embed image data, and keeps it. The lesson generalizes far beyond Pikachu: lookup tables, embedding constants, generated parsers, and hardware register maps all trip numeric-ratio rules while being entirely legitimate code. Every one of them is training signal a rule-based pipeline silently discards. When the filter's failure mode is "throws away unusual-but-correct code," you systematically narrow what your model can learn to write.

### 2. The clean script with the impossible condition

The inverse failure. A script that is beautifully formatted, well-commented, sensibly named — sails through every readability and comment-ratio rule — but contains `if temp < 0 and temp > 1:`, a condition that can never be true. The branch is dead code; the program is subtly broken. No surface rule catches this because every surface metric says the file is high quality. The LLM judge, reading for *clarity and correctness*, flags the impossible condition and scores the file down. This is the asymmetry that makes the model-centric approach worth its cost: rules can only measure form, and form is exactly what a broken-but-pretty file gets right. If you train a code model on pretty-but-broken code, it learns to write pretty-but-broken code.

### 3. Choosing Base over Instruct for an autocomplete plugin

A team building an in-editor completion plugin reached for Seed-Coder-8B-Instruct because "instruct is the better model," wired it into a FIM harness, and got verbose, chatty completions that tried to explain the code instead of just completing it. The fix was to drop to **Seed-Coder-8B-Base** and use the native `<[fim-prefix]>/<[fim-suffix]>/<[fim-middle]>` sentinels. Base is a continuation model; it completes the gap and stops. Instruct is a conversation model; it wants to talk. For cursor-position completion — where you control the prompt and want raw continuation — Base's 82.4 FIM exact-match is the right tool, and Instruct's chattiness is an active liability. The general rule: instruction tuning is not a free upgrade, it's a behavior change, and FIM is a use case where that behavior change hurts.

### 4. The MBPP number that didn't reconcile

An evaluation engineer tried to reproduce Seed-Coder-8B-Base's MBPP score and got a number that matched neither the paper's 68.3 nor a model-card's 82.0. The root cause: **MBPP has multiple splits** — the full set and the sanitized set — and different sources report different ones, sometimes without labeling which. The lesson is one every benchmarking effort relearns: a benchmark name is not a benchmark. Always pin the exact split, the exact prompt template, and the exact pass@k. The robust takeaways from Seed-Coder's Base evaluation are the MultiPL-E average (67.6) and HumanEval/HumanEval+ (77.4/82.0), which are stable across sources; the MBPP cell is the one to treat with suspicion until you've pinned the split yourself.

### 5. Wrong variant for a Codeforces bot

A hobbyist building a Codeforces-solving bot started with Seed-Coder-8B-Instruct, got mediocre results on Div-1 problems, and concluded the model "wasn't good at competitive programming." It is — but the **Reasoning** variant is, not Instruct. On LiveCodeBench the gap is 24.7% (Instruct) versus 53.3% (Reasoning); on a hard contest set that difference is the difference between solving and not solving. The Reasoning model's long chain-of-thought is the whole point: it reasons through the algorithm before emitting code. Instruct emits code immediately, which is great for "write me a function" and bad for "solve this DP problem with a non-obvious recurrence." Matching the variant to the task is the single highest-leverage decision when deploying this family.

### 6. The 64K-context surprise

A team feeding very long repository contexts to the Reasoning model for whole-codebase reasoning found it handled far longer inputs than the Instruct model — because **Reasoning extends context to 65,536 tokens** versus Instruct/Base's 32,768. The RL phase's progressive context schedule (16K → 32K, with the architecture supporting 64K) gives the Reasoning model room for both a long input *and* a long chain of thought. The gotcha is the inverse: don't assume all three variants share a context limit. If your workload needs >32K context, only the Reasoning variant fits it natively, and you pay for that with the CoT generation overhead whether you want the reasoning or not.

### 7. Distilling judgment instead of buying it

A team building their own data pipeline wanted model-based filtering but balked at the inference cost of running a frontier model over their corpus. The Seed-Coder pattern solved it: score a **sample** (222K files in the paper) with a strong oracle, then **distill a 1.3B regressor** from those scores and run the cheap model over everything. They cut their filtering inference bill by roughly two orders of magnitude versus scoring everything with the frontier model, with minimal quality loss because the regressor only has to imitate the oracle's *scores*, not its full reasoning. The transferable insight: model-centric filtering is not "run a big model on all your data" — it's "capture a big model's judgment in a small model, once."

### 8. The regression-head decision that saved a re-run

A pipeline initially trained a binary keep/drop *classifier* as its quality filter, then needed to tighten filtering from "drop bottom 10%" to "drop bottom 20%" and discovered they had to retrain the classifier with a new decision boundary baked in. Switching to a **regression head** outputting a continuous `[0,1]` score — Seed-Coder's choice — made the threshold a runtime knob. Re-filtering at a different strength became a one-line change, no retraining. The lesson: when you build a learned filter, predict a *score*, not a *label*. The continuous output preserves the flexibility that the paper specifically criticizes rule-based pipelines for lacking, and it's nearly free to add.

### 9. Repo-level data and the cross-file completion win

A team evaluating code models for a monorepo completion tool found Seed-Coder-8B-Base unusually strong at completing a function whose dependencies lived in *other files* — CrossCodeEval 53.7 EM, where file-level-only models stumble. The reason traces straight to the **topological dependency concatenation** in pretraining: Seed-Coder saw repositories assembled in import order, so it learned to use a symbol it "remembers" being defined earlier in the long sequence. A model pretrained only on shuffled individual files never gets that signal. The takeaway for anyone curating their own corpus: if you want cross-file completion, you have to *build cross-file context into the pretraining data*, not just the eval. Structure in the data becomes capability in the model.

### 10. The commits data and SWE-shaped tasks

A team fine-tuning for issue-to-patch workflows noticed Seed-Coder handled "given this commit message and context, produce the diff" surprisingly naturally out of the box. That's the **commits category** doing its job: 74M commits reframed as code-change-prediction tasks, complete with README, directory tree, and BM25-retrieved relevant files as context. The model was pretrained on exactly the shape of a SWE-bench task. The broader point is a design philosophy: if you know the downstream task shape, find a *naturally occurring* version of it in the wild (commits are issue-resolution traces in disguise) and pretrain on it, rather than relying entirely on synthetic SFT data to teach the format later. The 19.2% SWE-bench Verified shows there's still a ceiling at 8B, but the commits pretraining is why an 8B model is on the board at all.

### 11. The oracle-bias audit nobody ran

A team adopting the model-centric approach for an internal corpus skipped auditing what their oracle judge systematically preferred — and discovered months later that their filter had quietly downweighted a whole idiomatic style common in their codebase that the oracle, trained mostly on public GitHub, found "unusual." The Seed-Coder limitation this maps to is real: a model-centric pipeline inherits its judge's blind spots, and unlike a rule you can read and argue with, a distilled regressor's biases are opaque. The mitigation they landed on: periodically sample files the filter *dropped* and have a human spot-check whether the drops are defensible. Model-centric filtering removes the 130 rules you could audit and replaces them with one model you can't read — so you have to audit the *outputs* instead of the *rules*.

### 12. Picking Seed-Coder over a general model for cost reasons

A startup serving a code-completion feature compared Seed-Coder-8B-Instruct against a much larger general-purpose model and found that on their actual code tasks — BigCodeBench-style real-library usage, FIM completion — the 8B specialist matched or beat the generalist at a fraction of the serving cost, thanks to GQA-friendly 8B inference. The decision hinged on workload: their traffic was 95% code, 5% chitchat, and the code specialist won decisively on the 95%. For the 5% they fell back to a general model. The lesson is the whole case for specialist models: if your workload is narrow and the specialist is genuinely SOTA-at-scale on that narrow thing, the cost and latency wins are enormous, and Seed-Coder's data thesis is precisely what makes an 8B specialist competitive with much larger generalists on code.

### 13. The framework boilerplate the self-containedness score punished

A team adapting the model-centric recipe for an internal corpus heavy on a web framework noticed the filter was systematically downweighting their controllers and migrations — files that are, by design, *not* self-contained: they import a base class, inherit conventions, and only make sense inside the framework. The judge, scoring "reusability / self-containedness," read these as low-quality because in isolation they look like fragments with unexplained magic. The files were perfectly good; the rubric dimension was mis-aligned with the domain. This is the sharp edge of the whole approach: a scoring dimension that is right for general-purpose library code ("few dependencies, no hard-coded assumptions") is actively wrong for framework code, where depending on the framework *is* the point. The fix was not to abandon the filter but to *re-weight* it for the corpus — relax the self-containedness term for directories known to be framework code — which is only possible because the score is continuous and per-dimension rather than a single opaque keep/drop verdict. The lesson generalizes: the four dimensions are not universal constants, they are a default rubric, and when you port model-centric filtering to a domain with different norms you must re-examine whether each dimension still means what you want. A judge inherits the priors of whatever it was trained on, and "good code" is more domain-specific than the rubric admits.

### 14. The FIM sentinel mismatch that produced garbage completions

An infrastructure engineer wired Seed-Coder-8B-Base into a completion service and got plausible-looking but subtly wrong completions — the model would complete as if the suffix didn't exist, duplicating code that already followed the cursor. The bug was not the model; it was the prompt format. They had reused a FIM template from a different model that used `<fim_prefix>`/`<fim_suffix>`/`<fim_middle>` tokens, but Seed-Coder's tokenizer uses its own sentinels (`<[fim-prefix]>` / `<[fim-suffix]>` / `<[fim-middle]>`), so the "FIM" markers were tokenizing as ordinary text and the model never entered infilling mode — it just did left-to-right continuation of a weird-looking prompt. The fix was one config change to the correct sentinels, after which the suffix-aware completions worked as advertised (the 82.4 FIM exact-match is real, but only in the model's native format). The transferable lesson is mundane and bites everyone who deploys infilling: FIM is a *training-time token convention*, not a universal protocol, and a sentinel mismatch fails silently — you get completions, they're just completing the wrong problem. Always pull the exact special tokens from the model's own tokenizer config rather than assuming the format you used last time carries over.

## When to reach for Seed-Coder, and when not to

**Reach for Seed-Coder when:**

- Your workload is **predominantly code** — completion, generation, editing, FIM — and you want SOTA-at-8B quality without paying for a frontier general model. The data thesis is what makes the 8B competitive; lean into it.
- You need **cross-file or repository-level completion**. The topological-concatenation pretraining gives Base genuinely strong CrossCodeEval/RepoEval/FIM numbers that file-only models can't match.
- You're doing **competitive or algorithmic programming** — reach for the **Reasoning** variant specifically (CF ELO ~1553, IOI 146.5 beating 32B models, LiveCodeBench 53.3%).
- You want an **MIT-licensed, fully open-weights** model you can fine-tune and ship commercially without license friction.
- You're **building your own data pipeline** and want a proven recipe for model-centric filtering — the oracle→distilled-regressor→continuous-score pattern is the reusable artifact here, independent of the model itself.

**Skip Seed-Coder when:**

- You need a **generalist**. It's a code specialist trained on 6T code-heavy tokens; its NL understanding and broad-task ability trail general models trained on 36T-plus. Don't ask it to write prose or reason about non-code domains.
- Your task is **agentic software engineering at scale** (heavy SWE-bench-style issue resolution). At 19.2% Verified it's on the board but not leading; a larger general model with strong tool use will likely beat it.
- You need **very long context beyond 64K**, or 32K-plus context from the Base/Instruct variants (those cap at 32K; only Reasoning reaches 64K).
- You're picking the **Instruct model for raw FIM autocomplete** — that's a mismatch; use Base. Or picking **Instruct for hard contests** — use Reasoning. Variant selection is the most common way to be disappointed by this family.
- You require a **fully auditable, deterministic data filter** for compliance reasons. The model-centric pipeline trades the readability of 130 explicit rules for the quality of one opaque learned judge; if you must be able to point at the exact rule that dropped a file, the LLM-judge approach is the wrong fit.

The deeper takeaway outlives the specific model. Seed-Coder is a clean demonstration that **the bottleneck in code-LLM quality was never the architecture — it was the data filter**, and that the right move is to stop encoding your taste as rules and start distilling a model's judgment instead. The 8B numbers are impressive, but the recipe is the contribution: keep the structural filters you can't argue with, replace the quality heuristics you can, distill the expensive judgment into a cheap scorer, and let the code model curate data for itself.

It is worth sitting with why this is the direction the whole field is moving. For years the pretraining-data conversation was about *quantity* — scrape more, dedup, filter the obvious junk with regexes, and pour 30T-plus tokens through a bigger model. Seed-Coder is part of a turn toward *quality as a learned function*: the realization that a capable model is a better judge of "is this worth learning from" than any rule a human will write, and that you can capture that judgment cheaply enough to apply it at corpus scale. You can see the same idea in math-data pipelines, in instruction-data filtering, and in the synthetic-data flywheels other labs run — the common thread is a model in the curation loop, not just at the end of it. Seed-Coder's specific gift is that it does this *transparently and reproducibly* for code: the rubric is published, the oracle→regressor pattern is described concretely, and the resulting weights are MIT-licensed so you can verify the claim yourself. If you take one thing from this paper into your own work, let it be the mental reframe: stop thinking of your data filter as a preprocessing script and start thinking of it as a model you train, evaluate, and improve like any other — because at the scale modern pretraining runs, the filter is quietly one of the most important models in your stack.

## Further reading

- Seed-Coder paper — [arXiv:2506.03524, *Let the Code Model Curate Data for Itself*](https://arxiv.org/abs/2506.03524)
- Models on HuggingFace — [Base](https://huggingface.co/ByteDance-Seed/Seed-Coder-8B-Base), [Instruct](https://huggingface.co/ByteDance-Seed/Seed-Coder-8B-Instruct), [Reasoning](https://huggingface.co/ByteDance-Seed/Seed-Coder-8B-Reasoning)
- Code and benchmarks — [ByteDance-Seed/Seed-Coder on GitHub](https://github.com/ByteDance-Seed/Seed-Coder)
- ByteDance Seed model map — [the ByteDance Seed model universe](/blog/machine-learning/large-language-model/bytedance-seed-model-universe-by-use-case)
- The RL behind the Reasoning variant — [Seed1.5-Thinking: VAPO/DAPO reasoning RL](/blog/paper-reading/large-language-model/seed1-5-thinking-rl-reasoning-vapo-dapo)
- Contrast with fully-open training data — [OLMo 3 training and finetuning](/blog/paper-reading/large-language-model/olmo-3-training-finetuning-techniques)
- A cousin data pipeline for math — [DeepSeekMath data pipeline and the GRPO origin](/blog/machine-learning/training-techniques/deepseekmath-data-pipeline-and-grpo-origin)
