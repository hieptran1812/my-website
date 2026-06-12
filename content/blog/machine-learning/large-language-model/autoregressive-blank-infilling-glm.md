---
title: "GLM: Autoregressive Blank Infilling — The Objective That Unified Understanding and Generation"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "A close read of the 2021 GLM paper: how autoregressive blank infilling, a 2D positional scheme, and a hybrid attention mask fuse BERT-style understanding with GPT-style generation in a single transformer — and why the idea still echoes in today's hybrid-reasoning models."
tags:
  [
    "glm",
    "blank-infilling",
    "pretraining",
    "large-language-model",
    "transformer",
    "attention",
    "autoregressive",
    "paper-reading",
    "nlu",
    "text-generation",
  ]
category: "machine-learning"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: false
readTime: 30
---

> [!tldr]
> - **What it claims:** a single pretraining objective — *autoregressive blank infilling* — can match or beat BERT, GPT, and T5 at their own games (understanding, generation, seq2seq) using one model of the same size and data.
> - **Why it matters:** it dissolves the 2021 split between "understanding models" and "generation models." Every later GLM, up to the 355B agentic MoE, is still, mechanically, a blank-infilling model.
> - **Most surprising finding:** a hybrid attention mask lets the *same* transformer be a bidirectional encoder over the context and an autoregressive decoder over the masked spans — no separate encoder, no doubled parameters.
> - **Where it fails:** at the scale and on the English-only benchmarks of 2021, GLM's edge is real but modest; the objective's payoff compounds later (GLM-130B, GLM-4) more than the original paper could show.

This is the second article in a [series reading the entire GLM lineage](/blog/machine-learning/large-language-model/glm-lineage-frontier-llm-technique). The survey article made the case that GLM is an *accreting stack of techniques* — each release keeps the load-bearing parts of the last and adds one new layer. This article goes back to the genome: the 2021 paper [GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://arxiv.org/abs/2103.10360) (Du et al., ACL 2022), the objective that everything else is built on top of.

It is worth reading slowly, because the idea is deceptively simple and the consequences are large. By the end you should be able to implement the objective from scratch, explain why the 2D positional scheme is load-bearing, and see the straight line from this paper's "fill the blank" framing to GLM-4.5's thinking/non-thinking hybrid.

## Context: the 2021 split between understanding and generation

![A matrix showing that BERT, GPT, and T5 each cover only part of the capability space, while GLM covers all of it](/imgs/blogs/autoregressive-blank-infilling-glm-1.png)

The matrix above is the mental model for *why this paper exists*. In 2021 the field had effectively three pretraining religions, and each one was good at a different slice of the work:

- **Autoencoding (BERT).** Mask 15% of tokens, predict them from bidirectional context. The model develops a rich, bidirectional understanding of text — but it has no native way to *generate* a sequence, because it was never trained to produce one token after another. Great at NLU (classification, span extraction), structurally incapable of free generation.
- **Autoregressive (GPT).** Predict the next token left-to-right. The model is a natural generator — but its representation of any given token only sees the *left* context, which handicaps understanding tasks where the right context matters.
- **Encoder-decoder (T5, BART).** A bidirectional encoder feeds an autoregressive decoder. This genuinely does both — at the cost of roughly *doubling* the parameters, because you now maintain two stacks and a cross-attention bridge between them.

Read the matrix row by row and the gap is obvious. BERT's generation cells are red. GPT's understanding cells are amber. T5 does everything but pays a parameter tax. There was no single, parameter-efficient model that was simultaneously a first-class understander *and* a first-class generator. That is the gap the paper claims to fill.

The framing the authors use is sharp: the existing paradigms each impose a *structural* limitation, and those limitations are about which tokens can see which other tokens. BERT's tokens all see each other (bidirectional) but the model never generates. GPT's tokens see only leftward (causal) so it generates but reads weakly. The question GLM asks is: *can one attention pattern give you both, on the same tokens, in the same forward pass?* The answer — yes, with the right mask — is the whole paper.

It helps to remember the practical stakes of this split in 2021, because they're easy to forget now that one architecture (the causal decoder) has eaten almost everything. Back then, shipping an NLP product often meant maintaining *two or three* model families. You ran a BERT-family encoder for your classification, retrieval, and extraction services, and a GPT- or T5-family model for anything that generated text — and these were separate models, separate finetuning pipelines, separate serving stacks, separate everything. The cost wasn't only parameters; it was operational surface area. A single model that could do classification *and* summarization *and* extractive QA wasn't just an academic curiosity — it was a serious reduction in the number of things an ML platform team had to keep alive. GLM's pitch, read through that lens, is as much an infrastructure argument as a modeling one: collapse the zoo into one animal.

There's also a deeper reason the bidirectional/causal split was worth attacking head-on rather than papering over. Bidirectionality isn't a minor feature — for understanding tasks it's frequently decisive, because the meaning of a token genuinely depends on what comes after it (think of resolving a pronoun, or disambiguating a word whose sense is fixed by a later clause). A pure causal model is structurally handicapped on exactly these cases. So "just use a GPT for everything" was not, in 2021, an obviously correct answer; it traded away real understanding capability. GLM's contribution is to show you don't have to make that trade.

## Contributions

The paper's contributions, tightened from the authors' framing:

1. **Autoregressive blank infilling**, a pretraining objective that masks variable-length spans and reconstructs them autoregressively while reading the unmasked context bidirectionally — unifying autoencoding and autoregression in one model.
2. **A 2D positional encoding** that makes the model blind to the length of a masked span, turning blank-filling into true open-ended generation rather than fixed-length cloze.
3. **A multi-task pretraining setup** that mixes short-span, sentence-level, and document-level masking so the one objective covers NLU, seq2seq, and long-text generation.
4. **A finetuning reformulation** that recasts every downstream task — classification included — as a blank to fill, eliminating task-specific heads.
5. **Empirical results** showing a single GLM outperforms BERT, T5, and GPT given the same model size and data across understanding and generation benchmarks.

Notice how tightly these five contributions interlock — that's the sign of a well-designed paper rather than a bag of tricks. The objective (1) creates the need for length-blindness, which the positional scheme (2) supplies; the objective also admits the regime mixture (3), which is what makes one model cover three task families; and the finetuning reformulation (4) is only possible *because* the pretraining objective is "fill a blank," so downstream tasks can reuse the exact same interface. Pull any one out and the others wobble. A pure causal model couldn't use contribution (4) cleanly because it has no `[MASK]` to read a label from; a model without contribution (2) couldn't do open-ended generation. The empirical results (5) are then just the receipt that the interlocking design pays off. Keeping this dependency structure in mind makes the rest of the walkthrough easier: each piece exists to unblock another.

The rest of this post walks each of these, with the implementation in front of us.

## Method: how blank infilling actually works

### The corruption process

![A pipeline showing input text being corrupted into a masked Part A and a shuffled, autoregressive Part B](/imgs/blogs/autoregressive-blank-infilling-glm-2.png)

The pipeline above is the data transformation, end to end. Start from an input sequence and produce two parts:

- **Part A** is the input with several spans masked out, each span collapsed to a *single* `[MASK]` token regardless of how many tokens it covered.
- **Part B** is the spans themselves — the actual tokens that were masked — listed in *shuffled* order, each wrapped in a `[START]` (`[S]`) and `[END]` (`[E]`) sentinel.

The spans are sampled to cover ~15% of the tokens, with lengths drawn from a **Poisson distribution with λ = 3**. The choice of Poisson(3) is deliberate: it biases toward short spans (the mode is around 2–3 tokens) while occasionally producing longer ones, which keeps the task mostly local but sometimes demanding. Here is the whole corruption in runnable form:

```python
import numpy as np

def sample_spans(seq_len, mask_ratio=0.15, lam=3, rng=None):
    """Sample non-overlapping spans (Poisson lengths) until mask_ratio covered."""
    rng = rng or np.random.default_rng(0)
    target, taken, spans = int(round(mask_ratio * seq_len)), np.zeros(seq_len, bool), []
    while taken.sum() < target:
        length = max(1, int(rng.poisson(lam)))
        start = int(rng.integers(0, seq_len))
        if taken[start:start + length].any():
            continue
        spans.append((start, min(length, seq_len - start)))
        taken[start:start + length] = True
    return sorted(spans)

def corrupt(tokens, spans, rng=None):
    """Return (part_a, part_b). Part A collapses each span to [MASK];
    Part B = spans, SHUFFLED, each wrapped in [S] ... [E]."""
    rng = rng or np.random.default_rng(0)
    masked = {i for (s, l) in spans for i in range(s, s + l)}
    part_a, b_spans, i = [], [], 0
    while i < len(tokens):
        hit = next((sp for sp in spans if sp[0] == i), None)
        if hit:
            part_a.append("[MASK]")
            b_spans.append(["[S]"] + tokens[i:i + hit[1]] + ["[E]"])
            i += hit[1]
        elif i in masked:
            i += 1
        else:
            part_a.append(tokens[i]); i += 1
    rng.shuffle(b_spans)                                   # span order randomized
    return part_a, [t for span in b_spans for t in span]
```

Two design decisions in that code are easy to skim past and both matter. First, **the span order in Part B is shuffled**. The model never gets to rely on "the first blank comes first"; it must learn the dependency between a `[MASK]` in Part A and its reconstruction in Part B from content, not position. Second, **the `[END]` token must be generated**. Because each span terminates with an `[E]` the model itself produces, the model never knows in advance how long a span is — it decides when to stop. That single fact is what makes this a *generative* objective and not a fixed-width fill.

It's worth dwelling on the two hyperparameters here, because they're the kind of choice that looks arbitrary until you see the reasoning. The **15% masking ratio** is inherited straight from BERT — it's the well-worn "mask enough that the task is hard, little enough that there's context to reconstruct from" sweet spot, and GLM had no reason to relitigate it. The **Poisson(3)** span-length distribution is the more interesting choice. A geometric or uniform distribution would also produce variable spans, but Poisson(3) concentrates probability mass on spans of length 2–4 while leaving a long tail of occasionally-longer spans. That shape matters: most of the learning signal is on short, local reconstructions (which is where most useful linguistic structure lives — agreement, collocation, short-range syntax), but the tail forces the model to occasionally reconstruct a longer phrase from context, which is what builds the generative muscle. If you set λ too high, every example becomes a mini-summarization task and the model never learns crisp local infilling; too low and it degenerates toward single-token BERT masking and loses its generative edge. Poisson(3) is the dial set to "mostly local, sometimes not."

One more subtlety the code makes explicit: spans are sampled *without overlap* and the loop keeps drawing until the coverage target is hit. This means the number of spans per sequence is itself a random variable — a short document might get two `[MASK]`s, a long one a dozen — and the model has to handle a variable number of blanks per example. That variability is a feature: it prevents the model from learning a fixed "there are always K blanks" prior, the same way the shuffled order prevents a "first blank is leftmost" prior.

The loss only lands on Part B. During training you run the whole concatenated sequence through the transformer once, but you compute cross-entropy *only* on the Part B positions — the reconstructed span tokens and the `[END]` markers. Part A tokens get no loss; they exist purely as the bidirectional context the decoder conditions on. This is the same idea as masking the prompt tokens out of the loss in modern instruction tuning, and it's load-bearing for the same reason: you don't want to spend gradient teaching the model to "predict" tokens it was handed as input. A subtle consequence is that the *effective* number of supervised tokens per sequence is only ~15% of the original length (the masked fraction), which makes blank infilling somewhat less sample-efficient per forward pass than next-token prediction, where every position is supervised. GLM accepts that cost in exchange for the bidirectional context — a trade that looks expensive at small scale and pays off as the model grows and the quality of each gradient matters more than the quantity.

### The hybrid attention mask

The corruption gives you the data; the attention mask gives you the model. This is the load-bearing trick of the entire paper.

![A matrix of the hybrid attention mask, with Part A fully bidirectional and Part B autoregressive](/imgs/blogs/autoregressive-blank-infilling-glm-3.png)

Read each row of the matrix above as "what this query token may attend to":

- **Top-left block (Part A → Part A), all green.** Part A tokens attend to *every* other Part A token, in both directions. This is exactly BERT: a bidirectional encoder over the visible context.
- **Top-right block (Part A → Part B), all masked.** Part A tokens *cannot* see Part B. The context is never allowed to peek at the answer it's supposed to help reconstruct.
- **Bottom-left block (Part B → Part A), all green.** Every Part B token attends to *all* of Part A. The decoder gets the full bidirectional context for free, which is the source of GLM's strong conditioning.
- **Bottom-right triangle (Part B → Part B), lower-triangular.** Within Part B, token *i* sees only tokens ≤ *i*. This is exactly GPT: an autoregressive decoder.

So the upper-left quadrant of the matrix is an encoder and the lower-right quadrant is a decoder, and they share one set of weights and one forward pass. The construction is a single boolean matrix:

```python
def hybrid_mask(len_a, b_span_lens):
    """len_a = #Part-A tokens; b_span_lens = lengths of each Part-B span
    (incl [S] and [E]). True = query (row) may attend to key (col)."""
    n = len_a + sum(b_span_lens)
    m = np.zeros((n, n), dtype=bool)
    m[:len_a, :len_a] = True                  # Part A: bidirectional
    off = len_a
    for L in b_span_lens:
        m[off:off + L, :len_a] = True          # Part B sees all of Part A
        for i in range(L):
            m[off + i, off:off + i + 1] = True # autoregressive within span
        off += L
    return m                                    # top-right A->B stays False
```

If you have ever implemented a causal mask for a GPT and a full mask for a BERT, this is just both at once, partitioned by the Part A / Part B boundary. There is no architectural novelty in the transformer block itself — the novelty is entirely in *which entries of the attention matrix you allow*.

This is worth sitting with, because it's the source of GLM's parameter efficiency. A T5-style encoder-decoder achieves the same understand-and-generate capability by maintaining *two* transformer stacks — an encoder that runs bidirectional attention and a decoder that runs causal attention plus cross-attention back to the encoder. That's three distinct attention patterns across two parameter sets. GLM collapses all of it into one stack with one parameter set, and recovers the encoder/decoder distinction purely from the block structure of a single mask. There is no cross-attention module, because Part B attending to Part A *is* the cross-attention — it's just an off-diagonal block of the same attention matrix. So GLM is, in a precise sense, an encoder-decoder that shares its encoder and decoder weights and fuses them into one forward pass. That sharing is why a same-size GLM can beat BERT on understanding *and* do seq2seq without the parameter tax that made T5 expensive.

The flip side — the cost you pay for the elegance — is that the effective sequence the model processes is *longer* than the original text, because Part B re-lists the masked tokens after Part A. For a 15%-masked sequence the overhead is modest, but for the document-level regime (where Part B can be most of the text) the model is processing close to 2× the tokens. At training time that's an acceptable cost; it's one reason the later GLMs lean on efficient attention and aggressive [KV-cache management](/blog/machine-learning/large-language-model/kv-cache) once they scale.

### 2D positional encoding: staying blind to span length

There is a subtle problem the corruption creates. A `[MASK]` in Part A is a single token, but it expands into a variable-length span in Part B. How do positions line up so that the model can place a reconstructed span back where it belongs — *without* leaking how long the span is?

GLM's answer is **two position ids per token**:

- **Position 1 (inter-position)** is the token's index in the *corrupted Part A*. Every token of a reconstructed span in Part B shares the position id of the single `[MASK]` it fills. So a span knows *where* it belongs in the original text.
- **Position 2 (intra-position)** is 0 for all of Part A; within each Part B span it counts 1, 2, 3, … So a token knows *where it is inside its own span*.

The crucial property: nothing in the positional signal encodes the span's total length. A span of length 2 and a span of length 7 look identical from position 1 (they share their `[MASK]`'s index) and their position 2 simply counts up until the model emits `[E]`. The model is *structurally blind* to span length, which is exactly what you want for open-ended generation. Contrast this with a naive cloze setup where you pre-allocate N blanks — there, the model always knows it has exactly N slots to fill, and it has learned a fixed-length task. GLM never learns a length.

This is the detail most worth carrying forward. When GLM-130B later mixes a short-span `[MASK]` objective (30% of data) with a long generative `[gMASK]` suffix objective (70%), it is leaning on exactly this length-blindness: `[gMASK]` says "generate from here to as long as you like," and the model can do that because it was never trained to expect a fixed number of tokens.

A concrete worked example makes the two-id scheme click. Take the input `x1 x2 x3 x4` and mask the span `(x2 x3)`. Part A becomes `[x1, [MASK], x4]` and Part B becomes `[[S], x2, x3, [E]]`. Now lay out the position ids for the concatenated sequence `x1 [MASK] x4 [S] x2 x3 [E]`:

| Token | x1 | [MASK] | x4 | [S] | x2 | x3 | [E] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Position 1 (inter) | 1 | 2 | 3 | 2 | 2 | 2 | 2 |
| Position 2 (intra) | 0 | 0 | 0 | 1 | 2 | 3 | 4 |

Read the `[S] x2 x3 [E]` columns: their **position 1 is all 2** — the index of the `[MASK]` they're reconstructing — so the model knows this span belongs *where the second token of Part A was*. Their **position 2 counts 1, 2, 3, 4**, telling each token where it sits inside the span. Nothing here says "this span is 4 tokens long"; the model discovers that only when it generates `[E]`. Swap in a 7-token span and the position-1 values would all still be 2, and position-2 would just count higher. The scheme is, by construction, a function that is invariant to span length — which is the mathematical statement of "the model is blind to how much it will generate."

### One objective, three span regimes

A single masking scheme would bias the model toward one kind of task. GLM avoids that by *varying the span statistics* while keeping the objective identical.

![A tree showing the blank-infilling objective branching into token, sentence, and document regimes](/imgs/blogs/autoregressive-blank-infilling-glm-4.png)

The tree above shows the three regimes, all the *same* loss with different sampling:

- **Token-level** (the default): short spans, Poisson(3), 15% coverage. Trains the BERT-like infilling that powers understanding.
- **Sentence-level**: spans are constrained to be *complete sentences*, sampled to cover ~15% of tokens. This trains seq2seq behavior — summarization, paraphrase — because reconstructing a whole sentence from context is exactly that task.
- **Document-level**: a *single* span covering 50–100% of the text. This trains long, free-form generation, and is the direct conceptual ancestor of GLM-130B's `[gMASK]`.

The reusable idea here is one the whole lineage repeats: **the pretraining objective is a mixture, and the mixture weights are a design lever.** Tilt toward document-level and the model leans generative; tilt toward token-level and it leans toward understanding. You are not choosing an objective so much as choosing a *distribution over span shapes*, and that distribution is a knob you tune for the capability profile you want. The same instinct shows up far downstream when GLM-4 mixes reasoning and non-reasoning SFT data — see the [lineage survey](/blog/machine-learning/large-language-model/glm-lineage-frontier-llm-technique) for how this thread runs all the way to hybrid reasoning.

### Architecture changes around the objective

The transformer backbone is mostly standard, but the paper makes a few changes that matter for stability and that later editions kept or evolved:

- **Reordered LayerNorm and residual connection.** The order of the LayerNorm and the residual add is rearranged. The authors flag this as essential for numerical stability when scaling — a foreshadowing of the much more elaborate stability machinery (DeepNorm, embedding-gradient-shrink) that [GLM-130B](/blog/machine-learning/large-language-model/glm-lineage-frontier-llm-technique) would need at 130B.
- **A single linear output layer** for token prediction, rather than BERT's larger projection head.
- **GeLU activation** in place of ReLU.

None of these is exotic, but together with the objective they make GLM a clean, self-contained recipe rather than a research prototype. The model sizes in the paper run from GLM-Base (110M) up through GLM-Large (~340M, "1.25× BERT-Large") to a 10B variant, trained on BookCorpus + Wikipedia (small models) and RoBERTa-scale corpora (~158 GB) for the larger ones.

The reordered-LayerNorm detail deserves a moment, because it's the first appearance of a theme that becomes a *major* subplot two years later. Transformer stability is exquisitely sensitive to where exactly you put the LayerNorm relative to the residual connection — Pre-LN, Post-LN, and "Sandwich" variants train very differently, and at scale the difference between them is the difference between a converging run and a diverging one. The 2021 paper only needed a modest tweak because its models were small. But the same sensitivity, at 130B parameters and FP16 precision, would force the GLM-130B team into an elaborate stability stack (DeepNorm, embedding-gradient-shrink, FP32 attention softmax) that is the subject of the next article in this series. The seed of that whole saga is right here, in a one-line remark that reordering the norm and residual was "essential for stability." When a paper flags a small architectural choice as load-bearing, it's usually because the authors already watched the alternative blow up.

It's also worth noting what GLM *kept* standard, because the discipline of changing only what needs changing is part of why the recipe is reproducible. The attention mechanism is vanilla multi-head attention — no linear attention, no exotic sparsity. The optimizer is standard Adam-family. The tokenizer is conventional. The only genuinely novel pieces are the objective, the mask, and the 2D positions; everything else is deliberately boring. That restraint is a feature: it means a reader can attribute GLM's results to the three new ideas rather than to a pile of confounded changes, and it means the objective can be dropped into an otherwise-ordinary transformer codebase without a rewrite.

### Finetuning is just choosing the blank

The last piece of the method is the most elegant, and it is what makes GLM a *general* language model rather than a clever pretraining trick.

![A before-after showing three task-specific models replaced by one GLM that reformats each task](/imgs/blogs/autoregressive-blank-infilling-glm-5.png)

The before-after above is the payoff in one picture: instead of maintaining BERT for understanding, GPT for generation, and T5 for seq2seq — three architectures, three training runs — you keep *one* GLM and change the *task formatting*. The same weights answer a classification query and write a summary; only the cloze you wrap the input in changes.

![A grid showing classification, summarization, and QA each reformatted as a blank to fill](/imgs/blogs/autoregressive-blank-infilling-glm-6.png)

The grid above shows the move: **every downstream task is rewritten as a blank to fill**, so there is no task-specific head to bolt on.

- **Classification** becomes a cloze. For sentiment, append a templated continuation like *"It was really `[MASK]`."* and compare the probability the model assigns to filling `[MASK]` with "good" versus "bad". The classifier is the language model's own output distribution over the label words — no new parameters, no linear probe.
- **Conditional generation** (summarization, translation) appends a `[gMASK]` and lets the model decode the target span autoregressively.
- **Blank-filling QA** puts the question in context and a `[MASK]` where the answer goes.

This unification is why the paper can claim a single model is competitive across understanding *and* generation: it never leaves the pretraining objective's vocabulary of operations. Finetuning is not "add a head and hope the features transfer"; it is "phrase your task the way the model already thinks." The cost is some prompt engineering of the cloze template (a real cost — the choice of label words and template wording affects accuracy), but the benefit is a model that does NLU and NLG with literally the same machinery.

It's worth being concrete about that cost, because it's the part most likely to bite you in practice. For the sentiment example, "good" versus "bad" is an obvious label-word pair — but is it the *best* one? "great"/"terrible", "positive"/"negative", and "love it"/"hate it" all encode the same binary, and the model assigns them different probabilities; the wrong choice can cost several points of accuracy. The template wording matters too: *"It was really [MASK]."* primes a different distribution than *"All in all, [MASK]."*. This is why the GLM paper, and the prompt-based learning literature it sits alongside, treats label-word and template selection as a tuned hyperparameter, sometimes with multiple templates ensembled. The deeper point is that GLM didn't make the difficulty of NLU disappear — it *relocated* it, from "design and train a classification head" to "design a good cloze." Whether that's a win depends on your situation: it's a clear win when labeled data is scarce (a cloze leverages the pretrained distribution directly, so it works few-shot), and more of a wash when you have abundant labels and a linear head would train fine. The honest framing is that blank-infilling finetuning trades an *engineering* problem (head design) for a *prompt-design* problem, and the prompt-design problem happens to be the one that plays to a language model's strengths.

## Experiments: does one model really beat three specialists?

The headline claim is that GLM matches or beats the specialists at the same size and data. The numbers, on 2021 benchmarks:

| Task / benchmark | GLM | Best specialist baseline | Note |
| --- | --- | --- | --- |
| SuperGLUE (NLU), GLM-Large | **77.0** | BERT-Large 72.0 | +5.0 at equal size/data |
| Text infilling (Yahoo, 30% mask), BLEU | **64.2** | BLM 59.6 | the objective's home turf |
| LM (LAMBADA), GLM-515M | beats GPT-Large | GPT-Large | bidirectional Part A helps |
| CNN/DailyMail summarization, ROUGE-1/2/L | 43.8 / 21.0 / 40.5 | BART / T5-class | competitive seq2seq |
| XSum summarization, ROUGE-1/2/L | 45.5 / 23.5 / 37.3 | BART / T5-class | competitive seq2seq |

![A matrix comparing GLM to the best same-size baseline on SuperGLUE, LAMBADA, and summarization](/imgs/blogs/autoregressive-blank-infilling-glm-7.png)

The matrix above isolates the head-to-head that matters: GLM versus *the strongest specialist at the same size* on each task. Crucially, it's the same GLM in every row — one model, three tasks — facing a *different* specialist each time. That framing is the real claim. It's not hard to build a model that ties BERT on understanding if you're allowed to lose to GPT on generation; the difficulty is doing both with one set of weights, and the column of green cells is the evidence GLM does.

The most convincing cell is SuperGLUE: **GLM-Large beats BERT-Large by 5 points** on pure understanding, at the same parameter count and pretraining data. That is GLM winning on BERT's own turf — understanding — *despite* also being a generator. The summarization rows show the other half: GLM is competitive with BART and T5-class seq2seq models without the encoder-decoder parameter tax. And the LAMBADA result is a tidy illustration of *why* bidirectional Part A helps even for a generation-flavored benchmark: predicting the final word of a passage benefits from reading the whole passage, which GLM's encoder half does and a pure causal model cannot.

**What's load-bearing in their setup that might not transfer.** Three caveats worth stating plainly:

1. **Scale.** These are sub-1B and 10B models on 2021 corpora. The objective's biggest payoffs (a stable 130B, a strong agentic 355B) come later; the original paper can only show that the idea *works*, not how far it scales.
2. **Cloze template sensitivity.** The classification-as-cloze trick depends on the choice of label words and template. The paper engineers these; a careless template gives up accuracy. This is a real operational cost the headline numbers don't surface, and it's worth budgeting for if you plan to reproduce the understanding results: a chunk of GLM's SuperGLUE margin lives in the template, not only the weights.
3. **English-centric benchmarks.** The bilingual and multilingual strengths that would later define GLM-130B and GLM-4 aren't on display here. This matters because the objective is, in principle, language-agnostic — blank infilling has nothing English-specific about it — so the 2021 evaluation arguably *understates* the idea's reach. The team clearly believed this too, which is why the very next model in the lineage was bilingual at 130B scale rather than a larger English-only GLM.

## Why the objective outlived its benchmarks

The 2021 results are solid but not earth-shaking — a few points here, competitive ROUGE there. If that were the whole story, blank infilling would be a footnote. What makes the paper important is that the *mechanism* it introduced kept paying dividends long after the benchmarks it was measured on stopped mattering.

![A timeline showing the [END]-terminated span idea threading from 2021 GLM to 2025 hybrid reasoning](/imgs/blogs/autoregressive-blank-infilling-glm-8.png)

The timeline above traces one specific idea — *variable-length, `[END]`-terminated generation* — from this paper forward through the lineage:

- **2021, GLM:** a masked span is reconstructed token-by-token until the model emits `[END]`. The model decides the length.
- **2022, GLM-130B:** the document-level regime becomes `[gMASK]`, applied to 70% of training. "Generate a long suffix of unknown length" is now the dominant objective, not a minority regime. This is length-blindness scaled up.
- **2024, GLM-4:** open-ended generation drives autonomous tool use — the model generates a tool call, reads the result, and continues for as many steps as the task needs. "Decide when to stop" has become "decide how many tool-use turns to take."
- **2025, GLM-4.5:** the model generates a variable-length *thinking* span and then a *answer* span — and the cold-start training that teaches this is, structurally, blank infilling with a long reasoning blank followed by a short answer blank.

The through-line is that **"the model controls its own output length, terminated by a learned signal"** is not a 2025 invention bolted onto reasoning models — it's the 2021 `[END]` token, scaled and repurposed four times. That is why the GLM team never abandoned the blank-infilling backbone even after the field's center of gravity moved to pure causal decoders: the objective gave them a native handle on variable-length generation that they kept finding new uses for. A mechanism that survives four reinventions is doing real work, and that durability — more than any 2021 benchmark — is the paper's actual contribution. The full arc is the subject of the [lineage survey](/blog/machine-learning/large-language-model/glm-lineage-frontier-llm-technique).

## Critique: the senior-engineer lens

![A before-after scoring GLM's five 2021 ideas by whether they survived to 2026](/imgs/blogs/autoregressive-blank-infilling-glm-9.png)

The scorecard above is the honest five-year retrospective, and it's the right frame for the critique: not "was the paper good" but "which of its specific ideas earned their keep." Two of the five — the blank-infilling objective and the span-regime mixture — are still load-bearing in the 2025 flagships. One — the hybrid mask — evolved into the causal-plus-FIM patterns the field standardized on. Two — 2D positional encoding and cloze-style finetuning — were genuinely *replaced* (by RoPE and by chat formatting, respectively). A 3-out-of-5 survival rate over five years of a fast-moving field is excellent; most 2021 modeling papers have a survival rate near zero. The critique below is organized around understanding *why* the survivors survived and the replacements didn't.

### What's strong

The objective is genuinely elegant and genuinely general. "Unify understanding and generation" is the kind of claim papers make and don't deliver; GLM delivers it with a single mask change and a positional trick, and the SuperGLUE result is the proof that it isn't just generation with extra steps — it's competitive understanding *and* generation in one model.

The finetuning-as-cloze reformulation is the part I'd hand to a junior engineer as a model of good abstraction design. The usual transfer-learning story is "pretrain a feature extractor, attach a task-specific head, hope the features are useful." That story has a seam: the head is new, randomly initialized, and the model has never produced its output format. GLM removes the seam entirely by making the downstream interface *identical* to the pretraining interface — a blank to fill. There is nothing to transfer because there is nothing new. When an abstraction makes a whole category of engineering problems (designing task heads, tuning their learning rates, dealing with their cold start) simply not exist, that abstraction is doing real work. This is also why GLM finetunes stably with small data: the model isn't learning a new output behavior, just being nudged toward a particular kind of blank-filling it already knows how to do.

A second strength, easy to miss, is *honesty about cost*. The paper doesn't pretend the encoder-decoder fusion is free; the longer effective sequence and the cloze-template sensitivity are real, and the authors engineer around them rather than hiding them. Papers that report their costs tend to have done the work.

### What's weak or unfalsifiable

The span-shuffling and multi-task-regime choices are presented with ablations, but the *interactions* between them are under-explored — how much of the win is the mask, how much is the 2D positions, how much is the document/sentence/token mixture? The paper gives you the recipe but not a clean decomposition of which ingredient buys which capability. For a practitioner trying to port the idea to a new domain (say, protein sequences or code), that decomposition is exactly what you'd want: which parts are essential and which are English-NLP-specific tuning?

The comparison baselines, while fair, are same-era. The paper can't tell you whether the objective's advantage holds as the alternatives also scale — a question that the 2023+ wave of pure causal decoders (Llama, DeepSeek) arguably answered by winning on raw scale, and that GLM-130B then answered back by scaling the objective itself. So the 2021 paper sits in an awkward evidentiary spot: it proves the objective *works*, but the question that actually matters — does it keep winning at scale — is left to its sequels.

### What ablation is missing

The one I most want: hold the architecture and data fixed and sweep *only* the document/sentence/token mixture ratio, reporting NLU and NLG separately. That would turn "the mixture is a lever" from an assertion into a curve, and it would give downstream teams a principled way to set the ratio for their workload instead of copying GLM's. GLM-130B's later 30/70 `[MASK]`/`[gMASK]` split is clearly the descendant of such tuning, but the original paper leaves the dose-response implicit — you can see that the team *had* the curve internally (the 130B split is too specific to be a guess) without being shown it.

### What would change my mind

If a same-size, same-data pure causal decoder with a modern (2023+) training recipe matched GLM-Large's SuperGLUE number *and* its summarization ROUGE, I'd downgrade the objective from "load-bearing" to "a nice early unification that scale later made optional." That experiment is eminently runnable today and I'd genuinely like to see it. The strongest evidence *against* that outcome is behavioral: the GLM team kept the blank-infilling backbone all the way to GLM-4.5, rather than quietly switching to a vanilla causal objective once they had scale and could afford to. Teams abandon clever-but-inconvenient ideas the moment they stop paying rent. Five years of retention is a revealed preference that the objective keeps earning its keep.

## What I'd build with this

![A tree of four extensions of blank infilling, each leaning on a different property of the objective](/imgs/blogs/autoregressive-blank-infilling-glm-10.png)

The tree above maps four projects I'd actually run, organized by *which property of the objective each one exploits* — because the most reliable way to extend an idea is to find the property that makes it special and push on exactly that. Length-blindness powers the code-infilling project; the mixture-as-a-lever insight powers the ratio sweep; the cloze interface powers the template tooling; and `[END]`-termination powers the hybrid-reasoning bridge. None of these requires inventing a new objective — they're all "take blank infilling seriously about one of its own properties."

1. **A length-controlled infilling model for code** — length-blindness fits code fill-in-the-middle, where the hole length is unknown ahead of time, and `[END]`-terminated spans handle *multiple* holes per file cleanly. Compare a small GLM-style coder to causal FIM on repo completion.
2. **A mixture-ratio sweep** — run the ablation the paper skipped: sweep the token/sentence/document ratio on a fixed budget and plot NLU against NLG, yielding a practical recipe for any given workload.
3. **Cloze-template robustness tooling** — a small loop that searches label words and template wording against a held-out dev set, turning the paper's manual template engineering into something reproducible.
4. **A bridge to hybrid reasoning** — train a tiny model to emit a long `[gMASK]` *thinking* span followed by a short `[MASK]` *answer* span. If that works at small scale, it's evidence that GLM-4.5's thinking/non-thinking switch is blank infilling wearing a new name.

## References

- **Paper:** [GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://arxiv.org/abs/2103.10360) (Du, Qian, Liu, Ding, Qiu, Yang, Tang; ACL 2022). [Code: THUDM/GLM](https://github.com/THUDM/GLM).
- **Series survey:** [The GLM Lineage: Five Years of Frontier-LLM Technique](/blog/machine-learning/large-language-model/glm-lineage-frontier-llm-technique) — where this objective sits in the whole arc.
- **Next in series:** Engineering GLM-130B — how the objective survived the jump to 130B parameters on FP16.
- **Related on this blog:** [choosing the right LLM architecture for a task](/blog/machine-learning/large-language-model/choosing-right-llm-architecture-task) and [designing and choosing a tokenizer](/blog/machine-learning/large-language-model/designing-choosing-tokenizer-llm).
