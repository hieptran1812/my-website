---
title: "DeepCode: Turning Papers Into Codebases by Treating Context as a Bandwidth Problem"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "A close read of DeepCode (HKUDS): how reframing paper-to-repository synthesis as a noisy-channel problem — blueprint distillation, stateful code memory, adaptive retrieval, and closed-loop verification — pushes PaperBench replication past commercial agents and PhD-level humans."
category: paper-reading
subcategory: AI Agent
tags:
  - agentic-coding
  - code-generation
  - llm-agents
  - retrieval-augmented-generation
  - agent-memory
  - paper-reproduction
  - context-engineering
  - paperbench
author: Hiep Tran
featured: true
readTime: 30
image: ''
excerpt: ''
---

> [!tldr]
> - **The claim.** Reproducing a machine-learning paper as a runnable codebase is not blocked by model intelligence — it is blocked by **information flow**. DeepCode (from HKUDS at the University of Hong Kong) reframes "document → repository" as a **channel-optimization** problem and orchestrates four information operations — blueprint distillation, stateful code memory (CodeMem), adaptive retrieval (CodeRAG), and closed-loop verification — to keep task-relevant signal alive inside a finite context window.
> - **Why it matters.** On OpenAI's **PaperBench Code-Dev**, DeepCode scores **73.5%** replication versus **43.3%** for the best LLM agent (o1 + IterativeAgent) and **51.1%** for the prior specialized system (PaperCoder).
> - **The surprising bit.** Running on the *same base model* as Cursor and Claude Code (Claude Sonnet 4.5-thinking), DeepCode scores **0.854** on a five-paper subset where those commercial agents land at **0.584** and **0.587**. The gap is architecture, not the model.
> - **The headline.** On the three-paper human subset, DeepCode averages **75.9%** against the **72.4%** best-of-three from eight ML PhD students at Berkeley, Cambridge, and CMU.
> - **Where it's soft.** "Surpasses humans" is measured on *Code-Dev* — static grading of code structure and fidelity, **not** actually running the experiments and matching the paper's numbers — over only three papers, by an LLM judge. The information-theory framing is an intuition pump, not a measured bound.

When people argue about whether AI can "do research," they usually argue about reasoning. Can the model follow the proof? Can it design the ablation? That framing misses where autonomous reproduction actually breaks. Hand a strong model a 30-page ICML paper — dense prose, a dozen equations, a hyperparameter table buried in an appendix, three figures whose captions carry load-bearing detail — and ask for the full codebase. The model is smart enough. It just cannot **hold the whole specification in view** while generating thirty interdependent files, and the moment it loses the thread, the repository drifts: a base class defined in file three is silently re-invented in file eleven, a loss function quietly omits the term that made the paper work, and the whole thing fails to import.

DeepCode's thesis is that this is fundamentally a **plumbing** problem, not an intelligence problem — and that the right abstraction is information theory. A scientific paper is a high-entropy source. The LLM's context window is a finite-bandwidth channel. Naively concatenating the paper with a growing pile of generated code saturates that channel with redundant tokens, the signal-to-noise ratio collapses, and replication craters. The diagram below is the mental model for the entire paper: everything DeepCode does is an attempt to widen, or better *use*, that channel.

![The doc-to-repo bottleneck: a high-entropy paper squeezed through a finite context window loses signal, producing four failure modes that cap replication near 42%.](/imgs/blogs/deepcode-open-agentic-coding-1.webp)

This is a paper-reading post, so I will do three things: walk the architecture carefully (with the symbols defined, because the paper leans on notation), report the numbers honestly (including the ones that deserve an asterisk), and then put on the senior-engineer hat and ask what I actually believe. Source: [DeepCode: Open Agentic Coding](https://arxiv.org/abs/2512.07921), Li, Li, Guo, Ren, and Huang, December 2025; [code on GitHub](https://github.com/HKUDS/DeepCode).

## Context: what came before

The field has been climbing a ladder of ambition. The first rung was **code completion** — Copilot-style inline suggestions that assume a human still owns architecture and validation. The next rung is **agentic software engineering**: LLM agents expected to plan, orchestrate, and refine an entire project from a natural-language or document-level specification. The question shifts from "write this function" to "*can an artificial agent behave as an autonomous engineer that translates rich, informal specifications into comprehensive, robust systems?*"

The natural stress test for that question is **document-grounded program synthesis**, and the most stringent instance is reproducing a research paper as a working codebase. This is genuinely hard, and the prior numbers say so. OpenAI's **PaperBench** evaluates frontier models on 20 ICML 2024 papers; the strongest configuration in that work — o1 with an "IterativeAgent" scaffold — reaches only **42.4%** replication, against **72.4%** for human PhD experts. **PaperCoder** (also called Paper2Code), a specialized three-stage planning/analysis/generation pipeline, reaches **51.1%** on PaperBench. These are not bad systems; they are evidence that reliable, end-to-end reproduction was, as of early 2025, an open problem.

DeepCode names four concrete failure modes that open up under a finite context budget, and they are worth keeping in mind because the whole architecture is organized around defeating them:

1. **Specification preservation.** Papers describe the target through scattered, multimodal constraints. Faithfully mapping that fragmented spec into an implementation is hard, and detail leaks.
2. **Global consistency under partial views.** Repositories are interdependent, but generation proceeds file by file under limited context. Interfaces, types, and invariants drift.
3. **Completion of underspecified designs.** Papers state the algorithmic core and leave implementation details implicit. The agent must infer them — consequential but non-trivial.
4. **Executable faithfulness.** Plausible code is not running code. Long-horizon generation accumulates logic bugs, dependency conflicts, and fragile glue that breaks end-to-end execution.

The gap DeepCode claims to fill: a single, principled mechanism — *information-flow management* — that addresses all four at once, rather than four bolted-on heuristics.

## The one idea: synthesis is a noisy channel

Here is the load-bearing reframe. The task is a mapping function $\mathcal{F}_{gen} : \mathbb{D} \rightarrow \mathbb{P}$ from the space of specification documents $\mathbb{D}$ to the space of valid code repositories $\mathbb{P}$. A document $\mathcal{D} = (d_1, d_2, \dots, d_L)$ is a long sequence of multimodal elements — text blocks, equations, tables, figures, pseudocode snippets — and $L$ is large enough to strain any context window. The output is not a file but a structured repository, defined as a tuple:

$$\mathcal{P} = (\mathcal{T}, \mathcal{C}, \mathcal{M})$$

where $\mathcal{T}$ is the directory tree, $\mathcal{C} = \{c_1, c_2, \dots, c_N\}$ is the set of source files (and getting them to interoperate correctly is the cross-file-consistency problem), and $\mathcal{M}$ is the dependency manifest (`requirements.txt`, `package.json`, `README.md`) that lets the thing actually run.

The optimization target is then written as

$$\mathcal{P}^{*} = \arg\max_{\mathcal{P} \in \mathbb{P}} \; \text{Score}(\mathcal{P} \mid \mathcal{D})$$

and — this is the move — the score is decomposed into a **signal-to-noise objective across a synthesis channel**. The paper calls the design principle *contextual information maximization*: at every generation step, the system must actively maximize the density of task-relevant signal while suppressing irrelevant noise. Naive concatenation does the opposite — redundant tokens (the entire raw paper, the entire growing code history) mask the critical algorithmic constraints and the effective signal-to-noise ratio collapses.

That single objective splits into the four sub-objectives that map one-to-one onto the four operations:

| Objective | What must be true | The operation that delivers it |
|---|---|---|
| **Specification preservation** | The rigid algorithmic constraints hidden in multimodal source survive into code | **Source compression** — extract a high-signal blueprint from unstructured noise |
| **Global structural consistency** | Modules keep interface and type coherence without context saturation | **Structured indexing** — abstract the evolving codebase into compact retrievable summaries |
| **Domain knowledge grounding** | Abstract descriptions become concrete, correct engineering | **Conditional knowledge injection** — pull standard patterns from external sources only when needed |
| **Functional executability** | The repository is robust and runnable | **Error correction** — treat runtime feedback as a corrective signal |

I want to flag, up front, that this is the paper's strongest *and* weakest contribution. As a **design lens** it is genuinely clarifying — it tells you *why* each component exists and gives you a single principle to reason about context budgets. As **theory** it is a metaphor dressed in notation: there is no measured mutual information, no rate-distortion curve, no quantitative "channel capacity" anywhere in the paper. The $\arg\max$ is a way of writing down an intuition, not a bound you can compute. Keep that distinction; we will return to it in the critique.

## The architecture: three phases, four operations

The four operations are realized across three sequential phases. Phase 1 compresses the source. Phase 2 generates code while routing information through two mechanisms (a stateful memory and an adaptive retriever). Phase 3 corrects. The figure below is the spine of the whole system — read it top-to-bottom as "what each phase does" and bottom row as "which information operation it performs."

![DeepCode is three sequential phases — Blueprint, Code generation, Verify — each performing a distinct information operation: compress, index-and-inject, and correct.](/imgs/blogs/deepcode-open-agentic-coding-2.webp)

The reason this ordering matters: each phase exists to keep the *next* phase's context clean. Blueprint distillation means the coding agents never touch the raw paper. CodeMem means the agent generating file $t$ never sees the raw source of files $1 \dots t-1$. Verification means the executable-faithfulness problem is handled by a feedback loop rather than by hoping the first draft runs. It is context engineering all the way down.

## Phase 1: distilling the blueprint

The first job is **source compression**: turn the lengthy, unstructured paper $\mathcal{D}$ into a structured, machine-readable implementation blueprint $\mathcal{B}$ that is dense enough to replace the original entirely. It happens in three steps.

**Hierarchical content segmentation.** Instead of feeding the whole document to an LLM, DeepCode first parses it into a hierarchical content index. Structural parsing splits the document on explicit delimiters — section and subsection headings like "3. Methodology", "3.1 Model Architecture" — into chunks $S = \{s_1, s_2, \dots, s_K\}$. Each chunk $s_k$ is stored as a key–value pair $(h_k, c_k)$, where the heading $h_k$ is a natural semantic key and $c_k$ is the raw text of that section. This converts a *long-context comprehension* problem into a series of *targeted retrievals*: an agent can request "Model Architecture" and pull only that section, focusing its limited window on the most pertinent content and sidestepping the context-forgetting problem entirely.

**Multi-agent specification analysis.** Two specialized agents read the indexed document in parallel, each extracting a complementary layer without processing the whole paper at once:

- The **Concept Agent** builds the high-level map — the paper's conceptual structure, core contributions, and the components needed for a successful reproduction. It queries the index with broad keywords ("introduction", "method") and emits a *Conceptual Analysis Schema*: a paper-structure map, a method-decomposition map, an implementation map aligning claims to code requirements, and a reproduction roadmap with success criteria.
- The **Algorithm Agent** does the meticulous low-level extraction — every algorithm, mathematical formulation, model architecture, training procedure, and hyperparameter. It queries with technically dense keywords ("algorithm", "hyperparameter") and, notably, can perform **online search** to fetch reference implementations from the web. Its output is an *Algorithmic Implementation Schema*: verbatim pseudocode, exact equations and variables, layer-by-layer architectures, and a complete hyperparameter list with source locations.

**Blueprint synthesis.** A **Code Planning Agent** reconciles the conceptual overview with the granular technical spec, resolving inconsistencies via targeted index queries, into a single self-contained blueprint $\mathcal{B}$. The figure below shows that fork-then-join shape — two parallel readers feeding one planner that emits a five-part artifact.

![Phase 1 forks the paper into a Concept track and an Algorithm track, then a planner fuses them into one canonical five-part blueprint.](/imgs/blogs/deepcode-open-agentic-coding-3.webp)

The blueprint $\mathcal{B}$ is organized into five canonical sections, and this structure is what lets the coding phase never look at the paper again:

| Blueprint section | What it pins down |
|---|---|
| **Project File Hierarchy** | The prioritized directory layout and the *order* modules should be implemented |
| **Component Specification** | Per-module/class/function spec, each mapped to its pseudocode and math |
| **Verification Protocol** | The experimental setup, target metrics, and success criteria for reproduction |
| **Execution Environment** | Exact dependencies, library versions, and hardware needed to build and run |
| **Staged Development Plan** | A phased build order with staged verification checks for modular correctness |

This is the "source of truth" that downstream agents consume. The entire long-context challenge is, in principle, dissolved here: a dense, structured, actionable artifact obviates any need to re-read the original document.

## Phase 2: stateful generation with CodeMem

Now the system synthesizes the repository file by file. The danger is obvious and is exactly the failure DeepCode is built to avoid: a naive iterative loop appends each newly generated file's full source to the prompt for the next file, the context balloons, the signal-to-noise ratio collapses, and the model starts hallucinating interfaces that don't exist. DeepCode's answer is a **stateful Code Memory (CodeMem)** that indexes the evolving repository into compact summaries instead of raw code.

The generation is a loop over $t = 1, \dots, N$. At each step the system maintains the set of implemented files $\mathcal{C}_{t-1}$ and unimplemented files $\mathcal{U}_{t-1}$, and generates the current target file $\hat{c}_t$ (the blank file to be filled; $c_t$ is the result). Three sub-steps:

1. **Context formulation.** The generation context is built *not* from raw source but from the blueprint $\mathcal{B}$ plus a dynamically selected slice of memory:
$$\mathcal{X}_t = \big(\mathcal{B}, \; \text{SelectRelevantMemory}(\mathcal{M}_{t-1}, \hat{c}_t)\big)$$
The selector queries the memory bank $\mathcal{M}_{t-1}$ for just the summaries of files the current target depends on.
2. **Code generation.** The coding agent — the LLM function $\mathcal{L}$ — synthesizes the file from the curated context: $c_t = \mathcal{L}(\mathcal{X}_t)$.
3. **Memory update.** A specialized summarization agent $\mathcal{S}$ distills the freshly written $c_t$ into a structured memory entry $m_t$, and the bank grows: $\mathcal{M}_t = \mathcal{M}_{t-1} \cup \{m_t\}$.

The crucial property is in the animation below: the **memory bank grows** with each file, but the **live context window stays a fixed size** — only the blueprint and the handful of relevant summaries are ever in context. That is the signal-to-noise ratio being held high by construction, file after file.

<figure class="blog-anim">
<svg viewBox="0 0 760 384" role="img" aria-label="Iterative file generation: a highlight sweeps file by file while the code-memory bank fills with one compact summary per completed file, yet the context window stays a fixed size" style="width:100%;height:auto;max-width:820px">
<style>
.dc-cell{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.dc-mem{fill:var(--accent,#6366f1);stroke:none}
.dc-sweep{fill:var(--accent,#6366f1);opacity:.22}
.dc-ctx{fill:var(--surface,#f3f4f6);stroke:var(--accent,#6366f1);stroke-width:2}
.dc-t{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.dc-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.dc-mlbl{font:600 12px ui-sans-serif,system-ui;fill:#fff;text-anchor:middle}
.dc-sub{font:13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
@keyframes dc-move{0%{transform:translateX(0)}100%{transform:translateX(390px)}}
@keyframes dc-m1{0%,18%{opacity:.13}25%,100%{opacity:1}}
@keyframes dc-m2{0%,43%{opacity:.13}50%,100%{opacity:1}}
@keyframes dc-m3{0%,68%{opacity:.13}75%,100%{opacity:1}}
@keyframes dc-m4{0%,93%{opacity:.13}100%{opacity:1}}
.dc-anim{animation:dc-move 8s steps(4,jump-none) infinite}
.dc-c1{animation:dc-m1 8s ease-in-out infinite}
.dc-c2{animation:dc-m2 8s ease-in-out infinite}
.dc-c3{animation:dc-m3 8s ease-in-out infinite}
.dc-c4{animation:dc-m4 8s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.dc-anim{animation:none}.dc-c1,.dc-c2,.dc-c3,.dc-c4{animation:none;opacity:1}}
</style>
<text class="dc-t" x="20" y="28">Phase 2 — CodeMem: bounded context, growing memory</text>
<text class="dc-sub" x="40" y="56">files to implement (one per step)</text>
<text class="dc-sub" x="575" y="56">code memory M (grows)</text>
<rect class="dc-cell" x="40"  y="66" width="110" height="54" rx="8"/>
<rect class="dc-cell" x="170" y="66" width="110" height="54" rx="8"/>
<rect class="dc-cell" x="300" y="66" width="110" height="54" rx="8"/>
<rect class="dc-cell" x="430" y="66" width="110" height="54" rx="8"/>
<rect class="dc-sweep dc-anim" x="40" y="66" width="110" height="54" rx="8"/>
<text class="dc-lbl" x="95"  y="99">c1</text>
<text class="dc-lbl" x="225" y="99">c2</text>
<text class="dc-lbl" x="355" y="99">c3</text>
<text class="dc-lbl" x="485" y="99">c4</text>
<rect class="dc-mem dc-c1" x="575" y="66"  width="160" height="46" rx="7"/>
<rect class="dc-mem dc-c2" x="575" y="120" width="160" height="46" rx="7"/>
<rect class="dc-mem dc-c3" x="575" y="174" width="160" height="46" rx="7"/>
<rect class="dc-mem dc-c4" x="575" y="228" width="160" height="46" rx="7"/>
<text class="dc-mlbl" x="655" y="94">m1: purpose + API</text>
<text class="dc-mlbl" x="655" y="148">m2: + dep edges</text>
<text class="dc-mlbl" x="655" y="202">m3: + dep edges</text>
<text class="dc-mlbl" x="655" y="256">m4: + dep edges</text>
<text class="dc-sub" x="40" y="306">generate c_t  -&gt;  summarize  -&gt;  add one entry to M</text>
<rect class="dc-ctx" x="40" y="322" width="495" height="46" rx="8"/>
<text class="dc-lbl" x="287" y="351">context = blueprint + selected summaries (fixed size)</text>
</svg>
<figcaption>Each step generates one file, distills it to a compact memory entry, and appends it to the code memory M. The memory bank grows file by file, but the live context window stays a fixed size — only the blueprint plus the few summaries relevant to the current file are ever in context.</figcaption>
</figure>

What makes the summary load-bearing is its **structure**. The entry $m_t$ is not a free-text blurb; it is a typed object capturing exactly what inter-module communication needs:

```python
from dataclasses import dataclass, field

@dataclass
class MemoryEntry:
    """One CodeMem record per generated file — compact, but enough to
    keep the whole repository globally consistent without re-reading code."""
    core_purpose: str                 # P_t: one-line responsibility of the file
    public_interface: list[str]       # I_t: signatures, e.g. "TransformerBlock(d_model, n_heads): forward(x)"
    afferent_couplings: list[str]     # E_t (in):  what this file imports — internal + external deps
    efferent_couplings: list[str]     # E_t (out): which not-yet-written modules will consume this interface
    # Note: the "next file to implement" decision is kept OUT of the memory entry
    # and passed separately, so planning signal never pollutes the dependency index.
```

The dependency edges $\mathcal{E}_t$ are the clever part. Each entry records both **afferent couplings** (what this file imports — internal modules and external packages) and **efferent couplings** (which *unimplemented* modules are predicted to consume this file's public interface). That forward-looking edge is what lets a file written at step 3 correctly anticipate the contract a file written at step 11 will need. The generation loop, with all the context routing made explicit:

```python
def deepcode_generate(blueprint, files_in_build_order, L, S):
    """Phase 2 core loop. L = coding LLM, S = summarization agent.
    Context size stays ~constant in N; only memory bank M grows."""
    M = MemoryBank()                 # structured summaries, NOT raw code
    repo = {}
    for c_hat in files_in_build_order:        # order comes from the blueprint
        # 1. Context formulation: blueprint + ONLY the relevant summaries
        relevant = M.select_relevant(target=c_hat)     # query by dependency edges
        X_t = build_context(blueprint, relevant)        # <- bounded, high-signal

        # 2. Generation
        c_t = L.generate(X_t, target_file=c_hat)

        # 3. Memory update: clear context, distill c_t into one typed entry
        m_t = S.summarize(c_t)        # core_purpose / public_interface / dep edges
        M.add(m_t)
        repo[c_hat] = c_t
    return repo
```

This stateful, summary-based scheme is the answer to the **global-consistency** objective: a large repository stays logically cohesive because every file is generated against an accurate, compact picture of everything it touches — without ever paying the token cost of the full source.

### Why summaries beat raw history — and the number that proves it

The ablation here is the cleanest evidence in the paper that this is the right design. DeepCode compares CodeMem against a "Simple" baseline that naively evicts historical messages with a sliding window when the context approaches its limit. The result is exactly the failure the architecture predicts: unstructured eviction causes context saturation with **signal loss** — foundational class definitions get truncated *before* the dependent code that needs them is generated.

![Naive code-history concatenation saturates context and truncates dependencies; structured summaries preserve them — restoring scores from 0.33–0.43 to 0.70–0.92.](/imgs/blogs/deepcode-open-agentic-coding-5.webp)

Concretely: the Simple protocol scores only **0.33–0.43** on the `rice`, `fre`, and `mechanistic-understanding` tasks, precisely because those reproductions depend on long dependency chains that the sliding window severs. CodeMem's structured indexing restores those to **0.70–0.92**. Even on tasks where the baseline is already strong, structured memory delivers consistent gains — `test-time-model-adaptation` goes from 0.62 to 0.72, `all-in-one` from 0.66 to 0.76. The thesis holds: effective agentic coding requires *explicit information-flow management* to keep the signal-to-noise ratio high under context constraints, not just a bigger window or a smarter eviction heuristic.

## Phase 2, continued: knowledge grounding with CodeRAG

CodeMem keeps the repository *internally* consistent, but it does nothing about two other failures: model **hallucination** and the **omission of implicit domain knowledge**. Papers describe the algorithmic core and leave the standard engineering implicit — the exact way to wire a particular attention variant, the canonical call into a library, the idiomatic training-loop scaffolding. **CodeRAG** grounds generation in a pre-indexed corpus of relevant, high-quality code repositories.

It has two stages. **Repository indexing** ($\mathcal{I}_{\text{index}} : \mathcal{R} \times \mathcal{B} \rightarrow \mathcal{J}$) builds a queryable index $\mathcal{J}$ from a set of source repositories $\mathcal{R}$ (which can be the repos cited in the target paper's references, or repos found via online search):

1. **Relevance filtering** — for each repo, an LLM filter keeps only the source files $\mathcal{C}'_k \subset R_k$ most relevant to the blueprint's structure, focusing compute on promising assets.
2. **Code understanding** — each relevant file is independently summarized (purpose, key concepts, public interfaces), the same way CodeMem summarizes generated files.
3. **Relationship mapping** — the core step: for each source-file summary, an agent maps it to one or more target files in $\mathcal{B}$, emitting relationship tuples. Each tuple is $(c'_s, \hat{c}_t, \tau, \sigma, \gamma)$: source file $c'_s$, target file $\hat{c}_t$, relationship type $\tau$, a confidence score $\sigma$, and $\gamma$ — actionable context like helpful snippets, usage suggestions, and implementation patterns.

The second stage, **adaptive retrieval**, is where the information-flow discipline shows up again — DeepCode does not blindly stuff retrieved code into every prompt. At each generation step a binary decision is made:

$$r_t = \delta(\mathcal{X}_t, \hat{c}_t), \qquad r_t \in \{0, 1\}$$

based on the complexity of the target file and how much detail the blueprint already provides. If $r_t = 1$, it queries $\mathcal{J}$ for the highest-confidence relationship tuples for $\hat{c}_t$ and augments the context:

$$\mathcal{X}'_t = \mathcal{X}_t \cup \{\text{Retrieve}(\mathcal{J}, \hat{c}_t)\}$$

and generates with the enriched context, $c_t = \mathcal{L}(\mathcal{X}'_t)$. If $r_t = 0$, retrieval is skipped — the blueprint and memory already suffice, so injecting external code would only add noise.

![CodeRAG decides per file whether to retrieve; only files the blueprint underspecifies pull an external implementation tuple, the rest generate from memory alone.](/imgs/blogs/deepcode-open-agentic-coding-6.webp)

```python
def coderag_step(X_t, c_hat, J, L, delta):
    """Adaptive retrieval: pay the noise cost of injecting external code
    ONLY when the target file is underspecified by blueprint + memory."""
    if delta(X_t, c_hat):                      # r_t = 1?
        tuples = J.query(c_hat)                 # (c'_s, c_hat, tau, sigma, gamma)
        best = max(tuples, key=lambda t: t.sigma)   # highest-confidence mapping
        X_t = X_t + render_pattern(best.gamma)      # gamma = snippets + usage notes
    return L.generate(X_t, target_file=c_hat)
```

The ablation for CodeRAG is the most interesting result in the paper for a practitioner, because it tells you *when* retrieval matters. Decoupled on **Gemini-2.5-Flash**, CodeRAG delivers up to a **70% relative gain**, breaking the base model's performance ceiling from the 0.35–0.38 range. But applied to a frontier model like **Claude Sonnet 4.5**, the gains are **negligible**. The interpretation is sharp: reasoning giants already encode enough implementation patterns in their weights, so external retrieval is redundant for them; cost-efficient models suffer from genuine *knowledge gaps*, and CodeRAG is the bridge. The authors frame this as **democratizing high-fidelity replication** — letting a cheap model punch above its weight. That is a more honest and more useful claim than "retrieval always helps."

## Phase 3: closed-loop verification

The first two phases produce structurally sound code; they do not guarantee it *runs*. Phase 3 is an error-correction mechanism that treats execution outcomes as corrective signals, in two sequential stages.

**Static analysis and refinement.** An Analysis Agent $\mathcal{A}_{\text{static}}$ inspects the repository $\mathcal{P}$ against the blueprint $\mathcal{B}$ and produces a structured report $\mathcal{R}_{\text{static}} = \mathcal{A}_{\text{static}}(\mathcal{P}, \mathcal{B})$ flagging two issue classes: **structural discrepancies** (integrity violations like missing files or empty zero-byte source files the blueprint required) and **code-quality deficiencies** (an LLM quality score $q(c_i)$ per file, flagging poor style or maintainability). A Modification Agent $\mathcal{A}_{\text{modify}}$ then iterates through each issue and applies a targeted, *line-level* fix via a programmatic interface inspired by the Language Server Protocol (LSP) — modeled as $\Phi_{\text{LSP}}$, taking a file $c_i$ and an instruction and producing a corrected $c'_i$ — yielding the statically refined repository $\mathcal{P}' = \mathcal{A}_{\text{modify}}(\mathcal{P}, \mathcal{R}_{\text{static}})$. Line-level edits matter: they avoid the regression risk of regenerating whole files.

**Sandbox execution and correction.** The refined repository is then dynamically tested in a secure, isolated sandbox. A Sandbox Agent first verifies the environment setup (the dependencies declared in the `README.md` / blueprint) and auto-provisions it. It then executes the main entry points using auto-generated test data, producing an execution trace $\mathcal{T}_j = \mathcal{E}_{\text{sandbox}}(\mathcal{P}'_j)$ at iteration $j$. If the trace contains errors ($\mathcal{T}_j^{\text{error}} \neq \emptyset$), the agent localizes the faulty files, generates a fix, and patches via the same LSP interface:

$$\mathcal{P}'_{j+1} = \Phi_{\text{LSP}}(\mathcal{P}'_j, \mathcal{T}_j^{\text{error}})$$

The loop continues until execution succeeds or a maximum iteration count is reached; the final verified output is $\mathcal{P}^{*} = \mathcal{P}'_J$. The figure shows the closed loop — run, diagnose, patch, repeat — that turns "plausible code" into "running code."

![Phase 3 is a closed feedback loop: execute, read the error trace, LSP-patch the faulty file, and re-run until the trace is empty.](/imgs/blogs/deepcode-open-agentic-coding-7.webp)

How much does this final pass buy? Across three test papers, automated verification yields consistent gains of **3.7–6.5%**, lifting scores from the 0.69–0.81 range to 0.73–0.84. It primarily corrects three residual error types: typos in variable names, missing dependencies, and wrong command-line arguments. The modest size of the gain is itself informative — it means the earlier phases already achieved technical correctness, and verification is the final polish that eliminates the small but consequential defects that prevent otherwise-sound implementations from running reliably.

## How well does it work?

The evaluation uses **PaperBench Code-Dev**, OpenAI's benchmark of 20 ICML 2024 papers. Each paper carries an author-approved rubric that decomposes reproduction into **8,316** gradable, hierarchically-weighted components, scored by **SimpleJudge** (an automated judge built on o3-mini). The **Replication Score** is the bottom-up aggregation of binary leaf-node pass/fail up the rubric tree to a single root score, averaged over **three independent trials** per paper to tame stochasticity. Critically, a **source-code blacklist** is enforced during runs: the agent cannot access the authors' original repositories or known third-party implementations, so solutions must come from algorithmic reasoning, not retrieval of the answer.

A note on what Code-Dev measures, because it matters for interpreting "beats humans": Code-Dev focuses on **structural and functional correctness** of the generated code — static correctness, dependency validity, project structure, and algorithmic fidelity to the paper. It **does not include post-submission reproduction** — that is, the grade is on the *code*, not on re-running the experiments and matching the paper's reported numbers.

With that established, the headline comparison across automated systems:

![On PaperBench replication score, DeepCode (73.5) clears the best LLM agent (43.3), PaperCoder (51.1), and the human best-of-three (72.4).](/imgs/blogs/deepcode-open-agentic-coding-8.webp)

| System | Category | Replication score | Notes |
|---|---|---|---|
| BasicAgent (Claude-3.5-Sonnet) | LLM agent | 35.4 ± 0.8 | best no-scaffold LLM |
| o1 + IterativeAgent | LLM agent | **43.3 ± 1.1** | best general LLM-agent config |
| PaperCoder (Paper2Code) | Scientific agent | 51.1 ± 1.4 | prior specialized SOTA |
| **DeepCode** | This paper | **73.5 ± 2.8** | +70% relative over the best LLM agent |
| Human PhD (best-of-3)* | Human baseline | 72.4 | *three-paper subset |

DeepCode's 73.5 is a **70% relative improvement** over the best LLM-agent baseline and a **22-point** absolute jump over PaperCoder. On the three-paper subset where the human baseline was measured, DeepCode averages **75.9 ± 4.5** against the humans' **72.4** — it competes with and slightly exceeds the *best of three attempts* from PhD students. (Hold the overlapping confidence interval in mind; we will.)

The most architecturally convincing result is the head-to-head against commercial agents on a five-paper subset, because DeepCode uses **the same base model** (Claude Sonnet 4.5-thinking) as both Cursor and Claude Code. Any difference is therefore attributable to architecture, not model strength:

| Model | `fre` | `rice` | `bam` | `pinn` | `mech-u` | **Avg** |
|---|---|---|---|---|---|---|
| Codex (GPT-5 Codex-high) | 0.4095 | 0.3645 | 0.1937 | 0.5382 | 0.4926 | 0.3997 |
| Claude Code (Claude Sonnet 4.5-think) | 0.6286 | 0.3787 | 0.3829 | 0.7233 | 0.8222 | 0.5871 |
| Cursor (Claude Sonnet 4.5-think) | 0.6344 | 0.4186 | 0.3779 | 0.7748 | 0.7148 | 0.5841 |
| **DeepCode (Claude Sonnet 4.5-think)** | **0.8435** | **0.7380** | **0.8530** | **0.9474** | **0.8888** | **0.8541** |

DeepCode wins **every column**. The `bam` task is the tell: Claude Code and Cursor land at ~0.38 while DeepCode hits 0.85 — a paper whose reproduction depends on exactly the cross-file dependency chains that CodeMem preserves and a sliding-window context loses. Same model, +46% average. That is the single number I would put in front of anyone skeptical that agent architecture matters more than model choice for long-horizon coding.

## The backbone matters more than the scaffold

If architecture is necessary, the model is still the ceiling. DeepCode was run with five backbones on the three-paper subset, holding the agent architecture and tooling constant to isolate model capability:

| Backbone | Replication range | Read |
|---|---|---|
| **Claude-4.5-Sonnet** | 0.72 – 0.82 | best or near-best everywhere; strongest on long, underspecified, multi-stage specs |
| **GPT-5** | 0.69 – 0.81 | tracks Claude closely; small edge on `stay-on-topic` (0.81 vs 0.72) |
| Claude-3.5-Sonnet | 0.48 – 0.57 | recovers the skeleton, gaps on finer procedural steps |
| Gemini-2.5-Pro | 0.44 – 0.73 | high variance across task types |
| DeepSeek-R1 | ≈ 0.29 | reproduces only fragments of the target workflows |

The stable ranking across heterogeneous specs is the point: **under a fixed agent architecture, the underlying model becomes the primary factor determining the ceiling on automatic paper-level reproduction.** Two practical takeaways. First, this is why the same-base-model commercial comparison above is so load-bearing — it controls for exactly this. Second, it sets up the paper's most important open problem: the whole system currently *requires* a frontier model, which is expensive and slow.

## Ablations: which piece carries the load

Pulling the three ablations together tells a coherent story about where the value lives:

| Component | What it fixes | Gain | When it matters most |
|---|---|---|---|
| **CodeMem** | cross-file consistency; dependency truncation | 0.33–0.43 → 0.70–0.92 on dependency-heavy tasks | always; this is the structural backbone |
| **CodeRAG** | knowledge gaps; hallucinated patterns | up to **70%** relative on Gemini-2.5-Flash; ≈0 on Claude 4.5 | weak/cheap models, underspecified files |
| **Verification** | typos, missing deps, wrong CLI args | +3.7–6.5% | final-mile executable faithfulness |

Read it as a layered defense: CodeMem carries the load for *structure*, CodeRAG selectively patches *knowledge* (and matters far more the weaker your model is), and verification is the *final-mile* polish on executability. The components are not redundant — each targets a different one of the four failure modes from the introduction.

## Where this sits in the agentic-coding landscape

It helps to place DeepCode against the prior art, because its differentiator is *what it optimizes*, not the fact that it is a multi-agent system. The field has converged on a few recurring patterns:

| Approach | Representative systems | Organizing principle | What DeepCode does differently |
|---|---|---|---|
| **Org-structure mimicry** | ChatDev, MetaGPT, CodePoRi | Simulate a software company (CEO/CTO/dev roles) to manage tasks | Organizes around *information operations*, not human job titles |
| **Test-driven refinement** | AgentCoder, MapCoder | Programmer + test-designer + test-executor loops; example-retrieval → plan → generate → debug | Verification is one of four legs, not the whole loop; adds compression + memory + retrieval upstream |
| **Tooling / interface agents** | CodeAgent, SWE-agent, ToolGen | Give the agent richer tools (domain tools, an agent–computer interface) | Treats the *context window* as the bottleneck, not the toolset |
| **Scientific reproduction** | PaperCoder, CodeScientist, AI-Researcher, AlphaEvolve | Plan → analyze → generate; or generate-execute-reflect; or evolutionary search | Adds explicit source compression (blueprint) + stateful memory to keep the repo globally consistent |
| **Commercial assistants** | Cursor, Claude Code, Codex, Gemini CLI, Copilot, Cline | Whole-codebase understanding embedded in IDE/terminal | Document-grounded, long-horizon synthesis from a spec, not interactive editing |

The honest read of this table: most prior multi-agent coders either **mimic a human team** (ChatDev, MetaGPT) or **bolt a refinement loop onto generation** (AgentCoder, MapCoder). DeepCode's bet is that the binding constraint is neither org structure nor test coverage but **information flow under a finite context budget** — and the ablations are what make that bet credible rather than rhetorical. The commercial agents are the strongest baseline precisely because they have excellent whole-codebase understanding *interactively*; what they lack is a discipline for compressing a 30-page spec into a high-signal blueprint and then never drowning the generator in raw history. That is the gap the 0.854-vs-0.584 result measures.

It is worth being precise that the commercial tools are not designed for this task — they are interactive copilots, and benchmarking them on autonomous document-to-repository synthesis is somewhat off-label. The comparison is fair as an *architecture* test (same base model, same task) but it is not a claim that Cursor is "worse at coding"; it is a claim that DeepCode's pipeline extracts more from the same model on this *specific* long-horizon, document-grounded problem.

## Critique: what to believe and what to discount

Now the senior-engineer pass. I think this is a strong systems paper with one genuinely convincing result, wrapped in a framing that oversells.

**What's strong.** The same-base-model commercial comparison (0.854 vs 0.584/0.587) is the real contribution — it cleanly isolates *architecture* from *model*, which most agent papers fail to do. The ablations are unusually honest: instead of claiming every component universally helps, they show CodeRAG is near-useless on frontier models and most valuable on cheap ones. The blacklist methodology (no access to the original repos) is the right call and is easy to get wrong. And the central design lens — *manage the signal-to-noise ratio in a finite context budget* — is a genuinely useful way to think about any long-horizon agent, not just coding ones. This is the same problem the [memory-in-the-age-of-ai-agents survey](/blog/paper-reading/ai-agent/memory-in-the-age-of-ai-agents-a-survey) circles, and CodeMem is a clean, typed instance of it.

**What's soft.** Three things deserve an asterisk:

1. **The information theory is a metaphor, not a measurement.** There is no mutual information computed, no channel capacity estimated, no rate-distortion curve. "Channel optimization" and $\arg\max \text{Score}(\mathcal{P} \mid \mathcal{D})$ are intuition pumps. That is fine as *motivation*, but the paper occasionally leans on the framing as if it were a result ("we characterize the task through an information-theoretic lens"). It characterizes it through an information-theoretic *analogy*. Nothing in the method would change if you deleted the word "channel."
2. **"Surpasses humans" needs heavy qualification.** It is measured on Code-Dev — static grading of code structure and algorithmic fidelity, *not* running the experiments and matching the paper's numbers. The human comparison is over **three papers**, where DeepCode's 75.9 ± 4.5 has a confidence interval that comfortably overlaps the humans' 72.4. And the humans were not unaided experts in a vacuum — they had a four-week part-time window and were allowed to use ChatGPT and Copilot. "An automated system produces more rubric-conformant code than PhD students did in a part-time window, as judged by o3-mini" is a real and impressive claim, but it is not "AI out-reproduces human scientists."
3. **The judge is an LLM.** SimpleJudge is o3-mini scoring against a rubric. LLM-grades-LLM evaluations are known to have systematic biases (length, formatting, style affinity). Three-trial averaging tames variance, not bias.

**The ablation that's missing.** They ablate CodeMem, CodeRAG, and Verification — three of the four operations — but **not Phase 1, the blueprint distillation**, which is the entire *source-compression* leg of the thesis and arguably the most important component. There is no "blueprint vs. raw paper" comparison. The whole argument rests on compression being the thing that dissolves the long-context problem, yet that claim is never isolated. There is also no token/cost accounting: a multi-agent pipeline with parallel readers, per-file summarization, optional retrieval, and an iterative sandbox loop almost certainly costs far more per paper than a single-pass baseline, and that number is absent.

**What would change my mind.** If DeepCode were evaluated on **PaperBench full** — where reproduction requires actually executing the experiments and matching the paper's reported results, not just static code grading — and *still* matched humans, the "expert-level reproduction" claim would go from "impressive on a proxy metric" to "load-bearing." Likewise, a blueprint ablation showing that raw-paper generation collapses would convert the compression thesis from plausible to demonstrated. As it stands, I believe the **architecture wins** result completely, and I hold the **beats-humans** result loosely.

## What I'd build with this

Four concrete directions, two of which the authors flag themselves:

1. **Hybrid big-plan / small-code routing.** The backbone sweep shows the model is the ceiling, and the CodeRAG ablation shows cheap models can be rescued by retrieval. Combine them: use a frontier model for Phase 1 (blueprint) and the hard files, and route the routine files to a small fine-tuned model with CodeRAG always on — gated by exactly the $\delta(\mathcal{X}_t, \hat{c}_t)$ complexity signal that already exists. This is the "synergize models of varying scales" direction the discussion section gestures at, and the gate to implement it is already in the system.
2. **Bidirectional planning — let Phase 3 rewrite Phase 1.** Today the flow is linear Plan-then-Code; verification can patch *files* but not the *blueprint*. If the sandbox discovers that a design assumption was wrong (a missing component, an impossible interface), that signal should flow back and amend $\mathcal{B}$, not just $\Phi_{\text{LSP}}$ a file. The discussion explicitly calls out the fragility of a "stale plan"; closing the loop to planning is the fix. This is the same iterate-tests-and-patches instinct as [InfCode](/blog/paper-reading/ai-agent/infcode-adversarial-iterative-refinement-of-tests-and-patches-for-reliable-software-issue-resolution), pushed up a level from patches to plans.
3. **Cross-paper CodeMem — reflection into reusable skills.** CodeMem is reset per project. Persist it across papers: a Transformer block, a standard training loop, a data-loader pattern recur constantly, and a cross-project memory would let the agent retrieve its *own* prior summaries instead of re-deriving them. This is the "post-task reflection condensing traces into reusable skills" idea, and it is what would move these agents from episodic to genuinely *evolving* — the same trajectory [Kosmos](/blog/paper-reading/ai-agent/kosmos-an-ai-scientist-for-autonomous-discovery) pursues for autonomous discovery.
4. **Steal the evaluation discipline, not just the architecture.** The blacklist + author-rubric + LLM-judge protocol is a reusable template for any internal code-generation eval where "did it copy the answer" is a real risk. And the signal-to-noise / context-budget lens is worth applying to *every* long-horizon agent you build — it generalizes well beyond paper reproduction, which is the same scaling-the-scaffold-not-the-model thesis [Qwen3-Coder-Next](/blog/paper-reading/ai-agent/qwen3-coder-next-technical-report) reaches from the training side.

The one-line takeaway I'd carry out of this paper: for long-horizon code generation, *what is in the context window at each step* is a more tractable lever than *how smart the model is* — and a typed, summary-based memory plus an adaptive retriever is a remarkably effective way to pull that lever.

## References

- **Paper:** [DeepCode: Open Agentic Coding](https://arxiv.org/abs/2512.07921) — Zongwei Li, Zhonghang Li, Zirui Guo, Xubin Ren, Chao Huang (HKUDS, University of Hong Kong), December 2025. [PDF](https://arxiv.org/pdf/2512.07921).
- **Code:** [github.com/HKUDS/DeepCode](https://github.com/HKUDS/DeepCode)
- **Benchmark:** [PaperBench: Evaluating AI's Ability to Replicate AI Research](https://arxiv.org/abs/2504.01848) — Starace et al., OpenAI, 2025.
- **Prior specialized system:** [Paper2Code / PaperCoder](https://arxiv.org/abs/2504.17192) — Seo, Baek, Lee, Hwang, 2025.
- Related on this blog: [InfCode: adversarial iterative refinement of tests and patches](/blog/paper-reading/ai-agent/infcode-adversarial-iterative-refinement-of-tests-and-patches-for-reliable-software-issue-resolution) · [Kosmos: an AI scientist for autonomous discovery](/blog/paper-reading/ai-agent/kosmos-an-ai-scientist-for-autonomous-discovery) · [Qwen3-Coder-Next: scaling agentic training, not model size](/blog/paper-reading/ai-agent/qwen3-coder-next-technical-report) · [Memory in the age of AI agents: a survey](/blog/paper-reading/ai-agent/memory-in-the-age-of-ai-agents-a-survey)
