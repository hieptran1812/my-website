---
title: "Evaluating Memory in LLM Agents: Benchmarks, Metrics, and the Art of Not Fooling Yourself"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "A senior-level field guide to measuring agent memory: the write-manage-read model, the seven competencies, the benchmark landscape from LoCoMo to BEAM, which metrics lie, and a runnable harness that always includes the full-context baseline."
excerpt: "Your agent scores 92.5 on LoCoMo and still forgets the user's name. This is a deep dive into why memory evaluation is hard, what the benchmarks (LoCoMo, LongMemEval, MemBench, MemoryAgentBench, BEAM) actually measure, which metrics deceive you, and how to build an eval harness that catches the failures the leaderboards miss."
tags:
  [
    "agent-memory",
    "evaluation",
    "benchmarks",
    "llm-agents",
    "longmemeval",
    "locomo",
    "metrics",
    "long-context",
    "rag",
    "memory-systems",
    "llm-as-judge",
  ]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 51
---

Here is a sentence that should make you nervous: a memory system can post a 92.5 on the LoCoMo leaderboard and still forget that you switched jobs three sessions ago. Both things are true at once, and the gap between them is the entire subject of this article.

We have spent two years getting very good at making agents *score* on memory benchmarks and almost no time getting good at *measuring whether they actually remember*. Those are different skills. A leaderboard number is a single scalar summarizing one dataset under one scoring function, run once. "It remembers me" is a claim about retrieval, temporal reasoning, conflict resolution, and abstention holding up across hundreds of sessions, under a scoring function that is itself an LLM with its own failure modes. When a vendor says "state of the art on LoCoMo," the honest follow-up question is: *which version of LoCoMo, scored how, against which baseline, with how much variance, and does it survive a re-run by a skeptic?* As we will see, the answer to that last question has more than once been "no."

This is a field guide to closing that gap. We will build the mental model for what memory even is, enumerate the seven things a memory system has to do, tour the benchmark landscape from the 16k-token LoCoMo to the 10-million-token BEAM, dissect why the most popular metrics quietly mislead you, and then build a small evaluation harness you can point at your own agent. The thread running through all of it is a single discipline: **measure the thing you actually care about, include the baseline that could embarrass you, and report the variance.**

![Agent memory modeled as a write-manage-read loop: a user turn flows into a WRITE step that appends to the memory store, the store is consolidated by a MANAGE step that can evict stale facts, and a READ step retrieves top-k context for the LLM to generate a grounded answer](/imgs/blogs/evaluating-memory-llm-agents-benchmarks-metrics-1.webp)

The diagram above is the mental model for the whole post. Agent memory is not a database you query; it is a loop. Every turn *writes* something, the store gets *managed* between turns (consolidated, merged, sometimes forgotten), and every turn *reads* from it to ground generation. Each arrow in that loop is a separate place the system can fail, and — this is the key insight — each one can and should be measured independently. A benchmark that only checks the final answer is testing all three edges fused together, which is why a low score tells you almost nothing about *where* the memory broke.

A quick orientation before we dive in. This post is about *measuring* memory, not building it — when we mention mem0, Zep, MemGPT, or a graph store, it is as systems-under-test, never as recommendations. The techniques here are vendor-neutral by design: a good evaluation should score any of them, plus the baselines, on the same footing, because that is the only way a comparison means anything. And the bar we hold everything to is reproducibility — if a number cannot be regenerated from a config file and a pinned judge, it does not count, no matter whose leaderboard it sits on.

## Why a benchmark score and "it remembers me" are different claims

The trap is that long-context language models have gotten good enough that they paper over the difference until they suddenly don't. Below 30k tokens, a frontier model with the whole transcript in context will recall most facts, and a memory system layered on top looks redundant. The cliff arrives later, at hundreds of thousands or millions of tokens, exactly where you stopped testing.

| Assumption | The naive view | The measured reality |
| --- | --- | --- |
| "A high benchmark score means good memory." | One scalar on one dataset generalizes. | LoCoMo, LongMemEval, and BEAM disagree about which system wins; the metric you pick flips the ranking. |
| "Long context replaces memory." | A 1M-token window remembers everything in it. | Accuracy drops 30–60% as interaction history lengthens, and another ~25% from 1M to 10M tokens. |
| "Higher F1/BLEU means better answers." | Lexical overlap tracks correctness. | Lexical metrics punish paraphrase and abstractive memory; they rank systems differently than a judge does. |
| "The vendor's number is reproducible." | SOTA claims are audited. | A widely-cited 84% LoCoMo result was recomputed to ~58% by a competitor; a "68%" was recomputed to 75% by another. |
| "More memory operations = better recall." | Storing more helps. | Capacity degrades as the store grows; precision falls and retrieval latency climbs. |

Every row of that table is a story we will tell with numbers later. For now, internalize the shape of the problem: **memory evaluation is adversarial against your own optimism.** The defaults — final-answer accuracy, lexical overlap, a single run, no baseline — all bias upward. Good evaluation is the practice of systematically removing those biases.

There is a deeper reason the defaults flatter you, and it is worth saying plainly: every shortcut in evaluation trades measurement accuracy for convenience, and the trades all point the same direction. Final-answer scoring is convenient because it needs no retrieval instrumentation — and it hides *where* the failure was. Lexical metrics are convenient because they are deterministic — and they reward wording over correctness. A single run is convenient because compute is finite — and it gives you no error bars. Dropping the baseline is convenient because the baseline is boring to run — and it is the one comparison that could falsify your whole project. Evaluation discipline is mostly the refusal to take those convenient trades.

This post is the evaluation companion to two others on this blog. If you want the architecture side — what these memory systems *are*, how MemGPT, Letta, mem0, and Zep actually store and retrieve — read [Long-Term Memory for Conversational Agents: MemGPT and Beyond](/blog/machine-learning/ai-agent/long-term-memory-conversational-agents-memgpt) first. And if you want the broader discipline of grading agents beyond a final answer, [Evaluating Agent Trajectories: Beyond Final-Answer Accuracy](/blog/machine-learning/ai-agent/evaluating-agent-trajectories-beyond-final-answer) is the sibling to this piece; much of its statistical machinery applies directly here.

## 1. What we actually mean by "memory"

Before you can measure memory you have to agree on what it is, and the word is badly overloaded. "Memory" in an agent spans at least four distinct stores with different lifetimes, content, and failure modes. Conflating them is the first mistake; a benchmark that only tests one of them and reports a number called "memory" is the second.

![Taxonomy tree of agent memory: a root splits into working memory (in-context scratchpad, lives one turn), episodic memory (session logs, lives weeks), semantic memory (user facts, updated on change), and procedural memory (tool routines, reused across tasks)](/imgs/blogs/evaluating-memory-llm-agents-benchmarks-metrics-2.webp)

- **Working memory** is the in-context scratchpad: whatever fits in the current prompt. It lives exactly one turn (or one compaction cycle) and is gated by the context window, not by any storage system. Evaluating it is really evaluating long-context recall.
- **Episodic memory** is the log of past events — "what happened in session 7." It persists for weeks or months and is what most conversational benchmarks actually test. Retrieval over episodic memory is the bread and butter of LoCoMo and LongMemEval.
- **Semantic memory** is distilled fact: "the user is vegetarian," "the user's manager is named Priya." It is derived from episodes but stored as durable assertions, and crucially it gets *updated* — the user stops being vegetarian, gets a new manager. This is where knowledge-update and conflict-resolution failures live.
- **Procedural memory** is reusable skill: a verified tool sequence, a plan that worked. Voyager's Minecraft skill library is the canonical example — every routine that succeeds is stored as runnable code, indexed by a natural-language description, and composed later for novel tasks. Almost no public benchmark tests this well, which is itself a finding.

The 2026 survey *Memory for Autonomous LLM Agents* formalizes this with a three-axis taxonomy — **temporal scope** (how long a memory lives), **representational substrate** (raw text, vectors, knowledge graph, parameters), and **control policy** (how writes, merges, and evictions are decided) — and identifies five mechanism families that real systems use: context-resident compression, retrieval-augmented stores, reflective self-improvement, hierarchical virtual context, and policy-learned management. You do not need to memorize the families. You need to internalize the consequence: **a memory system makes choices on all three axes, and an evaluation that fixes those axes implicitly only tests one slice of the design space.** A benchmark built from short conversations with no fact updates cannot distinguish a good semantic-memory policy from a no-op, because nothing in the data ever needs updating.

The substrate axis matters for evaluation in a way that is easy to miss. A vector-store memory and a knowledge-graph memory can post identical end-to-end accuracy while failing completely differently: the vector store degrades by returning near-duplicate chunks that crowd out the relevant one, while the graph degrades by missing an edge that was never extracted in the first place. If your only metric is final-answer accuracy you cannot tell these apart, and you will reach for the wrong fix. This is the deeper argument for per-edge measurement in the loop — the substrate determines *how* each edge fails, so the diagnostic metric has to live at the edge, not at the output.

It helps to ground the four types in concrete probes, because the probe is where a fuzzy capability turns into a measurable one. For working memory: "what did I just ask you to do?" — answerable only from this turn's context. For episodic: "what restaurant did we talk about in our first conversation?" — a retrieval over logged events. For semantic: "am I allergic to anything?" — a distilled fact that may have been stated once, weeks ago, and must persist. For procedural: "deploy using the steps that worked last time" — a reusable routine the system has to recall and replay. A test set built entirely from episodic-style questions, as most are, says nothing about the other three.

> A memory benchmark is a hypothesis about what memory is for. Read its construction, not its leaderboard.

## 2. The seven things memory must do

Once memory is decomposed, "is the memory good?" decomposes too — into capabilities you can test one at a time. The literature has converged, from three different directions, on roughly the same competency list. LongMemEval names five core abilities. MemoryAgentBench names four core competencies. BEAM enumerates ten task categories. Stack them up and the union is a clean set of seven, and almost every benchmark you will encounter is some projection of these onto its own dataset.

1. **Accurate retrieval** — locate a specific fact in an arbitrarily long history. This is multi-hop needle-in-a-haystack: the fact is there, can the system find it under noise?
2. **Multi-session reasoning** — synthesize facts that live in *different* sessions to answer one question ("how long after starting therapy did the user change jobs?"). Single-fact retrieval is not enough.
3. **Temporal reasoning** — reason about *when* things happened, both from explicit dates and from session metadata timestamps. "What was the user driving *before* the Tesla?"
4. **Knowledge update** — recognize that a fact changed and prefer the new value. The user moved cities; the old city must not surface as the current answer.
5. **Conflict resolution** — when the store contains contradictory facts, return only the latest valid one. This is the read-time twin of knowledge update.
6. **Abstention** — decline to answer when the information was never in the history. A system that hallucinates a plausible answer for an unanswerable question is *worse* than one that says "not mentioned," and most systems are terrible at this.
7. **Test-time learning** — ingest a new rule, label, or procedure given mid-conversation and apply it reliably afterward, with no weight update. "From now on, tag anything about budget as 'finance'."

![Coverage matrix of seven memory competencies (rows) against five benchmarks (columns: LoCoMo, LongMemEval, MemBench, MemoryAgentBench, BEAM), each cell marked core, part, or none and color-coded, showing sparse and uneven coverage with no benchmark testing all seven](/imgs/blogs/evaluating-memory-llm-agents-benchmarks-metrics-3.webp)

The matrix above is the single most important figure for picking a benchmark. Read it as a coverage map. LoCoMo is strong on retrieval, multi-session, and temporal reasoning but has essentially no conflict-resolution coverage and no test-time learning. LongMemEval adds abstention and knowledge update as first-class categories. MemoryAgentBench is the one built explicitly around test-time learning and conflict resolution. BEAM is the broadest, with ten categories, but it pays for that breadth by being enormous and expensive to run. **No single column is fully green.** A team that reports only LoCoMo is, structurally, not testing whether their system handles a fact that changes — which is the single most common real-world memory operation.

The practical upshot: choose your benchmark from the *competency you are worried about*, not from which leaderboard is fashionable. If your agent is a long-running personal assistant, abstention and knowledge update dominate your risk; LoCoMo alone will lull you. If it is a coding agent learning project conventions, test-time learning dominates, and you want MemoryAgentBench-style incremental evaluation.

### Each competency as a concrete test

The competencies stay abstract until you write the probe that exercises each one. Here is the translation, with the failure mode each probe is designed to catch and the metric that scores it cleanly.

| Competency | Example probe | Typical failure | How to score |
| --- | --- | --- | --- |
| Accurate retrieval | "What shoe size did I say I wear?" | fact present but not retrieved under noise | recall@k, then judge the answer |
| Multi-session reasoning | "How many weeks between starting the diet and the checkup?" | retrieves both facts, fails to combine | judge; verify both facts in retrieved set |
| Temporal reasoning | "What phone did I use before this one?" | confuses order; ignores timestamps | judge with date-sensitive gold |
| Knowledge update | "Where do I work now?" | returns the superseded value | judge; gold = latest value only |
| Conflict resolution | "Am I vegetarian?" after a reversal | surfaces the higher-similarity old fact | judge; penalize the stale answer |
| Abstention | "What's my brother's name?" (never stated) | confidently fabricates a name | exact: did it say "not mentioned"? |
| Test-time learning | apply a labeling rule given mid-chat | forgets the rule after a few turns | accuracy on post-rule items |

The pattern in the last column is the one that surprises teams: **only retrieval is scored by a retrieval metric.** Everything else is an end-to-end judge call, because the failure is about *which* fact the system commits to, not whether the right chunk was somewhere in the top-k. This is exactly why a high recall@k can sit next to a low answer accuracy — the relevant memory was retrieved and then ignored, drowned out by neighbors, or overruled by a stale duplicate. The competency you under-test is the competency that fails you in production, and the cheapest insurance is a probe set that names all seven.

## 3. The benchmark landscape, 2024 → 2026

There are now enough memory benchmarks that the meta-problem is choosing among them. The good news is they occupy genuinely different points in a two-dimensional space: how much context they stress, and how many competencies they cover.

![Landscape scatter of memory benchmarks positioned by context scale on the horizontal axis and competency breadth vertically: LoCoMo at small scale and narrow coverage, LongMemEval-S and LongMemEval-M at medium and large scale, MemBench and MemoryAgentBench at broad coverage, and BEAM-1M and BEAM-10M at the largest scale and broadest coverage](/imgs/blogs/evaluating-memory-llm-agents-benchmarks-metrics-4.webp)

The figure plots the five families. Left to right is context scale, from LoCoMo's ~16k tokens to BEAM's 10 million; vertical position is rough competency breadth. The newest entrants push toward the top-right — broad *and* huge — because that is where the unsolved problems are. Here is the same landscape as a reference table, which you will want to keep open while choosing.

| Benchmark | Released | Scale | Size | Tasks / competencies | Default scoring | The catch |
| --- | --- | --- | --- | --- | --- | --- |
| **LoCoMo** | 2024-02 | ~16–26k tokens, ~32 sessions, ~600 turns | ~1,540 questions | single-hop, multi-hop, temporal, open-domain (+ summarization, multimodal) | F1, BLEU, ROUGE, MMRelevance, LLM-judge | Short enough that full-context wins; documented annotation errors |
| **LongMemEval** | 2024-10 | 115k tokens (S) to 1.5M (M) | 500 questions | extraction, multi-session, knowledge update, temporal, abstention | exact/judge per task | Synthetic histories; expensive at the M scale |
| **MemBench** | 2025-06 | moderate, multi-turn | factual + reflective | extraction, multi-hop, knowledge update, preference, temporal | accuracy + ops + capacity | Newer, less third-party replication |
| **MemoryAgentBench** | 2025-07 | incremental multi-turn | four competencies | accurate retrieval, test-time learning, long-range understanding, conflict resolution | per-competency accuracy | Tests the under-tested skills; smaller community |
| **BEAM** | 2025-10 | 1M and 10M tokens | ten categories | preference, instruction, extraction, knowledge update, multi-session, summarization, temporal, event ordering, abstention, contradiction | judge + category scores | Brutally expensive; the 10M track is the real frontier |

A few notes that the table cannot hold. **LoCoMo** (Maharana et al., out of Snap) was the field's first serious multi-session conversational benchmark and is still the most cited — which is exactly why its weaknesses matter so much, and we will spend a case study on them. **LongMemEval** introduced the indexing/retrieval/reading decomposition that maps cleanly onto our write-manage-read loop, and it is the benchmark that produced the headline "30–60% accuracy drop as history grows." **MemBench** is the one that insists memory has three measurable dimensions — effectiveness, efficiency, *and* capacity — rather than just accuracy, and it deliberately separates "participation" scenarios (the agent was in the conversation) from "observation" scenarios (the agent reads someone else's). **MemoryAgentBench** (ICLR 2026) evaluates through *incremental* multi-turn interaction rather than dumping a static history, which is the only setup that fairly tests test-time learning. **BEAM** (also ICLR 2026, "Beyond a Million Tokens") exists to make one point loudly: 10 million tokens is roughly a year of daily conversation, and you cannot solve it by buying a bigger context window.

It helps to see what a single item looks like in each. A **LoCoMo** instance is a multi-session dialogue between two personas with an event graph behind it, and a question like "Where did Caroline go for her birthday?" whose answer sits in session 4. A **LongMemEval** instance threads the target fact through a long history padded with unrelated sessions, and every question carries a type label, so temporal-reasoning questions are scored separately from abstention questions. A **MemoryAgentBench** instance does not hand you the history at all — it feeds the conversation *incrementally*, turn by turn, and probes mid-stream, which is the only fair way to test whether a rule given at turn 50 is still honored at turn 90. A **BEAM** instance is the same idea inflated to a year of interaction, with ten labeled categories so degradation can be attributed to a specific capability instead of smeared into one average.

Two construction choices deserve special attention because they silently change what a score means. The first is **participation versus observation**, which MemBench separates explicitly: did the agent take part in the conversation it is now being asked about, or is it reading a transcript of someone else's? The retrieval problem looks similar but the grounding differs, and a system tuned for one can be weak at the other. The second is **synthetic versus human-collected**. Most large memory benchmarks are synthetic — generated by an LLM — because collecting a year of real multi-session dialogue per user is impractical. Synthetic data scales but inherits the generator's blind spots and its annotation errors, which is the seed of the contamination problem we return to below.

There is also a culture problem worth naming. Once a benchmark becomes the leaderboard everyone cites, it becomes a target, and a target stops being a measure — Goodhart's law applied to memory. Systems get tuned to LoCoMo's specific question distribution and scoring quirks until the LoCoMo number stops predicting real-world performance. The defense is to treat any single benchmark as one noisy probe of a broad capability, rotate which benchmark you optimize against, and hold out a private, traffic-shaped test set that no one trains or tunes on. The number that matters is the one from data your system has never been pointed at.

## 4. Long context is not memory

This deserves its own section because it is the most expensive misconception in the field, measured in GPU-hours spent stuffing transcripts into context windows that then fail anyway.

![Before-after comparison titled long context is not memory: the assumption column shows short chat fits full context with GPT-class recall around 70 percent and the belief that more tokens equals more memory, while the measured reality column shows long histories on LongMemEval drop 30 to 60 percent and another 25 percent from 1M to 10M tokens on BEAM](/imgs/blogs/evaluating-memory-llm-agents-benchmarks-metrics-5.webp)

The assumption, drawn on the left, is seductive: context windows keep growing, so eventually you just paste everything in and the model remembers. And for short interactions it works — on a 16k-token conversation, a frontier model recalls most facts, which is precisely why a too-short benchmark cannot tell a memory system apart from a no-op. The reality, on the right, is a cliff. LongMemEval measured commercial assistants and long-context models dropping 30–60% in accuracy as interaction histories lengthened, even when the relevant fact was still technically inside the window. BEAM then showed that going from 1M to 10M tokens costs another ~25%, with temporal reasoning degrading the worst.

Why does a fact "in the window" still get missed? Three mechanisms, all measurable:

- **Attention dilution.** With a million distractor tokens, the relevant span competes for attention with everything else. This is the lost-in-the-middle failure, and it gets monotonically worse with length.
- **No update semantics.** Raw context has no notion that session 20 supersedes session 3. If the user changed jobs, both jobs are "in context" with equal standing, and the model has to infer recency from position — which it does unreliably.
- **Cost and latency.** Even when accuracy holds, paying to re-read 1M tokens every turn is economically absurd. A good memory system reads ~7k tokens per query instead, which is the entire point.

This is also why [Effective Context Engineering for AI Agents](/blog/machine-learning/ai-agent/effective-context-engineering-for-ai-agents) and memory are coupled disciplines: context engineering is about what you put in the window *this turn*, and memory is about deciding what is worth putting there across turns. An evaluation that does not measure read-time token consumption is missing half the story — a system that "remembers" by pasting everything is cheating, and only an efficiency metric exposes it.

## 5. Metrics: what to measure, and why most of them lie

We now have competencies and benchmarks. The remaining question is how to *score*, and this is where more evaluations go wrong than anywhere else. There are five metric families, and three of them will deceive you if used alone.

### Retrieval metrics: recall@k, precision@k, MRR

Before you score the answer, score the *read*. Retrieval metrics measure the READ edge of the loop in isolation, which is invaluable because they tell you whether a wrong answer is a retrieval failure or a reasoning failure. Given the set of gold-relevant memory items $R$ and the top-$k$ items the system returned, recall and precision are $\text{recall@}k = |\,\text{retrieved} \cap R\,| / |R|$ and $\text{precision@}k = |\,\text{retrieved} \cap R\,| / k$. Mean reciprocal rank adds *where* the first relevant item landed: $\text{MRR} = \frac{1}{N}\sum_i 1/\text{rank}_i$.

A worked example. Suppose a probe has two relevant memories, you retrieve ten, and the relevant ones land at ranks 2 and 8. Then recall@10 = 2/2 = 1.0, precision@10 = 2/10 = 0.2, and the MRR contribution is 1/2 = 0.5. Recall says "we found everything," precision says "but 80% of what we sent the model was noise," and MRR says "the first hit was near the top." They pull in different directions on purpose: raising $k$ improves recall and hurts precision, which is the capacity trade-off in miniature — more candidates, more noise. The right operating point is the $k$ that maximizes *downstream answer accuracy*, found by sweeping $k$ and watching the judge, not by maximizing recall in a vacuum. A system with recall@10 of 1.0 and answer accuracy of 0.5 has a reasoning or conflict problem; a system with recall@10 of 0.4 has nothing to reason over in the first place. Splitting those two cases is the first thing a per-edge harness buys you, and no end-to-end score can do it.

### Lexical overlap (F1, BLEU, ROUGE, exact match)

Token-level F1 is the historical default. For a candidate answer and a gold answer, precision $P$ is the fraction of candidate tokens that appear in gold, recall $R$ is the fraction of gold tokens that appear in the candidate, and

$$F_1 = \frac{2 \cdot P \cdot R}{P + R}.$$

It is cheap, deterministic, and reproducible — and it is wrong in a specific, predictable direction. Lexical metrics have **length bias**: they reward answers that match the reference *length and wording*, not the reference *meaning*. An abstractive memory system that says "the user prefers tea" when the gold is "the user said they switched from coffee to tea last month" scores poorly on F1 despite being correct. BLEU and ROUGE share this pathology. The literature is blunt: BLEU, ROUGE, exact match, and token-F1 "poorly reflect semantic correctness in open-ended generation and systematically favor outputs matching reference length, conflating memory fidelity with generation style." Use them for regression detection on a frozen system, never for ranking different systems.

### LLM-as-judge (the J score)

The modern default is to ask a strong model whether the candidate conveys the same fact as the gold, returning a binary correct/incorrect — the "J score" you see on LoCoMo leaderboards. This captures paraphrase and reasoning coherence that lexical metrics miss, and rankings produced by a judge often *invert* the rankings produced by F1.

![Matrix showing four memory systems scored by both F1 and an LLM judge: Lexical-tuned RAG ranks number one by F1 score 0.41 but number four by judge at 61 percent, while Abstractive memory ranks number four by F1 at 0.19 but number one by judge at 84 percent, with green coloring migrating diagonally from top-left to bottom-right](/imgs/blogs/evaluating-memory-llm-agents-benchmarks-metrics-6.webp)

The matrix above is the phenomenon in one picture (numbers illustrative, the pattern is real). Four systems, scored two ways. By F1, the lexical-tuned RAG system wins — it is *built* to match reference wording — and the abstractive memory system, which paraphrases, comes dead last with an F1 of 0.19. Switch to an LLM judge and the order completely flips: the abstractive system wins at 84% because its answers are *correct*, just worded differently, and the lexical-tuned system drops to last because matching wording is not the same as being right. The green cells migrate diagonally across the matrix. **If you had picked F1, you would have shipped the wrong system.** This is not hypothetical; researchers comparing F1-based and judge-based rankings across five representative memory architectures on LoCoMo found exactly this disconnect, with lexical metrics "failing to capture the strengths of abstractive memory systems."

But the judge has its own failure modes, and pretending otherwise is how the benchmark wars started:

- **Prompt sensitivity.** The judge's verdict depends on its rubric. A lenient rubric inflates everyone; a strict one deflates. Two teams using different judge prompts are not comparable, full stop.
- **Self-preference and model drift.** Judges favor outputs from their own model family, and a judge model that silently updates changes your historical numbers.
- **Position and verbosity bias.** Judges over-reward longer, more confident answers — the opposite of what you want for abstention.

The defense is to **calibrate the judge against human labels** on a sample (report its agreement rate, e.g. Cohen's kappa), pin the judge model and prompt by hash, and publish both. A J score without a stated rubric and judge version is a number with no units.

### Calibrating the judge against humans

If the judge decides who wins, the judge itself is a system under test, and you validate it the way you validate any classifier: against ground truth. Hand-label a sample of 100–200 (question, gold, candidate) triples as correct or incorrect, run the judge on the same triples, and measure agreement. Raw agreement is misleading when most answers are correct, so report Cohen's kappa, which corrects for chance:

```python
def cohens_kappa(human: list[bool], model: list[bool]) -> float:
    n = len(human)
    po = sum(h == m for h, m in zip(human, model)) / n      # observed agreement
    ph, pm = sum(human) / n, sum(model) / n
    pe = ph * pm + (1 - ph) * (1 - pm)                      # chance agreement
    return (po - pe) / (1 - pe)
```

A kappa above ~0.8 means the judge is trustworthy enough to rank systems; below ~0.6 your leaderboard is mostly measuring the judge's quirks, not the systems'. Two patterns show up constantly in the disagreements: the judge is too lenient on abstention, accepting a confident wrong answer as "close enough," and too strict on paraphrase when the gold is terse. Both are fixable by editing the rubric and re-measuring kappa — which is the reason you pin the rubric by hash, because a rubric change *is* a measurement change and an unversioned judge silently rewrites your history. Publish the kappa next to every J score. A judge accuracy with no calibration number is precisely the unaudited authority the benchmark wars were fought over.

### Efficiency (tokens per query, latency)

Accuracy is half the metric. The other half is what the accuracy *cost*. Two numbers matter:

- **Read-time token consumption** — how many tokens the memory system injects into the prompt per query. Mem0 reports averaging under ~7,000 tokens per retrieval call; the full-context baseline on the same LoCoMo conversations consumes ~26,000. A system that matches full-context accuracy at a quarter of the tokens is the actual win, and it is invisible unless you measure it.
- **Latency, reported as a distribution.** Median (p50) hides the tail that wrecks interactive UX. One published comparison put a memory system at p50 0.708s / p95 1.440s — usable — while a competitor under one evaluation showed a p95 search latency near *60 seconds*, which is a non-starter for a chat agent regardless of its accuracy.

### Capacity (degradation as the store grows)

The metric almost nobody reports, and the one that bites in production. As the memory store accumulates months of data, does accuracy hold? MemBench makes this a first-class axis. The failure is mechanical: more stored items means more candidates per retrieval, which means lower precision at fixed $k$, which means more noise in context, which means lower accuracy and higher latency. **A system evaluated at 30 sessions can quietly fall apart at 3,000.** If you only ever test on the benchmark's fixed-size histories, you have measured a snapshot, not the trajectory you actually care about.

The honest scorecard, then, is a vector, not a scalar: per-competency judge accuracy (calibrated), read tokens, p95 latency, and a degradation curve. Anyone who hands you a single number is hiding at least three of these.

### Putting the scorecard together: a worked example

The metric families only become useful when you read them *together*, as a vector. Take two systems evaluated on the same 500-probe set against the baseline (numbers illustrative; the shape is the lesson).

| Metric | System A (graph memory) | System B (lexical RAG) | Full-context baseline |
| --- | --- | --- | --- |
| Judge accuracy (overall) | 0.81 | 0.74 | 0.78 |
| — retrieval | 0.93 | 0.95 | 0.88 |
| — knowledge update | 0.79 | 0.41 | 0.55 |
| — abstention | 0.72 | 0.40 | 0.61 |
| Token-level F1 | 0.29 | 0.44 | 0.33 |
| Read tokens / query (p50) | 6,800 | 5,200 | 26,000 |
| Latency p95 (s) | 1.4 | 0.9 | 3.1 |

Read top to bottom and the story writes itself — a story no single number tells. System B wins on F1 and on raw retrieval, which is precisely why a lexical-metric-only evaluation would crown it. The judge disagrees: System B collapses on knowledge update and abstention, meaning it retrieves well but cannot tell which fact is current and cannot keep quiet when it knows nothing. System A wins overall on the judge while spending slightly more tokens. And the baseline is the gut-check that ties it together: full-context beats System B overall, so System B is not earning its existence; System A beats full-context on accuracy *and* uses roughly a quarter of the tokens, so System A is. That last comparison — against the baseline you could have shipped for free — is the one that decides whether any of this engineering was worth doing.

## 6. Build it yourself: a minimal memory-eval harness

Public benchmarks tell you how you stack up against the field. They do not tell you how your specific agent does on your specific users' data, and they will not catch the regression you ship next Tuesday. For that you need an in-house harness, and the good news is the core of one is about a hundred lines.

![Five-stage pipeline of a memory-eval harness: a multi-session JSON dataset feeds a WRITE stage that ingests sessions, a READ stage that retrieves per probe, an LLM judge that scores against the gold answer, and an aggregate stage producing per-competency results, with callout metrics ingest cost, recall at k, J score, and tokens p95 attached to each stage](/imgs/blogs/evaluating-memory-llm-agents-benchmarks-metrics-7.webp)

The harness is the write-manage-read loop turned into a measurable pipeline, and the figure shows the payoff: each stage emits its own metric. When the overall accuracy is bad, you do not have to guess whether the write step dropped the fact, the read step failed to retrieve it, or the judge mis-scored it — each stage has a tap. Let us build it.

### Get the probe set right before the code

The harness is the easy part; the probe set is where evaluations live or die. Three rules, each learned the expensive way.

**Balance answerable and unanswerable.** If every probe has an answer in the history, you never test abstention, and abstention is the competency that decides whether users trust the agent. Aim for at least a quarter of probes to be genuinely unanswerable — the fact was never stored — with a gold answer of "not mentioned." A system that scores 90% on answerable probes and 10% on unanswerable ones is a confident hallucinator hiding behind a good average.

**Make temporal and update probes require chronology.** A knowledge-update probe is only hard if the old value and the new value both appear, in order, far apart. Write the pair explicitly: session 1 says one thing, session N says another, and the gold answer is the value from session N. If your data contains no such pairs, your knowledge-update score is measuring nothing at all.

**Label every probe with its competency.** The single most useful thing you can do is tag each probe so the report breaks down per competency. An overall accuracy of 78% is uninformative; "94% retrieval, 81% temporal, 43% abstention, 38% knowledge update" tells you exactly what to fix next. The labels cost minutes and repay themselves the first time you debug a regression.

First, the data model. The only thing a memory system must expose to be evaluable is a `write`/`read` pair, which means the *same harness* evaluates mem0, Zep, a raw vector store, or "paste the whole transcript."

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import json, time

@dataclass
class Session:
    session_id: str
    timestamp: str            # ISO-8601; temporal-reasoning probes need this
    turns: list[dict]         # [{"role": "user"|"assistant", "content": str}, ...]

@dataclass
class Probe:
    question: str
    gold_answer: str
    competency: str           # "retrieval" | "multi_session" | "temporal" | ...
    answerable: bool = True   # False => the model SHOULD abstain

class MemorySystem(Protocol):
    """Anything with a write/read pair is evaluable: mem0, Zep, a bare
    vector store, or 'stuff the whole transcript into the prompt'."""
    def write(self, session: Session) -> None: ...
    def read(self, query: str, k: int = 10) -> list[str]: ...

def load_dataset(path: str) -> tuple[list[Session], list[Probe]]:
    raw = json.load(open(path))
    sessions = [Session(**s) for s in raw["sessions"]]
    probes   = [Probe(**p)   for p in raw["probes"]]
    return sessions, probes
```

Notice the `answerable` flag on the probe: half of evaluating memory is checking that the system *abstains* on questions whose answer was never stored. If your dataset has no unanswerable probes, you are not measuring abstention, and you will not notice when your system starts confidently hallucinating user preferences.

Next, the judge. This is the part you must calibrate; the rubric below is deliberately strict about judging the fact, not the prose, and explicitly handles abstention.

```python
import anthropic
client = anthropic.Anthropic()

JUDGE_RUBRIC = """You grade whether a candidate answer matches the gold answer
for a question about a user's conversation history. Output strict JSON:
{"correct": true|false, "reason": "<one sentence>"}.

Rules:
- correct iff the candidate conveys the same fact as gold; paraphrase is fine.
- If the question is unanswerable and the candidate abstains ("I don't know",
  "not mentioned"), mark correct=true.
- Ignore fluency, length, and hedging. Judge the FACT only."""

def judge(question: str, gold: str, candidate: str) -> bool:
    msg = client.messages.create(
        model="claude-opus-4-8",          # pin the judge model by exact ID
        max_tokens=200,
        system=JUDGE_RUBRIC,
        messages=[{"role": "user", "content":
            f"QUESTION:\n{question}\n\nGOLD:\n{gold}\n\nCANDIDATE:\n{candidate}"}],
    )
    verdict = json.loads(msg.content[0].text)
    return bool(verdict["correct"])
```

Now the answer-and-trace step. We retrieve, build a tightly-scoped prompt that *forces* abstention when the notes are empty, generate, and record three things per probe: the verdict, the read-time token count, and the latency.

```python
@dataclass
class Trace:
    correct: bool
    read_tokens: int          # tokens the memory returned into the prompt
    latency_s: float
    competency: str

def answer_with_memory(mem: MemorySystem, probe: Probe, k: int = 10):
    t0 = time.perf_counter()
    context = mem.read(probe.question, k=k)
    prompt = ("Use ONLY these notes to answer. If the answer is absent, "
              "reply exactly 'not mentioned'.\n\n"
              + "\n".join(f"- {c}" for c in context)
              + f"\n\nQ: {probe.question}")
    reply = client.messages.create(model="claude-opus-4-8", max_tokens=300,
                                   messages=[{"role": "user", "content": prompt}])
    latency = time.perf_counter() - t0
    read_tokens = sum(len(c) for c in context) // 4     # ~4 chars/token estimate
    return reply.content[0].text, read_tokens, latency

def evaluate(mem: MemorySystem, sessions, probes, k: int = 10) -> list[Trace]:
    for s in sorted(sessions, key=lambda s: s.timestamp):
        mem.write(s)                        # ingest in chronological order — this matters
    traces = []
    for p in probes:
        text, rtok, lat = answer_with_memory(mem, p, k)
        traces.append(Trace(judge(p.question, p.gold_answer, text), rtok, lat, p.competency))
    return traces
```

The `sorted(..., key=timestamp)` line is not cosmetic. Knowledge-update and conflict-resolution probes only make sense if writes arrive in chronological order, so the newer fact genuinely supersedes the older one. Ingesting out of order silently turns a hard temporal test into an easy one — a subtle way to inflate your own scores.

Now aggregation, broken out per competency, plus the one class that makes the whole exercise honest:

```python
import statistics as st
from collections import defaultdict

def report(traces: list[Trace]) -> dict:
    by_comp = defaultdict(list)
    for t in traces:
        by_comp[t.competency].append(t)
    out = {}
    for comp, ts in by_comp.items():
        lats = sorted(t.latency_s for t in ts)
        out[comp] = {
            "accuracy":        round(sum(t.correct for t in ts) / len(ts), 3),
            "n":               len(ts),
            "read_tokens_p50": int(st.median(t.read_tokens for t in ts)),
            "latency_p95":     round(lats[max(0, int(0.95 * len(lats)) - 1)], 3),
        }
    out["overall_accuracy"] = round(sum(t.correct for t in traces) / len(traces), 3)
    return out

class FullContext:
    """The baseline you must ALWAYS run: no memory system, just paste the
    whole transcript. If your fancy store can't beat this, it isn't earning
    its latency — and on short benchmarks, it usually can't."""
    def __init__(self):  self._buf = []
    def write(self, s):  self._buf.extend(t["content"] for t in s.turns)
    def read(self, query, k=10):  return self._buf      # ignores k; returns everything
```

That `FullContext` class is the most important fifteen lines in the harness. It is the baseline that has embarrassed published systems (next section), and running it costs you nothing. **If your memory system does not beat full-context on accuracy while using far fewer tokens, you have built a slower, more complex way to do nothing.**

### Adapting a real memory system

The protocol is deliberately minimal so a production system drops in behind it. A mem0-style store, for example, needs only a thin adapter:

```python
class Mem0Adapter:
    """Wrap any vendor SDK behind the MemorySystem protocol so the same
    harness scores it head-to-head with the baseline."""
    def __init__(self, client, user_id: str):
        self.client, self.user_id = client, user_id

    def write(self, session: Session) -> None:
        for turn in session.turns:
            self.client.add(turn["content"], user_id=self.user_id,
                            metadata={"ts": session.timestamp})   # NOT glued to text

    def read(self, query: str, k: int = 10) -> list[str]:
        hits = self.client.search(query, user_id=self.user_id, limit=k)
        return [h["memory"] for h in hits]

    def size(self) -> int:
        return len(self.client.get_all(user_id=self.user_id))
```

Two details in that adapter are exactly where the Mem0-versus-Zep dispute went wrong, so they are worth getting right. First, pass the timestamp through the vendor's *metadata* channel rather than gluing it onto the message text, or you sabotage the system's own temporal reasoning — the precise bug Zep alleged. Second, scope every call to a single `user_id`; collapsing two speakers into one identity, the other alleged bug, turns a clean evaluation into noise. The adapter is fifteen lines, but those fifteen lines are where a fair benchmark quietly becomes an unfair one. Write the adapter for the system you are comparing *against* as carefully as the one for your own — the entire benchmark-wars saga is what happens when a team does not.

Finally, wire it to a real benchmark and run multiple seeds so you report variance instead of a single lucky draw:

```bash
# Convert a public benchmark to the {sessions, probes} schema, then run 5 seeds
# so we report mean +/- CI, not one number that might be noise.
for seed in 1 2 3 4 5; do
  python -m memeval.run \
    --dataset data/longmemeval_s.jsonl \
    --system  mem0 \
    --k 10 --seed "$seed" \
    --out "runs/mem0_s${seed}.json"
done
python -m memeval.aggregate runs/mem0_s*.json   # mean +/- 95% CI per competency
```

And here is what a single probe record looks like — note the knowledge-update probe whose gold answer is the *second*, superseding value:

```json
{
  "sessions": [
    {"session_id": "s1", "timestamp": "2026-01-03T09:12:00Z",
     "turns": [{"role": "user", "content": "I just switched from coffee to matcha."}]},
    {"session_id": "s7", "timestamp": "2026-02-20T18:40:00Z",
     "turns": [{"role": "user", "content": "Actually I'm back on espresso now."}]}
  ],
  "probes": [
    {"question": "What does the user drink in the morning now?",
     "gold_answer": "espresso", "competency": "knowledge_update", "answerable": true}
  ]
}
```

A naive retrieval system returns *both* facts and lets the model guess; a good one knows session 7 supersedes session 1 and returns espresso. That single probe, multiplied across hundreds of updates, is what separates a real memory system from a vector store with good marketing.

### Reading the report

Run the harness and you get a structured report rather than a single number, which is the whole point. A typical run looks like this:

```json
{
  "overall_accuracy": 0.78,
  "retrieval":        {"accuracy": 0.94, "n": 180, "read_tokens_p50": 6800, "latency_p95": 1.31},
  "temporal":         {"accuracy": 0.81, "n": 90,  "read_tokens_p50": 7100, "latency_p95": 1.44},
  "knowledge_update": {"accuracy": 0.38, "n": 80,  "read_tokens_p50": 6900, "latency_p95": 1.40},
  "abstention":       {"accuracy": 0.43, "n": 70,  "read_tokens_p50": 5200, "latency_p95": 1.10}
}
```

The overall 0.78 is reassuring and useless. The breakdown is where the work is: retrieval is excellent, so the read edge is healthy, but knowledge update and abstention are near coin-flips. That points the finger squarely at the MANAGE edge and the read-time prompt — the system retrieves the right neighborhood of facts and then fails to prefer the current one or to stay silent when there is nothing. You would not learn any of that from 0.78. Always diff this report against the previous run, not the scalar, because a flat overall accuracy can hide a retrieval gain that exactly cancels an abstention regression.

### Evaluating the MANAGE edge directly

Everything above measures write and read through their effect on the final answer. The MANAGE edge — consolidation, deduplication, and forgetting — is harder to see because it produces no direct output, yet it is where stores rot over time. You can measure it without a language model at all, by probing the store itself across a capacity sweep.

```python
def capacity_sweep(make_system, sessions, probes, sizes=(10, 100, 1000)):
    """Re-run the eval at growing store sizes to expose degradation.
    A healthy MANAGE policy keeps accuracy flat and the store sub-linear;
    a broken one lets duplicates pile up until precision collapses."""
    rows = []
    for n in sizes:
        mem = make_system()
        subset = sorted(sessions, key=lambda s: s.timestamp)[:n]
        traces = evaluate(mem, subset, probes)
        rep = report(traces)
        rows.append({
            "sessions":    n,
            "accuracy":    rep["overall_accuracy"],
            "store_items": mem.size(),                       # exposed by the system
            "read_tokens": rep.get("retrieval", {}).get("read_tokens_p50"),
        })
    return rows
```

Two signals fall out of the sweep. If `store_items` grows roughly linearly with sessions while the true number of distinct facts is flat, the MANAGE policy is not deduplicating — it is hoarding, and precision will fall as the store grows. If `accuracy` declines as `sessions` climbs while `read_tokens` holds fixed, retrieval is being crowded out by accumulated noise. Neither shows up in a single fixed-size benchmark run, which is exactly why production memory systems regress in ways their benchmark numbers never predicted. A forgetting policy is a feature, and an unmeasured feature is a liability.

## 7. Cross-cutting concerns

Three issues cut across every benchmark and every metric. Ignoring them is how good-faith evaluations still produce wrong conclusions.

### Data contamination and annotation quality

Synthetic conversational benchmarks are generated by LLMs, and LLM-generated data has bugs. The benchmark you are scoring against may contain questions with missing ground truth, facts attributed to the wrong speaker, or multiple defensible answers. When that happens, a *better* system can score *lower* because it answers the genuinely-correct thing while the gold label is wrong. Before trusting a benchmark, sample 30 items by hand and check the labels; the rate of broken items is a hard ceiling on the resolution of any score derived from it. We will see a concrete, expensive instance of this in the case studies.

There is also leakage: if the benchmark was published before your model's training cutoff, the model may have memorized it. For a 2024 benchmark scored by a 2026 model, treat near-perfect scores with suspicion and prefer the newest, largest-scale benchmarks where memorization cannot substitute for retrieval.

### Statistical power and variance

A memory eval is a stochastic experiment: the judge is sampled, retrieval can be sampled, generation is sampled. A single run produces a single number with unknown variance, and two systems differing by "3 points" may be statistically indistinguishable. Run multiple seeds, report a confidence interval, and compute the minimum detectable effect for your eval-set size before you believe a ranking. This is the same discipline laid out in detail in [Evaluating Agent Trajectories: Beyond Final-Answer Accuracy](/blog/machine-learning/ai-agent/evaluating-agent-trajectories-beyond-final-answer) — the statistics of grading agents apply unchanged to grading their memory. A benchmark result quoted without error bars is an anecdote with a decimal point.

### The cost–latency–accuracy frontier

There is no single best memory system; there is a frontier. A system can win on accuracy by reading more tokens (slower, costlier) or win on latency by reading fewer (lower recall). The right way to present results is as points on a 2D frontier — accuracy versus tokens-per-query, or accuracy versus p95 latency — not as a single ranked list. A system that is 2 points more accurate but 80× slower is not "better"; it is a different point you might or might not want. Reporting only accuracy collapses this frontier and hides the trade-off the reader most needs to see.

### Privacy, consent, and multi-tenant leakage

Memory introduces a failure mode that has nothing to do with accuracy: a store that remembers across sessions can remember the *wrong* user's data, or surface a fact the user expected to be forgotten. Evaluation has to cover it. Add adversarial probes that ask for another tenant's information and score a leak as a hard failure, not a wrong answer — the cost of cross-user leakage is categorical, not a percentage point. Add deletion probes: write a fact, issue a delete, then query for it; the gold answer is "not mentioned," and a system that still returns the deleted fact fails a compliance test, not merely a quality one. The 2026 production retrospectives list privacy and consent among the explicitly unsolved problems, which means your evaluation is on its own here — no public benchmark will catch a leak in your multi-tenant deployment, so the probes have to be yours.

### Reproducibility: pin everything

The benchmark wars were, at bottom, reproducibility failures, so the defense is a checklist you run before quoting any number. Pin the **judge model by exact version ID** and store the **judge rubric by hash** — a silent model update or a reworded prompt changes every historical score. Pin the **dataset version**, because benchmarks get patched (LoCoMo's annotation fixes moved scores). Record the **harness commit**, the **retrieval $k$**, and the **random seeds**. Publish the **per-competency breakdown and the baselines**, not just the headline. And state the one thing vendors omit most often: *which question categories were included*, since quietly dropping a hard category is the easiest way to inflate an average without technically lying. A memory result you cannot reproduce from a config file is not a measurement; it is a press release.

## What a trustworthy memory-eval report contains

Pulling the threads together, a report you would actually stake a launch decision on has a fixed shape, and you can use it as a checklist when you read someone else's claims or write your own.

- **The baseline, run.** Full-context numbers on the same data, every time. Without it, "we beat the alternatives" omits the cheapest alternative.
- **Per-competency accuracy**, scored by a calibrated judge whose model ID and rubric hash are stated, with the judge's human-agreement rate (a Cohen's kappa, say) so the reader can trust the judge before trusting the scores.
- **Efficiency alongside accuracy**: read tokens per query (p50) and latency (p50 and p95). A point on the accuracy axis with no cost axis is half a result.
- **A capacity curve**, not just a fixed-size snapshot — accuracy and store size as the history grows, so the reader can see whether the system rots.
- **Variance**: multiple seeds, a confidence interval, and the eval-set size, so a 2-point "win" can be judged against the noise floor.
- **The dataset's known defects**, sampled and reported, so the reader knows the score's ceiling.

If a memory result is missing four of these six, you are not looking at an evaluation; you are looking at a number that survived selection. The rest of this section is what happens when the field forgets that.

## Case studies from the benchmark wars

The abstract failure modes above all have concrete, named instances. The two years from LoCoMo's release to BEAM's were a running argument: a SOTA claim, a rebuttal, a harder benchmark, repeat. The history is genuinely instructive, because every dispute was, at root, a disagreement about evaluation method — which is the subject of this post.

![Timeline of the agent-memory benchmark wars across two years: 2024-02 LoCoMo at 16k tokens, 2024-10 LongMemEval with 500 questions, 2025-04 Mem0 SOTA claim of about 68 percent on LoCoMo, 2025-05 Zep rebuttal recomputing 68 to 75 percent, 2025-07 MemBench plus MemoryAgentBench, and 2026-04 BEAM at 10M tokens with Mem0 at 92.5](/imgs/blogs/evaluating-memory-llm-agents-benchmarks-metrics-8.webp)

### 1. The Mem0-versus-Zep LoCoMo dispute

The headline incident. Mem0's paper reported a strong LoCoMo result (its best graph configuration around a 68% J score) and, in the course of it, evaluated competitor Zep and reported Zep behind. Zep published a rebuttal alleging three concrete harness bugs in how Mem0 had run *Zep*: the evaluation assigned the user role to *both* conversation participants, which confused Zep's per-user logic into treating the dialogue as one shifting identity; timestamps were appended to message text rather than passed through Zep's dedicated `created_at` field, sabotaging its temporal reasoning; and search was run sequentially instead of in parallel, inflating reported latency (0.778s versus a corrected 0.632s). With those fixed, Zep recomputed itself at 75.14% ± 0.17 — about 10% above Mem0. **The lesson is not who was right.** It is that a memory system's score is a function of how its API is driven, and a competitor evaluating a competitor has both the means and the motive to drive it badly. Never trust a system's number as reported by its rival; re-run it yourself, or read the harness line by line.

### 2. The full-context baseline that beat the memory system

Buried in the same dispute is the most damning number of all: a simple full-context baseline — feed the entire conversation to the model, no memory system at all — scored roughly **73%** on LoCoMo, *above* the specialized memory system's ~68%. Sit with that. The benchmark that launched a thousand memory papers was short enough (16–26k tokens) that doing nothing clever beat the clever thing. This is the single strongest argument for the `FullContext` baseline in our harness, and it generalizes: on any benchmark short enough to fit a modern context window, a memory system is fighting for scraps against a model that already remembers. If your evaluation does not include this baseline, your "win" may be a loss you never measured.

### 3. LoCoMo's broken Category 5

When researchers audited LoCoMo's annotations, they found structural defects: Category 5 was effectively unusable because of missing ground-truth answers; some multimodal questions referenced image content that the captioning model never produced; statements were attributed to the wrong speaker; and some questions had several defensible answers. Every one of these makes a *correct* system look *wrong*, capping the achievable score below 100% for reasons that have nothing to do with memory. The lesson is the one from the cross-cutting section made painfully concrete: **the benchmark is software, and software has bugs.** A score is only as trustworthy as the labels underneath it, and almost nobody checks the labels.

### 4. Zep's 84%-to-58% correction

The dispute cut both ways. A widely-circulated 84% LoCoMo figure associated with Zep was, under a corrected and independent evaluation, recomputed to roughly **58%**. A 26-point swing on the *same benchmark* and the *same system*, attributable entirely to evaluation methodology — judge prompt, question categories included, how the conversation was fed in. When two careful teams can produce numbers 26 points apart for one system on one dataset, the number is not the property of the system; it is the property of the evaluation. This is why "SOTA on LoCoMo" with no harness published is closer to a marketing claim than a measurement. The remedy is mechanical and unglamorous: every leaderboard row should ship the harness commit that produced it, so a third party can re-run it without reverse-engineering the setup. The teams that published their harness ended their disputes; the teams that did not kept them alive.

### 5. The F1-versus-judge ranking flip

A team comparing five representative memory architectures on LoCoMo scored them two ways: token-level F1 and an LLM judge. The two metrics produced *different system rankings*. Architectures that stored and returned abstractive summaries — paraphrased, condensed facts — were penalized by F1 for not matching reference wording, then vindicated by the judge for being correct. A team that had standardized on F1 (because it is cheap and deterministic) would have concluded the abstractive systems were worst and shipped the lexical one, which the judge ranked lower on actual correctness. The metric was the bug. This is exactly the inversion drawn in the metrics matrix earlier, and it is why we made the judge — calibrated, pinned, with a published rubric — the default scorer in the harness.

### 6. The LongMemEval cliff and the abstention hole

When LongMemEval ran commercial assistants and long-context models across lengthening histories, accuracy fell 30–60%, and the decomposition revealed *where*: abstention and temporal reasoning degraded fastest. Models that comfortably recalled a single recent fact would, asked about something that was never discussed, confidently fabricate a plausible answer instead of abstaining. Because abstention questions are a minority of most test sets, this failure is easy to average away into a respectable overall score — a system can look 85% good while being near-useless at the one behavior (saying "I don't know") that determines whether users trust it. If your eval set under-weights unanswerable questions, you are structurally blind to the hallucination your users will hit first.

### 7. BEAM's 1M-to-10M collapse

BEAM was built to test the regime everyone hand-waves: production scale. Its finding is sobering. Going from a 1M-token history to a 10M-token history — roughly a year of daily agent use — cost about **25%** accuracy, with temporal reasoning at scale identified as the single worst-degrading capability and explicitly named an unsolved problem. The systems that look excellent at benchmark scale are demonstrably not solved at deployment scale. The lesson for your own evaluation: **test at the size you will actually accumulate, not the size that is convenient to run.** A score at 30 sessions is a promissory note that may bounce at 3,000.

### 8. The stale-fact failure

The most mundane and most common production bug. A user updates a fact — new address, new job, new dietary restriction — and weeks later the agent answers with the *old* value, because retrieval surfaced the higher-similarity old memory and the system has no policy that newer supersedes older. This is the `MANAGE` edge of our loop failing silently: writes happened, reads happened, but consolidation and conflict resolution did not. It rarely shows up on short benchmarks (nothing changes in 16k tokens) which is why LoCoMo barely tests it and MemoryAgentBench was built partly to fix that. In your harness, this is precisely what the chronological-ingest plus the `knowledge_update` probe category exists to catch. A memory system that cannot tell you the user's *current* coffee order is not a memory system; it is a transcript with extra steps.

### 9. The token-efficiency illusion

By 2026 the best memory systems reported genuinely strong numbers — one hit 92.5 on LoCoMo and 94.4 on LongMemEval while averaging under ~7,000 tokens per query, against a full-context baseline burning ~26,000. That is a real and important result: matching or beating full-context accuracy at a quarter of the tokens is the entire economic case for memory, and it is the right thing to celebrate. But the headline invites a misread. The same report's own fine print listed the unsolved problems: a ~25% drop from 1M to 10M tokens, cross-session identity resolution, privacy and consent, and staleness in high-relevance facts. A 92.5 on a benchmark that fits inside a context window is not evidence that the 10M-token problem is solved — those are different regimes, and a strong small-scale number can lull a team into shipping confidently into the large-scale one. Read the efficiency win and the scale gap as two separate facts, because they are, and only the second one predicts what happens after a year in production.

### 10. Cross-session identity and the staleness tax

The failure that surfaces last in a deployment is identity. Over months, one user is described many ways — "my wife," "Sarah," "she" — and a memory system that does not resolve these to a single entity will fragment the person across the store, so a query about "Sarah" misses facts filed under "my wife." The 2026 production retrospectives name this explicitly, alongside *staleness*: a fact that was true and highly relevant when stored — a job, an address — silently goes wrong, and the very relevance that keeps it ranked highly is what keeps resurfacing the stale value. Both failures are invisible to a benchmark scored at a single point in time on a single coherent conversation. The only way to catch them is an evaluation that spans many sessions, refers to the same entities in different ways, and contains facts that change over time — which is to say, an evaluation that looks like real use rather than a clean dataset. The closer your probe set is to your actual traffic, the sooner you find the bug the public leaderboard will never show you.

## When to reach for a memory system — and when long context is enough

The honest conclusion of all this evaluation is sometimes "you don't need the thing you were about to build." A managed memory store adds writes, merges, evictions, retrieval, and an entire new surface of failure modes. It is worth that complexity only under specific conditions, and the decision is a short series of questions, not a default.

![Decision tree for whether to build a memory system: from state needed across turns, to whether it fits in a 128k context (use long context if yes), to whether reads span many sessions (use RAG over session logs if no), to whether facts update often (build a full memory system if yes, otherwise summary plus retrieval)](/imgs/blogs/evaluating-memory-llm-agents-benchmarks-metrics-9.webp)

Walk the tree top to bottom. The branches correspond directly to what your evaluation should stress.

The four leaves map to four engineering choices, each with its own cost and its own evaluation focus.

| Approach | Reach for it when | Main cost | What to test hardest |
| --- | --- | --- | --- |
| Plain prompt | no cross-turn state | none | nothing — there is no memory |
| Long context | history fits ~128k and you can pay to resend | tokens per turn | the lost-in-the-middle cliff |
| RAG over logs | reads span sessions but facts never change | retrieval infra | recall@k and abstention |
| Full memory system | reads span sessions *and* facts update | write + manage + evict complexity | knowledge update, conflict, capacity |

The table is also a warning: each step down the rows adds a failure surface, so do not descend a row until your evaluation shows the row above genuinely failing. Most teams who believe they "need a memory system" actually need the row above the one they are building, and only an honest eval — with the full-context baseline running — will tell them which.

**Reach for a dedicated memory system when:**

- Reads must span **many sessions or months**, far beyond any context window you can afford to re-read every turn.
- Facts **change over time** and the agent must prefer the current value — the knowledge-update and conflict-resolution competencies are load-bearing for your product.
- Read-time **cost or latency** matters and re-reading the full history per turn is economically or experientially unacceptable.
- You need **abstention** you can trust, which requires an explicit store you can check for presence rather than a model guessing from a giant blurry context.
- The agent accumulates **procedural skill** — verified tool sequences worth reusing — which raw context cannot index or compose.

**Skip it — use long context or plain retrieval — when:**

- The whole relevant history comfortably **fits in context** (under ~128k tokens) and you can afford to send it. Full-context is the simplest correct thing and, as the LoCoMo baseline showed, it is hard to beat at small scale.
- Interactions are **stateless or single-session**; there is nothing across turns to remember.
- Facts are **append-only and never updated**, in which case RAG over the raw session logs gives you retrieval without the cost of a write-and-manage policy.
- You have **no evaluation harness yet.** This is the one that catches teams: building a memory system before you can measure it means optimizing a number you cannot see. Build the harness — even the hundred-line one above — *first*. The benchmark you can run beats the architecture you can only hope about.

The discipline at the center of every section here is the same: a memory system is only as good as your ability to measure it, and the defaults all flatter you. Decompose memory into its loop. Test each competency, not just the final answer. Pick benchmarks by the competency you fear, run the full-context baseline that could embarrass you, score with a calibrated judge instead of lexical overlap, report tokens and tail latency alongside accuracy, and quote variance. Do that, and "it passed the benchmark" and "it remembers the user" finally start to mean the same thing.

## Further reading

- **LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory** (arXiv 2410.10813) — the five abilities, the 30–60% cliff, and the indexing/retrieval/reading decomposition.
- **Evaluating Very Long-Term Conversational Memory of LLM Agents** (LoCoMo, arXiv 2402.17753) — the foundational multi-session benchmark; read it alongside its critiques.
- **MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents** (arXiv 2506.21605) — effectiveness, efficiency, and capacity as three separate axes.
- **Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions** (MemoryAgentBench, arXiv 2507.05257) — the four competencies and the incremental-interaction setup.
- **Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs** (BEAM, arXiv 2510.27246) — the 1M and 10M tracks and why context size is not the answer.
- **Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers** (arXiv 2603.07670) — the survey that formalizes the write-manage-read loop and the three-axis taxonomy.
- On this blog: [Long-Term Memory for Conversational Agents: MemGPT and Beyond](/blog/machine-learning/ai-agent/long-term-memory-conversational-agents-memgpt), [Evaluating Agent Trajectories: Beyond Final-Answer Accuracy](/blog/machine-learning/ai-agent/evaluating-agent-trajectories-beyond-final-answer), [Demystifying Evals for AI Agents](/blog/machine-learning/ai-agent/eval-agents), and [Effective Context Engineering for AI Agents](/blog/machine-learning/ai-agent/effective-context-engineering-for-ai-agents).
