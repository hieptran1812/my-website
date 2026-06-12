---
title: "PaSa: Training an LLM Agent to Out-Search Google Scholar"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "How ByteDance turned comprehensive academic paper search into a long-horizon RL problem — a Crawler that searches and follows citations, a Selector that judges relevance, and a reward that beats Google+GPT-4o by 38 points of recall."
tags: ["llm-agent", "reinforcement-learning", "ppo", "academic-search", "information-retrieval", "qwen", "tool-use", "agent-training", "bytedance"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 51
---

If you have ever done a real literature survey — not "find me one paper about transformers" but "find me *every* paper that studies non-stationary reinforcement learning with UCB-style value-based algorithms" — you already know the dirty secret of academic search: a single query never finds everything. You search, you read the three papers that come back, you scroll to their related-work sections, you follow the citations, you search again with the terminology you just learned, and you repeat until the new papers stop being new. The search is not a lookup. It is a *traversal*, and the thing being traversed is the citation graph.

PaSa — short for **Pa**per **Sea**rch agent — is ByteDance Research and Peking University's answer to the question: what if we trained an LLM agent to do that traversal automatically, and trained it well enough that a 7-billion-parameter model beats every Google-based baseline by a wide margin? The headline number from their ACL 2025 paper is blunt: on real researcher queries, **PaSa-7b improves recall@20 by +37.78 percentage points and recall@50 by +39.90 points over the strongest Google baseline (Google + GPT-4o)**. It does this with two cooperating agents both fine-tuned from Qwen2.5-7B, a synthetic training set built for free out of conference related-work sections, and a reinforcement-learning reward that knows how to credit a search action for a paper it surfaced three hops later.

> **TL;DR.** Comprehensive scholarly search is a *long-horizon agentic task*, not a single retrieval call. PaSa splits it into a **Crawler** (an RL-trained policy with `[Search]`, `[Expand]`, and `[Stop]` tools that grows a candidate pool by following citations) and a **Selector** (a strong relevance classifier that also doubles as the Crawler's reward model). The Crawler is bootstrapped by imitating GPT-4o trajectories, then optimized with session-level PPO using a reward of `α·(new relevant papers) − action_cost` with cross-session value bonuses (`α=1.5`, `γ₁=0.1`). Training data — **AutoScholarQuery**, 35,347 (query, answer-set) pairs — is mined from related-work sections of ICLR/ICML/NeurIPS/ACL/CVPR papers, where the cited works *are* the ground-truth answers. The result: a 7B agent that out-recalls Google Scholar, ChatGPT search, GPT-o1, and even its own GPT-4o-driven variant.

This post is a full teardown: the two-agent architecture, the token-level MDP the Crawler operates in, the reward design and why naive RL would fail, how AutoScholarQuery turns published papers into free supervision, the full results and ablations, runnable pseudo-implementations of the agent loop and the reward, and an honest critique with an explicit "what would change my mind" line. By the end you should be able to reason about *why* the agentic formulation wins, where it breaks, and what you would change if you were shipping this in production.

## Why single-shot retrieval is the wrong tool

**The senior rule of thumb: if recall matters more than latency, never trust a single retrieval call.** Embedding search, BM25, and "Google + an LLM that writes one good query" all share the same structural ceiling — they map a query to a *fixed* ranked list in one shot. That is fine when the answer is one document. It is catastrophic when the answer is a *set* of two or three dozen papers scattered across sub-communities that use different vocabulary for the same idea.

Consider the AutoScholarQuery example from the paper: *"Which studies have focused on nonstationary RL using value-based methods, specifically Upper Confidence Bound (UCB) based algorithms?"* The ground-truth answer set includes papers like *"Reinforcement Learning for Non-Stationary Markov Decision Processes"* and *"Efficient Learning in Non-Stationary Linear Markov Decision Processes."* A single keyword query into Google might surface one or two of these. To find the rest, you have to read the ones you found, notice that they cite *each other* and a cluster of UCB-bandit papers, and follow those edges. That second hop is where single-shot retrieval simply has nothing to offer.

The numbers make the ceiling concrete. On RealScholarQuery — 50 queries from actual researchers — plain Google reaches recall@20 of **0.1834**, Google Scholar **0.1514**, and even Google with GPT-4o rewriting the query tops out at **0.2020**. Meanwhile PaSa-7b reaches **0.5798** recall@20 on the same benchmark. That is not a tuning difference; it is a difference in *kind*. The single-shot systems are doing one thing well; PaSa is doing a fundamentally different thing.

| Assumption (single-shot view) | Reality (agentic view) |
|---|---|
| The answer is the top-k of one ranked list. | The answer is a *set* spread across the citation graph; you must traverse to cover it. |
| A better query gets you better recall. | A better query gets you better *seeds*; recall comes from expansion after the seeds. |
| Reranking fixes precision; recall is the retriever's job. | Recall is a *search-policy* problem: when to expand vs. stop is learned, not fixed. |
| One model call per query. | Hundreds to thousands of model calls per query — read each candidate, decide each hop. |
| Latency budget is sub-second. | Latency budget is seconds-to-minutes; you are replacing a week of a PhD student's time. |

That last row is the one people underweight. PaSa is not competing with autocomplete. It is competing with a graduate student doing a literature review over an afternoon. Once you accept that the latency budget is "minutes, not milliseconds," the design space opens up enormously — you can afford to read every candidate paper with a 7B model, follow citations several layers deep, and let an RL policy decide where to spend the budget.

## The mental model

![PaSa end-to-end pipeline: a fine-grained query flows into the Crawler's search-and-expand loop, builds a candidate pool of hundreds to thousands of papers, then the Selector filters to a ranked answer set](/imgs/blogs/pasa-llm-paper-search-agent-1.webp)

The diagram above is the mental model, and the rest of this article is a tour of it. A user starts with a **fine-grained query** — not "RL papers" but a sentence-level information need. That query goes to the **Crawler**, which calls `[Search]` (Google restricted to `site:arxiv.org`) to fetch seed papers, then enters a loop: read a paper, decide whether to `[Expand]` it (pull the papers cited in specific subsections) or `[Stop]` (move on to the next paper in its queue). Each `[Expand]` adds more papers, which may themselves get expanded. The loop produces a **candidate pool** that can hold hundreds or even thousands of papers. That pool is then handed to the **Selector**, which reads each candidate's title and abstract against the query and emits a True/False relevance decision plus a rationale. The True papers, ranked, are the answer.

Two design choices in that picture are doing all the heavy lifting, and both are worth pausing on.

First: **the Crawler and Selector are separate agents with separate objectives.** The Crawler's job is recall — gather everything that *might* be relevant, err on the side of over-collecting. The Selector's job is precision — read the over-collected pool and throw out the noise. If you tried to do both in one model, you would constantly trade recall for precision inside a single forward pass, and you would have no clean signal to train either behavior. Splitting them means you can train the Crawler to maximize coverage and the Selector to maximize judgment, independently.

Second: **the Crawler is a policy, not a prompt.** The actions `[Search]`, `[Expand]`, `[Stop]` are tool calls the model learns *when* to emit. The interesting decisions are not "what to search for" (GPT-4o can write a fine query) but "given this paper, is its reference list worth the cost of expanding, or should I stop and move on?" That is a sequential decision under a budget, which is exactly what reinforcement learning is for. We will spend most of this post on how that policy is trained.

If you have read [Anthropic's building-effective-agents guide](/blog/machine-learning/ai-agent/building-effective-agents-hands-on-guide) on this blog, PaSa is a textbook case of the "orchestrator + specialized workers" pattern, except the orchestrator is itself trained with RL rather than prompted — which is what makes it interesting.

## 1. The Crawler: a policy over `[Search]`, `[Expand]`, `[Stop]`

**Senior rule of thumb: the hard part of an agent is not the tools, it is the stopping rule.** Anyone can wire up a search tool and a citation-expansion tool. The skill PaSa learns is *when to stop expanding a branch that has stopped paying off* and *when to keep digging into one that is still producing relevant papers.* That is what RL buys you.

The Crawler registers exactly three functions:

- **`[Search]`** — the model generates a search query string and invokes Google (restricted to `site:arxiv.org`). The returned papers are appended to a **paper queue**.
- **`[Expand]`** — the model picks subsection names from the paper it is currently reading (e.g., "Related Work", "Background", a specific method section) and the papers *cited in those subsections* are added to the queue. This is the citation-following step — the one that single-shot retrieval cannot do.
- **`[Stop]`** — the model resets its working context and pops the next paper off the queue to process. `[Stop]` is how a session ends.

The control loop is straightforward to state and subtle to learn. The Crawler starts with the query, issues one or more `[Search]` calls to seed the queue, then repeatedly: pop a paper, read it, and decide — `[Expand]` (and which subsections) or `[Stop]`. The expanded papers go back into the queue. The loop ends when the queue is exhausted (or a depth/budget limit is hit; the paper limits exploration depth to three). Everything in the queue at the end is the candidate pool.

Here is the loop as runnable-shaped Python pseudo-implementation. It is deliberately close to what `run_paper_agent.py` in the [PaSa repo](https://github.com/bytedance/pasa) does, minus the API plumbing:

```python
from dataclasses import dataclass, field
from collections import deque

@dataclass
class PaperNode:
    title: str
    abstract: str
    arxiv_id: str
    source: str          # "search" or "expand:<parent_id>"
    depth: int = 0       # citation hops from a seed

@dataclass
class CrawlerState:
    query: str
    queue: deque = field(default_factory=deque)   # papers waiting to be read
    seen: set = field(default_factory=set)         # arxiv_ids already queued (dedupe!)
    pool: list = field(default_factory=list)       # the candidate pool

MAX_DEPTH = 3        # the paper caps citation expansion at depth 3
MAX_PAPERS = 1500    # a practical budget; the queue can hold "hundreds to thousands"

def crawl(query, policy, google_search, fetch_paper):
    state = CrawlerState(query=query)

    # --- seed phase: the policy emits one or more [Search] actions ---
    for search_query in policy.initial_searches(query):
        for hit in google_search(search_query, site="arxiv.org"):
            _enqueue(state, fetch_paper(hit), depth=0, source="search")

    # --- expansion phase: read each paper, decide [Expand] or [Stop] ---
    while state.queue and len(state.pool) < MAX_PAPERS:
        paper = state.queue.popleft()
        state.pool.append(paper)

        # The policy reads (query, paper) and returns an action.
        action = policy.act(state.query, paper)

        if action.kind == "EXPAND" and paper.depth < MAX_DEPTH:
            # The policy names which subsections to follow.
            for cited in fetch_paper.citations(paper, action.subsections):
                _enqueue(state, cited, depth=paper.depth + 1,
                         source=f"expand:{paper.arxiv_id}")
        # action.kind == "STOP": fall through, pop the next paper.

    return state.pool

def _enqueue(state, paper, depth, source):
    if paper.arxiv_id in state.seen:    # dedupe: a paper found twice earns reward once
        return
    state.seen.add(paper.arxiv_id)
    paper.depth, paper.source = depth, source
    state.queue.append(paper)
```

Three details in that code are load-bearing and worth calling out, because they are exactly where a naive implementation would go wrong.

**Deduplication is not an optimization, it is a correctness requirement.** The citation graph is dense and cyclic; the same paper will be reached along many paths. If you let it into the queue twice, you waste budget reading it twice *and* — more importantly — you corrupt the training signal, because the reward (below) only pays for *newly discovered* relevant papers. The `seen` set is what makes "new" well-defined.

**Depth is capped.** Citation expansion is exponential. A paper cites ~30 others; each of those cites ~30 more. Without a depth cap, the queue explodes. The paper limits exploration depth to three, which is enough to cross sub-community boundaries (the thing single-shot search cannot do) without drowning in the long tail.

**`[Expand]` is selective about subsections, not whole bibliographies.** The model does not blindly add every reference. It names the subsections worth following — and learning *that* selectivity is a big part of what RL teaches. Expanding the "Related Work" of a survey is gold; expanding the "Acknowledgements" or a tangential "Limitations" paragraph is noise.

### Citation expansion as a branching search

![Citation expansion as a branching search tree: the query seeds two papers via Search, each Expand call either keeps a productive branch (green) toward relevant papers or stops a low-yield branch (red), with off-topic results discarded](/imgs/blogs/pasa-llm-paper-search-agent-2.webp)

The figure above shows what the loop actually *does* over a few hops. The query seeds two papers (`[Search]`). The policy keeps seed A and expands its related-work references — a productive branch that surfaces a relevant paper. Seed B branches two ways: one `[Expand]` into its method-section references (which surfaces a relevant paper but also an off-topic one that gets dropped downstream) and one `[Stop]` because the branch looks low-yield. The shape is a *branching search over the citation DAG*, and the policy's whole job is to allocate its finite budget toward the green branches and away from the red ones.

This is why I keep insisting the Crawler is a policy and not a prompt. A prompted agent can follow the loop, but it has no principled way to decide "this branch has stopped paying off." It either expands everything (blowing the budget and tanking precision) or expands timidly (tanking recall). The RL-trained Crawler learns a *calibrated* expansion policy — and the ablations later show that removing the learned expansion behavior is the single biggest recall hit in the whole system.

### The token-level MDP

To train the Crawler with RL we have to be precise about the decision process. PaSa frames it as a **token-level Markov Decision Process** where the LLM is the policy:

- **State** $s_t$: the current LLM context (the query, the paper being read, the actions taken so far) plus the paper queue.
- **Action** $a_t$: the next token from the LLM's vocabulary. The *action space is the entire vocabulary*. Most tokens just extend the text; but when the generated tokens spell out a registered function name — `[Search]`, `[Expand]`, `[Stop]` — the corresponding tool fires and mutates the state (enqueues papers, pops the queue).
- **Policy** $\pi_\theta(a_t \mid s_t)$: the LLM with parameters $\theta$.

Framing it at the token level (rather than treating each whole tool call as one atomic action) matters because the LLM *generates the action as text* — the search query string, the list of subsections to expand — and we want the reward to flow back to every token that contributed. This is the same modeling choice you see in modern agent RL across the field; if you want the broader picture of how trajectory-level signals get assigned to intermediate steps, the companion post on [evaluating agent trajectories beyond the final answer](/blog/machine-learning/ai-agent/evaluating-agent-trajectories-beyond-final-answer) is a good complement.

The catch with this clean formulation is that the trajectories are *enormous*. Reading hundreds of papers, each with its own expand/stop decision, produces a token sequence far longer than anything you would optimize with vanilla PPO. The reward is also *sparse* — most actions produce nothing, and the payoff (a relevant paper) arrives many steps after the action that caused it. Those two problems — long horizon and sparse, delayed reward — are what the training design has to solve, and they are the subject of section 4.

#### Second-order gotcha: context management across the queue

A non-obvious failure mode lurks in the `[Stop]` semantics. Recall that `[Stop]` *resets the context* to process the next paper in the queue. That reset is deliberate and important: if the Crawler carried the full text of every paper it had ever read into the context for every subsequent decision, the context would balloon past the model's window within a handful of papers, and the cost per decision would grow linearly with how far into the crawl you are. By resetting context on `[Stop]`, each expand/stop decision is made on a *bounded* context — the query plus the single paper currently being read. The state $S_{q+p}$ in the MDP is exactly that bounded context, which is also why session-level PPO can treat each paper-read as an independent short session.

The consequence is that the Crawler's per-paper decision is *Markovian by construction* — it does not remember the specific papers it read 50 steps ago, only the accumulated queue and seen-set. This is a feature: it keeps the policy's input distribution stationary (always "query + one paper"), which makes RL far more stable than it would be over an ever-growing context. But it is also a limitation — the Crawler cannot make a decision like "I have already found enough papers from this sub-area, stop expanding it" unless that information is encoded in the queue/seen state rather than in remembered context. PaSa accepts this trade for the stability and cost wins, and the results suggest the queue-based state carries enough signal. If you were designing a follow-up, *summarized cross-paper memory* (a compact running summary of what's been covered) would be the obvious thing to add — and the obvious thing to break stability if done carelessly. [Effective context engineering for agents](/blog/machine-learning/ai-agent/effective-context-engineering-for-ai-agents) on this blog is the right companion read for that design space.

## 2. The Selector: a relevance classifier that doubles as a reward model

**Senior rule of thumb: if your agent needs a reward signal at inference-impossible scale, build the reward model into the system you are already shipping.** PaSa's most elegant move is that the Selector is *not* an auxiliary component bolted on for training — it is the production relevance filter, and it happens to also be exactly the judge the Crawler needs during RL.

The Selector takes a (query, paper) pair — where "paper" is the title and abstract — and outputs two things:

1. A single **decision token**: `"True"` (relevant) or `"False"` (not relevant).
2. A **rationale**: a short explanation of the decision.

The single-token decision is the design detail that makes the Selector usable as a reward model. Because relevance collapses to one token, you can read out a *probability* of relevance directly from the logit, score thousands of candidate papers cheaply, and use that score as a dense reward signal during Crawler training. A free-text "yes, because..." answer would force you to parse natural language to extract a reward; a single token does not.

The paper experiments with decision-first vs. rationale-first ordering (chain-of-thought before the decision). Here are the Selector evaluation numbers as a relevance classifier:

| Selector variant | Precision | Recall | F1 |
|---|---|---|---|
| GPT-4o (prompted) | 0.96 | 0.69 | 0.80 |
| Qwen2.5-7B (prompted, untrained) | 1.00 | 0.38 | 0.55 |
| **PaSa-7b-Selector** | 0.95 | **0.78** | **0.85** |
| PaSa-7b-Selector (reason-first) | 0.94 | 0.76 | 0.84 |

The trained 7B Selector hits an F1 of **0.85**, beating prompted GPT-4o by 5 points and the untrained Qwen base by a full 30 points. Notice the *shape* of the win: precision is roughly tied across all the contenders (everyone is good at saying "no" to obviously irrelevant papers), but recall is where training pays off. The untrained Qwen is precision-1.0/recall-0.38 — it is a coward, only saying "True" when it is certain, which is great for precision and terrible for actually finding things. Training pushes recall to 0.78 while barely touching precision. That is the right trade for a system whose whole reason to exist is comprehensive coverage.

Here is the Selector as runnable-shaped code, including the read-out of the relevance score from the decision token — the part that lets it serve as a reward model:

```python
import torch
import torch.nn.functional as F

SELECTOR_PROMPT = """You are judging whether a paper is relevant to a scholarly query.
Query: {query}
Paper title: {title}
Paper abstract: {abstract}

Decide if the paper is relevant. Answer with a single token, True or False,
then give a one-sentence rationale.
Decision:"""

def selector_score(model, tokenizer, query, title, abstract):
    prompt = SELECTOR_PROMPT.format(query=query, title=title, abstract=abstract)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]   # logits for the next token

    true_id  = tokenizer.encode(" True",  add_special_tokens=False)[0]
    false_id = tokenizer.encode(" False", add_special_tokens=False)[0]

    # Softmax over just the two decision tokens -> P(relevant).
    pair = torch.tensor([logits[0, false_id], logits[0, true_id]])
    p_relevant = F.softmax(pair, dim=-1)[1].item()

    decision = p_relevant > 0.5
    return decision, p_relevant          # p_relevant doubles as the reward signal
```

The trick of reading a calibrated probability off a single decision token is not unique to PaSa — it is the standard way to turn a generative LLM into a cheap classifier or reward model — but PaSa's contribution is recognizing that *the same model you ship for filtering is the judge you need for training*. You do not have to train, host, and maintain a separate reward model. The Selector is the reward model. That is a real systems win, and it is the kind of consolidation that only becomes obvious once you have built a few of these pipelines and gotten tired of keeping three models in sync.

#### Decision-first vs. reason-first: a deliberate ordering choice

The table above has two PaSa Selector rows — the default (decision token first, then rationale) and a "reason-first" variant (chain-of-thought rationale, then the decision token). Conventional wisdom from the chain-of-thought literature says reasoning *before* answering should help: the model thinks, then commits. So why does PaSa default to decision-*first*, where the model commits before it explains?

Two reasons, and both are pragmatic rather than ideological. First, **the reward read-out needs the decision token at a fixed, early position.** If the decision comes after a variable-length rationale, you have to generate the whole rationale before you can read the relevance probability — multiplying the cost of every reward query during RL by the length of the rationale. With decision-first, the relevance logit is available at token one; you can score thousands of candidates without generating a single rationale token if you only need the score. For a reward model invoked millions of times across 16k PPO episodes, that is the difference between feasible and ruinous.

Second, the numbers say the ordering barely matters for *quality*: decision-first is F1 0.85, reason-first is 0.84. The rationale is not driving the decision much — the model already "knows" relevance from the title and abstract, and the rationale is mostly post-hoc justification. Given that the quality is a wash and decision-first is dramatically cheaper to use as a reward model, decision-first is the obvious default. This is a nice illustration of a senior instinct: when two designs are quality-equivalent, pick the one that is cheaper *in the loop you call most often.* The rationale still ships — it is useful for the human reading the final results — but it is generated lazily, only for the papers that survive into the answer set.

> When two designs tie on quality, the tiebreaker is which one is cheaper in your hottest loop — and for a reward model, the hottest loop runs millions of times.

### The Selector as reward model, in the loop

During Crawler RL, every time the Crawler discovers a new paper, the Selector scores it. If the Selector says "True" (or the paper is in the AutoScholarQuery ground-truth set), that discovery earns reward. The Selector's recall therefore directly caps the Crawler's effective reward signal — a Selector that misses relevant papers will fail to reward the Crawler for finding them, and the Crawler will learn not to bother. This coupling is why the Selector's 0.78 recall matters beyond the final filtering step: it is the quality of the teacher. The ablation where the Selector is *removed* as a reward model (and only ground-truth labels are used) shows a measurable recall drop, which we will quantify in section 5.

## 3. The two-agent architecture, end to end

![The two-agent architecture as a layered stack: the fine-grained query feeds the Crawler with Search/Expand/Stop tools, then the Selector reads title and abstract to emit True/False plus rationale, both fine-tuned from a shared Qwen2.5-7B base](/imgs/blogs/pasa-llm-paper-search-agent-6.webp)

The figure above stacks the whole system so you can see the division of labor. The input is a fine-grained scholarly query. The Crawler — `[Search]`/`[Expand]`/`[Stop]` tools driven by an RL-trained policy — gathers the candidate pool. The Selector — read title+abstract, emit True/False + rationale — judges each candidate. And critically, **both agents are fine-tuned from the same Qwen2.5-7B base**, but as *separate* fine-tunes: `bytedance-research/pasa-7b-crawler` and `bytedance-research/pasa-7b-selector` on HuggingFace.

Sharing a base model but splitting the fine-tunes is a pragmatic choice with real consequences. On the plus side, you get one family of weights to reason about, one tokenizer, one serving stack. On the minus side, you are hosting *two* 7B models at inference time, and the Crawler model is invoked once per paper read (potentially thousands of times per query) while the Selector is invoked once per candidate (also thousands of times). PaSa is not cheap to run — but remember the latency budget. It is replacing an afternoon of expert human time, not an autocomplete dropdown.

Why not a single model doing both jobs with different prompts? Two reasons. First, **objective conflict**: the Crawler wants to over-collect (recall) and the Selector wants to be strict (precision); training one set of weights toward both pulls in opposite directions. Second, **reward cleanliness**: the Selector has to serve as the Crawler's reward model, and you cannot cleanly use a model as its own reward signal during its own RL update without inviting reward hacking. Keeping them as separate fine-tunes keeps the training signal honest.

This is the same architectural instinct behind a lot of good multi-agent design — specialize the workers, keep their objectives from fighting — and it echoes the patterns covered in [advance tool use](/blog/machine-learning/ai-agent/advance-tool-use) and the broader [agent evaluation](/blog/machine-learning/ai-agent/eval-agents) discussions on this blog. PaSa is a clean, *trained* instance of the orchestrator/worker split.

## 4. Training the Crawler: imitation, then session-level PPO

This is the heart of the paper, and the place where most "let's just prompt an agent" projects fall down. You cannot prompt your way to a calibrated expand/stop policy. You have to train it. PaSa trains the Crawler in two stages.

![Crawler training pipeline timeline: GPT-4o rollouts generate 12,989 imitation trajectories, imitation SFT runs one epoch, then session-level PPO samples sub-sessions, scores them with the Selector reward, and applies PPO updates over 16k episodes](/imgs/blogs/pasa-llm-paper-search-agent-7.webp)

The timeline above is the whole pipeline. Let us walk it.

### Stage 1: imitation learning (the bootstrap)

You cannot start PPO from a model that does not even know the action format. So PaSa first generates **12,989 trajectories** by prompting GPT-4o to play the Crawler — 3,011 search sessions and 9,978 expand sessions — and fine-tunes Qwen2.5-7B to imitate them for **1 epoch** at learning rate `1e-5`. This is standard behavior cloning: it teaches the base model the action vocabulary, the loop structure, and a *mediocre but functional* search policy. After imitation, you have an agent that crawls — just not very well. (How well? The ablation "w/o RL training" below is exactly this imitation-only model, and it is decent but clearly behind the RL-trained version.)

The SFT hyperparameters from the repo, for both the Crawler and the Selector:

```bash
## examples/scripts/sft.py  (from the hyc2026/trl fork)
python sft.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --learning_rate 1.0e-5 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --max_seq_length 1024 \
  --weight_decay 0.01 \
  --warmup_ratio 0.01 \
  --attn_implementation flash_attention_2
```

### Stage 2: session-level PPO (the part that wins)

Now the long-horizon and sparse-reward problems bite. A full crawl trajectory — seed, then read hundreds of papers with an expand/stop decision each — is far too long to fit in a single PPO rollout, and the reward is concentrated on a few "found a relevant paper" events scattered across that long sequence.

PaSa's solution is **session-level PPO**. A **session** begins at an initial state and ends with a `[Stop]` action. There are two kinds of initial state:

- $S_q$ — query only (the seed/search session: the agent issues `[Search]` calls).
- $S_{q+p}$ — query plus a specific paper (an expand session: the agent reads one paper and decides expand/stop).

A complete trajectory is *partitioned* into these sub-sessions, and PPO optimizes over sessions rather than over whole trajectories. This is the key trick: it makes the horizon manageable (each session is short — read one paper, take a few actions, stop) and makes sampling computationally efficient (you can sample sub-sessions independently rather than rolling out an entire multi-hundred-paper crawl every update).

But partitioning into sessions creates a new problem. If you only reward a session for papers *it* found, you destroy the entire point of the system: an `[Expand]` action that queues a paper which turns out to be relevant *three sessions later* would get zero credit, because the reward shows up in a different session. That is the credit-assignment problem, and PaSa solves it with a value-function bonus across sessions.

### The reward function

![Reward shaping over a search trajectory: a Search action with cost 0.1 queues a paper that carries a value-function bonus, an Expand action earns +alpha=1.5 for each newly relevant paper (zero for duplicates), and the cross-session value bonus (gamma1=0.1) flows credit back to the action that queued the paper](/imgs/blogs/pasa-llm-paper-search-agent-8.webp)

The figure above traces how credit flows. The instantaneous reward for taking action $a_t$ in state $s_t$ is:

$$r(s_t, a_t) = \alpha \sum_i \mathbb{I}(q, p_i, t) - c(a_t)$$

where:

- $\mathbb{I}(q, p_i, t) = 1$ if paper $p_i$ matches query $q$ — *either* the Selector judges it relevant *or* it appears in the AutoScholarQuery ground-truth set — **and** it has not already been queued. Otherwise $0$. (The "not already queued" clause is why deduplication is correctness-critical: a duplicate earns nothing.)
- $\alpha = 1.5$ is the reward per newly discovered relevant paper.
- $c(a_t)$ is the action cost: $c(\texttt{[Search]}) = 0.1$, $c(\texttt{[Expand]}) = 0.1$, $c(\texttt{[Stop]}) = 0$.

Read that reward carefully, because every term is fighting a specific failure mode:

- The **$\alpha$ term rewards coverage** — find a relevant paper nobody has seen yet, get +1.5. This is the whole objective.
- The **$c$ cost on Search and Expand discourages thrashing** — every search or expand costs 0.1, so the policy cannot just expand everything for free. It must believe an expansion will pay off. This is what teaches the *stopping* behavior: expanding a low-yield branch costs 0.1 and returns nothing, so the policy learns to `[Stop]` instead.
- **`[Stop]` is free** ($c = 0$) — there is no penalty for moving on, which means the policy is never punished for being decisive about abandoning a dead branch.

### Cross-session credit assignment

The dedup-aware reward above only fires when a paper is *first queued*. But the action that deserves credit for a relevant paper is often the `[Expand]` that queued it, while the *confirmation* that it is relevant happens later. PaSa handles this with a discounted return that adds a **value-function bonus** for newly queued papers.

When an action queues new papers, the return includes a term

$$\gamma_1 \sum_j \hat{V}_\phi(S_{q + p_j})$$

where $\hat{V}_\phi(S_{q+p_j})$ is the value function's estimate of the expected future reward from the session that will eventually process the newly queued paper $p_j$, and $\gamma_1 = 0.1$ is the **across-session discount factor**. In words: *queuing a paper that the value function thinks will lead to relevant discoveries earns discounted credit right now*, even though the actual discovery happens in a future session. The in-session discount is $\gamma_0 = 1.0$ (no discounting within a short session) and the across-session discount is $\gamma_1 = 0.1$ (heavy discounting across sessions, because the causal link weakens with distance).

This is the mechanism that makes the whole "follow a citation that pays off three hops later" story trainable. Without it, session-level PPO would myopically optimize each session in isolation and the agent would never learn to expand toward distant payoffs. The dashed arrow in the figure above — from "paper queued" back to "return" via the value bonus — *is* that long-range credit path.

A **KL penalty** with coefficient $\beta = 0.1$ keeps the policy from drifting too far from the imitation baseline, which is the usual PPO regularizer against reward hacking and policy collapse.

Here is the reward computation as runnable-shaped code, which makes the dedup and value-bonus logic concrete:

```python
ALPHA   = 1.5     # reward per newly discovered relevant paper
GAMMA1  = 0.1     # across-session discount on the value bonus
COST    = {"SEARCH": 0.1, "EXPAND": 0.1, "STOP": 0.0}

def session_reward(action, newly_queued, query, selector, ground_truth,
                   seen_ids, value_fn):
    """Reward for one action in a session.

    newly_queued : papers this action just added to the queue (post-dedup)
    seen_ids     : arxiv_ids already in the queue BEFORE this action
    value_fn     : V_phi, estimates future reward of processing a queued paper
    """
    # 1) Immediate coverage reward: alpha for each NEW relevant paper.
    coverage = 0.0
    for p in newly_queued:
        if p.arxiv_id in seen_ids:        # duplicate -> indicator is 0
            continue
        relevant = (p.arxiv_id in ground_truth) or selector(query, p)
        if relevant:
            coverage += ALPHA
        seen_ids.add(p.arxiv_id)

    # 2) Action cost discourages thrashing; [Stop] is free.
    cost = COST[action.kind]

    # 3) Cross-session value bonus: discounted expected future reward of
    #    the papers this action just queued (the long-range credit path).
    value_bonus = GAMMA1 * sum(value_fn(query, p) for p in newly_queued)

    return ALPHA * 0 + coverage - cost + value_bonus
```

The PPO training command from the repo ties the hyperparameters to the equations above — note `expand_select_score` / `search_select_score = 1.5` (that is $\alpha$), the matching `0.1` costs, `gamma1 0.1`, and `kl_coef 0.1`:

```bash
## examples/scripts/ppo/ppo_tldr.py  (from the hyc2026/trl fork)
accelerate launch --config_file deepspeed_zero3.yaml ppo_tldr.py \
  --model_name_or_path <imitation_checkpoint> \
  --learning_rate 1e-6 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --total_episodes 16000 \
  --local_rollout_forward_batch_size 4 \
  --response_length 1024 \
  --gamma1 0.1 \
  --vf_coef 10.0 \
  --expand_select_score 1.5 --expand_cost 0.1 \
  --search_select_score  1.5 --search_cost  0.1 \
  --num_ppo_epochs 2 \
  --kl_coef 0.1 \
  --save_steps 10 --rounds 3
```

Two of those flags are worth flagging for anyone who tries to reproduce this. `vf_coef 10.0` is *enormous* by PPO standards (the usual range is 0.5–1.0). That tells you the value function is doing heavy lifting here — which makes sense, because the cross-session credit assignment *is* the value function. If $\hat{V}_\phi$ is bad, the long-range credit path is bad, and the agent never learns to expand toward distant payoffs. They are willing to spend a lot of optimization pressure keeping the critic accurate. The `learning_rate 1e-6` is also conservative, which is typical when you are doing PPO on top of an already-functional SFT policy and want small, stable updates.

### A worked example of the return computation

Numbers make the credit-assignment story click. Suppose the Crawler is reading seed paper A and decides to `[Expand]` its related-work section. That single action queues three new papers: $p_1$, $p_2$, $p_3$. None has been seen before. The Selector judges $p_1$ and $p_2$ relevant and $p_3$ irrelevant. The value function estimates the future reward of processing each queued paper as $\hat{V}_\phi(p_1) = 2.0$, $\hat{V}_\phi(p_2) = 0.5$, $\hat{V}_\phi(p_3) = 0.1$ (the critic thinks $p_1$ is a hub paper whose own references will surface more relevant work).

The immediate reward for this `[Expand]` is:

$$r = \alpha \cdot (\underbrace{1}_{p_1} + \underbrace{1}_{p_2} + \underbrace{0}_{p_3}) - c(\texttt{[Expand]}) = 1.5 \cdot 2 - 0.1 = 2.9$$

The cross-session value bonus added to the return is:

$$\gamma_1 \sum_j \hat{V}_\phi(p_j) = 0.1 \cdot (2.0 + 0.5 + 0.1) = 0.26$$

So the total return credited to this `[Expand]` action is $2.9 + 0.26 = 3.16$. Now contrast that with an `[Expand]` that queues three *already-seen* papers: the indicator is 0 for all three (duplicates earn nothing), there is no value bonus (already-queued papers contribute no new $\hat{V}$), and the return is just $-c = -0.1$. The policy *loses* 0.1 for re-expanding a mined branch. That negative return is precisely the gradient signal that teaches the Crawler to `[Stop]` instead of re-mining. The asymmetry between $+3.16$ for a productive expand and $-0.1$ for a wasteful one is the entire learning signal in miniature.

Notice what the value bonus does even when the immediate payoff is modest. Imagine an `[Expand]` that queues three papers the Selector currently rates as irrelevant (indicator 0, so immediate coverage reward is 0), but the critic thinks one of them ($\hat{V}_\phi = 4.0$) is a survey whose references will be a goldmine. The immediate reward is $-0.1$ (just the cost), but the return is $-0.1 + 0.1 \cdot 4.0 = +0.3$. The agent gets *positive* return for queuing a paper that is not itself relevant but leads to relevant work. This is exactly the "follow the citation that pays off three hops later" behavior, and it only exists because of the $\gamma_1 \hat{V}_\phi$ term. Strip that term out (the "w/o Selector as RM" ablation is adjacent to this) and the agent becomes myopic — it only expands toward papers that are *immediately* relevant, missing the bridges.

### Why session-level PPO and not GRPO or vanilla PPO

A reasonable question in 2026 is why PaSa used session-partitioned PPO rather than a simpler trajectory-level method or one of the critic-free variants (GRPO and friends) that became popular for LLM RL. The answer is the horizon. Critic-free methods estimate advantage from a group of full-trajectory samples and a baseline; they work beautifully when a trajectory is "generate one answer, get one reward." A PaSa trajectory is "read 800 papers, take 800+ expand/stop decisions, collect reward at dozens of scattered discovery events." Sampling enough *full* trajectories to get a low-variance group baseline would be astronomically expensive, and the per-action credit would be hopelessly diffuse.

Session partitioning sidesteps this by turning one giant trajectory into many short ones (read one paper, decide, stop), each of which is cheap to sample and has a tight, local reward. The price you pay is that you now need the explicit value function to carry credit *across* the session boundaries you just introduced — which is why PaSa keeps the critic (and weights it heavily) rather than going critic-free. It is a deliberate trade: shorter horizons and cheaper sampling, at the cost of needing a good critic. The `vf_coef 10.0` is the visible fingerprint of that trade. If you are designing RL for a long-horizon agent, this is the template to steal: partition into short, locally-rewarded sessions, then use a value function to stitch the credit back together.

## 5. AutoScholarQuery: free supervision from related-work sections

**Senior rule of thumb: the best training data is data that already exists for free, you just have to recognize the labels.** PaSa's data engine is its quietest brilliance. There is no human annotation in the training set at all — and yet every query has a verified ground-truth answer set.

![AutoScholarQuery construction as a tree: conference papers feed their related-work paragraphs and citation contexts, GPT-4o writes a scholarly query while the cited papers become the answer set, and arXiv resolution yields 35,347 (query, answer) pairs](/imgs/blogs/pasa-llm-paper-search-agent-4.webp)

The figure above shows the construction. The insight: **a related-work paragraph in a published paper is already a (query, answer-set) pair waiting to be extracted.** When an author writes "Several works have studied non-stationary RL with UCB-based methods [12, 27, 33]," they have implicitly posed a scholarly query ("what works study non-stationary RL with UCB methods?") and answered it (the papers they cited). PaSa harvests this at scale:

1. Take papers from top AI conferences: **ICLR 2023, ICML 2023, NeurIPS 2023, ACL 2024, and CVPR 2024.**
2. For each related-work paragraph / citation context, prompt **GPT-4o to write the scholarly query** that the paragraph is implicitly answering.
3. The **cited papers in that paragraph are the ground-truth answer set** — resolved to their arXiv entries.
4. Filter and verify.

The result is **AutoScholarQuery: 35,347 instances** — split 33,511 train / 1,000 dev / 1,000 test (the paper reports 33,551 train in one table and 33,511 in another; the rounding is in the source). Manual quality review found **94.0% of queries were qualified** and **93.7% had relevant corresponding papers** — high enough that the noise is a rounding error against the signal.

The answer sets are small and focused — the average number of answer papers per query, by venue:

| Source venue | Avg. answer papers / query |
|---|---|
| ICLR 2023 | 2.46 |
| ICML 2023 | 2.37 |
| NeurIPS 2023 | 2.59 |
| CVPR 2024 | 2.94 |
| ACL 2024 | 2.16 |

That ~2.5-papers-per-query average is important context for the recall metrics. These are *fine-grained* queries with small, specific answer sets — exactly the regime where single-shot retrieval struggles (the 2-3 right papers are needles, and they are scattered) and where agentic citation-following shines (find one needle, follow its citations to the others).

There is a subtle and honest limitation baked into this data engine, and PaSa is upfront about it: the ground-truth answer set is *only the papers the original author chose to cite*. A genuinely comprehensive search might find relevant papers the author missed or that were published after. So AutoScholarQuery's "recall" is recall against the author's citation list, not against the platonic ideal of all relevant papers. This is why they also built RealScholarQuery with human annotation — to check whether the agent's wins on the proxy metric hold up against real, fully-annotated relevance judgments. (They do, by even larger margins, which is the reassuring outcome.)

### Why this matters for generalization

Training on author-citation pairs teaches the Crawler something specific and transferable: *how relevant papers cluster in the citation graph*. The author who cited [12, 27, 33] together is telling the model "these three live near each other in citation space." The Crawler learns the topology of relevance — which is exactly the prior you want when you deploy it on a brand-new query whose answer set you do not know. It is not memorizing answers; it is learning the *shape* of how answers are distributed, which is why a model trained on 2023-2024 conference citations generalizes to 2024-10 real researcher queries.

## 6. Results: a 7B agent beats Google + GPT-4o

Now the payoff. PaSa is evaluated on two benchmarks: **AutoScholarQuery test** (1,000 held-out synthetic queries) and **RealScholarQuery** (50 real researcher queries with full human annotation — annotators reviewed an average of 76 candidate papers per query, pooled from PaSa, Google, Google Scholar, ChatGPT, and GPT-4o, with a query date of 2024-10-01 to fix the candidate universe).

![RealScholarQuery results matrix: Google (red, 0.18-0.25), Google+GPT-4o (amber, 0.20-0.29), PaSa-GPT-4o (amber, overall recall 0.31), and PaSa-7b (green, 0.58/0.66/0.69) across recall@20, recall@50, and recall@100](/imgs/blogs/pasa-llm-paper-search-agent-5.webp)

The matrix above is the RealScholarQuery story at a glance: red is the Google baselines, amber is the GPT-4o-driven methods, green is the trained 7B agent — and green dominates every column. (A note on the PaSa-GPT-4o cells: the paper reports overall recall and crawler recall for the GPT-4o-driven variant rather than recall@20/@50, so those cells show the overall figures with a marker; the clean @k comparison is the green vs. red/amber rows.) Here are the full tables.

### RealScholarQuery (real researcher queries, fully annotated)

| Method | Precision | Recall | Recall@20 | Recall@50 | Recall@100 |
|---|---|---|---|---|---|
| Google | — | — | 0.1834 | 0.2342 | 0.2535 |
| Google Scholar | — | — | 0.1514 | 0.2155 | 0.2809 |
| Google + GPT-4o | — | — | 0.2020 | 0.2573 | 0.2946 |
| ChatGPT (search) | 0.2280 | 0.2007 | — | — | — |
| GPT-o1 | 0.0580 | 0.0134 | — | — | — |
| PaSa-GPT-4o | 0.4721 | 0.3075 | — | — | — |
| **PaSa-7b** | **0.5146** | **0.6111** | **0.5798** | **0.6563** | **0.6929** |
| PaSa-7b-ensemble | 0.4938 | 0.6488 | 0.5986 | 0.6877 | 0.7281 |

### AutoScholarQuery test (1,000 held-out synthetic queries)

| Method | Crawler Recall | Precision | Recall | Recall@20 | Recall@50 | Recall@100 |
|---|---|---|---|---|---|---|
| Google | — | — | — | 0.1568 | 0.1891 | 0.2015 |
| Google Scholar | — | — | — | 0.0609 | 0.0970 | 0.1130 |
| Google + GPT-4o | — | — | — | 0.1921 | 0.2450 | 0.2683 |
| ChatGPT (search) | — | 0.0507 | 0.3046 | — | — | — |
| GPT-o1 | — | 0.0413 | 0.1925 | — | — | — |
| PaSa-GPT-4o | 0.7565 | 0.1457 | 0.3873 | — | — | — |
| **PaSa-7b** | **0.7931** | 0.1448 | 0.4834 | 0.5301 | 0.6334 | 0.6947 |
| PaSa-7b-ensemble | 0.8265 | 0.1410 | 0.4985 | 0.5326 | 0.6386 | 0.7099 |

The headline improvements, stated precisely:

- **PaSa-7b vs. Google + GPT-4o** (the best Google-based baseline) on AutoScholarQuery: **+37.78% recall@20, +39.90% recall@50, +39.83% recall@100.** These are absolute percentage-point gaps, and they are huge.
- **PaSa-7b vs. PaSa-GPT-4o** (same architecture, GPT-4o instead of the trained 7B as the agent): **+30.36% recall and +4.25% precision.** The trained 7B beats GPT-4o *in the same harness*. That is the single most important result in the paper — it says the win is not "we wrapped a frontier model in a good agent loop," it is "we *trained* a small model to do the agentic task better than a frontier model can do it zero-shot."

Let me dwell on that PaSa-7b-vs-PaSa-GPT-4o comparison because it is the one that should change how you think. If PaSa-GPT-4o had won, the story would be "agent scaffolding is what matters, the model is interchangeable." Instead, the *trained* 7B wins decisively. The agentic scaffold is necessary but not sufficient — the policy that decides when to expand and when to stop is a *learned skill*, and a 7B model trained specifically for it outperforms GPT-4o improvising. This is the strongest evidence I have seen that "agent skills are trainable and worth training," not just promptable.

A note on precision on AutoScholarQuery: it looks low (~0.14) across all the PaSa variants, including PaSa-GPT-4o. That is an artifact of the proxy ground truth — AutoScholarQuery only counts the author's ~2.5 cited papers as correct, so any *additional* genuinely-relevant paper the agent finds is scored as a false positive. On RealScholarQuery, where annotators judge actual relevance, PaSa-7b's precision jumps to **0.5146**. The proxy metric *understates* precision by construction; the human-annotated benchmark is the one to trust.

### Ablations: what each component is worth

This is where the paper earns its keep, because the ablations isolate exactly which design choices drive the result.

| Configuration | AutoScholarQuery Crawler Recall | AutoScholarQuery Recall | RealScholarQuery Crawler Recall | RealScholarQuery Recall |
|---|---|---|---|---|
| w/o `[Expand]` (no citation following) | 0.3355 | 0.2536 | 0.3359 | 0.2890 |
| w/o RL training (imitation only) | 0.6556 | 0.4210 | 0.4847 | 0.4115 |
| w/o Selector as reward model | 0.7041 | 0.4458 | 0.5994 | 0.5148 |
| **PaSa-7b (full)** | **0.7931** | **0.4834** | **0.7071** | **0.6111** |

Read top to bottom, this table is the whole thesis in numbers:

1. **Remove citation expansion** and the system collapses — crawler recall drops to 0.3355 on AutoScholarQuery (a 22.98% absolute decrease) and the RealScholarQuery hit is even worse at 32.21%. This is the single biggest ablation. *Following citations is not a nice-to-have; it is the mechanism.* Without `[Expand]`, PaSa is just a fancy single-shot searcher and lands in roughly the same league as Google.

2. **Remove RL training** (use the imitation-only Crawler) and recall drops 6.24% on AutoScholarQuery but **19.96% on RealScholarQuery.** That gap between the two benchmarks is telling: RL matters *much more on real queries than on the synthetic ones it was indirectly trained against.* The imitation policy learned to mimic GPT-4o on AutoScholarQuery-shaped trajectories, so it is okay there; but RL teaches a more *general* expand/stop policy that transfers to the messier real queries. RL is buying generalization, not just in-distribution fit.

3. **Remove the Selector-as-reward-model** (train only against ground-truth labels) and recall drops a few points on both. This confirms the coupling from section 2: the Selector's dense relevance signal is a better teacher than the sparse ground-truth-only signal, because it can reward the Crawler for finding relevant papers that happen not to be in the (incomplete) ground-truth set.

If you only remember one row from this entire post, make it the first one. The `[Expand]` action — the ability to read a paper and follow its citations — is what separates PaSa from everything else. Everything else (RL, the Selector reward, session-level PPO) is about making that expansion *well-calibrated*. But the expansion itself is the irreducible core.

### Why the frontier reasoning models lose

The most counterintuitive numbers in the tables are the frontier-model rows. GPT-o1 — a dedicated reasoning model — scores **0.0134 recall and 0.0580 precision** on RealScholarQuery. ChatGPT's search mode does better at 0.2007 recall / 0.2280 precision, but still loses to PaSa-7b by a mile. How does a 7B model crush a frontier reasoning model at *search*?

Because reasoning and searching are different skills. A reasoning model is extraordinary at *thinking about what it already knows* — it can reason its way to which papers it has memorized that fit the query. But its knowledge is frozen at training time, it has no live access to arXiv's full index, and it cannot follow a citation it has never seen. When GPT-o1 is asked "find all papers on non-stationary RL with UCB methods," it answers from memory, which means it surfaces the famous papers and hallucinates or omits the rest. Its recall is bounded by what it memorized, and for fine-grained queries that bound is brutally low.

PaSa, by contrast, is not answering from memory at all. It is *operating a search engine and a citation graph* in real time. Its recall is bounded by the graph, not by its parameters. This is the deepest lesson of the paper and it generalizes far beyond papers: **for tasks where the answer lives in an external, traversable structure, a small model that knows how to traverse beats a large model that has to recall.** The frontier model's parameters are wasted effort here; the relevant intelligence is procedural (how to search and expand), not declarative (what papers exist). The whole [tool-use](/blog/machine-learning/ai-agent/advance-tool-use) literature is, in a sense, the study of when procedural-via-tools beats declarative-via-parameters, and PaSa is a sharp data point: it beats GPT-o1 by ~45 points of recall precisely because the task rewards traversal over recall.

### The ensemble row

One more detail worth a sentence: PaSa-7b-ensemble (multiple Crawler rollouts, pooled and de-duplicated) consistently edges out single-run PaSa-7b — recall@100 of 0.7281 vs 0.6929 on RealScholarQuery. The gain is real but modest, and it costs you N× the crawl budget. The interesting implication is that a *single* Crawler run is already capturing most of the available recall; the policy is not wildly stochastic. If it were, the ensemble would help much more. A tight ensemble gap is a sign of a well-trained, relatively deterministic policy — which is what you want.

## 7. Naive single-shot vs. agentic multi-hop, quantified

![Before-after comparison: single-shot Google+GPT-4o uses one keyword query with no citation hops and reaches recall@20 of 0.1921, while agentic PaSa-7b does search-read-expand in a multi-hop citation loop and reaches recall@20 of 0.5798](/imgs/blogs/pasa-llm-paper-search-agent-3.webp)

The before/after above puts the two paradigms side by side with the actual numbers. On the left, the single-shot path: one keyword query, no citation hops, recall@20 = 0.1921 (Google + GPT-4o on AutoScholarQuery). On the right, the agentic path: search + read + expand in a multi-hop citation loop, recall@20 = 0.5798 (PaSa-7b on RealScholarQuery — and 0.5301 on AutoScholarQuery). The arithmetic is a 3× improvement in recall@20 from the same underlying search engine, purely by changing the *control flow* from one-shot to iterative-with-expansion.

This is the part that should reframe how you think about retrieval generally, not just for papers. We have spent years optimizing the *retriever* — better embeddings, better rerankers, better query rewriting — all of which improve single-shot quality. PaSa says: for high-recall tasks, the bigger lever is the *control flow*. A mediocre retriever (Google site:arxiv) wrapped in a good learned search *policy* beats a great retriever used once. If your RAG system is stuck on a recall ceiling, the question to ask is not "can I get a better embedding model?" but "can I afford to retrieve iteratively, follow the graph, and learn when to stop?" The companion posts on [vector databases](/blog/machine-learning/ai-agent/vector-database) and [building effective agents](/blog/machine-learning/ai-agent/building-effective-agents-hands-on-guide) come at this from the retriever and orchestration sides respectively; PaSa is the proof that the orchestration side has more headroom than most teams assume.

## 8. Case studies: what goes right and wrong in practice

Below are ten concrete scenarios — some drawn directly from the paper's reported behavior, some extrapolated from the mechanics and the ablations to illustrate how the system behaves at the edges. Each is the kind of thing you would hit if you actually deployed this.

### 1. The cross-community needle

A query asks for non-stationary RL with UCB methods. The seed search surfaces two RL papers. Neither is the full answer set — but one of them cites a *bandit theory* paper in its background section. The bandit community and the RL community use different terminology for overlapping ideas, so no single keyword query would bridge them. The Crawler's `[Expand]` on the background section crosses that vocabulary gap in one hop and surfaces the bandit-side answers. This is the canonical PaSa win: the answer set spans two sub-communities, and citation-following is the only bridge. Single-shot search scores ~0.19 here; PaSa scores ~0.58. The lesson: recall failures are usually *vocabulary-mismatch* failures, and citations route around vocabulary.

### 2. The expansion explosion (and why depth is capped)

Imagine you forgot the depth cap. The Crawler seeds 10 papers, expands each (≈30 citations apiece → 300), expands those (→ 9,000), expands those (→ 270,000). The queue explodes, the Selector has to score a quarter-million papers, and the run takes hours and costs a fortune. PaSa caps depth at 3, which keeps the candidate pool in the "hundreds to thousands" range the paper reports. The lesson: in any citation-following agent, an explicit depth or budget cap is not optional — exponential fan-out will eat you alive. The action cost (0.1 per expand) provides a soft brake; the depth cap provides the hard one.

### 3. The duplicate-reward bug that would silently wreck training

Suppose during RL you forget the "not already queued" clause in the indicator $\mathbb{I}$. Now a relevant paper reached along five citation paths earns $5\alpha$ instead of $\alpha$. The policy quickly learns to re-discover the same high-value papers over and over — re-expanding branches it has already mined — because that is where the reward is. Recall against *unique* papers craters even as the reward number climbs. This is a textbook reward-hacking failure, and the dedup clause is the one-line fix. The lesson: when your reward pays for "discovery," you must define discovery as *first* discovery, or the policy will farm duplicates.

### 4. The cowardly Selector

Recall the untrained Qwen Selector: precision 1.00, recall 0.38. Deploy *that* as your reward model and the Crawler gets rewarded only for finding the most obviously-relevant papers — the ones any system finds. It never gets reward for the hard, cross-community needles, so it never learns to chase them. The whole system regresses toward single-shot behavior. This is why the Selector's *recall* (0.78 after training) matters as much as its precision: the reward model's recall is the ceiling on the search policy's ambition. The lesson: a precision-obsessed judge makes a timid agent.

### 5. The proxy-metric trap

A team reproduces PaSa, sees precision ~0.14 on AutoScholarQuery, and panics — "our agent is 86% noise!" Then they look at RealScholarQuery and precision is 0.51. The 0.14 was never measuring noise; it was measuring "papers the agent found that the original author didn't happen to cite," most of which are genuinely relevant. The lesson: when your ground truth is a proxy (author citations), a "low precision" number can mean your system is *too good* — finding relevant papers the proxy doesn't credit. Always validate the proxy against a small, fully-annotated set before trusting its absolute numbers.

### 6. The value-function that wasn't learning

A reproduction sets `vf_coef` to the PPO default of 0.5 instead of PaSa's 10.0. The critic underfits, so the cross-session value bonus $\gamma_1 \hat{V}_\phi$ is garbage — it gives roughly equal (and noisy) credit to every expansion regardless of payoff. The agent never learns that *some* branches are worth following deep, so it expands uniformly and shallowly. Recall stalls near the imitation-only level. The lesson: in a system where the value function carries the long-range credit assignment, the critic is not a side-show — it is load-bearing, and PaSa's 20× boost to `vf_coef` is a deliberate statement of that.

### 7. The latency reality check

A product team wants to drop PaSa into an autocomplete-style "related papers" widget with a 300ms budget. It cannot work: PaSa reads hundreds to thousands of papers per query with two 7B models, taking seconds to minutes. The fix is to reframe the product — PaSa is a "run my literature review while I get coffee" tool, not a typeahead. Used in the right latency regime (a dedicated "comprehensive search" button at pasa-agent.ai), it is transformative; used in the wrong one, it is a non-starter. The lesson: match the agent to its latency budget, and never apologize for a minutes-long budget when the alternative is an afternoon of human time.

### 8. The stale-citation horizon

PaSa is trained on 2023-2024 conference citations and the RealScholarQuery candidate universe is frozen at 2024-10-01. Deploy it in 2026 and it will happily crawl arXiv's live index — but its learned sense of "how relevance clusters in citation space" was calibrated on a 2024 graph. For fast-moving fields (say, a 2026 architecture that didn't exist at training time), the seed search still works (Google is live) but the expansion priors are slightly stale — the model hasn't learned the citation topology around the new ideas. The lesson: agentic search degrades gracefully as the world drifts (the search tool stays current), but periodic re-training on fresh citations keeps the expansion policy sharp.

### 9. The over-broad query

A user types a query that is not fine-grained at all — "papers about deep learning." The answer set is effectively the entire field, so *every* expansion surfaces "relevant" papers and the Crawler never finds a reason to stop. The queue grows toward the budget cap, the Selector marks thousands of papers True, and the "ranked answer set" is an undifferentiated firehose. PaSa is built for *fine-grained* queries with small answer sets (~2.5 papers in training); it has no notion of "this query is too broad to answer comprehensively." The fix in a real product is a query-classification front door that detects over-broad queries and either narrows them interactively or routes them to a different surface (a survey-recommendation flow, say). The lesson: an agent tuned for a specific query distribution will behave pathologically off-distribution, and the cheapest defense is a classifier that keeps off-distribution inputs from reaching it.

### 10. The retracted or superseded paper

The Crawler follows a citation to a paper that has since been retracted, or to a v1 preprint that was substantially revised in v3. The Selector judges relevance from the title and abstract, which may not reflect the retraction or the revision, so it can confidently surface a paper that a careful human would flag. PaSa has no notion of paper *status* — it treats the arXiv graph as ground truth about existence and relevance, not about correctness or currency. For literature review this is usually tolerable (you would catch a retraction when you read the paper), but for any use where citing a retracted result is costly (clinical, legal), you need a status-checking post-filter outside the agent. The lesson: relevance is not the same as validity, and an agent optimized for the former says nothing about the latter — bolt validity checks on downstream, do not expect the search policy to learn them.

## 9. How to reason about deploying this

**Reach for a PaSa-style agentic searcher when:**

- **Recall is the dominant metric** and the answer is a *set*, not a single document — literature reviews, prior-art searches, due-diligence sweeps, comprehensive competitive analysis.
- **Your corpus has a link structure** — citations, hyperlinks, "see also" edges, code import graphs — that the agent can traverse. The `[Expand]` action needs edges to follow.
- **You can afford a minutes-long latency budget** because you are replacing expensive human effort, not a sub-second lookup.
- **You can build a cheap, calibrated relevance judge** (the Selector) — ideally one single-token decision you can read a probability off, so it doubles as a training-time reward model.
- **You have, or can synthesize, (query, answer-set) supervision** — and the AutoScholarQuery trick (mine answers from existing curated lists like citation paragraphs, playlists, or "awesome-X" repos) is how you get it for free.

**Skip it when:**

- **Latency is tight.** A 300ms budget rules out reading thousands of documents with 7B models. Use single-shot retrieval + a good reranker.
- **The answer is a single document.** Top-1 retrieval does not benefit from graph traversal; you are paying for expansion you do not need.
- **Your corpus has no usable link structure.** No edges, no `[Expand]`, no agentic advantage — you are back to single-shot.
- **You cannot define a reliable relevance signal.** Without a decent Selector, you cannot train the Crawler *or* filter its output, and the whole architecture loses its spine.
- **You need exhaustive, provable completeness.** PaSa maximizes recall but does not *guarantee* it found everything; for legal/regulatory exhaustiveness you still need deterministic, auditable search.

## Critique: where I am skeptical, and what would change my mind

I want to be fair to a genuinely strong paper while being honest about its limits.

**The proxy ground truth is a real confound for the AutoScholarQuery numbers.** Training and evaluating against author-citation sets bakes in the author's biases — what they knew about, what they chose to cite, what was published before their deadline. The agent is partly learning to *predict citations*, which is correlated with but not identical to *finding all relevant work*. RealScholarQuery mitigates this with human annotation, and the wins hold up there, which is reassuring — but RealScholarQuery is only **50 queries**. That is a small sample for a claim as strong as "beats every baseline by ~38 points." I would feel much better with a few hundred fully-annotated real queries across more fields.

**The evaluation is arXiv- and AI-centric.** Search is restricted to `site:arxiv.org`, the training conferences are all ML/CV/NLP, and the real queries come from PaSa's own demo users (a self-selected, ML-heavy crowd). It is genuinely unknown whether the approach transfers to fields with different citation cultures and venues — biology and medicine (PubMed, very different citation density), the humanities (books, not arXiv preprints), or law (case citations). The *mechanism* should transfer (citations are citations), but the *trained priors* almost certainly will not without domain-specific data.

**Cost and reproducibility are underspecified for production.** Two 7B models, thousands of invocations per query, a custom `trl`/`transformers` fork, `vf_coef 10.0` — this is a non-trivial system to stand up and an expensive one to run. The paper reports quality but is light on the dollars-and-seconds-per-query that a deploying team needs. The `vf_coef 10.0` in particular suggests the PPO training is delicate; I would want to see the variance across seeds before trusting that the RL gains reproduce cleanly.

**It is unclear how much of the win is RL vs. just having an `[Expand]` action at all.** The ablations show `[Expand]` is the biggest factor (a 23-32% recall swing) and RL is second (6-20%). A skeptic could argue that a *well-prompted* GPT-4o crawler with citation expansion already captures most of the value, and RL is a refinement. The counter — PaSa-7b beating PaSa-GPT-4o by 30% recall in the same harness — is the strongest rebuttal, but it is a single comparison.

**What would change my mind** — in either direction:

- If a fully-annotated, multi-domain benchmark of *several hundred* real queries (including non-arXiv fields like biomedicine and law) showed PaSa's lead shrinking to single digits or reversing on out-of-domain queries, I would downgrade the "agentic search is a paradigm shift" claim to "agentic search is a strong technique for arXiv-style ML literature." Conversely, if the lead held across domains, I would upgrade it to a general retrieval principle.
- If an ablation showed that a *prompted* (un-trained) GPT-4o Crawler with the same `[Search]`/`[Expand]`/`[Stop]` tools and a good Selector got within a few points of PaSa-7b on RealScholarQuery, I would conclude the trained policy is a marginal refinement and the real lever is the agent scaffold + a good judge. The reported 30% gap argues against this, but I would want to see it isolated on the *real* benchmark, not just the synthetic one.
- If reproductions across multiple seeds showed the PPO gains were within noise of the imitation baseline on RealScholarQuery, I would conclude the RL stage is fragile and not worth its complexity for most teams — imitation + a good Selector would be the pragmatic recommendation.

None of these is a knock on what the paper demonstrates. They are the experiments I would run before betting a product on it. As it stands, PaSa is the clearest existence proof I know that *comprehensive search is a trainable agentic skill*, that a small model can be trained to beat a frontier model at that skill, and that the data to train it has been sitting in plain sight in every related-work section ever written.

## References

- **Paper:** He, Yichen et al. *"PaSa: An LLM Agent for Comprehensive Academic Paper Search."* ACL 2025 (ByteDance Research + Peking University). [arXiv:2501.10120](https://arxiv.org/abs/2501.10120) · [HTML](https://arxiv.org/html/2501.10120v1)
- **Code:** [github.com/bytedance/pasa](https://github.com/bytedance/pasa) (Apache-2.0). Training uses forks: [hyc2026/trl](https://github.com/hyc2026/trl) (SFT + PPO scripts) and [hyc2026/transformers](https://github.com/hyc2026/transformers).
- **Models:** [bytedance-research/pasa-7b-crawler](https://huggingface.co/bytedance-research/pasa-7b-crawler) · [bytedance-research/pasa-7b-selector](https://huggingface.co/bytedance-research/pasa-7b-selector) (both fine-tuned from Qwen2.5-7B).
- **Demo:** [pasa-agent.ai](https://pasa-agent.ai)

### Related reading on this blog

- [ByteDance Research model atlas](/blog/machine-learning/bytedance-research-model-atlas) — the hub for this series.
- [Building effective agents: a hands-on guide](/blog/machine-learning/ai-agent/building-effective-agents-hands-on-guide) — the orchestrator/worker patterns PaSa instantiates.
- [Advanced tool use](/blog/machine-learning/ai-agent/advance-tool-use) — how agents wield tools like `[Search]` and `[Expand]`.
- [Evaluating agent trajectories beyond the final answer](/blog/machine-learning/ai-agent/evaluating-agent-trajectories-beyond-final-answer) — the trajectory/credit-assignment lens that PaSa's session-level PPO depends on.
- [Evaluating agents](/blog/machine-learning/ai-agent/eval-agents) — how to measure recall/precision tradeoffs in agentic systems.
- [Vector databases](/blog/machine-learning/ai-agent/vector-database) — the single-shot retrieval baseline PaSa's agentic loop out-recalls.
