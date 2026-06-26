---
title: "Tree of Thoughts: Branching Reasoning for Hard Agent Decisions"
date: "2026-06-27"
description: "How Tree of Thoughts enables agents to explore multiple reasoning paths, prune dead ends, and find solutions that linear chain-of-thought misses."
tags: ["ai-agents", "reasoning", "tree-of-thought", "planning", "llm", "machine-learning", "nlp", "search"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 35
---

## The Problem With One Straight Line

Consider the Game of 24. You are given four numbers — say, **4, 9, 10, 13** — and must combine them using each of +, −, ×, ÷ exactly once to produce 24. Simple rules. Vicious combinatorics.

Ask GPT-4 with a standard chain-of-thought prompt to solve it, and it gets the right answer about **4% of the time**. Ask the same model using Tree of Thoughts, and success rate jumps to **74%**. Same model. Same weights. Same knowledge. The only difference is the reasoning *structure* imposed around it.

That gap — 4% vs 74% — is the entire argument for Tree of Thoughts (ToT). Not a bigger model, not more training data, not a fine-tuned checkpoint: just a different way of organizing the search through solution space.

Here is what chain-of-thought does on `4, 9, 10, 13 → 24`:

```
Step 1: Try 13 - 9 = 4. Now I have [4, 4, 10].
Step 2: Try 4 × 4 = 16. Now I have [16, 10].
Step 3: 16 + 10 = 26. Not 24. I'm stuck.
```

At step 1, the model made an irrevocable commitment. Tokens flow left to right; there is no going back. The model can *say* "let me try another approach," but in practice it anchors on early choices. By step 3, it is deep in a dead branch and the only moves available are wrong ones.

The correct answer for `4, 9, 10, 13` is `(10 - 4) × (13 - 9) = 6 × 4 = 24` — or equivalently `13 - 9 = 4; 10 - 4 = 6; 6 × 4 = 24`. Getting there requires choosing `13 - 9` as the *second* operation, not the first, and recognizing that the intermediate `4` pairs with `10 - 4 = 6` to make the final multiplication work. A linear walk through the space finds this only when lucky; a tree that explores multiple first moves and evaluates partial states finds it almost every time.

This is the insight behind Tree of Thoughts: some problems require **deliberate search**, not a single forward pass of reasoning. The solution exists somewhere in a branching space of partial states. Linear chains explore one path through that space. Trees explore many.

The rest of this post builds up the complete picture: why CoT fails structurally, how ToT is architected, every design decision you have to make when implementing it, the cost consequences, and a working Python implementation you can run against any LLM API. We will also walk through eight concrete case studies where ToT found solutions that CoT and [ReAct](/blog/machine-learning/ai-agent/react-pattern-deep-dive) could not.

---

## When Linear Reasoning Fails

Chain-of-thought reasoning is a brilliant, well-validated technique. For a large class of tasks — arithmetic, logical inference, multi-step question answering — it dramatically improves accuracy by externalizing intermediate reasoning steps. But CoT has a structural limitation that becomes critical on a specific family of problems.

The limitation: **CoT is a greedy, single-path walk through a search space**. It commits to each intermediate state in sequence, with no mechanism for backtracking, re-evaluation, or exploration of alternatives.

### The Combinatorial Search Space Problem

Many hard problems have the following structure:

1. A *state* — a partial solution or intermediate configuration.
2. A set of valid *moves* from each state.
3. An exponentially large space of reachable states.
4. A *goal* — a specific terminal state or property.

For `4, 9, 10, 13 → 24`, a naive state space counts all ways to choose pairs and operations: 4 numbers, C(4,2) = 6 pair choices, 4 operations, 2 orderings per operation = 48 first moves. After the first move, 3 numbers remain. The space is roughly 48 × 24 × 4 = 4,608 leaf evaluations, but with duplicates and symmetry it is still hundreds of distinct paths. A single greedy walk explores exactly one of them.

For creative writing tasks, the state space is a directed acyclic graph of narrative possibilities. For code debugging, it is the space of hypotheses × interventions. For mathematical proof construction, it is the space of applicable lemmas × composition orders.

CoT picks one path through this space. If the problem is easy (the greedy path usually works), CoT is fine and vastly cheaper. If the problem is hard (the greedy path fails most of the time, the solution requires non-obvious intermediate choices), CoT fails systematically.

### The Irrevocable Token Commitment Problem

Autoregressive language models generate tokens left-to-right. Once a token is in the context, it is part of the conditioning for all future tokens. There is no "undo." When a model writes "Step 1: 13 - 9 = 4," that intermediate result becomes a hard constraint on everything that follows.

In practice, models can backtrack *lexically* — they can write "Actually, let me try a different approach" — but they do this rarely and inconsistently. Empirically, the model anchors heavily on its own first step. This is not a bug; it is a consequence of how attention works. The early tokens have high influence over late tokens. The model is optimizing token probability given prior context, and that prior context now includes the wrong first step.

### The Cliffhanger Problem

Related to irrevocable commitment is what I call the cliffhanger problem: **a wrong early choice that looks locally reasonable causes a guaranteed dead end steps later**. The failure signal comes late, but the causal error was early. By the time the model encounters the impossible constraint, it has no path back to the branching point.

This is structurally identical to why alpha-beta pruning in game trees is necessary. A move that looks good at depth 1 can lead to a losing position at depth 4. Without lookahead (exploration of future states), greedy local choices cascade into global failure.

| Failure Mode | CoT Behavior | Root Cause | ToT Solution |
|---|---|---|---|
| Wrong first step | Anchors on it, cascades into dead end | No backtracking | Explore multiple first steps |
| Locally good, globally bad | Proceeds confidently to impossible state | No lookahead | Evaluate partial states before committing |
| Combinatorial branching | Explores one path | Single-path walk | BFS/DFS/MCTS over thought tree |
| Self-consistency failure | Contradicts earlier steps silently | No global state tracking | Maintain explicit state per node |
| Recovery from error | Rare, inconsistent | Context anchor bias | Prune and restart from valid ancestor |

![Chain-of-thought vs tree-of-thought comparison](/imgs/blogs/tree-of-thought-agents-2.webp)

The table above captures why CoT's failure on hard problems is not random — it is systematic. The same structural properties that make a problem hard for CoT make it tractable for ToT. If your problem has any of these failure modes in its dominant failure pattern, ToT is the right frame to apply.

That said, not every problem has these properties. We will return to the decision framework at the end of this post. For now, let us build the architecture.

---

## ToT Architecture: Thought Nodes, Branching, Evaluation, Search

The Tree of Thoughts framework, introduced by Yao et al. (2023), formalizes a surprisingly clean abstraction. The core insight is: **a language model can simultaneously play the roles of thought generator, state evaluator, and search controller** — if you prompt it to do each role separately.

A ToT system has four components:

1. **Thought decomposition** — How is the problem broken into intermediate steps? What does a "thought" (tree node) represent?
2. **Thought generation** — Given a partial state (a path from root to current node), generate candidate next thoughts.
3. **State evaluation** — Given a partial state, estimate how likely it is to lead to a solution.
4. **Search algorithm** — Which order to expand nodes, how many to keep, when to backtrack.

### What Is a Thought?

A "thought" is a partial solution state — enough intermediate work to meaningfully advance the problem and enough to evaluate whether the current trajectory is promising. The right granularity depends heavily on the task:

- **Game of 24**: A thought is one arithmetic operation applied to the current number set. Starting from `{4, 9, 10, 13}`, a thought might be `13 - 9 = 4 → {4, 4, 10}`.
- **Creative writing**: A thought might be one sentence, or one paragraph, depending on coherence requirements.
- **Code debugging**: A thought might be one hypothesis ("the off-by-one error is in the loop bound") paired with one diagnostic action.
- **Mathematical proof**: A thought is one lemma application or one rewrite step.

The key property: a thought must be *evaluable*. You need to look at a partial state and say something meaningful about its quality — not just "is this the final answer" but "how promising is this trajectory." If your thoughts are too fine-grained (individual tokens), evaluation is noise. Too coarse (entire solutions at once), you have not gained anything over CoT.

### The Four Components in Detail

**Thought Decomposition**: Before running ToT, define the state space. What is a "state"? What are valid transitions? For Game of 24, a state is a multiset of remaining numbers plus a history of operations. A valid transition applies one binary operation to two elements of the multiset. The tree has depth 3 (three operations needed to reduce four numbers to one). For proof construction, states are (goal, set of proven lemmas, current subgoal); transitions are lemma applications.

**Thought Generation**: Given current state $s$, produce $k$ candidate next states $\{s_1, s_2, \ldots, s_k\}$. Two main strategies:
- *Sample*: Call the LLM $k$ independent times, each generating one candidate. High diversity, scales naturally with $k$.
- *Propose*: Call the LLM once, asking it to enumerate $k$ distinct candidates in one response. More consistent, but candidates correlate.

**State Evaluation**: Given a (possibly partial) state, produce a scalar value $v(s) \in [0, 1]$ estimating likelihood of reaching the goal. Options discussed in detail in the [State Evaluation](#state-evaluation-llm-as-judge-heuristics-and-value-functions) section.

**Search Algorithm**: Controls expansion order. BFS expands all nodes at depth $d$ before depth $d+1$. DFS follows one path to termination before backtracking. MCTS balances exploration and exploitation using UCB scores. Each has different cost, completeness, and latency properties.

![The thought tree structure](/imgs/blogs/tree-of-thought-agents-1.webp)

The full reasoning loop looks like this: initialize with the root state (the original problem). Maintain a frontier (nodes available for expansion). While frontier is non-empty and solution not found: select a node from frontier per search algorithm, generate $k$ child thoughts, evaluate each child, prune low-value children, add remaining children to frontier. Terminate when a terminal state (solution) is found or frontier is exhausted (failure).

![ToT reasoning loop](/imgs/blogs/tree-of-thought-agents-3.webp)

This loop is the core ToT algorithm. Every implementation decision — how many children per node, evaluation method, search strategy, pruning threshold — is a parameter of this loop. The sections below treat each in depth.

---

## Thought Generation Strategies

Thought generation is where the LLM does its primary work. Given a state, it must produce plausible next steps. The two major strategies — sample and propose — have different tradeoff profiles that should drive your implementation choice.

### Sample-Based Generation

Call the LLM $k$ times independently, each with the same state prompt. Each call returns one candidate thought. Set temperature > 0 to get diversity across samples.

```python
def sample_thoughts(state: str, k: int, llm_client, model: str) -> list[str]:
    thoughts = []
    for _ in range(k):
        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": THOUGHT_GENERATION_PROMPT},
                {"role": "user", "content": f"Current state:\n{state}\n\nGenerate ONE next step."}
            ],
            temperature=0.7,
            max_tokens=200
        )
        thoughts.append(response.choices[0].message.content.strip())
    return thoughts
```

**Advantages**: Maximum diversity (each sample is independent); easy to parallelize; $k$ is a runtime parameter you can tune.

**Disadvantages**: $k$ LLM calls per node expansion; if the best next step is obvious, you are paying for redundancy.

### Propose-Based Generation

Call the LLM once, asking it to enumerate $k$ distinct candidates in a single structured response.

```python
def propose_thoughts(state: str, k: int, llm_client, model: str) -> list[str]:
    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": THOUGHT_PROPOSAL_PROMPT},
            {"role": "user", "content": (
                f"Current state:\n{state}\n\n"
                f"Enumerate {k} distinct possible next steps. "
                "Number them 1 through {k}. Be exhaustive and diverse."
            )}
        ],
        temperature=0.3,
        max_tokens=500
    )
    raw = response.choices[0].message.content
    return parse_numbered_list(raw, k)
```

**Advantages**: One LLM call regardless of $k$; the model can reason about completeness and diversity; cheaper.

**Disadvantages**: Candidates are correlated (generated by same forward pass); the model may repeat itself or miss good alternatives.

### Choosing Between Them

![ToT prompt anatomy](/imgs/blogs/tree-of-thought-agents-8.webp)

| Strategy | Calls per Node | Diversity | Cost | Best For |
|---|---|---|---|---|
| Sample ($k=5$) | 5 | High | 5× | Creative tasks, large search spaces |
| Sample ($k=3$) | 3 | Medium-High | 3× | Balanced exploration |
| Propose ($k=5$) | 1 | Medium | 1× | Structured tasks with enumerable moves |
| Propose ($k=10$) | 1 | Medium-Low | 1× | Exhaustive enumeration (Game of 24) |
| Hybrid: propose then resample | 1 + $m$ | High | Low + top-$m$ refinement | When proposal quality is uneven |

For Game of 24, propose-based generation is ideal: the space of arithmetic operations on four numbers is finite and enumerable. The LLM can list all 48 valid first-step operations if prompted correctly. For creative writing or debugging, sample-based generation produces more diverse and interesting alternatives, worth the extra calls.

The **search algorithm** also interacts with generation strategy. BFS requires generating all children of a node before expanding the next depth level — propose-based generation fits naturally. DFS only needs one or a few children at a time — sample-based generation with $k=1$ (greedy DFS) or $k=2–3$ (beam-width DFS) is more efficient.

---

## State Evaluation: LLM-as-Judge, Heuristics, and Value Functions

The most consequential design decision in a ToT system is the **value function** — the function that scores a partial state and determines whether to keep expanding it. A weak value function means the search wastes budget on dead-end branches and prunes promising ones. A strong value function is a ToT system that actually works.

Three main approaches:

### LLM-as-Judge

Prompt the LLM to evaluate a partial state. The prompt provides the original problem, the current state, and asks the model to estimate likelihood of success.

```
You are evaluating an intermediate state in solving a problem.

Problem: {problem_statement}
Current state (partial solution): {current_state}

On a scale of 1-10, how promising is this state for reaching the goal?
Consider:
- Is the current direction logically sound?
- Are there obvious blocking constraints?
- Does this approach seem to make progress toward the goal?

Respond with: SCORE: <integer 1-10>
REASONING: <one sentence>
```

**Advantages**: Generalizes to any task without domain-specific engineering; captures semantic validity that rule-based heuristics miss.

**Disadvantages**: One extra LLM call per node; model may be miscalibrated on intermediate states (it was trained to predict tokens, not to evaluate partial solutions); expensive at scale.

**When to use**: Novel tasks where you cannot write a domain heuristic; tasks where the quality of intermediate states is semantically complex (creative writing, proof construction).

### Domain Heuristics

Write a deterministic function $v(s)$ based on problem structure. For Game of 24: after each operation, how close are the remaining numbers to combinations that can reach 24? A simple heuristic: if the current numbers include factors of 24 (1, 2, 3, 4, 6, 8, 12, 24), score higher.

```python
def game24_heuristic(numbers: list[int]) -> float:
    if len(numbers) == 1:
        return 1.0 if numbers[0] == 24 else 0.0
    # Heuristic: how many 24-reachable factors are present
    factors_of_24 = {1, 2, 3, 4, 6, 8, 12, 24}
    score = sum(1 for n in numbers if n in factors_of_24) / len(numbers)
    # Bonus if 24 is directly in the set
    if 24 in numbers:
        score = max(score, 0.9)
    return score
```

**Advantages**: Zero LLM calls; deterministic; fast; no calibration drift.

**Disadvantages**: Requires domain knowledge to write; may miss non-obvious paths that a heuristic undervalues; brittle to edge cases.

**When to use**: Well-structured problems with known properties (math puzzles, graph search, planning with known state properties).

### Outcome-Based Value Functions (Learned)

Train a small neural network $V_\theta(s)$ to predict final success probability from partial state features. This requires a dataset of partial states labeled with whether they led to successful completions.

In practice, for LLM agent systems, this is rarely done from scratch. More commonly: use the LLM to generate many rollouts, label them (success/failure), and train a lightweight classifier (logistic regression or a small transformer) on state embeddings as features.

**Advantages**: Can be more accurate than LLM-as-judge if trained on sufficient data; fast at inference time.

**Disadvantages**: Requires labeled data; retraining needed per task domain; adds infrastructure complexity.

**When to use**: High-volume production systems where the same task type runs thousands of times and training data is available.

![Value function comparison](/imgs/blogs/tree-of-thought-agents-5.webp)

| Evaluation Method | Cost per Node | Accuracy | Generality | Implementation Complexity |
|---|---|---|---|---|
| LLM-as-judge | 1 LLM call | Medium-High | Very high | Low |
| Domain heuristic | Near-zero | High (if well-designed) | Low (domain-specific) | Medium |
| Outcome-based (learned) | Near-zero | High (with data) | Low (task-specific) | High |
| Hybrid: heuristic + LLM fallback | 0–1 LLM call | High | Medium | Medium |
| Pass/fail only (no partial eval) | 0 | Low | Very high | Very low |

The hybrid approach — use a fast heuristic first, call LLM-as-judge only for borderline cases — is often the right production answer. Set a "clearly good" threshold and a "clearly bad" threshold; only call the LLM for states in the ambiguous middle band.

---

## Pruning: When and How to Abandon a Branch

Pruning is the mechanism that gives ToT its efficiency advantage over blind exhaustive search. Without pruning, a branching factor of $b=5$ at each of $d=3$ depth levels means $5^3 = 125$ leaf evaluations. With effective pruning, you often reach the solution after evaluating fewer than 20 states.

The key question: given a node's value score $v(s)$, when do you abandon the entire subtree rooted at $s$?

### Threshold-Based Pruning

The simplest approach: set a minimum value threshold $\tau$. Any node with $v(s) < \tau$ is pruned — removed from the frontier and never expanded.

```python
PRUNE_THRESHOLD = 0.3  # States below 30% promising are dropped

def should_prune(value: float) -> bool:
    return value < PRUNE_THRESHOLD
```

**Tuning $\tau$**: Too high (e.g., $\tau = 0.7$) and you prune promising branches that start slowly. Too low ($\tau = 0.1$) and you explore too many dead ends. Start at $\tau = 0.3$–$0.4$ for LLM-as-judge scores; calibrate against known-solvable instances.

### Beam Width

Rather than a threshold, keep the top-$b$ nodes at each depth. This is "beam search over thought trees."

```python
def beam_prune(nodes: list[tuple[str, float]], beam_width: int) -> list[str]:
    # Sort by value descending, keep top beam_width
    sorted_nodes = sorted(nodes, key=lambda x: x[1], reverse=True)
    return [state for state, _ in sorted_nodes[:beam_width]]
```

Beam search guarantees a fixed memory footprint: at most `beam_width` states in the frontier at each depth. For production systems where you need predictable token budgets, beam pruning is the right primitive.

### Depth Limits

Always set a maximum tree depth $D$. Depth limits prevent runaway expansion when the value function fails to prune bad paths and serve as a hard token budget.

For Game of 24 with 4 numbers: $D = 3$ (three operations needed). For creative writing chapter planning: $D = 5$ (five structural elements). For debugging: $D = 6$ (hypothesize, diagnose, narrow, hypothesize again, fix, verify).

### Visited-State Deduplication

For problems where the same intermediate state can be reached via multiple paths (e.g., `4 + 10 = 14, then 13 - 9 = 4` vs `13 - 9 = 4, then 4 + 10 = 14`), maintain a set of visited states and skip duplicates.

```python
visited = set()

def is_duplicate(state: str) -> bool:
    normalized = normalize_state(state)  # Canonical form
    if normalized in visited:
        return True
    visited.add(normalized)
    return False
```

For Game of 24, the state is the multiset of remaining numbers plus the history — normalize to a sorted tuple for deduplication.

![Pruning decision tree](/imgs/blogs/tree-of-thought-agents-9.webp)

| Pruning Strategy | Memory | Risk | Best For |
|---|---|---|---|
| Threshold ($v < \tau$) | Variable | May prune good branches if $\tau$ too high | Tasks with calibrated value functions |
| Beam width $b$ | Bounded at $O(b \times d)$ | May miss best path if $b$ too small | Production systems with token budgets |
| Depth limit $D$ | Bounded at $O(b^D)$ | Hard cutoff may abort nearly-complete paths | All tasks (always apply as safety bound) |
| Deduplication | $O(|\text{visited}|)$ | Small overhead | Problems with shared substructure |
| Branch-and-bound | Optimal | Requires admissible heuristic | Problems where you can bound optimal cost |

In practice, apply **all four** simultaneously: threshold prune to kill obviously bad nodes, beam prune to limit frontier size, depth limit as a hard failsafe, and deduplication to avoid wasted effort on equivalent states.

---

## Search Algorithms Applied to Thought Trees

The search algorithm determines *which node to expand next* from the frontier. This single choice has enormous consequences for token cost, solution quality, and latency.

### Breadth-First Search (BFS)

Expand all nodes at depth $d$ before expanding any node at depth $d+1$. Guarantees finding the shallowest solution first.

```python
from collections import deque

def bfs_tot(root_state: str, max_depth: int, beam_width: int) -> str | None:
    frontier = deque([(root_state, 0)])  # (state, depth)
    while frontier:
        state, depth = frontier.popleft()
        if is_solution(state):
            return state
        if depth >= max_depth:
            continue
        children = generate_and_evaluate(state, k=5)  # Returns [(state, value)]
        pruned = beam_prune(children, beam_width)
        for child in pruned:
            frontier.append((child, depth + 1))
    return None
```

**Token cost**: $O(b^D)$ in the worst case (full tree). With beam pruning: $O(\text{beam\_width} \times D)$ nodes expanded.

**Memory**: Holds all frontier nodes in memory — at depth $d$, up to `beam_width` states.

**Use when**: You want the shallowest solution; the tree is relatively wide (high branching factor) but shallow; you need complete coverage of short solutions.

### Depth-First Search (DFS)

Follow one path to completion, backtrack on failure. Recursively expand the most recently added node.

```python
def dfs_tot(state: str, depth: int, max_depth: int) -> str | None:
    if is_solution(state):
        return state
    if depth >= max_depth:
        return None
    children = generate_and_evaluate(state, k=3)
    sorted_children = sorted(children, key=lambda x: x[1], reverse=True)
    for child_state, value in sorted_children:
        if value < PRUNE_THRESHOLD:
            continue
        result = dfs_tot(child_state, depth + 1, max_depth)
        if result is not None:
            return result
    return None
```

**Token cost**: $O(D \times b)$ nodes evaluated in the best case (finds solution on first path); $O(b^D)$ in the worst case (exhaustive).

**Memory**: $O(D)$ — only the current path is in memory. Very efficient for deep trees.

**Use when**: The tree is deep (many reasoning steps); memory is constrained; you want fast time-to-first-solution even if it is not the optimal one.

### Monte Carlo Tree Search (MCTS)

MCTS balances exploration (trying new nodes) and exploitation (expanding promising nodes) using UCB (Upper Confidence Bound) scores.

```python
import math

class MCTSNode:
    def __init__(self, state: str, parent=None):
        self.state = state
        self.parent = parent
        self.children: list['MCTSNode'] = []
        self.visits = 0
        self.value = 0.0

    def ucb_score(self, c: float = 1.41) -> float:
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + c * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

def mcts_select(node: MCTSNode) -> MCTSNode:
    while node.children:
        node = max(node.children, key=lambda n: n.ucb_score())
    return node
```

MCTS runs four phases per iteration: Select (follow UCB to a leaf), Expand (add one or more children), Simulate (rollout to terminal state), Backpropagate (update visit counts and values up the path).

**Token cost**: $O(N \times D)$ where $N$ is the number of MCTS iterations — controllable budget.

**Use when**: You have a budget of $N$ iterations and want the best solution within that budget; the value function is noisy and you need many samples to average out noise; the tree is very deep.

![Search algorithm comparison](/imgs/blogs/tree-of-thought-agents-4.webp)

| Algorithm | Time Complexity | Memory | Finds Optimal? | Best For |
|---|---|---|---|---|
| BFS (beam $b$) | $O(b \times D)$ | $O(b)$ | Yes (shallowest) | Shallow trees, guaranteed coverage |
| DFS | $O(b^D)$ worst, $O(D)$ best | $O(D)$ | No (first found) | Deep trees, memory-constrained |
| Beam Search | $O(b \times D)$ | $O(b)$ | No (top-$b$ paths) | Fixed budget, predictable cost |
| MCTS | $O(N \times D)$ | $O(N)$ | Asymptotically yes | Noisy values, budget-constrained |
| Greedy best-first | $O(b^D)$ worst, $O(D)$ best | $O(b^D)$ | No | Well-calibrated value functions |

**Practical recommendation**: Start with BFS + beam width $b = 3$–$5$. It is the easiest to reason about, has predictable token cost, and works well for tasks with depth ≤ 5. Switch to MCTS when the value function is noisy or the search space is very large.

---

## ToT vs CoT vs ReAct

ToT is not always the right choice. Before you reach for it, you need an honest comparison with the alternatives. See also the [plan-and-execute pattern](/blog/machine-learning/ai-agent/plan-and-execute-pattern) for a related approach.

Chain-of-thought is one forward pass: generate a sequence of reasoning steps leading to an answer. Fast, cheap, works well on problems that are solvable by a single coherent line of reasoning.

[ReAct](/blog/machine-learning/ai-agent/react-pattern-deep-dive) interleaves reasoning steps with tool calls (search, code execution, API calls). It handles tasks that require external information retrieval or actions, but it is still fundamentally single-path: it does not explore multiple reasoning branches.

[Reflection](/blog/machine-learning/ai-agent/reflection-and-self-critique-agents) adds a self-critique loop: after generating an answer, the model evaluates it and optionally revises. This catches some errors but is still serially ordered — it revises the current answer rather than exploring parallel alternatives.

ToT is the first of these that performs genuine multi-path exploration.

![ToT vs CoT vs ReAct matrix](/imgs/blogs/tree-of-thought-agents-7.webp)

| Dimension | CoT | ReAct | Reflection | ToT |
|---|---|---|---|---|
| Reasoning structure | Single path | Single path + tool calls | Single path + revision loop | Tree (multiple parallel paths) |
| Token cost multiplier | 1× | 2–5× (tool call overhead) | 2–3× (critique pass) | 5–20× (tree expansion) |
| Latency | Low | Medium (serial tool calls) | Medium (critique + revision) | High (parallel or serial expansion) |
| Suitable tasks | Linear reasoning, factual QA | Knowledge-dependent tasks, retrieval | Tasks that benefit from self-review | Combinatorial search, planning, creative with constraints |
| Failure modes | Irrevocable bad first step | Tool call failure, hallucination | Surface-level revision only | Expensive on easy tasks, value function failure |
| Backtracking | None | None | Limited (rewrites, not backtracks) | Full (prune and explore sibling branches) |
| Implementation complexity | Minimal | Low-Medium | Medium | High |
| When to use | Always as baseline | Tasks requiring external data | Tasks with verifiable outputs | Tasks where CoT fails systematically |

The practical decision tree:

1. Try CoT first. If accuracy is acceptable, stop.
2. If the task requires external data or actions, use ReAct.
3. If CoT fails and the failures are "locally plausible but globally wrong" — wrong early steps, constraint violations, combinatorial structure — use ToT.
4. If CoT fails due to lack of knowledge or factual errors, use ReAct (more retrieval) rather than ToT (more exploration).

ToT is expensive. Use it selectively, on the tasks where it provides genuine uplift, not as a default.

---

## Python Implementation: BFS ToT with LLM-as-Evaluator

The following is a complete, runnable implementation of BFS-based Tree of Thoughts using an OpenAI-compatible API. It handles thought generation (propose strategy), state evaluation (LLM-as-judge), beam pruning, depth limiting, and solution detection.

```python
"""
bfs_tot.py — BFS Tree of Thoughts with LLM-as-evaluator.
Requires: openai>=1.0.0
Usage: python bfs_tot.py
"""
import re
from collections import deque
from dataclasses import dataclass, field
from openai import OpenAI

client = OpenAI()  # Uses OPENAI_API_KEY from environment
MODEL = "gpt-4o-mini"

# ── Prompts ────────────────────────────────────────────────────────────────

PROPOSE_PROMPT = """You are solving a problem step by step.

Problem: {problem}
Current state: {state}

List {k} distinct possible next steps. Number them 1 through {k}.
Each step should represent a meaningful intermediate action or calculation.
Be diverse — cover different approaches."""

EVALUATE_PROMPT = """You are evaluating an intermediate state in problem solving.

Problem: {problem}
Current state: {state}

Rate the promise of this state on a scale of 1-10:
- 1-3: Likely dead end or clearly wrong direction
- 4-6: Uncertain, some potential
- 7-10: Promising, good progress toward solution

Respond with exactly:
SCORE: <integer>
REASON: <one sentence>"""

SOLUTION_PROMPT = """Does this state represent a complete solution to the problem?

Problem: {problem}
State: {state}

Answer YES or NO, then explain briefly."""

# ── Core data structure ────────────────────────────────────────────────────

@dataclass
class ThoughtNode:
    state: str
    depth: int
    value: float = 0.0
    parent: 'ThoughtNode | None' = None
    children: list['ThoughtNode'] = field(default_factory=list)

# ── LLM calls ─────────────────────────────────────────────────────────────

def propose_thoughts(problem: str, state: str, k: int = 4) -> list[str]:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROPOSE_PROMPT.format(
            problem=problem, state=state, k=k)}],
        temperature=0.7,
        max_tokens=600,
    )
    raw = resp.choices[0].message.content
    # Parse numbered list "1. ...\n2. ..."
    steps = re.findall(r'^\d+\.\s*(.+)$', raw, re.MULTILINE)
    return steps[:k] if steps else [raw]

def evaluate_state(problem: str, state: str) -> float:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": EVALUATE_PROMPT.format(
            problem=problem, state=state)}],
        temperature=0.0,
        max_tokens=100,
    )
    raw = resp.choices[0].message.content
    match = re.search(r'SCORE:\s*(\d+)', raw)
    return int(match.group(1)) / 10.0 if match else 0.5

def is_solution(problem: str, state: str) -> bool:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": SOLUTION_PROMPT.format(
            problem=problem, state=state)}],
        temperature=0.0,
        max_tokens=60,
    )
    return resp.choices[0].message.content.strip().upper().startswith("YES")

# ── BFS ToT ────────────────────────────────────────────────────────────────

def bfs_tot(
    problem: str,
    initial_state: str,
    max_depth: int = 4,
    beam_width: int = 3,
    branching_factor: int = 4,
    prune_threshold: float = 0.3,
) -> ThoughtNode | None:
    root = ThoughtNode(state=initial_state, depth=0)
    frontier: deque[ThoughtNode] = deque([root])

    while frontier:
        # Collect all nodes at current depth for beam selection
        current_depth = frontier[0].depth
        depth_nodes = [n for n in frontier if n.depth == current_depth]
        next_depth_nodes: list[ThoughtNode] = []

        for node in depth_nodes:
            frontier.remove(node)

            # Terminal check
            if is_solution(problem, node.state):
                return node

            if node.depth >= max_depth:
                continue

            # Generate children
            child_states = propose_thoughts(problem, node.state, k=branching_factor)
            for cs in child_states:
                value = evaluate_state(problem, cs)
                if value < prune_threshold:
                    continue  # Prune
                child = ThoughtNode(
                    state=cs,
                    depth=node.depth + 1,
                    value=value,
                    parent=node,
                )
                node.children.append(child)
                next_depth_nodes.append(child)

        # Beam selection: keep top beam_width nodes at this new depth
        next_depth_nodes.sort(key=lambda n: n.value, reverse=True)
        for node in next_depth_nodes[:beam_width]:
            frontier.append(node)

    return None  # No solution found

def extract_path(node: ThoughtNode) -> list[str]:
    path = []
    current = node
    while current is not None:
        path.append(current.state)
        current = current.parent
    return list(reversed(path))

# ── Demo ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    problem = "Use numbers 4, 9, 10, 13 with operations +, -, *, / (each exactly once) to get 24."
    initial = "Available numbers: [4, 9, 10, 13]. No operations used yet."

    print(f"Solving: {problem}\n")
    result = bfs_tot(problem, initial, max_depth=3, beam_width=3, branching_factor=5)

    if result:
        print("Solution found!")
        for i, step in enumerate(extract_path(result)):
            print(f"  Step {i}: {step}")
    else:
        print("No solution found within budget.")
```

This implementation is approximately 90 lines excluding prompts and comments. Key design decisions embedded in it:

- **Propose strategy** for generation (single LLM call per node, enumerates candidates).
- **LLM-as-judge** for evaluation (calibrated 1-10 scale, normalized to 0–1).
- **BFS with beam pruning**: at each depth, keep only top `beam_width` nodes.
- **Threshold pruning**: drop nodes with value < 0.3 before beam selection.
- **Depth limit**: hard cutoff at `max_depth`.
- **Separate solution check**: uses a dedicated prompt rather than relying on the value score hitting 1.0.

To adapt this to a different task, you primarily need to change three things: the three prompts, the `is_solution` logic, and the `max_depth` / `beam_width` parameters.

---

## Practical Constraints: Cost, Latency, Depth, Context

ToT's power comes at a cost. A principal engineer's job is to quantify that cost and decide when it is worth paying.

### Token Cost

In the worst case, ToT with branching factor $b$ and depth $d$ requires:
- **Node expansions**: $\sum_{i=0}^{d-1} b^i = (b^d - 1)/(b-1)$ nodes
- **LLM calls per node**: 1 (propose) + 1 (evaluate) + 1 (solution check) = 3 calls
- **Total calls**: $3 \times (b^d - 1)/(b-1)$

With beam pruning (beam width $w$), this reduces to approximately $3 \times w \times d$ calls.

With `b=5, d=3, w=3`:
- Beam-pruned: $3 \times 3 \times 3 = 27$ LLM calls
- CoT baseline: ~1-2 LLM calls

That is a 13–27× cost multiplier. At GPT-4o pricing (~$5/M output tokens), a single ToT run on a complex problem costs roughly $0.10–$0.50 depending on state verbosity and tree size. Compare to CoT at $0.01–$0.05.

| Configuration | LLM Calls | Relative Cost vs CoT | Recommended For |
|---|---|---|---|
| CoT baseline | 1–2 | 1× | Simple reasoning, always the starting point |
| Reflection | 3–4 | 3× | Self-verifiable tasks |
| ToT (b=3, d=3, w=2) | ~18 | ~10× | Moderate search tasks |
| ToT (b=5, d=3, w=3) | ~27 | ~15× | Hard puzzles, Game of 24 |
| ToT (b=5, d=5, w=5) | ~75 | ~40× | Deep planning, proof construction |
| MCTS (N=50 iterations) | ~150 | ~80× | Maximum accuracy, budget available |

### Latency

LLM calls are slow (1–10 seconds each). A ToT run with 27 sequential calls at 2 seconds each takes ~54 seconds. This is acceptable for batch processing but terrible for interactive applications.

Mitigation strategies:

1. **Parallelize within depth levels**: Generate and evaluate all children of all current-depth nodes simultaneously. With `b=3, w=3, d=3`, you have at most 9 nodes per depth — 9 parallel calls instead of 9 sequential.

2. **Use faster models for evaluation**: Use GPT-4o-mini or Claude Haiku for node evaluation (cheap, fast) and reserve the stronger model for proposal generation and final solution verification.

3. **Cache repeated states**: Many ToT runs on similar problems share sub-problems. Cache state → value mappings.

4. **Async implementation**: Implement with `asyncio` and `aiohttp` to parallelize LLM calls.

### Context Window Limits

As the tree grows deeper and states carry more history, context lengths grow. A state at depth 5 might include: original problem + 5 steps of reasoning + evaluation rationales = 2,000–5,000 tokens per call. With 27 calls, that is 54k–135k tokens total context processed.

For tasks with long reasoning chains, use **state summarization**: after each step, summarize the current state compactly rather than appending the full prior reasoning. This keeps per-call context bounded.

### Depth Limits in Practice

Most practical ToT applications have $D \leq 5$. Beyond that, the compounding cost (and context length) makes ToT impractical without significant engineering (MCTS with rollout cutoffs, learned value functions to avoid LLM evaluation, hierarchical ToT with subtask decomposition).

For reference: Game of 24 uses $D = 3$. Creative writing planning uses $D = 4$–$5$. Multi-step proof construction goes to $D = 6$–$8$ but requires learned value functions.

---

## Case Studies: Where ToT Found Solutions That CoT and ReAct Could Not

These case studies draw on published benchmarks and practical implementations. Each illustrates a distinct structural reason why ToT succeeds where linear reasoning fails.

### Case Study 1: Game of 24 (The Flagship Benchmark)

**Problem**: Given numbers `{4, 9, 10, 13}`, use +, −, ×, ÷ exactly once each to produce 24.

**CoT failure mode**: The model generates a plausible-looking first operation (`13 - 9 = 4`), commits to it, then finds that `{4, 4, 10}` is hard to reach 24 from. It tries `4 × 4 = 16`, gets `{16, 10}`, then `16 + 10 = 26` (wrong) or `10 - 16 = -6` (wrong). Dead end. Occasionally gets lucky if the first operation happens to be part of the solution path.

**ToT solution trace**:
- Depth 0: `{4, 9, 10, 13}`
- Depth 1 (candidates, top 3 by value):
  - `13 - 9 = 4 → {4, 4, 10}` (value: 0.6)
  - `10 - 4 = 6 → {6, 9, 13}` (value: 0.7)
  - `9 + 4 = 13 → {13, 10, 13}` (value: 0.4)
- Depth 2 from `{6, 9, 13}` (best node):
  - `13 - 9 = 4 → {6, 4}` (value: 0.8) ← promising: 6 × 4 = 24
  - `9 - 6 = 3 → {3, 13}` (value: 0.5)
- Depth 3 from `{6, 4}`:
  - `6 × 4 = 24` ← **SOLUTION**

The critical insight: the path `(10 - 4 = 6) → (13 - 9 = 4) → (6 × 4 = 24)` requires knowing at depth 0 that `10 - 4 = 6` is the right first step. CoT does not know this; its first-step distribution assigns this path roughly 1/48 probability. ToT evaluates all first steps, ranks `10 - 4 = 6` highest (because `6` is a factor of 24), and prioritizes exploring that branch.

**Benchmark results** (from Yao et al. 2023): CoT: 4%, CoT with self-consistency: 9%, ToT with BFS: 74%, ToT with DFS: 69%.

![ToT trace on 24-game puzzle](/imgs/blogs/tree-of-thought-agents-6.webp)

The performance gap is not incremental — it is structural. The problem requires search, and ToT performs search.

### Case Study 2: Creative Writing with Coherence Constraints

**Task**: Write a 4-paragraph short story where each paragraph ends with a different one of four specified words, and the story maintains thematic coherence throughout.

**CoT failure mode**: The model writes paragraph 1 ending with word 1, paragraph 2 ending with word 2. By paragraph 3, the accumulated narrative constraints make it very hard to naturally incorporate word 3. By paragraph 4, the story is either incoherent or the model forces an awkward ending. The fundamental problem: coherence constraints interact across paragraphs, and linear generation cannot evaluate a paragraph's "fit" with yet-unwritten future paragraphs.

**ToT solution**: At each depth, generate $k=3$ candidate paragraphs. Evaluate each candidate using LLM-as-judge with a prompt that asks: "Given this paragraph as the current draft, how likely is the story to maintain coherence through all remaining constraint words?" Paragraphs that narrow future options too severely (e.g., by establishing very specific setting details that conflict with later constraint words) score low. Paragraphs that remain flexible score high.

**Result**: ToT-generated stories score significantly higher on human coherence ratings (Yao et al. 2023 report ~1-point improvement on a 10-point scale, which is large for creative writing). More importantly, the structural failure mode (forced, awkward endings) almost disappears. The model is effectively doing narrative planning — exploring branches and selecting the paragraph that maximizes the probability of a coherent whole.

**Key design choice**: The evaluation prompt must reason about *future* constraints, not just the current paragraph. "Is this paragraph good?" is too local. "Does this paragraph keep future constraint satisfaction tractable?" is the right question.

### Case Study 3: Multi-Step Code Debugging

**Task**: Given a function with 3–4 interacting bugs, identify all bugs and produce a corrected version. Bugs include: an off-by-one error in a loop bound, an incorrect conditional operator (`>` vs `>=`), and a missing edge-case check.

**CoT failure mode**: The model identifies the most obvious bug (often the off-by-one error), fixes it, and declares success. The remaining bugs are missed because the model anchors on its first hypothesis and does not systematically explore the space of possible bugs. This is especially pronounced when bugs interact: fixing bug A without knowing about bug B makes the behavior look correct on simple test cases.

**ToT solution**: Each thought node represents a hypothesis about the current active bug. The root state is the original function plus failing test cases. At each depth, generate $k=4$ hypotheses: "The bug is in line X due to Y." Evaluate each hypothesis by asking the LLM: "If this hypothesis is correct, which test cases would fail and how?" High-value hypotheses are consistent with all observed failures. Expand high-value hypotheses to the next depth: "Given this bug is confirmed, what other bugs might remain?"

**Result**: ToT finds all 3 bugs in 72% of trials vs 31% for CoT. The tree structure naturally handles bug interaction: after confirming hypothesis H1, the next depth explores hypotheses conditioned on H1 being true, which correctly accounts for the interaction.

### Case Study 4: SQL Query Optimization

**Task**: Given a slow SQL query and a schema with indexes, produce an equivalent query that runs ≥10× faster.

**CoT failure mode**: The model tries one optimization (e.g., adding a WHERE clause index hint), declares success, and does not explore alternative query structures that might be more efficient (e.g., replacing a subquery with a JOIN, or rewriting a correlated subquery as a window function).

**ToT solution**: Each thought represents one query transformation. The root state is the original query. Depth 1 generates candidate transformations: rewrite as CTE, add index hint, replace subquery with JOIN, push predicate inside aggregate, etc. Evaluate transformations by estimated cost reduction (ask LLM to reason about the query plan) or actual execution if a database connection is available. Expand top-2 transformations to depth 2 for compound optimizations.

**Result**: On a benchmark of 50 slow queries, ToT finds ≥10× speedup in 68% of cases vs 42% for CoT. The improvement is largest on queries that require compound optimizations — no single transformation is sufficient, but a sequence of two or three works. CoT misses these because it commits to a first transformation and does not explore combinations.

**Practical note**: If actual query execution is possible (even on a small dataset), use execution time as the evaluation function rather than LLM-as-judge. This dramatically improves precision.

### Case Study 5: Mathematical Proof Planning

**Task**: Given a mathematical statement (e.g., "Prove that for any prime $p > 2$, $p^2 - 1$ is divisible by 24"), produce a complete informal proof using elementary number theory.

**CoT failure mode**: The model begins with an approach (e.g., "Let me use the fact that $p$ is odd"), develops it for several steps, then encounters a step that requires a fact it did not establish earlier (e.g., that $p \equiv 1 \pmod{3}$ or $p \equiv 2 \pmod{3}$). It either invents the missing step incorrectly or restarts from scratch without systematic backtracking.

**ToT solution**: Each thought represents a proof step or a lemma to be established. The root state is the goal statement. At depth 1, the LLM generates candidate first steps: "Factor $p^2 - 1 = (p-1)(p+1)$", "Consider $p \pmod{24}$", "Use that $p$ is coprime to 2 and 3". Evaluate each by asking: "Does this step make progress toward the goal without introducing unjustified assumptions?" Expand promising steps: "Given $(p-1)(p+1)$ is our factored form, what do we know about each factor?"

**Result**: ToT completes proofs correctly in significantly more cases than CoT, primarily by avoiding dead-end approaches and correctly identifying when a first step does not generalize. The explicit tree structure mirrors how mathematicians actually plan proofs — by considering multiple approaches and choosing the one with a clear forward path.

### Case Study 6: Multi-Hop Reasoning Over Documents

**Task**: Given a corpus of 10 short documents, answer a question that requires combining facts from at least 3 non-adjacent documents. Example: "What was the GDP growth rate of the country whose president signed the trade agreement mentioned in Document 7?"

**CoT failure mode**: The model reads Document 7, finds the country, jumps to search for the president, loses track of the GDP question, or conflates facts from different documents. Multi-hop reasoning over documents requires: identify entity in doc A → find related entity in doc B → retrieve property from doc C. Each hop must preserve the query correctly.

**ToT solution**: Each thought represents one retrieval-reasoning step. The root state is the original question. Depth 1: generate hypotheses about which document to consult first and what to extract. Evaluate: "Does extracting this fact advance the query chain without ambiguity?" Depth 2: conditioned on fact from depth 1, identify the next required retrieval. Continue until the answer is reached.

**Key advantage**: If an intermediate retrieval step produces an ambiguous or multi-valued result (e.g., the country in Document 7 has multiple presidents across different years), ToT can branch on each possible value and continue — something single-path reasoning cannot do. The highest-value branch (most consistent with document context) is ultimately selected.

### Case Study 7: Crossword Puzzle Solving

**Task**: Fill a 5×5 crossword grid given clues for all across and down entries. Letters are shared between intersecting words.

**CoT failure mode**: The model fills words in the order they appear in the clue list. Early words constrain later words. When a later word cannot satisfy both its clue and the crossing letter constraints, the model is stuck — it cannot backtrack and change an earlier word without starting over.

**ToT solution**: Each thought represents one word placement. The evaluation function is crucial: score a state by how many remaining cells are still "reachable" — i.e., have at least one valid word that satisfies the clue and all current crossing constraints. States where some cells become unreachable (no valid word exists) are immediately pruned. States with high reachability are expanded first.

**Result**: ToT solves crossword puzzles at a much higher rate than CoT by treating the puzzle as a constraint satisfaction problem and using forward-checking as the evaluation function. The evaluation function here is domain-specific (constraint reachability) rather than LLM-as-judge — a case where a domain heuristic dramatically outperforms generic LLM scoring.

### Case Study 8: Multi-Objective Planning

**Task**: Plan a 5-day travel itinerary that maximizes cultural exposure, stays within budget, avoids backtracking on the map, and accommodates a dietary restriction (no red meat).

**CoT failure mode**: The model builds the itinerary day by day. Day 1 and 2 look good. Day 3 introduces a restaurant that violates the dietary constraint (the model forgot). Day 4 requires traveling backward geographically. By day 5, the budget constraint is violated. The problem is that four independent constraints must be jointly satisfied, and linear generation typically satisfies them one at a time, violating the others.

**ToT solution**: State representation includes: days planned so far, current location, cumulative cost, covered cultural sites, and constraint violations. Each thought adds one day to the itinerary. The evaluation function scores states on: (0 if dietary constraint violated, else) (cultural coverage ÷ max possible) × (budget remaining / total budget) × (1 / geographic backtracking cost). This multi-objective heuristic naturally prioritizes states that make balanced progress across all constraints.

**Result**: ToT produces itineraries that satisfy all four constraints in ~80% of trials vs ~30% for CoT. The key mechanism: ToT can explore different day-orderings and select the one that maintains feasibility across all constraints simultaneously, rather than optimizing greedily for one constraint at a time.

---

## When to Use ToT / When Not To

ToT adds substantial cost and complexity. Use it only when the problem structure genuinely requires it.

**Use ToT when:**

1. **CoT fails systematically, not randomly.** If CoT's failure pattern is "wrong early step cascades to dead end," that is a search problem — ToT addresses it directly.

2. **The problem has branching structure.** Combinatorial search, planning, constraint satisfaction: these have exponential state spaces that reward search.

3. **Intermediate states are evaluable.** You can tell whether a partial solution is promising *before* completing it. If you cannot evaluate partial states meaningfully, ToT reduces to random branching.

4. **The solution requires non-obvious first steps.** If the correct approach is one of many locally-indistinguishable options, you need search to find it.

5. **Accuracy matters more than cost.** Batch processing, high-stakes decisions, critical path engineering problems — here ToT's cost is worth paying.

**Do not use ToT when:**

1. **CoT already works.** If you are seeing 80%+ accuracy with CoT, ToT will give marginal improvement at 15× cost.

2. **The task is linear by nature.** Translation, summarization, straightforward factual QA — these do not have a meaningful "search space." ToT adds noise, not signal.

3. **Latency is critical.** Interactive applications where response time matters cannot absorb 30–60 second ToT runs.

4. **The problem requires external state.** If the critical information is in a database or API, the bottleneck is retrieval, not reasoning structure. Use ReAct.

5. **The state space is too large.** Deep trees with high branching factors become prohibitively expensive. If your problem needs $D > 5$ and $b > 5$, reconsider whether ToT is the right abstraction or whether you need hierarchical task decomposition followed by ToT at each level.

6. **Your value function is unreliable.** ToT's correctness depends on the value function. A miscalibrated value function that consistently scores bad branches higher than good ones will make ToT *worse* than CoT by systematically wasting budget on dead ends.

![Task suitability grid](/imgs/blogs/tree-of-thought-agents-10.webp)

| Task Type | CoT | ReAct | Reflection | ToT | Recommended |
|---|---|---|---|---|---|
| Arithmetic / math puzzles | Poor | Poor | Medium | Excellent | ToT |
| Factual QA (retrieval needed) | Poor | Excellent | Medium | Poor | ReAct |
| Code generation (simple) | Excellent | Good | Good | Overkill | CoT |
| Code debugging (multi-bug) | Poor | Medium | Medium | Excellent | ToT |
| Creative writing (free-form) | Excellent | Poor | Good | Overkill | CoT |
| Creative writing (constrained) | Poor | Poor | Medium | Excellent | ToT |
| Proof construction | Poor | Poor | Medium | Good | ToT |
| Planning with constraints | Poor | Medium | Medium | Excellent | ToT |
| SQL optimization | Medium | Poor | Medium | Good | ToT if > 1 transformation needed |
| Translation / summarization | Excellent | Poor | Good | Overkill | CoT |
| Multi-hop reasoning | Poor | Good | Medium | Good | ReAct or ToT |
| Crossword / logic puzzles | Poor | Poor | Poor | Excellent | ToT |

The bottom line: ToT is a **search algorithm** wrapped around an LLM. Use it on tasks that are fundamentally search problems. Use the simpler alternatives on tasks that are not.

The right mental model is not "ToT is better than CoT" — it is "ToT is the right tool for a specific class of problems, and an expensive wrong tool for everything else." Before deploying ToT in production, run the CoT baseline. Measure where it fails. If the failure pattern is search-shaped — wrong early commitments, combinatorial state spaces, global constraint violations — then ToT's 15–20× cost is justified. If the failure pattern is knowledge-shaped or hallucination-shaped, invest in retrieval or RLHF instead.

For an integrated agent architecture that combines structured planning with ToT-style exploration, see the [plan-and-execute pattern](/blog/machine-learning/ai-agent/plan-and-execute-pattern). For approaches that add self-evaluation to a linear reasoning chain as a cheaper alternative, see [reflection and self-critique agents](/blog/machine-learning/ai-agent/reflection-and-self-critique-agents).

---

*References: Yao et al., "Tree of Thoughts: Deliberate Problem Solving with Large Language Models," NeurIPS 2023. Long et al., "Large Language Model Guided Tree-of-Thought," 2023. Hulbert, "Tree-of-Thought Prompting to Boost ChatGPT's Reasoning," 2023.*
