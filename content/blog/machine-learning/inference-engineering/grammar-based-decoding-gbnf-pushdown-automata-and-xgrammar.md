---
title: "Grammar-based decoding: GBNF, pushdown automata, and the stack your FSM never had"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Prove why a finite-state machine can never validate nested JSON, build a real pushdown automaton over a GBNF grammar in nanoserve, and add the jump-forward path that emits forced tokens without ever calling the model."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "constrained-decoding",
    "structured-output",
    "grammar",
    "xgrammar",
    "decoding",
    "latency",
    "ml-systems",
    "vllm",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 50
---

The FSM you built in [constrained decoding from first principles](/blog/machine-learning/inference-engineering/constrained-decoding-from-first-principles-masking-logits-with-an-fsm) works. It compiles a flat schema into states, it produces a per-step allowed-token mask, and JSON mode stops emitting broken output. Then a product team ships a schema with a `properties` block whose value is itself an object, and inside that object is an array, and inside the array is another object. Your machine emits `{"plan": {"steps": [{"id": 1` and then quietly loses track of which closing bracket comes next, because it has nowhere to keep that information.

This is not a bug in your implementation. It is a theorem. A finite-state machine has a finite number of states and no auxiliary memory, and "how deep am I, and in which order did I open things" is unbounded information. You cannot patch around it, you cannot add a few more states, and the standard workaround — unrolling the schema to a fixed maximum depth — costs states exponentially in that depth. Figure 1 is the whole argument in one picture: the same three inputs, the machine that fails on the third, and the machine that does not.

![Two side by side columns comparing what a finite automaton and a stack machine can accept as nesting depth grows from flat to depth forty one](/imgs/blogs/grammar-based-decoding-gbnf-pushdown-automata-and-xgrammar-1.webp)

So this post climbs one rung. We move from regular languages to **context-free grammars**, and from a finite-state machine to a **pushdown automaton** — an FSM plus a stack. That single addition buys arbitrary nesting for free, and it costs you the ability to precompute a mask per state, because the state space is now infinite. Everything hard about grammar-based decoding follows from that trade.

By the end you will have written `nanoserve/decoding/gbnf.py` and `nanoserve/decoding/pda.py`: a small parser for llama.cpp's GBNF grammar notation, a stack machine that tracks the parse position, an `allowed_tokens(state)` that walks a trie over the vocabulary instead of testing all 128,256 tokens, a mask cache keyed by machine state, and — the part almost nobody implements and everybody should — a **jump-forward** path that detects when the grammar admits exactly one continuation and writes those characters straight into the output without calling the model at all. On a typical object schema that path is worth more than every kernel optimization in Track E combined, and it is architecturally impossible in some popular libraries. We will see exactly why.

If you have not read [what inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is), that post sets up the scoreboard this one is scored against: TTFT, TPOT, tokens per second, and goodput. Grammar decoding touches all four, and in opposite directions — compilation hurts TTFT, jump-forward helps TPOT, and a badly cached mask serializes your whole batch.

## 1. Why the FSM runs out of memory, proved rather than asserted

Let us be precise about what an FSM can and cannot do, because "it can't handle nesting" is the kind of statement that sounds like folklore until you see the proof.

A **deterministic finite automaton** (DFA) is a finite set of states $Q$, a start state, a set of accepting states, and a transition function $\delta: Q \times \Sigma \to Q$ that reads one input symbol and moves. That is the entire machine. There is no counter, no stack, no tape. Whatever the machine knows after reading a prefix is encoded in *which state it is in*, and there are only $|Q|$ of those.

Now take the simplest possible nesting language: strings of balanced square brackets, $L = \{\,[^n\,]^n : n \ge 0\,\}$.

**Claim.** No DFA recognizes $L$.

**Proof.** Suppose one does, with $|Q| = k$ states. Feed it the $k+1$ prefixes $[^0, [^1, \ldots, [^k$. That is $k+1$ prefixes and only $k$ states, so by the pigeonhole principle two of them land in the same state: there exist $i \lt j$ with $\delta^*([^i) = \delta^*([^j) = q$. The machine is now in state $q$ and has no way to tell whether it read $i$ brackets or $j$. Feed it $]^i$. Since $[^i\,]^i \in L$, the machine must accept. But the machine's behaviour from $q$ depends only on $q$, so it also accepts $[^j\,]^i$ — a string with $j \ne i$ that is not in $L$. Contradiction. $\square$

That is the pumping-lemma idea in its most honest form: **a finite machine cannot remember an unbounded count**, and nesting depth is exactly an unbounded count.

### The bounded-depth escape hatch, and why it is worse than it looks

The usual response is: "fine, cap the depth at 20 and unroll." That does work — bounded-depth JSON *is* a regular language, because now the depth is finite information. The question is how many states it costs.

JSON has two bracket types, `{` and `[`, and the closing order is determined by the opening order. So after reading a sequence of $d$ open brackets, the machine must know the *entire sequence*, not just its length — because `{[` requires `]}` and `[{` requires `}]`. There are $2^d$ distinct open-bracket sequences of length $d$, and by the Myhill–Nerode argument each needs its own equivalence class: any two distinct sequences have different legal completions, so no DFA can merge them.

$$|Q| \ \ge\ 2^{d}$$

Depth 3 needs at least 8 states just for the bracket bookkeeping. Depth 20 needs at least 1,048,576. Depth 41 needs about $2.2 \times 10^{12}$, which is the number in figure 1 and which is more states than you have bytes of RAM. The real grammar multiplies this by the per-level rule positions (are we expecting a key, a colon, a value, a comma?), so the true count is several times worse.

This is the honest reason production libraries do not unroll: it is not that unrolling is inelegant, it is that the state count is exponential in a parameter users legitimately want to set to 30.

![A two level taxonomy separating constraints that need only finite states from constraints that require a stack](/imgs/blogs/grammar-based-decoding-gbnf-pushdown-automata-and-xgrammar-2.webp)

Figure 2 sorts the constraints you will actually be asked for. The left branch — enums, ISO-8601 dates, fixed-width identifiers, phone numbers, anything a regular expression describes — is genuinely regular, and the FSM from post 18 is the right tool: cheap, fully precomputable, no surprises. The right branch is everything with recursion in it: nested objects and arrays, matched XML or HTML tags, balanced parentheses in a generated expression, and any JSON Schema whose definition refers to itself. Those are context-free, and they need a stack.

One clarification worth making early, because it saves an argument in review: **most real JSON schemas are not recursive**, and for those the FSM is sufficient in principle. The problem is that "not recursive" is a property you have to *prove* about every schema a user submits, and the depth at which the unrolled FSM blows up is small. Building the stack machine once is cheaper than auditing schemas forever.

## 2. The step up: a context-free grammar and a stack

A **context-free grammar** (CFG) is a set of rules of the form `A ::= α`, where `A` is a nonterminal and `α` is a sequence of terminals and nonterminals. The grammar generates a string if you can start from the root nonterminal and rewrite your way down to that exact string.

`llama.cpp` ships a concrete notation for this called **GBNF** (GGML BNF), and it is the closest thing the ecosystem has to a lingua franca — XGrammar takes GBNF as input, and llama.cpp's own grammar sampler is built on it. Here is a complete grammar for JSON values, small enough to read in one sitting:

```python
# nanoserve/decoding/grammars/json.gbnf  (loaded as a Python string for the demo)
JSON_GBNF = r'''
root    ::= value
value   ::= object | array | string | number | "true" | "false" | "null"
object  ::= "{" ws ( member ( "," ws member )* )? "}"
member  ::= string ws ":" ws value ws
array   ::= "[" ws ( value ws ( "," ws value ws )* )? "]"
string  ::= "\"" char* "\""
char    ::= [^"\\] | "\\" ["\\/bfnrt]
number  ::= "-"? int frac? exp?
int     ::= "0" | [1-9] [0-9]*
frac    ::= "." [0-9]+
exp     ::= [eE] [-+]? [0-9]+
ws      ::= [ \t\n]*
'''
```

Nine lines describe every legal JSON document of every depth. The recursion is the second line and the fifth: `value` can be an `object`, an `object` contains `member`s, and a `member` contains a `value`. That cycle is what a DFA cannot represent and what a stack handles trivially.

The machine that recognizes a CFG is a **pushdown automaton**: a finite control plus a last-in-first-out stack. On each input character it looks at its control state and the top of the stack, consumes the character, and may push or pop. The formal object is a 7-tuple; the working intuition is one sentence, and it is the sentence this whole post rests on:

> The parser's state is not a state name. It is a state name **plus the entire stack standing behind it**.

### Tracing the stack through a real value

Take the input `{"a": [1, {"b": 2}]}` and walk it. The stack holds *what we still owe* — the symbols we have promised to match before the enclosing rule can finish.

![An ordered left to right trace of seven parser events showing three pushes and three pops while a nested value is read](/imgs/blogs/grammar-based-decoding-gbnf-pushdown-automata-and-xgrammar-3.webp)

Figure 3 is that walk. Reading `{` pushes an object frame; the stack now says "I owe a member list and a closing brace." Reading `[` inside the member's value pushes an array frame on top of it. Reading `{` again pushes a third. When the inner `}` arrives, the top frame pops and the array frame is exposed again — and *that is why* `,` and `]` become legal at exactly that moment and `}` does not. Nothing about the characters already emitted tells you this. Only the stack does.

Here is the same trace as motion, because the pushing and popping is the entire mechanism and a still frame shows you one instant of it:

<figure class="blog-anim">
<svg viewBox="0 0 660 250" role="img" aria-label="A pushdown automaton stack grows to depth three and pops back to zero while the set of legal next characters changes at every depth" style="width:100%;height:auto;max-width:820px">
<style>
.g1-h{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.g1-str{font:600 20px ui-monospace,SFMono-Regular,Menlo,monospace;fill:var(--text-primary,#1f2937)}
.g1-set{font:600 17px ui-monospace,SFMono-Regular,Menlo,monospace;fill:var(--text-primary,#1f2937)}
.g1-base{stroke:var(--border,#d1d5db);stroke-width:2}
.g1-fr{fill:var(--accent,#6366f1)}
.g1-frt{font:600 13px ui-sans-serif,system-ui;fill:var(--background,#fff);text-anchor:middle}
.g1-rv{fill:var(--accent,#6366f1);opacity:.20;transform-box:fill-box;transform-origin:left center}
@keyframes g1-grow{0%,12%{transform:scaleX(.06)}15%,27%{transform:scaleX(.36)}30%,42%{transform:scaleX(.53)}45%,57%{transform:scaleX(.88)}60%,72%{transform:scaleX(.94)}75%,100%{transform:scaleX(1)}}
@keyframes g1-f1{0%{opacity:0}3%,72%{opacity:1}75%,100%{opacity:0}}
@keyframes g1-f2{0%,13%{opacity:0}16%,57%{opacity:1}60%,100%{opacity:0}}
@keyframes g1-f3{0%,28%{opacity:0}31%,42%{opacity:1}45%,100%{opacity:0}}
@keyframes g1-l1{0%,12%{opacity:1}14%,29%{opacity:0}31%,42%{opacity:1}44%,100%{opacity:0}}
@keyframes g1-l2{0%,14%{opacity:0}16%,27%{opacity:1}29%,100%{opacity:0}}
@keyframes g1-l3{0%,44%{opacity:0}46%,57%{opacity:1}59%,100%{opacity:0}}
@keyframes g1-l4{0%,59%{opacity:0}61%,72%{opacity:1}74%,100%{opacity:0}}
@keyframes g1-l5{0%,74%{opacity:0}76%,97%{opacity:1}99%,100%{opacity:0}}
.g1-a-rv{animation:g1-grow 14s ease-in-out infinite}
.g1-a-f1{animation:g1-f1 14s ease-in-out infinite}
.g1-a-f2{animation:g1-f2 14s ease-in-out infinite}
.g1-a-f3{animation:g1-f3 14s ease-in-out infinite}
.g1-a-l1{animation:g1-l1 14s ease-in-out infinite}
.g1-a-l2{animation:g1-l2 14s ease-in-out infinite}
.g1-a-l3{animation:g1-l3 14s ease-in-out infinite}
.g1-a-l4{animation:g1-l4 14s ease-in-out infinite}
.g1-a-l5{animation:g1-l5 14s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.g1-a-rv,.g1-a-f1,.g1-a-f2,.g1-a-f3,.g1-a-l1,.g1-a-l2,.g1-a-l3,.g1-a-l4,.g1-a-l5{animation:none}.g1-a-rv{transform:scaleX(.36)}.g1-a-f3,.g1-a-l1,.g1-a-l3,.g1-a-l4,.g1-a-l5{opacity:0}}
</style>
<text class="g1-h" x="24" y="32">emitted so far</text>
<rect class="g1-rv g1-a-rv" x="20" y="48" width="228" height="30" rx="5"/>
<text class="g1-str" x="24" y="70">{"a":[1,{"b":2}]}</text>
<text class="g1-h" x="24" y="132">legal next character, decided by the stack</text>
<text class="g1-set g1-a-l1" x="24" y="164">a key string, or }</text>
<text class="g1-set g1-a-l2" x="24" y="164">a value, or ]</text>
<text class="g1-set g1-a-l3" x="24" y="164">, or ]</text>
<text class="g1-set g1-a-l4" x="24" y="164">, or }</text>
<text class="g1-set g1-a-l5" x="24" y="164">end of value only</text>
<text class="g1-h" x="430" y="32">PDA stack</text>
<line class="g1-base" x1="424" y1="228" x2="640" y2="228"/>
<rect class="g1-fr g1-a-f1" x="430" y="178" width="196" height="44" rx="8"/>
<text class="g1-frt g1-a-f1" x="528" y="205">obj body · depth 1</text>
<rect class="g1-fr g1-a-f2" x="430" y="126" width="196" height="44" rx="8"/>
<text class="g1-frt g1-a-f2" x="528" y="153">array items · depth 2</text>
<rect class="g1-fr g1-a-f3" x="430" y="74" width="196" height="44" rx="8"/>
<text class="g1-frt g1-a-f3" x="528" y="101">obj body · depth 3</text>
</svg>
<figcaption>The stack is the state. As the value nests, frames push; as it closes, they pop — and the legal next character changes with the top frame, not with the characters already emitted.</figcaption>
</figure>

Watch what the legal-set line does at the 45% mark. Nothing was appended to the string except a `}`, and yet the set of legal next characters changed completely, because a frame disappeared. That coupling — output set determined by stack top — is what your mask function has to reproduce every single decode step.

### The size trade, stated as a formula

The DFA needed $\ge 2^d$ states for depth $d$. The PDA needs a *fixed* control — one entry per rule position, call it $R$, which for the JSON grammar above is a couple of dozen — plus a stack of length $d$. So:

$$\text{DFA size} = \Theta(2^{d}) \qquad\text{vs}\qquad \text{PDA size} = \underbrace{\Theta(R)}_{\text{control}} + \underbrace{\Theta(d)}_{\text{stack, at runtime}}$$

You have traded an exponential in machine size for a linear amount of *runtime* memory. That is a spectacular deal, and it is why every serious constrained-decoding library is a pushdown machine underneath.

The bill arrives in the next section. A DFA has finitely many states, so you can precompute one mask per state ahead of time and never think about it again. A PDA's state is (control position, stack), and there are infinitely many of those. **You cannot precompute the mask table.** Everything from here is about making the per-step computation cheap enough to survive production.

## 3. The token-level problem comes back, and it is harder now

Recall the mismatch from post 18. The grammar is defined over *characters*. The model emits *tokens* — multi-character strings from a vocabulary of, for Llama-3.1-8B, 128,256 entries. A token is legal at a given moment only if feeding all of its characters through the machine, one after another, leaves the machine alive.

With a DFA, you precomputed this: for each of the $|Q|$ states, test all $|V|$ tokens once at compile time, store $|Q| \times |V|$ bits, done. With a PDA there is no finite $|Q|$ to enumerate. The mask must be computed for a state you have never seen before, on the request path, between two forward passes.

### The cost of doing it naively

Let $|V|$ be the vocabulary size and $\bar{c}$ the mean characters per token. Testing every token against the machine costs

$$C_{\text{naive}} = |V| \cdot \bar{c} \ \text{character transitions per decode step}$$

For Llama-3.1-8B, $|V| = 128{,}256$; JSON-flavoured text tokenizes at roughly $\bar{c} \approx 3.5$ characters per token — a figure you should check against your own tokenizer with the script at the end of this post rather than take from me. That gives about 449,000 character transitions per token generated. Each transition in a Python implementation is a loop over the live stacks with a set membership test — call it a microsecond, which is an order-of-magnitude estimate and not a measurement. That is on the order of **hundreds of milliseconds per token**, against a decode step of a few milliseconds. Naive is not slow; naive is unusable.

Three techniques get you from there to viable, and every production library uses some mix of them.

**Technique 1 — walk a trie, not a list.** Build a prefix trie over the vocabulary once, at startup. Then compute the mask by depth-first search from the trie root, carrying the machine state down each edge. The moment an edge's character is illegal, the entire subtree beneath it is illegal too and you prune it. The work is no longer $|V| \cdot \bar{c}$; it is the number of trie nodes whose prefix is a legal continuation. In a state where only `,` and `}` are legal, that subtree is a few hundred nodes rather than half a million.

**Technique 2 — cache the mask by machine state.** Two requests at the same point in the same grammar have the same state, and the same state has the same mask. Key an LRU on the machine state and the hit rate on repetitive schemas is very high: a JSON object with ten fields visits the same "expecting a key" and "inside a string" states over and over. The mask itself is a bit per token — for a 128,256-token vocabulary that is $128{,}256 / 8 = 16{,}032$ bytes, just under 16 KiB. A thousand cached states is 16 MB of host RAM, which is nothing.

**Technique 3 — decompose the grammar into finite pieces.** This is the one that carries the field forward, and it is worth quoting the primary source exactly. In [Structured Decoding in vLLM](https://vllm.ai/blog/2025-01-14-struct-decode-intro) (2025-01-14), the vLLM team describes the XGrammar backend as using a pushdown automaton that is "a collection of FSMs, and each FSM represents a context-free grammar," which is what "enables recursive or nested structures" and "allows the PDA to handle multiple state transitions." The insight is that the *unbounded* part of the state is the stack, but each individual grammar rule is a finite object. Compile each rule to its own small FSM, precompute masks for those finite states, and let the stack decide which FSM is currently active. You get most of the precomputation back, and only the transitions that genuinely depend on the stack are computed live.

![A branching dataflow showing a decode step forking into a forced continuation path that skips the model and a masked path that computes or reuses a bitmask](/imgs/blogs/grammar-based-decoding-gbnf-pushdown-automata-and-xgrammar-4.webp)

Figure 4 is the resulting step. Note the shape: the fork happens *before* the mask is computed, because the cheapest mask is the one you never need. The upper branch is jump-forward, and it is the subject of the next section.

### The GPU side is not the problem

It is worth putting the GPU cost in its place, because people optimize the wrong half of this.

Applying a mask is `logits.masked_fill_(~mask, float("-inf"))` on a `[B, V]` tensor. At batch 32 with fp32 logits — and you do want fp32 here, for the numerical reasons the previous two posts in this track lay out — the traffic is $32 \times 128{,}256 \times 4 = 16.4$ MB read plus the same written, so 32.8 MB. An A100 80GB SXM lists 2.0 TB/s of HBM2e bandwidth on [NVIDIA's A100 datasheet](https://www.nvidia.com/en-us/data-center/a100/); at a realistic 1.5 TB/s achieved that is

$$t_{\text{mask}} = \frac{32.8\ \text{MB}}{1.5\ \text{TB/s}} \approx 22\ \mu s$$

against a decode step for an 8B model in bf16 of roughly $16\ \text{GB} / 2.0\ \text{TB/s} = 8$ ms. The mask application is **0.27% of the step**. It is free.

What is not free is producing the mask on the host and getting it to the device. vLLM's post is explicit about this being the pressure point: it names, as a v1 roadmap item, moving guided decoding to the scheduler level so the engine can "calculate the bitmask once and broadcast it to each GPU worker," and it flags that with the Outlines backend "we currently apply the token bitmask to all requests in the batch," which serializes everyone behind the slowest grammar. The lesson generalizes past vLLM: **the expensive part of constrained decoding is host-side bookkeeping and synchronization, not GPU arithmetic.**

## 4. Jump-forward decoding: the win nobody talks about

Here is the observation that makes grammar decoding *faster* than unconstrained decoding rather than slower.

At many positions in a structured output, the grammar admits exactly one legal next character. If your schema requires a property named `name` and it is the only required property, then after `{` the next character must be `"`, then `n`, then `a`, then `m`, then `e`, then `"`, then `:`. Seven characters with a branching factor of one. The model has no decision to make. Sampling from a distribution where 128,255 tokens have logit $-\infty$ and one token does not is a very expensive way to compute a constant.

So do not compute it. Detect the forced run and write it directly into the output.

<figure class="blog-anim">
<svg viewBox="0 0 660 210" role="img" aria-label="Nine grammar-forced characters appear in one go with no forward pass, then five model-chosen characters arrive one at a time" style="width:100%;height:auto;max-width:820px">
<style>
.j-h{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.j-f{fill:var(--accent,#6366f1)}
.j-m{fill:var(--surface,#f3f4f6);stroke:var(--accent,#6366f1);stroke-width:2}
.j-ft{font:600 17px ui-monospace,SFMono-Regular,Menlo,monospace;fill:var(--background,#fff);text-anchor:middle}
.j-mt{font:600 17px ui-monospace,SFMono-Regular,Menlo,monospace;fill:var(--text-primary,#1f2937);text-anchor:middle}
.j-dot{fill:var(--text-secondary,#6b7280)}
.j-bd{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.j-lg{font:600 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
@keyframes j-kf {0%,5%{opacity:.12}9%,93%{opacity:1}97%,100%{opacity:.12}}
@keyframes j-k1 {0%,27%{opacity:.12}31%,93%{opacity:1}97%,100%{opacity:.12}}
@keyframes j-k2 {0%,36%{opacity:.12}40%,93%{opacity:1}97%,100%{opacity:.12}}
@keyframes j-k3 {0%,45%{opacity:.12}49%,93%{opacity:1}97%,100%{opacity:.12}}
@keyframes j-k4 {0%,54%{opacity:.12}58%,93%{opacity:1}97%,100%{opacity:.12}}
@keyframes j-k5 {0%,63%{opacity:.12}67%,93%{opacity:1}97%,100%{opacity:.12}}
@keyframes j-kl {0%,74%{opacity:.12}78%,93%{opacity:1}97%,100%{opacity:.12}}
@keyframes j-b1 {0%,4%{opacity:0}8%,24%{opacity:1}28%,73%{opacity:0}77%,93%{opacity:1}97%,100%{opacity:0}}
@keyframes j-b2 {0%,28%{opacity:0}32%,70%{opacity:1}74%,100%{opacity:0}}
.j-af{animation:j-kf 12s ease-in-out infinite}
.j-a1{animation:j-k1 12s ease-in-out infinite}
.j-a2{animation:j-k2 12s ease-in-out infinite}
.j-a3{animation:j-k3 12s ease-in-out infinite}
.j-a4{animation:j-k4 12s ease-in-out infinite}
.j-a5{animation:j-k5 12s ease-in-out infinite}
.j-al{animation:j-kl 12s ease-in-out infinite}
.j-ab1{animation:j-b1 12s ease-in-out infinite}
.j-ab2{animation:j-b2 12s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.j-af,.j-a1,.j-a2,.j-a3,.j-a4,.j-a5,.j-al,.j-ab1,.j-ab2{animation:none}.j-a1,.j-a2,.j-a3,.j-a4,.j-a5,.j-al,.j-ab2{opacity:.12}.j-ab1{opacity:1}}
</style>
<text class="j-h" x="20" y="26">one required key in the schema, so nine characters have exactly one legal continuation</text>
<rect class="j-f j-af" x="18"  y="58" width="38" height="46" rx="6"/>
<rect class="j-f j-af" x="60"  y="58" width="38" height="46" rx="6"/>
<rect class="j-f j-af" x="102" y="58" width="38" height="46" rx="6"/>
<rect class="j-f j-af" x="144" y="58" width="38" height="46" rx="6"/>
<rect class="j-f j-af" x="186" y="58" width="38" height="46" rx="6"/>
<rect class="j-f j-af" x="228" y="58" width="38" height="46" rx="6"/>
<rect class="j-f j-af" x="270" y="58" width="38" height="46" rx="6"/>
<rect class="j-f j-af" x="312" y="58" width="38" height="46" rx="6"/>
<rect class="j-f j-af" x="354" y="58" width="38" height="46" rx="6"/>
<text class="j-ft j-af" x="37"  y="88">{</text>
<text class="j-ft j-af" x="79"  y="88">"</text>
<text class="j-ft j-af" x="121" y="88">n</text>
<text class="j-ft j-af" x="163" y="88">a</text>
<text class="j-ft j-af" x="205" y="88">m</text>
<text class="j-ft j-af" x="247" y="88">e</text>
<text class="j-ft j-af" x="289" y="88">"</text>
<text class="j-ft j-af" x="331" y="88">:</text>
<text class="j-ft j-af" x="373" y="88">_</text>
<rect class="j-m j-a1" x="396" y="58" width="38" height="46" rx="6"/>
<rect class="j-m j-a2" x="438" y="58" width="38" height="46" rx="6"/>
<rect class="j-m j-a3" x="480" y="58" width="38" height="46" rx="6"/>
<rect class="j-m j-a4" x="522" y="58" width="38" height="46" rx="6"/>
<rect class="j-m j-a5" x="564" y="58" width="38" height="46" rx="6"/>
<text class="j-mt j-a1" x="415" y="88">"</text>
<text class="j-mt j-a2" x="457" y="88">A</text>
<text class="j-mt j-a3" x="499" y="88">d</text>
<text class="j-mt j-a4" x="541" y="88">a</text>
<text class="j-mt j-a5" x="583" y="88">"</text>
<circle class="j-dot j-a1" cx="415" cy="44" r="4"/>
<circle class="j-dot j-a2" cx="457" cy="44" r="4"/>
<circle class="j-dot j-a3" cx="499" cy="44" r="4"/>
<circle class="j-dot j-a4" cx="541" cy="44" r="4"/>
<circle class="j-dot j-a5" cx="583" cy="44" r="4"/>
<rect class="j-f j-al" x="606" y="58" width="38" height="46" rx="6"/>
<text class="j-ft j-al" x="625" y="88">}</text>
<text class="j-bd j-ab1" x="20" y="150">forced run emitted in one go — zero forward passes</text>
<text class="j-bd j-ab2" x="20" y="150">model chooses the value — one forward pass per token</text>
<rect class="j-f" x="20" y="172" width="16" height="16" rx="3"/>
<text class="j-lg" x="44" y="185">forced by the grammar</text>
<rect class="j-m" x="200" y="172" width="16" height="16" rx="3"/>
<text class="j-lg" x="224" y="185">chosen by the model, dot marks a forward pass</text>
</svg>
<figcaption>Nine of the fourteen characters here have exactly one legal continuation, so the engine writes them straight into the output and only calls the model for the five that are genuinely uncertain.</figcaption>
</figure>

### The speedup, derived

Let $T$ be the number of tokens in the output, $f$ the fraction of those tokens that fall inside a forced run, and $L$ the mean forced-run length in tokens. Without jump-forward you pay $T$ decode steps. With it, you pay one step per free token plus one step per *run*:

$$T_{\text{steps}} = (1-f)\,T \;+\; \frac{f\,T}{L}$$

and the speedup is

$$S = \frac{T}{T_{\text{steps}}} = \frac{1}{(1-f) + f/L}$$

Two things fall out immediately. First, $S$ is bounded above by ${1/(1-f)}$ — no matter how long the runs are, the free tokens set the floor. Second, $L$ has strongly diminishing returns: going from $L=1$ to $L=4$ captures most of the available win, and $L=20$ is barely better than $L=8$. Long forced runs are pleasant but the *fraction* is what matters.

Why is a run of $N$ tokens one step and not zero? Because you still owe the KV cache. The jumped tokens have no keys and values in the cache, and the model needs them before it can produce token $N+1$. But that is one batched prefill over $N$ tokens, not $N$ sequential decode steps. Decode is memory-bound at roughly (weight bytes) / (HBM bandwidth) — see [the roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) — so processing 9 tokens in one pass reads the weights once instead of nine times. Better still, you can defer it entirely: the jumped tokens simply become the query of the *next* forward pass, which was going to happen anyway. Done that way, a forced run genuinely costs zero extra steps and the formula above is conservative.

#### Worked example: how forced is a real schema?

Take a four-field object with fixed key order and this output:

`{"name": "Ada Lovelace", "age": 36, "email": "ada@example.com", "active": true}`

That is 79 characters. Classify each one by whether the grammar left the model a choice:

| Span | Chars | Forced? | Why |
| --- | --- | --- | --- |
| `{"name": "` | 10 | forced | root is an object; first key is fixed; the value must be a string, so its opening quote is fixed |
| `Ada Lovelace"` | 13 | free | any string content, terminated when the model chooses |
| `, "age": ` | 9 | forced | second key is fixed |
| `36` | 2 | free | any integer |
| `, "email": "` | 12 | forced | third key plus the string's opening quote |
| `ada@example.com"` | 16 | free | any string content |
| `, "active": ` | 12 | forced | fourth key |
| `t` | 1 | free | `true` or `false` — one real bit of choice |
| `rue}` | 4 | forced | after `t` the boolean is determined, and the object then closes |

Forced: ${10+9+12+12+4 = 47}$ characters across 5 runs. Free: 32 characters. So **59% of the characters are decided by the schema, not the model**, and the mean forced run is 9.4 characters.

Convert to tokens at the $\bar{c} \approx 3.5$ characters-per-token figure above (again: verify this on your tokenizer, do not trust my constant). Total $T \approx 79/3.5 \approx 23$ tokens, forced $\approx 13$, so $f \approx 0.57$ and $L \approx 13/5 \approx 2.7$ tokens per run. Then

$$S = \frac{1}{(1-0.57) + 0.57/2.7} = \frac{1}{0.43 + 0.21} = 1.56$$

**A 1.56× speedup on a completely ordinary schema, from doing less work rather than faster work** — derived, with every input stated. And that is the *pessimistic* accounting where each run costs a full step. Schemas with more fields and shorter values push $f$ higher: a ten-field object of short enums can exceed 80% forced, where the ceiling ${1/(1-f)}$ is 5×.

### Why some libraries structurally cannot do this

Jump-forward requires the constraint machinery to sit *inside the decode loop* and be able to say "skip the next $N$ forward passes." If your integration point is a **logits processor** — a callback that receives a logits tensor and returns a modified logits tensor — you have no such authority. The engine has already decided to run a forward pass by the time you are called. You can zero out 128,255 entries, but the forward pass has happened.

That is not my inference; it is stated in vLLM's structured-decoding post as a named limitation of the Outlines backend: with the logit-processor approach, "we cannot do jump-forward decoding." The same post notes the more general shape of the problem for that backend — masks computed per request rather than shared, and CFG mode that "can potentially crash the engine."

This is worth internalizing as an architecture lesson beyond grammars. **An extension point that only lets you transform the output of a step can never let you skip the step.** If you are designing an inference engine, decide early whether your constraint API is a filter or a co-scheduler, because the ceiling on performance is set by that decision and not by how fast your masking code is.

## 5. Building it: `nanoserve/decoding/gbnf.py` and `pda.py`

Enough theory. The following code is complete enough to run against a real tokenizer.

### 5.1 A GBNF parser

The item vocabulary is deliberately tiny: a literal string, a character class, and a reference to another rule. Groups and repetition are desugared into fresh anonymous rules at parse time, which is what turns `x*` into recursion and therefore what makes the stack necessary.

```python
# nanoserve/decoding/gbnf.py
from __future__ import annotations
import re
from typing import Dict, List, Tuple

# An item is one of:
#   ("lit", "abc")             a literal string, consumed one char at a time
#   ("cls", frozenset, bool)   a character class; bool = negated
#   ("ref", "name")            a nonterminal
Item = Tuple
Alt = Tuple[Item, ...]
Rules = Dict[str, List[Alt]]

_TOK = re.compile(
    r"""(?P<skip>\s+|\#[^\n]*)
      | (?P<assign>::=)
      | (?P<name>[A-Za-z_][A-Za-z0-9_-]*)
      | (?P<str>"(?:[^"\\]|\\.)*")
      | (?P<cls>\[\^?(?:[^\]\\]|\\.)*\])
      | (?P<op>[|()*+?])""",
    re.VERBOSE,
)

_ESC = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "]": "]", "/": "/"}


def _unescape(s: str) -> str:
    out, i = [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            out.append(_ESC.get(s[i + 1], s[i + 1]))
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def _parse_class(tok: str) -> Item:
    body = tok[1:-1]
    neg = body.startswith("^")
    if neg:
        body = body[1:]
    chars, i = set(), 0
    while i < len(body):
        if body[i] == "\\":
            c, i = _ESC.get(body[i + 1], body[i + 1]), i + 2
        else:
            c, i = body[i], i + 1
        # a range like a-z
        if i < len(body) and body[i] == "-" and i + 1 < len(body):
            hi = body[i + 1]
            chars.update(chr(x) for x in range(ord(c), ord(hi) + 1))
            i += 2
        else:
            chars.add(c)
    return ("cls", frozenset(chars), neg)


class GbnfParser:
    def __init__(self, text: str):
        self.toks = [
            (m.lastgroup, m.group())
            for m in _TOK.finditer(text)
            if m.lastgroup != "skip"
        ]
        self.i = 0
        self.rules: Rules = {}
        self._anon = 0

    def _peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else (None, None)

    def _fresh(self, alts: List[Alt]) -> str:
        self._anon += 1
        name = f"_anon{self._anon}"
        self.rules[name] = alts
        return name

    def parse(self) -> Rules:
        while self.i < len(self.toks):
            kind, val = self.toks[self.i]
            assert kind == "name", f"expected a rule name, got {val!r}"
            self.i += 1
            assert self.toks[self.i][0] == "assign", "expected ::="
            self.i += 1
            self.rules[val] = self._alts()
        return self.rules

    def _alts(self) -> List[Alt]:
        out = [self._seq()]
        while self._peek() == ("op", "|"):
            self.i += 1
            out.append(self._seq())
        return out

    def _seq(self) -> Alt:
        items: List[Item] = []
        while True:
            kind, val = self._peek()
            if kind is None or val in ("|", ")"):
                break
            # a new rule definition starts with  name ::=
            if kind == "name" and self.toks[self.i + 1 : self.i + 2] == [
                ("assign", "::=")
            ]:
                break
            items.append(self._postfix(self._atom()))
        return tuple(items)

    def _atom(self) -> Item:
        kind, val = self._peek()
        self.i += 1
        if kind == "str":
            return ("lit", _unescape(val[1:-1]))
        if kind == "cls":
            return _parse_class(val)
        if kind == "name":
            return ("ref", val)
        if val == "(":
            alts = self._alts()
            assert self._peek() == ("op", ")"), "unclosed group"
            self.i += 1
            return ("ref", self._fresh(alts))
        raise SyntaxError(f"unexpected token {val!r}")

    def _postfix(self, item: Item) -> Item:
        kind, val = self._peek()
        if kind != "op" or val not in ("*", "+", "?"):
            return item
        self.i += 1
        if val == "?":
            return ("ref", self._fresh([(item,), ()]))
        # x*  ==>  R ::= x R | <empty>          (right recursion: needs a stack)
        star = self._fresh([])
        self.rules[star] = [(item, ("ref", star)), ()]
        if val == "*":
            return ("ref", star)
        return ("ref", self._fresh([(item, ("ref", star))]))
```

The three lines that matter are the last five: `x*` becomes a rule that refers to *itself*. Every repetition in GBNF is recursion in the compiled grammar, and recursion is what the stack pays for.

### 5.2 The pushdown machine

A stack is a tuple of items still owed, top first. Because a nonterminal can expand several ways, the machine state is a **frozen set of stacks** — every parse still alive. This is nondeterministic in the textbook sense and it is the simplest correct thing to write; §8 covers when it explodes and what to do about it.

```python
# nanoserve/decoding/pda.py
from __future__ import annotations
from functools import lru_cache
from typing import FrozenSet, Optional, Tuple

Stack = Tuple  # a tuple of Items, top first
State = FrozenSet[Stack]

MAX_STACK = 512  # guards left recursion and runaway nesting


class Pda:
    def __init__(self, rules, root: str = "root", max_depth: int = 64):
        self.rules = rules
        self.root = root
        self.max_depth = max_depth

    # --- expand nonterminals until every live stack shows a terminal on top ---
    def closure(self, stacks) -> State:
        out, work, seen = set(), list(stacks), set()
        while work:
            st = work.pop()
            if st in seen:
                continue
            seen.add(st)
            if len(st) > MAX_STACK:
                raise RecursionError(
                    "grammar stack exceeded MAX_STACK: left recursion or "
                    "an unbounded rule. Rewrite the rule right-recursively."
                )
            if not st:
                out.add(st)          # this parse can legally end here
                continue
            head = st[0]
            if head[0] == "ref":
                # depth cap: refuse to open another nesting level past max_depth
                if len(st) > self.max_depth * 4:
                    continue
                for alt in self.rules[head[1]]:
                    work.append(tuple(alt) + st[1:])
            else:
                out.add(st)
        return frozenset(out)

    def start(self) -> State:
        return self.closure([(("ref", self.root),)])

    def accepting(self, state: State) -> bool:
        return () in state

    def advance(self, state: State, ch: str) -> State:
        nxt = set()
        for st in state:
            if not st:
                continue
            head = st[0]
            if head[0] == "lit":
                s = head[1]
                if s and s[0] == ch:
                    rest = s[1:]
                    nxt.add(((("lit", rest),) if rest else ()) + st[1:])
            elif head[0] == "cls":
                _, chars, neg = head
                if (ch in chars) != neg:
                    nxt.add(st[1:])
        return self.closure(nxt) if nxt else frozenset()

    def single_char(self, state: State) -> Optional[str]:
        """The unique legal next character, or None if 0, 2+, or the value may end."""
        if self.accepting(state):
            return None            # ending is also legal, so nothing is forced
        cand = set()
        for st in state:
            head = st[0]
            if head[0] == "lit":
                cand.add(head[1][0])
            else:
                _, chars, neg = head
                if neg or len(chars) > 1:
                    return None    # a negated or wide class always admits many
                cand.update(chars)
            if len(cand) > 1:
                return None
        return next(iter(cand)) if cand else None
```

Two details are load-bearing and easy to get wrong.

`single_char` returns `None` whenever the state is accepting. If the parse could legally *stop* here, then "keep going" is a choice, and forcing a character would be the machine overruling the model. Skipping this check is how you ship an engine that never emits a short array.

`closure` raises rather than hangs on left recursion. Write `expr ::= expr "+" term` in GBNF and the expansion grows the stack forever. Right-recursive grammars terminate; left-recursive ones do not, under this simple algorithm. A loud error naming the fix beats a hung request.

### 5.3 A trie over the vocabulary, and the mask

```python
# nanoserve/decoding/vocab_trie.py
class TrieNode:
    __slots__ = ("kids", "ids")
    def __init__(self):
        self.kids = {}
        self.ids = []


def build_trie(tokenizer):
    """Map every ordinary token to its literal string and index it by prefix."""
    root, skipped = TrieNode(), []
    special = set(tokenizer.all_special_ids)
    for tid in range(len(tokenizer)):
        if tid in special:
            continue
        piece = tokenizer.convert_ids_to_tokens(tid)
        s = tokenizer.convert_tokens_to_string([piece])
        if not s or "�" in s:
            skipped.append(tid)      # byte-fallback fragments: not valid text
            continue
        node = root
        for ch in s:
            node = node.kids.setdefault(ch, TrieNode())
        node.ids.append(tid)
    return root, skipped
```

Byte-fallback tokens are the trap here — a token that is half of a UTF-8 sequence has no character-level meaning, so it cannot be checked against a character grammar. Excluding them is correct but has a consequence: the model can no longer spell out-of-vocabulary text byte by byte inside a constrained string. If your users write emoji into JSON values, work at the byte level instead of the character level, or accept the restriction knowingly. The mechanics of byte-fallback are covered in [the tokenizer boundary](/blog/machine-learning/large-language-model/bpe-tokenizer).

Now the mask, as a pruned DFS with the cost instrumented:

```python
# nanoserve/decoding/mask.py
import torch


def allowed_token_ids(pda, state, trie):
    """DFS the vocab trie, carrying the machine state. Returns (ids, nodes_visited)."""
    ids, visited = [], 0
    stack = [(trie, state)]
    while stack:
        node, st = stack.pop()
        visited += 1
        ids.extend(node.ids)
        for ch, kid in node.kids.items():
            nst = pda.advance(st, ch)
            if nst:
                stack.append((kid, nst))
    return ids, visited


class MaskCache:
    """Keyed on the PDA state, which is hashable because it is a frozenset."""

    def __init__(self, pda, trie, vocab_size, device, capacity=4096):
        self.pda, self.trie = pda, trie
        self.V, self.device, self.cap = vocab_size, device, capacity
        self.store, self.hits, self.misses, self.nodes = {}, 0, 0, 0

    def get(self, state) -> torch.Tensor:
        m = self.store.get(state)
        if m is not None:
            self.hits += 1
            return m
        self.misses += 1
        ids, visited = allowed_token_ids(self.pda, state, self.trie)
        self.nodes += visited
        mask = torch.zeros(self.V, dtype=torch.bool)
        if ids:
            mask[torch.tensor(ids, dtype=torch.long)] = True
        mask = mask.to(self.device, non_blocking=True)
        if len(self.store) >= self.cap:
            self.store.pop(next(iter(self.store)))
        self.store[state] = mask
        return mask
```

A `[128256]` bool mask is 128,256 bytes on device as stored here; if that matters, pack it to 16,032 bytes with `torch.tensor(..., dtype=torch.uint8)` bit-packing and unpack in the kernel — which is exactly the "bitmask" vocabulary the vLLM post uses. At 4,096 cached entries the unpacked version is about 525 MB, which is too much for VRAM; keep the cache on the host and transfer, or pack. This is a real decision, not a detail.

### 5.4 The processor, with the jump-forward path

```python
# nanoserve/decoding/grammar_processor.py
import torch


class GrammarProcessor:
    """Constrains one request. Owns the PDA state and the emitted text."""

    def __init__(self, pda, mask_cache, tokenizer, eos_id, min_jump=3):
        self.pda, self.masks, self.tok = pda, mask_cache, tokenizer
        self.eos_id, self.min_jump = eos_id, min_jump
        self.state = pda.start()
        self.text = ""
        self.stats = {"jump_chars": 0, "jump_runs": 0, "model_steps": 0}

    # ---- the fork from figure 4 -----------------------------------------
    def forced_prefix(self, limit: int = 64) -> str:
        st, out = self.state, []
        while len(out) < limit:
            ch = self.pda.single_char(st)
            if ch is None:
                break
            out.append(ch)
            st = self.pda.advance(st, ch)
            if not st:
                break
        return "".join(out)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        mask = self.masks.get(self.state)
        if self.pda.accepting(self.state):
            mask = mask.clone()
            mask[self.eos_id] = True          # only now may the model stop
        legal_mass = torch.softmax(logits.float(), -1)[mask].sum().item()
        self.stats["legal_mass"] = legal_mass
        return logits.masked_fill(~mask, float("-inf"))

    def accept_text(self, s: str) -> None:
        for ch in s:
            self.state = self.pda.advance(self.state, ch)
            if not self.state:
                raise ValueError(f"grammar rejected {ch!r} after {self.text!r}")
        self.text += s
```

And the decode loop that uses it. This is the part that a logits processor cannot express, because it decides *whether to call the model at all*:

```python
# nanoserve/decoding/loop.py
def _retokenize(tok, text: str):
    return tok.encode(text, add_special_tokens=False)


def _common_prefix_len(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def generate_constrained(engine, proc, prompt_ids, max_tokens=512):
    ids = list(prompt_ids)
    pending = []                      # tokens jumped but not yet in the KV cache
    while len(ids) - len(prompt_ids) < max_tokens:
        forced = proc.forced_prefix()
        if len(forced) >= proc.min_jump:
            new_ids = _retokenize(proc.tok, proc.text + forced)
            keep = _common_prefix_len(ids[len(prompt_ids):], new_ids)
            engine.truncate_kv(len(prompt_ids) + keep)      # boundary may shift
            ids = ids[: len(prompt_ids) + keep] + new_ids[keep:]
            pending = new_ids[keep:-1]  # hold back the last token: see below
            ids = ids[: len(ids) - 1]
            proc.accept_text(forced)
            proc.stats["jump_chars"] += len(forced)
            proc.stats["jump_runs"] += 1
            continue

        # the jumped tokens ride along as the query of this forward pass
        logits = engine.forward(ids, new_tokens=pending or ids[-1:])
        pending = []
        proc.stats["model_steps"] += 1
        tok = engine.sample(proc.apply(logits))
        if tok == proc.eos_id:
            break
        ids.append(tok)
        proc.accept_text(proc.tok.decode([tok]))
    return ids
```

Three things in there deserve their own paragraph.

**Retokenization is not optional.** The forced characters must be turned into token ids, and greedy BPE does not respect the boundary you happen to be standing on. If the model already emitted `"na` as one token and the forced continuation is `me": `, the correct tokenization of `"name": ` may split differently — the tokenizer merges across the seam. So you re-encode the whole constrained span and diff against what you emitted. Everything before the divergence point keeps its KV; everything after is recomputed. On a schema with long fixed keys this is a real cost, and it is why `min_jump` exists: a two-character jump is not worth a cache truncation.

**Hold back the last token of every jump.** The final token of the forced run sits at a merge boundary — the next character the model chooses may want to merge into it. Emitting it commits to a tokenization the model would never have produced, which pushes the sequence off the distribution the model was trained on and degrades the very next prediction. Dropping it and letting the model pick that token *under the mask* costs one step and keeps you on-policy. The mask makes the choice nearly deterministic anyway.

**The jumped tokens are free query tokens.** They ride into the next forward pass as extra query positions, so the KV they need is filled by a pass that was happening regardless. This is why the derivation in §4 is conservative.

### 5.5 What it prints

```python
from transformers import AutoTokenizer
from nanoserve.decoding.gbnf import GbnfParser
from nanoserve.decoding.pda import Pda

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rules = GbnfParser(PERSON_GBNF).parse()      # the 4-field schema from section 4
pda = Pda(rules, root="root")

st = pda.start()
for ch in '{"name": "Ada':
    st = pda.advance(st, ch)
    assert st, f"rejected at {ch!r}"
print("live parses:", len(st))
print("forced next:", GrammarProcessor(pda, ...).forced_prefix())
```

```console
live parses: 2
forced next:
```

Two live parses inside a string (the character class branch and the escape branch), and nothing forced — the model is genuinely choosing here. Run the same thing after `{` and `forced next` comes back as `"name": "`, nine characters the model never sees.

## 6. Compilation is a TTFT problem, not a throughput problem

Everything above assumed the grammar was already compiled. Compiling it is a separate cost, and it lands in the worst possible place: **on the request path, before the first token**.

![A layered view of four places a compiled grammar can be stored, from the raw schema text down to the per-step mask](/imgs/blogs/grammar-based-decoding-gbnf-pushdown-automata-and-xgrammar-5.webp)

Figure 5 is the hierarchy you want. The evidence that this cost is real rather than theoretical comes straight from vLLM's structured-decoding post, which lists among XGrammar's optimizations that the team moved "grammar compilation from Python to C using pthread." Nobody rewrites a component in C with explicit threading because it was already fast enough. The same post names FSM compilation as a significant contributor to time-to-first-token for the Outlines backend.

The arithmetic is unforgiving. If your TTFT service objective is 300 ms and a cold grammar compile takes even 50 ms, you have spent 17% of the budget before the prefill starts — and prefill was the thing you spent all of Track C optimizing. Worse, compile time is not uniform: it scales with the grammar's size and its alternation width, so the pathological schemas in §8 can cost far more than the median one.

Four places to put the cost, from best to worst:

1. **Deploy time.** Enumerate the schemas your product actually uses — usually a dozen tool definitions and a handful of response formats — and compile them at startup. Cost moves to a place where nobody is waiting.
2. **A process-wide cache keyed by grammar hash.** `sha256(canonical_schema_json)` as the key, compiled grammar as the value. The second request with the same schema pays nothing. This is the single highest-leverage line of code in the whole feature.
3. **A shared cache across replicas.** If schemas are user-supplied and numerous, a small Redis or on-disk cache of compiled artifacts turns a cold miss into a deserialization.
4. **On the request path.** Where you end up if you do none of the above.

```python
# nanoserve/decoding/registry.py
import hashlib, json, threading

class GrammarRegistry:
    def __init__(self, compile_fn, capacity=256):
        self._compile, self._cap = compile_fn, capacity
        self._store, self._lock = {}, threading.Lock()
        self.hits = self.misses = 0

    @staticmethod
    def key(schema: dict) -> str:
        canon = json.dumps(schema, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canon.encode()).hexdigest()

    def get(self, schema: dict):
        k = self.key(schema)
        with self._lock:
            g = self._store.get(k)
            if g is not None:
                self.hits += 1
                return g
        g = self._compile(schema)          # outside the lock: this is the slow part
        with self._lock:
            self.misses += 1
            if len(self._store) >= self._cap:
                self._store.pop(next(iter(self._store)))
            self._store[k] = g
        return g

    def warm(self, schemas):
        for s in schemas:
            self.get(s)
```

Note the lock discipline: compilation happens *outside* the lock. Holding a mutex across a multi-hundred-millisecond compile in an async server is how one slow schema stalls every request in the process, which is a different flavour of the same serialization hazard vLLM flagged for batch-wide masking.

#### Worked example: the TTFT budget with and without a warm cache

Say the schema is compiled per request at a cost $t_c$, prefill takes $t_p$, and everything else is $t_o$. TTFT is $t_c + t_p + t_o$. Measure your own $t_c$ with the script in §9 — it is a property of your schema and your library version, and quoting a number I have not run would be worthless. What you *can* reason about without measuring is the shape:

| Scenario | TTFT | Source |
| --- | --- | --- |
| Cold compile every request | $t_c + t_p + t_o$ | derived |
| Warm hash cache, hit | $t_p + t_o$ | derived |
| Warm cache, 1% miss rate | $0.01\,t_c + t_p + t_o$ on average, but p99 still $t_c + t_p + t_o$ | derived |
| Precompiled at deploy | $t_p + t_o$, always | derived |

The third row is the one people miss. A cache with a 99% hit rate fixes your *mean* TTFT and does nothing at all for your p99, because the 1% that miss are exactly the requests in the tail. If you have an SLO on p99 — and you should, per [admission control and latency collapse](/blog/machine-learning/inference-engineering/admission-control-backpressure-and-latency-collapse) — you need row four, not row two.

## 7. The four backends, compared without hand-waving

![A four by four comparison of Outlines, XGrammar, llguidance and llama.cpp GBNF across reach, jump-forward support, host engine and named limitations](/imgs/blogs/grammar-based-decoding-gbnf-pushdown-automata-and-xgrammar-6.webp)

Figure 6 is the summary; the table below is the same content with provenance attached to each cell, because the claims are of genuinely different kinds and blurring them would be dishonest.

| Backend | Expressiveness | Jump-forward | Where it runs | Named limitation | Source |
| --- | --- | --- | --- | --- | --- |
| **Outlines** | Regex-based FSM, plus a CFG mode | Not with a logit-processor integration | vLLM's fallback when XGrammar is insufficient | CFG mode "can potentially crash the engine"; one token per step; the batch mask is applied to all requests | cited: [vLLM, Structured Decoding in vLLM](https://vllm.ai/blog/2025-01-14-struct-decode-intro) |
| **XGrammar** | Context-free, via GBNF | Yes — a PDA design that owns the loop can skip steps | vLLM's default backend when the grammar allows | v0 accepted GBNF only: no regex, no complex JSON via regex patterns or numeric ranges | cited: same post |
| **llguidance** | Context-free, Lark-style syntax input | Yes — fast-forward is part of the design | A Rust core embedded by the host engine | The newest of the four; the smallest body of production reports | general property + observation, not a benchmark |
| **llama.cpp GBNF** | Context-free, GBNF files | Masking at the sampler, not step-skipping | Inside llama.cpp's own sampler chain | Takes GBNF, not JSON Schema — conversion is the caller's job | general property of a sampler-level integration |
| **lm-format-enforcer** | Schema-driven token filtering | Not applicable | A vLLM backend option | Reported to fail "to enforce correct outputs" in long-context scenarios | cited: same vLLM post |

Two rows carry the load.

**Expressiveness is the least interesting axis.** All four reach context-free. Choosing between them on "can it do nested JSON" is choosing on a dimension where they tie. The XGrammar v0 limitation the vLLM post names — GBNF only, so no regex and no complex JSON expressed through regex patterns or numeric ranges — is not about reach but about *input format*: a schema with `"pattern": "^[A-Z]{2}-\\d{4}"` or `"minimum": 0, "maximum": 99` needs a translation layer that v0 did not have, which is precisely why vLLM keeps Outlines as a fallback for grammars XGrammar cannot take.

**Jump-forward is the axis that changes your latency**, and it is decided by integration architecture rather than by grammar theory. This is why the same post reports that XGrammar delivers "up to 5x improvement in time per output token (TPOT) under load" — a cited claim, from the vLLM team, whose setup is described in their post rather than reproduced by me. Treat it as evidence of the mechanism's size, not as a number to put in your own capacity plan. Your number depends on your schemas' forced fraction, and §4 gives you the formula to predict it.

## 8. Where grammars turn hostile

Every failure of a constrained-decoding system is a disagreement about which strings are legal. There are exactly three parties who can disagree — the schema, the chat template, and the model — and each disagreement has a different signature.

![A three way taxonomy of grammar failures with concrete symptoms under each branch](/imgs/blogs/grammar-based-decoding-gbnf-pushdown-automata-and-xgrammar-7.webp)

### 8.1 The grammar is too permissive: unbounded recursion

```python
NESTED_GBNF = r'''
root ::= "[" root "]" | "x"
'''
```

Nothing in this grammar prefers termination. At every step both `[` and `x` are legal, so the model picks, and a model that has drifted slightly can pick `[` a hundred times in a row. The output is legal at every prefix and never finishes. It stops when `max_tokens` stops it, having burned the whole budget and produced nothing parseable, and the user sees a timeout.

The fix belongs in the grammar, not in the loop. Bound the depth explicitly by expanding the recursive rule a fixed number of times:

```python
def depth_bounded(template: str, max_depth: int) -> str:
    """root_k ::= "[" root_{k-1} "]" | "x" ;  root_0 ::= "x" """
    lines = [f'root ::= root{max_depth}']
    for k in range(max_depth, 0, -1):
        lines.append(f'root{k} ::= "[" root{k - 1} "]" | "x"')
    lines.append('root0 ::= "x"')
    return "\n".join(lines)
```

Yes, this is the exponential-unrolling problem from §1 in miniature — but bounded to a *small* depth chosen by you, on a single rule, at compile time. Depth 8 costs 9 rules, not $2^8$ states, because the PDA still handles the bracket matching; you are only forbidding deeper openings. `Pda.max_depth` in §5.2 does the same thing at runtime, which is the escape hatch when the grammar arrives from a user.

Also: a grammar is not a length bound. Always keep `max_tokens` as a hard second bound. Every constrained request needs both.

### 8.2 The grammar fights the chat template

This one produces the most confusing bug reports, because the output looks perfect and the request never returns.

Your engine stops on the EOS token. Your grammar does not know EOS exists — it describes JSON, and EOS is not a JSON character. So the mask sets EOS to $-\infty$ at every step. The model completes the object, the machine reaches an accepting state, and then every legal token is a character the grammar would allow *after* a complete value — perhaps nothing at all — and the mask is all $-\infty$. Softmax over an all-$-\infty$ row is NaN, and depending on your sampler you get a crash, a garbage token, or an infinite loop.

The fix is in `GrammarProcessor.apply` above: unmask EOS **exactly when the machine is accepting**, never before. One line, and it is the difference between a working feature and a class of tickets.

The related failure: models with structured chat templates want to emit template tokens — `<|eot_id|>`, a closing `</tool_call>`, a reasoning block terminator. If those live inside the span you are constraining, the grammar must include them, or you must exclude them from the constrained span. The practical rule is to constrain a *sub-span* of the output and hand the template tokens back to the ordinary stop-condition machinery:

```python
# constrain only between the markers the template produces
SPAN = r'''
root   ::= "<tool_call>" ws value ws "</tool_call>"
'''
```

For reasoning models the same applies to the thinking block: let the model think unconstrained, switch the processor on at the boundary. Constraining a chain of thought into JSON is a reliable way to make a model dumber.

### 8.3 The grammar is too strict: the model disagrees

The subtlest failure. The mask is correct, the output is valid, and the content is nonsense — because every token the grammar allowed had essentially zero probability, and masking renormalized noise into a confident-looking answer.

Instrument it. Before applying the mask, measure the probability mass the model assigned to the legal set:

$$m = \sum_{t \in A} \text{softmax}(z)_t$$

where $A$ is the allowed set. This is the `legal_mass` line in §5.4, and it is the single most valuable number a constrained decoder can log. What threshold? Derive one. Under a uniform distribution over a 128,256-token vocabulary, each token carries $7.8 \times 10^{-6}$ of mass, so an allowed set of 30 tokens would carry about $2.3 \times 10^{-4}$. If your measured legal mass falls *below* that, the model is doing worse than random on the legal set — it is actively trying to write something else. That makes $10^{-4}$ a defensible alarm threshold, derived rather than tuned.

```python
if proc.stats["legal_mass"] < 1e-4:
    log.warning(
        "grammar-model disagreement: legal_mass=%.2e after %r",
        proc.stats["legal_mass"], proc.text[-40:],
    )
```

Causes, roughly in order of frequency:

- **The prompt does not mention the schema.** Constrained decoding is not a substitute for asking. Put the schema in the prompt *and* enforce it; the mask then agrees with what the model already wanted.
- **The grammar forbids whitespace the model insists on.** `"{" ws member` where `ws ::= [ \t\n]*` is fine; `"{" member` with no whitespace rule will fight every instruction-tuned model on earth, because they were all trained to pretty-print JSON.
- **Key order.** A grammar with fixed field order is much more forced (good for jump-forward) but disagrees with a model that wants to emit fields in a different order. Permitting any order costs you the forced runs. This is a genuine trade, and it is worth measuring both ways for your schema.
- **The model is too small for the schema.** No mask fixes this.

### 8.4 Pathological grammars

**Huge alternation.** An enum with 5,000 string values compiles to 5,000 alternatives. At the first character the closure contains 5,000 live stacks, and the state — a frozenset of 5,000 tuples — is expensive to hash, expensive to compare, and useless as a cache key because it appears once. The fix is determinization: build a trie of the enum values and emit a grammar shaped like the trie, so the branching factor at each character is the number of distinct next characters, not the number of values. Alternation width is the parameter to watch when you accept user-supplied schemas; cap it.

**Deep nondeterminism.** The `frozenset` state representation is honest but pays for ambiguity. A grammar where many alternatives share long prefixes keeps many parses alive for a long time. Real libraries handle this with left-factoring, with Earley-style item sets, or with XGrammar's decomposition into per-rule FSMs. If your live-parse count (`len(state)` in §5.5) routinely exceeds a few dozen, your grammar needs factoring, not your code needs optimizing.

**Long literals.** A grammar containing a 4,000-character fixed string is wonderful for jump-forward — that whole string is one forced run — but it means the retokenization diff in §5.4 runs over 4,000 characters. Cap `forced_prefix(limit=...)` and jump in chunks.

### 8.5 The stress matrix

| Stress | What breaks first | Mitigation | Source |
| --- | --- | --- | --- |
| Batch 1, simple schema | Nothing; mask cost is hidden by the step | Ship it | derived from §3 timing |
| Batch 64, 64 distinct grammars | Host-side mask computation serializes the step | Per-request masks computed off the critical path; cache per grammar | cited: vLLM notes the batch-wide mask hazard |
| 128k-token context | Nothing new — the grammar state is independent of context length | None needed | derived |
| Recursive schema, depth unbounded | Request never terminates | Depth-bounded expansion (§8.1) plus `max_tokens` | derived |
| Enum with 5,000 values | Compile time and state hashing | Trie-shaped grammar; cap alternation width | derived |
| Byte-fallback token needed inside a string | Token excluded, model cannot spell the character | Work at byte level, or document the restriction | derived from §5.3 |
| Grammar with no EOS path | NaN softmax or infinite generation | Unmask EOS only when accepting | derived from §8.2 |

## 9. Measuring it honestly

Four numbers, and none of them is "tokens per second."

**1. Compile time per unique grammar.** Measure it once per schema, not per request, and record the distribution across your real schema population. This is a pure host measurement — no GPU, no synchronization needed.

```python
# bench/grammar_compile.py
import time, statistics

def compile_stats(schemas, compile_fn, reps=5):
    rows = []
    for name, s in schemas.items():
        compile_fn(s)                        # warm imports and any lazy init
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            compile_fn(s)
            ts.append((time.perf_counter() - t0) * 1e3)
        rows.append((name, statistics.median(ts), max(ts)))
    for name, med, mx in sorted(rows, key=lambda r: -r[1]):
        print(f"{name:32s} median {med:8.2f} ms   max {mx:8.2f} ms")
```

**2. Mask time per step, p50 and p99, with the cache hit rate beside it.** The p99 is what matters: a cache miss on a fresh state is the one that lands in a user's tail latency. Report `mask_cache.hits / (hits + misses)` and `mask_cache.nodes / misses` — mean trie nodes visited per miss — because that second number tells you whether your pruning is working.

**3. Jump-forward yield.** `jump_chars / total_chars` and `model_steps / total_tokens`. Plug the first into the formula from §4 and check the predicted speedup against the observed step count. If they disagree, your `min_jump` is too high or your retokenization is thrashing the cache.

**4. Legal mass, p1 not p50.** The median is always near 1 when things work. The first percentile is where the disagreements live.

For the end-to-end TTFT and TPOT comparison, the discipline from [setting up a reproducible benchmark](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) applies without modification: warm up until the step time is stationary, call `torch.cuda.synchronize()` before you read the clock, time GPU work with `torch.cuda.Event` rather than wall clock, lock your clocks, and drive load **open-loop** with Poisson arrivals. Closed-loop load hides exactly the queueing effect that a serialized mask computation creates — with a fixed number of client threads, a slow step slows the arrival rate too, and the problem disappears from your measurement while remaining in production.

And the comparison that actually answers the question:

| Configuration | What it isolates | Source |
| --- | --- | --- |
| Unconstrained | The baseline TPOT | reproduce: `bench/loadgen.py --no-grammar` |
| Grammar, jump-forward off | Pure masking overhead | reproduce: `bench/loadgen.py --grammar --no-jump` |
| Grammar, jump-forward on | The net effect users feel | reproduce: `bench/loadgen.py --grammar --jump` |
| Grammar, cold cache each request | The compile cost you are hiding | reproduce: `bench/loadgen.py --grammar --no-cache` |

Row two minus row one is your overhead. Row three minus row two is your win. If row three is not faster than row one on an object-heavy schema, your jump-forward path is not doing its job — go back to §4 and check the forced fraction your grammar actually produces, because a grammar that permits arbitrary key order has almost no forced runs.

## 10. Case studies and cited results

**vLLM's backend selection.** In [Structured Decoding in vLLM](https://vllm.ai/blog/2025-01-14-struct-decode-intro) (2025-01-14), the vLLM team describes running XGrammar as the default structured-decoding backend and falling back to Outlines when XGrammar is insufficient for a given grammar. The reason for the fallback is the v0 input limitation already discussed: GBNF only, with no regex support and therefore no complex JSON expressed via regex patterns or numeric ranges. This is a useful engineering pattern in its own right — a fast specialized path with a general slow path behind it, selected per request by capability rather than by configuration.

**The 5× TPOT claim.** The same post reports XGrammar achieving "up to 5x improvement in time per output token (TPOT) under load." Note the two qualifiers doing real work: *up to*, and *under load*. Under load is where the mechanism pays, because that is where a serialized host-side mask hurts most and where every skipped forward pass returns capacity to the batch. Do not carry this number into a capacity plan for your own schemas; carry the mechanism, and use the §4 formula with your own forced fraction to predict.

**Compilation as a first-class cost.** The move of grammar compilation "from Python to C using pthread" for XGrammar, reported in the same post, is the strongest available evidence that compile time is a production concern rather than a micro-optimization. The same post identifies FSM compilation as a significant TTFT contributor for the Outlines path. If a project rewrites its compiler in C with explicit threading, the cost was showing up in someone's latency graph.

**Long-context enforcement failures.** vLLM also lists `lm-format-enforcer` as a backend that fails "to enforce correct outputs" in long-context scenarios. The general lesson: an enforcement mechanism that works on short outputs is not thereby validated on long ones. Test your grammar path at the context lengths you actually serve, and put a validator on the output in production regardless — a parse failure that the constrained decoder was supposed to make impossible is exactly the alarm you want.

**The GBNF format itself.** llama.cpp's [`grammars/README.md`](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md) documents the notation and ships example grammars including one for JSON. It is worth reading as a specification: the format is small, and its choices — character classes, postfix repetition, no left recursion in practice — are the reason it maps cleanly onto a stack machine like the one in §5.

## 11. When to reach for this, and when not

**Use a full grammar when** your output has genuine recursion (nested objects, arrays of objects, matched tags, recursive schema references), or when your schema is user-supplied and you cannot audit its depth, or when you want jump-forward — which needs the grammar machinery even for shapes an FSM could handle.

**Stay with the FSM from post 18 when** your schema is flat and fixed. A regular constraint precomputes completely, has no per-step host cost, no compile-time surprise, and no live-parse count to worry about. Do not pay for a stack you never push.

**Skip constrained decoding entirely when** the output is prose, or a single classification label you can match with a cheap post-hoc check, or when a retry loop is cheaper than the machinery. For a one-field enum, sampling then validating and retrying on failure is a smaller system with better failure modes, and the accuracy question in [structured output in production](/blog/machine-learning/inference-engineering/structured-output-in-production-streaming-json-and-tool-calls) is worth reading before you constrain anything by default.

**Use vLLM's built-in structured outputs rather than your own code when** you are shipping a product. This post exists so you know what is happening inside `guided_json`, what its costs are, and which of its failure modes are architectural. It does not exist so you maintain a grammar compiler. The engine teams have solved the batch-wide mask problem, the fallback selection, and the compile caching, and they will keep solving them. Build yours to understand it; run theirs. [The vLLM deep dive](/blog/machine-learning/model-serving/vllm-deep-dive) is the map of what you would be reimplementing.

**Write your own when** you are building an engine (that is this series), when you need a constraint that no library expresses — a grammar coupled to external state, a schema that changes mid-generation, a domain-specific language with its own semantics — or when jump-forward matters enough that you need it inside a loop you control.

## 12. Key takeaways

1. **No finite-state machine recognizes nested structure.** The pigeonhole proof in §1 is short and complete; a DFA for JSON to depth $d$ needs at least $2^{d}$ states, so unrolling is not an engineering choice, it is a wall.
2. **A stack buys unbounded depth for a fixed control size** — $\Theta(2^d)$ states become $\Theta(R)$ control plus $\Theta(d)$ runtime stack. That is the entire reason grammar backends are pushdown machines.
3. **You cannot precompute the mask table for a PDA**, because the state includes the stack. Recover the precomputation by decomposing the grammar — vLLM describes XGrammar's PDA as "a collection of FSMs, and each FSM represents a context-free grammar," which is exactly this move.
4. **Compute masks by walking a vocabulary trie, not by testing every token.** Pruning an illegal subtree kills thousands of tokens per node visit.
5. **Cache masks on the machine state and grammars on a schema hash.** A 128,256-token bitmask is 16,032 bytes; caching a thousand of them is free, and it is the difference between a viable and an unusable implementation.
6. **Jump-forward is the biggest available win and it is architectural.** With a forced fraction $f$ and mean run length $L$, the speedup is $\frac{1}{(1-f)+f/L}$ — 1.56× on the ordinary four-field schema of §4. A logits-processor integration cannot do it at all; vLLM states this plainly for the Outlines backend.
7. **Unmask EOS exactly when the machine is accepting.** Earlier is a truncated object; never is a request that does not return and a NaN softmax.
8. **Log the legal probability mass every step.** Below about $10^{-4}$ on a 128k vocabulary the model is doing worse than uniform on the legal set, and your valid output is noise wearing a schema.
9. **Grammar compilation belongs at deploy time.** A 99%-hit cache fixes your mean TTFT and does nothing for your p99, because misses are the tail by construction.
10. **Bound the depth in the grammar and keep `max_tokens`.** A grammar constrains shape, never length.

## Further reading

- [Structured Decoding in vLLM: a gentle introduction](https://vllm.ai/blog/2025-01-14-struct-decode-intro) — the primary source for the FSM-versus-PDA framing, the XGrammar and Outlines backend comparison, the 5× TPOT claim, and the acknowledged limitations quoted throughout this post.
- [llama.cpp `grammars/README.md`](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md) — the GBNF specification and worked example grammars.
- [XGrammar](https://github.com/mlc-ai/xgrammar) — the reference implementation of the collection-of-FSMs pushdown design.
- [llguidance](https://github.com/guidance-ai/llguidance) — a Rust constraint engine with fast-forward built into its design.
- [Outlines](https://github.com/dottxt-ai/outlines) — the regex-and-FSM backend, and a good read for how far the regular approach can be pushed.
- [Constrained decoding from first principles: masking logits with an FSM](/blog/machine-learning/inference-engineering/constrained-decoding-from-first-principles-masking-logits-with-an-fsm) — the previous post, and the machine this one replaces.
- [Structured output in production: streaming JSON and tool calls](/blog/machine-learning/inference-engineering/structured-output-in-production-streaming-json-and-tool-calls) — the next post: partial JSON over SSE, tool-call parsing mid-stream, and whether constraining hurts accuracy.
- [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) — the capstone, where the decoding layer takes its place against the rest of the engine.
