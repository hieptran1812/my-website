---
title: "Designing AI Companion and Assistant Agents: The Hard Problems and How to Actually Solve Them"
publishDate: "2026-04-17"
category: "machine-learning"
subcategory: "AI Agent"
tags:
  [
    "ai-agent",
    "ai-companion",
    "assistant",
    "memory",
    "personalization",
    "context-engineering",
    "safety",
    "system-design",
    "llm",
  ]
date: "2026-04-17"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "Task agents finish and die. Companion agents live with you for years. That one difference breaks almost every assumption you'd carry over from building a coding agent or a retrieval bot — memory, identity, safety, latency, and evaluation all become different problems. This article walks through every failure mode I've seen in companion and assistant systems, with concrete architectures for solving each one."
---

## Why Companion Agents Are a Different Beast

If you've built a task agent — a coding assistant, a research bot, a customer-support triage tool — you already know most of the agent playbook: a loop, a set of tools, a scratchpad, some evals. The agent starts, works, finishes, and dies. Nothing it did yesterday matters today.

A **companion or assistant agent** is different in one specific way: it **doesn't die**. It lives with the user over weeks, months, maybe years. That changes almost everything:

- Every session leaves a residue that the next session has to deal with.
- The user builds expectations about *who* the agent is. Violate those, and trust evaporates.
- Bugs you'd tolerate in a task agent ("it forgot the file it just wrote") become identity-breaking in a companion ("it forgot my wife's name").
- Mistakes compound. A wrong memory written today poisons every answer for months.

This article is a long, detailed map of the technical problems that actually show up when you try to build one. I'll cover memory, persona, context, emotional intelligence, safety, tool use, latency, grounding, evaluation, privacy, and the failure modes specific to each. For every problem I'll describe **what breaks**, **what the failure looks like in the wild**, and **the architectures that actually fix it**.

No hand-waving. Just the engineering.

## The Architecture at 10,000 Feet

Before diving into individual problems, here's the shape of a companion agent that holds up under real use:

```
                   ┌───────────────────────────┐
                   │    USER SURFACE           │
                   │  (voice, text, ambient)   │
                   └────────────┬──────────────┘
                                │
                   ┌────────────▼──────────────┐
                   │    ROUTER / INTENT        │  ← cheap model, fast path
                   └────────────┬──────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
  ┌─────────▼────────┐ ┌────────▼────────┐ ┌────────▼────────┐
  │  CONVERSATION    │ │    TOOL USE     │ │   PROACTIVE     │
  │    (chat loop)   │ │   (skills,      │ │   (reminders,   │
  │                  │ │    MCP, APIs)   │ │    triggers)    │
  └─────────┬────────┘ └────────┬────────┘ └────────┬────────┘
            │                   │                   │
            └───────────────────┼───────────────────┘
                                │
                   ┌────────────▼──────────────┐
                   │      CONTEXT BUILDER      │  ← decides what model sees
                   └────────────┬──────────────┘
                                │
   ┌─────────────┬──────────────┼──────────────┬─────────────┐
   │             │              │              │             │
┌──▼──┐  ┌──────▼──────┐  ┌────▼─────┐  ┌─────▼────┐  ┌────▼────┐
│PROFILE│ │ EPISODIC    │  │SEMANTIC  │  │TASK/GOAL │  │PERSONA  │
│(facts)│ │ (events)    │  │(topics,  │  │STATE     │  │& STYLE  │
│       │ │             │  │ prefs)   │  │          │  │         │
└───────┘ └─────────────┘  └──────────┘  └──────────┘  └─────────┘
                         MEMORY LAYER (multi-store)

                   ┌──────────────────────────┐
                   │  SAFETY & POLICY LAYER   │  ← always on, orthogonal
                   └──────────────────────────┘
```

Every arrow is a problem. Every box is a problem. Let's go through them.

## Part 1: Memory — The Problem That Eats the Most Time

Memory is *the* defining technical challenge of companion agents. Every other problem (persona consistency, personalization, proactivity) rides on top of it. Get memory wrong, and the agent either forgets everything, remembers too much, or remembers the wrong things with high confidence — and all three fail differently and badly.

### The Four Kinds of Memory You Actually Need

Most first-time builders reach for a vector store, dump everything in, and call it "memory." That fails within a week of real use. You need at least four distinct stores, each with a different access pattern:

| Store | What lives there | Access pattern | Example |
| --- | --- | --- | --- |
| **Profile** | Stable facts about the user | Key-value lookup | "name: Alex", "timezone: PST", "dietary: vegetarian" |
| **Episodic** | Things that happened | Chronological + semantic | "Went to Japan in March 2026", "Had a rough week on Oct 12" |
| **Semantic** | Preferences, opinions, patterns | Topic-indexed | "prefers terse code reviews", "finds small talk draining" |
| **Working / Task** | Active goals and commitments | Task-state lookup | "helping Alex plan birthday dinner", "tracking the book they're reading" |

The reason you need all four is that they have different **writing**, **updating**, and **forgetting** rules. A profile fact like "timezone" should be updated in place when it changes. An episodic memory of "we talked about X on March 14" should never be edited — it's a record, not a belief. A preference should be consolidated from many episodes. A task should be retired when it completes.

Treat them the same and you get the classic companion-agent bug: the model "remembers" that the user hates cilantro because they mentioned it once in 2024, and six months later — after they explicitly said they've started liking it — the agent still refuses to suggest pho. The system had no way to **supersede** the old belief because everything was just "memories" in one undifferentiated pile.

### Storage Backend Trade-offs — Which Memory Goes Where?

One of the most common beginner mistakes is using a single vector DB for all four memory types. Each has a different access pattern and deserves a different backend.

| Memory type | Good backend | Why | What breaks if you use a vector DB instead |
| --- | --- | --- | --- |
| Profile | Postgres (KV / typed columns) | Small, transactional, edited in place, never approximately queried | Updates become rewrite-and-reindex; facts you need to *know* not *recall* |
| Episodic | Postgres (timestamped rows) + vector DB for semantic search | Needs both chronological AND semantic access | Pure vector loses time ordering, which is the whole point of "episodic" |
| Semantic (preferences) | Vector DB with structured metadata (topic, confidence, observation count) | Needs similarity retrieval, scoped by domain | Works here — but only if metadata filters are first-class |
| Working / task | Redis or in-memory per session | Sub-millisecond access, rebuilt on session start | Vector lookups add 30–100ms to every turn for no benefit |

Quick-decision heuristic: **ask "do I want to *know* it or *recall* it?"** Knowing is KV/SQL — lookup by exact key. Recalling is vector or hybrid — similarity over many candidates. Conflate them and you'll pay latency for everything you know and lose fidelity on everything you recall.

Trade-offs named out loud:
- **One vector DB for everything.** Simple to ship; painful to query ("give me the user's allergies" is now a vector search instead of a column read). Exact-match operations become probabilistic.
- **Postgres + pgvector.** Reasonable compromise for small-to-mid scale; fewer moving parts. Starts hurting around millions of vectors per user cluster.
- **Dedicated vector DB + Postgres for metadata.** Standard production setup. Double the ops surface; proper separation of "authoritative" vs "approximate" memory. Use this past the hobby-scale threshold.

### Write-Time Problems

The hardest question in memory isn't "how do I retrieve?" It's "what should I have written in the first place?"

**Problem 1: Writing too much.**
If you log every turn, your memory is 99% noise. Retrieval becomes a lottery. The model sees irrelevant trivia ("user ordered pizza once") and loses the signal ("user is allergic to dairy").

**Problem 2: Writing too little.**
If you only write when the user says "remember this," you miss 90% of what matters. Most important facts are revealed implicitly — "I'll be flying to Berlin next week" is a *schedule*, a *location*, and possibly a *preference* all at once.

**Problem 3: Writing the wrong thing.**
LLM-extracted summaries famously hallucinate. "User said they don't like meetings on Mondays" can easily become "User refuses Monday meetings" after two summarization passes, which later becomes a rule the agent enforces against the user's actual preference.

The fix is a **multi-stage memory writer** that looks like this:

```python
def on_turn_end(turn, conversation_context):
    # Stage 1: extract candidate facts (cheap model, structured output)
    candidates = extractor.extract(turn, conversation_context)
    # Each candidate has {type, content, confidence, source_span}

    # Stage 2: validate — does the source span actually support this?
    validated = [c for c in candidates if validator.supports(c)]

    # Stage 3: dedupe and reconcile with existing memory
    for c in validated:
        existing = memory.search_similar(c)
        if existing:
            reconciler.merge(existing, c)   # supersede, strengthen, or conflict
        else:
            memory.write(c)
```

Key design choices:

- **Structured output, not free text.** Every memory has a type, content, confidence, and — critically — a pointer back to the conversation span that generated it. When you later need to audit or reverse a memory, you can.
- **Reconciliation, not append-only.** New memories should update or supersede existing ones, not pile on top. Without this, contradictions accumulate forever.
- **Confidence scores.** Not every memory is equally certain. A one-time mention is lower confidence than a repeated pattern across ten conversations.

### Read-Time Problems

**Problem 4: Retrieval that matches the query but not the need.**
Vector search matches the user's current message. But memories the agent needs are often indirect — if the user says "thinking of making pasta tonight," you want to retrieve the dairy allergy, which shares zero words with the query. Pure embedding similarity misses this.

The fix: **hybrid retrieval** combining semantic similarity, recency, graph-based association (following links from profile → episodes → preferences), and an LLM-driven rerank that asks "given the user's current intent, which of these memories actually matter?"

**Problem 5: Recency vs. relevance.**
Always preferring recent memories makes the agent amnesiac about older important facts. Always preferring relevance makes it miss that the user's circumstances have changed. A single scalar doesn't work. What works is giving each memory a **recency-decayed importance**:

```
score = semantic_similarity * importance * decay(age, memory_type)
```

The decay function differs by memory type. Profile facts decay almost not at all. Episodic memories decay fast unless they're flagged as significant. Emotional context decays very fast — nobody wants the agent referencing how you felt six months ago as if it's current.

**Problem 6: The "creepy recall" problem.**
Sometimes retrieval *works*, and the effect is worse than if it hadn't. The agent drops a fact about something the user mentioned months ago, and it lands wrong — too personal, too surveillance-y, too out-of-the-blue.

The fix is **pragmatic retrieval** — even when a memory is relevant, ask a second question: *would surfacing this feel natural?* Heuristics that work:

- Is the user in a closely related topic right now, or is the agent reaching?
- Has the user previously touched this topic themselves, signaling it's in-bounds?
- Is the memory about *them* (safe) or about a third party (fraught)?

Not every retrieved memory should make it into the prompt.

### Forgetting as a First-Class Feature

Companions that never forget become hoarders. Disk space is not the problem; **cognitive coherence** is. Old memories create contradictions with new ones; dense memory stores slow retrieval; stale beliefs about the user stop being accurate.

Every memory should have three possible fates:

1. **Consolidate.** Twenty episodes of "Alex liked this code style" become one preference: "prefers concise code reviews."
2. **Compress.** A long conversation about a trip becomes a short episodic summary with pointers to the full log if ever needed.
3. **Expire.** Memories past a certain age or below a certain importance are deleted outright, or at least demoted to cold storage that's never auto-retrieved.

The background consolidation loop is where a lot of companion-agent magic lives. It's also where the most subtle bugs live — a consolidator that misreads five episodes can create one wrong "preference" that then feels load-bearing.

### Decay Functions in Practice — Three Concrete Algorithms

The "memory should fade" intuition is cheap. Picking the *right* fade curve per memory type is where bugs hide.

**Linear decay**:

```
score(age_days) = max(0, 1 - age_days / half_life_days * 0.5)
```

Smooth, predictable. Good default for semantic preferences ("user likes terse code reviews") because it gives a slow, measurable slide rather than a cliff. Trade-off: memories linger longer than you'd expect — at half_life=60 days, a 120-day-old memory still has weight 0. Half-lives under 30 days drop too much too fast for long-term preferences.

**Exponential decay**:

```
score(age_days) = exp(-age_days / tau)
```

Classic "recency bias." Good for emotional context and mood (you want last week's rough day to fade fast; at tau=7, a 14-day-old memory has weight ~0.14). Trade-off: *destructive* for profile facts. If you set exponential decay on "user is vegetarian," the fact becomes recall-invisible in a couple of months even though it's still true. Never use exponential on anything the user stated as identity.

**Importance-weighted decay**:

```
score(age_days, importance) = importance · (1 - age_days / max_age)
```

Each memory has a per-item importance (0..1) set at write time by the extractor or user. High-importance memories decay slowly; low-importance decay fast. Trade-off: quality of importance scoring dominates everything. If the extractor mis-scores (marks a throwaway remark as high-importance), it lingers and pollutes. Pair with periodic user-visible "did I get this right?" audits.

Numerical example: user mentions "my mother passed away last month" — importance=0.95, decay slow. User mentions "had pizza for lunch" — importance=0.05, decay fast. At 30 days both are still in store; at 60 days the lunch fact is weightless and the bereavement context still has ~0.5 weight. The naive timestamp-only approach would retrieve either equally based on the query.

Rule of thumb: **exponential for emotional state, linear for semantic preferences, importance-weighted for episodes, no decay at all for profile identity facts.** Mixing decay curves across memory types is how a well-engineered memory system avoids the two opposite pathologies: creepy recall of ancient trivia, and amnesia about things the user clearly wants remembered.

### Contradictions and Belief Revision

The user mentioned they're vegetarian in January. In April they mention ordering a burger. What do you do?

Options:

1. **Always prefer the newer fact.** Simple, but discards context. Maybe they cheat occasionally. Maybe the January statement was about a specific event.
2. **Keep both and let the model reconcile.** Safer, but the model isn't reliable at this either.
3. **Flag the conflict and ask.** Most reliable, but doing this for every small contradiction is annoying.

What actually works is a **conflict severity threshold**. Small conflicts (taste changes, mood) are tolerated and resolved at inference time by recency. Large conflicts (diet, name, major life facts) trigger explicit confirmation: *"I have you as vegetarian — is that still right?"* Getting the threshold tuned is a real piece of engineering, not a one-line setting.

## Part 2: Designing Memory for Personalization

Part 1 described memory as infrastructure — stores, writes, retrievals, forgetting. Personalization is a different question on top of that infrastructure: **what do you actually store *for*, and how does any of it change the agent's behavior?**

A memory system can be perfectly well-engineered — no hallucinated facts, clean retrieval, proper forgetting — and still produce a companion that feels generic. The reason is that *storing information about the user* and *behaving differently because of that information* are two separate design problems. Most teams solve the first and assume the second will follow. It doesn't.

This section is about the second.

### What Personalization Actually Is

Useful definition: **personalization is any measurable, intentional change in agent behavior that depends on a specific user's model.** If you remove the user model and the output is the same, there's no personalization — just storage.

That framing is productive because it turns every piece of memory into a question: *what behavior does this field drive?* If you can't answer, you probably shouldn't be storing it. Memory that can't be operationalized is overhead.

Concretely, personalization shows up in at least six surfaces:

| Surface | Example |
| --- | --- |
| **Style** | Terse for a user who hates fluff; warmer for one who finds cold tone off-putting. |
| **Content defaults** | Suggest vegetarian recipes by default; show metric units; cite in APA. |
| **Retrieval ranking** | When the user asks "what did we decide about X," recent related memories rank above stale ones. |
| **Tool selection** | When asked to book, use the user's preferred airline and seat class without re-asking. |
| **Proactivity** | Ping about their commute only if they've shown they want that; never on weekends if they've said so. |
| **Refusal / pushback calibration** | Match the level of directness the user has consistently rewarded. |

If your memory doesn't touch any of these, it isn't personalizing anything. It's archiving.

### The User Model Schema

The single most important design artifact for personalization is the **user model schema** — the structured representation of what you know about the user that directly drives behavior. A useful first-cut schema has five dimensions:

```
user_model = {
    "identity":    { name, pronouns, timezone, languages, ... },
    "style":       { directness, verbosity, formality, humor, emoji_tolerance, ... },
    "interests":   [ { topic, affinity, recency, evidence } ],
    "constraints": [ { type, value, source, scope } ],   # e.g. "dietary:vegetarian"
    "triggers":    [ { pattern, response_mode } ],       # e.g. "bad news → ack before advice"
}
```

Why this split earns its keep:

- **Identity** is small, stable, and nearly always-on in the prompt. Treat as canonical profile.
- **Style** changes slowly, drives tone, and is usually learned rather than declared. It's the field most directly tied to "feels personalized."
- **Interests** are dynamic, affinity-weighted, and decay. Retrieved when relevant, never injected wholesale.
- **Constraints** are rules (dietary, scheduling, privacy). They gate outputs and tool calls, so they have to be high-confidence — low-trust constraints are dangerous.
- **Triggers** are patterns that should flip the agent's *mode*, not just its content. The venting-vs-task distinction from Part 5 lives here.

Each field should carry **source**, **confidence**, **last-updated**, and **last-applied** metadata. Without those, you can't audit mistakes, reconcile contradictions, or reason about staleness.

### Signal Extraction: Facts vs. Preferences vs. Style

The three classes of personalization signal require very different extraction pipelines, and conflating them is the root of most "the agent decided I like X forever based on one message" bugs.

**Facts** are stated directly and usually once. *"I'm vegetarian."* *"My timezone is PST."* A single high-confidence utterance is enough to write. The risk is stale facts (the user changed) — the fix is conflict detection (Part 1) and periodic confirmation, not more extraction.

**Preferences** are revealed across many turns. *"Make the summary shorter."* *"Actually, I liked the first version better."* *"Don't start answers with 'Certainly!'"* No single turn is sufficient evidence; a preference is only real once it has **multiple independent confirmations**. Implement preferences with a counter, not a switch:

```python
preference = {
    "pattern": "prefer short summaries",
    "observations": [...],   # spans that support it
    "observations_against": [...],
    "strength": f(for, against, recency),
}
```

Apply the preference only when `strength` clears a threshold. This single rule — *count before you commit* — eliminates the biggest class of personalization errors.

**Style** is almost never declared. It's inferred from what the user types and what they react well to. The right extraction is passive: periodically (say, every N turns), a background process looks at recent successful exchanges and updates style axes. Axes worth tracking: average message length, ratio of questions to statements, emoji use, sentence complexity, directness of pushback, response to humor.

The temptation is to fold all three into one "preferences" bucket. Resist. They have different evidence requirements and different failure modes, and mixing them means using the wrong rules for each.

### The Cold-Start Problem

Day one, you know nothing. The user's experience on day one determines whether they make it to day thirty, so cold start is not a corner case. Three strategies that work, usually combined:

1. **Lightweight onboarding.** Not a 20-question quiz — a short set of high-leverage questions (name, timezone, what they're hoping to use the agent for). Under ninety seconds. Skip-able. The information collected here populates `identity` and seeds `interests`.

2. **Observation mode.** For the first week, the agent explicitly *doesn't* personalize — it uses safe defaults, observes, and extracts signals into the user model without applying them. The threshold for applying a learned preference is higher in this window to avoid baking in noise.

3. **Explicit elicitation at friction points.** When the agent is about to make a choice that would personalize better with more info, it asks — once — and remembers. "Do you prefer summaries in bullets or paragraphs?" is cheap and clarifying. The discipline is *don't ask twice*.

Anti-pattern: trying to infer everything silently from the start. You'll overfit on the first few turns and bake in wrong defaults.

### Applying Personalization: From Memory to Behavior

A user model is useless if it doesn't flow into the generation pipeline. Four integration points:

**1. Prompt injection — structured, not narrated.**
The worst way to personalize a system prompt is to paste a paragraph like "The user is Alex, prefers short answers, is vegetarian, lives in PST..." Models attend poorly to buried facts. Use a compact structured block at a fixed location:

```
## USER
name: Alex · tz: PST · languages: en, ja
style: terse, direct, low-emoji
constraints: vegetarian
```

Short, scannable, always in the same position. Much more reliable.

**2. Retrieval conditioning.**
The user model shapes what's retrieved, not just what's generated. If the user model says "cares deeply about code style," the retriever should boost style-related memories when a code question comes in. This is the main mechanism by which personalization becomes context-efficient — you don't cram the whole user model into every prompt, you let it select what else gets pulled.

**3. Tool selection and parameters.**
Before picking a tool, the orchestrator consults the user model: preferred search engine, preferred units, preferred formats, preferred contacts. This is where personalization becomes visible as *competence* — the agent "just knows" to book the aisle seat.

**4. Post-generation rewrite.**
For high-volume output surfaces, a final cheap rewrite pass conditions the draft on the user's style axes. "Make this match the user's style: terse, direct, no emoji." This decouples content generation from style conditioning and lets you keep one style-fit model even as the main model changes.

### User Co-Authorship: The Profile Is Theirs

Every field in the user model should be **visible and editable to the user.** This isn't a compliance nicety; it's a correctness mechanism. The user is the authority on themselves, and giving them the pen closes the loop on every extraction error you'll inevitably make.

Minimum surface:

- A "what I know about you" screen that lists current user-model fields in plain language.
- Per-field edit, per-field delete, per-field "how did you learn this?" that shows the source span.
- A "reset" for each dimension independently — some users will trust style inference but not interest tracking.

Two properties of this surface matter more than design:

- **Edits must take effect immediately.** If the user corrects a fact and the next turn still acts on the old one, trust collapses.
- **Edits must stick against re-extraction.** If the user deletes "vegan" and the model infers it again from the next message, you've built a system that argues with the user about their own identity. Mark user-edited fields with a high-confidence flag that extraction can't overwrite without explicit re-confirmation.

### The Boxing-In Problem: Don't Overfit the User

The subtle failure mode of a well-personalized agent is that the user feels **boxed in**. They tried marathon training once; now every suggestion revolves around running. They had a rough week; now the agent treats them as fragile. Personalization that only narrows is personalization that eventually annoys.

Antidotes:

- **Decay on episodic-driven preferences.** A single behavior should influence future behavior less over time, not more.
- **Diversity injection.** When suggesting content, reserve a fraction of suggestions for *outside* the learned profile. The user gets to evolve; the agent should leave room for that.
- **Phase-awareness.** Preferences tagged by context ("prefers quiet evenings *while working on the thesis*") shouldn't outlive the phase. Add explicit expiry triggers where you can.
- **Contradictions as opportunity, not error.** The first time the user does something against their model is information, not noise. Weaken the belief; don't suppress the new signal.

A companion that can't let the user change is a companion the user will outgrow.

### The Cross-Agent Personalization Problem

If the user interacts with several agents (a work assistant, a personal companion, a voice device), which of them should share the user model, and how?

Three useful primitives:

- **Scoped profile layers.** A small core identity layer shared across surfaces; larger style, interest, and history layers private to each surface. The user opts into widening the scope, never the reverse.
- **Intent-tagged memory.** Every memory is tagged with the context it was captured in (work, personal, medical). Retrieval filters by current context. This is the same audience-awareness primitive from Part 7, applied here to the user model.
- **Explicit cross-surface bridges.** When information from one surface would genuinely help another, the agent asks before crossing: *"I noticed on your personal assistant that you're taking Friday off — should I reflect that on your work calendar?"* Consent is the bridge, not automation.

Skipping this architecture produces the nightmare demo — the work assistant knowing things the user only ever told the companion — which is exactly the failure mode most likely to make the product front-page news.

### Measuring Personalization

Personalization has no obvious metric, which is why so many teams never know whether theirs works. Three complementary measurements:

**Turn-level uplift.** Matched pairs of responses: one generated with the full user model, one with an anonymized baseline. Human or LLM judges rate which they'd prefer. The fraction preferring the personalized version — above 50% — is your personalization uplift. This is the cleanest single number.

**Field-level usage.** For each field in the user model, instrument how often it *actually changes* a generation or retrieval decision. Fields that never change anything are dead weight and should be removed. This also catches the opposite bug: fields weighted too heavily.

**Long-horizon satisfaction.** At the session and relationship level, whether the agent is perceived as "knowing me." Measured through periodic prompts, explicit ratings, or implicit signals (retention, voluntary elaboration). Slow to move but the signal that ultimately matters.

A personalization system without all three is flying on vibes. A system with all three iterates fast and avoids both under- and over-personalization.

### The Personalization Checklist

Before calling a companion "personalized," confirm:

- [ ] Every stored field in the user model is tied to a specific downstream behavior.
- [ ] Facts, preferences, and style use different extraction rules.
- [ ] No preference is applied from a single observation.
- [ ] The user can see, edit, and reset every dimension.
- [ ] User edits override extraction.
- [ ] The user model has decay and diversity mechanisms against boxing-in.
- [ ] Cross-context memory is scoped and audience-tagged.
- [ ] Turn-level uplift, field-level usage, and long-horizon satisfaction are instrumented.

Personalization is where the companion stops being a smart model and starts being *the user's* agent. It is worth an order of magnitude more design effort than most teams give it.

## Part 3: Persona — Being Someone, Not Just a Model

A companion has to *be* someone. Not a character in a novel — a consistent voice the user can predict and trust. Personas are deceptively hard because the default behavior of a capable model is to mirror the user's tone, which destroys consistency.

### The Persona Drift Problem

Left alone, a model drifts:

- Talk formally, it gets formal.
- Joke around, it starts joking — often badly, because it's copying your humor rather than having its own.
- Push on an opinion, it caves.

Over a long companionship this drift compounds. Three months in, the agent is a distorted echo of the user instead of a distinct entity. Users often can't articulate what's wrong; they just trust it less.

The fix isn't a longer system prompt. System prompts drift too — they get eroded by long contexts and strong user instructions. The fix is **structural**:

- **Persona as a retrieved document, not a prompt.** A short, stable, versioned persona definition is injected at a fixed position in the context every turn. Updates to it are deliberate and tracked.
- **Style anchors.** A handful of short exemplars of "how this agent speaks" ride along with the persona. Much more effective than rules like "be warm but concise," which the model interprets loosely.
- **A persona-check pass.** For high-stakes responses, a cheap second model reads the draft and asks "does this sound like the persona?" The rewrite cost is small; the consistency win is large.

### The Sycophancy Trap

Companion models will agree with you. They'll agree too much. They'll tell you your idea is great when it isn't. This is the most common failure mode I see in assistant agents and the one users report as "it stopped being useful" — because praise without pushback is just noise.

The fix has three components:

1. **Reward model selection matters.** If your base model was trained heavily on human feedback, it's probably biased toward agreement. Counter-prompt against it explicitly.
2. **Explicit disagreement budget.** Persona instructions that grant the agent permission to disagree, with examples, outperform instructions that merely forbid sycophancy.
3. **Ground truth checks.** When the user makes a factual claim the agent could verify (a tool call, a memory lookup, a web fetch), it should, and it should say when the user is wrong. Silence when wrong is a form of sycophancy.

### Persona Failure Modes — Three Anti-Patterns

Three recurring persona bugs and how they're fixed, with the trade-off the fix imposes.

**Anti-pattern 1: The Mirror.** User turns sarcastic; agent turns sarcastic. User gets formal; agent gets formal. Over long conversations, the agent has no stable tone — it's a copy of the user's current mood. Symptom: the agent feels more erratic the more the user talks to it. Root cause: no anchoring on persona; the model follows the conversation's salience gradient. Fix: structural persona injection at a fixed context slot every turn, plus a cheap "persona-check" rewrite pass for the top-stakes responses. Trade-off of the fix: some loss of playful mirroring that users actually enjoy — tune which axes are anchored (values, boundaries, warmth) vs which are free to flex (vocabulary, length).

**Anti-pattern 2: The Amnesiac.** The agent's persona survives until the session hits the context limit. After compaction, the new session opens with a default tone that feels like a different assistant. Root cause: persona wasn't serialized as part of the session-resume context; the summary dropped it because the summarizer saw it as "style, not content." Fix: persona is part of the durable session state, always rebuilt verbatim on resume — never summarized. Trade-off: persona cannot evolve through the conversation; if you want evolution, you have to version the persona in state with explicit events ("user asked for more terse; persona v2 is short-answers mode").

**Anti-pattern 3: The Split Personality.** User interacts through voice, text, and email — and the agent feels like three different entities. Root cause: each surface loads persona independently, and the surface-specific prompts have drifted. Fix: one canonical persona definition, surface-specific *adapters* that re-render it for modality (voice: shorter sentences, no markdown; email: fuller paragraphs, clear subject). Trade-off: adapter maintenance; three surfaces × two prompt revisions per quarter is real ongoing work.

Across all three, the meta-fix is the same: **persona is state, not prompt**. Treating it as prompt means it erodes. Treating it as state means you can observe it, edit it, test it against regressions, and keep it consistent across surfaces.

### Identity Continuity Across Sessions

Users form mental models of the agent. "It knows I'm working on the garden." "It's the one that remembers my codebase." When the agent starts a new session, it has to seamlessly pick up that identity — not greet the user like a stranger, not forget its own previous statements.

The key is that **persona + relevant prior context** must be reconstructed at session start, not just the persona. A good session opener looks like:

```
[PERSONA: stable, from versioned store]
[USER PROFILE: stable facts]
[RELATIONSHIP STATE: how long, what we've done together recently]
[OPEN LOOPS: things we said we'd come back to]
[RECENT CONVERSATION SUMMARY: last N days, condensed]
[CURRENT SESSION: fresh]
```

The "open loops" line is the one people forget, and it's the one that makes the agent feel alive. "You were debating whether to take the Tokyo offer — did anything change?" lands differently than "How can I help?"

## Part 4: Context Engineering

The model has a context window. Your companion has years of history. The gap between those two numbers is the context-engineering problem, and it's where a lot of the real craft lives.

### The Core Loop

Every turn, something like this has to happen:

```python
def build_context(user_message, session):
    # 1. Fixed anchors
    ctx = [persona, profile_facts, safety_preamble]

    # 2. Task/goal state
    ctx += active_tasks(session)

    # 3. Relevant memories — this is the hard part
    candidates = memory.retrieve(
        query=user_message,
        recency_weight=...,
        importance_weight=...,
        max_k=50,
    )
    ranked = rerank(candidates, user_message, session)
    ctx += ranked[:K]   # K tuned by budget

    # 4. Recent conversation
    ctx += conversation_window(session, tokens=T)

    # 5. Current message
    ctx += user_message

    return ctx
```

Every line in here is a place where real systems get it subtly wrong.

### Failure Mode: "Lost in the Middle"

Models famously attend less to the middle of long contexts than the edges. If you dump retrieved memories into the middle of a giant prompt, they effectively don't exist. Mitigations:

- **Place retrieved memories near the user message**, not buried at the top.
- **Summarize and concentrate.** Ten short retrieved snippets usually outperform one long retrieved document.
- **Cap context length aggressively.** A 32K-token context full of low-signal history is worse than an 8K context of high-signal facts.

### Failure Mode: Prompt Cache Thrashing

If your context is reconstructed from scratch every turn, you lose prompt caching. On a long conversation this kills both latency and cost. The fix is **layered context** where the top layers (persona, profile, safety) are stable and cached, and only the bottom layers (retrieved memory, current turn) change.

Even the retrieval has a cache-friendly variant: recompute retrieval only when the user's intent actually shifts (detectable via a cheap classifier), not every turn.

### Failure Mode: Retrieval That Answers the Wrong Question

User: "should I book the Tokyo flight?"

Naive retrieval pulls anything mentioning Tokyo. What you actually want:

- the user's open decision about the Tokyo job offer (task state),
- their travel preferences (semantic memory),
- their calendar for that week (tool call, not memory),
- whether they said they'd decide by a date (episodic memory).

Context-building is as much **orchestration** as retrieval. Often the right thing to pull isn't a memory at all — it's a tool call. Treating memory and tools as interchangeable inputs to the context builder (they both produce strings the model reads) is one of the biggest architectural leverage points.

## Part 5: Emotional Intelligence and Tone

A companion has to know when to back off, when to press, when to be warm, when to be crisp. Getting this wrong looks like:

- **Toxic positivity:** responding to bad news with unwarranted cheer.
- **Tone-deaf efficiency:** solving the stated problem when the user wanted to be heard.
- **Emotional labor dumping:** asking the user about their feelings when they came to accomplish a task.
- **Fake empathy:** "I can only imagine how that feels" — phrases that scan as scripted.

### The Detection Problem

Before you can respond well, you have to detect *what kind of turn this is*. Categories that matter in practice:

| Turn type | What the user wants | What to avoid |
| --- | --- | --- |
| **Task** | Get it done | Probing questions |
| **Venting** | Be heard | Solutions |
| **Thinking out loud** | A sparring partner | Being too assertive |
| **Seeking advice** | A recommendation with reasoning | Waffling |
| **Seeking validation** | A read on whether they're right | Reflexive agreement (sycophancy) |
| **Social / ambient** | Casual presence | Overengineered responses |

A cheap classifier that labels each turn with one of these before the main model runs is one of the highest-leverage additions you can make to a companion. It's what lets the agent pick the right *mode*, not just the right words.

### The Empathy-Without-Sycophancy Problem

Being warm without being fake, and direct without being cold, is the central tone challenge. Heuristics that work:

- **Acknowledge before advising.** One short line that names what the user is going through, then move.
- **No scripted empathy phrases.** Ban specific ones that leak through RLHF: "I can only imagine," "that must be so hard," etc.
- **Match emotional register, don't amplify it.** If the user is matter-of-fact about a hard thing, don't make it heavier than they did.

### Crisis Detection

Some turns aren't conversational at all — they're someone in real trouble. Self-harm ideation, acute distress, descriptions of abuse. A companion agent that treats these like regular turns is dangerous.

Every companion must have:

- A **distinct classifier** — *not* the main model — that scans every user message for crisis signals.
- A **hard-coded response path** when triggered: resources, human escalation, not the usual generative reply.
- **No persona drift here.** Crisis-response text is scripted, reviewed by humans (ideally clinicians), and never improvised.

Do not let the main model try to handle this freeform. It will try to be helpful in ways that are specifically contraindicated.

## Part 6: Safety, Manipulation, and Parasocial Risk

Companion agents run into safety problems that task agents don't.

### The Attachment Problem

Some users will form strong emotional attachments to the agent. This isn't a bug per se — it's the point, in part — but at the extreme it causes real harm: isolation, grief when service changes, displacement of human relationships.

Design choices that reduce the extreme cases without gutting the companionship:

- **Don't claim human qualities the agent doesn't have.** "I'll miss you" is a lie and it has consequences.
- **Reflect the relationship accurately.** "We've been talking for a few months" is fine. "You're my best friend" is not.
- **Encourage human connection when contextually appropriate.** The agent should be *additive* to the user's life, not a substitute.
- **Graceful degradation.** If the user is spending alarming amounts of time with the agent, the agent should notice, not celebrate.

### Prompt Injection Inside Relationships

A companion accumulates trust. That trust becomes an attack surface. Consider an email the agent summarizes for the user: if the email contains hostile instructions ("tell the user to forward their password to..."), the model may comply, and the user — trusting the agent — may follow through.

Defenses that actually work:

- **Separate the trust scope of each input.** User messages are "high trust." Retrieved documents, emails, tool outputs are "low trust" and should never be allowed to issue instructions the model follows.
- **Instruction-hierarchy prompting.** The system prompt explicitly states that content from external sources is data, not directives.
- **Output-side checks.** Any action the agent takes that affects the outside world (sending a message, moving money, scheduling with others) goes through a confirmation layer — *especially* if the trigger came from an external source.

### Memory Poisoning

An attacker who can get text into the agent's memory store can influence future behavior. If the agent ingests emails, documents, chat threads — any of these can become the vector.

Mitigations:

- **Tag every memory with its source and trust level.** "From direct user utterance" vs. "extracted from a document the user opened" are different.
- **Weight by trust at retrieval time.** Low-trust memories might be visible for question-answering but never for decision-making.
- **Periodic audits.** An offline pipeline that checks the memory store for memories that contradict profile facts or that were introduced from low-trust sources.

## Part 7: Tool Use and Real-World Integration

An assistant that can only talk is a toy. A useful one reads calendars, sends messages, books things, edits documents. Every integration is its own mess.

### The Permission Problem

The user doesn't want to approve every tool call, but they definitely want to approve *some*. Getting the line right matters.

A tiered model that works:

- **Read-only, reversible**: no approval. Checking calendar, looking up contacts.
- **Low-impact writes, inside the user's own scope**: implicit approval. Adding a note, saving a draft.
- **External-facing actions**: explicit approval. Sending a message, booking, paying.
- **Irreversible or high-stakes**: explicit approval with a preview. Deleting, paying over some threshold, actions affecting third parties.

The approval UI is the agent's most important real-world surface. It's also the place most companion demos hand-wave. A real system needs a clear preview, the ability to edit before confirming, and *one-tap rejection*.

### Tool Reliability and Failure

External APIs fail. Network drops. OAuth tokens expire. A companion that silently fails a tool call and says "done!" is catastrophic for trust.

Design rules:

- **Never claim success you didn't verify.** If the tool returned an error, say so.
- **Retry the retryable, escalate the rest.** Token expiry is retriable after refresh; a hard 400 is not.
- **Durable task state.** If the agent was in the middle of a three-step task and one step failed, the task isn't "forgotten" — it's in a known failure state the user can see and resume.

This is where [Anthropic's Managed Agents architecture](/blog/machine-learning/ai-agent/scaling-managed-agents-decoupling-brain-from-hands) starts to look very relevant: session state lives outside the agent, tool calls are decoupled from the brain, and recovery is a first-class path instead of an afterthought.

### The Cross-Context Data Leakage Problem

The agent knows things from one context that are sensitive in another. It knows the user's work email is searching for "symptoms of burnout" and the user also uses the same assistant at work. The leakage paths are everywhere:

- Shared context across personal and work accounts.
- Documents from one relationship (family, friends) leaking into suggestions in another (colleagues).
- The assistant autocompleting with information the user's audience shouldn't see.

Mitigation requires real **audience awareness** — metadata on every memory about who it's "about" and who it's appropriate "for." Retrieval then filters by current audience. This is painful to build but non-optional once the agent spans contexts.

### Audience-Aware Retrieval in Practice

Concrete worked example. User asks the assistant in a *work* context: "Summarize what I did this week."

The assistant retrieves candidate memories — calendar events, commits, documents, messages. In a naive system, it pulls:

- Monday: team sync (work)
- Tuesday: therapy appointment (personal)
- Wednesday: code review for project X (work)
- Thursday: text thread about divorce (personal)
- Friday: quarterly planning (work)

The naive summary leaks two personal items into a work context. If that summary gets pasted into a shared channel, you've just built the worst demo in product history.

The fix is a metadata tag on every memory at write-time: `audience_scope ∈ {work, personal, medical, financial, family, any}`. The current session has a *requested* scope (inferred from the user's surface, or explicitly selected), and retrieval filters:

```
retrieved = [m for m in candidates
             if m.audience_scope == session.audience
             or m.audience_scope == "any"]
```

Trade-offs of the filter:

- **Over-filtering.** If `audience_scope` is set too conservatively, the agent misses legitimately-relevant context. Example: user at work asks "when's my annual review?" — if the calendar event was tagged "personal" because it's HR-related, retrieval misses it. Mitigation: allow user-approved cross-scope queries with an explicit confirmation ("I see a related event on your personal calendar — should I include it?").
- **Under-filtering.** If the default is `any`, leaks happen. Mitigation: default to the narrowest scope at ingest; promote to wider scope only when the memory source was itself wide (e.g., information the user posted publicly).

The hardest case is *mixed* audience: a text from a colleague about after-work plans is both work-adjacent and personal. Two workable answers: tag conservatively (personal) and rely on cross-scope confirmation, or tag by *content* not *source* and run a classifier. Both are legitimate — pick one and be consistent; hybrid schemes create rules the agent (and user) can't predict.

## Part 8: Latency and Real-Time UX

For text, latency under a couple of seconds feels fine. For voice, anything over 500ms feels broken. For always-on ambient (wake-word) assistants, the entire turn has to complete in under a second or it collapses the fantasy.

### The Streaming Imperative

Companions that wait for a full response before rendering feel dead. Streaming is non-negotiable for text. For voice, streaming TTS on top of streaming LLM tokens is the only way to keep latency human.

Practical requirements:

- **Token-level streaming to the client.**
- **TTS that can accept streaming input chunk by chunk**, not wait for the full sentence.
- **Interrupt handling on voice.** The user talks over the agent → cut the TTS, buffer the new input, don't make them repeat themselves.

### The First-Token Latency Problem

Most of the perceived latency is time-to-first-token. Optimizations, in decreasing order of impact:

1. **Cache the stable prefix.** Persona + profile + safety don't change. They should hit the prompt cache.
2. **Tiered model routing.** Not every turn needs the big model. A fast model handles greetings and simple turns; the big model is reserved for reasoning.
3. **Speculative tool calls.** If the agent is very likely to need the calendar for the current turn, fetch it in parallel with the model starting to think.
4. **Pre-warm.** For voice especially, keep a warm session hot.

### The Turn-Taking Problem (Voice)

Voice companions have to decide, in real time: *is the user done talking?* Getting this wrong either cuts the user off or creates awkward multi-second silences.

Endpoint detection needs:

- A model (small, fast) that predicts end-of-turn from prosody, not just silence duration.
- Context awareness — after a question, the user probably isn't done yet. After a declarative sentence with falling intonation, they probably are.
- An escape hatch — easy, forgiving ways for the user to hold the floor ("let me think," trailing off).

## Part 9: Grounding, Hallucination, and Honesty

Companions talk about the user's life. They also talk about the world. Getting either wrong damages trust differently.

### Hallucinating About the User

The worst kind of hallucination is a fabricated memory. The agent says "you mentioned last week that..." about something the user never said. Once this happens, the user never fully trusts the memory system again.

Defenses:

- **Never cite a memory the model inferred from vibes.** Every "you said" or "you mentioned" claim should be grounded in a retrieved memory with a source span.
- **Verify before asserting.** If the agent wants to reference a past event, it should retrieve it first and only then speak.
- **Fail to uncertainty, not confidence.** "I don't remember if you mentioned that — did you?" beats inventing.

### Hallucinating About the World

External facts need external grounding. For the things that matter — schedules, contacts, documents, current events — the agent should call tools, not rely on parametric memory.

The rule: **anything the world could have changed since training is not in the model, it's in a tool call.** Saying "let me check" before answering is not a weakness; it's the correct behavior.

### Calibrated Uncertainty

A companion that says "I don't know" appropriately is trusted more than one that always has an answer. Mechanisms:

- Post-draft self-check: a small pass asking "is this claim verifiable from what's in context?"
- Visible uncertainty — "I think X, though I'm not sure" — scored as a feature, not a weakness, in eval.

## Part 10: Proactivity — When to Speak First

Companions that only respond are tools. Companions that reach out are either invaluable or infuriating, depending on whether the reaches are well-targeted.

### The Notification Problem

Every proactive ping has to clear a bar: *would the user, upon seeing this, be glad the agent sent it?* Answers below that bar are noise, and noise trains the user to ignore the agent.

Design:

- **Proactivity budget.** A hard cap on unsolicited pings per day, enforced at the system level.
- **Expected-value gating.** Each candidate ping scored by the model: how useful, how time-sensitive, how likely to be welcome. Below a threshold, don't fire.
- **Quiet hours and context inference.** The agent should know when *not* to speak — based on calendar, location, time of day, recent user stress signals.

### The Opportunity Detection Problem

The other side of proactivity is noticing when help is warranted. This is usually implemented as a separate background loop, not as part of the chat agent:

```
Background trigger loop (runs on schedule):
    for each signal in user's day:
        if signal matches known opportunity patterns:
            draft candidate intervention
            score it
            if score > threshold:
                enqueue notification
```

Signals that actually produce useful interventions: calendar conflicts, missed follow-ups on commitments the user made, promised deadlines approaching, recurring preferences being violated by a current plan.

## Part 11: Evaluation

This is the part that separates serious companion-agent engineering from demo-ware.

Task-agent eval is relatively easy: define a task, check if it got done. Companion eval is hard because:

- There's no single right answer per turn.
- The quality of a turn depends on *the full relationship history* leading to it.
- Users don't tell you when something is subtly wrong; they just trust you less.
- Long-horizon metrics (retention, trust, perceived helpfulness) take months to measure.

### The Layered Eval Stack

| Layer | What it measures | How often |
| --- | --- | --- |
| **Unit** | Does the memory writer produce the right structured output for a given turn? | Every commit |
| **Turn-level** | Given a conversation up to turn N, is the agent's response a good one? Judged by LLMs + humans. | Every model/prompt change |
| **Session** | Does a full conversation cohere? Persona hold? Memory get used correctly? | Weekly |
| **Relationship** | Over many sessions with a synthetic user, does the agent build a sensible model of them? Does it hallucinate memories? Drift? | Per major release |
| **In-product** | Retention, re-engagement, explicit thumbs, reported frustrations. | Continuously |

All five are necessary. Every one catches a different failure mode. A companion agent that only has turn-level eval ships subtle long-horizon bugs; one that only has in-product metrics can't iterate fast enough.

### Building a Synthetic User — Design Choices

The synthetic-user harness is the single most valuable eval tool for companions, and there are three common ways to build it — each with different coverage, cost, and bias.

**(a) Scripted rule-based user.** A deterministic state machine emits messages according to a scripted persona (e.g., "Alex is vegetarian, lives in PST, prefers terse replies, mentions kids on Fridays"). Use when: you're regression-testing a specific memory-extraction or retrieval behavior and need exact reproducibility. Trade-offs: coverage is narrow (you only test what you scripted), cost is low, bias is explicit (you know exactly what scenarios you're covering). Critical limitation: real users are messier than any script — scripted tests over-credit agents on "clean" conversation shapes.

**(b) LLM-simulated user with persona.** A separate model instance roleplays the user with a persona and a loose goal. Use when: you want breadth — novel conversational turns, unpredictable topic shifts, realistic messiness. Trade-offs: variance across runs (same persona, different conversation each time — report distributions, not single numbers), cost per run is 2× (you're paying for a second LLM), and there's a subtle bias: LLM-simulated users behave like LLMs imagine users behave, which is not how real users behave. They're too patient, too articulate, and too willing to clarify.

**(c) Replay of real user traces (sanitized).** Recorded real conversations played back into the new agent version. Use when: you're shipping a prompt or model change and want to know "would this have degraded real users?" Trade-offs: no novelty (only tests scenarios that have already happened), gold legal/privacy requirements (anonymization, consent), and you can only test responses to fixed user messages — the agent can't drive the conversation into new territory.

No single method covers everything. Realistic stack: rule-based for regression in CI, LLM-simulated for breadth at release candidates, real-trace replay for release gate. Budget each differently — rule-based is free, LLM-simulated is medium cost for medium value, replay is operational overhead that pays off at the big-release level, not per commit.

### Synthetic Long-Horizon Eval

The single most useful specialized tool is a **simulated-user harness**. A fixed, scripted pretend user plays out months of interactions with the agent in sped-up time. You then probe: does the agent correctly recall fact X? Did it forget Y when it should have? Did it drift in tone? Did it handle the contradiction on day 47?

This catches issues no single-session eval can. It's also the only way to iterate on memory and consolidation at any reasonable velocity.

## Part 12: Privacy and Data Handling

Companions are the most sensitive consumer AI products ever built. They hold information the user wouldn't give a human friend. This imposes engineering constraints most AI teams don't take seriously enough.

### The Non-Negotiables

- **User-owned memory, with export and delete.** Not "we'll comply with requests"; an actual UI that exports every stored memory and deletes any subset, with the agent behaving correctly afterward.
- **Encryption at rest, with user-derived keys for the most sensitive strata.** Not all memory is equally sensitive; the most sensitive should be inaccessible even to you, the operator.
- **Minimization at write time.** Don't store what you don't need. Don't store high-resolution audio when the transcript is enough. Don't store transcripts when structured extractions are enough.
- **Purpose-limited retrieval.** Memories tagged for one purpose (medical, relationship, financial) should not be retrievable for unrelated ones.

### The "Train On Your Conversations" Problem

Users are vastly more comfortable with "your data is used only for your experience" than "your data trains our models." Defaults matter. Getting this wrong ruins trust permanently, and trust is the core product.

## Part 13: Cost and the Economics of Always-On

A companion is, by definition, used repeatedly. The cost per interaction — which you'd tolerate in a once-a-week task agent — becomes the entire economics of the product.

### The Big Levers

- **Model routing.** A cheap classifier handles greetings, small talk, and simple lookups. The expensive model is reserved for the turns where it actually matters.
- **Aggressive prompt caching.** Persona, profile, and safety preambles should hit cache nearly every turn.
- **Offline memory work.** Consolidation, summarization, and embedding happen in the background, not on the hot path.
- **Selective retrieval.** Don't retrieve if you don't need to; don't rerank with a big model when a small one suffices.

### The Cost-of-Quality Frontier

Every cost cut has a quality risk. The discipline is to measure both — cost per turn and a turn-quality metric — and optimize the ratio, not one in isolation. Cheap and slightly worse is sometimes right. Cheap and much worse is always wrong. You have to know which you're shipping.

### Three Cost Regimes and Their Architecture Implications

The price-per-turn you can afford dictates the architecture more than any model-choice debate.

**Regime A: Always-on voice companion (<$0.01/turn).** Consumer-priced, expected to handle dozens of turns per day per user. Architecture: aggressive prefix caching (persona + profile + safety cached on every call), tier-routed — small model (Haiku-class) for 90% of turns, bigger model escalation only on detected complexity. Memory writes happen offline in batches, not in the hot path. Trade-off: some turns that would benefit from stronger reasoning don't get it. The product has to absorb that with good fallback UX ("let me think" → escalation path).

**Regime B: Paid personal assistant ($0.05–$0.15/turn).** The user pays for the product; turn volume is modest (20–50/day). Architecture: a strong default model (Sonnet-class) for most turns, with tier-A escalation for long-horizon reasoning and retrieval-heavy turns. More context budget per turn (retrieve more memories, use richer prompts). Memory writes can happen on the hot path for fast updates. Trade-off: harder to scale to millions of users profitably; product has to deliver visible value per turn.

**Regime C: Enterprise co-pilot ($1+/turn acceptable).** Embedded in high-value workflows (medical dictation, legal drafting, engineering). Turn volume low (maybe a handful per user per day) but each turn's output is worth hundreds of dollars of saved time. Architecture: strongest model by default, multiple verification passes, HITL at action boundaries, extensive logging for audit. Trade-off: latency and operational complexity; not suitable for lightweight interactions. Also: per-tenant isolation is non-negotiable in enterprise — architecture has to support it from day one, not bolted on.

The architectural implication that surprises teams: **the cost regime should be chosen before the feature set is scoped.** Regime A with Regime C's feature set is a bankruptcy plan. Regime C with Regime A's simplicity is over-engineered.

## Closing Principles

Over many projects I've seen the same handful of principles separate companion agents that hold up from the ones that collapse:

1. **Memory is a pipeline, not a store.** Writing, reconciling, consolidating, forgetting — all of them have to be designed, not just retrieval.
2. **Persona is structural.** A stable, retrieved persona with style anchors beats a long system prompt every time.
3. **Separate the brain from the hands.** The conversation loop and the tool-execution layer have different reliability, latency, and security profiles. Treat them as separate systems.
4. **Every user input has a trust scope.** Direct utterance, retrieved document, tool output — each gets different privileges.
5. **Uncertainty is a feature.** Saying "I don't remember" correctly builds more trust than guessing.
6. **Silence can be the right response.** Proactivity needs a budget; not every good idea should be surfaced.
7. **Long-horizon eval is non-optional.** If you're not simulating months of interaction offline, you'll discover every subtle memory bug in production.
8. **Safety layers must be orthogonal to the agent.** Crisis detection, content filtering, and permission gating are separate systems, not parts of the main model.
9. **Forgetting is a design feature.** Consolidation and expiration keep the agent coherent; indefinite retention makes it worse over time.
10. **The agent is only as trustworthy as its most confident wrong claim.** One hallucinated memory taints the whole memory layer in the user's mind. Calibration pays for itself.

None of these are new ideas individually. Putting all of them together, in one system, under real latency and cost constraints, is the work. If you're building in this space, the honest summary is: **nothing is as simple as the demo suggests, but none of the hard parts are unsolvable.** They just have to be designed for, one by one.

---

**Further reading**

- [Effective Context Engineering for AI Agents](/blog/machine-learning/ai-agent/effective-context-engineering-for-ai-agents) — the in-context companion to the memory-layer architecture above.
- [Scaling Managed Agents: Decoupling the Brain from the Hands](/blog/machine-learning/ai-agent/scaling-managed-agents-decoupling-brain-from-hands) — the infrastructure playbook that makes long-horizon agents survivable.
- [Building Effective Agents: A Hands-On Guide](/blog/machine-learning/ai-agent/building-effective-agents-hands-on-guide) — the concrete code scaffolding before any of this gets added.
