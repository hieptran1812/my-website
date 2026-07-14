---
title: "The Expertise Multiplier: Getting Real Value from AI Coding Agents"
date: "2026-07-14"
publishDate: "2026-07-14"
description: "Why the biggest lever in AI-assisted engineering isn't the model — it's the domain judgment you bring and how precisely you frame the work, drawn from Anthropic's analysis of 400,000 Claude Code sessions."
tags:
  [
    "ai-assisted-development",
    "claude-code",
    "ai-agents",
    "developer-productivity",
    "prompting",
    "software-engineering",
    "workflow",
    "best-practices",
    "mental-models",
    "llm",
  ]
category: "software-development"
subcategory: "AI-Assisted Development"
author: "Hiep Tran"
featured: true
readTime: 18
image: "/imgs/blogs/the-expertise-multiplier-1.webp"
---

Two engineers open the same repository, point the same coding agent at the same bug, and type. Twenty minutes later one has a merged, verified fix and the other has a pile of confidently-wrong diffs and a growing suspicion that "the AI just isn't good at this." Same model. Same tools. Same codebase. The variable that separated them was not the agent — it was them.

This is Part 1 of a three-part series on working with AI as a team. This post is about the part no configuration file can fix: the mindset and the judgment that decide whether an agent multiplies you or wastes your afternoon. Part 2 turns that judgment into shared repository infrastructure so a whole team behaves consistently, and Part 3 is about running it day to day — verification, review, and rollout. But none of the machinery in Parts 2 and 3 matters if you get this part wrong, so we start here.

Here is the claim the whole series rests on, and it is uncomfortable if you have spent a career getting good at *writing* code: **the dominant variable in AI-assisted engineering is the human's domain judgment, not the model's coding ability or your typing speed.** Anthropic studied roughly 400,000 Claude Code sessions between October 2025 and April 2026 and found exactly this — the more domain expertise a person brought to a session, the more useful work the agent did per instruction ([Claude Code expertise research](https://www.anthropic.com/research/claude-code-expertise)). Users rated *expert* at a given task reached a verified successful outcome more than twice as often as novices — roughly 28–33% versus 15%. And it showed up in the agent's behavior: given a well-framed problem, Claude executed about **12 actions per prompt**; given a vague one, about **5**. Figure 1 is the shape of that finding, and it is the shape of this entire post.

![Bar chart showing verified success rate rising steeply from novice (~15%) to intermediate and then flattening toward expert (28-33%), annotated that the steepest gain is early where most of the ROI lives](/imgs/blogs/the-expertise-multiplier-1.webp)

## The multiplier is domain expertise, not typing speed

For thirty years the bottleneck in software was *production*: turning a clear idea into correct syntax took time, and the engineers who were fastest and most fluent at that translation were the most productive. A coding agent removes most of that bottleneck. It types faster than you, it never forgets an import, it will happily generate the ninth near-identical handler while you think about something else. If production were still the bottleneck, the best model would simply win and the human would be interchangeable.

It isn't, and they aren't. When you take production off the critical path, the thing that's left — the thing that now determines whether the output is *right* — is understanding the problem well enough to say what "right" even means. That's domain expertise, and it's precisely what the model doesn't have about *your* system. The agent knows how to write a database migration in the abstract; it does not know that your `orders` table has a soft-delete column that three downstream jobs quietly depend on. You know that, or you don't. If you do, you tell the agent, and it executes cleanly. If you don't, the agent writes a textbook-correct migration that breaks production, and no amount of model capability was ever going to save you, because the missing information lived in your head or nowhere at all.

Notice what the numbers in Figure 1 do *not* say. They do not say you must be a world expert. The steepest part of the curve is the climb from novice to intermediate; between intermediate and expert the slope flattens hard. Working, competent knowledge of the problem captures most of the available gain. That is genuinely good news — it means the payoff for learning *just enough* about an unfamiliar area is enormous, and we'll come back to how to bank that payoff fast. The research also found no meaningful occupational wall: people across very different jobs succeeded at coding tasks within a few points of professional software engineers. The skill that transfers is not "can code." It's "understands the problem."

## The division of labor: you plan, it executes

If expertise is the multiplier, where exactly does it get applied? The healthiest way to picture an agent session is as a division of labor with a hard seam down the middle. In the sessions that went well, the human did the overwhelming majority of the *deciding* — roughly 70% of the work was planning and judgment — and the agent did the *doing*. Figure 2 lays out the two columns.

![Two-column diagram of the division of labor: the human column owns planning and judgment (~70%) with decompose, set constraints, define acceptance criteria, enumerate edge cases; the agent column owns execution (~20%) with search the codebase, write and edit files, generate boilerplate, run mechanical refactors; a handoff arrow labeled framed task connects them](/imgs/blogs/the-expertise-multiplier-2.webp)

Your side of the seam is not optional overhead you do while waiting for the "real" work. It *is* the real work. **Decompose** the task into steps small enough that each has an obvious right answer. **Set the constraints** — the library you must use, the pattern the rest of the codebase follows, the thing that must not change. **Define acceptance criteria** so that both you and the agent know what done looks like before a line is written. **Enumerate the edge cases** you already suspect, because you have the scars and the model does not. Everything on the agent's side — searching the codebase, editing files, generating the boilerplate, running the mechanical refactor across forty call sites — is execution that becomes trivially parallelizable and fast *once your side is done well*.

The bar across the bottom of Figure 2 is the payoff for holding up your end: frame the task well and the agent runs about a dozen actions on a single prompt, chaining searches and edits and checks into real progress; frame it badly and it manages about five before it stalls, guesses, or asks you a question you should have answered up front. Same agent. The difference is entirely on your side of the seam.

## What "framing a problem precisely" actually looks like

"Frame it precisely" is easy to nod along to and hard to do, so let's make it concrete. Precision is not politeness and it is not length — a three-paragraph prompt full of "please" and vague aspiration is still vague. Precision is *constraint*: every sentence you add that rules out a wrong path makes the space of things the agent might do smaller, until what's left is mostly the thing you actually wanted. Figure 3 is the mechanism.

![Contrast diagram: a vague ask make it faster fans out into a large action space (rewrite in Rust, add a cache, new algorithm, shard the DB) leading to mostly-wrong rework, while a precise ask with constraints plus an acceptance test narrows to a small action space of one clear edit that passes the test and ships](/imgs/blogs/the-expertise-multiplier-3.webp)

Take the classic vague ask: **"make the checkout endpoint faster."** To the agent, that sentence is a wide-open fan of legitimate interpretations. Rewrite the hot path in a faster language? Add a cache? Swap the algorithm? Shard the database? Each is a defensible reading, most of them are wrong for your situation, and the agent has no way to know which — so it picks one, spends its budget building it, and you spend yours discovering it solved the wrong problem. That's the top row of Figure 3, and the rework is not the model being dumb. It's the model being asked to guess.

Now the same task, framed:

```md
The GET /checkout/summary endpoint p95 is 900ms; target is under 200ms.
Profiling shows 80% of the time is in N+1 queries in CartSerializer
(one query per line item). Fix it by batch-loading line items in a single
query — do NOT add a caching layer, we need live prices. Keep the JSON
response shape identical. Acceptance: the summary_spec integration test
still passes and the endpoint issues at most 3 DB queries total.
```

Look at how much of the fan that closes. The *where* (`CartSerializer`), the *why* (N+1), the *how* (batch load), an explicit **do-not** (no cache — with the reason, so the agent doesn't relitigate it), an invariant to preserve (response shape), and an **acceptance test** that turns "faster" from an opinion into a checkable fact ("at most 3 queries; the spec passes"). There is basically one correct edit left, and the agent will find it in one turn. That's the bottom row of Figure 3 — same task, the only variable is how you framed it.

The pattern generalizes past performance work. Consider a request that sounds harmless: **"add soft-delete to the users table."** Vague, it fans out again — a boolean flag or a `deleted_at` timestamp? Do existing queries need to change, or just new ones? What about the unique constraint on `email` that will now collide the moment someone re-registers with a "deleted" address? The agent will pick a reading and, more dangerously, will *not think to ask* about the email constraint, because it doesn't carry the operational scar that tells you soft-delete and unique constraints are natural enemies. Framed, the same task carries your knowledge into the prompt:

```md
Add soft-delete to the users table using a nullable deleted_at timestamp
(not a boolean). Update the default UserQuery scope to exclude rows where
deleted_at IS NOT NULL, and change the email unique constraint to a partial
index that only applies where deleted_at IS NULL, so a deleted user's email
can be reused. Leave the admin scope unfiltered. Acceptance: existing
user_spec passes, and a new test re-registers a deleted email successfully.
```

Every clause is a piece of domain judgment the model could not have supplied — the timestamp-over-boolean choice, the partial index, the admin-scope exception, the re-registration test. That is what your 70% looks like in practice. It is not writing the migration; it is knowing the four things that make this migration correct in *your* system and spending five sentences to say them.

The acceptance test is the highest-leverage sentence in that whole prompt, and it's the one novices leave out most. It does two jobs at once: it collapses the action space *before* execution, and it gives you an objective way to know the agent is done *after*. Turning a fuzzy ask into a testable specification is a skill senior engineers already practice on humans — it's the same muscle as [turning vague asks into requirements and SLOs](/blog/software-development/system-design/turning-vague-asks-into-requirements-and-slos). Working with an agent just raises the stakes, because the agent will never push back on the vagueness the way a good colleague would. It will simply pick an interpretation and run.

## The four habits of high performers

Across the sessions that ended in verified success, the same four behaviors kept showing up. None of them is about prompt-craft trickery; all of them are about supervision.

**They frame precisely.** Everything from the previous section — constraints, do-nots, an acceptance test — before the agent starts, not as a stream of corrections after it goes wrong.

**They catch the edge cases the agent misses.** The agent will produce something that works on the happy path and quietly ignores the empty list, the timezone boundary, the concurrent write, the row that's soft-deleted. High performers read the output *looking* for the case they know is lurking, because their domain knowledge tells them where the bodies are buried. "What happens when `items` is empty?" is a question you can only ask if you already know it can be empty.

**They recover from obstacles instead of abandoning.** When the agent hits a wall — a failing test, a confusing error, an approach that isn't working — the novice concludes "the AI can't do this" and gives up; the expert treats it as new information, re-frames, and steers. The difference between a session that produced a fix and one that produced a shrug was often just whether the human kept driving after the first setback. This is the same discipline as [reproducing a bug before you try to fix it](/blog/software-development/debugging/reproduce-it-first-or-youre-not-debugging): an obstacle is data, not a verdict.

**They explicitly confirm success.** They don't stop at "the agent said it's done." They run the thing, check the acceptance test, look at the actual behavior, and *then* declare victory. Confirmation is not ceremony — it's the step that catches the plausible-but-wrong output before it becomes a production incident, which is the entire subject of Part 3.

## The competence threshold: working knowledge, not mastery

Return to the shape of Figure 1, because it contains the most actionable idea in this post. If almost all the gain lives in the climb from novice to intermediate, then the highest-return move available to you — on any given task, in any unfamiliar area — is to get yourself to *intermediate on this specific problem, fast*, and then supervise. You do not need to become an authority on the payments gateway. You need enough working knowledge to frame the task, smell a wrong answer, and confirm a right one.

The good news is that "enough" is a low bar and reachable in minutes, and the agent itself is the fastest way to reach it. Before you ask it to *change* anything, use it to *learn* the terrain: have it walk you through how the module works, read the surrounding code yourself, run the thing once and watch what it actually does, and form a hypothesis about where your change belongs. This is deliberately the [hypothesize-and-falsify loop](/blog/software-development/debugging/hypothesize-and-falsify-not-stare-and-hope) applied to comprehension — a few minutes of building a mental model buys you the entire rest of the session, because now you can tell the agent the *where* and the *why* instead of hoping it guesses them.

The failure mode this guards against is subtle: an agent is more than capable of making a novice *feel* like an expert. It produces fluent, confident, well-structured code for a problem you don't actually understand, and if you never build the working model, you have no way to know whether that code is right — you're just forwarding it to review and hoping. The competence threshold isn't gatekeeping. It's the minimum understanding required to supervise at all.

## Anti-patterns that cap your ceiling

The habits above have mirror images, and if the multiplier is real then so is the anti-multiplier — ways of working that guarantee you stay on the flat, low part of the curve no matter how good the model gets.

**Prompt-and-pray.** Fire a one-line ask into the void, walk away, come back hoping. This is the vague-framing failure from Figure 3, industrialized. You are outsourcing the 70% that was yours to do.

**Over-delegating the decisions.** It's fine — desirable — to delegate *execution*. It is not fine to delegate the *choices*: which approach, which trade-off, what "correct" means. Those are the judgment calls your domain knowledge exists to make, and the agent has no basis for making them. When you let the agent decide the architecture because you weren't sure, you've handed the one thing you were uniquely there to provide to the one participant least equipped to provide it.

**Accept-without-reading.** Merging a diff you have not read because the tests are green and it looks plausible. Tests you didn't examine passing on code you didn't read is not evidence of anything. This one is dangerous enough that Part 3 is largely built around stopping it.

**Abandon-on-first-obstacle.** The inverse of "recover from obstacles." The first failing test is not the model telling you it can't; it's the model telling you something you didn't know. Quitting there leaves most of the value on the table.

**Mistaking motion for progress.** A long transcript, a big diff, lots of files touched — none of it is success. Output volume is the single most seductive vanity signal in agentic work. The agent that wrote 400 lines that don't pass your acceptance test did negative work; the one that changed 6 lines that do is the win.

What unites all five is that each one is a way of *withholding the judgment that was yours to supply* — either up front (prompt-and-pray, over-delegating), or after the fact (accept-without-reading, abandoning early), or by fooling yourself about whether anything was accomplished (motion for progress). That's why they cap your ceiling regardless of the model: a better agent executes faster, but if you're not framing, reading, steering, and confirming, faster execution of an unsupervised task just gets you to the wrong answer sooner. The anti-patterns aren't beginner mistakes you outgrow once; they're the default gravity of working with something that will do whatever you say without complaint. Staying off the flat part of the curve is an active, per-task choice, and the engineers who make it consistently are the ones who treat every session as *their* problem to solve with help — not a problem they've handed off.

## A mental model: the tireless junior who never pushes back

Here is the model I keep in my head, and it explains both the promise and the hazard in one picture. **A coding agent is a very fast, tireless, encyclopedically-read junior engineer who never gets bored, never sulks — and never pushes back.**

The speed and tirelessness are the promise: it will do the tedious thing, the tenth time, at 2am, without complaint. But "never pushes back" is the hazard, and it's easy to miss because it feels like a feature. A good human junior, handed a vague or wrong-headed task, will frown and ask "wait, why are we doing it this way?" — and that friction is a *safety mechanism*. The agent has no such friction. Hand it a bad plan and it will execute the bad plan beautifully, quickly, and completely. All of the judgment that a human colleague would have contributed by resisting, you must now supply yourself, up front, because nothing downstream will supply it for you.

That is why the whole thing lives or dies on the supervision loop in Figure 4. You **Frame** the task with enough constraint to be answerable. The agent **Executes**. Then — and this is the part that separates the two engineers from the opening — you **Inspect** the actual diff, **Correct** the course when it's off, and **Confirm** against your acceptance test before you call it done. Repeat until verified.

<figure class="blog-anim">
<svg viewBox="0 0 900 340" role="img" aria-label="The supervision loop cycles through Frame, Execute, Inspect, Correct, Confirm; a highlight sweeps each stage in turn and loops back, with Inspect and Correct marked as the steps novices skip" style="width:100%;height:auto;max-width:860px">
<title>Supervision loop: Frame, Execute, Inspect, Correct, Confirm — repeating until confirmed</title>
<style>
.em-stage{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.em-title{font:700 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.em-sub{font:400 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.em-note{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.em-loop{fill:none;stroke:var(--text-secondary,#6b7280);stroke-width:2}
.em-mark{fill:none;stroke:var(--accent,#6366f1);stroke-width:2.5}
.em-sweep{fill:var(--accent,#6366f1);opacity:.18}
@keyframes em-move{0%,16%{transform:translateX(0)}20%,36%{transform:translateX(170px)}40%,56%{transform:translateX(340px)}60%,76%{transform:translateX(510px)}80%,100%{transform:translateX(680px)}}
.em-anim{animation:em-move 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.em-anim{animation:none;transform:translateX(340px)}}
</style>
<defs>
<marker id="em-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" fill="var(--text-secondary,#6b7280)"/>
</marker>
</defs>
<path class="em-loop" d="M785,110 C785,34 105,34 105,110" marker-end="url(#em-arrow)"/>
<text class="em-note" x="445" y="26">repeat until confirmed</text>
<rect class="em-stage" x="30"  y="110" width="150" height="90" rx="10"/>
<rect class="em-stage" x="200" y="110" width="150" height="90" rx="10"/>
<rect class="em-stage" x="370" y="110" width="150" height="90" rx="10"/>
<rect class="em-stage" x="540" y="110" width="150" height="90" rx="10"/>
<rect class="em-stage" x="710" y="110" width="150" height="90" rx="10"/>
<rect class="em-sweep em-anim" x="30" y="110" width="150" height="90" rx="10"/>
<rect class="em-mark" x="370" y="110" width="150" height="90" rx="10"/>
<rect class="em-mark" x="540" y="110" width="150" height="90" rx="10"/>
<text class="em-title" x="105" y="150">Frame</text>
<text class="em-sub"   x="105" y="176">constrain it</text>
<text class="em-title" x="275" y="150">Execute</text>
<text class="em-sub"   x="275" y="176">agent edits</text>
<text class="em-title" x="445" y="150">Inspect</text>
<text class="em-sub"   x="445" y="176">read the diff</text>
<text class="em-title" x="615" y="150">Correct</text>
<text class="em-sub"   x="615" y="176">steer it</text>
<text class="em-title" x="785" y="150">Confirm</text>
<text class="em-sub"   x="785" y="176">prove it</text>
<text class="em-note" x="445" y="250">Inspect + Correct are the steps novices skip — where speed stops becoming leverage.</text>
</svg>
<figcaption>The supervision loop: frame the task, let the agent execute, then inspect, correct, and confirm — repeating until the work is verified. Skipping Inspect and Correct is exactly where novices lose the gains.</figcaption>
</figure>

The two engineers from the opening ran the *same* first two stages. What separated them was that one ran the last three and the other didn't. Inspect, correct, confirm — highlighted in the figure — are the steps novices skip, and skipping them is the exact point where raw speed stops converting into leverage and starts converting into confidently-wrong diffs.

## "But won't better models make this obsolete?"

It's the natural objection, and it's worth taking seriously: if the whole argument is "the model doesn't know your system," won't a smarter model eventually just *figure it out*? Partly, yes — future models will infer more from less, ask better clarifying questions, and need less hand-holding on the mechanical parts. The novice's floor will rise. But the ceiling is set by something a model cannot manufacture: **information that exists only in your head, your team's decisions, and your product's constraints.** No model, however capable, can know that your finance team needs the old rounding behavior preserved for an audit, that a "temporary" flag from 2023 is load-bearing, or that the metric your PM actually cares about is retention and not the click-through the code seems to optimize. That knowledge doesn't live in the training data or the repository; it lives in the organization. Framing is the act of transferring it into the task, and no upgrade removes the need to transfer it.

There's a second reason the multiplier persists. As models get more capable, we hand them *larger and more ambiguous* tasks — the frontier moves from "write this function" to "implement this feature" to "redesign this subsystem." Ambiguity grows faster than capability closes it, so the premium on the human who can decompose the ambiguous into the precise doesn't shrink; it moves up a level. The specific prompts in this post will date. The underlying skill — understand the problem, frame it so there's one right answer, verify against what you actually meant — is the durable part, and it's the part worth compounding.

## Where this series goes next

The uncomfortable, liberating conclusion is that getting more out of AI is mostly about getting better at the parts of engineering that were always the hard parts: understanding the problem, framing it precisely, knowing where the edge cases hide, and refusing to call something done until you've confirmed it. The model will keep improving, and it will keep getting easier to produce something that looks finished. Your judgment is still the multiplier, and it's the one part of this whole equation that you actually own.

Everything in this post is stated as if it's a solo skill — one engineer, one session, the context living in one person's head. And for one person it works. The trouble is it doesn't scale: the second engineer joins, then the fifth, and now the framing, the constraints, the do-nots, the acceptance conventions all live in scattered heads and nowhere the agent can read them. A team's agents are only as good as the shared context the team gives them, and a person cannot carry a team's context in their head.

So the next move is to get that context *out* of your head and *into the repository*, where every teammate's agent starts from the same map. That's [Part 2: Make Your Repo AI-Ready](/blog/software-development/ai-assisted-development/make-your-repo-ai-ready) — the concrete `CLAUDE.md`, permissions, commands, and guardrails that turn one person's good judgment into a team's default behavior. And once a whole team is moving fast, the discipline that keeps it safe — verification, review, and honest measurement — is [Part 3: Running AI in a Team](/blog/software-development/ai-assisted-development/running-ai-in-a-team).
