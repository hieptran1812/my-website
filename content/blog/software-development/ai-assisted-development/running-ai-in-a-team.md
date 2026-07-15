---
title: "Running AI in a Team: Verification, Review, and Rollout"
date: "2026-07-14"
publishDate: "2026-07-14"
description: "AI makes it cheap to produce code — and just as cheap to produce plausible-but-wrong code, so a team's real discipline moves to verifying it: how to review AI diffs, verify by behavior, onboard people, dodge the pitfalls, and roll it out without fooling yourself."
tags:
  [
    "ai-assisted-development",
    "claude-code",
    "code-review",
    "verification",
    "team-workflow",
    "security",
    "prompt-injection",
    "developer-productivity",
    "adoption",
    "best-practices",
  ]
category: "software-development"
subcategory: "AI-Assisted Development"
author: "Hiep Tran"
featured: true
readTime: 19
image: "/imgs/blogs/running-ai-in-a-team-1.webp"
---

[Part 1](/blog/software-development/ai-assisted-development/the-expertise-multiplier) gave you the mindset — domain judgment is the multiplier. [Part 2](/blog/software-development/ai-assisted-development/make-your-repo-ai-ready) put that judgment into the repository so a whole team inherits it. Both were, in a sense, about making the team *faster*. This final part is about the thing that goes wrong precisely when you succeed at that.

Here is the trap, and it's worth stating plainly because it catches good teams: **AI makes it cheap to produce code, and exactly as cheap to produce code that is confidently, plausibly wrong.** The agent that writes a correct fix in ninety seconds writes an incorrect one in ninety seconds too, and the incorrect one compiles, passes the tests it happened to touch, reads cleanly, and is wrong in a way you can only catch by looking. When production stops being the bottleneck, the bottleneck moves — to reading, running, and confirming. Figure 1 is that shift, and it is the whole subject of this post.

<figure class="blog-anim">
<svg viewBox="0 0 900 300" role="img" aria-label="Two horizontal bars for writing code and for reviewing and verifying; as the animation runs, the writing bar shrinks and the reviewing-and-verifying bar grows to fill the space, showing the bottleneck shifting from writing to verification once AI makes writing cheap" style="width:100%;height:auto;max-width:860px">
<title>The bottleneck shifts from writing code to reviewing and verifying it</title>
<style>
.rt-title{font:700 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.rt-lab{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:end}
.rt-note{font:600 12.5px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.rt-tag{font:600 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.rt-write{fill:#a5d8ff;stroke:#1971c2;stroke-width:1.5}
.rt-review{fill:#ffec99;stroke:#f08c00;stroke-width:1.5}
@keyframes rt-shrink{0%,16%{transform:scaleX(1)}84%,100%{transform:scaleX(0.4)}}
@keyframes rt-grow{0%,16%{transform:scaleX(0.36)}84%,100%{transform:scaleX(1)}}
.rt-shrink{transform-box:fill-box;transform-origin:left center;animation:rt-shrink 6s ease-in-out infinite}
.rt-grow{transform-box:fill-box;transform-origin:left center;animation:rt-grow 6s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.rt-shrink{animation:none;transform:scaleX(0.4)}.rt-grow{animation:none;transform:scaleX(1)}}
</style>
<text class="rt-title" x="450" y="30">As writing gets cheap, verifying becomes the bottleneck</text>
<text class="rt-tag" x="230" y="66">before AI</text>
<text class="rt-tag" x="700" y="66">after AI</text>
<text class="rt-lab" x="192" y="118">writing code</text>
<rect class="rt-write rt-shrink" x="210" y="96" width="620" height="46" rx="7"/>
<text class="rt-lab" x="192" y="176">reviewing</text>
<text class="rt-lab" x="192" y="194">&amp; verifying</text>
<rect class="rt-review rt-grow" x="210" y="162" width="620" height="46" rx="7"/>
<text class="rt-note" x="450" y="256">The same effort now produces far more code — so the scarce work is no longer writing it, but reading, running, and confirming it.</text>
</svg>
<figcaption>Before AI, writing was the wide bar and review a thin one. AI compresses the writing; the reviewing-and-verifying bar grows to fill the space. The team's bottleneck — and therefore its discipline — moves from production to verification.</figcaption>
</figure>

This is Part 3 of three, and it's about the discipline that keeps a fast team from shipping fast mistakes: how to verify, how to review AI-generated diffs, how to bring people up to speed, the pitfalls that show up only at team scale, and how to measure whether any of it is working — including how to roll the whole thing out without fooling yourself.

## Verification discipline: the non-negotiable

Start with the one rule that everything else hangs from: **never merge AI-written code you haven't verified.** Not "haven't glanced at" — verified. And verification does not mean the tests are green. Green tests on code you didn't read prove that the code passes the tests that already existed, which is a much weaker claim than "this does what I wanted," especially when the agent may have written those tests too, in the same session, with the same misunderstanding baked in.

Real verification is behavioral. You *run the thing* and watch what it actually does. Figure 2 is the loop: the agent generates, you drive the real flow, you observe the actual behavior, you compare it against what you *intended* — and only then do you accept it or send it back to be reframed. The comparison against intent is the crux. A change can be internally consistent, well-tested, and elegantly written, and still solve a subtly different problem than the one you had; the only way to catch that is to hold the behavior up against what you actually meant, which is knowledge that lives in your head, not in the diff.

![Branching flow diagram: generate the diff, then run and observe the real flow, then compare the behavior to intent, which forks to accept and merge if it matches or reject and reframe if it is off](/imgs/blogs/running-ai-in-a-team-1.webp)

This is not a new skill — it's the oldest debugging discipline there is, applied to a new source of code. You already know that you [reproduce a bug before you believe you've fixed it](/blog/software-development/debugging/reproduce-it-first-or-youre-not-debugging); verifying an AI change is the same move in the other direction — reproduce the *desired* behavior before you believe the code delivers it. On a team, make this a norm with teeth: the definition of done for an AI-assisted change is "a human ran it and confirmed the behavior against intent," and that human is accountable for it regardless of who — or what — wrote the code. The author of a diff is whoever puts their name on the PR, not the tool that generated the characters.

The *amount* of verification should scale with the blast radius of the change, not be uniform. A copy tweak or a log-line addition needs a glance and a build. A change to how money is calculated, how permissions are checked, or how data is migrated needs you to actually exercise the path with realistic inputs and watch the outputs — ideally the ugly inputs, the ones your domain knowledge tells you are load-bearing. What you must never do is let the *ease* of generating the change set the depth of verification: the whole hazard of a fast tool is that it makes a high-stakes change feel as cheap as a trivial one, and the calibration that used to come for free from "this was hard to write, so I'll check it carefully" is gone. You have to supply that calibration deliberately now, because the code no longer tells you how much it cost to produce.

## A different review lens for AI diffs

Human-written code and AI-written code fail differently, so reviewing them well requires looking for different things. When a tired human writes a bug, it usually *looks* like a mistake — a typo, an off-by-one, a forgotten case, something a reviewer's eye snags on. When an agent writes a bug, it tends to look *correct*, because fluent, idiomatic, confident-looking code is exactly what a language model is best at producing. The tell is gone. So the reviewer has to hunt deliberately for the specific ways AI diffs go wrong, shown in Figure 3.

![Tree diagram of four failure modes to hunt in an AI-generated diff branching from a root that reads read the diff for: plausible-but-wrong, silent scope creep, hallucinated API, and dropped edge case, each with a one-line tell](/imgs/blogs/running-ai-in-a-team-2.webp)

- **Plausible-but-wrong.** The code reads perfectly and is simply incorrect on real data — it handles the shape of the problem it assumed, not the one you actually have. This is the dangerous one, because nothing about the diff signals trouble. A concrete shape of it: you ask for a function that dedupes a list of orders, and the agent writes a clean, well-named, fully-tested implementation that dedupes by object identity — correct-looking, green tests, and useless, because your orders arrive as fresh objects and need deduping by `order_id`. The code is not buggy in any way a linter or a type-checker sees; it's answering a question adjacent to the one you asked. Only running it against real inputs (Figure 2) exposes it, which is exactly why "the tests pass" is not verification.
- **Silent scope creep.** You asked for a one-function fix and the diff touches nine files, "improving" things you didn't ask about — renaming, refactoring, "cleaning up" — each change plausible, the aggregate a review nightmare that hides the one line that matters. Scope discipline is a review criterion now: a change should do what was asked and nothing else.
- **Hallucinated API.** A method, flag, or config key that doesn't exist, or that existed two major versions ago, called with total confidence. Compilers and type-checkers catch some of these; the ones aimed at dynamically-typed or loosely-configured surfaces slip through to runtime.
- **Dropped edge case.** The happy path is immaculate and the empty list, the null, the concurrent write, the timezone boundary are silently unhandled. This is where your domain knowledge from Part 1 earns its keep: you know which edge cases exist because you have the scars, and the agent doesn't.

Reviewing an AI diff, in other words, is less like proofreading and more like [investigating](/blog/software-development/debugging/hypothesize-and-falsify-not-stare-and-hope): you approach it assuming a specific set of failures might be present and you go looking for them, rather than waiting for one to jump out. And a hard corollary: **keep AI diffs small.** A five-hundred-line AI-generated PR is not a productivity win; it's an un-reviewable object that no human will actually read, which means it ships unverified. Small diffs aren't a nicety here — they're what makes the verification loop physically possible.

## AI reviewing AI: where it helps, where it can't

If review is the bottleneck, can't the AI review its own output? Partly — and it's genuinely useful as a *first pass*. A code-review subagent (the kind we set up in Part 2) will reliably catch a category of mechanical issues: obvious bugs, style violations, missing null checks, a hallucinated method the type-checker missed. Running that pass before a human ever looks strips out the noise, so the human reviewer spends their attention on what's left.

But be precise about the limit. An AI reviewer is strong on *local correctness* and weak on *intent* — it can tell you the function is internally sound, but it cannot tell you the function solves the right problem, because "the right problem" is the thing that only lived in your head. It also shares blind spots with the author: the same model that confidently wrote a plausible-but-wrong change may confidently bless it, for the same reason. And on the two things that matter most at team scale — security and architecture — a human must stay in the loop, because those are judgment calls about *your* threat model and *your* system's future, not properties you can read off a diff. Use AI review to clear the underbrush; keep a human on intent, security, and design.

There's also a failure mode specific to AI review that's worth naming: **it can manufacture false confidence.** A thumbs-up from an automated reviewer *feels* like verification — a second opinion, a check passed — and it is very tempting to let it substitute for the human pass rather than precede it. But an AI reviewer that missed the same edge case the author missed, then approved the diff, has not made the change safer; it has added a green checkmark to an unverified change, which is worse than no checkmark at all, because now a human is even less likely to look hard. Treat the AI review the way you treat CI: a filter that removes obvious problems and shrinks the human's workload, never a signal that the human step can be skipped. The value is in what it lets a person *not* waste attention on, not in the approval itself.

## The PR workflow with AI

The pull request is where a team's verification discipline becomes institutional, so it's worth adapting for AI-assisted work. Three changes matter. First, **small diffs**, for the reason above — one reviewable change per PR. Second, **intent in the PR body**: state what the change is for and, crucially, *how it was verified* — the exact flow that was exercised and what was observed — so the reviewer inherits your verification instead of re-deriving it. Third, **provenance**: mark AI-assisted work as such. Not as a scarlet letter, but because a reviewer who knows a diff was largely agent-generated will correctly apply the sharper lens from the section above.

A PR template makes this the path of least resistance:

```md
## What & why
<one or two sentences: the change and the reason>

## How I verified
- Ran: `<the actual command / flow>`
- Observed: <the behavior you confirmed, vs. what you intended>

## Notes
- [ ] AI-assisted (agent generated part or all of this diff)
- [ ] Diff is scoped to the stated change — no drive-by refactors
- [ ] Edge cases considered: <empty / null / concurrent / …>
```

And CI is your backstop, not your verifier — the same distinction as before. Green CI means the automated checks passed; it does not mean the change is correct, and a team that lets "CI is green" stand in for "a human confirmed the behavior" has quietly deleted the most important step. CI catches regressions in things you already thought to test. It cannot catch the plausible-but-wrong change that satisfies every existing test while doing the wrong thing.

## Onboarding: level people up fast, because that's the ROI

Part 1's data had a striking implication that most teams miss. The gain from AI is *front-loaded* — the steep part of the curve is getting someone from novice to intermediate, and after that the returns flatten. Which means the single highest-return investment your team can make is **getting people over the novice hump quickly.** Not squeezing more out of your best engineer, but pulling everyone else up to competent. That's where the verified-success rate roughly doubles.

Concretely, onboarding into an AI-assisted team is less about teaching prompt syntax and more about transferring the habits from this series:

- **Pair on real work.** Sit a newcomer with someone who frames precisely, inspects the diff, and confirms behavior — the supervision loop from Part 1 is caught, not taught. Watching someone catch a plausible-but-wrong change teaches more than any doc.
- **Share the prompts that work.** A team accumulates a folklore of good framings — how we ask for a migration, how we scope a refactor. Put them in slash commands (Part 2) so they're not folklore but tools.
- **Use `CLAUDE.md` as the onboarding doc.** The file you wrote to orient the *agent* is, not coincidentally, an excellent orientation for a *human*: build commands, architecture, conventions, the do-nots. A new hire and their agent can read the same page on day one.
- **Treat review as teaching.** When you catch scope creep or a dropped edge case in a newcomer's AI-assisted PR, the review comment isn't just fixing that diff — it's installing the lens. Reviewing is the highest-bandwidth channel you have for leveling people up.

The payoff compounds: every engineer you move from novice to intermediate roughly doubles their verified output *and* becomes another person who can supervise safely, which is the only thing that lets the team move fast without the wheels coming off.

## Pitfalls at team scale

Some failure modes only appear once a team is moving fast and trusting the tool. Name them so you can watch for them.

- **Automation complacency.** The most insidious one. When the agent is right ninety times in a row, the ninety-first diff gets a rubber stamp — and that's the one with the subtle bug. Vigilance decays exactly as the tool earns trust, which is precisely when a wrong change is most likely to sail through. The defense is structural, not willpower: keep diffs small and keep the "a human ran it" gate non-optional, so review stays possible even when attention flags.
- **Context rot.** Long agent sessions accumulate stale assumptions, half-abandoned approaches, and contradictory instructions until the agent is reasoning over a polluted context and quality quietly degrades. The fix is hygiene: start fresh sessions for fresh tasks, and push exploration into subagents (Part 2) so the main context stays clean.
- **Config drift.** The `.claude/` setup from Part 2 is only a shared standard if it stays shared. Left ungoverned, everyone accretes personal tweaks, `deny` rules get loosened "just for this," and the fleet diverges. Review changes to `.claude/` in PRs like the load-bearing infrastructure they are.
- **Security: injection and leakage.** Everything an agent reads is untrusted input — the prompt-injection surface from Part 2 is a team problem, because now it's *your teammates'* agents connected to *your* issue tracker and database. A malicious issue or a poisoned web page can try to steer an agent into reading secrets or exfiltrating data; the `deny` rules and least-privilege MCP scopes from Part 2 are what contain it. Treat external content as hostile until proven otherwise.
- **Diffusion of responsibility.** "The AI wrote it" is not a defense, and the moment it becomes an accepted one, quality collapses — because now nobody owns the diff. Re-anchor the norm continuously: the human on the PR owns the change, full stop, exactly as if they'd typed every character.

Notice that these pitfalls share a root. Each one is a place where the team quietly *stops supplying the human judgment* that the whole system depends on — complacency stops the inspecting, context rot corrupts the framing, config drift erodes the shared rails, injection exploits misplaced trust, diffusion erases ownership. They are the team-scale versions of Part 1's individual anti-patterns, and they don't announce themselves; they creep in during the good stretches, when the tool has been reliable and vigilance feels like overhead. That's the cruel timing of it: the conditions that make a team *feel* safe with AI — a long run of correct output, a smooth pipeline, high trust — are exactly the conditions under which the next wrong change is most likely to slip through unexamined. The teams that stay safe treat the discipline as load-bearing precisely *because* things are going well, not as training wheels to remove once they're comfortable.

## Measuring adoption honestly

If you're going to invest in this, you'll want to know whether it's working — and this is where teams most often lie to themselves with the wrong numbers. **Lines of code written** and **percentage of code that is AI-authored** are worse than useless: they reward volume, which is the exact anti-signal from Part 1, and a team optimizing them will cheerfully generate more code to review and more surface to be wrong. Measure outcomes, not output:

- **Verified task success rate** — how often an AI-assisted task reaches a confirmed-correct outcome. This is the number Part 1's research tracked, and it's the one that matters.
- **Review rework rate** — how often AI diffs bounce back for correctness (not style). Rising rework is an early warning that verification is slipping or diffs are getting too big.
- **Cycle time** — from task start to merged-and-verified. This captures the *real* speedup, including the review it created, rather than the illusory speedup of raw generation.
- **Escaped defects** — bugs that reach production in AI-assisted changes. The ground truth. If this is climbing, your verification discipline is failing no matter how fast everything feels.

There's a subtlety worth flagging: some of these are lagging indicators. Escaped defects tell you the truth, but they tell you late — by the time production bugs climb, the erosion happened weeks ago. So pair a lagging measure (escaped defects) with a leading one (review rework rate, or how often a diff needs a second verification pass), because the leading indicator moves *first* and gives you a chance to correct before the lagging one does the telling. And resist the urge to turn any of these into an individual scorecard — the moment "verified success rate" becomes a number someone is graded on, it stops measuring reality and starts measuring how well people game it, and you're back to the volume problem wearing a different mask. Measure the team and the trend, to steer, not to rank.

The through-line: measure the thing you actually want (correct software, shipped sustainably), not the thing that's easy to count (characters produced). A metric that rewards volume will get you volume.

## Rollout: crawl, walk, run

You don't turn all of this on by decree on a Monday. Adoption that sticks grows in stages, shown in Figure 4.

![Ascending three-layer stack of adoption stages: crawl with one champion who proves it and shares what works, walk with team norms and a committed .claude directory plus review and verify discipline, and run as an org standard with shared guardrails, hooks, and honest metrics](/imgs/blogs/running-ai-in-a-team-3.webp)

**Crawl — one champion.** A single engineer who gets genuinely good at the supervision loop, proves value on real work, and — critically — writes down what worked. No mandates, no rollout deck. Just an existence proof and a growing set of good framings.

**Walk — team norms.** Now codify it: the committed `.claude/` from Part 2 (shared `CLAUDE.md`, permissions, commands, guardrails), plus the review-and-verify discipline from this post baked into the PR process. This is where individual skill becomes team default, and where the shared config starts paying for itself across everyone.

**Run — org standard.** Shared guardrails and hooks enforced across teams, security scopes governed centrally, and honest metrics tracked so the org can see what's actually working and adjust. The guardrails are deterministic (hooks, `deny` rules), the metrics are outcome-based, and the whole thing is reviewable infrastructure.

The mistake at every stage is skipping to the next one — mandating org-wide adoption before anyone has proven the loop, or standardizing config nobody has battle-tested. Let each stage earn the next.

Two things reliably stall a rollout, and both are worth pre-empting. The first is a **bad first impression at scale**: push the tool org-wide before the norms exist, a wave of unreviewed AI diffs causes a visible incident, and now "AI coding" is politically radioactive regardless of its actual merits — you've spent your credibility proving the wrong thing. Staging exists precisely to contain that blast radius to one champion's desk until the discipline is real. The second is **treating the champion's success as automatically transferable.** What the champion actually built was judgment and a set of habits; if you roll out only the *tooling* — "here's the `.claude/` folder, go" — without the review-and-verify culture that made it safe in their hands, you've distributed the fast part without the safe part, which is how you get the incident above. The transferable artifact is the whole practice: config *plus* the supervision loop, the review lens, and the ownership norm. Ship the practice, not just the folder.

## The series in one line

Three parts, one argument. **Expertise** (Part 1) is the multiplier — domain judgment, precise framing, and the discipline to inspect and confirm are what turn a fast tool into real leverage. **Setup** (Part 2) moves that judgment out of one person's head and into the repository, so a whole team's agents inherit it consistently and safely. And **workflow** (Part 3) is the verification, review, and honest measurement that let a team go fast without going wrong.

None of the three works alone. Expertise without shared setup doesn't scale past one person. Setup without verification discipline just industrializes the production of plausible mistakes. And verification without expertise is someone staring at a diff with no basis to judge it. Put all three together and the effect compounds — not because the model is doing your job, but because you've built a system where a fast, tireless, never-pushing-back tool is pointed at the right problems, bounded by the right rails, and checked by people who know what correct looks like. That's the whole game.

And it's worth ending on the part that doesn't change no matter how good the tools get. Better models will raise the floor — the novice's mistakes get rarer, the setup gets easier, the first-pass review catches more. But the ceiling stays where it has always been: with the humans who understand the problem, frame it precisely, and refuse to call something done until they've seen it work. The leverage was never in the generation. It was, and remains, in the judgment you bring to it and the discipline you build around it. Get those right, as a person and as a team, and the tool becomes exactly what it should be — an amplifier for engineers who already know what they're doing.
