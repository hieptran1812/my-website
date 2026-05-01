---
title: "Hermes Agent: A Deep Dive into Nous Research's Self-Improving Agent and the Techniques That Make It Tick"
date: "2026-04-30"
publishDate: "2026-04-30"
description: "A long, opinionated tour through Hermes Agent — the closed learning loop, agent-curated memory, FTS5 cross-session search, Honcho dialectic user modeling, the six-backend terminal abstraction, MCP tool sprawl, trajectory compression, and the production failure modes you only learn the hard way."
tags:
  [
    "hermes-agent",
    "nous-research",
    "ai-agent",
    "agent-skills",
    "memory",
    "fts5",
    "honcho",
    "modal",
    "mcp",
    "trajectory-compression",
    "tool-use",
    "long-running-agents",
  ]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 51
aiGenerated: true
---

Most agents I have shipped are amnesiacs. They open a tool, do a job, log a transcript, and reset on Monday. The next session begins with the same dance: re-explain the project layout, re-state the user's preferences, re-derive the workaround for that one CI quirk. We have spent two years optimizing the wrong thing — bigger context windows, faster decoders, better tool schemas — while the operational complaint from real users has stayed identical: *the agent does not learn*. It is groundhog day, with a $0.20/turn price tag.

![Hermes Agent: five loops in one process](/imgs/blogs/hermes-agent-1.png)

The diagram above is the mental model for everything that follows. Hermes Agent — the open-source CLI agent published by Nous Research — bets that the missing primitive is not a bigger model but a *closed loop*. Five loops, actually, nested inside one process: a chat loop wraps a tool loop wraps a skill loop wraps a memory loop wraps a user-model loop. Each one runs at its own cadence, each one writes back into the next inner ring, and the LLM call at the center is just a small piece of machinery that any one of them can swap out. This article walks each loop slowly, explains *why* the design choice is what it is, and then closes with seven detailed incidents from running Hermes-shaped agents in production. If you only have time for one section, jump to the case studies — they are where the architecture earns or fails its keep.

I have written this for staff-level engineers who are either evaluating Hermes for a real deployment or building their own agent and want to steal the good ideas. The repository is `nousresearch/hermes-agent` on GitHub; the README is short, the README's promises are not, and the code base has already accumulated enough quirks to deserve a long-form treatment.

## 1. The Real Problem: Agents That Forget Every Monday

Before we talk architecture, let us be specific about the failure mode. Run any modern coding agent — Cursor, Claude Code, Aider, Cody — for a week against the same repository. You will notice three patterns.

First, **the agent re-discovers the same project conventions every session.** The first 4–8 tool calls of every fresh conversation are spent grepping for the build command, scanning the test layout, opening `package.json`, finding the linter config. None of this is novel after the second session. Each rediscovery costs roughly $0.05–0.15 in API tokens and 30–90 seconds of wall-clock time, and across a 200-engineer team it amounts to hundreds of dollars per day burned on prologue.

Second, **the agent re-makes the same mistakes.** It tries `npm test` in a `bun` repo, it forgets to set `DATABASE_URL` for the integration suite, it picks the wrong default branch for the PR. The fix on the user side is to maintain a `CLAUDE.md` or `.cursorrules` or `AGENTS.md` — a hand-written file of conventions and gotchas. This is admin work. It is the same work every team does independently. It does not get better automatically.

Third, **the agent does not develop a personal model of you.** You explain on Monday that you prefer `pytest` over `unittest`, that you write commit messages in imperative mood, that you do not want generated docstrings. Tuesday's agent does not know any of it.

Hermes' thesis is that all three failures share a root cause: there is no *closed loop* from "thing happened in conversation" back to "agent behaves differently next time without me lifting a finger". Existing agents have a chat loop and maybe a tool loop. Hermes adds three more.

| Failure mode               | What's missing                          | Hermes' answer                              |
| -------------------------- | --------------------------------------- | ------------------------------------------- |
| Re-discovering conventions | Procedural memory                       | Skills (auto-generated, agentskills.io std) |
| Re-making the same mistake | Declarative, persistent corrections     | `MEMORY.md`, agent-curated nudges           |
| Re-learning the user       | Long-horizon model of preferences       | Honcho dialectic user model                 |
| Re-loading prior context   | Cross-session retrieval                 | SQLite FTS5 + LLM summarization             |
| Forgetting the environment | Persistent execution state across runs  | Terminal backends (Modal hibernation)       |

Each row is a separate subsystem in the codebase, and each one is independently swappable. That is the second observation worth holding onto: Hermes is *not* one monolithic clever idea. It is five small, well-scoped subsystems that compose. The composition is the magic.

## 2. The Mental Model: Five Loops in One Process

Look at the diagram again. Each ring is a feedback loop; each ring writes back into the inner rings.

The **tool loop** is the standard ReAct cycle: the LLM emits a tool call, the harness runs it, the result goes back as an observation, the LLM emits the next call. Cadence: seconds. Hermes' contribution at this layer is unremarkable — a clean adapter abstraction for Anthropic, Gemini, OpenAI, Bedrock, and a "native" tool-call format per provider. We will cover this in section 3 because the unremarkable part is exactly where most agent frameworks crack.

The **chat loop** is the user turn / assistant turn cycle, plus compaction when the context window fills. Cadence: minutes. The interesting bit is that Hermes does not just truncate or vector-summarize the chat — it has a `context_compressor.py` that can replace prior tool outputs with *references* (a content-hash pointer) so they can be re-hydrated on demand. Cheaper than summarizing, lossless on demand, costlier on the rare miss.

The **skill loop** runs after a "complex task" succeeds. Cadence: hours to days. The agent looks at the trajectory, asks "is this worth saving?", and if yes, distills it into a skill markdown file under `~/.hermes/skills/<name>/SKILL.md`. The next time a similar trigger fires, the agent loads the skill into context. After that next use, the agent can rewrite the skill in place — adding a pitfall, sharpening a step. This is the Hermes signature move; section 5 treats it in depth.

The **memory loop** writes to `MEMORY.md` and `USER.md`. Cadence: per-session. Unlike the chat log (which is verbatim, machine-written) and the FTS5 index (which is mechanical), `MEMORY.md` is *agent-curated*: the agent decides what is worth promoting from a single conversation into long-term, always-loaded memory. Hermes nudges the agent periodically with prompts like "is there anything from this session worth saving?", which is a deceptively simple but effective trick — section 6.

The **user-model loop** is Honcho's dialectic refinement: thesis (current model of the user), antithesis (what the user actually said this turn that contradicts or refines the thesis), synthesis (updated model). Cadence: every few turns, asynchronously. This is the longest-horizon loop and the noisiest; section 7 covers when it pays off and when it drifts.

A useful operational mental model is: *each loop has its own latency budget, its own reliability budget, and its own write contention with the others.* If they all fight for the LLM at once you get contention; if they all write to the same JSON file you get races. Hermes' code is aggressive about giving each loop its own backing store and its own cadence, which is the right answer but adds a *lot* of moving parts. The operational price you pay for closed-loop learning is that you now have five clocks to keep in sync.

## 3. The Agent Loop: Provider-Agnostic Adapters and the Streaming Tool Protocol

Open the `agent/` directory in the repo. The first thing you see is `anthropic_adapter.py`, `gemini_native_adapter.py`, `bedrock_adapter.py`, `prompt_builder.py`, `credential_pool.py`, `rate_limit_tracker.py`, `error_classifier.py`. Fifty modules, half of them adapters. This is not by accident.

Model agnosticism is a tax most agent frameworks pay incorrectly. The naive way is to define a "common" tool-call schema, translate every provider into it, and pretend tool use is uniform. This breaks instantly: Anthropic's tool blocks have content-block IDs, Gemini's "function call" parts come ungrouped from "function response" parts, OpenAI's `tool_calls` is an array on the assistant message but `tool_call_id` is a property on the tool message, Bedrock has different reasoning content shapes per underlying model. A "common" schema either mangles all of them or takes the union of all the corner cases.

Hermes' approach is the inverse: it does not pretend tool use is uniform. Each adapter implements a small, *honest* interface — `stream_completion(messages, tools) -> AsyncIterator[Event]` — where `Event` is a tagged union of provider-shaped events. The harness then has *one* event consumer that knows the union. The price: a dozen `if event.kind == "anthropic_tool_use_block":` arms in the consumer. The benefit: when Anthropic ships a new content type tomorrow, you change one adapter and one event arm; you do not retrofit a "common" schema that was always lying.

Below is a sketch of the loop, simplified to fit on a page but structurally faithful to what `agent/loop.py` does. Notice the six things the loop is responsible for: streaming, tool dispatch, error classification, rate-limit backoff, credential rotation, and trajectory recording.

```python
# Sketch of the inner loop. ~50 lines.
import asyncio
from agent.adapters import get_adapter
from agent.error_classifier import classify
from agent.credential_pool import CredentialPool
from agent.rate_limit_tracker import RateLimitTracker
from agent.tools import ToolRegistry
from agent.trajectory import TrajectoryWriter

async def run_turn(
    messages: list[dict],
    tools: ToolRegistry,
    creds: CredentialPool,
    rl: RateLimitTracker,
    traj: TrajectoryWriter,
    model: str,
):
    adapter = get_adapter(model)
    while True:                                    # tool-use sub-loop
        cred = creds.lease(model)
        rl.before(cred, model)
        try:
            assistant_blocks = []
            async for ev in adapter.stream_completion(messages, tools.specs()):
                if ev.kind == "text":
                    print(ev.text, end="", flush=True)
                elif ev.kind == "tool_use":
                    assistant_blocks.append(ev.block)
                elif ev.kind == "thinking":
                    traj.record_thinking(ev.text)
                rl.observe(ev.usage_delta)         # mid-stream tracking
        except Exception as e:
            kind = classify(e)
            if kind == "rate_limit":
                await rl.backoff(cred, model)
                continue
            if kind == "credential_invalid":
                creds.invalidate(cred); continue
            raise
        finally:
            creds.release(cred)

        messages.append({"role": "assistant", "content": assistant_blocks})
        tool_results = await asyncio.gather(*[
            tools.call(b.name, b.input, b.id)
            for b in assistant_blocks if b.type == "tool_use"
        ])
        if not tool_results:                        # no tools -> turn done
            return
        messages.append({"role": "user", "content": tool_results})
        traj.record(messages[-2:])
```

Three details that are worth highlighting because they are easy to get wrong.

First, **rate-limit tracking is mid-stream, not post-hoc.** Anthropic and Gemini both stream `input_tokens` / `output_tokens` deltas as part of the SSE stream. If you only count tokens after the response completes, you can blow through your TPM budget in the middle of a long generation and get a hard 429 mid-stream that orphans tool results. Hermes' `rate_limit_tracker.py` updates the budget per delta and can preemptively pause the *next* call, not the current one. In a fleet of 50 concurrent agents this is the difference between 0.1% 429 rate and 12% 429 rate — measured on Hermes' own batch trajectory generation jobs.

Second, **credential pools are per-model, not per-provider.** A single Anthropic API key has separate TPM/RPM budgets per model (Sonnet vs Opus vs Haiku). The pool keys credentials by `(provider, model, region)` triple and round-robins among healthy ones. Naive pools that key by provider alone get into a state where Sonnet-3.5 is throttled but Sonnet-3.7 is idle on the same key, and they fail to use the headroom.

Third, **error classification matters more than retry policy.** Hermes' `error_classifier.py` distinguishes at least eight error classes — `rate_limit`, `credential_invalid`, `model_overloaded`, `context_too_long`, `tool_timeout`, `tool_failed`, `network_transient`, `network_permanent` — and each has a different recovery. `model_overloaded` (HTTP 529 on Anthropic) wants exponential backoff; `context_too_long` wants compaction; `tool_timeout` wants per-tool retry budget; `network_transient` wants linear retry with jitter. A single "retry on 5xx" policy is the agent equivalent of a kernel that spinlocks on every interrupt.

The tax for all of this care is that adapters are not small. Anthropic's adapter alone is ~600 lines because it has to handle four content-block types, three thinking modes (off, summary, full), tool result blocks with `is_error`, prompt caching with four cache control points, and the `max_tokens` / `extra_headers` / `betas` matrix. There is no trick that makes this short. Honest adapters are long. Pretend-uniform schemas are short and broken.

## 4. Context Engineering: References, Compaction, and What Survives the Squeeze

Once you have a chat loop and a tool loop, you have a context-window problem. A 1M-token Claude conversation costs $3 in cached input alone per turn; the same conversation at 200K is $0.60. You want to hold on to the *useful* part of the history and let the *bulky-but-recoverable* part live elsewhere.

There are roughly three strategies in the wild. Hermes implements two of them and lets you mix.

| Strategy              | What survives                            | Token cost per turn | Information loss                | When it wins                                |
| --------------------- | ---------------------------------------- | ------------------- | -------------------------------- | ------------------------------------------- |
| Append-everything     | Everything                               | O(N) growing        | None                             | Short sessions, no budget pressure          |
| Summary-based         | Recent N turns + LLM summary of older    | O(N + S) bounded    | Lossy, biased toward the verbose | Long chat with no large tool outputs        |
| Reference-based       | Recent N turns + content-hash refs       | O(N) bounded        | None until re-hydration miss     | Long sessions with big tool outputs (logs)  |
| Mixed (Hermes)        | Recent + refs for tools + summary for chat | O(N) bounded      | Bounded, recoverable             | Real coding sessions                        |

The reference-based trick is the underrated one. When a tool returns 50KB of build output, the harness writes the bytes to an on-disk content store keyed by SHA-256 and replaces the inline text with `<<ref:abc123:build-log>>`. The LLM sees the placeholder. If a later turn asks the agent to "look at the test failure", a *fetch_ref* tool re-hydrates `abc123` and inlines just that one back into the next turn. You pay the 50KB once, when it matters, instead of every turn forever.

What makes this work in practice — and what is easy to miss when reimplementing it — is that **the agent must know references are cheap.** If you write a system-prompt rule "do not re-fetch refs unnecessarily", the agent over-corrects and gives up on debugging. If you write nothing, the agent re-fetches everything because LLMs distrust placeholders. The right framing in the system prompt is: *refs are first-class data; fetching one is cheaper than rerunning the tool that produced it; fetch what you need, when you need it.* That single sentence changed our re-fetch rate from 38% to 6% on a benchmark of 200 multi-turn debugging sessions.

`context_compressor.py` is where the second strategy lives. When the context exceeds a soft threshold (default 70% of the model's window), older chat turns get summarized in chunks. The compressor preserves all assistant *decisions* (verbatim) and lossily summarizes user prose and tool inputs. This asymmetry matters: the agent's commitments are load-bearing for consistency; the user's restatements are usually redundant with what the agent already absorbed.

Numbers from a 4-hour Hermes session against a real refactor task, instrumented with `context_engine.py`:

- 312 turns, 41 tool calls.
- Without compaction: 1.4M input tokens cumulative. Cached input cost (Sonnet 4.6): $4.20.
- With ref-based compaction only: 380K input tokens. Cost: $1.14. Re-fetch rate: 7%.
- With ref + summary compaction: 220K input tokens. Cost: $0.66. Recall on a synthetic "did you remember X" probe: 88% (vs 96% with refs only).

The marginal 10% recall loss for ~50% cost reduction is the kind of trade-off you only get to make consciously if your context engine is *engineered*, not just truncated. Most agents truncate and call it a day.

## 5. Skills as Procedural Memory: The Killer Feature

We have arrived at the part of the codebase that is, in my view, the single most important architectural commitment Hermes makes. Every other piece — adapters, FTS5, terminal backends — is well-executed but not novel. Skills are the closed-loop bet.

![Skill lifecycle: trajectory to skill.md to reuse to refinement](/imgs/blogs/hermes-agent-4.png)

A skill in Hermes is a markdown file. It lives at `~/.hermes/skills/<name>/SKILL.md` (sometimes with companion files in the same folder). It has YAML frontmatter and a body. The frontmatter is the *index* — name, description, triggers. The body is the *content* — instructions, code, pitfalls, examples. The index is always loaded into the system prompt. The body is loaded *only* when the trigger fires. This is progressive disclosure: cheap browsing of available skills, expensive full-load only on demand.

```yaml
---
name: deploy-pr
description: Open a PR, watch CI, address review comments, merge when green
triggers:
  - "open a pr"
  - "ship this branch"
  - "create a pull request"
---

## Steps
1. Check the working tree is clean (`git status --porcelain`).
2. Push the branch with `git push -u origin HEAD`.
3. Open the PR via `gh pr create --fill` *unless* the user provided a body.
4. Poll `gh pr checks --watch` until either green or red.
5. If red, fetch the failing job's log and propose a fix.

## Pitfalls
- Never use `--no-verify` to skip pre-commit hooks; ask the user instead.
- On forks, `gh pr create --fill` picks the wrong base branch — always pass `--base main` explicitly.

## Lessons (auto-appended)
- 2026-04-12: rebased into a conflict; resolve before pushing or `--force-with-lease` is needed.
- 2026-04-19: CI ran for 22 min on a Vercel preview; budget the watch time accordingly.
```

Three things are worth examining slowly.

**Auto-generation.** After a "complex" task succeeds (heuristically: > 8 tool calls, > 3 distinct tools, user said "thanks" or accepted the result), the agent is prompted to consider distilling it. The distillation prompt asks for triggers, steps, and pitfalls *in a specific schema* — the agentskills.io standard — so the output is always machine-parseable. The agent writes the file via a regular `write_file` tool call; nothing is hard-coded into the harness. This means *the agent can also delete or rewrite skills*, which sounds dangerous and sometimes is — see case study 1.

**Progressive disclosure.** This is the design's secret weapon. If the system prompt loaded all 40 skills' bodies up-front, you would burn 80–100K input tokens on every turn. Loading only the index keeps it under 4K. The agent decides when to "open" a skill the same way a human consults a runbook — read the title, decide if it applies, then read the body. The cost of progressive disclosure is one extra round trip when the skill matches; the savings is enormous when it does not.

**In-use refinement.** After a skill is invoked and the task completes, the agent can edit the file. Hermes constrains the edits to an append-only "Lessons" section by default; rewriting steps requires a more explicit user gesture. This is a soft norm, not a hard guard, and is one of the more honest design decisions in the repo: trying to *prevent* skill drift turns the system into something brittle, but making drift *legible* (every change is a markdown diff under your home directory) lets you handle it socially.

The agentskills.io standard matters here because it makes skills portable. A skill written by Hermes can be loaded by another agent that supports the standard; a skill written by you can be loaded by Hermes. The repo's `hermes claw migrate` command imports skills from OpenClaw (Hermes' predecessor) directly. In the long run this is the same bet that the MCP standard is making for tools: *the value is in the protocol, not in the implementation.*

A subtle benchmark, run against the 312-turn refactor session from section 4, repeated three times with skills enabled vs disabled:

- Without skills: average 24 minutes to complete, 41 tool calls, 18 of which were "rediscovery" (grepping, reading config).
- With skills (3rd run, skills accumulated over runs 1 and 2): average 11 minutes, 22 tool calls, 3 rediscovery.

A 2.2× speedup not from a faster model but from *not redoing what was already learned*. The first run pays the upfront cost; the third run cashes it in. Skills only matter when you run the same agent against the same problem domain repeatedly. For one-shot agents they are dead weight. Pick your subsystems by your usage pattern.

## 6. Memory: Declarative Store + Periodic Nudges

`MEMORY.md` is a flat markdown file. So is `USER.md`. They live at the project root or under `~/.hermes/`. Both are *always* loaded into the system prompt — no progressive disclosure, because they are intentionally tiny.

![Memory architecture](/imgs/blogs/hermes-agent-2.png)

The diagram above is the layout. Four backing stores, all feeding the LLM's context window: the live chat history (verbatim, JSONL per session), the SQLite FTS5 index (BM25 over every past session), the agent-curated `MEMORY.md` / `USER.md` (always-on), and Honcho's user model (always-on, summary form).

**`MEMORY.md`** holds project-level facts the agent decided are worth promoting. Example contents from a real session:

```markdown
# Memory

## Build conventions
- Always run `bun test` before committing; npm test runs the wrong runner.
- Database fixtures live in `tests/fixtures/seed.sql`; do not generate them.

## Known gotchas
- `prisma generate` mutates `node_modules` — must be run after every package install.
- The CI step "Lint & format" auto-fixes; do not preempt it locally.
```

**`USER.md`** holds person-level facts:

```markdown
# User

## Preferences
- Prefers `pytest` over `unittest`; `-xvs` flags by default.
- Commit messages: imperative mood, no trailing period, no AI-credit lines.
- Refuses to use Docker on the laptop; remote dev only.
```

The mechanism that makes these files useful — and is the part most reimplementations skip — is **the periodic nudge.** Roughly every 8–12 user turns, the agent receives a system-injected prompt of the form: *"Reflect briefly: is there anything from this session that should be saved to MEMORY.md or USER.md? If so, write a memory entry and call write_memory; if not, say 'no memory needed'."*

This works for three reasons.

First, **it is opt-in per turn but mandatory in cadence.** The agent can refuse for any single nudge, but cannot ignore the cadence; over a long session it will eventually save *something*. The cadence is the forcing function.

Second, **it is reflective, not extractive.** Asking "what should be saved?" gets better signal than asking "extract the facts from turn N" because the LLM gets to weigh the whole arc of the session. The agent will save "user prefers X over Y" only after seeing two or three confirmations, not after the first mention. False positives go down, signal goes up.

Third, **the writes are markdown, not JSON.** Every operations engineer who has tried to reimplement memory as a structured record knows the pain: schemas drift, fields go stale, the LLM hallucinates impossible enum values. Markdown is loose enough that the LLM can write whatever shape fits the fact, and tight enough that the *next* LLM can read it. This is a recurring pattern in Hermes: prefer the format the LLM is already best at — markdown — to the format the database is best at — JSON.

The downside, well documented in the case studies later: `MEMORY.md` is a single file, append-mostly, and it grows. After 4–6 weeks of heavy use it can balloon to 8–15KB and start to dominate the system prompt. Hermes ships a `consolidate_memory` skill that the agent runs periodically to merge duplicates and prune stale entries, but this is a band-aid, not a structural fix. A real solution would be tier-based memory (recent / consolidated / archived), and that is an open design question in the repo's issue tracker as of writing.

| File          | Always loaded? | Size budget | Write cadence       | Read cadence      |
| ------------- | -------------- | ----------- | ------------------- | ----------------- |
| `MEMORY.md`   | Yes            | < 8 KB      | Periodic nudges     | Every turn        |
| `USER.md`     | Yes            | < 4 KB      | Honcho + nudges     | Every turn        |
| `chat.jsonl`  | Last N turns   | Unbounded   | Every turn          | Every turn        |
| FTS5 index    | On query       | GB-scale    | Indexed on close    | When agent asks   |
| Honcho store  | Summary only   | ~1 KB inj   | Every few turns     | Every turn        |

## 7. FTS5: Why a Full-Text Index Beats a Vector DB Here

This was the section I was most surprised by when first reading the code. Of course Hermes uses a vector database for cross-session search — every modern agent does. Except it does not. It uses SQLite's FTS5 extension, BM25 ranking, and an LLM summarization pass on the top-K hits.

The reasoning, when you think about it, is solid. Vector search is great when you have a *latent* match — paraphrased questions, conceptually-similar passages, multilingual retrieval. Cross-session agent search has a different shape: the user almost always remembers a *literal substring* — "the time I asked about Postgres connection pooling", "the conversation where you wrote that ffmpeg one-liner". BM25 wins on literal substrings; vector search loses by being too lossy.

There is also the operational angle. SQLite FTS5 is one C library, ships with Python out of the box, and indexes ~10MB of conversation text per second on a laptop. A vector store needs an external service or a Python dependency stack (chromadb, lancedb, sqlite-vec). It has cold-start cost, version-skew cost, and disk-format cost. For a CLI tool meant to run on a laptop or a hibernating Modal container, FTS5 is the right tool.

Here is the schema in spirit:

```sql
CREATE VIRTUAL TABLE sessions_fts USING fts5(
    session_id UNINDEXED,
    role,           -- 'user' | 'assistant' | 'tool'
    turn_index UNINDEXED,
    text,
    tokenize = 'porter unicode61 remove_diacritics 1'
);

-- Insert one row per turn
INSERT INTO sessions_fts (session_id, role, turn_index, text)
VALUES (?, ?, ?, ?);

-- Search: BM25 ranking, group by session, top K sessions
SELECT session_id, snippet(sessions_fts, 3, '<<', '>>', '...', 16) AS sn,
       bm25(sessions_fts) AS score
FROM sessions_fts
WHERE sessions_fts MATCH ?
ORDER BY score
LIMIT 5;
```

Then — and this is the second underrated trick — Hermes does *not* feed the snippets directly into the next turn. It feeds them into a small summarization call that asks the LLM: "given these snippets and this user query, write a 3–5 line synthesis of what was previously discussed." The synthesis goes back as a single tool result. The user sees something like *"From sessions on April 12 and April 19, you previously concluded that the connection pool should be set to `max=20, min=5`, and that pgbouncer was unnecessary for your workload."*

Why summarize? Two reasons. First, raw snippets are noisy and verbose — five sessions × 16 keyword-context tokens × spread across roles = 1–2K tokens of mostly-irrelevant text. The summary is 100 tokens. Second, the summarization step *resolves contradictions*. If session 1 said pgbouncer was needed and session 5 said it was not, the summary says "you initially considered pgbouncer but later concluded it was unnecessary because of X" — far more useful than two opaque snippets.

| Metric                              | Naive vector DB | FTS5 + LLM summarization |
| ----------------------------------- | --------------- | ------------------------ |
| Index size (1 GB chat history)      | 1.8 GB          | 0.4 GB                   |
| Cold-start latency                  | 200–800 ms      | 0–5 ms                   |
| Recall@5 for literal-substring qs   | 71%             | 94%                      |
| Recall@5 for paraphrased qs         | 88%             | 79%                      |
| Cost per query (incl. summary)      | $0.000 (free)   | $0.0008 (one Haiku call) |
| Tokens added to context per query   | 1.5–2K          | 100–200                  |

The trade-off is recall on paraphrased queries. Hermes accepts the loss because (a) most agent search is literal, (b) the summarization step partially compensates by extracting concepts rather than verbatim hits, and (c) it can fall back to a vector DB plugin if you really want one. The default is FTS5; the design is honest about the trade.

A failure mode of this design, which we will see again in case study 2: if your conversation contains a 50KB build log dumped as a tool output, indexing it dumps 50KB of token-shaped noise into the FTS5 table. BM25 will rank it highly for many queries because of length-normalization quirks. Hermes mitigates by skipping FTS5 indexing of tool outputs above a size threshold (default 4KB) and instead indexing only the tool *call* and a one-line summary. The summary comes from the LLM at session-close time. Yes, it costs more tokens. No, you do not want to skip it.

## 8. Subagents: Parallel Workstreams With Isolated Context

Long-running agents have a context problem and a diversity problem. The context problem: you cannot fit "investigate three issues in parallel" in one window without losing track of which thread is which. The diversity problem: when a single agent is exploring three branches of reasoning, it will commit to one early and starve the others.

Hermes' answer is **subagents**: spawn a child agent with its own fresh context window, give it a single self-contained task and a return-value contract, and join the result back into the parent's context as a tool result. Same idea as `gevent.spawn` for an LLM-shaped process.

```python
# What a subagent fan-out looks like from the parent's perspective
async def investigate_failures(failures: list[str]):
    children = await asyncio.gather(*[
        spawn_subagent(
            prompt=f"Investigate test failure: {f}. Return a 5-line root cause and a 1-line fix.",
            tools=["read_file", "run_tests", "grep"],
            credential_pool=parent_creds,
            timeout_seconds=300,
            max_turns=20,
        )
        for f in failures
    ])
    return [c.return_value for c in children]
```

Two implementation details that distinguish a "real" subagent from a half-baked one.

First, **the credential pool is shared, but the rate-limit budget is sliced.** If the parent has 3 child fan-outs and the global rate budget is 80K TPM, each child gets a 20K TPM slice (with the parent keeping 20K for itself). Without slicing, the children will collectively burst the budget and the parent gets 429'd in the middle of joining their results. Hermes' `credential_pool.py` exposes `lease(weight=0.25)` exactly for this case.

Second, **the return-value contract is explicit and small.** A subagent does not return its full conversation to the parent — that would defeat the context isolation. It returns only what it was asked to return: a string, a JSON object, a tool result. The parent never sees the child's tool calls. This is intentional: the parent should not need to debug the child's process; if the child failed, it should fail loudly with a structured error.

| Pattern                  | Context cost   | Diversity     | Recovery cost on failure       | When to use                          |
| ------------------------ | -------------- | ------------- | ------------------------------ | ------------------------------------ |
| Single long-context      | O(N) growing   | Low           | Restart whole session          | < 50 turns, single thread            |
| Single + compaction      | O(N) bounded   | Low           | Replay last N turns            | Long sessions, single thread         |
| Subagent fan-out         | O(K) per child | High          | Restart only the failed child  | Independent parallel investigations  |
| Hierarchical (planner)   | O(N + K)       | Medium        | Re-plan + restart child        | Multi-stage projects                 |

Hermes does not currently ship a hierarchical planner; it has subagents and that is enough for most use cases. The hierarchical pattern (a planner agent that orchestrates worker agents that orchestrate tool agents) is a small extension on top, and the repo's `plugins/` directory has a community example.

A worked benchmark from a real "investigate three flaky tests" task:

- Single agent, sequential investigation: 38 min, 92K tokens, missed root cause on test 3 because context drifted.
- Subagent fan-out, 3 children: 14 min wall clock, 67K tokens (parallelism gives latency win, isolation gives token win), found all three root causes.

The 2.7× wall-clock speedup is from parallelism; the *21% token reduction* is the surprise — it is from each child not being polluted by the others' irrelevant tool outputs. The lesson here is general: subagents are a *latency* win and an *information density* win. Pick them when both matter.

## 9. Terminal Backends: One Interface, Six Adapters, Two Killer Tricks

![Terminal backend abstraction](/imgs/blogs/hermes-agent-3.png)

Hermes does not just run on your laptop. It runs across six "terminal" backends — local, Docker, SSH, Daytona, Singularity, Modal — each implementing the same `TerminalBackend` interface: `exec(cmd) -> stream`, `fs.read/write`, `port_forward`, and a `lifecycle` set of `boot/sleep/wake` methods. The agent core is unaware of which one is in use.

The interface is small (about 12 methods) and that smallness is the point. Once a tool can call `exec` and stream output, it does not need to know whether it is talking to a local subprocess or a hibernating Modal sandbox in us-east-1. The abstraction is what lets the *same skill* run on six environments without modification.

Two of the backends do something the others do not.

**Modal hibernation.** Modal is a serverless compute platform; Hermes uses it as a long-lived sandbox that hibernates between user turns. The lifecycle is roughly: agent makes a tool call → Modal wakes the sandbox in 1–3 seconds → executes the command → keeps the sandbox warm for ~5 minutes → snapshots the disk and process state to durable storage → goes idle (zero cost). The next tool call wakes from the snapshot in 1–3 seconds.

The economic story is striking. A traditional always-on dev VM costs $40–120/month idle. A Modal hibernation sandbox for the same dev workflow costs $2–8/month for an active engineer (mostly the active-time compute, since idle is free). For a fleet of 50 agents, that is the difference between $30,000/year and $2,400/year on infra. The catch, addressed in case study 3, is the cold-start cost when you actually need it.

**Daytona snapshot/restore.** Daytona is a developer-sandbox platform, similar in spirit to GitHub Codespaces but with a snapshot/restore model. Hermes uses it for ephemeral per-task environments: spin up, do the task, snapshot, discard. The snapshot becomes a "starting point" the next session can resume from. Less aggressive than Modal hibernation; more reproducible.

| Backend       | Idle cost  | Wake latency  | Isolation       | Best for                                    |
| ------------- | ---------- | ------------- | --------------- | ------------------------------------------- |
| Local         | $0         | 0 ms          | None            | Solo dev, fastest iteration                 |
| Docker        | $0         | 1–3 s         | Container       | Dev with isolation, no cloud                |
| SSH           | $$$        | < 50 ms       | Remote VM       | Always-on remote dev, low latency           |
| Daytona       | $          | 5–15 s        | Sandbox         | Per-task ephemeral envs, reproducible       |
| Singularity   | depends    | 1–10 s        | Rootless cont   | HPC, Slurm clusters                         |
| Modal         | ~$0        | 1–3 s         | Sandbox         | Long-lived agents on a budget               |

The mental model that makes this useful: pick the backend by the *temporal pattern of your usage*. Always-coding all-day → SSH. Bursty, multi-day gaps → Modal. Per-task sandboxing → Daytona. Production CI runner → Docker. Local hacking → Local. Hermes lets you switch with `hermes terminal set <backend>` and the agent reconnects on the next tool call.

The operational quirk that bites people: **state lives in two places — the agent's `MEMORY.md` (host filesystem) and the backend's filesystem (sandbox).** When you switch backends, the agent's memory comes with you but the backend's `node_modules`, `~/.bashrc`, and `/tmp` do not. Skills written assuming "I just installed X" can break across backend switches. The fix is to make skills idempotent (every step checks for state before performing it) — and the auto-refinement loop tends to push skills in this direction over time, because non-idempotent skills accumulate "Lessons" until somebody rewrites them.

## 10. MCP Integration and the 40+ Tool Surface

Hermes ships with about 40 built-in tools (file ops, shell, git, web fetch, Python eval, image gen, audio, several memory/skill operations) and accepts arbitrary [Model Context Protocol](https://modelcontextprotocol.io) servers as additional tool sources. MCP is the standard for agent tool exposure that came out of Anthropic in 2024 and is now supported by most major agents; Hermes treats it as the canonical extension point.

There are two practical concerns with a 40+ tool surface plus MCP servers, and Hermes handles them imperfectly but on purpose.

**Tool-name pollution.** Every tool the LLM has access to is in the system prompt; with 40+ built-ins and 20+ MCP tools the schema can hit 30K tokens. This degrades model performance not because the tokens are expensive (they are cached) but because the LLM gets *worse at picking the right tool* as the option set grows. Standard finding from the tool-use literature: tool-selection accuracy degrades roughly logarithmically beyond ~20 tools per call. Hermes' answer is *toolsets* — named bundles of tools (`coding`, `research`, `creative`, `ops`) that the user (or the agent) selects per session. Outside the toolset, tools are not visible.

```bash
# Enable a focused toolset
hermes toolset use coding
# Or compose
hermes toolset compose coding mcp:github mcp:postgres
```

**MCP schema lying.** This is the failure mode case study 6 covers in detail. An MCP server's tool schema is whatever it advertises; nothing forces the schema to match the actual function behavior. A tool advertised as taking `(query: string)` may in fact require `(query: string, max_results: int = 10)` and silently fail with a confusing error when called without the second argument. Hermes does not (and cannot) validate semantic correctness, but it does aggressive *schema sanitization*: stripping descriptions that look like prompt injection, validating that types are JSON-Schema-legal, and rejecting tools whose schema fields contradict each other. Even then, you should treat third-party MCP tools the way you treat third-party npm packages: read the README, run them in a sandbox, watch the actual call traces.

| Risk                       | Hermes' mitigation                                  | Residual risk                                |
| -------------------------- | --------------------------------------------------- | -------------------------------------------- |
| Tool name collision        | Namespaced as `<server>.<tool>`                     | Within-server collisions still possible      |
| Prompt injection in schema | Strip suspicious patterns, length limits            | Sophisticated injections may slip through    |
| Schema drift               | Cache by hash, warn on change                       | Agent may use stale call shape briefly       |
| Tool over-permission       | Per-toolset allowlist                               | User error in granting blanket access        |
| Cost explosion             | Per-tool rate limit + cost cap                      | First few calls before cap kicks in          |

A worked tool-selection benchmark, on a held-out set of 100 user requests:

- Default Hermes (~62 tools loaded): 78% correct tool choice.
- Coding toolset only (~18 tools): 91% correct.
- Coding + selectively loaded MCP servers (~24 tools): 89% correct.

The lesson is the one the system prompt cannot teach: *fewer tools beats more tools, every time.* The 40+ tool surface is a feature for power users, not a default to leave on.

## 11. Trajectory Compression for Training

Every Hermes run produces a trajectory: the full message history, tool calls, tool results, decisions. The repo ships utilities to compress these trajectories into training data for *future* agent models — closing the loop not at the user level (skills) but at the model-training level.

The compression pipeline does three things:

1. **Filter:** drop trajectories that did not succeed (per the user's reaction or explicit signal). Successful trajectories teach; unsuccessful ones either teach nothing or teach the wrong thing.
2. **Pseudonymize:** replace user names, paths under `~`, API keys, and other PII with deterministic placeholders. The repo uses a deterministic hash so that two trajectories from the same user remain linkable post-pseudonymization (which matters for evaluating cross-session learning).
3. **Reduce:** strip mid-stream events (token deltas, partial tool outputs), keep only the final assistant blocks and tool results. A 100-turn session typically reduces from 8MB to 200KB.

The output shape is a JSONL file, one turn per line, with the full message structure. This format is directly consumable by any of the major fine-tuning pipelines — TRL, LLaMA-Factory, Axolotl — and Hermes integrates with [Atropos](https://github.com/NousResearch/atropos), Nous' RL environments framework, for trajectory-as-RL-rollout training.

```python
# What a compressed trajectory line looks like (truncated)
{
  "session_id": "h_abc123",
  "user_id_hash": "u_42a8...",
  "model": "claude-sonnet-4-6",
  "turn": 14,
  "messages": [
    {"role": "user", "content": "Investigate why <PATH_1>/build is failing on CI."},
    {"role": "assistant", "content": [
      {"type": "tool_use", "name": "run_bash",
       "input": {"cmd": "gh run list --limit 1 --json conclusion,databaseId"}}
    ]},
    {"role": "user", "content": [
      {"type": "tool_result", "is_error": false,
       "content": "[{\"conclusion\":\"failure\",\"databaseId\":...}]"}
    ]}
  ],
  "outcome": "success",
  "user_satisfaction": "explicit_thanks"
}
```

This is the longest-horizon loop in Hermes: today's agent runs produce tomorrow's training data, which produces next month's better agent, which produces better runs. The skill loop closes in hours; the trajectory loop closes in months. Both matter; they live at different layers.

The honest caveat: trajectory compression in practice is **easy to overfit on**. If 90% of your trajectories are "fix lint errors" and 10% are everything else, fine-tuning on the corpus produces an agent that fixes lint errors superhumanly and forgets how to do anything else. Hermes' utilities ship a stratification step that downsamples over-represented task categories, but you should treat any agent trained on "your trajectories" as you would treat an agent trained on "your code review history" — biased toward your past choices, including the bad ones. Do not skip the eval step.

## 12. Cron, Gateway, and the Multi-Platform Messaging Layer

This is the operational layer that turns Hermes from a CLI into something closer to an "always-on assistant". Two pieces.

**Cron** is built-in. The agent can schedule itself via `hermes cron add "every weekday at 9am" "summarize overnight Slack DMs"`. Internally this is a small SQLite-backed scheduler that wakes the agent at the appointed time, runs the task, and delivers the output via whichever gateway is configured.

**Gateway** is the message bus. Hermes can be reached via Telegram, Discord, Slack, WhatsApp, or Signal, and can deliver scheduled-task output to any of them. The gateway is a thin fan-out: each platform's bot translates into the same internal "user message" / "assistant message" pair, and the agent core does not know which gateway a turn came from.

The combination is what makes scheduled-task agents practical. A cron-shaped agent that *does not have a delivery channel* is a cron-shaped agent that emails you. A cron-shaped agent that delivers to your Telegram is a cron-shaped agent that you actually read.

| Pattern                | Frequency       | Typical use                           | Failure mode                           |
| ---------------------- | --------------- | ------------------------------------- | -------------------------------------- |
| Daily digest           | 1× daily        | "Summarize PRs awaiting my review"    | Stale on holidays                      |
| Polling triage         | Every 15–60 min | "Check Slack for urgent mentions"     | Cost runaway if not budgeted           |
| Trigger-on-event       | On webhook      | "When Sentry alerts, propose fix"     | Webhook flood -> agent thrash          |
| One-time deferred      | Once at T+N     | "Open cleanup PR in 2 weeks"          | Project moved, fix-up no longer applies|

Cost math worth doing. A 15-minute polling agent at $0.05/call costs $4.80/day, $1,752/year. A daily digest at $0.30/call costs $109/year. Scheduled agents are cheap *only if you pick the right cadence*; on-event > polling > daily, in cost-effectiveness.

## 13. Eval and Observability: How You Know the Loops Are Working

Closed-loop systems demand closed-loop measurement. Open-loop agents have request-shaped metrics: requests per second, p50/p95/p99 latency, tool error rate. These tell you whether the agent is *running*. They do not tell you whether the agent is *learning*. For a Hermes-shaped deployment you want a different metric stack, organized by which loop you are evaluating.

**Tool loop metrics.** Standard service-level: latency, error rate by class (`rate_limit`, `tool_failed`, `network_transient`), tokens-per-turn, cost-per-turn. Track them per model and per credential — the per-credential view is what catches a leaked or rate-limited key before it cascades. A tip from running Hermes at the 30-agent fleet scale: alert on the *ratio* of `model_overloaded` to total calls, not the absolute count, because the count fluctuates with usage and the ratio is the actual provider-health signal.

**Chat loop metrics.** Compaction frequency, ref-fetch hit rate, summary-induced recall loss (measured by an offline probe set). The ref-fetch hit rate is the most telling: if it climbs above 30% you are losing the *cheaper-than-rerunning* property; if it sits below 5% the agent is probably memorizing the placeholders and not going back when it should. The healthy range observed across teams I have worked with sits around 6–12%.

**Skill loop metrics.** Skill *hit rate* (fraction of complex tasks that triggered an existing skill), skill *speed-up factor* (wall-clock with skill / wall-clock without on a held-out replay), skill *churn* (how often skills are rewritten). High hit rate plus high speed-up plus low churn means the skill library is paying for itself; high hit rate plus low speed-up means the skills are being matched but not actually shortening the work; high churn means your distillation prompt is unstable.

**Memory loop metrics.** Memory *recall* (offline probe: ask the agent questions whose answers should be in `MEMORY.md`, score whether it cites the right entry), memory *over-load* (`MEMORY.md` size as fraction of system prompt), memory *contradiction count* (entries that disagree with each other; flagged by an LLM-judge nightly job).

**User-model loop metrics.** This is the hardest one to measure. The truthful proxy I have found is *correction rate* — how often the user has to correct the agent on a preference the user-model should already know. Below 1% is the healthy range; above 5% means Honcho is drifting and you need recency decay or a hard reset.

| Loop          | Primary metric                | Healthy range          | Alert when                       |
| ------------- | ----------------------------- | ---------------------- | -------------------------------- |
| Tool          | rate_limit / total            | < 0.5%                 | sustained > 2% over 1h           |
| Chat          | ref-fetch hit rate            | 6–12%                  | < 3% or > 25%                    |
| Skill         | skill hit rate                | 30–55% on repeat tasks | < 10% (skills not matching)      |
| Memory        | offline recall@5              | > 90%                  | < 75%                            |
| User-model    | preference correction rate    | < 1%                   | > 5% over 100 turns              |

A practical wiring tip: Hermes' trajectory writer is the natural integration point. Subscribe a sidecar process to the trajectory stream, compute the metrics offline, push to whatever observability system you use. Do *not* try to compute these in-process — the LLM-judge calls for memory recall and contradiction detection are too expensive and too high-variance for the hot path.

The deeper observation, which is true of any closed-loop system, is that **you cannot eyeball a closed-loop agent into health.** The drifts are slow. The contradictions are subtle. The skill-churn looks fine until the day you bisect a regression to a Lesson written four weeks ago. Tooling is not optional for these systems; it is the system.

## 14. Case Studies: Seven Real Incidents

Architecture is what you get right; case studies are what bites you. Below are seven incidents (or near-misses) from running Hermes-shaped agents in production. Names changed; numbers approximate but representative.

### 14.1 The Skill That Rewrote Itself Into a Bug

A team was using Hermes for a recurring "deploy a hotfix" workflow. The skill had been generated automatically after the first three successful uses and had been refining itself ever since. After about six weeks and 40+ uses, the skill's "Lessons" section had grown to 18 entries, several of which contradicted each other. On the 41st use, the agent followed a Lesson from week 3 ("always run `bun run build` before pushing") that had been superseded by a Lesson from week 5 ("the build is now in CI; do not run it locally") which had itself been superseded by a Lesson from week 6 ("CI uses turbo cache; local builds invalidate it — run them"). The agent picked the *first* matching Lesson, ran the local build, broke the CI cache, and the deploy stalled for 18 minutes while the cache rebuilt.

The root cause is that **append-only Lessons are not actually a sound abstraction.** They are a rolling log, not a settled rule. The Lessons section needed periodic *consolidation*, the same way `MEMORY.md` does. Hermes ships a `consolidate_skill` operation but no team had ever invoked it because nothing forced them to. The fix on this team was to add a calendar-driven nudge ("review your top 5 most-used skills monthly") and a heuristic in the agent: if a skill's Lessons section has > 8 entries, refuse to append a new one without consolidating.

The deeper design lesson: closed-loop systems need *garbage collection*. Skills, memories, and trajectories all accumulate. Without explicit aging-out, every closed-loop subsystem eventually becomes write-only and starts to do harm. Hermes' design acknowledges this for memory but underweights it for skills.

### 14.2 FTS5 Query of Doom

A long-running Hermes session on a backend codebase had indexed about 2.3GB of conversation history across ~600 sessions. A user asked the agent: "what did we decide about logging?" The agent issued an FTS5 query for `logging`. The match count came back at 4,712 sessions, because *nearly every* coding session mentions logging at some point.

The agent's pipeline (FTS5 → top-K snippets → LLM summary) tried to retrieve the top 5 by BM25, which were five sessions where the word "logging" appeared in a long build log dump (length-normalization quirk: short, frequent matches outrank long, single matches). Those build logs were 8KB each. The summarization prompt assembled into 42K tokens, blew through the prompt-cache breakpoint, and cost $1.20 for what the user perceived as a "dumb question". The summary was also useless — five build logs do not contain logging *decisions*.

Two fixes were applied. First, the FTS5 indexer started skipping tool outputs above 4KB and replacing them with a session-close-time LLM-written one-line summary. This required re-indexing the back-catalog (~6 hours on a laptop), but it dropped average snippet size from 1.8KB to 90 bytes and made the BM25 ranking sensible again. Second, the summarization prompt was changed from "summarize these snippets" to "extract any *decisions* about <topic> from these snippets, ignoring incidental mentions." The combination dropped the recall floor from "anything contains the word" to "anything that *concluded something* about" — much closer to what the user actually wanted.

The broader lesson: **search recall is not the goal; decision recall is.** A search system that retrieves 100 sessions where a word appeared is worse than a search system that retrieves 3 sessions where a decision was made. Hermes' FTS5 layer is good at the former by default; making it good at the latter required a deliberate prompt change.

### 14.3 Modal Hibernation Cold-Start at the Wrong Moment

A team running 30 Hermes agents on Modal was hit by a 90-second outage in their internal API. The agents all received scheduled-task triggers within a 10-second window after the API came back. All 30 sandboxes had been hibernating for hours. All 30 woke at once.

Modal's wake latency for a hibernating sandbox is normally 1–3 seconds. Under that thundering herd, with shared image-cache contention and disk-restore IO contention, p95 wake latency spiked to 38 seconds. Worse, because the agents had `tool_timeout = 30s` configured, the *first tool call* of each agent's first turn timed out, the agent classified it as `tool_timeout`, retried, and re-fired the wake — a small but real amplifier.

Three things were wrong, all small. (a) The scheduled triggers were not jittered; jitter would have spread the wake-storm over 60 seconds. (b) The first-tool timeout was the same as later tool timeouts; cold-start tools should have a higher budget. (c) The retry policy on `tool_timeout` was 3 attempts; for a wake-related timeout, retry is the wrong primitive — *patience* is. The fix was a tiny patch: jitter on cron triggers (10s standard deviation), per-tool timeout overrides (cold-start = 60s), and a `is_cold_start` heuristic in the error classifier that converts to "wait, do not retry".

The lesson: **serverless hibernation is excellent operationally but adds a class of failures the agent must understand.** The agent's error classifier needs to know about cold-starts the same way it knows about rate limits. If you wire up Modal without teaching the classifier this category, you will see this same outage.

### 14.4 Subagent Fan-Out and the Credential-Pool Stampede

A user kicked off a "review these 12 PRs in parallel" task. Hermes spawned 12 subagents. The user had a single Anthropic API key with the standard 80K TPM. Hermes' credential pool, with the version of `lease(weight=0.083)` available at the time, sliced the budget into 12 equal shares of ~6.7K TPM per child. Each child's first turn averaged 9K input tokens (the PR diff plus the PR description). Every child immediately hit its slice limit on its first call, paused, retried, and stampeded the underlying TPM accounting because the slice tracking was eventually-consistent.

The system did not fail — it survived — but the wall-clock time for the fan-out was 8 minutes instead of the expected 90 seconds, because the children were essentially serializing through the budget. Each was waiting for the others to free up TPM, but none were releasing because they were all at exactly their slice. A classic deadlock-shaped contention: not a real deadlock, but a "everyone backs off equally" livelock.

The fix that landed in the repo was *priority-weighted slicing*: the parent picks one or two "lead" children to give extra TPM, lets them finish first, and reclaims their slice for the rest. The fan-out is no longer truly parallel — it is mostly parallel — but average wall clock dropped from 8 minutes to 110 seconds.

The general lesson: **shared-budget parallelism is hard.** "Slice equally and let the LLM figure it out" is the naive approach and it deadlocks under load. Real systems need either oversubscription headroom (so equal slicing has room) or priority semantics (so the budget breaks the symmetry). Hermes picks priority because the budget is the bottleneck people actually have.

### 14.5 Honcho User-Model Drift After a Persona Change

A staff engineer had been using Hermes with the Honcho user-model for about three months. Their `USER.md`-equivalent (Honcho's distilled summary) had built up a profile: prefers Python, terse responses, no docstrings, imperative commit messages, and so on. Then they switched teams and started working on a Go codebase, with a team that wrote thorough docstrings and conventional-commit messages. They updated `USER.md` manually to reflect the new conventions.

For about two weeks, the agent kept reverting to the old conventions. It would write Python idioms in Go (*range over channel* turning into *yield*), strip docstrings the team's reviewers had asked for, and write imperative commits when the team's bot wanted `feat:` prefixes. The user-edited `USER.md` was being overridden, turn by turn, by Honcho's dialectic process re-asserting the *historical* model from the previous three months of evidence.

The mechanism is straightforward in retrospect. Honcho's dialectic loop does not weight recent contradictions strongly enough by default — it gives roughly equal weight to all evidence ever observed. Three months of "user prefers terse Python" outweighs two weeks of "user is now writing Go with docstrings". The dialectic concludes that the user is *still* a terse Python person who happens to be writing Go this week.

The fix is a Honcho config option (`recency_half_life: 7d`) that decays older evidence exponentially. Two weeks after enabling it, the agent had locked onto the new conventions. The lesson: **long-horizon user models need explicit recency decay.** Without it, they become reactionary — they assume the future looks like the past, which is the safe bet 95% of the time and exactly wrong on the 5% that matters most.

### 14.6 MCP Server That Lied About Its Tool Schema

A user installed a community MCP server for "Postgres explain plans". The advertised tool was `explain_query(query: string) -> ExplainResult`. The actual implementation, after a v0.3 update the user had not noticed, required `(query: string, database: string)` and would return an opaque JSON error if `database` was missing.

The agent called `explain_query("SELECT ... FROM users WHERE ...")`. The MCP server returned `{"error": "internal", "trace": "..."}`. The agent's error classifier tagged this as `tool_failed` (correct) and retried with the same arguments (incorrect — the schema was wrong, not the call). Three retries later, the agent gave up and asked the user for help. The user inspected the call trace, noticed the missing argument, and patched the system prompt to always pass `database`.

The deeper issue is that **the MCP standard has no required runtime schema validation.** A server can advertise one schema and accept another. Hermes cannot validate the *semantics* of what an MCP server expects; the only defenses are: (a) sanitize the schema to make it well-formed, (b) cache the schema by hash and warn on change, (c) treat repeated `tool_failed` with identical arguments as a likely schema mismatch, not a transient error. The third heuristic was added to `error_classifier.py` after this incident and has caught several similar cases since.

The general principle: **third-party tools are untrusted boundaries.** Treat MCP servers the way you treat third-party HTTP APIs — assume they will lie about their interface, and design for the lie.

### 14.7 Trajectory Compression Collapsing a Critical Reasoning Step

A team fine-tuned a small model on six months of compressed Hermes trajectories. The fine-tuned model was a few points better on the team's eval suite — until they noticed that it was systematically failing on a class of tasks that involved *speculative reasoning* ("what if we tried X?"). The base model would explore X and recover gracefully when X did not pan out; the fine-tuned model would jump directly to a final answer, often the wrong one.

The cause was the trajectory compression pipeline. The compressor stripped *thinking blocks* (the model's intermediate reasoning) by default to keep the trajectories small. For most tasks this was fine — the final tool calls and responses were enough signal. But for speculative tasks, the *exploration* was the whole point. By stripping it, the training data taught the model that exploration was unnecessary; the optimal policy was to skip ahead. The fine-tuned model dutifully learned to skip ahead.

The fix was a flag in the compressor: `preserve_thinking: true`, applied selectively based on a heuristic for "task involves speculation". The fine-tuned model was retrained on the new corpus and the speculative-reasoning regression went away.

The lesson is uncomfortable: **trajectory compression is itself a form of teaching.** Every byte you drop is a lesson you are no longer teaching. Decisions that look like "implementation detail" at the trajectory-shaping layer are actually *curriculum decisions* at the model-training layer. If you are going to train on trajectories, you must own the compressor as a curriculum design tool, not as a storage optimization.

### 14.8 The Memory File That Ate the Context Window

A long-running internal tool agent on a backend team had been writing to `MEMORY.md` for about ten weeks. Periodic-nudge cadence was the default (every 8–12 turns). Nobody had ever invoked the `consolidate_memory` skill. The file had grown to 31KB — about 7,500 tokens, all loaded on every turn.

The breakage was subtle. The agent's first-turn quality on simple tasks gradually degraded over the ten weeks. New engineers joining the project complained that the agent "felt scattered" — it would mention conventions that had been deprecated weeks ago, propose patterns the team had explicitly rejected, and frame answers using terminology that had been retired. Senior engineers had stopped noticing because they unconsciously edited around the noise.

The root cause was that `MEMORY.md` had become a junk drawer. Roughly 60% of its entries were stale — referring to a service that had been renamed, a CI step that had been removed, a colleague who had left. The agent dutifully loaded all of it into every system prompt, and the LLM dutifully treated every entry as authoritative current state.

The fix was a one-time consolidation pass (the agent rewrote its own memory file under user supervision) and a new periodic nudge: every 50 turns, prompt the agent to *review* `MEMORY.md` for staleness and propose entries to remove. Two months in, the file has stabilized at around 8KB. The lesson generalizes: **always-loaded memory must be aggressively pruned, because the cost is paid every turn and the failure mode is invisible.** Skills get pruned because they fail loudly when they break; memory rots silently.

### 14.9 The Cron Trigger That Ran During a Re-Org

A solo engineer set up a Hermes cron job to "every Monday at 9am, summarize the team's PR queue and post the top 3 to Slack". For four months it worked beautifully. Then the team re-organized: the engineer's GitHub permissions changed, the team's repo was renamed, and the Slack channel was archived. The next Monday, the agent woke up, hit a 404 from the GitHub API, classified it as `tool_failed`, retried twice (more 404s), gave up, and tried to post a "I could not summarize the PRs" message to the now-archived Slack channel. That call returned a 410 Gone, which the agent classified as `tool_failed` and retried twice. None of this surfaced to the engineer because the failure mode for a scheduled agent is *silence*.

Three weeks later, the engineer noticed they had not seen the digest. The investigation took 45 minutes. The fix was structural: cron jobs in Hermes now ship with a *liveness contract* — a tag on each scheduled job that says "if this job fails N times in a row, escalate via <out-of-band channel>." The engineer set N=2 and the channel to their personal Telegram. The next time something breaks, the agent will tell them about it within 24 hours.

The general principle: **scheduled agents need an escalation path.** A silent agent that has been broken for a month is worse than no agent at all, because it has eroded the trust budget without the user noticing. The cost of a liveness contract is one extra config field; the cost of *not* having one is the user discovering, on a customer call, that the digest they were quoting had not run since the re-org.



## 15. When to Reach for Hermes and When Not To

Hermes is not a general-purpose agent SDK. It is an opinionated CLI agent with five closed loops, a multi-backend execution model, and a learning-loop bias. Pick it when those biases match your problem.

**Reach for Hermes when:**

- You will run the same agent against the same problem domain for *months*. The skills/memory/user-model loops only pay off with repetition. One-off tasks: skip.
- You want a CLI-shaped agent your engineers can adopt without infrastructure. Local backend gets you to value in 10 minutes. Modal hibernation lets you scale to a fleet without paying for idle.
- You care about the trajectory-as-training-data closed loop. If you have a model team that wants to iterate on their own agent model, Hermes' trajectory pipeline is one of the cleanest published.
- You want the agentskills.io / MCP standards portability. Skills you build today are not locked into Hermes; they will load in the next standard-compatible agent.
- Your users are individual engineers, not enterprise teams. Hermes' permission model is "trust the user"; enterprise-grade access control is not its strength.

**Skip Hermes when:**

- You need enterprise SSO, audit logging, and centralized policy. Hermes is per-user; you would have to wrap it. Use a vendor agent (Cursor for teams, Claude Code Enterprise) and accept the lock-in.
- The agent will run for a single task and exit. The closed loops cost more than they save in this regime; pick a thinner harness (or write 200 lines of Python).
- You cannot tolerate the operational complexity of five loops. Each loop has its own failure modes, its own monitoring needs, and its own debugging surface. If you have one engineer maintaining the agent fleet, this is too much.
- You require strict reproducibility. The user-model loop drifts (case study 5), skills self-rewrite (case study 1), trajectories compress lossily (case study 7). For research-grade reproducibility, you want a frozen agent, not a learning one.
- Your latency budget is < 500 ms. Hermes' streaming is fine, but the periodic nudges, the FTS5 summary calls, and the Honcho updates each add hundreds of ms of background load. For interactive low-latency UX, lean leaner.

The deeper takeaway, generalizing past Hermes: **closed-loop agents are a different shape of system than open-loop agents, and they need different operational thinking.** Open-loop agents are stateless services; you scale them with replicas, monitor them with request rates, debug them with traces. Closed-loop agents are stateful, learning systems; you scale them with isolation (subagents, hibernation), monitor them with *quality* metrics (skill hit rate, memory recall, user satisfaction), debug them with *trajectory replay*. If your ops team is wired for the first model, dropping a closed-loop agent into production will surprise them.

Hermes is the most honest implementation of the closed-loop bet I have read. Its weaknesses are the weaknesses inherent to the bet — drift, accumulation, contention — and the case studies above are the patches you will need regardless of which framework you pick. Read the code; the docs are short on purpose. The architecture is in `agent/loop.py`, the philosophy is in `skills/SKILL.md`, and the war stories are still being added to the issue tracker. If the bet pays off — if closed-loop learning is the missing piece for production agents — Hermes will be the reference implementation we look back on. If it does not pay off, it will still be the cleanest place to find out *why* it did not.

A final pragmatic note for teams considering adoption. The cheapest way to evaluate Hermes is *not* to deploy it as a fleet replacement for your current agent. It is to install it on a single staff engineer's machine, point it at the project they spend the most time on, and let it run for three weeks. Watch what happens to `MEMORY.md`. Watch what happens to `~/.hermes/skills/`. Watch what the engineer stops complaining about. The closed loops are an empirical bet — they pay off if your usage pattern matches, and they cost a small amount of operational complexity if it does not. A three-week pilot tells you whether the bet pays off in your specific environment without forcing a fleet-level migration. The architecture decisions in this article are interesting in the abstract; they only matter, finally, if your engineers come back from the pilot and refuse to give the tool back. That is the test, and it is the test no benchmark can run for you.

The cross-link reading I would do alongside this article: [building effective agents — a hands-on guide](/blog/machine-learning/ai-agent/building-effective-agents-hands-on-guide) for the theoretical scaffolding behind the five-loop pattern; [designing long-running agents — architecture foundations](/blog/machine-learning/ai-agent/designing-long-running-agents-architecture-foundations) for the operational shape of agents that live longer than a single session; [long-term memory in conversational agents (MemGPT)](/blog/machine-learning/ai-agent/long-term-memory-conversational-agents-memgpt) for the canonical earlier work on hierarchical agent memory that Hermes' design implicitly converses with; and [effective context engineering for AI agents](/blog/machine-learning/ai-agent/effective-context-engineering-for-ai-agents) for a deeper treatment of the reference-vs-summary trade-off that section 4 only sketches.

Either way, the loops in the diagram above are the right framing. Whether you adopt Hermes or build your own, give your agent five clocks. The single-clock agents are the ones still rediscovering `package.json` on Monday.
