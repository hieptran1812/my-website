---
title: "Web-Bench: Why LLMs Still Can't Build a Real Web App"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "A deep dive into Web-Bench, ByteDance's project-level benchmark of 50 web apps built across 20 dependent tasks each, why the best model passes only 25.1% of tasks, and what error compounding across a task chain tells us about agentic coding."
tags: ["llm", "ai-agent", "benchmark", "evaluation", "code-generation", "web-development", "bytedance", "agentic-coding", "pass-at-1", "frameworks"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 51
---

The most quietly misleading number on any model card is the coding score. A frontier model ships with a HumanEval Pass@1 in the high nineties and an MBPP number not far behind, and the implied claim is that the model "can code." Then you put it in front of a real task — *take this React app and add a filterable, sortable, paginated data table that talks to the existing store* — and it produces something that compiles, looks plausible, and breaks the third feature you ask for because it quietly renamed a selector the rest of the app depends on. The gap between the benchmark and the job is not a rounding error. It is the entire problem.

The reason is structural. HumanEval and MBPP grade **isolated functions**: one prompt, one self-contained function, one set of unit tests, no shared state with anything else. That is a parameter-shaped problem, and modern models are genuinely excellent at it — so excellent that the benchmarks are saturated. HumanEval Pass@1 has reached **99.4%** and MBPP **94.2%**. When a benchmark is at 99.4%, it has stopped measuring the models and started measuring the test harness. Real software, by contrast, is a **sequence of dependent edits to a growing codebase**: feature 2 reads the DOM contract feature 1 wrote; feature 7 imports the module feature 3 created; the data table you add at task 12 has to coexist with the routing you set up at task 3 and the state store you wired at task 5. State accumulates, and so do mistakes.

![One Web-Bench project shown as a pipeline of 20 dependent tasks: an init scaffold feeds task 1, which feeds task 2, through tasks 4 to 19, to task 20, then to a Playwright end-to-end gate](/imgs/blogs/web-bench-llm-web-development-benchmark-1.webp)

The diagram above is the mental model for the benchmark this post is about. [Web-Bench](https://arxiv.org/abs/2505.07473) — "Web-Bench: A LLM Code Benchmark Based on Web Standards and Frameworks," ByteDance — takes the *sequence* seriously. A Web-Bench instance is not a prompt; it is a **project**: 50 of them, each consisting of **20 tasks with sequential dependencies**, where task $N$ edits the exact code state task $N-1$ left behind. Each project is the kind of thing a senior engineer would spend **4 to 8 hours** building. The headline result is brutal and clarifying: the SOTA model — Claude 3.7 Sonnet, run with best-of-5 thinking — passes only **25.1%** of tasks on the first try. If you have been assuming that a 99.4% HumanEval number means the model can build a real app, Web-Bench is the benchmark that should change your mind.

## TL;DR

> **What Web-Bench is.** A project-level LLM coding benchmark from ByteDance: **50 projects × 20 sequentially-dependent tasks = 1,000 tasks**, designed by engineers with 5–10 years of experience. Each project is a real web app (average **~1,948 lines of code**, max ~6,180) covering both raw **Web Standards** (DOM, CSS Flex/Grid, Canvas, SVG, ECMAScript, TypeScript) and popular **Web Frameworks** (React, Vue, Angular, Svelte, Next, Nuxt, Express, Redux, Prisma, and more). Tasks are scored by **Playwright end-to-end tests** — on average **3.6 cases per task, 72.4 per project**, roughly **3,600 test cases** total.
>
> **The result that matters.** The SOTA, Claude 3.7 Sonnet (best-of-5, thinking), reaches **25.1% Pass@1**. For context, that is *lower (harder)* than the same model's SWE-Bench Verified (65.4%) and Full (33.8%) numbers — Web-Bench is the harder benchmark. Closed models average **15.08% Pass@1 / 20.79% Pass@2**; open models average **10.73% / 14.84%**. Project-level web development is far from solved.
>
> **Why it's hard.** Error compounds across the 20-task chain: a wrong DOM contract or a renamed export at task $N$ is silently inherited by every downstream task. Late tasks also carry long-context project state and must stay consistent with framework idioms and everything already shipped. The benchmark is built so that the *dependency structure itself* is the difficulty, not the individual task.
>
> **The metric that mirrors humans.** Pass@2 — retry once *with the build/test error context* — is the metric closest to how a real engineer works, and it lifts every model by several points. The gap between Pass@1 and Pass@2 is a measurement of how well a model uses its own error feedback.
>
> **Who should care.** Anyone choosing a model for an agentic coding product, anyone building a coding agent harness (the Web-Agent loop is a clean template), and anyone tired of saturated snippet benchmarks. Companion posts in this series: the [Model Atlas hub](/blog/machine-learning/bytedance-research-model-atlas), [PaSa](/blog/machine-learning/ai-agent/pasa-llm-paper-search-agent), the [ToolHop multi-hop benchmark](/blog/machine-learning/ai-agent/toolhop-multi-hop-tool-use-benchmark), and [evaluating agent trajectories beyond the final answer](/blog/machine-learning/ai-agent/evaluating-agent-trajectories-beyond-final-answer).

This is a deep-dive. We will work through *why* snippet benchmarks saturated and project benchmarks didn't, how Web-Bench is constructed and why the sequential dependency is the whole design, the Web-Agent harness that runs it (with runnable skeleton code), the full results once you read past the headline, the error-compounding dynamics that make late tasks so much harder than early ones, a set of named case studies of how projects actually break, and a critique that states plainly what would change my mind about the benchmark's conclusions.

## 1. Why snippet benchmarks saturated and projects didn't

> **Senior rule of thumb:** a benchmark that the best model clears 99% of the time is no longer measuring capability — it is measuring how cleanly the questions were written. The signal has moved to the residual 1%, which is mostly ambiguity, not difficulty.

Start with the assumption baked into most coding evals and the reality that breaks it.

| Assumption (snippet benchmarks) | Reality (real software) |
| --- | --- |
| One prompt produces one self-contained function | One feature edits a codebase that already exists and must keep existing |
| Tests check the function in isolation | Tests check the *whole app* end-to-end, including features built earlier |
| No shared state between problems | State accumulates: DOM structure, exports, routes, store shape, types |
| A mistake affects one score | A mistake at task $N$ corrupts the input to tasks $N+1 \ldots 20$ |
| Difficulty ≈ algorithmic cleverness | Difficulty ≈ consistency, context length, framework idiom, integration |
| Saturates near 100% | SOTA sits at ~25% Pass@1 |

HumanEval is 164 hand-written Python problems, each a docstring plus a function signature, graded by a handful of unit tests. MBPP is ~1,000 similar "basic programming problems." They were excellent benchmarks in 2021. The trouble is that the *shape* of the problem — produce one correct pure function from a clear spec — is exactly the shape that instruction-tuned code models are now overfit to. When a model has seen ten thousand variations of "write a function that returns the n-th Fibonacci number," the benchmark stops discriminating.

![A before-after comparison: the left column shows snippet benchmarks with one prompt producing one function, no shared state, HumanEval at 99.4% and MBPP at 94.2%; the right column shows Web-Bench with 20 dependent tasks per project, ~1948 lines carried, error compounding, and SOTA Pass@1 at 25.1%](/imgs/blogs/web-bench-llm-web-development-benchmark-2.webp)

The figure above is the core argument of the paper in one image. On the left is the saturated regime: a prompt, a function, isolated tests, scores in the nineties. On the right is what Web-Bench measures: twenty tasks that share a living codebase of roughly two thousand lines, where a mistake doesn't just fail one test — it *compounds* into every task that depends on the code state it corrupted. The right column is not a harder version of the left column. It is a different axis of difficulty entirely. You can be near-perfect at writing isolated functions and still fail to ship a coherent app, because shipping an app is a problem of *integration over time*, not *correctness in isolation*.

This is why the authors are careful to frame Web-Bench against **SWE-Bench** rather than HumanEval as the relevant comparison. SWE-Bench already moved the field toward real repositories and real GitHub issues, and that was the right direction. But SWE-Bench tasks are typically *one* fix to an *existing, mature* codebase — find the bug, patch it, pass the regression test. Web-Bench's twist is that the codebase doesn't pre-exist: the model **builds it**, task by task, and every task it gets wrong degrades the foundation for the rest. The relevant numbers make the point: Claude 3.7 Sonnet scores **65.4%** on SWE-Bench Verified and **33.8%** on SWE-Bench Full, but only **25.1%** on Web-Bench. Lower is harder. Web-Bench is harder than the benchmark the field currently treats as its frontier.

There is a second-order reason snippet benchmarks saturated that is worth naming: **contamination**. HumanEval and MBPP have been in the public domain long enough that their solutions are almost certainly in pretraining corpora. A benchmark whose answers leak into training measures memorization, not capability. Web-Bench's projects are newly authored, and — more importantly — even a leaked *project* doesn't help much, because the evaluation rewards building the project incrementally under a specific task decomposition with specific intermediate contracts. There is no single "answer" to memorize; there are twenty dependent answers, each conditioned on the model's own (possibly wrong) prior outputs. That structure is inherently resistant to contamination in a way a single function never is.

## 2. The anatomy of a Web-Bench project {#anatomy}

> **Senior rule of thumb:** the unit of real software is not the function — it is the *change set applied to a state*. Any benchmark that doesn't model state and sequence is measuring a different job than the one you actually do.

A Web-Bench project has three things working together: an **initial code scaffold**, **20 tasks in a fixed order**, and a set of **end-to-end tests** that grow as the project grows. Look again at the first figure: `init code` is a small starting repository (an empty Vite project, a bare HTML page, a framework scaffold). Then task 1 might be "build the page layout shell," task 2 "add the `<main>` content region," task 3 "wire up client-side routing and a state container," and so on, each one a feature an engineer would commit as a single PR. Task 20 is the final feature. Between every pair of tasks flows the **code state** — the actual files on disk that the next task must read and extend.

The crucial design decision is that **tasks are not independent prompts**. The paper's own Figure 1 makes this explicit with a minimal example: *Task 2 depends on the execution result (the `<main>` element) of Task 1*. If task 1 produced a `<main id="content">` element, task 2's tests assume that element exists and target it. If the model building task 1 instead emitted `<div class="content-area">`, task 1 might still pass its own loose tests, but task 2 now operates on a contract that doesn't exist. This is exactly how real codebases break: not with a syntax error, but with a *silent contract violation* that surfaces three commits later.

Some concrete numbers ground the scale. Across the 50 projects:

| Property | Value |
| --- | --- |
| Projects | 50 |
| Tasks per project | 20 (fixed, sequential) |
| Total tasks | 1,000 |
| Avg. lines of code per project | ~1,948 (max ~6,180) |
| Avg. test cases per task | 3.6 |
| Avg. test cases per project | 72.4 (max ~99) |
| Total test cases (approx.) | ~3,600 |
| Human reference time per project | 4–8 hours (senior engineer) |
| Authoring experience | engineers with 5–10 years |

Two thousand lines is not a toy. It is the size at which a single file no longer holds the whole app, at which you need cross-file consistency, at which renaming a thing in one place quietly breaks three others. And 72 test cases per project means the grading is not a smoke test — it is a battery of end-to-end assertions about behavior. A model can't pass by producing code that *looks* right; it has to produce code that *behaves* right when a headless browser drives it.

It is worth dwelling on why "4–8 hours for a senior engineer" is the right anchor. The benchmark is not trying to test whether a model can write a one-liner faster than a human; it is trying to test whether a model can sustain *correct incremental development* over the kind of time horizon where humans rely on memory of their own earlier decisions. A human building task 12 remembers that they named the store `useCartStore` back at task 5, that the router uses hash-mode, that the date formatter lives in `utils/format.ts`. The model has no such memory except what fits in its context window — which is precisely the constraint Web-Bench is built to stress.

### 2.1 The sequential dependency is the benchmark

If you strip away the web-specific surface, the deep idea is this: Web-Bench is an **autoregressive evaluation over code state**. The model's output at step $N$ becomes part of the input at step $N+1$. This is structurally the same trap that makes long-form generation hard — errors don't average out, they accumulate — except here the "tokens" are entire feature implementations and the "perplexity" is whether the end-to-end tests pass.

Formally, if $p_n$ is the probability that the model completes task $n$ correctly *given a correct prior state*, then the probability of completing the first $k$ tasks with no compounding help is bounded above by $\prod_{n=1}^{k} p_n$. Even with a generous per-task success probability of $p_n = 0.85$, the chance of getting twenty tasks right in a row is $0.85^{20} \approx 0.039$ — under 4%. This is the arithmetic that makes project-level benchmarks brutal: *individually easy tasks become collectively near-impossible the moment they depend on each other.* It is also why Pass@1, measured per-task, lands at 25% even though no single task is individually that hard for a senior engineer.

This is the same dynamic we explore from the trajectory side in [evaluating agent trajectories beyond the final answer](/blog/machine-learning/ai-agent/evaluating-agent-trajectories-beyond-final-answer): the final state is a poor proxy for the quality of the *path* that produced it, and a benchmark that only graded the finished app would miss exactly where models fall down. Web-Bench grades every task, which means it grades the path.

## 3. What the 50 projects actually cover {#coverage}

> **Senior rule of thumb:** "can it code web?" is the wrong question. The right question is "can it code web *the way it's actually built* — on top of frameworks, with their idioms, their build steps, and their footguns?"

Web development is not one skill. It is a stack of standards with an ecosystem of frameworks layered on top, and being fluent in one does not imply fluency in the other. Web-Bench's project set is deliberately split to test both.

![A taxonomy tree: the root Web-Bench 50 projects branches into Web Standards (~19 projects) covering DOM/BOM/Form, Flex/Grid/Float, and Canvas/SVG/WebGL; and Web Frameworks (~31 projects) covering React/Vue/Svelte, Redux/Zustand/Mobx, and Next/Nuxt/Prisma](/imgs/blogs/web-bench-llm-web-development-benchmark-5.webp)

The tree above is the coverage map. The **Web Standards** branch (roughly 19 projects) tests the model against the raw browser platform with no framework safety net: the **DOM** (document object model — node trees and events), the **BOM** (browser object model — `window`, `history`, `location`), HTML **forms** and validation, CSS layout (**Flexbox** for one-dimensional layout, **Grid** for two-dimensional, **Float** for the legacy flow model, plus selectors and color), graphics (**Canvas** for raster drawing, **SVG** for vector graphics, **WebGL** for GPU-accelerated rendering), and the language layer (**ECMAScript** modules and **TypeScript**). Project names in this branch include Table, Flex, Grid, Float, Color, Selector, BOM, DOM, Form, ESModule, Canvas, SVG, SVG-Solar, SVG-Chart, TypeScript, Survey, Draw, and ChartBuilder.

The **Web Frameworks** branch (roughly 31 projects) tests the model on the abstractions engineers actually ship on. The paper anchors these to real-world popularity by GitHub stars, which is a nice touch — it is testing the frameworks people use, not obscure ones:

- **UI frameworks:** React (~229K stars), Vue (~208K), Angular (~96K), Svelte (~79K), Three.js (~103K).
- **State management:** Redux (~61K), Zustand (~50K), MobX (~28K), Jotai (~20K).
- **Fullstack frameworks:** Next.js, Nuxt, Express.js, Fastify (~33K).
- **CSS preprocessing / utility:** SASS, LESS, Stylus, Tailwind.
- **Build tools:** Vite, Webpack, Parcel.
- **Databases / ORMs:** Prisma, Sequelize, Lowdb, MongoDB.

This breadth matters because framework competence is a *separate* failure surface from standards competence. A model can know the DOM cold and still write a React component that mutates state directly, or a Vue template that misuses reactivity, or a Next.js page that confuses server and client components. Each framework has idioms that are not derivable from the underlying standards, and the only way to test idiomatic fluency is to make the model build something nontrivial inside the framework's conventions.

To see the second axis of coverage — application domain, crossed with the tooling choice — consider the grid below.

![A coverage grid with three application domains as rows (rendering, interaction, data/fullstack) and tooling choices as columns: rendering crosses Canvas/SVG charts, Three.js 3D, and Tailwind/SASS; interaction crosses DOM/BOM/Form, React/Vue, and Redux/Zustand; data crosses ESModule/TypeScript, Next/Nuxt/Express, and Prisma/MongoDB](/imgs/blogs/web-bench-llm-web-development-benchmark-8.webp)

The grid makes the design intent legible: each project lives at the intersection of an **application domain** (what kind of app — a rendering-heavy visualization, an interaction-heavy form app, a data-heavy fullstack service) and a **concrete tooling choice** (which standard or framework). A model that is strong on interaction-domain DOM work might be weak on rendering-domain WebGL; a model that handles React state cleanly might fumble a Prisma schema migration. By spreading projects across this grid, Web-Bench resists the failure mode where a benchmark accidentally rewards one narrow competence.

### 3.1 Second-order: framework idioms are an adversarial surface

Here is the gotcha that breaks intuition. You might expect frameworks to make the model's job *easier* — they exist to reduce boilerplate, after all. In practice, frameworks are often *harder* for models because idiomatic framework code is a moving target across versions. React's recommended patterns changed with hooks, then with concurrent features, then with server components. A model trained on a corpus spanning all of these will happily mix class components with hooks, or call a hook conditionally, or pass a stale closure to an effect. The standards layer, by contrast, is far more stable — `document.querySelector` has meant the same thing for a decade. So the framework branch is not just "more of the same difficulty"; it adds a *version-confusion* failure mode that the standards branch largely avoids.

## 4. The Web-Agent: the harness that runs the benchmark {#web-agent}

> **Senior rule of thumb:** a coding benchmark is only as honest as its execution harness. If you grade generated code by reading it, you measure plausibility; if you grade it by running it against end-to-end tests, you measure behavior. Always grade behavior.

Running Web-Bench requires more than calling a model once. It requires a loop that builds a prompt, calls the model, extracts the files the model wants to write, applies them to the project, builds and runs the project, executes the end-to-end tests, and — critically — retries once with the error context if the first attempt fails. ByteDance calls this loop the **Web-Agent**.

![A five-layer stack of the Web-Agent evaluation pipeline, top to bottom: Build Prompt (system prompt + task description + current files + error context), Request LLM (OpenAI-style API, truncate if over context), Extract Files (parse response, rewrite project files), Init + Build (npm install, bundler, optional steps), and Score (Playwright E2E, ~3.6 cases per task, retry once)](/imgs/blogs/web-bench-llm-web-development-benchmark-6.webp)

The stack above shows the layers. Read it top to bottom because that is the order of execution within a single task attempt, but note that **scoring sits at the bottom on purpose**: the Playwright gate, not the model, decides whether a task passed. The model never grades itself.

1. **Build Prompt.** The agent assembles a prompt from the system prompt (SP), the current task's natural-language description, the *current state of the project files*, and — on a retry — the error messages from the failed build or test run. This is the step where the sequential dependency becomes concrete: the "current files" are whatever the previous tasks produced, mistakes and all.
2. **Request LLM.** The agent calls the model through an OpenAI-style API. If the assembled input exceeds the model's context length, **it is truncated** — which is itself a source of late-task failures, since a 2,000-line project plus task description plus error context can overflow a small context window.
3. **Extract Files.** The agent parses the model's response to extract the generated files. Models are prompted to emit files in a parseable format; the harness rewrites the corresponding files in the project directory.
4. **Init + Build.** Optionally re-initialize the environment (`npm install` for new dependencies) and run the build (the bundler — Vite/Webpack/Parcel — depending on the project). A build failure here is a failure the retry can use.
5. **Score.** Run the **Playwright** end-to-end test suite for the tasks completed so far. Playwright drives a real headless browser, so the tests assert *observed behavior* — does clicking the button add a row, does the chart render the right number of bars, does the form reject invalid input. On average 3.6 cases gate each task.

If the first attempt fails at build or test, the agent retries **once** with the error context appended to the prompt, then reports. Two attempts maximum. That two-attempt cap is what defines the two headline metrics.

### 4.1 A runnable Web-Agent loop

Here is the harness skeleton in Python, close enough to the real shape that you could adapt it to run your own project-level eval. The key ideas: the project directory is *mutated in place* across tasks, the prompt carries the current file state, and scoring is delegated to an external test runner.

```python
import json, subprocess, pathlib
from openai import OpenAI

client = OpenAI()  # OpenAI-style endpoint; swap base_url for Claude/DeepSeek/etc.

SYSTEM_PROMPT = (
    "You are a senior web engineer. You are given the current state of a "
    "project and a task. Implement ONLY the task. Return each file you "
    "create or modify as a fenced block:\n"
    "```file:<relative/path>\n<full file contents>\n```"
)

def read_project(root: pathlib.Path) -> dict[str, str]:
    """Snapshot every source file the model is allowed to see."""
    files = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in {".ts", ".tsx", ".js", ".jsx",
                                        ".html", ".css", ".json", ".vue"}:
            if "node_modules" not in p.parts:
                files[str(p.relative_to(root))] = p.read_text()
    return files

def build_prompt(task_desc: str, files: dict[str, str], err: str | None) -> str:
    blob = "\n".join(f"// === {name} ===\n{body}" for name, body in files.items())
    parts = [f"# Current project files\n{blob}", f"# Task\n{task_desc}"]
    if err:                       # retry path: feed the failure back in
        parts.append(f"# Previous attempt failed with:\n{err}\n"
                     f"Fix the task so all end-to-end tests pass.")
    return "\n\n".join(parts)

def extract_files(response: str) -> dict[str, str]:
    """Parse ```file:<path> ... ``` blocks out of the model output."""
    out, lines, cur, path = {}, response.splitlines(), [], None
    for ln in lines:
        if ln.startswith("```file:"):
            path = ln[len("```file:"):].strip(); cur = []
        elif ln.startswith("```") and path is not None:
            out[path] = "\n".join(cur); path = None
        elif path is not None:
            cur.append(ln)
    return out

def apply_files(root: pathlib.Path, files: dict[str, str]) -> None:
    for rel, body in files.items():
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(body)       # mutate the project IN PLACE — state carries forward

def run_e2e(root: pathlib.Path, up_to_task: int) -> tuple[bool, str]:
    """Delegate scoring to Playwright; the model never grades itself."""
    proc = subprocess.run(
        ["npx", "playwright", "test", f"--grep=@task<= {up_to_task}"],
        cwd=root, capture_output=True, text=True, timeout=600,
    )
    return proc.returncode == 0, (proc.stdout + proc.stderr)[-4000:]

def run_task(root, task_desc, model, idx) -> str:
    """One task = up to TWO attempts. Returns 'pass@1' | 'pass@2' | 'fail'."""
    err = None
    for attempt in (1, 2):                       # two-attempt cap
        files = read_project(root)               # current code STATE
        prompt = build_prompt(task_desc, files, err)
        if num_tokens(prompt) > context_limit(model):
            prompt = truncate(prompt, context_limit(model))   # late-task hazard
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
        ).choices[0].message.content
        apply_files(root, extract_files(resp))
        subprocess.run(["npm", "install"], cwd=root)          # init + build
        ok, err = run_e2e(root, up_to_task=idx)               # score
        if ok:
            return "pass@1" if attempt == 1 else "pass@2"
    return "fail"

def run_project(root, tasks, model) -> list[str]:
    return [run_task(root, t, model, i + 1) for i, t in enumerate(tasks)]
```

Three things in this skeleton are worth internalizing. First, `apply_files` mutates the project **in place** — there is no reset between tasks, which is what makes the state carry forward and the errors compound. Second, the retry on attempt 2 feeds `err` (the truncated tail of the build/test output) back into the prompt; this is the entire mechanism behind Pass@2. Third, `run_e2e` greps tests `@task<= idx`, meaning a later task can re-break an earlier task's tests — exactly the regression behavior real apps have.

### 4.2 Pass@1 vs Pass@2: the metric that mirrors a human

The two metrics are defined per-task and averaged:

$$\text{pass@1} = \frac{\text{tasks passing all E2E tests on the first attempt}}{\text{tasks}} \times 100\%$$

$$\text{pass@2} = \frac{\text{tasks passing all E2E tests within two attempts}}{\text{tasks}} \times 100\%$$

The paper frames it precisely: **Pass@2 (retry with error context) is closer to the behavior of human engineers.** No competent engineer ships the first thing they write without running it; they run it, read the error, and fix it. Pass@2 measures whether the model can do that loop — read a Playwright failure or a TypeScript build error and turn it into a correct fix. The gap between Pass@1 and Pass@2 is therefore not just "a second roll of the dice." It is a *capability measurement*: how well does the model use its own execution feedback?

This connects directly to the broader agent-evaluation literature. The ability to read an error and self-correct is one of the load-bearing skills in [building effective agents](/blog/machine-learning/ai-agent/building-effective-agents-hands-on-guide); a model that ignores its tool's error output is going to be unreliable in any agentic loop, not just this one. Web-Bench gives you a clean, quantified read on that skill — the Pass@2 minus Pass@1 delta — that most benchmarks don't isolate.

### 4.3 Why Playwright, and what "3.6 cases per task" buys you {#scoring}

It's worth slowing down on the scoring layer, because the choice of grader is what separates an honest coding benchmark from a vibes-based one. Web-Bench scores with **Playwright**, a browser-automation framework that drives a real headless Chromium (or Firefox/WebKit) instance against the running app. A Playwright test does not read the source code and judge it; it *operates the app* — it navigates to the page, clicks elements, fills inputs, waits for DOM mutations, and asserts on what it observes. A representative test for a "remove item" feature looks like this:

```typescript
import { test, expect } from "@playwright/test";

test("@task<=7 removing an item updates the list and the count", async ({ page }) => {
  await page.goto("http://localhost:3000");
  // state from earlier tasks: three seeded rows
  await expect(page.locator("li.item")).toHaveCount(3);
  // exercise the task-7 feature: click the delete button on the first row
  await page.locator("li.item").first().locator("button.remove").click();
  // behavioral assertion: the DOM actually changed
  await expect(page.locator("li.item")).toHaveCount(2);
  // integration assertion: the count display (task 20) stays consistent
  const total = await page.locator("#total-count").textContent();
  expect(total).toBe("2");
});
```

Notice three properties this test has that a unit test of a `removeItem()` function would not. First, it asserts on **observed behavior** through a real DOM — if the model's code throws, renders nothing, or mutates the wrong element, the assertion fails regardless of how plausible the source looks. Second, the `@task<=7` tag means this test runs for task 7 *and every task after it*, so a later task that breaks the remove feature is caught as a regression. Third, the final assertion is an **integration check** that couples task 7 (remove) to a feature defined much later (the total count) — the kind of cross-feature consistency that only end-to-end testing catches.

The "3.6 cases per task, 72.4 per project" figure is doing real work here. With only one test per task, a model could pass by satisfying a narrow happy-path assertion while leaving edge cases broken; the next task that hits the edge case would then fail mysteriously. With ~3.6 cases per task — happy path, an edge case or two, and often a regression check against earlier behavior — the grader pins down the *contract* of each feature tightly enough that a model can't slip a half-correct implementation through. The cost is real: ~3,600 browser-driven assertions across 50 projects is slow and resource-hungry compared to running a few thousand Python unit tests. That cost is the price of grading behavior instead of plausibility, and it's the right trade for a model-selection benchmark.

There's a subtle harness detail worth flagging for anyone building something similar. Because the project is mutated in place and the test suite grows, the scoring step for task $N$ re-runs the *cumulative* suite — tasks 1 through $N$ — not just task $N$'s new cases. This is what makes regressions count against the score. A naive harness that only ran the newest task's tests would report a flattering, regression-blind number; Web-Bench's cumulative grading is deliberately less forgiving, and it is the more faithful model of what a CI pipeline does on a real project.

## 5. The results, read past the headline {#results}

> **Senior rule of thumb:** the headline number tells you the ceiling. The *spread* tells you the story. When the best model is at 25% and the open-source average is at 11%, the interesting question is not "who won" but "why is everyone losing."

The headline is that Claude 3.7 Sonnet — best-of-5, with extended thinking — tops the leaderboard at **25.1% Pass@1**. The more informative numbers are the aggregates and the ranking.

![A matrix of models by metrics: rows are Claude 3.7 Sonnet, GPT family, Doubao, DeepSeek, closed average, and open average; columns are Pass@1, Pass@2 (retry), and rank tier; Claude tops at 25.1% Pass@1 and ~31% Pass@2, closed models average 15.08%/20.79%, open models average 10.73%/14.84%](/imgs/blogs/web-bench-llm-web-development-benchmark-3.webp)

The matrix above lays out the landscape. The aggregate numbers the paper reports are the ones to anchor on, because individual per-model figures are presented graphically in the paper's figures rather than in a clean table:

| Group | Pass@1 | Pass@2 | Notes |
| --- | --- | --- | --- |
| Claude 3.7 Sonnet (best-of-5, thinking) | **25.1%** | ~31% | SOTA; lower (harder) than its SWE-Bench |
| Closed-source models (average) | **15.08%** | **20.79%** | retry buys ~+5.7 points |
| Open-source models (average) | **10.73%** | **14.84%** | retry buys ~+4.1 points |

And the family ranking, which the paper states explicitly:

- **By Pass@1:** Claude > GPT > Doubao > DeepSeek > LLaMA > Gemini > Qwen > others.
- **By Pass@2:** Claude > Doubao > DeepSeek > GPT (the ranking *shifts* once retry is allowed).

That ranking shift is one of the most interesting findings in the whole paper, and it is easy to miss. On the first attempt, GPT-family models edge out Doubao and DeepSeek. But once you allow a single retry with error context, **Doubao and DeepSeek overtake GPT**. The plain reading: Doubao and DeepSeek are *better at using error feedback to self-correct* than the GPT models are, even though GPT is marginally better at one-shot generation. If you are choosing a model for an agentic coding harness — where the agent will always have a chance to read errors and retry — the Pass@2 ranking is the one that matters to you, not Pass@1. A model that one-shots slightly worse but recovers much better is the better agent backbone.

The Claude-on-top result is also worth taking seriously rather than dismissing as "of course." The same model leads on SWE-Bench. The consistency across two very different project-level benchmarks (real-repo bug-fixing vs. greenfield incremental building) suggests Claude's lead is about *sustained coding coherence over long contexts*, not about any single benchmark's quirks. That is a more durable kind of advantage than topping a snippet benchmark.

The closed-versus-open gap deserves its own read. Closed models average **15.08%** Pass@1 against open models' **10.73%** — a roughly 4.3-point gap that *widens* to nearly 6 points on Pass@2 (20.79% vs 14.84%). That widening is the interesting part: it says closed models are not just better at one-shot generation, they are *better at recovering from their own errors* than open models are, on average. If the gap were purely about raw generation quality, you'd expect it to stay constant under retry; the fact that it grows means closed models extract more value per unit of error feedback. For a team deciding between a hosted frontier model and a self-hosted open one for a coding agent, that compounding gap is the number to weigh — not the headline Pass@1, but how much each model improves when you give it a second look at its own mistakes. The open models are closing the one-shot gap faster than they're closing the recovery gap, which is a useful thing to know when you're betting on where the open ecosystem will be in six months.

One more nuance on the family ranking: it is reported at the *family* granularity (Claude, GPT, Doubao, DeepSeek, LLaMA, Gemini, Qwen) rather than per-checkpoint, which is the right level of abstraction for a model-selection decision but does smear over within-family variance. A reasoning-tuned variant of a family can behave quite differently from its base sibling on a benchmark this sensitive to self-correction, so treat the family ranking as a prior, not a verdict, and validate the specific checkpoint you intend to ship.

Now the number that should reframe the whole field: **25.1% is lower than SWE-Bench.** The SOTA on SWE-Bench Verified is in the mid-sixties. The community has, fairly, treated SWE-Bench as the hard frontier of "can models do real software engineering." Web-Bench says: the frontier is further out than that. Greenfield incremental development across twenty dependent tasks is *harder* than patching a bug in a mature repo, because there is no existing scaffold of correct code to lean on — the model has to produce that scaffold itself and then not break it.

### 5.1 Second-order: best-of-5 is doing real work

Notice the SOTA caveat: **best-of-5**. The 25.1% figure is the best of five sampled runs, not a single run. This is an honest disclosure, but it tells you something about variance: project-level coding is high-variance, and sampling multiple times and taking the best meaningfully raises the ceiling. For a practitioner this is actionable — if you are running an agent on a hard, long project, sampling multiple candidate implementations of a risky task and selecting by test-pass is a real lever, not a benchmark artifact. It is the inference-time analog of the retry mechanism: more shots at the same task, selected by an objective gate.

## 6. Error compounding: why late tasks are so much harder {#cascade}

> **Senior rule of thumb:** in a dependent task chain, the failure you should fear is not the one that crashes — it's the one that *passes its own test while violating a downstream contract.* The crash you fix in five minutes; the silent contract drift you debug for an hour at task 17.

This is the heart of the matter, and it is the reason Web-Bench is a fundamentally different kind of benchmark. In a snippet benchmark, every problem is independent — getting one wrong costs you exactly one point. In Web-Bench, getting one task wrong can cost you *every task downstream of it.*

![A dataflow graph showing failure cascade: task N (write the #main API) branches into a correct contract (green) or a wrong selector (red); both feed task N+1 (reads #main), which feeds task N+2 (appends child); task N+2 forks to E2E pass (green) or E2E fail cascade (red)](/imgs/blogs/web-bench-llm-web-development-benchmark-4.webp)

The graph above traces one cascade. At task $N$, the model writes the code that exposes a DOM contract — say, a `<div id="main">` that later tasks will query and append children to. If it writes the **correct contract** (green path), task $N+1$ reads `#main` successfully, task $N+2$ appends a child, and the end-to-end tests pass. If it writes a **wrong selector** (red path) — `<div class="main">` instead of `id="main"`, or `#content` instead of `#main` — then task $N+1$ *inherits* that broken contract. The query for `#main` returns null. Task $N+2$ tries to append a child to null and throws. The E2E test for $N+2$ fails — not because task $N+2$ was implemented wrong, but because the foundation it stood on was poured wrong two tasks earlier.

This is the cascade. And it has a particularly nasty property: the failure surfaces *downstream of its cause*. The model (and a naive harness) sees the test for task $N+2$ failing and naturally tries to fix task $N+2$ — which is correct. The actual bug is in task $N$, which already "passed." This is the single hardest debugging pattern in real software, and Web-Bench reproduces it faithfully because it grades the project incrementally with growing tests.

The timeline below shows what this does to pass rate as the task index climbs.

![A timeline of pass-rate decay across the 20 tasks: tasks 1-4 on a clean scaffold have a high pass rate (green), tasks 5-8 see first regressions (green), tasks 9-12 with cross-file dependencies dip (amber), tasks 13-16 face context pressure as state grows (amber), and tasks 17-20 with long context have the lowest pass rate (red)](/imgs/blogs/web-bench-llm-web-development-benchmark-7.webp)

The timeline above is the qualitative shape of difficulty over the task index. Early tasks (1–4) operate on a small, clean scaffold; there is little prior state to be inconsistent with, the context is short, and the pass rate is highest. As features stack (5–8), the first regressions appear — a new feature breaks an old test. By the middle (9–12), cross-file dependencies dominate and the pass rate dips. As the project state grows (13–16), the model is under **context pressure**: the project plus task description plus error context starts to crowd the context window, and the harness may truncate. By the final tasks (17–20), the model is reasoning over a long, possibly-truncated context full of code it didn't fully attend to, and the pass rate bottoms out.

Three distinct mechanisms drive this decay, and it's worth separating them because they have different mitigations:

1. **Inherited error (the cascade).** A wrong contract early poisons everything downstream. *Mitigation:* better early-task correctness; sampling and selecting risky foundational tasks; explicit contract tests early.
2. **Regression.** A new feature breaks an existing one because the model didn't account for the old behavior. *Mitigation:* feeding the full existing test suite's expectations into the prompt; running tests incrementally (which Web-Bench does).
3. **Context pressure.** The growing project exceeds the usable context, and the model attends to the wrong parts or works from truncated state. *Mitigation:* longer context windows, retrieval over the codebase, structured summaries of project state — the same toolkit you'd use for any long-context agent.

The third mechanism is why context-window size and long-context attention quality show up as a coding-agent capability, not just a chat feature. A model with a 200K-token window that genuinely attends across it has a structural advantage on the late tasks of a Web-Bench project over a model with an 8K window that truncates. This is also where retrieval-augmented approaches to *your own codebase* earn their keep; the same retrieval machinery we discuss for documents (and which underpins a [vector database](/blog/machine-learning/ai-agent/vector-database)) applies to keeping the relevant slice of a 2,000-line project in the prompt.

### 6.1 Second-order: the cascade rewards conservatism

There is a subtle behavioral consequence. Because a wrong contract is so expensive, the *optimal* policy for an agent on a dependent chain is to be **conservative about contracts** — to establish minimal, stable interfaces early and avoid renaming or restructuring them later. Models that are "creative" — that refactor aggressively, rename things to be cleaner, restructure the file layout mid-project — pay a heavy cascade tax, because every rename is a chance to break a downstream reference. This inverts a habit that's often good in isolated code generation (clean refactors are nice!) but is actively harmful in a dependent chain. The benchmark, without saying so, is rewarding the engineering discipline of *not breaking interfaces you've already shipped.*

### 6.2 The arithmetic of compounding, made concrete

Let me put numbers on the cascade so the intuition is unambiguous. Suppose a model's *intrinsic* per-task success probability — its chance of implementing a task correctly given a clean, correct prior state — is some constant $p$. If tasks were independent (the snippet-benchmark world), the expected fraction of tasks passed would simply be $p$. But they are not independent. Once a task fails and corrupts the state, downstream tasks that depend on the corrupted contract fail too, even if the model would have implemented them correctly on a clean state.

A crude but illuminating model: say each failure has probability $q$ of poisoning the immediate next task (a localized contract break), and that poisoning clears once the harness's retry or a later task happens to overwrite the broken contract. Even this gentle assumption drags the *observed* pass rate well below $p$, because every failure spawns a probabilistic chain of secondary failures. The gap between the intrinsic $p$ and the observed pass rate *is* the cascade tax, and it grows with both $q$ (how contagious failures are) and the chain length. Web-Bench fixes the chain length at 20, which is long enough for the tax to dominate: a model with an intrinsic $p$ of, say, 40% can easily post an observed Pass@1 in the twenties once compounding eats the difference.

This reframes the 25.1% SOTA number. It is *not* a claim that the best model gets only a quarter of individual web tasks right in isolation — it is almost certainly better than that on a clean state. It is a claim that, when forced to build on its own accumulating output across twenty steps, only a quarter of the resulting task-states are correct. The benchmark measures *integration over a chain*, and the chain is where the points are lost. This is exactly why the fresh-state ablation I argue for in the critique section would be so valuable: it would split the observed 25.1% into "intrinsic per-task skill" and "cascade contamination," and those two numbers point at completely different engineering fixes.

There is a hopeful corollary buried in this arithmetic. If the dominant loss is cascade contamination rather than intrinsic per-task weakness, then the highest-leverage improvement is not "make the model smarter at each task" but "make the model better at *not corrupting state* and at *recovering corrupted state*." Those are tractable, harness-level interventions: stronger early contract validation, explicit interface tests run after foundational tasks, retrieval that surfaces the exact prior contract a task depends on, and retry policies that suspect upstream causes when a downstream task fails. The Pass@2 numbers — which lift every model by 4–6 points purely from one feedback-driven retry — are direct evidence that recovery is a real, learnable lever, not a lost cause.

## 7. Case studies: how Web-Bench projects actually break {#case-studies}

These are illustrative failure patterns — composites grounded in the benchmark's design and the failure modes the paper's structure exposes, written the way I'd write a postmortem. Each names the symptom, the wrong first hypothesis, the actual root cause, the fix, and the lesson.

### 1. The renamed selector

**Symptom:** Tasks 1–6 pass. Task 7 ("add a `removeItem` button to each row") fails its E2E test: clicking the button does nothing. **Wrong first hypothesis:** the click handler is wired wrong, so the model rewrites task 7's event binding three times. **Root cause:** back at task 4, the model refactored the row container from `<li data-id="...">` to `<div class="row">`, dropping the `data-id` attribute the remove handler needs to identify which item to delete. Task 4 passed because its own tests only checked that rows *rendered*, not that they carried an id. **Fix:** restore the `data-id` contract at the point it was dropped — a task-4 change, surfacing at task 7. **Lesson:** the cascade puts the cause and the symptom in different tasks. A harness that only retries the failing task can never fix this; you need the model to suspect upstream state.

### 2. The Vue reactivity trap

**Symptom:** A Vue project's filter feature (task 11) renders the unfiltered list no matter what's typed in the search box. **Wrong first hypothesis:** the filter function has a bug. **Root cause:** task 8 stored the items array as a plain property assigned once in `created()`, not as a reactive `ref`/`reactive`. The filter at task 11 mutates a derived array, but Vue never re-renders because the source was never reactive. The code is idiomatic *React* thinking applied to Vue. **Fix:** make the source reactive at task 8. **Lesson:** framework idioms are a separate competence. A model fluent in React's mental model will write subtly broken Vue, and the break surfaces several tasks after the idiom violation.

### 3. The context-window cliff

**Symptom:** Tasks 1–14 of a fullstack Next.js project pass cleanly. Task 15 onward, quality collapses — the model starts re-declaring functions that already exist, importing from paths that don't, and ignoring the established API route conventions. **Wrong first hypothesis:** the model "got worse" at the harder later tasks. **Root cause:** the project crossed ~1,500 lines around task 14. The harness's prompt — project files + task + error context — exceeded the model's context limit and got **truncated**, cutting off exactly the files (the API route handlers) the late tasks needed to be consistent with. **Fix:** a larger context window, or retrieval that selects only the relevant files into the prompt. **Lesson:** late-task failure is often a *harness/context* failure, not a reasoning failure. Measure your token budget against your project growth.

### 4. The TypeScript build wall

**Symptom:** A TypeScript project passes all *behavioral* tests for task 9 but the task is scored as failed. **Wrong first hypothesis:** a flaky test. **Root cause:** the model introduced an implicit `any` and a missing return type that the project's `tsc --strict` build rejects. The Init+Build step failed before the E2E tests even ran, so the task failed at the build gate. **Fix:** on retry, the build error is fed back; the model adds the annotations. **Lesson:** Web-Bench scores the *build* as part of the task. "It works when I bypass the type-checker" is not a pass. This is exactly why Pass@2 (retry with the build error) recovers cases like this — and why models that read build errors well climb the Pass@2 ranking.

### 5. The regression nobody asked for

**Symptom:** Task 16 ("add dark-mode toggle") passes its own tests, but the E2E suite for task 3 (routing) now fails. **Wrong first hypothesis:** the dark-mode CSS is broken. **Root cause:** to add the toggle, the model wrapped the app root in a new theme provider component and, in doing so, moved the router outlet inside it — changing the DOM nesting the routing tests assert on. A new feature silently broke an old one. **Fix:** add the theme provider *without* relocating the router outlet. **Lesson:** Web-Bench runs the cumulative test suite, so regressions are caught and counted. This is the single most important way it differs from "build it once and grade the end state" — it punishes regressions the way production punishes them.

### 6. The hallucinated dependency

**Symptom:** Task 12 of a state-management project fails the build with `Cannot find module 'immer'`. **Wrong first hypothesis:** a missing `npm install`. **Root cause:** the model decided to use Immer for immutable updates — a reasonable library — but the project is a *Redux-without-toolkit* project that never declared Immer as a dependency, and the task description didn't authorize adding one. The model invented a dependency to make its life easier. **Fix:** write the immutable update by hand with spread syntax, no new dependency. **Lesson:** project benchmarks constrain the *toolset*, not just the behavior. A model that reaches for an unlisted convenience library fails the build, which is correct — real codebases don't let you add a dependency just because it's nicer.

### 7. The Canvas coordinate-space confusion

**Symptom:** A Canvas charting project (task 14: "add axis labels") draws labels in the wrong place — overlapping the chart, off by a consistent offset. **Wrong first hypothesis:** the label positioning math is wrong. **Root cause:** task 6 set up the chart by translating the canvas context origin (`ctx.translate(margin, margin)`) and never resetting it. Task 14's label code computes positions in absolute canvas coordinates, unaware that the context origin was shifted eight tasks ago. **Fix:** account for the translated origin, or `save()`/`restore()` around the chart drawing. **Lesson:** stateful APIs like Canvas carry hidden state across tasks (the transform matrix). The model has to remember a side effect it set up much earlier — precisely the kind of long-range consistency the benchmark is built to stress.

### 8. The retry that made it worse

**Symptom:** Task 10 fails on attempt 1. On attempt 2 (with error context), it fails *differently* and worse — now two tests fail instead of one. **Wrong first hypothesis:** the retry mechanism is broken. **Root cause:** the error context pointed at a real failure, but the model "fixed" it by rewriting a large chunk of the file from scratch rather than making a surgical change, and the rewrite broke a part that had been working. **Fix:** prompt the model to make minimal edits; in a real agent, apply a diff rather than a full-file rewrite. **Lesson:** the Pass@2 mechanism is only as good as the model's discipline in using error feedback. Some models genuinely recover (the Doubao/DeepSeek Pass@2 climb); others thrash. This is exactly the self-correction skill the [agent-trajectory evaluation](/blog/machine-learning/ai-agent/evaluating-agent-trajectories-beyond-final-answer) post argues you must measure directly.

### 9. The SVG namespace gotcha

**Symptom:** An SVG project's task 9 ("add a dynamically-created circle on click") runs without error but nothing appears on screen. **Wrong first hypothesis:** the click handler doesn't fire. **Root cause:** the model created the element with `document.createElement('circle')` instead of `document.createElementNS('http://www.w3.org/2000/svg', 'circle')`. An SVG element created in the HTML namespace is invisible — it's in the DOM but not rendered. **Fix:** use the SVG namespace. **Lesson:** the standards branch has its own deep footguns that don't exist in framework-land (where the framework handles namespaces for you). A model strong on React can be weak on raw SVG, and the benchmark's standards projects expose exactly that.

### 10. The off-by-one in incremental tests

**Symptom:** A model passes 19 of 20 tasks in a project but fails task 20 ("show a total count of items"). **Wrong first hypothesis:** the count logic is wrong. **Root cause:** the count is correct, but task 20's E2E test expects the count to update *after a delete* — and the delete feature (task 13) used an optimistic UI update that removes the row from the DOM but never decrements the model's item array. The count reads from the array, which is stale. **Fix:** keep the model array as the single source of truth and re-derive the DOM from it. **Lesson:** the final task often acts as an *integration test* over the whole project, and it surfaces a state-management bug planted many tasks earlier. Getting 19 of 20 is not a near-miss; the 20th task is frequently the one that audits the whole chain's consistency.

### 11. The Next.js server/client boundary

**Symptom:** A Next.js project's task 8 ("add a click-to-expand accordion") builds fine but the accordion doesn't respond to clicks in the browser. **Wrong first hypothesis:** the event handler isn't attached. **Root cause:** the component was authored as a React Server Component (the default in the app router), and Server Components don't ship interactivity — `onClick` handlers are stripped because the component never hydrates on the client. The model forgot the `"use client"` directive at the top of the file. **Fix:** mark the interactive component as a client component. **Lesson:** the server/client boundary is a Next.js idiom with no analog in plain React, and it fails *silently* — the code compiles, renders static markup, and simply ignores interactivity. A model fluent in classic React will write components that look right and do nothing, and Web-Bench's behavioral test (clicking and asserting the expansion) is the only thing that catches it.

### 12. The build-tool config drift

**Symptom:** A project using a path alias (`@/components/...`) passes tasks 1–9, then task 10 fails the build with `Cannot resolve '@/components/Chart'`. **Wrong first hypothesis:** the import path is wrong. **Root cause:** task 10 was the first task to import across the alias from a new directory the model created, and that directory wasn't covered by the `tsconfig.json` `paths` mapping that an earlier task had set up with a narrow glob. The alias worked for the original directories and broke for the new one. **Fix:** broaden the path mapping, or use a relative import. **Lesson:** build configuration is part of the project state too, not just source files. A model has to keep the bundler/tsconfig consistent with the file layout it's creating — a kind of consistency that's invisible until a new file falls outside the existing config's assumptions.

### 13. The async race in the data layer

**Symptom:** A fullstack project with a Prisma-backed list (task 17) intermittently shows an empty list on first load, then populates on refresh. **Wrong first hypothesis:** a flaky test or a slow database. **Root cause:** task 14 set up the data fetch as fire-and-forget in a `useEffect` without awaiting or tracking loading state; the initial render reads an empty array before the fetch resolves. Earlier tasks happened to render after the data was warm in a cache, so the race never showed until task 17 added a code path that rendered immediately on mount. **Fix:** track loading state and render from it; await the fetch before asserting populated state. **Lesson:** asynchrony is a state dimension that compounds invisibly. The bug was planted at task 14 and lay dormant until a later task changed the render timing — the cascade, but in the time domain rather than the contract domain. Playwright's `await expect(...).toHaveCount(n)` with its built-in retry actually *masks* mild races, so the test that fails here is one that asserts the count on the very first paint.

## 8. What this means for choosing and building coding agents

> **Senior rule of thumb:** pick your coding model by the metric that matches your harness. If your agent always gets to run tests and retry, optimize for Pass@2-style recovery, not Pass@1 one-shot brilliance.

Pulling the threads together, here's what Web-Bench actually tells a practitioner, beyond "models aren't as good as the marketing implies."

**The model ranking depends on your loop.** If you're building a one-shot codegen feature (generate-and-paste, no execution), Pass@1 is your metric and GPT-family models look strong. If you're building an *agent* that builds, runs, reads errors, and retries — which is what any serious coding agent does — Pass@2 is your metric, and Doubao/DeepSeek overtaking GPT there is a real signal. The discipline of [building effective agents](/blog/machine-learning/ai-agent/building-effective-agents-hands-on-guide) is precisely about closing this loop; Web-Bench quantifies which models close it well.

**Context budget is a first-class design constraint.** The late-task cliff is largely a context-pressure phenomenon. If your agent works on real projects, you cannot just stuff the whole repo into the prompt as it grows; you need retrieval, structured state summaries, or a genuinely long-context model that attends well. Treat the token budget as something you measure against project growth, not an afterthought.

**Conservatism beats cleverness in dependent chains.** The cascade tax means an agent that establishes stable contracts early and refrains from gratuitous refactors will outperform a "smarter" agent that rewrites things cleanly but breaks references. If you're prompting a coding agent, *tell it not to rename or restructure existing interfaces* unless the task demands it. This single instruction can meaningfully cut cascade failures.

**Grade behavior, not plausibility.** The Web-Agent's reliance on Playwright E2E tests — not on an LLM judge reading the code — is the right call, and you should copy it. An LLM-judge that reads generated code and pronounces it "looks correct" is measuring plausibility; a headless browser clicking the button and checking the row count is measuring behavior. Behavioral grading is more expensive and more honest. The same lesson runs through the broader [agent-evaluation](/blog/machine-learning/ai-agent/eval-agents) discussion: outcome-grounded evals beat judgment-grounded ones whenever you can afford them.

**Variance is real; sampling is a lever.** The best-of-5 SOTA disclosure tells you project-level coding is high-variance. For a risky foundational task — the kind whose contract everything downstream depends on — sampling multiple candidate implementations and selecting the one that passes the most tests is a legitimate inference-time strategy, not a benchmark trick. It's the inference-time mirror of the retry loop.

Here is a compact decision table for picking a model against this benchmark's findings:

| Your use case | Optimize for | Web-Bench reading |
| --- | --- | --- |
| One-shot codegen, no execution | Pass@1 | Claude leads; GPT strong |
| Agent with build+test+retry | Pass@2 / recovery | Claude leads; Doubao, DeepSeek climb past GPT |
| Long, large projects (>2K LOC) | long-context quality | favor models that don't truncate the late tasks |
| Framework-heavy work (React/Vue/Next) | framework idiom fluency | test the specific framework; standards skill ≠ framework skill |
| Standards-heavy work (DOM/Canvas/SVG) | platform fluency | watch for namespace/transform-state footguns |

## 9. Critique: what Web-Bench gets right, and what would change my mind {#critique}

> **Senior rule of thumb:** a benchmark earns trust by being honest about what it does *not* measure. Read the limitations section as carefully as the results.

Let me be clear about what I think Web-Bench gets unambiguously right, because it's a lot. The sequential-dependency design is the single best idea here: it captures the *integration-over-time* nature of real software that snippet benchmarks structurally cannot. Grading with Playwright E2E tests instead of an LLM judge is the honest choice. Anchoring frameworks to real GitHub popularity, splitting coverage across standards and frameworks, and disclosing best-of-5 are all signs of a benchmark built by people who've shipped software. And the headline finding — 25.1%, *lower* than SWE-Bench — is a genuinely useful recalibration of where the frontier sits.

Now the critiques, in descending order of how much they'd move my conclusions.

**1. Pass@1 conflates per-task difficulty with cascade contamination.** Because tasks are scored against the cumulative E2E suite and the model builds on its own prior (possibly wrong) state, a task can be marked failed *purely because an earlier task corrupted the foundation*, even if the model implemented the current task correctly given that bad state. This is realistic — it's how production works — but it means Pass@1 is not a clean estimate of "probability the model can do task $N$." It's "probability the model does task $N$ right *given the state its own earlier mistakes produced*." Both are legitimate metrics, but they answer different questions, and the benchmark would be stronger if it also reported a *fresh-state* per-task pass rate (give the model a correct reference state before each task) to decompose intrinsic difficulty from cascade contamination. **What would change my mind:** if a fresh-state ablation showed per-task pass rates also near 25%, that would prove the difficulty is intrinsic and not mostly cascade — and I'd update toward "models are genuinely bad at individual web tasks," not "models are fine at tasks but can't maintain state."

**2. The two-attempt cap is somewhat arbitrary.** Real engineers retry more than twice. Capping at two attempts makes Pass@2 a specific, reproducible metric, but it under-counts models that would succeed on attempt 3 or 4 with more error context. A Pass@$k$ curve as a function of $k$ would tell us whether models *converge* (each retry helps less) or *plateau early* (the second retry is the last useful one) — and that shape is decision-relevant for how many retries to budget in a real agent. **What would change my mind:** a Pass@$k$ curve showing most models keep improving through $k=5$ would tell me the two-attempt cap is meaningfully understating real-world agent performance.

**3. Project authorship may embed a house style.** Twenty tasks per project, decomposed by specific engineers, encode a particular way of building each app — a particular order, particular intermediate contracts. A model that would build the same app correctly via a *different* decomposition could be penalized for not matching the prescribed contract at each step. This is partly the point (real teams impose conventions), but it does mean the benchmark measures "can you build it *this* way" more than "can you build it *some* working way." **What would change my mind:** evidence that multiple valid decompositions of the same project yield similar model scores would reassure me the benchmark isn't over-indexing on one prescribed path.

**4. Web-only generalization is unproven.** Everything here is web (HTML/CSS/JS/TS, browser APIs, JS-ecosystem frameworks). The compelling claim — that *project-level, dependency-laden* development is the real frontier — would be far stronger with a non-web instantiation (a backend service in Go, a data pipeline in Python, an embedded firmware project in C). It's plausible the cascade dynamics generalize, but plausible isn't measured. **What would change my mind:** a sibling benchmark in a non-web domain showing the same SOTA-around-25%, error-compounding shape would convince me the finding is about *project-level development as such*, not about web specifically.

**5. Contamination resistance is asserted, not proven.** The argument that there's "no single answer to memorize" is good, but the frameworks and standards are public and heavily documented; a model could have memorized canonical implementations of, say, a React todo app. The incremental decomposition mitigates this, but a contamination study — does performance drop on projects authored *after* a model's training cutoff vs. before? — would settle it. **What would change my mind:** a post-cutoff vs. pre-cutoff split showing no performance gap would close the contamination question.

**6. Best-of-5 in the headline, single-run elsewhere, muddies comparison.** The 25.1% SOTA is best-of-5; many of the comparison points are not clearly best-of-5. Comparing a best-of-5 SOTA to single-run baselines slightly flatters the SOTA. The aggregate closed/open averages are the cleaner numbers to cite, which is why I've leaned on them. **What would change my mind:** a uniform best-of-$n$ across all models would make the leaderboard fully apples-to-apples.

None of these critiques undermine the central finding. They're the questions a second version of the benchmark should answer. The headline — that the best model passes a quarter of tasks on a benchmark *harder* than SWE-Bench — is robust, because even under the most charitable reading (cascade contamination inflates the difficulty), the *fresh-state* ceiling would still be well below the 99% of HumanEval. The gap between "writes a correct function" and "ships a coherent app" is not a measurement artifact. It's the gap.

## 10. When to reach for Web-Bench, and when not to

**Reach for Web-Bench when:**

- You're choosing a model to power an **agentic coding product** — especially one that builds features incrementally on a growing codebase, where the Pass@2 recovery ranking is directly predictive of how your agent will behave.
- You need to discriminate among **frontier models** whose snippet-benchmark scores are all saturated near 100% and therefore tell you nothing. Web-Bench's spread (25% down to 11%) actually separates them.
- You care specifically about **web development** capability and want coverage across both raw standards and the real framework stack, not a toy.
- You're designing your **own project-level eval harness** and want a proven template: in-place project mutation, prompt-carries-state, behavioral E2E grading, two-attempt retry with error context. Copy that shape.
- You want a benchmark with **structural contamination resistance** — twenty dependent answers conditioned on the model's own outputs are far harder to game than a leaked function.

**Skip Web-Bench when:**

- Your product is **non-web** (a Go backend, a data pipeline, embedded C). Web-Bench's generalization to other domains is unproven; use or build a domain-matched benchmark instead.
- You only ship **one-shot, no-execution codegen** (autocomplete, a generate-and-paste snippet feature). Then Pass@1 on a snippet benchmark, while saturated, is at least the right *shape* of measurement, and Web-Bench's cascade dynamics don't apply to you.
- You need a **fast, cheap** signal in CI. Running 50 projects × 20 tasks × Playwright E2E × two attempts is expensive — it's a periodic model-selection benchmark, not a per-commit gate.
- You're measuring **algorithmic reasoning** specifically (competitive-programming-style problems). Web-Bench measures integration and consistency, not clever algorithms; a different benchmark fits that goal.
- You want a number that will **look good in a launch post**. Web-Bench numbers are humbling by design. That's a feature for honest evaluation and a bug for marketing.

The deeper takeaway transcends the specific numbers. For four years the field optimized against benchmarks that graded isolated functions, and it got models that are superb at isolated functions and mediocre at shipping software. Web-Bench is part of a correction — alongside SWE-Bench, alongside the [ToolHop](/blog/machine-learning/ai-agent/toolhop-multi-hop-tool-use-benchmark) multi-hop work and the broader [Model Atlas](/blog/machine-learning/bytedance-research-model-atlas) effort to map model capability honestly — toward benchmarks that grade the *job*: stateful, sequential, integration-heavy, graded by behavior. The 25.1% is not a verdict that models are bad. It's a measurement of how much of real software work the current frontier actually does. A quarter. The other three-quarters is the work that's left.

## References

- **Paper:** [Web-Bench: A LLM Code Benchmark Based on Web Standards and Frameworks](https://arxiv.org/abs/2505.07473) — ByteDance. ([HTML version](https://arxiv.org/html/2505.07473v1)).
- **Code:** [github.com/bytedance/web-bench](https://github.com/bytedance/web-bench) — the Web-Agent harness, project definitions, and Playwright test suites (Apache 2.0).
- **Dataset:** [Web-Bench on HuggingFace](https://huggingface.co/datasets/bytedance-research/Web-Bench) and the [leaderboard Space](https://huggingface.co/spaces/bytedance-research/Web-Bench-Leaderboard).
- **HumanEval / MBPP** — the saturated snippet benchmarks Web-Bench is positioned against.

**Companion posts on this blog:**

- [The ByteDance Model Atlas hub](/blog/machine-learning/bytedance-research-model-atlas) — the series this benchmark belongs to.
- [ToolHop: stress-testing multi-hop tool use](/blog/machine-learning/ai-agent/toolhop-multi-hop-tool-use-benchmark) — the same "the benchmark that should change your mind" framing, for tool chains.
- [PaSa: an LLM paper-search agent](/blog/machine-learning/ai-agent/pasa-llm-paper-search-agent) — another ByteDance agent in the series.
- [Building effective agents, hands-on](/blog/machine-learning/ai-agent/building-effective-agents-hands-on-guide) — the build/run/read-error/retry loop that Pass@2 measures.
- [Evaluating agent trajectories beyond the final answer](/blog/machine-learning/ai-agent/evaluating-agent-trajectories-beyond-final-answer) — why grading the path, not just the end state, is the right call.
- [How to evaluate agents](/blog/machine-learning/ai-agent/eval-agents) — outcome-grounded vs. judgment-grounded evaluation.
