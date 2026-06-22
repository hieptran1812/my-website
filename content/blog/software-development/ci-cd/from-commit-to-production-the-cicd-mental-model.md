---
title: "From Commit to Production: The CI/CD Mental Model"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "The path from a commit to production is itself an engineering artifact you design, version, and measure. This is the mental model the whole series rests on: the precise CI/CD/CD definitions, the pipeline as a value stream, the four DORA metrics, and the two principles that make fast delivery safe."
tags:
  [
    "ci-cd",
    "devops",
    "continuous-delivery",
    "continuous-deployment",
    "dora-metrics",
    "deployment",
    "pipeline",
    "build-once-promote-everywhere",
    "everything-as-code",
    "platform-engineering",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/from-commit-to-production-the-cicd-mental-model-1.png"
---

A green build deployed a broken image to production on a Tuesday afternoon. The tests passed. The pipeline was green end to end. And yet within ninety seconds of the deploy, checkout was throwing 500s for every customer on the planet. The post-mortem found the cause in about twenty minutes: the staging environment had rebuilt the image from source with a slightly newer base image than the one we'd tested, the newer base image had bumped a TLS library, and the bump quietly changed how an outbound HTTPS call validated a certificate. The artifact that passed every test in CI was *not* the artifact that ran in production. They were built from the same commit, but they were not the same bytes. We had tested one thing and shipped another, and the pipeline cheerfully called it green.

That incident is the whole reason this series exists. The lesson is not "be more careful." Careful does not scale. The lesson is that **the path from a commit to production is itself an engineering artifact — something you design, version, test, and measure, exactly like the application code that flows through it.** Most teams treat the delivery path as plumbing: invisible until it bursts, owned by nobody, accreting hacks. The teams that ship thirty times a day with a two percent failure rate treat the path as a product. They know precisely how long it takes for a line of code to reach a customer, they know what fraction of their deploys cause an incident, and they can tell you to the minute how fast they recover when one does. They did not get there by being heroic. They got there by making the pipeline boring, repeatable, and observable.

This post is the map for the entire forty-post field manual. It is deliberately a little lighter on any single deep artifact than the posts that follow, because its job is to install the mental model that every later post assumes. By the end you will be able to say precisely what continuous integration, continuous delivery, and continuous deployment each mean — and why blurring them costs you arguments and outages. You will see the pipeline as a *value stream* of stages, each one buying you a unit of confidence, with **lead time** as the headline number. You will know the four DORA metrics that serve as the scoreboard, and the counterintuitive research finding that speed and stability are *not* a trade-off — the fastest teams are also the most stable. And you will internalize the two governing principles the whole series rests on: **build once, promote everywhere** and **everything as code**. Throughout, hold this spine in your head: **commit → build → test → package → deploy → operate.**

![A vertical stack diagram showing the six delivery stages from commit through build, test, package, deploy, and operate, each adding confidence](/imgs/blogs/from-commit-to-production-the-cicd-mental-model-1.png)

## 1. The path is the product

Here is the reframe that changes everything. When most engineers picture "the software we build," they picture the application: the service, the API, the frontend, the model. The delivery path — the CI config, the Dockerfiles, the deploy scripts, the infrastructure definitions — feels like a means to an end. It is the thing you tolerate so the real work can ship.

That instinct is backwards, and it is expensive. Consider what actually determines whether your team is effective. It is not how clever any single feature is. It is how reliably and how quickly the *stream* of features reaches customers, and how rarely that stream breaks production. Those properties — flow and stability — are properties of the *path*, not of any individual change. A brilliant feature stuck behind a six-week release train is worth nothing to a customer until week seven. A trivial one-line fix that takes two hours to ship safely is worth a great deal during an incident. The path is where flow and stability live, which means the path is where most of your delivery value is created or destroyed.

So treat it as a product. A product has an owner, a definition of done, tests, a changelog, and metrics. Your delivery path should have all of those. When the pipeline is slow, that is a defect in your product, and you triage it like one. When a deploy causes an incident, that is a quality escape in your product, and you write a post-mortem. When you can't tell how long it takes a change to reach production, that is a missing instrument, and you go install it. This is not a metaphor stretched for effect. The most consequential engineering decisions a platform team makes — caching strategy, parallelism, the promotion model, the rollback story — are decisions about *the path*, and they compound across every change every engineer ever ships.

The series is organized exactly this way. Each post takes one segment of the path and treats it as a designed artifact: how to wire the CI graph so feedback is fast, how to build a container image that is small and reproducible, how to model the Kubernetes deployment so a bad rollout self-heals, how to keep infrastructure in code so it can be reviewed and reverted, how to gate the supply chain so you ship only what you signed. Every one of them ties back to this single idea. Put plainly: you do not *have* a pipeline; you *engineer* one, and engineering means design, versioning, and measurement.

If you've never thought of the delivery path as something with *design choices*, here's a quick way to feel the weight of them. Every one of these is a real decision a team makes, explicitly or by neglect, and each one changes the numbers on the scoreboard: Do you rebuild per environment or build once and promote? Do you run your checks in series or in a parallelized graph? Do you cache the build or recompute it every run? Do you gate prod with a human, an automated canary, or nothing? Do you deploy by `kubectl apply` from a laptop, by a CI job holding prod credentials, or by a cluster pulling from Git? Do you keep infrastructure in a console or in reviewed code? Do you sign your artifacts or trust whatever's in the registry? None of these has a single right answer for every team — but every one of them is a *choice*, with costs and payoffs you can reason about, and the difference between a team that ships once a month and one that ships thirty times a day is mostly the accumulated weight of having made these choices deliberately rather than by default. The rest of the series is a guided tour of these decisions. This post gives you the frame to evaluate them: the value stream they live in, the metrics they move, and the two principles that tell you which way most of them should go.

## 2. CI vs CD vs CD: the definitions everyone blurs

Three terms get used interchangeably in casual conversation, and the sloppiness causes real damage — teams argue past each other, set the wrong goals, and claim capabilities they don't have. Let's nail them down, because the rest of the series depends on getting these right.

![A tree diagram branching from continuous practices into continuous integration, continuous delivery, and continuous deployment with their defining rules](/imgs/blogs/from-commit-to-production-the-cicd-mental-model-2.png)

**Continuous Integration (CI)** is a discipline about *merging*. The rule is: every engineer integrates their work into the shared mainline (the trunk) frequently — at least daily, ideally several times a day — and every merge automatically triggers a build and an automated test suite. The "continuous" is about the cadence of integration; the "automatic build and test" is what makes that cadence safe. CI is fundamentally a defense against the most expensive bug in software: the integration bug that hides until a giant merge at the end of a long-lived branch, when ten engineers' work collides at once and nobody can tell whose change broke what. Merge small, merge often, and let a machine prove each merge still builds and passes tests. That is CI. Note what CI is *not*: it is not about deploying anything. A team can have flawless CI and still ship to production once a quarter by hand.

**Continuous Delivery (CD)** is a discipline about *releasability*. The rule is: every change that passes CI produces an artifact that is *deployable to production at the push of a button*, and you keep the mainline in a perpetually releasable state. The deploy itself still requires a human to press the button — a release manager, a product owner, a team lead deciding the moment is right — but the *mechanics* of deploying are fully automated, tested, and reliable. There is no scramble, no manual checklist, no "let me remember the steps." If someone with the right permissions clicks deploy, the change goes live correctly. Continuous Delivery answers the question "*could* we ship this right now, safely, in one action?" with a confident yes.

**Continuous Deployment (also CD)** removes the human button. The rule is: every change that passes the full automated pipeline is *automatically* deployed to production, with no manual gate. The pipeline's green light *is* the decision to ship. This demands serious confidence in your automated checks, because there is no human safety net between a merge and a customer. It usually rides on top of progressive delivery — canaries, automated rollback, feature flags — so that even an auto-shipped bad change is caught and reverted by automation before it spreads. Continuous Deployment is the destination many teams aspire to, but it is emphatically *optional*. Plenty of excellent teams run Continuous Delivery deliberately, keeping a human gate on the prod step for regulatory, business-timing, or risk reasons. The distinction matters: Continuous *Delivery* makes every build *releasable*; Continuous *Deployment* makes every build *released*.

Here is the relationship laid out cleanly:

| Practice | The rule | Human gate before prod? | What it defends against | Prerequisite |
|---|---|---|---|---|
| Continuous Integration | Merge to trunk often; every merge auto-builds and auto-tests | N/A (no deploy) | Big-bang merge conflicts, integration drift | Fast trunk-based workflow, automated tests |
| Continuous Delivery | Every green build is deployable in one button-press; trunk always releasable | Yes — a person clicks deploy | Risky, manual, error-prone release mechanics | Solid CI + automated, tested deploy |
| Continuous Deployment | Every green build auto-ships to prod, no human gate | No — automation decides | Slow release cadence, batching of changes | Solid CD + progressive delivery + auto-rollback |

Why does the CD-vs-CD distinction matter so much in practice? Because it changes what "done" means and what you must build. If your goal is Continuous Delivery, your investment is in making the deploy *one reliable action* and keeping trunk releasable — you do not need a sophisticated automated canary analysis to gate prod, because a human is still in the loop choosing when to ship. If your goal is Continuous Deployment, you *must* build the automated safety net, because there is no human to catch a bad build. Teams that say "we do CI/CD" without knowing which CD they mean tend to under-invest in the safety net while claiming the autonomy — they auto-deploy without canaries and learn the hard way. Know which one you're targeting, and build accordingly. For the reliability theory of how to make automated production deploys safe — the canary analysis, the SLO gates, rollback-as-mitigation — this series links out to the SRE field manual's [deploying safely with progressive delivery](/blog/software-development/site-reliability-engineering/deploying-safely-progressive-delivery); here we own the *toolchain* that wires it into the pipeline.

A word on the practice that underpins all three: **trunk-based development.** CI is impossible without it, so it deserves a definition here. Trunk-based development means everyone integrates their work into a single shared branch (the trunk, usually `main`) frequently — at least daily — using short-lived feature branches that live hours or a day or two, never weeks. The opposite is long-lived feature branches that diverge from trunk for weeks before a giant, painful merge. Why does the cadence matter so much? Because the cost of a merge grows with how long the branches have diverged: a branch that's a day old has a handful of conflicts at most; a branch that's a month old has accumulated a thicket of conflicts with everyone else's month of work, and resolving them is both slow and *dangerous* (every manual conflict resolution is a chance to introduce a bug that no test was written for). Trunk-based development keeps every branch close to trunk, so merges are trivial and CI is always validating something close to the real integrated state. This is the cadence half of CI; the automated-build-and-test half is what makes the cadence safe. Teams that try to "do CI" while keeping long-lived branches get the worst of both worlds: they run builds on branches that don't reflect the integrated reality, and the integration bugs they were trying to prevent still ambush them at merge time. The version-control mechanics of short-lived branches and clean merges are covered in the [version-control field manual](/blog/software-development/version-control/using-git-like-senior-workflow-troubleshooting-playbook); here the point is that *trunk-based development is the prerequisite that makes continuous integration mean anything.*

One more piece of vocabulary the series leans on, because it dissolves a lot of confusion: **deploy is not the same as release.** *Deploy* means the new code is running on production infrastructure. *Release* means customers are exposed to its behavior. Feature flags decouple the two — you can deploy code that is dark (running, but flagged off) and release it later by flipping a flag, with zero new deploy. This decoupling is what lets a team practice Continuous Deployment (always deploying) while still controlling *when* customers see a feature (releasing on a schedule). The microservices field manual covers this at the fleet level in [deployment strategies: blue-green, canary, feature flags](/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags); we'll wire the flag and rollout machinery into pipelines in later posts.

## 3. The pipeline as a value stream

Borrow a word from manufacturing: the **value stream**. A value stream is the sequence of activities a unit of work passes through on its way from request to delivered value. In software delivery, the unit of work is a commit, and the value stream is the chain of stages it traverses: **commit → build → test → package → deploy → operate.** Each stage does one job, and — this is the key idea — each stage *adds a unit of confidence* that the change is safe to put in front of customers. By the time a commit reaches `operate`, it has accumulated enough confidence that you trust it with real traffic.

Walk the stages with that lens:

- **Commit.** An engineer pushes a small change to the trunk. Confidence added: the change exists, is attributed, and is captured in version control. Small is doing real work here — a small diff is easier to review, test, and revert than a large one, which is why trunk-based development with small batches is a recurring theme.
- **Build.** The pipeline compiles, bundles, or otherwise transforms source into a runnable artifact — a binary, a container image, a static bundle. Confidence added: the code at least *builds* in a clean, reproducible environment, not just on the author's laptop. This is also where "build once" begins: this single artifact is the one that will flow all the way to prod.
- **Test.** Automated checks run against the built artifact: unit tests, integration tests, contract tests, linters, type checks, security scans. Confidence added: the change behaves as specified and doesn't obviously regress. The design goal here is *fast feedback* — the sooner a test fails, the cheaper the fix, because the author still has full context in their head.
- **Package.** The validated artifact is sealed into an immutable, addressable form and pushed to a registry — a container image with a content digest, a versioned tarball, a signed bundle. Confidence added: the exact bytes that passed testing are now frozen and named, so the thing you deploy is provably the thing you tested.
- **Deploy.** The packaged artifact is rolled out to an environment — first staging, eventually production — by *promoting the same artifact*, never rebuilding. Confidence added: the artifact runs correctly in a production-like (then production) environment, ideally gated by health checks and progressive rollout.
- **Operate.** The change is live, serving traffic, and being watched. Confidence added: real-world signal — metrics, logs, traces, error rates, latency — confirms the change is healthy, or triggers a fast rollback if it isn't. Operate is not the end of the stream; it feeds back to the start, because what you learn in operate informs the next commit.

![A vertical stack contrasting nothing, but a horizontal directed graph of the pipeline where a commit fans out to parallel checks then converges on a single signed artifact before deploy](/imgs/blogs/from-commit-to-production-the-cicd-mental-model-5.png)

Notice two things about this stream. First, a real pipeline is not a straight line — it's a *directed acyclic graph*. Within the build and test stages, independent checks run in parallel (lint, unit tests, and integration tests don't depend on each other), then converge before the artifact is sealed. Designing that graph well — maximizing parallelism, ordering by failure-likelihood and cost so the cheap fast checks fail first — is a core CI skill we develop in later posts. Second, the entire stream is in service of two goals that pull in the same direction: **fast feedback** and **small batches.** Fast feedback means a problem surfaces while it's cheap to fix. Small batches mean each change is easy to reason about and to revert. Together they make the whole stream both faster *and* safer, which is the thread that runs through the DORA findings we get to next.

#### Worked example: where does the time actually go?

Let's make the value stream concrete with numbers. A team measures their lead time — the wall-clock from commit to running-in-prod — and finds it averages six hours. Six hours sounds like a lot of *work*. So they instrument each stage and find this decomposition:

- Commit to PR-review-started: the change sat in the review queue for **4 hours** (reviewers were busy).
- CI build and test: **6 minutes**.
- Review-approved to release-approval: waited **2 hours** for the release manager to batch it with other changes.
- Deploy run: **4 minutes**.

Add it up: 4h + 6min + 2h + 4min ≈ 6 hours and 10 minutes. The machines spent *ten minutes* doing actual work. The other six hours were *waiting* — in the review queue and in the release-batching queue. This is the single most important insight in lead-time work and it surprises almost everyone the first time: in most pipelines, **idle waiting dominates the wall clock, not computation.** If this team optimizes their CI from 6 minutes to 3 minutes, they save 3 minutes off a 6-hour lead time — a rounding error. If instead they shrink the review queue (smaller PRs that review faster, a review SLA) and remove the batch-release wait (ship each change as it's approved instead of pooling), they can cut six hours to under one. The lesson: *measure before you optimize, and optimize the biggest queue first.*

![A left-to-right timeline decomposing a six hour lead time into a four hour review wait, a six minute build, a two hour approval wait, and a four minute deploy](/imgs/blogs/from-commit-to-production-the-cicd-mental-model-6.png)

The arithmetic generalizes. If lead time $L$ is the sum of work time $W$ and wait time $Q$ across stages, $L = \sum_i (W_i + Q_i)$, then the marginal value of cutting any stage is proportional to that stage's share of $L$. Spend your effort where $Q_i$ is largest. Most teams instinctively optimize $W_i$ (faster builds, faster tests) because that's the part engineers control and enjoy, while $Q_i$ (queues, approvals, handoffs) is where the hours actually hide. We'll return to lead-time decomposition repeatedly; it's the diagnostic that tells you what to fix.

There's a deeper reason queues dominate, and it's worth understanding because it predicts where the time will hide *before* you measure. A queue forms wherever the rate of work arriving exceeds the rate at which it's served — and the wait in a queue grows non-linearly as utilization climbs toward 100%. A code-review queue where reviewers are 90% busy doesn't make changes wait *a little* longer than one where reviewers are 50% busy; it makes them wait *dramatically* longer, because near saturation a queue's wait time blows up. This is why "just hire faster reviewers" or "buy faster CI runners" often barely moves lead time: the problem isn't service speed, it's that the resource is near saturation and the queue is exploding. The structural fixes — smaller PRs (so each review takes less time and the queue drains faster), a review SLA, pull-based work instead of batch handoffs, and removing approval gates that exist only out of habit — attack the queue directly. The single most reliable way to shrink delivery lead time is not to make any stage *faster* but to make the *batches smaller* and the *handoffs fewer*, which drains every queue at once. Keep this in your pocket: when someone proposes a faster build to fix slow delivery, ask first where the actual hours are going. Nine times in ten it's a queue, not a computation.

One subtlety in measuring lead time honestly: there are two common definitions, and they answer different questions. *Delivery lead time* starts when work is committed to a deliverable (the first commit on the change) and ends when it's running in production — this is the DORA metric, and it measures your *engineering pipeline's* throughput. *Customer lead time* starts when the idea or request arrives and ends when the customer can use it — it includes product backlog time, design, and prioritization, which are usually far larger than the engineering portion. Don't conflate them. The four DORA metrics measure the *delivery* pipeline, the thing this series engineers. If your customer lead time is six weeks and your delivery lead time is six hours, your bottleneck is your product process, not your pipeline — and no amount of CI caching will fix it. Measure the right thing.

## 4. The four DORA metrics: the scoreboard

If the path is a product, it needs a scoreboard. The best one we have comes from the DORA (DevOps Research and Assessment) program and the *Accelerate* research — years of large-scale surveys of software teams correlating delivery practices with organizational performance. The research distilled delivery performance down to **four metrics**, and they have held up across a decade of annual State of DevOps reports. Memorize them; they recur in every post.

The four split cleanly into two about **speed** (throughput) and two about **stability**:

1. **Deployment frequency** — how often you successfully release to production. (Speed.) Elite teams deploy on demand, many times per day; low performers deploy between once a month and once every six months.
2. **Lead time for changes** — the time from code committed to code running in production. (Speed.) This is the value-stream wall-clock we just decomposed. Elite: less than an hour. Low: one to six months.
3. **Change failure rate** — the percentage of deploys that cause a failure in production requiring remediation (a rollback, a hotfix, a patch). (Stability.) Elite: 0–15%. Low: often 40–60%.
4. **Time to restore service** — how long it takes to recover from a production failure (sometimes written MTTR, mean time to restore). (Stability.) Elite: less than an hour. Low: more than a week.

![A matrix comparing elite and low performers across deploy frequency, lead time, change failure rate, and time to restore](/imgs/blogs/from-commit-to-production-the-cicd-mental-model-3.png)

| Metric | Dimension | Elite performers | Low performers |
|---|---|---|---|
| Deployment frequency | Speed | On-demand, multiple per day | Once per 1–6 months |
| Lead time for changes | Speed | Less than one hour | One to six months |
| Change failure rate | Stability | 0–15% | 40–60% |
| Time to restore service | Stability | Less than one hour | More than one week |

Now the counterintuitive part, and it is the single most important empirical finding in this whole field. The naive mental model says speed and stability *trade off* — that you can ship fast *or* ship safe, and going faster necessarily means breaking more. The DORA research found the **opposite**: speed and stability **move together.** Elite teams are not fast *at the cost of* being unstable; they are fast *and* stable, dramatically better than low performers on *all four* metrics simultaneously. The teams shipping many times a day also have the *lowest* change failure rates and the *fastest* recovery. The teams shipping once a quarter have the *highest* failure rates and the *slowest* recovery. There is no trade-off frontier; there is a cluster of teams that are good at everything and a cluster that is bad at everything.

Why? The causal story is about **batch size.** A team that deploys once a quarter ships an enormous batch of accumulated changes all at once. That batch is nearly impossible to reason about, test thoroughly, or review carefully — so it fails more often (high change-fail rate). And when it does fail, the failure is buried somewhere in a giant, weeks-old changeset, so finding and reverting the culprit takes forever (slow restore). A team that deploys many times a day ships *tiny* batches — often one change at a time. A tiny change is easy to review, easy to test, and if it breaks, the blast radius and the search space are both small: you know almost exactly what changed, and rolling back one small change is trivial. So small frequent changes are *both* faster to ship *and* safer per change. Speed and stability come from the *same* root cause — small batches and fast feedback — which is precisely why they correlate instead of trading off. This is the empirical justification for everything the series advocates: trunk-based development, small PRs, build-once-promote-everywhere, automated gates, fast rollback. They aren't separate good ideas; they're facets of one strategy that improves all four metrics at once.

It's worth being precise about *why* a small batch is safer, because the reason is not just "less code, fewer bugs" — the relationship is worse than linear. The number of *interactions* between changes grows quadratically: ten changes shipped together have roughly forty-five possible pairwise interactions, any of which could be the source of a regression that neither change exhibits alone. Ship those ten changes one at a time and each deploy has *zero* cross-change interactions to debug, because there's only one change in flight. So the debugging difficulty of a batch grows much faster than its size. Worse, a large batch defeats *bisection* — the single most powerful debugging technique for "it worked last week and it's broken now." Bisection works by binary-searching the history of changes for the one that introduced the break; it's only fast when each step of the search isolates a *small* set of changes. When you deploy a quarter's work at once, bisection within that deploy is nearly useless because every commit shipped simultaneously — you can't bisect *deploys* when there's only been one. Small frequent deploys keep the history finely sliced, so when something breaks, the culprit is one of a handful of recent small changes, and you find it in minutes instead of days. (The mechanics of bisecting a broken change live in [binary search your bug with bisection](/blog/software-development/debugging/binary-search-your-bug-with-bisection); the point here is that *batch size is the lever that makes bisection viable at all.*)

There's also a virtuous-cycle dynamic the metrics don't capture directly but every practitioner feels. When deploys are large and risky, engineers *fear* them, so they batch up changes to "make the scary deploy worth it" — which makes the next deploy larger and scarier, a vicious cycle that drives a team deeper into the low-performer cluster. When deploys are small and routine, nobody fears them, so engineers ship the moment a change is ready — which keeps batches small and deploys safe, a virtuous cycle that pulls a team toward elite. The two clusters DORA observes aren't just two skill levels; they're two *stable attractors*, and the thing that flips you from one to the other is reducing batch size until deploying stops being scary. That is why "deploy more often" is not reckless advice — it is, counterintuitively, the safest thing a stuck team can do, provided they pair it with the automated gates and fast rollback that make each small deploy low-stakes.

A word on measuring these honestly, because the metrics are easy to game. *Deployment frequency* should count production deploys that reach customers, not CI runs or staging deploys. *Lead time* should start at the *first* commit of a change (when work began flowing), not when the PR opened, and end when it's *serving traffic*, not when the deploy job turned green. *Change failure rate* needs an honest definition of "failure" agreed in advance — a rollback, a hotfix within some window, a customer-facing incident — applied consistently, not after-the-fact rationalized away. *Time to restore* is measured from when the failure *started affecting customers* (detection lag counts!) to when service is *restored*, which may be a rollback, not a root-cause fix — restoring service first and diagnosing later is itself a discipline, covered in the SRE manual's [mitigate first, diagnose later](/blog/software-development/site-reliability-engineering/mitigate-first-diagnose-later). The point of measuring is to drive behavior; a metric you can game is a metric that will drive the wrong behavior.

#### Worked example: the change-fail-rate and MTTR arithmetic

Make the stability metrics tangible. Suppose your team deploys 4 times a month, and 30% of those deploys cause an incident — so 0.3 × 4 ≈ 1.2 incidents per month. Each incident, because the batch is large and a month old, takes 90 minutes to restore. Monthly damage: roughly 1.2 × 90 ≈ 108 minutes of customer-facing degradation, plus the morale tax of every deploy being a coin-flip.

Now suppose you move to small-batch, deploy-on-demand delivery: 40 deploys a month, change-fail rate down to 4%, MTTR down to 12 minutes (because the failing change is tiny and recent, and rollback is one button). Incidents per month: 0.04 × 40 = 1.6. Slightly *more* incidents in raw count — but each is far cheaper: 1.6 × 12 ≈ 19 minutes of degradation per month, versus 108. You ship ten times as often, and your total downtime drops by more than 80%. And critically, each individual deploy is now low-stakes, which removes the fear that made the team batch up changes in the first place — breaking the vicious cycle. This is the DORA finding as arithmetic: deploying *more* made you *more* stable, because you changed the *size* of what you deploy, not just the frequency.

Put the comparison in a table so the shape is unmistakable — same team, two delivery models, and every number moves in the *good* direction at once even though they shipped ten times more often:

| Quantity | Quarterly-batch model | Small-batch on-demand model |
|---|---|---|
| Deploys per month | 4 | 40 |
| Change-failure rate | 30% | 4% |
| Incidents per month | ≈ 1.2 | ≈ 1.6 |
| Time to restore (MTTR) | 90 min | 12 min |
| Monthly degraded time | ≈ 108 min | ≈ 19 min |
| Per-deploy stakes | high (a quarter of work) | low (one small change) |
| Mean lead time | weeks | hours |

Two things to notice. First, the *change-failure rate* fell even though *raw incident count* ticked up slightly — these measure different things, and a naive read of "more incidents this month" would punish exactly the team that improved. Measure the *rate*, not the raw count, or you'll incentivize the wrong behavior. Second, the dominant win is MTTR collapsing from 90 minutes to 12, and that win comes *for free* from small batches: when the broken deploy contains one small change made an hour ago, the on-call engineer knows what to roll back and rolls it back in one action. When the broken deploy contains three months of accumulated changes, restoration is an archaeology project. The arithmetic says the same thing the DORA cluster says: the lever is batch size, and pulling it moves speed and stability together.

## 5. Principle one: build once, promote everywhere

Now the first of the two principles the whole series rests on. State it as a rule: **build the deployable artifact exactly once, then promote that same immutable artifact unchanged through every environment — dev, staging, production.** The corollary, and the part teams violate constantly: **never rebuild per environment.** The artifact you deploy to prod must be byte-for-byte the artifact that passed your tests in staging, which is byte-for-byte the artifact your build stage produced.

![A before and after comparison contrasting rebuilding a separate artifact per environment against promoting one immutable image by digest through every environment](/imgs/blogs/from-commit-to-production-the-cicd-mental-model-4.png)

Why does this matter enough to be a governing principle? Because of the Tuesday incident at the top of this post. The moment you rebuild per environment, you introduce **rebuild drift**: the prod build can differ from the staging build in ways nobody controls — a newer base image pulled by the dependency resolver, a transitive dependency that floated to a new patch version, a build-time environment variable that differs between CI runners, a clock-dependent or network-dependent step. Each rebuild is a fresh roll of the dice. The artifact that passed all your tests is *not* the artifact you ship, so your tests were testing the wrong thing. You can have a perfectly green pipeline and still deploy something you've never actually run. Build-once-promote-everywhere makes the guarantee airtight: *what you tested is what you ship*, with no asterisk, because they are the same bytes addressed by the same content digest.

The practice is concrete. You build a container image once, push it to a registry, and from then on you refer to it by its immutable **content digest** (a SHA-256 of the image contents, like `sha256:abc123...`), not by a mutable tag like `latest` or even `v1.2.3` (tags can be re-pointed; digests cannot). Promotion then means: take the image that's running in staging, identified by its digest, and tell production to run *that exact digest*. No rebuild. Here's what "deploy the exact digest" looks like in a Kubernetes manifest — note the `@sha256:` pinning rather than a tag:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: checkout
  labels:
    app: checkout
spec:
  replicas: 6
  selector:
    matchLabels:
      app: checkout
  template:
    metadata:
      labels:
        app: checkout
    spec:
      containers:
        - name: checkout
          # Pin by immutable digest, not a mutable tag — this is the
          # exact artifact that passed staging. Promotion changes ONLY
          # the digest, never a rebuild.
          image: ghcr.io/acme/checkout@sha256:abc123def456...
          ports:
            - containerPort: 8080
          readinessProbe:
            httpGet:
              path: /healthz
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 5
          resources:
            requests:
              cpu: "250m"
              memory: "256Mi"
            limits:
              memory: "512Mi"
```

Promotion across environments then becomes a tiny, auditable change — you update the pinned digest in staging's config, verify, then update prod's config to the same digest. In a GitOps world (a later track), promotion is literally a one-line Git commit changing a digest, reviewed and merged like any other change. Compare the two models:

| Property | Rebuild per environment | Build once, promote everywhere |
|---|---|---|
| What prod runs | A *fresh build* from source, possibly drifted | The *exact bytes* that passed staging |
| Reproducibility | Depends on the moment of rebuild | Guaranteed — same digest, same bytes |
| "Works in staging, breaks in prod" risk | High — silent drift between builds | Eliminated for the artifact itself |
| Build minutes spent | N per environment (3× for 3 envs) | One build, reused everywhere |
| Auditability | "Which build is in prod?" is hard to answer | The digest *is* the answer |
| Promotion = | A new build + deploy | A config change to a known digest |

There's a measured payoff beyond safety: you stop paying to rebuild the same code three times. A team building separately for dev, staging, and prod runs the build (say 6 minutes, plus image push) three times per change. Build once and you run it *once* and promote — cutting build minutes (and the CI bill) by roughly two-thirds for the redundant rebuilds, while *also* removing the drift risk. Faster *and* safer *and* cheaper, the recurring shape of good delivery engineering.

#### Worked example: the drift that build-once eliminates

Make the drift concrete, because "rebuild drift" sounds abstract until you've been burned by it. A team builds a Python service into a container three times per change — once per environment — using a Dockerfile whose first line is `FROM python:3.12-slim`. That tag is *mutable*: the registry re-points it whenever a new 3.12 patch ships. On Monday morning the dev build pulls `python:3.12-slim` resolving to patch 3.12.4. The staging build, run two hours later, pulls the *same tag* — but overnight the registry re-pointed it to 3.12.5, which bumped a bundled OpenSSL. The prod build, run that afternoon after the release approval, pulls 3.12.5 too. Tests passed against 3.12.4 in dev's image; prod runs 3.12.5. The OpenSSL bump changed a default cipher negotiation, and an outbound call to a payment provider that worked in dev now fails the TLS handshake in prod. Every test was green. The bug shipped anyway, because *the thing tested was never the thing deployed.* That is rebuild drift, and it is not a rare edge case — it is the *default* behavior of mutable base tags plus per-environment rebuilds.

The fix starts in the Dockerfile: pin the base image by digest so the build is reproducible, and use a multi-stage build so the runtime image is small and contains only what it needs (the containers track goes deep on this; here's the shape).

```dockerfile
# Pin the base by DIGEST, not the mutable tag — the build is now
# reproducible: the same source always resolves the same base bytes.
FROM python:3.12-slim@sha256:0a1b2c3d4e5f... AS build
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
COPY . .

# Multi-stage: the runtime image carries only the installed deps + app,
# not the build toolchain — smaller, smaller attack surface.
FROM python:3.12-slim@sha256:0a1b2c3d4e5f... AS runtime
WORKDIR /app
COPY --from=build /install /usr/local
COPY --from=build /app /app
USER 1000:1000               # never run as root
ENTRYPOINT ["python", "-m", "checkout"]
```

Now run build-once. The pipeline builds the image *one time* against that pinned base, pushes the result, and records the result's own digest. Dev, staging, and prod every one of them runs `ghcr.io/acme/svc@sha256:abc123` — the identical bytes. The OpenSSL version is frozen into that artifact; whatever patch the base happened to be at build time is the patch that runs in prod, *because it's the same image*. If 3.12.5 had a problem, you'd have caught it in dev and staging *on that exact image*, before it ever reached prod. The drift is not "reduced" — it is *structurally impossible*, because there is only one artifact. That structural guarantee is the entire reason build-once is a principle and not a nice-to-have. The cost of the guarantee is discipline (pin your bases, address by digest, never rebuild to promote); the payoff is that "green" finally means what everyone always assumed it meant.

Stress-test the principle. *What if dev and prod genuinely need different configuration* — a different database URL, a different log level, a different feature-flag default? That's fine and expected: build-once-promote-everywhere applies to the *artifact* (the code/binary/image), not to the *configuration*. Config is injected at deploy time via environment variables, ConfigMaps, or secrets — it lives *outside* the immutable image, exactly as the Twelve-Factor App prescribes ("strict separation of config from code"). The image is identical across environments; only the injected config differs. *What if you need an environment-specific build flag*, like a debug build for dev? Resist it where you can — the more your prod build differs from what you tested, the weaker the guarantee. Where it's unavoidable, treat each variant as a *distinct artifact* that's tested and promoted on its own track, not a silent per-env rebuild. The principle isn't "never have differences"; it's "make every difference explicit, injected, and outside the artifact you tested."

## 6. Principle two: everything as code

The second governing principle: **everything that defines how your software is built, tested, deployed, configured, and governed lives in Git as code — reviewable, versioned, and reproducible.** Not just application source. The *pipeline* definition (the CI/CD workflow YAML). The *infrastructure* (servers, networks, clusters, defined as Terraform or similar). The *configuration* (environment settings, feature flags, as declarative files). The *policy* (security and compliance rules, as code a gate can enforce). If it shapes how code reaches production, it belongs in version control, subject to the same review, history, and revert that your application code enjoys.

![A before and after comparison contrasting un-auditable console clickops against pipeline, infrastructure, config, and policy all versioned in Git](/imgs/blogs/from-commit-to-production-the-cicd-mental-model-7.png)

The alternative is **clickops** — configuring things by clicking through consoles and dashboards by hand. Clickops feels fast in the moment and is a disaster over time, for reasons that compound:

- **Un-reproducible.** A hand-configured environment is a *snowflake* — unique, undocumented, impossible to recreate exactly. When it dies, or when you need a second one, you're reverse-engineering it from memory. The phrase "works on my machine" has a deployment-scale cousin: "works in this environment," where nobody can build a matching one.
- **Un-auditable.** A console click leaves no durable record of *who* changed *what*, *when*, and *why*. When prod breaks at 2 a.m. and the question is "what changed?", clickops gives you no answer. A Git history gives you the exact commit, author, diff, and PR discussion.
- **Un-reviewable.** A click takes effect immediately with no second pair of eyes. A change as code goes through a pull request, where a reviewer can catch the Terraform plan that wants to *delete the production database* before it runs — a real, recurring near-miss that has destroyed real companies' data when it wasn't caught.
- **The Friday-deploy fear.** All of the above produce a culture of fear: deploys are scary because they're un-reversible and un-understood, so teams batch them, freeze them on Fridays, and ship rarely — which (per the DORA causal story) makes each deploy *bigger and riskier*, deepening the fear. Everything-as-code breaks the cycle: a bad change is reverted with `git revert`, the pipeline re-runs, and the previous known-good state is restored automatically.

The practice is to express each layer declaratively and put it under review. Here's the *pipeline itself* as code — a GitHub Actions workflow that is the spine we deepen across the whole series. Read it as the value stream made executable: build once, test, package by digest, then a gated deploy.

```yaml
name: deliver

on:
  push:
    branches: [main]   # trunk-based: every merge to main runs the pipeline

permissions:
  contents: read
  packages: write
  id-token: write       # for OIDC keyless auth + signing, no long-lived secret

concurrency:
  group: deliver-${{ github.ref }}
  cancel-in-progress: false   # don't cancel a deploy mid-flight

jobs:
  build-test:
    runs-on: ubuntu-latest
    outputs:
      digest: ${{ steps.push.outputs.digest }}   # the immutable artifact id
    steps:
      - uses: actions/checkout@v4

      - name: Set up Buildx
        uses: docker/setup-buildx-action@v3

      - name: Cache build layers
        uses: actions/cache@v4
        with:
          path: /tmp/.buildx-cache
          key: buildx-${{ github.sha }}
          restore-keys: buildx-

      - name: Lint and unit test
        run: |
          make lint
          make test          # fast checks first — fail cheap, fail early

      - name: Build and push image (ONCE)
        id: push
        uses: docker/build-push-action@v5
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.sha }}
          cache-from: type=local,src=/tmp/.buildx-cache
          cache-to: type=local,dest=/tmp/.buildx-cache,mode=max

  scan-and-sign:
    needs: build-test
    runs-on: ubuntu-latest
    steps:
      - name: Scan for critical vulnerabilities
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ghcr.io/${{ github.repository }}@${{ needs.build-test.outputs.digest }}
          severity: CRITICAL,HIGH
          exit-code: "1"      # gate: a critical vuln fails the pipeline

      - name: Keyless sign the artifact
        run: |
          cosign sign --yes ghcr.io/${{ github.repository }}@${{ needs.build-test.outputs.digest }}

  deploy-staging:
    needs: scan-and-sign
    runs-on: ubuntu-latest
    environment: staging      # auto-deploy the SAME digest to staging
    steps:
      - name: Promote to staging
        run: ./deploy.sh staging "${{ needs.build-test.outputs.digest }}"

  deploy-prod:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production   # protection rules add the human gate (Continuous Delivery)
    steps:
      - name: Promote SAME digest to prod
        run: ./deploy.sh production "${{ needs.build-test.outputs.digest }}"
```

Read what's encoded here. The build happens *once* in `build-test`, and its `digest` output is what every downstream job promotes — build-once-promote-everywhere, expressed in YAML. The `scan-and-sign` job *gates* on vulnerabilities and *signs* the artifact (the supply-chain track deepens this). The `deploy-prod` job is bound to a GitHub `environment: production` whose protection rules require a human approval — that single configuration choice is what makes this pipeline *Continuous Delivery* rather than *Continuous Deployment*. Delete the protection rule and it becomes Continuous Deployment; the only difference is one declarative setting. That's the power of everything-as-code: the most consequential policy decision in your delivery — human gate or not — is a reviewable, versioned config line, not tribal knowledge in someone's head. Notice too the `permissions: id-token: write` and the absence of any long-lived cloud secret: the pipeline uses OIDC keyless federation, so CI proves its identity to the cloud and signing service per-run instead of holding a credential that can leak (a recurring supply-chain theme).

The `deploy.sh` the workflow calls is itself just code — a thin, reviewable script that *verifies the signature before it deploys*, so a tampered artifact (the SolarWinds nightmare) can't reach prod even if it somehow landed in the registry:

```bash
#!/usr/bin/env bash
set -euo pipefail
env="$1"        # staging | production
digest="$2"     # the immutable artifact id, e.g. sha256:abc123...
image="ghcr.io/acme/checkout@${digest}"

# Refuse to deploy anything we didn't sign — verify provenance first.
cosign verify \
  --certificate-identity-regexp "https://github.com/acme/checkout/.*" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  "$image"

# Promote the SAME digest — no rebuild. Helm injects per-env config,
# keeping config out of the immutable image (Twelve-Factor).
helm upgrade --install checkout ./chart \
  --namespace "$env" \
  --set image="$image" \
  --values "values/${env}.yaml" \
  --wait --timeout 5m        # block on readiness; fail the deploy if unhealthy
```

Read what this guarantees. The `cosign verify` step turns the abstract idea "ship only what you signed" into a hard gate: if the artifact's signature doesn't trace to a build from this repo's CI, the deploy *aborts* before any change reaches the cluster. The `--set image="$image"` promotes the exact digest, never a rebuild. The `--values "values/${env}.yaml"` is where per-environment config lives — outside the immutable image, exactly as the build-once principle requires. And `--wait` makes the deploy *fail* if the new pods don't become healthy, so a broken rollout doesn't silently report success. Every one of those behaviors is encoded in a script that lives in Git, gets reviewed, and can be reverted — there is nothing to remember and nothing to click.

The honest measurement here is qualitative as much as quantitative: how long does it take a new engineer to stand up a matching environment? With everything as code, the answer is "run `terraform apply` and `helm install`" — minutes, reproducible. With clickops, the answer is "ask the one person who remembers, and pray." And the audit answer — "what changed before the incident?" — is `git log`, every time.

## 7. Why manual and clickops delivery fails

It's worth dwelling on the failure modes of manual delivery, because they're the *motivation* for everything the series builds, and because the symptoms are so common that many teams have normalized them as "just how deploys are."

The **snowflake server** is the archetype. Someone set up production by hand two years ago, tweaking it live over time to fix issues. Nobody knows the full set of changes. The configuration exists only in the running system, not in any file. The original engineer left. Now the box is a snowflake — unique, fragile, irreplaceable — and everyone is quietly terrified of touching it, rebooting it, or migrating off it. Snowflakes are how organizations end up with a production system *nobody dares to change*, which is the opposite of being able to deliver continuously.

The **"works on my machine" of deploys** is the runtime cousin. A change works in the environment it was tested in and breaks in production because the environments differ in ways nobody controlled — different OS patch level, different installed library, different env var, different network policy. This is *exactly* the drift that build-once-promote-everywhere kills for the artifact and that everything-as-code kills for the environment. When both principles hold, "works in staging" *means* "works in prod," because staging and prod run the same bytes in the same declaratively-defined environment.

The **un-reproducible release** is what happens when the deploy is a manual checklist: SSH in, pull the new code, run the migration, restart the service, clear the cache, flip the load balancer. Even a careful engineer skips a step under pressure. The steps drift from the wiki. The one person who knows the *real* steps becomes a bottleneck and a single point of failure. There's no record of exactly what was done, so a failed deploy can't be cleanly reverted — you don't know precisely what state to revert *to*.

And the cultural endpoint of all of this is the **Friday-deploy freeze** and the weekend that pages. Because deploys are risky, manual, and hard to reverse, teams stop deploying on Fridays (no one wants to break prod going into the weekend), then stop deploying often at all, then batch a month of changes into one terrifying release. That big batch fails more often, is harder to debug when it does, and takes longer to recover from — which makes deploys *even* scarier, which makes teams batch *even* more. It's a doom loop, and the DORA data shows exactly where it leads: the low-performer cluster, slow *and* unstable. The escape is counterintuitive but proven: deploy *more often*, in *smaller batches*, with *automated* and *reversible* mechanics — and the fear dissolves because each deploy is small, understood, and one `git revert` away from undone.

There's one more clickops failure mode that's so dangerous it's worth calling out specifically, because everything-as-code is what defends against it: the **destructive infrastructure change nobody reviewed.** Picture an engineer making an infrastructure change by hand in a cloud console — renaming a resource, adjusting a setting. With infrastructure-as-code, that same change goes through a tool like Terraform, which produces a *plan*: an explicit, reviewable diff of exactly what it will create, change, and — crucially — *destroy* before it does anything. A reviewer reading that plan can see the line that says it intends to *destroy* the production database (because a seemingly-innocent attribute change forces a resource replacement) and stop it cold in code review. There is no such safety net for a console click; the click just happens. This is not hypothetical — Terraform plans that wanted to delete production data stores have been caught in review at countless companies precisely *because* the plan was reviewable, and have destroyed real data at companies that ran `apply` without reading the plan. The IaC track devotes serious attention to `plan`-before-`apply` discipline and policy-as-code gates that *refuse* to apply a plan containing a destroy of a protected resource. The meta-point: putting infrastructure in code doesn't just make it reproducible; it makes the most dangerous changes *visible and stoppable* before they execute. A click is a change with no preview; a reviewed plan is a change you got to veto.

## 8. Fast feedback and the cost-of-delay of a failing test

Two principles govern the *structure* of the series — build once, everything as code. One property governs its *tempo*: **fast feedback.** It deserves its own section because it's the constraint that shapes how you build every stage of the pipeline, and because the reasoning behind it is precise enough to settle real design arguments.

The claim is that the *value* of a test or check is not just whether it catches a bug, but *how soon* it catches it. A bug caught by a type-checker the instant the author saves the file costs seconds to fix — the author has full context, the change is fresh, nothing has built on top of it. The *same* bug caught by a code reviewer an hour later costs minutes and a context-switch for two people. Caught by an integration test in CI a few hours later, it costs a re-run and a re-review. Caught in staging by QA the next day, it costs a bug report, a triage, a reproduction, and a second trip through the whole pipeline. Caught in *production* by a customer, it costs an incident, a rollback, a post-mortem, reputational damage, and possibly money. The cost of a defect grows by *orders of magnitude* with each stage it escapes — this is the well-worn "shift left" observation, and it has a direct consequence for pipeline design: **put the fastest, cheapest checks first, and make every check fail as early in the stream as it possibly can.**

This is why a well-engineered CI graph orders its checks by *cost-to-run* ascending and *likelihood-to-fail* descending, not arbitrarily. Linting and type-checking run first because they're fast (seconds) and catch a large class of mistakes. Unit tests run next (seconds to a couple minutes). Integration tests, which are slower and need more setup, run after. Expensive end-to-end tests and security scans run last, on the artifact that's already passed everything cheaper. The result: most failures surface in the first thirty seconds, the author gets feedback while they're still looking at the code, and you don't burn ten minutes of integration-test compute on a change that a one-second linter would have rejected. Fail cheap, fail fast, fail early — in that order.

Fast feedback also sets a hard target for *total* CI time, and the target is more aggressive than most teams assume. The relevant threshold is human attention: if CI returns a result in under ~10 minutes, an engineer will wait for it before moving on, keeping the change in their working memory. If CI takes 40 minutes, the engineer context-switches to other work, and when CI eventually fails they have to *reload* the entire mental context of the change — a tax that's far larger than the wall-clock difference suggests. So a 40-minute pipeline isn't four times worse than a 10-minute one; it's *categorically* worse, because it breaks the tight write-test-fix loop that makes development fast. This is why "get CI under 10 minutes" is one of the highest-leverage investments a platform team can make, and why the CI-foundations track spends so much energy on caching and parallelism. We'll cut a real 38-minute pipeline to 6 minutes in that track; here, just internalize *why* the number matters.

#### Worked example: the build-cache hit-rate payoff

Caching is the highest-leverage CI optimization, and its payoff has a clean formula worth understanding. Suppose a build has a fixed unavoidable cost $F$ (checkout, environment setup) and a cacheable cost $C$ (compilation, dependency installation, image layers). With a cache hit-rate $h$ (the fraction of the cacheable work that's served from cache), the expected build time is approximately $T = F + (1 - h)\,C$. The cache turns the *cacheable* portion off in proportion to how often it hits.

Plug in numbers. Say $F = 1$ minute and $C = 9$ minutes, for a 10-minute cold build. With no cache ($h = 0$): $T = 1 + 9 = 10$ minutes. With a 50% hit rate: $T = 1 + 4.5 = 5.5$ minutes — nearly halved. With a 90% hit rate (typical for a well-structured build where only the changed layers recompile): $T = 1 + 0.9 = 1.9$ minutes — a 5× speedup, for the price of configuring a cache. Compare that to the alternative everyone reaches for first — parallelism. Splitting that 10-minute build across two runners might get you to ~5 minutes, but it *doubles* your runner cost and adds coordination complexity, and it does nothing about the redundant work; you're now paying twice to recompute what a cache would have skipped for free. The lesson, and the reason the "when not to" section warns against sharding before caching: **fix the hit rate first.** A high cache hit-rate is both faster and cheaper; parallelism is faster but more expensive, and only worth adding to the work that's *left* after caching. Measure your hit rate (most CI systems report it), and if it's low, that's almost always where your minutes — and your CI bill — are going.

Stress-test fast feedback. *What if the cache is cold* — a fresh runner, a cache eviction, a dependency bump that invalidates everything? Then you pay the full cold-build cost, which is exactly why $F$ (the truly unavoidable part) should be small and why your cold build should still be tolerable; a cache that turns a 40-minute cold build into a 2-minute warm build is wonderful until the cache misses and you're back to 40. Engineer for a fast *cold* build too, and treat the cache as an accelerator, not a crutch. *What if two PRs merge at once* and both invalidate the cache? A merge queue (covered in the CI track) serializes integration so each merge tests against the actual post-merge state, preventing the "both were green alone, broken together" failure — and incidentally keeps the cache coherent. *What if a flaky test fails the pipeline* on a change that's actually fine? Then your fast feedback is *lying*, which is corrosive — engineers learn to ignore red, and a real failure slips through. Flaky tests must be quarantined and fixed, not retried-until-green; the [flaky-test playbook](/blog/software-development/debugging/the-flaky-test-find-it-fix-it-or-quarantine-it) covers the diagnosis, and the CI track covers the quarantine mechanics.

## 9. The whole-series map

This post is the hub. Here's where the rest of the field manual goes, organized along the value stream you now hold in your head. Think of it as walking the stream left to right — from how a commit becomes a trusted artifact, through how that artifact is packaged and shipped, to how you operate and scale the whole machine.

![A tree mapping the series into integrate and build, package and ship, and operate and scale tracks](/imgs/blogs/from-commit-to-production-the-cicd-mental-model-8.png)

- **CI foundations.** Trunk-based development, fast test pyramids, the CI DAG, caching and parallelism, merge queues, flaky-test quarantine, and cutting a 38-minute pipeline to 6. This is where we make the *feedback* fast. (Flaky-test diagnosis itself links out to [the flaky-test playbook](/blog/software-development/debugging/the-flaky-test-find-it-fix-it-or-quarantine-it).)
- **Containers and registries.** Multi-stage Dockerfiles, distroless and minimal base images, shrinking a 1.9 GB image to 80 MB, reproducible builds, BuildKit/buildx caching, registries (GHCR, ECR, Artifact Registry, Harbor), and addressing by digest. This is where the *immutable artifact* gets built well.
- **Kubernetes deploy.** Deployments, Services, readiness and liveness probes, rolling updates, pod disruption budgets, and the rollout mechanics that make a bad deploy self-heal instead of self-destruct. The *deploy* stage, done right.
- **Templating and overlays.** Helm charts and values, Kustomize overlays, and how to manage per-environment config without per-environment rebuilds — config separated from the immutable artifact.
- **GitOps.** Argo CD and Flux, the pull-based reconcile loop, app-of-apps, sync waves, self-heal, and drift detection — why the cluster *pulling* its desired state from Git beats CI *pushing* prod credentials around. This is everything-as-code taken to its conclusion for deploys.
- **Infrastructure as code.** Terraform state and backends, modules, workspaces, `plan`/`apply` discipline, golden images with Packer, and policy-as-code with OPA/Conftest — the *environment* as a reviewable, reproducible artifact.
- **Supply-chain security.** Signing with cosign/Sigstore, SBOMs with Syft, scanning with Trivy/Grype, SLSA provenance and attestations, dependency hygiene with Renovate/Dependabot, secrets with Vault and External Secrets and OIDC — so you ship only what you built and signed, and CI never holds the keys to prod.
- **Operate, measure, and scale the platform.** Wiring progressive delivery (Argo Rollouts, Flagger) into the pipeline, measuring the four DORA metrics for real, deployment observability, self-service platform engineering, and the org-scale concerns of running delivery for many teams. For the *reliability theory* underneath progressive delivery — error budgets, SLO gates, chaos — the series links out to the SRE manual's [the error budget, the currency of reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability) rather than re-deriving it.

Everything threads back to this post's frame: the value stream (commit → build → test → package → deploy → operate), the two principles (build once, promote everywhere; everything as code), and the four DORA metrics as the scoreboard. And it all converges on the future capstone, `capstone-the-cicd-playbook`, which assembles every track into one reference pipeline you can adapt end to end.

## 10. War story: what happens when delivery isn't engineered

Two real incidents make the stakes concrete, and they sit at opposite ends of the delivery-discipline spectrum.

**The Knight Capital deploy, 2012.** Knight Capital was a major market-maker. They deployed new trading software to their servers by *manually* copying the new code to each of eight production servers. On deploy day, the engineer copied it to seven of the eight and missed one. The new code repurposed an old feature flag that, on the *eighth* un-updated server, still triggered dead, long-disabled "power peg" trading logic. When the market opened, that one server began executing the old logic at machine speed — buying high and selling low across millions of orders. In about 45 minutes, Knight Capital lost roughly \$440 million and was effectively destroyed as an independent company within days. The proximate cause was a manual, per-server, non-reproducible deploy with no automated verification that all servers were in the identical, intended state. Build-once-promote-everywhere and everything-as-code don't merely make life nicer; for some businesses, their absence is existential. An automated deploy that promotes one verified artifact to all servers atomically — and refuses to proceed if any server diverges — does not make the eighth-server mistake.

**The SolarWinds build-system compromise, 2020.** Attackers breached SolarWinds' *build pipeline* and injected malicious code into the Orion software *during the build*, so the artifact that got signed and shipped to thousands of customers — including government agencies — contained a backdoor that the source code did not. This is the nightmare that build-once-promote-everywhere and supply-chain security together address: it's not enough to test your source; you must be able to prove that *the artifact you ship is the artifact you built from reviewed source*, with provenance and a signature attesting to the build's integrity. SolarWinds is *why* the supply-chain track exists — why we sign artifacts with cosign, generate SBOMs, attest build provenance with SLSA, and verify signatures at deploy time. A pipeline that builds an unverifiable artifact and a deploy that runs whatever's in the registry without checking a signature is a pipeline waiting for this attack. (The Codecov bash-uploader compromise and various dependency-confusion attacks are variations on the same theme: the supply chain *is* an attack surface, and the pipeline is where you defend it.)

**The GitOps drift incident, a composite from real outages.** A team runs their cluster with a GitOps tool that reconciles the live state toward what's declared in Git. One night, an on-call engineer hits a production problem and, under pressure, fixes it by hand — `kubectl edit` to bump a replica count and patch a config value directly in the cluster. It works; the page resolves; everyone goes back to sleep. Three hours later the GitOps reconcile loop runs, notices the live cluster no longer matches Git (which still says the *old* replica count and config), and dutifully *reverts the manual fix* to match the declared state — re-introducing the very problem the engineer fixed, at 4 a.m., with nobody watching. This is *drift*: the gap between what's declared and what's live, and a reconcile loop will always resolve drift in favor of Git. The lesson isn't "GitOps is dangerous" — it's that when your source of truth is Git, *every* change must go through Git, including emergency fixes. The fix the engineer should have made was a one-line PR bumping the replica count, merged fast-tracked, which the reconcile loop would have then *enforced* instead of reverted. Everything-as-code is only safe when it's *everything* — a single out-of-band manual change is a landmine. The GitOps track covers exactly this: how to handle emergencies within the model, and how self-heal turns drift from a silent risk into an enforced invariant.

The throughline: delivery that isn't engineered fails in ways that range from embarrassing to catastrophic. All three incidents were failures of *the path*, not of any feature — a manual per-server deploy, a compromised build pipeline, an out-of-band manual change against a declarative source of truth. All three were preventable by treating the path as the designed, versioned, verified artifact this series insists it must be. The features were fine. The *delivery* killed them.

## 11. How to reach for this (and when not to)

A field manual that only ever says "do more" is lying to you. Every practice in this series has a cost — in tooling, in cognitive load, in maintenance — and the right amount of delivery engineering depends on your context. Here's the honest guidance.

**Always, even for a solo project:** version control with a real branch/merge workflow, an automated build, an automated test run on every push, and a *scripted* (not manual) deploy. This is the floor. It's cheap, it pays for itself immediately, and skipping it is how three-month-old side projects become un-deployable. Even a single GitHub Actions file that builds, tests, and deploys is enough to get most of the value.

**Adopt as soon as you have a team and real users:** trunk-based development with small PRs, build-once-promote-everywhere (build the image once, promote by digest), and everything-as-code for your pipeline and config. The marginal cost is low and the payoff — no rebuild drift, reviewable changes, fast revert — shows up the first time prod breaks.

**Adopt when the scale justifies it:** Kubernetes, GitOps, full IaC, signing and SBOMs, progressive delivery with automated canary analysis, a self-service internal platform. These are *powerful* and *not free*. Concretely, here's when *not* to reach for them:

- **Don't build a GitOps platform for a three-person startup.** Argo CD, a service mesh, and a reconcile loop are wonderful at fifty engineers and forty services; at three engineers and one service they're a second product you now have to maintain instead of shipping your actual product. Use a managed PaaS (a platform that handles build and deploy for you) until the operational pain of *not* having the platform exceeds the cost of running it.
- **Don't add a canary to a service with no SLI to gate on.** A canary deploy works by routing a small slice of traffic to the new version and *automatically comparing a metric* (error rate, latency) against the old version. If you don't have a reliable service-level *indicator* to compare, the canary has nothing to decide on — it's theater. Get the metric first, then gate on it.
- **Don't shard or parallelize the pipeline before you've cached the build.** Splitting a slow pipeline across more runners adds coordination complexity and cost. Most slow pipelines are slow because they rebuild from scratch every time. Fix caching first (often a 5–10× win for near-zero complexity), *then* parallelize what's left if it's still slow.
- **Don't chase Continuous *Deployment* before your tests are trustworthy.** Auto-shipping every green build to prod with no human gate is only safe if green genuinely means safe. If your test suite is flaky or shallow, Continuous Deployment will faithfully ship your bugs to customers at machine speed. Earn the automated gate by first having Continuous *Delivery* with a human gate, watching your change-fail rate, and only removing the human once the data says the automation is trustworthy.

The meta-rule: **match the delivery engineering to the cost of getting it wrong and the rate at which you change.** A high-frequency, high-stakes system (payments, trading, healthcare) justifies the full apparatus early. A low-frequency, low-stakes internal tool does not. The goal is never "maximum tooling"; it's "the simplest path that gives you fast, safe, reversible delivery for *your* risk profile."

## Key takeaways

- **The path from commit to production is an engineering artifact** — designed, versioned, tested, and measured like a product, not tolerated as plumbing. Most of your delivery value (flow and stability) lives in the path, not in any single feature.
- **CI, CD-delivery, and CD-deployment are three distinct practices.** CI = merge to trunk often, auto-build-and-test every merge. Continuous *Delivery* = every green build is releasable in one button-press, human still decides when. Continuous *Deployment* = every green build auto-ships to prod, no human gate. Know which you mean.
- **The pipeline is a value stream** — commit → build → test → package → deploy → operate — where each stage adds confidence, and **lead time** is the wall-clock from commit to running-in-prod. Idle waiting (queues, approvals), not computation, usually dominates lead time, so measure before you optimize.
- **The four DORA metrics are the scoreboard:** deploy frequency, lead time, change-failure rate, time-to-restore. Speed and stability **move together** — elite teams are faster *and* more stable — because small frequent batches are both quicker to ship and safer per change.
- **Build once, promote everywhere.** Build the artifact exactly once and promote the same immutable bytes (by digest, not tag) through every environment. Never rebuild per environment — that reintroduces drift, so what you tested isn't what you ship.
- **Everything as code.** Pipeline, infrastructure, configuration, and policy all in Git — reviewable, versioned, reproducible. Clickops produces snowflakes, un-auditable changes, and the Friday-deploy fear; everything-as-code makes a bad change one `git revert` away from undone.
- **Match the engineering to the risk.** The floor (VCS + automated build/test + scripted deploy) is for everyone; GitOps, full IaC, signing, and automated canaries are for when scale justifies their cost. Don't build a platform for three people, and don't auto-deploy on tests you don't trust.

## Further reading

- *Accelerate: The Science of Lean Software and DevOps* (Forsgren, Humble, Kim) — the foundational research behind the four DORA metrics and the speed-and-stability-together finding.
- The annual **State of DevOps / DORA reports** — the latest elite-vs-low-performer benchmarks and the evolving research on what drives delivery performance.
- The **Twelve-Factor App** — the canonical statement of config-separated-from-code, the principle that makes build-once-promote-everywhere work.
- The **GitHub Actions**, **Argo CD**, **Terraform**, and **Kubernetes** official docs — the toolchain references for the artifacts in this series.
- The **SLSA framework** and **Sigstore/cosign** docs — the supply-chain provenance and signing standards behind the security track, and the answer to SolarWinds-style build-pipeline attacks.
- Within this series, the future capstone `capstone-the-cicd-playbook` assembles every track into one reference pipeline.
- SRE field manual: [deploying safely with progressive delivery](/blog/software-development/site-reliability-engineering/deploying-safely-progressive-delivery) and [the error budget, the currency of reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability) — the reliability theory underneath safe automated deploys.
- Microservices field manual: [CI/CD and independent deployability](/blog/software-development/microservices/ci-cd-and-independent-deployability) and [deployment strategies: blue-green, canary, feature flags](/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags) — the fleet-level view of the same ideas.
