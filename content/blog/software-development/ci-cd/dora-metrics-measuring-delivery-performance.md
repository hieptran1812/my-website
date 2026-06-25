---
title: "DORA Metrics: Measuring Delivery Performance"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Master the four DORA metrics—Deploy Frequency, Lead Time, Change Failure Rate, and Time to Restore—and build the instrumentation pipeline that turns them into a continuous improvement flywheel."
tags:
  [
    "ci-cd",
    "devops",
    "dora",
    "metrics",
    "delivery-performance",
    "deploy-frequency",
    "lead-time",
    "change-failure-rate",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/dora-metrics-measuring-delivery-performance-1.png"
---

Picture two teams. Team Alpha ships fifteen times a day. Their staging environment is always green. Their deploy pipeline completes in four minutes. The on-call engineer has not been paged at 3 AM in two months. Every engineer on the team can explain what is running in production right now, what changed last, and how to roll it back in under two minutes. Team Beta ships once a month. Their release nights are twelve-hour marathon affairs with a war room, a rollback plan, a post-mortem template already open before the first commit touches production, and a standing policy that nobody takes vacation the week after a release. By nearly every intuition a software organization possesses, Team Alpha is doing it right. The data agrees, and the data is now more extensive than any single team's experience.

Now picture a third team: Team Gamma. They have read all the same blog posts as Team Alpha. They ship twenty-three times a day. They are also breaking production on roughly one in every five deploys, and their mean time to restore service exceeds six hours when something goes wrong. A post-mortem last quarter found that ninety percent of their daily deployments were one-line configuration tweaks and version bumps, not features. Their engineers had learned — correctly, given their incentive structure — that smaller units called "deployments" inflated their deploy-frequency number, while the actual product work sat quietly in long-lived branches waiting for a quarterly release train to pick it up. Team Gamma did not learn to ship fast. They learned to game a metric.

This is the Goodhart's Law trap at the heart of every software delivery metrics program. Coined by economist Charles Goodhart in 1975 and generalized by anthropologist Marilyn Strathern, the principle states: when a measure becomes a target, it ceases to be a good measure. In software delivery, this manifests predictably. Teams that are evaluated on deploy frequency find creative ways to increase the count without shipping more value. Teams that are evaluated purely on change failure rate stop deploying anything except trivial changes. Teams that are graded on lead time for changes start marking tickets as "in progress" the moment a sprint starts and "done" the moment they merge, eliminating the inconvenient middle stages from the measurement. The metric improves while the underlying system gets worse.

The four DORA metrics — Deploy Frequency, Lead Time for Changes, Change Failure Rate, and Time to Restore Service — were designed with exactly this trap in mind. They form two complementary pairs, and any team that improves one pair while ignoring the other is almost certainly gaming rather than improving. That structural design is the reason the research holds up across 33,000+ professionals surveyed over seven years of DORA State of DevOps reports. The four metrics are self-checking. You cannot inflate deploy frequency without risking change failure rate. You cannot drive change failure rate to zero by never deploying without watching lead time balloon. The pairs are in productive tension with each other, and that tension is the measurement system's integrity.

Figure 1 shows the two-pair structure and the elite benchmark numbers from the DORA research. This is the map. The rest of this post is the territory.

![The DORA four-metric model showing throughput and stability pairs with elite benchmarks annotated](/imgs/blogs/dora-metrics-measuring-delivery-performance-1.png)

## Why Four Metrics? The Two-Pair Model

The DORA research program began with a question that sounds obvious in retrospect but had no rigorous answer in 2014: what does it mean to be good at software delivery? Most organizations measure what is easy to count — lines of code written, tickets closed per sprint, story points completed, pull request throughput. None of those predict outcomes. They measure activity, and activity is infinitely gameable without producing value. A team can close tickets by marking them done without doing the work. A team can increase story point velocity by inflating estimates. The metrics feel productive while the software rots.

Nicole Forsgren, Jez Humble, and Gene Kim — the authors of *Accelerate* and the DORA research program — identified the two dimensions that actually matter. The first is **throughput**: how quickly can a change move from idea to production? The second is **stability**: when changes reach production, how often do they cause problems, and how quickly can you recover when they do? These two dimensions are necessary and sufficient to characterize delivery performance. Everything else — test coverage, cycle time, PR size, deployment pipeline duration — is either a component of one of these dimensions or a leading indicator that predicts future movement in them.

The crucial research finding was that throughput and stability are **not a trade-off**. This contradicts the dominant intuition in engineering organizations, which holds that you can ship faster or you can ship reliably, but not both simultaneously. Low-performing organizations behave as though this trade-off is real: they slow down to improve stability (add more testing, longer release cycles, more approvals), or they speed up and accept more failures (move fast and break things, ship and hotfix). The DORA data shows that elite-performing organizations achieve high throughput and high stability together. The correlation between deploy frequency and change failure rate is not negative (faster = more failures) — in elite cohorts, it is actually positive (faster = fewer failures), because the causation runs through batch size. Small, frequent deployments are both faster and safer than large, infrequent ones.

### The Four Metrics as a System

Deploy Frequency measures how often a team deploys to production. Lead Time for Changes measures how long it takes a change to move from commit to production. These two together define the throughput of your delivery system — how fast value flows from engineers to users.

Change Failure Rate measures what percentage of deployments to production cause a service degradation requiring remediation. Time to Restore Service measures how long it takes to recover when a failure occurs. These two together define the stability of your delivery system — what happens when things go wrong, and how resilient your system is.

The four together give you a complete picture that a single metric, or even a pair, cannot provide:

- High DF + high LT: something is blocking changes from reaching production after merge (large batch sizes, long deploy queues, manual gates).
- Low DF + low LT: the pipeline is fast but the team is not using it (cultural barriers, risk aversion, lack of CD automation).
- Low CFR + high MTTR: failures are rare but catastrophic when they occur (insufficient monitoring, untested rollback procedures, poor runbooks).
- High CFR + low MTTR: failures are frequent but cheap (good incident response, practiced rollback, but insufficient quality gates).

Only by looking at all four simultaneously do you get an accurate diagnosis of where the constraint lives.

### The DORA Cohort Benchmarks at a Glance

The DORA research classifies teams into four performance cohorts. The numbers below are drawn from the 2019, 2021, and 2023 State of DevOps reports; the 2023 cohort definitions are the most current.

| Cohort | Deploy Frequency | Lead Time for Changes | Change Failure Rate | Time to Restore Service |
|--------|------------------|-----------------------|---------------------|-------------------------|
| Elite  | Multiple deploys/day | Less than 1 hour | 0%–5% | Less than 1 hour |
| High   | Once/day to once/week | 1 day to 1 week | 5%–10% | Less than 1 day |
| Medium | Once/week to once/month | 1 week to 1 month | 10%–15% | 1 day to 1 week |
| Low    | Once/month to once every 6 months | 1 month to 6 months | 15%–30%+ | More than 1 week |

![DORA cohort comparison showing the before-state of low-performing teams versus the after-state of elite teams across all four delivery metrics](/imgs/blogs/dora-metrics-measuring-delivery-performance-2.png)

A few important calibrations on this table. First, the Elite cohort on CFR (0%–5%) does not mean elite teams never cause failures. It means that out of every twenty deployments, no more than one causes a production degradation requiring remediation. Teams deploying a hundred times a day at the Elite CFR threshold have five or fewer incidents per day caused by deployments — not zero. They recover fast (under one hour by the MTTR Elite threshold), so the failure does not accumulate into user-visible downtime.

Second, the 2019 report published the most striking raw comparison: 208-fold higher deploy frequency, 106-fold faster lead time, 2,604-fold faster MTTR, and 7-fold lower CFR for Elite versus Low cohorts. These numbers feel like statistical artifacts but are not. They reflect the genuine operating-mode difference between "multiple times per day" and "between once per month and once every six months." The cumulative gap compounds across all four metrics simultaneously because the same underlying technical and cultural practices that enable fast deployment also enable fast recovery.

Third, the Medium-to-High boundary on CFR (10%–15%) is where most teams stall. A team with 13% CFR is in a stable trap: they are not failing badly enough to trigger an improvement initiative, but they are failing often enough that their delivery system cannot sustain higher deploy frequency without overwhelming their incident response capacity. The right response is better deploy-time validation, not slower deployments.

### The Organizational Performance Connection

The 2023 DORA State of DevOps report — the most recent in the series — found that elite delivery performers were 2.5 times more likely to exceed their organizational performance goals compared to low performers. These are not engineering goals. They are business goals: profitability, productivity, market share, customer satisfaction, and NPS. The link from software delivery performance to organizational performance is empirically validated, not assumed.

The 2019 report found a 208-fold higher deployment frequency and a 106-fold faster lead time for changes in elite cohorts compared to low performers. These numbers are often quoted in isolation, which can make them seem like statistical artifacts of how the cohorts are defined. They are not. The 208-fold gap is the compound result of the difference between "multiple times per day" and "between once per month and once every six months" — two genuinely different operating modes for a software delivery system.

The research uses Structural Equation Modeling (SEM) to test causal hypotheses in the observational data. SEM constructs a full causal model — from technical practices (continuous delivery, trunk-based development, test automation) through software delivery performance to organizational outcomes — and estimates the path coefficients. The finding is not merely that elite teams have better business outcomes; it is that the causal pathway from technical practices through delivery performance to business outcomes is statistically significant, even after controlling for industry, team size, years of experience, and application type.

## Deploy Frequency: Counting Deployments Correctly

Deploy frequency is the number of times your team deploys to production in a given time window, expressed as a rate per day, week, or month. That definition sounds unambiguous. In practice, the boundary conditions are where the Goodhart traps live.

**What counts as a deployment**: A deployment event must represent a meaningful change reaching the production environment that serves real user traffic. Specifically:

- A new version of the application artifact deployed to the production fleet: yes.
- A database migration that changes production data or schema: yes.
- A feature flag toggle that enables a new code path for production users: yes, if it represents a code path already deployed dark that is now active.
- A rollback that redeploys a previous artifact: debated. DORA's canonical definition counts it because it is a production state change. Most teams exclude rollbacks from their DF count to avoid artificial inflation. The important thing is consistency — choose a definition and keep it.
- A hotfix deployment: yes.
- An emergency patch deploying a security fix: yes.

**What does not count**:
- Deployments to staging, QA, or pre-production environments.
- CI pipeline runs that produce artifacts but do not deploy them.
- Feature flag toggles that have no code or config change behind them (A/B test reassignments, for example).
- Infrastructure changes that do not affect the application artifact (scaling up an instance type, adjusting autoscaling thresholds).

The discipline of drawing this boundary clearly matters more than exactly where you draw it. A team that counts staging deployments has inflated DF numbers that misrepresent their production release cadence. A team that counts config tweaks as deployments has inflated DF numbers that hide the fact that actual feature delivery is rare.

### Elite Benchmarks for Deploy Frequency

| Cohort | Deploy Frequency | Typical Mechanism |
|--------|-----------------|-------------------|
| Elite | Multiple times per day | Continuous deployment, trunk-based dev, automated gates |
| High | Between once per day and once per week | Continuous delivery, short-lived branches, manual final gate |
| Medium | Between once per week and once per month | Feature branch merges, scheduled releases, partial automation |
| Low | Between once per month and once every six months | Manual release process, long release cycles, heavy approval chains |

Elite organizations at the scale of Google, Netflix, Meta, and Amazon deploy to production thousands of times per day across their service fleets. For a monolithic application with a team of ten engineers, "multiple times per day" means three to seven intentional production deployments. The absolute number matters less than the trend, the cohort placement, and whether the DF score reflects genuine product delivery cadence.

### Why Deploy Frequency Predicts Quality

The causal mechanism is batch size. A team deploying once a month accumulates thirty days of changes per deployment. A team deploying five times per day accumulates roughly two to three hours of changes per deployment. When a monthly deployment fails, the root cause could be any of hundreds of commits across dozens of files. When a three-hour deployment fails, the root cause is almost certainly in the ten or twenty commits that went out since the last deploy.

This is why elite teams can sustain both high deploy frequency and low change failure rate. Each deployment is small enough that failures are both less likely (fewer compounding change interactions) and faster to diagnose (short commit history to bisect). The feedback loop is also tighter: a bug introduced this morning is caught this afternoon, not six weeks from now at the end of a quarterly cycle.

### Measuring Deploy Frequency in Practice

The most reliable instrumentation for deploy frequency is a webhook or CI step that emits a deployment event at the moment a new artifact begins receiving production traffic. Not at merge. Not at artifact build. At the moment the artifact is live and serving users.

```yaml
# GitHub Actions step: emit a deployment event to your DORA metrics collector
# Place this step at the END of your production deploy job, after health checks pass
- name: Emit DORA deployment event
  if: github.ref == 'refs/heads/main' && success()
  env:
    DORA_API_KEY: ${{ secrets.DORA_API_KEY }}
    DORA_COLLECTOR_URL: ${{ secrets.DORA_COLLECTOR_URL }}
  run: |
    DEPLOY_TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    curl --fail -s -X POST "$DORA_COLLECTOR_URL/events/deployment" \
      -H "Authorization: Bearer $DORA_API_KEY" \
      -H "Content-Type: application/json" \
      -d "{
        \"event_type\": \"deployment\",
        \"service\": \"${{ github.repository }}\",
        \"environment\": \"production\",
        \"deploy_ref\": \"${{ github.sha }}\",
        \"timestamp\": \"$DEPLOY_TIMESTAMP\",
        \"triggered_by\": \"${{ github.actor }}\",
        \"run_id\": \"${{ github.run_id }}\",
        \"pr_numbers\": ${{ toJSON(github.event.pull_request.number) }}
      }"
    echo "Deployment event emitted at $DEPLOY_TIMESTAMP"
```

This step runs only on pushes to main and only after preceding steps succeed — failed builds do not emit deployment events and do not inflate the count. The `pr_numbers` field connects the deployment to the pull request for later lead time computation.

For teams using Kubernetes and ArgoCD, the deployment event should be emitted by the ArgoCD sync webhook rather than the GitHub Actions step, since ArgoCD manages the actual production state transition:

```yaml
# ArgoCD notification template for DORA deployment events
# In your argocd-notifications-cm ConfigMap
template.dora-deployment-event: |
  webhook:
    dora-collector:
      method: POST
      path: /events/deployment
      body: |
        {
          "event_type": "deployment",
          "service": "{{.app.metadata.name}}",
          "environment": "production",
          "deploy_ref": "{{.app.status.sync.revision}}",
          "timestamp": "{{.app.status.operationState.finishedAt}}",
          "sync_status": "{{.app.status.sync.status}}"
        }

trigger.dora-on-sync-succeeded: |
  - when: app.status.operationState.phase in ['Succeeded']
    send: [dora-deployment-event]
```

### Connecting Deploy Frequency to the GitHub Deployments API

Teams using GitHub Enterprise can anchor DF measurement to the GitHub Deployments API rather than custom webhook calls. The Deployments API creates a first-class deployment record in GitHub that the built-in DORA dashboard can read directly. The pattern is to call the API at the end of your production deploy job instead of — or in addition to — the custom collector call:

```yaml
- name: Create GitHub Deployment record
  if: github.ref == 'refs/heads/main' && success()
  uses: actions/github-script@v7
  with:
    script: |
      const deployment = await github.rest.repos.createDeployment({
        owner: context.repo.owner,
        repo: context.repo.repo,
        ref: context.sha,
        environment: 'production',
        auto_merge: false,
        required_contexts: [],
        description: 'Production deployment via CI'
      });
      await github.rest.repos.createDeploymentStatus({
        owner: context.repo.owner,
        repo: context.repo.repo,
        deployment_id: deployment.data.id,
        state: 'success',
        environment_url: 'https://your-app.example.com',
        description: 'Deployment completed successfully'
      });
      core.setOutput('deployment_id', deployment.data.id);
```

This approach has two advantages: the GitHub DORA dashboard picks it up without any additional integration work, and the deployment record is permanently associated with the commit SHA for audit purposes. The disadvantage is that it requires explicit GitHub Deployments API calls from every service's pipeline — a non-trivial migration for organizations with dozens of services already using custom event pipelines.

## Lead Time for Changes: The Clock Has a Specific Start

Lead time for changes is the elapsed time from the first commit of a change to that change running in production and serving user traffic. Every word in that definition matters.

**"First commit"** — not when the PR is opened, not when the code review begins, not when the engineer thinks of the feature. The clock starts the moment code that will eventually ship touches the repository. For teams using trunk-based development with short-lived branches, this is usually within hours of starting work. For teams using long-lived feature branches, the first commit might happen three weeks before the PR is opened, and those three weeks are hiding in the lead time number whether they are measured or not.

**"Running in production"** — not deployed to staging, not passing CI, not merged to main. The clock stops when real user traffic is routing to the new code.

This precision matters because the measurement boundary determines what improvements are visible. Teams that measure from PR open to deployment might see a metric that looks healthy even though they have systematic delays in how long code sits in a branch before engineers bother opening a PR. Teams that measure from merge to deployment might have a beautiful "merge to production in 12 minutes" number while their actual time from feature complete to production-validated is four days.

![Lead time decomposition showing PR review wait time as the dominant bottleneck in most teams](/imgs/blogs/dora-metrics-measuring-delivery-performance-4.png)

### Decomposing Lead Time: The Four Stages

Lead time for changes decomposes into four stages. The distribution of time across these stages is almost always non-uniform, and the dominant stage is almost always the one that gets the least engineering attention.

**Stage 1: Coding time** — from the first commit of the change to the PR being opened. For small, well-scoped changes in a healthy trunk-based development environment, this is one to eight hours. For large changes, or in teams with long-lived feature branches, this can be days or weeks. If your coding time is high, the root cause is usually change scope (changes are too large), branch strategy (long-lived branches), or unclear technical direction (engineers rework).

**Stage 2: PR review wait time** — from the PR being opened to the first substantive review action (approval, change request, or comment that moves the review forward). This is the stage that most teams underestimate. Analysis of GitHub data across thousands of open source and enterprise repositories consistently shows that this stage accounts for 50–80% of total lead time in teams that have not specifically invested in reducing it. The reason is structural: reviewing code is reactive work. Engineers context-switch into it when the immediate pressure of their own work eases, which often means the end of the day, the next morning, or after the next sprint planning session.

**Stage 3: CI run duration** — from the PR being merged to the deployment artifact being ready and validated by CI. This is the stage that engineering teams most frequently try to optimize first, because it is the most tractable with engineering effort: parallelize test execution, cache dependencies, split slow test suites. A well-invested team can cut CI time from 45 minutes to 8 minutes. If PR review wait is averaging 18 hours, cutting CI by 37 minutes has reduced total lead time by 3.5%. This is not a bad optimization. It is just not the highest-leverage one.

**Stage 4: Deploy and validation time** — from artifact ready to production-validated. For teams with automated continuous deployment this is 5–20 minutes: canary rollout, health check window, synthetic test execution. For teams with manual deployment approval gates, this can add hours or days.

### Elite Benchmarks for Lead Time

| Cohort | Lead Time for Changes |
|--------|-----------------------|
| Elite | Less than 1 hour |
| High | Between 1 day and 1 week |
| Medium | Between 1 week and 1 month |
| Low | Between 1 month and 6 months |

The elite threshold of under one hour means a developer writes a change, opens a PR, gets it reviewed, merges, CI runs, and the change is live in production — all within 60 minutes. This is achievable with small changes, fast reviews (enabled by a team culture of immediate review SLOs), fast CI (10 minutes or less), and automated deployment. It requires trunk-based development and no manual gates. Teams that get there report it feels like a qualitative shift in development experience, not just a quantitative improvement.

#### Worked example: Decomposing a team's 3.5-day lead time

A platform engineering team pulls their data from GitHub's API and their deployment event log. They find their median lead time is 3.5 days, placing them solidly in the Medium cohort. They decompose the measurement by stage:

- Coding time (first commit to PR open): 6 hours median
- PR review wait (PR open to first approval): 58 hours median
- CI run duration (merge to artifact ready): 22 minutes median
- Deploy pipeline (artifact ready to production-validated): 18 minutes median

Total median: 84.6 hours. The CI run (22 minutes) and deploy pipeline (18 minutes) together are 40 minutes out of 84.6 hours — 0.8% of total lead time. The PR review wait is 58 hours out of 84.6 — 68.5%.

The team had been running a project to parallelize their test suite to cut CI time. They paused those experiments and instead introduced a PR review SLO: every PR must receive a first review within four hours during business hours (9 AM to 6 PM team-local). They added a Slack bot that notifies the team channel when a PR has been open for three hours without a review. They added GitHub CODEOWNERS entries so that every PR automatically requests a reviewer rather than waiting for someone to self-assign.

Six weeks after the SLO rollout, they re-measured. PR review wait was 4.2 hours median, down from 58 hours. Total lead time was 11.4 hours. They had moved from Medium to High cohort on lead time alone, without touching CI or deploy infrastructure.

#### Worked example: Arithmetic of an elite-track lead time

A payment-processing microservice team wants to understand whether they can reach the Elite cohort (under 1 hour). They break their current pipeline into timed segments:

- Average PR size: 47 lines changed across 3 files
- PR review wait (with a strict 2-hour SLO): 1 hour 55 minutes median
- CI pipeline: parallel unit tests (4 min) + integration tests (9 min) + artifact build (3 min) = 16 minutes total
- Canary deploy phase: 5% traffic for 8 minutes + full rollout + synthetic health check = 14 minutes
- Total: 1h 55min + 16min + 14min = 2 hours 25 minutes

The 2h 25min total puts them at the top of the High cohort. To reach Elite (under 1 hour), they need to cut 1 hour 25 minutes from the pipeline. The PR review wait is the only segment large enough to absorb that cut. They tighten the SLO to 30 minutes and pair-program the review with the author on the same Slack thread (author walks the reviewer through the change while screen-sharing). This brings PR review wait to a median of 28 minutes. New total: 28min + 16min + 14min = 58 minutes. They have entered the Elite cohort with a single cultural change and no infrastructure investment.

The arithmetic point here is precise: the individual stage numbers are 28 + 16 + 14 = 58 minutes. Every minute over 60 comes from review wait, so the only lever that moves the needle is review speed. CI optimization (reducing 16min to, say, 10min) saves 6 minutes — not enough to cross the threshold if review wait stays above 34 minutes.

### Measuring Lead Time with Automation

```bash
#!/bin/bash
# compute-lead-time.sh
# Run at deploy time to compute lead time for the batch of commits being deployed.
# Requires: git history, LAST_PROD_SHA env var (set from your deployment log)
#
# Usage: LAST_PROD_SHA=abc123 CURRENT_SHA=def456 bash compute-lead-time.sh

LAST_PROD_SHA="${LAST_PROD_SHA:?Set LAST_PROD_SHA to the SHA of the last production deploy}"
CURRENT_SHA="${CURRENT_SHA:-HEAD}"

# Find the earliest commit in this batch not yet in production
FIRST_COMMIT_IN_BATCH=$(git log --format="%H %ai" "${LAST_PROD_SHA}..${CURRENT_SHA}" | tail -1)

if [ -z "$FIRST_COMMIT_IN_BATCH" ]; then
  echo "No new commits since last production deploy"
  exit 0
fi

FIRST_COMMIT_SHA=$(echo "$FIRST_COMMIT_IN_BATCH" | awk '{print $1}')
FIRST_COMMIT_TIME=$(echo "$FIRST_COMMIT_IN_BATCH" | awk '{print $2}')
DEPLOY_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)

LEAD_TIME_SECONDS=$(( $(date -d "$DEPLOY_TIME" +%s) - $(date -d "$FIRST_COMMIT_TIME" +%s) ))
LEAD_TIME_HOURS=$(echo "scale=2; $LEAD_TIME_SECONDS / 3600" | bc)

echo "First commit in batch: $FIRST_COMMIT_SHA at $FIRST_COMMIT_TIME"
echo "Deploy time: $DEPLOY_TIME"
echo "Lead time: ${LEAD_TIME_HOURS} hours (${LEAD_TIME_SECONDS} seconds)"

# Push to DORA collector
curl --fail -s -X POST "${DORA_COLLECTOR_URL}/events/lead_time" \
  -H "Authorization: Bearer ${DORA_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{
    \"service\": \"${SERVICE_NAME}\",
    \"deploy_ref\": \"${CURRENT_SHA}\",
    \"first_commit_sha\": \"${FIRST_COMMIT_SHA}\",
    \"first_commit_time\": \"${FIRST_COMMIT_TIME}\",
    \"deploy_time\": \"${DEPLOY_TIME}\",
    \"lead_time_seconds\": ${LEAD_TIME_SECONDS},
    \"commit_count\": $(git rev-list --count ${LAST_PROD_SHA}..${CURRENT_SHA})
  }"
```

For teams using Linear or Jira for issue tracking, you can augment this with issue-level lead time that captures the time from ticket creation to production deploy:

```sql
-- PostgreSQL query: per-issue lead time for the last 30 days of deployments
-- Assumes your DORA collector has linked deployment events to issues via PR metadata
SELECT
  i.issue_key,
  i.title,
  i.created_at                                           AS issue_created,
  MIN(c.committed_at)                                    AS first_commit,
  d.deployed_at                                          AS production_deploy,
  ROUND(EXTRACT(EPOCH FROM (MIN(c.committed_at) - i.created_at)) / 3600, 1)
                                                         AS planning_time_hours,
  ROUND(EXTRACT(EPOCH FROM (d.deployed_at - MIN(c.committed_at))) / 3600, 1)
                                                         AS coding_to_deploy_hours,
  ROUND(EXTRACT(EPOCH FROM (d.deployed_at - i.created_at)) / 3600, 1)
                                                         AS total_lead_time_hours
FROM issues i
  JOIN issue_commits ic  ON ic.issue_id   = i.id
  JOIN commits c         ON c.sha         = ic.commit_sha
  JOIN deployment_issues di ON di.issue_id = i.id
  JOIN deployments d     ON d.id          = di.deployment_id
WHERE d.environment = 'production'
  AND d.deployed_at >= NOW() - INTERVAL '30 days'
GROUP BY i.issue_key, i.title, i.created_at, d.deployed_at
ORDER BY d.deployed_at DESC;
```

This query gives you both the technical lead time (first commit to production) and the total flow time (ticket created to production), which is a useful diagnostic for whether planning and scoping delays are eating into delivery performance.

### Configuring LinearB for Lead Time Measurement

LinearB connects to GitHub via OAuth and computes lead time stage breakdown automatically, but a few configuration decisions affect measurement accuracy. In LinearB's settings, set the "Coding start" anchor to "First commit" rather than "First push" or "PR opened" — the distinction matters for teams who commit locally before pushing. Set the "PR review" window to exclude weekends if your team does not work weekends, otherwise review time on a Friday afternoon PR gets inflated by the Saturday–Sunday gap. For the "deployment" event, configure LinearB to read from the GitHub Deployments API (most accurate) rather than inferring from merge-to-main events (which conflates merge time with deploy time for teams that do not deploy on every merge).

### Querying Faros AI for Lead Time Trend

Faros AI stores DORA events in a normalized graph model and exposes a GraphQL API. The query below retrieves 90-day weekly lead time percentiles for a specific service:

```graphql
query LeadTimeTrend {
  cicd_Pipeline_aggregate(
    where: {
      _and: [
        { service: { _eq: "payments-api" } },
        { environment: { _eq: "production" } },
        { deployed_at: { _gte: "2026-03-01T00:00:00Z" } }
      ]
    }
  ) {
    aggregate {
      count
    }
    nodes {
      deployed_at
      lead_time_seconds
      pr_review_wait_seconds
      ci_duration_seconds
      deploy_duration_seconds
    }
  }
}
```

Combine this with a Grafana data source that queries Faros via HTTP and you get a 90-day lead time percentile chart with stage breakdown, no custom pipeline required on the Faros side.

## Change Failure Rate: Precision in What You Count

Change Failure Rate is the percentage of deployments to production that result in a service degradation requiring remediation. It sounds simple. The definition of "failure" is the single most important decision you make when instrumenting it, and the wrong definition produces numbers that feel meaningful but mislead.

**What counts as a change-triggered failure**:
- A deployment that triggers a P1 or P2 incident — the incident opens, an engineer is paged, and remediation is required.
- A deployment followed within a configurable time window (typically 1–24 hours, depending on your incident detection latency) by an incident on the same service.
- A deployment where a feature flag is enabled and that flag enables a code path that triggers an incident.
- A deployment that is proactively rolled back because monitoring detected degradation before users reported it.

**What does not count**:
- Infrastructure failures unrelated to the deployment (AWS region outage, BGP route leak, third-party API going down).
- Incidents on services that were not recently deployed (the causal window has passed).
- Auto-resolving alerts that do not represent real user impact and do not require human intervention.
- Incidents that post-mortem analysis conclusively attributes to external causes.

The DORA canonical definition is: "a deployment to production that results in degraded service and subsequently requires remediation (e.g., hotfix, rollback, fix-forward patch)." The key criterion is "requires remediation" — an automatic health check that fires and auto-resolves in 30 seconds without user impact is not a change failure. An incident that wakes an engineer at 2 AM and requires a rollback is.

### The Batch Size Mechanism for CFR

CFR is causally linked to batch size through a compounding risk model. A deployment containing one commit that touches two files has one set of potential failure modes: the change introduced in those two files. A deployment containing 200 commits across 30 files has 200 sets of potential failure modes, and they can interact — a change in file A that is safe on its own might become dangerous when combined with a change in file B that went out in the same batch.

This is the mathematical argument for small batch sizes that the Accelerate research quantifies empirically. Teams with high deploy frequency, who are forced by necessity to keep batches small, consistently report lower CFR than teams with low deploy frequency who accumulate large batches. The CFR reduction is not because the individual changes are better; it is because smaller batches have smaller blast radius and shorter bisect windows when they fail.

The corollary is the Goodhart trap: a team that drives CFR down by freezing their deploy cadence — nothing goes to production unless it passes a three-day soak in staging — will see CFR improve while Lead Time triples. They have optimized the metric by degrading the system.

![DORA cohort performance matrix: Low, Medium, High, and Elite rows against Deploy Frequency, Lead Time, CFR, and MTTR columns](/imgs/blogs/dora-metrics-measuring-delivery-performance-3.png)

Figure 3 shows the matrix clearly: there is no cohort that is strong on stability but weak on throughput, or vice versa. The cohorts improve uniformly across all four metrics. Teams at the Low cohort are slow and unreliable. Teams at the Elite cohort are fast and reliable. The data does not show a throughput-stability trade-off at any cohort level.

### Measuring CFR with Prometheus

CFR measurement requires joining your deployment event stream with your incident event stream. The join key is service name and time: any incident that opens within your attribution window after a deployment to the same service is presumed deployment-caused unless tagged otherwise.

```yaml
# Prometheus recording rules: Change Failure Rate
# Assumes deployment_event and incident_event metrics are being pushed to your Prometheus instance
# incident_event labels: severity, service, caused_by (set by attribution logic)
groups:
  - name: dora_cfr
    interval: 1h
    rules:
      # 28-day rolling CFR (primary reporting metric)
      - record: dora:change_failure_rate_pct:28d
        expr: |
          100 * (
            count_over_time(
              incident_event{severity=~"P1|P2", caused_by="deployment"}[28d]
            )
            /
            count_over_time(
              deployment_event{environment="production"}[28d]
            )
          )

      # 7-day rolling CFR (operational alerting metric — detect trend shifts faster)
      - record: dora:change_failure_rate_pct:7d
        expr: |
          100 * (
            count_over_time(
              incident_event{severity=~"P1|P2", caused_by="deployment"}[7d]
            )
            /
            count_over_time(
              deployment_event{environment="production"}[7d]
            )
          )

  - name: dora_cfr_alerts
    rules:
      - alert: CFRExceedsHighCohortThreshold
        expr: dora:change_failure_rate_pct:7d > 15
        for: 24h
        labels:
          severity: warning
        annotations:
          summary: "Change failure rate exceeds High cohort threshold (15%)"
          description: "7-day CFR is {{ $value | humanize }}%. Review recent deployments for patterns."

      - alert: CFRExceedsLowCohortThreshold
        expr: dora:change_failure_rate_pct:7d > 30
        for: 12h
        labels:
          severity: critical
        annotations:
          summary: "Change failure rate in Low cohort (>30%) — consider deployment freeze"
          description: "7-day CFR is {{ $value | humanize }}%. Immediate investigation required."
```

The `caused_by="deployment"` label is set by a sidecar process that runs on a schedule, queries your incident system for incidents opened in the last 24 hours, looks up whether a deployment to the same service occurred in the preceding attribution window, and labels matching incidents accordingly. The sidecar also exposes an override API that lets the on-call engineer mark an incident as "external cause" to remove it from CFR attribution.

### Grafana Dashboard Queries for CFR

Once the Prometheus recording rules are in place, the Grafana dashboard panel for a 28-day rolling CFR with cohort threshold annotations uses these queries:

```
Panel: Change Failure Rate (28-day rolling)
Query A (CFR line):
  dora:change_failure_rate_pct:28d{service="$service"}

Query B (Elite threshold reference line — constant):
  vector(5)

Query C (High threshold reference line):
  vector(15)

Query D (Low threshold reference line):
  vector(30)

Legend overrides:
  Query B: "Elite threshold (5%)"  — green dashed
  Query C: "High threshold (15%)"  — yellow dashed
  Query D: "Low threshold (30%)"   — red dashed
```

Add a Grafana threshold override so the CFR panel background turns amber when the metric crosses 15% and red when it crosses 30%. This creates a visual signal that engineers notice on the team dashboard without needing to configure a separate PagerDuty alert for it.

## Time to Restore: Starting the Clock at Impact

Time to Restore Service (MTTR in conventional incident management terminology) is the elapsed time from when a production degradation begins affecting users to when the service is fully restored to its pre-incident state. Every word matters here too, and the start time is where most teams get it wrong.

The clock starts at **when the incident begins affecting users**. Not when the alert fires. Not when an engineer acknowledges the alert. Not when the on-call opens a war room. Not when a user reports a problem. The incident starts when the degradation begins. If your deploy goes bad at 14:00 and your synthetic monitors take 8 minutes to detect it and another 5 minutes for PagerDuty to page the on-call, your incident started at 14:00, not at 14:13. Those 13 minutes are part of your MTTR and they belong in the MTTR measurement.

In practice, you approximate the incident start time with the first alert timestamp, the first failing health check, or the first synthetic monitor failure — whichever is earliest. The ideal instrument is a synthetic monitor that checks production behavior every 30–60 seconds; that gives you at most a 60-second gap between actual degradation and detected start. Teams with 10-minute synthetic check intervals can have a 10-minute systematic undercount in MTTR, which flatters the metric without improving the user experience.

### Elite Benchmarks for MTTR

| Cohort | Time to Restore Service |
|--------|------------------------|
| Elite | Less than 1 hour |
| High | Less than 1 day |
| Medium | Between 1 day and 1 week |
| Low | More than 1 week |

The elite threshold of under one hour does not mean root cause identification within one hour. It means user impact restored within one hour, full stop. The fastest path to restoring service is almost never finding the root cause and fixing it. The fastest path is rolling back the deployment that caused the problem. A practiced rollback to the previous known-good artifact takes 2–5 minutes for a containerized service with a working CD pipeline. The root cause investigation happens after service is restored, on a stable system, without the urgency of a production fire making every decision feel consequential. This is the [Mitigate First, Diagnose Later](/blog/software-development/site-reliability-engineering/mitigate-first-diagnose-later) principle that good SRE practice formalizes.

The DORA research found a 2,604-fold difference in MTTR between elite and low-performing cohorts. Low-performing teams measured in weeks. Elite teams measured in minutes. This gap is not primarily about technical sophistication — it is about practiced response. Elite teams have runbooks, they have practiced rollback, they have clear incident severity definitions, and they have a culture where the default response to a production incident is mitigation-first, not investigation-first.

### The Connection to Error Budgets

MTTR is the second half of reliability accounting. The error budget framework described in [The Error Budget: The Currency of Reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability) defines how much downtime or degradation is acceptable per window. The relationship between CFR, MTTR, and error budget consumption is direct:

```
Error budget consumed per incident ≈ (incident duration) / (SLO window)
```

A team with a 99.9% monthly uptime SLO has 43.8 minutes of error budget per month. If every production incident lasts 45 minutes (MTTR), a single incident exhausts the entire monthly budget. If MTTR is 4 minutes (elite-level), the team can absorb 10 incidents per month and still meet their SLO.

This is why optimizing MTTR — building good rollback automation, practicing incident response, writing runbooks, investing in observability — is not just an operational hygiene exercise. It is a delivery enablement investment. A team that can restore service in 4 minutes has effectively purchased a much larger safety margin for experimentation.

![A deploy cycle timeline from PR open through CI through staging through production deploy to validation, showing where each DORA clock starts and stops](/imgs/blogs/dora-metrics-measuring-delivery-performance-6.png)

### Instrumenting MTTR

MTTR is measured from your incident management system. The key integration points are:

```yaml
# PagerDuty webhook handler: compute and record MTTR on incident resolution
# Deploy as a Cloud Function or Lambda that PagerDuty calls on incident state changes
incident_resolved_handler:
  trigger: pagerduty_webhook
  filter: event.type == "incident.resolve"
  actions:
    - compute_mttr:
        incident_start: event.incident.created_at
        # Use the earliest alert timestamp if available (closer to actual impact start)
        incident_actual_start: |
          MIN(
            event.incident.created_at,
            event.incident.first_trigger_log_entry.created_at
          )
        incident_end: event.incident.resolved_at
        duration_seconds: |
          (incident_end - incident_actual_start).total_seconds()

    - push_to_collector:
        url: "${DORA_COLLECTOR_URL}/events/incident_resolved"
        body:
          service: event.incident.service.name
          incident_id: event.incident.id
          severity: event.incident.urgency
          started_at: incident_actual_start
          resolved_at: incident_end
          mttr_seconds: duration_seconds
          caused_by_deployment: |
            lookup_recent_deployment(service, started_at, attribution_window_hours=4)
```

For teams not running a custom webhook handler, most DORA platforms (Faros.ai, LinearB, DX Platform) natively ingest PagerDuty and OpsGenie webhooks and compute MTTR automatically.

## The Correlation Proof: What Accelerate Actually Found

The strongest skeptical objection to DORA metrics is legitimate: maybe high-performing teams score better on these metrics because they are better-funded, have more senior engineers, work on less complex systems, or exist in industries with lower regulatory burden. The data might show that good teams have good metrics without the metrics causing good outcomes. The correlation might be real while the causal story is wrong.

The DORA research team took this objection seriously. The Accelerate book devotes a full methodological chapter to the Structural Equation Modeling approach used to test causal hypotheses in the data. The abbreviated version: SEM allows you to specify a causal model (A causes B, B causes C) and estimate the path coefficients and model fit statistics from observational data. The research tested multiple competing models — including models where organizational performance caused software delivery performance, rather than vice versa — and found that the data fit the causal model (technical practices → delivery performance → organizational outcomes) significantly better than the reverse.

Some key numbers across the research history:

| Year | Elite deploy frequency vs. low | Elite lead time vs. low | Elite MTTR vs. low | Elite CFR vs. low |
|------|-------------------------------|------------------------|-------------------|------------------|
| 2019 | 208× higher | 106× faster | 2,604× faster | 7× lower |
| 2021 | 973× higher | 6,570× faster | 6,570× faster | 3× lower |
| 2023 | Multiple/day vs. monthly | <1 hr vs. 1–6 months | <1 hr vs. >1 week | <5% vs. >30% |

The variation between years reflects changes in cohort definitions and survey methodology. The magnitude is consistently large. The direction is consistently the same.

The 2022 report introduced a finding that directly addresses the "maybe elite teams are just richer" objection: the research found that delivery performance is not correlated with industry, company size, or application complexity. A team at a 50-person startup and a team at a 50,000-person enterprise can both reach the Elite cohort. The determining factors are technical practices (continuous delivery, trunk-based development, test automation, loosely coupled architecture) and cultural practices (learning from failures, psychological safety, good change management processes) — not budget.

The 2023 report also found that deployment pain — defined as fear, anxiety, and time burned on deployment activities — was the single strongest predictor of poor delivery performance. Teams that dread deploying deploy less. They batch changes. Batches grow. Risk grows. Failures increase. Fear intensifies. DORA metrics make this spiral legible before it becomes irreversible.

## How to Instrument Your Pipeline for Honest DORA Data

The word "honest" is doing real work in this heading. DORA metrics computed from developer surveys, manual ticket updates, or ad hoc retrospective estimates are subject to systematic bias:

- Engineers overestimate their deploy frequency because they count deploys to staging.
- Engineers underestimate their lead time because they start the clock at PR open rather than first commit.
- Teams undercount incidents because marking an incident as "deployment-caused" feels like blame.
- MTTR is underestimated when engineers stop the clock at "incident acknowledged" rather than "service restored."

Honest DORA instrumentation requires an automated event pipeline where every measurement comes from system events, not human memory.

![DORA instrumentation pipeline showing the event flow from VCS commit through CI through deployment to metrics dashboard](/imgs/blogs/dora-metrics-measuring-delivery-performance-5.png)

The five components of an honest instrumentation pipeline:

**1. VCS event hook**: GitHub or GitLab webhooks emit on every push to main. The event payload includes commit SHA, author, timestamp, and parent SHAs (for computing the commit boundary between this and the previous deployment batch). This feeds lead time stage 1 (coding time, measured as time from this commit to PR open).

**2. CI metadata recorder**: Your CI system records the build result, test results, duration, and artifact SHA at job completion. This feeds lead time stage 3 (CI run time) and provides a secondary signal for CFR (a deploy with a partially failing CI run deserves closer scrutiny in incident attribution).

**3. Production deployment event**: This is the most critical event. It fires at the exact moment a new artifact begins serving production traffic. It must include the deploy timestamp, the artifact SHA, the list of commits since the previous deploy (for lead time computation), and the service name (for incident attribution). This event is the anchor for both deploy frequency and the CFR attribution window.

**4. Incident system integration**: PagerDuty or OpsGenie webhooks fire on incident creation, acknowledgment, and resolution. Your DORA collector computes MTTR from the resolution event (start time = incident.created_at approximated by first alert; end time = incident.resolved_at). It also runs the CFR attribution logic: for each incident, look up whether a deployment to the same service occurred within the attribution window.

**5. Metrics store and dashboard**: Purpose-built DORA collectors (Faros.ai, LinearB, Cortex, DX Platform) aggregate these events into the four metrics and serve them via API and dashboard. Custom implementations can use Prometheus pushgateway plus Grafana. The key requirement is that every number on the dashboard traces back to a specific event in the event log — no numbers that are computed from estimates.

### Choosing a DORA Tooling Stack

Three tiers of tooling exist, each with different cost and control trade-offs:

**Tier 1: Purpose-built DORA platforms** (Faros.ai, LinearB, DX Platform, Cortex). These connect to GitHub, Jira, PagerDuty, and your CI system via OAuth and compute all four metrics automatically. Setup takes 2–4 hours. The dashboard is ready on day one. The trade-off is vendor dependency (if they change their attribution model, your historical data changes meaning) and cost (\$15–50 per engineer per month at current pricing).

**Tier 2: GitHub DORA Insights** (available in GitHub Enterprise). GitHub now ships a built-in DORA metrics dashboard for Enterprise customers. It computes deploy frequency from the GitHub Deployments API and lead time from commit-to-deploy timestamp pairs. CFR and MTTR require a connected incident management system (PagerDuty integration is available). Free with GitHub Enterprise but requires intentional Deployments API instrumentation.

**Tier 3: Custom Prometheus + Grafana**. Maximum control, highest setup cost. You write the event pipeline as described in this post, push metrics to a Prometheus pushgateway, and build Grafana dashboards. The Prometheus recording rules for DF and lead time are straightforward. CFR requires a join between incident and deployment series, which PromQL does not handle natively — most custom implementations use a Go or Python sidecar that runs the join query on a schedule and publishes derived metrics.

For teams starting from zero, Tier 1 (a purpose-built DORA platform) is the right choice. The overhead of building a custom pipeline is justified only after you have enough experience with the metrics to know exactly which edge cases your attribution model needs to handle, and which custom dimensions (per-team, per-service, per-deployment-type) your organization needs.

### DX Platform: Correlating Metrics with Developer Experience

DX Platform (formerly DX Data) takes a distinct approach to DORA instrumentation: it combines automated metric collection from GitHub and PagerDuty with a developer survey that measures subjective friction. The value of this combination is identifying which pipeline friction most suppresses throughput. A team where CI is slow but developers rate it as "not a big deal" has a different problem than a team where CI is fast but developers rate code review as "consistently blocking."

The DX Platform survey cadence is two weeks, timed to be lightweight (fewer than ten questions per session). The correlation engine maps self-reported friction signals to objective metric movements: if developers report review friction as high in a two-week window and lead time is simultaneously elevated, that correlation is flagged automatically. This helps avoid the "we optimized CI for three months but the bottleneck was culture" failure mode described in the lead time section above.

## The Improvement Flywheel: Using Metrics to Find the Bottleneck

The DORA metrics are a diagnostic, not a report card. The goal is not to achieve a specific number to present to management. The goal is to find the highest-leverage bottleneck in your delivery system and remove it. The four metrics structure that search.

The improvement flywheel has four steps:

**Step 1: Establish a four-to-six-week baseline.** A single bad week inflates CFR and deflates deploy frequency. A holiday with no deployments depresses DF for the week. Anomalies like these require enough historical data to average out. Twenty to forty production deployments is a reasonable minimum for stable CFR estimation. Six weeks of data is a reliable baseline for most teams.

**Step 2: Identify the worst metric against the cohort bands.** Not by feel, not by engineer survey, by data. Plot all four metrics against the cohort thresholds. One metric is usually the clear outlier — a team on the boundary between Medium and High on three metrics but sitting solidly in Low on one has a specific bottleneck to diagnose.

**Step 3: Decompose the worst metric to find the root cause.** A high lead time could be caused by PR review wait, CI duration, manual approval gates, long-lived feature branches, or late ticket opening. Each has a different fix. A high CFR could be caused by large batch sizes, insufficient test coverage, flaky tests that trained engineers to ignore test failures, lack of canary validation, or insufficient feature flag discipline. Decompose before prescribing. Prescribing before diagnosing is how teams run "CI optimization" projects for three months while their lead time stays flat because the bottleneck is PR review.

**Step 4: Run a time-boxed experiment with a measurable hypothesis.** "If we introduce a PR review SLO of four hours within business hours, our median lead time will drop from 3.5 days to under one day within six weeks." This framing forces you to state a prediction, measure whether the intervention worked, and learn from the result — even if the result is that you were wrong about the root cause. Time-boxing to four to eight weeks prevents projects from dragging on indefinitely.

Then repeat. As you close the gap on the worst metric, another metric becomes the bottleneck. The flywheel keeps moving.

### Running a DORA Improvement Sprint

A DORA improvement sprint is a four-week focused cycle that follows the flywheel steps above with specific team rituals:

**Week 0 (prep)**: Pull the four metrics for the trailing six weeks. Build the per-stage decomposition for the worst metric. The team lead writes a one-page problem statement: which metric is the outlier, which stage within that metric is the dominant contributor, what the team believes the root cause is, and what intervention they will test. This document is the sprint contract.

**Week 1**: Implement the intervention. For a PR review SLO intervention: configure the GitHub bot, write CODEOWNERS, announce the SLO in the team channel, and measure day-by-day PR review wait time. The goal in week 1 is to confirm the intervention is running, not to show a metric improvement — most improvements take two to three weeks to show up in the rolling averages.

**Week 2–3**: Monitor the target metric and the correlated metrics. CFR is the canary for every throughput improvement: if you increase deploy frequency and CFR rises, the pipeline is not ready for the higher cadence. Watching both simultaneously prevents optimizing one metric into a worse system overall.

**Week 4 (retro)**: Compare the six-week trailing average at sprint start to the current four-week average. Did the metric move in the predicted direction? By how much versus the prediction? What does the decomposition show now — is the same stage still dominant, or has the bottleneck shifted? Write the findings into your team wiki. Even a failed hypothesis teaches something about where the constraint actually lives.

The most common failure mode of a DORA improvement sprint is measuring the wrong stage. A team that identifies "long lead time" as the problem but decomposes it only into "merge to production" rather than "first commit to production" will optimize the wrong segment. Always decompose from the DORA definition's start point (first commit), not from wherever your tooling makes it easy to start measuring.

### The Priority Order for Metric Improvement

![DORA improvement decision tree routing teams from worst metric to highest-leverage action, with CFR stabilization required before throughput optimization](/imgs/blogs/dora-metrics-measuring-delivery-performance-8.png)

When all four metrics are below the Elite threshold, improvement order matters because the metrics interact. The recommended sequencing from the DORA research and practitioner experience:

**First: stabilize CFR.** If CFR exceeds 15%, every improvement you make to deploy frequency compounds failures faster than the team can recover. A team moving from 2 deploys/week to 14 deploys/week at 20% CFR triples their incident load without having done anything about recovery. Stabilize CFR to under 10% before scaling throughput.

**Second: reduce MTTR.** With CFR in the High cohort (under 15%), the priority shifts to reducing MTTR. The reason: lower MTTR expands the error budget, which gives the team license to take on more delivery risk. A team that can restore service in 15 minutes can deploy riskier changes more confidently than a team that takes 4 hours to recover.

**Third: compress lead time.** Lead time compression is the highest-leverage throughput investment once the system is stable enough to handle the increased cadence. The PR review SLO intervention described above is the most common first action here.

**Fourth: increase deploy frequency.** Deploy frequency should increase as a natural consequence of reducing lead time and improving stability. Forcing DF upward without fixing the underlying constraints (batch size, quality gates, deployment automation) is exactly the gaming pattern that Team Gamma demonstrated in the introduction. DF that rises because changes are small, fast to review, and safe to deploy is real improvement. DF that rises because engineers split coherent changes into artificial fragments is Goodhart's Law in action.

#### Worked example: The six-month DORA improvement map

A B2B SaaS platform team of twelve engineers operates in the Medium cohort across all four metrics as of month zero:

- Deploy Frequency: 2× per week (target: High cohort = daily or better)
- Lead Time: 3.5 days median (target: under 1 day)
- CFR: 22% (target: under 15%)
- MTTR: 4.2 hours median (target: under 1 hour)

They analyze their incident history for the past quarter. Of 22 incidents, 18 opened within two hours of a deployment to the affected service. Of those 18, 14 were resolved by rollback (the failure was detectable at deploy time and the rollback resolved it). This means 14 of their 22 incidents were potentially catchable with better deploy-time validation.

They analyze their lead time decomposition and find PR review wait is 58 hours out of 84.6 hours total.

**Month 1–2 (Fix CFR bottleneck first — it is in the danger zone at 22%)**: They add production synthetic smoke tests to the deploy pipeline. A set of 12 synthetic checks runs against the newly deployed instance; if more than two fail within the first five minutes, the deployment auto-rolls-back. They add a canary deployment stage: 5% of traffic routes to the new version for 10 minutes before full rollout. CFR drops to 9% — the 14 "catchable at deploy time" incidents are now caught before full rollout.

**Month 2–4 (Attack lead time now that CFR is in the High cohort)**: They introduce a PR review SLO (first review within four hours during business hours) backed by a Slack bot. They add GitHub CODEOWNERS and automatic reviewer assignment. They run trunk-based development training and retire two long-lived feature branches. Lead time drops from 3.5 days to 11 hours median.

**Month 4–5 (Deploy frequency now matters — the pipeline is trustworthy enough to run it often)**: With CFR at 9% and lead time at 11 hours, they safely increase deploy frequency. They automate the manual final approval gate (it was confirming CFR below 15%; it now is). DF increases from 2× per week to 3–4× per day.

**Month 5–6 (MTTR is now the outlier)**: Analysis shows engineers spend 2–3 hours diagnosing before rolling back. They implement a strict "rollback within 15 minutes of confirmed degradation" policy with automated rollback triggers for the most critical services. MTTR drops from 4.2 hours to 28 minutes median.

At month six, the team is solidly in the High cohort and knocking on Elite for MTTR and DF.

![Before-and-after comparison of a team running a six-month DORA improvement program, showing metric progression from Medium cohort to High cohort across all four dimensions](/imgs/blogs/dora-metrics-measuring-delivery-performance-7.png)

## War Story: Gaming Deploy Frequency and How CFR Caught It

In 2022, a growth engineering team at a fintech startup instrumented DORA metrics for the first time and set quarterly OKRs tied to deploy frequency. The leadership target was to reach the High cohort (daily or better deployments). Within three weeks the team was reporting 5–7 deploys per day, well into High territory.

Two months into the quarter, their Change Failure Rate had climbed from 14% to 31%. The correlation was visible in the data but invisible in the team's daily experience. Each individual deploy seemed small and harmless. Post-incident analysis of a P2 incident in week eight revealed the pattern: engineers had been splitting changes that naturally belonged together — a database schema migration in one deploy, the application code that depended on it in a second deploy, the configuration that activated the feature in a third. Each piece was technically deployable in isolation. But the window between the schema migration and the application deploy, which sometimes spanned four to six hours, created an inconsistency that caused failures in the legacy code path that was still reading the old schema.

The anatomy of the gaming was worth studying in detail. Each of the three fragments passed CI. The schema migration deployed cleanly: it added a new column with a nullable default, so existing queries continued working. The application code deployed cleanly: in isolation, it was valid code that referenced the new column, which now existed. The configuration deploy that activated the feature deployed cleanly: the feature flag was wired to the right code path. What failed was the interaction: during the four-to-six-hour window between the schema deploy and the application code deploy, a background worker that ran on a tight loop was calling a stored procedure that read the old column, and the stored procedure had been updated in the schema migration to require the new column. The background worker had not been restarted. The result was a cascading failure of background job processing that was not caught by synthetic monitors because synthetic monitors tested the user-facing API, not the background worker fleet.

The deeper engineering failure was that the team had no concept of "atomic deployment group" — a set of changes that must be deployed together within a short time window to be safe. Their pipeline had no mechanism to express this constraint. When the incentive to increase DF was introduced, the natural response was to deploy each piece as soon as it was ready, without coordinating the window.

The root cause was not bad engineering judgment. It was misaligned incentives. The metric said "deploy frequently." The team deployed frequently. The metric said nothing about whether those deploys represented complete, coherent changes. CFR said something about it very clearly — and CFR had not been part of the OKR.

The fix was not to reduce deploy frequency. It was to redefine the deployment unit: only atomically complete changes counted as deployments in their DF metric. Schema migrations without their corresponding application code were tagged as "infrastructure deploys" and excluded from DF counting. This required a pipeline change to allow tagged deploys and a monitoring alert when infrastructure deploys were not followed by application deploys within a configurable window. DF dropped back to 1–2 true feature deploys per day. CFR returned to 11% within six weeks. Lead time — the metric they actually wanted to improve — had improved meaningfully because eliminating the artificial splitting reduced coordination overhead and reviewer confusion.

The structural insight from this story: the two pairs are self-correcting. You cannot sustainably improve throughput metrics while ignoring stability metrics, because stability is the constraint that makes throughput meaningful. The self-correcting property is the system design that makes DORA metrics hard to Goodhart permanently — you might game one metric for a quarter, but the other pair will expose it.

## When Not to Use DORA Metrics: The Goodhart Trap Inventory

DORA metrics are not universally the right instrument. Here are the failure modes to actively guard against.

### The Six Classic Gaming Patterns

Before listing the traps, it is useful to enumerate the most common gaming patterns with concrete examples, because recognizing them in practice is the first defense:

**Pattern 1: Deploy fragment inflation** (the war story above). Splitting one coherent change into N artificial deployment units to inflate DF. Defense: define "deployment" as a unit that represents a complete, independently releasable change. Add pipeline enforcement: schema migrations must be followed by application deploys within a window. Track "deployment cohesion" as a secondary metric (what percentage of deploys in a window are atomic versus fragment-of-larger-change).

**Pattern 2: Staging-to-production inflation**. Counting deploys to staging environments as production deployments. This is the simplest gaming pattern and the most common. Defense: emit deployment events only from the production environment tag in your CI/CD platform. Never count events from non-production environment contexts.

**Pattern 3: Ticket time manipulation for lead time**. Opening tickets immediately when a sprint starts (so the ticket is "in progress" from day one) but not committing code until day four. This inflates coding time and makes lead time appear shorter than it is when measured from first commit. Defense: measure from first commit, not from ticket open or sprint start. Use automated git-based timing, not ticket system timestamps.

**Pattern 4: Incident suppression for CFR**. Not opening incidents in the incident management system, resolving incidents quickly before they trigger the attribution window, or tagging incidents as "external cause" when they were clearly deployment-related. Defense: require on-call engineers to open an incident for any paging alert regardless of cause (even if the root cause turns out to be external, the incident documents the investigation). Require a second engineer to approve "external cause" override tagging. Review the override log quarterly.

**Pattern 5: MTTR clock-stopping manipulation**. Marking incidents as resolved when the on-call is confident the issue is fixed, before confirming that user-facing symptoms have actually stopped. The clock stops but the users are still experiencing degradation. Defense: define "resolved" as a customer-facing SLI returning to baseline (for example, error rate below 0.1% for five consecutive synthetic check windows). Automate the resolved timestamp from the SLI returning to normal, not from human judgment.

**Pattern 6: Batch suppression for CFR**. Reducing deploy frequency to reduce the denominator of the CFR ratio, artificially improving the percentage without changing the absolute number of production failures. Defense: track both CFR (percentage) and absolute failure count (incidents per week). Report them together. A team with 1 incident per month has very different health than a team with 4 incidents per month, even if both have 5% CFR at their respective deploy frequencies.

### The Six Traps

**Trap 1: Individual performance reviews.** The DORA research measures team-level and organizational-level delivery performance. An individual engineer's contribution to deploy frequency or lead time is not meaningfully separable from team behavior, pipeline infrastructure, and organizational culture. Using DORA metrics in individual performance evaluations — "your team's CFR was 18%, reflecting on your quality standards" — is methodologically invalid and produces exactly the gaming incentives the metrics are designed to prevent.

**Trap 2: Cross-team comparison without normalization.** A team maintaining a payment processing service and a team iterating on a marketing blog are not comparable on deploy frequency. The nature of the service, the risk tolerance of the business, the regulatory environment, and the change velocity requirements all affect what "good" looks like on each metric. Presenting raw DORA numbers in a cross-team leaderboard is a fast path to the gaming dynamics described in the war story above.

**Trap 3: Treating the Elite cohort as a mandatory target for all teams.** Multiple deploys per day is the right operating model for most consumer-facing web services and microservices. It is the wrong model for embedded firmware shipped to physical devices, regulated medical software with mandatory validation periods, database migration-heavy services where deploy risk is structurally elevated, or services that require coordinated cross-team releases. The value of DORA metrics is your team's trend over time and your positioning relative to comparable teams, not achievement of an absolute benchmark.

**Trap 4: Quarterly-only review.** Teams that review DORA metrics quarterly are using them as a history lesson. Teams that track them weekly — with alerts when any metric crosses a cohort threshold — are using them as a control system. The control system usage is the one with predictive power. A CFR that climbs from 5% to 12% over four weeks is actionable. A CFR that was 5% in Q1 and is 12% in Q2 is a post-mortem.

**Trap 5: Ignoring instrumentation quality.** DORA metrics are only as honest as the events feeding them. A team where engineers manually close incidents without marking them as deployment-caused has systematically understated CFR. A team whose CD pipeline emits a "deployment" event when the artifact builds rather than when it reaches production has overstated DF. Audit the event pipeline before trusting any trend.

**Trap 6: Optimizing CFR by slowing deployments.** This is the most common management-level failure mode. A team with 22% CFR is told to improve it. They add a mandatory 48-hour staging soak before every production deployment. CFR drops to 10% because the longer soak catches more failures before they reach users. Lead time climbs from 3.5 days to 5.5 days. The Goodhart trap has been obscured by the CFR improvement, but total delivery performance has declined. The correct response to high CFR is better quality gates earlier in the pipeline (automated testing, synthetic validation at deploy time, canary deployments) — not longer delays.

## Further Tools and Integration Patterns

The tooling landscape for DORA metrics has matured significantly since the Accelerate research first popularized the four metrics. Here is a current (mid-2026) snapshot of the major integration patterns:

**LinearB**: Connects to GitHub and Jira via OAuth. Computes DF, lead time, and a "coding time" breakdown without requiring custom pipeline instrumentation. CFR and MTTR require PagerDuty/OpsGenie integration. Strong on PR-level analytics (review time distribution, PR size vs. CFR correlation). Pricing: per-engineer SaaS.

**Faros.ai**: More infrastructure-adjacent than LinearB. Connects to CI systems, deployment platforms (ArgoCD, Spinnaker, Harness), and incident tools. Produces DORA metrics alongside a wider engineering effectiveness dashboard. Stronger for large engineering organizations with multiple CI/CD platforms. Pricing: enterprise contract.

**GitHub DORA Dashboard** (GHE): Available in GitHub Enterprise. Uses the GitHub Deployments API as the DF/LT anchor. Requires teams to call the Deployments API from their pipelines rather than relying on push events. Simple if your team already uses GitHub Environments.

**Google Cloud's Four Keys** (open source): A BigQuery-based data pipeline that ingests GitHub, GitLab, and PagerDuty events into a BigQuery dataset and serves them via a Looker Studio dashboard. Completely free, self-hosted. The reference implementation for "build your own DORA stack." Requires GCP. The BigQuery SQL transformations are readable and auditable, making them a good baseline for custom attribution logic.

**DX Platform**: Developer experience platform that combines DORA metrics with developer survey data and flow time analysis. Notable for correlating self-reported friction with objective delivery metrics — useful for the "which friction most suppresses throughput" question.

For most teams starting their DORA journey, the recommendation is to begin with a commercial platform (LinearB is the most common starting point) to get the baseline without infrastructure investment, then consider migrating to a custom or open-source stack once you understand your specific measurement edge cases.

### The Custom Prometheus + Grafana Stack in Practice

For teams that want full control, the complete custom stack involves five components: a deploy event webhook receiver, a lead time computation job, an incident attribution sidecar, a Prometheus pushgateway, and a set of Grafana dashboards. The total implementation cost for an experienced platform engineer is roughly three to five days for the first service, and thirty to sixty minutes per additional service once the boilerplate is established.

The lead time computation job runs at deploy time (triggered by the same deploy event webhook) and pushes a single Prometheus gauge per deployment:

```bash
# Push lead time as a Prometheus gauge via pushgateway
curl --fail -s -X POST \
  "http://pushgateway.internal:9091/metrics/job/dora_lead_time/service/${SERVICE_NAME}" \
  --data-binary "
# HELP dora_lead_time_seconds Lead time for changes in seconds
# TYPE dora_lead_time_seconds gauge
dora_lead_time_seconds{service=\"${SERVICE_NAME}\",deploy_ref=\"${CURRENT_SHA}\"} ${LEAD_TIME_SECONDS}
"
```

The Prometheus recording rule for median lead time uses `quantile_over_time`:

```
# Approximate 50th-percentile lead time over 28 days
# (requires Prometheus Agent with histogram support or a custom aggregation job)
record: dora:lead_time_p50_hours:28d
expr: |
  quantile_over_time(0.5,
    dora_lead_time_seconds{environment="production"}[28d]
  ) / 3600
```

Note: `quantile_over_time` in PromQL works on gauge time series but requires sufficient data density (at least 20–30 data points in the window) for the percentile estimate to be stable. For teams with fewer than 20 deployments per 28-day window, compute the percentile in a Python sidecar and push the result directly as a scalar gauge.

## Key Takeaways

1. **The two-pair structure is the system's integrity check.** Deploy Frequency and Lead Time measure throughput. Change Failure Rate and Time to Restore measure stability. Improving one pair without the other almost always reveals gaming. Measure all four simultaneously.

2. **Lead time starts at first commit, not at PR open.** Changes sitting in long-lived feature branches accumulate invisible lead time. If your measurement excludes pre-PR time, you have a systematically optimistic view of your delivery velocity.

3. **PR review wait time is almost always the lead time bottleneck.** Audit your decomposition before investing in CI optimization. Cutting 30 minutes from a 22-minute CI run when PR review wait is 58 hours is a category error in prioritization.

4. **CFR attribution must be automated with consistent rules.** Manual incident tagging underestimates CFR. Use time-window attribution (deployment within N hours before the incident) with manual override for confirmed external causes. Consistency over precision.

5. **MTTR is measured from impact start, not detection time.** Detection delays belong in your MTTR number. A 20-minute gap before paging is 20 minutes of user impact. Instrument first-alert timestamps as the incident start proxy.

6. **Rollback first, diagnose later.** Fast MTTR comes from practiced rollback, not faster root-cause analysis. The on-call's job in the first 15 minutes is to restore service; the post-mortem's job is to find the cause.

7. **Never use DORA metrics in individual performance reviews.** They are team and system metrics. Individual attribution corrupts the incentive structure immediately and thoroughly.

8. **Fix stability before scaling throughput.** If CFR exceeds 15%, increasing deploy frequency compounds failures faster than the team can absorb. Stabilize CFR first; then accelerate.

9. **The improvement flywheel is worst-metric-first.** Decompose the worst metric, run a time-boxed experiment, measure the result, repeat. Trying to improve all four simultaneously diffuses focus and makes it impossible to attribute causes.

10. **Honest instrumentation beats precise instrumentation.** A slightly less precise metric automatically collected from system events is far more valuable than a highly precise metric that depends on human memory to record. Automate the pipeline; trust the events.

11. **Gaming the pairs is self-defeating.** The two-pair design makes permanent Goodhart gaming structurally difficult: inflating DF raises CFR, suppressing CFR by slowing deploys inflates Lead Time. The metrics expose each other. Use all four in every review.

## Further Reading

- **Accelerate: The Science of Lean Software and DevOps** by Forsgren, Humble, and Kim (IT Revolution Press, 2018). The foundational book. The structural equation modeling chapter is the strongest rebuttal to the correlation-vs-causation objection. Every engineering leader who wants to argue from data about delivery practices should read it.

- **DORA State of DevOps Reports** (dora.dev/research). Annual survey data from 2014 through present. The 2019 report has the most striking raw numbers. The 2023 report has the most current cohort definitions and includes the Reliability metric. All reports are freely downloadable.

- **DORA DevOps Quick Check** (dora.dev/quickcheck). A five-minute self-assessment that places your team in the four cohorts based on self-reported metrics. The right starting point before you build out full pipeline instrumentation.

- **DORA Metrics Specification** (dora.dev/guides/dora-metrics-four-keys). The canonical definition of each metric, including the attribution edge cases that seven years of data collection have forced the research team to adjudicate. If you are building a custom collector, read this before you write the first line of attribution logic.

- **The Four Keys Project** (github.com/GoogleCloudPlatform/fourkeys). Google's open-source DORA metrics pipeline using BigQuery and Looker Studio. A working reference implementation of the instrumentation concepts described in this post. The SQL transformations are readable and a good baseline for custom attribution logic.

- **LinearB Engineering Metrics Guide** (linearb.io/blog/dora-metrics). Practical walkthrough for connecting Linear, GitHub, and PagerDuty into a commercial DORA dashboard. Useful for teams choosing Tier 1 tooling.

- **The Error Budget: The Currency of Reliability** at [/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability). The SRE framework that operationalizes the CFR and MTTR pair as a negotiated reliability contract between teams.

- **From Commit to Production** at [/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model). The pipeline architecture that DORA metrics instrument and the structural prerequisites for elite performance.
