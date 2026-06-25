---
title: "Pipeline Observability and the Flaky Pipeline"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Treat your CI/CD pipeline as a production system — instrument it with SLOs, flaky-test quarantine policies, and DORA dashboards — and cut flaky-caused failures from 35% to under 3% while slashing build time from 38 minutes to 6."
tags:
  [
    "ci-cd",
    "devops",
    "observability",
    "pipeline",
    "flaky-tests",
    "dora-metrics",
    "monitoring",
    "slo",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/pipeline-observability-and-the-flaky-pipeline-1.png"
---

It started with a Slack message that had become a reflex: "CI is red again — just retry it."

No one read the failure log anymore. The failure was almost certainly flaky. The retry would pass. The PR would merge. That was the deal the team had made, implicitly, over the course of several months. They had started reading failure logs, then learned that reading them was usually pointless (the test was just flaky), then stopped reading them, then collectively trained themselves that a red CI run was noise to be filtered rather than a signal to be investigated.

The day it caught up with them started quietly. A routine deploy: a small feature-flag refactor, low risk, had passed CI (on the second attempt, naturally). The staging canary triggered. The on-call engineer glanced at the alert, saw that CI for the commit was green (on a retry), and moved on. The canary kept firing. Thirty minutes later it was clearly not noise — the new code had a real regression in payment handling. By the time the incident was declared and the rollback was deployed, three hours had passed and \$47,000 in transactions had failed silently.

The post-mortem found one root cause: the team had stopped trusting CI. They had retried without reading. They had shipped because the build was "green enough." The underlying flaky test rate was 42%. Almost half of all CI failures, over the preceding two months, had been resolved by retry without any code change. Engineers had retried 3.4 times per PR on average. No one had a dashboard. No one owned the pipeline's reliability. No one had ever defined what "acceptable CI health" even meant.

This is not a hypothetical. It is a real incident pattern that the CI/CD research community has catalogued repeatedly, and it is the inevitable consequence of treating the pipeline as infrastructure-as-furniture — something that was built once, is vaguely unreliable, and is someone else's problem.

This post is about fixing that. We will treat the pipeline as a production system. We will instrument it. We will define SLOs for it. We will build a quarantine process that eliminates flaky-test noise systematically. We will profile the slow pipeline and cut it by a factor of six. We will wire up the DORA dashboard that makes delivery performance visible without any manual data entry. And we will do all of this with real artifacts — Prometheus rules, OpenTelemetry config, GitHub Actions YAML, Python scripts — not hand-waving.

![Pipeline telemetry flow: CI job emits spans to OTel collector, which fans out to Prometheus, Grafana Tempo, and a DORA event store, all visualised in Grafana with SLO alerting](/imgs/blogs/pipeline-observability-and-the-flaky-pipeline-1.png)

The telemetry architecture is simple: every CI job emits OpenTelemetry spans (for tracing) and Prometheus metrics (for alerting), and a deploy event to a lightweight event store (for DORA). The OTel collector fans them out to Grafana Tempo, Prometheus/Mimir, and the event store. Grafana queries all three. When an SLO burns too fast, Alertmanager pages the platform on-call. That is the whole system. You can build it in a single sprint.

---

## The Pipeline Is a Production System

Let us start with the assertion that most engineering organisations resist: your CI/CD pipeline is a production system, and it should be operated like one.

Consider what the pipeline does. It accepts a work item — a commit — from an engineer, processes it through a sequence of stages (build, test, scan, package, deploy), and produces a result: either a deployed service that is serving real traffic, or a clear failure signal that tells the engineer what is wrong and why. It has users: every engineer on your team, possibly hundreds of them. It has a latency SLA: engineers expect a result within a human-attention span (roughly 10–15 minutes for interactive development; 30 minutes at the outer bound). It has a reliability SLA: engineers expect that a green build genuinely means the code is safe to ship, and that a red build genuinely means something is wrong. And it has business impact when it fails: no deployments means no features, no bug fixes, no incident responses, no on-call rollbacks.

The reason most teams do not treat it this way is a combination of historical accident and category error. The pipeline was written as YAML. It lives in a `.github/workflows` folder. It is "just CI" — not the product, not production, not worth the same engineering investment as the API or the database. This is a category error. The pipeline is as much a part of the production system as the load balancer. It just happens to be in a different repo.

### The cost of unobserved pipelines

The costs of treating the pipeline as unobserved infrastructure are concrete and measurable.

**Lead time inflation.** The [DORA research](/blog/software-development/ci-cd/dora-metrics-measuring-delivery-performance) identifies lead time for changes — the time from first commit to production — as one of the strongest predictors of engineering performance. A pipeline that runs for 38 minutes instead of 6 adds 32 minutes of lead time to every change. For a team making 40 PRs per day with an average of 1.8 CI runs per PR (accounting for retries), that is 40 × 1.8 × 32 = 2,304 minutes of added lead time per day — 38 engineer-hours. You are paying for that, in engineering velocity, every single day.

**Trust erosion and change failure rate.** When engineers stop trusting CI, they start merging anyway. "The tests are probably flaky." When a real regression occurs, it gets through because the team's prior on any given red build being meaningful is low. The change failure rate rises. The MTTR rises because engineers don't recognise the regression quickly. The entire bottom half of the DORA scorecard degrades.

**Cognitive overhead.** An engineer who hits retry three times before merging has spent 10–15 minutes context-switching, waiting, wondering. Multiply that by the size of the team and the frequency of merges. The cognitive cost is real and cumulative.

**Invisible bottlenecks.** Without stage-level timing, you cannot know whether the test suite is the bottleneck or the Docker build or the dependency install. You cannot prioritise optimisation work. You cannot tell whether a change you made last week improved the pipeline or made it worse.

### SLOs, SLIs, and on-call for the pipeline

Applying the SRE model to the pipeline — which is described in depth in the [error budget post](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability) — means defining three things:

**SLIs (Service Level Indicators):** The raw measurements. For the pipeline:
- P95 build duration in seconds
- Pipeline success rate (fraction of runs that complete without failure)
- Flaky-test rate (fraction of failures resolved by retry without code change)
- Queue wait time in seconds
- Retry rate per PR

**SLOs (Service Level Objectives):** The targets. For the pipeline:
- P95 build duration < 600 seconds (10 minutes)
- Pipeline success rate > 95% over any rolling 24-hour window
- Flaky-caused failure rate < 2% of all failures
- P95 queue wait time < 60 seconds

**Error budget:** The allowable violation window. If your SLO is "95% success rate", your error budget is 5% of pipeline runs per month. When that budget is consumed, the platform team should stop feature work and fix the pipeline.

**On-call ownership:** Assign a rotation. When the pipeline SLO breaches, someone is paged. This is the step most teams skip, and it is the step that makes all the others real. An SLO without an owner is a number in a dashboard that nobody acts on.

The [CI/CD mental model post](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) frames this from first principles: the pipeline is the value stream, and anything that degrades the value stream degrades the business. Start there if you want the conceptual grounding before the mechanics.

---

## What to Instrument

Before writing any telemetry code, decide what you need to know. The following SLIs cover the full pipeline health picture. Each maps cleanly to a Prometheus metric or an OpenTelemetry attribute.

### The canonical SLI list

**Build duration (P50/P95/P99).** The most important number. P50 tells you what a typical build feels like. P95 tells you what the frustrating builds feel like — the ones that make engineers reach for coffee and check email. P99 identifies the pathological cases that block your slowest merges and often point to specific jobs that need attention (a particularly large service, a slow integration test suite, a Docker build that rebuilds from scratch).

**Queue wait time.** The time from when a pipeline is triggered to when a runner picks it up. In high-load CI environments, queue wait time can exceed actual build time. A team that upgraded their pipeline from 40 minutes to 15 minutes may find that P95 wait time is still 20 minutes because the runner pool is saturated. Queue wait time is often invisible in CI UIs because the clock starts when the job starts, not when it was queued.

**Pipeline success rate.** The fraction of pipeline runs that complete without a failure. This is your headline reliability number. A success rate of 80% means one in five pushes lands in red. An engineer doing 5 PRs per day hits a red CI run once a day. At a team of 20 engineers, that is 20 red CI runs per day that someone has to investigate or retry.

**Flaky-test rate.** The fraction of pipeline failures that are resolved by retry without any code change. This is the number the team in the opening story did not have. It is the single most diagnostic metric for CI health because it separates signal (real failures) from noise (flaky failures). A high flaky-test rate is not just a reliability problem — it is a trust problem.

**Retry rate.** The average number of retries per pipeline run. This should be close to 1.0 (one run, done). A retry rate above 1.1 means engineers are regularly hitting retry. A retry rate above 1.5 is a crisis — engineers are spending more time retrying than they are reviewing code.

**Stage-level duration.** Not just the total, but how long each stage takes: checkout, dependency install, build, test (broken by suite), scan, push. Without stage-level data, you cannot profile the pipeline or identify bottlenecks. You know the pipeline is slow; you do not know which part to fix.

**Runner availability and saturation.** The fraction of time your runner pool is fully occupied. Above 80% saturation, queue times grow rapidly (queuing theory: at 80% utilisation, the average queue length is 4 jobs; at 90%, it is 9). This is the CI equivalent of CPU saturation on a web server.

### The SLI-to-metric mapping

| SLI | Collection method | Prometheus metric name | Labels |
|---|---|---|---|
| Build duration | OTel span duration / post-job hook | `ci_job_duration_seconds` | `job`, `branch`, `status` |
| Queue wait time | Job start minus trigger timestamp | `ci_queue_wait_seconds` | `job`, `runner_group` |
| Pipeline success rate | Job exit code in post-job hook | `ci_job_success_total` + `ci_job_total` | `job`, `branch` |
| Flaky-test rate | Retry-passed flag in post-job script | `ci_flaky_failure_total` | `job`, `test_file` |
| Retry rate | Retry count in job metadata | `ci_retry_count_total` | `job`, `branch` |
| Stage duration | OTel child span per stage | `ci_stage_duration_seconds` | `job`, `stage` |
| Runner saturation | Runner self-metrics / GitHub API | `ci_runner_busy_ratio` | `runner_group` |

Instrumentation at this level does not require significant engineering investment. The post-job hook approach — a bash script that runs at the end of every job and pushes metrics to a Prometheus Pushgateway — can be implemented in a day and gives you 90% of the signal.

---

## Emitting Pipeline Telemetry

There are three complementary methods for getting telemetry out of a CI pipeline. Use all three: OTel for traces (debugging individual slow runs), Prometheus for metrics (alerting and trending), and structured events for DORA (delivery performance).

### OpenTelemetry spans for build jobs

OpenTelemetry lets you model a pipeline run as a distributed trace: a root span for the pipeline run, child spans for each stage, and attributes that carry context (commit SHA, branch, PR number, author, test counts). The idea is that the pipeline is a request, stages are subrequests, and the trace shows you where the time went — the same way you would debug a slow API call.

Here is a complete GitHub Actions workflow that wraps every stage in an OTel span using the `otel-cli` binary:

```yaml
# .github/workflows/ci.yml
name: CI
on:
  push:
    branches: [main, "feat/**"]
  pull_request:

jobs:
  build-test:
    runs-on: ubuntu-latest
    env:
      OTEL_EXPORTER_OTLP_ENDPOINT: ${{ secrets.OTEL_ENDPOINT }}
      OTEL_EXPORTER_OTLP_HEADERS: "Authorization=Bearer ${{ secrets.OTEL_TOKEN }}"
      OTEL_SERVICE_NAME: "ci-pipeline"
      OTEL_SERVICE_VERSION: ${{ github.sha }}
      OTEL_RESOURCE_ATTRIBUTES: >
        git.commit.sha=${{ github.sha }},
        git.branch=${{ github.ref_name }},
        ci.workflow=${{ github.workflow }},
        ci.run_id=${{ github.run_id }},
        ci.run_attempt=${{ github.run_attempt }},
        ci.actor=${{ github.actor }}

    steps:
      - name: Install otel-cli
        run: |
          curl -L https://github.com/equinix-labs/otel-cli/releases/latest/download/otel-cli-linux-amd64.tar.gz \
            | tar -xz -C /usr/local/bin

      - name: Start pipeline trace
        id: trace
        run: |
          # Start root span; export trace context for child spans
          TRACEPARENT=$(otel-cli span start \
            --name "pipeline/${{ github.workflow }}" \
            --service "ci-pipeline" \
            --kind producer \
            --attrs "ci.pr_number=${{ github.event.pull_request.number || '' }}" \
            --print-tp-header)
          echo "TRACEPARENT=$TRACEPARENT" >> $GITHUB_ENV
          echo "traceparent=$TRACEPARENT" >> $GITHUB_OUTPUT

      - name: Checkout
        run: |
          otel-cli exec \
            --name "stage/checkout" \
            --tp-carrier-env \
            -- git clone --depth=1 "${{ github.server_url }}/${{ github.repository }}" .

      - name: Cache dependencies
        id: cache
        uses: actions/cache@v4
        with:
          path: ~/.npm
          key: ${{ runner.os }}-npm-${{ hashFiles('**/package-lock.json') }}
          restore-keys: |
            ${{ runner.os }}-npm-

      - name: Install dependencies
        run: |
          otel-cli exec \
            --name "stage/install" \
            --attrs "cache.hit=${{ steps.cache.outputs.cache-hit || 'false' }}" \
            --tp-carrier-env \
            -- npm ci --prefer-offline

      - name: Build
        run: |
          otel-cli exec \
            --name "stage/build" \
            --tp-carrier-env \
            -- npm run build

      - name: Test
        id: test
        run: |
          otel-cli exec \
            --name "stage/test" \
            --tp-carrier-env \
            -- npm test -- --reporter=json --outputFile=test-results.json

      - name: End pipeline trace
        if: always()
        run: |
          STATUS="${{ job.status }}"
          OTEL_STATUS=$([ "$STATUS" = "success" ] && echo "OK" || echo "ERROR")
          otel-cli span end \
            --status-code "$OTEL_STATUS" \
            --attrs "ci.final_status=$STATUS,ci.run_attempt=${{ github.run_attempt }}"

      - name: Emit Prometheus metrics
        if: always()
        env:
          JOB_STATUS: ${{ job.status }}
          RUN_ATTEMPT: ${{ github.run_attempt }}
          PUSHGATEWAY_URL: ${{ secrets.PUSHGATEWAY_URL }}
        run: bash .github/scripts/emit-metrics.sh "build-test" "$JOB_STATUS" "$RUN_ATTEMPT"
```

With this in place, every CI run produces a structured trace in Grafana Tempo. Click into any slow run and see a waterfall of every stage — checkout, install, build, test — with exact durations and attributes.

### OTel SDK instrumentation in Python for more complex pipelines

When the pipeline logic itself is a script — a Python release orchestrator, a Gradle build plugin, a Go release tool — you can instrument it directly with the OTel SDK rather than wrapping external commands with `otel-cli`. The result is finer-grained traces with richer attributes, including span events for significant milestones and span links that connect a deploy trace to the CI trace that triggered it.

```python
#!/usr/bin/env python3
"""
pipeline_runner.py — instrument a multi-step build job with OTel spans.

Each pipeline stage becomes a child span of the root pipeline span.
Attributes carry metadata that makes traces useful in Grafana Tempo:
commit SHA, branch, attempt number, test counts, cache hit/miss.
"""

import os
import subprocess
import sys
import time
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind, StatusCode

# ── SDK bootstrap ──────────────────────────────────────────────────────────────

COMMIT_SHA = os.environ.get("GITHUB_SHA", "local")
BRANCH = os.environ.get("GITHUB_REF_NAME", "local")
RUN_ID = os.environ.get("GITHUB_RUN_ID", "0")
RUN_ATTEMPT = int(os.environ.get("GITHUB_RUN_ATTEMPT", "1"))
WORKFLOW = os.environ.get("GITHUB_WORKFLOW", "ci")

resource = Resource.create({
    SERVICE_NAME: "ci-pipeline",
    "service.version": COMMIT_SHA,
    "git.commit.sha": COMMIT_SHA,
    "git.branch": BRANCH,
    "ci.workflow": WORKFLOW,
    "ci.run_id": RUN_ID,
    "ci.run_attempt": str(RUN_ATTEMPT),
})

provider = TracerProvider(resource=resource)
otlp_exporter = OTLPSpanExporter(
    endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
    headers={"Authorization": f"Bearer {os.environ.get('OTEL_TOKEN', '')}"},
)
provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("ci.pipeline.runner")


# ── helpers ────────────────────────────────────────────────────────────────────

@contextmanager
def stage(name: str, **attrs):
    """Run a pipeline stage as a child OTel span.

    Usage:
        with stage("checkout", repo=REPO_URL) as sp:
            run_checkout(...)
    """
    with tracer.start_as_current_span(
        f"stage/{name}",
        kind=SpanKind.INTERNAL,
        attributes={f"ci.stage.{k}": str(v) for k, v in attrs.items()},
    ) as sp:
        t0 = time.monotonic()
        try:
            yield sp
            sp.set_status(StatusCode.OK)
        except Exception as exc:
            sp.set_status(StatusCode.ERROR, str(exc))
            sp.record_exception(exc)
            raise
        finally:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            sp.set_attribute("ci.stage.duration_ms", elapsed_ms)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess and raise on non-zero exit."""
    result = subprocess.run(cmd, check=True, text=True, capture_output=True, **kwargs)
    return result


# ── pipeline stages ────────────────────────────────────────────────────────────

def run_pipeline():
    with tracer.start_as_current_span(
        f"pipeline/{WORKFLOW}",
        kind=SpanKind.PRODUCER,
        attributes={
            "ci.pipeline.name": WORKFLOW,
            "ci.run_id": RUN_ID,
            "ci.run_attempt": str(RUN_ATTEMPT),
            "git.commit.sha": COMMIT_SHA,
            "git.branch": BRANCH,
        },
    ) as root_span:
        try:
            # Stage 1: checkout
            with stage("checkout"):
                run(["git", "fetch", "--depth=1", "origin", BRANCH])
                run(["git", "checkout", COMMIT_SHA])

            # Stage 2: install — record cache hit as a span attribute
            cache_key = os.environ.get("CACHE_HIT", "false")
            with stage("install", cache_hit=cache_key) as sp:
                run(["npm", "ci", "--prefer-offline"])
                sp.add_event("install_complete", {"cache.hit": cache_key == "true"})

            # Stage 3: build
            with stage("build"):
                run(["npm", "run", "build"])

            # Stage 4: test — emit span events for suite summaries
            with stage("test") as sp:
                result = run(["npm", "test", "--", "--reporter=json", "--outputFile=results.json"])
                import json
                try:
                    with open("results.json") as f:
                        data = json.load(f)
                    passed = data.get("numPassedTests", 0)
                    failed = data.get("numFailedTests", 0)
                    sp.set_attribute("test.passed", passed)
                    sp.set_attribute("test.failed", failed)
                    sp.add_event("suite_summary", {
                        "test.passed": passed,
                        "test.failed": failed,
                        "test.total": passed + failed,
                    })
                except Exception:
                    pass  # results file parse failure is non-fatal for tracing

            root_span.set_status(StatusCode.OK)
            root_span.set_attribute("ci.final_status", "success")

        except Exception as exc:
            root_span.set_status(StatusCode.ERROR, str(exc))
            root_span.set_attribute("ci.final_status", "failure")
            raise

        finally:
            provider.force_flush(timeout_millis=5000)


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception:
        sys.exit(1)
```

The key patterns here: every stage is a child span (`start_as_current_span` inherits the parent from context), span events record significant milestones within a stage (cache hit/miss, test suite totals), and `force_flush` at the end ensures spans are exported before the process exits. In Grafana Tempo, the resulting trace renders as a waterfall: the root span covers the full pipeline duration, each stage span is a horizontal bar inside it, and span events appear as markers on the timeline.

### Prometheus metrics from CI runners

The post-job metrics emission script pushes data to a Prometheus Pushgateway after every run. The Pushgateway is scraped by Prometheus on its normal interval. This gives you time-series data for trending, alerting, and SLO calculations.

```bash
#!/usr/bin/env bash
# .github/scripts/emit-metrics.sh
set -euo pipefail

JOB_NAME="$1"
JOB_STATUS="$2"   # "success" or "failure"
RUN_ATTEMPT="${3:-1}"

PUSHGATEWAY="${PUSHGATEWAY_URL:-}"
if [ -z "$PUSHGATEWAY" ]; then
  echo "PUSHGATEWAY_URL not set — skipping metric emission"
  exit 0
fi

BRANCH="${GITHUB_REF_NAME:-unknown}"
COMMIT="${GITHUB_SHA:-unknown}"
RUN_ID="${GITHUB_RUN_ID:-unknown}"
DURATION_SECONDS="${CI_JOB_DURATION:-0}"  # set by caller or computed

SUCCESS_COUNT=$([ "$JOB_STATUS" = "success" ] && echo 1 || echo 0)
FAILURE_COUNT=$([ "$JOB_STATUS" = "success" ] && echo 0 || echo 1)
IS_RETRY=$([ "$RUN_ATTEMPT" -gt 1 ] && echo 1 || echo 0)
# A flaky failure = previous attempt failed, this one succeeded
IS_FLAKY=$([ "$IS_RETRY" = "1" ] && [ "$SUCCESS_COUNT" = "1" ] && echo 1 || echo 0)

METRICS=$(cat <<PROM
# HELP ci_job_duration_seconds Wall-clock duration of a CI job in seconds
# TYPE ci_job_duration_seconds histogram
ci_job_duration_seconds_bucket{job="$JOB_NAME",branch="$BRANCH",le="60"} $([ "$DURATION_SECONDS" -le 60 ] && echo 1 || echo 0)
ci_job_duration_seconds_bucket{job="$JOB_NAME",branch="$BRANCH",le="180"} $([ "$DURATION_SECONDS" -le 180 ] && echo 1 || echo 0)
ci_job_duration_seconds_bucket{job="$JOB_NAME",branch="$BRANCH",le="360"} $([ "$DURATION_SECONDS" -le 360 ] && echo 1 || echo 0)
ci_job_duration_seconds_bucket{job="$JOB_NAME",branch="$BRANCH",le="600"} $([ "$DURATION_SECONDS" -le 600 ] && echo 1 || echo 0)
ci_job_duration_seconds_bucket{job="$JOB_NAME",branch="$BRANCH",le="900"} $([ "$DURATION_SECONDS" -le 900 ] && echo 1 || echo 0)
ci_job_duration_seconds_bucket{job="$JOB_NAME",branch="$BRANCH",le="+Inf"} 1
ci_job_duration_seconds_sum{job="$JOB_NAME",branch="$BRANCH"} $DURATION_SECONDS
ci_job_duration_seconds_count{job="$JOB_NAME",branch="$BRANCH"} 1

# HELP ci_job_success_total Total successful CI job runs
# TYPE ci_job_success_total counter
ci_job_success_total{job="$JOB_NAME",branch="$BRANCH"} $SUCCESS_COUNT

# HELP ci_job_failure_total Total failed CI job runs
# TYPE ci_job_failure_total counter
ci_job_failure_total{job="$JOB_NAME",branch="$BRANCH"} $FAILURE_COUNT

# HELP ci_job_total Total CI job runs (all outcomes)
# TYPE ci_job_total counter
ci_job_total{job="$JOB_NAME",branch="$BRANCH"} 1

# HELP ci_flaky_success_total Runs that succeeded only on retry (flaky indicator)
# TYPE ci_flaky_success_total counter
ci_flaky_success_total{job="$JOB_NAME",branch="$BRANCH"} $IS_FLAKY

# HELP ci_retry_total Total retry attempts made
# TYPE ci_retry_total counter
ci_retry_total{job="$JOB_NAME",branch="$BRANCH"} $IS_RETRY
PROM
)

echo "$METRICS" | curl --silent --fail \
  --data-binary @- \
  "$PUSHGATEWAY/metrics/job/ci_job/instance/${JOB_NAME}_${RUN_ID}"
```

### DORA events to the event store

The DORA event schema is minimal. You need five fields: event type, timestamp, service, environment, and the commit SHA plus a few timestamps from the version-control system.

```json
{
  "event_type": "deployment",
  "timestamp": "2026-06-22T14:32:00Z",
  "service": "checkout-api",
  "environment": "production",
  "commit_sha": "abc123def456",
  "commit_created_at": "2026-06-22T08:45:00Z",
  "pr_opened_at": "2026-06-22T09:15:00Z",
  "pr_merged_at": "2026-06-22T13:55:00Z",
  "triggered_by": "pipeline/deploy-prod",
  "pipeline_run_id": "9876543210",
  "status": "success",
  "deploy_duration_seconds": 147
}
```

Emit one event at the end of every deploy job. The PR timestamps come from the GitHub API (or GitLab, etc.) using the commit SHA to look up the associated PR.

```bash
#!/usr/bin/env bash
# .github/scripts/emit-dora-event.sh
set -euo pipefail

ENVIRONMENT="${1:-production}"
STATUS="${2:-success}"
SERVICE="${GITHUB_REPOSITORY##*/}"
COMMIT_SHA="${GITHUB_SHA}"
RUN_ID="${GITHUB_RUN_ID}"

# Fetch PR metadata from GitHub API
PR_DATA=$(curl -s \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  "https://api.github.com/repos/$GITHUB_REPOSITORY/commits/$COMMIT_SHA/pulls" \
  | jq '.[0] // {}')

PR_OPENED_AT=$(echo "$PR_DATA" | jq -r '.created_at // empty')
PR_MERGED_AT=$(echo "$PR_DATA" | jq -r '.merged_at // empty')

# Fetch first-commit timestamp for lead time
FIRST_COMMIT_AT=$(curl -s \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  "https://api.github.com/repos/$GITHUB_REPOSITORY/commits/$COMMIT_SHA" \
  | jq -r '.commit.author.date // empty')

EVENT=$(cat <<EOF
{
  "event_type": "deployment",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "service": "$SERVICE",
  "environment": "$ENVIRONMENT",
  "commit_sha": "$COMMIT_SHA",
  "commit_created_at": "${FIRST_COMMIT_AT:-}",
  "pr_opened_at": "${PR_OPENED_AT:-}",
  "pr_merged_at": "${PR_MERGED_AT:-}",
  "triggered_by": "pipeline/$GITHUB_WORKFLOW",
  "pipeline_run_id": "$RUN_ID",
  "status": "$STATUS"
}
EOF
)

# Post to your event store (Loki, Postgres, etc.)
curl -s --fail \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $DORA_API_TOKEN" \
  -d "$EVENT" \
  "${DORA_EVENTS_API:-https://dora.internal/api/events}"
```

---

## The Flaky Test Problem

A flaky test is a test that produces different outcomes on successive runs without any change to the code or environment that the test is supposed to be testing. That definition contains the word "supposed" deliberately: a test that is flaky because the code it tests has a real concurrency bug is revealing something true. But a test that is flaky because it tries to connect to an external API that is rate-limiting GitHub Actions runners is revealing nothing about the code — it is just noise.

The key property that makes flaky tests so destructive is that they corrupt the information content of CI results. CI works as a mechanism because it provides a reliable binary signal: this code is safe to ship, or it is not. Flaky tests degrade that signal. They introduce false negatives (code that is safe, flagged as unsafe) at a rate that trains engineers to discount the signal entirely.

The cognitive science here is straightforward: if a signal is unreliable at a rate above about 10–15%, humans stop using it as a decision input. They route around it. The system adapts to its unreliability by being ignored. This is precisely what happened in the opening story.

![Flaky test taxonomy: four root-cause categories (network, race condition, environment, order) with their detection signal, typical fix, and quarantine policy](/imgs/blogs/pipeline-observability-and-the-flaky-pipeline-3.png)

### The four classes of flaky tests

**Network-dependent tests** make real HTTP calls to external services — APIs, databases, Kafka brokers, Redis instances — that may be unavailable, slow, or rate-limited in CI. They fail with connection refused, timeout, or unexpected HTTP 429/503 errors. The stack trace points to a network layer, not to the business logic being tested.

Detection: look for `ECONNREFUSED`, `socket hang up`, `timeout`, `429 Too Many Requests` in failure logs. The error is consistent across all failures of this class, but the test passes when the external service happens to be responsive.

Fix: at the unit-test level, stub the external dependency with a mock or an in-memory fake (Wiremock, MSW, sinon). At the integration-test level, use a dedicated integration environment with retry logic and proper health checks before running tests.

**Race condition tests** expose real concurrency bugs — in the application code or in the test harness itself. They manifest as assertion failures where the actual value differs from expected non-deterministically, or as deadlocks that cause tests to time out. These are genuinely the hardest class because the root cause is a real defect, and the fix (adding proper synchronisation) can be non-trivial.

Detection: the failure is non-deterministic with no consistent error message. Passing and failing runs are interleaved with no pattern. Running the test in isolation with `--runInBand` (Jest) or `-p 1` (pytest-xdist) eliminates the failure (because you've removed the concurrency that triggers the race).

Fix: add the appropriate synchronisation primitive (mutex, `await`, `WaitGroup`, channel). Until the fix is in, quarantine is mandatory — the test is revealing a real bug, and its flakiness is dangerous, not just annoying.

**Environment-dependent tests** pass locally but fail on CI runners because of OS-level differences: file system case sensitivity, timezone configuration, available system libraries, locale settings, or tool version mismatches between developer machines and CI containers. These are often the sneakiest because they look like infrastructure failures when they are actually test fragility.

Detection: fails consistently on CI but passes locally. The error message often mentions a file, path, timezone, or locale. Reproducing the failure requires matching the CI environment (Docker container with the same base image as the CI runner).

Fix: pin all environmental dependencies explicitly. Use `TZ=UTC` in CI. Use case-insensitive path handling. Containerise the CI environment so it is identical to local development. Run the same Docker image locally for development.

**Order-dependent tests** pass when run in isolation but fail when run as part of a suite because a previous test modified shared state (a global variable, a database table, a file on disk, a singleton). They are particularly insidious in test suites that were written quickly without attention to isolation.

Detection: the test passes when run alone (`--testNamePattern="..." --runInBand`) but fails in the full suite. Shuffling the test execution order (Jest's `--randomize`, pytest's `--randomly-seed`) exposes these tests quickly because they fail in different positions with a non-random order but consistently fail in whatever order happens to put the polluting test before them.

Fix: proper test isolation — each test sets up its own state and tears it down. Use `beforeEach`/`afterEach` to reset shared state. Use database transactions that are rolled back after each test. Use in-memory state instead of file-system state where possible.

### Why flaky tests erode trust faster than you expect

The damage is not linear. It compounds with every incident of false alarm.

Each flaky-test failure that gets retried successfully teaches the team a lesson: CI failures are probably noise. After five such experiences, the team's prior on any given red build being meaningful is around 70%. After fifteen, it is around 30%. After twenty-five, engineers are not reading logs at all — they are just clicking retry as a reflex.

This is alarm fatigue applied to CI. It is the same phenomenon that leads to alert fatigue in monitoring systems: when too many alerts are false positives, engineers stop responding to alerts. The solution in both cases is the same — reduce the false positive rate below the threshold where the signal is still trusted (roughly 5–10% false positive rate is the danger zone; below 2% and most engineers still trust the signal).

The insidious part is that trust is asymmetric: it is lost quickly (a few flaky incidents train the new behaviour fast) and rebuilt slowly (engineers need months of consistent reliable CI to re-learn that red means red). This means that even after you fix the flaky tests, you will need to explicitly communicate the improvement to the team and show them the metrics before the cultural behaviour changes.

---

## Flaky Test Detection and Quarantine

The first step is detection. You cannot quarantine what you cannot identify. A flaky test has a specific signature: it fails in a build, and then the same commit passes in a retry without any code change. That signature is detectable from pipeline logs and metadata.

![Before and after flaky-test quarantine: 35% flaky-caused failures drop to under 3% with systematic tracking and fix sprints](/imgs/blogs/pipeline-observability-and-the-flaky-pipeline-2.png)

### Computing the flakiness score

Assign every test a flakiness score based on its recent run history. The score represents the fraction of consecutive-run pairs where the outcome changed (pass→fail or fail→pass) on the same commit. A score of 0.00 is perfectly deterministic. A score of 0.10 means that in 10% of consecutive pairs, the test gave a different result.

```python
# scripts/update_flakiness_scores.py
"""
Run after every CI job. Reads test-results.json, writes run records to Postgres,
and updates the per-test flakiness score.

Usage:
  python scripts/update_flakiness_scores.py \
    --results test-results.json \
    --commit $GITHUB_SHA \
    --run-id $GITHUB_RUN_ID \
    --attempt $GITHUB_RUN_ATTEMPT
"""

import argparse
import json
import os
import psycopg2
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--attempt", type=int, default=1)
    return p.parse_args()


def update_scores(args):
    with open(args.results) as f:
        results = json.load(f)

    conn = psycopg2.connect(os.environ["FLAKINESS_DB_URL"])
    cur = conn.cursor()

    # Create tables if they don't exist (idempotent)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS test_runs (
            id BIGSERIAL PRIMARY KEY,
            test_id TEXT NOT NULL,
            commit_sha TEXT NOT NULL,
            run_id TEXT NOT NULL,
            attempt INT NOT NULL DEFAULT 1,
            status TEXT NOT NULL,  -- 'pass' or 'fail'
            recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_test_runs_test_id ON test_runs(test_id);

        CREATE TABLE IF NOT EXISTS flakiness_scores (
            test_id TEXT PRIMARY KEY,
            score FLOAT NOT NULL DEFAULT 0.0,
            run_count INT NOT NULL DEFAULT 0,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """)

    for test_suite in results.get("testResults", []):
        suite_path = test_suite.get("testFilePath", "unknown")
        for case in test_suite.get("testCases", []):
            test_id = f"{suite_path}::{case['fullName']}"
            status = "pass" if case["status"] == "passed" else "fail"

            cur.execute("""
                INSERT INTO test_runs (test_id, commit_sha, run_id, attempt, status)
                VALUES (%s, %s, %s, %s, %s)
            """, (test_id, args.commit, args.run_id, args.attempt, status))

            # Compute flakiness score over last 200 runs for this test
            # Score = fraction of consecutive pairs with different outcomes
            cur.execute("""
                WITH ordered AS (
                    SELECT status,
                           LAG(status) OVER (ORDER BY recorded_at) AS prev_status
                    FROM test_runs
                    WHERE test_id = %s
                    ORDER BY recorded_at DESC
                    LIMIT 200
                ),
                transitions AS (
                    SELECT COUNT(*) FILTER (
                        WHERE status IS DISTINCT FROM prev_status
                          AND prev_status IS NOT NULL
                    ) AS flips,
                    COUNT(*) FILTER (WHERE prev_status IS NOT NULL) AS pairs
                    FROM ordered
                )
                SELECT
                    CASE WHEN pairs = 0 THEN 0.0
                         ELSE flips * 1.0 / pairs
                    END AS score,
                    pairs AS run_count
                FROM transitions
            """, (test_id,))
            row = cur.fetchone()
            score, run_count = (row[0] or 0.0, row[1] or 0) if row else (0.0, 0)

            cur.execute("""
                INSERT INTO flakiness_scores (test_id, score, run_count, updated_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (test_id) DO UPDATE
                    SET score = EXCLUDED.score,
                        run_count = EXCLUDED.run_count,
                        updated_at = EXCLUDED.updated_at
            """, (test_id, score, run_count))

    conn.commit()
    cur.close()
    conn.close()
    print(f"Updated flakiness scores for {sum(len(t.get('testCases',[])) for t in results.get('testResults',[]))} tests")


if __name__ == "__main__":
    update_scores(parse_args())
```

### The quarantine policy in GitHub Actions

The quarantine policy has three components: a quarantine list (tests above the threshold), a non-blocking test run for quarantined tests, and a PR comment that summarises the quarantine state.

```yaml
# .github/workflows/test-with-quarantine.yml
name: Test Suite (with flaky quarantine)
on: [push, pull_request]

jobs:
  fetch-quarantine-list:
    runs-on: ubuntu-latest
    outputs:
      quarantined_patterns: ${{ steps.fetch.outputs.patterns }}
      quarantined_count: ${{ steps.fetch.outputs.count }}
    steps:
      - id: fetch
        name: Fetch quarantine list
        run: |
          # Query your flakiness API for tests with score > 0.05
          QUARANTINED=$(curl -sf \
            -H "Authorization: Bearer ${{ secrets.FLAKINESS_API_TOKEN }}" \
            "${{ vars.FLAKINESS_API_URL }}/quarantined?threshold=0.05&format=jest-pattern" \
            || echo "")

          COUNT=$(echo "$QUARANTINED" | jq 'length' 2>/dev/null || echo 0)
          PATTERN=$(echo "$QUARANTINED" | jq -r '.[].pattern // empty' | paste -sd'|' -)

          echo "count=$COUNT" >> $GITHUB_OUTPUT
          echo "patterns=$PATTERN" >> $GITHUB_OUTPUT

  required-tests:
    needs: fetch-quarantine-list
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: "npm"
      - run: npm ci --prefer-offline

      - name: Run required tests (quarantined excluded)
        run: |
          EXCLUDE="${{ needs.fetch-quarantine-list.outputs.quarantined_patterns }}"
          if [ -n "$EXCLUDE" ]; then
            npx jest --testPathIgnorePatterns="$EXCLUDE" \
              --reporter=json --outputFile=test-results-required.json
          else
            npx jest --reporter=json --outputFile=test-results-required.json
          fi

      - name: Update flakiness scores
        if: always()
        run: |
          python scripts/update_flakiness_scores.py \
            --results test-results-required.json \
            --commit "${{ github.sha }}" \
            --run-id "${{ github.run_id }}" \
            --attempt "${{ github.run_attempt }}"
        env:
          FLAKINESS_DB_URL: ${{ secrets.FLAKINESS_DB_URL }}

  quarantined-tests:
    needs: fetch-quarantine-list
    if: needs.fetch-quarantine-list.outputs.quarantined_count > 0
    runs-on: ubuntu-latest
    continue-on-error: true  # CRITICAL: failures here never block merge
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: "npm"
      - run: npm ci --prefer-offline

      - name: Run quarantined tests (informational)
        run: |
          PATTERN="${{ needs.fetch-quarantine-list.outputs.quarantined_patterns }}"
          npx jest --testPathPattern="$PATTERN" \
            --reporter=json --outputFile=test-results-quarantined.json || true

      - name: Upload quarantine results
        uses: actions/upload-artifact@v4
        with:
          name: quarantine-results-${{ github.run_id }}
          path: test-results-quarantined.json

  post-quarantine-summary:
    needs: [fetch-quarantine-list, required-tests, quarantined-tests]
    if: always() && github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/github-script@v7
        with:
          script: |
            const count = parseInt('${{ needs.fetch-quarantine-list.outputs.quarantined_count }}');
            const reqStatus = '${{ needs.required-tests.result }}';
            const emoji = reqStatus === 'success' ? '✅' : '❌';

            const body = [
              `## CI Summary ${emoji}`,
              '',
              `| Check | Result |`,
              `|---|---|`,
              `| Required tests | ${reqStatus === 'success' ? '✅ Passed' : '❌ Failed'} |`,
              `| Quarantined tests (non-blocking) | ${count} tests excluded from gate |`,
              '',
              count > 0
                ? `> **${count} tests are currently quarantined** (flakiness score > 5%). They run but cannot block your PR. View the [flakiness dashboard](https://grafana.internal/d/ci-flakiness) to see which tests are in quarantine and track fix progress.`
                : `> No tests are currently quarantined. 🎉`,
              '',
              `_Build: \`${{ github.sha }}\` · Run: [${{ github.run_id }}](${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }})_`
            ].join('\n');

            await github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body
            });
```

### Automatic issue creation for newly quarantined tests

A quarantine entry that lives only in a database is easy to forget. The fix: when a test crosses the quarantine threshold for the first time, automatically open a GitHub issue tagged `flaky-test` and assigned to the author of the last commit that introduced the test file.

```yaml
# job fragment — append to test-with-quarantine.yml under required-tests
      - name: Auto-open issues for newly quarantined tests
        if: always()
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          FLAKINESS_DB_URL: ${{ secrets.FLAKINESS_DB_URL }}
        run: |
          python - <<'PYEOF'
          import os, json, subprocess, psycopg2, urllib.request, urllib.error

          conn = psycopg2.connect(os.environ["FLAKINESS_DB_URL"])
          cur = conn.cursor()

          # Find tests that just crossed threshold for the first time (no issue yet)
          cur.execute("""
              SELECT test_id, score
              FROM flakiness_scores
              WHERE score > 0.05
                AND github_issue_url IS NULL
              LIMIT 20
          """)
          newly_quarantined = cur.fetchall()

          token = os.environ["GITHUB_TOKEN"]
          repo = os.environ.get("GITHUB_REPOSITORY", "")
          api_url = f"https://api.github.com/repos/{repo}/issues"

          for test_id, score in newly_quarantined:
              title = f"[Flaky Test] {test_id.split('::')[-1]}"
              body = (
                  f"## Flaky test auto-detected\n\n"
                  f"**Test:** `{test_id}`\n"
                  f"**Flakiness score:** {score:.3f} (threshold: 0.05)\n\n"
                  f"This test has been moved to the quarantine tier and will no longer "
                  f"block PRs. Please investigate and fix the root cause.\n\n"
                  f"**Flakiness classes to check:**\n"
                  f"- Network dependency (mock the external call)\n"
                  f"- Race condition (run with `--runInBand` to confirm)\n"
                  f"- Environment sensitivity (diff CI vs local env)\n"
                  f"- Test ordering (run with `--randomize` to confirm)\n\n"
                  f"See the [flakiness dashboard](https://grafana.internal/d/ci-flakiness) "
                  f"for full run history."
              )
              payload = json.dumps({
                  "title": title,
                  "body": body,
                  "labels": ["flaky-test", "ci-health"],
              }).encode()

              req = urllib.request.Request(
                  api_url,
                  data=payload,
                  headers={
                      "Authorization": f"Bearer {token}",
                      "Content-Type": "application/json",
                      "Accept": "application/vnd.github+json",
                  },
              )
              try:
                  with urllib.request.urlopen(req) as resp:
                      issue = json.load(resp)
                      issue_url = issue["html_url"]
                      cur.execute(
                          "UPDATE flakiness_scores SET github_issue_url = %s WHERE test_id = %s",
                          (issue_url, test_id),
                      )
                      print(f"Opened issue for {test_id}: {issue_url}")
              except urllib.error.HTTPError as e:
                  print(f"Failed to open issue for {test_id}: {e}")

          conn.commit()
          cur.close()
          conn.close()
          PYEOF
```

### The fix sprint cadence

Quarantine is a holding pattern, not a solution. If you quarantine tests and never fix them, you accumulate a growing list of untrusted tests that reduces CI coverage. The fix is a recurring "flakiness sprint" — a dedicated engineering block each iteration, owned by a named engineer, to fix the top-N flakiest tests and remove them from quarantine.

A simple policy: every two-week sprint, one engineer owns "flakiness debt" as their primary responsibility for 20% of their time (one day per week). Their job is to take the top 10 flakiest tests from the dashboard, investigate the root cause, apply the fix, verify the fix (by checking that the score drops to 0.0 over 50+ runs), and mark the test as "graduated" from quarantine. With this cadence, a team that starts at 35% flaky-caused failures reaches below 3% within two to three quarters.

#### Worked example: Flaky rate from 35% to 3% — the quarantine sprint math

A team runs 4,200 test cases. Their first flakiness measurement (after running the score tracker for two weeks) finds 190 tests with a score above 0.05. Of those, 45 have a score above 0.15 — they fail in more than one in six consecutive-run pairs.

Under the quarantine policy, those 190 tests are moved to the informational-only category immediately. The required test suite has 4,010 tests. The flaky-caused failure rate for the required suite drops from roughly 35% (pre-quarantine baseline, where any of the 190 high-scoring tests could cause a pipeline failure) to approximately 2.5% (residual from tests with a score between 0.01 and 0.05 that sit just below the threshold).

The PromQL query that tracks progress during the sprint campaign:

```yaml
# Grafana panel: flaky-caused failure rate trend (rolling 24h window)
# Shows the weekly progress as the fix sprint closes issues
expr: >
  100 * (
    sum(rate(ci_flaky_success_total[24h]))
    / sum(rate(ci_job_total[24h]))
  )
legend: "Flaky rate %"
unit: "percent"
thresholds:
  - value: 10
    color: "red"
  - value: 5
    color: "yellow"
  - value: 2
    color: "green"
```

The fix sprint plan and timeline:

- **Week 1 (Sprint 1, day 1–3):** Fix the top 15 tests by score (all have score > 0.20). Average investigation time: 2.5 hours per test. 15 tests × 2.5h = 37.5 hours — approximately one engineer for one week. Flaky rate drops from 35% to 22% as the worst offenders are retired from quarantine.
- **Week 2 (Sprint 1, day 4–5 + Sprint 2, day 1–2):** Fix the next 15 tests (scores 0.15–0.20). Flaky rate drops to 15%.
- **Sprints 3–4 (weeks 3–6):** Fix 30 more tests (scores 0.08–0.15). Flaky rate: 15% → 7%.
- **Sprints 5–7 (weeks 7–12):** Fix remaining 130 tests at the 20%-time cadence (one day/sprint × 2 tests/day × 6 sprints = 24 tests). Flaky rate: 7% → ~3%.
- **Ongoing:** 20% sprint allocation per sprint, maintaining the quarantine list at < 10 tests. Flaky rate holds below 3%.

The business impact at the 3% mark: retry rate per PR falls from 3.4 to 1.1 (from 34% retries to 10%). For a team making 40 PRs/day, that is 40 × (3.4 - 1.1) × (average pipeline time) fewer pipeline runs. At 12 minutes average, that is 40 × 2.3 × 12 = 1,104 minutes (18.4 engineer-hours) of pipeline runtime saved per day, plus the cognitive overhead of engineers who no longer reflexively click retry.

---

## The Slow Pipeline: Profiling and Fixing

The flaky pipeline is a reliability problem. The slow pipeline is a performance problem. Both hurt lead time. A 38-minute pipeline that you trust completely is still costing you 32 minutes more than a 6-minute pipeline on every single merge. At scale, that is an enormous overhead.

![Pipeline flame graph showing 38-minute build: queue 4 min, checkout 2 min, dependency install 14 min, build 4 min, test 14 min, Docker push 4 min](/imgs/blogs/pipeline-observability-and-the-flaky-pipeline-5.png)

### Reading the pipeline flame graph

The stage-level timing data from `timed-step.sh` (or from OTel spans) renders in Grafana Tempo as a waterfall trace — effectively a flame graph for your build. Each horizontal bar is a stage. The width of the bar is the duration. The order from top to bottom follows the execution order, with parallel stages shown side by side.

When reading a pipeline flame graph, look for three patterns:

**The tall waterfall (serial bottleneck).** Every stage waits for the one above. The critical path equals the sum of all stage durations. The fix is always the same: identify which stages have no data dependency on each other, move them to parallel jobs.

**The wide single bar (one slow stage dominates).** One stage — typically the test suite or the Docker build — is 60–80% of total pipeline time. Everything else is noise. Fix the slow stage first. Optimising a 2-minute lint step when the test suite takes 14 minutes is wasted effort.

**The cold cache signature.** The dependency install stage is consistently 10+ minutes and shows no improvement on runs that follow a recent identical run. The cache restore step reports a cache miss. This is almost always a misconfigured cache key or a cache path that is incompatible across runner OS versions.

### Getting GitHub Actions timing data without OTel

If you have not yet instrumented with OTel, the GitHub Actions UI has a built-in timing view: in any job log, click the stopwatch icon next to each step name to see its duration. For aggregate analysis, the GitHub API exposes step-level timing:

```bash
#!/usr/bin/env bash
# Fetch step timings for the last N runs of a given workflow
REPO="${GITHUB_REPOSITORY:-org/repo}"
WORKFLOW_ID="${1:-ci.yml}"
N="${2:-20}"

curl -s \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/actions/workflows/$WORKFLOW_ID/runs?per_page=$N" \
  | jq -r '.workflow_runs[].id' \
  | while read RUN_ID; do
      curl -s \
        -H "Authorization: Bearer $GITHUB_TOKEN" \
        "https://api.github.com/repos/$REPO/actions/runs/$RUN_ID/jobs" \
        | jq -r '
          .jobs[] |
          .name as $job |
          .steps[] |
          select(.completed_at != null) |
          [
            $job,
            .name,
            ((.completed_at | fromdateiso8601) - (.started_at | fromdateiso8601))
          ] | @tsv
        '
    done
```

Run this script weekly to build a local dataset of step-level timings. The output is a TSV: job name, step name, duration in seconds. Feed it into any spreadsheet or pandas DataFrame to see which steps are consistently slow across all runs.

### The usual suspects

In practice, 80% of slow pipelines have one or more of these four problems:

**No dependency caching.** Every run reinstalls all packages from scratch. The cache key is missing, wrong, or invalidated too eagerly. Fix: cache the package manager's local cache directory (not `node_modules` directly), keyed on the lock file hash.

```yaml
# GitHub Actions: correct npm caching
- name: Setup Node.js with caching
  uses: actions/setup-node@v4
  with:
    node-version: "20"
    cache: "npm"  # This caches ~/.npm keyed on package-lock.json hash

# Or manually, for more control:
- uses: actions/cache@v4
  with:
    path: |
      ~/.npm
      .next/cache
    key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
    restore-keys: |
      ${{ runner.os }}-node-
```

**No Docker layer caching.** Every Docker build starts from the application source layer, rebuilding everything above it. Fix: use BuildKit with registry-based layer caching so only changed layers rebuild.

```yaml
- name: Build and push Docker image
  uses: docker/build-push-action@v5
  with:
    context: .
    push: true
    tags: |
      ghcr.io/${{ github.repository }}:${{ github.sha }}
      ghcr.io/${{ github.repository }}:latest
    cache-from: type=registry,ref=ghcr.io/${{ github.repository }}:buildcache
    cache-to: type=registry,ref=ghcr.io/${{ github.repository }}:buildcache,mode=max
    build-args: |
      BUILDKIT_INLINE_CACHE=1
```

**Serial test suite.** All 11,000 tests run on a single runner, one after another. Fix: shard the test suite across parallel runners.

```yaml
jobs:
  test:
    strategy:
      matrix:
        shard: [1, 2, 3, 4, 5, 6, 7, 8]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: "npm"
      - run: npm ci --prefer-offline

      - name: Run test shard ${{ matrix.shard }}/8
        run: |
          npx jest \
            --shard=${{ matrix.shard }}/8 \
            --reporter=json \
            --outputFile=test-results-${{ matrix.shard }}.json

      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: test-results-${{ matrix.shard }}
          path: test-results-${{ matrix.shard }}.json
```

**Serial stages that could be parallel.** Lint, type-check, security scan, and unit tests almost always have no data dependencies on each other. They can run simultaneously.

```yaml
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: "npm"
      - run: npm ci --prefer-offline
      - run: npm run lint

  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: "npm"
      - run: npm ci --prefer-offline
      - run: npm run typecheck

  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: |
          npm audit --audit-level=high
          npx snyk test --severity-threshold=high || true

  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: "npm"
      - run: npm ci --prefer-offline
      - run: npm run build
      - uses: actions/upload-artifact@v4
        with:
          name: build-artifacts
          path: dist/

  test:
    needs: build  # tests need build artifacts
    strategy:
      matrix:
        shard: [1, 2, 3, 4, 5, 6, 7, 8]
    runs-on: ubuntu-latest
    steps: [ ... ]

  deploy:
    needs: [lint, typecheck, security-scan, test]  # waits for all
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    runs-on: ubuntu-latest
    steps: [ ... ]
```

This structure runs lint, typecheck, security-scan, and build simultaneously. Tests run on 8 parallel shards after the build. Deploy waits for everything. The critical path is now: build (2 min) → test-shard-max (3.5 min) = 5.5 min, plus overhead.

---

## Pipeline SLOs: Setting, Alerting, and Owning

With metrics in Prometheus, the next step is formalising the SLOs and wiring up alerts that page someone when the pipeline is sick.

![Pipeline SLO stack: raw SLI measurement feeds an SLO target, which defines an error budget, which triggers a burn-rate alert, which pages the on-call if the budget runs critically low](/imgs/blogs/pipeline-observability-and-the-flaky-pipeline-4.png)

### Computing the error budget from pipeline data

Put plainly: the error budget is the monthly allowance of pipeline failures before the platform team is obligated to stop shipping features and fix the pipeline.

Given a reliability SLO of 95% and a team that runs 200 pipelines per day:
- Monthly pipeline runs: 200 × 30 = 6,000
- Error budget (5% of 6,000): 300 failed runs per month
- Daily error budget: 10 failed runs per day

Track budget consumption with a PromQL recording rule:

```yaml
# prometheus/rules/ci-slo-recording.yml
groups:
  - name: ci_slo_recording
    interval: 60s
    rules:
      # Monthly error budget remaining (expressed as fraction 0–1)
      - record: ci:error_budget_remaining:monthly
        expr: |
          1 - (
            (
              sum(increase(ci_job_failure_total[30d]))
              /
              (sum(increase(ci_job_total[30d])) * 0.05)
            )
          )

      # 7-day rolling success rate
      - record: ci:success_rate:7d
        expr: |
          sum(rate(ci_job_success_total[7d]))
          / sum(rate(ci_job_total[7d]))

      # P95 build duration (1h rolling)
      - record: ci:p95_duration_seconds:1h
        expr: |
          histogram_quantile(0.95,
            sum(rate(ci_job_duration_seconds_bucket[1h])) by (le)
          )
```

A Grafana stat panel that shows the error budget as a percentage remaining — green above 50%, yellow 20–50%, red below 20% — gives the platform team a single glance to know how close they are to exhausting the month's allowance.

### Defining the three core SLOs

The three SLOs that cover the most important dimensions of pipeline health:

| SLO | SLI | Target | Error budget | Who owns it |
|---|---|---|---|---|
| Build speed | P95 `ci_job_duration_seconds` for main pipeline | < 600 s (10 min) | 5% of builds may exceed | Platform team |
| Reliability | Success rate (non-flaky failures) | > 95% per rolling 24h | 5% failure rate allowed | Platform team |
| Flakiness | Flaky-caused failure rate | < 2% of all failures | < 2% monthly | Platform + test authors |

The error budget model: if your reliability SLO is 95%, then 5% of pipeline runs per month may fail. A month has roughly 43,200 minutes. If your team runs 200 pipelines per day, that is 6,000 pipeline runs per month. Your error budget is 300 failed runs per month. When you consume 300 failed runs, the error budget is exhausted, and the platform team should freeze feature work and focus on fixing the pipeline.

### Prometheus alerting rules

```yaml
# prometheus/rules/ci-slo.yml
groups:
  - name: ci_pipeline_slos
    interval: 60s
    rules:
      # ============================================================
      # Build Speed SLO (P95 < 600s)
      # ============================================================

      # Warning: P95 exceeds SLO for 15 minutes
      - alert: CIPipelineSlowP95Warning
        expr: |
          histogram_quantile(0.95,
            sum(rate(ci_job_duration_seconds_bucket{
              job=~"build-test|test-suite"
            }[1h])) by (le)
          ) > 600
        for: 15m
        labels:
          severity: warning
          team: platform
          slo: build-speed
        annotations:
          summary: "CI P95 build time {{ $value | humanizeDuration }} exceeds 600s SLO"
          description: >
            The P95 build duration for main CI jobs has been above 10 minutes for 15 minutes.
            Check for missing caches, runner saturation, or a new slow test suite.
          runbook: "https://runbooks.internal/ci/slow-pipeline"
          dashboard: "https://grafana.internal/d/ci-health"

      # Critical: P95 exceeds 20 minutes
      - alert: CIPipelineSlowP95Critical
        expr: |
          histogram_quantile(0.95,
            sum(rate(ci_job_duration_seconds_bucket{
              job=~"build-test|test-suite"
            }[1h])) by (le)
          ) > 1200
        for: 5m
        labels:
          severity: critical
          team: platform
          slo: build-speed

      # ============================================================
      # Reliability SLO (success rate > 95%)
      # ============================================================

      # Warning: success rate below 95% for 30 minutes
      - alert: CIPipelineReliabilityWarning
        expr: |
          sum(rate(ci_job_success_total[6h]))
          / sum(rate(ci_job_total[6h])) < 0.95
        for: 30m
        labels:
          severity: warning
          team: platform
          slo: reliability
        annotations:
          summary: "CI success rate {{ $value | humanizePercentage }} below 95% SLO"

      # ============================================================
      # Flakiness SLO (flaky rate < 2%)
      # ============================================================

      - alert: CIFlakyTestRateHigh
        expr: |
          sum(rate(ci_flaky_success_total[24h]))
          / sum(rate(ci_job_total[24h])) > 0.02
        for: 1h
        labels:
          severity: warning
          team: platform
          slo: flakiness
        annotations:
          summary: "CI flaky success rate {{ $value | humanizePercentage }} above 2% SLO"
          description: >
            More than 2% of CI runs are succeeding only on retry (flaky).
            Run the quarantine update script to move new flaky tests to the informational tier.
          dashboard: "https://grafana.internal/d/ci-flakiness"

      # ============================================================
      # Error budget burn rate (multiwindow)
      # ============================================================

      # Fast burn: 1h window at 14x baseline burn rate
      - alert: CIPipelineErrorBudgetFastBurn
        expr: |
          (
            1 - (
              sum(rate(ci_job_success_total[1h]))
              / sum(rate(ci_job_total[1h]))
            )
          ) > (14 * 0.05)
        for: 2m
        labels:
          severity: critical
          team: platform
          slo: reliability
        annotations:
          summary: "CI reliability error budget burning at 14x rate (1h window)"
          description: >
            The pipeline is failing at 14x the allowed baseline rate.
            If this continues for 1 hour, the monthly error budget will be exhausted.

      # Slow burn: 6h window at 6x baseline burn rate
      - alert: CIPipelineErrorBudgetSlowBurn
        expr: |
          (
            1 - (
              sum(rate(ci_job_success_total[6h]))
              / sum(rate(ci_job_total[6h]))
            )
          ) > (6 * 0.05)
        for: 15m
        labels:
          severity: warning
          team: platform
          slo: reliability

      # Queue saturation
      - alert: CIRunnerPoolSaturated
        expr: |
          avg(ci_runner_busy_ratio) > 0.85
        for: 10m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "CI runner pool at {{ $value | humanizePercentage }} saturation"
          description: >
            More than 85% of CI runners are occupied.
            Queue wait times will grow rapidly. Consider adding runner capacity.
```

---

## DORA Metrics from Pipeline Data

The pipeline is the richest source of DORA metric data in your organisation. Every commit, PR open event, deploy event, and rollback is already happening in the pipeline — you just need to capture those events consistently and query them.

![DORA metrics derived from pipeline events: commit SHA and deploy timestamps yield lead time, deploy counts yield frequency, and incident events yield change failure rate and MTTR](/imgs/blogs/pipeline-observability-and-the-flaky-pipeline-7.png)

### Lead time for changes

Lead time measures how long it takes for a committed line of code to reach production. From the pipeline perspective: `deploy_timestamp - first_commit_in_pr_timestamp`. This includes code review time, merge queue time, and pipeline duration.

The most important insight: for most teams, pipeline time dominates lead time more than they expect. A 38-minute pipeline with 1.8 CI runs per PR (accounting for retries) contributes 38 × 1.8 = 68 minutes to lead time per PR. For a PR that is reviewed and approved in 2 hours, the pipeline is responsible for more than a third of the total lead time.

### Deploy frequency

Deploy frequency is a count of successful production deployments per unit time. DORA defines four performance bands:

| Band | Deploy frequency |
|---|---|
| Elite | Multiple times per day |
| High | Between once per day and once per week |
| Medium | Between once per week and once per month |
| Low | Between once per month and once every 6 months |

The pipeline directly drives this metric: a faster, more reliable pipeline enables more frequent deploys because engineers can merge with confidence and the deploy cycle is short.

### Change failure rate and MTTR

These two metrics require correlating deploy events with incident events. A pragmatic approach for teams that do not yet have a formal incident management system: use the GitHub issue labelled "incident" or "outage" as the incident event source. Tag it with the deploy SHA that caused it. The change failure rate is `incidents_tagged_with_deploy / total_deploys`.

MTTR is `next_successful_production_deploy_after_incident.timestamp - incident.opened_at`. This is the time between "we have a problem" and "we have shipped the fix."

### PromQL queries for the DORA dashboard

The four DORA metrics can be surfaced entirely from pipeline event data using PromQL. The key is publishing named counters for each event class from the emit scripts above:

```yaml
# ── Deploy Frequency ─────────────────────────────────────────────────────────
# Grafana panel: deploy count per day (last 30 days)
expr: >
  increase(ci_deploy_total{environment="production",status="success"}[1d])
legend: "Production deploys"
unit: "short"

# DORA band classification (Elite = > 1/day, High = > 1/week, ...)
expr: >
  sum_over_time(
    increase(ci_deploy_total{environment="production",status="success"}[1d])[30d:1d]
  ) / 30
legend: "Avg deploys/day (30d)"

# ── Lead Time (P50 / P95) ────────────────────────────────────────────────────
# ci_lead_time_seconds is a histogram published by emit-dora-event.sh
# (deploy_timestamp - commit_created_at, in seconds)
expr: >
  histogram_quantile(0.50,
    sum(rate(ci_lead_time_seconds_bucket{environment="production"}[30d])) by (le)
  ) / 3600
legend: "Lead time P50 (hours)"

expr: >
  histogram_quantile(0.95,
    sum(rate(ci_lead_time_seconds_bucket{environment="production"}[30d])) by (le)
  ) / 3600
legend: "Lead time P95 (hours)"

# ── Change Failure Rate ──────────────────────────────────────────────────────
# ci_deploy_incident_total counts deploys that were followed by an incident
expr: >
  100 * (
    sum(increase(ci_deploy_incident_total{environment="production"}[30d]))
    /
    sum(increase(ci_deploy_total{environment="production",status="success"}[30d]))
  )
legend: "Change failure rate %"
thresholds:
  - value: 15
    color: "red"
  - value: 5
    color: "yellow"
  - value: 0
    color: "green"

# ── MTTR ─────────────────────────────────────────────────────────────────────
# ci_incident_recovery_seconds histogram: time from incident open to fix deploy
expr: >
  histogram_quantile(0.50,
    sum(rate(ci_incident_recovery_seconds_bucket[30d])) by (le)
  ) / 60
legend: "MTTR P50 (minutes)"
```

For teams that store DORA events in Postgres (rather than Prometheus), the SQL approach gives more flexible querying for custom time windows:

```sql
-- Deploy frequency: daily deploys to production (last 30 days)
SELECT
  DATE_TRUNC('day', timestamp) AS day,
  COUNT(*) AS deploys,
  COUNT(*) FILTER (WHERE status = 'success') AS successful_deploys
FROM deployment_events
WHERE environment = 'production'
  AND timestamp > NOW() - INTERVAL '30 days'
GROUP BY 1
ORDER BY 1;

-- Lead time distribution (hours) from first commit to deploy
SELECT
  PERCENTILE_CONT(0.25) WITHIN GROUP (
    ORDER BY EXTRACT(EPOCH FROM (timestamp - commit_created_at)) / 3600
  ) AS p25_hours,
  PERCENTILE_CONT(0.50) WITHIN GROUP (
    ORDER BY EXTRACT(EPOCH FROM (timestamp - commit_created_at)) / 3600
  ) AS p50_hours,
  PERCENTILE_CONT(0.95) WITHIN GROUP (
    ORDER BY EXTRACT(EPOCH FROM (timestamp - commit_created_at)) / 3600
  ) AS p95_hours,
  AVG(EXTRACT(EPOCH FROM (timestamp - commit_created_at)) / 3600) AS avg_hours
FROM deployment_events
WHERE environment = 'production'
  AND commit_created_at IS NOT NULL
  AND timestamp > NOW() - INTERVAL '30 days';

-- Change failure rate
SELECT
  ROUND(
    COUNT(DISTINCT i.deploy_sha) * 100.0 / NULLIF(COUNT(DISTINCT d.commit_sha), 0),
    2
  ) AS cfr_percent
FROM deployment_events d
LEFT JOIN incident_events i
  ON d.commit_sha = i.deploy_sha
  AND i.opened_at BETWEEN d.timestamp AND d.timestamp + INTERVAL '2 hours'
WHERE d.environment = 'production'
  AND d.timestamp > NOW() - INTERVAL '30 days';

-- MTTR distribution (minutes)
SELECT
  PERCENTILE_CONT(0.50) WITHIN GROUP (
    ORDER BY EXTRACT(EPOCH FROM (fix_deploy.timestamp - i.opened_at)) / 60
  ) AS p50_mttr_minutes,
  PERCENTILE_CONT(0.95) WITHIN GROUP (
    ORDER BY EXTRACT(EPOCH FROM (fix_deploy.timestamp - i.opened_at)) / 60
  ) AS p95_mttr_minutes
FROM incident_events i
CROSS JOIN LATERAL (
  SELECT timestamp
  FROM deployment_events d
  WHERE d.environment = 'production'
    AND d.timestamp > i.opened_at
    AND d.status = 'success'
  ORDER BY d.timestamp ASC
  LIMIT 1
) fix_deploy
WHERE i.opened_at > NOW() - INTERVAL '30 days';
```

---

## War Story: 38 Minutes to 6 Minutes

A B2B SaaS team of 22 engineers had a monorepo CI pipeline that had grown organically over three years. Nobody remembered when it became 38 minutes. There had been no single decision to make it slow — just a series of individually reasonable additions: first a lint stage, then a type-check stage, then an integration test suite, then a Docker build, then a security scan. Each stage was added in isolation, with no consideration for the overall pipeline shape.

The pipeline ran sequentially: checkout, then install, then lint, then type-check, then build, then unit tests, then integration tests, then Docker build, then push. Everything waited for everything else. The Docker layer cache was configured incorrectly (the `--cache-from` pointed to an image tag that no longer existed), so every Docker build was a cold start.

The flaky test problem had been accumulating in parallel. Of the 11,000 test cases, 340 had a flakiness score above 0.05. The retry rate was 3.4 per PR. Engineers had stopped reading failure messages. The causal link between the slow pipeline and the flaky pipeline was invisible to them: the flaky tests forced more retries, which meant more 38-minute runs, which meant 38 × 3.4 = 129 minutes of total CI time per PR on average, compared to a healthy baseline of roughly 10 minutes.

The platform engineer assigned to the pipeline improvement project spent two days profiling before writing a single line of YAML.

![Before (38-minute serial pipeline) vs after (6-minute parallel + cached pipeline) showing the per-stage breakdown and the restructured dependency graph](/imgs/blogs/pipeline-observability-and-the-flaky-pipeline-6.png)

**Finding 1: Dependency install — 14 minutes every run.** The cache was configured to key on `package-lock.json`, but the cache path was `node_modules` directly. When the runner OS version changed (GitHub bumped ubuntu-22.04 to ubuntu-24.04 mid-year), the `node_modules` cache was incompatible and every run was a cold install. The fix was to cache `~/.npm` instead — the package manager's download cache, which is OS-version-agnostic. On warm cache (90% of runs), install dropped from 14 minutes to 45 seconds. Saving: ~13 minutes on 90% of runs.

**Finding 2: Test suite — 16 minutes serial.** The test suite was CPU-bound (pure JavaScript, no I/O waits). Eight parallel shards, each with 1,375 tests, ran in 1.8–2.1 minutes each. Effective test time: ~2.1 minutes for the slowest shard, plus ~1.4 minutes of overhead (checkout + install on each runner). Total: ~3.5 minutes. Saving: ~12.5 minutes.

**Finding 3: Docker build — 4 minutes cold every run.** The `--cache-from` pointed to a tag that had been garbage-collected six months earlier. Every build was entirely cold. With registry-based layer caching (BuildKit `--cache-to type=registry,mode=max`), only the application layer (which changes every commit) rebuilds. The base image + npm deps layer (which rarely changes) is cached. Docker build dropped from 4 minutes to 50 seconds. Saving: ~3 minutes.

**Finding 4: Serial stages.** Lint, type-check, and security scan were running sequentially after the build, adding ~5 minutes of sequential overhead. Made them parallel with build. Net wall-clock saving: ~3.5 minutes (the critical path no longer includes them).

Total pipeline time: 38 → 6 minutes. The engineers on the team noticed the change the day it deployed. Two commented in Slack that they thought CI was broken because it finished so fast. Within a week, the retry rate dropped from 3.4 to 1.8 (the flaky tests were still there; the quarantine sprint ran in parallel and was complete two weeks later, bringing the retry rate to 1.1).

#### Worked example: Pipeline time cut from 38 min to 6 min — the full breakdown

The per-stage data before and after, showing exactly where 32 minutes came from and where they went:

| Stage | Before (serial) | After (parallel + cached) | Saving | Root cause fixed |
|---|---|---|---|---|
| Queue wait | 4 min | 2 min | 2 min | Added 4 more runners to pool |
| Checkout | 2 min | 1 min | 1 min | Shallow clone depth=1 |
| Dependency install | 14 min | 0.75 min | 13.25 min | Fixed cache path to ~/.npm |
| Lint | 3 min | 0 min (parallel) | 3 min | Moved to parallel job |
| Type-check | 2 min | 0 min (parallel) | 2 min | Moved to parallel job |
| Security scan | 1 min | 0 min (parallel) | 1 min | Moved to parallel job |
| Build | 4 min | 2 min | 2 min | Incremental build, cached outputs |
| Unit tests (serial) | 14 min | 2.1 min | 11.9 min | 8-way sharding |
| Docker build | 4 min | 0.8 min | 3.2 min | Fixed BuildKit registry cache |
| Docker push | 3 min | 1 min | 2 min | Reduced image size 820 MB → 180 MB |
| Integration test | 2 min | 1 min | 1 min | Parallelised 2-way |
| **Total (critical path)** | **38 min** | **6 min** | **32 min** | — |

The critical path after restructuring: queue (2 min) → checkout + install (1.75 min) → build (2 min) → [tests run in parallel with other checks] → slowest test shard (2.1 min) → Docker build + push (1.8 min) = 9.65 min, minus parallel savings = ~6 min wall-clock.

The \$ROI calculation: at an average loaded engineering cost of \$150/hour, 32 minutes × 40 PRs/day × 1.8 CI runs/PR = 38.4 engineering-hours saved per day. That is \$5,760/day in recovered engineering capacity — \$1.5 million per year — from one sprint of pipeline work.

#### Worked example: DORA improvement from pipeline work

Before the pipeline work, the team's DORA metrics were:
- Deploy frequency: 1.8 deploys/day (across 3 services)
- Lead time P50: 5.1 hours
- Lead time P95: 14.2 hours
- Change failure rate: 11%
- MTTR: 52 minutes

After pipeline work (6 min pipeline) + flakiness quarantine (1.1 retries/PR):

The average CI time per PR went from 38 × 3.4 = 129 min to 6 × 1.1 = 6.6 min — a reduction of 122 minutes of pipeline time per PR. For a PR that previously spent 5.1 hours in the pipeline (queue + CI + re-runs), the majority of that time was CI. The new P50 lead time:
- Old: 5.1h (PR review ~3h + pipeline ~2.1h)
- New: 3.3h (PR review ~3h + pipeline ~0.3h)

The 14.2-hour P95 was dominated by multi-retry PRs that consumed 4–5 CI slots of 38 minutes each. The new P95:
- Old: 14.2h (tail PRs with 4–5 retries: 38 × 5 = 190 min + review)
- New: 5.8h (tail PRs with 2 retries: 6 × 2 = 12 min + review)

Deploy frequency rose from 1.8 to 3.2 deploys/day (faster CI unblocked more merges). Change failure rate dropped from 11% to 7% (engineers trusted CI signal, stopped merging on noise). MTTR dropped from 52 to 28 minutes (fix deploys run faster).

The team moved from the "Medium" DORA band to the border of "High" on three metrics — from pipeline work alone, without touching application architecture.

---

## Pipeline Failure Triage

When a pipeline fails, the wrong default action is "retry and see". The right default action is to spend 30 seconds in a triage tree and know exactly what kind of failure this is and what action to take.

![Pipeline failure triage tree: root failure branches to flaky check, infra check, or code failure, each with a specific resolution path](/imgs/blogs/pipeline-observability-and-the-flaky-pipeline-8.png)

The triage tree has three branches:

**Branch 1 — Is it a quarantined flaky test?** The CI PR comment tells you how many quarantined tests are in the suite. If the failure message references a test in the quarantine list, no engineer action is needed. The quarantine policy handles it automatically: the required test gate passes (quarantined tests are excluded), and the flakiness score is updated. Merge the PR.

**Branch 2 — Is it an infrastructure failure?** Look for these signals: `SIGKILL` or OOM error in the logs (runner ran out of memory), Docker daemon errors (`failed to connect to docker daemon`), network errors to artifact registries (`failed to pull image`), or GitHub Actions runner errors (`Error: Runner System.IO Exception`). These are not code problems. Restart the runner or re-queue the job. If the infra failure is recurring (same runner, same error, multiple jobs), escalate to the platform team.

**Branch 3 — Is it a code failure?** If the failure is deterministic (both the original run and the retry fail), the error message points to a specific assertion or compilation error, and the error appears in code changed in the current PR, it is a genuine regression. Fix it forward (commit the fix to the branch) or revert the PR. Do not merge. Do not retry more than once — a deterministic failure does not become non-deterministic by retrying.

The triage tree should be the first link in every CI failure notification. Make it a runbook URL. Put it in the Slack bot that notifies the team of CI failures. Put it in the PR failure comment. Thirty seconds of triage saves ten minutes of "let me look at the logs" for 95% of failures.

---

## When NOT to Over-Invest in Pipeline Observability

Pipeline observability at the level described in this post — OTel traces, Prometheus metrics, DORA event store, Grafana dashboards, SLO alerts, flakiness score tracker — is right for teams where the pipeline is a significant source of friction. It is the wrong investment for teams where it is not.

A two-person startup with five GitHub Actions jobs and one service does not need any of this. They need: a fast, reliable pipeline, and the discipline to fix tests immediately when they flake. The overhead of the observability stack would cost more in engineering time than the problems it would solve.

Here is a practical sizing guide for when to add each layer:

| Team size / pipeline complexity | What you actually need |
|---|---|
| 1–5 engineers, < 10 CI jobs | Native CI UI timing, fix flaky tests as they appear, keep total time < 10 min |
| 5–20 engineers, 1–3 services | Add explicit stage timing (timed-step.sh), track retry rate informally, run fix sprints when retry rate > 1.3 |
| 20–50 engineers, 3–10 services | Add Prometheus + Grafana, flakiness score tracker, quarantine policy, explicit pipeline SLOs |
| 50+ engineers, 10+ services | Full OTel traces, DORA event store, pipeline SLO alerts with burn-rate paging, on-call rotation for pipeline |

The signal that you need the full stack is when you cannot answer "what fraction of our CI failures are flaky?" without manually reading logs, or when engineers are regularly retrying more than 1.3 times per PR. Below those thresholds, lighter tooling is sufficient and cheaper.

The anti-pattern to avoid: a team of eight engineers spending three weeks building a complete Prometheus/Grafana/OTel pipeline observability stack because they read a post about it. The ROI must come from actual pain. If your pipeline takes 5 minutes and flaky failures are below 5%, spend those three weeks shipping features instead.

---

## Key Takeaways

**The pipeline is a production system.** It has users, SLOs, failure modes, and business impact. Treat it with the same engineering discipline as your API.

**Measure before you fix.** Adding stage-level timing and a flakiness counter in week one is the highest-leverage action. You cannot prioritise what you cannot see.

**Flaky tests are a trust problem.** The cost is not the failed run — the cost is the cultural collapse of CI credibility. Every retried flaky failure trains engineers that red is probably noise. Restore trust first, by quarantining immediately, then fix.

**Quarantine is a gate, not a solution.** Move flaky tests to the informational tier on day one. Fix them in dedicated fix sprints at a sustainable cadence. An unmanaged quarantine list grows indefinitely.

**Parallelism and caching are almost always the bottleneck.** The vast majority of slow pipelines are slow because the test suite runs serially and/or dependencies reinstall from scratch on every run. Check these two things first — they require no code changes and typically yield 60–80% of the available speedup.

**SLOs create ownership.** A P95 build-time SLO with a burn-rate alert and a named on-call rotation means that when the pipeline degrades, someone acts. Without an owner, a dashboard is just a pretty number.

**DORA metrics are downstream of pipeline metrics.** A rising retry rate and a creeping build time today predict a worse lead time and deploy frequency next month. The pipeline metrics are your leading indicators.

---

## Further Reading

- [From Commit to Production: The CI/CD Mental Model](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) — the conceptual foundation this post builds on; read this first if you are new to the delivery pipeline as a system
- [DORA Metrics: Measuring Delivery Performance](/blog/software-development/ci-cd/dora-metrics-measuring-delivery-performance) — deeper coverage of how to compute, benchmark, and act on all four DORA metrics; the deploy-frequency and lead-time SQL queries in this post are simplified versions of what that post covers in full
- [The Error Budget: The Currency of Reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability) — the SRE error-budget model applied to pipeline SLOs; the burn-rate alerting formulas in this post come directly from that framework
- The software-development/debugging series covers flaky test debugging in depth — how to reproduce race conditions, isolate environment dependencies, and fix order-dependent tests at the teardown level; cross-reference when the quarantine triage points to a root cause that requires a systematic fix rather than a retry policy
- DORA State of DevOps Report (2024 edition) — the empirical research showing that deploy frequency, lead time, change failure rate, and MTTR are the best predictors of organisational performance among the hundreds of candidate metrics studied
- OpenTelemetry semantic conventions for CI/CD pipelines — the emerging standard for `ci.pipeline.name`, `ci.stage.name`, `vcs.ref.head.sha` attributes in pipeline spans; following these conventions makes your traces queryable across tools
- Google SRE Workbook, Chapter 5: Alerting on SLOs — the multiwindow burn-rate alerting model used in the SLO alert rules above; explains why a single-window alert is insufficient and how to calibrate the 14x / 6x multipliers for fast and slow burn
