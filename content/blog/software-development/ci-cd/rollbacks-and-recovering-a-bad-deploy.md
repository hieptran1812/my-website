---
title: "Rollbacks and Recovering a Bad Deploy"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Master the complete recovery playbook — from automated SLO-triggered rollback and Argo Rollouts canary analysis to the un-rollback-able deploy you must roll forward through — so every bad deploy costs minutes, not hours."
tags:
  [
    "ci-cd",
    "devops",
    "rollback",
    "incident-recovery",
    "argo-rollouts",
    "progressive-delivery",
    "slo",
    "mttr",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/rollbacks-and-recovering-a-bad-deploy-1.png"
---

It is 2:07 am on a Friday. You are on-call. Your phone is vibrating with a PagerDuty alert: error rate on the payments service has climbed from 0.1% to 8.4% in ninety seconds. You have a deployment that went out at 1:55 am — twelve minutes ago. The release was routine: a minor dependency bump and a config change, nothing that looked risky in the review. Your team's SLO agreement gives you a thirty-minute error budget window before customers are materially harmed. The clock is already running.

Your first question is not "what broke." Your first question is: **roll back or roll forward?**

![Recovery decision tree after a bad deploy](/imgs/blogs/rollbacks-and-recovering-a-bad-deploy-1.png)

Get this decision wrong and you add twenty minutes to your MTTR. Get the rollback wrong — try to revert a migration that has already run — and you make the outage worse. Have no automation in place and you spend the first forty-five minutes just figuring out what happened while users are sending angry tweets.

This post is the complete engineering guide to recovery. It covers the decision framework, the mechanisms (Argo Rollouts, Helm, kubectl, feature flags), the un-rollback-able situations that force a roll-forward, and the automated systems that make the entire process happen before a human even reads the first alert. By the end you will know how to cut a 90-minute MTTR down to under five minutes — and why that difference is not luck, it is architecture.

---

## Roll Back vs Roll Forward: The Decision Framework

The phrase "always roll forward" is a myth perpetuated by engineers who have never had to roll forward through a migration that corrupted ten million rows of user data. The phrase "always roll back" is equally naïve — sometimes the fastest path to recovery is a two-line hot-fix, not reverting to a binary that requires three minutes of container pull time plus a re-warm-up window.

Both slogans collapse into a false binary. The real world is more nuanced. Some outages are faster fixed by a roll-forward; others require a roll-back; a third class — the most underappreciated — requires neither, because a feature flag kill-switch can restore the previous behavior in under thirty seconds without touching the binary at all. A fourth class is genuinely un-rollback-able, and attempting a rollback makes things worse.

The right answer depends on four axes that you must evaluate simultaneously, ideally in under two minutes:

**Axis 1: Data mutability.** Did the deploy trigger a schema migration? Did it write new data formats to a message queue that older consumers cannot read? Did it rotate secrets such that old binaries will fail authentication? If any of these are true, a naive rollback of the application binary may not restore the system — or may actively break it further. The schema case is the most dangerous: rolling back the binary to a version that references a column that no longer exists will produce 100% errors where you previously had 8%. You have traded a partial outage for a total one.

**Axis 2: Time to fix.** If the bug is a one-line configuration change — a timeout that was accidentally set to 3s instead of 30s — rolling forward with a corrected config may be faster than the time required to revert, push, and watch Kubernetes roll out the previous image. If the bug is a five-hundred-line logic regression that will take two hours to diagnose and an hour to write a test for, roll back immediately and diagnose in calm daylight when your brain is working at full capacity.

**Axis 3: Blast radius.** Is this a stateless service that ten pods serve independently, with no shared mutable state? Roll back each pod in under two minutes. Is this a stateful workflow engine that has already enqueued work in a new serialisation format? Rolling back the binary without also draining the queue may leave you with unprocessable messages piling up behind a consumer that cannot deserialise them. Is this a service that downstream systems have already integrated with using the new API contract? Rolling back the API contract mid-flight breaks those consumers.

**Axis 4: Rollback availability.** Is the previous container image still in the registry? Is the previous Helm revision still in the release history? Have you kept the previous secrets version? If the answer to any of these is no, rolling back may not be possible in the first place. Teams that prune container images aggressively — deleting images older than three days to save registry storage costs — sometimes discover during an incident that the image they need is gone.

A clean decision framework for a 2am engineer looks like this:

**If the blast radius is contained, the data is not mutated, and the previous revision is available — roll back immediately without discussion.** Communicate the rollback on the incident channel, execute it, and verify. Do not spend ten minutes diagnosing the root cause before rolling back. The time to diagnose is after mitigation.

**If data is mutated but the mutation is backward-compatible** — for example, a migration that added a column but did not drop any — rolling back the binary is still safe. The new column will sit unused, and the old binary will ignore it. Verify that the previous binary will not fail on the schema change before executing.

**If data is mutated and the mutation is backward-incompatible** — a column rename, a required new field, a new serialisation format — rolling back the binary will break things. You have two options: ship a hotfix forward that bridges the old and new data formats, or if a feature flag controls the affected code path, flip the flag off and diagnose.

**If the change is a feature flag candidate** — the offending code path is already behind a flag — flip the flag off first. This is always the fastest path and requires the least infrastructure change. The binary rollback is a fallback if the flag cannot be flipped for some reason.

The third path — the feature flag kill-switch — is often the fastest of all and the one most teams under-invest in. If the offending code path is already behind a flag, a flag flip takes under thirty seconds, touches no infrastructure, and requires no CI pipeline. This is why progressive delivery practitioners argue that the flag flip is the real "rollback" and the binary revert is a fallback.

### Roll-Forward vs Roll-Back Decision Matrix

The matrix below formalises the decision. Use it during an incident to avoid debating in the #incident channel when the clock is ticking.

| Condition | Roll back | Roll forward | Feature flag |
|---|---|---|---|
| Stateless service, no schema change | First choice | Fallback if rev gone | If code is behind flag |
| Schema migration ran, backward-compatible | Safe | Also safe | Not applicable |
| Schema migration ran, backward-incompatible | **Dangerous — do not** | Required | If flag covers affected path |
| Events already published to Kafka in new format | **Dangerous — do not** | Required | Not applicable |
| Credentials rotated, old credential invalidated | **Dangerous — do not** | Required (or re-enable old credential) | Not applicable |
| Payment processed, email sent, user notified | Binary rollback safe, but user action is permanent | Required for user-facing recovery | Flag can prevent further harm |
| Secret rotation mid-deploy, both versions valid | Safe | Also safe | Not applicable |
| Previous container image deleted from registry | Not possible | Only option | If code is behind flag |
| Fix is one config line | Roll forward faster | **First choice** | If behind flag |
| Fix is complex logic, diagnosis unclear | **First choice** | Risk: ships broken code under pressure | Fastest if available |

When roll-back and roll-forward are both viable, pick the one that gets you below your error budget threshold faster. For most teams with Helm or Argo Rollouts, a binary rollback takes 3–5 minutes. A roll-forward that requires a PR review and CI pipeline takes 15–30 minutes. Roll back unless you have a specific technical reason not to.

---

## The Un-Rollback-able Deploy

Not every deploy can be rolled back. This is the most dangerous blindspot in rollback culture — teams that assume "we can always roll back" take risks that they cannot actually undo.

There are four classes of un-rollback-able situations, and they all share a common property: the deploy has changed shared persistent state in a way that the previous binary cannot handle, or it has triggered side effects in the external world that cannot be undone by reverting the code.

### Class 1: Incompatible Schema Already Migrated

Consider a deploy that renames a database column: `name` becomes `first_name` + `last_name`. The migration runs as part of the deployment process, typically via a Flyway or Liquibase migration hook that fires before the new binary starts receiving traffic. The migration completes. The old binary references a column called `name` which no longer exists. Rolling back the binary without rolling back the migration causes every database query in the old binary to fail with a `column "name" does not exist` error. You have traded a bad new deploy for a broken old one — and depending on the traffic pattern during the rolling update, you may not even notice immediately, because pods that have already been replaced are serving the new binary successfully while the rolled-back pods crash on startup.

A particularly dangerous variant is a migration that adds a NOT NULL column without a default value. The old binary does not supply this column on INSERT, so every write by the rolled-back binary fails with a constraint violation. Even a migration that merely sets NOT NULL on an existing nullable column creates this trap — the old binary may write NULL for that column, which now the database rejects.

**How to prevent it.** Use the **expand/contract migration** pattern described in detail in [Database Migrations in the Delivery Pipeline](/blog/software-development/ci-cd/database-migrations-in-the-delivery-pipeline). The old column is preserved until the new binary is confirmed stable, at which point a second migration in a subsequent deploy drops it. Every new NOT NULL column must include a DEFAULT value so that old binaries that do not supply the column still succeed on write. The pattern is:

1. **Expand:** Add the new column with a DEFAULT. Old binary writes NULLs for it (or the default fires). New binary writes real values.
2. **Verify:** Run the new binary for the rollback window. Confirm it is stable.
3. **Contract:** Drop the old column in a subsequent deploy, after the rollback window has closed.

The expand/contract approach requires running with redundant columns in production for days or weeks. Teams sometimes resist this because they find it messy. The alternative — a migration that breaks rollback-ability — costs them days of MTTR when something goes wrong.

**How to recover when you are already in the hole.** If the incompatible migration has already run and you cannot safely roll back the binary, you have three options:

- Create a SQL view with the old column name pointing to the new columns, restoring the old binary's query compatibility temporarily while you ship a hotfix.
- Add the old column back (temporarily) with the new column's value as its source, via a generated column or a trigger.
- Ship a roll-forward patch that updates the binary to tolerate both the old schema and the new schema simultaneously, then deploy it, then do the final schema cleanup.

None of these are pleasant. They are the cost of skipping expand/contract.

### Class 2: Events Already Published to Kafka

Message queues, caches, event stores, and blob stores compound the rollback problem in ways that schema migrations do not. If v2 of your service begins writing a new Protobuf schema version to a Kafka topic, rolling back to v1 leaves you with a queue that contains both v1 and v2 format messages. v1 consumers cannot deserialise the v2 messages. Depending on queue depth and consumer lag, you may face hours of backlog that the rolled-back binary simply cannot process.

The same problem appears with caches. If v2 begins writing cache entries in a new format — for example, it encodes a richer session object with new fields — and you roll back to v1, the cache will contain v2-format entries that v1 cannot parse. v1 may throw deserialisation exceptions, or it may silently return empty values (a cache miss masquerading as data loss), degrading performance until the cache naturally expires.

**How to prevent it.** Use a **schema registry with backward-compatible schema evolution**. Every Kafka message carries a schema version field. Consumers are required to handle version n-1 messages in addition to the current version. If you skipped that design, you are rolling forward — there is no safe binary rollback available.

For caches: use versioned cache keys. When v2 changes the format, it writes to a new key namespace (`session:v2:<id>` instead of `session:<id>`). Rolling back to v1 leaves it reading from the v1 namespace, which is empty but safe. Cache warm-up performance degrades, but correctness is maintained.

**How to recover.** If v2 messages are already in the queue and you need to roll back to v1 consumers, your options are:

- Write a consumer that reads v2 messages and re-publishes them in v1 format to a separate topic, which v1 consumers can handle. This bridge consumer runs until the queue depth clears.
- If you have a dead-letter queue, let v1 consumers NACK the unreadable v2 messages into the DLQ, and replay them once v2 is redeployed and stable.
- Accept the gap. If the messages are non-critical and you can afford to drop them, flush the queue and start fresh.

### Class 3: User-Visible Side Effects Already Committed

This is the class that no amount of binary rollback can fix, because the damage is in the external world rather than in your database or queues. Four concrete examples:

**Payment processed.** A bug in the new checkout flow charged users twice. Rolling back the binary stops the doubled charging, but the users who were already charged still have duplicate transactions. The rollback is still correct — stop the bleeding — but recovery requires a refund batch job, customer support outreach, and potential chargebacks, none of which are automated by the rollback.

**Email sent.** A campaign deploy had a bug that sent a promotional email to users who had opted out. Rolling back the binary prevents further sends, but the already-delivered emails are in inboxes permanently. Recovery involves an apology email, potential CAN-SPAM compliance review, and unsubscribe rate damage.

**Order confirmation sent.** An inventory bug confirmed orders for out-of-stock items. Rolling back the binary stops new confirmations, but customers already have confirmation emails for orders that cannot be fulfilled. Recovery is a customer communication problem, not a software problem.

**Account state changed.** A permissions bug granted admin access to a cohort of users who should not have it. Rolling back reverts the code path, but does not revoke the permissions that were already granted in the database.

**How to prevent it.** Dual-write and verify before committing side effects. Put payment processing, email delivery, and permission grants behind explicit confirmation steps that can be deferred. Use feature flags to gate new code paths in these sensitive areas, so you can ship the code and toggle it in a controlled window with validation rather than at 2am during a deploy.

**How to recover.** The binary rollback is still correct as the first step — stop creating new damage. Then treat the already-committed side effects as a separate remediation track: batch refund jobs, customer communication, permission audits. Document both tracks in the post-mortem.

### Class 4: Secret Rotation Mid-Deploy

Secret rotation events — API key rotations, certificate renewals, OAuth client credential changes — are often tied to deployments. The new binary receives the new credentials; the old binary is configured for the old ones. If the rotation has completed and the old credentials have been invalidated, rolling back to the old binary results in authentication failures across every downstream dependency call.

This scenario is particularly insidious because it can affect services beyond the one you deployed. If service A deployed with a new database password and the rotation completed, all other services that connect to the same database using the old password will also begin failing after rollback — even though those services were not deployed.

**How to prevent it.** Use a **two-phase rotation**: keep both old and new credentials valid simultaneously during the deploy window. Only expire the old credentials after the new binary is confirmed stable and the rollback window has passed. Most cloud secret managers support this: AWS Secrets Manager's `RotationLambda` pattern keeps the previous version active with the `AWSPREVIOUS` label until you explicitly deactivate it. HashiCorp Vault's dynamic secrets feature generates new credentials per request without invalidating existing ones.

**How to recover.** Re-enable the old credential version in your secret manager before executing the binary rollback. AWS Secrets Manager:

```bash
# Promote AWSPREVIOUS back to AWSCURRENT before rolling back the binary
aws secretsmanager update-secret-version-stage \
  --secret-id payments-db-password \
  --version-stage AWSCURRENT \
  --move-to-version-id <previous-version-id> \
  --remove-from-version-id <current-version-id>

# Now the old binary will authenticate successfully
kubectl rollout undo deployment/payments-service --namespace production
```

---

## Automated Rollback in Progressive Delivery: Argo Rollouts

The most powerful rollback mechanism in a modern Kubernetes platform is not a runbook — it is an automated analysis gate that reverts traffic before a human even knows there is a problem.

[Argo Rollouts](https://argoproj.github.io/rollouts/) is a Kubernetes controller that extends the native `Deployment` resource with canary and blue/green strategies, plus a mechanism called `AnalysisRun` that continuously queries metrics during a rollout and automatically triggers rollback when thresholds are breached.

![Argo Rollouts canary progression stack](/imgs/blogs/rollbacks-and-recovering-a-bad-deploy-3.png)

The canary flow works like this: traffic is shifted incrementally — 1%, then 10%, then 50%, then 100% — with an analysis gate between each step. The analysis gate queries Prometheus (or Datadog, or New Relic, or any metrics backend with an HTTP query API) and evaluates the query result against a success condition. If the condition fails more than the configured `failureLimit`, the rollout is automatically aborted and traffic is reverted to the stable revision.

### Complete Rollout YAML with AnalysisRun

Here is a complete `Rollout` manifest for a payments service with a three-step canary plus automated SLO analysis:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: payments-service
  namespace: production
spec:
  replicas: 20
  revisionHistoryLimit: 10
  selector:
    matchLabels:
      app: payments-service
  template:
    metadata:
      labels:
        app: payments-service
    spec:
      containers:
        - name: payments-service
          image: registry.example.com/payments-service:2.4.1
          ports:
            - containerPort: 8080
          resources:
            requests:
              memory: "256Mi"
              cpu: "250m"
            limits:
              memory: "512Mi"
              cpu: "500m"
  strategy:
    canary:
      stableService: payments-service-stable
      canaryService: payments-service-canary
      trafficRouting:
        nginx:
          stableIngress: payments-service-ingress
      steps:
        - setWeight: 1
        - pause:
            duration: 60s
        - analysis:
            templates:
              - templateName: payments-slo-check
            args:
              - name: service-name
                value: payments-service
        - setWeight: 10
        - pause:
            duration: 60s
        - analysis:
            templates:
              - templateName: payments-slo-check
            args:
              - name: service-name
                value: payments-service
        - setWeight: 50
        - pause:
            duration: 120s
        - analysis:
            templates:
              - templateName: payments-slo-check
            args:
              - name: service-name
                value: payments-service
        - setWeight: 100
      autoPromotionEnabled: false
```

Each field in this manifest has a purpose:

- `revisionHistoryLimit: 10` — retain the last 10 stable ReplicaSets, giving you a 10-revision rollback window.
- `stableService` and `canaryService` — two Kubernetes Services that Argo Rollouts manages. The NGINX ingress controller splits traffic between them based on the `setWeight` step. The stable service always routes to the current stable pods; the canary service routes to the new pods.
- `trafficRouting.nginx.stableIngress` — the name of the NGINX Ingress resource that Argo Rollouts will annotate with the canary weight. Argo Rollouts supports NGINX, Istio, AWS ALB, Traefik, and several other ingress providers.
- `steps` — the progression sequence. Each `setWeight` step shifts the traffic percentage. Each `pause` gives the analysis window enough time to collect data. Each `analysis` step launches an `AnalysisRun` that queries Prometheus; if it fails, the rollout halts and reverses.
- `autoPromotionEnabled: false` — the rollout stops at 100% weight and requires a manual promotion command (`kubectl-argo-rollouts promote payments-service`) before updating the stable ReplicaSet. This gives you a final human approval gate before the canary is fully "promoted" as the new stable.

### The AnalysisTemplate: Prometheus-Backed SLO Gate

The `AnalysisTemplate` defines what metrics to query and what thresholds trigger rollback. A well-designed analysis template checks the SLO components independently — availability (error rate) and latency (p99) — so that a degradation in either dimension triggers rollback:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: payments-slo-check
  namespace: production
spec:
  args:
    - name: service-name
  metrics:
    - name: error-rate
      interval: 30s
      count: 5
      failureLimit: 2
      successCondition: result[0] < 0.01
      provider:
        prometheus:
          address: http://prometheus.monitoring.svc.cluster.local:9090
          query: |
            sum(rate(http_requests_total{
              service="{{args.service-name}}",
              status=~"5.."
            }[2m]))
            /
            sum(rate(http_requests_total{
              service="{{args.service-name}}"
            }[2m]))
    - name: p99-latency-seconds
      interval: 30s
      count: 5
      failureLimit: 2
      successCondition: result[0] < 0.5
      provider:
        prometheus:
          address: http://prometheus.monitoring.svc.cluster.local:9090
          query: |
            histogram_quantile(0.99,
              sum(rate(http_request_duration_seconds_bucket{
                service="{{args.service-name}}"
              }[2m])) by (le)
            )
    - name: success-rate
      interval: 30s
      count: 5
      failureLimit: 2
      successCondition: result[0] > 0.995
      provider:
        prometheus:
          address: http://prometheus.monitoring.svc.cluster.local:9090
          query: |
            sum(rate(http_requests_total{
              service="{{args.service-name}}",
              status=~"2.."
            }[2m]))
            /
            sum(rate(http_requests_total{
              service="{{args.service-name}}"
            }[2m]))
```

Field explanations for the `error-rate` metric block:

- `interval: 30s` — evaluate the Prometheus query every 30 seconds.
- `count: 5` — run this metric for 5 evaluation cycles (5 × 30s = 2.5 minutes total observation window).
- `failureLimit: 2` — allow up to 2 failed evaluations before aborting the rollout. This prevents a single noisy data point from triggering a false rollback.
- `successCondition: result[0] < 0.01` — the query must return a value below 0.01 (1% error rate) to be considered passing.
- `provider.prometheus.query` — the PromQL expression. The `{{args.service-name}}` placeholder is replaced with the `value` from the `analysis.args` block in the Rollout manifest. This makes the same template reusable across all services.

When the `AnalysisRun` fails — either the error rate exceeds 1%, p99 latency exceeds 500ms, or success rate drops below 99.5%, failing twice within the 30-second evaluation interval — Argo Rollouts automatically executes the full rollback sequence:

1. Sets canary traffic weight back to 0% — all traffic returns to the stable service immediately
2. Scales down all canary pods to zero
3. Marks the `Rollout` as `Degraded` in its status
4. Fires a Kubernetes event of type `Warning` with reason `RollbackCompleted`
5. Updates the `Rollout` status conditions with a detailed failure message

The Kubernetes event can be forwarded to your alerting system. A simple event exporter configuration (e.g., `kubernetes-event-exporter`) will capture this and send it to Slack or PagerDuty:

```yaml
# kubernetes-event-exporter config snippet
receivers:
  - name: "slack-rollback"
    slack:
      token: "${SLACK_TOKEN}"
      channel: "#incident-response"
      message: |
        :rotating_light: *Argo Rollouts auto-rollback fired*
        Rollout: {{ .InvolvedObject.Name }}
        Namespace: {{ .InvolvedObject.Namespace }}
        Reason: {{ .Reason }}
        Message: {{ .Message }}

route:
  routes:
    - match:
        - receiver: "slack-rollback"
          reason: "RollbackCompleted"
```

The entire automated rollback completes in under two minutes. The on-call engineer receives a notification that says "canary rollback completed" rather than an alert that demands diagnosis. Their job is to acknowledge the incident, note that it was auto-resolved, open a post-mortem issue, and go back to sleep.

This is the core principle: **move the rollback decision from human judgment to a metric threshold**. Humans are slow, distracted at 2am, and prone to optimism bias ("it might recover on its own, let me wait another five minutes"). Metrics are fast, always awake, and have no feelings about the deploy they just reverted.

### Testing the AnalysisRun Before You Need It

The most common mistake teams make with Argo Rollouts is configuring the analysis template correctly but never verifying that it actually fires. The Prometheus query may return `null` instead of `0` for services with low traffic (division-by-zero case). The `failureLimit` may be set too high for the traffic volume. The metric name in the Prometheus query may differ between staging and production.

Test the analysis gate in staging before relying on it in production:

```bash
# Launch a test AnalysisRun manually with known-bad thresholds
kubectl apply -f - <<EOF
apiVersion: argoproj.io/v1alpha1
kind: AnalysisRun
metadata:
  name: slo-check-test
  namespace: staging
spec:
  metrics:
    - name: error-rate-test
      interval: 10s
      count: 3
      failureLimit: 1
      successCondition: result[0] < 0.0001
      provider:
        prometheus:
          address: http://prometheus.monitoring.svc.cluster.local:9090
          query: |
            sum(rate(http_requests_total{service="payments-service",status=~"5.."}[2m]))
            /
            sum(rate(http_requests_total{service="payments-service"}[2m]))
EOF

# Watch the AnalysisRun status — it should fail (this is expected)
kubectl get analysisrun slo-check-test -n staging --watch
kubectl describe analysisrun slo-check-test -n staging
```

If the AnalysisRun does not fail when you set an impossibly tight threshold, something is wrong with the query. Fix it in staging, not during a production incident.

---

## Helm Rollback: The Release History Approach

If your team uses Helm for templated Kubernetes deployments, rollback is built into the release model. Every `helm upgrade` creates a new revision. The previous revision's rendered manifests — the fully-expanded Kubernetes YAML — are stored in a Kubernetes Secret in the same namespace with a label `owner=helm`. Rolling back is a matter of pointing Helm at the previous revision's Secret and re-applying it.

To see the revision history for the `payments-service` release:

```bash
helm history payments-service --namespace production
```

Sample output:

```
REVISION  UPDATED                   STATUS     CHART                    APP VERSION  DESCRIPTION
1         Fri Jun 20 14:32:01 2026  superseded payments-service-2.3.0  2.3.0        Install complete
2         Fri Jun 20 22:15:44 2026  superseded payments-service-2.3.1  2.3.1        Upgrade complete
3         Sat Jun 21 01:55:12 2026  failed     payments-service-2.4.1  2.4.1        Upgrade "payments-service" failed: deployment exceeded progress deadline
```

Roll back to the last good revision (revision 2):

```bash
helm rollback payments-service 2 --namespace production
```

Or simply roll back to the immediately previous revision using revision `0` (shorthand for "previous"):

```bash
helm rollback payments-service 0 --namespace production
```

Wait for the rollback to complete and verify:

```bash
helm status payments-service --namespace production
helm history payments-service --namespace production
kubectl rollout status deployment/payments-service --namespace production
```

After a successful rollback, `helm history` will show a new revision 4 with description `Rollback to 2`. This is important: rollbacks are forward events in the Helm history. The revision numbering only ever increases. You never lose history through normal rollback operations.

### Helm Rollback Deep-Dive: Flags and Edge Cases

**The `--wait` flag** blocks the `helm rollback` command until all pods from the rolled-back revision are healthy. Without it, the command returns immediately after submitting the rollback request, and you need to separately poll `kubectl rollout status` to confirm completion. During an incident, use `--wait` so the command itself signals success or failure:

```bash
helm rollback payments-service 2 \
  --namespace production \
  --wait \
  --timeout 5m
```

If the rollback does not complete within the timeout, the command exits with a non-zero status, which allows your incident runbook script to detect the failure and escalate.

**The `--atomic` flag on `helm upgrade`** is a preventative measure that does not help with manual rollback, but dramatically reduces the number of incidents that require manual rollback in the first place. With `--atomic`, Helm automatically rolls back to the previous release if the upgrade fails (pods fail readiness checks within the timeout). This is the simplest form of automated rollback for Helm-based deployments:

```bash
helm upgrade payments-service ./chart \
  --namespace production \
  --atomic \
  --timeout 10m \
  --set image.tag=${NEW_IMAGE_TAG}
```

The `--atomic` flag combines `--wait` (wait for all resources to become ready) with automatic rollback on failure. The caveat: if the new version has a bug that only manifests under real traffic (not during pod startup), `--atomic` will not catch it — the pods start healthy, the upgrade succeeds, and the bug appears later. That is what Argo Rollouts with AnalysisRun solves.

**The `revisionHistoryLimit` trap.** Helm's underlying mechanism stores each revision as a Secret. By default, Helm keeps the last 10 revisions (`--history-max` flag). In a continuous deployment environment that deploys twenty times per day, you may have a rollback window of only 12 hours before revision 2 is garbage-collected. The lesson: set `revisionHistoryLimit` explicitly in your Helm release and ensure your rollback window aligns with how many revisions you keep.

Check whether a specific revision's Secret still exists before counting on it:

```bash
# List all Helm release Secrets, sorted by creation time
kubectl get secrets --namespace production \
  -l "owner=helm,name=payments-service" \
  --sort-by='.metadata.creationTimestamp' \
  -o custom-columns="NAME:.metadata.name,REVISION:.metadata.labels.version,CREATED:.metadata.creationTimestamp"
```

**The `--dry-run` option.** Before executing a rollback in a high-stakes situation, use `--dry-run` to see what Helm would apply:

```bash
helm rollback payments-service 2 --namespace production --dry-run
```

This renders the manifests for revision 2 and outputs them without applying, so you can verify that the previous configuration is what you expect before committing to the rollback.

**Rollback with cleanup.** If the failed upgrade left orphaned resources — for example, a Job that ran as part of the upgrade hook and failed halfway through — Helm's rollback will not automatically clean these up. You may need to manually delete them:

```bash
# List all Jobs created by the failed upgrade
kubectl get jobs --namespace production \
  -l "helm.sh/chart=payments-service-2.4.1"

# Delete the failed jobs (they will be recreated by rollback hooks if needed)
kubectl delete jobs --namespace production \
  -l "helm.sh/chart=payments-service-2.4.1"

# Now execute the rollback
helm rollback payments-service 2 --namespace production
```

---

## Kubernetes Rollout Undo: The Pod-Level Mechanism

For teams that manage deployments directly with `kubectl` rather than Helm, the native `kubectl rollout undo` command achieves the same result with a slightly simpler interface.

Check current rollout status and history:

```bash
kubectl rollout status deployment/payments-service --namespace production
kubectl rollout history deployment/payments-service --namespace production
```

Sample output (with `CHANGE-CAUSE` annotations properly set):

```
REVISION  CHANGE-CAUSE
1         deploy: payments-service@sha256:abc123 (2026-06-20 14:32)
2         deploy: payments-service@sha256:def456 (2026-06-20 22:15)
3         deploy: payments-service@sha256:789ghi (2026-06-21 01:55)
```

Roll back to the previous revision (revision 2):

```bash
kubectl rollout undo deployment/payments-service --namespace production
```

Roll back to a specific revision:

```bash
kubectl rollout undo deployment/payments-service --namespace production --to-revision=2
```

Watch the rollback progress in real time:

```bash
kubectl rollout status deployment/payments-service --namespace production --watch
```

Verify that the correct image is now running:

```bash
kubectl get deployment payments-service --namespace production \
  -o jsonpath='{.spec.template.spec.containers[0].image}'
```

**The `CHANGE-CAUSE` annotation.** By default, revision history entries show blank `CHANGE-CAUSE` fields unless you explicitly annotate deployments. The rollout history is nearly useless during an incident without this. Set the annotation in your CI pipeline deploy step:

```bash
kubectl annotate deployment/payments-service \
  kubernetes.io/change-cause="deploy: payments-service@${IMAGE_TAG} ($(date -u +%Y-%m-%dT%H:%M:%SZ)) by ${CI_USER:-ci}" \
  --namespace production \
  --overwrite
```

Or set it declaratively in your deployment manifest and let the CI pipeline template it:

```yaml
metadata:
  name: payments-service
  annotations:
    kubernetes.io/change-cause: "deploy: payments-service@${IMAGE_TAG}"
```

**The `revisionHistoryLimit` in Deployments.** Kubernetes Deployments also have a `revisionHistoryLimit` field, separate from Helm's. It defaults to 10. Each revision is stored as a `ReplicaSet` that is retained (scaled to 0 replicas) in the namespace. To roll back, Kubernetes simply scales up the desired `ReplicaSet`. If the `ReplicaSet` has been garbage-collected, rolling back to that revision is not possible.

Set this explicitly in your `Deployment` spec:

```yaml
spec:
  revisionHistoryLimit: 15
```

**Pod Disruption Budgets and rollback speed.** A `PodDisruptionBudget` (PDB) enforces availability constraints during voluntary disruptions, including rollbacks. If you have set `minAvailable: 90%` on a 20-pod deployment, Kubernetes will only terminate two pods at a time during the rollback, making the process take approximately 10x as long as it would without the PDB. In a 2am incident, waiting eighteen minutes for a rollback to complete while your error rate stays high is painful.

During a declared incident, temporarily relaxing the PDB is acceptable:

```bash
# Save the current PDB spec before deleting
kubectl get pdb payments-service-pdb --namespace production -o yaml > /tmp/payments-pdb-backup.yaml

# Delete the PDB (ONLY during a declared incident; restore immediately after)
kubectl delete pdb payments-service-pdb --namespace production

# Execute the rollback
kubectl rollout undo deployment/payments-service --namespace production

# Wait for rollback to complete
kubectl rollout status deployment/payments-service --namespace production --timeout=120s

# Restore the PDB
kubectl apply -f /tmp/payments-pdb-backup.yaml
```

This is a controlled, temporary measure. Document it in your incident timeline and make restoring the PDB a required step in your incident closure checklist.

---

## GitHub Actions: Auto-Revert on Failed Smoke Test

Automated rollback does not require Argo Rollouts. A simpler pattern for teams using GitHub Actions is to run a smoke test immediately after deploy and automatically trigger a revert if it fails. This is appropriate for smaller teams that cannot justify the operational overhead of a full progressive delivery controller.

```yaml
name: Deploy and Auto-Revert on Smoke Failure

on:
  push:
    branches: [main]

env:
  SERVICE_NAME: payments-service
  NAMESPACE: production
  REGISTRY: registry.example.com

jobs:
  deploy:
    runs-on: ubuntu-latest
    outputs:
      deploy_succeeded: ${{ steps.deploy.outputs.deploy_succeeded }}
    steps:
      - uses: actions/checkout@v4

      - name: Configure kubectl
        uses: azure/k8s-set-context@v3
        with:
          kubeconfig: ${{ secrets.KUBECONFIG }}

      - name: Deploy new image
        id: deploy
        run: |
          kubectl set image deployment/${SERVICE_NAME} \
            ${SERVICE_NAME}=${REGISTRY}/${SERVICE_NAME}:${{ github.sha }} \
            --namespace ${NAMESPACE}

          if kubectl rollout status deployment/${SERVICE_NAME} \
            --namespace ${NAMESPACE} --timeout=300s; then
            echo "deploy_succeeded=true" >> "$GITHUB_OUTPUT"
          else
            echo "deploy_succeeded=false" >> "$GITHUB_OUTPUT"
          fi

  smoke-test:
    needs: deploy
    runs-on: ubuntu-latest
    if: needs.deploy.outputs.deploy_succeeded == 'true'
    outputs:
      smoke_passed: ${{ steps.smoke.outputs.smoke_passed }}
    steps:
      - name: Health check and error-rate smoke test
        id: smoke
        run: |
          # Basic health check
          HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
            --max-time 10 \
            https://api.example.com/v1/health)

          # Error rate check via Prometheus query API
          ERROR_RATE=$(curl -s \
            "https://prometheus.example.com/api/v1/query" \
            --data-urlencode \
            'query=sum(rate(http_requests_total{service="payments-service",status=~"5.."}[5m]))/sum(rate(http_requests_total{service="payments-service"}[5m]))' \
            | jq -r '.data.result[0].value[1] // "0"')

          echo "HTTP status: ${HTTP_STATUS}"
          echo "Error rate: ${ERROR_RATE}"

          # Pass if health check returns 200 AND error rate is under 1%
          if [ "${HTTP_STATUS}" = "200" ] && \
             [ "$(echo "${ERROR_RATE} < 0.01" | bc -l)" = "1" ]; then
            echo "smoke_passed=true" >> "$GITHUB_OUTPUT"
          else
            echo "smoke_passed=false" >> "$GITHUB_OUTPUT"
            echo "FAILURE REASON: HTTP=${HTTP_STATUS}, error_rate=${ERROR_RATE}"
          fi

  auto-revert:
    needs: [deploy, smoke-test]
    runs-on: ubuntu-latest
    if: |
      always() &&
      (needs.deploy.outputs.deploy_succeeded == 'false' ||
       needs.smoke-test.outputs.smoke_passed == 'false')
    steps:
      - name: Configure kubectl
        uses: azure/k8s-set-context@v3
        with:
          kubeconfig: ${{ secrets.KUBECONFIG }}

      - name: Execute auto-revert
        run: |
          echo "Smoke test failed or deploy timed out. Auto-reverting to previous revision."
          kubectl rollout undo deployment/${SERVICE_NAME} \
            --namespace ${NAMESPACE}
          kubectl rollout status deployment/${SERVICE_NAME} \
            --namespace ${NAMESPACE} --timeout=180s
          echo "Rollback complete. Current image:"
          kubectl get deployment/${SERVICE_NAME} \
            --namespace ${NAMESPACE} \
            -o jsonpath='{.spec.template.spec.containers[0].image}'

      - name: Alert the team
        uses: slackapi/slack-github-action@v1.26.0
        with:
          payload: |
            {
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": ":warning: *AUTO-REVERT triggered for `${{ env.SERVICE_NAME }}`*\n\nDeploy `${{ github.sha }}` failed smoke test.\nReverted to previous stable revision automatically.\n\n*Action required:* Investigate the failing deploy before merging another change.\n\n<${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}|View deployment run>"
                  }
                }
              ]
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
          SLACK_WEBHOOK_TYPE: INCOMING_WEBHOOK
```

This workflow provides automated rollback without a custom Kubernetes controller. Its limitations compared to Argo Rollouts are real — it does not provide progressive traffic shifting, the smoke test window is short (5 minutes of Prometheus data), and it cannot run continuous analysis across canary steps — but it is a significant improvement over no automation at all.

---

## Feature Flags as Instant Rollback

The fastest rollback mechanism available is one that does not involve the deployment pipeline at all.

![Feature flag as zero-deploy rollback path](/imgs/blogs/rollbacks-and-recovering-a-bad-deploy-6.png)

When new code is shipped behind a feature flag that defaults to `off`, the production binary already contains the new code path — it is simply dormant. Turning the flag off in the feature flag service propagates in under a second (streaming) or under thirty seconds (polling), affecting all running instances simultaneously. No new container image is pulled. No rolling restart is initiated. No pods are disrupted. No deployment pipeline is invoked. The entire change is a key-value update in your flag service's backend store.

### Rollback Method Comparison Table

The table below gives you a direct comparison of every rollback path across the dimensions that matter during an incident.

| Rollback method | Time to safe | Data-safe on its own | Requires CI pipeline | Notes |
|---|---|---|---|---|
| Feature flag flip (streaming) | Under 1 minute | Yes — no binary or state change | No | Fastest path; requires code to be flag-gated |
| Feature flag flip (polling) | 1–2 minutes | Yes | No | 30–60s propagation lag |
| Argo Rollouts auto-rollback | 2–4 minutes | Yes if schema backward-compat | No | Fully automated; fires before human wakes up |
| `kubectl rollout undo` | 3–7 minutes | Yes if schema backward-compat | No | Requires previous ReplicaSet in namespace |
| `helm rollback` | 3–7 minutes | Yes if schema backward-compat | No | Requires revision in Helm history |
| `helm upgrade --atomic` (auto) | 5–10 minutes | Yes if schema backward-compat | No | Fires on pod readiness failure, not runtime metrics |
| GitHub Actions auto-revert | 8–15 minutes | Yes if schema backward-compat | Yes (re-uses pipeline) | Simpler ops model; slower than in-cluster |
| Git revert + CI pipeline | 15–30 minutes | Yes if schema backward-compat | Yes | GitOps-clean; slowest for pure mitigation |
| Roll-forward hotfix | 20–60 minutes | Yes — targeted fix | Yes | Only option for un-rollback-able deploys |
| Re-enable old credential (secret rotation) | 2–5 minutes | Yes | No | Applies only to secret rotation class |

The mechanics depend on your flag evaluation architecture:

**SDK polling (most common, e.g., LaunchDarkly polling mode, Unleash):** SDK clients poll the flag service every 30–60 seconds. Flag propagation takes 0–30 seconds depending on the interval. This is the standard model for most teams.

**Server-Sent Events / streaming (e.g., LaunchDarkly streaming mode, Flagsmith streaming):** Flag changes propagate in real time — typically under 200ms per SDK client. This is the model to use if your flags protect revenue-critical paths where a 30-second propagation delay is unacceptable.

**Database-backed (homegrown):** Flag values are stored in a database row. Each request evaluates the flag by reading from the database or from an in-process cache that refreshes every few seconds. Propagation speed depends on cache TTL. This model is cheap to build and adequate for low-frequency flag evaluations, but adds a cache read per request on the critical path.

**The critical operational requirement:** the flag service must be available during an incident. If your feature flag service is down, the flag flip is unavailable as a rollback path. This means the flag service itself must have a higher SLO than any service it protects, with a safe default configured for flag service outages. The safe default question is not obvious: "fail open" (treat missing flag as enabled) means new code runs during a flag service outage; "fail closed" (treat missing flag as disabled) means old code runs. The right choice depends on whether the default behavior or the new behavior is safer.

For revenue-critical paths, the recommendation is always "fail closed" — if you cannot confirm the flag is enabled, run the old code path. For non-critical paths, "fail open" may be acceptable.

See [Feature Flags: Decoupling Deploy from Release](/blog/software-development/ci-cd/feature-flags-decoupling-deploy-from-release) for the full treatment of flag architecture, rollout targeting, percentage rollouts, and the zero-deploy rollback path in a multi-service environment.

---

## The Recovery Runbook: A Step-by-Step On-Call Procedure

![On-call rollback decision tree](/imgs/blogs/rollbacks-and-recovering-a-bad-deploy-8.png)

A recovery runbook should require zero thought. Its job is to turn a panicked 2am engineer into a deterministic decision machine. The worst runbook is a prose document that requires judgment at every step. The best runbook is a flowchart with a concrete command at every leaf node.

Here is the complete runbook for a Kubernetes-based platform team, including the regulated-environment approval steps that most open-source runbooks omit.

### Step 0: Do not diagnose root cause yet (0 minutes)

This is the most important step. The SRE principle — codified in [Mitigate First, Diagnose Later](/blog/software-development/site-reliability-engineering/mitigate-first-diagnose-later) — is that your first job is to stop the bleeding, not to understand why it started. You will diagnose after the error rate is below the SLO threshold. Every minute you spend diagnosing before mitigating is a minute users spend experiencing the incident.

### Step 1: Confirm deployment correlation (2 minutes)

Check whether a deployment occurred within the last 30 minutes and whether its timing correlates with the alert:

```bash
# Check recent Helm deployments
helm history payments-service --namespace production | tail -5

# Check recent kubectl rollout events
kubectl rollout history deployment/payments-service --namespace production

# Check Kubernetes events for recent deploy activity
kubectl get events --namespace production \
  --sort-by='.lastTimestamp' \
  --field-selector reason=ScalingReplicaSet | tail -10

# Cross-reference with alert timestamp from monitoring
# (Check Prometheus alert annotations or PagerDuty for the exact alert time)
```

If no deployment occurred in the last 30 minutes, or if the alert timing does not correlate with the deploy time, this is not a deployment incident. Follow your general incident response procedure. Do not roll back a deploy that is not the cause — you are adding noise to a broken system.

### Step 2: Check for feature flag (30 seconds)

If the incident is deployment-correlated and the impacted feature is behind a flag, flip the flag off immediately:

1. Open your feature flag dashboard.
2. Find the flag controlling the affected code path.
3. Set it to 0% exposure (all traffic off).
4. Wait 30–60 seconds.
5. Check the error rate dashboard.

If the error rate drops, the flag flip is your mitigation. Declare the incident mitigated (with ongoing monitoring), mark a post-mortem for root cause analysis, and go back to sleep.

If the error rate does not drop after 60 seconds, proceed to Step 3.

### Step 3: Assess data mutability (60 seconds)

Before attempting a binary rollback, answer the three un-rollback-able questions:

```bash
# Did this deploy include a database migration?
# For Flyway:
psql -U admin -d payments -c \
  "SELECT version, description, installed_on FROM flyway_schema_history ORDER BY installed_on DESC LIMIT 5;"

# For Liquibase:
psql -U admin -d payments -c \
  "SELECT filename, dateexecuted FROM databasechangelog ORDER BY dateexecuted DESC LIMIT 5;"

# Did the migration run after the last known-good revision?
# Compare the migration timestamps with the deploy timestamp.
```

If a migration ran after the last good deploy timestamp, evaluate whether the migration is backward-compatible:
- **Compatible (added columns, added indexes, added NOT NULL with DEFAULT):** Safe to roll back the binary.
- **Incompatible (dropped columns, renamed columns, added NOT NULL without DEFAULT):** Do NOT roll back the binary. Proceed to roll-forward (Step 5).

### Step 4: Regulated-environment approval gate (0–5 minutes depending on SOX/PCI tier)

In regulated environments (SOX-scoped services, PCI-DSS cardholder data environments, HIPAA-covered services), an unilateral rollback by the on-call engineer without approval may violate change management policy. Most regulatory frameworks require:

- A documented change request (even a retroactive one opened mid-incident is usually acceptable under "emergency change" provisions)
- Approval from a second authorised person (typically the on-call manager or a designated approver on the incident call)
- An audit trail connecting the rollback action to the change record

The practical runbook for regulated environments:

1. Open an emergency change ticket immediately upon incident declaration. Most ITSM tools (ServiceNow, Jira Service Management) have an emergency change type that bypasses the standard approval window.
2. Post the ticket number in the incident channel.
3. Page the on-call manager or designated approver. In many organisations, verbal approval on the incident call — recorded in the call log — satisfies the "second approver" requirement.
4. Execute the rollback with the approver present on the call.
5. Document the rollback command and its output in the change ticket immediately after execution.

The key point: the approval process should take 2–5 minutes for a pre-planned rollback type, not 30 minutes. If your change management process makes emergency rollbacks take 30 minutes, the process itself is a safety hazard. Work with your compliance team to establish pre-approved rollback playbooks that require only verbal confirmation rather than a full CAB review during an incident.

### Step 5: Roll back (3–5 minutes)

If the deploy is rollback-safe and approved:

```bash
# Option A: Helm rollback
helm rollback payments-service 0 --namespace production --wait --timeout 5m
# (0 means "previous revision")

# Option B: kubectl rollout undo
kubectl rollout undo deployment/payments-service --namespace production

# Monitor progress
kubectl rollout status deployment/payments-service \
  --namespace production --watch
```

### Step 6: Roll forward if rollback is unsafe

If the deploy is not rollback-safe, write and deploy a bridging hotfix:

1. Create a branch from `main`.
2. Write the minimum code change required to restore compatibility (e.g., a SQL view that aliases the old column name to the new one, or a request translation layer in the application).
3. Mark the PR as `hotfix: incident-<id>` to trigger fast-track review.
4. Merge and deploy as soon as one other engineer approves.
5. Verify error rate drops.

This path takes 15–45 minutes depending on the complexity of the bridge. It is slower than a rollback but it is the only safe path when data compatibility is broken.

### Step 7: Verify recovery (2 minutes)

Regardless of which path you took, verify the error rate before declaring the incident mitigated:

```bash
# Query current error rate from Prometheus
curl -s "http://prometheus.monitoring.svc:9090/api/v1/query" \
  --data-urlencode \
  "query=sum(rate(http_requests_total{service=\"payments-service\",status=~\"5..\"}[2m]))/sum(rate(http_requests_total{service=\"payments-service\"}[2m]))" \
  | jq -r '.data.result[0].value[1] // "0"'

# Check pod health
kubectl get pods --namespace production \
  -l app=payments-service \
  -o custom-columns="NAME:.metadata.name,STATUS:.status.phase,READY:.status.conditions[?(@.type==\"Ready\")].status,RESTARTS:.status.containerStatuses[0].restartCount"
```

When the error rate is below your SLO threshold (typically the same threshold as your alerting rule) for at least two consecutive minutes, the incident is mitigated.

### Step 8: Write the post-mortem stub immediately

Do not go back to sleep without writing a post-mortem stub. Five minutes now saves two hours of memory reconstruction tomorrow:

```
Incident: payments-service error spike 2026-06-21 01:55–02:07 UTC
Trigger: Deploy of payments-service v2.4.1
Root cause: [UNKNOWN — investigate tomorrow]
Timeline:
  01:55 — v2.4.1 deployed
  02:07 — PagerDuty alert fired (error rate 8.4%)
  02:09 — Feature flag flip attempted (no flag for this code path)
  02:11 — helm rollback payments-service 2 executed
  02:13 — Error rate returned to baseline (0.1%)
MTTR: 18 minutes (delay: 2min detection, 2min flag check, 7min decision, 2min rollback, 2min verify, 3min Helm history review)
Action items: [to be filled in post-mortem]
```

### The Post-Mortem: Closing the Loop

A post-mortem is not a blame exercise. It is a structured investigation into the systemic conditions that made an incident possible, with the goal of eliminating those conditions. The standard post-mortem structure for a deploy-triggered incident:

**Timeline (exact, to the minute):** Every action taken from the first anomalous signal to the final resolution. Use your monitoring dashboards and incident channel timestamps, not memory.

**Proximate cause:** What specifically broke? (e.g., "the new timeout value of 3s caused upstream calls to fail under normal p99 latency of 4.2s"). This is the technical root cause.

**Contributing factors:** Why was the proximate cause not caught before production? Common findings: the timeout value was not in staging configuration, load tests do not exercise p99 latency realistically, the config change was bundled with a dependency bump and reviewers focused on the code diff, not the config.

**Action items with owners and due dates:** The only output of a post-mortem that matters is action items that change the system. "Add Argo Rollouts AnalysisRun to payments-service" with a specific engineer and a specific week is an action item. "Improve monitoring" is not.

**The five-minute post-mortem review:** Schedule a 20-minute post-mortem review meeting within 48 hours of the incident, while memory is fresh. Assign a facilitator who was not on-call during the incident (they have fresh eyes). The outcome is a refined action item list, not a rehash of the timeline.

---

## MTTR Measurement and the 18x Improvement

MTTR — Mean Time to Recovery — is the key metric for rollback effectiveness. But it is often measured wrong. Many teams measure MTTR as "time from alert to resolution" and stop there. The more useful decomposition is the TTD+TTA+TTM framework:

**MTTR = TTD + TTA + TTM**

- **TTD (Time to Detect):** Time from the first bad request to the alert firing. Driven by alerting thresholds, evaluation windows, and metric collection latency.
- **TTA (Time to Acknowledge):** Time from alert firing to an engineer actively working the incident. Driven by on-call response time and pager escalation policy. At 2am on a Friday, TTA can be 5–15 minutes even with a healthy on-call rotation.
- **TTM (Time to Mitigate):** Time from first action to error rate below threshold. Driven by rollback mechanism speed and decision confidence. This is the component that automated rollback attacks.

Typical breakdown with purely manual rollback on a well-staffed team:
- TTD: ~2 minutes (Prometheus alerting with 1-minute evaluation window)
- TTA: ~5 minutes (pager response at 2am)
- TTM: ~83 minutes (15 min diagnosis, 10 min decision, 5 min rollback execution, 15 min verify, 38 min buffer for false starts, communication overhead, and human uncertainty)
- **MTTR: ~90 minutes**

With Argo Rollouts automated analysis + automated rollback on canary SLO breach:
- TTD: ~2 minutes (AnalysisRun evaluation interval fires within one evaluation window)
- TTA: 0 minutes (no human action required for mitigation — the system reverts automatically)
- TTM: ~1.5 minutes (Argo Rollouts executes the rollback sequence, pods restart, traffic shifts back)
- **MTTR: ~3.5 minutes**

This is the 18x improvement. The improvement is almost entirely in TTM — automation eliminates the diagnosis time and decision uncertainty that consume the bulk of manual MTTR. The TTD component could theoretically be reduced further by tightening the AnalysisRun evaluation interval from 30 seconds to 10 seconds, but this increases the false-positive rate. Most teams find 30-second intervals a good balance between speed and noise.

![MTTR comparison: manual vs automated rollback](/imgs/blogs/rollbacks-and-recovering-a-bad-deploy-2.png)

The business impact of this improvement is significant. At 90 minutes MTTR for a payments service, even a single incident per quarter can consume 30% of a monthly error budget. At 3.5 minutes MTTR, the same incident consumes under 2% of the monthly error budget, leaving room for other failures without SLO breach. This is why automated rollback is not a nice-to-have — it is the difference between an SLO you can defend and one you cannot.

#### Worked example: The arithmetic of 90 min → 5 min

Put plainly, the MTTR improvement translates directly into error budget preservation. Take a service with a 99.9% availability SLO over a 30-day window. That SLO permits 43.2 minutes of downtime per month. A single 90-minute incident burns through the entire monthly budget twice over — the SLO is breached the moment the incident hits the 43-minute mark, and the team spends the rest of the month in "error budget debt."

Now apply automated rollback:

- Before: 1 incident per month × 90 min MTTR = 90 min downtime. SLO breach: yes, by 2.1×.
- After: 1 incident per month × 5 min MTTR = 5 min downtime. SLO breach: no (5 min < 43.2 min allowance).

The monetary implication follows directly. Suppose the payments service processes \$2 million per hour in transaction volume. At 90 minutes of elevated error rate (even at 8% errors, not a complete outage), the revenue impact is approximately \$2M × 1.5 hours × 0.08 error rate = \$240,000 in failed transactions. At 5 minutes, the same calculation yields \$2M × (5/60) hours × 0.08 = \$13,333. The 18x MTTR reduction maps to an 18x reduction in revenue exposure per incident.

The investment to get there: one sprint of an SRE configuring Argo Rollouts and the AnalysisTemplate. The amortisation period for that investment, on a service processing \$2M/hour, is measured in incident-hours, not quarters.

---

## Rollback Method Comparison

Choosing the wrong rollback mechanism under incident pressure wastes precious minutes. The matrix below is the reference for picking the right tool:

![Rollback method trade-off matrix](/imgs/blogs/rollbacks-and-recovering-a-bad-deploy-5.png)

A secondary comparison organised by incident type:

| Incident type | Best rollback method | Approximate MTTR | Key requirement |
|---|---|---|---|
| New code bug, stateless service | `kubectl rollout undo` or `helm rollback` | 3–7 min | Previous revision available in history |
| New code bug, code behind flag | Feature flag flip | 0.5–2 min | Flag service available, correct flag in place |
| Schema migration regression | Roll forward with bridging hotfix | 15–45 min | CI pipeline available, engineer available |
| Config change (env var, ConfigMap) | Edit ConfigMap + rolling restart | 5–10 min | Config change is isolated to the service |
| Secret rotation regression | Two-phase rotation (re-enable old credential) | 2–10 min | Old credential version retained in secret store |
| Canary rollout quality regression | Argo Rollouts automated rollback | 2–5 min | AnalysisRun configured with correct thresholds |
| Multi-service API contract change | Coordinated flag flip across services | 5–15 min | Flags available across all affected services |
| Full production regression, complex state | Feature flag + full postmortem before deploy | 1–3 min (flag) | High confidence flag is available and covers blast radius |

---

## The Knight Capital War Story: The Canonical Un-Rollback-able Deploy

If you want to understand why rollback strategy matters at its most consequential, Knight Capital Group's trading disaster of August 1, 2012 is the required reading. It is the canonical case study of every failure mode that rollback architecture is designed to prevent — and it happened before most of the tools described in this post existed.

![Knight Capital incident timeline](/imgs/blogs/rollbacks-and-recovering-a-bad-deploy-4.png)

Knight Capital was one of the largest electronic market makers in the US, handling approximately 10–15% of US equity volume on any given day. On the morning of August 1, 2012, they deployed a new version of their automated trading system, SMARS (Smart Market Access Routing System), across their production trading servers. The deploy team manually copied the new software to each of the nine production servers one by one. Eight of the nine received the new build. The ninth server was missed — it continued running an older version of SMARS that contained a piece of code called "Power Peg," a repurposed legacy order routing mechanism that had been deactivated years earlier and was supposed to have been removed.

The new deployment had reused a configuration flag that previously activated Power Peg. In the new code, the flag controlled a different feature. On the eight correctly-deployed servers, the flag controlled the new feature harmlessly. On the ninth server, the same flag activated the dormant Power Peg code.

When the market opened at 9:30 am EST, the eight correctly-deployed servers began running the new SMARS code as intended. The ninth server, running the old code with Power Peg active, began treating incoming retail orders as triggers to buy and then immediately sell large blocks of stock on the open market, generating market-making positions at a rate that was orders of magnitude larger than intended.

Over the next 45 minutes, Knight Capital's ninth server executed approximately 4 million trades across 154 securities, generating roughly \$7 billion in unwanted long and short positions. The firm was buying high and selling low in a continuous loop — the opposite of profitable market-making.

When engineers noticed anomalous behavior at approximately 9:41 am — eleven minutes after market open — they began attempting to diagnose and stop the problem. Here is where the un-rollback-able nature of the incident became decisive. Engineers attempted to roll back the software on the problem server. But because the other eight servers were running different code, there was no coherent "previous state" to restore to. The rollback procedures they had were designed for a normal scenario where all servers were on the same version. In this scenario, every action they took to fix one dimension of the problem risked creating another inconsistency.

At approximately 9:58 am, the engineering team shut down the new code on the eight correctly-deployed servers — reverting them to the old code — not realising that this would cause all nine servers to now be running the old code, all with Power Peg active, and dramatically increasing the volume of erroneous orders.

Trading was finally halted manually at approximately 10:16 am, 45 minutes after the incident began. By that time, Knight Capital had crystallised a \$440 million loss. The firm's entire net capital was approximately \$365 million. The loss exceeded their capital. Knight Capital was sold within four days to Getco for a fraction of its previous value. It never operated as an independent entity again.

#### Worked example: Why Knight Capital could not roll back

The mechanism of the un-rollback-ability is worth unpacking precisely, because it maps directly to the framework in this post.

The trades Knight Capital had already executed were real market transactions — user-visible side effects of the worst kind. By 9:41 am, the ninth server had already executed approximately 2.5 million trades. These were actual fills on the New York Stock Exchange. Reversing the binary on the ninth server did not undo those trades. The damage from those 2.5 million trades — roughly \$220 million in adverse market positions — was locked in regardless of what the software did next.

The second problem was the multi-version state inconsistency: the nine servers were running two different versions of the binary with fundamentally incompatible behaviour. There was no "previous uniform state" to roll back to. A system rollback was only possible if all nine servers moved simultaneously to the same version. The engineers who attempted to do this at 9:58 am made the correct conceptual choice, but chose the wrong target version — they unified all nine servers on the version with Power Peg enabled rather than the version without it.

The third problem was the absence of a kill switch. In today's deployment architecture, a Kubernetes Deployment rollback would have taken 2–3 minutes to execute across all nine equivalent pods. The fact that Knight's deployment was a manual file copy to individual servers — with no deployment controller verifying uniform state — meant there was no mechanism to guarantee that a rollback command reached all nine servers simultaneously. This is exactly the class of failure that container orchestration exists to prevent.

The post-mortem lessons map directly to the principles in this post:

1. **Verify uniform state after deploy.** A version endpoint query across all nine servers before market open would have caught the discrepancy. Today: `kubectl get pods -l app=smars -o jsonpath='{range .items[*]}{.spec.containers[0].image}{"\n"}{end}'` — one line, run after every deploy.
2. **Delete legacy code, do not just disable it.** Power Peg lived in the codebase for years after it was retired, protected only by a configuration flag. When the flag was repurposed, nobody audited what the old code behind it did.
3. **Position-rate circuit breakers are mandatory for any code that moves money.** An automated kill switch that fired when order volume exceeded 10,000/minute would have stopped the incident at 9:31 am instead of 10:16 am.
4. **Rollback that operates on inconsistent system state makes things worse.** Always verify the current state before acting. The rollback decision must start with "what version is each component running?" — not "what do we think should be running?"

---

## Rollback-ability as a Design Property

The key insight that separates mature delivery teams from everyone else is this: **rollback-ability is a design property, not an emergency procedure.** You design rollback-ability in during development, not during an incident.

The operational checklist for every deploy that touches persistent state:

```
[ ] Schema change?
    → Use expand/contract — never break old binary compatibility
    → Verify previous binary handles expanded schema (no NOT NULL without default)

[ ] Message format change?
    → Version the schema — consumers must handle version n-1 messages
    → Register the schema in your schema registry before deploy

[ ] New credentials?
    → Keep old credentials valid in secret manager until rollback window closes
    → Rollback window = time between deploy and schema cleanup deploy (days, not hours)

[ ] New code behind a flag?
    → Default the flag to OFF — ship dark
    → Test the flag flip OFF in staging before shipping

[ ] Deployment tooling confirms uniform version across all instances?
    → After deploy, query all pods' version endpoint
    → Fail the deploy pipeline if any pod is not on the new version after rollout timeout

[ ] AnalysisRun thresholds defined and tested in staging?
    → Run the AnalysisRun template in staging with known-bad thresholds to verify it triggers rollback
    → Confirm the Prometheus queries return reasonable values, not null

[ ] PDB allows rollback velocity?
    → Set minAvailable low enough that rollback can complete in under 5 minutes
    → A 90% PDB on a 10-pod deployment means 1 pod at a time — 10 minutes for a rollback

[ ] Previous container image retained in registry for ≥7 days?
    → Check your image pruning policy
    → The image from the last known-good deploy must still be pullable
```

This checklist converts rollback from a reactive emergency procedure into a boring, routine operation. The goal is for every engineer on the team to execute a rollback without hesitation — because they have designed every component to make it safe.

---

## Before-After: Rollback-able vs Un-Rollback-able Deploy

![Rollback-able expand/contract vs un-rollback-able direct migration](/imgs/blogs/rollbacks-and-recovering-a-bad-deploy-7.png)

The before/after contrast above captures the single most important architectural difference between a team that can always roll back and one that cannot: whether the database migration preserves backward compatibility with the previous binary.

Teams that habitually use expand/contract never face the schema-driven un-rollback-able scenario. The pattern requires more migration steps and carries slightly more schema complexity during the overlap window, but the payoff is enormous: the rollback window on any deploy extends as long as needed — days, weeks — because the old binary continues to work against the expanded schema indefinitely. The cleanup migration (dropping the old column) is a low-risk operation performed in calm daylight with no incident pressure.

The pattern also has a secondary benefit: it forces you to think about backward compatibility at migration authoring time, before the deploy, rather than under incident pressure at 2am. Encoding that constraint in a pattern — "we always expand before contract, no exceptions" — removes the need to evaluate it case by case.

---

## Integrating With Progressive Delivery and GitOps

Rollback does not live in isolation. It is the failure path of progressive delivery and the safety net of GitOps. The [CI/CD pipeline overview](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) positions rollback as a first-class part of the delivery pipeline, not an afterthought. The [Progressive Delivery and GitOps integration](/blog/software-development/ci-cd/progressive-delivery-meets-gitops) explains how Argo CD and Argo Rollouts together make automated rollback an expression of declarative Git state.

In a GitOps model, a rollback is simply a Git operation: revert the commit that changed the image tag, push to the repository, and let Argo CD reconcile the cluster state back to the reverted manifest. This is elegant because the rollback is auditable (it appears in the Git commit history), reproducible (anyone with Git access can see exactly what was reverted and why), and triggers the same delivery pipeline as a forward deploy — smoke tests, analysis gates, and notifications all fire.

The downside is speed. A Git revert plus CI/CD pipeline execution typically takes 5–15 minutes. For a pure binary rollback where `kubectl rollout undo` takes 2 minutes, the GitOps path is slower. Many teams adopt a hybrid: use `kubectl rollout undo` or `helm rollback` for immediate incident mitigation, then create the Git revert commit as a follow-up to keep the repository state consistent. The key is that the imperative rollback does not become the permanent state of the cluster — the GitOps commit closes the loop and ensures that the cluster state is again fully described by a Git commit.

---

## Key Takeaways

**The myth of "always roll forward" and "always roll back."** Neither is universally correct. The right answer depends on data mutability, time-to-fix, blast radius, and revision availability. Codify the decision as a deterministic checklist, not a judgment call.

**Automated rollback on SLO breach is the highest-MTTR-reduction investment available.** Argo Rollouts with AnalysisRun gates compresses MTTR from 90 minutes to under 5 minutes — but only if you configure the thresholds, test the analysis templates, and verify the Prometheus queries before the incident.

**Four classes of un-rollback-able deploys exist.** Incompatible schema already migrated (especially NOT NULL without DEFAULT), events already published to Kafka in a new format, user-visible side effects already committed (payments, emails, orders), and credentials rotated with old versions invalidated. All four are preventable by design: expand/contract migrations, versioned message schemas, two-phase credential rotation, and deferred side effects. Build these patterns in at development time.

**Feature flags are the fastest rollback path.** Sub-30-second recovery, zero infrastructure change, available even when the CI/CD pipeline is broken. Wire flags into your critical paths before you need them.

**The recovery runbook must be deterministic.** A 2am engineer following a decision tree makes better decisions than one improvising under adrenaline. Codify the rollback decision as a sequence of yes/no checks with a concrete command at every leaf. Include the regulated-environment approval gate so it does not surprise you during an incident.

**Knight Capital is the canonical case study.** Inconsistent deployment state + no automated kill switch + rollback without state verification = \$440M loss in 45 minutes. Every failure in that incident is preventable with tools that exist today. The specific mechanism — trying to roll back through inconsistent multi-version state without first mapping which server was running what — is the blueprint for what your deployment controller must never permit.

**Rollback-ability is a design property.** Expand/contract migrations, schema versioning, two-phase credential rotation, feature flags defaulting to off, PDBs sized for rollback velocity — these decisions made at development time determine whether a 2am rollback takes 3 minutes or 35 minutes.

---

## Further Reading

- [From Commit to Production: The CI/CD Pipeline Overview](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) — the full delivery pipeline context that positions rollback as a first-class stage
- [Progressive Delivery Meets GitOps](/blog/software-development/ci-cd/progressive-delivery-meets-gitops) — how Argo CD and Argo Rollouts combine for declarative progressive delivery with automated rollback
- [Database Migrations in the Delivery Pipeline](/blog/software-development/ci-cd/database-migrations-in-the-delivery-pipeline) — the expand/contract pattern and zero-downtime migrations in full depth
- [Feature Flags: Decoupling Deploy from Release](/blog/software-development/ci-cd/feature-flags-decoupling-deploy-from-release) — flag architecture, rollout targeting, and the zero-deploy rollback path in a multi-service environment
- [Mitigate First, Diagnose Later](/blog/software-development/site-reliability-engineering/mitigate-first-diagnose-later) — the SRE argument for speed-of-mitigation over correctness-of-diagnosis that underlies every rollback runbook
- Argo Rollouts documentation — the authoritative reference for AnalysisRun configuration, metric providers, and rollback hooks
- SEC Administrative Proceeding File No. 3-15570 (Knight Capital Group, 2014) — the official regulatory report on the 2012 incident, detailed and publicly available
- "Site Reliability Engineering" (Beyer et al., Google) — Chapters 12–14 cover incident management, postmortems, and the error budget model that makes MTTR a first-class metric
- DORA (DevOps Research and Assessment) State of DevOps Report — the empirical data showing that elite teams have 168x faster MTTR than low performers, driven primarily by deployment automation and automated rollback capability
