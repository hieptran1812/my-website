---
title: "Runners, Caching, and the CI Cost Problem"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Audit your CI pipeline like an engineer, not an accountant — learn exactly where build minutes go, how caching and autoscaling eliminate 70% of them, and how to take a $45k/month bill down to $12k."
tags:
  [
    "ci-cd",
    "devops",
    "ci-runners",
    "caching",
    "github-actions",
    "cost-optimization",
    "build-performance",
    "docker",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/runners-caching-and-the-ci-cost-problem-1.png"
---

The message landed in Slack on a Tuesday morning. The CTO had just reviewed the monthly cloud bill. Subject line: "Can someone explain the \$45,782 GitHub Actions charge?"

The engineering leads gathered in a call thirty minutes later. Nobody had a clean answer. The platform team lead pulled up the usage dashboard and started reading numbers aloud: 5.7 million runner-minutes in the past month, across 47 repositories, touching 212 distinct workflows. The CTO asked the obvious question: "Where do the minutes actually go?"

That question launched a two-week audit. What the team found was predictable in hindsight, but painful in practice. Sixty percent of their CI spend was being consumed by two problems: cold dependency installs on every run, and a Docker build process that reconstructed the same image layers from scratch on every push. The remaining forty percent was a combination of a serial test suite that could have been parallelized months ago and dozens of workflows triggered redundantly on branches that had not changed relevant code.

The fix took another three weeks. At the end of it, the monthly bill was \$11,940. Same engineering organization, same release velocity, same quality bar. The minutes just stopped going to waste.

![Anatomy of a slow CI build: where the minutes actually go](/imgs/blogs/runners-caching-and-the-ci-cost-problem-1.png)

This post is the engineering behind that audit. We will walk through the anatomy of a slow build, the full landscape of runner options and their cost profiles, every caching strategy worth implementing, how to shard tests intelligently, and the cost arithmetic that lets you predict your bill before you get surprised by it. By the end you will have the tools to do the same audit on your own pipeline.

---

## Where CI Minutes Actually Go: Anatomy of a Slow Build

Before we talk solutions, we need to agree on the problem. CI cost is not distributed evenly across a pipeline. It concentrates in a small number of well-understood places. If you have not measured where your minutes go, you are optimizing blind.

**Cold dependency installation** is almost always the largest single line item. A Node.js project with a non-trivial dependency tree takes eight to fourteen minutes to run `npm install` from scratch. A Python data-science project with numpy, pandas, torch, and their transitive dependencies can take longer. If you run one hundred pipelines per day — a moderate number for a team of twenty engineers — that is 800 to 1,400 runner-minutes per day, just on dependency installation. At GitHub-hosted standard runner rates (\$0.008/min), that is \$192 to \$336 per day, or \$5,760 to \$10,080 per month, on a task that yields zero value unless something in `package.json` actually changed.

**Docker layer rebuilds** are the second major contributor. Every Dockerfile has a layer structure. If you are not exploiting that structure — building with `--no-cache`, or not using a registry cache backend — you are paying to run `apt-get install`, `pip install`, and `npm install` inside your container on every single pipeline run. A production Dockerfile for a Node.js API server might spend 15 minutes on dependency installation. A Python ML training container might spend 20 minutes pulling and installing CUDA-adjacent dependencies. These layers change rarely. Rebuilding them constantly is pure waste.

**Serial test execution** is the third lever. Test suites grow organically. Nobody deliberately designs a 38-minute test suite — it accumulates test by test over years. Most teams run their test suite as a single job on a single runner because that is what the example workflows show. The actual test runtime is parallelizable: if you have 1,200 tests and each takes two seconds, that is 40 minutes on one runner, or five minutes on eight runners. The compute cost is identical. The wall-clock time is eight times better.

**Redundant pipeline triggers** are the fourth and most insidious contributor. Monorepos with a naive workflow structure trigger every pipeline on every push, regardless of which code changed. An engineer pushes a documentation fix and re-runs the full backend test suite, the full frontend test suite, and the full integration test suite. None of those runs can produce a different outcome than the previous run. They exist purely because nobody restricted trigger scope. The [monorepo pipeline scaling post](/blog/software-development/ci-cd/monorepo-vs-polyrepo-and-scaling-the-pipeline) covers affected-target builds in detail — but the cost impact is real even in simpler multi-repository setups where a library change triggers downstream service rebuilds that could be skipped with proper change detection.

Understanding which of these four problems is your primary cost driver is the single most important thing you can do before touching any configuration. Measure first. The next section gives you the arithmetic.

---

## The Cost Arithmetic: Computing Your Bill Before It Arrives

CI cost has a simple formula at its core. The complexity is in correctly estimating each input.

```
monthly_cost = minutes_per_run × runs_per_day × days_per_month × cost_per_minute
```

For GitHub-hosted runners in 2024:
- Linux 2-core standard: \$0.008/min
- Linux 4-core: \$0.016/min
- Linux 8-core: \$0.032/min
- Linux 16-core: \$0.064/min
- Windows 2-core: \$0.016/min (2× Linux)
- macOS M1: \$0.08/min (10× Linux)

The hidden multiplier is concurrency. When you shard tests across 8 runners, you consume 8× the minutes in 1/8 the wall-clock time. Your bill stays the same; your developers wait less. This is important: parallelism does not save minutes, it saves time. Caching saves minutes (and therefore money).

#### Worked example: Computing the baseline bill

A mid-sized engineering org has these characteristics:
- 8 repositories, each running full CI on every push
- Average pipeline: 10 min dep install + 18 min Docker build + 12 min tests = 40 min total
- Average 80 pushes per day across all repos
- Using GitHub-hosted standard 2-core Linux runners

```
minutes_per_day = 40 min/run × 80 runs/day = 3,200 min/day
minutes_per_month = 3,200 × 30 = 96,000 min/month
cost_per_month = 96,000 × $0.008 = $768/month
```

That looks manageable — until scale hits. Fast-forward 18 months: the team has grown, repositories have multiplied, test suites have grown, and Docker images have gotten heavier.

- 22 repositories
- Average pipeline: 12 min dep install + 22 min Docker build + 18 min tests = 52 min total
- Average 270 pushes per day

```
minutes_per_day = 52 × 270 = 14,040 min/day
minutes_per_month = 14,040 × 30 = 421,200 min/month
cost_per_month = 421,200 × $0.008 = $3,370/month
```

Still not \$45k. The jump to five figures happens when you add matrix builds (multiplying runs by 3–5×), mandatory integration tests triggered on every PR rather than just main-branch merges, and a caching setup that stopped working six months ago because someone rotated a secret and forgot to update the cache key.

The four levers that reduce the bill are: runner type, cache hit rate, parallelism strategy, and job elimination. We will work through each.

### Self-Hosted Break-Even Analysis

Before investing in self-hosted runner infrastructure, you need to know whether the scale justifies it. The core calculation compares the fully-loaded cost of self-hosted against the marginal cost of managed runners.

The managed runner cost is simple: monthly minutes multiplied by \$0.008/min. The self-hosted cost has two components: compute cost and operational overhead.

**Compute cost** for self-hosted runners on AWS EC2 in us-east-1 (approximate on-demand pricing):
- m5.large (2 vCPU, 8 GB): \$0.096/hour = \$0.0016/min
- m5.xlarge (4 vCPU, 16 GB): \$0.192/hour = \$0.0032/min
- c5.2xlarge (8 vCPU, 16 GB): \$0.34/hour = \$0.0057/min

Spot instances reduce these costs by 60–80%, bringing a self-hosted 2 vCPU runner to roughly \$0.0003–\$0.0006/min. However, spot interruptions introduce job failures, so you need retries and a slightly more complex setup.

**Operational overhead** is the harder number to estimate. Running ARC on Kubernetes, maintaining runner images, handling security patches, debugging scaling issues — this is real engineering time. A conservative estimate for a small organization with no existing Kubernetes expertise is 10–20 engineer-hours per month once the system is set up. At \$100/engineer-hour fully loaded, that is \$1,000–\$2,000/month in hidden cost. An organization with an existing platform team that already runs Kubernetes might spend 2–5 hours/month maintaining CI runners — essentially free given existing salary commitments.

**The break-even math:**

```
managed_cost = M × $0.008
self_hosted_cost = M × $0.0016 + operational_overhead

break_even when:
M × $0.008 = M × $0.0016 + $1,500  (assuming $1,500/mo operational overhead)
M × ($0.008 - $0.0016) = $1,500
M × $0.0064 = $1,500
M = 234,375 minutes/month
```

At 234,375 minutes/month, managed and self-hosted cost the same, assuming \$1,500/month operational overhead. Below this threshold, managed runners cost less once you factor in engineering time. Above it, self-hosted wins — and the savings compound as volume grows.

Put plainly: the break-even is around 230,000–300,000 minutes per month for most organizations, depending on how cheaply they can operate the infrastructure. If you are spending \$1,900/month on GitHub Actions (about 237,000 minutes), you are right at the threshold. If you are spending \$5,000/month (about 625,000 minutes), self-hosted saves roughly \$3,500/month after overhead — worth the investment.

---

## GitHub-Hosted vs Self-Hosted Runners: The Real Trade-off

The managed-vs-self-hosted decision is not primarily about cost per minute. It is about where you want to spend engineering time.

**GitHub-hosted runners** give you zero operational burden. You write a workflow, specify `runs-on: ubuntu-latest`, and GitHub provisions a fresh virtual machine, runs your job, and terminates it. You pay \$0.008/min and have nothing to maintain. For teams under roughly 50 concurrent jobs, this is almost certainly the right answer. The operational overhead of running your own runner fleet — security patching, networking, ephemeral state management, secret handling, node pool sizing — is not free, and it needs an owner.

**Self-hosted runners** start making economic sense when your monthly GitHub Actions bill crosses somewhere between \$2,000 and \$5,000 per month. At that point, the cost differential between \$0.008/min managed and roughly \$0.002/min self-hosted (including the EC2 or Kubernetes compute) starts to compound into meaningful savings. The break-even is roughly 250,000–350,000 minutes per month, assuming your infra team can absorb the maintenance cost.

The operational overhead is real. You need to:
- Keep runner images patched and up to date
- Handle secrets injection securely without leaking them across jobs from different PRs
- Ensure ephemeral runners (a new VM per job) so one job cannot see another job's filesystem
- Manage network egress costs for pulling Docker images and packages
- Handle scaling: too few runners means queue bottlenecks; too many means idle waste

The ephemeral runner requirement is non-negotiable for public repositories and strongly recommended for private ones. Persistent runners accumulate state across jobs — environment variables, filesystem artifacts, cached credentials — and that state can leak between workflows from different contributors. Every serious self-hosted runner deployment uses ephemeral runners: the runner registers, picks up exactly one job, runs it, and then terminates. The scaling infrastructure is responsible for keeping a warm pool of ready runners that matches current queue depth.

![CI bill before and after optimization: $45k to $12k](/imgs/blogs/runners-caching-and-the-ci-cost-problem-2.png)

---

## Self-Hosted Runner Options in Depth

There are four serious options for self-hosted runners in the GitHub Actions ecosystem. Each has a different operational profile.

### Actions Runner Controller on Kubernetes

ARC (Actions Runner Controller) is the Kubernetes-native solution for self-hosted GitHub Actions runners. It runs as a controller deployment in your cluster, watches the GitHub API for queued jobs, and creates runner pods on demand. When a job completes, the pod terminates.

ARC supports two scaling modes:
- **RunnerDeployment**: maintains a fixed pool of runners, scaled by HPA or KEDA
- **RunnerScaleSet**: the newer architecture, uses GitHub's built-in scale-set API for tighter queue-depth-based scaling

The scale-set approach is recommended for new deployments. It requires fewer permissions and produces more predictable scaling behavior because the queue depth signal comes directly from the GitHub API rather than being inferred from metrics.

ARC is the best choice if your organization already runs Kubernetes and has engineers comfortable operating it. The runners inherit your cluster's existing networking, secret management (via ExternalSecrets or Vault), and RBAC. You pay for the underlying node pool compute at cloud rates — typically \$0.0016–\$0.003/vCPU-min for on-demand compute, or 60–80% less for spot.

**RunnerDeployment with HorizontalRunnerAutoscaler** — the older but still widely deployed pattern:

```yaml
apiVersion: actions.summerwind.dev/v1alpha1
kind: RunnerDeployment
metadata:
  name: my-runner-deployment
  namespace: actions-runner-system
spec:
  replicas: 2
  template:
    spec:
      repository: your-org/your-repo
      # Use org-level runners for cross-repo sharing:
      # organization: your-org
      labels:
        - self-hosted
        - linux
        - x64
      image: summerwind/actions-runner:latest
      resources:
        requests:
          cpu: "2"
          memory: "4Gi"
        limits:
          cpu: "4"
          memory: "8Gi"
      # Ephemeral: pod terminates after one job
      ephemeral: true
      nodeSelector:
        node.kubernetes.io/instance-type: "m5.xlarge"
      tolerations:
        - key: "ci-runners"
          operator: "Exists"
          effect: "NoSchedule"
```

```yaml
apiVersion: actions.summerwind.dev/v1alpha1
kind: HorizontalRunnerAutoscaler
metadata:
  name: my-runner-autoscaler
  namespace: actions-runner-system
spec:
  scaleTargetRef:
    kind: RunnerDeployment
    name: my-runner-deployment
  minReplicas: 2
  maxReplicas: 50
  metrics:
    - type: TotalNumberOfQueuedAndInProgressWorkflowRuns
      repositoryNames:
        - your-org/your-repo
  scaleDownDelaySecondsAfterScaleOut: 120
  scaleUpTriggers:
    - githubEvent:
        workflowJob: {}
      amount: 1
      duration: "5m"
```

The `scaleUpTriggers` block tells ARC to scale up by 1 replica immediately when a `workflow_job` event arrives, and to maintain that extra replica for 5 minutes before allowing scale-down. This prevents thrashing when jobs arrive in rapid succession. The `scaleDownDelaySecondsAfterScaleOut: 120` adds a 2-minute cooldown after any scale-out event before scale-in can occur.

The `minReplicas: 2` keeps two warm runners ready at all times — avoiding the cold-start wait when a developer pushes code. For organizations with a predictable morning CI surge, setting a scheduled minimum (via KEDA CronScaler) to 8 replicas from 8 AM to 12 PM and back to 2 replicas overnight effectively eliminates queue waits during the peak window.

### AWS CodeBuild

AWS CodeBuild runners are a serverless option that eliminates the runner infrastructure problem entirely. You configure a CodeBuild project as the runner, and GitHub Actions triggers it via the CodeBuild source provider. AWS provisions compute on demand, runs your job, and deallocates it. You pay per build minute at roughly \$0.005/min for a 2 vCPU instance (varies by region and instance type), with no idle cost.

CodeBuild works well for organizations already deep in AWS who want serverless runner economics without Kubernetes operational overhead. The cold-start time is 30–90 seconds per job, which adds to wall-clock time but not to billed minutes (you only pay for the build duration, not provisioning).

### Buildkite Agents

Buildkite uses a different model: a SaaS CI orchestration platform where you run your own agent processes on whatever compute you want, and Buildkite provides the orchestration, scheduling, and UI. You pay Buildkite a platform fee (roughly \$15–\$30/user/month) and pay separately for your compute.

The advantage is operational simplicity relative to running ARC: Buildkite handles job distribution, retry logic, and pipeline visualization. The disadvantage is the additional platform cost and the fact that you are now dependent on two external services (Buildkite plus your compute provider) instead of one.

### EC2 Auto-Scaling Fleet (Spot Instances)

The cost-optimal but most operationally demanding option. You run GitHub Actions runners on EC2 spot instances, using an auto-scaling group sized by a custom Lambda function that polls the GitHub API for queue depth. At 70–90% spot discounts over on-demand, this can bring compute cost below \$0.001/min for a 2 vCPU instance.

The operational challenges are significant:
- Spot interruptions require checkpointing or accepting job failures and retries
- You need to build or adopt tooling for the GitHub registration/deregistration lifecycle
- Networking and security configuration is entirely your responsibility
- The scaling logic needs careful tuning to avoid both queuing delays and excess idle capacity

Several open-source tools exist for this: `philips-labs/terraform-aws-github-runner` is the most widely used and handles most of the lifecycle complexity. It is a reasonable starting point if you have Terraform expertise and AWS infrastructure already in place.

![Runner type comparison matrix: cost, setup, burst, maintenance](/imgs/blogs/runners-caching-and-the-ci-cost-problem-3.png)

---

## Autoscaling Runners: Matching Capacity to Demand

Autoscaling is the mechanism that prevents you from paying for idle runners during quiet periods and from queueing jobs during CI traffic spikes. Getting it right is the difference between a cost-efficient self-hosted fleet and one that is either wasteful or slow.

The fundamental control loop is straightforward. The autoscaler watches queue depth — the number of jobs waiting for a runner. When queue depth exceeds a threshold (typically 0 or 1), it provisions new runners. When runners become idle (no job picked up within a timeout, typically 2–5 minutes), they are terminated. The challenge is the provisioning latency: a new EC2 instance or Kubernetes pod takes 30–120 seconds to become ready, and that latency is directly visible to developers as queue wait time.

### Predictive vs Reactive Scaling

Purely reactive scaling means developers always wait at least one provisioning cycle before their job starts running. For teams with predictable CI patterns — heavy pushes in the morning, quiet overnight — a predictive warm pool can eliminate this wait. You maintain a floor of pre-warmed runners sized to handle typical morning traffic, and reactive scaling handles the overflow.

The warm pool has a cost: you pay for idle runners during quiet periods. The right warm pool size minimizes the sum of (idle runner cost) + (developer wait time cost). For most teams, a floor of 2–5 runners with reactive scaling above that produces good results.

### ARC RunnerScaleSet Configuration

```yaml
# values.yaml for ARC RunnerScaleSet Helm chart
githubConfigUrl: "https://github.com/your-org"
githubConfigSecret: "arc-github-token"

minRunners: 2
maxRunners: 50

template:
  spec:
    containers:
      - name: runner
        image: ghcr.io/actions/actions-runner:latest
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
    nodeSelector:
      node.kubernetes.io/instance-type: "m5.xlarge"
    tolerations:
      - key: "ci-runners"
        operator: "Exists"
        effect: "NoSchedule"
```

The `minRunners: 2` keeps two warm runners ready at all times. `maxRunners: 50` caps the scale-out at 50 concurrent runners — you tune this based on your maximum reasonable concurrent job count, which you can read from historical usage data. The `nodeSelector` ensures runners land on nodes in a dedicated node pool (you do not want CI runners competing with production workloads for CPU).

### Queue-Depth Scaling with KEDA

For more sophisticated scaling, KEDA (Kubernetes Event-Driven Autoscaler) can drive runner scale-out based on GitHub's queue depth API directly:

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: github-runner-scaler
spec:
  scaleTargetRef:
    name: runner-deployment
  minReplicaCount: 2
  maxReplicaCount: 50
  triggers:
    - type: github-runner
      metadata:
        githubAPIURL: "https://api.github.com"
        owner: "your-org"
        runnerScope: "org"
        labels: "self-hosted,linux,x64"
        targetWorkflowQueueLength: "1"
```

`targetWorkflowQueueLength: "1"` means KEDA will scale the runner count such that at most one job is waiting in queue at any given time. This is an aggressive setting that prioritizes developer experience over cost. Setting it to `"3"` or `"5"` allows more queuing but reduces the number of idle runners you maintain.

### The Warmup Time Problem

The most common complaint about ARC-based autoscaling is the warmup latency. A new runner pod on Kubernetes takes time to schedule, pull the runner container image, register with GitHub, and accept its first job. The timeline typically looks like:

```
t=0s   Job queued in GitHub Actions
t=2s   KEDA detects queue depth increase, triggers scale-out
t=8s   Kubernetes scheduler places pod
t=35s  Runner container image pulled (cached on node pool) or 90s (cold pull)
t=40s  Runner registers with GitHub API
t=42s  Job dispatched to new runner
t=43s  First workflow step begins executing
```

For a 42-second warmup, a developer who pushes code and switches to the Actions tab immediately will see "Queued" for up to a minute. This is acceptable. A 3-minute warmup (common when the node pool needs to scale out to fit the runner pod) is not acceptable for developer experience.

The solutions, in order of effectiveness:
1. Pre-warm a floor of runners with `minRunners: 2–5` so most jobs hit an already-registered runner immediately
2. Use a dedicated node pool with the runner container image pre-pulled (node prep userdata installs `docker pull ghcr.io/actions/actions-runner:latest`)
3. Set resource requests conservatively so runners fit on existing nodes without triggering node scale-out
4. For critical workflows (main-branch, production deploys), use a larger warm pool floor specifically for those labels

![Autoscaler control loop: queue-depth driven scale-out and scale-in](/imgs/blogs/runners-caching-and-the-ci-cost-problem-5.png)

---

## Caching Strategies That Actually Move the Needle

Caching is where most teams leave the most money on the table. There are four distinct cache layers in a modern CI pipeline, each operating at different granularity and providing different cost-reduction leverage.

![The four-layer CI caching stack](/imgs/blogs/runners-caching-and-the-ci-cost-problem-4.png)

### L1: Local Runner Disk Cache

On self-hosted persistent runners (not the ephemeral kind — a tradeoff worth noting), the local disk retains state across jobs. If your runners are ephemeral, L1 does not exist. For the purposes of cost optimization, ephemeral runners are almost always the right choice for security reasons, so we focus on L2–L4.

### L2: GitHub Actions Cache (actions/cache) — Deep Dive

`actions/cache` is the workhorse dependency cache for GitHub Actions workflows. It uploads a tarball of the specified paths to GitHub's cache storage (backed by Azure Blob Storage) at the end of a job with a cache miss, and downloads it at the start of subsequent jobs with a matching cache key.

The cache hit rate is the critical metric. A cache that hits 90% of the time saves 9× more minutes than one that hits 10% of the time. The key design is everything.

**How the key and restore-keys chain works**

The `key` field is an exact-match lookup. If no entry exists for that key, `actions/cache` tries the `restore-keys` list in order — each entry is a prefix match against all existing cache keys. The first matching prefix wins. This means you can implement a fallback chain that always finds the closest available cache even when the exact key misses.

Put plainly, the restore-keys chain works like this: the action tries the most specific key first, and if that misses, it tries progressively less specific prefixes until it either finds a partial match or finds nothing. A partial match is still a cache hit — the restore action downloads the cached archive, your package manager runs and detects what is already present versus what needs to be downloaded, and only the delta is fetched from the internet.

**Cache eviction policy**: GitHub Actions caches are subject to a 7-day LRU eviction policy. A cache entry that has not been accessed in 7 days is deleted. Additionally, the total cache size per repository is capped at 10 GB. When the 10 GB limit is reached, the oldest entries are evicted to make room for new ones. These two policies interact in ways that can silently destroy your caching setup. If your feature branches each have their own cache entries and 40 engineers are actively branching, you may fill the 10 GB limit within days, evicting the main-branch cache that all other workflows depend on.

**Practical counter-measure**: monitor cache size via the Actions UI (Settings > Actions > Caches) or the REST API, and keep per-branch cache entries small by scoping them carefully. The main branch cache — the one based on the lockfile hash — should be the most frequently accessed and therefore the last to be evicted.

**Node.js dependency caching:**

```yaml
- name: Cache node modules
  uses: actions/cache@v4
  with:
    path: |
      ~/.npm
      node_modules
    key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
    restore-keys: |
      ${{ runner.os }}-node-

- name: Install dependencies
  run: npm ci
```

The key `runner.os-node-{hash(package-lock.json)}` means: this cache is valid as long as `package-lock.json` has not changed. When it does change (new dependency added), the cache misses, `npm ci` runs fresh, and the new cache is saved. The `restore-keys` fallback (`runner.os-node-`) allows a partial cache hit when the exact key misses — npm will still need to download new packages, but the vast majority of unchanged packages are already present.

**Python/pip caching:**

```yaml
- name: Cache pip
  uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-

- name: Install Python deps
  run: pip install -r requirements.txt
```

**Maven/Gradle caching:**

```yaml
# Maven
- name: Cache Maven repository
  uses: actions/cache@v4
  with:
    path: ~/.m2/repository
    key: ${{ runner.os }}-maven-${{ hashFiles('**/pom.xml') }}
    restore-keys: |
      ${{ runner.os }}-maven-

# Gradle
- name: Cache Gradle packages
  uses: actions/cache@v4
  with:
    path: |
      ~/.gradle/caches
      ~/.gradle/wrapper
    key: ${{ runner.os }}-gradle-${{ hashFiles('**/*.gradle*', '**/gradle-wrapper.properties') }}
    restore-keys: |
      ${{ runner.os }}-gradle-
```

**Cache size awareness**: GitHub Actions cache has a 10 GB limit per repository. The cache eviction policy is LRU (least recently used). If you cache large build artifacts alongside dependency caches, you risk evicting the dependency cache. Monitor cache size in the Actions UI and be selective about what you cache.

**Advanced pattern — cascading restore keys across branches**:

When working with feature branches, you want CI on a feature branch to benefit from the main-branch cache as a starting point, even when the feature branch has no cache of its own:

```yaml
- name: Cache pip with branch fallback
  uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ github.ref_name }}-${{ hashFiles('requirements.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-${{ github.ref_name }}-
      ${{ runner.os }}-pip-main-
      ${{ runner.os }}-pip-
```

This chain tries: exact match for the branch+lockfile, then the branch without lockfile constraint, then the main-branch cache, then any pip cache at all. The first push from a feature branch will restore from the main-branch cache — significantly faster than a cold install.

### L3: Docker Layer Cache (BuildKit) — Deep Dive

Docker layer caching is the highest-leverage optimization for teams building container images. The mechanics are straightforward: Docker builds images layer by layer, and each layer is identified by a hash of its content. If a layer's hash matches a cached layer, Docker skips rebuilding it.

The problem with standard Docker builds in CI is that the layer cache is stored on the runner's local disk. On ephemeral runners, that cache is lost when the runner terminates after each job. BuildKit's registry cache mode solves this by pushing intermediate layers to a container registry (your existing ECR, GCR, or Docker Hub registry) and pulling them on subsequent builds.

**GitHub Actions workflow with BuildKit registry cache:**

```yaml
- name: Set up Docker Buildx
  uses: docker/setup-buildx-action@v3

- name: Login to container registry
  uses: docker/login-action@v3
  with:
    registry: ghcr.io
    username: ${{ github.actor }}
    password: ${{ secrets.GITHUB_TOKEN }}

- name: Build and push image
  uses: docker/build-push-action@v5
  with:
    context: .
    push: true
    tags: ghcr.io/${{ github.repository }}:${{ github.sha }}
    cache-from: type=registry,ref=ghcr.io/${{ github.repository }}:buildcache
    cache-to: type=registry,ref=ghcr.io/${{ github.repository }}:buildcache,mode=max
```

`cache-from` tells BuildKit to pull cached layers from the `:buildcache` tag before building. `cache-to` pushes updated layers back to `:buildcache` after building. `mode=max` caches all intermediate layers, not just the final image layers — this maximizes cache hit rate at the cost of more registry storage.

**Dockerfile layer ordering for maximum cache efficiency:**

```dockerfile
# GOOD: dependency layers first, source code last
FROM node:20-alpine AS base
WORKDIR /app

# Layer 1: OS dependencies (changes rarely)
RUN apk add --no-cache git

# Layer 2: package files (changes when deps change)
COPY package.json package-lock.json ./

# Layer 3: npm install (invalidated only when package-lock changes)
RUN npm ci --only=production

# Layer 4: source code (changes on every commit)
COPY src/ ./src/

# Layer 5: build step (must run when source changes)
RUN npm run build
```

```dockerfile
# BAD: source code copied before npm install
FROM node:20-alpine AS base
WORKDIR /app

# This invalidates every subsequent layer on every commit
COPY . .

RUN npm ci --only=production
RUN npm run build
```

The bad pattern is extremely common in real-world Dockerfiles. A single `COPY . .` before `npm install` means that npm install re-runs on every single code change, even when no dependencies changed. Fix this by splitting the copy into two steps: first copy only the lock file and install dependencies, then copy the application source.

**Multi-stage Dockerfile with BuildKit cache mounts**

BuildKit's `RUN --mount=type=cache` directive allows a build step to access a persistent cache directory that survives across builds without being baked into the image layers. This is fundamentally different from layer caching: `--mount=type=cache` gives you a shared mutable directory that the package manager uses as its local cache, entirely outside the layer graph.

```dockerfile
# syntax=docker/dockerfile:1.4
FROM python:3.11-slim AS builder

# Install system deps (cached layer — rarely changes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first (cached layer until requirements change)
COPY requirements.txt ./

# pip cache mount: /root/.cache/pip persists across builds
# The pip cache is NOT included in the image, just used during the build
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Copy source after dependencies
COPY . .

# --- Final stage: only runtime artifacts ---
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app/src ./src

CMD ["python", "-m", "src.main"]
```

The `RUN --mount=type=cache,target=/root/.cache/pip` line is the key. When BuildKit encounters it, it mounts a persistent cache directory at `/root/.cache/pip` for the duration of the `RUN` step. pip uses this directory as its download cache — on the first run it downloads everything; on subsequent runs it finds packages already cached and only downloads new or changed packages. The cache directory is never written to the image layers, so your final image stays lean.

The equivalent for Node.js and npm:

```dockerfile
# syntax=docker/dockerfile:1.4
FROM node:20-alpine AS deps

WORKDIR /app
COPY package.json package-lock.json ./

# npm cache mount: persistent across builds, not in the image
RUN --mount=type=cache,target=/root/.npm \
    npm ci --cache /root/.npm

FROM node:20-alpine AS builder

WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:20-alpine AS runtime

WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=deps /app/node_modules ./node_modules

EXPOSE 3000
CMD ["node", "dist/index.js"]
```

And for Gradle (Java/Kotlin):

```dockerfile
# syntax=docker/dockerfile:1.4
FROM gradle:8-jdk21 AS build

WORKDIR /app
COPY build.gradle.kts settings.gradle.kts ./
COPY gradle ./gradle

# Gradle cache mount: ~/.gradle persists across builds
RUN --mount=type=cache,target=/root/.gradle \
    gradle dependencies --no-daemon

COPY src ./src

RUN --mount=type=cache,target=/root/.gradle \
    gradle bootJar --no-daemon

FROM eclipse-temurin:21-jre-alpine AS runtime

WORKDIR /app
COPY --from=build /app/build/libs/*.jar app.jar
ENTRYPOINT ["java", "-jar", "app.jar"]
```

Note that `RUN --mount=type=cache` requires BuildKit (`DOCKER_BUILDKIT=1` or `# syntax=docker/dockerfile:1.4` at the top of the Dockerfile) and a BuildKit builder (`docker buildx build` rather than `docker build`). In GitHub Actions, `docker/setup-buildx-action@v3` handles this automatically.

The performance impact is significant. A Python project with 40 packages in `requirements.txt` might take 8 minutes for the pip install on a cold build. With `--mount=type=cache`, subsequent builds that only change source code take under 10 seconds for the pip step, because all packages are already in the cache mount. Combined with registry layer caching, this is the pattern that drove Docker build times from 19 minutes to under 4 minutes in the case study above.

![Docker build layer caching: naive 18-min build vs BuildKit 3-min cached build](/imgs/blogs/runners-caching-and-the-ci-cost-problem-7.png)

### L4: Remote Build Cache (Bazel, Gradle, Turborepo)

For organizations using monorepo build systems — Bazel, Gradle with build cache, or Turborepo for JavaScript monorepos — a remote build cache provides team-wide content-addressable caching. When engineer A builds a package, the output is stored in the remote cache. When engineer B (or a CI runner) builds the same package with the same inputs, it retrieves the cached output instead of rebuilding.

**Turborepo remote cache configuration:**

```json
{
  "pipeline": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": ["dist/**"]
    },
    "test": {
      "dependsOn": ["build"],
      "outputs": []
    }
  },
  "remoteCache": {
    "signature": true
  }
}
```

```yaml
# GitHub Actions with Turborepo remote cache
- name: Run build with Turborepo
  run: npx turbo run build test
  env:
    TURBO_TOKEN: ${{ secrets.TURBO_TOKEN }}
    TURBO_TEAM: ${{ vars.TURBO_TEAM }}
```

With Turborepo's Vercel Remote Cache (or a self-hosted equivalent like `ducktape` or `turborepo-remote-cache`), unchanged packages produce 0-second build times for any engineer or CI runner who hits the cache. For large monorepos, this can eliminate the majority of build time.

---

## Parallel Test Sharding: Parallelism Without Extra Cost

Test parallelization is the only optimization that simultaneously reduces wall-clock time and cost (if you can reduce total runner-minutes, not just wall time). The mechanism is matrix jobs: you split your test suite across N runner jobs, run them concurrently, and collect results.

The naive approach is splitting by file count. A more intelligent approach uses historical timing data to split by expected duration, ensuring each shard takes approximately the same wall-clock time (minimizing the time the slowest shard blocks the pipeline).

**Round-robin file splitting with GitHub Actions matrix:**

```yaml
strategy:
  matrix:
    shard: [1, 2, 3, 4, 5, 6, 7, 8]

steps:
  - name: Run tests (shard ${{ matrix.shard }} of 8)
    run: |
      # Collect all test files
      TEST_FILES=$(find tests/ -name "*.test.js" | sort)
      TOTAL=$(echo "$TEST_FILES" | wc -l)
      SHARD_SIZE=$(( (TOTAL + 7) / 8 ))
      START=$(( (${{ matrix.shard }} - 1) * SHARD_SIZE + 1 ))
      
      SHARD_FILES=$(echo "$TEST_FILES" | sed -n "${START},$((START + SHARD_SIZE - 1))p")
      npx jest $SHARD_FILES --ci
```

**Jest built-in sharding (Jest 29+):**

```yaml
strategy:
  matrix:
    shard: [1, 2, 3, 4, 5, 6, 7, 8]

steps:
  - name: Run tests
    run: npx jest --shard=${{ matrix.shard }}/8 --ci
```

**Vitest sharding** follows the same pattern:

```yaml
strategy:
  matrix:
    shard: [1, 2, 3, 4, 5, 6, 7, 8]

steps:
  - name: Run Vitest shard
    run: npx vitest run --reporter=junit --outputFile=test-results-${{ matrix.shard }}.xml --shard=${{ matrix.shard }}/8
```

**Pytest with timing-based splitting:**

```yaml
strategy:
  matrix:
    group: [1, 2, 3, 4]

steps:
  - name: Run pytest shard
    run: |
      pip install pytest-split
      pytest --splits 4 --group ${{ matrix.group }} \
             --store-durations --durations-path=.test-durations.json
```

`pytest-split` records test duration data to a JSON file and uses it on subsequent runs to assign tests to shards such that each shard's expected duration is equal. The first run uses round-robin splitting; subsequent runs use timing-based splitting. This is significantly more efficient than round-robin when test duration variance is high (some tests take 10 ms, others take 10 seconds).

**Timing-based splitting vs round-robin — the math**

Suppose you have 240 tests with the following distribution: 200 tests averaging 100 ms each, 30 tests averaging 5 seconds each, and 10 tests averaging 30 seconds each. Total runtime: 200×0.1 + 30×5 + 10×30 = 20 + 150 + 300 = 470 seconds.

With round-robin splitting across 4 shards, by chance one shard might receive 3 of the 10 slow tests (90 seconds of slow tests) while another receives only 1 (30 seconds). The shard with 3 slow tests runs for roughly 90 + 37.5 (its share of the medium tests) + 5 (its share of fast tests) = 132 seconds. The pipeline waits for this slowest shard before proceeding. Round-robin wall-clock time: 132 seconds.

With timing-based splitting, `pytest-split` allocates exactly 470/4 = 117.5 seconds of work to each shard. The fastest shard finishes in about 117 seconds; the slowest in about 118 seconds. Pipeline unblocks in 118 seconds. The difference is modest in this example — 11% faster — but in real suites with high variance (a few tests taking several minutes, many tests taking milliseconds), timing-based splitting can be 2–3× faster than round-robin.

**Collecting results from sharded jobs:**

```yaml
test:
  strategy:
    matrix:
      shard: [1, 2, 3, 4, 5, 6, 7, 8]
  steps:
    - run: npx jest --shard=${{ matrix.shard }}/8 --ci --json --outputFile=test-results-${{ matrix.shard }}.json

merge-results:
  needs: test
  steps:
    - uses: actions/upload-artifact@v4
      with:
        name: test-results
        path: test-results-*.json
    
    - name: Merge JUnit reports
      run: npx junit-report-merger merged-results.xml test-results-*.json
```

![Pipeline optimization journey: 38 minutes to 6 minutes](/imgs/blogs/runners-caching-and-the-ci-cost-problem-6.png)

---

## The Runner Selection Decision Tree

Choosing the right runner strategy before committing to infrastructure investment avoids a costly migration six months later.

![Runner selection decision tree](/imgs/blogs/runners-caching-and-the-ci-cost-problem-8.png)

The decision has three questions, in order:

**1. Do you exceed 50 concurrent jobs at peak?** If no, managed runners (GitHub-hosted or Buildkite managed) are almost certainly the right answer. The cost savings from self-hosted do not outweigh the operational overhead for organizations running fewer than 250,000–300,000 minutes per month. If yes, continue.

**2. Is your workload bursty?** If your CI traffic is smooth and predictable — a flat rate of pushes across the workday — a static or lightly-scaled self-hosted fleet works well. If your traffic is spiky (everyone pushes before standup, CI is quiet overnight), you need aggressive autoscaling, and the question becomes whether your team can operate it.

**3. Does your team have Kubernetes expertise?** ARC on Kubernetes is the most operationally efficient self-hosted option for organizations already running k8s. If you do not run Kubernetes, the overhead of learning and operating it for CI runners is not justified. AWS CodeBuild (serverless, low operational burden) or an EC2 auto-scaling fleet with Terraform automation is a better fit.

The matrix of all four options:

| | GitHub-hosted | ARC on k8s | Buildkite | AWS CodeBuild |
|---|---|---|---|---|
| Cost/min | \$0.008 | \$0.002–0.003 | \$0.003–0.004 | \$0.004–0.005 |
| Setup effort | None | High | Medium | Medium |
| Burst speed | Fast | Very fast | Fast | Medium |
| Maintenance | Zero | High | Medium | Low |
| Best for | Small teams | k8s shops | Mixed infra | AWS-native |

---

## The \$45k → \$12k Case Study

Let me walk through the actual audit that produced those numbers. This is a composite of a real engagement, details changed to protect the innocent.

The organization was a Series B SaaS company, 45 engineers, 28 repositories (a mix of backend services, frontend apps, and shared libraries), using GitHub Actions exclusively. Monthly bill: \$45,782 for the month before the audit.

### The Audit

Step one: pull the usage breakdown from the GitHub API.

```bash
# Get per-workflow usage for the past month
gh api /orgs/your-org/actions/cache/usage \
  --paginate \
  --jq '.actions_caches_size_in_bytes'

# Get per-repository minutes
gh api /orgs/your-org/settings/billing/actions \
  --jq '.total_minutes_used, .included_minutes, .total_paid_minutes_used'
```

The API gives you aggregate numbers. For per-workflow breakdown, you need to use the workflow run API with time filtering:

```bash
# Minutes consumed by a specific workflow in the last 30 days
gh api "/repos/your-org/api-service/actions/workflows" \
  --jq '.workflows[] | {name: .name, id: .id}'

gh api "/repos/your-org/api-service/actions/workflows/12345/timing" \
  --jq '.billable'
```

What the audit found, sorted by cost:

| Repository/Workflow | Monthly minutes | Monthly cost | Root cause |
|---|---|---|---|
| api-service / build-and-test | 412,000 | \$3,296 | No dep cache, no Docker layer cache |
| frontend-app / build-and-test | 389,000 | \$3,112 | Same — 14 min npm install × 90 runs/day |
| data-pipeline / integration-tests | 334,000 | \$2,672 | Serial test suite, 52-min runs |
| shared-lib / on-push | 298,000 | \$2,384 | Triggers on every push including docs |
| *18 other repos* | 3,300,000 | \$26,400 | Mixed — mostly cold installs |
| macOS runner jobs | 610,000 | \$4,880 | iOS build — necessary but misconfigured |
| Windows runner jobs | 380,000 | \$3,040 | Unnecessary — Linux equivalents existed |
| **Total** | **5,723,000** | **\$45,782** | |

Three findings dominated the fix list:

**Finding 1**: The top four workflows had zero dependency caching. `npm ci` and `pip install -r requirements.txt` ran fresh on every single run. Fix: add `actions/cache` to each workflow. Cache hit rate after one week: 91% for the top two, 88% for the data pipeline.

**Finding 2**: The Docker builds for twelve backend services had `COPY . .` before `npm install`, destroying layer cache effectiveness. Additionally, no workflow was using `BuildKit --cache-from`. Fix: reorder Dockerfile layers + add BuildKit registry cache. Average Docker build time dropped from 19 minutes to 3.5 minutes per service.

**Finding 3**: Fourteen workflows were using Windows runners for tasks that had no Windows-specific requirements — they had been copy-pasted from a historical workflow that did need Windows. Windows runners cost 2× Linux. Fix: change `runs-on: windows-latest` to `runs-on: ubuntu-latest` for these fourteen workflows.

The data pipeline's serial test suite got sharded to 6 parallel runners, dropping from 52 minutes to 10 minutes.

### The Result

After three weeks of changes (about 40 hours of engineering time across two engineers):

| Category | Before | After | Savings |
|---|---|---|---|
| Dependency install | \$18,200 | \$1,800 | \$16,400 |
| Docker builds | \$14,100 | \$3,200 | \$10,900 |
| Test suite runtime | \$9,300 | \$4,100 | \$5,200 |
| Runner type waste | \$4,180 | \$840 | \$3,340 |
| **Total** | **\$45,782** | **\$9,940** | **\$35,842** |

The final bill came in at \$9,940 — slightly below the \$12k estimate because the Dockerfile layer reordering had more impact than projected. The engineering investment paid back in 1.5 months at the savings rate.

#### Worked example: The minute-by-minute audit and four interventions

This worked example walks through exactly how the cost reduction was calculated before a single line of YAML was changed. This is the pre-implementation estimate that justified the engineering investment.

**Step 1 — Measure the actual per-step durations**

The team pulled step-level timing from the most expensive workflow (api-service / build-and-test, costing \$3,296/month):

```
Checkout code:               18 seconds
Set up Node.js:              22 seconds
Install dependencies (npm):  14 min 03 sec  ← TARGET
Build application:            3 min 11 sec
Run linter:                   1 min 44 sec
Run unit tests:               8 min 22 sec
Build Docker image:          19 min 07 sec  ← TARGET
Push Docker image:            2 min 18 sec
Run integration tests:       12 min 44 sec
Total per run:               ~62 minutes
```

Runs per day: 90 (tracked from the workflow run API). Monthly minutes: 62 × 90 × 30 = 167,400 minutes. Cost at \$0.008/min: \$1,339/month. The table shows \$3,296 because the workflow also ran on PRs (additional 90 runs/day at full duration on a 4-core runner at \$0.016/min).

**Step 2 — Estimate savings for each intervention**

Intervention A — Add `actions/cache` to the npm install step:
- Expected cache hit rate: 88% (based on how frequently package-lock.json changes: roughly once per week across 90 runs/day = 1/90 ≈ 1.1% of runs are genuine updates)
- On a cache hit, `npm ci` is replaced by a ~30-second cache restore
- Expected new step duration: 0.12 × 14 min + 0.88 × 0.5 min = 1.68 + 0.44 = 2.12 min
- Time saved per run: 14 - 2.12 = 11.88 min
- Monthly savings: 11.88 × 90 × 30 × \$0.008 = \$257/month on this one workflow

Intervention B — Reorder Dockerfile + add BuildKit registry cache:
- Current Docker build time: 19 min 07 sec (dominated by `npm install` inside Docker)
- With layer reordering, the install layer only rebuilds on package-lock.json changes
- With registry cache, even a cold layer pull takes 2–3 min instead of 19 min
- Expected post-optimization Docker build time: 3.5 min (cache hit, ~90% of runs) / 9 min (cache miss)
- Expected average: 0.90 × 3.5 + 0.10 × 9.0 = 3.15 + 0.90 = 4.05 min
- Time saved per run: 19.12 - 4.05 = 15.07 min
- Monthly savings: 15.07 × 90 × 30 × \$0.008 = \$326/month on this workflow

Intervention C — Shard the integration tests (12 min 44 sec) across 4 runners:
- Current: 12.73 min × 90 runs/day × 30 days × \$0.008/min = \$275/month
- After sharding to 4 runners: wall-clock drops to ~3.5 min, but runner-minutes stay the same
- Savings: zero in dollar terms, but the pipeline shortens by 9 minutes — that is developer wait time saved, not compute cost
- Note: parallelism saves time, caching saves money. Both matter.

Intervention D — Eliminate Windows runners from non-Windows workflows:
- 14 workflows accidentally on Windows runners
- Average workflow duration: 35 minutes
- Runs/day across 14 workflows: ~120
- Windows cost: 120 × 30 × 35 × \$0.016/min = \$20,160/month
- Linux equivalent cost: 120 × 30 × 35 × \$0.008/min = \$10,080/month
- Savings: \$10,080/month — the single largest intervention

Combined pre-implementation estimate: \$257 + \$326 + \$10,080 + [similar calculations for the other 24 repos] = \$29,000–\$33,000/month savings. The actual outcome (\$35,842 savings) exceeded the estimate because several secondary workflows had the same patterns and were fixed as a side effect of the Dockerfile changes.

---

#### Worked example: From 38 minutes to 6 minutes — per-stage before/after with cache hit math

This example covers the data-pipeline integration test workflow, the one that went from 52 minutes total (38 minutes of test execution) down to 6 minutes of test execution.

**Before — the serial baseline**

```
Stage                        Duration    Cacheable?
----------------------------------------------------
Checkout                     22 sec      No
Set up Python 3.11           18 sec      No
Install pip deps             23 min      Yes — cache miss every run
Install test fixtures        4 min       Yes — cache miss every run
Provision test DB (Docker)   6 min       No (docker pull cached separately)
Run full integration suite   38 min      No
Generate coverage report     2 min       No
Upload artifacts             1 min       No
----------------------------------------------------
Total                        ~74 min
```

Wait — the table above shows 52 minutes, not 74 minutes. The discrepancy is because the workflow had already done partial optimization: docker pull was cached, and the checkout step used a shallow clone. The 52-minute figure was what the team measured after those prior improvements. The raw baseline before any work was 74 minutes.

**The four changes applied:**

Change 1 — Add `actions/cache` for pip:

```yaml
- name: Cache pip dependencies
  uses: actions/cache@v4
  with:
    path: |
      ~/.cache/pip
      .venv
    key: ${{ runner.os }}-pip-${{ hashFiles('requirements-test.txt', 'requirements.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-
```

This brought the pip install step from 23 minutes to ~2 minutes on a cache hit. The requirements files changed roughly once per 12 working days for this project, giving a cache hit rate of approximately 1 - (1/120) = 99.2% per push. In practice, measured hit rate over 2 weeks was 94% (some cache eviction due to the 10 GB limit being shared with other repos).

Savings from this one step: 23 min × 94% = 21.6 minutes saved per run.

Change 2 — Cache test fixtures (a 3.2 GB archive of anonymized production data used for integration tests):

The team added a separate cache step for the fixtures directory, keyed on a hash of the fixture generation script:

```yaml
- name: Cache test fixtures
  id: cache-fixtures
  uses: actions/cache@v4
  with:
    path: tests/fixtures/
    key: test-fixtures-${{ hashFiles('scripts/generate-fixtures.py') }}

- name: Generate test fixtures
  if: steps.cache-fixtures.outputs.cache-hit != 'true'
  run: python scripts/generate-fixtures.py
```

The fixtures script changed about once per month. Cache hit rate: ~97%. The 4-minute fixture generation step effectively disappeared from the pipeline for normal pushes.

Change 3 — Shard tests across 6 runners using `pytest-split`:

The integration test suite had 340 tests. Timing data (from a full run with `--durations=0`) showed the distribution was highly skewed: 10 tests touching external API mocks took 2–4 minutes each; the remaining 330 tests took under 30 seconds combined.

With timing-based splitting into 6 groups:
- Groups 1–4: each received ~2 of the slow API mock tests + ~80 fast tests = ~7 min each
- Groups 5–6: received the remaining fast tests = ~2 min each
- Pipeline waits for the slowest shard: ~7 minutes

Without timing-based splitting (round-robin), one shard would likely receive 5–6 slow tests and take 12–14 minutes, while others finished in 2 minutes. The wall-clock improvement from switching to timing-based splitting: 12 min → 7 min.

Change 4 — Parallelize the Docker provision step with the cache restore step:

```yaml
jobs:
  setup:
    steps:
      - uses: actions/checkout@v4
  
  test:
    needs: setup
    strategy:
      matrix:
        group: [1, 2, 3, 4, 5, 6]
    steps:
      - name: Cache pip
        uses: actions/cache@v4
        # ... (runs in parallel with docker pull on same runner)
      
      - name: Start test database
        run: docker compose up -d postgres redis
        # Docker images already pulled and cached
      
      - name: Wait for services
        run: |
          until docker exec test-postgres pg_isready; do sleep 1; done
```

By running the docker service startup in parallel with the cache restore (both are I/O-bound, not CPU-bound), the effective "setup" time before tests begin dropped from 6 + 2 = 8 minutes to max(6, 2) = 6 minutes.

**After — the optimized state**

```
Stage                        Duration     Notes
----------------------------------------------------
Checkout                     22 sec       Unchanged
Set up Python 3.11           18 sec       Unchanged
Restore pip cache            45 sec       Was 23 min cold install
Restore fixture cache        30 sec       Was 4 min generate
Start test services          6 min        Parallel with cache restores
Run test shard (of 6)        6 min 45 sec Was 38 min serial
Generate coverage report     2 min        Unchanged
Upload artifacts             1 min        Unchanged
----------------------------------------------------
Total                        ~16 min (wall clock, slowest shard)
```

The wall-clock time dropped from 52 minutes to 16 minutes. The per-shard test execution dropped from 38 minutes serial to 6 min 45 sec parallel — matching the headline of "38 min to 6 min" for the test execution phase specifically.

**Cache hit rate math for monthly savings:**

```
Daily runs: 45
Monthly runs: 45 × 30 = 1,350

Before optimization:
  pip install:    23 min × 1,350 × $0.008 = $2,484/month
  fixtures:        4 min × 1,350 × $0.008 = $432/month
  test execution: 38 min × 1,350 × $0.008 = $4,104/month (serial)
  Total test workflow cost:                  $7,020/month

After optimization (6 shards = 6× runner-minutes for test phase):
  pip restore:   0.75 min × 1,350 × $0.008 × 94% hit = $7.60/month
  pip miss:         23 min × 1,350 × $0.008 × 6% miss = $149.04/month
  fixture restore: 0.5 min × 1,350 × $0.008 × 97% hit = $5.24/month
  fixture miss:      4 min × 1,350 × $0.008 × 3% miss = $12.96/month
  test shards:    6.75 min × 1,350 × 6 shards × $0.008 = $4,374/month
  Total:                                                  $4,548/month
  
Monthly savings: $7,020 - $4,548 = $2,472/month on this one workflow
```

The test sharding itself costs the same in runner-minutes as the serial run (38 min × 1 runner ≈ 6.75 min × 6 runners = 38 runner-min both ways). The cost reduction comes entirely from caching, not parallelism. The parallelism benefit is the 32-minute reduction in developer wait time — roughly 45 developers × 32 min/day × 0.25 (fraction of pushes that hit this workflow) = 360 developer-minutes per day recovered.

---

## War Story: The Team Whose CI Bill Doubled in Three Months

This one still comes up in conversations with platform engineers as a cautionary tale about CI cost ownership.

A growth-stage fintech with 30 engineers had been running comfortably at around \$8,000/month on GitHub Actions for about a year. Then, over three months, the bill climbed to \$11,400, \$14,200, and finally \$17,800. The engineering manager noticed during budget reviews. Nobody had gotten an alert. Nobody owned the number.

When a platform engineer finally sat down to investigate, the story unfolded in layers.

**Layer one — the new service with a heavyweight dependency stack:**

A new microservice had been added — a fraud-detection service with a Python dependency stack (numpy, scipy, scikit-learn, tensorflow-cpu). The `pip install` step took 24 minutes. The service had 35 pushes per day across several engineers actively building it. Nobody had added caching because the workflow had been copied from a template that predated the org-wide caching guidelines. Cost breakdown:

```
35 runs/day × 24 min × 30 days × $0.008/min = $2,016/month
```

Two thousand dollars per month from a single workflow with no caching, on a service that had been running for less than two months. The tensorflow-cpu package alone was 480 MB — 12 minutes of the 24-minute install.

**Layer two — Playwright tests with no path filter:**

A frontend engineer had added end-to-end Playwright tests to the main frontend workflow. The tests took 28 minutes on a 2-core runner because browser-based tests are CPU-bound. The tests ran on every push to any branch, including documentation commits, style tweaks, and README updates. There was no `paths:` filter to skip the workflow when only non-source files changed. The fix was a single YAML addition:

```yaml
on:
  push:
    paths:
      - 'src/**'
      - 'public/**'
      - 'package.json'
      - 'package-lock.json'
      - '.github/workflows/playwright.yml'
```

Before this change: 90 runs/day × 28 min × 30 × \$0.008 = \$6,048/month. After the `paths:` filter, roughly 45% of pushes were skipped (documentation and README commits). New cost: 49.5 runs/day × 28 × 30 × \$0.008 = \$3,326/month. Savings: \$2,722/month from six lines of YAML.

**Layer three — cache eviction that nobody noticed:**

The repository-level Actions cache had been filling up slowly as new branches were created and never cleaned up. After 8 weeks, the 10 GB limit was hit. The LRU eviction algorithm started removing the least-recently-accessed entries — which turned out to be the main-branch Node.js dependency cache, because it had been superseded by a newer entry when the frontend dependencies were updated two weeks prior.

The result: `npm ci` had been running fresh on every push to main and every PR for two weeks. The workflow logs showed the install step climbing from 3 minutes (cached) back to 13 minutes (cold). Nobody noticed because the step still completed successfully — it just took longer.

The team added a cache monitoring step to the workflow:

```yaml
- name: Check cache status
  uses: actions/cache@v4
  id: npm-cache
  with:
    path: node_modules
    key: ${{ runner.os }}-npm-${{ hashFiles('package-lock.json') }}

- name: Report cache status
  run: |
    if [ "${{ steps.npm-cache.outputs.cache-hit }}" != "true" ]; then
      echo "::warning::npm cache MISS — cold install running. Check cache eviction."
    fi
```

This surfaces cache misses as workflow warnings in the Actions UI, making unexpected cold installs immediately visible.

**Layer four — a silently broken BuildKit cache:**

A CI secret rotation in month two had broken the Docker registry authentication for the BuildKit cache push step. The step silently failed (the action's default behavior is `continue-on-error` for cache writes), and the job continued. The cache was no longer being updated, so every subsequent run missed the cache. Docker builds that had been taking 4 minutes were now taking 22 minutes.

The fix: add explicit failure handling to the cache push step, and alert on Docker build duration anomalies.

```yaml
- name: Build and push with cache
  uses: docker/build-push-action@v5
  with:
    context: .
    push: true
    cache-from: type=registry,ref=ghcr.io/${{ github.repository }}:buildcache
    cache-to: type=registry,ref=ghcr.io/${{ github.repository }}:buildcache,mode=max

- name: Verify cache was pushed
  run: |
    # Verify the cache tag exists and was updated recently
    CACHE_CREATED=$(docker manifest inspect ghcr.io/${{ github.repository }}:buildcache \
      --verbose 2>/dev/null | jq -r '.[0].OCIManifest.annotations["org.opencontainers.image.created"]')
    echo "BuildKit cache last updated: $CACHE_CREATED"
    
    # Fail if cache is more than 2 hours old (indicates push failure)
    CACHE_AGE=$(( $(date +%s) - $(date -d "$CACHE_CREATED" +%s) ))
    if [ $CACHE_AGE -gt 7200 ]; then
      echo "::error::BuildKit cache appears stale (${CACHE_AGE}s old). Check registry authentication."
    fi
```

**The total fix: 2 days of work**

- Added `actions/cache` to the fraud-detection service workflow (fixes layer one)
- Added `paths:` filter to the Playwright workflow (fixes layer two)
- Deleted stale branch caches via the GitHub API and added cache monitoring (fixes layer three)
- Fixed the Docker registry secret rotation, re-enabled BuildKit cache, added staleness check (fixes layer four)
- Added a CI cost alert: Slack notification via the GitHub billing API when weekly spend exceeds 120% of the 4-week rolling average

The bill the following month: \$8,400.

The lasting lesson is not technical. It is organizational. **CI cost needs an owner.** It needs a metric, a threshold, and an alert. Without those three things, cost grows silently until a CTO asks the uncomfortable question. The four-layer failure above happened not because the engineers were careless — each individual change was reasonable in isolation. It happened because nobody was watching the aggregate cost trend.

---

## When NOT to Optimize

This post has described a significant number of optimizations. None of them should be applied indiscriminately.

**At low pipeline volume, managed runners are the right answer.** If your team runs fewer than 500 pipeline jobs per day, the engineering time required to set up and maintain ARC on Kubernetes, configure BuildKit registry caching, and implement test sharding will not pay back within a reasonable timeframe. The fully loaded cost of an engineer-day is \$800–1,200. If your optimizations save \$50/month, you need 16–24 months to break even on a single day of engineering effort. At low volume, the right optimization is writing better code and adding useful tests, not tuning CI infrastructure.

**Self-hosted runners require infrastructure ownership.** If your organization does not have an operations or platform engineering function, do not run self-hosted runners. The security surface area alone (ephemeral runner configuration, secret isolation, network egress controls) requires someone who understands it. A misconfigured self-hosted runner can leak secrets between workflows, enable repository exfiltration, or create network pivot points into your internal infrastructure.

**Caching adds complexity to debugging.** A stale cache that returns incorrect results is harder to debug than a cache miss. Always provide clear cache invalidation mechanisms, document your cache key strategy, and ensure that engineers know how to force a cache bypass (`workflow_dispatch` with a cache-bust input, or manually clearing the cache via the GitHub UI) when debugging cache-related issues.

**Test sharding adds reporting complexity.** Aggregating results from 8 parallel test shards is more work than reading a single test report. Ensure your test reporting infrastructure (JUnit aggregation, coverage collection, flaky test detection) handles sharded results before you shard. Coverage data in particular requires merging multiple coverage reports, which adds a post-shard step to your workflow.

**Premature optimization is still premature.** A 10-minute pipeline for a 5-person team is not a problem worth solving. It becomes a problem when the organization grows and the pipeline is on the critical path of 30 engineers' daily work. Optimize when the cost (in developer time waiting, or in actual dollars) justifies the investment.

---

## Putting It Together: The CI Cost Audit Checklist

When you sit down to audit your CI costs, work through these in order. Each step provides the information you need for the next.

```bash
# Step 1: Get total monthly usage
gh api /orgs/YOUR_ORG/settings/billing/actions

# Step 2: Get per-repository breakdown
gh api /repos/YOUR_ORG/YOUR_REPO/actions/workflows \
  --jq '.workflows[] | {name:.name, id:.id}'

# Step 3: Get timing for the top-cost workflows
gh api /repos/YOUR_ORG/YOUR_REPO/actions/workflows/WORKFLOW_ID/timing

# Step 4: Check cache hit rates
gh api /repos/YOUR_ORG/YOUR_REPO/actions/cache/usage

# Step 5: Check for expensive runner types
gh api /repos/YOUR_ORG/YOUR_REPO/actions/runs \
  --jq '.workflow_runs[] | {name:.name, conclusion:.conclusion, 
        created_at:.created_at}'
```

Then work through the checklist:

- [ ] Is `actions/cache` configured for `package-lock.json`, `requirements.txt`, `pom.xml`, `build.gradle`?
- [ ] Is the cache key based on the dependency lockfile hash?
- [ ] Are cache hit rates above 80%? (Check in Actions UI > Caches)
- [ ] Are Docker builds using BuildKit with `--cache-from` pointing to a registry?
- [ ] Is the Dockerfile ordering dependencies before source code in each layer?
- [ ] Are any workflows using Windows or macOS runners for tasks that do not require them?
- [ ] Are large test suites (>15 min) using matrix sharding?
- [ ] Are workflow triggers scoped with `paths:` filters where possible?
- [ ] Is there a cost alert configured to notify when spend exceeds a threshold?
- [ ] Does someone own the CI cost metric?

The last two items are the most likely to be missing, and they are the ones that prevent the bill from silently doubling again six months later.

---

## The Pipeline as an Engineering System

The foundational shift this post argues for is treating the CI pipeline as an engineering system with a cost profile, not as a utility that runs in the background. Every software system you build gets capacity-planned, performance-monitored, and optimized. Your CI pipeline consumes real compute resources and real money. It deserves the same rigor.

The analogy to [pipeline observability and SLOs](/blog/software-development/ci-cd/pipeline-observability-and-the-flaky-pipeline) is direct. You would not run a production service without latency tracking and alerting. Your CI pipeline needs equivalent instrumentation: p95 pipeline duration, cache hit rates, runner utilization, cost per pipeline run, and weekly spend trends. When a metric moves unexpectedly, you want to know within hours, not at the end of the billing month.

The [CI/CD mental model post](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) frames the pipeline as the connective tissue between developer intent and production state. Slow, expensive pipelines add friction to that connection. Every minute a developer waits for CI feedback is a minute of interrupted flow. At scale, this is not just a cost problem — it is an engineering productivity problem. The \$45k → \$12k reduction was not just about the bill. The same pipeline changes that reduced the cost reduced the feedback cycle from 38 minutes to 6 minutes. Developers pushed more often, iterated faster, and caught bugs earlier. The cost reduction was the accounting manifestation of a pipeline that was working the way it should have been all along.

The [CI/CD capstone playbook](/blog/software-development/ci-cd/capstone-the-cicd-playbook) synthesizes these ideas across the entire delivery lifecycle — from commit to production. The cost problem we worked through here is one facet of a larger discipline: building delivery systems that are fast, reliable, and economical.

---

## The Long-Term CI Ownership Model

Fixing a \$45k bill once is an event. Keeping it fixed is a discipline. The organizations that sustain CI efficiency over years do so because they treat CI infrastructure as a product with owners, SLOs, and a review cadence — not as a shared utility that everyone uses and nobody manages.

### Who Owns CI Cost: Platform Team vs. Distributed Ownership

The two common models are full platform ownership and federated ownership. In the platform team model, a central infrastructure or developer-experience team owns the runner fleet, the caching infrastructure, and the billing dashboard. Individual product teams write workflows but do not manage the underlying machinery. In the federated model, each product team owns their own workflows end-to-end, including performance and cost.

Platform ownership scales more efficiently above about 30 engineers. A central team can enforce consistent caching patterns via shared reusable workflows, negotiate reserved instance pricing for self-hosted runners, and catch cost anomalies before they compound. The tradeoff is that product teams lose direct control over their runner configuration and must file requests to change runner specs or caching strategies.

Federated ownership works well at smaller scales and in organizations with strong engineering autonomy culture. The risk is that each team optimizes locally without visibility into the aggregate cost picture. One team's decision to add a 4-core runner for a 10-second speed improvement has no individual cost consequence, but if 12 teams make the same decision, the collective impact is \$3,000–\$5,000/month in unnecessary spend.

The practical resolution at mid-size organizations (50–200 engineers) is a hybrid: a platform team that owns the runner fleet and shared workflow templates, with per-team cost visibility that creates accountability without requiring platform team approval for every workflow change.

### The CI Budget Owner Role

Someone must own the CI cost number the way a product manager owns feature delivery — with a target, a review cadence, and accountability when the number moves unexpectedly.

The CI budget owner does not need to be a dedicated role. At most organizations it is a platform engineer or an engineering manager who owns it as 10–15% of their responsibilities. Their core responsibilities are:

Monthly cost review: pull the per-repository and per-workflow usage data via the GitHub billing API, compare against the prior month, and flag any line item that grew more than 20% month-over-month. A \$300 workflow that grew to \$600 in a month warrants investigation. A \$3,000 workflow that grew to \$3,100 probably does not.

Per-team cost allocation: the raw billing API gives per-repository data, which maps cleanly to team ownership when you have a team-to-repository ownership map. Export this monthly and share it with engineering managers. Teams that can see their own CI spend develop natural cost awareness without needing mandates from above. A team that sees their workflow consuming \$4,200/month — three times the next-highest team — will often self-correct once the number is visible.

Threshold alerting: configure a weekly Slack notification via the GitHub billing API that fires when weekly spend exceeds 125% of the four-week rolling average. This catches regressions within a week rather than at month-end billing.

### Establishing a CI SLO

A CI SLO — service-level objective — defines what "good" looks like for your pipeline. Without explicit targets, "good" defaults to "it passed," which conflates correctness with performance.

A practical starter set of CI SLOs covers three dimensions. Build time: P95 pipeline duration for your main-branch workflow should complete within a target that matches your organization's release cadence. For a team shipping multiple times per day, a P95 above 20 minutes is a productivity bottleneck. For a weekly release team, 40 minutes may be acceptable. Set the target deliberately, not by accident.

Success rate: the fraction of pipeline runs that complete without a transient failure (infrastructure errors, flaky tests, runner timeouts, cache download errors). A healthy pipeline should have a success rate above 95% on non-code failures. If your success rate is 88%, 12% of developer pushes are generating noise that erodes trust in the CI signal.

Queue depth SLA: for self-hosted runners, the time from job queued to job started should stay below a threshold — commonly 60 seconds at P95. A queue depth SLA surfaces runner capacity problems before developers start complaining. If you measure it and it breaches, the autoscaler configuration or the warm pool floor needs adjustment.

Track these three metrics over time in a dashboard (Grafana with a Prometheus exporter for the GitHub API is a common setup). A metric you cannot see is a metric you cannot improve.

### The Quarterly CI Audit Cycle

Beyond the monthly cost review, a quarterly deep-dive catches structural inefficiencies that do not show up in week-over-week deltas.

The quarterly audit covers four areas. Cache hit rates: pull cache hit rates for every major workflow. Any workflow with a hit rate below 75% has a broken or misscoped cache key. The most common causes are an overly specific key that invalidates on irrelevant file changes, a lock file path pattern that does not match the actual file location, or cache eviction pressure from too many branch-specific entries.

Cold starts: track how many jobs waited more than 90 seconds for a runner. A steady rise in cold-start frequency indicates that the autoscaler's minimum warm pool is too small for current traffic patterns — usually because the team has grown since the pool floor was set.

Idle runner time: for self-hosted fleets, measure the fraction of runner-minutes billed while no job was running. Above 15% idle time suggests the scale-down delay is too long or the minimum replica count is too high for off-peak hours. A scheduled minimum that drops from 5 to 2 runners overnight can recover \$200–\$800/month on a mid-sized fleet with no impact on daytime developer experience.

Job parallelism headroom: compare the historical peak concurrent job count against the maximum runner count. If peak concurrency ever hit 85% of the runner maximum, you are one CI traffic spike away from queue saturation. Either raise the maximum or add a monitoring alert that fires before the cap is hit.

### Decision Checklist Before Any CI Cost Optimization Initiative

Before investing engineering time in a CI optimization project, answer these five questions. They determine whether the initiative is worth running and what success looks like.

First: what is the current monthly cost of the workflow or behavior you are targeting, in actual dollars? If you cannot calculate this from the billing API, you are optimizing blind. Do the arithmetic before writing any YAML.

Second: what is the realistic savings estimate after the optimization, and what assumptions drive that estimate? A 90% cache hit rate on a 14-minute install step saves 12.6 minutes per run — calculate the monthly dollar figure from your actual run count. If the savings are under \$200/month, the engineering time to implement and maintain the optimization likely does not pay back within six months.

Third: what new failure mode does this optimization introduce? Caching adds the risk of stale state. Test sharding adds the risk of split coverage reports. Registry-cached Docker builds add a dependency on registry authentication. Every optimization has a corresponding failure mode; identify it before shipping.

Fourth: who will own this configuration going forward? An optimization that requires ongoing maintenance — cache key updates when lockfile formats change, autoscaler tuning as team size grows — needs a named owner. Unowned configurations drift silently back into inefficiency.

Fifth: how will you measure whether the optimization is working three months from now? Define the metric and the target before you implement. Without a pre-committed success metric, it is easy to ship the change, observe a short-term improvement, and then lose visibility as the benefit erodes over time.

---

## Key Takeaways

**CI cost is mostly build minutes, and build minutes are mostly cache misses and serial test suites.** Fix those two things before anything else. The savings are immediate, measurable, and require no infrastructure changes beyond YAML edits.

**The four levers are: runner type, cache hit rate, parallelism, and job elimination.** In roughly that order of implementation complexity and roughly the reverse order of cost impact per engineering hour. Cache hit rate is the highest-ROI lever for most teams.

**Docker layer ordering is a Dockerfile discipline problem, not a CI configuration problem.** `COPY . .` before `npm install` is the single most common cause of Docker build inefficiency. Fix it once in the Dockerfile and it stays fixed.

**Autoscaling is essential for self-hosted runners.** Idle runners are money you are paying for nothing. Insufficient runners are developer time you are burning on queue waits. The autoscaler control loop keeps these in balance.

**Self-hosted runners save money only when you have the volume and the operational capacity to justify them.** The break-even is roughly 250,000–300,000 minutes per month. Below that, managed runners and aggressive caching produce better ROI.

**CI cost needs an owner, a metric, and an alert.** Without these three things, costs grow silently until they become a budget crisis. With them, cost anomalies surface within days and can be investigated before they compound.

---

## Further Reading

- [Actions Runner Controller documentation](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners-with-actions-runner-controller/about-actions-runner-controller) — the definitive reference for ARC deployment and configuration
- [Docker BuildKit documentation](https://docs.docker.com/build/cache/) — comprehensive guide to BuildKit caching modes and registry backends
- [philips-labs/terraform-aws-github-runner](https://github.com/philips-labs/terraform-aws-github-runner) — production-grade Terraform module for EC2-based GitHub Actions runners
- [KEDA GitHub Runner Scaler](https://keda.sh/docs/2.13/scalers/github-runner/) — queue-depth-based autoscaling for GitHub Actions runners via KEDA
- [Turborepo remote caching](https://turbo.build/repo/docs/core-concepts/remote-caching) — team-wide build cache for JavaScript/TypeScript monorepos
- [pytest-split documentation](https://pytest-split.readthedocs.io/) — timing-based test sharding for Python projects
- [GitHub Actions cache management API](https://docs.github.com/en/rest/actions/cache) — programmatic access to cache hit rates, sizes, and eviction
- [CI/CD & Delivery series overview](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) — the mental model behind the entire series
