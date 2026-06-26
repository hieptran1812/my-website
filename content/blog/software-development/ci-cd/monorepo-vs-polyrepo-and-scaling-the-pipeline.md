---
title: "Monorepo vs Polyrepo: Scaling the Delivery Pipeline"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Learn how to eliminate the 2-hour monorepo CI bottleneck with affected-target builds, remote caching, and matrix parallelism — and decide which repo structure actually fits your team."
tags:
  [
    "ci-cd",
    "devops",
    "monorepo",
    "build-systems",
    "turborepo",
    "nx",
    "bazel",
    "pipeline-scaling",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 64
image: "/imgs/blogs/monorepo-vs-polyrepo-and-scaling-the-pipeline-1.png"
---

It is 10:47 AM on a Tuesday. A developer on the platform team merges a one-line CSS fix to a shared component library. Twelve seconds later, GitHub Actions fires up a fresh ephemeral runner, clones the repository, installs dependencies, and begins building every single package in the repo. Two hours and four minutes later, the pipeline finishes green — having compiled, linted, and tested all 40 packages regardless of which ones actually changed. The developer who merged that one line could have gone to lunch and come back. She did. She is on her third cup of coffee and has already forgotten what the original fix was for.

This is the 2-hour monorepo problem. It is not a fundamental limitation of the monorepo structure. It is not evidence that monorepos are a bad idea. It is what happens when you point a "build everything" pipeline at a repository that has grown beyond a handful of packages without investing in the tooling that makes monorepos scale. The team that shipped 40 packages in a single repo chose that structure deliberately — atomic refactoring, shared tooling, unified dependency upgrades — and gave up all of those advantages the moment CI became the bottleneck that blocked every merge.

The fix is not to split into a polyrepo. The fix is to build a build graph.

One team did exactly that. They were running a 42-package TypeScript monorepo for a Series B fintech startup. The platform engineer responsible for CI spent two days migrating to Turborepo, wiring up GitHub Actions matrix jobs, and connecting a remote cache backed by Vercel. Their next pipeline run: 7 minutes. The run after that: 6 minutes, because the remote cache had populated from the previous run and most unchanged packages restored their build artifacts instead of rebuilding. Same 42 packages. Same test suite. The difference was that Turborepo computed which three packages were actually affected by the CSS change and only ran those — the other 39 packages were cache hits.

![Build graph showing package dependency structure: when pkg-A changes only its downstream dependents pkg-C and pkg-E are rebuilt, while pkg-B and pkg-D become cache hits](/imgs/blogs/monorepo-vs-polyrepo-and-scaling-the-pipeline-1.png)

This post walks through the full picture. We start with the fundamental question — where does the coordination tax live in each structure? — then move through the mechanics of affected-target builds, the build tools that make them work (Bazel, Nx, Turborepo, Gradle, Pants), remote caching in depth, and how to scale CI with dynamic matrix jobs. We finish with the decisive criteria for choosing between monorepo and polyrepo, two worked examples with real numbers, and the honest case for when each structure works against you.

---

## The Fundamental Choice: Where Does the Coordination Tax Live?

The monorepo versus polyrepo debate is not about code organization. It is not about how many files live in a single Git repository. It is about where you pay the coordination tax — and the coordination tax is the engineering cost of keeping multiple components that depend on each other aligned as they evolve.

Every non-trivial software system has multiple components that depend on each other. When those dependencies change — a shared library gets a new interface, a shared utility adds a required argument, a common configuration format changes — something has to coordinate the update across all consumers. That coordination has a cost, and you pay it either upfront at the repo boundary or later at the build boundary.

**In a polyrepo:** the coordination tax shows up at dependency management. Each service publishes versioned artifacts — npm packages, Maven artifacts, Docker images, Python wheels. Breaking changes require explicit version bumps and separate PRs across every consuming repository. CI for each individual repo is fast because it only builds that one service. But the total engineering cost of keeping 40 services aligned with a changing library is measured in person-weeks per quarter. You pay the tax in a diffuse, hard-to-observe way: scattered across dozens of repositories, piling up in open PRs that nobody reviews quickly, accumulating as a permanent tail of repos stuck on old library versions.

**In a monorepo:** the coordination tax shows up at the build layer. All code lives in one place, so refactoring is atomic — one PR touches the library and all call sites simultaneously, and CI proves that nothing broke before anyone merges. But if your CI naively builds everything on every commit, the build time grows proportionally to the number of packages. A 40-package monorepo with a 3-minute average build per package will take 120 minutes of wall-clock time with no parallelism. Even with 8 parallel runners, the job that hits the most expensive package takes the critical path — typically 15–20 minutes. You pay the tax visibly, on every commit, in a way that every developer on the team notices and complains about.

The key insight that most monorepo/polyrepo arguments miss: **neither structure is inherently better**. The question is whether you invest in the tooling to make your chosen structure pay off. A polyrepo without a dependency update automation strategy (Renovate, Dependabot, or a custom bot) accumulates version drift. A monorepo without affected-target builds and a build graph tool accumulates CI minutes. Both outcomes are bad. The right answer is the structure that matches your tooling investment and team topology — not the structure that worked for Google or Vercel or Netflix.

Put plainly: a monorepo without a build graph is a disaster waiting to happen. A polyrepo without automated dependency management is also a disaster, it just takes longer to notice. Pick the disaster you have better tooling to prevent.

## Monorepo: The Build-Graph Problem in Depth

A monorepo is a single version-control repository that contains multiple distinct projects, packages, or services that may be independently deployable but share code and tooling. The archetypal example in the JavaScript world is a repository with `packages/ui`, `packages/api-client`, `packages/auth`, and `apps/web` — four distinct artifacts built from one repo. In the JVM world, it might be a Gradle multi-project build with `services/auth`, `services/payments`, `libs/common`, and `libs/testing`. In Go, a single module with multiple `cmd/` binaries and shared `pkg/` libraries.

### The naive approach and why it fails

The simplest possible CI pipeline for a monorepo is:

```yaml
# .github/workflows/ci.yml — the naive approach
name: CI (naive — do not use at scale)
on: [push]
jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
      - run: npm ci
      - run: npm run lint --workspaces
      - run: npm run build --workspaces
      - run: npm run test --workspaces
```

This is correct for a repo with three packages. It is a 2-hour pipeline for a repo with 40. The fundamental flaw is that `--workspaces` has no concept of what changed — it runs every package unconditionally on every push. The same 40-package build fires whether you changed one line in one package or rewrote 10 packages from scratch.

The naive approach has two compounding failure modes. First, build time grows linearly with package count — adding more packages slows CI for everyone, even for PRs that do not touch those packages. This creates a perverse incentive: teams avoid extracting shared code into packages because each new package makes CI slower. Second, when builds fail, they fail in packages that had nothing to do with the change — triggering false positives that erode trust in the CI signal and train developers to re-run CI without investigating failures.

### What a build graph gives you

A build graph is a directed acyclic graph (DAG) where each node is a buildable artifact — a package, library, service, or test suite — and each edge represents a dependency relationship. When you run a build for a specific target, the build tool:

1. Traverses the dependency graph to find all transitive dependencies of the target.
2. Computes a content hash for each node based on its source files, the hashes of its declared dependencies, and the versions of its build tools and compiler.
3. Checks whether a cached artifact exists for that hash (locally in a cache directory, or remotely in a cache server).
4. Runs only the nodes where no valid cache entry exists, in topological order (dependencies before dependents).
5. Stores the build output in the cache so future runs with the same inputs can skip the computation entirely.

The affected set from a change to package A is: A itself, plus every node that transitively depends on A. If B depends on A, and C depends on B, then changing A affects A, B, and C — but not D, E, or any package that has no dependency path to A. Packages with no dependency path to the changed package are unaffected and get cache hits — their stored outputs are valid and do not need to be recomputed.

This is not a minor optimization. On a 40-package monorepo where the average PR touches 1–3 packages, affected-target builds reduce average pipeline time by 85–95%. The 2-hour pipeline becomes an 8-minute pipeline.

![Naive CI pipeline takes 2 hours 4 minutes building all 40 packages versus affected-target CI takes 8 minutes building only 3 affected packages](/imgs/blogs/monorepo-vs-polyrepo-and-scaling-the-pipeline-2.png)

The reduction is multiplicative because each eliminated package contributes its full build, lint, and test time to the saved total. A 3-minute test suite × 37 unaffected packages = 111 minutes of test time that does not need to run.

### The anatomy of a content hash

Understanding what goes into the content hash is critical for getting the most out of a build cache. The hash typically includes:

- **Source files**: all `.ts`, `.js`, `.go`, `.py`, etc. files in the package directory. Any source change invalidates the cache for that package and all its dependents.
- **Dependency hashes**: the content hash of each declared dependency. If a dependency's hash changes (because that dependency was rebuilt), all packages that depend on it also get cache misses.
- **Lock file**: `package-lock.json`, `yarn.lock`, `go.sum`, `Cargo.lock`. Changes to external dependency versions propagate cache invalidations through the entire build.
- **Build tool configuration**: `turbo.json`, `nx.json`, `BUILD` files, `build.gradle.kts`. Changes to task definitions invalidate caches for the affected tasks.
- **Environment variables declared as inputs**: Node version, `CC`, `GOARCH`, and other variables you declare as relevant. Undeclared environment variables do not affect the hash — this is both a feature (reproducibility) and a risk (secret values that should affect builds must be explicitly declared).

Well-calibrated input declarations are the single highest-leverage configuration you can make. Every file you correctly exclude from the inputs is a file that, when changed, does not bust the build cache. README updates, CHANGELOG entries, test fixtures for unrelated tests, and documentation images should all be excluded from production build inputs.

## Polyrepo: The Dependency-Management Problem in Depth

A polyrepo (sometimes called a multi-repo) structure puts each service or library in its own repository with its own CI/CD pipeline, its own versioning, and its own release cadence. This is the default organizational state — most engineering organizations start with polyrepo and accumulate repos as they grow.

### Why polyrepos feel fast — and are genuinely fast, initially

Individual repo CI is fast by construction. The repo contains one service. Building one service takes 3 minutes. The entire CI pipeline — install, lint, build, test, package — completes in 5 minutes. There is no ambiguity about what to build: you always build everything in the repo, and "everything" is one artifact.

This is genuinely good. A 5-minute feedback loop is fast enough that developers do not feel the pain of waiting for CI before starting their next task. The loop is tight enough to catch regressions quickly. Teams that have never experienced a 2-hour CI run have never been tempted to start merging without waiting for CI.

Polyrepo isolation also provides a real benefit for organizations with strong team ownership boundaries. Team A can break their own repo's CI without affecting Team B's ability to merge and deploy. A major refactor in `service-payments` does not slow down merges in `service-notifications`. Teams can independently upgrade their language version, switch test frameworks, or restructure their CI pipeline without coordinating across the organization.

### Where polyrepos accumulate debt: version drift

The coordination tax in a polyrepo is invisible on any individual day. It becomes visible only when you aggregate the engineering time spent on dependency management across all repos over a quarter.

The mechanism is straightforward. You publish `@company/auth-utils@2.0.0` with a breaking API change. You notify all 40 service teams. Day 1: five high-priority services merge their upgrade PRs quickly. Day 3: ten more services have open PRs, some stalled waiting for review. Day 7: fifteen services are upgraded, the other 25 are stuck — some have merge conflicts because their branch was created before the bump was backported, some have CI failures from transitive conflicts, some just have not been triaged yet.

Three months later, `@company/auth-utils@3.0.0` is needed for a security patch. You now have 25 services that are two major versions behind. The upgrade path for those services is now `1.x → 2.x → 3.x`, requiring two migrations instead of one. The engineering cost is nonlinear: a service that was neglected for three months costs three times as much to bring current as one that upgrades incrementally.

### The numbers: polyrepo coordination cost at 40 services

A concrete baseline measurement from a real organization (anonymized): 42 service repos, 8 shared libraries, quarterly major version bumps per library. Measured engineering cost:

| Activity | Time per event | Events per quarter | Quarterly cost |
|---|---|---|---|
| Author upgrade PRs (40 repos × 8 libs) | 15 min/PR | 320 | 80 eng-hours |
| Review and merge upgrade PRs | 20 min/PR | 320 | 107 eng-hours |
| Resolve merge conflicts from late upgrades | 45 min avg | 120 (38% conflict rate) | 90 eng-hours |
| Audit: verify all 40 repos upgraded | 30 min/lib | 8 | 4 eng-hours |
| Handle services stuck on old versions | 2 hours avg | 15 | 30 eng-hours |
| **Total per quarter** | | | **311 eng-hours ≈ 8 weeks** |

At a loaded engineer cost of \$100/hour, that is \$31,100 per quarter — \$124,400 per year — in pure coordination overhead. The organization could have funded a full-time build infrastructure engineer to migrate to a monorepo and recoup the investment in year one.

The same organization after migrating to a monorepo: each major version bump is one automated Renovate PR that updates all consumers simultaneously, with affected CI proving it passes. The quarterly coordination cost drops to 4 engineer-hours.

![Monorepo vs polyrepo across five dimensions: atomic refactoring, dependency upgrades, CI speed without tooling, repo isolation, and onboarding new developers](/imgs/blogs/monorepo-vs-polyrepo-and-scaling-the-pipeline-3.png)

## Build Tools: The Ecosystem in Full Depth

Affected-target builds require a tool that understands the build graph. Without a tool that can track which packages depend on which others and compute content-addressed hashes for each, you cannot know which packages are affected by a given change. The build tool is not optional — it is the enabling technology that makes the monorepo coordination strategy work.

### Bazel: Hermetic correctness at Google scale

Bazel is Google's open-source build system, extracted from their internal Blaze tool and released in 2015. It is the most powerful and most complex option in the ecosystem. Bazel uses a content-addressed, hermetically sealed build model: every build step explicitly declares its inputs and outputs in `BUILD` files, and the build tool guarantees that two builds with the same inputs produce byte-for-byte identical outputs regardless of which machine, at what time, or in what environment the build ran.

The hermetic model is what makes Bazel correct at Google's scale. If any undeclared input could affect a build step, cache hits would be incorrect — a package might be served a cached artifact that was built with a different version of a transitive dependency than the one currently in the repo. Bazel prevents this by failing builds that read from undeclared inputs, forcing engineers to be explicit about every dependency.

A Bazel `BUILD` file for a Go microservice:

```python
# //services/auth/BUILD
load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library", "go_test")

go_library(
    name = "auth_lib",
    srcs = glob(["*.go"], exclude = ["*_test.go"]),
    importpath = "github.com/company/platform/services/auth",
    deps = [
        "//libs/tokens:tokens_lib",
        "//libs/logging:logging_lib",
        "@com_github_golang_jwt_jwt//:jwt",
        "@com_github_grpc_grpc_go//codes:codes",
    ],
    visibility = ["//visibility:public"],
)

go_binary(
    name = "auth",
    embed = [":auth_lib"],
)

go_test(
    name = "auth_test",
    srcs = glob(["*_test.go"]),
    embed = [":auth_lib"],
    deps = [
        "//libs/testutils:testutils_lib",
        "@com_github_stretchr_testify//assert:assert",
    ],
)
```

The `.bazelrc` for a CI environment with remote caching:

```bash
# .bazelrc
# Remote cache configuration
build --remote_cache=grpcs://cache.example.com:443
build --google_default_credentials
build --remote_timeout=3600

# Remote build execution (optional, requires a BES-compatible cluster)
build:remote --remote_executor=grpcs://rbe.example.com:443
build:remote --jobs=200

# Performance
build --jobs=auto
build --local_ram_resources=HOST_RAM*.75
build --local_cpu_resources=HOST_CPUS*.5

# Test settings
test --test_output=errors
test --keep_going
test --test_timeout=120

# CI-specific: always use remote cache
build:ci --remote_cache=grpcs://cache.example.com:443
build:ci --noremote_accept_cached=false
build:ci --remote_upload_local_results=true
```

#### How Bazel's query language computes rdeps

The `bazel query` language is SQL-like. It operates on the build graph and returns labels (targets) that satisfy a predicate. The most important operator for affected-target computation is `rdeps` — reverse dependencies.

The grammar of a `rdeps` query:

```bash
# rdeps(universe, start_set [, depth])
# Returns all targets in `universe` that depend on any target in `start_set`.
# Optional `depth` limits the reverse-dependency traversal depth.

# Find everything in //... that depends on //libs/tokens:tokens_lib
bazel query "rdeps(//..., //libs/tokens:tokens_lib)"

# Find everything that directly depends on tokens (depth=1 only)
bazel query "rdeps(//..., //libs/tokens:tokens_lib, 1)"

# Find all targets affected by any change in libs/auth
bazel query "rdeps(//..., //libs/auth/...)"

# Combine multiple changed packages with union()
bazel query "rdeps(//..., union(//libs/tokens/..., //libs/auth/...))"

# Filter to only test targets in the affected set
bazel query "kind('.*_test', rdeps(//..., //libs/tokens:tokens_lib))"
```

The query engine reads the in-memory build graph that Bazel builds from all `BUILD` files in the workspace. This means the query is authoritative — it computes from the actual declared dependency graph, not from heuristics or file path matching. A target is included in the `rdeps` result only if there exists a declared dependency edge, either direct or transitive, from that target back to the changed package.

In practice, a CI script converts changed file paths to Bazel package labels, runs a `rdeps` query to find the full affected set, then passes those labels directly to `bazel test`:

```bash
#!/usr/bin/env bash
BASE="${1:-origin/main}"
HEAD="${2:-HEAD}"

CHANGED=$(git diff --name-only "$BASE"..."$HEAD")

if [ -z "$CHANGED" ]; then
  echo "No changed files detected."
  exit 0
fi

CHANGED_PACKAGES=$(echo "$CHANGED" | \
  sed 's|/[^/]*$||' | \
  sort -u | \
  sed 's|^|//|' | \
  sed 's|$|:all|')

AFFECTED_TARGETS=$(bazel query \
  "rdeps(//..., union(${CHANGED_PACKAGES//\n/,}))" \
  --output=label \
  2>/dev/null)

echo "Affected targets:"
echo "$AFFECTED_TARGETS"

if [ -n "$AFFECTED_TARGETS" ]; then
  # shellcheck disable=SC2086
  bazel test $AFFECTED_TARGETS --keep_going --test_output=errors
fi
```

Bazel is the right answer when you have a large polyglot monorepo (Go, Java, Python, C++, Rust coexisting), hermetic reproducibility is a hard requirement (pharmaceutical, financial, government software that requires provenance attestation), or you are already at the scale where a dedicated build infrastructure team is justified. The operational overhead — maintaining BUILD files, running a remote cache server, managing external dependency declarations — is real and should not be underestimated.

For organizations below roughly \$500,000/year in CI compute costs, Bazel's operational overhead almost always outweighs its benefits. For organizations above \$2M/year in CI costs (approximately 50+ engineers running CI heavily), the economics favor Bazel significantly.

### Nx: The JavaScript and TypeScript powerhouse

Nx is a build system and development toolkit focused on JavaScript and TypeScript monorepos. It understands the npm/yarn/pnpm workspace model, integrates with popular frameworks (Next.js, React, Angular, NestJS, Fastify), and provides first-class affected-target computation via its project graph.

Unlike Bazel, Nx does not require explicit BUILD files. It infers the project graph by analyzing `package.json` `dependencies` fields across workspaces and (for Angular/React apps) by static analysis of import statements. This makes Nx significantly easier to adopt in an existing JavaScript monorepo.

#### How Nx computes the project graph

Nx builds its project graph in two phases. In the first phase, it reads every `package.json` in the workspace and constructs an edge for each dependency listed in `dependencies` and `devDependencies` that refers to another workspace package. In the second phase, for projects that have an Nx executor configured (such as `@nx/webpack:webpack`, `@nx/jest:jest`, or a custom executor), Nx performs static analysis of TypeScript import statements to detect implicit dependencies that were not listed in `package.json`.

```bash
# Visualize the full project graph in the browser
npx nx graph

# Visualize only the projects affected by current changes
npx nx graph --affected --base=origin/main

# Output the graph as JSON for programmatic use
npx nx graph --file=graph.json

# Show which projects are affected (text output)
npx nx show projects --affected --base=origin/main --head=HEAD
```

The `nx graph` command opens a browser window with an interactive DAG visualization. Nodes are projects, edges are dependency relationships, and the affected set (the projects with cache misses) is highlighted. This visualization is the primary debugging tool when the affected set seems wrong — you can trace exactly why a particular project was or was not included.

When the affected set includes a project you did not expect, the most common cause is an undeclared implicit dependency — a TypeScript `import` that Nx detected via static analysis but that is not reflected in `package.json`. The fix is to add the dependency to `package.json`, which makes the relationship explicit and therefore correct in the graph.

The `nx.json` for a full-featured setup:

```json
{
  "$schema": "./node_modules/nx/schemas/nx-schema.json",
  "nxCloudAccessToken": "your-nx-cloud-token",
  "defaultBase": "main",
  "targetDefaults": {
    "build": {
      "dependsOn": ["^build"],
      "inputs": ["production", "^production"],
      "outputs": ["{projectRoot}/dist"],
      "cache": true
    },
    "test": {
      "inputs": [
        "default",
        "^production",
        "{workspaceRoot}/jest.preset.js",
        "{workspaceRoot}/jest.config.ts"
      ],
      "outputs": ["{projectRoot}/coverage"],
      "cache": true
    },
    "lint": {
      "inputs": [
        "default",
        "{workspaceRoot}/.eslintrc.json",
        "{workspaceRoot}/.eslintignore"
      ],
      "cache": true
    },
    "type-check": {
      "inputs": ["default", "^production", "{workspaceRoot}/tsconfig.base.json"],
      "cache": true
    }
  },
  "namedInputs": {
    "default": ["{projectRoot}/**/*", "sharedGlobals"],
    "production": [
      "default",
      "!{projectRoot}/**/?(*.)+(spec|test).[jt]s?(x)?(.snap)",
      "!{projectRoot}/tsconfig.spec.json",
      "!{projectRoot}/.eslintrc.json",
      "!{projectRoot}/**/*.stories.[jt]s?(x)"
    ],
    "sharedGlobals": [
      "{workspaceRoot}/nx.json",
      "{workspaceRoot}/tsconfig.base.json"
    ]
  },
  "parallel": 3,
  "cacheDirectory": ".nx/cache"
}
```

Running affected targets with Nx — the core commands:

```bash
# Show which projects are affected by changes since main
npx nx show projects --affected --base=origin/main --head=HEAD

# Run build for all affected projects (respects dependency order)
npx nx affected --target=build --base=origin/main --head=HEAD

# Run multiple targets for all affected projects
npx nx affected --target=build,test,lint --parallel=4

# Run a specific target for a specific project and all its dependencies
npx nx run-many --target=build --projects=web --with-deps

# Graph the affected projects visually
npx nx graph --affected --base=origin/main
```

The GitHub Actions workflow with Nx and dynamic SHA computation:

```yaml
# .github/workflows/ci.yml — Nx affected pipeline
name: CI

on:
  push:
    branches: [main]
  pull_request:

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  nx-affected:
    name: Nx affected build + test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # required for nx/set-shas to find the correct base

      - uses: nrwl/nx-set-shas@v4
        # sets NX_BASE and NX_HEAD environment variables
        # NX_BASE = last successful commit on the base branch
        # NX_HEAD = current HEAD

      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm

      - run: npm ci

      - name: Run affected lint, build, test
        run: |
          npx nx affected \
            --target=lint,type-check,build,test \
            --base=$NX_BASE \
            --head=$NX_HEAD \
            --parallel=4 \
            --max-parallel=4
        env:
          NX_CLOUD_ACCESS_TOKEN: ${{ secrets.NX_CLOUD_ACCESS_TOKEN }}
```

The `nrwl/nx-set-shas@v4` action is what makes affected computation correct in CI. It finds the last successful run on the base branch and uses that commit as `NX_BASE`, so Nx computes the affected set relative to "what was last green on main" rather than just "what changed in this PR". This catches cases where a PR on main failed and the next PR would otherwise skip affected targets because nothing changed relative to the current HEAD.

Nx Cloud provides distributed task execution — distributing individual tasks across multiple agents — in addition to remote caching. For very large monorepos (100+ projects), DTE can reduce task execution time beyond what local parallelism achieves.

### Turborepo: The pragmatic sweet spot for JS/TS

Turborepo occupies the sweet spot between "too simple" (npm workspaces with no graph awareness) and "too complex" (Bazel with full hermetic sealing). It uses the npm/yarn/pnpm workspace topology as its project graph source and requires only a `turbo.json` to describe the pipeline task dependency structure.

The critical insight of Turborepo's design: in JavaScript/TypeScript monorepos, the dependency graph is already encoded in `package.json` files. The `"dependencies"` and `"devDependencies"` fields of each workspace package define the edges of the build graph. Turborepo reads this topology automatically — no BUILD files, no explicit graph declarations. The `turbo.json` only needs to describe *how* to execute tasks and what the dependency ordering is between task types.

#### How Turborepo hashes its inputs

Turborepo's hash function is deterministic and layered. For a given task (say, `build` in `packages/ui`), the cache key is constructed as follows:

1. **File hashes**: Turborepo scans all files matching the `inputs` glob patterns for that package directory. For each file it computes a SHA-256 of the file content and appends the relative file path. The list of (path, sha256) pairs is sorted lexicographically, then the sorted list is hashed again to produce a single package source hash.

2. **Declared environment variable values**: Each environment variable listed in `env` for the task is read and its value is included in the hash. `NODE_ENV=production` and `NODE_ENV=development` produce different hashes for the same source files.

3. **Dependency task hashes**: For each workspace package listed in `dependencies` or `devDependencies`, Turborepo recursively computes that package's hash for the `build` task (since `dependsOn: ["^build"]` means "my build depends on my deps' builds"). The hash of `packages/tokens`'s build output is included when computing the hash of `packages/ui`'s build.

4. **Global hash contributions**: Files listed in `globalDependencies` (typically `.env.*local`, `tsconfig.base.json`, tool config files at the root) are hashed and included in every task's cache key. Changing `tsconfig.base.json` invalidates every TypeScript package's build cache.

The final cache key is the SHA-256 of the concatenated list of all the above contributions. Turborepo stores this key as the directory name under `.turbo/cache/` locally, and as the path under the remote cache S3 bucket.

A concrete example of what the hash inputs look like in practice:

```bash
# Turborepo stores cache entries as:
# .turbo/cache/<hash>/
#   ├── .meta.json     — task metadata and timing
#   ├── outputs/       — captured build artifacts
#   └── logs/          — stdout/stderr of the task

# To inspect what Turborepo is hashing for a task:
turbo run build --filter=packages/ui --dry=json | jq '.tasks[0]'
# Output includes:
# {
#   "taskId": "packages/ui#build",
#   "task": "build",
#   "package": "packages/ui",
#   "hash": "abc123def456...",
#   "inputs": { ... },
#   "hashOfExternalDependencies": "xyz789..."
# }
```

The `--dry=json` output exposes the hash without running the task, which is useful for debugging cache misses: compare the hash across two runs to identify which input changed.

The `turbo.json` for a production setup with all common task types:

```json
{
  "$schema": "https://turbo.build/schema.json",
  "globalDependencies": [
    "**/.env.*local",
    "tsconfig.base.json"
  ],
  "globalEnv": ["NODE_ENV", "VERCEL_ENV"],
  "pipeline": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": ["dist/**", ".next/**", "!.next/cache/**"],
      "env": ["DATABASE_URL", "API_URL"],
      "cache": true
    },
    "test": {
      "dependsOn": ["^build"],
      "outputs": ["coverage/**", "test-results/**"],
      "cache": true
    },
    "lint": {
      "outputs": [],
      "cache": true
    },
    "type-check": {
      "dependsOn": ["^build"],
      "outputs": [],
      "cache": true
    },
    "dev": {
      "cache": false,
      "persistent": true
    }
  }
}
```

The `"dependsOn": ["^build"]` syntax means "run the `build` task of all my workspace dependencies before running my own `build`." The `^` prefix is the topological ordering operator. Without it, all instances of the task would run in parallel without respecting dependency order. With it, Turborepo computes the correct topological sort of the build graph and executes tasks layer by layer.

Running affected builds with Turborepo's `--filter` flag:

```bash
# Build affected packages since main (using git diff internally)
turbo run build --filter=...[origin/main]

# Build affected packages AND their dependencies
turbo run build --filter=...[origin/main]...

# Dry run: show what would be executed without running
turbo run build --filter=...[origin/main] --dry=json | jq '.tasks[].package'

# Run on a specific set of packages with glob
turbo run build --filter=./packages/*

# Run on a single app and all its transitive dependencies
turbo run build --filter=web...
```

The `--filter=...[origin/main]` syntax is the most common CI invocation. The `...` prefix means "this package and all packages that depend on it transitively." The `[origin/main]` part means "packages that have changed since origin/main." Together: "all packages that have changed since main, plus all packages that depend on them."

Configuring Turborepo remote cache — self-hosted via the open-source cache server:

```yaml
# docker-compose.yml for self-hosted Turborepo remote cache
version: "3"
services:
  turborepo-cache:
    image: ducktors/turborepo-remote-cache:latest
    ports:
      - "3000:3000"
    environment:
      - TURBO_TOKEN=your-shared-secret-token
      - STORAGE_PROVIDER=s3
      - STORAGE_PATH=turborepo-cache
      - S3_ACCESS_KEY=${AWS_ACCESS_KEY_ID}
      - S3_SECRET_KEY=${AWS_SECRET_ACCESS_KEY}
      - S3_REGION=us-east-1
      - S3_ENDPOINT=https://s3.amazonaws.com
```

```yaml
# .github/workflows/ci.yml — Turborepo affected pipeline
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  compute-matrix:
    name: Compute affected packages
    runs-on: ubuntu-latest
    outputs:
      matrix: ${{ steps.affected.outputs.matrix }}
      has-packages: ${{ steps.affected.outputs.has-packages }}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
      - run: npm ci
      - id: affected
        run: |
          PACKAGES=$(npx turbo run build \
            --filter=...[origin/${{ github.base_ref || 'main' }}] \
            --dry=json \
            --token=${{ secrets.TURBO_TOKEN }} \
            --team=${{ vars.TURBO_TEAM }} | \
            jq -c '[.tasks[].package | select(. != null)] | unique')
          echo "matrix={\"package\":$PACKAGES}" >> "$GITHUB_OUTPUT"
          COUNT=$(echo "$PACKAGES" | jq 'length')
          echo "has-packages=$([ "$COUNT" -gt 0 ] && echo true || echo false)" >> "$GITHUB_OUTPUT"
          echo "Affected packages ($COUNT): $PACKAGES"

  build-test:
    name: Build + test ${{ matrix.package }}
    needs: compute-matrix
    if: needs.compute-matrix.outputs.has-packages == 'true'
    runs-on: ubuntu-latest
    strategy:
      matrix: ${{ fromJson(needs.compute-matrix.outputs.matrix) }}
      fail-fast: false
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
      - run: npm ci
      - name: Run build and test
        run: |
          npx turbo run lint build test \
            --filter=${{ matrix.package }} \
            --token=${{ secrets.TURBO_TOKEN }} \
            --team=${{ vars.TURBO_TEAM }}
        env:
          TURBO_TOKEN: ${{ secrets.TURBO_TOKEN }}
          TURBO_TEAM: ${{ vars.TURBO_TEAM }}
```

### Gradle and Maven: JVM world affected builds

For Java, Kotlin, and Android monorepos, Gradle is the dominant build tool. Gradle's build cache — both local and remote — is mature, well-documented, and integrated with the Develocity platform (formerly Gradle Enterprise).

A Gradle multi-project repository `settings.gradle.kts`:

```kotlin
rootProject.name = "platform"

// Include all service subprojects
include("services:auth")
include("services:payments")
include("services:notifications")
include("services:user-management")

// Include all library subprojects
include("libs:common")
include("libs:testing")
include("libs:proto-generated")
```

Root `build.gradle.kts` with Develocity remote cache:

```kotlin
plugins {
    id("com.gradle.develocity") version "3.17"
}

develocity {
    buildScan {
        termsOfUseUrl.set("https://gradle.com/terms-of-service")
        termsOfUseAgree.set("yes")
        publishing.onlyIf { System.getenv("CI") == "true" }
    }
    buildCache {
        local {
            isEnabled = true
            directory = File(rootDir, ".gradle/build-cache")
        }
        remote<com.gradle.develocity.agent.gradle.buildcache.DevelocityBuildCache> {
            isPush = System.getenv("CI") == "true"
            isEnabled = true
        }
    }
}
```

For affected-target computation in Gradle, a common approach uses a shell script to find changed subproject directories and map them to Gradle project paths:

```bash
#!/usr/bin/env bash
# scripts/gradle-affected.sh
# Compute affected Gradle subprojects from git diff

BASE_BRANCH="${1:-origin/main}"
HEAD="${2:-HEAD}"

CHANGED=$(git diff --name-only "$BASE_BRANCH"..."$HEAD")

# Find unique top-level subproject paths
AFFECTED_PROJECTS=$(echo "$CHANGED" | \
  grep -E '^(services|libs)/' | \
  sed 's|/.*||' | \
  sort -u)

# Map directory names to Gradle project paths
GRADLE_PROJECTS=()
while IFS= read -r dir; do
  subdir=$(echo "$CHANGED" | grep "^$dir/" | head -1 | \
    sed "s|^$dir/||" | cut -d/ -f1)
  GRADLE_PROJECTS+=(":$dir:$subdir")
done <<< "$AFFECTED_PROJECTS"

# Build only affected projects and their dependents
if [ ${#GRADLE_PROJECTS[@]} -gt 0 ]; then
  ./gradlew ${GRADLE_PROJECTS[@]/%/:test} \
    --build-cache \
    --parallel \
    --configuration-cache
fi
```

Gradle's configuration cache (introduced in Gradle 7.x, stable in 8.x) is a separate performance feature worth enabling. It caches the result of the configuration phase — the Groovy/Kotlin script evaluation that constructs the task graph — so that incremental runs skip re-parsing and re-evaluating all build scripts. On a 30-subproject JVM monorepo, configuration cache can cut the startup overhead from 45 seconds to under 3 seconds.

For projects still using Maven, the Develocity Maven extension provides remote caching at the Mojo level. Maven's reactor ordering already handles topological build order, so affected builds require only a custom goal to identify changed modules:

```bash
# Maven affected build using Develocity predictive test selection
mvn -T 4 verify \
  -pl $(./scripts/maven-affected.sh origin/main HEAD) \
  -am \
  --no-transfer-progress
```

### Pants: Python and polyglot affected builds with import inference

Pants (Pants v2) brings Bazel-style hermetic builds to Python, Go, and Java without the full BUILD file authoring burden. Its key differentiator is **import inference**: Pants reads Python `import` statements, Go `import` paths, and Java package declarations to automatically infer dependency edges in the build graph. You do not declare `BUILD` files at all for most code — Pants generates them.

```toml
# pants.toml
[GLOBAL]
pants_version = "2.20.0"
backend_packages = [
    "pants.backend.python",
    "pants.backend.python.lint.flake8",
    "pants.backend.python.lint.black",
    "pants.backend.python.lint.isort",
    "pants.backend.python.typecheck.mypy",
]

[source]
root_patterns = ["src", "tests", "scripts"]

[python]
interpreter_constraints = ["CPython>=3.11,<3.12"]

[python-infer]
use_rust_parser = true
```

Running affected targets with Pants:

```bash
# Test only directly changed files and their dependents
pants --changed-since=origin/main test

# Test changed files and ALL transitive dependents
pants --changed-since=origin/main --changed-dependees=transitive test

# Lint and type-check affected files only
pants --changed-since=origin/main lint check

# Full affected pipeline in CI
pants --changed-since=origin/$BASE_BRANCH \
      --changed-dependees=transitive \
      lint check test
```

For Python monorepos with shared utility code — common in data engineering, ML platform, and backend service organizations — Pants provides the correct affected-target computation without requiring BUILD file authoring. The import inference accuracy is generally 95%+ for well-structured Python code.

### Buck2: Meta's next-generation build tool

Buck2 is Meta's open-source successor to Buck (their first-generation Bazel-like tool). Released in 2023, Buck2 is the system that builds all of Meta's internal code across their monorepo — Facebook.com, Instagram, WhatsApp, and Meta's internal infrastructure.

The key technical differentiations from Bazel are:

- **Starlark-native rules**: All build rules are written in Starlark (the same language Bazel uses), but Buck2 evaluates them lazily and in parallel using a Rust-based executor. A large graph that takes Bazel 90 seconds to analyze can be analyzed by Buck2 in under 10 seconds.
- **Remote execution by default**: Buck2 was designed from the start to use remote execution (RE) as the standard mode. Every build action is sent to a RE cluster by default; local execution is the fallback. This inverts the Bazel model where local execution is the default.
- **Action graph deduplication**: Buck2 deduplicates identical build actions across the graph. If two targets compile the same source file with the same flags, the compilation happens once. Bazel performs the same deduplication but only via the cache; Buck2 detects it in the action graph itself before execution.

For organizations that are evaluating build tools and have polyglot requirements at very large scale, Buck2 is worth investigating — but its operational maturity outside of Meta's internal infrastructure is still maturing as of this writing.

### Full build tool comparison

With all tools surveyed, a comprehensive comparison:

| Tool | Primary languages | Graph source | Affected mechanism | Remote cache | Setup complexity | Best fit |
|---|---|---|---|---|---|---|
| Turborepo | JS/TS | `package.json` workspaces | `--filter=[base]` | Vercel / self-hosted S3 | Low (2–4 hrs) | JS/TS monorepos under 100 packages |
| Nx | JS/TS + Angular/React | `package.json` + static import analysis | `nx affected` | Nx Cloud / self-hosted | Medium (1–2 days) | JS/TS monorepos over 50 packages or with framework tooling |
| Gradle | Java, Kotlin, Android | `settings.gradle.kts` subproject declarations | Custom script + build cache | Develocity / S3 | Medium (1–2 days) | JVM monorepos of any size |
| Maven | Java (legacy) | POM reactor ordering | Custom script | Develocity Maven ext | Low-Medium | Legacy JVM projects migrating to build caching |
| Pants | Python, Go, Java | Auto-inferred from imports | `--changed-since` + `--changed-dependees` | Built-in remote (GCS/S3) | Medium | Python data/ML monorepos |
| Bazel | Polyglot (all languages) | Explicit `BUILD` files | `bazel query rdeps()` | Any BES-compatible server | High (weeks) | Large polyglot monorepos, hermetic reproducibility required |
| Buck2 | Polyglot | Explicit `BUCK` files | `buck2 query rdeps()` | RE-integrated | High (weeks) | Very large polyglot monorepos with a build infra team |

The default recommendation for new JavaScript/TypeScript monorepos is **Turborepo**. Setup time is 2–4 hours, the remote cache works out of the box with Vercel, and the `turbo.json` pipeline model is simple enough that every developer on the team can understand it without a build systems background.

For JS/TS monorepos with more than 50 projects or complex framework requirements (Angular Enterprise, NestJS + multiple React frontends, custom code generators), **Nx** provides richer tooling at the cost of more configuration.

For JVM monorepos, **Gradle** with the configuration cache and Develocity remote cache is the standard. The affected-target computation requires more custom scripting than Turborepo or Nx, but the build cache performance is excellent once configured.

For large Python monorepos (data engineering, ML platforms, backend service organizations), **Pants** is the best option — it eliminates the BUILD file authoring burden that makes Bazel painful for Python by inferring dependencies from import statements.

For organizations that are already at Google, Uber, or Stripe scale with multiple languages — and that can staff a build infrastructure team — **Bazel** is the correct long-term answer.

## Remote Caching: The Multiplier on Affected-Target Builds

Affected-target builds eliminate redundant work within a single CI run. Remote caching eliminates redundant work across runs — across different PRs, across different branches, and across different runners. Together they produce the largest speedup.

![Turborepo pipeline execution stack: parallel workers sit on top of affected filter, then build graph analysis, then remote cache lookup, then local cache at the base](/imgs/blogs/monorepo-vs-polyrepo-and-scaling-the-pipeline-4.png)

### How content-addressed remote caching works

Remote caching is built on a content-addressed storage model — the same model used by Docker layer caches, Nix package store, and IPFS. For a given build task, the cache key is a hash of its declared inputs. If a future run has the same inputs, the hash matches and the cached output is downloaded and restored instead of rebuilding.

The hash computation is deterministic and complete: the same source files + the same lockfile + the same tool versions + the same declared environment variables = the same hash = the same cached output is valid. Any change to any declared input produces a different hash, which produces a cache miss.

This is the key difference between remote caching and a simple "cache the node_modules directory" approach. Node_modules caching saves time on dependency installation but does not skip builds. Content-addressed task caching skips the entire build computation for packages whose inputs have not changed.

#### What a cache key looks like in practice

A Turborepo cache key is a 20-byte SHA-256 truncated to 40 hex characters. The key for `packages/ui#build` at a given commit might look like:

```
3a9f1c82e5b047d6f2190a4e8c3d5b7f1a2e9c4d
```

This key is the hash of the concatenated inputs described earlier. The cache entry is stored at:

```
# Remote (S3-backed):
s3://turborepo-cache/v8/3a9f1c82e5b047d6f2190a4e8c3d5b7f1a2e9c4d.tar.zst

# Local:
.turbo/cache/3a9f1c82e5b047d6f2190a4e8c3d5b7f1a2e9c4d/
  ├── .meta.json
  └── outputs/
      └── dist/
          ├── index.js
          ├── index.d.ts
          └── index.css
```

When a CI runner queries the remote cache, it sends a HEAD request to the S3 object URL. A 200 response means cache hit; the runner downloads the tar archive and extracts it to restore the build outputs. A 404 means cache miss; the runner executes the task and then uploads the output archive.

The end-to-end latency for a cache hit is dominated by the download: for a typical 2–10 MB dist artifact, a download from S3 in the same AWS region takes 200–800 ms. Compare that to a 3–8 minute build: the cache hit is 99%+ faster.

#### Cache hit rate math with real numbers

The benefit of a remote cache is best expressed as average pipeline time, accounting for the probabilistic mix of hits and misses across all runs in a typical week.

Consider a 40-package monorepo where:
- Average build time per package (build + lint + test): 3 minutes
- Average affected packages per PR: 3 out of 40
- Remote cache hit rate on unaffected packages: 98% (they haven't changed)
- Remote cache hit rate on affected packages: 0% (they changed, so their hash is new)
- PRs per day: 20 (active team)

Without remote cache, with affected builds only:
- Time = 3 affected packages × 3 min + overhead = ~10 minutes per run

With remote cache (accounting for the fact that affected packages are cache misses but unaffected packages are near-100% hits):
- The 37 unaffected packages: cache restore latency ≈ 0.5 min total (37 × 0.8 s)
- The 3 affected packages: full build = 9 min
- Total: ~9.5 min per run (the remote cache saves little here because affected builds already skip unaffected packages)

The remote cache's impact is largest for cross-cutting changes that affect many packages:

| Scenario | Affected packages | Without remote cache | With remote cache | Saving |
|---|---|---|---|---|
| CSS fix in one component | 3 / 40 | 10 min | 9.5 min | 5% |
| API contract change | 15 / 40 | 46 min | 18 min | 61% |
| Lock file upgrade | 40 / 40 | 120 min | 110 min | 8% (first run only) |
| Same PR re-run (identical source) | 3 / 40 | 10 min | 0.5 min | 95% |
| Rebase onto main, no code change | 3 / 40 | 10 min | 0.5 min | 95% |

The "same PR re-run" and "rebase onto main" cases deliver the most dramatic improvements. These are among the most common CI patterns — address review feedback, force-push, re-run CI. Without remote cache, every re-run is a full rebuild. With remote cache, re-runs after a code-identical change are sub-minute.

At 20 PRs/day with an average of 2.5 re-runs per PR (addressing review comments, rebasing), remote caching eliminates 50 full rebuilds per day. At 10 minutes per rebuild × 50 rebuilds = 500 runner-minutes saved daily = 10,500 runner-hours saved per month. At \$0.008/runner-minute, that is \$84,000/month in saved CI compute — for a mid-size active team.

![Remote cache decision flow: CI job hashes its inputs, queries the remote cache store, downloads the artifact on a hit or runs a full build and uploads the result on a miss](/imgs/blogs/monorepo-vs-polyrepo-and-scaling-the-pipeline-5.png)

### Typical hit rates in practice

Remote cache hit rates depend primarily on two factors: how frequently a package's inputs change (source churn rate) and how well the input declarations are scoped (whether irrelevant files are correctly excluded).

A typical production measurement across different scenario types:

| Scenario | Hit rate | Wall-clock saving |
|---|---|---|
| Main branch: documentation-only change | 95–99% | Save all build time |
| Feature PR: touches 2–3 packages out of 40 | 70–85% | Save 37+ packages' build time |
| Cross-cutting refactor: touches 15 packages | 35–50% | Still save 25+ packages |
| First run after a full lockfile upgrade | 5–15% | Minimal; all packages need rebuilds |
| Same package, same code, new PR | 100% | Instant cache hit |

The "same package, same code, new PR" case is the most impactful in practice. Developers frequently create PRs that touch the same set of files as a previous PR (after addressing review feedback, rebasing onto main, or reverting and re-applying a change). Without remote caching, each new PR run triggers a full rebuild. With remote caching, the second run with the same inputs is free — under 2 seconds to restore the cached artifact.

### Measuring cache effectiveness

To measure your remote cache hit rate with Turborepo:

```bash
# Run with --summarize to get per-task cache status
turbo run build --summarize

# Parse the summary for cache statistics
turbo run build --dry=json | jq '
  .tasks | 
  group_by(.cache.status) | 
  map({
    status: .[0].cache.status,
    count: length,
    packages: map(.package)
  })'
```

For Nx, the cache analytics are available in the Nx Cloud dashboard with per-task hit rates, time saved, and historical trends.

### Setting up and tuning the remote cache

For Turborepo, the minimal remote cache configuration:

```bash
# .env.local (gitignored)
TURBO_TOKEN=your-vercel-api-token
TURBO_TEAM=your-team-name

# Or for a self-hosted cache server:
TURBO_API=https://cache.your-company.com
TURBO_TOKEN=your-shared-secret
```

For Nx Cloud:

```bash
# Connect this workspace to Nx Cloud (interactive, one-time setup)
npx nx connect

# Or manually in nx.json:
# "nxCloudAccessToken": "your-nx-cloud-token"
```

For the self-hosted Turborepo cache server with S3 storage, the production configuration:

```bash
# Environment variables for the cache server process
TURBO_TOKEN=shared-secret-matches-client-config
STORAGE_PROVIDER=s3
STORAGE_PATH=turborepo-ci-cache
S3_REGION=us-east-1
S3_ACCESS_KEY=your-iam-access-key
S3_SECRET_KEY=your-iam-secret-key
PORT=3000
BODY_LIMIT=104857600  # 100 MB max artifact size
```

The cache server is stateless — all state lives in S3. You can run multiple instances behind a load balancer for high availability, and the only operational concern is the S3 bucket cost (typically \$10–30/month at medium scale, far less than the CI compute you save).

## Scaling CI for the Large Monorepo: Beyond Affected Builds

Once you have affected-target builds and remote caching in place, the remaining scaling lever is parallelism — running the affected packages concurrently across multiple runners. The goal is to fan out the affected packages across enough runners that the wall-clock time approaches the build time of the slowest single package.

![Pipeline optimization journey: 124-minute baseline cut to 38 minutes with affected targets, 14 minutes with parallelism, 11 minutes with local cache, and 8 minutes with remote cache](/imgs/blogs/monorepo-vs-polyrepo-and-scaling-the-pipeline-6.png)

### The compute-affected → fan-out pattern

The canonical GitHub Actions pattern for dynamic matrix parallelism:

```yaml
name: CI — Affected Matrix

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  NODE_VERSION: "20"

jobs:
  # Job 1: Compute the affected set and output a JSON matrix
  compute-affected:
    name: Compute affected packages
    runs-on: ubuntu-latest
    outputs:
      matrix: ${{ steps.set-matrix.outputs.matrix }}
      has-packages: ${{ steps.set-matrix.outputs.has-packages }}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: npm

      - run: npm ci --ignore-scripts

      - id: set-matrix
        shell: bash
        run: |
          BASE_REF="${{ github.base_ref || 'main' }}"
          git fetch origin "$BASE_REF" --no-tags

          # Use turbo dry-run to get affected packages
          PACKAGES=$(npx turbo run build \
            --filter="...[origin/${BASE_REF}]" \
            --dry=json 2>/dev/null | \
            jq -c '[.tasks[].package | select(. != null)] | unique | sort')

          COUNT=$(echo "$PACKAGES" | jq 'length')
          echo "Detected $COUNT affected package(s): $PACKAGES"

          echo "matrix={\"package\":$PACKAGES}" >> "$GITHUB_OUTPUT"
          echo "has-packages=$([ "$COUNT" -gt 0 ] && echo true || echo false)" \
            >> "$GITHUB_OUTPUT"

  # Job 2: Fan out across the matrix — one job per affected package
  build-and-test:
    name: "${{ matrix.package }}"
    needs: compute-affected
    if: needs.compute-affected.outputs.has-packages == 'true'
    runs-on: ubuntu-latest
    strategy:
      matrix: ${{ fromJson(needs.compute-affected.outputs.matrix) }}
      fail-fast: false  # don't cancel other packages if one fails

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: npm

      - run: npm ci --ignore-scripts

      # Restore Turborepo local cache for this runner
      - name: Restore Turborepo cache
        uses: actions/cache@v4
        with:
          path: .turbo
          key: turbo-${{ matrix.package }}-${{ runner.os }}-${{ hashFiles('**/package-lock.json') }}-${{ github.sha }}
          restore-keys: |
            turbo-${{ matrix.package }}-${{ runner.os }}-${{ hashFiles('**/package-lock.json') }}-
            turbo-${{ matrix.package }}-${{ runner.os }}-

      - name: Run lint, type-check, build, test
        run: |
          npx turbo run lint type-check build test \
            --filter="${{ matrix.package }}" \
            --concurrency=4
        env:
          TURBO_TOKEN: ${{ secrets.TURBO_TOKEN }}
          TURBO_TEAM: ${{ vars.TURBO_TEAM }}

  # Merge job: required status check that passes only when all matrix jobs pass
  ci-success:
    name: CI — all affected packages passed
    needs: [compute-affected, build-and-test]
    if: always()
    runs-on: ubuntu-latest
    steps:
      - name: Check matrix status
        run: |
          if [ "${{ needs.build-and-test.result }}" = "failure" ]; then
            echo "One or more affected packages failed CI"
            exit 1
          fi
          echo "All affected packages passed CI"
```

The merge job pattern — a `ci-success` job that `needs` the matrix jobs and is the actual required status check — is important for repository branch protection rules. GitHub's required status checks work by name; if the matrix generates jobs with names like `"@company/auth"`, `"@company/payments"`, those names are dynamic and cannot be pre-registered as required checks. The `ci-success` job has a stable name and can be the single required check.

### Controlling the blast radius: input exclusions

In practice, not all files in a package should bust the build cache. A common source of unnecessary cache invalidation is non-source files that are part of the package directory but irrelevant to the build:

```json
{
  "pipeline": {
    "build": {
      "dependsOn": ["^build"],
      "inputs": [
        "src/**",
        "package.json",
        "tsconfig.json",
        "tsconfig.build.json",
        "!src/**/*.spec.ts",
        "!src/**/*.test.ts",
        "!src/**/*.stories.tsx"
      ],
      "outputs": ["dist/**"]
    },
    "test": {
      "inputs": [
        "src/**",
        "jest.config.ts",
        "!src/**/*.stories.tsx"
      ],
      "outputs": ["coverage/**"]
    }
  }
}
```

Excluding test files from build inputs means that adding or modifying a test does not bust the build cache — only the test cache. Excluding `.stories.tsx` files from both build and test inputs means that adding Storybook stories does not invalidate CI builds or tests. These exclusions compound: on a 40-package monorepo, every file category you correctly exclude from inputs is one more file type that developers can change without triggering unnecessary rebuilds.

### Sharding for very large affected sets

For cross-cutting changes that affect 20+ packages, even parallelism has limits — 20 concurrent jobs consume 20 runners simultaneously, which at \$0.008/runner-minute and 15 minutes each costs \$2.40 per run. Beyond 30–40 concurrent jobs, runner availability on shared GitHub-hosted runners becomes a practical bottleneck (job queue delays).

The solution is sharding: instead of one job per package, group packages into shards of N and assign a runner per shard:

```yaml
- id: build-matrix
  run: |
    AFFECTED_PACKAGES=${{ needs.compute-affected.outputs.packages }}
    SHARD_SIZE=5

    # Group packages into shards of SHARD_SIZE
    SHARDS=$(echo "$AFFECTED_PACKAGES" | jq -c \
      --argjson size "$SHARD_SIZE" '
        to_entries |
        group_by(.key / $size | floor) |
        map(map(.value))
      ')
    echo "matrix={\"shard\":$SHARDS}" >> "$GITHUB_OUTPUT"
```

Each shard job runs `turbo run build test --filter=package-A --filter=package-B ...` across its assigned packages in parallel within a single runner, using Turborepo's `--concurrency` flag to utilize all available CPUs.

## Two Worked Examples

The abstract performance claims above become real when you trace through specific scenarios with specific numbers.

#### Worked example:

**A 20-package TypeScript monorepo — PR changes 1 package, affected set = 3.**

The repository has the following package topology (simplified):

```
packages/
  tokens/         # design tokens
  ui/             # component library — depends on tokens
  auth/           # auth utilities
  api-client/     # typed API client — depends on auth
  forms/          # form components — depends on ui
  analytics/      # analytics utilities (no local deps)
  testing-utils/  # test helpers
  ...13 more independent packages

apps/
  web/            # Next.js app — depends on ui, api-client, forms, analytics
  admin/          # React admin — depends on ui, auth, analytics
  docs/           # Docusaurus — depends on ui, tokens
```

A developer opens a PR that changes one file in `packages/ui`: `src/Button.tsx` — a one-line change to the default padding value.

Turborepo computes the affected set:

1. `packages/ui` — source file changed, hash is new → cache miss.
2. `packages/forms` — declared `"@company/ui"` in `package.json` dependencies → its hash input now includes the new `ui` hash → cache miss.
3. `apps/web` — declared `"@company/ui"` and `"@company/forms"` as dependencies → cascades from both → cache miss.
4. `apps/admin` — declared `"@company/ui"` → cache miss.
5. `apps/docs` — declared `"@company/ui"` → cache miss.

Five packages are affected out of 20. The other 15 — `tokens`, `auth`, `api-client`, `analytics`, `testing-utils`, and the 9 remaining independent packages — have no dependency path to `packages/ui`. Their hashes are unchanged. They are cache hits: Turborepo downloads their previously computed build artifacts in ~1 second each and marks them as restored.

Pipeline time breakdown:

| Phase | Without Turborepo | With Turborepo + remote cache |
|---|---|---|
| `packages/tokens` through 15 unaffected packages | 15 × 2 min = 30 min | 15 × 1 sec = 0.25 min (cache restore) |
| `packages/ui` build | 3 min | 3 min (cache miss, full build) |
| `packages/forms` + `apps/*` (4 packages, parallelized) | 4 × 3 min = 12 min | 4 × 3 min = 12 min (all cache misses) |
| Overhead (checkout, install, upload) | 3 min | 3 min |
| **Total wall-clock time** | **~45 min** (with 4× parallelism) | **~8 min** |

The \$0.008/runner-minute cost: the naive 45-minute pipeline costs \$0.36 per run × 60 daily runs = \$21.60/day = \$648/month. The affected + cached pipeline at 8 minutes costs \$0.064 per run × 60 daily runs = \$3.84/day = \$115/month. Monthly saving: \$533.

---

#### Worked example:

**A polyrepo with 40 services — coordinating a shared library breaking change with and without Renovate.**

The organization has 40 service repos, all depending on `@company/http-client`. The library adds a new required option `timeout` to every outbound request, replacing an undeclared default. It is a security requirement (the undeclared default was infinite, and the security team requires all services to declare an explicit timeout).

**Scenario A: Manual coordination, no tooling.**

The library maintainer publishes `http-client@3.0.0`, writes a migration guide, and opens 40 PRs manually (using a script that clones each repo, applies a sed replacement, and pushes a branch).

- Day 1: 40 branches pushed, 40 PRs opened. 8 are merged by the end of the day (the highly active teams).
- Day 3: 18 merged. 12 have CI failures — the sed replacement missed a few call sites in services that had wrapper functions around the HTTP client. Each requires a manual follow-up.
- Day 7: 26 merged. 6 services report that their test suites were not covering the HTTP client call sites and the sed replacement went undetected but wrong.
- Day 14: 33 merged. 5 services have open PRs with review comments still unaddressed. 2 services have not reviewed the PR at all.
- Day 21: 37 merged. 3 services are stuck — two have their own breaking changes in flight that conflict with this PR; one team lead has been out on leave.

Total engineering cost: the library maintainer spent 3 days writing and reviewing PRs. Each team spent an average of 1.5 hours on their own PR. Total: 24 hours (maintainer) + 40 × 1.5 hours (teams) = 84 engineering-hours. Wall-clock time to full compliance: 3 weeks. 3 services still not compliant at the end of the window.

**Scenario B: Renovate bot, monorepo migration not yet done.**

The same organization with Renovate configured. Renovate opens automated PRs in all 40 repos within 4 hours of the `3.0.0` release on npm. Each PR includes the version bump and, because the migration guide includes a jscodeshift codemod referenced in the `package.json` `codemod` field, Renovate applies the codemod automatically.

- Day 1: 40 automated PRs open, CI running. 22 pass CI immediately — the codemod handled all call sites correctly. They are auto-merged within 6 hours (configured `automerge: true` for minor/patch, but this is a major version so it requires manual approval).
- Day 2: 8 more approved and merged after a 5-minute review confirming the codemod diff looks correct. 10 have CI failures — deeper call-site patterns that the codemod missed. Renovate labels them `requires-manual-attention`.
- Day 3: 8 of the 10 failing PRs are fixed by their team leads in 30–45 minutes each (the codemod missed call sites in deeply nested async utility functions). 2 require more extensive refactoring.
- Day 5: 38 of 40 services on `3.0.0`. 2 services have scheduled remediation.

Total engineering cost: 38 services × 15 minutes (reviewing an automated PR) = 9.5 hours. 10 services × 45 minutes (fixing CI) = 7.5 hours. Total: ~17 engineering-hours. Wall-clock time: 5 days, 95% compliance.

**Scenario C: Monorepo with Turborepo, same change.**

After migrating all 40 services into a single monorepo:

```bash
# Apply the codemod across all services at once
npx jscodeshift -t codemods/http-client-v3.js \
  services/ apps/ \
  --extensions=ts,tsx \
  --parser=tsx

# Verify the codemod's changes
git diff --stat  # shows 40 changed service directories

# Run affected CI locally to verify before opening PR
npx turbo run build test --filter=...[HEAD~1]
# Turborepo identifies all 40 services as affected (they all import http-client)
# Runs all 40 in parallel across all CPUs: ~18 minutes

# Open one PR covering all 40 services
gh pr create --title "security: migrate http-client to v3.0.0 (require explicit timeout)"
```

CI on the PR builds and tests all 40 affected services in parallel across 8 concurrent runners. The review is a single diff showing the mechanical codemod pattern applied consistently across all services.

Total engineering cost: 4 hours (library maintainer) + 2 hours (PR review) = 6 engineering-hours. Wall-clock time: 6 hours from library change to merged PR with full compliance proven by CI.

| Approach | Engineering cost | Time to full compliance | Remaining non-compliant |
|---|---|---|---|
| Manual polyrepo coordination | 84 hours | 3 weeks | 3 services (7.5%) |
| Renovate + polyrepo | 17 hours | 5 days | 2 services (5%) |
| Monorepo + Turborepo | 6 hours | 6 hours | 0 services (0%) |

The monorepo approach is 14× faster in wall-clock time and 14× cheaper in engineering cost than manual polyrepo coordination. The compliance rate is also categorically better: the monorepo PR cannot be merged until all 40 services pass CI. There is no "we will get to it later" — the PR is either green (all services pass) or it is not merged.

---

## War Story: Google, Meta, and the Real Limits of Each Structure

### Google's monorepo: 1 billion lines of code

Google has maintained a single monorepo — the "Google3 repository" — since the late 1990s. As of published reports circa 2016, it contained approximately 1 billion lines of code across 25 languages, used by 25,000+ engineers making 45,000+ commits per day. The enabling technology is Blaze (the internal predecessor to open-source Bazel).

The key insight from Google's experience is not that monorepos are universally better. It is that **the investment in explicit BUILD file maintenance scales linearly with repo size, but the benefit of correct affected-target computation grows superlinearly**. At 25,000 engineers, even a 0.1% false positive rate in the affected set — building packages that did not actually need to be rebuilt — wastes 25 engineer-build-minutes per commit. At 45,000 commits per day, 0.1% false positives cost 1,125 hours of compute per day. The hermetic precision of Bazel BUILD files is worth the authoring cost at that scale.

Google also runs Critique (their code review system), Borg (their cluster scheduler), and an internal CI system called TAP (Test Automation Platform) that runs affected tests within minutes of a commit landing — across the entire billion-line repository. The scale is only possible because every build step is precisely declared and can be distributed across thousands of machines.

Google's monorepo experience also produced practical organizational insights that are worth carrying to any scale:

- **Trunk-based development is the only viable model at monorepo scale.** Long-lived feature branches on a billion-line repo would produce merge conflicts that take weeks to resolve. Google enforces merging to trunk daily, using feature flags to disable incomplete features in production.
- **Ownership metadata is required.** Every directory in Google's monorepo has an `OWNERS` file that lists who must approve changes to that directory. Without this, code review at 25,000 engineers would be chaos. The `OWNERS` model is now encoded in the open-source Gerrit code review system and GitHub's CODEOWNERS.
- **Atomic refactoring requires scale tooling.** Google developed a system called "Rosie" that takes a large-scale automated refactoring, shards it into hundreds of independent PRs (each touching one independent subtree of the codebase), reviews them in parallel, and lands them atomically in aggregate. For most organizations, a single PR with a codemod is sufficient — but at Google's scale, even a single codemod PR would be too large for any human reviewer to reason about.

For organizations at less than 1/1000th of Google's scale — which is most — the full BUILD file authoring discipline of Bazel is not justified. But the principle carries: the investment in a build graph tool (whatever its complexity level) pays off at any scale where the naive "build everything" pipeline has become a bottleneck.

### Meta's Mercurial monorepo and the Sapling fork

Meta has maintained a monorepo for its core services and products for over a decade. Their experience illuminates a scaling problem that most organizations never encounter but should be aware of: **source control performance degrades before build performance does**.

At around 50,000 files, standard `git status` starts to feel sluggish — 3–5 seconds instead of sub-second. At 200,000 files, `git checkout` and `git blame` become painful. At Meta's scale (millions of files), they are unusable without special tooling. Meta responded by contributing to Mercurial (adding lazy loading and virtual filesystem features), then ultimately forking Mercurial into an open-source project called Sapling (`sl`). Sapling uses partial checkout (clients only materialize the subtrees they work on) and a virtual filesystem (Eden) to serve source files on-demand rather than syncing the entire repository to each developer's machine.

For most organizations — below 100 services, below 500,000 files — standard Git is fine. The pain point arrives much later than build performance issues. But if you are planning a monorepo that might grow to Meta's scale, the source control layer needs architectural investment too, not just the build layer.

A practical mitigation available to ordinary GitHub users today: sparse checkout and partial clone:

```bash
# Partial clone: skip downloading large blobs until needed
git clone --filter=blob:none git@github.com:company/monorepo.git

# Sparse checkout: only materialize the subtrees you need
git sparse-checkout set services/auth libs/common
git sparse-checkout add services/payments

# Sparse checkout in cone mode (faster, simpler)
git sparse-checkout init --cone
git sparse-checkout set services/auth libs/common
```

Sparse checkout is available in Git 2.25+ and works with GitHub, GitLab, and Bitbucket without any special server-side infrastructure. For a 200-service monorepo where a developer works on 2–3 services, sparse checkout reduces their working tree from 200 service directories to 3, making standard Git operations fast again.

### The realistic mid-scale migration story

A fintech startup grows from 5 to 50 engineers over 18 months. They start with 8 service repos, accumulate to 32 service repos plus 8 shared library repos. Every quarter, the platform team spends 3 weeks migrating all 32 services to the latest versions of the 8 shared libraries. The trigger for a monorepo migration is a security incident: an auth library has a critical vulnerability. The patch requires a breaking API change. The platform team needs to roll out the fix across all 32 services within 48 hours to meet their security SLA.

In the polyrepo model: 32 PRs opened on day 1. By hour 24, 19 are merged. The other 13 are stalled — merge conflicts from other branches in flight, CI failures from test suites that had not been run since the previous bump, and three teams whose on-call engineers are occupied with incidents. The security SLA is missed.

After the migration to a monorepo with Turborepo and affected CI:

```bash
# Patch the auth library
vim libs/auth/src/validator.ts  # apply the security fix

# Run a codemod to update call sites if the API changed
npx @codemod/cli --codemod auth/validate-v3 ./services ./apps

# CI automatically runs affected builds and tests
# One PR, one review, merged within 4 hours
```

**Measured outcomes at the end of year 1:**

| Metric | Polyrepo (before) | Monorepo (after) |
|---|---|---|
| Security patch rollout time | 48–96 hours | 4–6 hours |
| Quarterly dependency upgrade cost | 3 weeks engineering | 4 hours (automated) |
| Average CI wait time per PR | 45 minutes | 9 minutes |
| Monthly CI compute cost | \$3,200 | \$280 |
| Broken deps detected at merge time | 8% of merges | <0.1% of merges |

The cost reduction in CI compute is 91%. The reduction in quarterly upgrade cost is 95%. The security posture improvement — measured by time-to-patch SLA compliance — is the most strategically valuable outcome.

### Microsoft's Windows monorepo: Git at a different scale

Microsoft faced the same VFS problem as Meta but from the opposite direction: they had a massive polyrepo-like codebase (the Windows source tree) that they wanted to migrate to Git without rebuilding their source control infrastructure from scratch. The Windows tree contains over 3 million files and 3.5 TB of source code — too large to clone conventionally.

Microsoft's solution was Git Virtual File System (GVFS), later renamed to Scalar and donated to the Git project. Scalar works by intercepting filesystem calls from Git and serving only the files that the developer actually accesses, fetching the rest on demand from the server. A developer whose feature touches 50 files never downloads the other 2,999,950 files to their local machine.

The lessons from Microsoft's experience are directly applicable to large monorepos at more modest scales. The core techniques — background prefetch (fetching likely-needed data proactively), filesystem monitoring (tracking which files Git needs to scan), and partial clone (skipping large blob downloads at clone time) — are all available in standard Git 2.38+ via the `--scalar` option or as separate configuration.

```bash
# Clone a large monorepo with Scalar optimizations
git clone --filter=blob:none --sparse \
  git@github.com:company/monorepo.git

# Enable Scalar maintenance (background prefetch, commit-graph, etc.)
git maintenance start

# Enable filesystem monitor for faster status/add/commit
git config core.fsmonitor true
git config core.untrackedCache true
```

These optimizations are transparent to the build layer — Turborepo, Nx, and Bazel continue to work identically whether or not Scalar optimizations are enabled. They address only the source control performance layer, not the build graph layer.

## Dependency Upgrade: A Full Worked Example

To make the polyrepo coordination tax concrete, here is a detailed walkthrough of a breaking change migration in both structures.

**Setup:** 40 service repos, all depend on `@company/auth-utils`. The library needs a breaking change: `validateToken(token: string)` becomes `validateToken(token: string, options: ValidationOptions)`. `options` is required, not optional, because the existing callers that don't pass it were inadvertently using insecure defaults.

**Polyrepo upgrade — step by step:**

Day 1:
- Author writes the breaking change, publishes `auth-utils@2.0.0`, writes a migration guide.
- Opens PRs in the 5 highest-priority services. Each PR: find the 3–8 call sites, update the signatures, ensure tests pass.
- 4 hours of work for 5 PRs.

Day 3:
- 8 more PRs opened. 2 of the original 5 merged. The other 3 have review comments requesting additional test coverage.
- One PR fails CI because the service had already bumped to `auth-utils@1.4.0-beta.1` via Dependabot and now has a conflict.
- 2 service teams report that `validateToken` is called in a shared utility they own — they need to figure out whether their utility should be updated to pass through `options` or hardcode the secure defaults. This is a design decision that requires a synchronous discussion.

Day 7:
- 20 PRs merged. 12 open. 8 not started.
- A support ticket arrives: `service-payments` is running `auth-utils@1.8.2` which has a different security vulnerability. The bump to `2.0.0` would fix it — but `service-payments` has a hard freeze on external dependency changes until their Q3 financial reporting period ends in 10 days.

Day 15:
- 30 PRs merged. 8 services still on v1.x.
- The library maintainer sends a deprecation notice for `auth-utils@1.x` with a 60-day sunset.

Day 22:
- 36 of 40 services on v2.0.0.
- 4 services are still on v1.x: two have genuinely complex migration paths (their call sites pass `token` through 6 nested call-stack layers), one is frozen for external reasons, one has been silent for two weeks.

Total: 88 engineering-hours over 22 days, with 4 services not yet compliant at the end of the scheduled migration window.

**Monorepo upgrade — complete sequence:**

```bash
# Step 1: Make the breaking change in-tree
# Edit libs/auth-utils/src/index.ts to update the function signature
# Edit libs/auth-utils/src/validator.ts with the new implementation

# Step 2: Run a codemod to update all call sites mechanically
# (write or find a codemod for this specific API change)
cat > codemods/auth-utils-v2.js << 'EOF'
module.exports = function transformer(fileInfo, api) {
  const j = api.jscodeshift;
  return j(fileInfo.source)
    .find(j.CallExpression, {
      callee: { type: 'Identifier', name: 'validateToken' }
    })
    .filter(path => path.node.arguments.length === 1)
    .forEach(path => {
      path.node.arguments.push(
        j.objectExpression([
          j.property(
            'init',
            j.identifier('requireExpiry'),
            j.literal(true)
          )
        ])
      );
    })
    .toSource();
};
EOF

npx jscodeshift -t codemods/auth-utils-v2.js \
  services/ apps/ \
  --extensions=ts,tsx \
  --parser=tsx \
  --dry  # preview first

npx jscodeshift -t codemods/auth-utils-v2.js \
  services/ apps/ \
  --extensions=ts,tsx \
  --parser=tsx  # apply

# Step 3: Verify the automated changes look correct
git diff --stat  # should show changes in every service that calls validateToken

# Step 4: Run CI locally to verify
npx turbo run build test --filter=...[HEAD~1]

# Step 5: Open one PR
gh pr create \
  --title "security: migrate auth-utils to v2.0.0 (require explicit expiry)" \
  --body "Breaking change migration handled by codemod. CI passes on all 37 affected packages."
```

CI runs automatically on the PR, building and testing all 37 affected packages in parallel. The reviewer sees a clean diff with a mechanical codemod pattern and a passing CI run covering every affected service simultaneously.

Timeline: 16 engineering-hours. One PR. Full compliance on merge.

![Polyrepo upgrade spans 3 engineering-weeks with 40 separate manual PRs versus monorepo upgrade completes in 1 automated PR with all affected services updated simultaneously](/imgs/blogs/monorepo-vs-polyrepo-and-scaling-the-pipeline-7.png)

The difference is structural. In the monorepo, the upgrade is provably complete at merge time — CI covered all consumers. In the polyrepo, the upgrade is declared done when no one is watching anymore and the author stops sending reminders.

## Build Tool Selection Framework

With the full ecosystem surveyed, a decision framework:

![Build tool selection tree: first branch on hermetic reproducibility, then by language ecosystem, leading to Bazel, Nx, Turborepo, Gradle/Maven, or Pants](/imgs/blogs/monorepo-vs-polyrepo-and-scaling-the-pipeline-8.png)

The first question is always: do you need hermetic reproducibility — guaranteed byte-for-byte identical builds given the same inputs? This is a hard requirement in regulated industries and best practice for software with provenance attestation requirements. If yes, Bazel is the only production-proven option at significant scale.

If hermetic sealing is not required, the decisive factor is language ecosystem:

| Tool | Primary language | Complexity | Affected-target mechanism | Remote cache |
|---|---|---|---|---|
| Turborepo | JS/TS | Low | `--filter=[base]` | Vercel / self-hosted |
| Nx | JS/TS + Angular/React | Medium | `nx affected` | Nx Cloud / self-hosted |
| Gradle | Java, Kotlin, Android | Medium | Custom script + build cache | Develocity |
| Maven | Java (legacy projects) | Low-Medium | Custom script + local/S3 | Develocity / custom |
| Pants | Python, Go, Java | Medium | `--changed-since` + `--changed-dependees` | Built-in remote |
| Buck2 | Polyglot | High | `buck2 query rdeps()` | RE-integrated |
| Bazel | Polyglot | High | `bazel query rdeps` | BES-compatible server |

The default recommendation for new JavaScript/TypeScript monorepos is **Turborepo**. Setup time is 2–4 hours, the remote cache works out of the box with Vercel, and the `turbo.json` pipeline model is simple enough that every developer on the team can understand it without a build systems background.

For JS/TS monorepos with more than 50 projects or complex framework requirements (Angular Enterprise, NestJS + multiple React frontends, custom code generators), **Nx** provides richer tooling at the cost of more configuration.

For JVM monorepos, **Gradle** with the configuration cache and Develocity remote cache is the standard. The affected-target computation requires more custom scripting than Turborepo or Nx, but the build cache performance is excellent once configured.

For large Python monorepos (data engineering, ML platforms, backend service organizations), **Pants** is the best option — it eliminates the BUILD file authoring burden that makes Bazel painful for Python by inferring dependencies from import statements.

For organizations that are already at Google, Uber, or Stripe scale with multiple languages — and that can staff a build infrastructure team — **Bazel** is the correct long-term answer.

## When NOT to Use Each Structure

This section deserves explicit treatment because the Internet is full of uncritical advocacy for both approaches.

### When monorepo works against you

**Without affected-target builds, a monorepo is actively harmful.** A monorepo running naive `npm run build --workspaces` does not give you the benefits of a monorepo — it gives you the costs (long CI, merged blast radius) with none of the benefits. If your organization is considering a monorepo migration but is not prepared to invest in a build tool and a remote cache, the correct answer is to stay polyrepo. The migration investment is real: plan for 1–4 engineering-weeks depending on repo size and build tool choice.

**With widely divergent tech stacks, a monorepo adds friction without commensurate benefit.** If one team uses Python with Poetry, another uses Go modules, and a third uses Java with Maven, a monorepo does not automatically unify their builds. You need a polyglot build tool (Bazel or Pants) to get cross-language affected-target builds. Without it, you have three separate build systems coexisting in one repo, which is more confusing than three separate repos. The colocation benefit (easier cross-team code sharing) may not outweigh the unified-tooling cost at 3 services.

**At 2–3 services, a monorepo is premature investment.** The coordination tax of a polyrepo at 2 services is essentially zero. Bumping a shared library across 2 repos takes 30 minutes. The investment in monorepo tooling at this scale returns nothing for years. Default to polyrepo for early-stage projects and migrate deliberately when the dependency coordination pain becomes measurable — a concrete threshold is when you spend more than 2 engineering-days per month on cross-repo dependency upgrades.

**With strong team ownership boundaries that require independent deployment, monorepo governance becomes complex.** If Team A and Team B have independent deployment schedules, separate security review requirements, and different SLAs, sharing a monorepo creates policy surface area. CODEOWNERS files, required reviewers per directory, merge queue policies that gate cross-team changes — these are solvable, but they add operational overhead that does not exist in a polyrepo. The benefit must be worth it.

**With secret-dependent builds that cannot share a remote cache,** the remote cache benefit is lost. If each team's build requires secrets that cannot be shared (API keys, internal certificates, customer-specific configuration), you cannot use a shared remote cache — each team's cache must be isolated, which reduces hit rates significantly.

### When polyrepo works against you

**With more than ~10 shared libraries, polyrepo dependency management becomes a full-time job.** The coordination cost scales with the number of (library, consumer) pairs. At 5 shared libraries × 20 consumers = 100 pairs: manageable. At 15 libraries × 40 consumers = 600 pairs: not manageable without dedicated automation infrastructure that is equivalent in complexity to a monorepo build system.

**When you need atomic cross-service refactoring, polyrepo is structurally impossible.** Renaming a gRPC service method called from 30 services requires 30 separate PRs in a polyrepo. In a monorepo, it is one PR with a codemod and CI proof. If your architecture evolves through cross-cutting refactors — which is common in the first 3 years of a product — polyrepo imposes a structural friction that compounds over time.

**When onboarding is a bottleneck, polyrepo discovery overhead is significant.** A new engineer joining a team with 40 repos needs to discover which repos are relevant, clone them, understand their interdependencies, and set up their local development environment for each. With a monorepo and sparse checkout, the same engineer clones one repository and checks out the 3–5 subtrees relevant to their work. This difference compounds at scale: a 100-person engineering organization onboards 20+ engineers per year, and onboarding friction has a real cost.

**When your CI is already fast (< 5 minutes per repo), polyrepo's speed advantage is negligible.** The strongest argument for polyrepo CI is speed. If each service builds in 2 minutes, the "polyrepos are faster" argument holds. If each service takes 15 minutes because of large test suites and docker builds, the speed advantage is gone — you now have 40 slow pipelines instead of one slow pipeline. The monorepo with affected builds would be faster.

## The Decisive Criteria: A Decision Framework

Strip away the tooling discussions and the war stories. Three questions determine the correct answer:

**1. How frequently do you make cross-cutting changes?**

If more than 10% of your significant PRs touch more than one independently deployed unit, the coordination cost of polyrepo will compound. Monorepo wins.

If your services are genuinely independent — each team deploys on its own schedule, uses its own tech stack, and rarely modifies shared code — the coordination cost is low. Polyrepo isolation is a net positive.

A rough measurement: count the number of PRs in the last quarter that required synchronized changes across multiple repos (coordinating merges, bumping versions in both repos simultaneously). If that number exceeds 20% of total PRs, the polyrepo coordination cost is already significant.

**2. Are you willing and able to invest in a build tool?**

A monorepo without Turborepo, Nx, Bazel, or Gradle with caching enabled is just a slow polyrepo. The tooling investment is non-optional. If the answer is "we will get to it later," the practical answer is "stay polyrepo for now."

The investment estimate by organization size:

- 3–15 engineers: 1 engineer-week to set up Turborepo + Vercel cache + GitHub Actions matrix.
- 15–50 engineers: 2–3 engineer-weeks for Nx or Turborepo with self-hosted cache.
- 50–200 engineers: 4–8 engineer-weeks for full Nx/Bazel setup, ongoing 0.25 FTE maintenance.
- 200+ engineers: dedicated build infrastructure team (2–5 engineers).

**3. What does your organizational topology look like?**

Team topology is the often-overlooked factor. Conway's Law applies to repo structure: the structure you choose will be reflected in your software architecture and organizational communication patterns. A monorepo encourages (and technically enables) cross-team code sharing and cross-cutting refactors. A polyrepo enforces (and technically mandates) explicit API contracts between teams.

### Decisive criteria comparison table

The following table collapses the decision space to the factors that most reliably predict which structure will work for a given organization:

| Factor | Favors Monorepo | Favors Polyrepo |
|---|---|---|
| Team size | 5–200 engineers on a shared platform | Under 5 (premature) or over 500 (consider federated) |
| Repo age | Starting fresh or migrating deliberately | Mature polyrepo with well-established team ownership |
| Language ecosystem | Single language (JS/TS, Python, JVM) | Multiple languages with no polyglot build tool |
| Shared library count | More than 5 actively maintained shared libs | Fewer than 3 shared libs, stable interfaces |
| Cross-cutting change frequency | More than 10% of PRs touch multiple services | Under 5% of PRs cross service boundaries |
| Deployment coupling | Services typically deploy together or in waves | Services deploy fully independently on different cadences |
| Team ownership | Fluid teams, shared platform ownership | Strong ownership boundaries, independent team SLAs |
| Security/compliance | Unified policy enforcement required | Isolated compliance scopes per service/team |
| Onboarding volume | More than 10 engineers/year, high discovery cost | Small stable team with deep per-repo knowledge |
| Build tool investment willingness | Team is ready to invest 1–4 weeks in tooling | No dedicated build infrastructure capacity |

No single factor is decisive in isolation. The table is a checklist: count the columns. If 7+ factors favor monorepo, choose monorepo. If 7+ factors favor polyrepo, choose polyrepo. For a split decision, the cross-cutting change frequency and shared library count are the two highest-signal predictors.

| Team topology | Recommendation |
|---|---|
| Single product team, < 20 engineers | Monorepo + Turborepo |
| Multiple product teams sharing a platform | Monorepo + Nx, or federated monorepos per domain |
| Independent product teams, minimal shared code | Polyrepo per team |
| Acquisitions integrating into an existing org | Polyrepo temporarily, migrate on a schedule |
| Open-source project with external contributors | Polyrepo (isolation is a feature for external contributors) |

The federated model — multiple domain-scoped monorepos (platform monorepo, product monorepo, data monorepo) with explicit versioned API contracts between domains — is what large organizations converge on when a single monorepo becomes too large for one build graph but polyrepo dependency chaos across domains is also unacceptable.

## Key Takeaways

The monorepo versus polyrepo debate has exactly one correct answer: it depends on your tooling investment and coordination topology. Neither structure is inherently superior. Each structure imposes a coordination tax; the question is which tax you are better equipped to pay.

The 2-hour monorepo pipeline is not a fundamental property of monorepos. It is a symptom of a missing build graph — a "build everything" pipeline applied to a repository that has grown beyond 5–10 packages without the tooling investment to match. The fix is a build graph tool (Turborepo, Nx, Bazel, Gradle with caching, or Pants) and a remote cache. The fix is not to split into a polyrepo.

Affected-target builds — Turborepo's `--filter=[base]`, Nx's `nx affected`, Bazel's `rdeps`, Pants' `--changed-since` — reduce CI time by 85–95% on average for real-world codebases. They work by computing which packages are in the reverse-dependency subgraph of the changed packages and running only those.

Remote caching extends the benefit across runs and across runners. A 74% remote cache hit rate on ephemeral runners is typical for a stable codebase and translates to a 74% reduction in build compute cost. At \$0.008/runner-minute, even a 40-package monorepo with 80 daily runs saves over \$2,000/month.

The polyrepo coordination tax compounds with the number of (shared library, consumer) pairs. At 600+ pairs, manual dependency upgrade coordination costs 300+ engineering-hours per quarter. Monorepos make that coordination atomic and automatic.

Choose monorepo when: cross-cutting changes are frequent, shared library proliferation is already causing upgrade overhead, atomic refactoring across services is needed, or onboarding friction from repo discovery is measurable. Invest in the build tool before migrating — the tooling is the prerequisite, not an afterthought.

Choose polyrepo when: services are genuinely independent with distinct tech stacks, the team structure has strong ownership boundaries that require blast-radius isolation, you have fewer than 3–5 services with minimal shared code, or you are not yet ready to invest in a build tool.

When you choose monorepo, invest in affected-target builds on day one. The 2-hour pipeline is not a growing pain to be tolerated — it is a direct cost to developer productivity and CI budget that has a known, well-documented solution.

## Further Reading

Posts in this series most directly connected to this topic:

- [From Commit to Production: The CI/CD Mental Model](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) — the foundational model for the entire delivery pipeline. If you are new to CI/CD, start here before diving into monorepo-specific scaling.
- [Runners, Caching, and the CI Cost Problem](/blog/software-development/ci-cd/runners-caching-and-the-ci-cost-problem) — the mechanics of runner selection, artifact caching strategies, and controlling CI spend at scale. Read this alongside the remote cache section of this post.
- [CI/CD and Independent Deployability](/blog/software-development/microservices/ci-cd-and-independent-deployability) — how microservice architecture principles interact with CI/CD pipeline design. Directly relevant if your monorepo contains independently deployable services with different release cadences.
- The CI/CD Playbook capstone (coming soon at `/blog/software-development/ci-cd/capstone-the-cicd-playbook`) — the complete reference for every topic in this series: affected builds, caching, deployment strategies, GitOps, rollbacks, and progressive delivery.

**Primary sources and reference material:**

Turborepo's `turbo.json` pipeline reference and remote cache server documentation are at [turbo.build/repo/docs](https://turbo.build/repo/docs). The `--filter` syntax documentation is the most important reference for affected-target builds in Turborepo.

Nx's affected computation documentation and project graph visualization are at [nx.dev/nx-api/nx/documents/affected](https://nx.dev/nx-api/nx/documents/affected). The `nrwl/nx-set-shas` GitHub Action documentation explains how `NX_BASE` is computed in CI.

Bazel's remote caching documentation at [bazel.build/remote/caching](https://bazel.build/remote/caching) covers the cache protocol, authentication, and the open-source compatible cache server implementations.

Rachel Potvin and Josh Levenberg, "Why Google Stores Billions of Lines of Code in a Single Repository" (ACM Communications, 2016) is the definitive academic description of Google's monorepo approach. It covers the BUILD file authoring discipline, the Piper source control system, and the cultural practices that make the model work at 25,000 engineers.

The [Sapling SCM documentation](https://sapling-scm.com) describes Meta's approach to source control at monorepo scale, including the Eden virtual filesystem, partial checkout, and the EdenAPI server protocol.

[Microsoft Scalar](https://github.com/microsoft/scalar) provides Git partial clone, background prefetch, and filesystem monitoring optimizations for large monorepos. It is what enables the Windows source tree (3+ million files) to be hosted in a single Git repository on GitHub.

The `turborepo-remote-cache` open-source cache server at `github.com/ducktors/turborepo-remote-cache` is the self-hosted alternative to Vercel's managed cache. It supports S3, GCS, and local filesystem storage backends and is deployable as a Docker container in any environment.
