# CI/CD & Cloud-Native Delivery, From Commit to Production — A Field Manual

Series of **40 deep-dive posts** in NEW folder **`content/blog/software-development/ci-cd/`** (subcategory
`CI/CD & Delivery`). **blog-writer** voice (principal platform/release engineer; intuition → principle → runnable
pipeline artifact → measured proof → war stories), **English**, ≥ 11,000 words, 8 figures each, `.png` embeds +
`optimize-blog-images`. Commit + push **each wave** (explicit paths, never `git add -A`). Date 2026-06-22.

## Angle (non-negotiable)
**"Get code from a commit to production safely and fast — and make the path itself a versioned, measurable
engineering artifact."** Three-things-per-post: principle (why) + practice (runnable YAML/Dockerfile/Terraform/Helm/
Argo) + proof (before→after: lead time, deploy frequency, change-fail rate, MTTR, build minutes, image size, CI
cost). Spine: **commit → build → test → package → deploy → operate**, governed by **"build once, promote everywhere"**
+ **"everything as code"**, measured by the **four DORA metrics**.

**ANTI-OVERLAP:** the *delivery toolchain* layer — how to ENGINEER the pipeline. Cross-links OUT (never re-derives) to
the shipped SRE series (reliability theory of deploying: progressive-delivery-as-SLO-gate, rollback-as-mitigation,
chaos, error budgets), microservices (`ci-cd-and-independent-deployability`, `deployment-strategies-...`),
system-design, database (`zero-downtime-schema-migrations`), version-control (git), debugging (flaky tests). Kit:
`.cache/blog-writer/_cicd-series-kit.md`; render reuses `_render-debug.sh` (slug-driven). Intro =
`from-commit-to-production-the-cicd-mental-model`; capstone = `capstone-the-cicd-playbook`.

---

## Track A — Foundations: the path from commit to production (Wave 1, 6)
A1 `from-commit-to-production-the-cicd-mental-model` — **Series intro.** CI vs continuous delivery vs continuous deployment; the pipeline as a value stream; the four DORA metrics; "build once, promote everywhere" + "everything as code"; the master map the series fills in.
A2 `continuous-integration-merge-early-merge-often` — Real CI: trunk-based vs long-lived branches, small batches, integration hell, PR checks + branch protection, why merging often lowers change-fail rate.
A3 `the-build-stage-reproducible-fast-and-cacheable` — Build systems, layer/dependency caching, reproducible & hermetic builds, the slow-build tax, incremental + remote cache (Bazel/Gradle/`actions/cache`).
A4 `the-test-stage-fast-feedback-vs-confidence` — Test stages in the pipeline: the pyramid in CI, parallelization/sharding, fail-fast, required vs advisory checks, the flaky-test gate. (cross-link debugging flaky-test)
A5 `build-once-promote-everywhere-artifacts-and-versioning` — The immutable artifact promoted through environments; artifact registries, semantic versioning, digests, the rebuild-per-environment anti-pattern.
A6 `pipeline-as-code-the-anatomy-of-a-pipeline` — Pipeline-as-code: stages/jobs/steps, triggers, the DAG, runners, GitHub Actions / GitLab CI / Jenkins models side by side.

## Track B — Containers & images (Wave 2, 5)
B1 `containers-from-first-principles-for-delivery` — What a container really is (namespaces, cgroups, union FS, OCI image) and why it's the unit of delivery. (cross-link system-design/microservices, don't re-derive orchestration)
B2 `writing-a-production-dockerfile` — Multi-stage builds, layer-cache ordering, distroless/minimal base, non-root, build-vs-runtime split, `.dockerignore`, the 2 GB image anti-pattern.
B3 `image-registries-tagging-and-promotion` — Registries, tagging strategy (never `latest` in prod), digests vs tags, immutable tags, image promotion across envs, retention/GC.
B4 `building-images-fast-and-securely-in-ci` — BuildKit cache mounts + remote cache, building without Docker-in-Docker (kaniko/buildah), reproducible image builds, the cold-cache problem.
B5 `image-security-scanning-and-a-minimal-attack-surface` — Vulnerability scanning (Trivy/Grype) as a gate, distroless/minimal, base-image patching, the image as a supply-chain entry point.

## Track C — Kubernetes & deployment (Wave 3, 6)
C1 `kubernetes-for-delivery-the-objects-that-matter` — Deployment/Service/Ingress/ConfigMap/Secret, the reconcile loop, declarative desired-state as the CD target. (cross-link, don't re-derive k8s internals)
C2 `rolling-updates-and-zero-downtime-deploys` — Rolling-update mechanics, readiness/liveness gates, maxSurge/maxUnavailable, PodDisruptionBudget, graceful shutdown + connection draining.
C3 `configuration-and-secrets-in-kubernetes` — ConfigMaps/Secrets, env vs mounted, twelve-factor config, externalized config, config drift, the secret-in-the-manifest mistake.
C4 `helm-vs-kustomize-templating-your-manifests` — Helm charts vs Kustomize overlays, per-environment values, templating trade-offs, chart versioning, the over-templated chart.
C5 `progressive-delivery-in-the-pipeline-canary-and-blue-green` — Wiring Argo Rollouts/Flagger INTO the pipeline (the toolchain side), automated analysis gates, the Rollout CRD. (cross-link SRE deploying-safely for the reliability theory)
C6 `multi-environment-promotion-dev-staging-prod` — Environment topology, promotion gates, environment parity, ephemeral/preview environments, the promotion pipeline.

## Track D — Infrastructure as Code (Wave 4, 5)
D1 `infrastructure-as-code-from-clickops-to-declarative` — Why IaC, declarative vs imperative, the clickops/snowflake problem, idempotency, the plan/apply model.
D2 `terraform-in-practice-state-modules-and-workspaces` — Remote state + locking (state is sensitive), modules, workspaces, drift, `import`, the read-before-you-apply discipline.
D3 `managing-terraform-safely-at-scale` — Plan-in-CI, policy-as-code (OPA/Conftest/Sentinel), blast-radius + layered state, the apply-wanted-to-destroy-prod fear, reviewing a plan.
D4 `immutable-infrastructure-and-golden-images` — Immutable infra, golden images (Packer), cattle-not-pets, rebuild-don't-patch, the image bake pipeline.
D5 `iac-for-the-whole-stack-and-config-drift` — IaC beyond compute (DNS/IAM/managed DBs), drift detection + reconciliation, the manual-change-broke-it incident.

## Track E — GitOps & continuous deployment (Wave 5, 5)
E1 `gitops-git-as-the-source-of-truth` — GitOps principles, the pull-based reconcile-from-Git model, desired state in Git, why GitOps vs push-based CD.
E2 `argo-cd-and-flux-in-practice` — Argo CD / Flux, App-of-Apps, sync/drift/self-heal, sync waves, the GitOps repo structure.
E3 `push-vs-pull-cd-and-who-holds-the-keys` — Push-based (CI pushes to prod) vs pull-based (agent pulls); the credential/security argument; separating CI from CD.
E4 `promoting-releases-with-gitops` — Promotion via a Git PR to the env repo, rendered manifests, promoting across clusters, the config repo pattern.
E5 `progressive-delivery-meets-gitops` — Argo Rollouts + Argo CD, automated analysis-gated promotion, the GitOps + canary combination.

## Track F — Supply-chain security & secrets (Wave 6, 5)
F1 `software-supply-chain-security-the-new-frontier` — The threat (SolarWinds, Codecov, dependency confusion, typosquatting), the attack surface commit→prod, the SLSA framework.
F2 `signing-and-provenance-with-sigstore-and-slsa` — Artifact signing (cosign/sigstore), provenance/attestations (in-toto), SLSA levels, verifying-what-you-deploy at admission.
F3 `sbom-and-dependency-management` — SBOM (SPDX/CycloneDX), lockfiles + pinning, automated updates (Dependabot/Renovate), the transitive-dependency risk.
F4 `secrets-management-in-the-pipeline` — Secrets never in Git, Vault/cloud-KMS/External-Secrets/Sealed-Secrets, rotation, the leaked-credential incident, OIDC keyless to cloud.
F5 `securing-the-pipeline-itself` — The pipeline as an attack target (a compromised runner = prod), least-privilege + ephemeral CI credentials (OIDC federation), the poisoned-pipeline-execution attack, protecting deploy keys.

## Track G — Operate, measure & recover (Wave 7, 5)
G1 `dora-metrics-measuring-delivery-performance` — The four DORA metrics deep-dive: how to measure them, what elite looks like, using them to improve. (cross-link SRE)
G2 `feature-flags-decoupling-deploy-from-release` — Flags as a delivery tool (deploy ≠ release), flag platforms, progressive rollout by flag, flag debt + cleanup. (cross-link SRE)
G3 `database-migrations-in-the-delivery-pipeline` — Schema changes through the pipeline, expand/contract, online migrations, decoupling migration from deploy, the can't-roll-back migration. (cross-link database + SRE)
G4 `rollbacks-and-recovering-a-bad-deploy` — Rollback strategies in the pipeline, roll-forward vs roll-back, the un-rollback-able deploy, automated rollback on SLO regression. (cross-link SRE mitigate-first)
G5 `pipeline-observability-and-the-flaky-pipeline` — Observing the pipeline itself, build/deploy failure triage, pipeline flakiness, the slow pipeline, pipeline SLOs + DORA dashboards.

## Track H — Scale, platform & the playbook (Wave 8, 4)
H1 `monorepo-vs-polyrepo-and-scaling-the-pipeline` — Monorepo vs polyrepo CI/CD, affected-target builds, the 2-hour pipeline, build graphs (Bazel/Nx/Turborepo), scaling CI.
H2 `runners-caching-and-the-ci-cost-problem` — Self-hosted vs managed runners, autoscaling runners, caching to cut time + cost, the \$45k/mo CI bill, where the minutes go.
H3 `platform-engineering-and-the-internal-developer-platform` — The IDP, golden paths / paved roads, Backstage, self-service delivery, platform-as-product (the modern evolution of DevOps).
H4 `capstone-the-cicd-playbook` — **The master map.** Commit→prod walked end-to-end, the reference pipeline architecture, the delivery maturity model, DORA targets; ties all 39 posts into one field manual.

---

## Build conventions (carry over from prior series — Debugging & SRE)
- Author each wave via parallel agents (prose + Excalidraw DSL); **main session** renders (`_render-debug.sh <slug>`),
  runs `optimize-blog-images`, runs real `verify-post.sh`, cross-link audits, commits **only that wave's** `.md` +
  `.webp` (explicit paths), `git pull --rebase --autostash`, pushes to main.
- Slugs must NOT contain a verify trigger word. Set readTime to the verifier's recomputed value.
- Recurring fix loop (from SRE/Debugging): **4-strong-kinds → demote one to neutral** (most common); `tree` needs a
  `parent` field on every non-root node (NOT edges) else "multiple roots"; a near-linear `graph` (one node per layer,
  or branch-then-remerge-one-deeper) FAILs → recast as `stack`; drop `yes`/`no` edge labels on diamonds (text
  overlap); side-annotation nodes feeding a chain cause arrow-crossings → fold into the node label; ` ```text `
  forbidden → ` ```bash `/`yaml`/`hcl`; abstraction-trigger words far from a figure → reword; bare-folder links
  `/blog/software-development/<series>/` → repoint to a real slug. zsh: `status` is read-only; use arrays.
- Marker `.claude/plans/.cicd-done` on completion; update memory.
