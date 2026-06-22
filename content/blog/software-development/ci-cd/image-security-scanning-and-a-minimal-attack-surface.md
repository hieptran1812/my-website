---
title: "Image Security Scanning and a Minimal Attack Surface: Ship Less, Scan What's Left"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Every image you ship is a bag of OS packages and dependencies, each a potential CVE. The cheapest defense is to ship less and scan what's left — minimize the attack surface, gate the build on critical CVEs, and keep scanning what's already deployed."
tags:
  [
    "ci-cd",
    "devops",
    "image-security",
    "vulnerability-scanning",
    "trivy",
    "distroless",
    "attack-surface",
    "supply-chain",
    "containers",
    "cve",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/image-security-scanning-and-a-minimal-attack-surface-1.png"
---

A scanner once told me an image we deployed to production every single day carried two hundred and eighty known vulnerabilities, eleven of them rated critical. The natural reaction — the one I had — is to argue. We are not careless engineers. We did not write two hundred and eighty bugs. We barely wrote enough code to *have* eleven bugs. But the scanner was right and I was wrong, and the reason is the whole subject of this post: almost none of those vulnerabilities were in our code. They were in the operating system packages, the language runtime, the package manager, a half-dozen image-processing libraries we pulled in for one endpoint, and the dozens of transitive dependencies that came along for the ride because our `Dockerfile` said `FROM node:18` and `npm install`. We were not shipping a web service. We were shipping an entire Linux distribution, a full Node toolchain, and a software development kit, and then running a thirty-megabyte service inside it. Every one of those packages was a line item in the scanner's report. Every one was a door an attacker might try.

Here is the part that should bother you more than the number itself. That image was not unusual. It was the *normal* image, the one the tutorial gives you, the one shipping at thousands of companies right now. And the report it produced was useless in exactly the way that makes security theater out of vulnerability scanning: nobody can triage two hundred and eighty findings. The team did what every team does with an un-actionable wall of CVEs — they glanced at it, felt vaguely bad, and ignored all of it, including the eleven critical ones that mattered. A scanner that produces a list nobody reads is worse than no scanner, because it launders inaction into the appearance of diligence.

![Before and after comparison of a node full base image scanning at hundreds of CVEs across nine hundred packages next to a distroless runtime image scanning at near zero with only three packages and no shell](/imgs/blogs/image-security-scanning-and-a-minimal-attack-surface-1.png)

This post closes that gap with two moves that reinforce each other, and they are the entire thesis. **First, minimize the attack surface** — ship less. An image with three packages has almost nothing in it to be vulnerable, almost nothing to patch, and almost nothing for an attacker to exploit. The distroless base that carries no shell and no package manager is not a security feature you bolt on; it is the absence of the thing you would otherwise have to defend. **Second, scan what's left as a pipeline gate** so you never *knowingly* ship a critical CVE with a fix available — and keep scanning what is already deployed, because new vulnerabilities are discovered against old images every day, and a clean scan at build time goes stale the moment you walk away from it. By the end of this post you will be able to wire a Trivy scan into a GitHub Actions pipeline that fails the build on exactly the findings worth blocking, write a `.trivyignore` that documents accepted risk with an expiry instead of burying it forever, configure continuous registry scanning so a CVE disclosed six months after build still pages you, and automate the base-image rebuilds that patch those CVEs without a single application change. And you will be able to explain *why* the distroless image with three packages is the move that makes every other step on this list cheap.

A quick word on vocabulary, because we will lean on it constantly. A **CVE** (Common Vulnerabilities and Exposures) is a publicly catalogued software vulnerability with a unique identifier like `CVE-2024-3094`. A **vulnerability scanner** inspects the packages inside an image and reports which of them are versions that appear in CVE databases. **Severity** is a rating — typically critical, high, medium, low — derived from a score like CVSS that estimates how bad a CVE is. **Attack surface** is the sum of everything in your image an attacker could exploit: every package, every binary, every shell, every library. **Distroless** is a family of base images that contain your application's runtime dependencies and nothing else — no shell, no package manager, no `coreutils`. An **SBOM** (Software Bill of Materials) is a machine-readable inventory of everything in the image. **Provenance** is a signed record of how and from what an artifact was built. We will define the rest as we go.

This post lives in the series **"CI/CD & Cloud-Native Delivery, From Commit to Production."** If you have not read [the CI/CD pipeline map](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model), start there — it lays out the spine the whole series returns to: **commit → build → test → package → deploy → operate**, governed by **"build once, promote everywhere"** and **"everything as code,"** measured by the four DORA metrics. Image scanning sits at the *package* stage, right where the immutable artifact is born. It composes directly with [writing a production Dockerfile](/blog/software-development/ci-cd/writing-a-production-dockerfile), which builds the minimal image we will scan, and with [building images fast and securely in CI](/blog/software-development/ci-cd/building-images-fast-and-securely-in-ci), which runs the build that produces it. And it is one half of a larger story about software supply chains — scanning tells you *what is in* the image; signing and provenance prove it *is the image you built*. We will keep that distinction sharp.

## 1. What a vulnerability scanner actually does

Before we argue about gates and thresholds, you need an honest mental model of the machine. A scanner is not magic and it is not deep. Put plainly, it does three things in sequence, and understanding them tells you exactly what it can and cannot find.

![Flow diagram showing a scanner inventory the OS packages and language dependencies from image layers then match each version against CVE feeds to produce findings sorted by severity](/imgs/blogs/image-security-scanning-and-a-minimal-attack-surface-2.png)

**Step one: build an inventory.** The scanner walks the image's filesystem layers and finds every package it can identify. For the operating system, that means reading the package database — `/var/lib/dpkg/status` on Debian and Ubuntu, the `rpm` database on Red Hat family distros, `/lib/apk/db/installed` on Alpine — and extracting the name and exact version of every installed system package. For your application, it parses the language-specific manifests: `package-lock.json` and `node_modules` for Node, `requirements.txt` or installed `*.dist-info` for Python, `go.sum` and embedded module metadata for Go binaries, `pom.xml` and the contents of JAR files for Java. The output of this step is a list — often hundreds of lines long — of `(package, version)` pairs. That list is, essentially, an SBOM.

**Step two: match against CVE databases.** The scanner downloads vulnerability feeds — the National Vulnerability Database (NVD), the GitHub Advisory Database (GHSA), distro-specific security trackers like the Debian and Alpine security feeds, and language ecosystem advisories. Each CVE in those feeds names an affected package and a version range. The scanner does a join: for each `(package, version)` pair in your inventory, it asks "does any known CVE name this package with a version range that includes this version?" Every hit is a finding.

**Step three: report by severity.** Each finding carries the CVE id, the affected package, the installed version, the version that fixes it (if one exists), and a severity. The scanner sorts and presents them. That's the whole machine.

Notice what this architecture implies. The scanner's accuracy is bounded entirely by two things: the quality of its inventory, and the quality of its feeds. If it cannot identify a package — say you compiled a library from source and dropped the binary in by hand, with no package metadata — it is invisible to the scan, vulnerable or not. If a CVE has not yet been disclosed, or has been disclosed but not yet added to the feeds, no match happens. And if the feed says a package is fixed in version 1.4.2 but the distro backported the patch to their 1.4.0-3 build, a naive scanner will report a false positive. The good scanners — Trivy in particular — understand distro backport semantics and suppress those, which is one reason the same image scanned by two tools can give two different counts.

That last point matters for trust, so let me name the major tools and their character before we pick one.

| Tool | Engine | Strengths | Notes |
| --- | --- | --- | --- |
| **Trivy** (Aqua) | Go, single binary | Fast, no daemon, OS + language + IaC + secrets, good distro backport handling | The default for most CI pipelines; what we use below |
| **Grype** (Anchore) | Go, single binary | Fast, pairs with Syft for SBOM, strong language coverage | Very similar to Trivy; pick by ergonomics |
| **Clair** (Quay/Red Hat) | Server + API | Registry-integrated continuous scanning, mature | Heavier to run; great as a registry-side scanner |
| **Snyk** | SaaS + CLI | Reachability analysis, fix PRs, rich UI, dependency graphs | Commercial; strong on the noise problem |

For a CI gate, a single-binary scanner you can run in one step with no server to maintain is the right shape, and that is Trivy or Grype. For continuous scanning of everything in your registry, a server-side scanner like Clair or the scanning built into your registry (ECR, GHCR, Harbor, Artifact Registry all have it) is the right shape, because it can re-evaluate stored images against updated feeds on a schedule. We will use both, because they answer different questions at different times.

One more property of the machine deserves emphasis, because it changes how you reason about a scan result: **the scan is a function of two inputs, the image and the feed, and only one of them is under your control.** You freeze the image the moment you build it — that artifact is immutable, and "build once, promote everywhere" depends on it staying that way. But the feed is a living thing that grows every day as new CVEs are catalogued. So the *same* immutable image produces a *different* scan result over time, purely because the feed changed underneath it. This is not a bug; it is the defining characteristic of vulnerability scanning, and almost every mistake teams make with scanners traces back to forgetting it. A "passing" scan is a passing scan *against today's feed*. Tomorrow's feed has not seen tomorrow's CVE yet. Hold onto that asymmetry — the image is fixed, the knowledge about it is not — because §5 is entirely an exploration of its consequences.

There is also a practical question of *when* the scanner gets its feed. In CI, the scanner downloads (or updates) the vulnerability database at the start of the run, which adds a few seconds and, occasionally, a flaky network dependency. The robust pattern is to cache the vulnerability database between runs (Trivy supports `--cache-dir` and a separate database-download step) and to pre-warm it, so a transient outage of the upstream feed mirror does not turn into a failed build. A scanner that cannot reach its feed should *warn loudly*, not silently pass with a stale database — a stale database is exactly how you ship a critical you would otherwise have caught. Configure the scan so a feed-fetch failure is a visible, non-fatal warning on pull requests and a hard failure on the release path, where shipping against a stale database is unacceptable.

Here is the simplest possible run, so the machine is concrete:

```bash
# Scan an image, show only fixable CRITICAL and HIGH, fail if any are found
trivy image \
  --severity CRITICAL,HIGH \
  --ignore-unfixed \
  --exit-code 1 \
  ghcr.io/acme/checkout:1.8.3
```

`--severity` filters which ratings to consider. `--ignore-unfixed` drops findings that have no patched version available — a point we will return to, because it is the single most important flag for keeping the gate sane. `--exit-code 1` is what turns the scan into a *gate*: a nonzero exit code fails the CI step. Run that against a `node:18` base and you get a wall. Run it against a distroless image and you get a sentence. The difference between those two outcomes is not the scanner. It is the image.

## 2. What scanning catches — and what it cannot

The most dangerous thing you can believe about a scanner is that a clean scan means a secure image. It does not, and the gap is not subtle. Scanning is *one* layer of defense, and it covers exactly one category of risk: **known vulnerabilities in software you did not write.** Everything outside that category, it is blind to.

![Matrix showing that a scanner sees known OS and dependency CVEs but is blind to application logic bugs, zero-day vulnerabilities, and most misconfiguration which other layers must catch](/imgs/blogs/image-security-scanning-and-a-minimal-attack-surface-3.png)

Walk the categories honestly:

- **Known CVEs in OS packages** — a vulnerable `openssl`, `glibc`, `zlib`, `bash`. *Caught.* This is the scanner's home turf.
- **Known CVEs in your language dependencies** — a vulnerable `lodash`, `log4j`, `requests`, `jackson-databind`. *Caught*, as long as the dependency is identifiable in a manifest the scanner parses.
- **Your own logic bugs** — an authorization check you forgot, an injection you left open, a secret you logged. *Not caught.* The scanner has no idea what your code does. There is no CVE for your business logic.
- **Zero-day vulnerabilities** — a flaw nobody has disclosed yet, including the famous case where the flaw is *in* a package and was deliberately planted. *Not caught*, by definition, until it is disclosed and added to a feed.
- **Misconfiguration** — running as root, a world-readable secret baked into a layer, an overly permissive `securityContext`, a debug port left open. *Mostly not caught* by image scanning, though some scanners (Trivy included) bolt on a separate config-scanning mode and a secret-scanning mode that help here.

The honest summary is the one engraved over the door of every security team: **scanning is necessary but not sufficient.** It is a smoke detector. A smoke detector is genuinely valuable — it catches the most common, most preventable category of fire, the one where someone left a vulnerable `log4j` on the stove. It will not catch an arsonist, and it will not stop you from building the house out of kerosene. So we run the scanner as one layer, and we pair it with the layers that catch what it misses: code review and tests for logic bugs, runtime defenses and rapid patching for zero-days, policy-as-code and admission control for misconfiguration, and — the subject of the next section — a minimal attack surface so there is less for any attacker to reach in the first place.

There is a second subtle limit worth internalizing. A finding is not the same as an exploit. The scanner tells you a vulnerable *package version* is present. It does not tell you whether the vulnerable *function* in that package is ever called, whether it is reachable from any input an attacker controls, or whether your deployment context neutralizes it (a CVE in a TLS library you never use for inbound connections, say). This is the gap that produces the noise problem — hundreds of "vulnerabilities" that are technically present and practically irrelevant — and it is why reachability analysis, which we will cover, is the most valuable filter you can add. But hold that thought; reachability only matters once you have decided to take the report seriously, and you only take the report seriously when it is short.

## 3. The minimal attack surface: the first and best defense

Here is the principle that organizes everything else, and it is almost embarrassingly simple: **you cannot exploit what is not there.** Every package in your image is a potential CVE, a potential patch, and a potential foothold. The most reliable way to reduce vulnerabilities is not to find them and fix them. It is to not have them. Ship less.

![Stack diagram showing defense in depth layers around the image from a minimal base of three packages up through a scan gate, a nonroot read only runtime, dropped capabilities, and signing for provenance](/imgs/blogs/image-security-scanning-and-a-minimal-attack-surface-4.png)

Let me make the principle quantitative, because that is what makes it persuasive to a security review. Suppose the probability that any given package in your image has at least one known critical CVE at a random point in time is some small number $p$. If your image contains $n$ independent packages, the expected number of packages carrying a critical CVE is $n \cdot p$, and the probability that *at least one* package carries a critical CVE is $1 - (1-p)^n$. The exact value of $p$ does not matter for the argument; what matters is the shape. With $n = 900$ packages, even a tiny per-package risk compounds into near-certainty that *something* in the image is critically vulnerable at any given moment. With $n = 3$, the expected count $3p$ is small enough that the image is usually clean and, when it is not, the list is short enough to fix the same afternoon. The attack surface is roughly linear in $n$; the management cost is roughly linear in the *number of findings*, which is also roughly linear in $n$. Cutting $n$ from 900 to 3 cuts both by two orders of magnitude. That is the entire case for distroless in one inequality.

The practice is the base-image spectrum we covered in [writing a production Dockerfile](/blog/software-development/ci-cd/writing-a-production-dockerfile), now seen through the security lens. You move down a ladder, and each rung sheds packages — and therefore CVEs.

| Base | Approx packages | Has shell | Has package manager | Typical CVE surface |
| --- | --- | --- | --- | --- |
| `ubuntu` full | ~900+ | yes (`bash`) | yes (`apt`) | hundreds |
| `debian:slim` | ~120 | yes | yes | dozens |
| `alpine` | ~25 | yes (`ash`) | yes (`apk`) | a handful |
| `gcr.io/distroless/base` | ~5–10 | **no** | **no** | 0–2 |
| `scratch` | 0 | no | no | 0 |

The jump that matters is the one to distroless. A distroless image contains your runtime and its direct shared libraries and *nothing else*. No shell means an attacker who achieves code execution cannot drop into `/bin/sh` and start poking around — there is no `sh` to drop into, no `curl` to pull down a second-stage payload, no `cat /etc/passwd`, no `apt-get install` to bring in tools. No package manager means there is no `apt` or `apk` with its own CVE history and its own ability to mutate the running image. You have not added a defense; you have removed the things that needed defending.

It is worth being precise about *why* a missing shell is such a large win, because it is the most counterintuitive part of the argument. Most real-world container compromises are not a single clean "attacker runs arbitrary code" event. They are a *chain*: the attacker finds a foothold (a deserialization bug, an SSRF, a vulnerable library), and then *pivots* — uses that foothold to explore the filesystem, read credentials, install reconnaissance tools, establish persistence, and move laterally. Almost every step in that pivot wants a shell, or `wget`/`curl`, or a package manager, or a scripting interpreter that is not your application's. A distroless image denies the attacker the *tooling for the pivot*. The initial foothold may still be possible — distroless does not patch your logic bug — but the attacker who lands in a distroless container lands somewhere with no shell, no network tools, no way to download a payload, and a read-only filesystem they cannot write to. They have a foothold and nowhere to go. That is the difference between an incident and a breach. Minimizing the surface does not just reduce the count of *findings*; it reduces the *exploitability* of the findings that remain and the *value* of any foothold an attacker does achieve.

The honest counterargument, and you should weigh it: a shell-less image is harder to *debug*. When something is wrong in production and your instinct is to `kubectl exec -it pod -- /bin/sh` and look around, distroless says no — there is no shell to exec into. The answer is not to add a shell back; it is to debug differently. Use an *ephemeral debug container* (`kubectl debug -it pod --image=busybox --target=app`) that attaches a throwaway, fully-tooled container sharing the target's process namespace *only when you need it and only for as long as you need it*, leaving the running image minimal. Or rely on logs, metrics, and traces that you should have anyway. The debuggability cost is real, but it is a one-time workflow change, not an ongoing risk, and the ephemeral-debug-container pattern recovers almost all of the convenience without keeping a shell resident in production.

The concrete ways you shed surface:

- **Multi-stage builds** leave the entire build toolchain — compilers, `make`, dev headers, the package manager that installed them — in a build stage you throw away. The runtime stage copies only the compiled artifact. The CVE-heavy SDK never ships.
- **Remove shells and `setuid` binaries.** A shell is the single most useful tool an attacker can find. `setuid` binaries (files that run with the owner's privileges, often root) are a classic privilege-escalation vector; a minimal base has essentially none.
- **Remove build tools and unused libraries.** If you only call one endpoint that needs `imagemagick`, and `imagemagick` drags in fifteen image-codec libraries each with their own CVE stream, every one of those is surface you carry for one feature. Question whether it belongs in the image at all.
- **Pin and remove dev dependencies.** Test frameworks, linters, and mocks have CVEs too, and they have no business in a production image. A proper dependency install in the runtime stage uses the production-only flag (`npm ci --omit=dev`, `pip install --no-dev` patterns, Go's natural exclusion of test files).

And then, because no surface reduction is ever total, you add the defense-in-depth runtime hardening — which is the same hardening we argued for in the Dockerfile post, now stated as a security control. **Run as a non-root user** so that even successful code execution is unprivileged. **Mount the root filesystem read-only** so an attacker cannot write a payload to disk. **Drop Linux capabilities** the process does not need. Here is the Kubernetes `securityContext` that encodes all three, which you should treat as the default for every workload:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: checkout
spec:
  template:
    spec:
      securityContext:
        runAsNonRoot: true          # refuse to start if the image's user is root
        runAsUser: 10001            # a numeric, non-root UID
        runAsGroup: 10001
        fsGroup: 10001
        seccompProfile:
          type: RuntimeDefault      # restrict syscalls to a sane default set
      containers:
        - name: checkout
          image: ghcr.io/acme/checkout@sha256:9f2c...   # pin by digest, not tag
          securityContext:
            allowPrivilegeEscalation: false   # block setuid-based escalation
            readOnlyRootFilesystem: true      # nothing can be written to the image fs
            capabilities:
              drop: ["ALL"]                   # drop every Linux capability, add back none
          volumeMounts:
            - name: tmp
              mountPath: /tmp        # writable scratch space, since root fs is read-only
      volumes:
        - name: tmp
          emptyDir: {}
```

`runAsNonRoot: true` makes Kubernetes *refuse to start the pod* if the image would run as root, which turns a Dockerfile mistake into a deploy-time failure instead of a silent risk. `readOnlyRootFilesystem: true` is the one that most often requires a small change — you grant a writable `emptyDir` for `/tmp` and any cache directory the app genuinely needs — but the payoff is large: an attacker who gets execution cannot persist anything to the image filesystem. `capabilities: drop: ["ALL"]` removes powers like binding to low ports or modifying the network stack that almost no application server actually needs. Read [designing for failure](/blog/software-development/site-reliability-engineering/designing-for-failure) for the reliability reasoning behind treating every container as something that will eventually be compromised or crash; the security and reliability arguments converge on the same minimalism.

#### Worked example: attack-surface reduction kills the CVE list

Take the service from the intro. The starting point is the normal Dockerfile:

```dockerfile
FROM node:18
WORKDIR /app
COPY . .
RUN npm install
CMD ["node", "server.js"]
```

Scan it. Trivy reports roughly **280 vulnerabilities**, of which about **11 are critical** and another forty or so high. The image is **1.9 GB**. It has `bash`, `apt`, `dpkg`, a C compiler, Python (Node's build scripts pull it in), `git`, and around 900 OS packages — plus every dev dependency in `node_modules`, because `npm install` does not omit them. The scanner's report is four screens long. Nobody is going to triage four screens. The findings are, in practice, un-actionable, and the team ignores all of them.

Now apply the surface reduction — multi-stage build, production-only dependency install, distroless runtime, non-root user:

```dockerfile
# ---- build stage: full toolchain, thrown away ----
FROM node:18 AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci                          # full deps for building
COPY . .
RUN npm run build && npm prune --omit=dev   # build, then strip dev deps

# ---- runtime stage: distroless, nonroot, minimal ----
FROM gcr.io/distroless/nodejs18-debian12:nonroot AS runtime
WORKDIR /app
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist
COPY --from=build /app/package.json ./
USER nonroot
CMD ["dist/server.js"]
```

Scan the result. Trivy now reports **0 to 2 vulnerabilities**, typically **0 critical**, because the distroless base carries a tiny set of pinned libraries that Google patches aggressively, and your production `node_modules` is a far shorter list than the dev tree. The image is around **80 MB**. There is no shell, no `apt`, no compiler. The report is, at most, a few lines. The same scanner that produced an ignored wall now produces a list you can read in ten seconds and fix in an afternoon.

The point is not just that the second image is more secure — though it is, dramatically. The point is that **the second image makes scanning *meaningful*.** A scan that returns three findings gets read and acted on. A scan that returns two hundred and eighty gets ignored, including the eleven that would have hurt you. Minimizing the attack surface is what converts scanning from theater into a working control. You cannot fix a list nobody reads; the first job of attack-surface reduction is to make the list readable.

## 4. Scanning as a pipeline gate

A scan that you run by hand once a quarter, or that posts a report nobody opens, changes nothing. The point of scanning in CI is to make it a **gate** — a required check that *fails the build* when the image carries a vulnerability you have decided is unacceptable to ship. The gate is the mechanism that turns "we know about this CVE" into "we cannot ship while this CVE is here." It is the difference between knowing and *not knowingly shipping*.

![Flow diagram showing the scan running after the build and routing on a policy where a critical CVE with an available fix blocks and fails the build while a low severity or unfixable finding only warns and the image proceeds to be pushed signed to the registry](/imgs/blogs/image-security-scanning-and-a-minimal-attack-surface-5.png)

The entire design problem of a gate is the **threshold**, and it has two failure modes that bracket the right answer:

- **Block everything.** Fail the build on any finding of any severity. This sounds rigorous and is a disaster in practice. Your build is now red because of a medium-severity CVE in a system library that has *no fix available* and that you cannot do anything about. The team's only options are to ignore the failed gate (so the gate is dead) or to disable it (so the gate is gone). Blocking on unfixable findings makes the gate the enemy, and an enemy gate gets routed around.
- **Block nothing.** Scan and report but never fail. This is the most common state, and it is functionally identical to not scanning at all. A report that cannot stop a deploy does not stop a deploy. People mean to look at it. They don't.

The right threshold is in between, and it is sharper than "high severity": **block on CRITICAL (and optionally HIGH) severity findings that have a fix available.** Three conditions, all of which must hold to block:

1. **Severity is critical or high.** A critical CVE is the kind that gets exploited in the wild within days of disclosure. A medium or low can wait for the regular patch cadence.
2. **A fix is available.** This is the crucial qualifier and the most-skipped one. If the vulnerable package has no patched version, blocking the build accomplishes nothing except blocking the team — there is no version to upgrade to. You *track* the unfixable finding; you do not *gate* on it. The flag is `--ignore-unfixed`, and it is the single most important setting for keeping the gate trusted.
3. **It is newly introduced** (a refinement, not a requirement). The most advanced setup compares the finding set against the previous image and blocks only on *new* critical-with-fix CVEs, so a long-standing accepted risk does not block an unrelated change. We will reach that, but severity-plus-fixability gets you most of the value.

The policy in one line: **block CRITICAL-with-fix, warn on everything else.** Here is the gate as a real GitHub Actions step, slotted into the build job right after the image is built and before it is pushed:

```yaml
name: build-and-scan
on:
  pull_request:
  push:
    branches: [main]

jobs:
  image:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write          # to push to GHCR
      security-events: write   # to upload SARIF to the Security tab
    steps:
      - uses: actions/checkout@v4

      - name: Build image
        run: |
          docker build -t ghcr.io/acme/checkout:${{ github.sha }} .

      # Gate: fail the build on fixable CRITICAL/HIGH only
      - name: Scan image (gate)
        uses: aquasecurity/trivy-action@0.24.0
        with:
          image-ref: ghcr.io/acme/checkout:${{ github.sha }}
          severity: CRITICAL,HIGH
          ignore-unfixed: true        # do not block on findings with no patch
          exit-code: "1"              # nonzero => fail the job
          trivyignores: .trivyignore  # honor documented exceptions
          format: table

      # Also upload a full report (all severities) for visibility, non-blocking
      - name: Scan image (report all, non-blocking)
        if: always()
        uses: aquasecurity/trivy-action@0.24.0
        with:
          image-ref: ghcr.io/acme/checkout:${{ github.sha }}
          severity: CRITICAL,HIGH,MEDIUM,LOW
          ignore-unfixed: false
          exit-code: "0"              # never fail on the report run
          format: sarif
          output: trivy-results.sarif

      - name: Upload SARIF to Security tab
        if: always()
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: trivy-results.sarif

      - name: Push image (only reached if the gate passed)
        if: github.ref == 'refs/heads/main'
        run: |
          echo "${{ secrets.GITHUB_TOKEN }}" | docker login ghcr.io -u ${{ github.actor }} --password-stdin
          docker push ghcr.io/acme/checkout:${{ github.sha }}
```

Two scans run, and the split is deliberate. The **gate scan** considers only fixable critical and high findings and sets `exit-code: 1`, so it is the step that can fail the build and block the push. The **report scan** considers everything, never fails, and uploads a SARIF file to the GitHub Security tab so the full picture — including the unfixable mediums and lows — is visible and trackable without blocking anyone. This is the shape that survives contact with a real team: the gate is narrow enough to be trusted (it only fires on things you can and should fix), and the visibility is wide enough to be useful (you can still see and prioritize the long tail). Make the gate scan a **required status check** on the `main` branch protection rule, and now the policy is enforced by the platform, not by goodwill.

There is one more decision: whether to gate on HIGH or only CRITICAL. My recommendation is to start with **CRITICAL-with-fix as a hard gate** and **HIGH-with-fix as a gate with a short grace period** — block CRITICAL immediately, but allow HIGH to warn for, say, seven days before it starts blocking, so the team has a window to patch without an emergency. The grace period is policy you encode in your scanning tooling or in a small wrapper script; the principle is that severity should map to *urgency*, and urgency should map to *how fast the gate tightens*.

The same gate translates cleanly to GitLab CI, which ships a built-in container-scanning template, but here is the explicit version so you can see the wiring rather than a black box:

```yaml
stages: [build, scan, deploy]

build_image:
  stage: build
  image: docker:24
  services: [docker:24-dind]
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

scan_gate:
  stage: scan
  image:
    name: aquasec/trivy:0.55.0
    entrypoint: [""]
  variables:
    TRIVY_CACHE_DIR: ".trivycache/"   # cache the DB between runs
  cache:
    paths: [".trivycache/"]
  script:
    # Pull the DB first so a feed outage is a clear failure here, not mid-scan
    - trivy image --download-db-only
    # The gate: fixable CRITICAL/HIGH only; nonzero exit fails the stage
    - >
      trivy image
      --severity CRITICAL,HIGH
      --ignore-unfixed
      --exit-code 1
      --ignorefile .trivyignore
      $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
    # Full report, never blocks — emitted as an artifact for visibility
    - >
      trivy image
      --severity CRITICAL,HIGH,MEDIUM,LOW
      --exit-code 0
      --format json --output gl-report.json
      $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  artifacts:
    paths: [gl-report.json]
    when: always

deploy:
  stage: deploy
  needs: [scan_gate]          # deploy only runs if the gate stage succeeded
  rules:
    - if: $CI_COMMIT_BRANCH == "main"
  script:
    - echo "promote $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA"
```

The `needs: [scan_gate]` on the deploy job is the structural enforcement: the DAG simply will not reach `deploy` unless the gate stage exited zero. That is the same property as a required status check in GitHub — the platform, not a human's vigilance, refuses to ship a blocked image. Whichever CI system you are on, the invariant is identical: the scan stands *between* build and any path to production, and a blocking finding stops the pipeline cold.

### Stress-testing the gate

A gate is only as good as its behavior under the awkward cases, so reason through them before they happen in production:

- **What if the scanner's feed is down mid-run?** As above: fail loud on the release path, warn on PRs, and never silently pass with a stale database. Cache the DB so the common case does not depend on the network at all.
- **What if two PRs introduce the same unfixable critical at once?** Neither should block (it is unfixable — `--ignore-unfixed` drops it), and both should surface it on the watchlist. If you are gating on *newly introduced* findings by diffing against the base image, make sure the base of comparison is the target branch's last good image, not each PR's parent, or you will double-count.
- **What if a fix becomes available between the PR scan and the merge scan?** The previously-unfixable critical is now fixable and the gate fires on merge. This is correct behavior — the moment a fix exists, the finding becomes actionable — but it means a green PR can go red at merge time through no change of the author's. Communicate that this is the gate working, not a flake, and let the base-bump automation in §5 supply the fix.
- **What if someone adds a `.trivyignore` entry to force a red build green?** This is the failure you design against. Require that `.trivyignore` changes go through code review like any other change, that every entry carries an `expired_at` and a justification, and ideally that a separate owner (a security reviewer) approves additions. An exception is a decision, and decisions get reviewed.
- **What if the gate is flaky and the team starts re-running it to get green?** A flaky security gate is a dead security gate — re-run-until-green trains the team to disbelieve it. Track the gate's flake rate like any other check; if it flakes, the cause is almost always the feed fetch, which the DB cache fixes. See the broader treatment of [flaky tests in the pipeline](/blog/software-development/debugging/the-flaky-test-find-it-fix-it-or-quarantine-it) — a flaky gate poisons trust exactly the way a flaky test does.

### Allowlisting accepted risk — with an expiry

Sometimes a critical-with-fix finding genuinely should not block you. Maybe the vulnerable code path is unreachable in your usage. Maybe the "fix" is a major version bump that breaks your API and you have a mitigation in place while you plan the migration. The mechanism for this is an allowlist, and Trivy's is a `.trivyignore` file. The danger is that an allowlist becomes a silent forever-ignore — someone adds a CVE id to the file to make a build green at 2 a.m., and three years later that CVE is still suppressed and nobody remembers why. **A bare allowlist entry is a security debt you have hidden from yourself.**

The discipline that fixes this is simple and non-negotiable: **every allowlist entry carries an expiry and a justification.** Trivy supports both, and you should use both every time:

```yaml
# .trivyignore.yaml — accepted-risk exceptions, each with an owner, reason, and expiry.
# An entry without an expiry is a code-review reject. Re-justify on renewal.
vulnerabilities:
  - id: CVE-2024-12345
    # libtiff CVE in an image codec we ship but never invoke on untrusted input;
    # the vulnerable decode path is gated behind an admin-only endpoint.
    # Owner: platform-security. Reviewed: 2026-06-20.
    statement: "Unreachable: decode path is admin-only, not exposed to user input."
    expired_at: 2026-09-20      # ~90 days; build starts failing again after this
  - id: CVE-2024-67890
    # Fix requires a breaking major bump of the framework; migration tracked in JIRA-4821.
    # Mitigation: WAF rule blocks the exploit pattern at the edge.
    # Owner: checkout-team. Reviewed: 2026-06-20.
    statement: "Mitigated at edge by WAF rule; migration scheduled for Q3."
    expired_at: 2026-08-15
```

The `expired_at` field is the whole point. After that date, Trivy stops honoring the exception and the finding starts blocking the build again. This forces a *renewal decision*: someone has to look at the CVE again, confirm the justification still holds (has the fix shipped? is the path still unreachable? is the migration still on track?), and either patch it or consciously re-accept it with a new expiry. The accepted risk can never silently become permanent. The `statement` is the justification — it is what a future engineer (or auditor) reads to understand *why* this was acceptable. An allowlist with expiries and justifications is a living risk register. An allowlist of bare CVE ids is a graveyard. Use the first kind.

#### Worked example: the gate that halved change-fail incidents

A team I worked with shipped roughly thirty deploys a week and was averaging about one security-related incident a quarter traced to a known, fixable CVE that had been in the image at deploy time — a `log4j`-style "we shipped a version we knew was bad" event. We added the gate above: CRITICAL-with-fix blocks, HIGH-with-fix blocks after a seven-day grace, everything else reports to the Security tab. We also added the distroless base from §3, which dropped the median finding count from the high hundreds to single digits, so the gate almost never fired on noise.

Over the following two quarters, the count of incidents traceable to a known fixable CVE present at deploy went to **zero** — not because the gate caught a dramatic exploit, but because the *combination* of a short finding list and a hard gate meant the team patched fixable criticals as a routine part of the build instead of discovering them in an incident review. The gate fired about twice a month, each time on a genuine fixable critical, each time fixed by a base-image bump or a dependency upgrade within hours. The cost was real but small: roughly two short interruptions a month, plus the one-time work of getting `readOnlyRootFilesystem` to play nicely with the app's temp-file usage. Measured against a quarterly security incident — each of which cost days of investigation, a postmortem, and a customer-facing disclosure — the trade was overwhelmingly positive. The honest framing for your own measurement: count "deploys carrying a known fixable critical CVE" before and after. That number going to zero is the gate working.

## 5. The base-image patching problem: a clean scan goes stale

Here is the part most teams miss, and it is the part that turns a good build-time gate into a false sense of security. **A clean scan at build time is a snapshot, and the world keeps moving.** CVEs are discovered against software *that already exists* — including the exact base layers and library versions inside the image you shipped to production six months ago. The image has not changed. Your code has not changed. But the *knowledge about that image* has changed: vulnerabilities that were unknown when you built it are now public, scored, and possibly being exploited. Your build-time scan said "clean." It is no longer true.

![Timeline showing an image that scanned clean on build day accumulating one new high CVE at month two, two by month four, and four new critical CVEs by month six which a continuous registry scan flags and an automated base rebase patches back to zero](/imgs/blogs/image-security-scanning-and-a-minimal-attack-surface-6.png)

This has two consequences, and you need both controls to handle them.

**Consequence one: you must keep scanning what is already deployed.** A scan is a join against a feed, and the feed updates daily. So the same image scanned today and three months from now can produce different results *with no change to the image* — because the feed has more CVEs in it now. This is why a build-time gate is necessary but not sufficient: it only ever sees the feed as it was on build day. You need **continuous scanning in the registry**, re-evaluating stored images against the current feed on a schedule. Every major registry supports this. Here is enhanced scanning configured on AWS ECR via Terraform, which rescans images on push *and* continuously as new CVEs are disclosed:

```hcl
resource "aws_ecr_registry_scanning_configuration" "this" {
  scan_type = "ENHANCED"   # uses Amazon Inspector; rescans on new CVE disclosures

  rule {
    scan_frequency = "CONTINUOUS_SCAN"   # not just on push — re-evaluate over time
    repository_filter {
      filter      = "*"
      filter_type = "WILDCARD"
    }
  }
}

# Route new findings on already-pushed images to a notification channel
resource "aws_cloudwatch_event_rule" "inspector_findings" {
  name        = "ecr-image-new-cve"
  description = "Fire when Inspector finds a new vulnerability in a stored image"
  event_pattern = jsonencode({
    source      = ["aws.inspector2"]
    detail-type = ["Inspector2 Finding"]
    detail = {
      severity = ["CRITICAL", "HIGH"]
      status   = ["ACTIVE"]
    }
  })
}
```

`CONTINUOUS_SCAN` is the keyword that matters: the registry does not wait for you to re-push: it re-evaluates the stored image whenever the vulnerability feed gains a relevant entry. When a critical CVE is disclosed against a base layer in an image you shipped months ago, the registry scan fires and routes a finding to your alerting channel. You find out within a day of disclosure instead of within a day of the incident. If you run your own registry, Harbor with its built-in Trivy scanner does the same thing; on GitHub, GHCR integrates with Dependabot and the Advisory Database to flag stored images. The control is the same regardless of vendor: *scan the registry continuously, not just the build.*

**Consequence two: you must rebuild to patch.** Finding the stale CVE is half the job; the other half is fixing it. And here is the elegant part — most base-image CVEs are fixed not by changing your code but by **rebasing onto a patched base layer.** Google ships a new `distroless` digest with the patched library; Debian ships a new `slim` build; Alpine bumps `apk`. You do not need an application change. You need to *rebuild your image against the newer base and re-deploy.* This is why a regular rebuild cadence is itself a security control: an image you rebuild weekly from a freshly-pulled base picks up the latest base patches automatically, so a CVE disclosed and fixed in the base never lives long in your fleet.

The automation that makes this sustainable is the same dependency-bump tooling you use for application dependencies — Renovate or Dependabot — pointed at your `Dockerfile`'s base image and your dependency manifests. (We go deep on this in [SBOM and dependency management](/blog/software-development/ci-cd/sbom-and-dependency-management); here is the image-specific piece.) Renovate opens a pull request when a newer base digest or dependency version is available, your existing CI gate scans the rebuilt image, and if it is clean it merges and deploys — no human in the loop for the routine case:

```json
{
  "$schema": "https://docs.renovatebot.com/renovate-schema.json",
  "extends": ["config:recommended"],
  "packageRules": [
    {
      "description": "Auto-merge patch/minor base-image and dep bumps once CI (incl. scan) is green",
      "matchUpdateTypes": ["patch", "minor", "digest"],
      "matchManagers": ["dockerfile", "npm"],
      "automerge": true,
      "automergeType": "pr",
      "platformAutomerge": true
    },
    {
      "description": "Pin base images by digest so a rebase is an explicit, reviewable change",
      "matchManagers": ["dockerfile"],
      "pinDigests": true
    }
  ],
  "vulnerabilityAlerts": {
    "labels": ["security"],
    "automerge": true,
    "schedule": ["at any time"]
  },
  "schedule": ["before 6am on monday"]
}
```

The `vulnerabilityAlerts` block is the security accelerator: when a CVE is disclosed against a dependency you use, Renovate raises a fix PR *immediately*, off the regular weekly schedule, and auto-merges it once the scan gate passes. `pinDigests` keeps your base reference as an immutable `@sha256:...` digest so that every rebase is a visible diff in a PR rather than a silent change under a moving tag — which matters both for reproducibility and for knowing exactly what you are deploying. Read [managing third-party and dependency risk](/blog/software-development/site-reliability-engineering/managing-third-party-and-dependency-risk) for the broader reliability framing of treating every dependency as a liability you must actively manage; the rebuild cadence here is the delivery-pipeline implementation of that posture.

#### Worked example: the six-month-old clean image that grew four criticals

A backend service was built and deployed in January. Its build-time Trivy scan was clean — zero critical, zero high — because it used a distroless base and a tight dependency set, and on build day there were no known CVEs against any of it. The team, reasonably, considered it done and moved on. The image sat in production, unchanged, for six months.

By June, four critical CVEs had been disclosed against libraries in that image's base layer — `glibc` and a TLS library among them — none of which existed on the build feed in January. The image was now carrying four exploitable criticals, and the build-time gate had no way to know, because it had run once, in January, against January's feed. What caught it was the **continuous registry scan**: the ECR enhanced scanning re-evaluated the stored image against June's feed, found the four new criticals, and fired a CloudWatch event to the security channel. From there the fix was almost entirely automated. Renovate's `vulnerabilityAlerts` had already opened a PR bumping the base image to a patched digest (Google had shipped the patched distroless base weeks earlier). CI rebuilt the image against the new base, the scan gate confirmed it was clean again, and the change auto-merged and deployed. **Total application code change: zero lines.** The fix was a rebase, not a refactor.

The lesson is the cadence, and it is worth stating as a rule: a build-time scan tells you the image was clean *the day you built it*; a registry scan tells you whether it is *still* clean today; and an automated base-bump rebuild is how you make "still clean" true again without an engineer touching the application. A team that only scans at build time is defending against January's threats in June. Scan the registry, rebuild on a cadence, and automate the bump, and the gap between "CVE disclosed" and "CVE patched in prod" shrinks from months to about a day.

#### Worked example: the cost arithmetic of a rebuild cadence

It is fair to ask whether a weekly automated rebuild of every image is worth the CI minutes. Run the numbers. Suppose you have 40 services, each image rebuilds in about 4 minutes on a cached build, and you rebuild weekly. That is $40 \times 4 = 160$ build-minutes per week, plus a few seconds of scan each, call it 200 minutes a week, roughly 870 minutes a month. On a managed runner at a few cents a minute, that is on the order of \$25–\$50 a month in CI cost for the entire fleet's security-rebuild cadence. Now weigh it against the alternative. A single security incident traced to an unpatched base-image CVE — the investigation, the emergency patch, the postmortem, the customer disclosure, the engineer-days consumed — runs into the tens of thousands of dollars in loaded engineering time alone, before any reputational or contractual cost. The rebuild cadence is, in expectation, one of the cheapest insurance policies in your entire delivery pipeline. The CI spend is a rounding error against a single avoided incident, and the automation means the marginal human cost per rebuild is approximately zero. The decision is not close. The only reason teams skip it is that the cost is *visible* (a line on the CI bill every month) while the benefit is *invisible* (the incidents that did not happen) — which is precisely the cognitive trap that good measurement is supposed to defeat. Track "mean age of deployed base layer" and "count of deployed images with a known fixable critical," watch both stay near zero, and the invisible benefit becomes a number you can point at.

## 6. The noise problem: triage by severity, fixability, and reachability

Even with a minimal base, the moment you turn on full-spectrum scanning across a fleet of services you will be staring at a number with a comma in it. Hundreds of lows, dozens of mediums, a long tail of "won't fix" findings in distro packages, CVEs in test fixtures, vulnerabilities in code paths you never execute. This is the **noise problem**, and it is the second way scanning dies — not by being ignored from the start, but by drowning the signal until the team learns that "scanner findings" means "background hum" and stops looking. Alert fatigue is not a personal failing; it is the predictable result of a tool that reports everything with no prioritization.

![Decision tree showing raw findings of more than three hundred filtered first by severity then by fixability then by reachability where a critical with a fix blocks now, an unfixable finding is watched, an unreachable path is deprioritized, and a short real list of three remains to fix](/imgs/blogs/image-security-scanning-and-a-minimal-attack-surface-7.png)

The cure is triage, and there is a precise order to apply the filters, because each one cuts the list further:

1. **Severity first.** Critical and high get attention; medium and low get a backlog and a cadence, not an interrupt. This is not because mediums don't matter — it is because attention is finite and you spend it where the expected damage is highest. A critical-with-fix is a today problem. A low in a rarely-loaded library is a someday problem.
2. **Fixability second.** Of the criticals and highs, separate the ones with an available patch from the ones without. The fixable ones are *actionable* — you can upgrade. The unfixable ones go onto a watchlist (and possibly an allowlist with an expiry and a mitigation), because there is no upgrade to apply yet; blocking on them only punishes the team. The moment a fix ships, they jump to the top of the actionable list.
3. **Reachability third, and this is the one that cuts the deepest.** A finding tells you a vulnerable *package version* is present. **Reachability analysis** asks a sharper question: is the vulnerable *function* in that package actually called from your code, transitively, along any path? An enormous fraction of findings are in code paths your application never executes — a CVE in a CSV parser of a library you only use for its JSON support, a vulnerability in a deprecated method nothing calls. Tools like Snyk, and increasingly the open-source scanners, can do call-graph analysis to mark findings as "reachable" or "not reachable from your code." A finding that is not reachable is real but not exploitable *in your image*, and deprioritizing it is legitimate. Reachability commonly cuts an actionable list by half or more, which is the difference between a team that triages and a team that gives up.

Notice how §3 and §6 compound. A minimal base means step 1 starts with a short list instead of a long one. **Distroless is what makes triage tractable in the first place** — when the scan returns three findings, you barely need a triage process; you just read them. The noise problem is, to a large degree, an *artifact of a bloated image*. Ship less, and most of the noise was never generated. Triage handles the residue.

The output of triage is a short, ordered, *actionable* list — the handful of CVEs that are severe, fixable, and reachable — and that list is what your gate enforces and your team fixes. Everything else is tracked, not blocked. Here is a triage scan that filters down to the actionable set in one command, suitable for a daily report:

```bash
# Daily actionable report: critical+high, fixable only, machine-readable
trivy image \
  --severity CRITICAL,HIGH \
  --ignore-unfixed \
  --format json \
  ghcr.io/acme/checkout:latest \
  | jq '[.Results[].Vulnerabilities[]?
         | {id: .VulnerabilityID, pkg: .PkgName,
            installed: .InstalledVersion, fixed: .FixedVersion,
            severity: .Severity}]'
```

That pipeline takes the raw scan and emits only the findings that pass the severity-and-fixability filter, as clean JSON you can post to a channel or feed into a ticket. Add reachability (via a tool that supports it) and the list shrinks again. The goal is always the same: hand a human a list short enough that they will actually read it and act on it. A scanner's job is not to report vulnerabilities. It is to *get the right ones fixed*, and a finding nobody triages is a finding nobody fixes.

There is a deeper point about *ownership* hiding inside the noise problem, and it is what separates teams that triage from teams that drown. A pile of findings with no owner is a pile nobody fixes — the diffusion of responsibility is total. The fix is to route findings to the team that owns the image, automatically, with enough context to act, and to give each severity tier a service-level objective for remediation: critical-with-fix patched within, say, 48 hours; high within a week; medium on the regular monthly cadence; low tracked but unscheduled. Those numbers are policy, and you should set them to something you can actually meet rather than something aspirational that everyone ignores. A remediation SLO you hit 90% of the time is worth more than a stricter one you hit 30% of the time, because the first one is a real commitment and the second is a fiction that trains the team to disbelieve the whole program. Measure the SLO honestly — the metric is "time from CVE disclosure (or from the fix becoming available) to the patched image running in production" — and the registry-scan plus base-bump automation from §5 is what lets you hit aggressive numbers without heroics.

One subtlety on reachability worth flagging so you do not over-trust it: reachability analysis is a *static* call-graph approximation, and it can be wrong in both directions. It can mark a finding "unreachable" when the path is reached via reflection, dynamic dispatch, or a configuration-driven plugin the static analysis could not follow — a false sense of safety. And it can mark a finding "reachable" when the path is only hit by dead code or a disabled feature flag — needless noise. So reachability is a *prioritization* signal, not a *gate* signal: use it to order your actionable list and to justify deprioritizing low-severity unreachable findings, but do not use "unreachable" as a reason to permanently ignore a *critical* finding, because the cost of the static analysis being wrong about a critical is too high. Severity and fixability gate; reachability sorts.

| Filter | Question it answers | Typical reduction | Action on what passes |
| --- | --- | --- | --- |
| Severity | Is it bad enough to interrupt? | drops the long tail of lows | critical/high → continue |
| Fixability | Can I even do anything? | drops unfixable (`--ignore-unfixed`) | fixable → continue; unfixable → watchlist |
| Reachability | Is the vulnerable code ever called? | often halves the remainder | reachable → fix now; unreachable → deprioritize |
| **Result** | **What must I fix today?** | **hundreds → single digits** | **the gate's block list** |

## 7. The image as one link in the supply chain

I want to place this post precisely, because it is easy to over-claim what scanning gives you. Scanning answers exactly one question about your image: **what is in it.** It inventories the packages and tells you which carry known vulnerabilities. That is genuinely valuable — it is the smoke detector — but it is half of a larger story about whether you can *trust* the image you are about to run.

![Matrix contrasting what scanning catches against what signing catches across a known dependency CVE, an image tampered with after build, a fake image planted in the registry, and an unverified base, showing the two controls cover different attacks](/imgs/blogs/image-security-scanning-and-a-minimal-attack-surface-8.png)

The other half is **provenance and integrity**, and it is answered by signing and attestation, which we cover in depth in [software supply chain security, the new frontier](/blog/software-development/ci-cd/software-supply-chain-security-the-new-frontier). The questions scanning *cannot* answer:

- **Is this the image we built?** A scanner is perfectly happy to scan a malicious image. If an attacker pushes a tampered image to your registry under the same tag, the scanner will dutifully report its (possibly clean) findings. **Signing** — with `cosign` from the Sigstore project — answers this: you sign the image at build time, and you *verify* the signature at deploy time, so the cluster refuses to run anything you did not sign.
- **What was it built from, and by what?** **Provenance** (an in-toto / SLSA attestation) is a signed record of the build: which source commit, which builder, which inputs. It lets you verify not just that the image is yours, but that it came out of your trusted pipeline and not someone's laptop.
- **What exactly is inside, in a form I can re-check later?** The **SBOM** — which the scanner already generates as a byproduct of its inventory — is the durable artifact you attach to the image so that, when a CVE is disclosed tomorrow, you can query "which of my deployed images contain the affected package?" without re-scanning everything.

The clean way to hold the distinction: **scanning tells you what is in the image; signing proves it is the image you built; provenance proves how it was built; the SBOM lets you re-ask the question later.** They are complementary controls against different attacks. Scanning catches a known-vulnerable dependency you pulled in honestly. Signing catches an attacker who swaps your image for theirs. Neither substitutes for the other — an image can be perfectly signed and full of critical CVEs, or perfectly clean and unsigned (and therefore untrustworthy). This post is the "what's in it" half; treat it as one layer in the stack, and read the supply-chain post for the layers that prove origin. The microservices view, [CI/CD and independent deployability](/blog/software-development/microservices/ci-cd-and-independent-deployability), frames why this matters at fleet scale: when a hundred services each pull from a shared base, a single poisoned base layer is a single point of compromise for the whole fleet, which is exactly why provenance on the base image is not optional.

## 8. War story: when the supply chain is the attack

Scanning's limits and its value both come into focus in the real supply-chain incidents of recent years. These are accurate accounts; I'll flag where I'm generalizing.

**SolarWinds (2020).** Attackers compromised the *build system* of SolarWinds' Orion product and inserted a backdoor into the software during compilation. The shipped builds were legitimately signed by SolarWinds' own signing infrastructure, so signature verification *passed* — the malicious code came from inside the trusted pipeline. A vulnerability scanner would not have caught it either: the backdoor was not a known CVE in a third-party package; it was novel, planted code. This is the case that motivates *provenance*: it is not enough to verify that the image is signed by the vendor if the vendor's build system is the thing that was compromised. You need to verify *what the build did*, not just *who signed the output*. Scanning and signing both missed SolarWinds; only build-integrity attestation addresses it.

**Codecov (2021).** Attackers modified the Codecov Bash uploader script — a tool many CI pipelines piped directly from a URL into a shell. The modified script exfiltrated environment variables, including secrets and cloud credentials, from every CI run that used it. The relevant lessons for *this* post: the attack rode in through a build-time dependency (the uploader), and the broad exposure was amplified by CI jobs running with far more privilege and far more secrets in scope than they needed. A minimal CI surface — pinning the uploader by digest, not piping `curl | bash` from a mutable URL, scoping secrets tightly with OIDC instead of long-lived keys — would have blunted both the entry and the blast radius.

**The `xz`/`liblzma` backdoor (CVE-2024-3094, 2024).** A multi-year social-engineering effort planted a backdoor into the `xz` compression library, which is a dependency of `liblzma`, which is linked into `sshd` on many distributions. It was a near-miss of staggering scope — caught essentially by luck (a Postgres engineer noticed `sshd` was a half-second slow) days before it would have shipped into stable distributions worldwide. Here is the sobering part for scanning advocates: when it was *unknown*, no scanner could catch it — it was a zero-day, by definition. The moment it was disclosed and assigned `CVE-2024-3094`, every scanner caught it instantly, and the teams that found out fastest were the ones running *continuous registry scanning* against the current feed, not just a build-time gate. The `xz` story is the clearest possible argument for both halves of this post: minimize what you ship (a distroless image that does not even contain `xz` was never exposed) and scan continuously (so the instant it was disclosed, you knew which of your running images contained it).

The thread through all three: the image is one link in a chain that runs from a developer's commit through your build system, your dependencies, your registry, and into your cluster. Scanning hardens one link by telling you what known-bad software is inside. It does not harden the others. A complete posture minimizes the surface (fewer links to attack), scans continuously (catch known-bad fast), signs and attests (prove origin and integrity), and scopes the pipeline's privileges tightly (bound the blast radius if a link breaks). Scanning without the rest is a smoke detector in a house with no locks.

## 9. How to reach for this (and when not to)

Every control has a cost, and the discipline is matching the control to the risk instead of cargo-culting the maximal setup. Here is how I would sequence it, and where I would stop.

**Always do these, even for a tiny team — they are nearly free and high-leverage:**

- **Use a minimal base image.** This is the single highest-return move and it costs you a few lines in a Dockerfile. Distroless or `alpine` or `scratch` (for static binaries) eliminates most of your CVE surface before you scan anything. Do this first; it makes everything downstream cheaper.
- **Run as non-root with a read-only root filesystem.** Pure configuration, no new tooling. `runAsNonRoot`, `readOnlyRootFilesystem`, `drop: ["ALL"]`.
- **Add a build-time scan gate on CRITICAL-with-fix.** One CI step, one required check. Start narrow (critical only, fixable only) so the gate is trusted from day one.

**Do these as soon as you have a registry and a fleet:**

- **Continuous registry scanning.** Turn on the scanning your registry already offers. It is usually a config flag, and it is the only thing that catches the stale-clean-scan problem.
- **Automated base-image bumps** via Renovate or Dependabot, with auto-merge gated on the scan passing. This is what makes patching sustainable instead of a quarterly fire drill.
- **An allowlist discipline with expiries.** The moment you need your first exception, set up `.trivyignore` *with* `expired_at` and a justification, so you never start the habit of silent forever-ignores.

**Add these when scale or compliance demands it:**

- **Reachability analysis** (commercial tooling like Snyk, or emerging open-source support). Worth it when your actionable list is still long *after* a minimal base and triage — typically a larger org with many services. For a small team on distroless, the list is already short enough that reachability is gold-plating.
- **Signing, provenance, and SBOMs.** Necessary for a serious supply-chain posture and for SLSA-level compliance, but a separate body of work — see the supply-chain post.

**When NOT to do something:**

- **Do not gate on every severity.** Blocking on unfixable mediums and lows produces a gate the team routes around, which is worse than no gate. Block CRITICAL-with-fix; report the rest.
- **Do not treat a clean build scan as "secure."** It is one layer against one category of risk. A clean scan and a logic bug ship a compromised service just fine.
- **Do not bolt on reachability analysis before you have done the free stuff.** A bloated base with a fancy reachability tool is more expensive and less effective than a distroless base with a basic scan. Surface reduction first, always.
- **Do not let the allowlist become a graveyard.** No expiry, no entry. An exception without a renewal date is a vulnerability you have decided to forget.

The whole posture compresses to a sentence: **ship less, scan what's left as a gate, keep scanning what's deployed, and never knowingly ship a fixable critical.** The cheapest of those — shipping less — is also the most effective, which is the happy result you almost never get in engineering.

## 10. Key takeaways

- **A scanner inventories your image's OS packages and language dependencies and matches them against CVE feeds.** It catches *known* vulnerabilities in software you did not write. It is blind to your logic bugs, to zero-days, and to most misconfiguration. Scanning is one layer, not security.
- **Minimizing the attack surface is the first and best defense.** You cannot exploit what is not there. A distroless image with three packages scans at 0–2 CVEs; a full-distro base with 900 packages scans at hundreds. Ship less, and most vulnerabilities were never in the image to begin with.
- **A minimal base is also what makes scanning *meaningful*.** A scan that returns three findings gets read and fixed; a scan that returns two hundred and eighty gets ignored, including the eleven that matter. Surface reduction converts scanning from theater into a working control.
- **Run the scan as a pipeline gate, and block on the right thing: CRITICAL (and HIGH) severity *with a fix available*.** Block-everything makes a gate the team routes around; block-nothing is the same as no gate. Make it a required check.
- **Allowlist accepted risk with an expiry and a justification, never silently.** `.trivyignore` entries with `expired_at` force a renewal decision, so an accepted risk can never quietly become permanent.
- **A clean build-time scan goes stale.** CVEs are disclosed against images you already shipped. Scan the registry continuously so a vulnerability disclosed six months after build still pages you.
- **Patch by rebuilding, not refactoring.** Most base-image CVEs are fixed by rebasing onto a patched base layer — zero application change. Automate the base-image bump with Renovate/Dependabot and a regular rebuild cadence.
- **Tame the noise by triaging on severity, then fixability, then reachability.** A finding that is unreachable or unfixable is not a today problem. The output is a short, actionable list; the rest is tracked, not blocked.
- **The image is one link in the supply chain.** Scanning tells you *what is in* the image; signing proves it *is the image you built*; provenance proves *how* it was built. They are complementary controls against different attacks — none substitutes for another.

## Further reading

- **[From commit to production, the CI/CD map](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model)** — the series map: commit → build → test → package → deploy → operate, and where image scanning sits in it.
- **[Writing a production Dockerfile](/blog/software-development/ci-cd/writing-a-production-dockerfile)** — how to build the minimal, multi-stage, distroless, non-root image that this post scans.
- **[Building images fast and securely in CI](/blog/software-development/ci-cd/building-images-fast-and-securely-in-ci)** — the CI build that produces the image and where the scan step slots in.
- **[Software supply chain security, the new frontier](/blog/software-development/ci-cd/software-supply-chain-security-the-new-frontier)** — the other half: signing, provenance, and proving the image is the one you built.
- **[SBOM and dependency management](/blog/software-development/ci-cd/sbom-and-dependency-management)** — the durable inventory and the dependency-bump automation that makes patching sustainable.
- **[Managing third-party and dependency risk](/blog/software-development/site-reliability-engineering/managing-third-party-and-dependency-risk)** — the reliability framing of treating every dependency as an active liability.
- **Trivy documentation** (`aquasecurity/trivy`), **Grype** (`anchore/grype`), and **Clair** — the scanner docs for the gate and the registry scan.
- **The SLSA framework** (`slsa.dev`) and **Sigstore / cosign** (`sigstore.dev`) — the provenance and signing standards that complete the supply-chain picture.
- **Google's distroless project** (`GoogleContainerTools/distroless`) — the minimal base images that make scanning a short, actionable list.
