---
title: "Securing the pipeline itself: hardening CI/CD against attackers"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Learn how to defend your CI/CD pipeline against poisoned pipeline execution, credential theft, and runner compromise using GitHub Actions hardening, ephemeral runners, and the OWASP CICD-SEC taxonomy."
tags:
  [
    "ci-cd",
    "devops",
    "security",
    "github-actions",
    "supply-chain-security",
    "pipeline-hardening",
    "owasp",
    "least-privilege",
    "ephemeral-runners",
    "secrets-management",
    "ppe-attack",
    "ci-security",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/securing-the-pipeline-itself-1.png"
---

The call came in at 2:47 AM on a Tuesday. An alert fired on an unexpected registry push — a new image had been pushed to the production container registry from a workflow run that nobody on the team had authorized. By the time the on-call engineer traced the audit trail, the story was all there in the GitHub Actions logs: a contributor had opened a pull request ten hours earlier with what looked like a documentation update. The workflow triggered. A step in that workflow called a shell script that the PR had quietly modified. The shell script read the `REGISTRY_TOKEN` environment variable — which was a repository-level secret visible to all workflow runs — and posted it to an external URL. The attacker then used that token to push a backdoored image tagged as `v2.1.4`.

The team had a supply chain attack in production by 5 AM. The actual exploit was three lines of bash. The root cause was not a missing dependency scan, not an SBOM gap, not a missing Sigstore signature. The root cause was that nobody had treated the CI runner itself as an attack surface.

The remediation took two weeks: rotate every secret the pipeline had ever had access to (six credentials across four services), rebuild every image pushed in the past 30 days from scratch to ensure no backdoored image remained in any environment, work through the dependency tree of every service that pulled the compromised image, notify the security team of potential exposure, file a post-incident report, and implement the controls described in this post to prevent recurrence. The cost — in engineering time, in incident response, in customer trust erosion — was dramatically higher than the cost of preventing the breach would have been. That is the economic argument for pipeline hardening: prevention is cheap, remediation is expensive.

This post is about that attack surface. It covers the poisoned pipeline execution (PPE) family of attacks, why the GitHub Actions permissions model has two trigger keywords that behave completely differently, how to pin actions by commit SHA, why persistent self-hosted runners are a ticking clock, how to configure ephemeral runners via Actions Runner Controller (ARC), and how to apply the full OWASP CICD-SEC taxonomy to audit your own pipeline. You will finish with a concrete hardened GitHub Actions workflow you can drop into a real repository today.

![Diagram showing how a pull request attack branches into safe pull_request trigger versus dangerous pull_request_target trigger that exposes secrets and leads to prod compromise](/imgs/blogs/securing-the-pipeline-itself-1.png)

---

## The CI runner as an attack surface

Every pipeline security conversation eventually arrives at the same uncomfortable fact: a compromised CI runner has roughly the same access as a compromised production server, except it also has access to your source code, your build cache, your signing keys, and the credentials to every environment you deploy to.

Put the DORA metrics frame on it. You have worked hard to push deploy frequency from once a month to thirty times a day. Each of those thirty daily deploys runs through a CI pipeline. The pipeline holds:

- The `GITHUB_TOKEN` (or an equivalent CI service account token), which can create releases, push commits, and update deployment environments.
- Registry credentials for pushing images — credentials that, if leaked, let an attacker push an arbitrary image to your production tag.
- Cloud deploy credentials: an AWS IAM key, a GCP service account JSON, or an OIDC token that can call `aws sts assume-role` and get keys to your prod account.
- Secrets for external services: database passwords, third-party API keys, Slack webhook URLs.

The pipeline also runs code that it downloads at runtime: third-party GitHub Actions from the marketplace, npm packages, pip packages, Docker base images. Any of those can be the vector.

The question is not "is our pipeline a target." It is. The SolarWinds breach of 2020 started in a build system. The Codecov breach of 2021 was a bash script that was downloaded as part of a CI step and quietly exfiltrated environment variables — specifically `CODECOV_TOKEN`, which then gave attackers access to the source code of thousands of companies. The 3CX breach of 2023 traced back to a compromised upstream software dependency that was compiled into a CI build.

The CICD-SEC-4 category from OWASP names this specifically: Poisoned Pipeline Execution. But the full OWASP CICD-SEC taxonomy covers ten distinct risk classes, and understanding all ten is the right frame for hardening your pipeline comprehensively. We will work through the most critical ones, starting with the one that bites teams most often.

### Why pipeline security is different from application security

Most security programs invest heavily in the running application: WAFs, runtime protection agents, network segmentation, secret scanning in code review. The pipeline is harder to reason about because it is not a persistent service — it runs, finishes, and disappears. But that transient nature is exactly what makes it dangerous. The pipeline:

1. **Runs privileged code by design.** The whole point of a build pipeline is to compile code, run tests, push artifacts, and trigger deployments. Every capability it needs to do its job is also a capability an attacker could abuse.
2. **Has write access to the artifact store.** Whatever the pipeline produces is what gets deployed. If an attacker can influence the artifact, they own the deployment.
3. **Is often the only path to production credentials.** In well-designed systems, no human has direct production access. The pipeline is the bridge. That bridge is the target.
4. **Runs code from multiple untrusted sources simultaneously.** Your source code, third-party actions, npm packages, pip packages, and Docker base layers all execute in the same build environment. Any one of them can be the vector.

The 2023 CISA advisory on CI/CD security noted that pipeline attacks were the fastest-growing category in supply chain compromise. The investment in runtime security has pushed attackers upstream — if you can own the build, you do not need to own the running application.

### Quantifying the blast radius

A useful mental exercise before designing controls is to enumerate what a fully compromised pipeline can do. Take a typical GitHub Actions workflow for a Go microservice deployed to AWS EKS:

- **Source code access**: the pipeline clones the full repository. A compromised step can exfiltrate all source code, including any embedded secrets or API keys that were accidentally committed.
- **Dependency access**: the pipeline downloads all dependencies. A step can enumerate what packages are in use — valuable for targeted supply chain attacks against your specific stack.
- **Artifact write access**: the pipeline pushes the built image to the registry. A compromised step can push a backdoored image with the same tag as a legitimate release.
- **Signing access**: if the pipeline runs `cosign sign`, it has access to the signing key (or OIDC identity). Signing a malicious image makes it indistinguishable from a legitimate one to any downstream verification step.
- **Deployment credentials**: the pipeline has cloud credentials (IAM key or OIDC token). A compromised step can call `aws s3 cp`, `gcloud kms encrypt`, or `kubectl exec` against production.
- **Cross-repository access**: a `GITHUB_TOKEN` with `contents: write` at the organization level can push commits to other repositories in the organization. One compromised pipeline can poison every repository it has access to.

The blast radius is not uniform — it depends on exactly which credentials the pipeline holds and what they are scoped to. The controls in this post are all about reducing that scope so that a single compromised step cannot reach all of these targets simultaneously.

---

## The poisoned pipeline execution (PPE) attack

PPE attacks exploit the fact that a CI pipeline must, by design, execute code from the repository. The job of the pipeline is to build and test that code. An attacker who can influence the code that the pipeline executes can therefore influence what the pipeline does — including stealing the secrets it has access to.

There are three variants.

**Direct PPE** is the simplest. The attacker has write access to the repository (either a legitimate contributor, or someone who has stolen a contributor's credentials) and directly edits the pipeline YAML file — `.github/workflows/build.yml`, `.gitlab-ci.yml`, `Jenkinsfile`. They add a step that reads secrets and posts them to an external URL, or replaces the legitimate build command with one that installs a backdoor. Direct PPE requires write access to the main branch, so a well-configured branch protection rule (requiring review and status checks before merge) stops it. Most teams configure this. The more dangerous variants are the ones that bypass branch protection.

**Indirect PPE** exploits the fact that pipeline YAML files often call external scripts: `bash scripts/build.sh`, `python tools/release.py`, `make deploy`. An attacker who can modify those scripts — even in a branch, even in a fork — can influence what the pipeline does without touching the YAML file itself. The YAML file appears clean in a review. The shell script one directory deeper contains the exploit.

**Pull-request-from-fork attacks** are indirect PPE against public repositories. A contributor forks your public repository, modifies a build script in their fork, and opens a pull request. Your pipeline runs against their code. If the workflow is configured with the right trigger, their code executes with access to your repository's secrets.

The critical variable in all three variants is the GitHub Actions trigger keyword.

### `pull_request` vs `pull_request_target`

GitHub Actions has two triggers that fire when a pull request is opened against a repository:

- `on: pull_request` — the workflow runs in the context of the **head branch** (the PR branch), but with **read-only** access. For pull requests from forks, secrets are not available, and the `GITHUB_TOKEN` has limited read permissions only. This is the safe default.

- `on: pull_request_target` — the workflow runs in the context of the **base branch** (the branch being merged into), but against the **head branch code**. This means the workflow has full write permissions and access to all repository secrets — including secrets from the target repository, not just the fork. This was introduced to allow CI checks that need to post results back to the PR (because a forked PR run cannot write a comment or update a status check). But it is extraordinarily dangerous when combined with steps that execute code from the PR branch.

The pattern that leads to a full compromise looks like this:

```yaml
# DANGEROUS — do not use this pattern
on:
  pull_request_target:
    types: [opened, synchronize]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha }}  # checks out the PR branch code
      - run: bash scripts/test.sh  # runs attacker-controlled code with full secrets access
```

The `actions/checkout` step with `ref: ${{ github.event.pull_request.head.sha }}` checks out the forked branch's code. The `run: bash scripts/test.sh` executes whatever is in that code. The workflow is running under `pull_request_target`, so `secrets.REGISTRY_TOKEN` is available. Three lines later, the attacker has your token.

The safe alternative is to never combine `pull_request_target` with steps that check out PR code. If you need `pull_request_target` for write-back (posting comments, updating check status), split the workflow into two separate files: one that runs the dangerous build/test work under `pull_request` (no secrets, read-only), and a second that runs under `pull_request_target` for the write-back and receives only the results from the first run as an artifact — never the code.

The safe two-file pattern looks like this:

```yaml
# .github/workflows/pr-check.yml — runs the actual work, no secrets
on:
  pull_request:
    branches: [main]

permissions: {}

jobs:
  test:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
      - name: Run tests
        run: make test
      - name: Save test results
        run: echo "${{ job.status }}" > test-result.txt
      - uses: actions/upload-artifact@ea165f8d65b6e75b540449e22b4372ded7b2a89b # v4.6.0
        with:
          name: test-results-${{ github.event.pull_request.number }}
          path: test-result.txt
```

```yaml
# .github/workflows/pr-comment.yml — posts results back, NO checkout of PR code
on:
  workflow_run:
    workflows: ["pr-check"]
    types: [completed]

# This workflow runs in the base branch context and CAN post comments
permissions:
  pull-requests: write

jobs:
  comment:
    runs-on: ubuntu-latest
    steps:
      - name: Download test results
        uses: actions/download-artifact@v4
        with:
          name: test-results-${{ github.event.workflow_run.pull_requests[0].number }}
          run-id: ${{ github.event.workflow_run.id }}
          github-token: ${{ secrets.GITHUB_TOKEN }}
      - name: Post PR comment
        uses: actions/github-script@v7
        with:
          script: |
            const result = require('fs').readFileSync('test-result.txt', 'utf8').trim();
            github.rest.issues.createComment({
              issue_number: ${{ github.event.workflow_run.pull_requests[0].number }},
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `Test run completed with status: ${result}`
            });
```

This pattern completely decouples the dangerous operation (executing PR code) from the privileged operation (writing a PR comment). The first workflow checks out PR code but has no secrets. The second workflow has write permissions but never touches PR code. An attacker who modifies the PR cannot influence what the second workflow does — it only reads a pre-uploaded artifact.

### Detecting PPE attempts

Prevention is the primary control, but detection is the backstop. Signs that a PPE attempt occurred or was attempted:

- A `pull_request_target` workflow job made an outbound HTTPS connection to a domain not in the build's normal egress list.
- A `pull_request_target` workflow accessed a secret that it did not access in any prior run of the same workflow.
- A workflow step exited with a non-zero status while also making an outbound connection (the script may have tried to exfiltrate but failed).
- A new YAML file appeared in `.github/workflows/` in a PR from a first-time contributor.

That last point deserves attention. A branch protection rule that requires review of changes to workflow files is a practical mitigation for direct PPE. GitHub's CODEOWNERS feature lets you specify that `.github/workflows/` requires approval from a specific team — typically the platform or security team — before any workflow change can be merged.

```bash
# .github/CODEOWNERS
# All workflow file changes require review from the platform team
.github/workflows/  @myorg/platform-team
.github/           @myorg/platform-team
scripts/            @myorg/platform-team
Makefile            @myorg/platform-team
```

The `scripts/` and `Makefile` lines are the often-missed ones. Indirect PPE works by modifying scripts that the pipeline calls. Protecting those scripts with the same CODEOWNERS rule closes the indirect PPE vector.

---

## Pinning GitHub Actions by commit SHA

Every `uses:` line in a GitHub Actions workflow downloads and executes code from an external repository. The version specifier in that line determines which code gets downloaded.

```yaml
# Mutable — what the tag points to can change
uses: actions/checkout@v4

# Also mutable, just slower to change
uses: actions/setup-node@v4.1.0

# Immutable — this SHA cannot be re-pointed
uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683
```

A Git tag is a pointer. A pointer can be moved. A malicious actor who gains write access to the `actions/checkout` repository (by compromising an Anthropic or GitHub maintainer account, by finding a credential in a commit, by social engineering a repository transfer) can move the `v4` tag to point at a new commit that contains malicious code. Every pipeline that uses `@v4` will then download and execute that malicious code.

This is not theoretical. In 2024, the `tj-actions/changed-files` action was compromised. The attacker modified the action's code to print all CI environment variables — including secrets — to the workflow log. Because the action was pinned by tag in thousands of repositories, the compromise affected all of them simultaneously. The blast radius was limited only by how quickly maintainers could identify affected repositories and rotate secrets.

Pinning to a full commit SHA makes this attack impossible. A SHA cannot be re-pointed. The cryptographic properties of SHA-1 (and SHA-256 for newer Git operations) mean that two different code states cannot produce the same hash. If you pin `actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683`, you will always get exactly that commit, no matter what happens to the `v4` tag.

![Diagram comparing mutable action tag which can be re-pointed to malicious code versus immutable commit SHA that always executes the exact same code](/imgs/blogs/securing-the-pipeline-itself-2.png)

The practical objection is maintenance: tracking which SHA corresponds to which version of every action you use is tedious. The answers are tooling:

- **Dependabot for Actions** — Add `package-ecosystem: "github-actions"` to your `.github/dependabot.yml`. Dependabot will open PRs that update action SHAs when new versions are released, with a comment showing the tag they correspond to.
- **`pin-github-action`** — A CLI tool from Ratchet and similar tools that reads your workflow files and replaces all `uses: action@tag` references with their current SHA, adding the tag as a comment for human readability.
- **OpenSSF Scorecard** — A tool that audits your repository against a checklist of security practices, including whether your workflows pin actions to SHAs. It produces a score and flags specific workflows that use mutable tags.

A Dependabot configuration that covers both package dependencies and GitHub Actions:

```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    groups:
      actions:
        patterns:
          - "*"

  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
    versioning-strategy: increase-if-necessary
```

After running `pin-github-action` on your workflows, the `uses:` lines look like this:

```yaml
steps:
  - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
  - uses: actions/setup-node@1d0ff469b12461b29b1af44df672f99c28b50387 # v4.2.0
  - uses: docker/build-push-action@48aba3b46d1b1fec4febb7c5d0c644b249a11355 # v6.7.0
```

The comment preserves human readability; the SHA preserves security.

---

## Least-privilege CI permissions

The GitHub Actions permissions model gives every workflow run a `GITHUB_TOKEN` — an automatically generated short-lived token scoped to the repository. Before mid-2023, the default scope for this token was broad write access. GitHub changed the default, but many repositories still carry old workflows that were written under the permissive default, and many teams have not audited what their `GITHUB_TOKEN` can actually do.

The `permissions:` key in a workflow file controls exactly what the token can do. It can be set at the workflow level (applying to all jobs) and overridden per job. The principle of least privilege says: start at `read-all`, then explicitly add the minimum write scope each job actually needs.

![Diagram of CI privilege layers showing GITHUB_TOKEN read-only at the base escalating through explicit write grants, environment secrets, deploy OIDC role, up to prod access at the top](/imgs/blogs/securing-the-pipeline-itself-3.png)

Here is what a hardened workflow looks like:

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

# Default: no permissions at all — every job must declare what it needs
permissions: {}

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    permissions:
      contents: read          # checkout the repo
      packages: read          # pull images from GHCR
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
      - name: Run tests
        run: make test

  publish-image:
    runs-on: ubuntu-latest
    needs: [build-and-test]
    if: github.ref == 'refs/heads/main'
    permissions:
      contents: read
      packages: write         # push image to GHCR — only this job needs it
      id-token: write         # request OIDC token for keyless signing
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
      - name: Build and push
        run: |
          docker buildx build --push -t ghcr.io/myorg/myapp:${{ github.sha }} .

  deploy-staging:
    runs-on: ubuntu-latest
    needs: [publish-image]
    environment: staging       # requires environment approval; secrets scoped here
    permissions:
      contents: read
      id-token: write          # OIDC federation to AWS
    steps:
      - uses: aws-actions/configure-aws-credentials@e3dd6a429d7300a6a4c196c26e071d42e0343502 # v4.0.2
        with:
          role-to-assume: arn:aws:iam::123456789012:role/GHAStaging
          aws-region: us-east-1
      - name: Deploy to staging
        run: kubectl rollout status deployment/myapp -n staging
```

Several patterns here are worth unpacking.

**`permissions: {}`** at the workflow level explicitly revokes the default token scope. Every job that needs any permission must declare it explicitly. This is more aggressive than `permissions: read-all` but guarantees that no accidental write access exists.

**Job-level permissions override the workflow default.** The `build-and-test` job gets `contents: read` and `packages: read`. The `publish-image` job gets `packages: write` and `id-token: write`. The `deploy-staging` job gets `id-token: write` for OIDC federation. Each job has only what it needs.

**`environment:` for secrets scoping.** Environment-level secrets are only available to jobs that reference that environment. Repository-level secrets are available to all workflow runs. Anything genuinely sensitive — a production deploy credential, a signing key — should be an environment secret, not a repository secret.

**OIDC federation instead of long-lived keys.** The `id-token: write` permission allows the job to request a short-lived OIDC token from GitHub's OIDC provider. AWS, GCP, and Azure all support trust relationships with GitHub's OIDC provider. The job exchanges the OIDC token for a short-lived cloud credential, and no static secret ever needs to be stored in GitHub. This eliminates the entire class of "leaked IAM key" incidents. For more on this pattern, see the dedicated post on [secrets management in the pipeline](/blog/software-development/ci-cd/secrets-management-in-the-pipeline).

### OIDC trust policy: locking it down correctly

OIDC federation is only as secure as the trust policy. A common mistake is to configure an overly broad trust:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::123456789012:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:myorg/*:*"
        }
      }
    }
  ]
}
```

This trust policy allows any workflow in any repository under `myorg` to assume the role. An attacker who can create a repository in `myorg` — or compromise any workflow in any existing repository — can assume this role. The correct constraint locks down to the specific repository, specific environment, and optionally specific workflow:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::123456789012:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:sub": "repo:myorg/myapp:environment:production",
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        }
      }
    }
  ]
}
```

The `sub` claim with `:environment:production` means this role can only be assumed by a workflow job running in the `production` environment — which itself requires explicit approval from the `prod-approvers` team. The constraint chain is: GitHub account → organization → specific repository → specific environment → specific required reviewers.

### Secrets scoping across environments

The separation between repository secrets and environment secrets is a significant security boundary. Here is the practical difference:

| Secret type | Accessible to | When to use |
|-------------|--------------|-------------|
| Repository secret | All workflow runs in the repository | Non-sensitive shared values; internal registry URLs |
| Environment secret | Only jobs referencing that environment | Deploy credentials; signing keys; prod API tokens |
| Organization secret | Selected repositories in the org | Shared monitoring/observability credentials |
| OIDC (no static secret) | The specific job that requests it | Cloud deploy credentials — always prefer OIDC |

The principle is: move secrets as close to the point of use as possible. A production deploy credential that is a repository-level secret is accessible to a workflow that runs on a PR from a first-time contributor — because `pull_request` against public repos still runs workflows (though without secrets for forks). An environment-level secret scoped to `production` with required reviewers is accessible only when a human explicitly approves the deployment.

For the pattern of injecting secrets from HashiCorp Vault or AWS Secrets Manager at runtime — rather than storing them in GitHub at all — see the [secrets management in the pipeline](/blog/software-development/ci-cd/secrets-management-in-the-pipeline) post. That pattern removes GitHub as a secrets store entirely, reducing the attack surface further.

### Separating build jobs from deploy jobs

The deploy job should be a separate job in the workflow, not a step appended to the build job. The reasons are both operational and security-oriented:

- Deploy jobs run against specific environments, which can require manual approval gates (GitHub's `environment:` feature supports required reviewers).
- Deploy jobs need different permissions than build jobs — specifically, cloud credentials that build jobs should never hold.
- If a build fails, the deploy job is skipped automatically via `needs:` dependency. You never need a conditional like `if: ${{ success() }}` to prevent a bad build from deploying.
- A compromised build job cannot escalate to deploy permissions because the deploy job's credentials are not in scope during the build job's execution.

---

## Ephemeral self-hosted runners

GitHub-hosted runners (the `ubuntu-latest`, `windows-latest`, etc. images) are ephemeral by design. Each workflow job gets a fresh virtual machine that is destroyed after the job completes. Nothing persists between runs. This is a significant security property that many teams underestimate until they move to self-hosted runners.

Self-hosted runners are popular for cost reduction (GitHub-hosted runners are billed per minute; large organizations can spend \$40k–\$100k per month), for access to specific hardware (GPUs, specific CPU architectures, on-premises networks), or for compliance reasons (regulated industries that cannot send source code to GitHub's infrastructure). The problem is that the most common self-hosted runner configuration — a long-running VM or bare-metal server with the runner agent installed — is a persistent environment that accumulates state.

A persistent runner accumulates:

- **Cached credentials** — Docker credential helpers that were configured during a prior run may still be active. A `docker login` in one job can leave credentials in `~/.docker/config.json` that a subsequent job can read without re-authenticating.
- **Environment variables** — Some CI setups source environment variables from files on the runner host. A malicious job can write to those files; a subsequent job reads them.
- **Build cache artifacts** — Gradle, pip, npm, and Cargo caches left over from prior builds. A malicious package that was included in one build's cache can be read by a subsequent build.
- **Process state** — Long-running background processes started by one job that are still running when the next job starts.

The correct fix is to run self-hosted runners as ephemeral containers: each job gets a fresh container, the container is destroyed when the job completes, and nothing persists to the next job. The production-grade tooling for this on Kubernetes is **Actions Runner Controller (ARC)**.

![Diagram comparing persistent self-hosted runner that accumulates state across builds versus ephemeral ARC runner that starts clean and is destroyed after each job](/imgs/blogs/securing-the-pipeline-itself-6.png)

### Why this matters: the state accumulation attack

A persistent runner that has been running builds for two months is a repository of build history. Here is a concrete attack sequence:

1. A build job six weeks ago ran `docker login ghcr.io` to push an image. The Docker credential helper wrote the credentials to `~/.docker/config.json` on the runner host.
2. The runner team never explicitly ran `docker logout` after the job — the workflow assumed a fresh environment.
3. A new malicious PR is opened that contains a build step which runs `cat ~/.docker/config.json` and posts the output to an external URL.
4. The `pull_request` trigger fires. The step executes. The attacker receives valid registry credentials. They never needed to touch `secrets.REGISTRY_TOKEN` — the token was sitting unguarded on the runner's filesystem.

This attack is not hypothetical. It is the runner-level equivalent of a shared database password that never gets rotated. The fix is ephemeral runners: each job gets a fresh container, the container knows nothing about previous builds.

### Setting up ARC for ephemeral runners

ARC runs as a Kubernetes operator. It watches for workflow jobs queued in your GitHub repository or organization and provisions ephemeral runner pods on demand. Each pod runs one job and is destroyed immediately after.

```yaml
# arc-runner-deployment.yaml — deploy to your Kubernetes cluster
apiVersion: actions.summerwind.dev/v1alpha1
kind: RunnerDeployment
metadata:
  name: myapp-runners
  namespace: arc-runners
spec:
  replicas: 0          # scale-to-zero when no jobs are queued
  template:
    spec:
      repository: myorg/myapp
      image: summerwind/actions-runner:latest
      ephemeral: true           # destroy pod after job completes
      serviceAccountName: arc-runner-sa
      # Resource limits prevent a runaway job from consuming cluster resources
      resources:
        requests:
          cpu: "500m"
          memory: "1Gi"
        limits:
          cpu: "2"
          memory: "4Gi"
      # Run as non-root
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        readOnlyRootFilesystem: false
        allowPrivilegeEscalation: false
```

```yaml
# arc-scale.yaml — HorizontalRunnerAutoscaler scales up to meet demand
apiVersion: actions.summerwind.dev/v1alpha1
kind: HorizontalRunnerAutoscaler
metadata:
  name: myapp-runners-autoscaler
  namespace: arc-runners
spec:
  scaleTargetRef:
    name: myapp-runners
  minReplicas: 0
  maxReplicas: 10
  metrics:
    - type: TotalNumberOfQueuedAndInProgressWorkflowRuns
      repositoryNames:
        - myapp
```

To use these runners in your workflow, set the `runs-on` label:

```yaml
jobs:
  build:
    runs-on: [self-hosted, arc-runner]   # matches the runner label
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683
```

The operational effect is that every job runs in a container that was just provisioned from the base image. No state from any previous job exists. A malicious job that writes credentials to disk, modifies environment files, or installs backdoored binaries has zero effect on any subsequent job. The ephemeral property gives you the same security posture as GitHub-hosted runners while keeping the cost and access advantages of self-hosted.

---

## Network isolation for runners

Even with ephemeral runners, the runner has network access during the job. A compromised step can make outbound connections: to exfiltrate secrets, to download additional malicious tooling, or to phone home to a command-and-control server.

The "the build downloaded a malicious package" attack is a variant of this. A legitimate-looking npm package (or PyPI package, or RubyGem) contains a post-install script that reads environment variables and makes an outbound HTTPS request to a domain the attacker controls. The build script runs, the package installs, the hook fires, and the secrets are gone — all before any security scanner runs, because the scanner runs after the install step.

Network isolation for runners means: define an allowlist of outbound destinations (your artifact registry, your package mirrors, your cloud APIs), and block everything else. On a Kubernetes cluster, this is a NetworkPolicy:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: arc-runner-egress
  namespace: arc-runners
spec:
  podSelector:
    matchLabels:
      app: arc-runner
  policyTypes:
    - Egress
  egress:
    # Allow DNS
    - ports:
        - protocol: UDP
          port: 53
    # Allow HTTPS to GitHub (for the runner agent and checkout)
    - to:
        - ipBlock:
            cidr: 140.82.112.0/20
      ports:
        - protocol: TCP
          port: 443
    # Allow HTTPS to your internal artifact registry
    - to:
        - namespaceSelector:
            matchLabels:
              name: registry
      ports:
        - protocol: TCP
          port: 443
    # Allow HTTPS to your package mirror
    - to:
        - ipBlock:
            cidr: 10.0.100.0/24   # internal npm/pip/maven mirror
      ports:
        - protocol: TCP
          port: 443
    # Block everything else — no direct npm.npmjs.com, no pypi.org
```

The dependency injection here is that you need an internal package mirror if you block direct access to public registries. This is an investment — running a Nexus, Artifactory, or Verdaccio instance — but it also provides other supply chain benefits: you control which package versions are available, you can pre-scan packages before they reach runners, and your builds are not dependent on the availability of public registries (a surprisingly common CI failure mode).

For teams that cannot run a full internal mirror, the middle ground is a proxy: configure your package manager to route through a caching proxy that logs all requests and blocks known-malicious domains. All outbound requests still ultimately go to npm.npmjs.com, but you get visibility into what the build is downloading and can alert on unexpected new domains.

### Dependency confusion attacks: the network isolation connection

Dependency confusion is a specific supply chain attack that network isolation directly mitigates. The attack works because most package managers search public registries before (or in addition to) private registries. An attacker who discovers that your private npm registry contains a package called `myorg-internal-utils` can publish a package with the same name to npm.npmjs.com with a higher version number. When a runner with public internet access runs `npm install`, it may install the public (malicious) package instead of the private (legitimate) one.

The defense has two layers:

1. **Registry configuration**: configure your package manager to use only your internal registry (not as an additional source, but as the only source). For npm, this is `.npmrc`:

```bash
# .npmrc — route ALL installs through the internal registry
registry=https://registry.internal.myorg.com/
always-auth=true
//registry.internal.myorg.com/:_authToken=${NPM_TOKEN}
```

2. **Network isolation**: even if the configuration is correct, a malicious build script can `curl https://registry.npmjs.org/...` directly. Network isolation at the runner level means that connection is blocked before it can complete — the defense-in-depth layer that survives a misconfigured `.npmrc`.

The dependency confusion attack affected Shopify, Microsoft, Apple, and hundreds of other companies when security researcher Alex Birsan published his findings in 2021 — all by registering public packages with the same names as private internal packages he had discovered through leaked `package-lock.json` files. The technique is publicly known and actively exploited. Network isolation and private registry configuration are the primary mitigations.

---

## Audit logging and anomaly detection

The Codecov attack went undetected for months. The reason was not that the attackers were subtle. The reason was that nobody was watching. The audit trail existed — GitHub's audit log, the Codecov API logs, the CI environment variable logs — but nobody had wired up alerting on the patterns that would have flagged it.

GitHub's audit log is the primary data source for CI security events. It records every significant action in the platform: workflow runs, secret accesses, permission changes, repository visibility changes, and authentication events. The critical gap most teams have is that the audit log is not exported anywhere — it sits in the GitHub UI, with a 90-day retention window for free plans and 12 months for Enterprise. By the time an incident requires forensic analysis, the relevant log entries may be gone.

Export the audit log continuously to durable storage. GitHub provides streaming export via the Audit Log Streaming feature (Enterprise) or REST API polling. Route it to your SIEM — Datadog, Splunk, Elastic, Panther, or even AWS S3 + Athena for a budget option. The schema is JSON with a consistent structure that maps well to most SIEM query languages.

```bash
# Example: poll the GitHub audit log API and write to S3
# (run as a scheduled Lambda or Kubernetes CronJob)
curl -H "Authorization: Bearer $GITHUB_AUDIT_TOKEN" \
  "https://api.github.com/orgs/myorg/audit-log?per_page=100&after=$CURSOR" \
| jq '.[].@timestamp, .[].action, .[].actor, .[].repo' \
| aws s3 cp - s3://myorg-audit-logs/$(date +%Y/%m/%d)/audit-$(date +%H%M%S).json
```

The patterns to alert on:

**Secret access outside CI hours.** If your team deploys Monday through Friday between 8 AM and 8 PM, and the CI audit log shows `secret.access` events at 3 AM on a Saturday, that is worth investigating. GitHub's audit log includes secret access events for environment and repository secrets. Export the log to your SIEM (Splunk, Datadog, Elastic, Panther, or even a basic S3 + Athena setup) and write a rule: `event=secret.access AND NOT (weekday IN [Mon,Fri] AND hour BETWEEN 8 AND 20)`.

**Registry pushes from unexpected workflows.** Your production registry should receive pushes only from a specific workflow — `publish.yml`, `release.yml`, whatever you call it. A push from `test.yml` or `pr-check.yml` is anomalous. Most container registries expose an API for push events; wire those to your SIEM.

**Jobs running on unexpected repositories.** If you use organization-level runners, a new repository in the organization can use those runners. A rule that alerts when a runner job runs for the first time on a repository that was not registered as an authorized consumer catches cases where someone creates a repo specifically to exploit shared runner infrastructure.

**OIDC token claims outside expected ranges.** OIDC tokens issued by GitHub carry claims including the repository name, the workflow name, the branch name, and the environment. Your cloud IAM trust policy can validate these claims, but you can also log and alert on them. An OIDC token claiming to be from `github.com/myorg/myapp` but with a branch name of `attacker-branch` is suspicious.

**Unusual egress volume during a build.** A build step that exfiltrates secrets will usually make a small number of HTTPS requests to a domain outside your normal build footprint. Network flow logs from the runner cluster, correlated with the build timeline, can surface this: "job 12345 made a connection to 185.220.101.x that no prior build has ever made."

### OpenSSF Scorecard integration

The OpenSSF Scorecard is a free, automated tool that evaluates a repository against a set of security best practices and produces a numeric score from 0 to 10. Running it as part of your CI pipeline gives you a regression detector: if a PR reduces the score by pinning fewer actions or loosening branch protection rules, the drop surfaces in the PR check.

```yaml
# .github/workflows/scorecard.yml
name: OpenSSF Scorecard

on:
  push:
    branches: [main]
  schedule:
    - cron: "0 6 * * 1"   # also run weekly on Mondays

permissions: read-all

jobs:
  scorecard:
    runs-on: ubuntu-latest
    permissions:
      security-events: write    # to upload SARIF results to GitHub Security tab
      id-token: write
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
        with:
          persist-credentials: false

      - uses: ossf/scorecard-action@62b2cac7ed8198b15735ed49ab1e5cf35480ba46 # v2.4.0
        with:
          results_file: scorecard-results.sarif
          results_format: sarif
          publish_results: true

      - uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: scorecard-results.sarif
          category: ossf-scorecard
```

The scorecard checks include: token-permissions (do your workflows use least privilege?), branch-protection (are main-branch pushes protected?), pinned-dependencies (are actions and package dependencies pinned?), dangerous-workflow (does any workflow use `pull_request_target` unsafely?), vulnerabilities (are there known CVEs in your dependencies?), and ten other checks. Each check returns a score between 0 and 10 with detailed remediation guidance.

Treating a scorecard score drop as a blocking CI gate is an aggressive but effective approach for high-security repositories. A more practical policy for most teams: run scorecard on every push, alert on drops below 6.0, and require a security review for any PR that drops the score by more than 1.0 point.

---

## The OWASP CICD-SEC taxonomy in full

OWASP's CICD-SEC project (published 2022, updated since) defines ten risk categories for CI/CD security. Understanding all ten gives you a comprehensive audit checklist rather than whack-a-mole patching.

![Tree diagram of OWASP CICD-SEC risks grouped under pipeline integrity, credential risks, and supply chain risk categories with specific numbered risks as leaves](/imgs/blogs/securing-the-pipeline-itself-7.png)

| # | Risk | Description | Primary mitigation |
|---|------|-------------|-------------------|
| CICD-SEC-1 | Insufficient flow control mechanisms | No branch protection, no required reviews, no status-check gates | Enforce branch protection; require at least 2 reviewers for main |
| CICD-SEC-2 | Inadequate identity and access management | Shared CI service accounts; no per-pipeline identity; excessive group access | OIDC per-job identity; principle of least privilege; no shared tokens |
| CICD-SEC-3 | Dependency chain abuse | Malicious or typosquatted packages; dependency confusion attacks; compromised upstream packages | Private registry; dep pinning; Dependabot; vulnerability scanning with Trivy/Grype |
| CICD-SEC-4 | Poisoned pipeline execution | Attacker-controlled code runs in CI with access to secrets | Restrict `pull_request_target`; review all workflow files; separate untrusted code from secrets |
| CICD-SEC-5 | Insufficient PBAC (pipeline-based access controls) | Jobs have access to resources they should not need; no environment gating | `permissions:` block; environment secrets; approval gates for prod environments |
| CICD-SEC-6 | Insufficient credential hygiene | Long-lived tokens; secrets in environment variables; credentials in code | OIDC keyless; short-lived tokens; external secrets operator; credential rotation |
| CICD-SEC-7 | Insecure system configuration | Default-permissive runner settings; unpatched runner software; publicly-accessible pipeline dashboards | Harden runner images; patch regularly; network policy; restrict pipeline visibility |
| CICD-SEC-8 | Ungoverned usage of third-party services | Arbitrary third-party actions; npm packages from unknown publishers; integrations with no audit trail | Actions allowlist; package provenance verification; SBOM; third-party action review |
| CICD-SEC-9 | Improper artifact integrity validation | Unsigned artifacts; no provenance; no chain-of-custody from source to deploy | Sigstore/cosign signing; SLSA provenance; artifact checksums; verify-before-deploy |
| CICD-SEC-10 | Insufficient logging and visibility | CI audit logs not exported; no alerting on anomalous events; no record of who changed what | SIEM integration; structured audit logging; alert on secret access, anomalous pushes |

The framework is also a prioritization guide. CICD-SEC-4 (PPE) and CICD-SEC-6 (credential hygiene) are the most commonly exploited in real breaches. CICD-SEC-9 (artifact integrity) and CICD-SEC-3 (dependency chain) are the hardest to exploit but the hardest to recover from — a compromised build artifact in production with no provenance record is an incident that takes weeks to fully scope.

For deeper coverage of CICD-SEC-9, see the companion post on [signing and provenance with Sigstore and SLSA](/blog/software-development/ci-cd/signing-and-provenance-with-sigstore-and-slsa). For CICD-SEC-6 credential management, see [secrets management in the pipeline](/blog/software-development/ci-cd/secrets-management-in-the-pipeline).

### Applying the taxonomy as an audit checklist

The CICD-SEC framework's practical value is as a structured audit checklist. For each of the ten categories, ask three questions:

1. What is our current control? (What are we actually doing today to address this risk?)
2. How do we know the control is working? (Is it detectable when this control fails?)
3. What is the blast radius if this control fails? (What is the worst-case outcome?)

If the answer to question 2 is "we don't know" or "we would find out from a customer report," that is a detection gap. If the answer to question 3 is "the attacker gets full production access," that is a scope gap. Work top-down from the risks with the largest blast radius and the weakest detection. In most teams, that means CICD-SEC-4 (PPE) and CICD-SEC-6 (credentials) first, then CICD-SEC-9 (artifact integrity), then the rest.

A quarterly security sprint dedicated to working through this checklist — updating controls, testing detection, measuring blast radius — is more effective than waiting for a breach to motivate investment. The DORA research shows that high-performing teams have lower change-failure rates partly because they invest in continuous security posture improvement, not just post-incident remediation. Security and delivery velocity are complements, not trade-offs: a team that ships confidently 30 times a day with hardened pipelines is more secure and faster than a team that ships once a month with manual deploy gates and ad hoc credentials.

---

## War story: the real incidents behind these controls

### SolarWinds (2020)

The SolarWinds breach is the canonical build-system attack. Attackers with access to the SolarWinds internal network injected malicious code into the Orion build process. The CI pipeline compiled that code into a legitimate Orion update. The update was signed with SolarWinds' legitimate code-signing certificate. It was distributed to roughly 18,000 organizations through normal software update channels.

The specific failure was CICD-SEC-4 (the build process was a vector for injecting code) compounded by CICD-SEC-9 (signed artifacts gave false legitimacy — signing proves the artifact came from the expected build system, but if the build system itself is compromised, signing is insufficient). The correct response — SLSA provenance — creates a verifiable record of *what source code* was compiled to produce *which artifact*. If SolarWinds had had SLSA level 3 provenance, an auditor could have verified that the Orion artifact at version X was built from the exact source tree at commit Y with no unexpected modifications. Absent that record, the signed artifact was indistinguishable from a clean one.

### Codecov (2021)

The Codecov breach is the canonical "malicious CI step downloads a compromised script" attack. Codecov's bash uploader script — `bash <(curl -s https://codecov.io/bash)` — was modified by an attacker who had gained access to Codecov's GCS bucket. The modification added two lines that read the CI environment and posted it to a URL the attacker controlled.

The specific failure was: teams were running a shell script from an external URL in their CI pipeline without pinning it to a specific version or checksum, and with full access to their CI environment variables. Every team that ran `bash <(curl -s https://codecov.io/bash)` without a checksum verification was running arbitrary code from the internet with access to their secrets. The fix — pinning to a specific version with a verified checksum, or better, using the official GitHub Action pinned to a SHA — would have contained the impact to teams that manually updated to the compromised version.

### tj-actions/changed-files (2024)

In March 2024, the `tj-actions/changed-files` GitHub Action was compromised. Attackers modified the action's code to dump CI runner memory to the workflow log, which included environment variables containing secrets. Because the action was pinned by tag (e.g., `@v45`) in thousands of repositories, all of them automatically picked up the malicious version.

The immediate remediation was to audit and rotate any secrets that had appeared in workflow logs. The long-term fix was the one described earlier: pin all actions to a full commit SHA. If every consumer had been pinned to the SHA corresponding to `v44.6.0`, the attackers' update of the `v45` tag would have had no effect on them.

What made the incident particularly instructive was the detection timeline. GitHub's audit log showed the malicious workflow runs. Researchers who analyzed the logs found that the action had been dumping memory to workflow logs for approximately 72 hours before the compromise was publicly discovered. During those 72 hours, secrets from thousands of repositories appeared in workflow log output. Many of those repositories had logging that was not monitored; secrets were exposed in a form that was effectively public (workflow logs for public repositories are visible to anyone) without triggering any alerts.

The incident is a case study in all three of CICD-SEC-6 (insufficient credential hygiene — secrets in logs), CICD-SEC-8 (ungoverned usage of third-party services — no audit of what the action did), and CICD-SEC-10 (insufficient logging — nobody was watching the logs for unexpected content). SHA pinning stops the malicious code from running. Audit log export and secret-scanning on workflow logs is the detection layer that catches it even when prevention fails.

### The limits of what pipeline hardening alone can do

These three incidents share a common thread: the pipeline was the vector, but the root cause was a combination of trust assumptions (we trust this CI step to do what it says), visibility gaps (nobody was watching the CI logs for anomalies), and insufficient compartmentalization (a single pipeline ran with credentials to everything it might ever need, rather than only what each specific step needed).

Pipeline hardening — the controls in this post — addresses the compartmentalization and detection layers. But it cannot eliminate the need for the trust verification layer that SLSA provenance and artifact signing provide. The full defense stack is:

1. **Pipeline hardening** (this post): least-privilege, SHA pinning, ephemeral runners, PPE prevention
2. **Artifact integrity** ([signing and provenance post](/blog/software-development/ci-cd/signing-and-provenance-with-sigstore-and-slsa)): cosign signatures, SLSA provenance attestations, verify-before-deploy
3. **Supply chain security** ([supply chain post](/blog/software-development/ci-cd/software-supply-chain-security-the-new-frontier)): SBOM generation, dependency vulnerability scanning, private registries, dep pinning
4. **Secrets management** ([secrets post](/blog/software-development/ci-cd/secrets-management-in-the-pipeline)): OIDC keyless auth, secret rotation, vault integration

Each layer stops a different class of attack. A hardened pipeline with no artifact signing still lets a sophisticated attacker push a malicious image if they compromise the build environment at a layer below the runner (kernel-level, hypervisor-level, or supply chain compromise of the base image). A signed artifact with SLSA provenance stops that attack even if the pipeline is compromised — because you can verify the provenance record against the expected source tree.

---

## Bringing it all together: a hardened workflow

#### Worked example:

A team operates a Go microservice deployed to AWS EKS. Before hardening, their workflow was a single file that ran on `pull_request_target` (for PR comment posting), used `@v4` tags for all actions, had `permissions: write-all` from a legacy configuration, ran on a persistent self-hosted runner, and stored the AWS deploy credentials as repository-level secrets. In an audit, three separate attack surfaces were identified: PPE via `pull_request_target`, arbitrary code execution via mutable action tags, and lateral movement via a persistent runner that had residual AWS credentials in its Docker credential store.

The hardened replacement:

```yaml
# .github/workflows/ci.yml — hardened version
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
    # NOTE: pull_request (not pull_request_target) — no secret access for fork PRs

# Workflow-level default: no permissions
permissions: {}

jobs:
  lint-and-test:
    runs-on: [self-hosted, arc-runner, ephemeral]
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
        with:
          persist-credentials: false    # do not persist GITHUB_TOKEN in git config

      - uses: actions/setup-go@f111f3307d8850f501ac008e886eec1fd1932a34 # v5.3.0
        with:
          go-version-file: go.mod
          cache: true

      - name: Lint
        run: golangci-lint run ./...

      - name: Test
        run: go test -race -coverprofile=coverage.out ./...

      - name: Upload coverage
        uses: codecov/codecov-action@0565863a31f2c772f9f0395e7f7f2ba8b668e28 # v5.4.0
        with:
          files: coverage.out
          # codecov action uses OIDC — no CODECOV_TOKEN needed

  build-image:
    runs-on: [self-hosted, arc-runner, ephemeral]
    needs: [lint-and-test]
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    permissions:
      contents: read
      packages: write
      id-token: write      # for keyless cosign signing
    outputs:
      image-digest: ${{ steps.build.outputs.digest }}
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
        with:
          persist-credentials: false

      - uses: docker/setup-buildx-action@b5ca514318bd6ebf23d188b2f6b0e7a4a9b7571 # v3.10.0

      - uses: docker/login-action@74a5d142397b4f367a81961eba4e8cd7edddf772 # v3.4.0
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        id: build
        uses: docker/build-push-action@48aba3b46d1b1fec4febb7c5d0c644b249a11355 # v6.7.0
        with:
          context: .
          push: true
          tags: |
            ghcr.io/myorg/myapp:${{ github.sha }}
            ghcr.io/myorg/myapp:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Sign image with cosign (keyless)
        uses: sigstore/cosign-installer@3454372f43399081ed03b604cb2d021dabca52bb # v3.8.1

      - run: |
          cosign sign --yes \
            ghcr.io/myorg/myapp@${{ steps.build.outputs.digest }}

  deploy-staging:
    runs-on: [self-hosted, arc-runner, ephemeral]
    needs: [build-image]
    environment: staging     # requires approval from staging-approvers team
    permissions:
      contents: read
      id-token: write
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
        with:
          persist-credentials: false

      - uses: aws-actions/configure-aws-credentials@e3dd6a429d7300a6a4c196c26e071d42e0343502 # v4.0.2
        with:
          role-to-assume: arn:aws:iam::123456789012:role/GHAStaging
          role-session-name: GHADeploy-${{ github.run_id }}
          aws-region: us-east-1

      - name: Update EKS deployment
        run: |
          aws eks update-kubeconfig --name myapp-staging --region us-east-1
          kubectl set image deployment/myapp \
            myapp=ghcr.io/myorg/myapp@${{ needs.build-image.outputs.image-digest }} \
            -n myapp
          kubectl rollout status deployment/myapp -n myapp --timeout=300s

  deploy-production:
    runs-on: [self-hosted, arc-runner, ephemeral]
    needs: [deploy-staging]
    environment: production   # requires explicit approval from prod-approvers team
    permissions:
      contents: read
      id-token: write
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
        with:
          persist-credentials: false

      - uses: aws-actions/configure-aws-credentials@e3dd6a429d7300a6a4c196c26e071d42e0343502 # v4.0.2
        with:
          role-to-assume: arn:aws:iam::987654321098:role/GHAProduction
          role-session-name: GHADeploy-${{ github.run_id }}
          aws-region: us-east-1

      - name: Update EKS deployment
        run: |
          aws eks update-kubeconfig --name myapp-production --region us-east-1
          kubectl set image deployment/myapp \
            myapp=ghcr.io/myorg/myapp@${{ needs.build-image.outputs.image-digest }} \
            -n myapp
          kubectl rollout status deployment/myapp -n myapp --timeout=600s
```

This workflow eliminates all three attack surfaces. The `pull_request` trigger (not `pull_request_target`) means fork PRs have no secret access. All actions are pinned to SHAs with tag comments for readability. The `permissions: {}` default with per-job overrides means no job has more access than it needs. The ARC ephemeral runners destroy the execution environment after each job. The OIDC federation (`id-token: write` + `aws-actions/configure-aws-credentials`) means no static AWS credentials are stored in GitHub at all — the credentials are generated on demand and valid for the duration of the job only.

Before this hardening, the blast radius of a compromised step in any job was: read all repository secrets, push any image to the registry, deploy to any AWS account that the shared static credential touched. After hardening, a compromised step in `lint-and-test` can read the repository contents and nothing else. A compromised step in `build-image` can push one image to GHCR and sign it. A compromised step in `deploy-staging` can modify the staging EKS cluster. No single compromised step can reach production.

---

## CI security controls comparison

| Control | Threat addressed | Cost | When to skip |
|---------|-----------------|------|-------------|
| Pin actions to SHA | Compromised third-party action | Low — tooling automates it | Never — this is always worth doing |
| `permissions: {}` + explicit grants | Overprivileged CI token | Low | Never — takes 10 minutes to add |
| `pull_request` not `pull_request_target` | PPE via fork | Low | Only if you absolutely need write-back from forked PRs |
| Ephemeral runners | Cross-build credential leak, runner persistence | Medium — requires ARC setup | Small teams under 50 builds/day on GitHub-hosted runners |
| Environment secrets + approval gates | Credential exfiltration to prod | Low | Never for prod; optional for non-prod |
| OIDC keyless auth | Long-lived secret leak | Medium — requires IAM trust policy | Legacy cloud environments that don't support OIDC |
| Network egress filtering | Secret exfiltration via malicious package | High — requires internal mirror | Teams that cannot run an internal registry |
| CI audit log export + alerting | Delayed detection of breaches | Medium — SIEM integration | Very early-stage teams with no SIEM |
| OpenSSF Scorecard | Ongoing posture regression | Low — runs as a GitHub Action | Never — it is free and automated |

---

## Before-after: the security posture change

#### Worked example:

The before state: a team running a monorepo with forty microservices. Each service has a workflow that uses `pull_request_target` for PR comment posting, all actions on mutable `@v3` / `@v4` tags, `permissions: write-all` inherited from the org default, a shared persistent self-hosted runner VM, and an `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` pair stored as repository secrets. The keys have `AdministratorAccess` because it was "easier than figuring out the right policy."

After a week of hardening work (split across three engineers):

- All 40 workflows migrated from `pull_request_target` to `pull_request` for PR checks, with a separate `workflow_run` file for posting PR comments (which runs in the base branch context without checking out PR code).
- `pin-github-action` ran across all 40 workflows; Dependabot configured to open weekly SHAsupdate PRs.
- `permissions: {}` added at org level; each job has explicit minimal scopes.
- ARC deployed to the team's EKS cluster; all jobs migrated to ephemeral ARC runners.
- OIDC trust policies configured for AWS; static credentials deleted from GitHub.
- Audit log export to Datadog; three alert rules created (secret access outside business hours, registry push from non-release workflow, new repository using shared runners).

Before: a single PPE attack in any of the 40 services would have exposed credentials with AWS admin access across the entire account. After: a PPE attack in any service exposes only what that job's explicit permissions allow — typically, the ability to read that repository. The blast radius reduction is from "entire AWS account" to "one repository read access."

The OpenSSF Scorecard score went from 3.2 to 8.7 (out of 10). Not because of any single change, but because the scorecard aggregates exactly these hardening practices.

![Matrix diagram showing CI security controls mapped against risk categories of PPE attack, credential leak, runner persistence, and dep confusion with corresponding prevention and detection controls](/imgs/blogs/securing-the-pipeline-itself-4.png)

---

## Stress-testing the hardened pipeline

Once you have a hardened pipeline, the right question is: what breaks it?

**What if an internal ARC runner image is compromised?** The runner base image (`summerwind/actions-runner:latest` in the example above) is itself a third-party artifact. If it is compromised, every job on every ephemeral runner is compromised regardless of whether the runner is ephemeral. The fix is to maintain your own runner base image built from source and stored in your private registry, pinned to a specific SHA-based tag. Rebuild it weekly; update the ARC RunnerDeployment spec via a Dependabot-equivalent process.

**What if a developer's GitHub account is compromised?** Account compromise allows direct PPE (editing the pipeline YAML on main). Branch protection with required reviews mitigates this — the attacker would need to compromise two accounts, or compromise one account that has bypass permissions. Rotate to hardware security keys (YubiKey) for GitHub authentication and require them for organization members with write access to main.

**What if the OIDC provider is unavailable?** GitHub's OIDC endpoint going down would fail all `id-token: write` jobs. Have a break-glass procedure: a manually-rotated deploy credential stored in a password manager (not in GitHub) that can be used by a human for emergency deploys. Document the procedure; test it quarterly.

**What if your internal package mirror is compromised?** This is the "attacker inside your perimeter" scenario. Mitigate with: signed packages (npm provenance, PyPI trusted publisher), checksums verified on install, SBOM generation before and after each build to detect unexpected new dependencies.

**What if the approval gate for production is bypassed?** GitHub environments with required reviewers are enforced server-side; bypassing them requires GitHub admin access to the repository or organization. A separate CODEOWNERS rule that requires security-team review of any change to workflow files adds an additional gate.

**What if a legitimate developer's machine is compromised and they push a malicious workflow change?** Branch protection with required reviews means a second reviewer must approve the change. With CODEOWNERS configured for `.github/workflows/`, that reviewer must be from the platform team. A malicious change cannot merge unless two accounts are compromised: the developer and a platform-team member. For very high-security repositories, require GPG-signed commits — commits from the developer's compromised machine would lack the signing key that is stored on their hardware token.

**What if the malicious change is subtle enough that reviewers approve it?** This is the insider threat scenario. SHA pinning, least-privilege permissions, and ephemeral runners limit the blast radius even if the malicious change merges. A workflow change that adds a new step but cannot do anything because it has no write permissions and runs in an ephemeral container that is destroyed after the job — the attacker has wasted a very expensive social engineering effort for zero gain.

#### Worked example:

A fintech team runs a payment processing service. They needed to quantify the security improvement from pipeline hardening for a compliance audit. They ran an adversarial simulation before and after:

**Before hardening:**
- Simulated attacker opens a PR with a modified `scripts/test.sh` that reads `$PAYMENT_SERVICE_API_KEY` (a repository-level secret) and posts it to an attacker-controlled server
- Workflow runs on `pull_request_target` (legacy configuration from when they needed to post PR comments)
- The secret is exfiltrated in the first CI run, T+4 minutes after the PR is opened
- Detection: none (no audit log monitoring, no egress filtering)
- Blast radius: \$PAYMENT_SERVICE_API_KEY can be used to make arbitrary payment API calls; potential financial loss unbounded

**After hardening:**
- Same PR is opened, same modified `scripts/test.sh`
- Workflow now runs on `pull_request` — no secrets available
- The script runs but `$PAYMENT_SERVICE_API_KEY` is empty; nothing is exfiltrated
- Egress filtering would have blocked the outbound connection anyway
- Detection: audit log shows a workflow run from a new contributor, which triggers a review-required alert
- The PR is reviewed by the platform team (CODEOWNERS), who notice the suspicious modification to `scripts/test.sh`
- PR is rejected; contributor account is investigated

The hardened configuration changed the outcome from "full credential exfiltration in 4 minutes" to "caught in review before any harm was done." The compliance audit accepted this as a Level 2 CICD-SEC control implementation.

The measured metrics before and after hardening:
- OpenSSF Scorecard: 2.8 → 8.1
- Time-to-detect a simulated PPE attack: never detected → 12 minutes (via CODEOWNERS review alert)
- Number of long-lived cloud credentials stored in GitHub: 4 → 0 (all replaced with OIDC)
- Blast radius of a compromised build job: full AWS account → single EKS namespace

---

## PPE attack lifecycle: from PR to prod

The timeline from an attacker's first action to full impact is short. Understanding it helps you think about where alerting has to fire to actually be useful.

![Timeline diagram showing PPE attack lifecycle from malicious PR opened through secrets exfiltrated to attacker to registry compromised, with detection in audit logs arriving hours later](/imgs/blogs/securing-the-pipeline-itself-5.png)

The window that matters is the gap between "secrets exfiltrated" (T+3 minutes in a fast pipeline) and "detected in audit logs" (T+hours, or never, without alerting). Every minute of that gap is time for the attacker to use the stolen credentials. The Codecov breach ran undetected for two months.

The controls that shrink this gap:

1. **Real-time secret access alerting** — configure your SIEM to page on the first anomalous secret access event, not on a daily digest.
2. **Short-lived credentials** — OIDC tokens are valid for 15–60 minutes. Even if stolen, they expire. A static AWS access key, once stolen, is valid indefinitely.
3. **Registry push monitoring** — most registries can trigger webhooks or Eventbridge events on push. An alert that fires on any push that was not expected (not from the `publish-image` job, not during business hours) gives you a second tripwire.

---

## Permissions before and after: the blast radius math

A useful way to reason about the value of least-privilege is to enumerate the blast radius — what an attacker can do with a compromised job — under the two configurations.

![Diagram comparing overpermissioned write-all CI token giving access to any repo and registry versus scoped permissions block that limits blast radius to exactly what the specific job needs](/imgs/blogs/securing-the-pipeline-itself-8.png)

Under `permissions: write-all` (or an equivalent broad scope):

- A compromised `lint` job can push commits to any branch, create releases, update deployments, push images to the registry, and trigger other workflow runs.
- The theoretical blast radius is: all repositories the token has write access to, plus any downstream deployment that reads from the registry.

Under `permissions: {}` with explicit grants:

- A compromised `lint` job has `contents: read`. It can read the repository. Nothing else.
- A compromised `build-image` job has `contents: read` + `packages: write` + `id-token: write`. It can push one image to GHCR and request one OIDC token. It cannot modify the source code, cannot push to other repositories, cannot access deploy credentials.
- A compromised `deploy-staging` job has `id-token: write` and an OIDC role scoped to the staging environment. It can modify the staging cluster. It cannot reach production.

The blast radius reduction is not theoretical. It is the difference between "the attacker can pivot to production" and "the attacker is contained to the specific resource this job needs."

---

## How to reach for this (and when not to)

Pipeline hardening has a real cost. Here is how to think about prioritization.

**Do immediately, regardless of team size:**

- Add `permissions: {}` at the workflow level and explicit per-job scopes. This takes 30 minutes per workflow and has zero operational overhead.
- Pin all `uses:` lines to SHA. `pin-github-action` does this in one command. Add Dependabot to keep them current.
- Audit your workflows for `pull_request_target` — replace or neutralize any use of it that checks out PR code.

**Do when you have self-hosted runners:**

- Migrate to ephemeral runners via ARC. This requires Kubernetes and a few hours of setup, but the security benefit is substantial. The cost is that some workflows that relied on persistent caches will need to be updated to use ARC's cache integration.

**Do at medium-to-large scale:**

- Export CI audit logs to a SIEM and write alerting rules. The value is proportional to the blast radius — a startup where everyone knows everyone's builds does not need automated anomaly detection. A 500-engineer org where any developer can trigger a workflow run does.
- Network egress filtering on runners. This requires an internal package mirror, which is a real investment. Worth it for teams in regulated industries or with very high-value secrets.

**Do not do:**

- Do not add environment approval gates to every environment, including dev. Approval gates on development environments destroy developer productivity for minimal security benefit. Reserve approval gates for staging and production.
- Do not mandate SHA pinning for internal actions you own and maintain. If you run a private action in your own organization's repositories, you control the tag — the threat of a malicious tag update is essentially zero. Pin third-party actions; tag-based references are fine for internal ones.
- Do not run a full internal package mirror before you have the operational capacity to maintain it. An unmaintained mirror becomes a stale dependency graveyard that creates build failures and blocks security patches. Start with a caching proxy; graduate to a full mirror when you have a team that can own it.

The core principle, woven through every control in this post, matches the [CI/CD mental model](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model): everything as code, explicitly versioned, with minimal privilege at each step. The pipeline is not an afterthought to the system it builds and deploys — it IS part of the attack surface of that system, and it deserves the same security investment. And as covered in the [software supply chain security](/blog/software-development/ci-cd/software-supply-chain-security-the-new-frontier) post, the pipeline's security posture is upstream of every artifact it produces.

---

## Key takeaways

1. **A compromised CI runner has prod access.** Every secret, credential, and deploy key the pipeline uses is available to code running in the pipeline. Treat the runner as a production server.

2. **`pull_request_target` is almost always the wrong trigger.** Unless you have a very specific need for write-back from fork PR contexts, use `pull_request`. If you need both, split them into two workflow files.

3. **Pin all third-party actions to commit SHA.** Tags are mutable. SHAs are not. Dependabot keeps them current automatically.

4. **Start your `permissions:` block at `{}`** — explicitly grant only what each job needs. The blast radius of a compromised job is bounded exactly by its declared permissions.

5. **Self-hosted runners MUST be ephemeral.** A persistent runner accumulates state, credentials, and artifacts that bleed across build boundaries. ARC makes ephemeral Kubernetes runners straightforward.

6. **OIDC keyless federation eliminates the long-lived-credential risk class entirely.** No static AWS/GCP/Azure key in GitHub means no static credential to steal.

7. **Export audit logs and alert on anomalies.** Secret access outside business hours, registry pushes from unexpected workflows, and OIDC claims outside expected ranges are the early-warning signals for an active compromise.

8. **The OWASP CICD-SEC taxonomy is a comprehensive audit checklist.** Work through all ten categories, not just the headline PPE risk. Credential hygiene (CICD-SEC-6) and artifact integrity (CICD-SEC-9) are underinvested in most teams.

9. **Hardening has a real cost — prioritize.** SHA pinning and permissions blocks are free; ephemeral runners and network isolation have setup cost. Do the cheap things first; scale the expensive ones as your blast radius grows.

10. **Build once, promote everywhere, sign everywhere.** The artifact that passes your hardened pipeline is the artifact that reaches production. Every link in that chain — source, build, registry, deploy — needs to be integrity-verified. Pipeline hardening secures the chain; artifact signing proves the chain was not broken. The two work together.

The underlying principle, which runs through the entire [CI/CD series](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model), is that security is not a gate you add at the end of the pipeline — it is a property you engineer into every step of it. A pipeline where least-privilege, SHA pinning, and ephemeral runners are defaults is a pipeline where security is the path of least resistance, not an afterthought that slows delivery down.

---

## Further reading

- **OWASP CICD-SEC Top 10** — [owasp.org/www-project-top-10-ci-cd-security-risks](https://owasp.org/www-project-top-10-ci-cd-security-risks/) — the canonical taxonomy for pipeline security risks
- **GitHub Actions security hardening guide** — [docs.github.com/en/actions/security-guides](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions) — official documentation for permissions, OIDC, and secret handling
- **OpenSSF Scorecard** — [github.com/ossf/scorecard](https://github.com/ossf/scorecard) — automated repository security posture scoring
- **Actions Runner Controller (ARC)** — [github.com/actions/actions-runner-controller](https://github.com/actions/actions-runner-controller) — official Kubernetes-based ephemeral runner operator
- **SLSA Supply Chain Framework** — [slsa.dev](https://slsa.dev) — provenance and build integrity levels; the companion to pipeline hardening
- **Accelerate (Forsgren, Humble, Kim, 2018)** — the empirical foundation for DORA metrics and why security and delivery velocity are complementary, not opposed
- **Within this series:**
  - [From commit to production: the CI/CD mental model](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model)
  - [Software supply chain security: the new frontier](/blog/software-development/ci-cd/software-supply-chain-security-the-new-frontier)
  - [Secrets management in the pipeline](/blog/software-development/ci-cd/secrets-management-in-the-pipeline)
  - [Signing and provenance with Sigstore and SLSA](/blog/software-development/ci-cd/signing-and-provenance-with-sigstore-and-slsa)
