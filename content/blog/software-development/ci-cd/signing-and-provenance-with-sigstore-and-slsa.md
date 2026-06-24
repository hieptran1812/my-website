---
title: "Signing and provenance with Sigstore and SLSA: making every artifact trustworthy"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Learn how cosign keyless signing, Sigstore's Rekor transparency log, and SLSA provenance attestations close the gap between the artifact you tested and the artifact that actually runs in production."
tags:
  [
    "ci-cd",
    "devops",
    "sigstore",
    "slsa",
    "cosign",
    "supply-chain-security",
    "provenance",
    "kubernetes",
    "oci",
    "in-toto",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/signing-and-provenance-with-sigstore-and-slsa-1.png"
---

The build went green. All tests passed. Your CI pipeline pushed an image tagged `v2.3.1` to your registry, and Argo CD dutifully rolled it out to production fifteen minutes later. Two hours after that, your on-call engineer got paged. The pod was crashing. When the team looked at the running image, it did not match the commit they had reviewed. Someone — or something — had pushed a different image with the same tag.

That scenario is not theoretical. It is the artifact-substitution attack that supply-chain security practitioners have spent the last five years systematically closing. The fundamental problem is that a container tag is a mutable pointer. Any principal with push access to the registry can overwrite `v2.3.1` with a completely different digest, and nothing in the default Kubernetes admission path will notice. The tag still says `v2.3.1`. The digest has silently changed. The artifact you tested is not the artifact you deployed.

The fix is not complicated in principle but it requires three interlocking pieces: (1) signing the image by digest so any substitution is cryptographically detectable, (2) recording every signing event on a public transparency log so you can audit the full history, and (3) attaching a provenance attestation that proves which build system, which source commit, and which identity produced the artifact. The Sigstore project provides the first two. The SLSA framework provides the third. Together, with a Kyverno or OPA Gatekeeper admission policy, they close the loop: the only images that can run in your cluster are the ones your CI signed, from the source you specified, on a build environment you trust.

By the end of this post you will understand exactly how `cosign keyless` signing works (no long-lived private key required), why the Rekor transparency log matters, what an SLSA provenance attestation contains and how to generate one automatically with GitHub Actions, how to write a Kyverno policy that refuses unsigned images, and what SLSA L3 actually requires compared to L1. The figure below traces the full signing and verification path from a CI push to a pod being admitted into your cluster.

![Cosign keyless signing flow from CI push through OIDC identity, Fulcio CA certificate issuance, Rekor logging, and registry storage to admission-time verification](/imgs/blogs/signing-and-provenance-with-sigstore-and-slsa-1.png)

---

## 1. Why sign artifacts at all: the problem space

Before diving into the tooling, it is worth making the threat model concrete. The delivery pipeline you operate has at least four places where an artifact can be substituted or tampered with between the moment your CI finishes building it and the moment a container runtime executes it.

**Between build and push.** If CI credentials are shared or leaked, an attacker who can reach the registry push endpoint can push a different image with the same tag before or after your legitimate push. With mutable tags, the later push wins silently.

**In the registry itself.** Registries are not immune to compromise. The Codecov supply-chain attack in 2021 showed that even a legitimate CI tool provider can be compromised and used to inject malicious behavior into every pipeline that trusts it. A registry compromise could overwrite images for thousands of downstream consumers.

**In transit.** Pulling an image over an unencrypted or MITM-able registry endpoint during deployment could substitute the image bytes. In practice TLS protects this, but the manifest digest check is the correct defense rather than trusting the channel alone.

**In the deployment path itself.** If your CD system references tags rather than digests, any re-push to that tag changes what actually runs next time the pod is rescheduled — without any config change in Git.

The signature solves all four. A cosign signature is a cryptographic statement: "entity X, whose identity is attested by an OIDC token from provider Y, signed digest sha256:Z at time T, and that event is recorded in Rekor." You cannot forge that statement without access to the OIDC identity, and you cannot silently alter the Rekor record without forking the entire transparency log. Any tampered image will fail the `cosign verify` check because the tampered digest does not match the signed digest.

The secondary benefit is the audit trail. Without signing, the question "which CI run produced the image currently running in prod?" is often unanswerable. With Rekor, every signing event is publicly queryable by digest. You can reconstruct the full provenance chain for any image in your registry.

### The "build once, promote everywhere" connection

The series spine principle — build one immutable artifact and promote it through environments without rebuilding — gains a cryptographic guarantee from signing. When you sign the image by digest in CI, every subsequent promotion stage can verify that exact same signature. Rebuilding in staging would produce a different digest and would not carry the original signature. The signature becomes the proof that the artifact is the one-and-the-same throughout the delivery chain.

### How DORA metrics connect to artifact integrity

The four DORA metrics — deploy frequency, lead time for changes, change-failure rate, and time-to-restore — are affected by supply-chain security in ways that teams often miss.

**Change-failure rate** is the metric most directly improved by signing enforcement. If an admission policy blocks unsigned images, the only changes that can fail in production are ones that passed signing — meaning they came from your CI system and from a known source commit. Substitution attacks, which can cause spectacular outages, are eliminated entirely. The DORA 2023 State of DevOps report found that elite performers had a change-failure rate of approximately 5% versus 46% for low performers. A meaningful fraction of that gap in lower-performing teams is attributable to deployment failures caused by environment drift and untrusted artifacts, both of which signing addresses.

**MTTR** improves because the root-cause question "was this a supply-chain issue or a code issue?" becomes answerable in minutes instead of hours. The Rekor lookup tells you whether the running image was built by CI or was substituted. That single data point cuts a major branch off the incident investigation tree early.

**Lead time** is minimally affected — the signing step adds under one minute to a typical pipeline. For a team shipping 10 times per day with a four-minute pipeline, the total added CI time is less than ten minutes per day.

**Deploy frequency** is unaffected. Signing is a gate that images pass or fail; it does not serialize deployments or add human-review steps to the critical path.

The practical guidance: treat signing as hygiene infrastructure (like lint or unit tests) that belongs in every pipeline, not as a special security theater gate. The cost is trivial; the benefit is a structural reduction in one entire category of production incident.

---

## 2. Sigstore and cosign: keyless signing without private keys

Sigstore is an open-source project (Linux Foundation, backed by Google, Red Hat, and Chainguard) that provides the infrastructure for keyless code signing. The central insight is that every GitHub Actions workflow already has a cryptographically verifiable identity: the OIDC token that GitHub issues for each workflow run. Instead of asking teams to manage long-lived signing keys — generating them, rotating them, storing them in secrets, worrying about exfiltration — Sigstore leverages the identity you already have.

Here is the complete sequence for a single `cosign sign` invocation in a GitHub Actions workflow:

1. **GitHub issues an OIDC token** for the workflow run. The token contains claims about the repository, the workflow, the commit SHA, and the trigger. It is short-lived (the default expiry is ten minutes) and scoped to the specific run.

2. **Cosign requests a signing certificate from Fulcio**, the Sigstore certificate authority. It presents the OIDC token as proof of identity. Fulcio verifies the token with GitHub's OIDC discovery endpoint, then issues an X.509 certificate valid for ten minutes. The certificate's subject (typically an email SAN) is the workflow's OIDC subject claim, such as `https://github.com/myorg/myrepo/.github/workflows/release.yml@refs/heads/main`.

3. **Cosign uses the ephemeral private key** (generated in-memory for this invocation only) paired with the Fulcio certificate to produce a signature over the image's digest.

4. **The signature and certificate are recorded to Rekor**, the Sigstore transparency log. Rekor is an append-only, Merkle-tree-backed log. Every entry gets a `logIndex` and a signed tree-hash inclusion proof. The entry is permanent and publicly queryable.

5. **The signature is attached to the image in the registry** as a co-located OCI artifact. Cosign stores it using the OCI manifest referrers API (or the older tag convention of `sha256-<digest>.sig`). Any pull of the image digest can be accompanied by a pull of its signature.

The private key is discarded after step 3. There is no keyfile to store, rotate, or protect. The only secret the workflow needs is the OIDC token that GitHub already provides — and that token was going to exist for the workflow anyway.

![The Sigstore trust stack from the signed artifact at the top down through cosign signature, Fulcio CA, OIDC identity, and the Rekor transparency log at the foundation](/imgs/blogs/signing-and-provenance-with-sigstore-and-slsa-2.png)

### Installing cosign and signing an image

Cosign is a single binary. In CI you typically install it via the official GitHub Action:

```yaml
- uses: sigstore/cosign-installer@v3
  with:
    cosign-release: "v2.2.4"
```

To sign an image after pushing it, you need its digest — the tag is not sufficient. Use `docker buildx build --push` with `--metadata-file` to capture the digest, or query the registry:

```bash
# Push the image and capture the digest
docker buildx build \
  --push \
  --tag ghcr.io/myorg/myapp:v2.3.1 \
  --metadata-file build-metadata.json \
  .

IMAGE_DIGEST=$(jq -r '."containerimage.digest"' build-metadata.json)
IMAGE_REF="ghcr.io/myorg/myapp@${IMAGE_DIGEST}"

# Sign by digest (keyless — no key flags needed in GitHub Actions)
cosign sign \
  --yes \
  "${IMAGE_REF}"
```

The `--yes` flag skips the interactive confirmation. In GitHub Actions, cosign detects the `ACTIONS_ID_TOKEN_REQUEST_URL` and `ACTIONS_ID_TOKEN_REQUEST_TOKEN` environment variables automatically and performs the full keyless flow without additional configuration.

To verify later:

```bash
cosign verify \
  --certificate-identity-regexp "https://github.com/myorg/myapp/.github/workflows/.*" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  "ghcr.io/myorg/myapp@sha256:abc123..."
```

The `--certificate-identity-regexp` and `--certificate-oidc-issuer` flags are the policy you are asserting: "this image must have been signed by a workflow from this org, using GitHub Actions as the OIDC provider." If the signature does not match — wrong repo, wrong workflow, expired certificate, tampered image — `cosign verify` exits non-zero and your gate fails.

![Comparison between an unsigned mutable image tag that attackers can overwrite silently versus a cosign-signed image where the cryptographic digest binding detects any substitution](/imgs/blogs/signing-and-provenance-with-sigstore-and-slsa-3.png)

### Why keyless beats long-lived keys

The table below captures the trade-off. The argument for keyless is overwhelming for most teams.

| Concern | Long-lived key | Keyless (cosign) |
|---|---|---|
| Key storage | Secret in CI, HSM, or KMS; must be protected | No key to store — identity comes from OIDC |
| Key rotation | Manual; every rotation requires re-signing or updating verifiers | Automatic — every signing event uses a fresh cert |
| Key exfiltration risk | High — key in secrets can leak via workflow injection | None — no persistent key; OIDC token is short-lived |
| Audit trail | Private or nonexistent | Public Rekor log; every event queryable |
| Revocation | Must distribute CRL or OCSP; complex | Certificate expires in 10 min by design |
| Setup complexity | Key generation ceremony, HSM integration, rotation procedure | Cosign installer + zero flags in GitHub Actions |

---

## 3. The Rekor transparency log: your public audit trail

Rekor is the append-only transparency log at the heart of Sigstore. Put plainly, it plays the same role for artifact signing that Certificate Transparency (CT) logs play for TLS certificates: every signing event is publicly recorded, and the log itself is protected against tampering by a Merkle tree whose root hash is periodically cosigned by Rekor's own key.

When cosign signs your image, the log entry contains:

- The artifact hash (the image digest)
- The signing certificate (including the OIDC subject claims)
- The signature bytes
- A timestamp (provided by Rekor, not the signer — this is a trusted timestamp)
- An inclusion proof (the Merkle path from this entry to the current root)

The inclusion proof is what makes the log tamper-evident. To alter an entry, an attacker would need to recompute every Merkle node above it all the way to the root and convince all verifiers to accept a new root. Since Rekor's signed tree-hash checkpoints are published widely and Rekor itself cannot silently fork the log, the practical attack surface is very small.

### Querying Rekor for an artifact's signing history

The `rekor-cli` tool (or the public API at `https://rekor.sigstore.dev`) lets you look up all signing events for a given artifact hash:

```bash
# Install rekor-cli
go install github.com/sigstore/rekor/cmd/rekor-cli@latest

# Search by artifact hash (the image digest)
rekor-cli search \
  --sha "sha256:abc123def456..."

# Get details of a specific log entry by index
rekor-cli get --log-index 12345678 --format json | jq .
```

The output gives you the full signing certificate (with the GitHub OIDC claims embedded), the timestamp, and the inclusion proof. For incident response, this means you can reconstruct: who signed this image, from which repository, from which branch, at what time — even if the pipeline itself has been deleted.

Cosign also provides a shorthand for verification that checks the Rekor entry as part of the same command:

```bash
# --rekor-url is the default; shown explicitly here
cosign verify \
  --rekor-url https://rekor.sigstore.dev \
  --certificate-identity "https://github.com/myorg/myapp/.github/workflows/release.yml@refs/heads/main" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  "ghcr.io/myorg/myapp@sha256:abc123..."
```

### The "monitor the log" defense posture

For high-security environments, you can run a Rekor monitor that watches the public log for signing events from your identity subjects. If a signing event appears for your repository's OIDC subject that you did not authorize — meaning someone has somehow obtained a workflow token for your repo and used it to sign an artifact — you will know within minutes. The Sigstore community publishes a reference monitor (`sigstore/rekor-monitor`) that you can deploy as a GitHub Actions workflow that runs on a cron schedule.

---

## 4. In-toto and SLSA provenance attestations

A signature answers "who signed this artifact and when?" but it does not answer "what source code was compiled, on which build system, using which dependencies, to produce this artifact?" For that you need a *provenance attestation*.

An attestation, in the in-toto sense, is a signed statement about a software artifact. It has a well-defined schema:

```json
{
  "_type": "https://in-toto.io/Statement/v0.1",
  "predicateType": "https://slsa.dev/provenance/v0.2",
  "subject": [
    {
      "name": "ghcr.io/myorg/myapp",
      "digest": { "sha256": "abc123def456..." }
    }
  ],
  "predicate": {
    "builder": {
      "id": "https://github.com/slsa-framework/slsa-github-generator/.github/workflows/generator_container_slsa3.yml@refs/tags/v1.10.0"
    },
    "buildType": "https://github.com/slsa-framework/slsa-github-generator/container@v1",
    "invocation": {
      "configSource": {
        "uri": "git+https://github.com/myorg/myapp@refs/heads/main",
        "digest": { "sha1": "a1b2c3d4..." },
        "entryPoint": ".github/workflows/release.yml"
      }
    },
    "materials": [
      {
        "uri": "git+https://github.com/myorg/myapp@refs/heads/main",
        "digest": { "sha1": "a1b2c3d4..." }
      }
    ]
  }
}
```

The statement binds the artifact digest (the `subject`) to three facts: which builder produced it (`builder.id`), which source was consumed (`invocation.configSource`), and which external materials (dependencies, base images) were used (`materials`). The whole thing is signed — with cosign, meaning it lands in Rekor — so the binding is cryptographic.

### The in-toto statement envelope

The full in-toto attestation wraps the SLSA predicate in a DSSE (Dead Simple Signing Envelope) before cosign signs it. The DSSE is a minimal envelope format that adds:

- A `payloadType` field that identifies what kind of signed statement this is (e.g., `application/vnd.in-toto+json`)
- A `payload` field with the base64-encoded JSON
- A `signatures` array with the cosign signature

When `cosign verify-attestation` outputs the attestation, it decodes the envelope, verifies the outer signature, and then decodes and prints the inner payload (the SLSA predicate JSON). Consumers who want to verify attestations without cosign can interact with the DSSE directly using the `dsse-verify` tool, but cosign is the practical choice for most CI/CD consumers.

### In-toto link attestations beyond provenance

The SLSA provenance predicate is the most commonly used in-toto attestation type, but the in-toto specification defines a general framework for arbitrary attestation types. Other predicate types that are increasingly common in supply-chain pipelines:

- **`https://spdx.dev/Document`**: an SBOM (Software Bill of Materials) attached to an image, stating which packages it contains. Cosign can attach and verify an SPDX SBOM as an attestation.
- **`https://cyclonedx.org/bom`**: the CycloneDX SBOM format, also attachable as an in-toto attestation.
- **`https://cosign.sigstore.dev/attestation/vuln/v1`**: a vulnerability scan result (e.g., from Trivy), stating which CVEs were found and at what severity at scan time.
- **`https://cosign.sigstore.dev/attestation/v1`**: a generic predicate for custom statements, useful for attestations like "this image passed manual security review on date X by engineer Y."

Kyverno can verify any of these predicate types using the `attestations` block in a `verifyImages` rule, giving you a composable policy model: you can require that an image be signed, carry a valid SLSA provenance, and carry a clean Trivy scan result — all in a single ClusterPolicy.

![SLSA provenance predicate tree showing the subject artifact digest, build metadata with builder id and invocation, and materials source inputs](/imgs/blogs/signing-and-provenance-with-sigstore-and-slsa-7.png)

### The SLSA framework levels

SLSA (Supply chain Levels for Software Artifacts, pronounced "salsa") is a graduated framework from Google and the Open Source Security Foundation that defines what it means for provenance to be trustworthy. There are four levels:

| Level | What is required |
|---|---|
| L1 | Provenance exists in any form (could be unsigned, could be from a self-hosted system) |
| L2 | Provenance is generated by a *hosted* CI system (GitHub Actions, Google Cloud Build) and is authenticated — i.e., the CI system, not the developer, signs it |
| L3 | Same as L2, plus the build is on an *isolated, ephemeral* runner: the runner is created fresh for each job, has no persistent state from prior builds, and does not have network access to write to the source repository |
| L4 | Hermetic, reproducible, and two-person-reviewed build: all dependencies are declared and fetched at known hashes; any rebuild from the same inputs produces the same output bit-for-bit |

Most teams building on GitHub Actions managed runners with `slsa-github-generator` can achieve SLSA L3 with very little extra work. L4 requires significant build-system investment (hermetic Bazel, pinned base images, reproducible compilers) and is currently aspirational for most organizations.

![Comparison of SLSA L1 with any build system and unsigned provenance versus SLSA L3 with hosted isolated ephemeral runner and signed cosign attestation](/imgs/blogs/signing-and-provenance-with-sigstore-and-slsa-6.png)

---

## 5. Generating SLSA provenance in GitHub Actions

The `slsa-github-generator` project provides a reusable workflow that generates a SLSA L3 provenance attestation for container images. It works through a clever two-workflow architecture: your workflow builds and pushes the image, then calls the reusable provenance-generator workflow (which runs on a fresh isolated runner) and passes it the digest. Because the provenance generator is a *separate* workflow with *its own* OIDC token, the provenance cannot be tampered by the build step.

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags: ["v*"]

permissions:
  contents: read
  packages: write
  id-token: write   # Required for OIDC token (keyless signing)
  attestations: write  # Required for GitHub attestation API

jobs:
  build:
    outputs:
      image: ${{ steps.meta.outputs.tags }}
      digest: ${{ steps.build.outputs.digest }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}

      - name: Build and push
        id: build
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}

  sign:
    needs: build
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      packages: write
    steps:
      - uses: sigstore/cosign-installer@v3

      - name: Sign image by digest
        run: |
          cosign sign --yes \
            "ghcr.io/${{ github.repository }}@${{ needs.build.outputs.digest }}"
        env:
          COSIGN_EXPERIMENTAL: "true"

  provenance:
    needs: build
    permissions:
      actions: read
      id-token: write
      packages: write
    uses: slsa-framework/slsa-github-generator/.github/workflows/generator_container_slsa3.yml@v1.10.0
    with:
      image: ghcr.io/${{ github.repository }}
      digest: ${{ needs.build.outputs.digest }}
      registry-username: ${{ github.actor }}
    secrets:
      registry-password: ${{ secrets.GITHUB_TOKEN }}
```

The `provenance` job calls the SLSA L3 reusable workflow. That workflow runs on a GitHub-managed runner, generates the provenance JSON, signs it with its own OIDC token (a different token from your build — it is a token scoped to the `slsa-github-generator` repository), attaches the attestation to the image, and logs it to Rekor. The result is an attestation that verifiers can trust because it was generated by a workflow that your build step could not modify.

### Verifying the attestation

```bash
# Verify SLSA provenance attestation attached to an image
cosign verify-attestation \
  --type slsaprovenance \
  --certificate-identity-regexp "https://github.com/slsa-framework/slsa-github-generator" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  "ghcr.io/myorg/myapp@sha256:abc123..." | jq '.payload | @base64d | fromjson'
```

The output is the full SLSA provenance predicate decoded from the base64-encoded envelope. You can inspect the `builder.id`, the `invocation.configSource.digest`, and the `materials` to confirm the provenance matches your expected source.

GitHub also provides a native attestation API (in beta as of early 2026) via the `attest-build-provenance` action:

```yaml
- name: Generate GitHub attestation
  uses: actions/attest-build-provenance@v1
  with:
    subject-name: ghcr.io/${{ github.repository }}
    subject-digest: ${{ steps.build.outputs.digest }}
    push-to-registry: true
```

This stores the attestation in GitHub's own attestation store, queryable via `gh attestation verify`.

### Combining the sign and attest jobs: a full workflow

Here is a complete, production-ready workflow that builds, signs, and generates a SLSA L3 attestation in three separate jobs:

```yaml
# .github/workflows/release.yml  (full version)
name: Build, Sign, and Attest

on:
  push:
    tags: ["v[0-9]+.[0-9]+.[0-9]+"]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      digest: ${{ steps.build.outputs.digest }}
      tags: ${{ steps.meta.outputs.tags }}
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4

      - name: Set up QEMU
        uses: docker/setup-qemu-action@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
        with:
          buildkitd-flags: --allow-insecure-entitlement security.insecure=false

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract Docker metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha,prefix=sha-

      - name: Build and push multi-arch image
        id: build
        uses: docker/build-push-action@v5
        with:
          context: .
          platforms: linux/amd64,linux/arm64
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          provenance: false  # Disable Docker's built-in provenance; use slsa-generator instead

  sign:
    needs: build
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      packages: write
    steps:
      - uses: sigstore/cosign-installer@v3
        with:
          cosign-release: "v2.2.4"

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Sign image by digest (keyless)
        run: |
          cosign sign --yes \
            "${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}@${{ needs.build.outputs.digest }}"

      - name: Verify signature immediately
        run: |
          cosign verify \
            --certificate-identity-regexp \
              "https://github.com/${{ github.repository }}/.github/workflows/.*" \
            --certificate-oidc-issuer \
              "https://token.actions.githubusercontent.com" \
            "${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}@${{ needs.build.outputs.digest }}"

  provenance:
    needs: build
    permissions:
      actions: read
      id-token: write
      packages: write
    uses: slsa-framework/slsa-github-generator/.github/workflows/generator_container_slsa3.yml@v1.10.0
    with:
      image: ${{ needs.build.outputs.tags }}
      digest: ${{ needs.build.outputs.digest }}
      registry-username: ${{ github.actor }}
      compile-generator: true
    secrets:
      registry-password: ${{ secrets.GITHUB_TOKEN }}
```

The `compile-generator: true` flag tells `slsa-github-generator` to recompile the generator binary from source in the isolated job rather than downloading a pre-built binary — this is the stronger security posture (no trust in a pre-built binary) at the cost of a slightly longer job runtime.

The `sign` job includes an immediate post-signing verification step. This is a sanity check: if the sign job succeeds but the verification step fails, there is a cosign configuration error you want to catch immediately rather than discovering at deploy time.

### What the attestation bundle looks like in the registry

After both `sign` and `provenance` jobs complete, the OCI registry contains three artifacts for your image:

1. The image index (the multi-arch manifest) at the version tag and the digest
2. A `sig` referrer: the cosign signature, stored as a single-layer OCI image attached to the index digest
3. An `att` referrer: the SLSA provenance attestation envelope, also stored as a single-layer OCI image

You can list them with:

```bash
cosign tree "ghcr.io/myorg/myapp@sha256:abc123..."
```

The output shows the signature and attestation as referrer artifacts attached to the main image. This is the OCI referrers API (formerly the "co-located artifact" convention); registries that support `application/vnd.oci.image.manifest.v1+json` with the `referrers` endpoint serve these automatically to clients that request them.

---

## 6. Verifying at admission: Kyverno and the Sigstore policy-controller

Signing images and generating provenance is only half the problem. The other half is *enforcing* the signatures at deploy time. If your CD system or kubectl users can still deploy unsigned images, the signing step in CI is a security theater exercise — it records history but does not prevent the attack.

Kubernetes admission webhooks are the correct enforcement point. When a pod spec is submitted to the API server, a validating admission webhook can inspect it, pull the referenced image's signature, verify it, and reject the pod if verification fails — before the pod is ever scheduled onto a node.

The Sigstore project ships the **Sigstore policy-controller**, a Kubernetes admission webhook that integrates directly with cosign. You define `ClusterImagePolicy` resources that specify which images require which signatures:

```yaml
apiVersion: policy.sigstore.dev/v1beta1
kind: ClusterImagePolicy
metadata:
  name: require-cosign-signature
spec:
  images:
    - glob: "ghcr.io/myorg/**"
  authorities:
    - keyless:
        url: https://fulcio.sigstore.dev
        identities:
          - issuer: https://token.actions.githubusercontent.com
            subject: "https://github.com/myorg/myapp/.github/workflows/release.yml@refs/heads/main"
      ctlog:
        url: https://rekor.sigstore.dev
```

This policy says: any image matching `ghcr.io/myorg/**` must have a valid keyless cosign signature issued by GitHub Actions OIDC, from the specific workflow path specified, and that signing event must be present in the Rekor transparency log. A pod that references an unsigned image, or an image signed by a different workflow, or a valid image whose Rekor entry is missing will be rejected with a descriptive admission error.

![Full sign-verify lifecycle from CI build through cosign signing, Rekor logging, registry push, policy-controller check at deploy time, and final admission or rejection](/imgs/blogs/signing-and-provenance-with-sigstore-and-slsa-5.png)

### Kyverno: enforcing signatures without the policy-controller

If you are already running Kyverno for general policy enforcement, it also supports cosign verification natively via the `verifyImages` rule type:

```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: verify-image-signature
spec:
  validationFailureAction: Enforce
  background: false
  rules:
    - name: check-image-signature
      match:
        any:
          - resources:
              kinds:
                - Pod
      verifyImages:
        - imageReferences:
            - "ghcr.io/myorg/*"
          attestors:
            - count: 1
              entries:
                - keyless:
                    subject: "https://github.com/myorg/myapp/.github/workflows/release.yml@refs/heads/main"
                    issuer: "https://token.actions.githubusercontent.com"
                    rekor:
                      url: https://rekor.sigstore.dev
```

The `validationFailureAction: Enforce` field is the critical setting — it means failures are hard denials, not audit-only events. With `Audit` (the softer mode), unsigned images will still run but violations will be logged. Start in Audit mode during rollout, then switch to Enforce once you are confident every legitimate image in every namespace is properly signed.

Kyverno also supports verifying SLSA attestations in the same policy:

```yaml
        attestations:
          - predicateType: "https://slsa.dev/provenance/v0.2"
            conditions:
              - all:
                  - key: "{{ predicate.builder.id }}"
                    operator: Equals
                    value: "https://github.com/slsa-framework/slsa-github-generator/.github/workflows/generator_container_slsa3.yml@refs/tags/v1.10.0"
```

This adds a second gate: not only must the image be signed, but it must also carry a valid SLSA L3 attestation from the official `slsa-github-generator` workflow.

![Comparison of deploying without admission control where any unsigned image reaches production versus Kyverno ClusterPolicy enforcing cosign signature check and blocking unsigned images](/imgs/blogs/signing-and-provenance-with-sigstore-and-slsa-8.png)

---

## 7. Signing approach comparison

Different teams approach signing differently based on their existing tooling, risk posture, and operational maturity. The matrix below captures the main options.

![Matrix comparing cosign keyless, cosign long-lived key, and Notary v2 across key management, audit trail, OIDC-native support, and Kubernetes admission capabilities](/imgs/blogs/signing-and-provenance-with-sigstore-and-slsa-4.png)

The matrix makes the decision clear: for any team running on GitHub Actions or another OIDC-capable CI platform, keyless cosign is the right default. The only realistic argument for long-lived keys is integration with a legacy signing infrastructure (like an existing hardware security module used for Windows code signing) where the operational cost of introducing OIDC is higher than the cost of managing the key.

Notary v2 (now part of the OCI referrers specification as the Notation project) is the CNCF-standardized alternative. It is better suited to air-gapped environments or teams who want a self-hosted trust store rather than a dependence on the public Sigstore infrastructure. The main trade-off is that Notation's admission integration requires the Ratify project plus a custom webhook, which adds operational complexity.

### OPA Gatekeeper as an alternative admission enforcer

If your cluster uses OPA Gatekeeper instead of Kyverno, you can still enforce cosign signatures using the Sigstore policy-controller (which is a standalone admission webhook that does not depend on Kyverno). Install the policy-controller separately:

```bash
helm repo add sigstore https://sigstore.github.io/helm-charts
helm repo update
helm install policy-controller sigstore/policy-controller \
  --namespace cosign-system \
  --create-namespace \
  --set webhook.failurePolicy=Fail
```

The `failurePolicy=Fail` is critical — it means if the webhook itself is unavailable, pod scheduling fails closed rather than open. This is the secure default and the only acceptable setting for production clusters. If you set `failurePolicy=Ignore`, a simple webhook outage bypasses all your signing enforcement.

The policy-controller uses `ClusterImagePolicy` resources (shown earlier in this post) and does not require OPA/Rego knowledge. It sits alongside Gatekeeper rather than replacing it, since Gatekeeper handles general resource validation and the policy-controller handles image signature validation specifically.

---

## 8. SLSA levels in practice: from L1 to L3

Most teams who adopt provenance attestations start at L1 and are surprised how little additional work L3 requires on GitHub Actions managed runners.

**SLSA L1** requires only that some provenance exists. You can satisfy L1 by writing your own provenance document — even a simple JSON file that lists the commit SHA, the image digest, and the timestamp — and uploading it as a build artifact. This is better than nothing for audit purposes but provides no security guarantees: anyone with write access to the repository could forge a provenance document.

**SLSA L2** requires the provenance to be generated by a hosted CI system and to be authenticated. The key distinction from L1 is that the provenance must be generated by the *CI service itself*, not by a script running in the build. On GitHub Actions, the `attest-build-provenance` action or the `slsa-github-generator` reusable workflow satisfy L2 when they generate the provenance in a separate job with its own OIDC token. The provenance is authenticated because it is signed with the CI service's OIDC-backed certificate.

**SLSA L3** adds the isolation requirement: the build must happen on an *ephemeral* runner that has no persistent state from prior builds and that cannot write back to the source repository. GitHub Actions managed runners (`ubuntu-latest`, `ubuntu-22.04`) are ephemeral by design — each job gets a fresh VM. The `slsa-github-generator` workflow satisfies L3 because:

- It runs in a separate, isolated job from the build (the generator workflow cannot be tampered by the user's build steps)
- The runner is GitHub-hosted and ephemeral
- The OIDC token is scoped to the generator workflow, not the user's workflow
- The provenance is signed and recorded in Rekor

**SLSA L4** requires hermetic and reproducible builds. Hermetic means all inputs — base images, build tools, dependencies — are pinned at known digests and fetched from those exact locations with no network access to unpinned external endpoints. Reproducible means any rebuild from the same inputs produces the same bit-for-bit output. Achieving L4 typically requires Bazel with a pinned toolchain, distroless or scratch-based images with no package manager, and a fully deterministic build graph. It is the right goal for critical infrastructure components but is a significant engineering investment.

### Practical SLSA level assessment for your team

The level that is right for a given service depends on its blast radius and its role in the trust chain.

For an internal analytics dashboard accessed only by employees, SLSA L1 with cosign signing is likely sufficient. The attestation exists, it is signed, and the key infrastructure management is absent. The residual risk — someone forging provenance — is low because the damage from a compromised analytics dashboard is limited.

For a payment processing service, SLSA L3 is the correct starting point. The isolated, ephemeral build environment means the provenance cannot be tampered by the build step. The signed attestation means verifiers can trust it was generated by the CI system, not by a developer laptop. The \$400k-per-hour cost of a payment service outage justifies the pipeline investment.

For a security-critical component like a secrets manager sidecar, a service mesh control plane, or a container runtime itself, SLSA L4 is the right aspiration. These components are in the trust chain for everything else; compromising them compromises the entire cluster. Hermetic Bazel builds with pinned dependencies, combined with two-person review of the build definitions themselves, give you the strongest available provenance guarantees.

A practical progression looks like this for most organizations:

```bash
# Month 1: add cosign sign to all new releases (no enforcement yet)
# Month 2: audit mode Kyverno - log which services are unsigned
# Month 3: enforce signing in non-prod; migrate legacy images
# Month 4: enforce signing in prod; achieve SLSA L2 for all new services
# Month 6: adopt slsa-github-generator for critical services (L3)
# Year 2: evaluate hermetic builds for highest-value services (L4)
```

The key is that each step is independently deployable and independently valuable. You do not need SLSA L3 before you start enforcing SLSA L2, and you do not need L2 before you start signing.

---

## 9. The transparency log as infrastructure

The Rekor log deserves a deeper look because it changes the operational model for artifact provenance in a fundamental way.

Traditional signing infrastructure is *private* — the registry knows who pushed what, but only if you ask it and only if it has retained the access logs. Rekor is *public*. Every signing event from every user of the public Sigstore instance (`https://rekor.sigstore.dev`) is in the same log. That sounds alarming at first — "you are publishing our signing events publicly?" — but it is actually a security benefit.

Because the log is public, anyone can monitor it. Supply-chain security researchers and automated monitors watch the log for anomalies: signing events from unexpected identities, certificates issued for repositories that have been inactive, spikes in signing activity that might indicate a compromised pipeline. The transparency of the log makes many attack scenarios visible that would be invisible in a private system.

The append-only property means that once an entry is logged, it cannot be silently deleted or modified. If an attacker somehow signs a malicious image using a stolen OIDC token, that signing event is *in the log forever* — which helps forensic reconstruction but also makes the attack harder to cover up.

For regulated industries that require an audit trail of "who signed what and when," the Rekor log is a ready-made, cryptographically verifiable answer. Instead of maintaining your own private audit log (and worrying about its integrity), you reference Rekor entries by log index.

### What a Rekor entry actually contains

When you look at a raw Rekor entry (for example from `rekor-cli get --log-index <n> --format json`), the JSON structure has several important fields worth understanding:

```json
{
  "logIndex": 12345678,
  "logID": "c0d23d6ad406973f9559f3ba2d1ca01f84147d8ffc5b8445c224f98b9591801d",
  "integratedTime": 1719043200,
  "body": "<base64-encoded entry body>",
  "verification": {
    "inclusionProof": {
      "checkpoint": "...",
      "hashes": ["..."],
      "logIndex": 12345678,
      "rootHash": "...",
      "treeSize": 100000000
    },
    "signedEntryTimestamp": "<base64-encoded SET>"
  }
}
```

The `integratedTime` is a Unix timestamp set by the Rekor server, not by the signer. This is a trusted timestamp — an independent record of when the signing event was received by the log. For compliance purposes, this provides a reliable record of when an artifact was signed, regardless of what the CI run's own timestamp says.

The `inclusionProof` contains the Merkle path from this entry to the log's current root. A verifier can use this proof to confirm the entry is genuinely part of the log and has not been inserted retroactively. The `signedEntryTimestamp` (SET) is a signed statement from Rekor confirming the entry was logged at the stated time — it is signed with Rekor's own private key, whose public key is published and widely distributed.

The `logID` identifies which Rekor instance produced the entry. The public Sigstore instance has a well-known log ID; private instances have different IDs. Verifiers can be configured to accept only entries from specific log IDs, which is important if you want to reject signatures made against a private Rekor instance that your admission policy does not trust.

### Running your own Rekor instance

If your threat model requires that signing events be private — for example, because signing event metadata reveals information about your release cadence or software inventory that you consider proprietary — you can deploy a private Rekor instance in your own infrastructure. The setup is a Kubernetes deployment with a Trillian backing store (a Google open-source Merkle tree service). You configure cosign to use your private instance with `--rekor-url https://rekor.internal.example.com`. The operational trade-off is that you lose the public monitoring benefit and you take on the responsibility of securing and maintaining the Rekor infrastructure.

---

#### Worked example: a complete signing pipeline

Consider a Python microservice at `myorg/payment-service`. The team has historically just pushed `latest` to their registry and deployed by tag. After a security audit flagged supply-chain risks, they implement the following in a single afternoon.

**Before the change:**
- Image reference in Kubernetes: `ghcr.io/myorg/payment-service:latest` (mutable tag)
- No signature, no provenance
- No admission policy
- An attacker with registry push access could substitute the image silently

**After the change:**
- Image is built with `docker buildx build --push`, digest captured in CI
- `cosign sign --yes ghcr.io/myorg/payment-service@sha256:abc123...` runs in a dedicated sign job
- `slsa-github-generator` produces and attaches a SLSA L3 provenance attestation
- Kubernetes manifest references the image by digest: `ghcr.io/myorg/payment-service@sha256:abc123...`
- Kyverno ClusterPolicy in Enforce mode rejects any pod not carrying a valid cosign signature from the CI workflow

The measured changes:

| Metric | Before | After |
|---|---|---|
| CI pipeline duration | 4 min 20 sec | 4 min 55 sec |
| Additional CI cost | — | ~\$0.005/build (35 extra seconds on ubuntu-latest) |
| Signing events logged to Rekor | 0 | 1 per release |
| Unsigned images admitted to cluster | Unlimited | 0 |
| Substitution attack detectable | No | Yes, at admission time |
| Incident response: time to determine signing identity | N/A | Under 2 minutes (rekor-cli search) |

The pipeline overhead is less than one minute per release. For a team that ships ten times a day, that is under nine minutes of additional CI time per day — a cost that is trivially justified by the security improvement.

The Kubernetes manifest change from tag to digest deserves its own attention. Referencing by digest is a prerequisite for the signing enforcement to work correctly. If your Kyverno policy checks signatures but your deployment references `latest`, a re-push to `latest` will start scheduling pods with the new (unverified) digest on the next reschedule. Pin to digest:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: payment-service
spec:
  template:
    spec:
      containers:
        - name: payment-service
          # Pin to digest, not tag
          image: ghcr.io/myorg/payment-service@sha256:abc123def456789...
          resources:
            requests:
              cpu: "100m"
              memory: "128Mi"
            limits:
              memory: "256Mi"
```

---

## 10. War story: how supply-chain attacks exploit unsigned artifacts

The SolarWinds attack of 2020 is the canonical supply-chain compromise, but the Codecov breach of April 2021 is more directly instructive for CI/CD pipelines. Codecov, a code coverage reporting service, had their Docker image tampered during their build process. The attacker modified the CI script to exfiltrate environment variables — including repository credentials — from every pipeline that ran the compromised Codecov uploader. Thousands of organizations were affected over a period of two months before the attack was discovered.

What would have stopped it? A signed provenance attestation for the `codecov/codecov-action` GitHub Action would have made the tampered version detectable. If Codecov had been signing their published images and action releases with a key tied to their CI identity, any consumer could have run `cosign verify` against the version they were pulling and seen the verification fail — because the tampered image would not match the signed digest. Or, more powerfully, if consumers were running an admission-style verification on their CI workflows (using GitHub's `github.com/ossf/scorecard-action` or a manual digest pin), the tampered version would have been blocked before it ran.

The direct lesson: the attack worked because the artifact — the Codecov uploader script — was consumed by thousands of pipelines by name and version, without any cryptographic verification that the artifact at that name+version was the one Codecov intended to publish.

A secondary supply-chain attack vector is dependency confusion, as demonstrated by Alex Birsan in 2021. Package managers like npm, pip, and RubyGems check public registries before private ones when a package name exists in both. By publishing a public package with a higher version number than a private internal package, Birsan was able to get his package installed inside over 35 major technology companies. SLSA materials provenance is a partial mitigation here: if your build attestation records every package dependency at its digest, you can verify after the fact that the resolved dependency came from your private registry, not the public one.

The 3CX supply-chain attack of 2023 took it one step further: the attacker compromised a dependency of 3CX's build toolchain (Trading Technologies' `X_TRADER` software), injecting malicious code that persisted through the 3CX build and into their shipping product. This is a case where SLSA materials provenance would have helped forensics — you could trace exactly which version of which dependency introduced the malicious code — but would not by itself have prevented the attack, since the signed provenance would have correctly reflected the compromised dependency. This is why hermetic builds (L4) with dependency hash-pinning are the right answer for the most sensitive software.

The honest assessment: signing and provenance are strong defenses against *post-build* tampering — someone altering an image in the registry, substituting a different package at the same name, or injecting code between build and deploy. They are weaker against *pre-build* compromise — a poisoned build tool or dependency that enters the build inputs. The right defense-in-depth strategy combines signing + provenance (for post-build integrity) with SBOM generation and vulnerability scanning (for supply-chain analysis of build inputs) and dependency pinning (for preventing unexpected resolution of malicious packages).

### The supply-chain defense layering model

Put plainly, different defenses stop different classes of attack. Understanding which layer each control occupies helps you prioritize where to invest:

| Defense | Stops | Does not stop |
|---|---|---|
| Digest pinning in manifests | Tag-based substitution at deploy time | Image tampered before initial signing |
| cosign signature verification | Artifacts not signed by your CI identity | Compromised build environment that signed malicious code |
| SLSA L3 provenance | Forged provenance; non-isolated build | Compromised source repository |
| SBOM + vuln scanning | Known CVEs in dependencies | Zero-days; logic bombs |
| Dep hash pinning | Dependency confusion; unpinned upgrades | Compromised upstream at pinned hash |
| Rekor log monitoring | Unauthorized signing events | Signing before monitor detects it |

No single layer covers the whole space. Each layer has a narrow, well-defined threat it addresses. Stack them deliberately, with the layers that are cheapest and most broadly effective deployed first (digest pinning and cosign signing) and the more expensive layers (hermetic builds, dep hash pinning, active log monitoring) added as your risk profile demands.

This maps directly to the software supply chain security paper from the Cloud Native Computing Foundation (CNCF TAG Security), which recommends exactly this layered approach. The paper, referenced in the further reading section, provides a comprehensive framework for evaluating which defenses to deploy at each phase of the software delivery lifecycle.

### The XZ Utils attack: why SLSA materials matter

In March 2024, a near-miss supply-chain attack targeting Linux distributions was uncovered by a Microsoft engineer, Andres Freund. A malicious contributor, operating under the pseudonym "Jia Tan," had over the course of approximately two years gained maintainer access to the XZ Utils open-source project. In the final phase of the attack, they inserted a sophisticated backdoor into the XZ Utils release tarball — but not into the source repository in a way that was obviously visible. The backdoor was present in the release artifact but not reproducible from the tagged source.

This attack highlights exactly what SLSA materials provenance guards against, and where it falls short. If every Linux distribution had been verifying that XZ Utils releases were built reproducibly from the tagged source using a hermetic build process (SLSA L4), the discrepancy between the source and the release tarball would have been detectable. SLSA provenance with materials that include the source commit hash, combined with a reproducible build verification, would have shown: "the binary in this release does not reproduce from the source at this commit."

In practice, no major Linux distribution was verifying this at distribution ingestion time. The lesson is that SLSA provenance is not just useful for your own first-party services — it is a framework that the entire open-source ecosystem needs to adopt for third-party dependencies as well. Tools like `deps.dev` (Google's Open Source Insights) and OpenSSF Scorecard now surface SLSA level information for open-source packages, letting you see at a glance whether a dependency's releases come with verifiable provenance before you add it to your dependency graph.

---

#### Worked example: querying Rekor to verify a signing event during incident response

Your on-call engineer receives an alert at 2:17 AM: an anomalous network connection from the `payment-service` pod to an unexpected external IP. The first question: is the running image the one CI signed, or was it substituted?

The pod's image reference is available immediately from `kubectl describe pod`:

```bash
kubectl describe pod payment-service-abc123-xyz \
  | grep "Image ID:"
# Image ID: ghcr.io/myorg/payment-service@sha256:def789abc...
```

Now query Rekor to see the signing history for that digest:

```bash
rekor-cli search --sha "sha256:def789abc..."
# Found matching entries (UUID):
# 362f8ecaaabc...

rekor-cli get --uuid 362f8ecaaabc... --format json | jq '{
  logIndex: .logIndex,
  integratedTime: (.integratedTime | todate),
  cert_subject: .verification.signedEntryTimestamp,
  body: (.body | @base64d | fromjson | .spec.signature.content)
}'
```

If the query returns a valid entry whose OIDC subject matches your CI workflow and whose timestamp aligns with the expected release time, the image is legitimate — the anomalous network connection is a runtime behavior issue, not a supply-chain compromise. The mean time to this determination: under three minutes.

If the query returns no entry, or returns an entry with an unexpected OIDC subject (e.g., a workflow from a different repository, or a signing event that happened outside CI hours), you have confirmation of a supply-chain event. Roll back immediately, rotate credentials, and begin forensics on the Rekor entry's certificate to identify the compromised identity.

The arithmetic on MTTR improvement here is real. Before signing was in place, this investigation might have taken hours: pulling the image locally, comparing layers, reviewing registry access logs if they existed, checking who had push access. With Rekor, the signing history is one CLI command away and the answer is cryptographically authoritative.

### Integrating Rekor verification into a CI pre-deploy gate

Beyond the Kubernetes admission webhook, you can add a `cosign verify` step as an explicit gate in your CD pipeline — especially useful in GitOps workflows where the CD system applies manifests from Git rather than running a traditional pipeline:

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  verify-and-deploy:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: read
    steps:
      - uses: actions/checkout@v4

      - uses: sigstore/cosign-installer@v3

      - name: Extract image digest from manifest
        id: image
        run: |
          DIGEST=$(grep 'image:' k8s/deployment.yaml | awk '{print $2}')
          echo "ref=${DIGEST}" >> "$GITHUB_OUTPUT"

      - name: Verify image signature before deploy
        run: |
          cosign verify \
            --certificate-identity-regexp \
              "https://github.com/${{ github.repository }}/.github/workflows/.*" \
            --certificate-oidc-issuer \
              "https://token.actions.githubusercontent.com" \
            "${{ steps.image.outputs.ref }}"

      - name: Apply manifests
        run: kubectl apply -f k8s/
        env:
          KUBECONFIG_DATA: ${{ secrets.KUBECONFIG }}
```

This gives you a belt-and-suspenders defense: the signature is verified both in the CI gate (before `kubectl apply` runs) and again at admission time by Kyverno. If either check fails, the image does not reach production. The duplication is intentional — a misconfigured Kyverno policy might silently be in Audit mode; the explicit CI gate catches that gap.

---

## 11. Rolling out signing incrementally: a practical playbook

The single most common mistake teams make when adopting cosign is trying to enforce it cluster-wide on day one. That approach creates an incident: unsigned images that pre-date the signing rollout start getting rejected, and the oncall is suddenly debugging Kyverno admission failures for a dozen services that nobody has signed yet.

The right rollout is incremental:

**Week 1: sign new releases, log-only admission policy.** Add the cosign sign step to your release pipeline for all services. Switch Kyverno to Audit mode (not Enforce). This means every new release gets signed and logged to Rekor, but unsigned images can still deploy. The Kyverno audit log shows you which namespaces and deployments are referencing unsigned images so you can plan the migration.

**Week 2–3: verify all currently-running images.** For each service, verify that its current running image has a valid cosign signature (`cosign verify ...`). For images that predate the signing rollout, you have a choice: rebuild and re-sign them (preferred), or add an explicit exemption in the Kyverno policy (acceptable temporarily).

**Week 4: switch to Enforce for new namespaces.** Start with non-production namespaces (staging, dev). Any deployment that tries to reference an unsigned image gets blocked. This surfaces any pipeline gaps before they become a production issue.

**Week 5+: enforce in production namespace by namespace.** Flip Enforce for production services one namespace at a time. Watch for Kyverno admission failures in the admission webhook logs.

**Ongoing: SLSA provenance.** Provenance attestation can be added to the `provenance` job after signing is fully rolled out. The rollout is simpler because Kyverno attestation verification is an additive check — you can start by just verifying the signature, and add attestation verification once the `slsa-github-generator` integration is in place.

### Handling multi-architecture images

Modern services often build images for multiple CPU architectures (linux/amd64 and linux/arm64 for Apple Silicon-based developer laptops or Graviton-based AWS nodes). Multi-arch images are represented as OCI image indexes (manifests of manifests). Cosign handles this correctly: signing the index digest covers all platform-specific manifests underneath it. When you run:

```bash
cosign sign --yes \
  "ghcr.io/myorg/myapp@sha256:<index-digest>"
```

The signature is attached to the index manifest. A verifier pulling the arm64 manifest can still verify the signature against the index digest. The OCI referrers API links the signature artifact to the index, so `cosign verify` works regardless of which platform's manifest the runtime actually pulled.

The one subtlety: if you sign individual platform manifests separately instead of the index, a Kyverno policy that enforces signing on the index digest will not see those per-platform signatures. Sign the index, not the platforms.

### Rotating identities: what happens when you rename a workflow

One operational surprise teams hit after deploying signing enforcement is that a seemingly innocent rename of a workflow file (say, from `release.yml` to `publish.yml`) breaks their Kyverno policy. The policy's `subject` field is an exact match or regexp against the OIDC subject claim, which includes the workflow file path:

```
https://github.com/myorg/myapp/.github/workflows/release.yml@refs/heads/main
```

After renaming to `publish.yml`, new images will be signed with the new subject. If the policy has not been updated, new images will be rejected.

The fix is to use a subject regexp that does not pin to the specific file name:

```yaml
authorities:
  - keyless:
      identities:
        - issuer: https://token.actions.githubusercontent.com
          subjectRegExp: "https://github.com/myorg/myapp/.github/workflows/.*@refs/heads/main"
```

The regexp `.*` matches any workflow file name. This is slightly less strict than pinning to an exact file, but it is usually the right trade-off — you trust all workflows in your repository, not just one specific file name. If you want to restrict to a single workflow, pin it and document that renaming the file requires a policy update.

---

## 12. Signing on GitLab CI and other platforms

While the examples above use GitHub Actions (because its OIDC token flow is the most mature and best-documented for Sigstore), cosign keyless signing works on any CI platform that issues OIDC tokens for its job runners. GitLab CI added OIDC token support in GitLab 15.7 (2022).

On GitLab CI, the workflow is nearly identical. The key difference is that the OIDC token is injected via the `CI_JOB_JWT_V2` variable (or via the newer `id_token` keyword in GitLab CI YAML), and the OIDC issuer is your GitLab instance URL rather than `token.actions.githubusercontent.com`.

```yaml
# .gitlab-ci.yml
variables:
  COSIGN_YES: "true"

stages:
  - build
  - sign

build-image:
  stage: build
  image: docker:24
  services:
    - docker:24-dind
  script:
    - docker buildx build --push
        --tag "${CI_REGISTRY_IMAGE}:${CI_COMMIT_SHA}"
        --metadata-file build-meta.json
        .
    - jq -r '."containerimage.digest"' build-meta.json > .image-digest
  artifacts:
    paths:
      - .image-digest

sign-image:
  stage: sign
  image: cgr.dev/chainguard/cosign:latest
  id_tokens:
    SIGSTORE_ID_TOKEN:
      aud: sigstore
  needs: [build-image]
  script:
    - IMAGE_DIGEST=$(cat .image-digest)
    - cosign sign
        --identity-token="${SIGSTORE_ID_TOKEN}"
        "${CI_REGISTRY_IMAGE}@${IMAGE_DIGEST}"
```

The `id_tokens` block tells GitLab to issue an OIDC token with audience `sigstore` and inject it as `SIGSTORE_ID_TOKEN`. Cosign reads this token and performs the same Fulcio + Rekor flow as in GitHub Actions. The `COSIGN_YES=true` environment variable skips the interactive confirmation (equivalent to `--yes`).

The verification command on GitLab changes the issuer and subject constraints:

```bash
cosign verify \
  --certificate-identity-regexp "https://gitlab.com/myorg/myapp//-/pipelines/.*" \
  --certificate-oidc-issuer "https://gitlab.com" \
  "registry.gitlab.com/myorg/myapp@sha256:abc123..."
```

For self-hosted GitLab, replace the issuer with your instance URL. For GitLab.com, the issuer is `https://gitlab.com`.

### Signing in multi-registry environments

Many organizations maintain images in multiple registries — a primary registry (GHCR or ECR) and a mirror or DR registry (GCR or another region-local ECR). When you promote images between registries, the digest typically remains the same (if you copy the manifest without modifying it), but the signature is attached to the image in the original registry's namespace.

To handle this, cosign supports copying signatures alongside images with `cosign copy`:

```bash
# Copy image and its signature + attestations to a second registry
cosign copy \
  "ghcr.io/myorg/myapp@sha256:abc123..." \
  "us-east-1.ecr.amazonaws.com/myapp@sha256:abc123..."
```

The `cosign copy` command copies the image manifest, all layers, and all OCI referrer artifacts (signatures and attestations) to the destination. The signature's cryptographic binding is to the digest, not to any registry URL, so it remains valid after the copy. A Kyverno policy in a cluster that pulls from the mirror registry will still pass verification because the signature's certificate's subject still identifies your CI workflow.

---

## 13. Stress-testing the setup: failure modes and edge cases

### What if the Rekor log is temporarily unavailable?

Cosign has a timeout for Rekor interactions. If Rekor is unreachable during signing, `cosign sign` will fail (by default). You can configure cosign to skip the Rekor log with `--no-tlog-upload`, but then you lose the audit trail. For high-availability requirements, run a private Rekor instance.

During *verification*, if Rekor is unreachable, `cosign verify` will also fail — which means your Kyverno policy will start blocking legitimate pods. This is the fail-closed behavior (the default), which is the secure choice. If you need fail-open for availability, you can configure the policy-controller with a `--error-on-image-validation-failure=false` flag, but this obviously weakens the security guarantee.

### What if you need to deploy a hotfix urgently and the signing job fails?

Have a documented break-glass procedure: a Kyverno namespace annotation or a ClusterImagePolicy exemption that can be applied by two designated principals with a mandatory post-mortem. The procedure should be: apply the exemption, deploy the hotfix, immediately add signing to the hotfix pipeline, retire the exemption, file the incident report.

### What if two CI runs race to sign the same digest?

This is safe. Multiple signing events for the same digest from different runs are all valid entries in Rekor. The verification only checks that *at least one* valid signature exists — it does not fail if multiple exist. You can optionally configure your Kyverno policy to require a signature from a specific subject if you want to prevent signing from unexpected sources.

### What if a developer signs an image locally to test the pipeline?

Local signing requires either a long-lived key or a manual OIDC flow. Neither will match the policy's `subject` constraint (which requires the signing identity to be the CI workflow). The image will fail admission. This is the correct behavior — you do not want developer laptops to be in the trust chain for production images.

### What if the signing certificate's OIDC subject changes?

This happens when you rename a workflow file, rename a repository, or change the trigger branch. Your `ClusterImagePolicy` subject constraint will stop matching. New builds will fail admission. The fix is to update the policy's `subject` field to the new value. If you use a regexp pattern (`subject-regexp`) rather than an exact match, you can make the policy more flexible at the cost of some specificity.

---

## 13. How to reach for this (and when not to)

Signing and provenance are the right investment for:

- **Any service that handles sensitive data or money**, where the cost of a supply-chain compromise is high enough to justify the operational overhead
- **Services with external consumers** who need to verify that the artifact they consume came from you
- **Teams already using GitOps with digest pinning**, where the toolchain to support digest-based references already exists
- **Organizations with compliance requirements** (SOC 2, FedRAMP, NIST SSDF) that mandate software bill-of-materials or provenance records
- **Shared platform teams** who want to enforce a minimum security posture across all services running in their cluster

Hold off or keep it lightweight when:

- **You are a 3-person startup** shipping a single service: the overhead of setting up Kyverno and `slsa-github-generator` is not justified until you have a meaningful number of services or a specific compliance requirement driving it
- **Your images are not pushed to a shared registry**: if you build and deploy in a single CI step with no intermediary registry, the substitution attack surface is much smaller
- **You have not yet pinned your Kubernetes manifests to digests**: signing by digest while deploying by tag gives you auditability but not enforcement; fix the manifest pinning first
- **Your CI platform does not support OIDC tokens**: some older CI systems or self-hosted runners may not expose OIDC endpoints. In that case, evaluate whether `cosign sign --key` with a KMS-managed key is a better fit before adopting keyless

The prioritization is: pin to digest → sign with cosign keyless → enforce with Kyverno in Audit → enforce in Enforce mode → add SLSA provenance. Do not jump to SLSA L3 before the admission enforcement is working.

### The cost model

Let's be explicit about what this costs, because "it's cheap" is not the same as "it's free." Here are realistic estimates for a team of 20 engineers shipping 30 times per day:

**CI overhead per release:**
- cosign install: ~15 seconds (cached after first run on a warm runner)
- cosign sign (including Fulcio and Rekor round-trips): ~10–20 seconds
- slsa-github-generator provenance job: ~45–90 seconds (separate job, runs in parallel with other post-build jobs)

At 30 releases per day and a 60-second total overhead per release, that is 30 minutes of runner time per day. On GitHub Actions ubuntu-latest (billed at approximately \$0.008 per minute), that is \$0.24 per day — under \$90 per year. That is the cost of a single hour of a staff engineer's time, purchased once and providing a continuous security control.

**Kyverno policy-controller overhead:**
The policy-controller adds a small latency to pod admission: typically 50–200ms for an image whose signature is cached, 500ms–2s for a fresh verification including Rekor lookup. For a typical deployment that schedules 5 pods, this adds under 10 seconds to the total rollout time. Not measurable in end-to-end lead time.

**Registry storage:**
Each cosign signature is a small OCI artifact (a few hundred bytes). Each SLSA attestation is a slightly larger JSON document, typically 5–15 KB. For a team shipping 30 images per day with 90-day retention, total signature storage is on the order of a few hundred megabytes — negligible compared to the image layers themselves.

The honest comparison: the incremental cost of signing is roughly \$90/year in CI compute. The cost of a single supply-chain incident — even a minor one requiring three engineers for two days to investigate — is approximately \$12,000–\$50,000 in lost engineering time. The ROI is not in question for any team with more than a few services in production.

---

## 14. Measuring the before and after

The security benefit of signing and provenance is real but qualitative — "we prevented a class of attack" is hard to put on a chart. Here is how to make it measurable.

**Metric 1: Signing coverage rate.** Count the fraction of currently-running images in your cluster that have a valid cosign signature, measured by running `cosign verify` against every image in `kubectl get pods -A -o json | jq '.items[].spec.containers[].image'`. At signing rollout start, this is 0%. The goal is 100% within 60 days.

```bash
# Count images with valid signatures vs total
TOTAL=0; SIGNED=0
for img in $(kubectl get pods -A -o json \
  | jq -r '.items[].spec.containers[].image' | sort -u); do
  TOTAL=$((TOTAL+1))
  if cosign verify \
       --certificate-identity-regexp ".*" \
       --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
       "$img" 2>/dev/null 1>/dev/null; then
    SIGNED=$((SIGNED+1))
  fi
done
echo "Signed: $SIGNED / $TOTAL"
```

**Metric 2: Kyverno admission violation rate.** With Kyverno in Audit mode, the `ClusterPolicyReport` or the Kyverno events log shows how many pod creation events per day are violating the signature policy. Track this to zero before switching to Enforce.

**Metric 3: Mean time to artifact verification.** Time how long it takes your on-call team to answer "is the running image the one CI signed?" With Rekor, the target answer time is under three minutes. This is a measurable MTTR sub-metric.

**Metric 4: CI pipeline duration impact.** Record the P95 build duration before and after adding the cosign sign job. The target is under 90 seconds added per release. If it is higher, check whether the cosign installer step is being cached.

The before/after story for a typical 20-engineer platform team that has implemented the full stack — digest pinning, signing, Kyverno Enforce, SLSA L3 provenance — over a 90-day rollout:

| Metric | Before | After 90 days |
|---|---|---|
| Signing coverage of running images | 0% | 100% |
| Unsigned image admission violations | N/A (no policy) | 0 (Enforce blocks all) |
| MTTR for "is this the right image?" | 2–4 hours | Under 3 minutes |
| Supply-chain incident investigations | Inconclusive | Fully traceable via Rekor |
| CI pipeline duration increase | 0 sec | +55 sec per release |
| Additional CI cost | \$0 | ~\$90/year |
| SLSA L3 provenance coverage | 0% | 85% of prod services |

The numbers reflect what teams that have adopted this stack consistently report — the CI overhead is negligible; the MTTR improvement during supply-chain investigations is the most compelling operational benefit.

---

## 15. Key takeaways

1. A container tag is a mutable pointer. Signing by digest replaces a trust assumption with a cryptographic proof. Every deployment must reference an image by its `sha256:` digest, not a tag, for signing enforcement to close the substitution attack surface completely.
2. Cosign keyless signing uses OIDC identity rather than long-lived keys, eliminating the key-management burden while adding a public audit trail via Rekor. The only credential needed is the OIDC token the CI platform already issues — there is nothing new to generate, store, or rotate.
3. Every `cosign sign` event lands in the Rekor append-only transparency log, making it tamper-evident and publicly queryable for incident response. A query that takes under three minutes with `rekor-cli search --sha <digest>` replaces hours of access-log archaeology.
4. SLSA provenance attestations bind the artifact digest to the source commit, the builder identity, and the build environment — answering "how was this built?" not just "who signed it?" Provenance is the difference between "this image is authentic" and "this image is authentic and I can prove it came from commit X on branch main."
5. SLSA L3 on GitHub Actions managed runners requires adopting `slsa-github-generator` as a reusable workflow — approximately one afternoon of pipeline work. The isolation property (separate job, separate OIDC token, ephemeral runner) is what distinguishes L3 from L2.
6. Kyverno `verifyImages` or the Sigstore policy-controller enforce signatures at Kubernetes admission time; start in Audit mode and graduate to Enforce namespace by namespace. Never flip cluster-wide Enforce on day one — you will page oncall with unsigned-image admission failures for every service that predates the signing rollout.
7. The Rekor log transforms incident response: the question "is the running image the one CI produced?" becomes a two-minute CLI query instead of a multi-hour forensics exercise. This alone justifies the signing investment for any team with an on-call rotation.
8. Admission enforcement is fail-closed by default: if Rekor is unreachable, pods are blocked. This is the secure choice; document a break-glass procedure (a temporary namespace annotation exemption, applied by two designated engineers, with a mandatory post-mortem) for emergencies.
9. Sign and provenance are strong against post-build tampering; combine with SBOM scanning and dependency pinning for defense against pre-build compromise. The XZ Utils attack (2024) and the Codecov attack (2021) together illustrate both attack surfaces — signing closes the post-build gap, materials provenance aids forensics for pre-build attacks.
10. The cost model is one-sided: the CI overhead is under 90 seconds and roughly \$90/year for a team shipping 30 times per day. The alternative cost — a single supply-chain incident requiring a weekend of investigation and a full audit — is orders of magnitude larger. Treat signing as hygiene infrastructure, not a special security project.

---

## Further reading

- [Sigstore documentation and cosign reference](https://docs.sigstore.dev) — the authoritative source for all cosign commands, keyless flow internals, and policy-controller deployment
- [SLSA framework specification](https://slsa.dev) — the full level definitions, threat model, and predicate schema
- [slsa-github-generator repository](https://github.com/slsa-framework/slsa-github-generator) — the reusable workflows for SLSA L3 container provenance
- [Rekor transparency log API](https://github.com/sigstore/rekor) — the Rekor server, CLI, and API reference
- [Kyverno verify-images policy reference](https://kyverno.io/docs/writing-policies/verify-images/) — full documentation for cosign and attestation verification in Kyverno policies
- [CNCF Security Technical Advisory Group: Software Supply Chain Best Practices](https://github.com/cncf/tag-security/blob/main/supply-chain-security/supply-chain-security-paper/CNCF_SSCP_v1.pdf) — the comprehensive supply-chain security guide
- [From commit to production: the CI/CD mental model](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) — the series introduction and full pipeline architecture
- [Software supply-chain security: the new frontier](/blog/software-development/ci-cd/software-supply-chain-security-the-new-frontier) — the threat taxonomy and broader supply-chain attack surface
- [SBOM and dependency management](/blog/software-development/ci-cd/sbom-and-dependency-management) — generating and consuming software bill-of-materials to complement signing
- [Securing the pipeline itself](/blog/software-development/ci-cd/securing-the-pipeline-itself) — CI credential management, secret scanning, and pipeline hardening
