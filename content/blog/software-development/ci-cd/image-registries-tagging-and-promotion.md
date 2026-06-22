---
title: "Image Registries, Tagging, and Promotion: Deploy by Digest, Tag for Humans"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Treat the registry as the warehouse between build and deploy, understand why tags are mutable pointers and digests are immutable identities, and learn to deploy by digest while you tag and promote for humans so build-once-promote-everywhere actually holds."
tags:
  [
    "ci-cd",
    "devops",
    "container-registry",
    "docker",
    "image-tagging",
    "digest",
    "promotion",
    "retention",
    "supply-chain",
    "oci",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/image-registries-tagging-and-promotion-1.png"
---

A few years ago I helped a team chase a ghost. They deployed a service called `checkout` to production on a Wednesday afternoon, the way they always did: merge to main, wait for the green build, click promote. The staging environment had been running the same image for two days and the full integration suite was green against it. Within four minutes of the production deploy, the error rate on the checkout endpoint climbed from a flat 0.1% to a spiky 6%. We rolled back, the errors stopped, and then we spent the rest of the day arguing about something that should have been impossible. The code was the same commit. Staging had tested it. How does an image that staging blessed fail the moment it hits prod?

The answer, when we finally found it, was embarrassingly simple and it had nothing to do with the code. The deployment manifest said `image: registry.internal/checkout:latest`. Staging had pulled `:latest` on Monday and pinned a copy locally. Between Monday and Wednesday, two other PRs had merged, and CI — doing exactly what it was told — had pushed new images and re-pointed `:latest` at each one. So the bytes that staging tested and the bytes that prod pulled were *not the same bytes*. They shared a tag. They did not share a digest. The tag is a sticky note. Somebody had moved the sticky note. We had carefully validated one image and then shipped a different one, under the banner of "it's the same tag, so it's the same thing." It was not the same thing, and the thing that runs in production is what matters.

That incident is the entire reason this post exists. The registry — the warehouse that sits between your build stage and your deploy stage — is where "build once, promote everywhere" either holds or quietly breaks, and the deciding factor is almost always how you **tag** and **promote** the images you store there. Get it wrong and you ship untested bytes with a clean conscience. Get it right and the artifact that production runs is, to the byte, the artifact you validated in staging. This post is about getting it right.

![A diagram of a container registry as a content-addressed store where a manifest lists layers and each layer is written once into the blob store keyed by its sha256 digest](/imgs/blogs/image-registries-tagging-and-promotion-1.png)

Here is what you will be able to do by the end. You will be able to explain, to a skeptical colleague, exactly why `latest` in a production manifest is a loaded gun, and what to use instead. You will be able to design a tagging scheme that gives humans the friendly names they want (`v1.2.3`, `stable`, `prod`) without ever letting those friendly names compromise safety. You will be able to promote the same image from a dev registry to a prod registry with a single command that copies bytes rather than rebuilding them — `crane copy` or `skopeo copy`, the promotion-as-a-registry-operation pattern. You will be able to write a retention policy that bounds your storage bill without ever garbage-collecting an image a running pod still needs. And you will understand where the registry sits on the supply-chain attack surface and how to harden it. All of this maps onto the spine of this whole series — **commit → build → test → package → deploy → operate** — and onto the two governing principles, **build once, promote everywhere** and **everything as code**. The registry is the "package" link in that chain, and tagging-and-promotion is how the package survives the journey from build to operate without changing.

If you have not yet read the series intro, [the CI/CD model from commit to production](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) sets up the spine and the DORA metrics this post leans on. And the immediately preceding sibling, [build once, promote everywhere](/blog/software-development/ci-cd/build-once-promote-everywhere-artifacts-and-versioning), establishes *why* one immutable artifact must flow through every environment. This post is the operational sequel: now that you have decided to build once, *how* do you store, name, move, and retire that one artifact in a registry so the principle survives contact with reality?

## 1. What a registry actually is: a content-addressed store

Let me define the word before I lean on it, because the whole post depends on understanding what a registry *is* mechanically, not just what it does at a high level.

A **container registry** is a server that stores and serves OCI images. OCI stands for Open Container Initiative — the standards body that defined the image format and the distribution protocol that Docker popularized and everyone else adopted. The registries you will actually use are Docker Hub (the public default), GitHub Container Registry (GHCR, `ghcr.io`), Amazon Elastic Container Registry (ECR), Google Artifact Registry, and self-hosted options like Harbor and Quay. They differ in access control, retention features, replication, and price, but they all speak the same wire protocol and they all store images the same way underneath.

The way they store images is the single most important fact in this post, so let me be precise. An OCI image is not one file. It is a small JSON document called a **manifest**, plus a set of **layers** (tarballs of filesystem changes), plus a **config** blob (the metadata: entrypoint, env, architecture). The manifest does not *contain* the layers; it *references* them, and it references each one by its **digest** — a SHA-256 hash of the layer's exact bytes, written as `sha256:` followed by 64 hex characters. The registry's blob store is keyed by those digests. Push a layer whose content hashes to `sha256:a1b2…`, and the registry files it under that key. Push a second image that happens to include the very same base layer, and the registry sees the digest is already present and stores nothing new — it just records another reference. This is **content addressing**: the address of a blob *is* a fingerprint of its content. Two blobs with identical bytes have identical addresses and are therefore the same blob, stored once.

That is why a registry is sometimes called a content-addressed store, and figure 1 above shows the shape of it: a client pushes a manifest, the manifest lists layers, and each layer lands in the blob store exactly once even when many images share it. This deduplication is not a nice-to-have; it is the reason a registry holding 10,000 images built on the same `python:3.12-slim` base does not store that 80 MB base 10,000 times. It stores it once and hands out 10,000 references.

The push/pull protocol is worth a sentence because it shapes everything downstream. To **push** an image, the client uploads each layer the registry does not already have (checking by digest first, so unchanged layers are skipped), then uploads the config blob, then uploads the manifest and associates it with a **tag** or a digest. To **pull**, a client asks the registry to resolve a reference — either a tag like `myapp:v1.2.3` or a digest like `myapp@sha256:9f2b…` — into a manifest, then downloads any layers it does not already have locally. The crucial detail: a pull *by tag* asks the registry "what manifest does this tag point to *right now*?" The registry answers with whatever the tag currently points at. A pull *by digest* asks "give me the manifest whose content hashes to exactly this," and the registry can only answer with one thing, because the digest *is* the content.

Hold onto that distinction, because the next section is built entirely on it. The protocol gives us two ways to name an image: a tag, which the registry looks up and can answer differently over time, and a digest, which the registry can only ever resolve to one set of bytes. Those two naming schemes have completely different safety properties, and conflating them is how teams ship untested code.

A registry, then, is three things stacked together: a blob store keyed by digest (the bytes), a set of manifests (the recipes that list which blobs make an image), and a tag namespace (the mutable, human-friendly pointers into the manifests). When I say "the registry is the warehouse between build and deploy," this is the warehouse: build writes immutable bytes and recipes into it, and deploy reads them back out. The only question that matters is whether deploy reads back *exactly* what build wrote, and that comes down to whether you address by tag or by digest.

One more protocol detail pays off later, when we talk about retention. The registry distinguishes between *deleting a tag*, *deleting a manifest*, and *garbage-collecting a blob*. Untagging an image removes a pointer but leaves the manifest and its layers in place. Deleting the manifest removes the recipe but does not immediately reclaim the layer bytes — those blobs are now unreferenced ("orphaned") but they still occupy disk until a separate garbage-collection pass walks the blob store, finds blobs that no surviving manifest references, and reclaims them. This two-phase model (un-reference, then sweep) is why "I deleted the image but my storage bill did not drop" is a frequent confusion: deleting the manifest only sets up the blob for collection; the collection is a distinct operation, often run on a schedule or requiring the registry to be read-only during the sweep. Keep this in mind for section 6.

The registries you will choose among differ less in *how* they store images — they all implement the same content-addressed model above — and more in the surrounding features: access control, retention tooling, replication, scanning, and price. A quick orientation:

| Registry | Hosting | Notable features | Typical fit |
|---|---|---|---|
| Docker Hub | SaaS | The public default; rate-limited anonymous pulls | Public images, small projects |
| GHCR (`ghcr.io`) | SaaS | Tied to GitHub repos and Actions OIDC; per-repo visibility | Teams already on GitHub |
| ECR | AWS-managed | IAM-native, scan-on-push, immutable-tag flag, lifecycle policies | AWS workloads |
| Artifact Registry | GCP-managed | Declarative cleanup policies, per-repo, multi-format | GCP workloads |
| Harbor | Self-hosted | Tag-immutability rules, replication, built-in scanning and signing policy | On-prem, regulated, multi-tenant |
| Quay | SaaS or self-hosted | Robot accounts, fine-grained retention, scanning | Red Hat / OpenShift shops |

The point of the table is not to pick a winner — pick the one that matches where your workloads run — but to notice that the features that *vary* are exactly the ones this post cares about: immutable-tag enforcement, lifecycle/retention policies, and access control. Those are not afterthoughts; they are the levers that make tagging-and-promotion safe, so weigh them when you choose.

## 2. Tags vs digests: the most important distinction in the registry

Here is the distinction stated as plainly as I can state it. A **tag** is a mutable, human-friendly pointer. A **digest** is an immutable, cryptographic identity. They look superficially similar — both go after a colon-or-at-sign in an image reference — but they behave like opposites.

A tag is `v1.2.3` or `latest` or `staging`. It is a label you stick on a manifest. The registry lets you move that label to a different manifest at any time, simply by pushing a new image with the same tag. There is no law of the universe that says `myapp:v1.2.3` today is the same image as `myapp:v1.2.3` tomorrow. By default, the registry will happily let you overwrite the tag. The tag is a name; names can be reassigned.

A digest is `sha256:9f2bda1c…`. It is not a label you assign; it is computed from the manifest's bytes. You cannot "move" a digest to point at different content, because the digest *is* a fingerprint of the content. If the content changes by a single byte, the digest changes completely. `myapp@sha256:9f2b…` today and `myapp@sha256:9f2b…` in a year are guaranteed identical bytes, or the year-from-now registry would refuse to serve them under that digest (the hash would not match). The digest is an identity; identities cannot be reassigned without becoming a different identity.

Put plainly, the tag is a sticky note on a box in the warehouse and the digest is the box's actual contents. You can peel the sticky note off this box and stick it on that box. You cannot do the same with the contents.

It helps to lay the two side by side across the properties that actually matter when you decide which to use where. Figure 5 does exactly that for the four reference forms you will encounter: `:latest`, a semver tag, a git-SHA tag, and a raw sha256 digest. Read it as a decision aid — the property a tag wins on (human-friendliness) is precisely the property that makes it unsafe to deploy by, and the property a digest wins on (immutability) is precisely the property that makes it unreadable to humans. Neither is "better"; they are tools for different jobs, and the entire tagging discipline in this post is about using each for the job it is good at.

![A matrix comparing the latest tag, a semver tag, a git SHA tag, and a sha256 digest across whether each is mutable, human-friendly, and safe to deploy in production](/imgs/blogs/image-registries-tagging-and-promotion-5.png)

The matrix surfaces a point worth stating directly: there is no single reference form that is both maximally human-friendly *and* maximally safe. The `:latest` tag is the most friendly and the least safe — it always moves and is never safe to deploy. The raw sha256 digest is the safest and the least friendly — it never moves and you would never want to type it from memory or read it in a changelog. The two immutable middle options, a semver tag and a git-SHA tag, are the bridge: friendly enough for humans to read, and (with immutability enforced) safe enough to reason about, while still being one level above the digest that the deploy actually pins. A good scheme uses all four — `:latest` and friends for "what's current," semver for "which release," the git SHA for "which commit," and the digest for "which exact bytes ship." The mistake is not using a tag; the mistake is *deploying* by anything other than the bottom row.

![A before and after comparison of deploying by mutable tag versus deploying by pinned digest, showing that the latest tag moved between the staging test and the prod deploy while the digest stayed fixed](/imgs/blogs/image-registries-tagging-and-promotion-2.png)

### Why a mutable tag in production is dangerous

Now I can explain the checkout incident in precise terms, and figure 2 shows it as a before-and-after. The danger is a classic **time-of-check to time-of-use (TOCTOU)** bug, applied to deployments. You *check* the artifact at one moment — you run staging's integration suite against the image that `:latest` points to on Monday. You *use* the artifact at a later moment — you deploy whatever `:latest` points to on Wednesday. Between the check and the use, the pointer moved. The thing you checked is not the thing you used.

This is not a rare edge case. It is the *default* behavior of `latest` and of any environment tag or moving convenience tag. The whole point of those tags is that they move; that is what makes them convenient. And the convenience is exactly what makes them unsafe to deploy. Every CI push that re-points `:latest`, every nightly build, every hotfix that reuses the `prod` tag — each one is a chance for the pointer to move between your validation and your release. The window can be days, as in the checkout story, or it can be seconds: two PRs merge near-simultaneously, both pipelines push `:latest`, and the order in which the registry processes those two pushes — a race you do not control — decides which image your deploy step pulls.

There is a sharper version of this danger that has nothing to do with your own pipeline. If your base image is `FROM ubuntu:22.04` and you rebuild, the `ubuntu:22.04` tag upstream may now point at a different patch than it did last month. Your "unchanged" Dockerfile produces a different image because a mutable tag underneath you moved. (That is one reason the [build stage post](/blog/software-development/ci-cd/the-build-stage-reproducible-fast-and-cacheable) argues for pinning base images by digest, too — `FROM ubuntu@sha256:…`. The same principle applies all the way down.)

So the rule, which I will repeat because it is the thesis of the post: **tag for humans, deploy by digest.** Tags exist so people can find and talk about images — "we shipped v1.2.3," "roll staging back to last week's build." Digests exist so machines can deploy exactly the bytes that were validated. Your deployment manifest should never reference a tag. It should reference a digest, and that digest should be the one your tests ran against.

### What "pin the digest" looks like

Concretely, a Kubernetes Deployment that deploys by tag looks like this — and this is the dangerous version:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: checkout
spec:
  template:
    spec:
      containers:
        - name: checkout
          # DANGER: this is a mutable pointer. The bytes can change
          # between the test that validated :v1.2.3 and this rollout.
          image: registry.internal/checkout:v1.2.3
```

And the safe version pins the digest. Note that you can keep the human-readable tag in the reference for legibility — the registry ignores the tag part when a digest is present and resolves strictly by digest:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: checkout
spec:
  template:
    spec:
      containers:
        - name: checkout
          # SAFE: the digest is immutable. This is the exact manifest
          # staging tested. The :v1.2.3 part is decoration for humans;
          # resolution is by sha256 only.
          image: registry.internal/checkout:v1.2.3@sha256:9f2bda1c4e7a8f0b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293
```

How does the digest get *into* the manifest? You do not hand-copy 64 hex characters. The push step in CI emits the digest, and your deploy step reads it. With `docker buildx`, `--metadata-file` writes a JSON file containing the digest. With GitHub Actions, the `docker/build-push-action` outputs `outputs.digest`. You capture that and template it into the manifest. Here is the GitHub Actions shape:

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    outputs:
      digest: ${{ steps.push.outputs.digest }}
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - id: push
        uses: docker/build-push-action@v6
        with:
          push: true
          # tag for humans here; we will deploy by the output digest below
          tags: ghcr.io/acme/checkout:${{ github.sha }}
      - name: Record digest for downstream deploy
        run: echo "Pushed digest is ${{ steps.push.outputs.digest }}"
```

That `outputs.digest` is now the single source of truth for the rest of the pipeline. The test job runs against `ghcr.io/acme/checkout@${{ needs.build.outputs.digest }}`, the deploy job pins the same value, and there is no window in which a tag can move out from under you. Check and use refer to the identical bytes because they both refer to a digest, and a digest cannot move.

## 3. A tagging strategy that serves humans without endangering deploys

If digests are what we deploy, why tag at all? Because nobody can hold `sha256:9f2bda1c4e7a8f…` in their head, talk about it in a standup, or scan a list of them and know which is the hotfix. Tags are the human interface to the registry. The goal of a tagging strategy is to give humans every convenient name they want while making sure none of those names can ever silently change the bytes a release pulls.

![A layered stack diagram of a tagging strategy showing the immutable digest at the base, an immutable git SHA tag and semver above it, and moving convenience tags and environment tags floating on top](/imgs/blogs/image-registries-tagging-and-promotion-3.png)

Figure 3 shows the layering I recommend, from immutable foundation to mutable convenience. There are three tiers, and the discipline is knowing which tier each tag belongs to.

**Tier 1 — the immutable build tag.** Every image gets exactly one tag that is *never reused for any other image, ever*: the git commit SHA, or a monotonic build id like `2026.06.22.1487`. This is the tag-level analog of the digest. It is immutable not because the registry enforces it (though it should — see below) but because the SHA names a specific commit and you push that image exactly once. The build tag answers "which source produced these bytes?" and gives you traceability from a running container back to a line of code. This is the tag your pipeline should treat as canonical internally.

**Tier 2 — semantic-version tags for releases.** When a build becomes a release, it earns a semver tag: `v1.2.3`. SemVer (`MAJOR.MINOR.PATCH`) communicates intent to humans — a `MINOR` bump adds features compatibly, a `MAJOR` bump breaks something. The full `v1.2.3` should be immutable: once you publish `v1.2.3`, that exact string must forever mean those exact bytes, the same as the build tag. This is the tag people put in release notes and changelogs.

**Tier 3 — moving convenience tags.** On top of the immutable foundation, you layer the tags that are *meant* to move: `v1` and `v1.2` (always point at the latest patch in that line), `stable` or `latest` (point at the current blessed release), and environment tags like `staging` and `prod` (point at whatever is deployed there). These are pure conveniences for humans and tooling that wants "the current X." They move constantly and that is fine — *as long as nothing deploys by them.*

The cardinal sin is mixing tiers: treating a moving tag as if it were immutable, or deploying by a Tier-3 tag. The checkout incident was deploying by a Tier-3 tag (`latest`). The discipline is: Tier 1 and Tier 2 are immutable and traceable; Tier 3 moves and is for discovery only; and the deploy reads a *digest*, which is below all of them.

Here is a tagging step that implements all three tiers in one push. The same digest gets four tags — one immutable build tag, one immutable semver, and two moving convenience tags — and they all resolve to the identical manifest:

```bash
DIGEST_REPO=ghcr.io/acme/checkout
SHA=$(git rev-parse --short HEAD)        # e.g. abc123f
VERSION=v1.2.3                            # from your release process

# Build once, push with the immutable build tag.
docker buildx build --push -t "$DIGEST_REPO:$SHA" .

# Re-tag the SAME bytes (no rebuild) with semver and moving tags.
# crane copies by digest, so these are guaranteed the same image.
crane tag "$DIGEST_REPO:$SHA" "$VERSION"   # immutable release tag
crane tag "$DIGEST_REPO:$SHA" "v1.2"       # moving: latest 1.2.x patch
crane tag "$DIGEST_REPO:$SHA" "v1"         # moving: latest 1.x patch
crane tag "$DIGEST_REPO:$SHA" "stable"     # moving: current blessed release

# Capture the canonical digest for the deploy manifest.
crane digest "$DIGEST_REPO:$SHA"
# -> sha256:9f2bda1c... (this is what the k8s manifest pins)
```

### Immutable-tag enforcement: make the registry stop you

Discipline is good; enforcement is better, because discipline fails at 2 a.m. during an incident. Most serious registries can mark a repository's tags **immutable**, meaning once a tag is published, the registry *refuses* to let it be overwritten. This converts "we agreed not to move `v1.2.3`" into "the registry will reject the push that tries to." It closes the gap between policy and reality.

The mechanism differs per registry but the idea is identical. On ECR, you set the repository's tag mutability to `IMMUTABLE`. Here is the Terraform, because [everything as code](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) is a principle of this series and a registry config is exactly the kind of thing that should live in version control, not be clicked in a console:

```hcl
resource "aws_ecr_repository" "checkout" {
  name                 = "checkout"
  image_tag_mutability = "IMMUTABLE" # published tags cannot be overwritten

  image_scanning_configuration {
    scan_on_push = true # scan every pushed image for CVEs
  }
}
```

With `IMMUTABLE` set, a second `docker push` of a different image under an existing tag fails with `tag invalid: The image tag already exists`. That is the registry refusing to let `v1.2.3` silently become different bytes. It is a small line of config that eliminates an entire class of incident.

The subtlety: immutable-tag enforcement and moving convenience tags are in tension. If `v1.2.3` is immutable but `v1.2` must move to follow patches, you cannot make the whole repo immutable. The common resolutions are (a) two repositories — one with immutable tags for releases, one mutable for floating tags — or (b) a registry like Harbor or GHCR that supports tag-immutability *rules* (immutable patterns like `v*.*.*` and `*-rc*`, mutable everything else). The principle to preserve is: the tags you might ever deploy by, or cite in a changelog, are immutable; the tags that are explicitly conveniences may move. And underneath all of it, you deploy by digest, so even a misconfigured mutability rule cannot ship you untested bytes.

### Threading the digest through Helm and Kustomize

If you template your manifests — and most teams do — the digest needs a clean way in. Hardcoding 64 hex characters into a values file by hand is both error-prone and defeats the automation. The pattern is to make the digest a *value* that the pipeline sets at promotion time. With Helm, the chart references an image by repository plus digest, and the pipeline passes the digest as a `--set`:

```yaml
# values.yaml — the chart pins by digest, not tag
image:
  repository: registry.prod.acme.com/checkout
  # digest is supplied at deploy time by the pipeline; never a tag
  digest: ""

# deployment.yaml template
# image: "{{ .Values.image.repository }}@{{ .Values.image.digest }}"
```

```bash
# pipeline supplies the exact tested digest
helm upgrade --install checkout ./charts/checkout \
  --set image.digest="$DIGEST" \
  --namespace prod --wait
```

With Kustomize, the `images` transformer rewrites the digest declaratively, which is GitOps-friendly because the change is a committed diff:

```yaml
# kustomization.yaml in the prod overlay
images:
  - name: registry.prod.acme.com/checkout
    digest: sha256:9f2bda1c4e7a8f0b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293
```

In a GitOps setup, the promotion step does not run `kubectl` at all — it commits this `digest:` change to the environment's repo, and Argo CD or Flux reconciles the cluster to match. The promotion is then a *pull request against the prod overlay that bumps one digest*, which is auditable, reviewable, and revertable. That is "everything as code" applied to promotion: the record of what is deployed where is a Git history of digest bumps, not a mutable tag whose history the registry may or may not retain.

#### Worked example: tracing a 3 a.m. page back to a commit

It is 3 a.m. and the `checkout` service is throwing errors. You `kubectl get pod checkout-xxxxx -o jsonpath='{.spec.containers[0].image}'` and get back `ghcr.io/acme/checkout@sha256:9f2bda1c...`. With a digest in hand, the trace is mechanical. `crane ls ghcr.io/acme/checkout | xargs -I{} sh -c 'echo {} $(crane digest ghcr.io/acme/checkout:{})'` lists every tag and its digest; grep for `9f2bda1c` and you find this digest is also tagged `abc123f` (the build tag) and `v1.2.3`. The build tag `abc123f` is a git short SHA, so `git show abc123f` shows you the exact commit, author, and diff that produced the bytes currently failing in production. Total time: about ninety seconds, and zero ambiguity. Now suppose the manifest had said `:latest` instead. You query the running image, it says `:latest`, and `latest` currently points at a *newer* build than the one running (because CI pushed twice since the deploy). You have no idea what is actually running without digging into pod creation timestamps and CI logs. The digest-plus-immutable-build-tag scheme turns a forensic investigation into a `git show`. That traceability is not a side benefit; it is half the reason the scheme exists.

## 4. Promotion: moving the same bytes, not rebuilding them

We have an immutable artifact in a registry, named with digests for safety and tags for humans. Now we need to move it through environments — dev, staging, prod — on its way to production. This is **promotion**, and the single most important rule is: *promotion never rebuilds*. Promotion is a registry operation, not a build operation. You take the digest that passed the gate for the previous environment and you make it available to the next one. The bytes do not change. That is the whole point of "build once, promote everywhere" — the [sibling post](/blog/software-development/ci-cd/build-once-promote-everywhere-artifacts-and-versioning) makes the case for the principle; here is the mechanics.

![A timeline showing one image digest built once, pushed to the dev registry, passing tests, copied to the prod registry with crane copy, and deployed to prod as the same digest](/imgs/blogs/image-registries-tagging-and-promotion-4.png)

There are two flavors of promotion, and most mature setups use both.

**Promotion within one registry, by re-tagging.** The simplest case: dev, staging, and prod all pull from the same registry, and "promotion" means moving an environment tag (`prod`) to point at a digest that has cleared staging — and, more importantly, updating the *deploy manifest* in the prod environment to pin that digest. The bytes never move; only a pointer and a manifest reference change. Figure 4 shows this as a timeline: build once, push to dev, test, then the same digest flows to prod with no rebuild in between.

**Promotion across registries, by copying the digest.** The more robust case, especially for security: separate registries per trust tier — a permissive dev/build registry that any CI run can push to, and a locked-down prod registry that only accepts promoted-and-signed images. Promotion here means *copying* the digest from the dev registry to the prod registry. And the key insight is that you copy bytes, you do not pull-rebuild-push. The tools that do this are `crane` (from Google's `go-containerregistry`) and `skopeo` (from the Red Hat / containers ecosystem). Both can copy an image from one registry to another *without a local Docker daemon and without unpacking it* — they stream the layers registry-to-registry, preserving the digest exactly.

Here is the promotion command, the artifact the prompt asks for:

```bash
# Promote a specific digest from the dev registry to the prod registry.
# crane copies layer-by-layer, registry to registry, preserving the
# digest. No pull, no rebuild, no push of locally-fabricated bytes.
crane copy \
  ghcr.io/acme-dev/checkout@sha256:9f2bda1c4e7a8f0b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293 \
  registry.prod.acme.com/checkout@sha256:9f2bda1c4e7a8f0b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293

# Verify the digest survived the copy (it must be identical):
crane digest registry.prod.acme.com/checkout:v1.2.3
# -> sha256:9f2bda1c...  (same 64 hex chars; the bytes did not change)
```

The equivalent with `skopeo`, which some teams prefer because it handles signatures and multiple transports cleanly:

```bash
skopeo copy --all \
  docker://ghcr.io/acme-dev/checkout@sha256:9f2bda1c... \
  docker://registry.prod.acme.com/checkout:v1.2.3
```

The `--all` flag tells skopeo to copy the *entire* manifest list (all architectures), which matters for multi-arch images — more on that shortly. The thing to internalize: because the copy is keyed by digest and preserves bytes exactly, the image in the prod registry is provably the same image that passed tests in dev. There is no rebuild, so there is no rebuild drift. Promotion is a `cp` of immutable content, decorated with a verification.

It is worth contrasting the promotion approaches explicitly, because teams reach for the wrong one all the time — most commonly "rebuild from the same tag," which is not promotion at all but a fresh build wearing promotion's clothes:

| Approach | Bytes guaranteed identical? | Rebuild drift risk | Cross-registry? | Verdict |
|---|---|---|---|---|
| Rebuild per environment | No — new build each time | High ($1-p^{n-1}$ across n envs) | N/A | Anti-pattern; defeats build-once |
| Move a mutable tag | Only if you also pin the digest | None if digest-pinned | Same registry only | OK within one registry |
| `crane copy` / `skopeo copy` by digest | Yes — bytes preserved | None | Yes (the point) | Best for cross-trust-tier promotion |
| Re-tag the same digest (`crane tag`) | Yes — same manifest | None | Same registry only | Best within one registry |

The two right-hand rows are real promotion: they move or reference the *same digest*. The top row is the trap. The Knight Capital and checkout stories are both, at root, "the thing deployed was not the thing validated," and copy-by-digest is the mechanism that makes that mismatch impossible rather than merely discouraged.

### Why "promote, don't rebuild" is provably safer

Let me make the why provable rather than asserted, because the series demands it. Suppose your build is reproducible with probability $p$ — that is, two builds of the same source produce byte-identical images with probability $p$. In practice $p < 1$ for almost every real build, because of timestamps, dependency resolution against moving upstream tags, non-deterministic compiler ordering, and the network. Now consider a release that flows through $n$ environments.

If you **rebuild per environment**, you perform $n$ builds, and the probability that *all* $n$ produce the same artifact is $p^n$. The probability that production's rebuild differs from the artifact staging tested is $1 - p^{\,n-1}$ for the prod-vs-staging comparison alone. Even at a generous $p = 0.99$, across a four-environment flow the chance of a drift somewhere is $1 - 0.99^{3} \approx 3\%$. Three percent of releases ship something a prior environment did not test. That is a *lot* of incidents at thirty deploys a day.

If you **build once and promote**, you perform exactly one build. The probability that prod runs the artifact staging tested is $1$, by construction, because it is literally the same digest. The drift probability collapses to zero — not "low," zero, because you removed the rebuild that introduced it. The arithmetic is the entire argument: rebuilding multiplies your exposure to non-reproducibility by the number of environments; promoting eliminates it. You do not need your build to be perfectly reproducible if you only build once.

#### Worked example: the moving-tag burn, before and after

This is the checkout incident, written out with numbers so you can see the fix in the manifest. **Before:** the prod Deployment specifies `image: registry.internal/checkout:latest`. On Monday at 14:00, staging pulls `:latest`, which resolves to digest `…41ab` (build 41), and runs the full suite — green. On Wednesday at 09:30, two PRs have merged since Monday; CI pushed build 42 (`…42cd`) at Tuesday 16:00 and re-pointed `:latest`. The Wednesday prod deploy pulls `:latest`, which now resolves to `…42cd`. Build 42 contains an untested change to JSON error handling. Prod runs `…42cd`. Staging tested `…41ab`. The error rate climbs from 0.1% to 6%; at, say, 400 requests/second on checkout, that is roughly 23 failed checkouts per second until rollback — call it eight minutes to detect, page, and roll back, so on the order of 11,000 failed checkouts. The cost is real and it traces entirely to one mutable pointer.

**After:** the pipeline changes two things. First, the deploy job pins the digest the test job validated: the test job ran against `…41ab`, so the prod manifest says `image: registry.internal/checkout@sha256:…41ab`. Second, immutable-tag enforcement is on for the release repo so nobody can overwrite `v1.2.3`. Now the Wednesday deploy resolves `@sha256:…41ab` — there is no tag to move, and the registry can only return the bytes that hash to `…41ab`. Prod runs exactly what staging tested. Build 42, the untested change, never reaches prod by accident; it gets its own digest, its own staging validation, and its own explicit promotion. The before-and-after manifest diff is two lines — `:latest` becomes `@sha256:…41ab` — and it closes the TOCTOU window completely. Two lines of YAML versus 11,000 failed checkouts is, in my experience, the best return on investment in all of delivery engineering.

### Promotion as a gated registry operation

Here is the full promotion job as it appears in a real pipeline, tying together copy-by-digest, environment registries, and a gate. This is a GitLab CI shape using DAG `needs`:

```yaml
stages: [build, test, promote-prod, deploy-prod]

build:
  stage: build
  script:
    - docker buildx build --push -t "$DEV_REG/checkout:$CI_COMMIT_SHA" .
    - crane digest "$DEV_REG/checkout:$CI_COMMIT_SHA" > digest.txt
  artifacts:
    paths: [digest.txt]

test:
  stage: test
  needs: [build]
  script:
    - DIGEST=$(cat digest.txt)
    - ./run-integration-suite.sh "$DEV_REG/checkout@$DIGEST"

promote-prod:
  stage: promote-prod
  needs: [test]
  rules:
    - if: '$CI_COMMIT_BRANCH == "main"'   # only main is promotable
      when: manual                         # human approval gate
  script:
    - DIGEST=$(cat digest.txt)
    # copy the EXACT tested bytes into the locked-down prod registry
    - crane copy "$DEV_REG/checkout@$DIGEST" "$PROD_REG/checkout@$DIGEST"
    - crane tag "$PROD_REG/checkout@$DIGEST" "v$CI_PIPELINE_IID"

deploy-prod:
  stage: deploy-prod
  needs: [promote-prod]
  script:
    - DIGEST=$(cat digest.txt)
    - kubectl set image deploy/checkout checkout="$PROD_REG/checkout@$DIGEST"
```

Notice the `test` and `deploy-prod` jobs both reference `@$DIGEST` — the exact same value, flowing through a pipeline artifact (`digest.txt`). That is the thread that guarantees you deploy what you tested. The `when: manual` rule is the human approval gate; the `crane copy` is the promotion; the prod registry is a separate trust tier. (For the *why this is safe* in reliability terms — gating a promotion on signals, bounding blast radius — see SRE's [deploying safely with progressive delivery](/blog/software-development/site-reliability-engineering/deploying-safely-progressive-delivery). This post owns the registry mechanics; SRE owns the reliability theory.)

### Stress-testing the promotion path

A design is only as good as how it behaves when things go wrong, so let me poke at this one the way I would in a review.

*What if two PRs merge to main at almost the same time?* Both pipelines build, both push to the dev registry under their own git-SHA tags (which never collide, because the SHAs differ), and both produce distinct digests. There is no race on the digest, because each build's digest is a function of its own bytes. The only thing they might race on is a *moving* tag like `stable` or an environment tag — and since nothing deploys by those, a race there is cosmetic, not dangerous. The deploy of each pins its own digest. This is precisely the failure that deploy-by-digest immunizes you against: concurrent pushes cannot cross-contaminate, because immutable identities cannot collide.

*What if the registry is down mid-deploy?* If the prod registry is unreachable when a pod tries to pull, the pod stays in `ImagePullBackOff` and the rollout stalls — but, crucially, it does not deploy *wrong* bytes; it deploys *no* bytes. The already-running pods keep serving. This is a safe failure mode (availability degraded, correctness preserved). The mitigations are operational: a pull-through cache or registry replica close to the cluster so a transient outage of the primary does not block pulls, and `imagePullPolicy: IfNotPresent` so nodes that already cached the digest can start without the registry. Because the reference is a digest, `IfNotPresent` is *safe* here — a cached digest is guaranteed to be the right bytes, whereas caching by mutable tag would risk serving a stale image. Deploy-by-digest is what makes aggressive image caching safe.

*What if the `crane copy` fails halfway?* Layer copies are idempotent and content-addressed: a re-run of `crane copy` skips layers already present in the destination (it checks digests first) and only re-uploads what is missing. A half-finished copy leaves the destination without the final manifest, so the digest is simply not yet resolvable there — the deploy that depends on it will not find the image and will not proceed. Re-running the promotion completes it. There is no partial-image-runs state, because the manifest is written last and a manifest without all its layers is not servable.

*What if a secret — say the prod-registry push credential — leaks?* This is exactly what the per-environment split in the next section bounds. A leaked dev-registry push credential lets an attacker pollute the dev registry, but the prod registry only admits signed-and-verified digests through the promotion gate, so the leaked credential cannot land a malicious image in prod. A leaked *prod* push credential is more serious — which is why that credential should be short-lived and OIDC-federated (no static secret to leak) and why nothing but the promotion role holds it.

*What if you need to roll back and the previous image was garbage-collected?* This is the retention hazard, and it is the subject of section 6. The short version: your retention policy must protect every digest a workload references and keep a rollback horizon at least as long as your incident-response horizon, or rollback-as-mitigation becomes rollback-to-an-image-that-no-longer-exists.

The pattern across all of these: because the identity you deploy is an immutable digest, the failure modes degrade toward *not deploying* rather than *deploying the wrong thing*. That is the property you want. A pipeline that fails closed — stalls rather than ships untested bytes — is a pipeline you can trust to run thirty times a day.

## 5. Per-environment registries and access control

The promotion job above quietly introduced a powerful pattern: separate registries per trust tier. Let me make the case for it explicitly, because it is where registry hygiene meets supply-chain security.

![A graph showing a build pushing a digest to a permissive dev registry, then tests and signing, then a policy gate that rejects unsigned digests and admits verified ones into a prod registry that accepts signed images only](/imgs/blogs/image-registries-tagging-and-promotion-7.png)

The idea, shown in figure 7, is to split your registries by how much you trust what is in them:

- A **dev / build registry** that any CI run on any branch can push to. It is permissive on purpose — it is where work-in-progress, feature-branch builds, and experiments live. It is also where most of your storage churn happens, so it gets aggressive retention (more in the next section).
- A **prod registry** that is locked down. *Nothing* can push to it directly. The only way in is the promotion job, and the promotion job only runs after tests pass, the image is signed, and a policy gate verifies the signature. The prod registry's pull permissions are scoped to the prod cluster's service account and nothing else.

Why bother? Because it converts "we hope only good images reach prod" into "only promoted-and-signed images *can* reach prod." The blast radius of a leaked CI token shrinks dramatically: a token that can push to the dev registry cannot, by itself, get anything into the prod registry. To land code in prod, an attacker would need to get it past tests *and* sign it with the key the gate trusts *and* pass the promotion policy — a much taller wall than "push to one registry everything pulls from."

The signing-and-verification step is where this composes with the supply-chain work. The full treatment of Sigstore, `cosign`, and SLSA provenance is its own post (planned in this series as `signing-and-provenance-with-sigstore-and-slsa`); here, the relevant slice is that the prod registry's admission gate verifies a signature *over a digest*. You sign the digest, the gate verifies the signature against that digest before promotion, and a Kubernetes admission policy (Kyverno or the Sigstore policy-controller) re-verifies at deploy time that the image being pulled is signed. Three sketch commands:

```bash
# In CI, after tests pass, sign the digest (keyless, OIDC-backed):
cosign sign --yes "$DEV_REG/checkout@$DIGEST"

# In the promotion gate, verify before copying into prod:
cosign verify \
  --certificate-identity-regexp "https://github.com/acme/.*" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  "$DEV_REG/checkout@$DIGEST"

# Only if verification passes does the gate run the crane copy.
```

The access-control layer underneath is just IAM, scoped tightly. On ECR, the prod repository's resource policy grants `ecr:BatchGetImage` and `ecr:GetDownloadUrlForLayer` to exactly the prod cluster's node role, and `ecr:PutImage` to *only* the promotion role. On GHCR, package visibility is private and the prod cluster authenticates with a narrowly-scoped pull token. The principle: dev is write-permissive and read-broad; prod is write-locked to the promotion path and read-locked to the prod consumers. The registry stops being a single shared bucket and becomes a set of trust boundaries that promotion crosses deliberately and verifiably.

| Decision | Single shared registry | Per-environment registries |
|---|---|---|
| Who can push | Any CI job | Dev: any CI job. Prod: only the promotion role |
| Blast radius of a leaked CI token | Anything anywhere | Bounded to dev; prod unreachable directly |
| Promotion mechanism | Move a tag | `crane copy` digest across a trust boundary |
| Prod admission | Whatever was pushed | Signed-and-verified digests only |
| Operational cost | Low (one registry) | Higher (two registries, copy step, signing) |
| When it is worth it | Small team, low threat | Regulated, multi-tenant, or high-value targets |

The honest trade-off in that last row matters. Per-environment registries plus signing add real operational weight: another registry to run or pay for, a copy step that can fail, a key or OIDC trust to manage. For a three-person startup shipping an internal tool, this is over-engineering — one private registry, deploy-by-digest, and immutable release tags get you 90% of the safety at 10% of the cost. The full split earns its keep when you are a high-value target, when you have regulatory pressure (the gate is your auditable control), or when many teams share infrastructure and you cannot trust every CI job equally. Reach for the boundary when the threat model justifies it, not before.

## 6. Retention and garbage collection: bounding cost without breaking rollback

Images accumulate forever if you let them, and forever is expensive. Every build pushes layers; most of those layers are never deployed; and unless something deletes them, your registry's storage bill climbs without bound. Retention and garbage collection are how you bound the cost — and they are also a beautifully easy way to cause an outage if you are careless, because the wrong GC deletes an image a running pod still needs.

![A before and after comparison showing keeping every image forever growing storage to 4.5 terabytes versus a keep ninety days plus deployed policy holding storage near 400 gigabytes while preserving rollback targets](/imgs/blogs/image-registries-tagging-and-promotion-8.png)

Let me start with the cost, because the numbers motivate everything else. Figure 8 contrasts the two regimes.

#### Worked example: sizing a retention window

A busy service team pushes about 50 builds per day. Each build's *new* layers (the application layer, the changed bits — not the shared base, which is deduplicated) average about 90 MB after compression. So new image content lands at roughly $50 \times 90\ \text{MB} = 4.5\ \text{GB/day}$, or about $4.5 \times 30 \approx 135\ \text{GB/month}$, or $\approx 1.6\ \text{TB/year}$. Run that for a couple of years across several services and you are paying for multiple terabytes of images, the overwhelming majority of which are feature-branch builds nobody will ever deploy again. At, say, \$0.10/GB-month for registry storage, a year of one team's unmanaged churn is on the order of $1{,}600\ \text{GB} \times \$0.10 = \$160/\text{month}$ and climbing — small per team, but it multiplies across teams and never stops growing. Unbounded retention is a slow leak that becomes a real line item.

Now compare two policies. **Keep-last-20** keeps only the 20 most recent images and deletes the rest. Storage is bounded tightly: about $20 \times 90\ \text{MB} = 1.8\ \text{GB}$ of unique content per repo. Cheap — but dangerous, because 20 builds at 50/day is *less than half a day* of history. If you need to roll back to last Tuesday's release, it is gone. Keep-last-N counts builds, not time, and a busy repo burns through N fast. **Keep-90-days-plus-deployed** keeps anything pushed in the last 90 days *and* anything currently or recently deployed, and sweeps untagged builds older than that. Storage settles at roughly the 90-day churn that has not aged out — for this team, on the order of a few hundred GB rather than terabytes — while guaranteeing that every image you might realistically need to roll back to is still present. For most teams, a time window plus a "protect what is deployed" rule beats a raw count, because rollback safety is a function of *time* ("roll back to last week") not of *build count*.

### What to keep and what to sweep

The discipline is to split images into keep-rules and sweep-rules, and to be conservative about what you keep. Figure 6 lays out the taxonomy.

![A tree diagram of a retention policy splitting into keep rules that preserve tagged releases and digests deployed in the last ninety days, and sweep rules that remove untagged builds and orphan layers](/imgs/blogs/image-registries-tagging-and-promotion-6.png)

**Keep rules** (never delete these):

- **Tagged releases.** Anything with a semver tag (`v1.2.3`) is a release someone may cite or roll back to. Keep the last N of these, where N is generous (say, the last 50 releases) — releases are infrequent compared to builds, so this is cheap.
- **Anything deployed recently.** Any digest that is, or recently was, running in any environment. This is the rollback-safety rule and it is the one teams most often forget. If a pod references a digest, that digest must not be garbage-collected, full stop.

**Sweep rules** (safe to delete):

- **Untagged builds older than a window.** Feature-branch builds, superseded CI pushes, anything with no semver tag and no recent deployment, past (say) 14 days. This is the bulk of the savings.
- **Orphan layers.** Blobs no manifest references anymore. Once the manifests that pointed to a layer are gone, the layer is unreachable and the registry's GC reclaims it. This is the layer-level GC that actually frees disk; deleting a manifest only un-references its layers, and a separate GC pass collects the now-orphaned blobs.

Here is the keep-90-days-plus-tagged policy as configuration. GHCR / GitHub Packages expresses retention through cleanup rules; here is the equivalent on Google Artifact Registry, which has a clean declarative form:

```bash
# Google Artifact Registry cleanup policy: delete untagged images
# older than 14 days, but KEEP the most recent 50 tagged releases.
gcloud artifacts repositories set-cleanup-policies checkout \
  --location=us-central1 \
  --policy=cleanup-policy.json

# cleanup-policy.json:
# [
#   { "name": "delete-old-untagged",
#     "action": { "type": "Delete" },
#     "condition": { "tagState": "UNTAGGED",
#                    "olderThan": "1209600s" } },   # 14 days
#   { "name": "keep-recent-releases",
#     "action": { "type": "Keep" },
#     "mostRecentVersions": { "keepCount": 50,
#                             "packageNamePrefixes": ["v"] } }
# ]
```

And the ECR lifecycle policy, expressed as JSON rules, which is the form many readers will actually use:

```json
{
  "rules": [
    {
      "rulePriority": 1,
      "description": "Keep last 50 tagged releases (semver)",
      "selection": {
        "tagStatus": "tagged",
        "tagPrefixList": ["v"],
        "countType": "imageCountMoreThan",
        "countNumber": 50
      },
      "action": { "type": "expire" }
    },
    {
      "rulePriority": 2,
      "description": "Expire untagged images older than 14 days",
      "selection": {
        "tagStatus": "untagged",
        "countType": "sinceImagePushed",
        "countUnit": "days",
        "countNumber": 14
      },
      "action": { "type": "expire" }
    }
  ]
}
```

### The GC-deletes-an-image-you-still-need hazard

Now the hazard, because it is the one that turns a cost-saving policy into an outage. Suppose your retention is "keep last 20 untagged builds" and a long-running service has been stable for two weeks — 700 builds ago. Its running pods reference a digest that aged out of the last-20 window 690 builds back. The GC, doing exactly what you configured, deletes that digest. Everything is fine *until* a pod restarts — a node drains, an autoscaler scales up, a crash triggers a restart — and the kubelet tries to pull the image. The pull fails: `ErrImagePull`, manifest unknown. The service cannot start new replicas. You have a partial outage, and the cause is not a bug in the code; it is that you deleted the bytes the code was running.

This is why the "keep anything deployed in the last 90 days" rule is not optional. A pure count-based or time-since-push policy does not know what is deployed. The robust pattern is to *exempt deployed digests from GC*, which means the registry retention policy must be informed by what the cluster is actually running. There are two common ways to do this:

1. **Pin and protect.** A small reconciler (a CronJob) lists the images currently referenced across all Deployments/StatefulSets in all clusters, and applies a "protected" label or skips them in the cleanup policy. Pinned digests are never swept regardless of age.
2. **Generous time window.** Make the window so long (90 days, 180 days) that no realistically-still-running image ages out before it would have been redeployed. Cruder, but simple, and it composes with the protect-deployed rule as a backstop.

The rule to write on the wall: **never garbage-collect a digest a running workload references, and never let your rollback horizon shrink below your incident-response horizon.** If you might need to roll back to last month's release during an incident, last month's release must still be in the registry. Retention that breaks rollback is worse than no retention, because it fails silently and only bites you during a restart — which is exactly when you can least afford it. (Keeping rollback targets is a reliability concern; the SRE post on [mitigate first, diagnose later](/blog/software-development/site-reliability-engineering/mitigate-first-diagnose-later) treats rollback as the primary mitigation, and a missing image is a mitigation that is not there when you reach for it.)

| Policy | Storage | Rollback safety | Failure mode |
|---|---|---|---|
| Keep forever | Unbounded, grows ~1.6 TB/yr per busy team | Total | Storage bill never stops climbing |
| Keep last 20 | Tight, ~1.8 GB/repo | Poor (under half a day) | GC deletes a still-running digest |
| Keep 90d + tagged | Bounded, settles a few hundred GB | Good (90-day horizon) | None, if deployed digests are protected |
| Keep 90d + tagged + protect deployed | Bounded plus a few pinned olds | Best (covers every live workload) | Slightly more complex to operate |

## 7. Multi-arch images, briefly

A quick but practical detour, because promotion tooling and digests behave a little differently for multi-architecture images and it trips people up. When you build for both `linux/amd64` and `linux/arm64` (common now that ARM instances and Apple Silicon laptops are everywhere), `docker buildx` does not produce one image; it produces two, plus a **manifest list** (also called a manifest index) that ties them together. The manifest list is itself an object with its own digest, and it maps each architecture to that architecture's image digest. When a node pulls `myapp:v1.2.3`, the registry serves the manifest list, and the container runtime selects the entry matching the node's architecture.

The implications for this post are three. First, when you pin a digest, pin the *manifest list's* digest, not a single-arch digest — the manifest-list digest resolves correctly on both amd64 and arm64 nodes, while a single-arch digest will fail to run on the wrong architecture. `buildx` outputs the manifest-list digest as the push digest, so this is the natural thing to capture. Second, when you promote with `skopeo copy`, use `--all` (or `crane copy`, which copies the whole index by default) so you copy *both* architectures, not just the one matching the machine running the copy. Forgetting `--all` is a classic "works on the build host, fails on half the cluster" bug. Third, the manifest-list digest is still immutable and content-addressed — everything in this post about deploy-by-digest applies unchanged; you just pin the index digest. Multi-arch does not weaken the model; it adds one layer of indirection that the tooling handles for you if you pass the right flag.

You can confirm what you have with `docker buildx imagetools inspect registry/checkout@sha256:…`, which prints the manifest list and the per-architecture digests it points to. If you see entries for `linux/amd64` and `linux/arm64`, your index is complete; if a promotion only carried one, this is where you will catch it before a single-arch node lands in `CrashLoopBackOff` with an `exec format error`. The retention discussion in section 6 also touches multi-arch: garbage collection must treat the index and its per-arch children as a unit, so a policy that protects the index digest must not orphan the architecture-specific child digests it references. Mature registries handle this correctly, but it is worth verifying that "protect this digest" follows the index down to its children rather than stopping at the top-level manifest.

## 8. The registry as an attack surface

The registry is not just a warehouse; it is a network service that hands executable code to your production cluster, which makes it a juicy target. A few of the ways it gets attacked, and the defenses, since this is where registry hygiene meets the supply chain.

**Anonymous pull and public exposure.** A registry repository left public, or a cluster configured to pull without authentication, lets anyone enumerate and download your images — and your images often contain secrets baked in by mistake, internal hostnames, and a map of your architecture. Defense: private repositories by default, authenticated pulls scoped to the consumers that need them, and a scan in CI for secrets accidentally committed to image layers (Trivy and others detect embedded credentials).

**The typosquatted public image.** Your Dockerfile says `FROM node:20` and pulls from Docker Hub; an attacker publishes a malicious image under a name one keystroke away (`FROM ndoe:20` or a lookalike org) hoping a copy-paste or autocomplete error pulls it. More broadly, depending on a mutable public tag means you are trusting whoever controls that tag upstream. Defense: pin base images by digest (`FROM node:20@sha256:…`), mirror the bases you depend on into your own registry, and scan pulled images before they enter your build.

**Pull-through caches and dependency confusion.** A pull-through cache proxies a public registry and caches images locally — convenient and fast, but if it is misconfigured to fall back to the public registry for a name you intended to resolve internally, an attacker can publish a public image with your internal name and get it pulled (the registry analog of the npm/PyPI dependency-confusion attacks). Defense: explicit registry prefixes (never bare image names that could resolve either place), namespace your internal images so they cannot collide with public ones, and configure the cache to *not* fall through for your private namespaces.

There is also a defense that enforces the thesis of this whole post at the cluster boundary: a Kubernetes admission policy that *refuses to run an image referenced by a mutable tag at all*. If the only images the cluster will admit are digest-pinned (and, ideally, signed), then a manifest that sneaks in `:latest` is rejected before it can schedule a pod — turning "we agreed to deploy by digest" into "the cluster will not run anything else." A Kyverno policy expresses it compactly:

```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-image-digest
spec:
  validationFailureAction: Enforce
  rules:
    - name: images-must-use-digest
      match:
        any:
          - resources:
              kinds: [Pod]
      validate:
        message: "Images must be referenced by sha256 digest, not a tag."
        pattern:
          spec:
            containers:
              - image: "*@sha256:*"
```

That single policy closes the loophole at the last possible moment: even if a tagged reference slips through code review and the pipeline, the cluster rejects it. Combined with a signature-verification policy (the Sigstore policy-controller or Kyverno's `verifyImages` rule), the cluster will only run images that are both digest-pinned and signed — defense in depth around the registry.

The deeper defense ties back to section 5: a locked-down prod registry that only admits signed-and-verified digests means even a compromised dev pull or a poisoned cache cannot get a malicious image into production without also forging a signature the admission gate trusts. The registry trust boundaries you built for promotion safety double as supply-chain defenses. (The real-world stakes here are not hypothetical — the [war story](#9-war-story-the-codecov-and-dependency-confusion-lessons) section covers what these attacks look like in practice.)

## 9. War story: the Codecov and dependency-confusion lessons

Two real incidents make the registry-as-attack-surface concrete, and a third illustrates the deploy-the-wrong-bytes failure at catastrophic scale.

**Codecov (2021).** Attackers gained access to Codecov's Bash Uploader script — distributed and pulled by thousands of CI pipelines — and modified it to exfiltrate environment variables, including registry credentials and cloud keys, from every CI run that executed it. The relevant lesson for registries: a CI pipeline holds the keys to push to your registries, and anything that runs in CI can steal those keys. This is precisely why per-environment registries with a locked-down prod tier matter — a stolen dev-registry credential is bounded; a stolen credential that can push directly to prod is catastrophic. It is also why keyless, short-lived OIDC-federated credentials beat long-lived stored registry passwords: there is no static secret to exfiltrate.

**Dependency confusion (2021, Alex Birsan's research).** A researcher demonstrated that publishing public packages with the same names as companies' *internal* packages caused build systems at Apple, Microsoft, and dozens of others to pull the attacker's public package instead of the internal one, executing attacker code in trusted build environments. The container-registry analog is exactly the pull-through-cache and typosquatting risk from section 8: if your tooling can resolve an image name to either an internal or a public source, an attacker controls which. The defense is the same — explicit registry prefixes, namespacing, and never trusting a bare image name.

**Knight Capital (2012).** Not a registry attack, but the purest illustration of "deploying the wrong bytes." A deploy pushed new code to seven of eight servers and left old, dead code running on the eighth — and the new code reused a flag that the old code interpreted differently. The mismatched-version fleet executed millions of erroneous trades in 45 minutes, and the firm lost roughly \$440 million and effectively ceased to exist. The registry-and-promotion lesson is stark: a deploy is only safe when *every* target runs the *exact same validated artifact*. Deploy-by-digest with an immutable identity is the mechanism that makes "every server runs the same bytes" verifiable rather than hoped-for. If those eight servers had each been pinned to one digest and the rollout had been atomic and verified, the partial-deploy state that killed the firm could not have existed.

These are accurate accounts; the dollar figures for Knight Capital and the scope of Codecov and dependency-confusion are documented in the public record. The numbers in the *worked examples* earlier (build rates, storage, error rates) are illustrative and labeled as such — defensible orders of magnitude, not a specific company's audited metrics.

## 10. How to reach for this (and when not to)

Every practice in this post has a cost, so here is the decisive guidance on what to adopt and when to skip it.

**Always deploy by digest. No exceptions, at any team size.** This is the cheapest, highest-leverage change in the whole post — two characters of YAML (`:tag` becomes `@sha256:…`) plumbed through your pipeline. It costs nothing operationally and it eliminates the single most common "we shipped what we didn't test" incident. If you take one thing from this post, take this. There is no team small enough to justify deploying by a mutable tag.

**Always use an immutable build tag (git SHA) for traceability.** Also free, also universal. It turns a 3 a.m. forensic investigation into a `git show`. Layer your moving convenience tags on top of it freely.

**Turn on immutable-tag enforcement once you have real releases.** Low cost, high value the moment more than one person can push. Skip it only on throwaway/sandbox repos.

**Use per-environment registries with a signed-promotion gate when your threat model justifies it — not before.** This is the expensive practice. It earns its keep for high-value targets, regulated environments, and shared multi-tenant infrastructure. For a small internal-tooling team, it is over-engineering; one private registry, deploy-by-digest, and immutable release tags get you most of the safety. Do not build a two-registry signed-promotion platform before you have deploy-by-digest working, because you will have spent your complexity budget on the second-most-important thing while the first is still broken.

**Always have a retention policy, but tune it to your rollback horizon, not a build count.** Unbounded retention is a slow, certain cost leak; keep-last-N is a fast, certain rollback-safety risk on a busy repo. Keep-by-time-window plus protect-what-is-deployed is the safe default. The one rule you cannot violate: never GC a digest a running workload references.

**Pin base images by digest, mirror the ones you depend on, and scan on push** as soon as you are pulling third-party bases — which is essentially day one. The cost is small and the typosquatting/dependency-confusion downside is severe.

The meta-rule: adopt these in order of leverage-per-cost. Deploy-by-digest and immutable build tags first (free, universal). Retention policy and immutable release tags next (cheap, near-universal). Per-environment registries, signing, and admission gates last and only when the threat model calls for them. Spending complexity where it is not yet warranted is its own failure mode.

## 11. Key takeaways

- A registry is a **content-addressed store**: layers are keyed by their sha256 digest and stored once, shared across every image that includes them. Understanding this makes the tag-vs-digest distinction obvious.
- A **tag is a mutable, human-friendly pointer; a digest is the immutable, cryptographic identity** of exact bytes. They are opposites. Conflating them is how teams ship untested code.
- **`latest` (and any mutable tag) in a production manifest is a TOCTOU bug.** The pointer can move between the test and the deploy, so you validate one image and ship another.
- **Tag for humans, deploy by digest.** Tags are for discovery and conversation; the deploy manifest must pin `@sha256:…` — the exact digest your tests ran against, threaded through the pipeline as a captured output.
- Use a **three-tier tagging scheme**: an immutable git-SHA build tag (never reused), immutable semver release tags, and moving convenience tags (`v1`, `stable`, `staging`, `prod`) layered on top. Enforce immutability in the registry, not just by agreement.
- **Promotion never rebuilds.** Move the same digest across environments with `crane copy` or `skopeo copy --all` — bytes-preserving, daemonless, drift-free. Building once and promoting collapses rebuild-drift probability to zero.
- **Per-environment registries with a signed-promotion gate** bound the blast radius of a leaked CI token: a permissive dev registry and a prod registry that admits only verified digests. Worth it when the threat model justifies the cost.
- **Retention policy by time window plus protect-what-is-deployed** bounds storage without breaking rollback. Never garbage-collect a digest a running workload references; never let your rollback horizon shrink below your incident-response horizon.
- The registry is an **attack surface**: private by default, pin and mirror base images by digest, namespace internal images, and never trust a bare image name or a fall-through pull-through cache.

## 12. Further reading

- [From commit to production: the CI/CD mental model](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) — the series intro that sets up the commit→build→test→package→deploy→operate spine and the DORA metrics this post leans on.
- [Build once, promote everywhere: artifacts and versioning](/blog/software-development/ci-cd/build-once-promote-everywhere-artifacts-and-versioning) — the principle this post operationalizes; why one immutable artifact must flow through every environment.
- [The build stage: reproducible, fast, and cacheable](/blog/software-development/ci-cd/the-build-stage-reproducible-fast-and-cacheable) — where digests are first produced; the case for pinning base images by digest.
- The planned sibling `multi-environment-promotion-dev-staging-prod` extends this post's copy-by-digest mechanics into a full dev→staging→prod promotion flow with gates and approvals; and `signing-and-provenance-with-sigstore-and-slsa` covers the cosign/SLSA layer the prod-registry admission gate depends on.
- The planned sibling `writing-a-production-dockerfile` covers the multi-stage builds and distroless bases that produce the small, layer-friendly images a registry stores efficiently.
- [Deploying safely with progressive delivery](/blog/software-development/site-reliability-engineering/deploying-safely-progressive-delivery) and [mitigate first, diagnose later](/blog/software-development/site-reliability-engineering/mitigate-first-diagnose-later) — the SRE reliability theory behind gated promotion and rollback-as-mitigation, which this post's mechanics serve.
- The OCI Image Format and Distribution specifications (the standards that define manifests, layers, digests, and the push/pull protocol); the `go-containerregistry`/`crane` and `skopeo` documentation for the promotion tooling; and the Sigstore/`cosign` and SLSA framework docs for the signing-and-provenance layer.
