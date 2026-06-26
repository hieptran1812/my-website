---
title: "Platform Engineering and the Internal Developer Platform"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Build an Internal Developer Platform that collapses new-service lead time from three weeks to four hours by turning your deployment golden path into self-service scaffolding every product engineer reaches for by default."
tags:
  [
    "ci-cd",
    "devops",
    "platform-engineering",
    "developer-experience",
    "backstage",
    "internal-developer-platform",
    "golden-path",
    "self-service",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/platform-engineering-and-the-internal-developer-platform-1.png"
---

Picture this. A 80-engineer company, six product squads, shipping features every day. On paper, they have adopted DevOps: each team owns its services end to end, builds its own pipelines, manages its own infrastructure. In practice, team Alpha writes raw `kubectl apply` scripts. Team Beta maintains a bespoke Helm wrapper that nobody outside the team understands. Team Gamma has a Bash deploy script that lives in a private Gist. Team Delta set up Argo CD but skipped the configuration conventions everyone else uses. Team Epsilon inherited a Jenkins pipeline from 2019 that no one dares touch. Team Zeta wrote a fresh GitHub Actions workflow last quarter — the fourth distinct GHA approach across the company.

A new engineer joins. She is productive in her domain and eager. She spends day one trying to understand which approach her squad uses. Day three she copies a Dockerfile from a different squad because nobody can find the canonical one. Day five she opens a ticket with the infrastructure team to request a database. The ticket sits in a queue. She spends the next two weeks chasing approvals, debugging YAML that was correct in the old cluster version but is not in the new one, and waiting for a secrets rotation she did not know was in progress. Her first service reaches production on day 22. Three weeks for hello world.

Nobody did anything wrong. Every team made rational local decisions. The DevOps principle of "you build it, you run it" was followed faithfully. What was missing was the layer between "you build it" and "you know how to build it in a way that works here." That layer is the Internal Developer Platform (IDP), and building it is platform engineering.

![The Internal Developer Platform component map, showing how the developer portal ties together the service catalog, scaffolder, golden path, and self-service infrastructure](/imgs/blogs/platform-engineering-and-the-internal-developer-platform-1.png)

This post is a field guide for the platform engineer who has to build that layer. You will get real Backstage config, a working scaffolder template, a Crossplane XRD, a reusable GitHub Actions workflow, and the organizational model that makes the whole thing sustainable for a team of three supporting eighty.

---

## The DevOps Tax Nobody Budgeted For

"You build it, you run it" is one of the best ideas in modern software delivery. Giving teams end-to-end ownership removes the handoff bottleneck, aligns incentives with reliability, and shortens feedback loops. The problem is that the full sentence — the one Werner Vogels actually meant at Amazon — was about ownership of the *product lifecycle*, not about every team solving the same infrastructure problems from scratch.

When DevOps adoption is implemented as "every team figures out CI/CD themselves," you do not get faster delivery. You get fragmentation. The math is simple: if each of N teams spends F hours per quarter reinventing shared infrastructure (Dockerfiles, Helm charts, CI pipelines, secret management, observability wiring), the organisation spends N×F hours on work that produces zero user value. At 6 teams and 80 hours per team per quarter, that is 480 engineer-hours — about 60 engineer-days — per quarter spent on accidental complexity.

The hidden cost is worse than the direct cost. Each team's bespoke setup becomes tribal knowledge. When an engineer leaves, the setup leaves with them. Onboarding new engineers requires re-learning N different conventions. Security and compliance remediations must be applied to N different implementations. Incidents surface in N different formats across N different monitoring setups.

The DevOps tax is not the price of empowerment. It is the price of empowerment without infrastructure abstraction.

Platform engineering is the discipline that pays down that tax. The platform team builds and operates the internal infrastructure that lets product teams own their services without owning the cognitive overhead of deploying, securing, and observing those services from scratch.

---

## What an IDP Actually Is

An Internal Developer Platform is not a tool. It is a product. It is the collection of capabilities — a service catalog, self-service templates, golden-path workflows, developer portal, and self-service infrastructure provisioning — assembled into a coherent experience that developers reach for because it is faster, safer, and less frustrating than any alternative.

The components that make up a mature IDP:

**Service Catalog.** A searchable, always-current registry of every service, library, data pipeline, and infrastructure resource in your organisation. Each entry records who owns it, what it does, what it depends on, what SLOs it carries, and where its runbook lives. This sounds trivial until you are debugging a 2 AM incident and the first 20 minutes are spent in Slack asking "who owns the payment service?"

**Scaffolder Templates.** Parameterised project generators that create a new service — complete with repository, CI pipeline, Kubernetes manifests, observability wiring, and a catalog entry — in one click or one CLI command. The scaffolder is how the platform team encodes the golden path once and amortises that encoding across every new service forever.

**Golden Path Workflows.** The approved, supported, recommended way to build and deploy a service. Not a mandate (more on the off-ramp in a later section), but a path so smooth that choosing it is the line of least resistance. Reusable GitHub Actions workflows, standardised Dockerfiles, shared Helm chart values, pre-configured OPA policies.

**Self-Service Infrastructure.** Developers requesting a PostgreSQL database, a Redis cluster, an S3 bucket, or a message queue without opening a ticket and waiting three days. Implemented via Crossplane XRDs, Terraform modules surfaced through a portal, or Port self-service actions backed by automation.

**Developer Portal.** The front door to all of the above. Backstage is the dominant open-source option. Port, Cortex, and OpsLevel are SaaS alternatives. The portal is where developers register services, trigger templates, view docs, check DORA metrics, and browse infrastructure.

![Time-to-first-deploy before and after the IDP: three weeks of manual coordination collapses to four hours of self-service scaffolding](/imgs/blogs/platform-engineering-and-the-internal-developer-platform-2.png)

These components do not need to arrive simultaneously. The most effective IDP build sequences them: catalog first (visibility), then scaffolder templates (speed), then self-service infra (independence), then metrics (improvement). Most teams get stuck by trying to build everything at once. Build in that order and you deliver value at each stage.

---

## Backstage: The Developer Portal That Spotify Built

In 2020 Spotify open-sourced Backstage at KubeCon NA, the internal developer portal they had been running since 2016. The announcement described a system managing roughly 2,000 services, 280 engineers, and a catalog that covered every data pipeline, library, and team in the organisation. Spotify donated Backstage to the CNCF in September 2022, where it became an incubating project. By 2024, hundreds of companies including American Airlines, Netflix, and Zalando were running it in production.

The origin was unglamorous. Spotify's infrastructure was fragmenting in the same way described in the opening of this post: each tribe (Spotify's term for what other companies call squads) was building tools in isolation. The catalog was the first piece — a simple index of what services existed and who owned them. The scaffolder came next. TechDocs followed. The plugin system grew as internal teams at Spotify began contributing tools that lived alongside the catalog.

Backstage is a React-based framework (not a SaaS product) for building a developer portal. The framework itself provides the shell, the routing, and the plugin system. Everything useful — the catalog, the scaffolder, TechDocs, CI/CD status, cost management — is delivered via plugins, either official or community-built.

### The Software Catalog: Every Kind Explained

The catalog is populated by `catalog-info.yaml` files that live alongside each entity in its repository. Backstage scrapes these files (via GitHub, GitLab, or Bitbucket integrations) and builds a live index.

Backstage defines six catalog kinds. Each serves a distinct purpose in the system map:

**Component** — a deployable piece of software: a microservice, a frontend app, a data pipeline, a library. The most common kind.

**API** — a published interface that other services consume: a REST API, a gRPC service, a GraphQL schema, a message topic. An API entity links the provider (a Component) to the consumers.

**Resource** — a piece of infrastructure that components depend on: a database, a queue, a blob storage bucket, a CDN. Resources have owners and appear in the dependency graph.

**System** — a collection of components, APIs, and resources that together deliver a product capability. The `checkout` system might contain the `payment-service` Component, the `payment-api` API, and the `payments-postgres` Resource.

**Domain** — a grouping of Systems that belong to a business domain. Useful at large scale (200+ services) to group Systems into "Commerce", "Identity", "Platform".

**Template** — a Backstage scaffolder template. Templates appear in the catalog so developers can discover and trigger them from the portal.

Here is a complete `catalog-info.yaml` for a real service, annotated to show how the schema fields connect to operational workflows:

```yaml
apiVersion: backstage.io/v1alpha1
kind: Component
metadata:
  name: payment-service
  description: Processes checkout payments via Stripe and manages refund workflows.
  annotations:
    github.com/project-slug: acme-corp/payment-service
    backstage.io/techdocs-ref: dir:.
    pagerduty.com/service-id: P1234XY
    grafana/dashboard-selector: "service=payment-service"
  tags:
    - payments
    - stripe
    - typescript
  links:
    - url: https://grafana.internal/d/payment
      title: Grafana Dashboard
      icon: dashboard
    - url: https://runbook.internal/payment-service
      title: Incident Runbook
      icon: help
spec:
  type: service
  lifecycle: production
  owner: group:payments-team
  system: checkout
  dependsOn:
    - component:order-service
    - resource:payments-postgres
  providesApis:
    - payment-api
```

The `annotations` block is where Backstage plugins pull context. The `pagerduty.com/service-id` field lets the PagerDuty plugin display the current on-call engineer and open incidents directly on the catalog page. The `grafana/dashboard-selector` field lets the Grafana plugin embed the live dashboard panel. When the payments team is paged at 2 AM, the on-call engineer opens the catalog, clicks `payment-service`, and has the Grafana link, the PagerDuty service ID, and the runbook URL available in under 10 seconds.

The companion `API` kind for the same system:

```yaml
apiVersion: backstage.io/v1alpha1
kind: API
metadata:
  name: payment-api
  description: REST API for initiating and querying payment transactions.
  annotations:
    backstage.io/techdocs-ref: dir:./docs/api
spec:
  type: openapi
  lifecycle: production
  owner: group:payments-team
  system: checkout
  definition:
    $text: ./openapi.yaml
```

The `definition.$text` field causes Backstage to inline the OpenAPI spec from the repository and render an interactive SwaggerUI panel inside the portal. Consumers of `payment-api` discover the spec, try endpoints with a live token, and find the owning team — all without leaving the developer portal.

The `Resource` kind for the database:

```yaml
apiVersion: backstage.io/v1alpha1
kind: Resource
metadata:
  name: payments-postgres
  description: Primary PostgreSQL database for the payment service.
  annotations:
    acme.com/aws-arn: "arn:aws:rds:us-east-1:123456789:db:payments-postgres-prod"
spec:
  type: database
  lifecycle: production
  owner: group:platform-team
```

And the `System` that groups them:

```yaml
apiVersion: backstage.io/v1alpha1
kind: System
metadata:
  name: checkout
  description: End-to-end checkout flow, from cart to payment confirmation.
spec:
  owner: group:checkout-leads
  domain: commerce
```

With these four files in their respective repositories, the Backstage catalog can render a full dependency graph for the `checkout` system: the `payment-service` component depends on the `order-service` component and the `payments-postgres` resource, and it exposes the `payment-api`. This graph is navigable in the portal and answers "what breaks if payments-postgres goes down?" in seconds.

### The Scaffolder: A Complete Node.js Template

The scaffolder is Backstage's template engine. A software template is a YAML file that defines the UI form a developer fills out in the portal, and the sequence of steps — fetch a skeleton, substitute parameters, publish to GitHub, create a Jira ticket, register in the catalog — that execute when the developer clicks Create.

Here is a production-ready scaffolder template for a Node.js microservice:

```yaml
apiVersion: scaffolder.backstage.io/v1beta3
kind: Template
metadata:
  name: nodejs-microservice
  title: Node.js Microservice
  description: Creates a production-ready Node.js service with CI, Helm chart, and catalog registration.
  tags:
    - nodejs
    - typescript
    - microservice
spec:
  owner: group:platform-team
  type: service

  parameters:
    - title: Service Details
      required:
        - serviceName
        - serviceDescription
        - owner
      properties:
        serviceName:
          title: Service Name
          type: string
          description: Lowercase, hyphen-separated (e.g. payment-service)
          pattern: '^[a-z][a-z0-9-]{2,63}$'
        serviceDescription:
          title: Description
          type: string
          description: One sentence describing what this service does.
        owner:
          title: Owning Team
          type: string
          description: GitHub team that owns this service
          ui:field: OwnerPicker
          ui:options:
            allowedKinds:
              - Group

    - title: Infrastructure
      properties:
        namespace:
          title: Kubernetes Namespace
          type: string
          enum:
            - production
            - staging
          default: staging
        needsDatabase:
          title: PostgreSQL Database
          type: boolean
          default: false

  steps:
    - id: fetch-skeleton
      name: Fetch Skeleton
      action: fetch:template
      input:
        url: ./skeleton
        values:
          serviceName: ${{ parameters.serviceName }}
          serviceDescription: ${{ parameters.serviceDescription }}
          owner: ${{ parameters.owner }}
          namespace: ${{ parameters.namespace }}

    - id: publish
      name: Create GitHub Repository
      action: publish:github
      input:
        allowedHosts: ['github.com']
        description: ${{ parameters.serviceDescription }}
        repoUrl: github.com?repo=${{ parameters.serviceName }}&owner=acme-corp
        defaultBranch: main
        repoVisibility: private
        gitAuthorName: backstage-bot
        gitAuthorEmail: backstage-bot@acme.com
        topics:
          - microservice
          - ${{ parameters.owner }}

    - id: register
      name: Register in Catalog
      action: catalog:register
      input:
        repoContentsUrl: ${{ steps['publish'].output.repoContentsUrl }}
        catalogInfoPath: /catalog-info.yaml

  output:
    links:
      - title: Repository
        url: ${{ steps['publish'].output.remoteUrl }}
      - title: Open in Catalog
        entityRef: ${{ steps['register'].output.entityRef }}
```

When a developer fills out this form and clicks Create, Backstage fetches the `./skeleton` directory (a template tree with `${{ values.serviceName }}` placeholders throughout), substitutes the developer's values, publishes the result to a new GitHub repository under `acme-corp`, and registers the service in the catalog. The developer never touches GitHub settings, never writes a Dockerfile, and never sets up a CI workflow — all of that lives in the skeleton.

The skeleton directory structure that the template fetches typically looks like:

```
skeleton/
  .github/
    workflows/
      deploy.yml          # Calls the platform reusable workflow
  helm/
    Chart.yaml
    values.yaml           # Pre-wired for the golden-path Helm chart
  src/
    index.ts
    health.ts
  Dockerfile
  catalog-info.yaml       # Pre-populated with ${{ values.serviceName }}
  mkdocs.yml              # For TechDocs auto-build
  docs/
    index.md
```

Each file in the skeleton uses Nunjucks templating. The `catalog-info.yaml` in the skeleton, for example, becomes:

```yaml
apiVersion: backstage.io/v1alpha1
kind: Component
metadata:
  name: ${{ values.serviceName }}
  description: ${{ values.serviceDescription }}
  annotations:
    github.com/project-slug: acme-corp/${{ values.serviceName }}
    backstage.io/techdocs-ref: dir:.
spec:
  type: service
  lifecycle: production
  owner: group:${{ values.owner }}
```

After substitution and publication, the developer has a repository with a working Node.js service, a GitHub Actions workflow that calls the platform's reusable deploy workflow, Kubernetes manifests under `helm/`, and a fully populated catalog entry — all without writing a single line of infrastructure YAML.

### TechDocs Integration

TechDocs is Backstage's documentation-as-code plugin. It reads `mkdocs.yml` from each service repository, builds the static site using MkDocs, and serves it inside the portal alongside the catalog entry. Developers write docs in Markdown, commit them alongside code, and the portal builds and hosts the site automatically.

The `mkdocs.yml` in the skeleton is minimal:

```yaml
site_name: ${{ values.serviceName }}
nav:
  - Home: index.md
  - Architecture: architecture.md
  - Runbook: runbook.md
plugins:
  - techdocs-core
```

On every push to `main`, a GitHub Actions step (included in the skeleton) runs `techdocs-cli generate` and pushes the built site to the configured TechDocs storage bucket. The portal serves the site from there. No separate Confluence page, no stale wiki — the documentation lives where the code lives and is rebuilt on every commit.

### The Backstage Plugin Ecosystem

The plugin ecosystem is where Backstage earns its integration value. At the time of writing, the CNCF ecosystem lists over 100 community and vendor-maintained plugins. The ones worth deploying first at a mid-sized company:

**Kubernetes plugin** — shows live pod status, recent deployments, and resource consumption for each catalog service. The plugin queries the Kubernetes API using the service label selectors from the catalog entry. When a developer views their service in the portal, they see the current deployment status without switching to `kubectl` or a separate cluster dashboard.

**PagerDuty plugin** — surfaces the current on-call engineer, open incidents, and recent alert history on each service page. Requires the `pagerduty.com/service-id` annotation in the catalog entry. During an incident, a responder can page the right team from inside the portal.

**Grafana plugin** — embeds live dashboard panels in catalog pages. Requires the `grafana/dashboard-selector` annotation. On-call engineers get their key metrics visible in one place alongside ownership and runbook links.

**GitHub Actions plugin** — shows the status of recent CI runs and deployment workflows for each service. Helps developers track whether their last push is building, failing, or waiting for approval — without leaving the portal.

**Lighthouse plugin** — runs periodic Google Lighthouse audits on frontend services and tracks performance scores over time inside the portal.

The cost of each plugin is two things: the initial configuration time (typically half a day) and the ongoing maintenance overhead if the plugin falls behind upstream Backstage versions. Running a plugin audit quarterly — checking that each plugin is compatible with the installed Backstage version — is a non-negotiable part of platform maintenance.

---

## Self-Service Infrastructure: Crossplane Deep Dive

The service catalog and scaffolder reduce the friction of starting a service. Self-service infrastructure eliminates the ticket queue for ongoing resource requests. The two dominant approaches are Kubernetes-native (Crossplane) and portal-driven (Port self-service actions).

### Crossplane XRDs and Compositions: The Full Picture

Crossplane extends Kubernetes with Custom Resource Definitions that represent cloud infrastructure. A `CompositeResourceDefinition` (XRD) defines the developer-facing schema — the fields a developer fills in when making a resource request. A `Composition` defines the mapping from that schema to actual provider resources (RDS instances, security groups, parameter groups in AWS; equivalent resources in GCP or Azure).

The developer never writes provider YAML. They write a claim that matches the developer-facing schema. Crossplane's controller resolves the claim against the Composition and creates the real infrastructure.

Here is a complete XRD for a self-service PostgreSQL database:

```yaml
# XRD — the developer-facing API schema
apiVersion: apiextensions.crossplane.io/v1
kind: CompositeResourceDefinition
metadata:
  name: xpostgresqlinstances.platform.acme.com
spec:
  group: platform.acme.com
  names:
    kind: XPostgreSQLInstance
    plural: xpostgresqlinstances
  claimNames:
    kind: PostgreSQLInstance
    plural: postgresqlinstances
  versions:
    - name: v1alpha1
      served: true
      referenceable: true
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              properties:
                parameters:
                  type: object
                  properties:
                    storageGB:
                      type: integer
                      minimum: 20
                      maximum: 500
                    instanceClass:
                      type: string
                      enum:
                        - db.t3.micro
                        - db.t3.medium
                        - db.r5.large
                    team:
                      type: string
                  required:
                    - storageGB
                    - instanceClass
                    - team
```

The matching Composition maps that schema to an AWS RDS instance, a subnet group, and a parameter group:

```yaml
apiVersion: apiextensions.crossplane.io/v1
kind: Composition
metadata:
  name: postgresql-aws
  labels:
    crossplane.io/xrd: xpostgresqlinstances.platform.acme.com
    provider: aws
spec:
  compositeTypeRef:
    apiVersion: platform.acme.com/v1alpha1
    kind: XPostgreSQLInstance
  resources:
    - name: rds-instance
      base:
        apiVersion: rds.aws.upbound.io/v1beta1
        kind: Instance
        spec:
          forProvider:
            region: us-east-1
            engine: postgres
            engineVersion: "15.4"
            autoMinorVersionUpgrade: true
            backupRetentionPeriod: 7
            deletionProtection: true
            storageEncrypted: true
            multiAz: false
            publiclyAccessible: false
            dbSubnetGroupNameSelector:
              matchLabels:
                platform.acme.com/subnet-group: private
      patches:
        - type: FromCompositeFieldPath
          fromFieldPath: spec.parameters.storageGB
          toFieldPath: spec.forProvider.allocatedStorage
        - type: FromCompositeFieldPath
          fromFieldPath: spec.parameters.instanceClass
          toFieldPath: spec.forProvider.instanceClass
        - type: FromCompositeFieldPath
          fromFieldPath: spec.parameters.team
          toFieldPath: metadata.labels["acme.com/team"]
        - type: ToCompositeFieldPath
          fromFieldPath: status.atProvider.address
          toFieldPath: status.address

    - name: connection-secret
      base:
        apiVersion: rds.aws.upbound.io/v1beta1
        kind: InstanceRootPassword
        spec:
          forProvider:
            instanceIdentifierSelector:
              matchControllerRef: true
          writeConnectionSecretToRef:
            namespace: crossplane-system
```

A product engineer files a `PostgreSQLInstance` claim in their namespace:

```yaml
apiVersion: platform.acme.com/v1alpha1
kind: PostgreSQLInstance
metadata:
  name: payment-db
  namespace: payments-team
spec:
  parameters:
    storageGB: 50
    instanceClass: db.t3.medium
    team: payments-team
  writeConnectionSecretToRef:
    name: payment-db-credentials
```

Crossplane sees this claim, triggers the Composition, creates the RDS instance in the platform AWS account, and writes the connection credentials into a Kubernetes Secret named `payment-db-credentials` in the `payments-team` namespace. The engineer never touches the AWS console, never opens a ticket, and gets a credential secret she can reference directly in her Deployment manifest. The platform team has enforced backup retention, encryption at rest, deletion protection, and subnet placement in the Composition — none of which the product engineer needs to be aware of.

![The four abstraction layers of an Internal Developer Platform, from raw cloud infrastructure up to the self-service developer layer](/imgs/blogs/platform-engineering-and-the-internal-developer-platform-4.png)

The Composition also enforces the `acme.com/team` label on the RDS instance. The cost allocation system reads this label to produce per-team cloud spend reports — another compliance requirement that the platform team handles once in the Composition rather than asking every developer to remember it.

### Port Self-Service Actions

Port takes a different approach: instead of extending Kubernetes, Port lets you define self-service actions in the portal that execute backend webhooks or automation. A "Create Database" action in Port triggers a webhook to a Lambda or a GitHub Actions workflow that runs Terraform, then updates the Port catalog with the new resource entity.

Port self-service actions are easier to set up for teams without Kubernetes expertise, and they work equally well for non-Kubernetes infrastructure. The tradeoff is that the automation sits outside Kubernetes, so you lose the reconciliation loop that Crossplane provides — if someone manually modifies the RDS instance in the console, Crossplane will revert it; Port will not.

---

## The Golden Path: Making the Right Way the Easy Way

The golden path is the central idea in platform engineering, and it is worth spending time on exactly what it means in practice.

A golden path is not a mandate. It is not a governance policy that says "thou shalt use this CI pipeline." It is a designed experience that makes the recommended approach so much faster, safer, and better-documented than alternatives that developers choose it without being forced.

Spotify's original framing was "paved roads." A paved road is not a required route. You can walk off-road. But you would have to choose to leave the smooth, well-lit, signposted path for uncertain terrain. Most engineers, most of the time, will take the paved road.

The practical implication is that the golden path must be genuinely good. If the golden path is slow, underdocumented, or inflexible, engineers will route around it. Every routing-around creates a new snowflake to maintain. The platform team's primary job is to make the golden path good enough that routing around it is the exception, not the norm.

### What "Paved Road" Means Operationally

Put plainly: a paved road is a set of defaults so well-chosen that developers never need to override them for standard cases. It has three layers.

**Layer 1: Mandatory guardrails.** These are non-negotiable. Every service on the golden path must use TLS for all external traffic. Every service must expose a `/health` endpoint with a defined format. Every service must emit structured JSON logs to stdout. Every container image must be scanned for HIGH and CRITICAL CVEs before it reaches production. These are implemented as admission controllers, CI gates, and Kubernetes policies (OPA/Gatekeeper) that enforce the requirements automatically — developers cannot accidentally skip them.

**Layer 2: Opinionated defaults.** These are strongly recommended but overridable. The base Docker image is `node:20-alpine`. The default memory limit is 512Mi. The log retention period is 30 days. The CI pipeline runs unit tests, a lint check, and a SAST scan. A developer who needs to override a default does so in their `values.yaml` or in the three-line calling workflow, not by forking the platform's reusable workflow.

**Layer 3: Optional extras.** These are capabilities the platform provides that services can opt into: distributed tracing via OpenTelemetry, feature flags via Unleash, structured rate limiting via Envoy. They are pre-configured and documented in the portal; adopting them is a one-line addition to the values file.

The critical boundary is between Layer 1 and Layer 2. If you put too many requirements in Layer 1, engineers feel constrained and the off-ramp request queue fills up. If you put too few, the mandatory baseline provides insufficient security and compliance coverage. A good heuristic: Layer 1 should contain only things whose absence has caused a production security incident or regulatory finding at your company.

### Evolving the Golden Path Without Breaking Users

The golden path is not static. Languages deprecate. Container runtimes update. Security policies tighten. The challenge is evolving the path without disrupting the 40 services that are already using it.

The pattern that works is version-pinned references with deprecation windows. The reusable workflow is versioned:

```yaml
jobs:
  deploy:
    uses: acme-corp/platform-actions/.github/workflows/deploy-service.yml@v2
```

When the platform team ships `v3`, they announce it in the platform newsletter, update the scaffolder template to generate `@v3` references for new services, and set a deprecation date for `@v2` (typically 90 days). During the window, `@v2` continues to work. After the window, `@v2` calls emit a warning but still succeed for another 30 days, then hard-fail. Services that have not migrated are flagged in the Backstage scorecard as "golden path: outdated."

The scorecard system (a Backstage plugin or a Port scoring rule) is what makes this tractable at scale. Instead of tracking 40 services in a spreadsheet, the platform team defines the criterion ("uses platform-actions >= v3") as a catalog fact. The portal shows a compliance dashboard: 32 of 40 services are on v3, 8 are on v2, 3 have a documented exception. The platform team can see at a glance where follow-up is needed without manually auditing each repository.

### Reusable GitHub Actions Workflow

Here is the reusable workflow that is the heart of the golden-path CI/CD pipeline at a mid-sized company. Product teams call this workflow from their own pipeline YAML; they do not copy any of the build or deploy logic.

```yaml
# .github/workflows/deploy-service.yml (in the platform-actions repository)
name: Platform Deploy

on:
  workflow_call:
    inputs:
      service_name:
        required: true
        type: string
      environment:
        required: true
        type: string
        default: staging
      dockerfile_path:
        required: false
        type: string
        default: Dockerfile
    secrets:
      REGISTRY_TOKEN:
        required: true

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
      id-token: write    # for OIDC to ECR or GCR
    outputs:
      image_tag: ${{ steps.meta.outputs.tags }}
      image_digest: ${{ steps.build.outputs.digest }}

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.REGISTRY_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/acme-corp/${{ inputs.service_name }}
          tags: |
            type=sha,prefix=,format=short
            type=ref,event=branch

      - name: Build and push
        id: build
        uses: docker/build-push-action@v5
        with:
          context: .
          file: ${{ inputs.dockerfile_path }}
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          provenance: true
          sbom: true

      - name: Run Trivy vulnerability scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ghcr.io/acme-corp/${{ inputs.service_name }}:${{ github.sha }}
          severity: HIGH,CRITICAL
          exit-code: 1

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    environment: ${{ inputs.environment }}

    steps:
      - uses: actions/checkout@v4

      - name: Configure kubectl
        uses: azure/setup-kubectl@v3

      - name: Deploy via Helm
        run: |
          helm upgrade --install ${{ inputs.service_name }} \
            oci://ghcr.io/acme-corp/charts/microservice \
            --namespace ${{ inputs.environment }} \
            --set image.tag=${{ needs.build-and-push.outputs.image_tag }} \
            --set service.name=${{ inputs.service_name }} \
            --atomic \
            --timeout 5m

      - name: Verify rollout
        run: |
          kubectl rollout status deployment/${{ inputs.service_name }} \
            -n ${{ inputs.environment }} \
            --timeout=300s
```

A product team calls this from their own repository with three lines:

```yaml
# In the product team's repository: .github/workflows/deploy.yml
name: Deploy Payment Service
on:
  push:
    branches: [main]

jobs:
  deploy:
    uses: acme-corp/platform-actions/.github/workflows/deploy-service.yml@main
    with:
      service_name: payment-service
      environment: staging
    secrets:
      REGISTRY_TOKEN: ${{ secrets.REGISTRY_TOKEN }}
```

The product team has no build logic, no vulnerability scanning config, no Helm release logic. They maintain three lines. The platform team maintains the reusable workflow. When the platform team upgrades the base Docker image, bumps the Trivy severity threshold, or changes the registry, the change propagates to every service on the next push. This is the golden-path leverage ratio at work.

![Backstage scaffolder end-to-end flow, from developer triggering a template to receiving a deploy URL after repo creation, CI wiring, and infra provisioning](/imgs/blogs/platform-engineering-and-the-internal-developer-platform-5.png)

---

## Platform-as-Product: The Organizational Model That Makes It Work

The most common reason platform engineering initiatives fail is that the platform team thinks of itself as an infrastructure team. Infrastructure teams respond to tickets. They measure themselves by uptime and ticket throughput. They learn about developer needs through escalations.

A platform team that behaves like an infrastructure team will build tools that developers tolerate rather than love. Tolerating a tool is the first step toward building a replacement.

The platform team needs to behave like a product team. Product teams have users. They talk to users. They have a roadmap. They measure user satisfaction. They have on-call. They treat reliability as a product feature.

### Staffing the Platform Team

The canonical question is: what is the right ratio of platform engineers to product engineers?

The range seen in practice is 1:15 to 1:30. A team of 3 platform engineers can sustainably support 45–90 product engineers, provided the platform is built as a product (not a set of one-off scripts) and the ticket intake is genuinely replaced by self-service. Below 1:30, platform engineers are stretched too thin to do proactive investment alongside reactive support. Above 1:15, the platform team is overstaffed relative to the leverage available — the tools built by a 10-person platform team supporting 80 engineers have diminishing returns beyond a certain point.

The composition of a three-person platform team matters as much as its size. The most effective configuration is:

- **One person with deep infrastructure expertise** (Kubernetes, Terraform, cloud networking, Crossplane). This person builds and operates the self-service infrastructure layer. They understand the blast radius of Composition changes and can debug RDS provisioning failures.
- **One person with developer tooling and developer experience focus** (CI/CD, Backstage, GitHub Actions, scripting). This person owns the scaffolder templates, the golden-path workflow, and the portal. They are the primary user-researcher — they sit with developers and watch them deploy.
- **One person who can bridge both worlds** (technically broad, strong communicator). This person owns the platform roadmap, runs the quarterly DevSat survey, handles RFC reviews, and is the face of the platform team in engineering all-hands. In small teams this person often carries the title "platform lead" or "staff platform engineer."

All three share on-call for the platform. The rotation gives each person visibility into how the platform behaves under pressure and what developers reach out about during incidents.

### The Relationship to DevOps and SRE

Platform engineering, DevOps, and SRE are three practices that overlap in a way that confuses many teams.

DevOps is a philosophy — "you build it, you run it" — about ownership and culture. It is not a team structure. The platform team's existence does not conflict with DevOps; the platform team makes DevOps more tractable by removing the overhead that would otherwise make "you build it, you run it" too expensive per team.

SRE is a practice for managing production reliability with engineering discipline — error budgets, SLOs, toil reduction, postmortems. An SRE team typically owns the reliability of platform services and the reliability practices used by product teams. The platform team and the SRE team often overlap in tools (both care about observability, alerting, incident management) but have different scopes. The SRE team owns production reliability; the platform team owns developer productivity. In smaller organisations these teams are the same people wearing different hats on different days.

In Team Topologies language, the platform team is a "platform team" (obviously), and it maintains an "X-as-a-Service" interface with stream-aligned product teams. The SRE team, if separate, is a "complicated-subsystem team" for production reliability. The relationship is constructive when both teams share the DORA metrics as a common north star.

### The Inner Source Model

As the platform matures, product teams accumulate opinions about what should change. A payments engineer discovers that the golden-path Dockerfile build does not cache the `node_modules` directory correctly for monorepos. A backend engineer has a Grafana dashboard template that should be part of the platform. A security engineer wants to add a secrets detection pre-commit hook to the skeleton.

The inner source model handles this: the platform repositories are open to all engineers to read, and product engineers can file pull requests against platform repositories. Platform team members review these PRs and act as maintainers. The product engineer contributes; the platform team reviews for architectural consistency and cross-cutting impact.

The mechanics are the same as open source: a `CONTRIBUTING.md` in the platform repository, a review SLA (typically 5 business days), and a clear decision process for rejections (the platform team writes a short explanation of why the contribution does not fit the current direction, with suggestions for alternatives). The culture it creates is ownership: product engineers feel partial ownership of the platform, which increases adoption and reduces the adversarial dynamic that sometimes develops between product and platform teams.

### User Research for Platform Teams

The platform team's users are developers. The most valuable thing a platform team can do in its first month is sit with developers and watch them deploy a service. Not ask them to demo. Watch them work. Note every context switch, every Slack message to ask "what's the right command here?", every 15-minute detour to find the right documentation.

The patterns that emerge from five of these observation sessions will define your first six months of roadmap work more accurately than any survey.

Standard user-research artifacts for platform teams:

- **Developer journey map:** The sequence of steps a developer takes from "new service idea" to "first deploy to production." Map every decision point, every tool switch, every waiting period.
- **Jobs to be done:** What are developers actually trying to accomplish? "I need to deploy my service" is not the job. "I need to validate that my change is safe to ship to the 50,000 users who depend on this service" is the job.
- **Developer satisfaction score (DevSat):** A quarterly survey, four questions, measuring satisfaction with deploy speed, observability, documentation quality, and platform reliability. Track the score over time. Make it public.

### Platform SLOs

The platform is a product, so the platform has SLOs. Some examples from real platform teams:

| SLO | Target |
|---|---|
| CI pipeline p95 build time | \< 8 minutes |
| Scaffolder template success rate | \> 99.5% |
| Service catalog staleness | \< 24 hours |
| Self-service infra provisioning time | \< 15 minutes |
| Platform portal uptime | \> 99.9% |
| Time-to-first-deploy for new service | \< 4 hours |

These SLOs are published in the developer portal alongside the product team SLOs. When the CI pipeline starts running slow, the platform team is paged. When a scaffolder template fails at a rate above 0.5%, the platform team investigates before developers file tickets.

### The Platform Roadmap

A healthy platform roadmap has the same structure as any product roadmap: now, next, later. The "now" column contains work that has been scoped, assigned, and has acceptance criteria. The "next" column is validated by user research and roughly sized. The "later" column is a holding area for ideas and requests not yet validated.

The critical difference from an infrastructure ticket queue: the platform roadmap is driven by developer impact, not by ticket volume. Ten tickets asking for SSH access to production Kubernetes nodes is a signal that developers lack confidence in their observability tooling, not a signal to open SSH access. The platform roadmap addresses root causes.

---

## Measuring Developer Experience: SPACE and DORA

Platform engineering exists to improve developer experience. To know whether it is working, you need to measure developer experience. There are two complementary frameworks: DORA for delivery outcomes, and SPACE for the human experience of development.

### DORA Metrics

The DORA research program (originally at Google, now part of DORA at DevOps Research and Assessment) identifies four metrics that predict software delivery performance:

**Deployment Frequency.** How often your organisation deploys to production. Elite performers deploy multiple times per day. The IDP improves deployment frequency by removing friction from the deploy path — a developer who can deploy in 10 minutes deploys more often than one who deploys in 4 hours.

**Lead Time for Changes.** The time from a commit being merged to that commit running in production. Elite performers achieve less than one hour. Self-service templates, golden-path workflows, and automated testing pipelines all reduce lead time.

**Change Failure Rate.** The percentage of deployments that cause a production incident. Elite performers are below 5%. The golden path includes automated testing, security scanning, and rollback mechanisms that reduce change failure rate.

**Time to Restore Service.** How long it takes to recover from a production incident. Elite performers restore in less than one hour. The service catalog (who owns this? where is the runbook?) and observability integration (dashboards pre-linked in the catalog) directly reduce restoration time.

DORA metrics are lagging indicators. They tell you what happened. SPACE adds leading indicators: the human experience that predicts future DORA outcomes.

### SPACE Framework

The SPACE framework (Microsoft Research, 2021) defines five dimensions of developer productivity:

- **Satisfaction:** How satisfied are developers with their tools, processes, and work environment? Measured by DevSat surveys.
- **Performance:** Does the developer's work achieve the outcomes it was meant to achieve? Measured by production incident rate, code review turnaround.
- **Activity:** Volume of work completed. Pull requests merged, deploys shipped. Useful as a comparative signal but misleading in isolation.
- **Communication:** How effectively do developers share knowledge and coordinate? Measured by code review latency, documentation quality scores.
- **Efficiency:** How few interruptions and context switches happen during focused work? Measured by flow time (uninterrupted coding sessions > 2 hours), meeting load.

The platform team's primary SPACE levers are Satisfaction and Efficiency. Self-service infrastructure reduces context switches (no ticket queues). Golden-path templates reduce cognitive load during new service creation. TechDocs integration reduces documentation-hunting interruptions.

### How Spotify Measures Developer Experience

Spotify's developer experience team (the team responsible for Backstage internally, distinct from the engineering teams that use it) tracks a composite metric called the Developer Experience Index. It combines four components, each measured quarterly via a short survey:

1. **Ease of deployment:** "How easy is it to get your code to production?"
2. **Discoverability:** "When you need to understand a system you don't own, how easy is it to find what you need?"
3. **Cognitive load:** "How much mental overhead does your daily work involve that is not core to the product problems you are solving?"
4. **Tool reliability:** "How reliably do the internal tools you use work?"

Each is scored 1–5. The index is the average across all four. Spotify publishes the index to all engineers quarterly and uses it to prioritise the Backstage team's roadmap. When discoverability drops after a major re-org that shuffled service ownership, the team ships a catalog ownership audit tool. When cognitive load spikes after a Kubernetes version upgrade that changed required annotation formats, the team ships a migration guide with automated PR generation.

The measurement practice itself is a form of trust-building. Developers who see their feedback reflected in the platform roadmap within one or two quarters start to believe that the platform team is working for them rather than maintaining a system they have to tolerate.

### Cognitive Load as a Platform KPI

The concept of cognitive load — how much mental overhead a task requires — is the connecting tissue between the SPACE framework's Efficiency dimension and the concrete IDP investments that reduce it.

Each piece of accidental complexity a developer carries adds to cognitive load. Knowing which kubectl context maps to which environment. Remembering the incantation to rotate a secret. Understanding which Helm chart version is current. Figuring out which squad owns an unfamiliar service during an incident.

The IDP reduces each of these:

- The catalog replaces "remember who owns X" with a two-second lookup.
- The scaffolder template replaces "remember how to set up a new service" with a form fill.
- The golden-path workflow replaces "remember the correct build and deploy sequence" with a three-line calling workflow.
- The Crossplane XRD replaces "open a ticket and wait eight days" with a ten-line YAML file.

Measuring cognitive load reduction directly is hard. A practical proxy is time-on-task: how long does it take a developer to complete a specific task (deploy a new service, find who owns a failing service, provision a database)? Benchmark these tasks before and after each IDP investment. The reduction in time-on-task is a direct measure of the cognitive load the IDP has absorbed.

#### Worked example: Three weeks to four hours — the mechanics

The 3-week-to-4-hour headline requires unpacking. In concrete terms: what took three weeks, and what makes four hours achievable?

Here is the breakdown of a new service creation before the IDP, measured across eight projects at a company that later built a platform:

| Step | Median time |
|---|---|
| Create GitHub repository with correct settings | 45 minutes (manual) |
| Write Dockerfile | 3 hours |
| Set up CI pipeline | 6 hours |
| Write Kubernetes manifests | 4 hours |
| Request database ticket | Filed day 1 |
| Database ticket resolved | Day 8 (median queue time) |
| Configure secrets management | 3 hours |
| Write observability wiring | 2 hours |
| Register service in monitoring system | 1 hour |
| First deploy to staging | Day 10–12 (waiting on database) |
| Security review | Day 14–16 |
| First deploy to production | Day 20–22 |

The three weeks is not entirely wasted time. Much of it is waiting: waiting for database tickets, waiting for security reviews, waiting for a senior engineer to review infrastructure YAML. But waiting is still a three-week delay.

After the IDP:

| Step | Time |
|---|---|
| Fill scaffolder template form | 5 minutes |
| Backstage creates repo, CI, Helm chart, catalog entry | 3 minutes (automated) |
| File Crossplane PostgreSQLInstance claim | 10 minutes |
| Crossplane provisions RDS instance | 8 minutes |
| First CI run completes | 6 minutes |
| First deploy to staging | 30 minutes from template completion |
| Security scan result (integrated in CI) | Same CI run |
| First deploy to production | 2–4 hours (includes review of the first staging run) |

The four hours is real if the security review is replaced by automated scanning in the golden-path CI (which a mature platform does), and if the database provisioning is self-service (which Crossplane provides). The platform team encoded three weeks of sequential human decisions into a 30-minute automated sequence.

---

## War Story: The Sixty-Engineer Company That Built It Right

A B2B SaaS company — call them Meridian — grew from 15 to 55 engineers in 18 months. At 15 engineers, everyone knew where everything lived. At 30 engineers, things started slipping: two incidents where the on-call engineer took 20 minutes to find who owned the failing service. At 45 engineers, their infrastructure team (three people) was drowning in "can you create a database for us?" tickets. Their mean ticket resolution time was 11 days. One sprint the infrastructure team resolved zero product work because they spent the entire sprint on infrastructure tickets.

The CTO hired a fourth infrastructure engineer specifically to lead platform work. Her first action was not to install Backstage. It was to spend two weeks doing what she called a "cognitive load audit." She sat with engineers from six different squads for two hours each and asked them to walk her through deploying their most recent change. She found:

1. No two squads had the same CI pipeline structure. One squad had Dockerfile build caching; the others did not. Build times varied from 4 to 22 minutes for similar-sized services.
2. Three squads did not know the correct kubectl context to use for staging. They had been deploying to production by accident, correcting with a rollback, and treating this as a normal cost of deployment.
3. Every squad had its own approach to secrets: one used AWS Secrets Manager directly, one used Kubernetes Secrets (unencrypted), one had hardcoded secrets in environment variables checked into git (found during the audit; rotated immediately).
4. Zero squads had a service catalog entry. When she asked "who owns the notification service?" the room went quiet for 15 seconds before someone said "I think it's Priya's team, but she left two months ago."

Armed with this data, she wrote a one-page proposal: three months to build the foundation, six months to reach self-service. Leadership approved.

**Month 1–3: The Catalog and the Standard.** She deployed Backstage, wrote `catalog-info.yaml` files for all 47 services (most of them herself, filling in ownership from git blame and Slack archaeology), and defined the golden path: a standardised Dockerfile, a reusable GitHub Actions workflow, and a Helm chart values template. No mandates yet. She published the catalog at `platform.meridian.internal` and sent one Slack message: "You can now see who owns every service at this URL."

Usage was immediate. The catalog link appeared in post-mortem documents, Slack incident channels, and onboarding docs within a week.

**Month 4–6: The Scaffolder.** She built one scaffolder template: the Node.js microservice. It took three weeks of iteration — collecting developer feedback on the form, adjusting the skeleton to match the actual deployment environment, debugging Backstage scaffolder YAML that silently failed on edge cases. The fourth week she announced it in the engineering all-hands. "New service in 4 hours, first deploy included. Here's the demo."

Nine teams used the template in the first month. Eleven new services were created that quarter versus three in the previous quarter. The increased service creation rate was not the goal — reduced friction was — but it was a visible signal that something had changed.

**Month 7–9: Self-Service Databases.** Using Crossplane, she created XRDs for PostgreSQL instances and Redis clusters. The infrastructure ticket queue fell from 47 open tickets to 8. Mean resolution time for the remaining tickets (edge cases, large instances, cross-account networking) dropped to 2 days because the team was no longer buried in routine provisioning.

**Month 10–12: Metrics and SLOs.** She added the DORA metrics dashboard to Backstage, wired to GitHub Actions and their incident tracking tool. Leadership could now see deployment frequency (moved from 3/week to 12/week across the engineering organisation), lead time (from 4 days to 6 hours), and change failure rate (down from 14% to 5%).

At the end of the year, Meridian's CTO was asked in an interview what had improved developer productivity most in the past 12 months. He said the migration from RDS to Aurora. But the platform engineer's engineering colleagues told a different story. The most common phrase in the quarterly developer satisfaction survey was "deploying is not a chore anymore."

#### Worked example: A platform team of three supporting eighty engineers

The economics of this configuration are worth examining in detail, because the standard objection to platform investment is "we can't afford to have three engineers not shipping product features."

**The cost.** A three-person platform team at a mid-market tech company has a fully-loaded annual cost of approximately \$600,000–\$750,000 (salaries, benefits, overhead). Call it \$700,000/year.

**What the platform provides to 80 product engineers.**

First, time saved on new service creation. At Meridian, the baseline was 3 weeks per new service. After the IDP: 4 hours. Net time saved: approximately 116 hours per service. At 12 new services per year, that is 1,392 hours — roughly 174 engineer-days. At an average fully-loaded cost of \$225,000/year (\$108/hour), the value recovered is approximately \$150,000/year.

Second, time saved on incident ownership lookup. At Meridian's pre-IDP baseline, 20 minutes per incident to identify ownership. Post-IDP: 30 seconds. Net: 19.5 minutes per incident. At 200 incidents per year, that is 65 hours saved. Value: approximately \$7,000/year. Small in dollar terms but significant for MTTR.

Third, time saved on infrastructure provisioning. Pre-IDP: 8-day median ticket wait per request, with approximately 60 requests per year. At 8 engineering-hours of actual work per request (waiting, re-filing, follow-up), that is 480 hours per year across the organisation. Post-IDP: 30 minutes per self-service claim. Net: 450 hours saved. Value: approximately \$48,500/year.

Fourth, reduced CI/CD fragmentation maintenance. Pre-IDP: 6 distinct CI/CD systems requiring maintenance across squads. Estimated 40 hours per system per year in updates, security patches, version bumps. Total: 240 hours. Post-IDP: one golden-path workflow maintained by the platform team, zero per-squad maintenance. Value recovered: approximately \$26,000/year.

**Total directly measurable return: approximately \$231,500/year against a \$700,000 investment.**

The numbers do not pencil out in year one on direct productivity savings alone. The real return is in the unmeasured factors: engineer satisfaction (lower attrition; replacing one senior engineer costs \$100,000–\$200,000 in recruiting and ramp-up alone), faster feature shipping velocity (more deploys per day), and the compound effect of a consistent security and compliance baseline replacing six bespoke systems.

The ROI calculation is not the right frame for year one. The right frame is: at 80 engineers and growing, is the cost of not having a platform — in fragmentation, cognitive load, and security inconsistency — greater than the cost of the platform team? At Meridian, the infrastructure team was already spending one entire sprint per quarter on routine provisioning tickets. That is one person-quarter per year consumed by work that should not exist. The platform team replaces that cost and creates new leverage.

---

## Measuring Developer Experience: SPACE and DORA

Platform engineering exists to improve developer experience. To know whether it is working, you need to measure developer experience. The DORA and SPACE frameworks are covered above. The key point for operational practice is this: measure both, publish both, and review both in your quarterly roadmap planning.

The time-to-first-deploy KPI is the single most important leading indicator of platform health. It is directly actionable, directly attributable to platform investment, and directly visible to engineering leadership. If time-to-first-deploy rises (because a scaffolder template broke, because the Crossplane composition started timing out, because a new required approval step was added), the platform team hears about it immediately.

The developer satisfaction score (DevSat) is the most important lagging indicator. It captures the cumulative effect of all platform investments on engineer experience. A rising DevSat score, tracked quarterly alongside DORA metrics, is the argument for continued platform investment that survives any single-quarter productivity dip.

![Platform team maturity journey from month zero (golden path defined) through month twelve (full self-service delivery with DORA metrics)](/imgs/blogs/platform-engineering-and-the-internal-developer-platform-6.png)

---

## Build vs Buy: Backstage, Port, Cortex, and OpsLevel

The IDP tooling landscape has fragmented rapidly since Spotify open-sourced Backstage in 2020. You have a genuine choice between building on an open-source framework and buying a SaaS product. Here is how to think about the decision.

![IDP platform comparison matrix across Backstage, Port, Cortex, and OpsLevel on catalog, templates, infra provisioning, self-hosted, and price dimensions](/imgs/blogs/platform-engineering-and-the-internal-developer-platform-3.png)

### Backstage

Backstage is the most flexible option and the most work. You are not buying a product; you are adopting a framework and building on it. The framework provides the plugin system, the catalog schema, the scaffolder engine, and the authentication layer. Everything else — plugins for your CI/CD provider, your incident management tool, your cloud cost dashboard — is either a community plugin you find, adapt, and maintain, or a custom plugin you write.

The plugin ecosystem is large (the CNCF ecosystem page lists over 100 plugins), but quality varies. Some plugins are actively maintained by vendors (Datadog, PagerDuty, GitHub, Confluence). Others are community-maintained and go stale. Part of running Backstage is curating and maintaining your plugin set.

The development overhead is real. A minimal production Backstage deployment requires:
- A Node.js application server (run in Kubernetes or on a managed service)
- A PostgreSQL database for catalog state
- A GitHub/GitLab integration for catalog discovery
- An authentication provider (OAuth, SAML, LDAP)
- A CI/CD pipeline for Backstage itself (yes, your developer portal needs a deploy pipeline)

Most companies allocate 0.5–1 platform engineer full-time to Backstage maintenance. The investment is worth it if you want full customisation and cannot share company data with a SaaS vendor.

### Port

Port is the most feature-complete SaaS IDP option. It offers a service catalog, software templates (called "self-service actions"), infrastructure blueprints, and a scorecard system for measuring service health. Port has a generous free tier and charges per engineer per month beyond the free limit.

Port's key advantage over Backstage is time-to-value. A basic Port catalog is operational within hours, not days. The self-service actions are configured in Port's UI without writing YAML templates from scratch. The scorecard system (tracking which services have a production runbook, have a defined owner, pass security scans) is built-in and takes minutes to configure versus days in Backstage.

Port's limitations: no self-hosted option (your service metadata lives in Port's cloud), less extensible than Backstage for deeply custom workflows, and the per-engineer pricing becomes significant at larger companies.

### Cortex and OpsLevel

Both Cortex and OpsLevel focus on the catalog-and-scorecard use case with lighter infrastructure provisioning capabilities. They are strong choices if your primary goal is service ownership visibility and engineering standards enforcement, and you do not need the full scaffolder/template capability. Both are SaaS-only. Cortex has a particularly strong integration story with major observability and incident management vendors.

**The decision rule:** If you have fewer than 5 platform engineers and need to show value in under 30 days, start with Port or Cortex. If you have time, a dedicated Backstage maintainer, and hard data-residency requirements, invest in Backstage. If your primary gap is service ownership visibility and standards enforcement (not templates), OpsLevel or Cortex may be sufficient without the broader IDP complexity.

---

#### Worked example: Port self-service action for environment provisioning

At a company without the Kubernetes expertise to run Crossplane, Port self-service actions offer a practical alternative. Here is a Port action definition that lets a developer provision a staging environment by filling out a form in the Port portal:

```json
{
  "identifier": "create_staging_env",
  "title": "Create Staging Environment",
  "description": "Provisions a dedicated staging environment with database and queue",
  "trigger": {
    "type": "self-service",
    "operation": "CREATE",
    "userInputs": {
      "properties": {
        "service_name": {
          "type": "string",
          "title": "Service Name",
          "description": "The service that needs a staging environment"
        },
        "postgres_size": {
          "type": "string",
          "title": "Postgres Instance Size",
          "enum": ["small", "medium", "large"],
          "default": "small"
        },
        "ttl_days": {
          "type": "integer",
          "title": "Environment TTL (days)",
          "description": "Auto-delete after N days",
          "minimum": 1,
          "maximum": 30,
          "default": 7
        }
      },
      "required": ["service_name"]
    }
  },
  "invocationMethod": {
    "type": "GITHUB",
    "org": "acme-corp",
    "repo": "platform-automation",
    "workflow": "provision-staging-env.yml",
    "omitPayload": false,
    "reportWorkflowStatus": true
  }
}
```

The action triggers a GitHub Actions workflow (`provision-staging-env.yml`) that runs Terraform to create the environment, writes the environment URL and credentials to Port as entity properties, and notifies the developer via Slack when provisioning is complete. The developer never touches Terraform, never opens the AWS console, and gets a fully-wired staging environment in 8 minutes.

The `reportWorkflowStatus: true` flag means Port polls the GitHub Actions workflow for status and updates the self-service action run log in the portal in real time. The developer can watch provisioning progress without leaving the portal.

---

## The Off-Ramp: When Teams Need to Leave the Golden Path

A golden path with no off-ramp is a mandate. Mandates create shadow IT. Engineers will go around a platform that forces them into an approach that genuinely does not fit their use case.

Every golden path needs a designed off-ramp: a documented, governed process for requesting an exception to the standard approach.

The off-ramp process at a well-run platform team typically looks like:

1. **Request.** The team files an RFC (Request for Comment) in the platform team's repository, describing the exception they need, why the golden path does not work for their use case, and what they propose to use instead.
2. **Review.** The platform team reviews the RFC within 5 business days. They may suggest a modification to the golden path that accommodates the use case without creating a one-off exception. They may approve the exception with conditions (for example, "you can use custom Dockerfile but you must still use our reusable workflow for the CI pipeline").
3. **Decision.** If approved, the exception is documented in the catalog entry for the service. The owning team commits to maintaining the exception, including updating it when security vulnerabilities are found in the non-standard approach.
4. **Review cadence.** Exceptions are reviewed quarterly. If the original reason for the exception no longer applies (for example, the golden path was extended to cover the use case), the team migrates back.

The off-ramp serves two purposes. It preserves genuine flexibility for edge cases. And it gives the platform team signal about where the golden path needs to evolve. If 30% of teams are requesting the same exception, the exception should become the golden path.

![Deploy fragmentation across six teams without a platform, versus the single golden path with a governed off-ramp that platform engineering produces](/imgs/blogs/platform-engineering-and-the-internal-developer-platform-7.png)

---

## When NOT to Build an Internal Developer Platform

This post has presented the IDP as an unambiguous good. It is not. There are situations where building a platform is the wrong investment. The decision tree is in figure 8.

![Platform investment decision tree: whether to build an IDP depends on team count, platform team existence, and golden-path maturity](/imgs/blogs/platform-engineering-and-the-internal-developer-platform-8.png)

**Fewer than 10 product teams.** Below this threshold, the coordination overhead of a shared platform often exceeds the tax it eliminates. Engineers know each other. Tribal knowledge is sufficient. The catalog adds maintenance burden without enough discovery value. The scaffolder template is maintained by one person who knows all 8 engineers personally. The investment does not pencil out.

**No dedicated platform team.** A platform built as a side project by infrastructure engineers who are also on-call for production and handling tickets will not be maintained. The scaffolder template will have six open bugs. The catalog will be 40% stale. The golden-path workflow will be two major versions behind the CI/CD provider's best practices. Engineers will learn that the platform is unreliable and route around it. You will have spent the investment cost and gotten no adoption. Hire a platform team before building the platform.

**Golden path not yet defined.** The platform encodes the golden path at scale. If the golden path has not been defined, the platform will encode whatever the first platform engineer thought was reasonable. The scaffolder template will scaffold the wrong thing. The reusable workflow will embed the wrong opinions. You will spend the next year fighting engineers who correctly identify that the platform is encoding bad practices. Define the golden path — even as a document — before building the tooling that encodes it.

**The right way is actively contested.** Some organisations have genuine architectural disagreements: should services be written in TypeScript or Go? Should they use Kafka or SQS? Should they deploy to Kubernetes or Lambda? A platform that encodes one answer before leadership has reached a decision will be contested, forked, or ignored. Resolve the architectural disputes before encoding them in scaffolding.

**The team is still discovering the right deployment model.** Early-stage companies and teams in deep architectural transition should not build a platform. The platform is most valuable when you know what "right" looks like and want to make it easy. If "right" is still evolving — you just adopted Kubernetes, you are migrating from monolith to microservices, you are moving from on-prem to cloud — building a platform is premature. Stabilise the architecture first.

---

## Comparison: Platform Engineering vs Traditional Infrastructure Teams

| Dimension | Traditional Infra Team | Platform Engineering Team |
|---|---|---|
| Primary output | Uptime, capacity | Developer productivity, self-service |
| Work intake | Ticket queue | Roadmap driven by user research |
| Success metric | MTTR, uptime % | Deployment frequency, time-to-first-deploy |
| Developer interaction model | On-call escalation | Product feedback loops, office hours |
| Infrastructure changes | Manual or scripted | Self-service via IDP |
| Documentation | Runbooks | TechDocs embedded in developer portal |
| Toil philosophy | Manage it | Automate it via golden path |
| Investment trigger | Production incidents | Scaling friction (10+ teams) |

The distinction is not about the technical skills involved — both teams run Kubernetes, manage cloud accounts, write Terraform. The distinction is about the problem being solved. An infrastructure team solves reliability. A platform team solves developer leverage.

Neither is better in absolute terms. Early-stage companies need reliable infrastructure before they need developer portals. But past a certain scale, an infrastructure team that has not evolved toward platform thinking will become a bottleneck: every developer improvement requires an infrastructure ticket, every new service requires manual onboarding, every incident requires someone to know the tribal knowledge that was never written down.

---

## Cross-Series Context

This post is part of the CI/CD & Cloud-Native Delivery series. The foundational mental model for how a commit travels to production is in [From Commit to Production: The CI/CD Mental Model](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model). The reusable workflow in this post uses GitHub-hosted and self-hosted runners; the economics and caching strategies behind runner infrastructure are covered in [Runners, Caching, and the CI Cost Problem](/blog/software-development/ci-cd/runners-caching-and-the-ci-cost-problem). The full series playbook, including how platform engineering fits into the broader CI/CD maturity model, is the [CI/CD Capstone](/blog/software-development/ci-cd/capstone-the-cicd-playbook). For the microservices perspective — how independent deployability interacts with the platform scaffolding model — see [CI/CD and Independent Deployability](/blog/software-development/microservices/ci-cd-and-independent-deployability).

---

## Common Platform Anti-Patterns

Building a platform is hard enough. Building one that developers actually adopt is harder still. The following failure modes appear repeatedly across organisations at every size, and they share a common root: optimising for the platform team's convenience rather than the developer's.

**The "build it and they will come" trap.** The most common early mistake is installing Backstage before defining what the golden path looks like. The result is a catalog with forty blank entries, a scaffolder section with no templates, and a portal that developers visit once, find nothing useful, and never return to. Backstage is a framework for encoding a golden path at scale — it cannot invent the path for you. Before a single line of Backstage YAML is written, the platform team should have a documented answer to "what does deploying a new service correctly look like, end to end?" That answer — the golden path document — is what the catalog, templates, and workflows encode. Ship the document first. Treat the portal as a distribution mechanism for the decisions already made.

**Platform team as gatekeeper.** Self-service platforms occasionally evolve in the opposite direction: rather than removing approval queues, they route every infrastructure request through a new queue called "platform review." A team that must open a ticket for every new Crossplane claim, every new scaffolder template use, every new namespace creation has not gained self-service — it has gained a rebranded ticket queue with a two-week SLA. The test is simple: can a product engineer provision a database and deploy a service without any synchronous interaction with the platform team? If the answer is no, the platform is a gatekeeper, not an enabler. Approvals belong inside automated policy gates (OPA, admission controllers, cost budgets), not inside a human review queue.

**Versioning without deprecation.** Golden-path workflows and scaffolder templates accumulate versions as the platform evolves. The failure mode is releasing v3 of the reusable deploy workflow without retiring v1 or v2. After eighteen months, the fleet has services on four different workflow versions, each with different security controls, different caching behaviour, and different output formats. Security remediations must be applied four times instead of one. The fix is a formal deprecation window on every golden-path version: announce, set a hard deadline, flag stale users in the scorecard, and hard-fail after the grace period. A platform that cannot deprecate old versions cannot improve.

**Optimising for platform team convenience, not developer convenience.** It is easy to build a platform that is easy to operate and maintain from the platform team's perspective, while remaining friction-heavy for the developers it serves. A scaffolder template that outputs the correct YAML but requires a developer to understand twelve Helm values before it works correctly has solved the platform team's encoding problem, not the developer's onboarding problem. The right SLA is not platform deployment frequency, uptime of the portal, or number of templates shipped. The right SLA is developer time-to-value: how long from "I want a new service" to "my first change is in production." Measure this directly, publish it, and let it drive the platform roadmap. If time-to-value is rising, something in the platform is getting harder for developers — regardless of what the platform team's internal metrics show.

**The "not invented here" spiral.** At a certain scale, a platform team accumulates enough infrastructure knowledge to build everything from scratch. This is a trap. When the platform team decides to write a custom service catalog instead of adopting Backstage because "our requirements are unique," or to build a home-grown Kubernetes operator for database provisioning instead of evaluating Crossplane, they are exchanging a known operational cost for an unknown one. The rule is to exhaust adoption before building. Backstage, Crossplane, Port, and Argo CD collectively cover 95% of IDP use cases for most organisations. The 5% of requirements that genuinely cannot be met by existing tooling are worth building custom solutions for. The other 95% are not unique — they feel unique because the platform team is too close to the problem to see how well the standard solutions already fit. Before committing to a custom build, spend one week seriously attempting to satisfy the requirement within the existing tool, with the help of the tool's documentation and community. If that week ends with a genuine gap, build. If it ends with a working solution that required configuration rather than code, adopt.

---

## Key Takeaways

**The DevOps tax is real.** "You build it, you run it" without a shared infrastructure layer forces every team to reinvent the same CI/CD, Dockerfile, and secrets management wheel. At 10+ teams, this is a measurable tax on engineering time.

**An IDP is a product, not a project.** It has users (developers), a roadmap, SLOs, and on-call. If it is built as a project — shipped and handed off — it will rot. It requires a dedicated team that behaves like a product team.

**Catalog before scaffolder before self-service.** Visibility comes first. Once teams can see what exists and who owns it, templates become useful. Once templates are adopted, self-service infrastructure removes the last ticket queue.

**The golden path works because it earns adoption, not because it mandates compliance.** Make the right way faster, better-documented, and more secure than any alternative. Design a governed off-ramp for genuine exceptions. Track off-ramp frequency to find where the path needs to evolve.

**Backstage's catalog kinds model the whole system.** Component, API, Resource, System, Domain — each kind serves a distinct role. The dependency graph that emerges from properly annotated `catalog-info.yaml` files answers incident triage questions in seconds.

**Crossplane gives the platform team infrastructure-as-code with a reconciliation guarantee.** Unlike Terraform applied once, Crossplane continuously reconciles. Manual console changes are reverted. Compliance requirements encoded in a Composition cannot be bypassed by any developer claim.

**Staff a platform team of 1:15–1:30 against product engineers.** Three engineers can support eighty product engineers if the platform is product-managed. The inner source model distributes contribution across the organisation.

**Measure what matters.** DORA metrics track delivery outcomes. SPACE measures human experience. Time-to-first-deploy and incident ownership-finding time are your two best platform-specific leading indicators. DevSat is the lagging signal that validates the whole programme.

**Backstage is the most powerful and the most work.** Port, Cortex, and OpsLevel offer faster time-to-value with less extensibility. Choose based on team size, time-to-value requirements, and data-residency constraints — not hype.

**Don't build a platform prematurely.** Fewer than 10 teams, no dedicated platform team, no defined golden path — any of these is a stop sign. Fix the prerequisite first.

---

## Further Reading

- **Team Topologies** (Skelton & Pais, 2019) — The foundational framework for stream-aligned, platform, enabling, and complicated-subsystem teams. The intellectual home of modern platform engineering.
- **CNCF Platforms White Paper** (2023) — The CNCF's definition of platform engineering, capability taxonomy, and maturity model. Free at cncf.io.
- **Backstage documentation** — docs.backstage.io covers catalog schema, scaffolder template authoring, TechDocs, and plugin development in depth.
- **Crossplane documentation** — crossplane.io/docs covers XRD authoring, Composition design, and provider configuration.
- **DORA State of DevOps Report** (annual) — The primary longitudinal study of software delivery performance. The source data for the four DORA metrics and their correlation with organisational outcomes.
- **The SPACE of Developer Productivity** (Forsgren et al., 2021) — The original Microsoft Research paper defining the SPACE framework. Available via ACM Digital Library.
- **Internal Developer Platforms** by Luca Galante (platformengineering.org) — A practitioner guide with case studies from companies that have shipped IDPs at scale.
- **Humanitec's Platform Engineering report** (annual) — Survey data on IDP adoption, tooling choices, and ROI measurement across hundreds of engineering organisations.
